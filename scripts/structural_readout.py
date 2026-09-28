"""Structural (no-training) readout experiment over frozen Qwen3 weights.

Layer aggregation, format/order ensembles and a fixed layer contrast. Protocol
and closed candidate list: docs/HISTORY.md (reading tricks). Phases, each a separate
invocation and in this order:

  collect --splits development calibration   per-layer logits, dev+cal only
  select                                     selection.json from dev+cal only
  collect --splits test                      refuses to run without selection.json
  confirm                                    test summary, once

No parameter is created or fitted. Intermediate layers are read through the
model's own final RMSNorm and existing LM-head rows for the answer codes (logit
lens). No change to typed_decisions/. Set TD_BASELINE_REF to a git ref to assert
that typed_decisions/ is unchanged since that ref before collecting.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from typed_decisions.tree import pack_tree
from scripts.audit_scenarios import infer, metrics, questions_for
from scripts.evaluate import resource_record

FORMATS = ("letter", "json")
BASELINE = "letter"
CORPUS = ROOT / "examples/audit-003-scenarios.jsonl"
JEV_RUN = ROOT / "reports/audit-003/jev-scenarios/predictions.jsonl"  # per-item baseline outputs: not distributed
PARITY_TOLERANCE = 1e-3
BOOTSTRAP_SEED, BOOTSTRAP_N = 20260925, 10_000
PROMOTION_MIN_GAIN, PROMOTION_MAX_DOMAIN_LOSS = 3, 1
DEVCAL, TEST = "predictions-devcal.jsonl", "predictions-test.jsonl"


def candidates(layers):
    """Closed, pre-registered list: name -> [(weight, format, order, layer)]."""
    late = range(layers - layers // 4 + 1, layers + 1)

    def mean(views):
        return [(1 / len(views), *v) for v in views]
    both = [(f, o) for f in FORMATS for o in (0, 1)]
    return {
        "letter": mean([("letter", 0, layers)]),
        "json": mean([("json", 0, layers)]),
        "letter_orders": mean([("letter", o, layers) for o in (0, 1)]),
        "letter_json": mean([(f, 0, layers) for f in FORMATS]),
        "letter_json_orders": mean([(f, o, layers) for f, o in both]),
        "letter_late_layers": mean([("letter", 0, l) for l in late]),
        "all_late_layers": mean([(f, o, l) for f, o in both for l in late]),
        "letter_dola": [(1., "letter", 0, layers), (-1., "letter", 0, layers // 2)],
    }


def log_softmax(z):
    m = max(z)
    s = m + math.log(sum(math.exp(x - m) for x in z))
    return [x - s for x in z]


def pooled(by_format, spec):
    """Weighted sum of per-view log-probabilities; used as logits downstream."""
    first = next(iter(by_format.values()))
    total = [0.] * len(first["keys"])
    for weight, fmt, order, layer in spec:
        view = log_softmax(by_format[fmt]["layer_logits"][layer - 1][order])
        total = [t + weight * v for t, v in zip(total, view)]
    return {k: first[k] for k in ("id", "domain", "split", "target_kind", "keys", "target")} | {"logits": [total]}


def argmax(values):
    return max(range(len(values)), key=values.__getitem__)


def correct(row):
    return argmax(row["logits"][0]) == argmax(row["target"])


def hard(rows):
    return [r for r in rows if r["target_kind"] != "known_distribution"]


def score(rows):
    labelled = hard(rows)
    return {"hard_n": len(labelled), "hard_correct": sum(map(correct, labelled)),
            "hard": metrics(labelled), "random": metrics([r for r in rows if r not in labelled])}


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.
    return min(1., 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def paired_bootstrap(base, challenger):
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(base)
    diffs = sorted(sum(challenger[i] - base[i] for i in (rng.randrange(n) for _ in range(n))) / n
                   for _ in range(BOOTSTRAP_N))
    return [diffs[int(.025 * BOOTSTRAP_N)], diffs[int(.975 * BOOTSTRAP_N) - 1]]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


BASELINE_REF = os.environ.get("TD_BASELINE_REF")


def frozen_check():
    """Optional: fail if typed_decisions/ differs from the git ref in TD_BASELINE_REF."""
    if BASELINE_REF:
        subprocess.run(["git", "-c", "core.excludesFile=.gitignore", "diff", "--exit-code",
                        BASELINE_REF, "--", "typed_decisions"], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)


def baseline_commit():
    if not BASELINE_REF:
        return None
    return subprocess.check_output(["git", "rev-parse", BASELINE_REF + "^{commit}"], cwd=ROOT).decode().strip()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def grouped(records):
    cases = {}
    for r in records:
        cases.setdefault(r["id"], {})[r["format"]] = r
    assert all(set(v) == set(FORMATS) for v in cases.values())
    return cases


def layer_forward(model, case, fmt):
    """One backbone forward (same tree as production); read every layer's output."""
    t = model.torch
    model.readout = fmt
    questions, variants = questions_for(case, "letter")
    layers = model.model.model.layers
    norm = model.model.model.norm
    started = time.perf_counter()
    with model.lock, t.inference_mode():
        sequences, sizes, prefix_length = model._sequences(case["state"], questions)
        packed = pack_tree(sequences, prefix_length, model.max_packed)
        captured = []

        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured.append(hidden[0, packed.readout_positions])
        handles = [layer.register_forward_hook(hook) for layer in layers]
        try:
            out = model.model.model(
                input_ids=t.tensor([packed.ids]), position_ids=t.tensor([packed.positions]), use_cache=False,
                attention_mask={"full_attention": None}, tree_layout=packed,
                tree_bucket_branches=model.bucket_branches)
        finally:
            for handle in handles:
                handle.remove()
        final = out.last_hidden_state[0, packed.readout_positions]
        assert len(captured) == len(layers)
        assert (norm(captured[-1]) - final).abs().max().item() < 1e-5
        per_layer = [model._readout(norm(h), sizes) for h in captured]
    elapsed = (time.perf_counter() - started) * 1000
    keys = list(case["question"]["criteria"])
    layer_logits = []
    for rows in per_layer:
        by_variant = []
        for row, options in zip(rows, variants):
            scores = dict(zip((k for k, _ in options), row))
            by_variant.append([scores[k] for k in keys])
        layer_logits.append(by_variant)
    return {"id": case["id"], "domain": case["domain"], "split": case["split"], "format": fmt,
            "target_kind": case["target_kind"], "keys": keys, "target": [case["target"][k] for k in keys],
            "layer_logits": layer_logits, "elapsed_ms": elapsed, "prefix_tokens": prefix_length,
            "computed_input_tokens": len(packed.ids)}


@contextmanager
def low_ram_watchdog(estimated_gib, description, stop_below_gib):
    """Run below the usual 4 GB free-RAM margin, guarded by an in-process watchdog.

    The watchdog exits (only this process) if free RAM drops below stop_below_gib.
    """
    import psutil
    done = threading.Event()

    def watch():
        while not done.wait(0.5):
            free = psutil.virtual_memory().available / 2**30
            if free < stop_below_gib:
                print(json.dumps({"watchdog_stop": True, "free_gib": free, "run": description}), flush=True)
                os._exit(3)
    threading.Thread(target=watch, daemon=True).start()
    try:
        yield
    finally:
        done.set()


def collect(args):
    frozen_check()
    splits = tuple(args.splits)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if splits == ("development", "calibration"):
        name, phase = DEVCAL, "devcal"
    elif splits == ("test",):
        name, phase = TEST, "test"
        if not (out / "selection.json").exists():
            raise RuntimeError("test collection requires a persisted selection.json first")
    else:
        raise ValueError("splits must be 'development calibration' or 'test'")
    raw = out / name
    if raw.exists():
        raise FileExistsError(raw)
    cases = [c for c in read_jsonl(CORPUS) if c["split"] in splits]
    parity = {}
    if args.parity_run:
        parity_provenance = json.loads((args.parity_run / "provenance.json").read_text(encoding="utf-8"))
        parity = {(r["id"], r["method"]): r for r in read_jsonl(args.parity_run / "predictions.jsonl")
                  if r["split"] in splits}
    source = out / "source"
    source.mkdir(exist_ok=True)
    for name_ in ("structural_readout.py", "audit_scenarios.py"):
        (source / name_).write_bytes((ROOT / "scripts" / name_).read_bytes())
    provenance = {
        "phase": phase, "splits": list(splits), "protocol": "docs/HISTORY.md (reading tricks)",
        "baseline_commit": baseline_commit(),
        "head_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
        "corpus_sha256": sha256(CORPUS), "parity_run": str(args.parity_run) if args.parity_run else None,
        "dtype": args.dtype, "parity_asserted": bool(args.parity_run) and args.dtype == "float32",
        "low_ram_allowed": args.allow_low_ram,
        "weights_trained": False, "parameters_fitted": 0, "thinking": False,
        "source_hashes": {p.name: sha256(p) for p in sorted(source.iterdir())},
    }
    if phase == "test":
        provenance["selection_sha256"] = sha256(out / "selection.json")
    records = []
    guard = (low_ram_watchdog(args.memory_gib, f"structural {phase} per-layer readout", args.stop_below_gib)
             if args.allow_low_ram else resource_record(args.memory_gib, f"structural {phase} per-layer readout"))
    with guard:
        import torch
        import transformers
        from typed_decisions.qwen import QwenDecisionModel
        loader = transformers.AutoModelForCausalLM.from_pretrained

        def with_dtype(*a, **kw):
            # Only the storage/compute precision changes; typed_decisions/ is untouched.
            return loader(*a, **kw | {"dtype": getattr(torch, args.dtype)})
        with patch.object(transformers.AutoModelForCausalLM, "from_pretrained", side_effect=with_dtype):
            model = QwenDecisionModel(lock_file=args.lock)
        assert model.tree_kernel == "branched"
        assert next(model.model.parameters()).dtype == getattr(torch, args.dtype)
        if parity:
            assert model.lock_data == parity_provenance["model"], "parity run used a different model"
        provenance["model"] = model.lock_data
        provenance["num_hidden_layers"] = model.model.config.num_hidden_layers
        (out / f"provenance-{phase}.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        with raw.open("x", encoding="utf-8") as log, \
             patch.object(model.model, "generate", side_effect=AssertionError("No generation")):
            for fmt in FORMATS:
                warm = layer_forward(model, cases[0], fmt)  # warm-up, not recorded
                if not parity:
                    # No prior run for this model: check the hook path against production predict.
                    model.readout = fmt
                    live = infer(model, cases[0], "letter")["logits"]
                    error = max(abs(a - b) for zs, zr in zip(warm["layer_logits"][-1], live) for a, b in zip(zs, zr))
                    assert error < PARITY_TOLERANCE, (fmt, error)
                    provenance.setdefault("live_parity_max_logit_error", {})[fmt] = error
                    (out / f"provenance-{phase}.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
                print(json.dumps({"warmup": fmt, "ms": round(warm["elapsed_ms"])}), flush=True)
            for index, case in enumerate(cases):
                formats = FORMATS if index % 2 == 0 else tuple(reversed(FORMATS))
                for fmt in formats:
                    record = layer_forward(model, case, fmt)
                    error = None
                    if parity:
                        reference = parity[(case["id"], fmt)]["logits"]
                        error = max(abs(a - b) for zs, zr in zip(record["layer_logits"][-1], reference)
                                    for a, b in zip(zs, zr))
                        assert math.isfinite(error)
                        if args.dtype == "float32":
                            assert error < PARITY_TOLERANCE, (case["id"], fmt, error)
                        ref_choice = argmax(reference[0])
                        record["parity_argmax_same"] = argmax(record["layer_logits"][-1][0]) == ref_choice
                    record["parity_max_logit_error"] = error
                    records.append(record)
                    log.write(json.dumps(record, ensure_ascii=False) + "\n")
                    log.flush()
                print(json.dumps({"phase": phase, "case": case["id"], "records": len(records)}), flush=True)
    errors = [r["parity_max_logit_error"] for r in records if r["parity_max_logit_error"] is not None]
    print(json.dumps({"complete": True, "phase": phase, "records": len(records),
                      "max_parity_error": max(errors) if errors else None,
                      "parity_argmax_changed": sum(not r.get("parity_argmax_same", True) for r in records)}),
          flush=True)


def select(args):
    out = args.output
    if (out / "selection.json").exists():
        raise FileExistsError(out / "selection.json")
    if (out / TEST).exists():
        raise RuntimeError("test predictions already exist; selection must precede them")
    records = read_jsonl(out / DEVCAL)
    assert {r["split"] for r in records} == {"development", "calibration"}
    cases = grouped(records)
    layers = len(records[0]["layer_logits"])
    specs = candidates(layers)
    scores = {name: score([pooled(c, spec) for c in cases.values()]) for name, spec in specs.items()}
    challengers = [n for n in specs if n != BASELINE]
    ranked = sorted(challengers, key=lambda n: (-scores[n]["hard_correct"], scores[n]["hard"]["nll"], len(specs[n])))
    layer_curve = {f"{fmt}_L{l}": score([pooled(c, [(1., fmt, 0, l)]) for c in cases.values()])["hard_correct"]
                   for fmt in FORMATS for l in range(1, layers + 1)}
    selection = {
        "selected_challenger": ranked[0], "baseline": BASELINE, "ranking": ranked,
        "rule": "dev+cal hard-label correct count, then mean NLL at T=1, then fewer views; baseline excluded",
        "source_file": DEVCAL, "source_sha256": sha256(out / DEVCAL),
        "candidates": {n: [list(v) for v in s] for n, s in specs.items()},
        "devcal": {n: {"hard_correct": s["hard_correct"], "hard_n": s["hard_n"], "nll": s["hard"]["nll"],
                       "random_tv": s["random"]["mean_total_variation"]} for n, s in scores.items()},
        "diagnostic_layer_curve_hard_correct": layer_curve,
        "diagnostic_note": "Per-layer curve is descriptive only; single layers are not candidates.",
        "promotion_criterion": {"min_test_gain_cases": PROMOTION_MIN_GAIN,
                                "max_cases_lost_in_any_domain": PROMOTION_MAX_DOMAIN_LOSS},
    }
    (out / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(json.dumps({k: selection[k] for k in ("selected_challenger", "ranking")} |
                     {"devcal": {n: v["hard_correct"] for n, v in selection["devcal"].items()}}), flush=True)


def confirm(args):
    out = args.output
    if (out / "summary.json").exists():
        raise FileExistsError(out / "summary.json")
    selection = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    assert sha256(out / DEVCAL) == selection["source_sha256"]
    test_provenance = json.loads((out / "provenance-test.json").read_text(encoding="utf-8"))
    assert test_provenance["selection_sha256"] == sha256(out / "selection.json")
    records = read_jsonl(out / TEST)
    assert {r["split"] for r in records} == {"test"}
    cases = grouped(records)
    specs = {n: [tuple(v) for v in s] for n, s in selection["candidates"].items()}
    rows = {n: [pooled(c, s) for c in cases.values()] for n, s in specs.items()}
    jev = {r["id"]: r for r in read_jsonl(JEV_RUN)}
    challenger = selection["selected_challenger"]

    def described(name):
        labelled = hard(rows[name])
        domains = {}
        for d in sorted({r["domain"] for r in labelled}):
            domains[d] = sum(correct(r) for r in labelled if r["domain"] == d)
        agree = 0
        for r in rows[name]:
            j = jev[r["id"]]
            assert j["keys"] == r["keys"]
            agree += r["keys"][argmax(r["logits"][0])] == j["returned_choices"][0]
        return score(rows[name]) | {"domains_correct": domains, "jev_argmax_agreement": agree,
                                    "n": len(rows[name])}

    table = {n: described(n) for n in specs}
    jev_rows = [{"target_kind": j["target_kind"], "target": j["target"], "logits": [j["logits"][0]],
                 "domain": j["domain"]} for j in jev.values()]
    jev_domains = {d: sum(correct(r) for r in hard(jev_rows) if r["domain"] == d)
                   for d in sorted({r["domain"] for r in hard(jev_rows)})}
    base = [correct(r) for r in hard(rows[BASELINE])]
    chal = [correct(r) for r in hard(rows[challenger])]
    b = sum(x and not y for x, y in zip(base, chal))
    c = sum(y and not x for x, y in zip(base, chal))
    gain = sum(chal) - sum(base)
    domain_delta = {d: table[challenger]["domains_correct"][d] - table[BASELINE]["domains_correct"][d]
                    for d in table[BASELINE]["domains_correct"]}
    promote = gain >= PROMOTION_MIN_GAIN and min(domain_delta.values()) >= -PROMOTION_MAX_DOMAIN_LOSS
    median_ms = {fmt: sorted(r["elapsed_ms"] for r in records if r["format"] == fmt) for fmt in FORMATS}
    summary = {
        "selection": {k: selection[k] for k in ("selected_challenger", "rule", "source_sha256")},
        "confirmatory": {"baseline": BASELINE, "challenger": challenger,
                         "baseline_correct": sum(base), "challenger_correct": sum(chal), "hard_n": len(base),
                         "gain_cases": gain, "baseline_only_correct": b, "challenger_only_correct": c,
                         "mcnemar_exact_p": mcnemar_exact(b, c),
                         "paired_bootstrap_95ci_accuracy_diff": paired_bootstrap(base, chal),
                         "domain_delta": domain_delta, "promotion_criterion_met": promote},
        "jev_test_domains_correct": jev_domains,
        "descriptive_all_candidates_not_selected_here": table,
        "median_forward_ms": {f: v[len(v) // 2] for f, v in median_ms.items()},
        "max_parity_error": max((r["parity_max_logit_error"] for r in records
                                 if r["parity_max_logit_error"] is not None), default=None),
    }
    if args.reference_run:
        # Cross-model comparison of the unchanged production readout, paired by case id.
        ref_cases = grouped(read_jsonl(args.reference_run / TEST))
        ref_layers = len(next(iter(ref_cases.values()))[BASELINE]["layer_logits"])
        ref_spec = candidates(ref_layers)[BASELINE]  # the reference model's own last layer
        ref = {r["id"]: correct(r) for r in hard([pooled(c, ref_spec) for c in ref_cases.values()])}
        mine = {r["id"]: correct(r) for r in hard(rows[BASELINE])}
        assert set(ref) == set(mine)
        ids = sorted(mine)
        b = sum(ref[i] and not mine[i] for i in ids)
        c = sum(mine[i] and not ref[i] for i in ids)
        summary["versus_reference_model"] = {
            "reference_run": str(args.reference_run), "readout": BASELINE,
            "reference_correct": sum(ref.values()), "this_model_correct": sum(mine.values()), "hard_n": len(ids),
            "gain_cases": sum(mine.values()) - sum(ref.values()), "reference_only_correct": b,
            "this_only_correct": c, "mcnemar_exact_p": mcnemar_exact(b, c),
            "paired_bootstrap_95ci_accuracy_diff": paired_bootstrap([ref[i] for i in ids], [mine[i] for i in ids]),
            "promotion_criterion_met": sum(mine.values()) - sum(ref.values()) >= PROMOTION_MIN_GAIN}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("confirmatory", "versus_reference_model") if k in summary},
                     indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=["collect", "select", "confirm"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--splits", nargs="+")
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--parity-run", type=Path)
    parser.add_argument("--reference-run", type=Path)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--allow-low-ram", action="store_true",
                        help="run below the 4 GB free-RAM margin, with an in-process watchdog")
    parser.add_argument("--stop-below-gib", type=float, default=2.5)
    parser.add_argument("--memory-gib", type=float, default=5)
    args = parser.parse_args()
    {"collect": collect, "select": select, "confirm": confirm}[args.phase](args)
