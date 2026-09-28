"""Production fast-path baselines on the frozen test-2 corpus (Choice, Score, Noul), one run per model.

Protocol: docs/METHODOLOGY.md (test-2). Uses `QwenDecisionModel.predict` exactly as `--serve` does (branched tree
kernel, letter readout, thinking off, original option order), one question per call. Reports accuracy against
label A (corpus author) and label B (blind second annotator), exact McNemar against the official Jev baseline run
(per-item baseline outputs are not distributed: supply your own run at JEV), TV on the
known distributions, level MAE for score. Nothing trained.
"""
import argparse
import hashlib
import json
import math
try:  # Unix only; on Windows the peak RSS is recorded as null
    import resource
except ImportError:
    resource = None
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.structural_readout import mcnemar_exact, paired_bootstrap, read_jsonl, sha256

CORPUS = ROOT / "examples/test-2-scenarios.jsonl"
JEV = ROOT / "reports/test-2/jev-official/predictions.jsonl"
LABEL_B = ROOT / "examples/test-2-labels-b.jsonl"


def argmax(v):
    return max(range(len(v)), key=v.__getitem__)


def vector(answer, question, keys):
    if question["type"] == "noul":
        return [1 - answer["noul"], answer["noul"]]  # keys are ["false", "true"]
    return [answer["probabilities"][k] for k in keys]


def label_b_index(case, labels_b):
    """Index of label B in the case's key order; falls back to label A outside the annotated domains."""
    keys = list(case["target"])
    if case["id"] in labels_b:
        lab = labels_b[case["id"]]
        return keys.index(str(lab)) if str(lab) in keys else keys.index(lab)
    return argmax([case["target"][k] for k in keys])


def evaluate(rows, cases, jev, labels_b):
    by_id = {c["id"]: c for c in cases}
    lab = [r for r in rows if r["target_kind"] != "known_distribution"]
    if not lab:  # e.g. the `random` domain: only the distribution metric applies
        dist = rows
        return {"n_labelled": 0, "tv_known_distribution_mean": sum(0.5 * sum(abs(p - t) for p, t in zip(r["probabilities"], r["target"])) for r in dist) / len(dist)}
    a = [argmax(r["probabilities"]) == argmax(r["target"]) for r in lab]
    b = [argmax(r["probabilities"]) == label_b_index(by_id[r["id"]], labels_b) for r in lab]
    jev_mean = {j["id"]: [sum(v[i] for v in j["probabilities"]) / len(j["probabilities"]) for i in range(len(j["keys"]))] for j in jev}
    ja = [argmax(jev_mean[r["id"]]) == argmax(r["target"]) for r in lab]
    out = {"n_labelled": len(lab), "correct_A": sum(a), "correct_B": sum(b), "jev_correct_A": sum(ja),
           "discordant_model_only_A": sum(x and not y for x, y in zip(a, ja)), "discordant_jev_only_A": sum(y and not x for x, y in zip(a, ja)),
           "mcnemar_vs_jev_A": mcnemar_exact(sum(x and not y for x, y in zip(a, ja)), sum(y and not x for x, y in zip(a, ja))), "bootstrap_vs_jev_A": paired_bootstrap(a, ja),
           "nll_A": -sum(math.log(max(r["probabilities"][argmax(r["target"])], 1e-300)) for r in lab) / len(lab)}
    scores = [r for r in lab if r["question_type"] == "score"]
    if scores:
        out["score_mae_levels_A"] = sum(abs(argmax(r["probabilities"]) - argmax(r["target"])) for r in scores) / len(scores)
    nouls = [r for r in lab if r["question_type"] == "noul"]
    if nouls:
        for cls, idx in (("false", 0), ("true", 1)):
            pos = [r for r in nouls if argmax(r["target"]) == idx]
            out[f"noul_recall_{cls}"] = (sum(argmax(r["probabilities"]) == idx for r in pos) / len(pos)) if pos else None
    dist = [r for r in rows if r["target_kind"] == "known_distribution"]
    if dist:
        out["tv_known_distribution_mean"] = sum(0.5 * sum(abs(p - t) for p, t in zip(r["probabilities"], r["target"])) for r in dist) / len(dist)
    return out


def tracked_files(directory):
    """Files under `directory` that git tracks (committed reports); [] outside a repo or without git."""
    try:
        out = subprocess.check_output(["git", "ls-files", "--", str(directory)], cwd=ROOT, stderr=subprocess.DEVNULL).decode()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return []
    return [line for line in out.splitlines() if line]


def guard_output(out, force, tracked=None):
    """Never resume over, or rewrite, committed reports: ask for a new --tag, or --force to overwrite on purpose."""
    tracked = tracked_files(out) if tracked is None else list(tracked)
    if tracked and not force:
        raise SystemExit(f"{out} has {len(tracked)} committed file(s); use another --tag (e.g. --tag {out.name.replace('fastpath-', '')}-repro) or --force to overwrite")


def domain_counts(domains):
    """(correct_A, n_labelled) per domain; `random` has only distributions, so no correct_A (None)."""
    return {d: (v.get("correct_A"), v.get("n_labelled")) for d, v in domains.items()}


def run(args):
    out = ROOT / "reports/test-2" / f"fastpath-{args.tag}"
    guard_output(out, getattr(args, "force", False))
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "predictions.jsonl"
    cases = read_jsonl(CORPUS)
    assert len(cases) == 168 and all(c["split"] == "test" for c in cases)
    jev = read_jsonl(JEV)
    labels_b = {r["id"]: r["label"] for r in read_jsonl(LABEL_B)}
    from typed_decisions.qwen import QwenDecisionModel
    model = QwenDecisionModel(lock_file=args.lock, threads=args.threads)
    provenance = {"protocol": "docs/METHODOLOGY.md (test-2 baselines)", "lock": str(args.lock.name), "model": model.lock_data,
                  "threads": args.threads, "corpus_sha256": sha256(CORPUS), "jev_official_sha256": sha256(JEV),
                  "label_b_sha256": sha256(LABEL_B), "readout": model.readout, "tree_kernel": model.tree_kernel,
                  "thinking": False, "head_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
                  "typed_decisions_sha256": {p.name: sha256(p) for p in sorted((ROOT / "typed_decisions").glob("*.py"))},
                  "script_sha256": sha256(__file__), "weights_trained": False}
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    rows = read_jsonl(raw) if raw.exists() else []
    done = {r["id"] for r in rows}
    with raw.open("a", encoding="utf-8") as log:
        for case in cases:
            if case["id"] in done:
                continue
            keys = list(case["target"])
            started = time.perf_counter()
            result = model.predict(case["state"], {"q": case["question"]})
            elapsed = (time.perf_counter() - started) * 1000
            probs = vector(result["answers"]["q"], case["question"], keys)
            row = {"id": case["id"], "domain": case["domain"], "question_type": case["question"]["type"],
                   "target_kind": case["target_kind"], "keys": keys, "target": [case["target"][k] for k in keys],
                   "probabilities": probs, "logits": result["logits"]["q"], "elapsed_ms": elapsed,
                   "input_tokens": result["usage"]["input_tokens"]}
            rows.append(row)
            log.write(json.dumps(row, ensure_ascii=False) + "\n")
            log.flush()
            print(json.dumps({"id": case["id"], "done": len(rows), "ms": round(elapsed)}), flush=True)
    summary = {"tag": args.tag, "all": evaluate(rows, cases, jev, labels_b),
               "domains": {d: evaluate([r for r in rows if r["domain"] == d], cases, jev, labels_b) for d in sorted({r["domain"] for r in rows})},
               "median_ms": sorted(r["elapsed_ms"] for r in rows)[len(rows) // 2],
               "input_tokens": sum(r["input_tokens"] for r in rows),
               "peak_rss_gib": (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20 if resource else None)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"all": summary["all"], "domains": domain_counts(summary["domains"])}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lock", type=Path, required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--force", action="store_true", help="overwrite a reports/test-2/fastpath-<tag> folder that has committed files")
    run(p.parse_args())
