"""Thinking-enabled decision readout over frozen Qwen3 weights (no training).

Protocol: see docs/HISTORY.md (thinking). The model generates its <think> block with
Qwen's recommended thinking-mode sampling and a fixed per-case seed, up to a
token budget (closed by force if exhausted); the decision is then read exactly
like production: logits of the option letter codes at the answer position.

Phases: `smoke` (development cases only; speed and pipeline check) and `test`
(the 45 labelled test cases, original option order, run once). No selection:
the method is fixed in advance, so no development/test split decision exists.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from typed_decisions.qwen import SYSTEM, option_descriptions, render
from scripts.structural_readout import (CORPUS, argmax, baseline_commit, frozen_check, low_ram_watchdog,
                                        mcnemar_exact, paired_bootstrap, read_jsonl, sha256)
from scripts.evaluate import resource_record

SAMPLING = {"do_sample": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20}
SEED = 20260925
CLOSE = "\n</think>\n\n"


def prompt_text(model, case):
    """Same system prompt, state and option rendering as production, thinking on."""
    q = case["question"]
    options = "\n".join(f"{model.aliases[i]}: {s}" for i, s in enumerate(option_descriptions(q)))
    content = (f"STATE:\n{render(case['state'])}\n\nQUESTION:\n" + render(q["instructions"]) +
               "\n\nOPTIONS:\n" + options + "\n\nAnswer with the letter code only.")
    return model.tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=True)


def think_and_read(model, case, budget, index, assistant=None, min_think=0):
    t, tok = model.torch, model.tok
    text = prompt_text(model, case)
    ids = tok.encode(text, add_special_tokens=False)
    end_think = tok.encode("</think>", add_special_tokens=False)
    assert len(end_think) == 1
    t.manual_seed(SEED + index)
    started = time.perf_counter()
    with t.inference_mode():
        out = model.model.generate(t.tensor([ids]), attention_mask=t.ones(1, len(ids), dtype=t.long),
                                   max_new_tokens=budget, min_new_tokens=min_think, eos_token_id=end_think[0], **SAMPLING,
                                   **({"assistant_model": assistant} if assistant is not None else {}))
    generated = out[0, len(ids):].tolist()
    generation_s = time.perf_counter() - started
    closed = end_think[0] in generated
    if closed:
        thought = tok.decode(generated[:generated.index(end_think[0])])
    else:
        thought = tok.decode(generated)
    # Rebuild the text so both closed and forced cases share one answer boundary.
    answer_prefix = text + thought.rstrip() + CLOSE
    answer_ids = tok.encode(answer_prefix, add_special_tokens=False)
    size = len(case["question"]["criteria"])
    with t.inference_mode():
        # Project only the last position (same values as slicing full logits, far less memory).
        hidden = model.model.model(input_ids=t.tensor([answer_ids]), use_cache=False).last_hidden_state[0, -1]
        logits = model.model.get_output_embeddings()(hidden)
        letters = logits[model.alias_token_ids[:size]].float().tolist()
        full = logits.float().log_softmax(-1)
        letter_mass = full[model.alias_token_ids[:size]].exp().sum().item()
    keys = list(case["question"]["criteria"])
    return {"id": case["id"], "domain": case["domain"], "split": case["split"], "target_kind": case["target_kind"],
            "keys": keys, "target": [case["target"][k] for k in keys], "logits": [letters],
            "thinking_tokens": len(generated) if not closed else generated.index(end_think[0]),
            "closed_naturally": closed, "budget": budget, "letter_mass": letter_mass,
            "generation_s": generation_s, "total_s": time.perf_counter() - started, "thought": thought}


def correct(row):
    return argmax(row["logits"][0]) == argmax(row["target"])


def run(args):
    frozen_check()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    raw = out / f"predictions-{args.phase}.jsonl"
    if raw.exists() and not args.resume:
        raise FileExistsError(raw)
    cases = [c for c in read_jsonl(CORPUS) if c["target_kind"] != "known_distribution"]
    if args.phase == "smoke":
        cases = [c for c in cases if c["split"] == "development"][:args.smoke_n]
    else:
        cases = [c for c in cases if c["split"] == "test"]
        assert len(cases) == 45
    source = out / "source"
    source.mkdir(exist_ok=True)
    for name in ("thinking_readout.py", "structural_readout.py"):
        (source / name).write_bytes((ROOT / "scripts" / name).read_bytes())
    provenance = {"phase": args.phase, "protocol": "docs/HISTORY.md (thinking)", "budget": args.budget,
                  "sampling": SAMPLING, "seed_base": SEED, "min_think": args.min_think, "forced_close": CLOSE,
                  "baseline_commit": baseline_commit(),
                  "head_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
                  "corpus_sha256": sha256(CORPUS), "weights_trained": False, "thinking": True,
                  "source_hashes": {p.name: sha256(p) for p in sorted(source.iterdir())}}
    guard = (low_ram_watchdog(args.memory_gib, f"thinking {args.phase}", args.stop_below_gib)
             if args.allow_low_ram else resource_record(args.memory_gib, f"thinking {args.phase}"))
    # Resume after an interruption: keep finished cases; seeds stay SEED + original index,
    # so the remaining cases are generated exactly as in an uninterrupted run.
    records = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines()] if raw.exists() else []
    done = {r["id"] for r in records}
    assert all(r["id"] in {c["id"] for c in cases} for r in records)
    if records:
        provenance["resumed_after_cases"] = len(records)
    with guard:
        from typed_decisions.qwen import QwenDecisionModel
        model = QwenDecisionModel(lock_file=args.lock, tree_kernel="dense")
        provenance["model"] = model.lock_data
        (out / f"provenance-{args.phase}.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        with raw.open("a" if records else "x", encoding="utf-8") as log:
            for index, case in enumerate(cases):
                if case["id"] in done:
                    continue
                record = think_and_read(model, case, args.budget, index, min_think=args.min_think)
                records.append(record)
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
                log.flush()
                print(json.dumps({"case": case["id"], "tokens": record["thinking_tokens"],
                                  "closed": record["closed_naturally"], "s": round(record["total_s"], 1),
                                  "tok_per_s": round(record["thinking_tokens"] / record["generation_s"], 2)}), flush=True)
    if args.phase == "test" and args.no_thinking_run:
        # Paired against the same model's production readout (thinking off), original order.
        ref = {}
        for r in read_jsonl(args.no_thinking_run / "predictions-test.jsonl"):
            if r["format"] == "letter" and r["target_kind"] != "known_distribution":
                ref[r["id"]] = argmax(r["layer_logits"][-1][0]) == argmax(r["target"])
        mine = {r["id"]: correct(r) for r in records}
        assert set(ref) == set(mine)
        ids = sorted(mine)
        b = sum(ref[i] and not mine[i] for i in ids)
        c = sum(mine[i] and not ref[i] for i in ids)
        domains = sorted({r["domain"] for r in records})
        summary = {
            "no_thinking_correct": sum(ref.values()), "thinking_correct": sum(mine.values()), "n": len(ids),
            "gain_cases": sum(mine.values()) - sum(ref.values()),
            "no_thinking_only_correct": b, "thinking_only_correct": c, "mcnemar_exact_p": mcnemar_exact(b, c),
            "paired_bootstrap_95ci_accuracy_diff": paired_bootstrap([ref[i] for i in ids], [mine[i] for i in ids]),
            "domains_thinking": {d: sum(mine[r["id"]] for r in records if r["domain"] == d) for d in domains},
            "domains_no_thinking": {d: sum(ref[r["id"]] for r in records if r["domain"] == d) for d in domains},
            "promotion_criterion_met": sum(mine.values()) - sum(ref.values()) >= 3,
            "closed_naturally": sum(r["closed_naturally"] for r in records),
            "median_thinking_tokens": sorted(r["thinking_tokens"] for r in records)[len(records) // 2],
            "median_seconds_per_question": sorted(r["total_s"] for r in records)[len(records) // 2],
            "min_letter_mass": min(r["letter_mass"] for r in records),
            "no_thinking_run": str(args.no_thinking_run), "provenance": provenance}
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in summary.items() if k != "provenance"}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=["smoke", "test"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--smoke-n", type=int, default=2)
    parser.add_argument("--no-thinking-run", type=Path)
    parser.add_argument("--memory-gib", type=float, default=5)
    parser.add_argument("--allow-low-ram", action="store_true", help="run below the 4 GB free-RAM margin, with a watchdog")
    parser.add_argument("--stop-below-gib", type=float, default=2.5)
    parser.add_argument("--min-think", type=int, default=0, help="forbid </think> before this many tokens")
    parser.add_argument("--resume", action="store_true", help="continue an interrupted run, skipping finished cases")
    run(parser.parse_args())
