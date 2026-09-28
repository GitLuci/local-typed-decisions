"""Qwen3-4B Q8_0 (the quantized 4B thinker selected in the quantization study) thinks over the frozen test-2 corpus, once, in 4 fragments; reads its own letters.

Protocol: docs/METHODOLOGY.md (test-2). Same configuration as the 4B-Thinking Q4 run (test2_thinking4b_q4.py) (own chat template,
system prompt of production, lettered options, sampling T 0.6 / top_p 0.95 / top_k 20, seed 20260925 + case index in
the test-2 file, cap 1024, forced close on cap). Primary reading = the thinker itself (letter logits after </think>);
secondary = Qwen3-0.6B FP32 reading the same thought in its own <think> block. Nothing trained.

Phases:
  think --fragment k --fragments 4   cases with index % 4 == k -> reports/test-2/thinking-4b-q4/thoughts-k{k}.jsonl
  merge                              join the fragments (168 unique ids, in corpus order) -> thoughts.jsonl
  read                               Qwen3-0.6B secondary reading -> readings-06b.jsonl
  summary                            summary.json (own = primary, 0.6B = secondary), criteria of the plan
"""
import argparse
import json
import math
from pathlib import Path
try:  # Unix only; on Windows the peak RSS is recorded as null
    import resource
except ImportError:
    resource = None
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.devcal_thinkers import CAP, GGUFThinker, LETTERS, SEED, THREADS, letter_ids, base_provenance, split_thought
from scripts.structural_readout import argmax, mcnemar_exact, paired_bootstrap, read_jsonl, sha256
from scripts.test2_fastpath import evaluate, label_b_index
from scripts.thinking_readout import CLOSE, SAMPLING, prompt_text

CORPUS = ROOT / "examples/test-2-scenarios.jsonl"
CORPUS_SHA256 = "cb9cd5be55e59db86f73b42a39cf48791bc81c8bb62bd27df27593e308b0b828"
JEV = ROOT / "reports/test-2/jev-official/predictions.jsonl"
LABEL_B = ROOT / "examples/test-2-labels-b.jsonl"
FASTPATH_Q8 = ROOT / "reports/test-2/fastpath-4b-q8_0/predictions.jsonl"
FASTPATH_NAME = "4B Q8 fast path"
GGUF = ROOT / "models/gguf/qwen3-4b-Q8_0.gguf"
GGUF_SHA256 = "78bea492bbdc8667db4bc7f3affcbc619d3a28b88daf212cd1334441ab6e5a70"
LOCK = ROOT / "qwen-4b.lock.json"
THINKING_Q4 = ROOT / "reports/test-2/thinking-4b-q4/thoughts.jsonl"  # paired comparison, 129/138
OUT = ROOT / "reports/test-2/thinking-4b-q8"
FRAGMENTS = 4
PRIMARY_GAIN, JEV_MAX_GAP = 7, 5  # test-2 thinking pre-registration, amendment 2, on the 138 cases without `numeric`


def cases_test2():
    rows = read_jsonl(CORPUS)
    assert len(rows) == 168 and all(r["split"] == "test" for r in rows)
    assert sha256(CORPUS) == CORPUS_SHA256, "test-2 corpus differs from the sealed SHA-256"
    for i, c in enumerate(rows):
        c["index"] = i
    return rows


def keys_for(case):
    q = case["question"]
    if q["type"] == "choice":
        keys = list(q["criteria"])
    elif q["type"] == "score":
        keys = [str(i) for i in range(len(q["criteria"]))]
    else:
        keys = ["false", "true"]  # option_descriptions renders A = No, B = Yes
    assert set(keys) == set(case["target"]) and len(keys) == len(case["target"]), case["id"]
    return keys


def softmax(z):
    m = max(z)
    p = [math.exp(v - m) for v in z]
    s = sum(p)
    return [x / s for x in p]


def row_base(case):
    keys = keys_for(case)
    return {"id": case["id"], "index": case["index"], "domain": case["domain"], "question_type": case["question"]["type"],
            "target_kind": case["target_kind"], "keys": keys, "target": [case["target"][k] for k in keys]}


class Thinker(GGUFThinker):
    def __init__(self):
        assert sha256(GGUF) == GGUF_SHA256, "GGUF differs from the sealed hash (rebuild: scripts/build_gguf.sh)"
        lock = json.loads(LOCK.read_text(encoding="utf-8"))
        super().__init__(str(ROOT / lock["path"]), GGUF)
        self.letters = letter_ids(self.tok, len(LETTERS))


def phase_think(args):
    assert 0 <= args.fragment < args.fragments
    OUT.mkdir(parents=True, exist_ok=True)
    raw = OUT / f"thoughts-k{args.fragment}.jsonl"
    cases = [c for c in cases_test2() if c["index"] % args.fragments == args.fragment]
    done = {r["id"] for r in read_jsonl(raw)} if raw.exists() else set()
    thinker = Thinker()
    prov = base_provenance("qwen3-4b-q8_0", {"backend": "gguf"}, json.loads(LOCK.read_text(encoding="utf-8")))
    prov.update({"protocol": "docs/METHODOLOGY.md (test-2, 4B Q8_0 thinking)", "corpus": str(CORPUS.name), "corpus_sha256": CORPUS_SHA256,
                 "fragment": args.fragment, "fragments": args.fragments, "cases": [c["id"] for c in cases],
                 "thinker_info": thinker.info, "resumed_after_cases": len(done), "reader_primary": "own"})
    (OUT / f"provenance-k{args.fragment}.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    with raw.open("a", encoding="utf-8") as log:
        for case in cases:
            if case["id"] in done:
                continue
            row = row_base(case)
            row.update(thinker.think(case, case["index"]))
            row.update(thinker.read_own(case, row["thought"]))
            row["probabilities"] = softmax(row["own_logits"])
            row["own_correct"] = argmax(row["own_logits"]) == argmax(row["target"])
            row["peak_rss_gib"] = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20 if resource else None)
            log.write(json.dumps(row, ensure_ascii=False) + "\n")
            log.flush()
            print(json.dumps({"case": case["id"], "index": case["index"], "tokens": row["thinking_tokens"], "closed": row["closed"],
                              "s": round(row["generation_s"], 1), "tok_per_s": round(row["generated_tokens"] / row["generation_s"], 2),
                              "own_correct": row["own_correct"]}), flush=True)


def phase_merge(args):
    cases = cases_test2()
    rows = []
    for k in range(FRAGMENTS):
        part = OUT / f"thoughts-k{k}.jsonl"
        assert part.exists(), f"missing fragment {k}"
        rows.extend(read_jsonl(part))
    by_id = {r["id"]: r for r in rows}
    assert len(rows) == len(by_id) == 168, f"{len(rows)} rows, {len(by_id)} unique ids"
    assert set(by_id) == {c["id"] for c in cases}
    merged = [by_id[c["id"]] for c in cases]
    for c, r in zip(cases, merged):
        assert r["index"] == c["index"]
    target = OUT / "thoughts.jsonl"
    target.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in merged), encoding="utf-8")
    print(json.dumps({"merged": len(merged), "sha256": sha256(target),
                      "fragments_sha256": {k: sha256(OUT / f"thoughts-k{k}.jsonl") for k in range(FRAGMENTS)}}), flush=True)


def phase_read(args):
    rows = read_jsonl(OUT / "thoughts.jsonl")
    assert len(rows) == 168
    target = OUT / "readings-06b.jsonl"
    if target.exists():
        raise FileExistsError(target)
    from typed_decisions.qwen import QwenDecisionModel
    from scripts.crossmodel_thinking import read_letters
    reader = QwenDecisionModel(lock_file=ROOT / "qwen.lock.json", tree_kernel="dense")
    by_id = {c["id"]: c for c in cases_test2()}
    prov = {"protocol": "docs/METHODOLOGY.md (test-2, 4B Q8_0 thinking)", "phase": "read", "reader": reader.lock_data, "placement": "own_think",
            "thoughts_sha256": sha256(OUT / "thoughts.jsonl"), "head_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
            "script_sha256": sha256(Path(__file__)), "weights_trained": False}
    (OUT / "provenance-read.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    with target.open("w", encoding="utf-8") as log:
        for row in rows:
            case = by_id[row["id"]]
            text = prompt_text(reader, case) + "<think>\n" + row["thought"].rstrip() + CLOSE
            logits, seconds, tokens = read_letters(reader, text, len(row["keys"]))
            rec = {"id": row["id"], "domain": row["domain"], "question_type": row["question_type"], "target_kind": row["target_kind"],
                   "keys": row["keys"], "target": row["target"], "logits": logits, "probabilities": softmax(logits),
                   "correct": argmax(logits) == argmax(row["target"]), "read_s": seconds, "reader_tokens": tokens}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps({"case": row["id"], "correct_06b": rec["correct"]}), flush=True)


def paired(mine, ref, ids):
    b = sum(ref[i] and not mine[i] for i in ids)
    c = sum(mine[i] and not ref[i] for i in ids)
    return {"reference_correct": sum(ref[i] for i in ids), "thinking_correct": sum(mine[i] for i in ids), "reference_only": b,
            "thinking_only": c, "mcnemar_exact_p": mcnemar_exact(b, c),
            "paired_bootstrap_95ci": paired_bootstrap([ref[i] for i in ids], [mine[i] for i in ids])}


def subset_report(rows, cases, jev, labels_b, q8_correct, jev_correct, name, keep):
    sub = [r for r in rows if keep(r) and r["target_kind"] != "known_distribution"]
    ids = [r["id"] for r in sub]
    mine = {r["id"]: argmax(r["probabilities"]) == argmax(r["target"]) for r in sub}
    ev = evaluate([r for r in rows if keep(r)], cases, jev, labels_b)
    return {"n": len(ids), "correct_A": ev["correct_A"], "correct_B": ev["correct_B"], "jev_correct_A": ev["jev_correct_A"],
            "vs_jev": paired(mine, jev_correct, ids), "vs_fastpath_q8": paired(mine, q8_correct, ids)}


def phase_summary(args):
    cases = cases_test2()
    rows = read_jsonl(OUT / "thoughts.jsonl")
    readings = read_jsonl(OUT / "readings-06b.jsonl")
    assert len(rows) == len(readings) == 168
    jev = read_jsonl(JEV)
    labels_b = {r["id"]: r["label"] for r in read_jsonl(LABEL_B)}
    jev_mean = {j["id"]: [sum(v[i] for v in j["probabilities"]) / len(j["probabilities"]) for i in range(len(j["keys"]))] for j in jev}
    jev_correct = {j["id"]: argmax(jev_mean[j["id"]]) == argmax(j["target"]) for j in jev}
    q8_correct = {r["id"]: argmax(r["probabilities"]) == argmax(r["target"]) for r in read_jsonl(FASTPATH_Q8)}
    t4b_correct = {r["id"]: argmax(r["probabilities"]) == argmax(r["target"]) for r in read_jsonl(THINKING_Q4)} if THINKING_Q4.exists() else None
    subsets = {"156_all": lambda r: True, "138_no_numeric": lambda r: r["domain"] != "numeric",
               "120_no_numeric_no_tone": lambda r: r["domain"] not in ("numeric", "subjective_tone")}

    def report(pred_rows):
        out = {"all": evaluate(pred_rows, cases, jev, labels_b),
               "domains": {d: evaluate([r for r in pred_rows if r["domain"] == d], cases, jev, labels_b) for d in sorted({r["domain"] for r in pred_rows})},
               "subsets": {name: subset_report(pred_rows, cases, jev, labels_b, q8_correct, jev_correct, name, keep) for name, keep in subsets.items()}}
        if t4b_correct:
            for name, keep in subsets.items():
                sub = [r for r in pred_rows if keep(r) and r["target_kind"] != "known_distribution"]
                mine = {r["id"]: argmax(r["probabilities"]) == argmax(r["target"]) for r in sub}
                out["subsets"][name]["vs_thinking_4b_q4"] = paired(mine, t4b_correct, [r["id"] for r in sub])
        s138 = out["subsets"]["138_no_numeric"]
        out["criteria"] = {"primary_thinking_pays": s138["vs_fastpath_q8"]["thinking_correct"] >= s138["vs_fastpath_q8"]["reference_correct"] + PRIMARY_GAIN,
                           "primary_rule": f"thinking >= {FASTPATH_NAME} + {PRIMARY_GAIN} on the 138 labelled cases without numeric",
                           "jev_level": (s138["vs_jev"]["paired_bootstrap_95ci"][0] <= 0 <= s138["vs_jev"]["paired_bootstrap_95ci"][1]
                                         and s138["vs_jev"]["reference_correct"] - s138["vs_jev"]["thinking_correct"] <= JEV_MAX_GAP),
                           "jev_rule": f"95% CI of the paired difference contains 0 and Jev - thinking <= {JEV_MAX_GAP} on the 138"}
        return out

    summary = {"protocol": "docs/METHODOLOGY.md (test-2, 4B Q8_0 thinking)", "n": 168, "primary_reader": "own (4B Q8_0)",
               "own": report(rows), "reader_06b_secondary": report(readings),
               "agreement_own_vs_06b": sum(argmax(r["probabilities"]) == argmax(x["probabilities"]) for r, x in zip(rows, readings)),
               "closed_naturally": sum(r["closed"] for r in rows), "hit_cap": sum(not r["closed"] for r in rows),
               "median_thinking_tokens": sorted(r["thinking_tokens"] for r in rows)[84],
               "median_generation_s": sorted(r["generation_s"] for r in rows)[84],
               "mean_generation_s": sum(r["generation_s"] for r in rows) / 168,
               "median_tok_per_s": sorted(r["generated_tokens"] / r["generation_s"] for r in rows)[84],
               "peak_rss_gib_think": max(r["peak_rss_gib"] for r in rows),
               "thoughts_sha256": sha256(OUT / "thoughts.jsonl"), "readings_sha256": sha256(OUT / "readings-06b.jsonl"),
               "head_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    brief = {k: summary[k] for k in ("closed_naturally", "median_thinking_tokens", "median_generation_s", "agreement_own_vs_06b")}
    for who in ("own", "reader_06b_secondary"):
        brief[who] = {name: (s["correct_A"], s["n"], "jev", s["jev_correct_A"], "q8", s["vs_fastpath_q8"]["reference_correct"])
                      for name, s in summary[who]["subsets"].items()} | {"criteria": summary[who]["criteria"],
                      "domains": {d: (v["correct_A"], v["n_labelled"]) for d, v in summary[who]["domains"].items()}}
    print(json.dumps(brief, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("phase", choices=["think", "merge", "read", "summary"])
    p.add_argument("--fragment", type=int, default=0)
    p.add_argument("--fragments", type=int, default=FRAGMENTS)
    p.add_argument("--threads", type=int, default=4, help="llama.cpp threads (recorded in provenance; 2 when two fragments share a 4-vCPU VM)")
    a = p.parse_args()
    import scripts.devcal_thinkers as _cd
    _cd.THREADS = a.threads
    {"think": phase_think, "merge": phase_merge, "read": phase_read, "summary": phase_summary}[a.phase](a)
