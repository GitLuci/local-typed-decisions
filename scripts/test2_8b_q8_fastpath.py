"""Qwen3-8B Q8_0 (llama.cpp) on the test-2 fast path, once, no thinking.

Protocol: docs/METHODOLOGY.md (test-2). Same reader and report format as `scripts/test2_q8_fastpath.py` (4B Q8_0):
production prompt built by `QwenDecisionModel._sequences` over the HF tokenizer of the 8B (weights not loaded in
PyTorch), letter readout, thinking off, original order, token ids checked against the llama.cpp tokenizer per case.
Extra: the GGUF SHA-256 must match the sealed one; accuracy is also reported on the 138 without `numeric` and the
120 without `numeric` nor `subjective_tone`. Nothing trained; no imatrix.
"""
import argparse
import hashlib
import json
try:  # Unix only; on Windows the peak RSS is recorded as null
    import resource
except ImportError:
    resource = None
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.structural_readout import read_jsonl
from scripts.test2_fastpath import CORPUS, JEV, LABEL_B, evaluate
from scripts.test2_q8_fastpath import PromptOnly, softmax

GGUF_SHA256 = "bc7efafeb86690a3efe8c0f9babc48a4e747cd4d9cec490c4bec1e298f446506"
CORPUS_SHA256 = "cb9cd5be55e59db86f73b42a39cf48791bc81c8bb62bd27df27593e308b0b828"
OUT = ROOT / "reports/test-2/fastpath-8b-q8_0"


def sha256(path):
    """Chunked (structural_readout.sha256 reads the whole file; the GGUF is 8.7 GB on a 15 GB VM)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def run(args):
    OUT.mkdir(parents=True, exist_ok=True)
    raw = OUT / "predictions.jsonl"
    assert sha256(CORPUS) == CORPUS_SHA256, "corpus hash mismatch"
    cases = read_jsonl(CORPUS)
    assert len(cases) == 168
    rows = read_jsonl(raw) if raw.exists() else []
    if len(rows) < 168:
        gguf_sha = sha256(args.gguf)
        assert gguf_sha == GGUF_SHA256, f"GGUF hash mismatch: {gguf_sha}"
        from llama_cpp import Llama
        builder = PromptOnly(str(args.model_dir))
        llm = Llama(model_path=str(args.gguf), n_ctx=2048, n_threads=args.threads, logits_all=False, verbose=False, seed=0)
        provenance = {"protocol": "docs/METHODOLOGY.md (test-2, 8B Q8_0 fast path)", "gguf": args.gguf.name, "gguf_sha256": gguf_sha,
                      "gguf_bytes": args.gguf.stat().st_size, "model_dir": str(args.model_dir), "lock": "qwen-8b.lock.json",
                      "threads": args.threads, "corpus_sha256": CORPUS_SHA256, "readout": "letter", "thinking": False,
                      "head_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
                      "typed_decisions_sha256": {p.name: sha256(p) for p in sorted((ROOT / "typed_decisions").glob("*.py"))},
                      "script_sha256": sha256(__file__), "weights_trained": False, "imatrix": False,
                      "llama_cpp_python": __import__("llama_cpp").__version__}
        (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        done = {r["id"] for r in rows}
        with raw.open("a", encoding="utf-8") as log:
            for case in cases:
                if case["id"] in done:
                    continue
                keys = list(case["target"])
                sequences, sizes, _ = builder._sequences(case["state"], {"q": case["question"]})
                ids = sequences[0]
                gg = llm.tokenize(builder.tok.decode(ids).encode("utf-8"), add_bos=False, special=True)
                started = time.perf_counter()
                llm.reset()
                llm.eval(ids)
                elapsed = (time.perf_counter() - started) * 1000
                logits = np.ctypeslib.as_array(llm._ctx.get_logits(), shape=(llm._n_vocab,)).copy()
                letters = [float(logits[t]) for t in builder.alias_token_ids[:sizes[0]]]
                row = {"id": case["id"], "domain": case["domain"], "question_type": case["question"]["type"],
                       "target_kind": case["target_kind"], "keys": keys, "target": [case["target"][k] for k in keys],
                       "probabilities": softmax(letters), "logits": letters, "tokens_match_hf": gg == ids,
                       "elapsed_ms": elapsed, "input_tokens": len(ids)}
                rows.append(row)
                log.write(json.dumps(row, ensure_ascii=False) + "\n")
                log.flush()
                print(json.dumps({"id": case["id"], "done": len(rows), "ms": round(elapsed)}), flush=True)
    jev = read_jsonl(JEV)
    labels_b = {r["id"]: r["label"] for r in read_jsonl(LABEL_B)}
    no_numeric = [r for r in rows if r["domain"] != "numeric"]
    no_numeric_no_tone = [r for r in no_numeric if r["domain"] != "subjective_tone"]
    summary = {"tag": "8b-q8_0", "tokens_match_hf": sum(r["tokens_match_hf"] for r in rows),
               "all": evaluate(rows, cases, jev, labels_b),
               "no_numeric": evaluate(no_numeric, cases, jev, labels_b),
               "no_numeric_no_tone": evaluate(no_numeric_no_tone, cases, jev, labels_b),
               "domains": {d: evaluate([r for r in rows if r["domain"] == d], cases, jev, labels_b) for d in sorted({r["domain"] for r in rows})},
               "median_ms": sorted(r["elapsed_ms"] for r in rows)[len(rows) // 2],
               "peak_rss_gib": (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20 if resource else None)}
    nn = summary["no_numeric"]
    summary["criteria"] = {"primary_138_ge_129": nn["correct_A"] >= 129,
                           "jev_level_gap_le_5": nn["jev_correct_A"] - nn["correct_A"] <= 5,
                           "jev_138": nn["jev_correct_A"], "model_138": nn["correct_A"]}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"all": (summary["all"]["correct_A"], summary["all"]["n_labelled"]),
                      "no_numeric": (nn["correct_A"], nn["n_labelled"]),
                      "no_numeric_no_tone": (summary["no_numeric_no_tone"]["correct_A"], summary["no_numeric_no_tone"]["n_labelled"]),
                      "domains": {d: (v.get("correct_A"), v["n_labelled"]) for d, v in summary["domains"].items()},
                      "criteria": summary["criteria"], "median_ms": summary["median_ms"]}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, default=ROOT / "models/qwen3-8b")
    p.add_argument("--threads", type=int, default=4)
    run(p.parse_args())
