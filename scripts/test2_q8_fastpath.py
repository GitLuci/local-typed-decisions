"""Qwen3-4B Q8_0 (llama.cpp) on the test-2 fast path, once; optional comparison against the local FP32 run.

Protocol: docs/METHODOLOGY.md (test-2). Same production prompt as `QwenDecisionModel` (built by its own `_sequences`
over the HF tokenizer, weights not loaded in PyTorch), letter readout, thinking off, original order. Token ids are
checked against the llama.cpp tokenizer per case. With --compare <predictions.jsonl of the FP32 run made by
scripts/test2_fastpath.py>, reports argmax agreement and mean TV against the pre-registered criterion.
"""
import argparse
import json
import math
try:  # Unix only; on Windows the peak RSS is recorded as null (test3_harness only uses PromptOnly)
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
from typed_decisions.qwen import QwenDecisionModel, SYSTEM
from scripts.structural_readout import read_jsonl, sha256
from scripts.test2_fastpath import CORPUS, JEV, LABEL_B, evaluate, argmax


class PromptOnly(QwenDecisionModel):
    """The production prompt builder without loading PyTorch weights."""

    def __init__(self, path):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        self.max_len, self.max_packed = 2048, 4096
        self.aliases, self.alias_token_ids = self._aliases()
        self.readout = "letter"
        marker = "TYPED_DECISIONS_CONTENT_SENTINEL"
        template = self.tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": marker}],
                                                tokenize=False, add_generation_prompt=True, enable_thinking=False)
        assert template.count(marker) == 1
        self._shells = {"letter": tuple(template.split(marker))}


def softmax(z):
    m = max(z)
    p = [math.exp(v - m) for v in z]
    return [x / sum(p) for x in p]


def compare(rows, fp32_path):
    fp32 = {r["id"]: r for r in read_jsonl(fp32_path)}
    pairs = [(r, fp32[r["id"]]) for r in rows if r["id"] in fp32 and r["tokens_match_hf"]]
    same = sum(argmax(a["probabilities"]) == argmax(b["probabilities"]) for a, b in pairs)
    tv = [0.5 * sum(abs(x - y) for x, y in zip(a["probabilities"], b["probabilities"])) for a, b in pairs]
    return {"fp32_file": str(fp32_path), "fp32_sha256": sha256(fp32_path), "compared": len(pairs), "argmax_equal": same,
            "argmax_equal_fraction": same / len(pairs), "tv_mean": sum(tv) / len(tv), "tv_max": max(tv),
            "criterion": "argmax >= 98% of 168 and tv_mean <= 0.02",
            "passes": len(pairs) == 168 and same >= math.ceil(0.98 * 168) and sum(tv) / len(tv) <= 0.02}


def run(args):
    out = ROOT / "reports/test-2/fastpath-4b-q8_0"
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "predictions.jsonl"
    cases = read_jsonl(CORPUS)
    assert len(cases) == 168
    rows = read_jsonl(raw) if raw.exists() else []
    if len(rows) < 168:
        from llama_cpp import Llama
        builder = PromptOnly(str(ROOT / "models/qwen3-4b"))
        llm = Llama(model_path=str(args.gguf), n_ctx=2048, n_threads=args.threads, logits_all=False, verbose=False, seed=0)
        provenance = {"protocol": "docs/METHODOLOGY.md (test-2, 4B Q8_0 fast path)", "gguf": args.gguf.name, "gguf_sha256": sha256(args.gguf),
                      "threads": args.threads, "corpus_sha256": sha256(CORPUS), "readout": "letter", "thinking": False,
                      "head_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
                      "typed_decisions_sha256": {p.name: sha256(p) for p in sorted((ROOT / "typed_decisions").glob("*.py"))},
                      "script_sha256": sha256(__file__), "weights_trained": False, "imatrix": False,
                      "llama_cpp_python": __import__("llama_cpp").__version__}
        (out / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
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
    summary = {"tag": "4b-q8_0", "tokens_match_hf": sum(r["tokens_match_hf"] for r in rows),
               "all": evaluate(rows, cases, jev, labels_b),
               "domains": {d: evaluate([r for r in rows if r["domain"] == d], cases, jev, labels_b) for d in sorted({r["domain"] for r in rows})},
               "median_ms": sorted(r["elapsed_ms"] for r in rows)[len(rows) // 2],
               "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20 if resource else None}
    if args.compare:
        summary["vs_fp32_local"] = compare(rows, args.compare)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"all": summary["all"], "domains": {d: (v.get("correct_A"), v["n_labelled"]) for d, v in summary["domains"].items()},
                      "vs_fp32_local": summary.get("vs_fp32_local")}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", type=Path, required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--compare", type=Path, default=None)
    run(p.parse_args())
