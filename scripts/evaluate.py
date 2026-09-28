"""Incremental offline evaluation, with cross-domain results and process monitoring."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from typed_decisions.runtime import LocalDecisionModel
from typed_decisions.metrics import summarize, cluster_bootstrap
import psutil


@contextmanager
def resource_record(estimated_gib=3, description="offline Laya diagnostic"):
    """Refuse to start without enough free RAM for the model plus a 4 GiB margin."""
    if psutil.virtual_memory().available < max(8, estimated_gib+4)*1024**3:
        raise RuntimeError(f"Insufficient available RAM for {description} (model plus 4 GiB margin)")
    yield


def distribution(answer, question):
    if question["type"] == "noul":
        return [1-answer["noul"], answer["noul"]]
    return list(answer["probabilities"].values())


def run(args):
    source = args.corpus.read_bytes()
    corpus = [json.loads(line) for line in source.decode("utf-8-sig").splitlines() if line]
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    # New run only: never silently mix old predictions with a changed model or corpus.
    raw_file = out / "predictions.jsonl"
    if raw_file.exists():
        raise FileExistsError(f"Choose a new output directory; {raw_file} already exists")
    print(json.dumps({"pid": os.getpid(), "free_gib": psutil.virtual_memory().available/1024**3}), flush=True)
    estimated = 4 if args.backend == "qwen" else 3
    with resource_record(estimated, f"offline {args.backend} diagnostic"):
        started = time.perf_counter()
        if args.backend == "qwen":
            from typed_decisions.qwen import QwenDecisionModel
            model = QwenDecisionModel(lock_file=args.qwen_lock)
        else:
            model = LocalDecisionModel()
        load_s = time.perf_counter()-started
        first = corpus[0]
        model.predict(first["state"], first["questions"])  # explicit warmup, excluded
        rows, latencies = [], []
        with raw_file.open("x", encoding="utf-8") as f:
            for case in corpus:
                if psutil.virtual_memory().available < 4*1024**3:
                    raise RuntimeError("Stopped own evaluation: available RAM below 4 GiB")
                result = model.predict(case["state"], case["questions"])
                record = {"case": case, "result": result}
                f.write(json.dumps(record, ensure_ascii=False)+"\n")
                f.flush()
                latencies.append(result["metadata"]["elapsed_ms"])
                for qid, q in case["questions"].items():
                    p = distribution(result["answers"][qid], q)
                    target = case["targets"][qid]
                    if not isinstance(target, list):
                        index = int(target) if q["type"] != "choice" else list(q["criteria"]).index(target)
                        target = [float(i == index) for i in range(len(p))]
                    rows.append({"case_id": case["id"], "domain": case["domain"], "split": case["split"],
                                 "type": q["type"], "p": p, "target": target})
                print(json.dumps({"case": case["id"], "ms": round(latencies[-1]), "rss_gib": psutil.Process().memory_info().rss/1024**3}), flush=True)
        # Test question independence against genuine model calls, not mocks.
        grouped = model.predict(first["state"], first["questions"])
        max_delta = 0.
        single_ms = []
        for qid, question in first["questions"].items():
            one = model.predict(first["state"], {"renamed": question})
            single_ms.append(one["metadata"]["elapsed_ms"])
            max_delta = max(max_delta, max(abs(a-b) for a,b in zip(
                distribution(grouped["answers"][qid], question), distribution(one["answers"]["renamed"], question))))
        summary = {"model": getattr(model, "lock_data", None) or json.loads((ROOT/"model.lock.json").read_text()),
                   "corpus_sha256": hashlib.sha256(source).hexdigest(), "load_s": load_s,
                   "hardware": {"device": "cpu", "threads": 4, "processor": os.environ.get("PROCESSOR_IDENTIFIER"),
                                "ram_bytes": psutil.virtual_memory().total},
                   "latency_ms": {"median": statistics.median(latencies), "p95": sorted(latencies)[int(.95*(len(latencies)-1))]},
                   "independence_max_probability_delta": max_delta,
                   "batch_3_ms": grouped["metadata"]["elapsed_ms"], "sum_3_single_ms": sum(single_ms),
                   "peak_working_set_bytes": getattr(psutil.Process().memory_info(), "peak_wset", None),
                   "calibration": "unvalidated", "scope": "synthetic pilot; not generalization proof",
                   "domains": {}, "primitives": {}}
        for domain in sorted({r["domain"] for r in rows}):
            subset = [r for r in rows if r["domain"] == domain and r["split"] == "test"]
            summary["domains"][domain] = {**summarize(subset), "bootstrap_95": cluster_bootstrap(subset)}
        for t in ("choice", "score", "noul"):
            subset = [r for r in rows if r["type"] == t and r["domain"] != "aleatoric" and r["split"] == "test"]
            summary["primitives"][t] = summarize(subset)
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({"done": str(out), "latency_ms": summary["latency_ms"], "independence_delta": max_delta}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=ROOT/"examples/pilot.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=["laya", "qwen"], default="laya")
    parser.add_argument("--qwen-lock", type=Path)
    run(parser.parse_args())
