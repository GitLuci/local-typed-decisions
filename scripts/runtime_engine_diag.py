"""Post-hoc diagnostic (declared as such): the 4B fast path through the test-3 HARNESS (code independent of the runtime)
on the same llama-server b11205 Vulkan, over the 168 test-2 cases. Separates "runtime defect" from "engine drift"
(test-2 predictions = llama-cpp-python 0.3.35 on CPU; runtime and harness = b11205 on GPU).

    python scripts/runtime_engine_diag.py [--server-exe tools/llama-b11205-vulkan/llama-server.exe]
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.test3_harness import Builder, LlamaServer, read_letters  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-exe", type=Path, default=ROOT / "tools/llama-b11205-vulkan/llama-server.exe")
    ap.add_argument("--out", type=Path, default=ROOT / "reports/runtime-measurement/engine-diag-4b-harness.jsonl")
    args = ap.parse_args()
    cases = [json.loads(l) for l in (ROOT / "examples/test-2-scenarios.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    srv = subprocess.Popen([str(args.server_exe), "-m", str(ROOT / "models/qwen3-4b-gguf/qwen3-4b-Q8_0.gguf"), "-ngl", "99",
                            "--host", "127.0.0.1", "--port", "8189", "-t", "4", "-tb", "4", "-c", "2048", "-np", "1", "--no-webui"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    out = []
    try:
        for _ in range(150):
            try:
                httpx.get("http://127.0.0.1:8189/health", timeout=3).raise_for_status()
                break
            except Exception:
                time.sleep(2)
        b, be = Builder(ROOT / "models/qwen3-4b"), LlamaServer("http://127.0.0.1:8189")
        for c in cases:
            q = c["question"]
            keys = list(q["criteria"]) if q["type"] == "choice" else (["false", "true"] if q["type"] == "noul" else [str(i) for i in range(len(q["criteria"]))])
            ids, size = b.fast_ids(c["state"], q)
            r = read_letters(be, b, ids, size, 0)
            p = r["probabilities"]
            out.append({"id": c["id"], "argmax": keys[max(range(len(p)), key=p.__getitem__)], "probabilities": dict(zip(keys, p)),
                        "letters_missing": r["letters_missing"]})
    finally:
        srv.terminate()
        srv.wait(timeout=30)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(o, ensure_ascii=False) + "\n" for o in out), encoding="utf-8")
    print("done", len(out))


if __name__ == "__main__":
    main()
