"""test-3: run ONE arm of the harness (`scripts/test3_harness.py run`) with its own `llama-server` (measured on an RX 7600).

Starts the llama.cpp b11205 Vulkan server with the flags used for the published numbers:
`-ngl` (30 for 8B Q8, 99 for 4B Q8), `-t 4 -tb 4`, `-c 2048` (4096 only for the 4B thinking arm, cap 1024), `-np 1`,
`--no-webui`. With 30 layers on an 8 GB RX 7600 the 8B does not fit a 4096 KV cache (ErrorOutOfDeviceMemory); the
8B short arm fits in 2048 (longest prompt 282 tokens + 128 thinking tokens + readout). Waits for `/health`, runs the
harness and stops the server at the end, including on error or Ctrl+C.

    python scripts/test3_arm.py --arm 8b-q8-fast --split dev
    python scripts/test3_arm.py --arm 4b-q8-think --server-exe tools/llama-b11205-vulkan/llama-server.exe
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "tools/llama-b11205-vulkan/llama-server.exe"
GGUF = {"8b": ROOT / "models/qwen3-8b-gguf/qwen3-8b-Q8_0.gguf", "4b": ROOT / "models/qwen3-4b-gguf/qwen3-4b-Q8_0.gguf"}
NGL = {"8b": 30, "4b": 99}
THREADS = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["8b-q8-fast", "4b-q8-fast", "8b-q8-short", "4b-q8-think"])
    ap.add_argument("--split", default="test", choices=["test", "dev"])
    ap.add_argument("--port", type=int, default=8187)
    ap.add_argument("--server-exe", type=Path, default=SERVER, help="llama-server executable (llama.cpp b11205 Vulkan)")
    a = ap.parse_args()
    size = a.arm.split("-")[0]
    cmd = [str(a.server_exe), "-m", str(GGUF[size]), "-ngl", str(NGL[size]), "--host", "127.0.0.1",
           "--port", str(a.port), "-t", str(THREADS), "-tb", str(THREADS), "-c", str(4096 if a.arm == "4b-q8-think" else 2048),
           "-np", "1", "--no-webui"]
    log = ROOT / "reports/test-3" / f"server-{a.arm}-{a.split}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    url = f"http://127.0.0.1:{a.port}"
    with log.open("ab") as lf:
        server = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
        try:
            for _ in range(300):
                if server.poll() is not None:
                    raise SystemExit(f"llama-server exited with {server.returncode} before becoming ready; see {log.name}")
                try:
                    httpx.get(f"{url}/health", timeout=5).raise_for_status()
                    break
                except Exception:
                    time.sleep(2)
            else:
                raise SystemExit("llama-server was not ready within 10 min")
            harness = [sys.executable, str(ROOT / "scripts/test3_harness.py"), "run", "--arm", a.arm,
                       "--server", url, "--gguf", str(GGUF[size])] + (["--split", "dev"] if a.split == "dev" else [])
            return subprocess.call(harness, cwd=ROOT)
        finally:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()


if __name__ == "__main__":
    sys.exit(main())
