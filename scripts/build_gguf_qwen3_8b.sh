#!/usr/bin/env bash
# Reproduce the Qwen3-8B GGUFs used for the fast/slow modes (8B variant of the quantization recipe; qwen-8b.lock.json).
# No imatrix, no calibration data. Low-disk recipe (30 GB VM): Q8_0 straight from the safetensors with
# convert_hf_to_gguf.py --outtype q8_0, then Q4_K_M requantized from that Q8_0 (llama-quantize --allow-requantize).
# Expected SHA-256 and the pinned llama.cpp commit (9588757 = 95887577ab5fead779581a7030a83c7752ff3234) are in
# reports/manifests/gguf-manifest-qwen3-8b.json. Needs: cmake, gcc, python3 with torch (CPU) + gguf, ~31 GB free
# (16.4 GB safetensors + 8.7 GB Q8_0 + 5.0 GB Q4_K_M) or pass --delete-safetensors to free 16.4 GB after the Q8_0.
# Usage: scripts/build_gguf_qwen3_8b.sh <llama.cpp dir> <output dir> [--delete-safetensors]
set -euo pipefail
LLAMA=${1:?llama.cpp checkout at 9588757}; OUT=${2:?output dir}; ROOT=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$OUT"
( cd "$LLAMA" && git rev-parse --short HEAD | grep -q '^9588757' ) || { echo "llama.cpp must be at commit 9588757"; exit 1; }
( cd "$LLAMA" && cmake -B build -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DCMAKE_BUILD_TYPE=Release \
  && cmake --build build --config Release -j4 --target llama-quantize )
python3 "$LLAMA/convert_hf_to_gguf.py" "$ROOT/models/qwen3-8b" --outtype q8_0 --outfile "$OUT/qwen3-8b-Q8_0.gguf"
if [ "${3:-}" = "--delete-safetensors" ]; then rm -f "$ROOT"/models/qwen3-8b/*.safetensors; fi
"$LLAMA/build/bin/llama-quantize" --allow-requantize "$OUT/qwen3-8b-Q8_0.gguf" "$OUT/qwen3-8b-Q4_K_M.gguf" Q4_K_M 4
sha256sum "$OUT"/qwen3-8b-Q8_0.gguf "$OUT"/qwen3-8b-Q4_K_M.gguf
