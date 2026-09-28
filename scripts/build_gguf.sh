#!/usr/bin/env bash
# Reproduce the Qwen3-4B GGUF files used in the quantization study (docs/HISTORY.md). No imatrix, no data.
# Expected SHA-256 are in reports/manifests/gguf-manifest-qwen3-4b.json; llama.cpp commit is pinned there too.
# Usage: scripts/build_gguf.sh <llama.cpp dir> <output dir>      (needs: cmake, gcc/clang, python3 with gguf+torch)
set -euo pipefail
LLAMA=${1:?llama.cpp checkout}; OUT=${2:?output dir}; ROOT=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$OUT"
( cd "$LLAMA" && cmake -B build -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF && cmake --build build --config Release -j --target llama-quantize )
python3 "$LLAMA/convert_hf_to_gguf.py" "$ROOT/models/qwen3-4b" --outtype bf16 --outfile "$OUT/qwen3-4b-bf16.gguf"
"$LLAMA/build/bin/llama-quantize" "$OUT/qwen3-4b-bf16.gguf" "$OUT/qwen3-4b-Q8_0.gguf" Q8_0 4
"$LLAMA/build/bin/llama-quantize" "$OUT/qwen3-4b-bf16.gguf" "$OUT/qwen3-4b-Q4_K_M.gguf" Q4_K_M 4
rm -f "$OUT/qwen3-4b-bf16.gguf"
sha256sum "$OUT"/qwen3-4b-Q8_0.gguf "$OUT"/qwen3-4b-Q4_K_M.gguf
