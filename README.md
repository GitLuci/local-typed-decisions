# local-typed-decisions

Typed decisions — `Choice`, `Score` and `Noul` (yes/no) — with a full probability distribution over the allowed
answers, computed **locally** with open-weight models on a consumer GPU (llama.cpp, Vulkan) or on CPU, with **no
fine-tuning**. The model never writes free text for the caller: the answer is always one of the options you pass,
read from the probabilities of the option letters.

The reference we compare against is **Jev 1.13 by TypeSafe, used as reference via its API**; the HTTP request/response
shape follows its public primitives so clients can switch backends ([docs/API.md](docs/API.md)). This is a research
project, not a production service, and it is not affiliated with TypeSafe.

## What it does

```json
POST /v1/systemone
{"state": "The parcel arrived on Tuesday. The customer thanked the support team.",
 "questions": {"tone": {"type": "choice", "instructions": "What is the tone?",
                         "criteria": {"positive": "Positive", "negative": "Negative"}}}}

→ {"answers": {"tone": {"type": "choice", "choice": "positive",
                        "probabilities": {"positive": 0.98, "negative": 0.02}, "confidence": 0.86}}, ...}
```

- The prompt shows the state, the question and the options labeled with letters; the answer is the softmax of the
  model's logits restricted to those letters, at the answer position. Optionally the model first *thinks*
  (`<think>…</think>`, bounded budget) and the letters are read after the thought.
- Each question is answered independently (no cross-question leakage; checked by the conformance suite).
- Weights are frozen: Qwen3-4B and Qwen3-8B, quantized to Q8_0 **without an importance matrix** (no data, no
  gradients).
- `confidence` = 1 − normalized entropy. **Probabilities are not calibrated.**

## Modes

Measured on an **AMD Radeon RX 7600 (8 GB)** with llama.cpp b11205 Vulkan, batch 1. Accuracy on the 900 sealed
questions of test-3 (Jev: 841/900).

| mode | model | how it decides | test-3 /900 | median s/question | per request in `serve`¹ |
|---|---|---|---:|---:|---:|
| `ultra` | Qwen3-4B Q8_0 (4.3 GB) | one forward, read the letter | 726 | 0.29 | 0.48–0.54 s |
| `fast` | Qwen3-8B Q8_0 (8.7 GB, 30/36 layers on GPU) | one forward, read the letter | 782 | 0.48 | 0.57–0.63 s |
| `medium` | Qwen3-4B Q8_0 | think ≤ 1024 tokens (median 269), then read | **816** | 7.3 | 6.0–8.7 s |
| `slow` | Qwen3-8B Q8_0 | think ≤ 128 tokens, then read | 804 | 11.6 | 12.4–12.5 s |
| `routed` | 4B/8B by domain | per-domain choice of the four above | 829² | 0.50 (mean 4.5) | — |

¹ `python -m typed_decisions serve` smoke, 3 requests per mode after the first one (the first request loads the model:
11–22 s). Peak VRAM +4.4 GB (4B) / +6.5 GB (8B); see `reports/serve-smoke/summary.json`.
² Computed from the four arms' test-3 answers with a map chosen on test-3 itself, so it is optimistic. On test-2 it scores 144/156 (= Jev)
from existing predictions and **145/156 measured end to end** with the released server, mean 4.5 s per question
including 5 model switches ([RESULTS §4](docs/RESULTS.md#4-routed-mode)).

## Results

### test-3: stress test with independent gold (900 sealed questions)

9 domains × 100 questions, difficulty levels N1–N3 fixed before any model ran, gold independent of Jev and of the
evaluated models ([METHODOLOGY](docs/METHODOLOGY.md#4-test-3-independent-gold)). "Jev level" = lower bound of the
conservative 95 % CI of Δ above −3 points (pre-registered). **No local arm reaches Jev level overall.**

| arm (mode) | correct /900 | Δ vs Jev (points) | 95 % CI | Jev level |
|---|---:|---:|---|---|
| Jev 1.13 (API) | 841 | — | — | reference |
| routed (combined from the four arms, post hoc)³ | 829 | −1.3 | — | not tested |
| 4B Q8 think (`medium`) | **816** | **−2.8** | [−5.0; −0.6] | no |
| 8B Q8 short think (`slow`) | 804 | −4.1 | [−6.2; −2.1] | no |
| 8B Q8 fast (`fast`) | 782 | −6.6 | [−8.9; −4.3] | no |
| 4B Q8 fast (`ultra`) | 726 | −12.8 | [−15.6; −10.0] | no |

Per domain (correct /100):

| domain | Jev | `fast` | `ultra` | `slow` | `medium` | routed to |
|---|---:|---:|---:|---:|---:|---|
| factual | 100 | 91 | 91 | 96 | 93 | `slow` |
| numeric | 80 | 77 | 72 | 77 | **98** (beats Jev, +18 [9.7; 27.0]) | `medium` |
| deterministic | 95 | 82 | 88 | 93 | 92 | `slow` |
| sentence | 98 | 81 | 82 | 85 | 90 | `medium` |
| sentiment | 90 | 87 | 86 | 88 | 85 | `slow` |
| subjective_tone | 97 | 92 | 52 | 92 | 87 | `fast` |
| robotic_style | 88 | 84 | 76 | 84 | 84 | `fast` |
| noul_refund | 93 | 92 | 84 | 93 | 91 | `fast` |
| score_urgency | 100 | 96 | 95 | 96 | 96 | `fast` |

³ Sum of the per-domain arm in the "routed to" column below; no CI or verdict because the map was selected on
these same results.

The routed column is, per domain, the best arm on this table with ties going to the faster arm (one exception:
`noul_refund` goes to `fast`, 1 point below `slow`, i.e. within noise and 20× cheaper). It was chosen **post hoc**,
see caveats.

### test-2 (168 cases frozen by a second author; 156 labeled)

| configuration | /156 | /138 without `numeric` |
|---|---:|---:|
| Jev 1.13 (API) | 144 | 135 |
| **routed** (existing predictions / measured end to end) | **144 / 145** | 132 / — |
| 8B Q8 short think (`slow`) | 144 | 133 |
| 4B Q8 think (`medium`) | 141 | 129 |
| 8B Q8 fast (`fast`) | 132 | 126 |
| 4B Q8 fast (`ultra`) | 123 (120 on the GPU engine) | 115 |

Per-domain test-2 tables, paired comparisons (routed − Jev: +0.0 [−3.8; 3.8]), earlier models and every caveat:
[docs/RESULTS.md](docs/RESULTS.md).

### Caveats (short)

- Gold of the authored test-3 items was produced by LLM labelers (blind double labeling + adjudication, 97.4 %
  agreement) **without a human audit**; results carry an "LLM gold, not human-validated" warning.
- Engine drift CPU (llama-cpp-python) vs GPU (llama.cpp Vulkan): ~3 % of argmax change on test-2.
- The routed map was chosen post hoc on test-3 and only confirmed on test-2 (which had been seen before).
- Probabilities are uncalibrated. Per-item Jev outputs are withheld (commercial API); only aggregates are shipped.

## Quickstart

Requirements: Python 3.11+, a Vulkan-capable GPU driver (or CPU only, slower), ~13 GB of disk for the two Q8_0 GGUF
files plus the tokenizer files.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements-server.txt
```

1. **llama.cpp:** download the **b11205** release for your platform from
   <https://github.com/ggml-org/llama.cpp/releases/tag/b11205> (Vulkan build) and extract it so that
   `tools/llama-b11205-vulkan/llama-server(.exe)` exists. On Linux/macOS, edit `launch.executable` in
   `examples/llama-*.json` (the shipped configs point to `llama-server.exe`).
2. **Models:** the GGUF files are on the Hugging Face Hub at
   [`bssgoat/local-typed-decisions`](https://huggingface.co/bssgoat/local-typed-decisions) (configured in
   `release-models.json` and `examples/models-hf.json`); tokenizers come from the official Qwen repos at pinned
   revisions. The downloader installs only files whose size and SHA-256 match the committed manifest, never
   overwrites a mismatching local file, and fetches only the two Q8_0 files the runtime uses. While the model
   repository is private, log in first (`hf auth login` or `HF_TOKEN`).

   ```bash
   python scripts/fetch_models.py                # download + verify
   python scripts/fetch_models.py --verify-only  # offline check of an existing install
   ```
3. **Serve** (loopback only; the first request of each model starts llama-server lazily; one model resident at a
   time; Ctrl+C stops the servers it started):

   ```bash
   python -m typed_decisions serve --config examples/llama-routed.json --port 8000
   python -m typed_decisions serve --config examples/llama-routed.json --mode medium --port 8000   # fixed mode
   ```
4. **Ask:**

   ```bash
   curl http://127.0.0.1:8000/health
   curl -X POST http://127.0.0.1:8000/v1/systemone -H "Content-Type: application/json" \
        --data-binary @examples/routed-request.json
   ```

   In `routed` mode each question needs a domain in the `domains` map (there is no domain classifier yet); in fixed
   modes omit `domains`. Always inspect `errors` even on HTTP 200: a missing option probability affects only that
   question. Details and limits: [docs/RUNTIME.md](docs/RUNTIME.md).

For CPU-only use set `gpu_layers: 0` in the config (not benchmarked for this runtime). The older research backends
(Laya heads, Qwen FP32 shared-prefix tree via PyTorch) are still available with
`python -m typed_decisions --backend qwen ...`; they need `requirements.lock.txt`.

## Tests and reproduction

```bash
python -m pytest -q          # unit tests with simulated HTTP/processes; no weights, no GPU, no network
```

- `docs/METHODOLOGY.md` — pre-registration, sealed sets and hashes, independent gold, levels, statistics.
- `docs/RESULTS.md` — every number with its source report; `reports/` holds the aggregate JSON/MD.
- `docs/HISTORY.md` — what was tried and what failed (small models, reading tricks, thinking, quantization, candidates,
  prompts).
- `docs/OPEN_WORK.md` — open fronts (vision schema with small VLMs, speed, full offload, domain inference, new sealed
  set for the router).
- Reproduce test-3 arms: `python scripts/test3_seal.py --check`, then `scripts/test3_arm.py` per arm and
  `scripts/test3_harness.py analyze` (see the script help). test-2 runners: `scripts/test2_*.py`. Runtime
  measurement: `scripts/runtime_measure.py`. Building the GGUF files from the Qwen safetensors:
  `scripts/build_gguf.sh`, `scripts/build_gguf_qwen3_8b.sh` (llama.cpp commit `9588757`, no imatrix).

## Layout

`typed_decisions/` runtime and server · `conformance/` backend-independent contract oracle · `scripts/` benchmark
harnesses, model download/build · `tests/` · `examples/` configs, requests and the frozen benchmark sets ·
`labels/` human-audit sample · `reports/` aggregate results and manifests · `docs/`.

The benchmark data is mostly Portuguese (pt-PT and pt-BR) plus English items from public datasets; see
[DATA_LICENSES.md](DATA_LICENSES.md).

## License

Code and authored data: [Apache-2.0](LICENSE). Model weights: Qwen3 (Apache-2.0), redistributed as GGUF
quantizations. Public-dataset items: see [DATA_LICENSES.md](DATA_LICENSES.md). "Jev" and "TypeSafe" are names of a
third-party product and company, used here only to identify the comparison baseline.
