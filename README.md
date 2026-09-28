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

Modes are named by speed. Measured on an **AMD Radeon RX 7600 (8 GB)** with llama.cpp b11205 Vulkan, batch 1;
accuracy on the 900 sealed questions of test-3. Sorted by accuracy.

| mode | model | how it decides | test-3 /900 | median s/question | per request in `serve`¹ |
|---|---|---|---:|---:|---:|
| *Jev 1.13 (API, baseline)* | — | — | *841* | network | — |
| `routed` | 4B/8B by domain | per-domain choice of the four below | 829² | 0.50 (mean 4.5) | — |
| `medium` | Qwen3-4B Q8_0 | think ≤ 1024 tokens (median 269), then read | **816** | 7.3 | 6.0–8.7 s |
| `slow` | Qwen3-8B Q8_0 | think ≤ 128 tokens, then read | 804 | 11.6 | 12.4–12.5 s |
| `fast` | Qwen3-8B Q8_0 (8.7 GB, 30/36 layers on GPU) | one forward, read the letter | 782 | 0.48 | 0.57–0.63 s |
| `ultra-fast` | Qwen3-4B Q8_0 (4.3 GB) | one forward, read the letter | 726 | 0.29 | 0.48–0.54 s |

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

![test-3 accuracy](docs/img/test3_accuracy.png)

![accuracy vs latency](docs/img/accuracy_vs_latency.png)

| system | correct /900 | Δ vs Jev (points) | 95 % CI | Jev level |
|---|---:|---:|---|---|
| Jev 1.13 (API) | 841 | — | — | reference |
| routed (combined from the four arms, post hoc)³ | 829 | −1.3 | — | not tested |
| `medium` (4B Q8, think ≤ 1024) | **816** | **−2.8** | [−5.0; −0.6] | no |
| `slow` (8B Q8, think ≤ 128) | 804 | −4.1 | [−6.2; −2.1] | no |
| `fast` (8B Q8) | 782 | −6.6 | [−8.9; −4.3] | no |
| `ultra-fast` (4B Q8) | 726 | −12.8 | [−15.6; −10.0] | no |

Per domain (correct /100; columns ordered by overall accuracy):

![test-3 per domain](docs/img/test3_per_domain.png)

| domain | Jev | `medium` | `slow` | `fast` | `ultra-fast` | routed to |
|---|---:|---:|---:|---:|---:|---|
| factual | 100 | 93 | 96 | 91 | 91 | `slow` |
| numeric | 80 | **98** (beats Jev, +18 [9.7; 27.0]) | 77 | 77 | 72 | `medium` |
| deterministic | 95 | 92 | 93 | 82 | 88 | `slow` |
| sentence | 98 | 90 | 85 | 81 | 82 | `medium` |
| sentiment | 90 | 85 | 88 | 87 | 86 | `slow` |
| subjective_tone | 97 | 87 | 92 | 92 | 52 | `fast` |
| robotic_style | 88 | 84 | 84 | 84 | 76 | `fast` |
| noul_refund | 93 | 91 | 93 | 92 | 84 | `fast` |
| score_urgency | 100 | 96 | 96 | 96 | 95 | `fast` |

³ Sum of the per-domain arm in the "routed to" column below; no CI or verdict because the map was selected on
these same results.

The routed column is, per domain, the best arm on this table with ties going to the faster arm (one exception:
`noul_refund` goes to `fast`, 1 point below `slow`, i.e. within noise and 20× cheaper). It was chosen **post hoc**,
see caveats.

### test-2 (168 cases frozen by a second author; 156 labeled)

| configuration | /156 | /138 without `numeric` |
|---|---:|---:|
| **routed** (measured end to end / from existing predictions) | **145 / 144** | — / 132 |
| Jev 1.13 (API) | 144 | 135 |
| `slow` (8B Q8, think ≤ 128) | 144 | 133 |
| `medium` (4B Q8, think ≤ 1024) | 141 | 129 |
| `fast` (8B Q8) | 132 | 126 |
| `ultra-fast` (4B Q8) | 123 (120 on the GPU engine) | 115 |

![test-2 accuracy](docs/img/test2_accuracy.png)

Per-domain test-2 tables, paired comparisons (routed − Jev: +0.0 [−3.8; 3.8]), earlier models and every caveat:
[docs/RESULTS.md](docs/RESULTS.md).

### Comparison with similar projects

Same sealed sets, same gold, same scoring; sorted by test-3 accuracy. The numbers for the other systems come from
a pre-registered comparison run; its aggregate results (per level, per domain, paired CIs against Jev, `fast` and
`medium`, test-2 subsets) are in [`reports/comparison/results.md`](reports/comparison/results.md) and `results.json`, summarised in
`reports/comparison.json` (which `scripts/make_charts.py` reads). Per-item predictions are not shipped. Latency is
median seconds per question on the stated hardware.

| system | type | test-3 /900 | Δ vs Jev on test-3 [95 % CI] | test-2 /156 | median s/question | hardware |
|---|---|---:|---|---:|---:|---|
| Jev 1.13 by TypeSafe (baseline, via its API) | hosted typed-decision API | 841 | — | 144 | network | remote |
| local-typed-decisions `routed` | per-domain mode | 829 (post hoc) | −1.3 (no CI: post hoc) | 145 (measured) | 0.50 (mean 4.5) | RX 7600 8 GB |
| local-typed-decisions `medium` (Qwen3-4B Q8_0, think ≤ 1024) | local LLM, thinking + readout | 816 | −2.8 [−5.0; −0.6] | 141 | 7.3 | RX 7600 8 GB |
| local-typed-decisions `slow` (Qwen3-8B Q8_0, think ≤ 128) | local LLM, thinking + readout | 804 | −4.1 [−6.2; −2.1] | 144 | 11.6 | RX 7600 8 GB |
| local-typed-decisions `fast` (Qwen3-8B Q8_0) | local LLM, letter readout | 782 | −6.6 [−8.9; −4.3] | 132 | 0.48 | RX 7600 8 GB |
| local-typed-decisions `ultra-fast` (Qwen3-4B Q8_0) | local LLM, letter readout | 726 | −12.8 [−15.6; −10.0] | 123 | 0.29 | RX 7600 8 GB |
| Laya multilingual (~322M, decision heads, third-party) | small model, typed heads | 462 | −42.1 [−45.9; −38.1] | 86 | 0.13 | CPU (Ryzen 5 5600X) |
| DeBERTa-v3-large zero-shot classifier | zero-shot NLI | 374 | −51.9 [−55.3; −48.2] | 59 | 12.8 | CPU (Ryzen 5 5600X) |
| BART-large-MNLI zero-shot classifier | zero-shot NLI | 358 | −53.7 [−57.1; −50.1] | 57 | 0.95 | CPU (Ryzen 5 5600X) |
| mDeBERTa-v3 zero-shot classifier (multilingual) | zero-shot NLI | 353 | −54.2 [−57.7; −50.8] | 55 | 4.2 | CPU (Ryzen 5 5600X) |
| Julia-1 (144M encoder + decision head, third-party) | small encoder, typed API | 322 | −57.7 [−61.3; −54.0] | 62 | 0.06 | CPU (Ryzen 5 5600X) |

The other systems were run by the maintainers on CPU (AMD Ryzen 5 5600X), pre-registered, on the same sealed sets
with the same gold and scoring, each model as released; the Jev, `medium` and `fast` reference values were reproduced
in the same analysis. Every one differs from Jev, `fast` and `medium` with McNemar p < 1e-60 on test-3, with no
errors, unanswered or truncated questions. mDeBERTa-v3 was added because the test-3 texts are mostly Portuguese.
Their latencies are CPU numbers and are not directly comparable with our GPU numbers. **GLiClass was not included:**
it follows the same zero-shot label-matching paradigm as the NLI classifiers, has no `Score` or `Noul` equivalent,
and needs a third-party library.

test-2 numbers for the local modes are from the CPU engine (llama-cpp-python) except `routed`, which was measured
with the released GPU runtime; on the GPU engine `ultra-fast` scores 120. The `routed` test-3 number is combined
from the four modes with a map chosen on test-3, so it is optimistic.

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
- test-3 ships with the 44 GoEmotions texts removed (licence, see [DATA_LICENSES.md](DATA_LICENSES.md)); restore
  them with `python scripts/fetch_goemotions.py` (pinned upstream file, verified against the sealed hash).
- Charts: `python scripts/make_charts.py` (needs matplotlib) regenerates `docs/img/*.png` from `reports/`.
- Reproduce test-3 arms: `python scripts/test3_seal.py --check`, then `scripts/test3_arm.py` per arm and
  `scripts/test3_harness.py analyze` (see the script help). test-2 runners: `scripts/test2_*.py`. Runtime
  measurement: `scripts/runtime_measure.py`. Building the GGUF files from the Qwen safetensors:
  `scripts/build_gguf.sh`, `scripts/build_gguf_qwen3_8b.sh` (llama.cpp commit `9588757`, no imatrix).

## Layout

`typed_decisions/` runtime and server · `conformance/` backend-independent contract oracle · `scripts/` benchmark
harnesses, model download/build · `tests/` · `examples/` configs, requests and the frozen benchmark sets ·
`labels/` human-audit sample · `reports/` aggregate results and manifests · `docs/`.

The benchmark data is mostly Portuguese (pt-PT and pt-BR) plus English items from public datasets (GoEmotions texts
are fetched by script, not shipped); see
[DATA_LICENSES.md](DATA_LICENSES.md).

## License

Code and authored data: [Apache-2.0](LICENSE). Model weights: Qwen3 (Apache-2.0), redistributed as GGUF
quantizations. Public-dataset items: see [DATA_LICENSES.md](DATA_LICENSES.md). "Jev" and "TypeSafe" are names of a
third-party product and company, used here only to identify the comparison baseline.
