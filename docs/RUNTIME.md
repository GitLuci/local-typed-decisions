# Runtime: local llama-server backend

CLI and HTTP API with four decision profiles over two Q8_0 GGUF files, plus a `routed` mode that picks the profile
of each question from an explicit domain map in the JSON configuration. The runtime never downloads or trains
models, and questions are always answered independently of each other.

| Mode | Model | Decision | Example port |
| --- | --- | --- | --- |
| `ultra-fast` | Qwen3-4B Q8_0 | no thinking; read the option-letter probabilities | 8791 |
| `fast` | Qwen3-8B Q8_0 | no thinking; read the option-letter probabilities | 8790 |
| `medium` | Qwen3-4B Q8_0 | think up to 1024 tokens, then read the letters | 8791 |
| `slow` | Qwen3-8B Q8_0 | think up to 128 tokens, then read the letters at that point | 8790 |
| `routed` | per domain | explicit domain → mode map in the config | both, one at a time |

In thinking modes a natural `</think>` before the budget is respected; if the budget runs out, the thought is closed
(forced) before the readout. `slow` is the recipe of the "8B short think" arm of test-3.

## Configure and run

Example configs: `examples/llama-{ultra-fast,fast,medium,slow,routed}.json`. They expect the
**llama.cpp b11205 Vulkan** `llama-server` executable at `tools/llama-b11205-vulkan/llama-server(.exe)`, the GGUF
files under `models/qwen3-{4b,8b}-gguf/` and the tokenizers under `models/qwen3-{4b,8b}/` (all installed by
`scripts/fetch_models.py`). Relative paths are resolved against the JSON file. If your binary lives elsewhere,
edit `servers.<model>.launch.executable`.

Each `servers` entry has `url`, `tokenizer_path` and optionally `launch` (`executable`, `model_path`, `gpu_layers`,
`threads`, `startup_timeout` in seconds). The launcher uses an argument list (no shell) and a hidden window on
Windows:

```text
llama-server -m <GGUF> -ngl <layers> -c 2048 -t 4 --parallel 1 --host 127.0.0.1 --port <port>
```

The 8B example uses `-ngl 30` (fits an 8 GB card); the 4B uses `-ngl 99`. `gpu_layers: 0` runs on CPU (slower; not
benchmarked for this runtime). With partial offload (fewer than 37 layers for these 36-block models plus output) the
launcher adds `--load-mode none` (the b11205 spelling); see [OPEN_WORK.md](OPEN_WORK.md) for the full-offload
follow-up.

```bash
# Routed API (the mode comes from the config)
python -m typed_decisions serve --config examples/llama-routed.json --port 8000
# A fixed profile on top of the same config (requests without "domains")
python -m typed_decisions serve --config examples/llama-routed.json --mode ultra-fast --port 8000
# One request from the CLI
python -m typed_decisions examples/routed-request.json --backend llama-server --config examples/llama-routed.json
```

Start-up and `/health` load no weights. For each request the domains, schema and token budget of **all** questions
are validated first; only then is the needed server started. Tokenizers are loaded locally for that preparation.
At most **one owned llama-server process** is resident: the previous one is stopped before another model is started.
Modes on the same GGUF reuse the same server. The last server stays up until a model switch or until the API/CLI
exits (there is no idle timer). Questions are grouped by model inside a request to avoid repeated loads; the answer
keeps the original ID order.

The pool only terminates PIDs it created. A busy port is an error (it never adopts or stops a foreign server). A
start-up timeout or failure cleans up the owned process. On Windows each owned child is attached to a
non-inheritable Job Object with `KILL_ON_JOB_CLOSE`, so the OS also kills it if the parent exits abruptly (there is a
short window between `Popen` and the assignment that is not covered). Outside Windows, abrupt exits require external
process management. For servers started and managed externally, omit `launch`; the runtime will then neither start
nor stop them.

The legacy single-profile config (`mode`, `url`, `tokenizer_path`) is still supported. `n_probs`, `timeout` and
`seed` are common options.

## Per-question domains (routed mode)

In `routed` mode the request envelope accepts `domains`, a map from question ID to domain. The primitive definitions
do not change. There is **no domain classifier** yet (see [OPEN_WORK.md](OPEN_WORK.md)); the caller supplies the
domain. Example: `examples/routed-request.json`.

Shipped map (`examples/llama-routed.json`):

| Domains | Mode |
| --- | --- |
| numeric, sentence | medium |
| factual, deterministic, sentiment | slow |
| subjective_tone, robotic_style, noul_refund, score_urgency | fast |
| random | fast |

The map was chosen **post hoc** on test-3 (best arm per domain, ties to the faster one) and then checked on test-2
(see [RESULTS.md](RESULTS.md)). `random` (known probability distributions, scored by total-variation distance, not
accuracy) was added after the runtime measurement found the route missing; it uses `fast` with no accuracy claim.

Unknown domain, extra ID or a question without a domain → HTTP 422 before any inference or server start. A
`default_domain` may be declared in the config (the shipped examples do not). Fixed profiles reject `domains` over
HTTP. `metadata.routing` records the domain, mode and model used for each question; `metadata.modes` keeps the
per-mode details. `usage` sums all completions.

## Readout, errors and independence

- The production `SYSTEM` prompt (`typed_decisions/qwen.py`) is used unchanged. The seed is derived from the prompt
  and the base seed, independent of question ID, position and other questions (it differs from the per-index seeds
  of the benchmark scripts; no stochastic parity with those runs is claimed).
- Serial execution: one completion per question without thinking, two with thinking. `cache_prompt: false`, no
  history, no automatic retry.
- Option letters are identified by token ID in the `n_probs` list. Defaults: `n_probs=20` without thinking, `256`
  with thinking. An explicit `n_probs` in the config overrides both. The parser accepts `completion_probabilities` or
  `probs`, with `top_logprobs`/`top_probs`/`probs` and `logprob`/`prob`.
- Probabilities are renormalised over the options only. Confidence = `1 − normalized entropy`, **not calibrated**.
- `/props` checks the GGUF name and a context of at least 2048 per slot (it does not attest hashes or offload — verify
  the manifests at install time). `/tokenize` must reproduce the local token IDs.
- Budget overflow: 422, never truncated. Protocol/model/tokenizer failure, reported truncation or busy port: 502.
  Start-up/HTTP timeout: 504. These abort the request.

### Missing option probabilities

llama-server b11205 returns only the top-N tokens; it cannot be asked for the probabilities of arbitrary token IDs.
Raising N to 256 in thinking modes is an engineering mitigation for the case seen in the runtime measurement (1 of
168 questions in `slow` mode); it does not guarantee coverage. If an option letter is still missing, the other
questions are answered and the response (HTTP 200) carries:

```json
"errors": {
  "q": {
    "code": "missing_option_probabilities",
    "message": "Option probabilities are missing; increase n_probs in config.",
    "missing_options": ["right"],
    "n_probs": 256
  }
}
```

`missing_options` holds original Choice IDs, Score level indices as text, or `false`/`true` for Noul. Every requested
ID appears exactly once in `answers` or `errors`. Clients must check `errors` even on HTTP 200. No probability mass,
confidence or answer is invented and inference is not repeated.

## Verification

Unit tests use simulated HTTP and processes: the four profiles, mixed domains, configurable map, independence,
preflight before start-up, reuse/switch/close, busy port, timeout, Vulkan flags, shipped configs, top-N handling,
per-question errors and the opt-in contract extension.

```bash
python -m pytest tests/test_llama_probability_errors.py tests/test_llama_routing.py tests/test_llama_server.py \
  tests/test_contract_conformance.py tests/test_runtime_validation.py tests/test_contract.py tests/test_qwen_contract.py \
  -q --td-backend-factory=test_llama_routing:make_routed_backend --td-generates-tokens
```

Optional tokenizer-only test (no weights): set `TD_TOKENIZER_PATH` to an installed Qwen3 tokenizer directory and run
`tests/test_llama_tokenizer.py`.

The real-hardware measurement of this runtime (latency, model switches, RAM/VRAM) is in
[RESULTS.md](RESULTS.md#runtime-measurement) and `reports/runtime-measurement/`.
