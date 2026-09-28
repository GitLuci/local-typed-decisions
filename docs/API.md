# API contract and local conformance suite (v1)

The HTTP API follows the request/response shape of the typed primitives (`Choice`, `Score`, `Noul`) of
Jev 1.13 by TypeSafe, which this project uses only as a comparison baseline. The public reference pages are
[API](https://docs.typesafe.ai/api), [Choice](https://docs.typesafe.ai/primitives/choice),
[Score](https://docs.typesafe.ai/primitives/score) and [Noul](https://docs.typesafe.ai/primitives/noul).

The contract below is an **explicit local profile**. It does not claim equality of probabilities, calibration,
accuracy, speed, internal architecture or every limit of the remote service. Unknown fields are rejected by the
local profile; that is a local policy, not observed remote behaviour. Keys are case-sensitive, preserve Unicode and
do not depend on JSON property order. This project is not affiliated with TypeSafe.

## Request

HTTP: `POST /v1/systemone`, UTF-8 JSON (the path mirrors the reference API so that clients can switch backends).
Python backend call: `predict(state, questions) -> dict`.

| Field | Local profile v1 rule |
| --- | --- |
| `state` | Required; JSON text, object or array; may be empty. Not null, bool or number at the root. |
| `questions` | Non-empty object mapping a non-empty ID to a question; 1–32 entries. Question IDs are for routing and are never sent to the model. |
| `model` | Optional; non-empty string; the transport accepts `local` or the installed model ID. |
| `question.type` | Lower-case literal `choice`, `score` or `noul`. |
| `question.instructions` | Non-empty text, object or array. |
| `question.criteria` | Depends on the primitive (below). |

Every nested value must be finite JSON (no NaN/Infinity). Booleans are legitimate data inside the state. The backend
must not mutate the request.

- **Choice:** `criteria` maps 2–255 non-empty option IDs to descriptions (text/object/array/null). Option names carry
  meaning. A backend may declare a smaller capacity.
- **Score:** `criteria` is a list of 2–10 descriptions (text/object/array) for levels 0 … K−1. Never rescale to 0–1
  or infer numbers from the words.
- **Noul:** criteria are optional; the public form is an object with optional `true`/`false` descriptions. The local
  profile also accepts null and legacy structured clarifications (`Profile(noul_clarifications=False)` disables this).

## Response

Envelope: `model` (non-empty string identifying the actual executor), `answers` (exactly the requested IDs),
`usage` (non-negative integers `input_tokens`/`output_tokens`), optional `metadata` (local extension).

| Primitive | Fields and invariants |
| --- | --- |
| Choice | `type`, `choice`, `probabilities`, `confidence`; the chosen option belongs to the criteria and maximises the probability (any winner of an exact tie is valid). |
| Score | `type`, `score`, `probabilities`, `legend`, `confidence`; probability and legend keys are the strings `"0"`…`"K-1"`; `score = sum(i * p[i])` in [0, K−1]. |
| Noul | Only `type` and `noul`: a number in [0, 1] = P(yes), no separate confidence. |

Each probability is a finite number in [0, 1]. Choice/Score list **all** options/levels (zeros included) and no
extra keys. Sums must be 1 within an absolute tolerance of **1e-6**; the Score expectation also within 1e-6. The
validator reports deviations; it never renormalises.

`confidence` is in [0, 1]. The local implementation uses `1 − normalized Shannon entropy` of the option
distribution. It is a concentration measure, **not** P(correct), and it is **not calibrated**.

Structured legend: the local profile preserves each original JSON level description exactly (a declared local
extension). `logits` may exist in the native Python result for diagnostics but is removed from the HTTP envelope.
Backends that generate tokens (the llama-server adapter uses `n_predict=1`, plus the thought tokens in thinking
modes) report positive `output_tokens`; only `Profile(fast_path=True)` requires zero.

### Local extensions

- **Routing (`mode: routed` only):** the envelope accepts `domains`, a map from question ID to domain. `routes` and an
  optional `default_domain` live in the server configuration. Extra IDs or unknown/missing domains without an
  explicit default are rejected with 422 before any inference. See [RUNTIME.md](RUNTIME.md).
- **Per-question errors (llama-server runtime):** if the top-N list returned by llama-server lacks an option token,
  the response keeps the other answers (HTTP 200) and adds `errors[question_id]` with
  `code: "missing_option_probabilities"`. That question is omitted from `answers`; the two maps together cover the
  request exactly. `usage` still counts the failed questions. The base contract rejects this field;
  `Profile(question_errors=True)` enables it in the oracle.

## Errors and limits

Local HTTP profile: 422 for invalid JSON, invalid envelope/question, unavailable model, or capacity violations; the
JSON body has a non-empty `detail`. A body larger than **256000 bytes** (UTF-8) returns 413. Errors never become an
empty HTTP 200. The llama-server runtime also returns 502 (protocol/model/tokenizer failure, busy port) and 504
(start-up/HTTP timeout).

| Limit | Scope |
| --- | --- |
| 32 questions; 256000 bytes | Local policy. |
| Choice up to 255; Score up to 10 | Primitive documentation; a backend may declare less. |
| Laya backend: 20 options; 1024 tokens/path | Historical local checkpoint. |
| Qwen FP32 tree backend: 2048 tokens/path; 4096 packed | Historical local configuration. |
| llama-server runtime: prompt + thinking budget ≤ 2048 tokens | Rejected with 422, never truncated. |

## Fixtures, oracle and how to run

`tests/fixtures/contract/cases.json` holds four hand-written synthetic request/response pairs (mixed types, Unicode,
structured data, ties, zeros, fractional expectation, local extensions). They are never model outputs.
`conformance/contract.py` imports no inference code; any Python backend can be checked with it. `exercise_backend`
checks repetition, permutation of IDs/order, isolated questions and a query after a different state, and verifies
that the request is not mutated (distribution tolerance 1e-5, configurable). These checks detect some leakage between
questions; they do not prove the attention mask for every input.

```bash
python -m pytest tests/test_contract_conformance.py tests/test_contract.py -q
```

To check your own backend, provide `my_adapter:make_backend` returning an object with `predict(state, questions)`
and optionally `close()`:

```bash
python -m pytest tests/test_contract_conformance.py -k selected_backend \
  --td-backend-factory=my_adapter:make_backend --td-max-options=255 -q
```

The test network guard blocks non-loopback connections; the suite never calls the remote reference API.
