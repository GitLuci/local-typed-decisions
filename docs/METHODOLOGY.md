# Methodology

How every number in [RESULTS.md](RESULTS.md) was produced. The reference system is **Jev 1.13 by TypeSafe, used as
reference via its API**. No model in this project was fine-tuned or trained.

## 1. Rules that apply to every number

- **Pre-registration.** Each experiment had a plan (question, data, criterion, reading of each outcome) committed
  before any result was computed. Changes after the fact were written as dated addenda and labeled as post hoc.
  Verdicts that went against a sealed criterion are reported as such, with any later decision stated separately.
- **Select on dev, confirm once on test.** Prompts, budgets, readers and models were chosen on development/calibration
  data only. Each final configuration was run **once** on a test set.
- **Frozen weights.** Public checkpoints are used as published. Quantization to Q8_0 or Q4_K_M **without an
  importance matrix** (plain rounding of the weights, with no data and no gradient) counts as frozen weights.
- **Provenance.** Model revisions, file hashes (safetensors, GGUF), engine versions, seeds and commands are recorded
  next to each run (`reports/manifests/`, `*.lock.json`, `release-models.json`).
- **Paired statistics.** Every comparison is paired case by case (exact McNemar, paired confidence intervals); no
  claims from unpaired accuracy differences.

## 2. The corpora ladder

| set | size | built by | role |
|---|---|---|---|
| **corpus 1: dev+cal** | 45 labeled (+ known-distribution items) | one author, synthetic (`examples/audit-003-scenarios.jsonl`) | all selection happens here |
| **corpus 1: test-1** | 45 labeled | same | seen once per configuration; later read many times, so only used for paired comparisons |
| **test-2** | 168 cases (156 labeled, 12 known distributions) | a second author, frozen before any model saw it (`examples/test-2-scenarios.jsonl`) | one run per final configuration |
| **test-3** | 900 sealed primary questions (+50 `random`, +90 dev) | independent gold, see §4 (`examples/test-3/`) | one run per arm; stress test |

Corpus 1 has 108 synthetic cases in 6 domains, all `Choice` with 3 options. It showed that dev and test can disagree
badly: a JSON readout gained +20 cases on dev+cal and only +2–4 on test.

The benchmark texts are in **Portuguese** (European and Brazilian) and English. They are data and are kept verbatim.

## 3. test-2

- **Frozen file:** `examples/test-2-scenarios.jsonl`, SHA-256
  `cb9cd5be55e59db86f73b42a39cf48791bc81c8bb62bd27df27593e308b0b828`. No model, local or Jev, saw any state before the
  commit that froze it.
- **Domains:** `factual`, `sentence` (communicative function), `sentiment`, `subjective_tone`, `robotic_style`,
  `numeric` (does a stated total match?) — 18 `Choice` cases each, balanced 6/6/6; `noul_refund` (does the customer
  ask for money back? — `Noul`, 24 cases, 11 yes / 13 no); `score_urgency` (`Score`, 4 levels, 24 cases, 6 per
  level); `random` (dice, coins, cards, urns; 12 cases with an exact target distribution, scored by total variation
  distance only).
- **Labels.** Label A = the corpus author, with a rationale per case. Label B = a second labeler who labeled the 78
  subjective cases (`sentiment`, `subjective_tone`, `robotic_style`, `score_urgency`) blind to label A, before any
  model ran (`examples/test-2-labels-b.jsonl`). Agreement 77/78.
- **Metrics:** accuracy per domain and total on the 156 labeled cases; paired McNemar against Jev; TV on `random`;
  mean absolute level error for `score`; per-class recall for `noul` (a class collapse had been seen before).
- **Subsets.** `numeric` was excluded from the primary criterion by a documented scope decision (single-forward
  arithmetic is not what the fast path is for), giving the 138 set; 120 = without `numeric` and `subjective_tone`.
- **Resolution.** With 156 cases one case is 0.64 points and the minimum credible difference is ~6–8 cases; smaller
  differences are "not distinguishable".
- **Jev once**, with each `Choice` question sent under two letter assignments and averaged; `noul`/`score` as given.
  An earlier replica run is kept as a consistency check (`reports/test-2/jev-replica/`).

## 4. test-3: independent gold

test-2 had Jev at ≥ 97 % in several domains and a single author's labels. test-3 was designed to separate systems
where test-2 could not, with a truth that depends neither on Jev nor on any evaluated model.

### 4.1 Sealed files

Sealed before any model touched a case (`examples/test-3/manifest.json`):

| file | SHA-256 |
|---|---|
| `states.jsonl` | `2de4e0bb3461c42d30928e58702b4aa672606005d95998e8e5bcf2ad7c2011a5` |
| `questions.jsonl` | `ceef675200224d0ec30edffd52f4f90f923bd9d50b752b893b6e73cdb0d85c83` |
| `gold.jsonl` | `5a9dbb80e83228c2bab74b7db8d0e5f62986c37a0f4999c32482da4485db752a` |
| `levels.jsonl` | `96ed571ca6bd218bcdd88bb266b71034ad621ff60d94686a6a4243b20735e914` |

The manifest also records hashes of the source files (generator outputs, author files, labeler files) from the
development history; those source files are not part of this release.

### 4.2 Three sources of gold (`source.kind`)

**(a) Constructed (300 primary + 50 `random`).** A code generator fixes the answer before the text: the factual
relation (explicit, contradicted, rescheduled, absent, negated, quoted, with distractor), the arithmetic (`numeric`),
the rule (`deterministic`: dates, list membership, counting, ordering) or the exact distribution (`random`). The label
is the generator's specification. A verifier **V**, without seeing any model output, confirmed that each text
realizes its specification; cases V did not confirm were fixed or dropped before sealing.

**(b) Public datasets (74 primary).** Banking77 (`noul_refund`, English) and GoEmotions *simplified* (`sentiment`,
English), used only after confirming their licenses (see [../DATA_LICENSES.md](../DATA_LICENSES.md)). The original
dataset labels are **not** used as gold: these items went through the same blind double labeling as (c). GoEmotions
emotions were mapped to polarity with a fixed table (positive / negative / neutral; `confusion`, `curiosity`,
`realization`, `surprise` discard the item), and items with mixed polarity were dropped. Neighboring Banking77 intents
(e.g. refund not showing up) were never assigned a label by inheritance. **Contamination:** these public texts may be
in the pretraining data of both Qwen and Jev, so they form a separate stratum and results are reported with and
without it.

**(c) Authored and judged (526 primary).** For `sentence`, Portuguese `sentiment`, `subjective_tone`,
`robotic_style`, `score_urgency`, Portuguese `noul_refund`:
1. An author (**A1**: sentence, tone, robotic style; **A2**: Portuguese sentiment, refund, urgency, neighboring
   intents) writes the state and records an intended label in a file hidden from the labelers.
2. Two labelers (**R1**, **R2**) label blind to each other, to the authors and to any model output. Each label has a
   confidence (`sure` / `hesitated`) and an optional `ambiguous` flag.
3. If R1 = R2, that is the gold. Otherwise an adjudicator (**J**), who sees both labels and justifications but no
   model output, decides or marks the case `ambiguo_final` (ambiguous_final).
4. `ambiguo_final` cases leave the primary set (4 in total).
5. Author intent does not count for gold; its agreement with gold is reported as a diagnostic.
6. A 60-item human audit sample (stratified by domain and level) was prepared; below 85 % human–gold agreement the
   results would carry a warning. The audit has not been done, so results carry the warning "LLM gold, not
   human-validated".

A1, A2, V, R1, R2 and J are distinct LLM sessions; none ran Jev or a Qwen on the cases, and none saw model outputs
before sealing. The truth is therefore independent of the evaluated systems but **not human**.

Gold provenance across all 1 043 questions: generator 380, R1 = R2 647, adjudicator 16.

### 4.3 Format and size

Jev's format: one context with one or several questions of mixed types. About 35 % of the (a) and (c) states carry
2–3 questions; the unit of measurement is the question, and state-level dependence is handled by a state-level
bootstrap. Power analysis on test-2 disagreement rates (d ≈ 0.08; McNemar, α 0.05, power 0.80) gave a minimum
detectable difference of ~2.6 points at 900 questions, ~4.6 at 300 (one level) and ~7.8 at 100 (one domain), so the
per-domain reading is coarse by design (`reports/test-3/power.json`).

9 labeled domains × 100 sealed test questions + 10 dev each; `random` 50 (TV only). Languages: pt-PT (269 states),
pt-BR (237), en (207). The 90 dev questions were used only to check that runners parse the format; no tuning.

### 4.4 Difficulty levels N1–N3 (fixed before any model ran)

`nivel = max(structural level, labeling level)`; no evaluated model enters the computation.

- **Structural:** count of hard patterns (negation, contrast, sarcasm/irony, quoted question, insufficient
  information, distractor, arithmetic with ≥ 2 steps, request phrased as a question), number of sentences and number of
  questions in the state. N1: 0 patterns, ≤ 2 sentences, 1 question. N2: 1 pattern, or 3–6 sentences, or 2 questions.
  N3: ≥ 2 patterns, or > 6 sentences, or 3 questions. Generators and authors declare patterns; V (for a) and R1 (for c)
  confirm them.
- **Labeling (c only):** N1: R1 = R2, both sure. N2: R1 = R2 with at least one "hesitated". N3: R1 ≠ R2, resolved by J.
- **Quotas per domain:** N1 20 %, N2 40 %, N3 40 %. If labeling pushed a domain more than 10 points off quota, new
  cases were written **before** sealing; 31 items were excluded to restore quotas (listed in the manifest) and one
  item was excluded for privacy. Final sealed test: **N1 170 / N2 363 / N3 367**.

### 4.5 Decision rules (written before results)

- **Primary metric:** accuracy per question (argmax = gold) on the 900 sealed labeled questions, without
  `ambiguo_final`. For each arm, Δ = accuracy(model) − accuracy(Jev), paired per question.
- **CI:** 95 % paired Newcombe (hybrid score) interval, confirmed by a state-level bootstrap (10 000 resamples);
  the **conservative** interval is the union of both.
- **"Jev level" (non-inferiority):** lower bound of Δ > −3 points globally; > −5 per level; > −8 per domain.
- **"Beats Jev":** lower bound of Δ > 0 and one-sided p < 0.05 after **Holm** (over the arms globally; over 3 levels;
  over 9 domains). Exact McNemar is reported alongside.
- **Ceiling:** in a stratum where Jev is ≥ 97 %, "beats" is reported as not measurable.
- **Primary arm:** 8B Q8 fast (the product's fast path); the other arms are secondary. A fourth arm (8B short think
  at 128 tokens) was added by an addendum before any local arm ran.
- **Secondary, no verdict:** mean level error on `score`; `noul` per-class recall (< 0.80 = collapse); TV on
  `random`; accuracy per stratum (a/b/c) and on unanimous cases.

### 4.6 Runs

- One run per arm on an **AMD RX 7600 (8 GB)** with **llama.cpp b11205 (Vulkan)**, cold prefill, batch 1, seed
  `20260927 + question index`. The 8B uses 30 of 36 layers on GPU; the 4B all layers.
- Jev once over the sealed test via its API.
- Earlier experiments (corpus 1, test-2) ran on CPU: a 4-vCPU Xeon VM with **llama-cpp-python 0.3.35** (llama.cpp
  upstream `9588757`) or PyTorch/transformers for FP32, and a 6-core desktop CPU for some FP32 runs.

## 5. Decision procedure of the models

1. **Prompt:** fixed English system prompt (`typed_decisions/qwen.py::SYSTEM`) + state + question + options labeled
   with letters (`Noul`: A = no, B = yes; `Score`: A = level 0), Qwen3 chat template.
2. **Fast path:** thinking disabled, one forward; the decision is the logits of the option-letter tokens at the answer
   position, normalized over the valid letters only.
3. **Thinking path:** sample a `<think>` block (temperature 0.6, top_p 0.95, top_k 20, fixed seed per case) up to the
   budget (1024 for 4B, 128 for 8B short); stop at `</think>` or force the close; then the same letter readout by the
   same model.
4. **Probabilities:** softmax over the option letters; confidence = 1 − normalized Shannon entropy; not calibrated.

## 6. Re-verification and reproducibility

- **Independent re-analysis:** the test-3 analysis script recomputes every table from the per-question predictions
  and the sealed gold; the seal check (`scripts/test3_seal.py --check`) verifies the four sealed files against the
  manifest hashes.
- **Clean-machine reproduction:** on a fresh clone in a clean Linux VM (no models, no GGUF, no cache), the documented
  path was followed end to end — download Qwen3-4B at the pinned revision, build the Q8_0 GGUF with
  `scripts/build_gguf.sh`, run the 4B Q8 fast path on the 168 test-2 cases — and reproduced the same GGUF byte for byte
  (identical hash) and the same summary (123/156) with all 168 letter-logit vectors identical to the committed ones. The thinking path was not covered (sampling is only
  bit-exact on the same machine/build).
- **Reproducibility run:** the released runtime was re-measured end to end on test-2 with its own pre-registration.
  Prompts matched the CPU harness token for token on all 168 cases; argmax agreement with the old CPU predictions was
  163/168 (4B) and 166/168 (8B). An independent harness on the same GPU build agreed with the runtime 168/168,
  attributing the differences to the engine change (CPU llama-cpp-python → GPU llama.cpp b11205), i.e. engine drift of
  ~3 %, not to the runtime. From then on the GPU engine is the reference.
- **Quantization parity:** Q8_0 matched FP32 argmax on 54/54 fast-path cases (TV 0.012) and was non-inferior as a
  thinker (41/45 = FP32, argmax agreement 43/45); Q4_K_M failed the agreement criterion by one case.

## 7. Glossary of sealed-file field names

The sealed test-3 files keep their original Portuguese field names (changing them would change the hashes):

| field | meaning |
|---|---|
| `nivel` | level (N1–N3) |
| `nivel_estrutural`, `nivel_rotulagem` | structural level, labeling level |
| `estrato` | stratum (a = constructed, b = dataset, c = judged) |
| `primario` | counts in the primary set |
| `ambiguo_final` | ambiguous after adjudication (excluded from primary) |
| `padroes_usados`, `padroes_de` | hard patterns used; who confirmed them |
| `nota` | note |
| `target_exacto` | exact target distribution as fractions |
| `autor` | author (A1/A2) |
| `gerador` | generator script |
| `n_frases` | number of sentences |
