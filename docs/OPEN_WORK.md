# Open work

Fronts that are started but not finished, and known problems. Contributions are welcome on any of them; please follow
the pre-registration rule of [METHODOLOGY.md](METHODOLOGY.md) (plan committed before results, selection on
development data, one confirmation run on a sealed set).

## 1. Vision: the same typed schema over images, with small VLMs

Goal: `Choice` / `Score` / `Noul` questions about one or more images (plus optional text state), answered by reading
the option-letter probabilities of a small vision-language model, with no fine-tuning — the same recipe as the text
fast path.

What exists (not shipped in this repository yet; the harness will be added once the benchmark is sealed):
- A development benchmark of 150 questions over public image datasets (VQAv2 yes/no and counting, A-OKVQA
  multiple choice, ScienceQA-IMG), with images referenced by dataset/split/offset and hashed, and several questions per
  image. ScienceQA's original license is non-commercial and must be confirmed or replaced before any non-academic
  use.
- Findings from a 20-case smoke test on CPU:
  - The readout must be placed after an explicit `Answer:` prefix; without it SmolVLM puts only ~14 % of the mass on
    the letters and falls below chance. With it the mass on the letters is ~99.9 %.
  - Extra questions on the same image are almost free when the image prefix is cached (suffix ≈ 1 % of the prefix
    cost; logit parity 5e-5).
  - The cost is the image tokens: ~830 image tokens → ~10 s per question for a 500M model on a 4-vCPU VM; 64 tokens
    → ~1 s, losing 1 case in 20.
  - SmolVLM-256M/500M do not beat a zero-shot SigLIP2-base floor (10/20) on this smoke; the serious candidate is
    larger.
- **Qwen3-VL-2B-Instruct** (Apache-2.0) on the 61 development questions: **51/61 with the image capped at 256
  visual tokens, 50/61 at 1024 tokens**. The higher resolution buys nothing on this set, so the cheaper setting is
  the current choice.

Next steps: seal a test split (with multi-image contexts), run Qwen3-VL-2B once on it with the 256-token setting,
compare against the SigLIP2 floor, and measure latency on the RX 7600 through llama.cpp's multimodal server path.
Calibration will be an issue: wrong answers often get p_max > 0.95.

## 2. Speed

The accurate modes are slow: `medium` 7.3 s and `slow` 11.6 s median per question on an RX 7600, against < 0.5 s for
the fast modes. Priority is lower latency **without trading accuracy for time** (no ensembles, no longer thinking).
Directions, none measured end to end yet:
- **Prefix reuse across questions on the same state.** On CPU, reusing the KV cache of the shared prefix is exact
  (bit-identical letter logits) and gives 1.3–1.4× with ≥ 4 questions per state (asymptote ~1.7×). The runtime
  currently uses `cache_prompt: false` and answers questions serially.
- **Continuous batching** (`llama-server --parallel B`): 1.6–2.4× throughput with ≥ 93 % agreement in earlier CPU
  tests; not yet wired into the runtime (which uses `--parallel 1` and a fixed per-question seed).
- **Full GPU offload of the 8B:** at 8 GB of VRAM only 30/36 layers fit with a 2048 context. A card with ≥ 10 GB, or
  a smaller KV cache, would remove the CPU part of the 8B modes.
- Shorter prompts (the option list is ~55 % of the tokens on short states) and a shorter system prompt.
- Prompt lookup decoding was *slower* in tests; speculative decoding with a 0.6B draft did not help on CPU.

## 3. `--load-mode none` with full offload

The launcher adds `--load-mode none` only with partial offload (fewer than 37 GPU layers). With full offload the
4B GGUF stays memory-mapped in RAM as well, which is why `ultra-fast`/`medium` show the highest RAM peak in the serve
smoke (5.0 GB) although the whole model is on the card. With partial offload the flag already makes a large
difference for the 8B (≈ 2.6 GB of RAM instead of ≈ 10.4 GB). Applying `--load-mode none` in all cases should cut RAM; it
needs a measurement (RAM, load time, identical answers) before becoming the default.

## 4. Domain inference for the routed mode

`routed` requires the caller to declare each question's domain (`domains` map); unknown or missing domains are
rejected. For general use the runtime needs either:
- a cheap domain classifier (e.g. the `ultra-fast` model answering a `Choice` over the domain list, or rules on the
  question schema), measured for routing accuracy and for the end-to-end accuracy it induces; or
- a `default_domain` policy documented per deployment.

Mixed-domain traffic also causes more model switches (5–12 s each on the RX 7600, not measured under shuffled
traffic); grouping by model inside a request helps only within a request.

## 5. A new sealed set to confirm the router

The routed map was chosen post hoc on test-3 and confirmed only on test-2, which had been seen before. A new sealed
set built with the test-3 protocol (independent gold, levels, quotas) is needed to confirm routed vs Jev and vs the
best single arm. Ideally it also includes the missing **human audit** of the authored-item gold (the 60-item sample
in `labels/test-3/audit-sample.jsonl` is ready to be labeled).

## 6. Known problems

- **Missing option probabilities.** llama-server b11205 returns only the top-N tokens. After a thought an option
  letter can fall outside the top-20 (seen once in 168); the default is now N = 256 in thinking modes and the
  question is reported in `errors` rather than failing the request, but coverage is not guaranteed.
- **Uncalibrated probabilities.** `confidence` is a concentration measure, not P(correct).
- **Tone.** The 4B fast path calls insults `neutral` (52/100 on tone in test-3); a revised system prompt helped only
  on development data (3→5 of 9) and was not confirmed on a sealed set.
- **Engine drift.** CPU and GPU builds of llama.cpp differ on ~3 % of fast-path argmax; numbers are only comparable
  within one engine.
- **Seeds.** The runtime derives the thinking seed from the prompt, not from the item index used by the benchmark
  harnesses; per-item stochastic parity with the published runs is not claimed.
