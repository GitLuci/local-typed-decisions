# History: what was tried, what failed

A condensed record of the research path, in order. Counts are correct answers on corpus 1 (45 labeled cases, dev+cal
or test-1; one case = 2.2 points) unless stated. Every step had a pre-registered criterion; see
[METHODOLOGY.md](METHODOLOGY.md). Final numbers are in [RESULTS.md](RESULTS.md). Reference: Jev 1.13 by TypeSafe,
used as reference via its API (43/45 on test-1).

## 1. Dedicated decision heads (Laya)

Started with the public **Laya** multilingual model (`convaiinnovations/laya-multilingual`, ~322M, pretrained
decision heads). The typed contract (`Choice` / `Score` / `Noul` with probabilities, never free text) worked
end to end, but quality was far from the reference (68 % agreement with it on a pilot). Kept as the `laya` backend.

## 2. Small Qwen3, fast path

Qwen3-0.6B and 1.7B in FP32, one forward: the state is encoded once and each question is a branch of a shared-prefix
attention tree over the same KV (parity with the normal forward < 1e-4). The decision is read from the logits of the
option letters. test-1: **21** (0.6B) and **27** (1.7B).

## 3. Reading tricks: all within ±2 cases

Intermediate layers, layer ensembles, DoLa-style contrast, a JSON readout, option orderings, persona prompts, a
contrastive 4B − 0.6B readout, and permutation consistency. None moved accuracy by more than ±2 cases. The answer
signal lives only in the last ~10 layers. Conclusion: the bottleneck is what the model represents, not how it is read.
The JSON readout was the cautionary tale: +20 on dev+cal, only +2–4 on test.

## 4. Capacity

Qwen3-4B FP32 fast path: **35** on test-1 (p = 0.021 vs 1.7B). Model size was the only consistent lever — at the cost
of 16 GB RAM and ~7 s per case on a desktop CPU.

## 5. Thinking

Let the model generate a `<think>` block, then read the letter: 0.6B **30**, 1.7B **32**, **4B 41** — but ~183 s
per question for the 4B in FP32 on CPU.
- **The gain belongs to the thinker.** A small reader copies the thinker's conclusion in 98–100 % of cases.
  "Small model thinks, big model reads" does not work.
- **Things that did not help:** stopping the thought early by letter confidence (loses 5 cases at τ = 0.9; confidence
  is high early even when the answer later changes); running only the first 20 or 27 of 36 layers during thinking
  (incoherent text); speculative decoding 0.6B → 4B (exact but only ~1.2× faster on CPU).
- An empty `<think></think>` is not neutral for the 4B (29 vs 35 without the thinking template).

## 6. Speed diagnosis and quantization

CPU generation is **memory-bandwidth bound**: the three Qwen sizes all read ~20–25 GB of weights per second. That
explained why speculative decoding, partial layers and bf16 did not pay, and pointed to weight quantization.

Quantization **without an importance matrix** (no data, no gradient):
- Q8_0: fast-path argmax equal to FP32 on 54/54 (TV 0.012); as a thinker 41/45 and argmax agreement 43/45 →
  **non-inferior**. RAM 16 GB → 4.3 GB.
- Q4_K_M: 53/54 on the fast path; as a thinker 41/45 but agreement 41/45 → **failed by one case**.

## 7. Batching, prompt lookup, budgets, prefix reuse

- Continuous batching (llama-server, B = 4–8) gave 1.6–2.4× throughput with argmax agreement ≥ 93 %.
- Prompt-lookup decoding was **slower**.
- Short thinking budgets (64/128 tokens) only matched the fast path outside the tone domain.
- Reusing the shared prefix KV is exact (bit-identical logits) and gives 1.3–1.4× with ≥ 4 questions per state
  (asymptote ~1.7× because the question + options are longer than these short states).

## 8. test-2 baselines

Fast paths: 0.6B **77**, 1.7B **89**, 4B Q8 **123** of 156 — the 4B called insults `neutral` (tone 7/18). Jev
**144**. Scope decisions, dated before the thinking runs: `numeric` out of the primary criterion; tone in.

## 9. Candidate thinkers

On dev+cal (45), each thinking once, read by itself and by a 0.6B reader:
- LFM2.5-1.2B-Thinking: 35–36 — not enough, and not 2× faster than the 1.7B.
- Qwen3.5-2B: 39–40, but slow (~181 s/case) and often hits the 1024 cap.
- Qwen3-1.7B forced to think (≥ 128 tokens): 40–41 on dev+cal but **120/156** on test-2.
- **Qwen3-4B-Thinking-2507 Q4_K_M:** 42 on dev+cal and **142/156** on test-2 (129/138, same as 4B Q8 think) at
  ~118 s/case versus ~46.
- Qwen3-8B Q8_0 thinking: 43 on dev+cal = its own fast path, with different errors (p = 1.0); decides well with short
  thoughts (sealed budget rule picked 64 tokens).

## 10. Julia-1, a small third-party decision model

[Julia-1](https://huggingface.co/SupersonicLabs/Julia-1) (`SupersonicLabs/Julia-1`, Apache-2.0; a 144M mmBERT-small
encoder with a decision head), run as released with its own code and our question schema: 18/45 on
dev+cal (near chance on three domains; confident and wrong probabilities, ECE 0.46) at ~0.06 s per question. Dropped
before test-2.

## 11. Prompt changes

A revised system prompt aimed at the tone domain (`a_revised`) was selected on dev+cal for the fast path: 38 → 41,
tone 3 → 5 of 9, no losses elsewhere. On the 4B thinker it passed its dev+cal criterion (tone 8 → 9 of 9, one
changed case), but it was never confirmed on test-2, so all reported thinking numbers use the original prompt.

## 12. test-2 verdict

4B Q8 think: **141/156, 129/138** (+14 over its fast path, p = 0.001). Jev − model = 6 on the 138 set against a
pre-registered limit of 5: "Jev level" **missed by one case**. The maintainers decided afterwards to accept it as the
recommended accurate configuration ("medium"); the sealed verdict is unchanged. Five of the six separating cases are
tone items where the thought quotes the rubric's caveat.

8B Q8 fast: 132/156, 126/138 at ~4 s on CPU — below the thinker's criterion (≥ 129), kept as the fast mode.

## 13. GPU and test-3

Moving to llama.cpp b11205 Vulkan on an AMD RX 7600 made the fast paths sub-second and thinking a few seconds. On the
900-question stress test with independent gold: Jev 841; 4B Q8 think 816 (Δ −2.8 [−5.0; −0.6]); 8B short think 804;
8B fast 782; 4B fast 726. **No arm reaches Jev level**; 4B Q8 think beats Jev on `numeric` (+18). The maintainers
judged the quality sufficient for local use and moved the focus to the product: a server with per-domain routing
(chosen post hoc on test-3, validated on test-2, then measured end to end).
