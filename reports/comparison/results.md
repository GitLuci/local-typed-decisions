# Comparison with similar systems (test-3 and test-2)

Generated from `results.json` in this folder. Comparison of the local modes with similar public systems on the sealed test-3 (900 primary questions) and test-2 (156 labeled). Pre-registered before running. CPU (AMD Ryzen 5 5600X), 4 threads, one question per call, cold. Every model used as released; nothing trained or tuned. Unsupported or failing questions would count as wrong (none occurred). Statistics as in the test-3 harness: conservative 95 % CI = union of paired Newcombe and state-level bootstrap; exact McNemar. References: Jev 1.13 by TypeSafe (via its API) and our modes fast (8B Q8) and medium (4B Q8 think), whose values were reproduced in the same analysis. Per-item predictions are not shipped.

## test-3: 900 primary questions

| system | correct /900 | vs Jev (841): Δ [95 % CI] | vs fast (782): Δ [CI] | vs medium (816): Δ [CI] | N1 · N2 · N3 | median s/question (CPU) |
|---|---:|---|---|---|---|---:|
| Laya multilingual (convaiinnovations/laya-multilingual @ e4e9ddf2) | 462 | −42.1 [−45.9; −38.1] | −35.6 [−39.6; −31.5] | −39.3 [−43.4; −35.2] | 105/170 · 186/363 · 171/367 | 0.13 |
| DeBERTa-v3-large zero-shot (MoritzLaurer/deberta-v3-large-zeroshot-v2.0 @ cf44676c28ba) | 374 | −51.9 [−55.3; −48.2] | −45.3 [−49.0; −41.5] | −49.1 [−52.7; −45.3] | 49/170 · 161/363 · 164/367 | 12.76 |
| BART-large-MNLI zero-shot (facebook/bart-large-mnli @ d7645e127eaf) | 358 | −53.7 [−57.1; −50.1] | −47.1 [−50.8; −43.4] | −50.9 [−54.6; −47.1] | 56/170 · 154/363 · 148/367 | 0.95 |
| mDeBERTa-v3 zero-shot, multilingual (MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 @ b5113eb38ab6) | 353 | −54.2 [−57.7; −50.8] | −47.7 [−51.7; −43.7] | −51.4 [−55.0; −47.8] | 60/170 · 150/363 · 143/367 | 4.16 |
| Julia-1 (SupersonicLabs/Julia-1 @ a85b127321, 144M) | 322 | −57.7 [−61.3; −54.0] | −51.1 [−54.8; −47.4] | −54.9 [−58.7; −51.1] | 46/170 · 136/363 · 140/367 | 0.06 |

Every difference to Jev, fast and medium has exact McNemar p ≤ 2.5e-60 (all below 1e−60). Errors, unanswered and truncated questions: 0 in total.

Per domain (correct /100):

| domain | Laya multilingual | DeBERTa-v3-large zero-shot | BART-large-MNLI zero-shot | mDeBERTa-v3 zero-shot, multilingual | Julia-1 |
|---|---:|---:|---:|---:|---:|
| factual | 44 | 26 | 62 | 52 | 40 |
| numeric | 33 | 33 | 34 | 38 | 30 |
| deterministic | 54 | 16 | 49 | 48 | 33 |
| sentence | 68 | 52 | 36 | 38 | 39 |
| sentiment | 72 | 64 | 44 | 46 | 43 |
| subjective_tone | 67 | 42 | 30 | 33 | 32 |
| robotic_style | 35 | 30 | 30 | 34 | 30 |
| noul_refund | 48 | 40 | 38 | 38 | 52 |
| score_urgency | 41 | 71 | 35 | 26 | 23 |

## test-2: 156 labeled cases

References: Jev 144, fast 132.

| system | /156 | /138 (no numeric) | /120 (no numeric, no tone) | vs Jev: only Jev / only model, McNemar p |
|---|---:|---:|---:|---|
| Laya multilingual (convaiinnovations/laya-multilingual @ e4e9ddf2) | 86 | 80 | 67 | 60 / 2, p = 8e-16 |
| Julia-1 (SupersonicLabs/Julia-1 @ a85b127321, 144M) | 62 | 56 | 47 | 87 / 5, p = 2e-20 |
| DeBERTa-v3-large zero-shot (MoritzLaurer/deberta-v3-large-zeroshot-v2.0 @ cf44676c28ba) | 59 | 53 | 47 | 91 / 6, p = 1e-20 |
| BART-large-MNLI zero-shot (facebook/bart-large-mnli @ d7645e127eaf) | 57 | 51 | 45 | 93 / 6, p = 4e-21 |
| mDeBERTa-v3 zero-shot, multilingual (MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 @ b5113eb38ab6) | 55 | 49 | 43 | 90 / 1, p = 7e-26 |

## Reading

- No similar system comes close to our modes or to Jev on this schema. The best, Laya multilingual, reaches ~51 % on test-3 and 86/156 on test-2, below even our weakest mode (`ultra-fast`, 726/900).
- The zero-shot NLI classifiers (the classic "classify against options" baseline) score 39–42 % on test-3 and about a third on test-2; the best, DeBERTa-v3-large (374/900), is also the slowest (12.8 s per question on CPU). They cannot use the instructions ("use only the facts in the text", tone and style rubrics) nor treat Score/Noul as decisions. The multilingual NLI model does not help over the English ones.
- Julia-1 is very fast (~0.06 s on CPU) but near chance in most domains.
- Laya and Julia-1 on CPU are faster than our `ultra-fast` mode on CPU, but `ultra-fast` and `fast` on the GPU (~0.5–0.6 s) are already under a second, with +33 to +50 points of accuracy.

Limits: sealed corpus labels; models used as released, with no prompt tuning to this data; CPU times on one machine, cold.

Models (pinned revisions): Laya `convaiinnovations/laya-multilingual` (Apache-2.0); DeBERTa-v3-large `MoritzLaurer/deberta-v3-large-zeroshot-v2.0` (MIT); BART `facebook/bart-large-mnli` (MIT); mDeBERTa `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` (MIT); Julia-1 `SupersonicLabs/Julia-1` (Apache-2.0). The comparison runner is not included: it depends on evaluation code that is not part of this repository. NLI arms score each option as the hypothesis `<instructions> + newline + <option description>` against the state as premise and softmax the entailment logits over the options (Noul options: No / Yes descriptions).
