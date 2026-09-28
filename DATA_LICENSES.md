# Data licenses and attribution

This repository ships evaluation data. This file states where each part comes from and under what terms it is
redistributed.

## 1. Authored and constructed items — Apache-2.0 (this repository's license)

- `examples/audit-003-scenarios.jsonl` (corpus 1), `examples/test-2-scenarios.jsonl` (test-2),
  `examples/test-2-labels-b.jsonl`, `examples/pilot.jsonl`, `examples/audit-002.jsonl`, the example requests in
  `examples/*.json`, and in `examples/test-3/` every state whose `source.kind` is `constructed` or `judged`, plus all
  questions, gold labels and levels.
- These texts are synthetic: written for this project by code generators or authoring sessions. They contain no real
  personal data; any item with a real, non-public person's name was discarded before sealing.
- `labels/test-3/audit-sample.jsonl` is a 60-item sample of the authored items, prepared for a human audit.

## 2. Banking77 — CC BY 4.0

- **Source:** `PolyAI/banking77` (test split), https://huggingface.co/datasets/PolyAI/banking77
- **License:** Creative Commons Attribution 4.0 International — stated in the `LICENSE` file of the original release
  (https://github.com/PolyAI-LDN/task-specific-datasets, verified 2026-09-28) and on the dataset card. CC BY 4.0
  allows redistribution with attribution, so the 43 texts are shipped.
- **Used in:** 43 states in `examples/test-3/states.jsonl` (`state_id` prefix `s3-b77-`, `noul_refund` domain),
  each carrying `source.dataset`, `source.license` and `source.orig_id` (row in the original `test.csv`).
- **Changes:** texts are used verbatim; the question wrapped around them and the gold label are ours. The original
  intent labels are **not** used as gold (items were labeled blind by two labelers and adjudicated).
- **Attribution:** Iñigo Casanueva, Tadas Temčinas, Daniela Gerz, Matthew Henderson, Ivan Vulić. "Efficient Intent
  Detection with Dual Sentence Encoders." Proceedings of the 2nd Workshop on NLP for ConvAI, ACL 2020.
  arXiv:2003.04807.

## 3. GoEmotions (simplified) — texts NOT shipped

- **Source:** `google-research-datasets/go_emotions`, `simplified` configuration (test split),
  https://huggingface.co/datasets/google-research-datasets/go_emotions ; original release in
  https://github.com/google-research/google-research/tree/master/goemotions.
- **Why the texts are not shipped:** the only licence statement is on the Hugging Face card ("The GitHub repository
  which houses this dataset has an Apache License 2.0"). The upstream README states no licence or terms for the data
  itself, and the repository's Apache-2.0 licence is written for software. The texts are Reddit comments. Since the
  redistribution terms for the texts are not explicit, this repository ships only ids and attribution (checked
  2026-09-28).
- **What is shipped:** the 44 states in `examples/test-3/states.jsonl` with `state_id` prefix `s3-goe-`
  (`sentiment` domain) keep `source.dataset` and `source.orig_id` (the dataset's comment id) but have `text: null`;
  2 items of `labels/test-3/audit-sample.jsonl` likewise have `state_text: null` and a `state_source` reference.
  Questions, gold labels and levels (ours) are shipped.
- **Restoring the texts:** `python scripts/fetch_goemotions.py` downloads `goemotions/data/test.tsv` from the upstream
  repository at pinned commit `2adf640a14f11025ae5a9d0ec493b78530d276d3`, checks its SHA-256, fills each text by id
  (surrounding whitespace stripped, as when the set was built) and verifies that the restored `states.jsonl` is
  byte-identical to the sealed file (hash `2de4e0bb…` in `examples/test-3/manifest.json`). The shipped, redacted
  file has its own hash in `manifest["shipped_redacted"]`; `scripts/test3_seal.py --check` accepts both, and the
  model harnesses refuse to run until the texts are restored.
- **Privacy:** only the *simplified* configuration was used, in which person names are masked as `[NAME]` by the
  dataset authors; the *raw* configuration (with Reddit usernames) was not used. One further item was excluded before
  sealing for privacy reasons.
- **Changes:** texts used verbatim (whitespace-stripped); the 27 emotion labels were only used to pre-select items;
  gold comes from blind double labeling, not from the dataset.
- **Content warning:** as the dataset authors note, GoEmotions contains biases and potentially offensive content.
- **Attribution:** Dorottya Demszky, Dana Movshovitz-Attias, Jeongwoo Ko, Alan Cowen, Gaurav Nemade and Sujith Ravi.
  2020. "GoEmotions: A Dataset of Fine-Grained Emotions." In *Proceedings of the 58th Annual Meeting of the
  Association for Computational Linguistics (ACL 2020)*, pages 4040–4054. ACL Anthology 2020.acl-main.372; arXiv:2005.00547.

Of these 87 dataset-derived states (43 shipped, 44 restored by script), **74 questions** are in the primary sealed test (38 Banking77, 36 GoEmotions);
the rest are dev items or excluded (quota, `ambiguo_final`). Because both datasets are public, they may be in the
pretraining data of any evaluated system; results report them as a separate stratum.

## 4. Baseline outputs — withheld

Per-item outputs of the reference system (Jev 1.13 by TypeSafe, used as reference via its API) are **not**
distributed, because they come from a commercial API. Only aggregate numbers (accuracy, per-domain counts, paired
statistics) are included in `reports/`. Scripts that need per-item baseline outputs accept a path to predictions you
generate yourself with your own API credential.

## 5. Model weights

The GGUF files (Qwen3-4B and Qwen3-8B, Q8_0 and Q4_K_M, quantized without an importance matrix) are derivatives of
`Qwen/Qwen3-4B` and `Qwen/Qwen3-8B`, released by the Qwen team under Apache-2.0, and are distributed on the Hugging
Face Hub under the same license. Tokenizer files are downloaded from the official Qwen repositories at pinned
revisions.
