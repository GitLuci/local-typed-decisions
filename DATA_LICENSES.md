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

## 6. test-4 sources — aggregates only, no texts shipped

test-4 draws 10 000 items from the 40 public datasets below; items were built, sealed and run in the development
repository. **This repository ships no test-4 items, no per-item predictions and no reasoning traces — only aggregate
numbers** (`reports/test-4/results.json`, produced by `scripts/export_test4.py`). The column *texts redistributable*
records whether the dataset's stated terms allow redistributing its texts; for the sources marked **no** (no licence
stated, non-commercial, academic-only, or tweets under platform terms) only ids and a rebuild script are kept in the
development repository. Licences as stated on the dataset card or the original release, checked 2026-09-30.

| source (as used) | dataset | language | items | licence as stated | texts redistributable |
|---|---|---|---:|---|---|
| banking77 | [`legacy-datasets/banking77`](https://huggingface.co/datasets/legacy-datasets/banking77) | en | 300 | CC-BY-4.0 | yes |
| clinc150 | [`clinc/clinc_oos`](https://huggingface.co/datasets/clinc/clinc_oos) | en | 200 | CC-BY-3.0 | yes |
| massive-pt | [`mteb/amazon_massive_intent`](https://huggingface.co/datasets/mteb/amazon_massive_intent) | pt-PT | 500 | CC-BY-4.0 | yes |
| sst2 | [`stanfordnlp/sst2`](https://huggingface.co/datasets/stanfordnlp/sst2) | en | 150 | unknown | **no** |
| sst5 | [`SetFit/sst5`](https://huggingface.co/datasets/SetFit/sst5) | en | 200 | unknown | **no** |
| sst2-pt (MT) | [`PORTULAN/extraglue`](https://huggingface.co/datasets/PORTULAN/extraglue) | pt-PT | 150 | MIT | yes |
| b2w-nota | [`ruanchaves/b2w-reviews01`](https://huggingface.co/datasets/ruanchaves/b2w-reviews01) | pt-BR | 400 | CC BY-NC-SA 4.0 (original release) | **no** |
| tweetsentbr | [`eduagarcia/tweetsentbr_fewshot`](https://huggingface.co/datasets/eduagarcia/tweetsentbr_fewshot) | pt-BR | 300 | none stated | **no** |
| goemotions | [`google-research-datasets/go_emotions`](https://huggingface.co/datasets/google-research-datasets/go_emotions) | en | 300 | Apache-2.0 | **no** (see §3) |
| tweeteval-emo | [`cardiffnlp/tweet_eval`](https://huggingface.co/datasets/cardiffnlp/tweet_eval) | en | 300 | unknown | **no** |
| goemotions-pt (MT) | [`antoniomenezes/go_emotions_ptbr`](https://huggingface.co/datasets/antoniomenezes/go_emotions_ptbr) | pt-BR | 200 | Apache-2.0 | **no** (see §3) |
| civil | [`google/civil_comments`](https://huggingface.co/datasets/google/civil_comments) | en | 300 | CC0-1.0 | yes |
| tweeteval-hate | [`cardiffnlp/tweet_eval`](https://huggingface.co/datasets/cardiffnlp/tweet_eval) | en | 100 | unknown | **no** |
| tweeteval-off | [`cardiffnlp/tweet_eval`](https://huggingface.co/datasets/cardiffnlp/tweet_eval) | en | 100 | unknown | **no** |
| hatebr | [`eduagarcia/portuguese_benchmark`](https://huggingface.co/datasets/eduagarcia/portuguese_benchmark) | pt-BR | 300 | academic/research use only | **no** |
| toldbr | [`JAugusto97/ToLD-Br`](https://huggingface.co/datasets/JAugusto97/ToLD-Br) | pt-BR | 250 | CC BY-SA 4.0 (tweets) | **no** |
| fortuna | [`eduagarcia/portuguese_benchmark`](https://huggingface.co/datasets/eduagarcia/portuguese_benchmark) | pt | 150 | unknown | **no** |
| assin2-rte | [`nilc-nlp/assin2`](https://huggingface.co/datasets/nilc-nlp/assin2) | pt-BR | 350 | unknown | **no** |
| assin-rte | [`nilc-nlp/assin`](https://huggingface.co/datasets/nilc-nlp/assin) | pt-PT | 200 | unknown | **no** |
| faquad-nli | [`ruanchaves/faquad-nli`](https://huggingface.co/datasets/ruanchaves/faquad-nli) | pt-BR | 150 | CC-BY-4.0 | yes |
| rte-pt (MT) | [`PORTULAN/extraglue`](https://huggingface.co/datasets/PORTULAN/extraglue) | pt-PT | 150 | MIT | yes |
| snli | [`stanfordnlp/snli`](https://huggingface.co/datasets/stanfordnlp/snli) | en | 300 | CC-BY-SA-4.0 | yes |
| mnli | [`nyu-mll/multi_nli`](https://huggingface.co/datasets/nyu-mll/multi_nli) | en | 150 | CC-BY/SA, MIT | yes |
| boolq | [`google/boolq`](https://huggingface.co/datasets/google/boolq) | en | 400 | CC-BY-SA-3.0 | yes |
| boolq-pt (MT) | [`PORTULAN/extraglue`](https://huggingface.co/datasets/PORTULAN/extraglue) | pt-PT | 400 | MIT | yes |
| arc | [`allenai/ai2_arc`](https://huggingface.co/datasets/allenai/ai2_arc) | en | 250 | CC-BY-SA-4.0 | yes |
| csqa | [`tau/commonsense_qa`](https://huggingface.co/datasets/tau/commonsense_qa) | en | 200 | MIT | yes |
| mmlu | [`cais/mmlu`](https://huggingface.co/datasets/cais/mmlu) | en | 300 | MIT | yes |
| enem | [`maritaca-ai/enem`](https://huggingface.co/datasets/maritaca-ai/enem) | pt-BR | 250 | Apache-2.0 | yes |
| bluex | [`portuguese-benchmark-datasets/BLUEX`](https://huggingface.co/datasets/portuguese-benchmark-datasets/BLUEX) | pt-BR | 200 | none stated | **no** |
| oab | [`eduagarcia/oab_exams`](https://huggingface.co/datasets/eduagarcia/oab_exams) | pt-BR | 200 | MIT (original release) | yes |
| mmmlu-pt | [`openai/MMMLU`](https://huggingface.co/datasets/openai/MMMLU) | pt-BR | 100 | MIT | yes |
| agnews | [`fancyzhx/ag_news`](https://huggingface.co/datasets/fancyzhx/ag_news) | en | 250 | non-commercial | **no** |
| dbpedia | [`fancyzhx/dbpedia_14`](https://huggingface.co/datasets/fancyzhx/dbpedia_14) | en | 250 | CC-BY-SA-3.0 | yes |
| b2w-categoria | [`ruanchaves/b2w-reviews01`](https://huggingface.co/datasets/ruanchaves/b2w-reviews01) | pt-BR | 400 | CC BY-NC-SA 4.0 (original release) | **no** |
| stsb | [`sentence-transformers/stsb`](https://huggingface.co/datasets/sentence-transformers/stsb) | en | 300 | none stated | **no** |
| assin2-sts | [`nilc-nlp/assin2`](https://huggingface.co/datasets/nilc-nlp/assin2) | pt-BR | 300 | unknown | **no** |
| assin-sts | [`nilc-nlp/assin`](https://huggingface.co/datasets/nilc-nlp/assin) | pt-PT | 200 | unknown | **no** |
| sms | [`ucirvine/sms_spam`](https://huggingface.co/datasets/ucirvine/sms_spam) | en | 300 | CC BY 4.0 (UCI repository) | yes |
| sms-pt (MT) | [`dbarbedillo/SMS_Spam_Multilingual_Collection_Dataset`](https://huggingface.co/datasets/dbarbedillo/SMS_Spam_Multilingual_Collection_Dataset) | pt | 200 | GPL | yes |

20 of the 40 sources are marked **no**. (MT) = machine-translated Portuguese. ToLD-Br is fetched from its GitHub
release (`JAugusto97/ToLD-Br`), not from the Hugging Face card linked above. Attribution: each dataset's own citation, as given on its card.
Because all 40 datasets are public, they may be in the pretraining data of any evaluated system.
