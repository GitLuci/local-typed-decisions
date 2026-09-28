# Contributing

Thanks for your interest. This is a research codebase with a strict evaluation discipline; please read
[docs/METHODOLOGY.md](docs/METHODOLOGY.md) before proposing a new result.

## Ground rules

- **No fine-tuning, frozen weights.** Changes must work with the published weights. Quantization without an
  importance matrix counts as frozen; anything that uses data or gradients to change weights is out of scope.
- **Pre-register before you measure.** For any claim about accuracy, commit a short plan (what runs, which set,
  the decision rule and the threshold) *before* running on a test set. Select prompts/budgets/models on development
  data only; confirm once on a sealed set.
- **Never edit sealed files.** `examples/test-3/{states,questions,gold,levels}.jsonl` and
  `examples/test-2-scenarios.jsonl` are byte-exact; `python scripts/test3_seal.py --check` must pass. Their
  Portuguese field names are part of the seal.
- **Numbers come from committed reports.** Every number in the docs must be produced by a committed script and
  backed by a committed aggregate report under `reports/`. Do not hand-edit tables.
- **Baseline outputs.** Do not commit raw responses from commercial APIs; commit aggregates only.
- **Independence.** Each question must be answered without information from other questions in the same request;
  the conformance suite (`tests/test_contract_conformance.py`) checks this for any backend.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest -q
```

The test suite needs no model weights, GPU or network (HTTP and processes are simulated). Tests that exercise the
PyTorch research backends (`tests/test_tree.py`) need `torch` from `requirements.lock.txt` and are
skipped without it. Optional tokenizer-only checks run when `TD_TOKENIZER_PATH` points to an installed Qwen3
tokenizer.

## Pull requests

- Keep changes focused; include tests for runtime changes.
- Code, comments, docs and commit messages in English. Benchmark data may be in other languages.
- State the hardware and engine build for any latency or memory number (e.g. "RX 7600 8 GB, llama.cpp b11205
  Vulkan").
- Open fronts are listed in [docs/OPEN_WORK.md](docs/OPEN_WORK.md).

By contributing you agree that your contributions are licensed under the Apache License 2.0.
