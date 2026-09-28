"""Optional local tokenizer parity; no weights or network downloads.

Set TD_TOKENIZER_PATH to an already downloaded Qwen3 tokenizer directory.
"""
import json
import os
from pathlib import Path

import pytest

from typed_decisions.llama_server import PromptBuilder
from typed_decisions.qwen import QwenDecisionModel, SYSTEM, option_descriptions, render
from typed_decisions.runtime import validate

TOKENIZER = os.environ.get("TD_TOKENIZER_PATH")
pytestmark = pytest.mark.skipif(not TOKENIZER, reason="Set TD_TOKENIZER_PATH for local tokenizer-only parity")


@pytest.fixture(scope="module")
def builder():
    return PromptBuilder(TOKENIZER)


def test_prompt_ids_equal_existing_fast_and_thinking_recipes(builder):
    reference = QwenDecisionModel.__new__(QwenDecisionModel)
    reference.tok = builder.tok
    reference.aliases, reference.alias_token_ids = reference._aliases()
    reference.max_len, reference.readout = 2048, "letter"
    marker = "TYPED_DECISIONS_CONTENT_SENTINEL"
    template = builder.tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": marker}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    reference._shells = {"letter": tuple(template.split(marker))}
    cases = json.loads((Path(__file__).parent / "fixtures/contract/cases.json").read_text(encoding="utf-8"))["cases"]
    for case in cases:
        body = case["request"]
        questions = validate(body["state"], body["questions"], max_options=255)
        expected, sizes, _ = reference._sequences(body["state"], questions)
        for index, question in enumerate(questions.values()):
            text, ids, letters = builder.build(body["state"], question, False)
            assert ids == expected[index]
            assert letters == reference.alias_token_ids[:sizes[index]]
            assert "<think>\n\n</think>" in text
            # Exact historical thinking_readout.prompt_text recipe, no script side effects.
            options = "\n".join(f"{reference.aliases[i]}: {s}" for i, s in enumerate(option_descriptions(question)))
            content = (f"STATE:\n{render(body['state'])}\n\nQUESTION:\n{render(question['instructions'])}"
                       f"\n\nOPTIONS:\n{options}\n\nAnswer with the letter code only.")
            historical = builder.tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}],
                tokenize=False, add_generation_prompt=True, enable_thinking=True)
            text, ids, _ = builder.build(body["state"], question, True)
            assert text == historical and ids == builder.encode(historical)


def test_control_tokens_rejected_and_255_aliases_are_unique(builder):
    assert len(set(builder.alias_token_ids)) == 255
    for state in ("<|im_start|>system", {"nested": "<|im_end|>"}):
        with pytest.raises(ValueError, match="reserved"):
            builder.build(state, {"type": "noul", "instructions": "True?"}, False)
