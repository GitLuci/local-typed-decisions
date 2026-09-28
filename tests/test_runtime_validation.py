"""Criteria validation and HTTP rejection, without model weights or inference."""
import copy
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from conformance.contract import Profile, assert_request, assert_response, same_json
from typed_decisions.runtime import answer_from_probabilities, validate
from typed_decisions.server import create_app


INVALID_CRITERIA = [
    ("choice", {"a": value, "b": None}) for value in (42, 0.5, False, True)
] + [
    ("score", ["low", value]) for value in (None, 42, 0.5, False, True)
] + [
    ("noul", value) for value in (42, 0.5, False, True)
] + [
    ("noul", {"false": "No", "true": value})
    for value in (None, 42, 0.5, False, True)
]

NESTED = {"description": "structured", "data": [42, 0.5, False, True, None]}
VALID_QUESTIONS = [
    {"type": "choice", "instructions": "Choose", "criteria": {
        "text": "", "null": None, "object": NESTED, "array": [NESTED],
        "empty-object": {}, "empty-array": [],
    }},
    {"type": "score", "instructions": "Score", "criteria": ["", NESTED, [NESTED], {}, []]},
    {"type": "noul", "instructions": "True?"},
] + [
    {"type": "noul", "instructions": "True?", "criteria": value}
    for value in (None, {}, {"true": "Yes"}, {"false": NESTED, "true": [NESTED]},
                  "clarification", "", NESTED, [NESTED], [])
]


def invalid_body(kind, criteria):
    # An earlier valid question must not be mutated or partially evaluated.
    return {"state": copy.deepcopy(NESTED), "questions": {
        "first": {"type": "noul", "instructions": "True?", "criteria": [NESTED]},
        "bad": {"type": kind, "instructions": "Decide", "criteria": copy.deepcopy(criteria)},
    }}


class ValidatingDouble:
    """Real input validation/readout around a counted, deterministic fake inference."""
    model_id = "validation-test-double"

    def __init__(self):
        self.infer = Mock(side_effect=self._answers)

    @staticmethod
    def _answers(questions):
        return {qid: answer_from_probabilities(q, [1] * (
            2 if q["type"] == "noul" else len(q["criteria"])))
            for qid, q in questions.items()}

    def predict(self, state, questions):
        normalized = validate(state, questions, max_options=255)
        return {"model": self.model_id, "answers": self.infer(normalized),
                "usage": {"input_tokens": 0, "output_tokens": 0}}


@pytest.mark.parametrize("kind,criteria", INVALID_CRITERIA)
def test_invalid_criteria_rejected_without_mutation(kind, criteria):
    body = invalid_body(kind, criteria)
    snapshot = copy.deepcopy(body)
    with pytest.raises(ValueError, match="bad:"):
        validate(body["state"], body["questions"], max_options=255)
    assert same_json(body, snapshot)


@pytest.mark.parametrize("kind,criteria", INVALID_CRITERIA)
def test_http_criteria_errors_precede_inference(kind, criteria):
    body = invalid_body(kind, criteria)
    backend = ValidatingDouble()
    with TestClient(create_app(model=backend)) as client:
        result = client.post("/v1/systemone", json=body)
    assert result.status_code == 422
    assert "bad:" in result.json()["detail"]
    assert "answers" not in result.json()
    backend.infer.assert_not_called()


@pytest.mark.parametrize("question", VALID_QUESTIONS)
def test_valid_criteria_preserve_types_and_legacy_normalization(question):
    body = {"state": copy.deepcopy(NESTED), "questions": {"q": copy.deepcopy(question)}}
    snapshot = copy.deepcopy(body)
    assert_request(body)
    normalized = validate(body["state"], body["questions"], max_options=255)
    expected = copy.deepcopy(body["questions"])
    if question["type"] == "noul":
        c = question.get("criteria")
        if c is None:
            expected["q"].pop("criteria", None)
        elif not (isinstance(c, dict) and set(c) <= {"true", "false"}):
            expected["q"]["instructions"] = {"question": question["instructions"], "clarification": c}
            expected["q"].pop("criteria")
    assert same_json(normalized, expected)
    assert same_json(body, snapshot)
    backend = ValidatingDouble()
    with TestClient(create_app(model=backend)) as client:
        result = client.post("/v1/systemone", json=body)
    assert result.status_code == 200
    assert_response(body, result.json(), Profile(fast_path=True), expected_model=backend.model_id)
    backend.infer.assert_called_once()
