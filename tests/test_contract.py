import math
import pytest
from fastapi.testclient import TestClient

from typed_decisions.runtime import answer_from_probabilities, validate, MODEL_ID
from typed_decisions.metrics import summarize
from typed_decisions.server import create_app


def test_score_is_expectation_and_uniform_confidence_is_zero():
    q = {"type": "score", "criteria": ["low", "middle", "high"]}
    a = answer_from_probabilities(q, [.1, .3, .6])
    assert a["score"] == pytest.approx(1.5)
    assert list(a["legend"]) == ["0", "1", "2"]
    assert answer_from_probabilities(q, [1, 1, 1])["confidence"] == pytest.approx(0)


def test_noul_is_probability_true_without_separate_confidence():
    assert answer_from_probabilities({"type": "noul"}, [.7, .3]) == {"type": "noul", "noul": .3}


def test_choice_keys_and_distribution_are_preserved():
    q = {"type": "choice", "criteria": {"naïve": {"description": "A"}, "B": "B"}}
    a = answer_from_probabilities(q, [2, 8])
    assert a["choice"] == "B"
    assert a["probabilities"] == {"naïve": .2, "B": .8}
    assert a["confidence"] != .8


@pytest.mark.parametrize("p", [[math.nan, 1], [-1, 2], [0, 0], [1], [math.inf, 1]])
def test_invalid_distributions_rejected(p):
    with pytest.raises(ValueError):
        answer_from_probabilities({"type": "noul"}, p)


def test_noul_structured_criteria_becomes_explicit_clarification():
    q = {"q": {"type": "noul", "instructions": "Is it urgent?", "criteria": ["urgent means today"]}}
    r = validate({}, q)
    assert r["q"]["instructions"]["clarification"] == ["urgent means today"]
    assert "criteria" in q["q"]  # input not mutated


def test_bad_requests_rejected():
    for state, qs in [(None, {}), ({}, {}), ({}, {"q": {"type": "chat", "instructions": "hi"}}),
                      ({}, {"q": {"type": "choice", "instructions": "pick", "criteria": {"only": "one"}}})]:
        with pytest.raises(ValueError):
            validate(state, qs)


def test_calibration_metrics_known_values_and_p_equals_one_bin():
    rows = [{"p": [.8, .2], "target": [1., 0.]}, {"p": [.8, .2], "target": [0., 1.]}]
    m = summarize(rows)
    assert m["accuracy"] == .5
    assert m["ece_10"] == pytest.approx(.3)
    assert m["brier_sum"] == pytest.approx(.68)
    assert sum(b["n"] for b in summarize([{"p": [1, 0], "target": [1, 0]}])["reliability"]) == 1


def test_aleatoric_target_is_not_a_sampled_outcome():
    m = summarize([{"p": [1/6]*6, "target": [1/6]*6}])
    assert m["brier_sum"] == 0
    assert m["ece_10"] == 0
    assert m["accuracy"] == 1/6  # expected accuracy, not observed hits


class StubModel:
    def predict(self, state, questions):
        validate(state, questions)
        return {"model": MODEL_ID, "answers": {"q": {"type": "noul", "noul": .75}},
                "usage": {"input_tokens": 10, "output_tokens": 0}, "logits": {"q": [0, 1]}}


def test_http_contract_rejects_unknown_model_and_hides_logits():
    body = {"state": "example", "questions": {"q": {"type": "noul", "instructions": "True?"}}}
    with TestClient(create_app(StubModel())) as client:
        assert client.get("/health").json()["offline"] is True
        result = client.post("/v1/systemone", json=body)
        assert result.status_code == 200
        assert "logits" not in result.json()
        assert result.json()["answers"]["q"]["noul"] == .75
        assert client.post("/v1/systemone", json={**body, "model": "other-model-latest"}).status_code == 422
        assert client.post("/v1/systemone", content="{" ).status_code == 422
        assert client.post("/v1/systemone", content=b"x"*256001).status_code == 413
