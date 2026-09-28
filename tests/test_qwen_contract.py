import pytest
from fastapi.testclient import TestClient

from typed_decisions.runtime import validate
from typed_decisions.server import create_app


def test_explicit_null_noul_criteria_is_normalized():
    q = validate("state", {"q":{"type":"noul", "instructions":"Is it true?", "criteria":None}})
    assert "criteria" not in q["q"]


def test_qwen_schema_admits_255_dynamic_options_but_rejects_256():
    q = {"q": {"type": "choice", "instructions": "Select the matching code",
               "criteria": {f"label_{i}": f"Description {i}" for i in range(255)}}}
    assert len(validate("state", q, max_options=255)["q"]["criteria"]) == 255
    q["q"]["criteria"]["label_255"] = "extra"
    with pytest.raises(ValueError):
        validate("state", q, max_options=255)


@pytest.mark.parametrize("identity",["local-typed-decisions-qwen3-0.6b-tree","local-typed-decisions-qwen3-1.7b-tree"])
def test_server_advertises_and_enforces_the_loaded_model_identity(identity):
    class Model:
        model_id = identity

        def predict(self, state, questions):
            return {"model": self.model_id, "answers": {"q": {"type": "noul", "noul": .5}},
                    "usage": {"input_tokens": 1, "output_tokens": 0}, "logits": {"q": [0,0]}}

    body = {"state": "A", "questions": {"q": {"type": "noul", "instructions": "B?"}}}
    with TestClient(create_app(Model())) as c:
        assert c.get("/health").json()["model"] == Model.model_id
        for model_id in ("local", Model.model_id):
            r = c.post("/v1/systemone", json={**body, "model": model_id})
            assert r.status_code == 200
            assert r.json()["model"] == Model.model_id
        assert c.post("/v1/systemone", json={**body, "model": "local-typed-decisions-laya-multilingual"}).status_code == 422
