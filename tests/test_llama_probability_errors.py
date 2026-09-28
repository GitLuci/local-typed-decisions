"""Missing top-N option probabilities, simulated without model weights or GPU."""
from copy import deepcopy
import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from conformance.contract import ContractError, Profile, assert_response, exercise_backend
from typed_decisions.llama_server import LlamaServerDecisionModel, load_config
from typed_decisions.server import create_app
from test_llama_server import FakeBuilder, FakeHTTP, QUESTION
from test_llama_routing import make_routed_backend

PARTIAL = Profile(question_errors=True)
BAD = {**QUESTION, "instructions": "MISSING: pick"}
BODY = {"state": "x", "questions": {"before": QUESTION, "bad": BAD, "after": QUESTION}}


class TopNHTTP(FakeHTTP):
    """Simulate one option at rank 100 or outside every requested top-N."""
    def __call__(self, request):
        body = json.loads(request.content) if request.content else {}
        if request.url.path == "/completion":
            prompt = "".join(map(chr, body["prompt"]))
            self.missing = "MISSING" in prompt or ("RANK100" in prompt and body["n_probs"] < 100)
        return super().__call__(request)


def backend(mode="fast", **options):
    fake = TopNHTTP(mode)
    model = LlamaServerDecisionModel(mode=mode, builder=FakeBuilder(),
        transport=httpx.MockTransport(fake), **options)
    model.fake = fake
    return model


@pytest.mark.parametrize("mode,expected", [("ultra", 20), ("fast", 20),
    ("medium", 256), ("slow", 256)])
def test_default_top_n_and_config_override(mode, expected):
    for options, count in [({}, expected), ({"n_probs": 37}, 37)]:
        model = backend(mode, **options)
        try:
            result = model.predict("x", {"q": QUESTION})
            assert "errors" not in result
            reads = [b for p, b in model.fake.calls if p == "/completion" and b["n_predict"] == 1]
            assert len(reads) == 1 and reads[0]["n_probs"] == count
        finally:
            model.close()


def test_wider_top_n_recovers_simulated_rank100_without_retry():
    for options, recovered in [({"n_probs": 20}, False), ({}, True)]:
        model = backend("slow", **options)
        body = {"state": "x", "questions": {"q": {**QUESTION, "instructions": "RANK100: pick"}}}
        try:
            result = model.predict(**body)
            assert_response(body, result, PARTIAL)
            assert ("q" in result["answers"]) is recovered
            calls = [b for p, b in model.fake.calls if p == "/completion"]
            assert len(calls) == 2  # one thought, one read, no retry
            assert result["usage"]["output_tokens"] == 6
        finally:
            model.close()


@pytest.mark.parametrize("mode", ["fast", "slow"])
def test_partial_batch_preserves_siblings_counts_all_usage_and_is_independent(mode):
    model = backend(mode)
    try:
        result = model.predict(**BODY)
        assert_response(BODY, result, PARTIAL)
        assert list(result["answers"]) == ["before", "after"]
        assert result["answers"]["before"] == result["answers"]["after"]
        assert list(result["errors"]) == ["bad"]
        assert result["errors"]["bad"]["missing_options"] == ["right"]
        calls = [b for p, b in model.fake.calls if p == "/completion"]
        assert len(calls) == (6 if mode == "slow" else 3)
        assert result["usage"] == {
            "input_tokens": sum(len(b["prompt"]) for b in calls),
            "output_tokens": 18 if mode == "slow" else 3,
        }
        assert exercise_backend(model.predict, BODY, PARTIAL)["status"] == "passed"
        with pytest.raises(ContractError):
            assert_response(BODY, result)  # base Jev contract remains strict
    finally:
        model.close()


@pytest.mark.parametrize("question,missing", [(BAD, ["right"]),
    ({"type": "score", "instructions": "MISSING", "criteria": ["low", "high"]}, ["1"]),
    ({"type": "noul", "instructions": "MISSING"}, ["true"])])
def test_all_error_http_200_is_explicit_and_uses_original_option_ids(question, missing):
    model = backend()
    body = {"state": "x", "questions": {"q": question}}
    try:
        with TestClient(create_app(model)) as client:
            response = client.post("/v1/systemone", json=body)
        assert response.status_code == 200
        result = response.json()
        assert_response(body, result, PARTIAL)
        assert result["answers"] == {}
        assert result["errors"]["q"]["missing_options"] == missing
        assert result["usage"]["output_tokens"] == 1
    finally:
        model.close()


def test_router_preserves_order_and_usage_across_partial_mode_groups():
    router = make_routed_backend(default_domain=None)
    original = router.factory
    def factory(**options):
        model = original(**options)
        model.client.close()
        model.fake = TopNHTTP(model.mode)
        model.client = httpx.Client(base_url=options["url"], transport=httpx.MockTransport(model.fake))
        return model
    router.factory = factory
    body = {"state": "x", "questions": {
        "bad8": BAD, "ok8": QUESTION, "bad4": BAD, "ok4": QUESTION}}
    domains = {"bad8": "factual", "ok8": "subjective_tone", "bad4": "numeric", "ok4": "numeric"}
    try:
        with TestClient(create_app(router)) as client:
            response = client.post("/v1/systemone", json={**body, "domains": domains})
        assert response.status_code == 200
        result = response.json()
        assert_response(body, result, PARTIAL)
        assert list(result["answers"]) == ["ok8", "ok4"]
        assert list(result["errors"]) == ["bad8", "bad4"]
        assert result["usage"]["output_tokens"] == 19
        calls = [b for model in router.made for p, b in model.fake.calls if p == "/completion"]
        assert result["usage"]["input_tokens"] == sum(len(b["prompt"]) for b in calls)
        assert [router.backends[m].n_probs for m in ("fast", "medium", "slow")] == [20, 256, 256]
    finally:
        router.close()


@pytest.mark.parametrize("fault", ["overlap", "uncovered", "extra", "unknown_option", "duplicate",
                                  "unknown_code", "invalid_count", "empty", "invented_probability"])
def test_partial_contract_rejects_corruptions(fault):
    model = backend()
    try:
        result = deepcopy(model.predict(**BODY))
    finally:
        model.close()
    error = result["errors"]["bad"]
    if fault == "overlap":
        result["errors"]["before"] = deepcopy(error)
    elif fault == "uncovered":
        result["errors"].clear()
    elif fault == "extra":
        result["errors"]["unrequested"] = deepcopy(error)
    elif fault == "unknown_option":
        error["missing_options"] = ["invented"]
    elif fault == "duplicate":
        error["missing_options"] *= 2
    elif fault == "unknown_code":
        error["code"] = "something_else"
    elif fault == "invalid_count":
        error["n_probs"] = True
    elif fault == "empty":
        error["missing_options"] = []
    else:
        error["probabilities"] = {"right": 0.01}
    with pytest.raises(ContractError):
        assert_response(BODY, result, PARTIAL)


def test_shipped_thinking_configs_do_not_override_profile_top_n():
    root = Path(__file__).resolve().parents[1]
    for mode in ("medium", "slow", "routed"):
        config = load_config(root / "examples" / f"llama-{mode}.json")
        assert "n_probs" not in config
