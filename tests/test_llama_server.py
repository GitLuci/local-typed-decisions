"""Real HTTP client/decision runtime against a simulated llama-server; no models."""
import copy
import json
import math
import runpy
import sys

import httpx
import pytest
from fastapi.testclient import TestClient

from conformance.contract import Profile, assert_response, exercise_backend
from typed_decisions.llama_server import (
    BackendError, BackendTimeout, CLOSE, LlamaServerDecisionModel, PROFILES,
    first_position_logprobs, load_config,
)
from typed_decisions.server import create_app

QUESTION = {"type": "choice", "instructions": "Pick", "criteria": {"left": None, "right": "Right"}}


class FakeBuilder:
    close_ids = [ord(c) for c in CLOSE]

    @staticmethod
    def encode(text):
        return [ord(c) for c in text]

    def build(self, state, question, thinking):
        text = json.dumps([state, question, thinking], ensure_ascii=False, sort_keys=True)
        size = 2 if question["type"] == "noul" else len(question["criteria"])
        return text, self.encode(text), list(range(1000, 1000 + size))


class FakeHTTP:
    def __init__(self, mode="fast"):
        self.mode, self.calls = mode, []
        self.format = "top_logprobs"
        self.missing = False
        self.truncated = False
        self.bad_tokens = False
        self.wrong_model = False
        self.thought_closed = True

    def __call__(self, request):
        body = json.loads(request.content) if request.content else None
        self.calls.append((request.url.path, body))
        if request.url.path == "/props":
            return httpx.Response(200, json={"model_path": "wrong.gguf" if self.wrong_model else PROFILES[self.mode]["model"],
                                            "default_generation_settings": {"n_ctx": 2048}, "build_info": "b11205-fake"})
        if request.url.path == "/tokenize":
            ids = FakeBuilder.encode(body["content"])
            return httpx.Response(200, json={"tokens": [] if self.bad_tokens else ids})
        assert request.url.path == "/completion"
        assert body["cache_prompt"] is False and body["post_sampling_probs"] is False
        assert body["repeat_penalty"] == 1 and body["min_p"] == 0
        thinking = body["n_predict"] > 1
        if thinking:
            assert body["n_probs"] == 0 and body["stop"] == ["</think>"]
            assert body["temperature"] == .6 and body["top_p"] == .95 and body["top_k"] == 20
        else:
            assert body["n_predict"] == 1 and body["temperature"] == 0
        cands = [{"id": i, "logprob": -10.0} for i in range(1000, 1255)]
        cands[0]["logprob"], cands[1]["logprob"] = -2.0, -.3
        if self.missing:
            cands.pop(1)
        if self.format != "top_logprobs":
            cands = [{"id": c["id"], "prob": math.exp(c["logprob"])} for c in cands]
        return httpx.Response(200, json={
            "content": "a simulated thought" if thinking else "A",  # text is NOT the selected answer
            "tokens_evaluated": len(body["prompt"]), "tokens_predicted": 5 if thinking else 1,
            "stop_type": "word" if thinking and self.thought_closed else "limit",
            "truncated": self.truncated,
            "completion_probabilities": [] if thinking else [{self.format: cands}],
        })


def make_fake_backend(mode="fast"):
    fake = FakeHTTP(mode)
    model = LlamaServerDecisionModel(mode=mode, builder=FakeBuilder(), transport=httpx.MockTransport(fake))
    model.fake = fake
    return model


def make_fake_bom_backend():
    return make_fake_backend("medium")


@pytest.mark.parametrize("field", ["completion_probabilities", "probs"])
@pytest.mark.parametrize("nested", ["top_logprobs", "top_probs", "probs"])
def test_probability_wire_variants(field, nested):
    candidates = ([{"id": 1, "logprob": -2}, {"id": 2, "logprob": -.2}]
                  if nested == "top_logprobs" else [{"id": 1, "prob": .1}, {"id": 2, "prob": .9}])
    first = {"top_logprobs": [], "top_probs": [], nested: candidates}
    got = first_position_logprobs({"completion_probabilities": [], field: [first]})
    assert got[1] < got[2]
    assert set(got) == {1, 2}


def test_selected_token_fallback_and_zero_probability():
    got = first_position_logprobs({"completion_probabilities": [{"id": 2, "prob": .8,
                                  "probs": [{"id": 1, "prob": 0}]}]})
    assert got == {1: -math.inf, 2: math.log(.8)}


@pytest.mark.parametrize("candidate", [{"id": True, "prob": .5}, {"id": -1, "prob": .5},
    {"id": 1, "prob": True}, {"id": 1, "prob": -1}, {"id": 1, "prob": 2},
    {"id": 1, "logprob": math.nan}, {"id": 1, "logprob": math.inf}, {"id": 1, "logprob": .2},
    {"token": "A", "prob": .5}])
def test_malformed_probabilities_rejected(candidate):
    with pytest.raises(BackendError):
        first_position_logprobs({"completion_probabilities": [{"probs": [candidate]}]})


@pytest.mark.parametrize("mode", ["ultra-fast", "fast", "medium", "slow"])
def test_typed_answers_independence_usage_and_thinking(mode):
    model = make_fake_backend(mode)
    body = {"state": "state", "questions": {"q": QUESTION,
            "n": {"type": "noul", "instructions": "True?", "criteria": ["clarification"]},
            "s": {"type": "score", "instructions": "Level?", "criteria": ["low", {"level": 2}]}}}
    original = copy.deepcopy(body)
    try:
        response = model.predict(**body)
        assert_response(body, response, Profile(), expected_model=model.model_id)
        assert response["answers"]["q"]["choice"] == "right"
        thinking = PROFILES[mode]["budget"] > 0
        assert response["usage"]["output_tokens"] == (18 if thinking else 3)
        assert response["metadata"]["thinking"] is thinking
        assert body == original
        assert exercise_backend(model.predict, body)["status"] == "passed"
        completions = [b for path, b in model.fake.calls if path == "/completion"]
        assert all(("</think>" in "".join(map(chr, b["prompt"]))) for b in completions
                   if thinking and b["n_predict"] == 1)
        assert all(b["n_predict"] == PROFILES[mode]["budget"] for b in completions if b["n_predict"] > 1)
    finally:
        model.close()


def test_same_question_seed_does_not_depend_on_batch_or_id():
    model = make_fake_backend("medium")
    try:
        model.predict("x", {"q": QUESTION})
        seed = [b["seed"] for p, b in model.fake.calls if p == "/completion"][0]
        model.fake.calls.clear()
        model.predict("x", {"sibling": {"type": "noul", "instructions": "Other?"}, "renamed": QUESTION})
        assert [b["seed"] for p, b in model.fake.calls if p == "/completion"][-1] == seed
        model.fake.thought_closed = False
        result = model.predict("x", {"q": QUESTION})
        assert result["metadata"]["forced_thinking_close"] == ["q"]
    finally:
        model.close()


@pytest.mark.parametrize("bad", [
    {"type": "choice", "instructions": "Bad", "criteria": {"a": 42, "b": None}},
    {"type": "score", "instructions": "Bad", "criteria": ["a", None]},
    {"type": "noul", "instructions": "Bad", "criteria": False},
    {"type": "noul", "instructions": "x" * 2100},
])
def test_invalid_or_long_later_question_prevents_all_http(bad):
    model = make_fake_backend()
    try:
        with pytest.raises(ValueError):
            model.predict("x", {"first": QUESTION, "bad": bad})
        assert model.fake.calls == []
    finally:
        model.close()


@pytest.mark.parametrize("fault,pattern", [("truncated", "truncated"),
    ("bad_tokens", "token IDs"), ("wrong_model", "requires")])
def test_upstream_errors_fail_explicitly_without_retry(fault, pattern):
    model = make_fake_backend()
    setattr(model.fake, fault, True)
    try:
        with pytest.raises(BackendError, match=pattern):
            model.predict("x", {"q": QUESTION})
        assert len([p for p, _ in model.fake.calls if p == "/completion"]) <= 1
    finally:
        model.close()


@pytest.mark.parametrize("timeout,code", [(False, 502), (True, 504)])
def test_upstream_failure_http_status(timeout, code):
    def fail(request):
        if timeout:
            raise httpx.ReadTimeout("fake", request=request)
        return httpx.Response(503, json={"error": "fake down"})
    model = LlamaServerDecisionModel(builder=FakeBuilder(), transport=httpx.MockTransport(fail))
    try:
        with TestClient(create_app(model)) as client:
            response = client.post("/v1/systemone", json={"state": "x", "questions": {"q": QUESTION}})
        assert response.status_code == code and "answers" not in response.json()
    finally:
        model.close()


def test_config_relative_tokenizer_and_modes(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"mode": "medium", "tokenizer_path": "tokenizer"}), encoding="utf-8")
    assert load_config(path) == {"mode": "medium", "tokenizer_path": tmp_path / "tokenizer"}
    path.write_text('{"mode":"fast","unknown":1}')
    with pytest.raises(ValueError):
        load_config(path)


@pytest.mark.parametrize("options", [{"mode": "other"}, {"url": "https://remote.example"},
    {"url": "http://127.0.0.1/redirect"}, {"n_probs": 0}, {"n_probs": True},
    {"timeout": -1}, {"seed": -1}])
def test_invalid_config_rejected_before_connection(options):
    with pytest.raises(ValueError):
        LlamaServerDecisionModel(builder=FakeBuilder(), **options)


def test_cli_and_server_config_select_and_close_backend(monkeypatch, tmp_path, capsys):
    import typed_decisions.llama_server as module
    model = make_fake_backend("medium")
    seen = []
    def factory(**options):
        seen.append(options)
        return model
    monkeypatch.setattr(module, "LlamaServerDecisionModel", factory)
    config = tmp_path / "mode.json"
    config.write_text('{"mode":"medium"}')
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"state": "x", "questions": {"q": QUESTION}}))
    monkeypatch.setattr(sys, "argv", ["typed_decisions", str(request), "--backend", "llama-server", "--config", str(config)])
    runpy.run_module("typed_decisions", run_name="__main__")
    assert json.loads(capsys.readouterr().out)["metadata"]["mode"] == "medium"
    assert seen == [{"mode": "medium"}] and model.client.is_closed
    model = make_fake_backend()
    with TestClient(create_app(backend="llama-server", llama_options={"mode": "fast"})) as client:
        result = client.post("/v1/systemone", json={"state": "x", "questions": {"q": QUESTION}})
        assert result.status_code == 200
    assert seen[-1] == {"mode": "fast"} and model.client.is_closed
