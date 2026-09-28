"""Routing, wire extension and lazy process ownership, all with fake HTTP/processes."""
import copy
import json
from pathlib import Path
import subprocess

import httpx
import pytest
from fastapi.testclient import TestClient

from conformance.contract import assert_response, exercise_backend
from typed_decisions.llama_server import BackendError, BackendTimeout, LlamaServerDecisionModel, PROFILES, create_backend, load_config
from typed_decisions.llama_routing import RoutedDecisionModel, ServerPool
from typed_decisions.server import create_app
from test_llama_server import FakeBuilder, FakeHTTP, QUESTION

ROUTES = {"numeric": "medium", "sentence": "medium", "factual": "slow",
          "deterministic": "slow", "sentiment": "slow", "subjective_tone": "fast",
          "robotic_style": "fast", "noul_refund": "fast", "score_urgency": "fast", "random": "fast"}
SERVERS = {"4b": {"url": "http://127.0.0.1:8791"}, "8b": {"url": "http://127.0.0.1:8790"}}


class FakePool:
    def __init__(self):
        self.ensured, self.closed = [], False

    def ensure(self, key):
        self.ensured.append(key)

    def close(self):
        self.closed = True


def make_routed_backend(default_domain="sentiment", mode="routed"):
    pool = FakePool()
    made = []
    def factory(**options):
        fake = FakeHTTP(options["mode"])
        options.setdefault("builder", FakeBuilder())
        model = LlamaServerDecisionModel(**options, transport=httpx.MockTransport(fake))
        model.fake = fake
        made.append(model)
        return model
    router = RoutedDecisionModel(mode=mode, servers=SERVERS,
        **({"routes": ROUTES, "default_domain": default_domain} if mode == "routed" else {}),
        backend_factory=factory, server_pool=pool)
    router.made = made
    return router


def test_mixed_domains_route_to_right_modes_and_preserve_answer_order():
    router = make_routed_backend(default_domain=None)
    body = {"state": "some data", "questions": {key: copy.deepcopy(QUESTION) for key in ROUTES}}
    domains = {key: key for key in ROUTES}
    snapshot = copy.deepcopy(body)
    try:
        assert router.made == [] and router.pool.ensured == []
        result = router.predict(**body, domains=domains)
        assert_response(body, result)
        assert list(result["answers"]) == list(body["questions"])
        assert {qid: meta["mode"] for qid, meta in result["metadata"]["routing"].items()} == ROUTES
        assert result["usage"]["output_tokens"] == 5 * 6 + 5
        assert body == snapshot
        assert router.pool.ensured == ["4b", "8b", "8b"]
        assert len(router.made) == 3
        assert router.backends["fast"].builder is router.backends["slow"].builder
        for model in router.made:
            completions = [b for path, b in model.fake.calls if path == "/completion"]
            budget = PROFILES[model.mode]["budget"]
            assert all(b["n_predict"] in (1, budget) for b in completions)
    finally:
        router.close()
    assert all(model.client.is_closed for model in router.made) and router.pool.closed


def test_explicit_domains_independent_of_id_order_siblings_and_previous_state():
    router = make_routed_backend(default_domain=None)
    try:
        first = router.predict("x", {"original": QUESTION}, domains={"original": "numeric"})
        router.predict("other state", {"other": QUESTION}, domains={"other": "subjective_tone"})
        second = router.predict("x", {"sibling": QUESTION, "renamed": QUESTION},
                                domains={"sibling": "factual", "renamed": "numeric"})
        assert first["answers"]["original"] == second["answers"]["renamed"]
        assert first["metadata"]["routing"]["original"] == second["metadata"]["routing"]["renamed"]
        calls = [b for p, b in router.backends["medium"].fake.calls if p == "/completion"]
        assert calls[0]["seed"] == calls[-1]["seed"]
    finally:
        router.close()


@pytest.mark.parametrize("domains", [None, {}, {"q": "unknown"}, {"q": []}, {"extra": "numeric"}, []])
def test_missing_or_bad_domains_do_not_create_backends_or_processes(domains):
    router = make_routed_backend(default_domain=None)
    try:
        with pytest.raises(ValueError):
            router.predict("x", {"q": QUESTION}, domains=domains)
        assert router.made == [] and router.pool.ensured == []
    finally:
        router.close()


def test_all_budgets_checked_before_first_model_launch():
    router = make_routed_backend(default_domain=None)
    try:
        with pytest.raises(ValueError, match="budget"):
            router.predict("x", {"ok": QUESTION, "long": {"type": "noul", "instructions": "x" * 1800}},
                           domains={"ok": "subjective_tone", "long": "numeric"})
        assert router.pool.ensured == []
        assert all(model.fake.calls == [] for model in router.made)
    finally:
        router.close()


@pytest.mark.parametrize("mode", ["ultra-fast", "fast", "medium", "slow"])
def test_fixed_mode_with_lazy_server_config_and_alias(mode):
    router = make_routed_backend(mode=mode)
    try:
        assert exercise_backend(router.predict, {"state": "x", "questions": {"q": QUESTION}})["status"] == "passed"
        assert set(router.pool.ensured) == {PROFILES[mode]["model_key"]}
        assert router.mode == mode
    finally:
        router.close()


def test_http_domains_local_extension_and_bad_requests():
    router = make_routed_backend(default_domain=None)
    body = {"state": "x", "questions": {"q": QUESTION}, "domains": {"q": "numeric"}}
    try:
        with TestClient(create_app(router)) as client:
            assert client.post("/v1/systemone", json={**body, "domains": {}}).status_code == 422
            assert router.pool.ensured == []
            response = client.post("/v1/systemone", json=body)
            assert response.status_code == 200
            assert response.json()["metadata"]["routing"]["q"]["mode"] == "medium"
        with TestClient(create_app(LlamaServerDecisionModel(builder=FakeBuilder(), transport=httpx.MockTransport(FakeHTTP())))) as client:
            assert client.post("/v1/systemone", json=body).status_code == 422
    finally:
        router.close()


def test_route_map_is_config_data_and_default_is_explicit():
    router = make_routed_backend()
    router.close()
    custom = RoutedDecisionModel(mode="routed", servers={"4b": SERVERS["4b"]},
        routes={"custom-domain": "ultra-fast"}, default_domain="custom-domain", server_pool=FakePool(),
        backend_factory=lambda **kw: LlamaServerDecisionModel(**kw, builder=FakeBuilder(), transport=httpx.MockTransport(FakeHTTP("ultra-fast"))))
    try:
        assert custom.predict("x", {"q": QUESTION})["metadata"]["routing"]["q"]["mode"] == "ultra-fast"
    finally:
        custom.close()


@pytest.mark.parametrize("change", [{"routes": {}}, {"routes": {"domain": "unknown"}},
    {"default_domain": "unknown"}, {"servers": {"4b": SERVERS["4b"]}},
    {"servers": {"4b": {"url": "http://127.0.0.1:8790"}, "8b": SERVERS["8b"]}}])
def test_invalid_router_configuration(change):
    config = {"mode": "routed", "servers": SERVERS, "routes": ROUTES}
    with pytest.raises(ValueError):
        create_backend(**{**config, **change})


class FakeProcess:
    def __init__(self, events, key):
        self.events, self.key, self.code = events, key, None
        self.stubborn = False

    def poll(self):
        return self.code

    def terminate(self):
        self.events.append(("terminate", self.key))
        if not self.stubborn:
            self.code = 0

    def wait(self, timeout):
        if self.code is None:
            raise subprocess.TimeoutExpired("fake", timeout)
        return self.code

    def kill(self):
        self.events.append(("kill", self.key))
        self.code = -9


@pytest.fixture
def managed(monkeypatch, tmp_path):
    import typed_decisions.llama_routing as module
    executable = tmp_path / "llama-server.exe"
    executable.write_text("fake, never executed")
    servers = copy.deepcopy(SERVERS)
    for key, mode in (("4b", "ultra-fast"), ("8b", "fast")):
        model = tmp_path / PROFILES[mode]["model"]
        model.write_text("fake GGUF, never loaded")
        servers[key]["launch"] = {"executable": executable, "model_path": model, "startup_timeout": 1}
    events, processes, args_seen = [], [], []
    def popen(args, **kwargs):
        key = "4b" if "qwen3-4b" in args[2] else "8b"
        assert all(p.poll() is not None for p in processes)
        assert kwargs["shell"] is False and kwargs["stdout"] == subprocess.DEVNULL
        if module.os.name == "nt":
            assert kwargs["creationflags"] == subprocess.CREATE_NO_WINDOW
        events.append(("start", key))
        args_seen.append(args)
        process = FakeProcess(events, key)
        processes.append(process)
        return process
    monkeypatch.setattr(module.subprocess, "Popen", popen)
    class FakeJob:
        def attach(self, process):
            assert process is processes[-1]
        def close(self):
            pass
    monkeypatch.setattr(module, "ProcessJob", FakeJob)
    monkeypatch.setattr(ServerPool, "_port_free", staticmethod(lambda url: True))
    original_client = httpx.Client
    def client(**kw):
        kw.setdefault("transport", httpx.MockTransport(lambda req: httpx.Response(200, json={"status": "ok"})))
        return original_client(**kw)
    monkeypatch.setattr(module.httpx, "Client", client)
    pool = ServerPool(servers)
    yield pool, events, processes, args_seen
    pool.close()


def test_lazy_process_reuse_switch_flags_and_ownership(managed):
    pool, events, processes, args_seen = managed
    assert events == []
    pool.ensure("8b")
    pool.ensure("8b")
    pool.ensure("4b")
    assert events == [("start", "8b"), ("terminate", "8b"), ("start", "4b")]
    assert args_seen[0][args_seen[0].index("-ngl") + 1] == "30"
    assert args_seen[1][args_seen[1].index("-ngl") + 1] == "99"
    assert all(args[args.index("-c") + 1] == "2048" for args in args_seen)
    assert args_seen[0][-2:] == ["--load-mode", "none"]
    assert "--load-mode" not in args_seen[1]
    pool.close()
    assert events[-1] == ("terminate", "4b")
    pool.close()
    assert len(events) == 4


def test_occupied_port_not_adopted_or_terminated(managed, monkeypatch):
    pool, events, _, _ = managed
    monkeypatch.setattr(ServerPool, "_port_free", staticmethod(lambda url: False))
    with pytest.raises(BackendError, match="occupied"):
        pool.ensure("8b")
    assert events == []


def test_only_owned_stubborn_process_is_killed_before_switch(managed):
    pool, events, processes, _ = managed
    pool.ensure("8b")
    processes[0].stubborn = True
    pool.ensure("4b")
    assert events[:4] == [("start", "8b"), ("terminate", "8b"), ("kill", "8b"), ("start", "4b")]


def test_startup_timeout_cleans_owned_process(managed, monkeypatch):
    import typed_decisions.llama_routing as module
    pool, events, processes, _ = managed
    tick = iter([0, 2])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(tick))
    with pytest.raises(BackendTimeout):
        pool.ensure("8b")
    assert events == [("start", "8b"), ("terminate", "8b")]
    assert pool.process is None


def test_external_servers_are_never_started_or_stopped(monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: pytest.fail("unexpected process"))
    pool = ServerPool(SERVERS)
    pool.ensure("4b")
    pool.ensure("8b")
    pool.close()
    assert pool.process is None


def test_router_with_managed_pool_launches_two_models_for_four_modes(managed):
    pool, events, _, _ = managed
    def factory(**options):
        options.setdefault("builder", FakeBuilder())
        return LlamaServerDecisionModel(**options, transport=httpx.MockTransport(FakeHTTP(options["mode"])))
    router = RoutedDecisionModel(mode="routed", servers=pool.servers,
        routes={mode: mode for mode in ("ultra-fast", "fast", "medium", "slow")},
        backend_factory=factory, server_pool=pool)
    assert events == []
    try:
        qs = {mode: QUESTION for mode in ("fast", "medium", "slow", "ultra-fast")}
        result = router.predict("x", qs, domains={mode: mode for mode in qs})
        assert list(result["answers"]) == list(qs)
        assert events == [("start", "4b"), ("terminate", "4b"), ("start", "8b")]
    finally:
        router.close()
    assert events[-1] == ("terminate", "8b")


def test_config_resolves_nested_paths_and_keeps_routes(tmp_path):
    path = tmp_path / "config.json"
    config = {"mode": "routed", "routes": ROUTES, "servers": copy.deepcopy(SERVERS)}
    config["servers"]["4b"].update({"tokenizer_path": "../tokenizer", "launch": {
        "executable": "llama.exe", "model_path": "qwen3-4b-Q8_0.gguf"}})
    path.write_text(json.dumps(config), encoding="utf-8")
    loaded = load_config(path)
    assert loaded["routes"] == ROUTES
    assert loaded["servers"]["4b"]["launch"]["executable"] == tmp_path / "llama.exe"
    assert loaded["servers"]["4b"]["tokenizer_path"] == tmp_path.parent / "tokenizer"


@pytest.mark.parametrize("mode", ["ultra-fast", "fast", "medium", "slow", "routed"])
def test_shipped_configs_create_no_backend_or_process(mode):
    config = load_config(Path(__file__).resolve().parents[1] / "examples" / f"llama-{mode}.json")
    runtime = create_backend(**config)
    try:
        assert runtime.backends == {} and runtime.pool.process is None
        if mode == "routed":
            assert runtime.routes == ROUTES and runtime.default_domain is None
    finally:
        runtime.close()
