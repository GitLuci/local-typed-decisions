"""Release entry point, model downloads and Windows lifetime guard; no real models."""
from copy import deepcopy
import ctypes as ct
import hashlib
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from typed_decisions.llama_server import BackendError, create_backend, load_config
from typed_decisions.process_job import ExtendedLimits, ProcessJob
from scripts import fetch_models as fetch
from test_llama_routing import managed  # simulated processes, including lifetime guard

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("mode", [None, "ultra", "fast", "medium", "slow", "routed"])
def test_serve_selects_config_mode_without_starting_models(monkeypatch, mode):
    import typed_decisions.server as server
    import uvicorn
    captured = {}
    sentinel = object()
    def app(**options):
        captured.update(options)
        runtime = create_backend(**options["llama_options"])
        try:
            assert runtime.backends == {} and runtime.pool.process is None
        finally:
            runtime.close()
        return sentinel
    def run(instance, **options):
        assert instance is sentinel and options == {"host": "127.0.0.1", "port": 8000}
    monkeypatch.setattr(server, "create_app", app)
    monkeypatch.setattr(uvicorn, "run", run)
    argv = ["typed_decisions", "serve", "--config", str(ROOT / "examples/llama-routed.json"), "--port", "8000"]
    if mode:
        argv += ["--mode", mode]
    monkeypatch.setattr(sys, "argv", argv)
    runpy.run_module("typed_decisions", run_name="__main__")
    assert captured["backend"] == "llama-server"
    assert captured["llama_options"]["mode"] == (mode or "routed")


@pytest.mark.parametrize("argv", [["serve"], ["serve", "--port", "0"],
    ["--mode", "ultra"], ["serve", "--config", "unused.json", "--backend", "qwen"]])
def test_invalid_serve_arguments_fail_before_backend(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["typed_decisions", *argv])
    with pytest.raises(SystemExit) as error:
        runpy.run_module("typed_decisions", run_name="__main__")
    assert error.value.code == 2


@pytest.fixture
def downloads(tmp_path):
    payload = b"fake GGUF bytes"
    cached = tmp_path / "cached"
    cached.write_bytes(payload)
    item = {"kind": "gguf", "filename": "qwen.gguf", "path": "models/qwen.gguf",
            "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    manifest = {"version": 1, "files": [item]}
    calls = []
    def download(**options):
        calls.append(options)
        return cached
    def info(**options):
        calls.append(options)
        return SimpleNamespace(private=True, sha="a" * 40)
    return tmp_path, cached, manifest, calls, download, info


def test_download_verifies_before_install_and_is_offline_on_second_run(downloads):
    root, _, manifest, calls, download, info = downloads
    config = {"repo_id": "owner/private", "revision": "main"}
    assert fetch.obtain(config, manifest, root=root, download=download, repo_info=info) == {"verified": 1, "downloaded": 1}
    assert calls[0] == {"repo_id": "owner/private", "revision": "main", "token": None}
    assert calls[1]["revision"] == "a" * 40 and calls[1]["token"] is None
    assert (root / "models/qwen.gguf").read_bytes() == b"fake GGUF bytes"
    calls.clear()
    assert fetch.obtain({}, manifest, root=root, verify_only=True) == {"verified": 1, "downloaded": 0}
    assert fetch.obtain({}, manifest, root=root, download=download) == {"verified": 1, "downloaded": 0}
    assert calls == []


@pytest.mark.parametrize("fault", ["hash", "size", "existing"])
def test_bad_artifact_not_installed_or_overwritten(downloads, fault):
    root, cached, manifest, _, download, info = downloads
    target = root / "models/qwen.gguf"
    if fault == "hash":
        cached.write_bytes(b"FAKE GGUF bytes")
    elif fault == "size":
        cached.write_bytes(b"short")
    else:
        target.parent.mkdir()
        target.write_bytes(b"existing file preserved")
    with pytest.raises(ValueError):
        fetch.obtain({"repo_id": "owner/private"}, manifest, root=root, download=download, repo_info=info)
    assert target.read_bytes() == b"existing file preserved" if fault == "existing" else not target.exists()
    assert not list(root.glob("models/.download-*"))


@pytest.mark.parametrize("fault", ["unpinned", "missing_repo", "credential_field", "escape", "duplicate", "verify_missing"])
def test_download_preflight_rejects_invalid_setup_without_fetch(downloads, fault):
    root, _, manifest, calls, download, info = downloads
    config, verify_only = {"repo_id": "owner/private"}, False
    if fault == "unpinned":
        info = lambda **kw: SimpleNamespace(private=False, sha=None)
    elif fault == "missing_repo":
        config = {}
    elif fault == "credential_field":
        config["token"] = "not-a-real-credential"
    elif fault == "escape":
        manifest["files"][0]["path"] = "models/../../escape.gguf"
    elif fault == "duplicate":
        manifest["files"] *= 2
    else:
        verify_only = True
    with pytest.raises(ValueError):
        fetch.obtain(config, manifest, root=root, verify_only=verify_only, download=download, repo_info=info)
    assert calls == []


def test_public_tokenizers_use_pinned_revision_without_private_token(downloads):
    root, _, manifest, calls, download, info = downloads
    manifest["files"][0].update(kind="tokenizer", repo_id="Qwen/Qwen3-4B", revision="b" * 40)
    fetch.obtain({}, manifest, root=root, download=download, repo_info=info)
    assert len(calls) == 1 and calls[0]["token"] is False and calls[0]["revision"] == "b" * 40


def test_release_manifest_matches_recorded_provenance_and_runtime_paths():
    manifest = fetch.read(ROOT / "release-models.json")
    fetch.plan(manifest, ROOT)
    for size in ("4b", "8b"):
        gguf = fetch.read(ROOT / f"reports/manifests/gguf-manifest-qwen3-{size}.json")["files"]
        tokenizer = fetch.read(ROOT / f"reports/manifests/model-manifest-qwen3-{size}.json")["files"]
        lock = fetch.read(ROOT / f"qwen-{size}.lock.json")
        matching = [i for i in manifest["files"] if f"qwen3-{size}" in i["path"]]
        assert len(matching) == 6
        for item in matching:
            recorded = (gguf if item["kind"] == "gguf" else tokenizer)[Path(item["filename"]).name]
            assert (item["bytes"], item["sha256"]) == (recorded["bytes"], recorded["sha256"])
            if item["kind"] == "tokenizer":
                assert item["revision"] == lock["revision"] and item["repo_id"] == lock["repo_id"]
        config = load_config(ROOT / "examples/llama-routed.json")["servers"][size]
        assert any((ROOT / i["path"]).resolve() == config["launch"]["model_path"] for i in matching)


def test_release_repo_and_subfolder_layout(downloads):
    manifest = fetch.read(ROOT / "release-models.json")
    config = fetch.read(ROOT / "examples/models-hf.json")
    assert manifest["repo_id"] == config["repo_id"] == "bssgoat/local-typed-decisions"
    ggufs = [i for i in manifest["files"] if i["kind"] == "gguf"]
    assert {i["filename"] for i in ggufs} == {
        "qwen3-4b-gguf/qwen3-4b-Q8_0.gguf", "qwen3-8b-gguf/qwen3-8b-Q8_0.gguf"}
    root, _, fake, calls, download, info = downloads
    fake.update(repo_id=manifest["repo_id"], revision="main")
    fake["files"][0]["filename"] = ggufs[0]["filename"]
    fetch.obtain({}, fake, root=root, download=download, repo_info=info)
    assert calls[0]["repo_id"] == manifest["repo_id"]
    assert calls[1]["filename"] == ggufs[0]["filename"]


class FakeKernel:
    def __init__(self, fail=None):
        self.fail, self.closed, self.assigned = fail, [], []
    def CreateJobObjectW(self, security, name):
        assert security is None and name is None
        return 0 if self.fail == "create" else 42
    def SetInformationJobObject(self, handle, kind, data, size):
        assert handle == 42 and kind == 9 and size == ct.sizeof(ExtendedLimits)
        assert ct.cast(data, ct.POINTER(ExtendedLimits)).contents.basic.flags == 0x2000
        return self.fail != "configure"
    def AssignProcessToJobObject(self, job, process):
        self.assigned.append((job, process))
        return self.fail != "assign"
    def CloseHandle(self, handle):
        self.closed.append(handle)
        return True


def test_windows_job_is_noninherited_kill_on_close_and_uses_process_handle():
    api = FakeKernel()
    job = ProcessJob(api=api)
    job.attach(SimpleNamespace(_handle=987))
    job.close()
    job.close()
    assert api.assigned == [(42, 987)] and api.closed == [42]


@pytest.mark.parametrize("failure", ["create", "configure", "assign"])
def test_job_failures_are_explicit_and_close_allocated_handle(failure):
    api = FakeKernel(failure)
    job = None
    with pytest.raises(OSError):
        try:
            job = ProcessJob(api=api)
            job.attach(SimpleNamespace(_handle=123))
        finally:
            if job:
                job.close()
    assert api.closed == ([] if failure == "create" else [42])


def test_guard_assignment_failure_stops_only_new_owned_child(managed, monkeypatch):
    import typed_decisions.llama_routing as routing
    pool, events, processes, _ = managed
    class FailureJob:
        def attach(self, process):
            raise OSError("assignment failed")
        def close(self):
            events.append(("close_guard", "8b"))
    monkeypatch.setattr(routing, "ProcessJob", FailureJob)
    with pytest.raises(BackendError):
        pool.ensure("8b")
    assert processes[0].poll() is not None and pool.process is None and pool.job is None
    assert events == [("start", "8b"), ("terminate", "8b"), ("close_guard", "8b")]


def test_interrupted_startup_releases_child(managed, monkeypatch):
    import typed_decisions.llama_routing as routing
    pool, _, processes, _ = managed
    def interrupt():
        raise KeyboardInterrupt()
    monkeypatch.setattr(routing.time, "monotonic", interrupt)
    with pytest.raises(KeyboardInterrupt):
        pool.ensure("8b")
    assert processes[0].poll() is not None and pool.process is None and pool.job is None
