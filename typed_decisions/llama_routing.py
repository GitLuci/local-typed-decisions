"""Configuration-based routing and lazy ownership of local llama-server processes."""
import copy
import os
from pathlib import Path
import socket
import subprocess
import threading
import time
from urllib.parse import urlsplit

import httpx

from .llama_server import BackendError, BackendTimeout, LlamaServerDecisionModel, PROFILES
from .runtime import validate
from .process_job import ProcessJob


def canonical(mode):
    if not isinstance(mode, str) or mode not in PROFILES:
        raise ValueError("route mode must be ultra-fast, fast, medium or slow")
    return mode


def server_url(key, config):
    port = 8791 if key == "4b" else 8790
    url = config.get("url", f"http://127.0.0.1:{port}")
    if not isinstance(url, str):
        raise ValueError("server URL must be text")
    parsed = urlsplit(url)
    if (parsed.scheme != "http" or parsed.hostname not in ("localhost", "127.0.0.1", "::1")
            or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in ("", "/")):
        raise ValueError("server URL must be a loopback HTTP origin")
    if parsed.port is not None and not 1 <= parsed.port <= 65535:
        raise ValueError("invalid server port")
    return url.rstrip("/")


class ServerPool:
    """At most one owned model resident; never adopts or terminates other processes.

    No launch configuration means an externally managed server. Construction
    validates configuration only. ensure() is called after request preflight.
    """
    def __init__(self, servers):
        self.servers = copy.deepcopy(servers)
        self.process = None
        self.job = None
        self.active_key = None
        self.lock = threading.RLock()
        origins = set()
        for key, config in self.servers.items():
            if key not in ("4b", "8b") or not isinstance(config, dict):
                raise ValueError("servers must define 4b and/or 8b objects")
            if set(config) - {"url", "tokenizer_path", "launch"}:
                raise ValueError("unknown server configuration fields")
            url = server_url(key, config)
            parsed = urlsplit(url)
            # All allowed names are loopback: reserve distinct ports per model.
            endpoint = parsed.port or 80
            if endpoint in origins:
                raise ValueError("models must use distinct server ports")
            origins.add(endpoint)
            launch = config.get("launch")
            if launch is None:
                continue
            if not isinstance(launch, dict) or set(launch) - {"executable", "model_path", "gpu_layers", "threads", "startup_timeout"}:
                raise ValueError("invalid launch configuration")
            if not {"executable", "model_path"} <= launch.keys():
                raise ValueError("launch requires executable and model_path")
            expected = PROFILES["ultra-fast" if key == "4b" else "fast"]["model"]
            if Path(launch["model_path"]).name != expected:
                raise ValueError(f"{key} requires {expected}")
            for name, default, low, high in (("gpu_layers", 99 if key == "4b" else 30, 0, 99), ("threads", 4, 1, 8),
                                              ("startup_timeout", 120, 1, 600)):
                value = launch.get(name, default)
                if type(value) is not int or not low <= value <= high:
                    raise ValueError(f"invalid launch {name}")

    def _stop(self):
        if self.process is None:
            if self.job is not None:
                self.job.close()
                self.job = None
            return
        process = self.process
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired as error:
                    raise BackendError("owned llama-server did not stop; refusing another load") from error
        self.process = None
        self.active_key = None
        if self.job is not None:
            self.job.close()
            self.job = None

    def close(self):
        with self.lock:
            self._stop()

    @staticmethod
    def _port_free(url):
        parts = urlsplit(url)
        try:
            connection = socket.create_connection((parts.hostname, parts.port or 80), timeout=0.2)
        except OSError:
            return True
        connection.close()
        return False

    def ensure(self, key):
        with self.lock:
            if self.active_key == key and self.process is not None and self.process.poll() is None:
                return
            config = self.servers[key]
            launch = config.get("launch")
            if launch is None:
                self._stop()
                return  # external process stays entirely under caller ownership
            url = server_url(key, config)
            executable, model = Path(launch["executable"]), Path(launch["model_path"])
            if not executable.is_file() or not model.is_file():
                raise BackendError("llama-server executable or GGUF is missing")
            if not self._port_free(url):
                raise BackendError("configured llama-server port is occupied; refusing to adopt it")
            self._stop()  # release owned model before allocating the other model
            parts = urlsplit(url)
            layers = launch.get("gpu_layers", 99 if key == "4b" else 30)
            args = [str(executable.resolve()), "-m", str(model.resolve()), "-ngl",
                    str(layers), "-c", "2048", "-t",
                    str(launch.get("threads", 4)), "--parallel", "1", "--host", parts.hostname,
                    "--port", str(parts.port or 80)]
            # b11205 spelling: --no-mmap is not supported in this build.
            if layers < 37:  # both locked Qwen3 models: 36 blocks plus output
                args.extend(["--load-mode", "none"])
            try:
                self.job = ProcessJob()
                self.process = subprocess.Popen(args, shell=False, stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                self.job.attach(self.process)
                self.active_key = key
                deadline = time.monotonic() + launch.get("startup_timeout", 120)
                with httpx.Client(timeout=1, trust_env=False, follow_redirects=False) as client:
                    while time.monotonic() < deadline:
                        if self.process.poll() is not None:
                            raise BackendError("owned llama-server exited during startup")
                        try:
                            if client.get(url + "/health").status_code == 200:
                                return
                        except httpx.HTTPError:
                            pass
                        time.sleep(0.1)
                raise BackendTimeout("llama-server startup timed out")
            except BaseException as error:
                self._stop()
                if isinstance(error, OSError):
                    raise BackendError("could not launch local llama-server") from error
                raise


class RoutedDecisionModel:
    """Routes are data, never inferred from another question or its answer."""
    def __init__(self, mode, servers, routes=None, default_domain=None, timeout=3600, seed=20260925,
                 n_probs=None, *, backend_factory=None, server_pool=None):
        self.mode = mode if mode == "routed" else canonical(mode)
        self.accepts_domains = self.mode == "routed"
        self.routes = copy.deepcopy(routes or {})
        if not isinstance(self.routes, dict):
            raise ValueError("routes must map domain names to modes")
        if self.accepts_domains:
            if not self.routes or any(not isinstance(k, str) or not k for k in self.routes):
                raise ValueError("routed requires an explicit nonempty domain map")
            self.routes = {domain: canonical(mode) for domain, mode in self.routes.items()}
            if default_domain is not None and (not isinstance(default_domain, str) or default_domain not in self.routes):
                raise ValueError("default_domain must occur in routes")
        elif routes is not None or default_domain is not None:
            raise ValueError("routes/default_domain require mode routed")
        self.default_domain = default_domain
        if not isinstance(servers, dict) or not servers:
            raise ValueError("servers must be a nonempty object")
        self.servers = copy.deepcopy(servers)
        modes = set(self.routes.values()) if self.accepts_domains else {self.mode}
        required = {PROFILES[mode]["model_key"] for mode in modes}
        if not required <= self.servers.keys():
            raise ValueError("missing server configuration for a routed model")
        # Validate all server settings even with a test pool supplied.
        validated_pool = ServerPool(self.servers)
        self.pool = server_pool if server_pool is not None else validated_pool
        self.factory = backend_factory or LlamaServerDecisionModel
        self.options = {"timeout": timeout, "seed": seed, "n_probs": n_probs}
        self.backends = {}
        self.lock = threading.RLock()
        self.model_id = f"local-typed-decisions-{self.mode}-llama-server"
        self.closed = False

    def _backend(self, mode):
        if mode not in self.backends:
            key = PROFILES[mode]["model_key"]
            config = self.servers[key]
            # Modes on the same model share a tokenizer, client pool owns weights.
            peer = next((b for m, b in self.backends.items() if PROFILES[m]["model_key"] == key), None)
            self.backends[mode] = self.factory(mode=mode, url=server_url(key, config),
                tokenizer_path=config.get("tokenizer_path"), **self.options,
                **({"builder": peer.builder} if peer else {}))
        return self.backends[mode]

    def predict(self, state, questions, *, domains=None):
        validate(state, questions, max_options=255)
        if domains is None:
            domains = {}
        if not isinstance(domains, dict) or set(domains) - set(questions):
            raise ValueError("domains must map only question IDs in this request")
        if domains and not self.accepts_domains:
            raise ValueError("domains require mode routed")
        groups, selections = {}, {}
        for qid, question in questions.items():
            domain = domains.get(qid, self.default_domain)
            if self.accepts_domains:
                if not isinstance(domain, str) or domain not in self.routes:
                    raise ValueError(f"{qid}: missing or unmapped domain")
                mode = self.routes[domain]
            else:
                mode = self.mode
            groups.setdefault(mode, {})[qid] = question
            selections[qid] = {"domain": domain, "mode": mode, "model_key": PROFILES[mode]["model_key"]}
        with self.lock:
            if self.closed:
                raise BackendError("runtime is closed")
            # All domains/schema/budgets checked before launching any process.
            prepared = {mode: self._backend(mode).prepare(state, qs) for mode, qs in groups.items()}
            answers, errors, input_tokens, output_tokens = {}, {}, 0, 0
            details = {}
            # Group by physical model so two modes do not cause repeated reloads.
            for mode in sorted(groups, key=lambda m: PROFILES[m]["model_key"]):
                self.pool.ensure(PROFILES[mode]["model_key"])
                result = self.backends[mode]._predict_prepared(prepared[mode])
                answers.update(result["answers"])
                errors.update(result.get("errors", {}))
                input_tokens += result["usage"]["input_tokens"]
                output_tokens += result["usage"]["output_tokens"]
                details[mode] = {"model": result["model"], **result["metadata"]}
        result = {"model": self.model_id, "answers": {qid: answers[qid] for qid in questions if qid in answers},
                "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
                "metadata": {"backend": "llama-server", "mode": self.mode, "routing": selections,
                             "modes": details, "question_execution": "independent_serial_grouped_by_model",
                             "calibration": "unvalidated", "shared_state_encoding": False}}
        if errors:
            result["errors"] = {qid: errors[qid] for qid in questions if qid in errors}
        return result

    def close(self):
        with self.lock:
            try:
                self.pool.close()
            finally:
                for backend in self.backends.values():
                    backend.close()
                self.closed = True
