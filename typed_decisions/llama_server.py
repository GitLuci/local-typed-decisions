"""HTTP decision core for llama-server b11205, without cross-question history.

The optional factory delegates lazy process management to llama_routing.
"""
import hashlib
import json
import math
from pathlib import Path
import threading
from urllib.parse import urlsplit

import httpx

from .qwen import QwenDecisionModel, SYSTEM, option_descriptions, render
from .runtime import ROOT, answer_from_probabilities, validate

CLOSE = "\n</think>\n\n"
PROFILES = {
    "ultra": {"model": "qwen3-4b-Q8_0.gguf", "tokenizer": "models/qwen3-4b", "port": 8791, "model_key": "4b", "budget": 0},
    "fast": {"model": "qwen3-8b-Q8_0.gguf", "tokenizer": "models/qwen3-8b", "port": 8790, "model_key": "8b", "budget": 0},
    "medium": {"model": "qwen3-4b-Q8_0.gguf", "tokenizer": "models/qwen3-4b", "port": 8791, "model_key": "4b", "budget": 1024},
    "slow": {"model": "qwen3-8b-Q8_0.gguf", "tokenizer": "models/qwen3-8b", "port": 8790, "model_key": "8b", "budget": 128},
}


class BackendError(RuntimeError):
    """An upstream/protocol failure, not an invalid Jev request."""
    status_code = 502


class BackendTimeout(BackendError):
    status_code = 504


def load_config(path):
    path = Path(path)
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    allowed = {"mode", "url", "tokenizer_path", "timeout", "seed", "n_probs", "servers", "routes", "default_domain"}
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError("unknown llama configuration fields")
    if "mode" not in config:
        raise ValueError("llama config requires mode")
    if "tokenizer_path" in config:
        config["tokenizer_path"] = (path.resolve().parent / config["tokenizer_path"]).resolve()
    if "servers" in config:
        if not isinstance(config["servers"], dict):
            raise ValueError("servers must be a model-keyed object")
        for server in config["servers"].values():
            if not isinstance(server, dict):
                raise ValueError("each server configuration must be an object")
            if "tokenizer_path" in server:
                server["tokenizer_path"] = (path.resolve().parent / server["tokenizer_path"]).resolve()
            if "launch" in server:
                if not isinstance(server["launch"], dict):
                    raise ValueError("launch must be an object")
                for key in ("executable", "model_path"):
                    if key in server["launch"]:
                        server["launch"][key] = (path.resolve().parent / server["launch"][key]).resolve()
    return config


def create_backend(**config):
    if config.get("mode") == "routed" or "servers" in config:
        from .llama_routing import RoutedDecisionModel
        return RoutedDecisionModel(**config)
    return LlamaServerDecisionModel(**config)


def first_position_logprobs(response):
    """Accept b11205 top_logprobs/top_probs and legacy probs, by token ID."""
    positions = response.get("completion_probabilities") or response.get("probs")
    if not isinstance(positions, list) or not positions or not isinstance(positions[0], dict):
        raise BackendError("llama-server returned no first-position probabilities")
    first = positions[0]
    candidates = first.get("top_logprobs") or first.get("top_probs") or first.get("probs") or []
    if not isinstance(candidates, list):
        raise BackendError("invalid candidate probability list")
    result = {}
    for candidate in [*candidates, first]:
        if not isinstance(candidate, dict):
            raise BackendError("invalid probability candidate")
        if "id" not in candidate:
            continue  # never identify tokens by display text (spaces/bytes can differ)
        tid = candidate["id"]
        if type(tid) is not int or tid < 0:
            raise BackendError("invalid probability token ID")
        if "logprob" in candidate:
            value = candidate["logprob"]
            if type(value) not in (int, float) or not math.isfinite(value) or value > 0:
                raise BackendError("invalid token logprob")
        elif "prob" in candidate:
            p = candidate["prob"]
            if type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1:
                raise BackendError("invalid token probability")
            value = math.log(p) if p else -math.inf
        else:
            continue
        if tid in result and result[tid] != value:
            if not math.isclose(result[tid], value, abs_tol=1e-6):
                raise BackendError("conflicting probabilities for one token")
        result[tid] = value
    if not result:
        raise BackendError("no probabilities indexed by token ID")
    return result


class PromptBuilder:
    """Only the local tokenizer; same prompt/aliases as the measured Qwen path."""
    def __init__(self, path=None, *, tokenizer=None):
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
        self.tok = tokenizer
        self.aliases, self.alias_token_ids = QwenDecisionModel._aliases(self)
        self.close_ids = self.encode(CLOSE)
        # Verify aliases at the actual answer boundary, not just in isolation.
        suffix = self.tok.apply_chat_template(
            [{"role": "user", "content": "Q"}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False).split("<|im_start|>assistant")[-1]
        ids = self.encode(suffix)
        if any(self.encode(suffix + alias) != ids + [tid]
               for alias, tid in zip(self.aliases, self.alias_token_ids)):
            raise ValueError("tokenizer option codes are not single tokens at the answer boundary")

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False)

    def build(self, state, question, thinking):
        for special in self.tok.all_special_tokens:
            if special in render(state) or special in render(question):
                raise ValueError("input contains a reserved tokenizer control token")
        descriptions = option_descriptions(question)
        options = "\n".join(f"{self.aliases[i]}: {s}" for i, s in enumerate(descriptions))
        content = (f"STATE:\n{render(state)}\n\nQUESTION:\n{render(question['instructions'])}"
                   f"\n\nOPTIONS:\n{options}\n\nAnswer with the letter code only.")
        text = self.tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}],
            tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
        return text, self.encode(text), self.alias_token_ids[:len(descriptions)]


class LlamaServerDecisionModel:
    def __init__(self, mode="fast", url=None, tokenizer_path=None, timeout=3600,
                 seed=20260925, n_probs=None, *, builder=None, transport=None):
        if not isinstance(mode, str) or mode not in PROFILES:
            raise ValueError("mode must be ultra, fast, medium or slow")
        if n_probs is None:
            n_probs = 256 if PROFILES[mode]["budget"] else 20
        if type(seed) is not int or not 0 <= seed < 2**32 - 1:
            raise ValueError("seed must be an integer in [0, 2**32-2]")
        if type(n_probs) is not int or not 1 <= n_probs <= 200000:
            raise ValueError("n_probs must be an integer in [1, 200000]")
        if type(timeout) not in (float, int) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        self.profile = PROFILES[mode]
        url = url or f"http://127.0.0.1:{self.profile['port']}"
        parts = urlsplit(url)
        if (parts.scheme != "http" or parts.hostname not in ("127.0.0.1", "localhost", "::1")
                or parts.username or parts.password or parts.query or parts.fragment or parts.path not in ("", "/")):
            raise ValueError("llama-server URL must be a plain loopback HTTP origin")
        self.builder = builder or PromptBuilder(tokenizer_path or ROOT / self.profile["tokenizer"])
        self.client = httpx.Client(base_url=url.rstrip("/"), timeout=timeout, trust_env=False,
                                   follow_redirects=False, transport=transport)
        self.mode, self.seed, self.n_probs = mode, seed, n_probs
        self.model_id = f"local-typed-decisions-{self.profile['model'][:-5].lower()}-{mode}-llama-server"
        self.lock = threading.Lock()

    def close(self):
        self.client.close()

    def _request(self, method, path, body=None):
        try:
            response = self.client.request(method, path, json=body)
            response.raise_for_status()
            result = response.json()
        except httpx.TimeoutException as error:
            raise BackendTimeout("llama-server timed out") from error
        except (httpx.HTTPError, ValueError) as error:
            raise BackendError("llama-server unavailable or returned invalid HTTP/JSON") from error
        if not isinstance(result, dict) or "error" in result:
            raise BackendError("llama-server returned an invalid response")
        return result

    def _check_server(self):
        props = self._request("GET", "/props")
        name = str(props.get("model_path", "")).replace("\\", "/").rsplit("/", 1)[-1]
        if name != self.profile["model"]:
            raise BackendError(f"mode {self.mode} requires {self.profile['model']}")
        settings = props.get("default_generation_settings", {})
        ctx = settings.get("n_ctx") if isinstance(settings, dict) else None
        if type(ctx) is not int or ctx < 2048:
            raise BackendError("llama-server requires at least 2048 tokens per slot")
        return props

    def _tokenize_check(self, text, ids):
        result = self._request("POST", "/tokenize", {"content": text, "add_special": False, "parse_special": True})
        if result.get("tokens") != ids:
            raise BackendError("local tokenizer and llama-server token IDs differ")

    def _complete(self, ids, count, seed, *, thinking=False, size=0):
        body = {"prompt": ids, "n_predict": count, "seed": seed, "cache_prompt": False,
                "stream": False, "n_probs": 0 if thinking else max(self.n_probs, size),
                "post_sampling_probs": False, "temperature": 0.6 if thinking else 0.0,
                "top_p": 0.95 if thinking else 1.0, "top_k": 20 if thinking else 0,
                "min_p": 0.0, "repeat_penalty": 1.0, "presence_penalty": 0.0,
                "frequency_penalty": 0.0, "stop": ["</think>"] if thinking else []}
        result = self._request("POST", "/completion", body)
        if result.get("truncated"):
            raise BackendError("llama-server truncated the context")
        for key in ("tokens_evaluated", "tokens_predicted"):
            if type(result.get(key)) is not int or result[key] < 0:
                raise BackendError(f"llama-server returned invalid {key}")
        return result

    def prepare(self, state, questions):
        """Validate/tokenize a whole batch without HTTP or loading any weights."""
        normalized = validate(state, questions, max_options=255)
        budget = self.profile["budget"]
        thinking = budget > 0
        plans = []
        # Prepare and bound EVERY question before any inference (no partial batch).
        for qid, q in normalized.items():
            text, ids, letters = self.builder.build(state, q, thinking)
            reserve = budget + len(self.builder.close_ids) + 1 if thinking else 1
            if len(ids) + reserve > 2048:
                raise ValueError(f"{qid}: prompt plus generation budget exceeds 2048 tokens")
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            seed = (self.seed + int.from_bytes(digest[:4], "big")) % (2**32 - 1)
            plans.append((qid, q, text, ids, letters, seed))
        return plans

    def predict(self, state, questions):
        return self._predict_prepared(self.prepare(state, questions))

    def _predict_prepared(self, plans):
        budget = self.profile["budget"]
        thinking = budget > 0
        with self.lock:
            props = self._check_server()
            for _, _, text, ids, _, _ in plans:
                self._tokenize_check(text, ids)
            answers, errors, input_tokens, output_tokens, forced = {}, {}, 0, 0, []
            for qid, q, text, ids, letters, seed in plans:
                if thinking:
                    thought = self._complete(ids, budget, seed, thinking=True)
                    content = thought.get("content")
                    if not isinstance(content, str):
                        raise BackendError("llama-server returned invalid thought text")
                    input_tokens += thought["tokens_evaluated"]
                    output_tokens += thought["tokens_predicted"]
                    if thought.get("stop_type") != "word" and not thought.get("stopped_word"):
                        forced.append(qid)
                    text = text + content.split("</think>", 1)[0].rstrip() + CLOSE
                    ids = self.builder.encode(text)
                    if len(ids) + 1 > 2048:
                        raise BackendError("thought/readout exceeds context budget")
                    self._tokenize_check(text, ids)
                response = self._complete(ids, 1, seed, size=len(letters))
                input_tokens += response["tokens_evaluated"]
                output_tokens += response["tokens_predicted"]
                probs = first_position_logprobs(response)
                missing = [i for i, tid in enumerate(letters) if tid not in probs]
                if missing:
                    keys = (list(q["criteria"]) if q["type"] == "choice" else
                            [str(i) for i in range(len(letters))] if q["type"] == "score" else ["false", "true"])
                    errors[qid] = {"code": "missing_option_probabilities",
                                   "message": "Option probabilities are missing; increase n_probs in config.",
                                   "missing_options": [keys[i] for i in missing],
                                   "n_probs": max(self.n_probs, len(letters))}
                    continue
                values = [probs[tid] for tid in letters]
                maximum = max(values)
                if not math.isfinite(maximum):
                    raise BackendError("all option probabilities are zero")
                answers[qid] = answer_from_probabilities(q, [math.exp(v - maximum) for v in values])
        result = {"model": self.model_id, "answers": answers,
                "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
                "metadata": {"backend": "llama-server", "mode": self.mode, "thinking": thinking,
                             "build_info": props.get("build_info", "unknown"), "forced_thinking_close": forced,
                             "question_execution": "independent_serial", "shared_state_encoding": False,
                             "calibration": "unvalidated", "confidence_definition": "1 - normalized Shannon entropy"}}
        if errors:
            result["errors"] = errors
        return result
