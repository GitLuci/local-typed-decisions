"""Use pretrained decision heads directly: no decoding and no cloud inference."""
import json
import math
import os
from pathlib import Path
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "local-typed-decisions-laya-multilingual"


def configure(threads=4):
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = str(threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["USE_TF"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HOME"] = str(ROOT / ".cache" / "huggingface")


def validate(state, questions, *, max_options=20):
    if not isinstance(state, (str, dict, list)):
        raise ValueError("state must be text, an object or an array")
    json.dumps(state, allow_nan=False)
    if not isinstance(questions, dict) or not 1 <= len(questions) <= 32:
        raise ValueError("questions must contain 1 to 32 entries")
    normalized = {}
    for qid, question in questions.items():
        if not isinstance(qid, str) or not qid or not isinstance(question, dict):
            raise ValueError("each question needs a nonempty string ID and an object definition")
        q = dict(question)
        if set(q) - {"type", "instructions", "criteria"}:
            raise ValueError(f"{qid}: unknown question fields")
        t = q.get("type")
        if t not in {"choice", "score", "noul"}:
            raise ValueError(f"{qid}: use choice, score or noul")
        if not isinstance(q.get("instructions"), (str, dict, list)) or not q["instructions"]:
            raise ValueError(f"{qid}: instructions must be nonempty text, object or array")
        c = q.get("criteria")
        if t == "choice":
            if not isinstance(c, dict) or not 2 <= len(c) <= max_options:
                raise ValueError(f"{qid}: this checkpoint supports 2 to {max_options} choice options")
            if any(not isinstance(k, str) or not k for k in c):
                raise ValueError(f"{qid}: option keys must be nonempty strings")
            if any(v is not None and not isinstance(v, (str, dict, list)) for v in c.values()):
                raise ValueError(f"{qid}: choice descriptions must be text, objects, arrays or null")
        elif t == "score":
            if not isinstance(c, list) or not 2 <= len(c) <= 10:
                raise ValueError(f"{qid}: score needs 2 to 10 ordered descriptions")
            if any(not isinstance(v, (str, dict, list)) for v in c):
                raise ValueError(f"{qid}: score descriptions must be text, objects or arrays")
        elif c is None:
            q.pop("criteria", None)
        elif isinstance(c, dict) and set(c) <= {"false", "true"}:
            if any(not isinstance(v, (str, dict, list)) for v in c.values()):
                raise ValueError(f"{qid}: noul descriptions must be text, objects or arrays")
        else:
            if not isinstance(c, (str, dict, list)):
                raise ValueError(f"{qid}: noul criteria must be text, an object, an array or null")
            # Preserve legacy local clarifications; Laya only accepts false/true maps.
            q["instructions"] = {"question": q["instructions"], "clarification": c}
            q.pop("criteria")
        json.dumps(q, allow_nan=False)
        normalized[qid] = q
    return normalized


def answer_from_probabilities(question, probabilities):
    p = [float(v) for v in probabilities]
    if not p or any(not math.isfinite(v) or v < 0 for v in p) or sum(p) <= 0:
        raise ValueError("invalid model distribution")
    total = sum(p)
    p = [v / total for v in p]
    t = question["type"]
    expected = 2 if t == "noul" else len(question["criteria"])
    if len(p) != expected:
        raise ValueError("model distribution does not match the supplied options")
    if t == "noul":
        return {"type": t, "noul": p[1]}
    concentration = max(0., min(1., 1 + sum(v * math.log(v) for v in p if v) / math.log(len(p))))
    keys = list(question["criteria"]) if t == "choice" else [str(i) for i in range(len(p))]
    result = {"type": t, "probabilities": dict(zip(keys, p)), "confidence": concentration}
    if t == "choice":
        result["choice"] = keys[max(range(len(p)), key=p.__getitem__)]
    else:
        result["score"] = sum(i * v for i, v in enumerate(p))
        result["legend"] = dict(zip(keys, question["criteria"]))
    return result


class LocalDecisionModel:
    model_id = MODEL_ID

    def __init__(self, model_path=None, threads=4, max_len=1024):
        if not 1 <= threads <= 4 or not 64 <= max_len <= 1024:
            raise ValueError("local CPU profile: 1-4 threads, 64-1024 tokens")
        configure(threads)
        import torch
        from laya import Agent
        torch.set_num_threads(threads)
        self.torch = torch
        lock = json.loads((ROOT / "model.lock.json").read_text(encoding="utf-8"))
        path = Path(model_path or ROOT / lock["path"]).resolve()
        if not (path / "model.safetensors").exists():
            raise FileNotFoundError("Model missing. Run scripts/download_model.py first.")
        self.agent = Agent(str(path), device="cpu", compile=False)
        self.max_len = max_len
        self.head_max_len = int(self.agent.cfg.get("head_max_len", 256))
        self.lock = threading.Lock()
        self.revision = lock["revision"]

    def _prepare(self, state, questions):
        from laya.common import render_options, serialize_state
        tok = self.agent.tok
        state_text = serialize_state(state)
        if tok.mask_token in state_text:
            raise ValueError("state contains a reserved model marker")
        state_n = len(tok(state_text, add_special_tokens=False)["input_ids"])
        internal = {}
        for qid, question in questions.items():
            q = self.agent._to_internal(question)
            options = render_options(q)
            if any(tok.mask_token in s for s in [q["ins"], *options]):
                raise ValueError(f"{qid}: reserved model marker in question")
            sizes = [len(tok(" " + s, add_special_tokens=False)["input_ids"]) for s in options]
            head_n = len(tok(f'{q["t"]} question: {q["ins"]}', add_special_tokens=False)["input_ids"])
            option_n = sum(n + 1 for n in sizes)
            if max(sizes) > 48 or option_n + max(16, head_n) > self.head_max_len:
                raise ValueError(f"{qid}: question/options exceed checkpoint budget; shorten descriptions")
            if state_n + head_n + option_n + 4 > self.max_len:
                raise ValueError(f"{qid}: input exceeds {self.max_len} tokens; no silent truncation")
            internal[qid] = q
        items = self.agent._encode_state(state, list(questions), internal, self.max_len, self.head_max_len)
        return items, internal

    def predict(self, state, questions):
        questions = validate(state, questions)
        from laya.common import collate_items, QTYPES, temp_bucket
        started = time.perf_counter()
        with self.lock, self.torch.inference_mode():
            items, internal = self._prepare(state, questions)
            batch = collate_items([items], self.agent.tok.pad_token_id)
            logits, _ = self.agent._forward(batch)
            answers, raw = {}, {}
            for i, (qid, q) in enumerate(questions.items()):
                k = len(items[i]["markers"])
                values = [float(v) for v in logits[i, :k]]
                bucket = temp_bucket(QTYPES[q["type"]], k)
                temperature = self.agent.temperature_by_options.get(bucket, self.agent.temperature[QTYPES[q["type"]]])
                z = [v / temperature for v in values]
                weights = [math.exp(v - max(z)) for v in z]
                answers[qid] = answer_from_probabilities(q, weights)
                raw[qid] = values
        return {
            "model": MODEL_ID, "answers": answers,
            "usage": {"input_tokens": sum(len(i["ids"]) for i in items), "output_tokens": 0},
            "metadata": {"backend": "laya", "revision": self.revision, "device": "cpu",
                         "elapsed_ms": (time.perf_counter() - started) * 1000,
                         "calibration": "unvalidated", "confidence_definition": "1 - normalized Shannon entropy",
                         "shared_state_encoding": False},
            "logits": raw,
        }
