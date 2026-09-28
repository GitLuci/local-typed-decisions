"""Loopback API; one bounded inference worker and no remote fallback."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import json

from fastapi import FastAPI, HTTPException, Request

from .runtime import LocalDecisionModel, MODEL_ID
from .llama_server import BackendError


def create_app(model=None, threads=4, backend="laya", qwen_options=None, llama_options=None):
    pool = ThreadPoolExecutor(max_workers=1)
    gate = asyncio.Semaphore(1)

    @asynccontextmanager
    async def lifespan(app):
        if model is not None:
            app.state.model = model
        elif backend == "llama-server":
            from .llama_server import create_backend
            app.state.model = create_backend(**(llama_options or {}))
        elif backend == "qwen":
            from .qwen import QwenDecisionModel
            app.state.model = QwenDecisionModel(threads=threads, **(qwen_options or {}))
        elif backend == "laya":
            app.state.model = LocalDecisionModel(threads=threads)
        else:
            raise ValueError("unknown backend")
        try:
            yield
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
            if model is None and hasattr(app.state.model, "close"):
                app.state.model.close()

    app = FastAPI(title="Local typed decisions (experimental)", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": getattr(app.state.model, "model_id", MODEL_ID), "offline": True, "calibration": "unvalidated"}

    @app.post("/v1/systemone")
    async def systemone(request: Request):
        raw = bytearray()
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw) > 256_000:
                raise HTTPException(413, "request too large")
        try:
            body = json.loads(raw)
            if not isinstance(body, dict) or not {"state", "questions"} <= body.keys():
                raise ValueError("state and questions are required")
            allowed = {"state", "questions", "model"}
            if getattr(app.state.model, "accepts_domains", False):
                allowed.add("domains")
            if set(body) - allowed:
                raise ValueError("unknown request fields")
            model_id = getattr(app.state.model, "model_id", MODEL_ID)
            if body.get("model", model_id) not in {model_id, "local"}:
                raise ValueError(f"available model: {model_id}")
            async with gate:
                options = {"domains": body["domains"]} if "domains" in body else {}
                result = await asyncio.get_running_loop().run_in_executor(
                    pool, lambda: app.state.model.predict(body["state"], body["questions"], **options))
            result.pop("logits", None)
            return result
        except BackendError as error:
            raise HTTPException(error.status_code, str(error)) from error
        except (ValueError, TypeError, RecursionError) as error:
            raise HTTPException(422, str(error)) from error

    return app
