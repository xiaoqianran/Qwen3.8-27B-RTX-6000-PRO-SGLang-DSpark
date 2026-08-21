"""Always-on HTTP gateway for the Modal SGLang backend.

The gateway keeps Cherry Studio/OpenAI-compatible clients away from Modal's
transient 502/503 responses while the GPU Server scales from zero. During a
cold start it returns valid OpenAI-shaped responses using the synthetic
``cold-starting`` model. Once the backend is healthy it transparently proxies
requests to SGLang.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

BACKEND_URL = os.environ["QWEN38_BACKEND_URL"].rstrip("/")
COLD_MODEL_NAME = "cold-starting"
DEFAULT_ESTIMATE_SECONDS = max(
    1, int(os.environ.get("QWEN38_COLD_START_ESTIMATE_SECONDS", "120"))
)
RETRYABLE_BACKEND_STATUS = {502, 503, 504}
HISTORY_SIZE = 8
OVERDUE_REMAINING_FLOOR_SECONDS = 15

_HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
_RESPONSE_HEADERS = {
    "cache-control",
    "content-disposition",
    "content-type",
    "x-request-id",
}


class ColdStartTracker:
    """Track one GPU cold-start cycle and recent observed durations."""

    def __init__(self, default_estimate_seconds: int = DEFAULT_ESTIMATE_SECONDS):
        self.default_estimate_seconds = default_estimate_seconds
        self._lock = threading.Lock()
        self._status = "unknown"
        self._started_at: float | None = None
        self._history: deque[float] = deque(maxlen=HISTORY_SIZE)

    def mark_starting(self, now: float | None = None) -> None:
        now = time.time() if now is None else now
        with self._lock:
            if self._status != "cold_starting":
                self._status = "cold_starting"
                self._started_at = now

    def mark_ready(self, now: float | None = None) -> None:
        now = time.time() if now is None else now
        with self._lock:
            if self._status == "cold_starting" and self._started_at is not None:
                duration = now - self._started_at
                if 1 <= duration <= 30 * 60:
                    self._history.append(duration)
            self._status = "ready"
            self._started_at = None

    def snapshot(self, now: float | None = None) -> dict[str, int | float | str]:
        now = time.time() if now is None else now
        with self._lock:
            if self._status != "cold_starting" or self._started_at is None:
                self._status = "cold_starting"
                self._started_at = now

            elapsed = max(0.0, now - self._started_at)
            if self._history:
                estimate = statistics.median(self._history)
            else:
                estimate = float(self.default_estimate_seconds)

            # If a boot takes longer than the historical estimate, do not show
            # a misleading zero-second remainder while it is still starting.
            estimate = max(
                estimate,
                elapsed + OVERDUE_REMAINING_FLOOR_SECONDS
                if elapsed >= estimate
                else estimate,
            )
            estimate_seconds = max(1, math.ceil(estimate))
            elapsed_seconds = max(0, math.floor(elapsed))
            remaining_seconds = max(1, estimate_seconds - elapsed_seconds)

            return {
                "status": "cold_starting",
                "model": COLD_MODEL_NAME,
                "estimated_total_seconds": estimate_seconds,
                "elapsed_seconds": elapsed_seconds,
                "remaining_seconds": remaining_seconds,
                "observed_cold_starts": len(self._history),
            }


tracker = ColdStartTracker()


def _cold_message(snapshot: dict[str, int | float | str]) -> str:
    return (
        "模型正在冷启动，请稍候。"
        f"预计冷启动总时长约 {snapshot['estimated_total_seconds']}s，"
        f"当前已冷启动 {snapshot['elapsed_seconds']}s，"
        f"预计还需约 {snapshot['remaining_seconds']}s。"
    )


def _request_headers(request: Request) -> dict[str, str]:
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in _HOP_BY_HOP_HEADERS
    }
    # Keep the backend leg explicitly HTTP/1.1-friendly and uncompressed so
    # raw streaming bytes can be relayed without content-encoding ambiguity.
    headers["accept-encoding"] = "identity"
    return headers


def _response_headers(response: httpx.Response) -> dict[str, str]:
    return {
        key: value
        for key, value in response.headers.items()
        if key.lower() in _RESPONSE_HEADERS
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.backend = httpx.AsyncClient(
        http2=False,
        timeout=httpx.Timeout(connect=10.0, read=None, write=60.0, pool=10.0),
        follow_redirects=False,
    )
    try:
        yield
    finally:
        await app.state.backend.aclose()


app = FastAPI(title="Qwen3.8 Modal cold-start gateway", lifespan=lifespan)


async def _backend_ready(request: Request) -> bool:
    client: httpx.AsyncClient = request.app.state.backend
    try:
        response = await client.get(
            f"{BACKEND_URL}/health",
            timeout=httpx.Timeout(2.0),
        )
        ready = response.status_code == 200
    except httpx.HTTPError:
        ready = False

    if ready:
        tracker.mark_ready()
    else:
        # The probe itself is intentional: hitting the zero-scaled Modal Server
        # triggers allocation while this CPU gateway can immediately answer 200.
        tracker.mark_starting()
    return ready


async def _proxy(request: Request) -> Response | None:
    """Proxy to SGLang; return None if the backend fell back to cold-starting."""

    client: httpx.AsyncClient = request.app.state.backend
    body = await request.body()
    query = request.url.query
    target = f"{BACKEND_URL}{request.url.path}"
    if query:
        target = f"{target}?{query}"

    upstream_request = client.build_request(
        request.method,
        target,
        headers=_request_headers(request),
        content=body,
    )
    try:
        upstream = await client.send(upstream_request, stream=True)
    except httpx.HTTPError:
        tracker.mark_starting()
        return None

    if upstream.status_code in RETRYABLE_BACKEND_STATUS:
        await upstream.aclose()
        tracker.mark_starting()
        return None

    async def relay() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()

    return StreamingResponse(
        relay(),
        status_code=upstream.status_code,
        headers=_response_headers(upstream),
    )


def _cold_health_response() -> JSONResponse:
    snapshot = tracker.snapshot()
    return JSONResponse({**snapshot, "message": _cold_message(snapshot)}, status_code=200)


def _cold_models_response() -> JSONResponse:
    snapshot = tracker.snapshot()
    created = int(time.time())
    return JSONResponse(
        {
            "object": "list",
            "data": [
                {
                    "id": COLD_MODEL_NAME,
                    "object": "model",
                    "created": created,
                    "owned_by": "system",
                    "status": "cold_starting",
                }
            ],
            "cold_start": {
                **snapshot,
                "message": _cold_message(snapshot),
            },
        },
        status_code=200,
    )


def _chat_chunk(
    request_id: str,
    created: int,
    *,
    delta: dict[str, str] | None = None,
    finish_reason: str | None = None,
    usage: dict[str, int] | None = None,
) -> dict:
    payload = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": COLD_MODEL_NAME,
        "choices": (
            []
            if usage is not None
            else [
                {
                    "index": 0,
                    "delta": delta or {},
                    "finish_reason": finish_reason,
                }
            ]
        ),
    }
    if usage is not None:
        payload["usage"] = usage
    return payload


def _cold_chat_response(payload: dict) -> Response:
    snapshot = tracker.snapshot()
    message = _cold_message(snapshot)
    request_id = f"chatcmpl-cold-{uuid.uuid4().hex[:16]}"
    created = int(time.time())
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    if not payload.get("stream", False):
        return JSONResponse(
            {
                "id": request_id,
                "object": "chat.completion",
                "created": created,
                "model": COLD_MODEL_NAME,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": message},
                        "finish_reason": "stop",
                    }
                ],
                "usage": usage,
                "cold_start": snapshot,
            },
            status_code=200,
        )

    include_usage = bool(
        isinstance(payload.get("stream_options"), dict)
        and payload["stream_options"].get("include_usage")
    )

    async def events() -> AsyncIterator[str]:
        first = _chat_chunk(
            request_id,
            created,
            delta={"role": "assistant", "content": message},
        )
        final = _chat_chunk(request_id, created, finish_reason="stop")
        yield f"data: {json.dumps(first, ensure_ascii=False, separators=(',', ':'))}\n\n"
        yield f"data: {json.dumps(final, ensure_ascii=False, separators=(',', ':'))}\n\n"
        if include_usage:
            usage_chunk = _chat_chunk(request_id, created, usage=usage)
            yield (
                "data: "
                f"{json.dumps(usage_chunk, ensure_ascii=False, separators=(',', ':'))}"
                "\n\n"
            )
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        events(),
        status_code=200,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/_gateway/health", include_in_schema=False)
async def gateway_health() -> JSONResponse:
    """Local gateway liveness check that deliberately does not wake the GPU."""

    return JSONResponse({"status": "gateway_ready"})


@app.get("/health")
async def health(request: Request) -> Response:
    if await _backend_ready(request):
        proxied = await _proxy(request)
        if proxied is not None:
            return proxied
    return _cold_health_response()


@app.get("/v1/models")
async def models(request: Request) -> Response:
    if await _backend_ready(request):
        proxied = await _proxy(request)
        if proxied is not None:
            return proxied
    return _cold_models_response()


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    try:
        payload = json.loads(await request.body())
        if not isinstance(payload, dict):
            payload = {}
    except (json.JSONDecodeError, UnicodeDecodeError):
        payload = {}

    if await _backend_ready(request):
        proxied = await _proxy(request)
        if proxied is not None:
            return proxied
    return _cold_chat_response(payload)


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
)
async def passthrough_or_cold(request: Request, path: str) -> Response:
    """Proxy all other SGLang routes when ready; stay HTTP 200 while cold."""

    del path
    if await _backend_ready(request):
        proxied = await _proxy(request)
        if proxied is not None:
            return proxied
    snapshot = tracker.snapshot()
    return JSONResponse({**snapshot, "message": _cold_message(snapshot)}, status_code=200)
