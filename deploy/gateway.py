"""Scale-to-zero HTTP gateway for the Modal SGLang backend.

Cost policy:
- GET /health never touches the GPU Server.
- GET /v1/models never touches the GPU Server.
- Only real inference POSTs are allowed to create/refresh GPU demand.

A tiny Modal Dict heartbeat maintained by the GPU container lets the gateway
advertise `idle`, `starting`, or `ready` without probing the Server URL. This
prevents Cherry Studio's background model refreshes and health checks from
accidentally waking an RTX PRO 6000.
"""

from __future__ import annotations

import json
import math
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from deploy.backend_state import mark_triggered_async, read_state_async

BACKEND_URL = os.environ["QWEN38_BACKEND_URL"].rstrip("/")
SERVED_MODEL_NAME = os.environ.get("QWEN38_SERVED_MODEL_NAME", "qwen3.8-27b")
COLD_MODEL_NAME = "cold-starting"
DEFAULT_ESTIMATE_SECONDS = max(
    1, int(os.environ.get("QWEN38_COLD_START_ESTIMATE_SECONDS", "120"))
)
OVERDUE_REMAINING_FLOOR_SECONDS = 15
RETRYABLE_BACKEND_STATUS = {502, 503, 504}

# These POST routes are user-driven inference operations and may wake the GPU.
# Read-only discovery/health endpoints are deliberately excluded.
WAKEABLE_INFERENCE_PATHS = {
    "/v1/chat/completions",
}

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


def _request_headers(request: Request) -> dict[str, str]:
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in _HOP_BY_HOP_HEADERS
    }
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


app = FastAPI(title="Qwen3.8 Modal cost-aware gateway", lifespan=lifespan)


def _cold_snapshot(state: dict[str, Any], now: float | None = None) -> dict[str, Any]:
    now = time.time() if now is None else now
    status = state.get("status", "idle")
    if status == "starting":
        try:
            started_at = float(state["started_at"])
        except (KeyError, TypeError, ValueError):
            started_at = now
        elapsed = max(0.0, now - started_at)
    else:
        elapsed = 0.0

    estimate = float(DEFAULT_ESTIMATE_SECONDS)
    if elapsed >= estimate:
        estimate = elapsed + OVERDUE_REMAINING_FLOOR_SECONDS

    estimated_total = max(1, math.ceil(estimate))
    elapsed_seconds = max(0, math.floor(elapsed))
    remaining = max(1, estimated_total - elapsed_seconds)
    return {
        "status": "cold_starting" if status == "starting" else "idle",
        "model": COLD_MODEL_NAME,
        "estimated_total_seconds": estimated_total,
        "elapsed_seconds": elapsed_seconds,
        "remaining_seconds": remaining,
        "gpu_wake_allowed": False,
    }


def _cold_message(snapshot: dict[str, Any]) -> str:
    if snapshot["status"] == "idle":
        return (
            "模型当前处于休眠状态。发送真实聊天/生成请求后会自动启动 GPU；"
            f"预计冷启动总时长约 {snapshot['estimated_total_seconds']}s。"
        )
    return (
        "模型正在冷启动，请稍候。"
        f"预计冷启动总时长约 {snapshot['estimated_total_seconds']}s，"
        f"当前已冷启动 {snapshot['elapsed_seconds']}s，"
        f"预计还需约 {snapshot['remaining_seconds']}s。"
    )


def _ready_health(state: dict[str, Any]) -> JSONResponse:
    return JSONResponse(
        {
            "status": "ready",
            "model": SERVED_MODEL_NAME,
            "gpu": "ready",
            "heartbeat_age_seconds": state.get("heartbeat_age_seconds"),
            "gpu_wake_allowed": False,
            "message": "Gateway 与 GPU 模型均已就绪。此健康检查不会续命 GPU。",
        },
        status_code=200,
    )


def _cold_health_response(state: dict[str, Any]) -> JSONResponse:
    snapshot = _cold_snapshot(state)
    return JSONResponse({**snapshot, "message": _cold_message(snapshot)}, status_code=200)


def _models_response(state: dict[str, Any]) -> JSONResponse:
    created = int(time.time())
    if state.get("status") == "ready":
        return JSONResponse(
            {
                "object": "list",
                "data": [
                    {
                        "id": SERVED_MODEL_NAME,
                        "object": "model",
                        "created": created,
                        "owned_by": "sglang",
                        "status": "ready",
                    }
                ],
                "backend_status": "ready",
                "gpu_wake_allowed": False,
            },
            status_code=200,
        )

    snapshot = _cold_snapshot(state)
    return JSONResponse(
        {
            "object": "list",
            "data": [
                {
                    "id": COLD_MODEL_NAME,
                    "object": "model",
                    "created": created,
                    "owned_by": "system",
                    "status": snapshot["status"],
                }
            ],
            "cold_start": {**snapshot, "message": _cold_message(snapshot)},
            "gpu_wake_allowed": False,
        },
        status_code=200,
    )


async def _proxy(request: Request) -> Response | None:
    """Proxy a real request; retryable upstream errors become a cold response."""

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
        return None

    if upstream.status_code in RETRYABLE_BACKEND_STATUS:
        await upstream.aclose()
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


def _chat_chunk(
    request_id: str,
    created: int,
    *,
    delta: dict[str, str] | None = None,
    finish_reason: str | None = None,
    usage: dict[str, int] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
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


def _cold_chat_response(payload: dict[str, Any], state: dict[str, Any]) -> Response:
    snapshot = _cold_snapshot(state)
    # A chat request is allowed to wake the GPU; make that explicit in metadata.
    snapshot["gpu_wake_allowed"] = True
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


async def _user_inference(request: Request) -> Response:
    """The only path family authorized to create or refresh GPU demand."""

    state = await read_state_async()
    if state.get("status") == "starting":
        payload = await _json_payload(request)
        return _cold_chat_response(payload, state)

    if state.get("status") == "ready":
        proxied = await _proxy(request)
        if proxied is not None:
            return proxied
        # A stale control-plane state can race with scale-down. The real user
        # inference request we just attempted is itself authorized to wake GPU.
        state = await mark_triggered_async()
        payload = await _json_payload(request)
        return _cold_chat_response(payload, state)

    # GPU is idle/stale. Mark the cold-start *before* sending the actual user
    # request upstream. This avoids background GET probes entirely. If Modal has
    # already kept the GPU alive, the POST succeeds immediately; if it is truly
    # scaled to zero, its expected 503 is swallowed and returned as friendly 200.
    state = await mark_triggered_async()
    proxied = await _proxy(request)
    if proxied is not None:
        return proxied
    payload = await _json_payload(request)
    return _cold_chat_response(payload, state)


async def _json_payload(request: Request) -> dict[str, Any]:
    try:
        payload = json.loads(await request.body())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


@app.get("/_gateway/health", include_in_schema=False)
async def gateway_health() -> JSONResponse:
    """Pure gateway liveness. Never reads or wakes the GPU backend."""

    return JSONResponse({"status": "gateway_ready", "gpu_wake_allowed": False})


@app.get("/health")
async def health() -> Response:
    """Read heartbeat state only. This endpoint can never wake/keep GPU alive."""

    state = await read_state_async()
    if state.get("status") == "ready":
        return _ready_health(state)
    return _cold_health_response(state)


@app.get("/v1/models")
async def models() -> Response:
    """Return a static/state-driven model list without contacting SGLang."""

    return _models_response(await read_state_async())


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    return await _user_inference(request)


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
)
async def passthrough_or_no_wake(request: Request, path: str) -> Response:
    """Unknown/read-only routes never wake GPU; selected inference POSTs may."""

    full_path = "/" + path
    if request.method == "POST" and full_path in WAKEABLE_INFERENCE_PATHS:
        return await _user_inference(request)

    state = await read_state_async()
    return JSONResponse(
        {
            "error": {
                "message": "Endpoint does not wake a sleeping GPU backend.",
                "type": "gateway_no_wake_policy",
                "backend_status": state.get("status", "idle"),
            }
        },
        status_code=404,
    )
