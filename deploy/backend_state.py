"""Shared, non-waking runtime state for the Modal GPU backend.

The public CPU gateway must be able to answer health/model-list requests without
calling the GPU Server URL, because *any* call to a zero-scaled Modal Server can
create GPU demand. A tiny Modal Dict stores lifecycle state instead. The GPU
container refreshes a heartbeat while alive; stale state is treated as idle.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import modal

STATE_DICT_NAME = "qwen38-27b-runtime-state"
STATE_KEY = "backend"
HEARTBEAT_INTERVAL_SECONDS = 10
HEARTBEAT_STALE_SECONDS = 35


def runtime_state_dict() -> modal.Dict:
    return modal.Dict.from_name(STATE_DICT_NAME, create_if_missing=True)


def _coerce_state(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return dict(value)


def normalized_state(value: Any, now: float | None = None) -> dict[str, Any]:
    """Return a trustworthy lifecycle snapshot without touching the GPU.

    `starting` and `ready` require a recent heartbeat. If a container dies
    without running its exit hook, the stale heartbeat automatically decays to
    `idle` instead of falsely advertising a live GPU forever.
    """

    now = time.time() if now is None else now
    state = _coerce_state(value)
    status = state.get("status")
    updated_at = state.get("updated_at")
    try:
        heartbeat_age = max(0.0, now - float(updated_at))
    except (TypeError, ValueError):
        heartbeat_age = float("inf")

    if status in {"starting", "ready"} and heartbeat_age <= HEARTBEAT_STALE_SECONDS:
        state["heartbeat_age_seconds"] = round(heartbeat_age, 3)
        state["stale"] = False
        return state

    return {
        "status": "idle",
        "updated_at": state.get("updated_at", 0),
        "started_at": state.get("started_at"),
        "ready_at": state.get("ready_at"),
        "heartbeat_age_seconds": (
            None if heartbeat_age == float("inf") else round(heartbeat_age, 3)
        ),
        "stale": bool(status in {"starting", "ready"}),
    }


async def read_state_async() -> dict[str, Any]:
    store = runtime_state_dict()
    value = await store.get.aio(STATE_KEY, {})
    return normalized_state(value)


def read_state_sync() -> dict[str, Any]:
    return normalized_state(runtime_state_dict().get(STATE_KEY, {}))


async def mark_triggered_async(now: float | None = None) -> dict[str, Any]:
    """Mark a user inference request as the beginning of a cold-start cycle."""

    now = time.time() if now is None else now
    store = runtime_state_dict()
    current = normalized_state(await store.get.aio(STATE_KEY, {}), now=now)
    if current.get("status") in {"starting", "ready"}:
        return current

    state = {
        "status": "starting",
        "started_at": now,
        "ready_at": None,
        "updated_at": now,
        "source": "gateway_user_inference",
    }
    await store.put.aio(STATE_KEY, state)
    return normalized_state(state, now=now)


@dataclass
class BackendStateReporter:
    """GPU-side lifecycle reporter with a low-frequency control-plane heartbeat."""

    served_model_name: str
    heartbeat_interval_seconds: int = HEARTBEAT_INTERVAL_SECONDS

    def __post_init__(self) -> None:
        self._store = runtime_state_dict()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._status = "starting"
        self._started_at = self._existing_started_at()
        self._ready_at: float | None = None

    def _existing_started_at(self) -> float:
        now = time.time()
        try:
            current = normalized_state(self._store.get(STATE_KEY, {}), now=now)
        except Exception:
            current = {"status": "idle"}
        if current.get("status") == "starting":
            try:
                return float(current["started_at"])
            except (KeyError, TypeError, ValueError):
                pass
        return now

    def _snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "status": self._status,
                "model": self.served_model_name if self._status == "ready" else None,
                "started_at": self._started_at,
                "ready_at": self._ready_at,
                "updated_at": time.time(),
                "source": "gpu_backend",
            }

    def _write(self) -> None:
        try:
            self._store.put(STATE_KEY, self._snapshot())
        except Exception as exc:
            # State reporting must never take down inference. A stale heartbeat
            # safely degrades the public gateway to `idle` instead.
            print(f"Backend state heartbeat failed: {exc!r}", flush=True)

    def start(self) -> None:
        self._write()
        self._thread = threading.Thread(
            target=self._heartbeat_loop,
            name="qwen38-backend-state-heartbeat",
            daemon=True,
        )
        self._thread.start()

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.heartbeat_interval_seconds):
            self._write()

    def mark_ready(self) -> None:
        with self._lock:
            self._status = "ready"
            self._ready_at = time.time()
        self._write()

    def mark_failed(self, reason: str) -> None:
        self._stop.set()
        state = {
            "status": "idle",
            "started_at": self._started_at,
            "ready_at": self._ready_at,
            "updated_at": time.time(),
            "source": "gpu_backend",
            "last_exit_reason": reason,
        }
        try:
            self._store.put(STATE_KEY, state)
        except Exception as exc:
            print(f"Backend state failure update failed: {exc!r}", flush=True)

    def stop(self, reason: str = "scaledown") -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2)
        state = {
            "status": "idle",
            "started_at": self._started_at,
            "ready_at": self._ready_at,
            "updated_at": time.time(),
            "source": "gpu_backend",
            "last_exit_reason": reason,
        }
        try:
            self._store.put(STATE_KEY, state)
        except Exception as exc:
            print(f"Backend state shutdown update failed: {exc!r}", flush=True)
