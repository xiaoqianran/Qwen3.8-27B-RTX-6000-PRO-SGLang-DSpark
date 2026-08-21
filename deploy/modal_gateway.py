"""Independent scale-to-zero Modal ASGI gateway for Qwen3.8.

This App is intentionally separate from the RTX PRO 6000 backend App. Gateway
changes can therefore be deployed without rolling/restarting an active GPU
container. Only real chat-completion requests are allowed to contact the GPU
Server; discovery and health traffic read shared lifecycle state instead.
"""

from __future__ import annotations

import os

import modal

from deploy.modal_config import CONFIG

HOUR = 60 * 60
GATEWAY_COLD_START_ESTIMATE_SECONDS = 120
GATEWAY_CPU = 0.125
GATEWAY_SCALEDOWN_SECONDS = 5
GATEWAY_MAX_INPUTS = 100

app = modal.App(CONFIG.gateway_app_name, include_source=True)
backend = modal.Server.from_name(CONFIG.app_name, "Qwen38Backend")

gateway_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "fastapi==0.116.1",
        "httpx==0.28.1",
    )
    .add_local_python_source("deploy")
)


@app.function(
    name="Qwen38Gateway",
    image=gateway_image,
    cpu=GATEWAY_CPU,
    min_containers=0,
    max_containers=1,
    scaledown_window=GATEWAY_SCALEDOWN_SECONDS,
    timeout=HOUR,
)
@modal.concurrent(max_inputs=GATEWAY_MAX_INPUTS)
@modal.asgi_app(label="qwen38server")
def Qwen38Gateway():
    """Public OpenAI-compatible gateway, independently deployable from GPU."""

    backend_url = backend.get_url()
    if not backend_url:
        raise RuntimeError(
            f"Backend Server Qwen38Backend is unavailable in App {CONFIG.app_name!r}"
        )

    os.environ["QWEN38_BACKEND_URL"] = backend_url
    os.environ["QWEN38_SERVED_MODEL_NAME"] = CONFIG.served_model_name
    os.environ["QWEN38_COLD_START_ESTIMATE_SECONDS"] = str(
        GATEWAY_COLD_START_ESTIMATE_SECONDS
    )
    print(
        "Starting independent scale-to-zero gateway:",
        f"backend_app={CONFIG.app_name}",
        f"backend={backend_url}",
        f"cpu={GATEWAY_CPU}",
        f"scaledown_window={GATEWAY_SCALEDOWN_SECONDS}s",
        f"max_inputs={GATEWAY_MAX_INPUTS}",
        flush=True,
    )

    from deploy.gateway import app as gateway_app

    return gateway_app
