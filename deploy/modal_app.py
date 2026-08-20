"""Modal-native Qwen3.8-27B + DFlash2 deployment."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
import urllib.request

import modal

from deploy.modal_benchmark import run_single_stream_benchmark
from deploy.modal_config import (
    COMPILE_CACHE_PATH,
    CONFIG,
    MODEL_STORE_PATH,
    SGLANG_CACHE_PATH,
    TORCHINDUCTOR_CACHE_PATH,
    TRITON_CACHE_PATH,
    build_sglang_command,
)
from deploy.model_prepare import download_models, validate_model_store

MINUTE = 60
HOUR = 60 * MINUTE

app = modal.App(CONFIG.app_name)
model_store = modal.Volume.from_name(CONFIG.model_volume_name, create_if_missing=True)
compile_cache = modal.Volume.from_name(
    CONFIG.compile_cache_volume_name, create_if_missing=True
)

# CPU-only build: download model snapshots into the persistent Volume.
# run_function includes the source package of download_models automatically.
prepare_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("huggingface_hub[hf_xet]==1.26.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    .run_function(
        download_models,
        volumes={MODEL_STORE_PATH: model_store},
        args=(
            CONFIG.model_id,
            CONFIG.model_revision,
            CONFIG.draft_model_id,
            CONFIG.draft_model_revision,
            CONFIG.download_max_workers,
        ),
        cpu=CONFIG.download_cpu,
        timeout=CONFIG.download_timeout_hours * HOUR,
    )
)


@app.function(
    image=prepare_image,
    volumes={MODEL_STORE_PATH: model_store.read_only()},
    cpu=1,
    timeout=60,
)
def model_store_ready() -> bool:
    validate_model_store()
    return True


# GPU runtime: model network access is disabled and /models is mounted read-only.
# Legacy Modal builders downgrade typing_extensions to 4.12.2; restore the
# version shipped by the SGLang image before validating the full DFlash2 import.
sglang_image = (
    modal.Image.from_registry(CONFIG.sglang_image)
    .entrypoint([])
    .uv_pip_install("typing_extensions==4.16.0")
    .env(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "TRITON_CACHE_DIR": TRITON_CACHE_PATH,
            "TORCHINDUCTOR_CACHE_DIR": TORCHINDUCTOR_CACHE_PATH,
            "SGLANG_CACHE_DIR": SGLANG_CACHE_PATH,
        }
    )
    .run_commands(
        "python3 -c \"from sglang.srt.models.dflash import "
        "DFlash2DraftModel; print('DFlash2 import: OK')\""
    )
)


def _http_json(url: str, payload: dict, timeout: float = 180) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _wait_ready(process: subprocess.Popen) -> None:
    deadline = time.monotonic() + CONFIG.startup_timeout_minutes * MINUTE
    health_url = f"http://127.0.0.1:{CONFIG.port}/health"

    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"SGLang exited during startup: {process.returncode}")
        try:
            with urllib.request.urlopen(health_url, timeout=2):
                return
        except Exception:
            time.sleep(2)
    raise TimeoutError("SGLang startup timed out")


def _warmup() -> None:
    payload = {
        "model": CONFIG.served_model_name,
        "messages": [{"role": "user", "content": "Explain GPU inference continuously."}],
        "temperature": 0.0,
        "max_tokens": 128,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    url = f"http://127.0.0.1:{CONFIG.port}/v1/chat/completions"
    for i in range(3):
        result = _http_json(url, payload)
        tokens = (result.get("usage") or {}).get("completion_tokens", "?")
        print(f"warmup {i + 1}/3: {tokens} tokens", flush=True)


@app.server(
    image=sglang_image,
    gpu=CONFIG.gpu,
    cpu=CONFIG.server_cpu,
    port=CONFIG.port,
    volumes={
        MODEL_STORE_PATH: model_store.read_only(),
        COMPILE_CACHE_PATH: compile_cache,
    },
    startup_timeout=CONFIG.startup_timeout_minutes * MINUTE,
    target_concurrency=CONFIG.modal_target_concurrency,
    min_containers=0,
    max_containers=CONFIG.modal_max_containers,
    exit_grace_period=CONFIG.exit_grace_period_seconds,
    unauthenticated=True,
)
class Qwen38Server:
    @modal.enter()
    def startup(self):
        validate_model_store()

        subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            check=False,
        )

        command = build_sglang_command()
        print("Launching:", " ".join(command), flush=True)
        self.process = subprocess.Popen(command, env=os.environ.copy())
        _wait_ready(self.process)
        _warmup()

    @modal.exit()
    def shutdown(self):
        process = getattr(self, "process", None)
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=CONFIG.exit_grace_period_seconds)
        except subprocess.TimeoutExpired:
            process.kill()


@app.local_entrypoint()
async def main(max_tokens: int = 1024):
    if max_tokens < 128:
        raise ValueError("--max-tokens must be >= 128")

    # Finish CPU preparation before requesting the single RTX PRO 6000.
    await model_store_ready.remote.aio()
    url = await Qwen38Server.get_url.aio()
    await asyncio.to_thread(
        run_single_stream_benchmark,
        url,
        CONFIG.served_model_name,
        max_tokens,
    )
