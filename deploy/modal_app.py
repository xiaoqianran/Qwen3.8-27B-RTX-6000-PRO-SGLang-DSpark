"""Modal-native Qwen3.8-27B + DFlash2 deployment."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
import urllib.error
import urllib.request

import modal

from deploy.modal_benchmark import run_single_stream_benchmark
from deploy.modal_config import (
    COMPILE_CACHE_PATH,
    CONFIG,
    DRAFT_MODEL_PATH,
    MODEL_STORE_PATH,
    SGLANG_CACHE_PATH,
    TARGET_MODEL_PATH,
    TORCHINDUCTOR_CACHE_PATH,
    TRITON_CACHE_PATH,
    build_sglang_command,
)
from deploy.model_prepare import download_models, validate_model_store

MINUTE = 60
HOUR = 60 * MINUTE

app = modal.App(CONFIG.app_name, include_source=False)
model_store = modal.Volume.from_name(CONFIG.model_volume_name, create_if_missing=True)
model_store_ro = model_store.with_mount_options(read_only=True)
compile_cache = modal.Volume.from_name(
    CONFIG.compile_cache_volume_name, create_if_missing=True
)

# CPU-only build step. Model/revision/path settings are kwargs so they are part
# of Modal's image-build cache key.
prepare_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("huggingface_hub[hf_xet]==1.26.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    .run_function(
        download_models,
        volumes={MODEL_STORE_PATH: model_store},
        kwargs={
            "target_repo": CONFIG.model_id,
            "target_revision": CONFIG.model_revision,
            "target_dir": TARGET_MODEL_PATH,
            "draft_repo": CONFIG.draft_model_id,
            "draft_revision": CONFIG.draft_model_revision,
            "draft_dir": DRAFT_MODEL_PATH,
            "model_store_dir": MODEL_STORE_PATH,
            "max_workers": CONFIG.download_max_workers,
        },
        cpu=CONFIG.download_cpu,
        timeout=CONFIG.download_timeout_hours * HOUR,
    )
    .add_local_python_source("deploy")
)


def _validate_models() -> None:
    validate_model_store(
        MODEL_STORE_PATH,
        TARGET_MODEL_PATH,
        DRAFT_MODEL_PATH,
        CONFIG.model_id,
        CONFIG.draft_model_id,
    )


@app.function(
    image=prepare_image,
    volumes={MODEL_STORE_PATH: model_store_ro},
    cpu=1,
    timeout=60,
)
def model_store_ready() -> bool:
    _validate_models()
    return True


# GPU runtime. The repository's qwen38 image predates DFlash2, so bake the
# existing patch/sglang compatibility layer into the Modal image. This is the
# same backport used by start.sh, without running Docker inside Modal.
sglang_image = (
    modal.Image.from_registry(CONFIG.sglang_image)
    .entrypoint([])
    # Legacy Modal builders may downgrade this dependency inside third-party
    # images; restore the version required by the SGLang/Pydantic stack.
    .uv_pip_install("typing_extensions==4.16.0")
    .add_local_dir(
        "patch/sglang",
        "/tmp/sglang-patch",
        copy=True,
        ignore=["README.md"],
    )
    .run_commands(
        "cp -a /tmp/sglang-patch/kernels/. "
        "/sgl-workspace/sglang/python/sglang/kernels/ && "
        "cp -a /tmp/sglang-patch/srt/. "
        "/sgl-workspace/sglang/python/sglang/srt/ && "
        "rm -rf /tmp/sglang-patch",
        "python3 -c \"from sglang.srt.models.dflash import "
        "DFlash2DraftModel; print('DFlash2 import: OK')\"",
    )
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
    .add_local_python_source("deploy")
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


def _wait_for_public_server(base_url: str) -> None:
    """Trigger zero-to-one scaling and retry expected cold-start failures."""
    deadline = time.monotonic() + CONFIG.startup_timeout_minutes * MINUTE
    health_url = base_url.rstrip("/") + "/health"

    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=5):
                return
        except urllib.error.HTTPError as exc:
            if exc.code not in {502, 503, 504}:
                raise
        except urllib.error.URLError:
            pass
        time.sleep(2)

    raise TimeoutError("Modal server did not become reachable before timeout")


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
        MODEL_STORE_PATH: model_store_ro,
        COMPILE_CACHE_PATH: compile_cache,
    },
    startup_timeout=CONFIG.startup_timeout_minutes * MINUTE,
    min_containers=0,
    max_containers=CONFIG.modal_max_containers,
    exit_grace_period=CONFIG.exit_grace_period_seconds,
    unauthenticated=True,
)
class Qwen38Server:
    @modal.enter()
    def startup(self):
        _validate_models()

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
            process.wait(timeout=5)


@app.local_entrypoint()
async def main(max_tokens: int = 1024):
    if max_tokens < 128:
        raise ValueError("--max-tokens must be >= 128")

    # CPU preparation finishes before the first request can allocate the single
    # RTX PRO 6000 container.
    await model_store_ready.remote.aio()
    url = await Qwen38Server.get_url.aio()
    await asyncio.to_thread(_wait_for_public_server, url)
    await asyncio.to_thread(
        run_single_stream_benchmark,
        url,
        CONFIG.served_model_name,
        max_tokens,
    )
