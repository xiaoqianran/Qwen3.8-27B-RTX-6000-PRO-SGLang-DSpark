"""Modal-native Qwen3.8-27B + DFlash2 deployment."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request

import modal

from deploy.modal_benchmark import run_concurrency_benchmark
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

# Keep the App source module enabled. When this file is invoked as
# `modal run deploy/modal_app.py`, Modal records the defining module as
# `modal_app`; disabling source inclusion makes remote @app.server hydration
# fail with `ModuleNotFoundError: No module named 'modal_app'`.
app = modal.App(CONFIG.app_name, include_source=True)
model_store = modal.Volume.from_name(CONFIG.model_volume_name, create_if_missing=True)
model_store_ro = model_store.with_mount_options(read_only=True)
compile_cache = modal.Volume.from_name(
    CONFIG.compile_cache_volume_name, create_if_missing=True
)

# CPU-only model preparation. If revisions are unpinned, rerun this cheap build
# step on deploy so Hugging Face's moving default branch cannot be hidden behind
# Modal's image cache. snapshot_download still reuses the persistent Volume.
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
        force_build=(
            CONFIG.model_revision is None
            or CONFIG.draft_model_revision is None
        ),
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


# The cookbook qwen38 image predates DFlash2. Bake the repository's existing
# compatibility overlay into the image, then validate the integration before a
# GPU can ever be allocated.
sglang_image = (
    modal.Image.from_registry(CONFIG.sglang_image)
    .entrypoint([])
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
        "python3 -c \""
        "from sglang.kernels.ops.speculative.fused_kv_materialize import FusedKVMaterializeHelper; "
        "from sglang.srt.layers.logits_processor import should_apply_lm_head_quant_method; "
        "from sglang.srt.models.dflash import DFlash2DraftModel; "
        "from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2; "
        "print('DFlash2 runtime imports: OK')\"",
        "python3 -m sglang.launch_server --help | grep -q -- '--max-mamba-cache-size'",
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


def _parse_concurrency_levels(spec: str) -> list[int]:
    levels: list[int] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            level = int(item)
        except ValueError as exc:
            raise ValueError(f"invalid concurrency value: {item!r}") from exc
        if not 1 <= level <= CONFIG.max_running_requests:
            raise ValueError(
                f"concurrency must be between 1 and {CONFIG.max_running_requests}"
            )
        if level not in levels:
            levels.append(level)
    if not levels:
        raise ValueError("--concurrency must contain at least one integer")
    return levels


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
    target_concurrency=CONFIG.max_running_requests,
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
        self.process = subprocess.Popen(
            command,
            env=os.environ.copy(),
            start_new_session=True,
        )
        _wait_ready(self.process)
        _warmup()
        compile_cache.commit()
        print("Compile cache committed.", flush=True)

    @modal.exit()
    def shutdown(self):
        process = getattr(self, "process", None)
        if process is None or process.poll() is not None:
            return

        process.terminate()
        try:
            process.wait(timeout=CONFIG.shutdown_timeout_seconds)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)


@app.local_entrypoint()
async def main(max_tokens: int = 4096, concurrency: str = "1"):
    if max_tokens < 128:
        raise ValueError("--max-tokens must be >= 128")
    levels = _parse_concurrency_levels(concurrency)

    # CPU preparation finishes before the first request can allocate the single
    # RTX PRO 6000 container.
    await model_store_ready.remote.aio()
    url = await Qwen38Server.get_url.aio()
    await asyncio.to_thread(_wait_for_public_server, url)

    for level in levels:
        await asyncio.to_thread(
            run_concurrency_benchmark,
            url,
            CONFIG.served_model_name,
            max_tokens,
            level,
        )

    print("Benchmark complete; the temporary `modal run` Server will now stop.")
