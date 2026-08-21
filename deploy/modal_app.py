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
    TARGET_MODEL_PATH,
    build_sglang_command,
    runtime_cache_identity,
)
from deploy.model_prepare import download_models, validate_model_store
from deploy.runtime_cache import flashinfer_entry_count, prepare_runtime_cache

MINUTE = 60
HOUR = 60 * MINUTE

app = modal.App(CONFIG.app_name, include_source=True)
model_store = modal.Volume.from_name(CONFIG.model_volume_name, create_if_missing=True)
model_store_ro = model_store.with_mount_options(read_only=True)
compile_cache = modal.Volume.from_name(
    CONFIG.compile_cache_volume_name, create_if_missing=True
)

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


# Only require experimental CLI flags when the corresponding experiment is
# explicitly enabled. This keeps the validated default image path fail-closed
# without making an optional newer-SGLang feature a hard dependency.
language_only_check = (
    "python3 -m sglang.launch_server --help | grep -q -- '--language-only'"
    if CONFIG.language_only
    else "true"
)

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
        "python3 -m sglang.launch_server --help | grep -q -- '--cuda-graph-bs-prefill'",
        "python3 -m sglang.launch_server --help | grep -q -- '--cuda-graph-bs-decode'",
        language_only_check,
    )
    .env(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "SGLANG_FLASHINFER_AUTOTUNE_CACHE": "1",
            # SGLang's default /health path performs a real 1-token generation.
            # We already run three explicit OpenAI warmups after readiness, so
            # disable that redundant generation and let /health be status-only.
            "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "0",
            "QWEN38_COLD_START_PROFILE": CONFIG.cold_start_profile,
            "QWEN38_VERIFY_GRAPH_PROFILE": CONFIG.verify_graph_profile,
            "QWEN38_LANGUAGE_ONLY": "1" if CONFIG.language_only else "0",
            "QWEN38_RUNTIME_CACHE_EPOCH": CONFIG.runtime_cache_epoch,
            "QWEN38_SGLANG_IMAGE": CONFIG.sglang_image,
        }
    )
    .add_local_python_source("deploy")
)

# Modal documents this as a small but useful cold-start prefetch hint for SGLang.
# This is already active in main and deliberately remains before server startup.
with sglang_image.imports():
    import sglang  # noqa: F401


GATEWAY_PORT = 8080
GATEWAY_COLD_START_ESTIMATE_SECONDS = 120

gateway_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "fastapi==0.116.1",
        "httpx==0.28.1",
        "uvicorn==0.35.0",
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


def _wait_ready(process: subprocess.Popen) -> float:
    started = time.monotonic()
    deadline = started + CONFIG.startup_timeout_minutes * MINUTE
    health_url = f"http://127.0.0.1:{CONFIG.port}/health"

    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"SGLang exited during startup: {process.returncode}")
        try:
            with urllib.request.urlopen(health_url, timeout=2):
                return time.monotonic() - started
        except Exception:
            time.sleep(2)
    raise TimeoutError("SGLang startup timed out")


def _wait_for_public_server(base_url: str) -> None:
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
    h2_enabled=False,
)
class Qwen38Backend:
    @modal.enter()
    def startup(self):
        startup_started = time.perf_counter()
        _validate_models()

        subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            check=False,
        )

        self.runtime_cache = prepare_runtime_cache(
            compile_cache_dir=COMPILE_CACHE_PATH,
            model_store_dir=MODEL_STORE_PATH,
            identity=runtime_cache_identity(),
        )
        print(
            "Runtime cache:",
            f"profile={CONFIG.cold_start_profile}",
            f"verify_profile={CONFIG.verify_graph_profile}",
            f"language_only={CONFIG.language_only}",
            f"key={self.runtime_cache.key}",
            f"flashinfer_entries_before={self.runtime_cache.flashinfer_entries_before}",
            flush=True,
        )

        command = build_sglang_command()
        process_env = os.environ.copy()
        process_env.update(self.runtime_cache.env)
        # The base image currently carries a deprecated compatibility variable;
        # it is not required by this deployment and only produces warning noise.
        process_env.pop("SGLANG_FLASHINFER_PR4266_SOURCE", None)
        print("Launching:", " ".join(command), flush=True)
        process_started = time.perf_counter()
        self.process = subprocess.Popen(
            command,
            env=process_env,
            start_new_session=True,
        )
        health_wait = _wait_ready(self.process)
        ready_at = time.perf_counter()
        _warmup()
        warm_at = time.perf_counter()

        flashinfer_after = flashinfer_entry_count(self.runtime_cache)
        compile_cache.commit()
        committed_at = time.perf_counter()
        print(
            "Runtime cache committed:",
            f"key={self.runtime_cache.key}",
            f"flashinfer_entries={self.runtime_cache.flashinfer_entries_before}->{flashinfer_after}",
            flush=True,
        )
        print(
            "Cold-start timing:",
            f"pre_launch={process_started - startup_started:.2f}s",
            f"health_wait={health_wait:.2f}s",
            f"engine_ready={ready_at - startup_started:.2f}s",
            f"warmup={warm_at - ready_at:.2f}s",
            f"cache_commit={committed_at - warm_at:.2f}s",
            f"total={committed_at - startup_started:.2f}s",
            flush=True,
        )

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


def _wait_gateway_ready(process: subprocess.Popen) -> None:
    deadline = time.monotonic() + 60
    health_url = f"http://127.0.0.1:{GATEWAY_PORT}/_gateway/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"Cold-start gateway exited during startup: {process.returncode}"
            )
        try:
            with urllib.request.urlopen(health_url, timeout=2):
                return
        except Exception:
            time.sleep(0.5)
    raise TimeoutError("Cold-start gateway did not become ready")


@app.server(
    image=gateway_image,
    cpu=1,
    port=GATEWAY_PORT,
    startup_timeout=2 * MINUTE,
    target_concurrency=100,
    min_containers=1,
    max_containers=1,
    exit_grace_period=30,
    unauthenticated=True,
    h2_enabled=False,
)
class Qwen38Server:
    """Always-on public gateway; preserves the existing qwen38server URL slug."""

    @modal.enter()
    def startup(self):
        backend_url = Qwen38Backend.get_url()
        process_env = os.environ.copy()
        process_env.update(
            {
                "QWEN38_BACKEND_URL": backend_url,
                "QWEN38_COLD_START_ESTIMATE_SECONDS": str(
                    GATEWAY_COLD_START_ESTIMATE_SECONDS
                ),
            }
        )
        print(
            "Launching cold-start gateway:",
            f"backend={backend_url}",
            f"estimate={GATEWAY_COLD_START_ESTIMATE_SECONDS}s",
            flush=True,
        )
        self.process = subprocess.Popen(
            [
                "python3",
                "-m",
                "uvicorn",
                "deploy.gateway:app",
                "--host",
                "0.0.0.0",
                "--port",
                str(GATEWAY_PORT),
                "--no-access-log",
            ],
            env=process_env,
            start_new_session=True,
        )
        _wait_gateway_ready(self.process)

    @modal.exit()
    def shutdown(self):
        process = getattr(self, "process", None)
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)


@app.local_entrypoint()
async def main(
    max_tokens: int = 4096,
    concurrency: str = "1",
    cache_only: bool = False,
):
    if max_tokens < 128:
        raise ValueError("--max-tokens must be >= 128")
    levels = _parse_concurrency_levels(concurrency)

    await model_store_ready.remote.aio()
    url = await Qwen38Backend.get_url.aio()
    await asyncio.to_thread(_wait_for_public_server, url)

    if cache_only:
        print(
            "Runtime cache prepared; the temporary `modal run` Server will now stop."
        )
        return

    for level in levels:
        await asyncio.to_thread(
            run_concurrency_benchmark,
            url,
            CONFIG.served_model_name,
            max_tokens,
            level,
        )

    print("Benchmark complete; the temporary `modal run` Server will now stop.")
