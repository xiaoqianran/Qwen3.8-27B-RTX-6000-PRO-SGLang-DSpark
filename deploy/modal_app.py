"""Independent Modal deployment for Qwen3.8-27B + DFlash2.

The Modal deployment is isolated from the forked bare-metal/Docker runtime.
It does not read/import/copy start.sh, stop.sh, patch/, or README.md.

Model lifecycle:
  1. CPU-only Modal image-build step downloads both Hugging Face repositories
     into a persistent Modal Volume.
  2. The GPU server mounts that Volume and runs in Hugging Face offline mode.
  3. RTX PRO 6000 time is therefore spent on model loading, CUDA/SGLang
     initialization, warmup, and inference -- never model network download.

Run:
    uv run modal run deploy/modal_app.py --max-tokens 2048

Deploy:
    uv run modal deploy deploy/modal_app.py
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import subprocess
import time
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


MINUTE = 60
HOUR = 60 * MINUTE

app = modal.App(CONFIG.app_name)

model_store = modal.Volume.from_name(
    CONFIG.model_volume_name,
    create_if_missing=True,
)
compile_cache = modal.Volume.from_name(
    CONFIG.compile_cache_volume_name,
    create_if_missing=True,
)


def _download_models_to_volume(
    target_repo_id: str,
    target_revision: str | None,
    target_dir: str,
    draft_repo_id: str,
    draft_revision: str | None,
    draft_dir: str,
    max_workers: int,
) -> None:
    """CPU-only image-build step that materializes both repositories.

    Keep all download logic in this function. Modal includes this function's
    source and its arguments in the Image build definition, so changing model
    IDs/revisions/paths deliberately invalidates the corresponding build step.
    """
    import json as _json
    import os as _os
    from pathlib import Path as _Path

    from huggingface_hub import snapshot_download

    _os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

    downloads = (
        ("target", target_repo_id, target_revision, target_dir),
        ("draft", draft_repo_id, draft_revision, draft_dir),
    )

    manifest: dict[str, dict[str, str | None]] = {}

    for role, repo_id, revision, local_dir in downloads:
        destination = _Path(local_dir)
        destination.mkdir(parents=True, exist_ok=True)

        print(
            f"[CPU download] {role}: {repo_id}"
            + (f" @ {revision}" if revision else ""),
            flush=True,
        )
        resolved_path = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=destination,
            max_workers=max_workers,
        )

        config_path = destination / "config.json"
        if not config_path.is_file():
            raise RuntimeError(
                f"{role} download finished but {config_path} is missing"
            )

        manifest[role] = {
            "repo_id": repo_id,
            "requested_revision": revision,
            "path": str(resolved_path),
        }
        print(f"[CPU download] {role} ready at {resolved_path}", flush=True)

    manifest_path = _Path(MODEL_STORE_PATH) / ".modal-model-manifest.json"
    manifest_path.write_text(
        _json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[CPU download] wrote {manifest_path}", flush=True)
    print("[CPU download] model store preparation complete", flush=True)


# Start from the current upstream SGLang image.
#
# The run_function step is deliberately CPU-only (gpu=None by default). It
# executes before any RTX PRO 6000 container is created, and writes model
# weights to the persistent model_store Volume. This mirrors Modal's current
# recommended model-serving pattern.
image = (
    modal.Image.from_registry(CONFIG.sglang_image)
    .entrypoint([])
    .env(
        {
            "HF_HOME": f"{MODEL_STORE_PATH}/.hf",
            "HF_HUB_CACHE": f"{MODEL_STORE_PATH}/.hf/hub",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .run_commands(
        "python3 -c \"from sglang.srt.models.dflash import "
        "DFlash2DraftModel; print('DFlash2 upstream check: OK')\""
    )
    .run_function(
        _download_models_to_volume,
        volumes={MODEL_STORE_PATH: model_store},
        args=(
            CONFIG.model_id,
            CONFIG.model_revision,
            TARGET_MODEL_PATH,
            CONFIG.draft_model_id,
            CONFIG.draft_model_revision,
            DRAFT_MODEL_PATH,
            CONFIG.download_max_workers,
        ),
        cpu=CONFIG.download_cpu,
        timeout=CONFIG.download_timeout_hours * HOUR,
    )
    # Everything below is runtime policy for the GPU server.
    # The GPU container must use local snapshots only.
    .env(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TRITON_CACHE_DIR": TRITON_CACHE_PATH,
            "TORCHINDUCTOR_CACHE_DIR": TORCHINDUCTOR_CACHE_PATH,
            "SGLANG_CACHE_DIR": SGLANG_CACHE_PATH,
        }
    )
    .add_local_python_source("deploy")
)


def _http_json(
    url: str,
    payload: dict | None = None,
    timeout: float = 10.0,
) -> dict:
    if payload is None:
        request = urllib.request.Request(url, method="GET")
    else:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _wait_ready(process: subprocess.Popen) -> None:
    timeout_seconds = CONFIG.startup_timeout_minutes * MINUTE
    deadline = time.monotonic() + timeout_seconds
    health_url = f"http://127.0.0.1:{CONFIG.port}/health"

    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"SGLang exited during startup with code {return_code}"
            )

        try:
            with urllib.request.urlopen(health_url, timeout=2):
                return
        except Exception:
            time.sleep(2)

    raise TimeoutError(
        f"SGLang did not become healthy within {timeout_seconds} seconds"
    )


def _validate_local_model_store() -> None:
    required = (
        Path(TARGET_MODEL_PATH) / "config.json",
        Path(DRAFT_MODEL_PATH) / "config.json",
        Path(MODEL_STORE_PATH) / ".modal-model-manifest.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(
            "CPU-prepared model Volume is incomplete; refusing to let the GPU "
            "fall back to network downloads. Missing: " + ", ".join(missing)
        )


def _warmup() -> None:
    url = f"http://127.0.0.1:{CONFIG.port}/v1/chat/completions"
    payload = {
        "model": CONFIG.served_model_name,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Write a continuous technical paragraph about GPU inference. "
                    "Continue until the output limit."
                ),
            }
        ],
        "temperature": 0.0,
        "max_tokens": 128,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    for index in range(3):
        result = _http_json(url, payload=payload, timeout=180)
        usage = result.get("usage") or {}
        print(
            f"warmup {index + 1}/3: "
            f"{usage.get('completion_tokens', '?')} completion tokens",
            flush=True,
        )


@app.server(
    image=image,
    gpu=CONFIG.gpu,
    cpu=CONFIG.server_cpu,
    port=CONFIG.port,
    volumes={
        MODEL_STORE_PATH: model_store,
        COMPILE_CACHE_PATH: compile_cache,
    },
    startup_timeout=CONFIG.startup_timeout_minutes * MINUTE,
    target_concurrency=CONFIG.modal_target_concurrency,
    min_containers=0,
    exit_grace_period=CONFIG.exit_grace_period_seconds,
    unauthenticated=True,
)
class Qwen38Server:
    @modal.enter()
    def startup(self):
        # Fail closed: if CPU preparation is missing/corrupt, never let an
        # expensive GPU container try an implicit Hugging Face download.
        _validate_local_model_store()

        print("=" * 80, flush=True)
        print("Independent Modal deployment", flush=True)
        print(f"GPU:              {CONFIG.gpu}", flush=True)
        print(f"SGLang image:     {CONFIG.sglang_image}", flush=True)
        print(f"Target repo:      {CONFIG.model_id}", flush=True)
        print(f"Target local:     {TARGET_MODEL_PATH}", flush=True)
        print(f"Draft repo:       {CONFIG.draft_model_id}", flush=True)
        print(f"Draft local:      {DRAFT_MODEL_PATH}", flush=True)
        print("HF network:       OFFLINE inside GPU container", flush=True)
        print(
            f"Concurrency:      Modal={CONFIG.modal_target_concurrency}, "
            f"SGLang={CONFIG.max_running_requests}",
            flush=True,
        )
        print("=" * 80, flush=True)

        subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            check=False,
        )

        subprocess.run(
            [
                "python3",
                "-c",
                (
                    "import sglang; "
                    "print('SGLang version:', "
                    "getattr(sglang, '__version__', 'unknown'))"
                ),
            ],
            check=False,
        )

        command = build_sglang_command()
        print("\nLaunching SGLang:\n" + " ".join(command) + "\n", flush=True)

        self.process = subprocess.Popen(
            command,
            env=os.environ.copy(),
        )

        _wait_ready(self.process)
        print("SGLang healthy; warming DFlash2/CUDA paths...", flush=True)
        _warmup()
        print("Warmup complete.", flush=True)

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
    """Build/download on CPU, then start GPU server and benchmark one stream."""
    if max_tokens < 128:
        raise ValueError("--max-tokens must be >= 128")

    url = await Qwen38Server.get_url.aio()

    print()
    print(f"Temporary Modal endpoint: {url}")
    print(
        "Running one active stream with thinking disabled; "
        f"max_tokens={max_tokens}."
    )

    await asyncio.to_thread(
        run_single_stream_benchmark,
        url,
        CONFIG.served_model_name,
        max_tokens,
    )
