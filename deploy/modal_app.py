"""Independent Modal deployment for Qwen3.8-27B + DFlash2.

The deployment lives in this repository but is intentionally isolated from the
forked Docker implementation. It does not read/import/copy:
  - start.sh
  - stop.sh
  - patch/
  - README.md
or any other upstream-owned runtime file.

Run:
    uv run modal run deploy/modal_app.py

Benchmark with a longer generation:
    uv run modal run deploy/modal_app.py --max-tokens 2048

Persistent endpoint:
    uv run modal deploy deploy/modal_app.py
"""

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
    CONFIG,
    HF_CACHE_PATH,
    TRITON_CACHE_PATH,
    build_sglang_command,
)


MINUTE = 60

app = modal.App(CONFIG.app_name)

hf_cache = modal.Volume.from_name(
    "qwen38-27b-hf-cache",
    create_if_missing=True,
)
triton_cache = modal.Volume.from_name(
    "qwen38-27b-triton-cache",
    create_if_missing=True,
)


# DFlash2 is now upstream in SGLang. This build-time compatibility check makes
# a moving dev image fail before a paid GPU container is started if upstream
# ever changes/removes the model class we rely on.
image = (
    modal.Image.from_registry(CONFIG.sglang_image)
    .entrypoint([])
    .env(
        {
            "HF_HOME": HF_CACHE_PATH,
            "HF_HUB_CACHE": HF_CACHE_PATH,
            "HF_XET_HIGH_PERFORMANCE": "1",
            "TRITON_CACHE_DIR": TRITON_CACHE_PATH,
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .run_commands(
        "python3 -c \"from sglang.srt.models.dflash import "
        "DFlash2DraftModel; print('DFlash2 upstream check: OK')\""
    )
    # Modal 1.x no longer automounts arbitrary local helper modules.
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
    port=CONFIG.port,
    volumes={
        HF_CACHE_PATH: hf_cache,
        TRITON_CACHE_PATH: triton_cache,
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
        print("=" * 80, flush=True)
        print("Independent Modal deployment", flush=True)
        print(f"GPU:          {CONFIG.gpu}", flush=True)
        print(f"SGLang image: {CONFIG.sglang_image}", flush=True)
        print(f"Target:       {CONFIG.model_id}", flush=True)
        print(f"Draft:        {CONFIG.draft_model_id}", flush=True)
        print(
            f"Concurrency:  Modal={CONFIG.modal_target_concurrency}, "
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
    """Start a temporary Modal Server and run one single-stream benchmark."""
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
