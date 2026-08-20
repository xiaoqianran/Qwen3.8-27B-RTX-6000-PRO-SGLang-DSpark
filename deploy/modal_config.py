"""Configuration for the independent Modal deployment.

Only this module should need routine edits when changing the GPU, SGLang
image, model IDs, or serving parameters. It deliberately has no dependency on
start.sh, stop.sh, patch/, or any other upstream-fork implementation file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ServingConfig:
    app_name: str = "qwen38-27b-modal"
    gpu: str = "RTX-PRO-6000"
    port: int = 8000

    # Use upstream SGLang directly. Override with a dated/pinned image tag
    # without editing code:
    #   QWEN38_SGLANG_IMAGE=lmsysorg/sglang:<tag> uv run modal run ...
    sglang_image: str = os.environ.get(
        "QWEN38_SGLANG_IMAGE",
        "lmsysorg/sglang:dev-cu13",
    )

    model_id: str = "RadixArk/Qwen3.8-27B-NVFP4"
    draft_model_id: str = "z-lab/Qwen3.8-27B-DFlash2"
    served_model_name: str = "qwen3.8-27b"

    # RTX PRO 6000 / Qwen3.8 reference choices.
    mem_fraction_static: float = 0.90
    context_length: int = 262_144
    kv_cache_dtype: str = "fp8_e4m3"
    attention_backend: str = "flashinfer"
    chunked_prefill_size: int = 4096
    max_prefill_tokens: int = 4096

    # This Modal profile is deliberately optimized for a single active decode.
    # Change both values together if the goal changes to multi-user throughput.
    max_running_requests: int = 1
    modal_target_concurrency: int = 1

    speculative_algorithm: str = "DFLASH"
    speculative_num_draft_tokens: int = 8
    speculative_draft_quantization: str = "unquant"
    speculative_draft_attention_backend: str = "flashinfer"

    min_free_slots_delay: int = 1
    decode_log_interval: int = 50

    startup_timeout_minutes: int = 30
    exit_grace_period_seconds: int = 15


CONFIG = ServingConfig()

HF_CACHE_PATH = "/root/.cache/huggingface"
TRITON_CACHE_PATH = "/root/.triton"


def build_sglang_command(port: int | None = None) -> list[str]:
    """Build the SGLang command from the Modal-owned configuration only."""
    c = CONFIG
    listen_port = c.port if port is None else port

    return [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        c.model_id,
        "--served-model-name",
        c.served_model_name,
        "--trust-remote-code",
        "--attention-backend",
        c.attention_backend,
        "--kv-cache-dtype",
        c.kv_cache_dtype,
        "--mem-fraction-static",
        str(c.mem_fraction_static),
        "--context-length",
        str(c.context_length),
        "--max-running-requests",
        str(c.max_running_requests),
        "--chunked-prefill-size",
        str(c.chunked_prefill_size),
        "--max-prefill-tokens",
        str(c.max_prefill_tokens),
        "--speculative-algorithm",
        c.speculative_algorithm,
        "--speculative-draft-model-path",
        c.draft_model_id,
        "--speculative-num-draft-tokens",
        str(c.speculative_num_draft_tokens),
        "--speculative-draft-model-quantization",
        c.speculative_draft_quantization,
        "--speculative-draft-attention-backend",
        c.speculative_draft_attention_backend,
        "--min-free-slots-delay",
        str(c.min_free_slots_delay),
        "--reasoning-parser",
        "qwen3",
        "--tool-call-parser",
        "qwen3_coder",
        "--sampling-defaults",
        "model",
        "--decode-log-interval",
        str(c.decode_log_interval),
        "--host",
        "0.0.0.0",
        "--port",
        str(listen_port),
    ]
