"""Configuration for the Modal deployment."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ServingConfig:
    app_name: str = "qwen38-27b-modal"
    gpu: str = "RTX-PRO-6000"
    port: int = 8000

    # Match the Qwen3.8 cookbook image. DFlash2 is supplied by the repository's
    # shared patch/sglang backport until a verified official image includes it.
    sglang_image: str = os.environ.get(
        "QWEN38_SGLANG_IMAGE",
        "lmsysorg/sglang:qwen38-27b",
    )

    model_id: str = "RadixArk/Qwen3.8-27B-NVFP4"
    draft_model_id: str = "z-lab/Qwen3.8-27B-DFlash2"
    model_revision: str | None = os.environ.get("QWEN38_MODEL_REVISION") or None
    draft_model_revision: str | None = (
        os.environ.get("QWEN38_DRAFT_MODEL_REVISION") or None
    )
    served_model_name: str = "qwen3.8-27b"

    model_volume_name: str = "qwen38-27b-model-store"
    compile_cache_volume_name: str = "qwen38-27b-compile-cache"

    download_cpu: int = 8
    download_max_workers: int = 16
    download_timeout_hours: int = 4
    server_cpu: int = 8

    # Current SGLang RTX PRO 6000 + NVFP4 baseline.
    mem_fraction_static: float = 0.85
    context_length: int = 262_144
    kv_cache_dtype: str = "fp8_e4m3"
    attention_backend: str = "flashinfer"
    chunked_prefill_size: int = 2048
    max_prefill_tokens: int = 16_384

    # Qwen3.8's hybrid-GDN target needs five base recurrent-state slots per
    # active request with extra_buffer + overlap. Pin both the strategy/dtype
    # and 8*5=40 slots so later SGLang defaults cannot change this budget.
    max_running_requests: int = 8
    mamba_slots_per_request: int = 5
    mamba_radix_cache_strategy: str = "extra_buffer"
    mamba_ssm_dtype: str = "float32"

    # Keep exactly one GPU replica. Modal may route up to the same eight
    # concurrent HTTP requests into that replica before the autoscaler queues.
    modal_max_containers: int = 1

    speculative_algorithm: str = "DFLASH"
    speculative_num_draft_tokens: int = 8
    speculative_draft_quantization: str = "unquant"
    speculative_draft_attention_backend: str = "flashinfer"

    decode_log_interval: int = 50
    startup_timeout_minutes: int = 30

    # Let an in-flight long generation finish before Modal tears down a Server.
    # The @modal.exit handler itself uses a much shorter subprocess wait.
    exit_grace_period_seconds: int = 300
    shutdown_timeout_seconds: int = 20


CONFIG = ServingConfig()

MODEL_STORE_PATH = "/models"
TARGET_MODEL_PATH = f"{MODEL_STORE_PATH}/target"
DRAFT_MODEL_PATH = f"{MODEL_STORE_PATH}/draft"

COMPILE_CACHE_PATH = "/compile-cache"
TRITON_CACHE_PATH = f"{COMPILE_CACHE_PATH}/triton"
TORCHINDUCTOR_CACHE_PATH = f"{COMPILE_CACHE_PATH}/torchinductor"
SGLANG_CACHE_PATH = f"{COMPILE_CACHE_PATH}/sglang"


def build_sglang_command(port: int | None = None) -> list[str]:
    c = CONFIG
    listen_port = c.port if port is None else port

    return [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        TARGET_MODEL_PATH,
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
        "--max-mamba-cache-size",
        str(c.max_running_requests * c.mamba_slots_per_request),
        "--mamba-radix-cache-strategy",
        c.mamba_radix_cache_strategy,
        "--mamba-ssm-dtype",
        c.mamba_ssm_dtype,
        "--chunked-prefill-size",
        str(c.chunked_prefill_size),
        "--max-prefill-tokens",
        str(c.max_prefill_tokens),
        "--speculative-algorithm",
        c.speculative_algorithm,
        "--speculative-draft-model-path",
        DRAFT_MODEL_PATH,
        "--speculative-num-draft-tokens",
        str(c.speculative_num_draft_tokens),
        "--speculative-draft-model-quantization",
        c.speculative_draft_quantization,
        "--speculative-draft-attention-backend",
        c.speculative_draft_attention_backend,
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