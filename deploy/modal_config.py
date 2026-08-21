"""Configuration for the Modal deployment."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


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

    # Cold-start A/B profile. `full` is the v0.1.0 behavior. `fast` keeps a
    # sparse power-of-two prefill CUDA-graph set while preserving 2048 coverage.
    # Graph profile is intentionally NOT part of the persistent disk-cache key:
    # CUDA graphs are process-local, while Triton/Inductor/FlashInfer artifacts
    # are reusable by both full and fast graph profiles.
    cold_start_profile: str = os.environ.get(
        "QWEN38_COLD_START_PROFILE", "full"
    ).strip().lower()
    runtime_cache_epoch: str = os.environ.get(
        "QWEN38_RUNTIME_CACHE_EPOCH", "2"
    ).strip()
    fast_prefill_cuda_graph_tokens: tuple[int, ...] = (
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
    )

    # Experimental decode/verify graph A/B. SGLang's decode CUDA-graph runner
    # pads a raw batch to the smallest captured bucket >= raw batch size. Keep
    # the validated default (`full`) unless explicitly testing `sparse`.
    verify_graph_profile: str = os.environ.get(
        "QWEN38_VERIFY_GRAPH_PROFILE", "full"
    ).strip().lower()
    sparse_decode_cuda_graph_bs: tuple[int, ...] = (1, 2, 4, 8)

    # Experimental text-only A/B. Off by default because the validated baseline
    # loads the original conditional-generation checkpoint path unchanged.
    language_only: bool = _env_bool("QWEN38_LANGUAGE_ONLY", False)

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

    # One GPU replica only. app.py uses max_running_requests as Modal's
    # autoscaling target too; excess demand can queue but cannot add a 2nd GPU.
    modal_max_containers: int = 1

    # Keep the expensive GPU warm across normal pauses between interactive chat
    # turns. This is intentionally much longer than the ~90-120 s cold boot.
    # Override with QWEN38_GPU_SCALEDOWN_SECONDS when a different cost/latency
    # trade-off is desired.
    gpu_scaledown_window_seconds: int = int(
        os.environ.get("QWEN38_GPU_SCALEDOWN_SECONDS", "600")
    )

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

    def __post_init__(self) -> None:
        if self.cold_start_profile not in {"full", "fast"}:
            raise ValueError(
                "QWEN38_COLD_START_PROFILE must be either 'full' or 'fast'"
            )
        if self.verify_graph_profile not in {"full", "sparse"}:
            raise ValueError(
                "QWEN38_VERIFY_GRAPH_PROFILE must be either 'full' or 'sparse'"
            )
        if not self.runtime_cache_epoch:
            raise ValueError("QWEN38_RUNTIME_CACHE_EPOCH must not be empty")
        if self.gpu_scaledown_window_seconds < 1:
            raise ValueError("QWEN38_GPU_SCALEDOWN_SECONDS must be >= 1")
        if self.fast_prefill_cuda_graph_tokens[-1] != self.chunked_prefill_size:
            raise ValueError(
                "fast prefill CUDA-graph coverage must reach chunked_prefill_size"
            )
        if self.sparse_decode_cuda_graph_bs[-1] != self.max_running_requests:
            raise ValueError(
                "sparse decode CUDA-graph coverage must reach max_running_requests"
            )


CONFIG = ServingConfig()

MODEL_STORE_PATH = "/models"
TARGET_MODEL_PATH = f"{MODEL_STORE_PATH}/target"
DRAFT_MODEL_PATH = f"{MODEL_STORE_PATH}/draft"

COMPILE_CACHE_PATH = "/compile-cache"


def runtime_cache_identity() -> dict[str, object]:
    """Identity only for disk-persistable GPU compilation/autotune artifacts.

    Deliberately exclude CUDA-graph profile and shape lists. CUDA graphs are
    process-local and recaptured on every cold boot, while disk artifacts can be
    shared safely across graph-shape A/B profiles for the same model/runtime.
    """
    c = CONFIG
    return {
        "epoch": c.runtime_cache_epoch,
        "sglang_image": c.sglang_image,
        "gpu": c.gpu,
        "language_only": c.language_only,
        "attention_backend": c.attention_backend,
        "kv_cache_dtype": c.kv_cache_dtype,
        "mem_fraction_static": c.mem_fraction_static,
        "context_length": c.context_length,
        "chunked_prefill_size": c.chunked_prefill_size,
        "max_prefill_tokens": c.max_prefill_tokens,
        "max_running_requests": c.max_running_requests,
        "mamba_cache_size": c.max_running_requests * c.mamba_slots_per_request,
        "mamba_radix_cache_strategy": c.mamba_radix_cache_strategy,
        "mamba_ssm_dtype": c.mamba_ssm_dtype,
        "speculative_algorithm": c.speculative_algorithm,
        "speculative_num_draft_tokens": c.speculative_num_draft_tokens,
        "speculative_draft_quantization": c.speculative_draft_quantization,
        "speculative_draft_attention_backend": c.speculative_draft_attention_backend,
    }


def build_sglang_command(port: int | None = None) -> list[str]:
    c = CONFIG
    listen_port = c.port if port is None else port

    command = [
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
    ]

    if c.language_only:
        command.append("--language-only")

    if c.cold_start_profile == "fast":
        command.extend(
            [
                "--cuda-graph-bs-prefill",
                *[str(value) for value in c.fast_prefill_cuda_graph_tokens],
            ]
        )

    if c.verify_graph_profile == "sparse":
        command.extend(
            [
                "--cuda-graph-bs-decode",
                *[str(value) for value in c.sparse_decode_cuda_graph_bs],
            ]
        )

    command.extend(
        [
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
    )
    return command
