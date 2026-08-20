# Modal deployment

`deploy/` is an independent Modal-native backend. It does not execute or import
`start.sh`, `stop.sh`, or `patch/`.

## Structure

```text
deploy/
├── modal_config.py     # serving/storage settings
├── model_prepare.py    # CPU-only Hugging Face download + validation
├── modal_app.py        # Modal orchestration + SGLang GPU server
└── modal_benchmark.py  # single-stream decode benchmark
```

## Architecture

```text
CPU preparation image (8 CPU, no GPU)
        │
        ├── Qwen3.8-27B-NVFP4
        └── Qwen3.8-27B-DFlash2
        │
        ▼
qwen38-27b-model-store (persistent Modal Volume)
        │ read-only
        ▼
RTX PRO 6000
        │
        └── SGLang + DFlash2 + FlashInfer
```

The GPU server uses only local paths:

```text
--model-path /models/target
--speculative-draft-model-path /models/draft
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

If the model Volume is incomplete, GPU startup fails instead of downloading.
GPU-side model storage is mounted read-only.

A separate `qwen38-27b-compile-cache` Volume persists Triton, TorchInductor,
and SGLang compilation artifacts.

## Required Modal image builder

Use Modal Image Builder `2025.06`. Older builders can inject legacy Modal
runtime dependencies into the SGLang image and break its Python environment.

```bash
uv run modal workspace settings set image-builder-version 2025.06
```

This only needs to be configured once per workspace.

## Local client

The root project uses:

```text
modal[api-proxy-support]==1.5.3
```

so local Modal API traffic can use proxy support while the remote SGLang image
remains independent.

## Temporary benchmark

```bash
uv run modal run deploy/modal_app.py --max-tokens 2048
```

Before requesting the RTX PRO 6000, the local entrypoint waits for the CPU model
preparation/readiness function. The benchmark then starts one GPU server, warms
it three times, measures one decode stream, prints `DECODE TOK/S`, and exits.

## Persistent endpoint

```bash
uv run modal deploy deploy/modal_app.py
```

Deploy builds the small CPU preparation image and the SGLang runtime image as
separate build graphs. Model weights remain in the Volume across deployments.
The RTX PRO 6000 is allocated only when the deployed server receives traffic.

## Default inference profile

```text
GPU                       RTX-PRO-6000
Target                    RadixArk/Qwen3.8-27B-NVFP4
Draft                     z-lab/Qwen3.8-27B-DFlash2
DFlash2 draft tokens      8
Attention                 flashinfer
KV cache                  fp8_e4m3
Context                   262144
mem-fraction-static       0.90
SGLang running requests   1
Modal target concurrency  1
```

Change serving values only in `modal_config.py`.

For reproducibility, pin model revisions and the SGLang image with environment
variables:

```powershell
$env:QWEN38_MODEL_REVISION="<commit>"
$env:QWEN38_DRAFT_MODEL_REVISION="<commit>"
$env:QWEN38_SGLANG_IMAGE="lmsysorg/sglang:<known-good-tag>"
uv run modal deploy deploy/modal_app.py
```

Upstream serving improvements should be deliberately ported into
`modal_config.py`; the Modal backend should never source configuration from
`start.sh` at runtime.
