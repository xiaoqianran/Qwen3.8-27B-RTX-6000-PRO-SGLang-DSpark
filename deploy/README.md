# Modal deployment

`deploy/` is the Modal-native lifecycle for this repository. It does not execute
`start.sh` or `stop.sh`, but intentionally reuses the repository's existing
`patch/sglang` DFlash2 compatibility layer so the Docker and Modal backends run
the same SGLang recipe instead of maintaining duplicate patch code.

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
lmsysorg/sglang:qwen38-27b
        +
shared patch/sglang DFlash2 backport
        │
        ▼
1 × RTX PRO 6000 maximum
        │
        └── SGLang + DFlash2 + FlashInfer
              └── up to 8 active requests
```

The GPU server uses only local model paths and runs Hugging Face offline:

```text
--model-path /models/target
--speculative-draft-model-path /models/draft
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

CPU preparation resolves each Hugging Face revision, downloads all shards, and
writes a manifest. Before GPU allocation, readiness checks verify the configured
repositories and every safetensors shard referenced by an index.

A separate `qwen38-27b-compile-cache` Volume persists Triton, TorchInductor,
and SGLang compilation artifacts.

## Why the shared patch is still required

DFlash2 landed upstream after the `qwen38-27b` image was built. The currently
available moving `dev-cu13` image can also lag the GitHub main branch, so checking
main source is not sufficient to prove the registry image contains DFlash2.

Modal therefore bakes the existing `patch/sglang` tree into the SGLang image at
build time and immediately verifies:

```python
from sglang.srt.models.dflash import DFlash2DraftModel
```

If a future official image contains DFlash2, this compatibility layer can be
removed once that exact image has been tested.

## One-time Modal workspace setup

Use Image Builder `2025.06`; older builders can inject legacy Modal runtime
dependencies into third-party Python images.

```bash
uv run modal workspace settings set image-builder-version 2025.06
uv run modal workspace settings list
```

The code also restores `typing_extensions==4.16.0` as a compatibility guard for
legacy builders.

## Temporary benchmark

```bash
uv run modal run deploy/modal_app.py --max-tokens 2048
```

Order:

```text
CPU model preparation/readiness
→ first public health request
→ one RTX PRO 6000 cold start
→ SGLang load
→ three warmups
→ single-stream benchmark
```

The local benchmark retries expected 502/503/504 responses during zero-to-one
cold start.

## Persistent endpoint

```bash
uv run modal deploy deploy/modal_app.py
```

Runtime limits:

```text
min_containers              0
max_containers              1
Modal target_concurrency    unset
SGLang max running requests 8
```

All 1–8 active requests are handled by one SGLang process on one RTX PRO 6000.
Modal is never allowed to start a second GPU container.

## Default inference profile

```text
GPU                       RTX-PRO-6000
SGLang image              lmsysorg/sglang:qwen38-27b
DFlash2 compatibility     patch/sglang
Target                    RadixArk/Qwen3.8-27B-NVFP4
Draft                     z-lab/Qwen3.8-27B-DFlash2
DFlash2 draft tokens      8
Attention                 flashinfer
KV cache                  fp8_e4m3
Context                   262144
mem-fraction-static       0.90
SGLang running requests   8
Modal max containers      1
```

Serving values live in `modal_config.py`. The endpoint is currently
`unauthenticated=True`; add authentication before exposing a paid production
endpoint publicly.
