# Modal deployment

This directory is an **independent Modal deployment layer** for this fork.

It intentionally does **not** import, execute, copy, or patch the repository's
existing `start.sh`, `stop.sh`, `patch/`, or Docker workflow. Upstream changes
to those files can therefore be merged independently of the Modal deployment.

## Architecture

```text
deploy/
├── modal_app.py        # CPU model preparation + Modal lifecycle + SGLang Server
├── modal_config.py     # serving/storage/download knobs in one place
├── modal_benchmark.py  # local streaming decode benchmark
└── README.md
```

Runtime dependency chain:

```text
uv -> Modal -> official SGLang image -> persistent Modal model Volume
                                         ├── Qwen3.8-27B NVFP4
                                         └── DFlash2
```

The default profile is optimized for **single-user decode**:

- GPU: `RTX-PRO-6000`
- target: `RadixArk/Qwen3.8-27B-NVFP4`
- draft: `z-lab/Qwen3.8-27B-DFlash2`
- DFlash2 draft tokens: `8`
- target/draft attention backend: `flashinfer`
- KV cache: `fp8_e4m3`
- static memory fraction: `0.90`
- native context: `262144`
- SGLang max running requests: `1`
- Modal target concurrency: `1`

All deployment settings live in `modal_config.py`.

## Important: CPU downloads, GPU only loads/serves

Model download is intentionally separated from GPU execution.

On the first build, Modal executes an `Image.run_function(...)` build step with:

```text
GPU: none
CPU: 8 cores
Hugging Face download workers: 16
```

That CPU-only step materializes both complete Hugging Face snapshots into the
persistent Volume:

```text
qwen38-27b-model-store
└── /models
    ├── target/   # RadixArk/Qwen3.8-27B-NVFP4
    └── draft/    # z-lab/Qwen3.8-27B-DFlash2
```

Only after the image/model preparation succeeds can Modal start the RTX PRO
6000 server.

The GPU server launches SGLang with **local paths**:

```text
--model-path /models/target
--speculative-draft-model-path /models/draft
```

and explicitly sets:

```text
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

Therefore the GPU container cannot silently fall back to downloading the model
from Hugging Face. If the model Volume is incomplete, startup fails instead.

This means the expensive RTX PRO 6000 is used for:

1. reading already-persisted weights from the Modal Volume,
2. loading weights into GPU memory,
3. CUDA/SGLang initialization and graph/kernel preparation,
4. warmup and inference.

Network download/hashing of model shards happens before GPU allocation.

## Persistent compile cache

A second Volume is used for GPU-specific compilation artifacts:

```text
qwen38-27b-compile-cache
```

It stores:

```text
/compile-cache/triton
/compile-cache/torchinductor
/compile-cache/sglang
```

Some CUDA/kernel preparation necessarily requires the actual GPU on the first
cold start, but later containers can reuse these caches instead of rebuilding
everything.

## Local Modal client with API proxy support

The root `pyproject.toml` depends on:

```text
modal[api-proxy-support]==1.5.3
```

Equivalent manual install:

```bash
uv pip install 'modal[api-proxy-support]'
```

Use `uv run` so the proxy-enabled Modal client is always selected.

## Run a temporary benchmark

From the repository root:

```bash
uv run modal setup
uv run modal run deploy/modal_app.py --max-tokens 2048
```

The first ever invocation may spend time in the **CPU image-build/model-download
stage**, but it does not allocate the RTX PRO 6000 for that download.

After the model Volume exists, later runs reuse it.

`modal run` then starts a temporary GPU server, performs three warmup requests,
runs one streaming benchmark, prints `DECODE TOK/S`, and exits.

## Deploy a persistent endpoint

```bash
uv run modal deploy deploy/modal_app.py
```

Image build/model preparation still happens before the GPU server is eligible
to start, so `modal deploy` does not require a separate manual download command.

The deployed server is OpenAI-compatible:

```text
POST <modal-url>/v1/chat/completions
```

Model name:

```text
qwen3.8-27b
```

The current server is `unauthenticated=True`; add application/proxy
authentication before exposing a paid production endpoint.

## Model revisions

By default the model repositories use their current default revisions.

For reproducible deployment, pin exact Hugging Face revisions locally:

PowerShell:

```powershell
$env:QWEN38_MODEL_REVISION="<target-commit-sha>"
$env:QWEN38_DRAFT_MODEL_REVISION="<draft-commit-sha>"
uv run modal deploy deploy/modal_app.py
```

Bash:

```bash
QWEN38_MODEL_REVISION="<target-commit-sha>" \
QWEN38_DRAFT_MODEL_REVISION="<draft-commit-sha>" \
uv run modal deploy deploy/modal_app.py
```

Changing these build arguments causes the CPU preparation step to target the
new snapshots.

## SGLang image policy

Default:

```text
lmsysorg/sglang:dev-cu13
```

DFlash2 support is checked during Modal image construction:

```python
from sglang.srt.models.dflash import DFlash2DraftModel
```

For stronger reproducibility, use a known-good pinned SGLang image:

```powershell
$env:QWEN38_SGLANG_IMAGE="lmsysorg/sglang:<known-good-tag>"
uv run modal run deploy/modal_app.py --max-tokens 2048
```

## Storage cleanup

The old deployment used the Volume:

```text
qwen38-27b-hf-cache
```

The optimized deployment no longer uses it. After verifying the new
`qwen38-27b-model-store` works, the old cache Volume can be removed manually if
you no longer need it.

Do not delete `qwen38-27b-model-store` unless you want the CPU build step to
download the model snapshots again.

## Updating upstream

Normal fork maintenance remains separate:

```text
upstream changes
    |
    +--> README.md / start.sh / stop.sh / patch/
              |
              +---- no runtime dependency ----> deploy/
```

When upstream finds a better serving recipe, port only the desired parameter
changes into `deploy/modal_config.py`. Do not source values dynamically from
`start.sh`; that would recreate the coupling this directory is designed to
avoid.
