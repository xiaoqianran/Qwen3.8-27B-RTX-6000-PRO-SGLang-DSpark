# Modal deployment

This directory is an **independent deployment layer** for this fork.

It intentionally does **not** import, execute, copy, or patch the repository's
existing `start.sh`, `stop.sh`, `patch/`, or Docker workflow. Upstream changes
to those files can therefore be merged independently of the Modal deployment.

## Architecture

```text
deploy/
├── modal_app.py        # Modal lifecycle + SGLang Server
├── modal_config.py     # all serving knobs in one place
├── modal_benchmark.py  # local streaming decode benchmark
└── README.md
```

The runtime dependency chain is:

```text
uv -> Modal -> official SGLang image -> Hugging Face target/draft checkpoints
```

The default profile is optimized for **single-user decode latency/throughput**:

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

All of those settings live in `modal_config.py`.

## Run

From the repository root:

```bash
uv run modal setup
uv run modal run deploy/modal_app.py
```

For a longer decode measurement:

```bash
uv run modal run deploy/modal_app.py --max-tokens 2048
```

`modal run` starts a temporary server, performs three warmup requests, then runs
one streaming benchmark and reports `DECODE TOK/S`.

## Deploy a persistent endpoint

```bash
uv run modal deploy deploy/modal_app.py
```

The server is OpenAI-compatible:

```text
POST <modal-url>/v1/chat/completions
```

Model name:

```text
qwen3.8-27b
```

The current server is declared `unauthenticated=True`, which means a deployed
URL is public. Add application-level authentication before exposing a paid
production endpoint.

## SGLang image policy

The default is:

```text
lmsysorg/sglang:dev-cu13
```

DFlash2 is checked during the Modal image build:

```python
from sglang.srt.models.dflash import DFlash2DraftModel
```

If a future moving `dev-cu13` image becomes incompatible, image construction
fails before an RTX PRO 6000 container is started.

For stronger reproducibility, point the deployment at a known-good dated SGLang
tag without changing source code:

```powershell
$env:QWEN38_SGLANG_IMAGE="lmsysorg/sglang:<known-good-tag>"
uv run modal run deploy/modal_app.py --max-tokens 2048
```

or on bash:

```bash
QWEN38_SGLANG_IMAGE="lmsysorg/sglang:<known-good-tag>" \
  uv run modal run deploy/modal_app.py --max-tokens 2048
```

## Cache behavior

Two Modal Volumes are independent of the repository:

- `qwen38-27b-hf-cache` — Hugging Face model cache
- `qwen38-27b-triton-cache` — Triton/JIT cache

The first run downloads the model checkpoints. Later containers reuse the
Volumes rather than re-downloading the weights.

## Updating upstream

Normal fork maintenance remains separate:

```text
upstream changes
    |
    +--> README.md / start.sh / stop.sh / patch/
              |
              +---- no runtime dependency ----> deploy/
```

When upstream finds a better serving recipe, port the desired parameter change
into `deploy/modal_config.py` deliberately. Do not make `deploy/` source values
from `start.sh`; that would recreate the coupling this directory is designed to
avoid.
