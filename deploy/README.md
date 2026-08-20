# Modal deployment

`deploy/` is an independent Modal-native backend. It does not execute or import
`start.sh`, `stop.sh`, or `patch/`.

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
writes a manifest. Before the temporary benchmark requests a GPU, a CPU
readiness function verifies the configured repositories and every safetensors
shard referenced by an index. GPU startup validates the same store again before
launching SGLang.

A separate `qwen38-27b-compile-cache` Volume persists Triton, TorchInductor,
and SGLang compilation artifacts.

## One-time Modal workspace setup

Use Image Builder `2025.06`; older builders can inject legacy Modal runtime
dependencies into the SGLang Python environment.

```bash
uv run modal workspace settings set image-builder-version 2025.06
uv run modal workspace settings list
```

The root project pins `modal[api-proxy-support]==1.5.3` for local proxy support.

## Temporary benchmark

```bash
uv run modal run deploy/modal_app.py --max-tokens 2048
```

The order is CPU preparation/readiness → first HTTP health request → one RTX PRO
6000 cold start → three warmups → single-stream benchmark. Modal Servers return
503 while scaling from zero; the local benchmark retries health checks until the
server is ready.

## Persistent endpoint

```bash
uv run modal deploy deploy/modal_app.py
```

The deployed server has:

```text
min_containers              0
max_containers              1
Modal target_concurrency    unset (no horizontal autoscaling target)
SGLang max running requests 8
```

Concurrency belongs to SGLang, not the Modal autoscaler. All 1–8 active requests
are handled by the same SGLang process on the same RTX PRO 6000. The hard
`max_containers=1` cap prevents a second GPU container from being started; this
also means rolling redeploys cannot use a temporary spare replica.

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
SGLang running requests   8
Modal max containers      1
```

Serving settings live only in `modal_config.py`.

For reproducibility, exact model revisions and a known-good SGLang image can be
pinned without changing source:

```powershell
$env:QWEN38_MODEL_REVISION="<commit>"
$env:QWEN38_DRAFT_MODEL_REVISION="<commit>"
$env:QWEN38_SGLANG_IMAGE="lmsysorg/sglang:<known-good-tag-or-digest>"
uv run modal deploy deploy/modal_app.py
```

The endpoint is currently `unauthenticated=True`, which is convenient for direct
OpenAI-compatible clients but makes possession of the URL sufficient to trigger
the single paid GPU. Add Modal Proxy authentication before treating this as a
public production endpoint.

Upstream serving improvements should be deliberately ported into
`modal_config.py`; the Modal backend never sources runtime configuration from
`start.sh`.
