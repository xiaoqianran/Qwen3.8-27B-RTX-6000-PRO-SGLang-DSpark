# Cold-start optimization

This document tracks cold-start work on top of the validated `v0.1.0` Modal
baseline. Decode settings are intentionally unchanged while startup work is
optimized and measured.

## Baseline

The validated RTX PRO 6000 boot reported approximately:

```text
weight load                 13.82 s
FlashInfer autotune         ~12 s
prefill CUDA graph          17.94 s
target verify CUDA graph     3.47 s
draft verify CUDA graph      6.76 s
scheduler startup           62.17 s
```

The goal is to reduce startup without changing the validated NVFP4 + DFlash2
serving path or its 1/2/4/8-request decode behavior.

## What is cacheable

Disk-backed artifacts are persisted in the Modal compile-cache Volume:

```text
Triton cache
TorchInductor cache
SGLang cache
FlashInfer autotune JSON
```

CUDA Graph captures are process/GPU-state objects and are not treated as disk
artifacts. Their startup cost is reduced by capturing fewer shapes instead of
trying to serialize them like wheels.

## Versioned runtime cache

Each GPU runtime creates a cache namespace under:

```text
/compile-cache/runtime/<16-char-key>/
```

The key includes the resolved target and draft Hugging Face revisions plus the
runtime settings that can invalidate compiled/autotuned artifacts:

```text
runtime cache epoch
SGLang image
GPU type
cold-start profile
prefill CUDA-graph shapes
attention backend
KV dtype
memory/context settings
concurrency/Mamba settings
DFlash2 settings
```

The namespace contains:

```text
manifest.json
triton/
torchinductor/
sglang/
  flashinfer/autotune/...
```

`SGLANG_FLASHINFER_AUTOTUNE_CACHE=1` is explicit. Startup prints the number of
FlashInfer cache JSON files before and after SGLang warmup, then explicitly
commits the Modal Volume.

If cache compatibility must be invalidated manually, bump the epoch:

```powershell
$env:QWEN38_RUNTIME_CACHE_EPOCH="2"
```

A different epoch produces a different runtime cache key without deleting old
artifacts.

## Profiles

### `full` (default)

`full` preserves the validated v0.1.0 CUDA-graph behavior. SGLang owns its full
prefill graph shape list.

```powershell
Remove-Item Env:QWEN38_COLD_START_PROFILE -ErrorAction SilentlyContinue
uv run modal run deploy/modal_app.py --cache-only
```

### `fast`

`fast` keeps the same 2048-token chunked-prefill ceiling but captures a sparse
power-of-two prefill set:

```text
4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048
```

This preserves small-prompt buckets and full 2048 coverage while reducing the
42-shape baseline capture set to 10 shapes.

```powershell
$env:QWEN38_COLD_START_PROFILE="fast"
uv run modal run deploy/modal_app.py --cache-only
```

The image build fails before GPU allocation if the selected SGLang base image
does not expose `--cuda-graph-bs-prefill`.

## A/B protocol

For each profile, run it twice. The first run seeds the versioned disk cache;
the second run measures a cache hit.

```powershell
# full, seed + measure
$env:QWEN38_COLD_START_PROFILE="full"
uv run modal run deploy/modal_app.py --cache-only
uv run modal run deploy/modal_app.py --cache-only

# fast, seed + measure
$env:QWEN38_COLD_START_PROFILE="fast"
uv run modal run deploy/modal_app.py --cache-only
uv run modal run deploy/modal_app.py --cache-only
```

Key log lines:

```text
Runtime cache: profile=... key=... flashinfer_entries_before=...
Running FlashInfer autotune with cache: ...
Runtime cache committed: key=... flashinfer_entries=before->after
Cold-start timing: engine_ready=... warmup=... cache_commit=... total=...
```

A useful cache hit should show a non-zero `flashinfer_entries_before` and a lower
FlashInfer/autotune startup contribution. A useful `fast` profile should also
show a materially shorter prefill CUDA-graph capture while preserving correct
requests and decode throughput.

## Acceptance criteria

Do not make `fast` the default until it passes all of the following against
`full`:

```text
server reaches /health
3 warmups complete
4096-token single-stream benchmark completes with SSE [DONE]
1/2/4/8 concurrency sweep completes
no extra GPU replica is created
no meaningful decode-throughput regression
no material TTFT regression for representative short/medium prompts
```

The stable `v0.1.0` tag remains the rollback point while these experiments are
performed on `main`.
