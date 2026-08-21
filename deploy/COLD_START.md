# Cold-start optimization

This document tracks cold-start work on top of the validated `v0.1.0` Modal
baseline. Decode settings remain unchanged unless an experiment is explicitly
enabled.

## Verified observations from v04-v07

Four cache-only runs covered full/fast × cold/warm cache:

```text
run   graph profile   disk cache   total
v04   full            cold         164.86 s
v05   full            warm         148.83 s
v06   fast            cold         154.53 s
v07   fast            warm         130.40 s
```

Important measured components:

```text
full cold prefill graph   33.58 s
full warm prefill graph   26.66 s
fast cold prefill graph   18.27 s
fast warm prefill graph    8.82 s

v07 target verify graph    5.05 s
v07 draft verify graph    13.36 s
v07 weight load           21.11 s
v07 scheduler_e2e         72.16 s
```

The first optimization stage therefore reduced total startup by about 34.5 s
from v04 to v07, and reduced prefill CUDA-graph capture by about 74%.

## SGLang import prefetch

The Modal image already uses the documented import prefetch hint:

```python
with sglang_image.imports():
    import sglang
```

Keep this enabled. It is a low-risk startup optimization and is independent of
the CUDA-graph experiments below.

## What is cacheable

Disk-backed artifacts are persisted in the Modal compile-cache Volume:

```text
Triton cache
TorchInductor cache
SGLang cache
FlashInfer autotune JSON
```

CUDA Graph captures are process/GPU-state objects and are not treated as disk
artifacts. They must be captured for each new GPU process.

## Shared versioned runtime cache

As of runtime-cache epoch `2`, full and fast graph profiles share the same disk
cache namespace:

```text
/compile-cache/runtime/<16-char-key>/
```

The key includes the resolved target/draft Hugging Face revisions plus settings
that can invalidate compiled/autotuned artifacts:

```text
runtime cache epoch
SGLang image
GPU type
language-only mode
attention backend
KV dtype
memory/context settings
concurrency/Mamba settings
DFlash2 settings
```

The key deliberately does **not** include:

```text
cold-start graph profile
prefill CUDA-graph shape list
decode/verify CUDA-graph shape list
```

Those graphs are recaptured every process anyway. Keeping graph-only choices out
of the key allows `full` and `fast` to reuse the same Triton/Inductor/FlashInfer
disk artifacts instead of paying a second JIT/autotune seed cost.

The namespace contains:

```text
manifest.json
triton/
torchinductor/
sglang/
  flashinfer/autotune/...
```

`SGLANG_FLASHINFER_AUTOTUNE_CACHE=1` remains explicit. FlashInfer cache hits are
useful but do not remove the autotune forward entirely; v04-v07 measured roughly
18 s cold versus roughly 15 s warm.

To invalidate disk compatibility deliberately:

```powershell
$env:QWEN38_RUNTIME_CACHE_EPOCH="3"
```

## Health readiness optimization

SGLang normally makes `/health` perform a real one-token generation after the
server leaves `Starting`. The deployment already performs three explicit
128-token OpenAI warmups, so that generation is redundant.

The image now sets:

```text
SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0
```

`/health` therefore becomes a status-only readiness check; the three explicit
OpenAI warmups remain the functional inference validation. Startup logs now
split timing into:

```text
pre_launch
health_wait
engine_ready
warmup
cache_commit
total
```

This makes the health-path saving directly measurable on the next run.

## Prefill graph profiles

### `full` (default)

Preserves the v0.1.0 prefill graph behavior:

```powershell
$env:QWEN38_COLD_START_PROFILE="full"
uv run modal run deploy/modal_app.py --cache-only
```

### `fast`

Captures:

```text
4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048
```

instead of the 42-shape baseline:

```powershell
$env:QWEN38_COLD_START_PROFILE="fast"
uv run modal run deploy/modal_app.py --cache-only
```

v07 measured 8.82 s for this prefill graph set versus 26.66 s for the warm full
profile.

## Experimental sparse decode/verify graphs

Current SGLang's decode CUDA-graph runner pads a raw batch to the smallest
captured bucket greater than or equal to that batch size. For an eight-request
server, an explicit sparse set can therefore represent:

```text
1, 2, 4, 8
```

with 3 padding to 4 and 5/6/7 padding to 8.

The validated default remains the full 1..8 capture. To A/B the sparse path:

```powershell
$env:QWEN38_VERIFY_GRAPH_PROFILE="sparse"
uv run modal run deploy/modal_app.py --cache-only
```

This adds:

```text
--cuda-graph-bs-decode 1 2 4 8
```

Do not make it the default until 1/2/4/8 request inference passes without a
throughput or correctness regression. The persistent disk cache is shared with
the full verify profile because CUDA graph objects themselves are not persisted.

## Experimental language-only mode

Current SGLang documents `--language-only` for VLM-style checkpoints to load the
language model only. This deployment keeps it **off by default** because the
stable baseline uses the original Qwen3.8 conditional-generation path.

A/B only:

```powershell
$env:QWEN38_LANGUAGE_ONLY="1"
uv run modal run deploy/modal_app.py --cache-only
```

When enabled, image build first requires the selected SGLang image to expose the
flag; otherwise the build fails before GPU allocation. The mode is part of the
disk-cache identity because it can change model construction and compiled paths.

Potential benefits to measure:

```text
avoid multimodal processor discovery/initialization
avoid the 1024 MiB multimodal CUDA IPC reservation
more KV/cache headroom
possibly shorter startup
```

Do not merge this behavior into the default until NVFP4 + DFlash2 model load,
three warmups, and the normal concurrency benchmark all pass.

## Recommended next runs

First establish the new shared cache and health-readiness baseline with fast
prefill but normal verify graphs:

```powershell
$env:QWEN38_COLD_START_PROFILE="fast"
$env:QWEN38_VERIFY_GRAPH_PROFILE="full"
Remove-Item Env:QWEN38_LANGUAGE_ONLY -ErrorAction SilentlyContinue

uv run modal run deploy/modal_app.py --cache-only
uv run modal run deploy/modal_app.py --cache-only
```

Both runs should print the **same runtime cache key**. The second should show a
non-zero `flashinfer_entries_before`.

Then A/B sparse verify graphs while reusing that same disk cache:

```powershell
$env:QWEN38_VERIFY_GRAPH_PROFILE="sparse"
uv run modal run deploy/modal_app.py --cache-only
```

Finally, test language-only separately:

```powershell
$env:QWEN38_VERIFY_GRAPH_PROFILE="full"
$env:QWEN38_LANGUAGE_ONLY="1"
uv run modal run deploy/modal_app.py --cache-only
```

Language-only intentionally gets a different runtime cache key.

## Acceptance criteria

No experiment becomes the default until it passes:

```text
server reaches /health
3 explicit OpenAI warmups complete
4096-token single-stream benchmark receives SSE [DONE]
1/2/4/8 concurrency sweep completes
no extra GPU replica is created
no meaningful decode-throughput regression
no material TTFT regression for representative prompts
clean shutdown with zero remaining requests
```

The stable `v0.1.0` tag remains the rollback point while these experiments are
performed on `main`.
