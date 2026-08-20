# Modal deployment

`deploy/` is the Modal-native lifecycle for this repository. It does not execute
`start.sh` or `stop.sh`; it reuses the existing `patch/sglang` DFlash2 backport
so the Docker and Modal backends do not maintain duplicate compatibility code.

## Architecture

```text
CPU preparation (8 CPU, no GPU)
        │
        ├── RadixArk/Qwen3.8-27B-NVFP4
        └── z-lab/Qwen3.8-27B-DFlash2
        ▼
qwen38-27b-model-store (persistent Volume)
        │ read-only
        ▼
lmsysorg/sglang:qwen38-27b + patch/sglang
        ▼
max 1 × RTX PRO 6000
        ▼
SGLang + DFlash2 + FlashInfer
        └── max 8 active requests
```

The GPU runtime is offline from Hugging Face and uses only:

```text
--model-path /models/target
--speculative-draft-model-path /models/draft
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

CPU preparation verifies model metadata and all safetensors shards before GPU
allocation. If model revisions are not pinned, the CPU preparation step is
re-evaluated on each deploy; `snapshot_download` still reuses the persistent
Volume and only fetches missing/changed data.

## SGLang runtime choice

The Qwen3.8 cookbook image predates DFlash2. The Modal image therefore bakes the
repository's existing DFlash2 backport into `lmsysorg/sglang:qwen38-27b` and
fails during image build unless all of these integration points import:

```text
DFlash2DraftModel
DFlashWorkerV2
FusedKVMaterializeHelper
should_apply_lm_head_quant_method
```

It also verifies that the base server exposes `--max-mamba-cache-size` and
`--cuda-graph-bs-prefill`. No RTX PRO 6000 is allocated for these checks.

## RTX PRO 6000 profile

The defaults follow the current SGLang RTX PRO 6000 + NVFP4 baseline and the
configuration that completed real Modal boots with DFlash2:

```text
GPU                          RTX-PRO-6000
SGLang image                 lmsysorg/sglang:qwen38-27b
Target                       RadixArk/Qwen3.8-27B-NVFP4
Draft                        z-lab/Qwen3.8-27B-DFlash2
Attention                    flashinfer
KV cache                     fp8_e4m3
Context                      262144
mem-fraction-static          0.85
chunked-prefill-size         2048
max-prefill-tokens           16384
Mamba radix strategy         extra_buffer
Mamba SSM dtype              float32
DFlash2 draft tokens         8
SGLang max running requests  8
max-mamba-cache-size         40
Modal target concurrency     8
Modal max containers         1
Modal shutdown request grace 300 s
```

Qwen3.8 is a hybrid-GDN model. With `extra_buffer` and the overlap scheduler,
the target needs five base recurrent-state slots per active request. `8 × 5 =
40`, so the explicit Mamba-cache pin prevents the state pool from silently
reducing the requested eight-way SGLang concurrency. The strategy and SSM dtype
are also explicit so this memory assumption cannot drift with a future SGLang
default. DFlash2's draft model is pure attention, but speculative verification
still uses target-model recurrent state.

The validated boot auto-sized the shared KV pool to 958,178 tokens. The 262,144
context setting is a per-request upper bound, not a promise that eight requests
can each keep a full 256K context resident simultaneously. Concurrent requests
share that KV pool; SGLang remains responsible for scheduling/retraction under
long-context pressure.

`--min-free-slots-delay` is intentionally not overridden. The RTX PRO 6000
profile keeps the 2048-token prefill chunk and the observed 16384-token maximum
prefill budget.

## Cold-start optimization

Cold-start work on `main` is isolated from the stable `v0.1.0` tag. See
[`COLD_START.md`](./COLD_START.md) for the full design and A/B protocol.

Disk-backed runtime artifacts now use a versioned cache namespace keyed by the
resolved target/draft revisions and runtime configuration. The cache persists
Triton, TorchInductor, SGLang, and FlashInfer autotune artifacts and is explicitly
committed after warmup.

The default profile remains the validated full CUDA-graph profile. A `fast`
profile reduces the 42 prefill graph shapes to these 10 buckets while preserving
the 2048-token chunk ceiling:

```text
4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048
```

Prepare the default/full cache without running a long benchmark:

```powershell
Remove-Item Env:QWEN38_COLD_START_PROFILE -ErrorAction SilentlyContinue
uv run modal run deploy/modal_app.py --cache-only
```

Prepare/test the fast profile:

```powershell
$env:QWEN38_COLD_START_PROFILE="fast"
uv run modal run deploy/modal_app.py --cache-only
uv run modal run deploy/modal_app.py --max-tokens 4096 --concurrency 1,2,4,8
```

Run each `--cache-only` command twice when measuring disk-cache reuse: the first
run seeds the versioned cache and the second run measures a hit. Startup prints:

```text
Runtime cache: profile=... key=... flashinfer_entries_before=...
Runtime cache committed: key=... flashinfer_entries=before->after
Cold-start timing: engine_ready=... warmup=... cache_commit=... total=...
```

Set `QWEN38_RUNTIME_CACHE_EPOCH` to a new value to deliberately invalidate all
runtime artifacts without deleting older cache namespaces.

## Lifecycle

SGLang runs as a subprocess in its own process session. Modal's autoscaling
target is aligned with SGLang at eight concurrent requests; it is not a hard
request cap. `max_containers=1` guarantees that demand can never allocate a
second GPU, and excess work can queue instead.

`exit_grace_period=300` gives an in-flight long generation time to finish when a
Server is being removed. The `@modal.exit` handler then gives SGLang 20 seconds
to terminate and force-kills its process group only if graceful shutdown stalls.

`modal run` is temporary. After its local benchmark returns, Modal tears down the
temporary Server, so a final SGLang `SIGTERM`/shutdown sequence in remote logs is
expected. The validated v02 run exited with zero remaining requests and no
shutdown traceback. `modal deploy` creates the persistent endpoint; it remains
deployed while still allowing the GPU replica count to scale between zero and
one.

## One-time Modal workspace setup

Use Image Builder `2025.06`:

```bash
uv run modal workspace settings set image-builder-version 2025.06
uv run modal workspace settings list
```

The image also restores `typing_extensions==4.16.0` as a compatibility guard.

## Run

The benchmark default is **4096 output tokens per request**:

```bash
uv run modal run deploy/modal_app.py
```

Long single-stream benchmark:

```bash
uv run modal run deploy/modal_app.py --max-tokens 8192
```

Specific eight-way concurrency at 4096 tokens per request:

```bash
uv run modal run deploy/modal_app.py --max-tokens 4096 --concurrency 8
```

Full one-container concurrency sweep at 4096 tokens per request:

```bash
uv run modal run deploy/modal_app.py --max-tokens 4096 --concurrency 1,2,4,8
```

Heavier 8K-token sweep:

```bash
uv run modal run deploy/modal_app.py --max-tokens 8192 --concurrency 1,2,4,8
```

The comma-separated concurrency levels are executed sequentially against the
same warm GPU container. Each level launches its requests together and never
exceeds SGLang's configured maximum of eight. Prompts include a per-request ID
to avoid benchmarking an accidentally identical full prefix.

Every request must supply both the OpenAI SSE `[DONE]` marker and
`usage.completion_tokens`; a truncated request fails the whole level. The
streaming benchmark timeout is 60 minutes so longer 4K/8K multi-request tests
are not prematurely terminated by the local client. The benchmark reports
per-request TTFT and decode tok/s plus average user decode speed and aggregate
end-to-end throughput.

Persistent endpoint:

```bash
uv run modal deploy deploy/modal_app.py
```

Runtime limits are deliberately simple:

```text
min_containers       0
max_containers       1
target_concurrency   8
GPU replicas         0 or 1
SGLang requests      up to 8 active inside that one GPU
```

The local benchmark retries expected 502/503/504 responses while the Modal
Server scales from zero. The endpoint is currently `unauthenticated=True`; add
authentication before exposing a paid production endpoint publicly.
