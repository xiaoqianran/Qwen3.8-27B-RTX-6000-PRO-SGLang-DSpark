# Modal deployment

`deploy/` is the Modal-native lifecycle for this repository. It does not execute
`start.sh` or `stop.sh`; it reuses the existing `patch/sglang` DFlash2 backport
so the Docker and Modal backends do not maintain duplicate compatibility code.

## Architecture

```text
CPU download (2 x 4 CPU writers, no GPU)
        │
        ├── RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead
        └── z-lab/Qwen3.8-27B-DFlash2 (only when DFLASH enabled)
        ▼
qwen38-27b-model-store (persistent Volume)
        │ read-only
        ▼
max 1 × RTX PRO 6000 (`Qwen38HTTP`) ← OpenAI-compatible client
        │
        └── lmsysorg/sglang:qwen38-27b + patch/sglang
                └── SGLang + DFlash2 + FlashInfer
        └── max 8 active requests
```

The GPU runtime is offline from Hugging Face and uses only:

```text
--model-path <published target path under /models>
--speculative-draft-model-path <published draft path under /models>  # only DFLASH
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

The downloader uses modal-benchmark 0.1.5 on CPU. Its Reader validates
persisted files, and an independent publish step writes the effective model
paths to the Volume only after verification. GPU startup never downloads.
When speculative decoding is disabled, the DFlash2 draft is not downloaded.

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

The defaults now match the Docker RTX PRO 6000 recipe; the BF16 lm_head and prefill changes still require Modal GPU validation:

```text
GPU                          RTX-PRO-6000
SGLang image                 lmsysorg/sglang:qwen38-27b
Target                       RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead
Draft                        z-lab/Qwen3.8-27B-DFlash2
Attention                    flashinfer
KV cache                     fp8_e4m3
Context                      262144
mem-fraction-static          0.90
chunked-prefill-size         4096
max-prefill-tokens           16384
Mamba radix strategy         extra_buffer
Mamba SSM dtype              float32
DFlash2 draft tokens         8
SGLang max running requests  8
max-mamba-cache-size         40
Modal target concurrency     8
Modal max containers         1
```

Qwen3.8 is a hybrid-GDN model. With `extra_buffer` and the overlap scheduler,
the target needs five base recurrent-state slots per active request. `8 × 5 =
40`, so the explicit Mamba-cache pin prevents the state pool from silently
reducing the requested eight-way SGLang concurrency. The strategy and SSM dtype
are also explicit so this memory assumption cannot drift with a future SGLang
default. DFlash2's draft model is pure attention, but speculative verification
still uses target-model recurrent state.

The previous NVFP4 Modal baseline auto-sized the shared KV pool to 958,178 tokens. The new BF16 lm_head and 0.90 static-memory configuration requires a fresh KV pool measurement. The 262,144 context setting is a per-request upper bound, not a promise that eight requests
can each keep a full 256K context resident simultaneously. Concurrent requests
share that KV pool; SGLang remains responsible for scheduling/retraction under
long-context pressure.

`--min-free-slots-delay` is intentionally not overridden. The RTX PRO 6000
profile now matches the Docker recipe: 4096-token prefill chunk and 16384-token maximum
prefill budget.

## Cold-start optimization

Cold-start work on `main` is isolated from the stable `v0.1.0` tag. See
[`COLD_START.md`](./COLD_START.md) for the full design and A/B protocol.

Disk-backed runtime artifacts now use a versioned cache namespace keyed by the
resolved target/draft revisions and runtime configuration. The cache persists
Triton, TorchInductor, SGLang, and FlashInfer autotune artifacts and is explicitly
committed after warmup.

The default profile remains full CUDA-graph capture, but the updated prefill settings require validation. The `fast` profile uses these 11 buckets:

```text
4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096
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

## Direct HTTP and cold starts

Clients connect to the `Qwen38HTTP` Modal Server URL with `/v1` appended.
SGLang serves model discovery, chat completions, and streaming directly over
HTTP/1.1. The model ID is `qwen3.8-27b`.

The first request can trigger a GPU cold start. Allow time for the web endpoint to become ready before retrying. Discovery and health checks also reach the GPU Server, so
continuous polling can wake the GPU or keep it warm.

If upgrading from the two-App deployment, deploy the Server, point clients at
its URL, and verify a real chat request before retiring the old
`qwen38-27b-gateway` App. The old `qwen38server` Web Function address belongs
to that App.

The RTX PRO 6000 backend explicitly uses a 600-second idle scaledown window
(default, overridable with `QWEN38_GPU_SCALEDOWN_SECONDS`). This keeps the GPU
warm across ordinary pauses between Cherry Studio turns and is long enough that
a 90-120 second cold boot cannot consume the entire idle lifetime. See
[`AUTOSCALING.md`](./AUTOSCALING.md) for the lifecycle and shutdown semantics.

## Lifecycle

SGLang runs as a subprocess in its own process session. Modal's autoscaling
target is aligned with SGLang at eight concurrent requests; it is not a hard
request cap. `max_containers=1` guarantees that demand can never allocate a
second GPU, and excess work can queue instead.

The `@modal.exit` handler gives SGLang 20 seconds to terminate and force-kills its process group only if graceful shutdown stalls.

`modal run` is temporary. After its local benchmark returns, Modal tears down the
temporary Server, so a final SGLang `SIGTERM`/shutdown sequence in remote logs is
expected. The validated v02 run exited with zero remaining requests and no
shutdown traceback. `modal deploy` registers the persistent Server URL and
allows its GPU container count to scale to zero.

## One-time Modal workspace setup

This project pins Modal SDK `1.6.4` and uses Image Builder `2026.10`.
The CPU downloader uses Debian with Python 3.12. The GPU image
continues to use the SGLang registry image and the DFlash2 patch. If a
workspace uses a different default, select the builder for the current shell:

```powershell
$env:MODAL_IMAGE_BUILDER_VERSION = "2026.10"
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

Deploy the Server and look up its URL:

```bash
uv run modal deploy deploy/modal_app.py
uv run python -c "import modal; print(modal.Cls.from_name('qwen38-27b-modal', 'Qwen38HTTP')().serve.get_web_url())"
```

Do **not** routinely redeploy `deploy/modal_app.py` while users are actively
generating. Modal deployment version changes can roll the Server revision after
in-flight requests drain.

Runtime limits are deliberately simple:

```text
GPU backend          min 0 / max 1 / concurrent inputs 8 / 600 s scaledown
GPU replicas         0 or 1
SGLang requests      up to 8 active inside that one GPU
```

The local benchmark targets the same Server URL as external clients and retries
expected 502/503/504 health responses while the GPU Server scales from zero.
Set `QWEN38_REQUIRE_PROXY_AUTH=1` when deploying to require a
[Modal Proxy Token](https://modal.com/docs/guide/webhook-proxy-auth). An
OpenAI-compatible client can use `<token-id>.<token-secret>` as its API key.
For a custom Bearer key, set `QWEN38_API_KEY_SECRET` to the name of a Modal
Secret containing `SGLANG_HTTP_API_KEY`, and leave `QWEN38_REQUIRE_PROXY_AUTH`
unset. SGLang checks the key in the GPU container, so requests can trigger
GPU startup before this check runs.

To serve Pi LoRA, set `QWEN38_PI_LORA=1`. The cookbook image requires
`QWEN38_SPECULATIVE_ALGORITHM=NONE`. For DFlash2 with dynamic LoRA, use
`QWEN38_SPECULATIVE_ALGORITHM=DFLASH` and
`QWEN38_SGLANG_IMAGE=lmsysorg/sglang:nightly-dev-cu13-20260823-eec794bc`.
This image uses its native DFlash2 worker and skips the cookbook overlay.
The adapter is mounted read-only. Select `qwen3.8-27b:qwen3.8-27B-pi`
in API requests. On RTX PRO 6000 with builder 2026.10 this combination passed
chat, SSE, a 130,029-token prompt and a Pi CLI file-read tool round trip.

## 使用 modal-benchmark 0.2.0 下载权重

新流程完全独立于 GPU：一个 Coordinator、两个直接 Writer，不再有额外的 Discover/Inspect/Reader/Report/Publish 容器或远程队列。Writer 使用 Xet 下载和分批 Commit；Coordinator 刷新并检查 Volume，完成后一次性发布 `/models/.modal-model-manifest.json`。GPU 只读取清单，不复制权重。

```powershell
# 离线预览（不启动云端 CPU/GPU）
uv run python -m deploy.download_models

# 首次准备或补齐目标模型；启用 DFLASH 时同时下载 Draft；可选 --workers 1..5
uv run python -m deploy.download_models --execute --workers 2

# 下载成功后再部署；部署与 GPU 冷启动均不触发下载
uv run modal deploy deploy/modal_app.py
```

默认 Volume：`qwen38-27b-model-store`。下载器需要 Modal 认证和包含 `HF_TOKEN` 的 `huggingface-secret` Secret；它会产生 CPU、网络和存储费用。现有 Volume 的旧权重不会被删除，但没有 v2 发布清单的旧缓存不能直接启动 GPU。Pi LoRA 仍挂载旧 `modal-vllm-sglang` Volume，本次暂未改变适配器下载路径。`modal-benchmark` 依赖固定于 v0.2.0 Git commit `60c3e31f5812c98c2cd51618bdfcaed061616bdf`。
