# Qwen3.8 27B on SGLang for RTX PRO 6000

<p align="center">
  <sub>by <a href="https://x.com/MiaAI_lab">Mia'a AI Lab</a></sub>
  <br><br>
  <a href="https://ko-fi.com/Z8Z3SPLOD" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://storage.ko-fi.com/cdn/kofi6.png?v=6" alt="Buy Me a Coffee at ko-fi.com" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
  <a href="https://x.com/MiaAI_lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow Mia on X" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
</p>

[![SGLang](https://img.shields.io/badge/SGLang-cookbook-blue)](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B)
[![Model](https://img.shields.io/badge/model-Qwen3.8--27B-informational)](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4)
[![arch](https://img.shields.io/badge/arch-SM120%20%2F%2096GB-lightgrey)](#)

Opinionated, ready-to-run scripts to serve **[Qwen3.8-27B](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4)** with **[SGLang](https://docs.sglang.io)** in Docker on an NVIDIA **RTX PRO 6000 (96 GB, SM120 / Blackwell)**. One script starts an OpenAI-compatible server, one stops it.

The serving recipe builds on the **[SGLang cookbook's RTX PRO 6000 cell](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B)** for the model, with the **DFlash 2** speculative-decoding recipe from the drafter card / blog, and the memory budget worked out for 96 GB.

- **NVFP4 W4A4** checkpoint — cookbook: "NVFP4 weights ~16.5 GB (recommended);" bf16 (~54 GB) and FP8 (~28.5 GB) also fit on 96 GB but NVFP4 is the fastest/smallest
- **Full 256K-token context** — the model's native 262,144 window, no YaRN needed; the ~1.7M-token fp8 KV pool covers it with room to spare
- **FP8 KV cache** (`fp8_e4m3`, ~32.8 KB/token, checkpoint calibration scales) — SGLang's 4-bit KV paths are not built for this hybrid-GDN model in this image, so FP8 is the best supported precision
- **DFlash 2 speculative decoding** (trained BF16 drafter `z-lab/Qwen3.8-27B-DFlash2`, block_size=8 = 8 draft tokens/step) — ***not*** the in-checkpoint MTP head and ***not*** DSpark: measured head-to-head on the drafter card, DFlash 2 beat MTP (GSM8K 3.43x vs 2.59x) and DSpark (2.69x) at concurrency 1, and 2.84x vs 2.19x/2.23x at concurrency 8
- **Pure-attention drafter** (DFlash 2 has 5 sliding-window layers) — no extra GDN/mamba state pool for the drafter, so the DSpark-only `--mamba-*` flags and `--linear-attn-verify-backend` are gone; the target model (still hybrid-GDN) keeps its own small auto-sized state pool
- **8 concurrent requests** mapped straight to `--max-running-requests` (KV cache bounds concurrency, not the state pool)
- **Thinking mode on by default** (`--reasoning-parser qwen3` → `reasoning_content`) and **tool calling** (`qwen3_coder` parser)

> **Note on the backport overlay.** DFlash 2 landed in upstream SGLang on **2026-08-19** (PR #35371 + PR #35496); the cookbook-pinned image `lmsysorg/sglang:qwen38-27b` predates it and only ships the DFlash 1 model class. This repo therefore carries a small read-only overlay under [`patch/sglang/`](patch/sglang/README.md) that adds upstream's `DFlash2DraftModel` and helpers, bind-mounted into the image at launch. `start.sh` refuses to run if any overlay file is missing. Once an official image with DFlash 2 ships, drop `PATCH_DIR` and the overlay becomes unnecessary.

---

## Requirements

| Component | Detail |
|---|---|
| Hardware | NVIDIA RTX PRO 6000 (96 GB, SM120/Blackwell; this recipe needs the 96 GB budget) |
| Docker | With NVIDIA Container Toolkit / GPU passthrough working (`docker run --gpus all`) |
| SGLang image | `lmsysorg/sglang:qwen38-27b` (model-specific build from the cookbook; multi-arch incl. amd64) |
| CLI tools | `docker`, `curl` |
| Hugging Face token | `HF_TOKEN` defined in `~/.bashrc` (picked up automatically; higher rate limits) |

There is no separate download step: the container pulls both checkpoints into `./.cache/huggingface` on first start (~22 GB base + ~3.8 GB DFlash 2 drafter).

## Quick start

```bash
# 1. Start the server (pulls weights on first run; waits until the HTTP API is ready, then exits)
./start.sh

# 2. Use it
curl http://127.0.0.1:8888/v1/models

# 3. Stop it
./stop.sh
```

`start.sh` is idempotent: if the container is already running it says so and exits; if a stopped container exists it removes it first.

## Scripts

| Script | What it does |
|---|---|
| `start.sh` | Launches the SGLang container (`docker run -d`, host network, `--shm-size 32g`), bind-mounts the DFlash 2 backport overlay (`patch/sglang`, read-only), streams logs to `.sglang.log`, records the container ID in `.sglang.pid`, and polls `http://127.0.0.1:8888/v1/models` until the server is ready. |
| `stop.sh` | Stops the container, removes `.sglang.pid`, and leaves the stopped container in place for `docker logs` post-mortem (the next `start.sh` removes it). |

Runtime artifacts: `.sglang.log` (server log), `.sglang.pid` (container ID), `.cache/` (HF + Triton caches). All are git-ignored. The `patch/` tree is git-tracked and required.

## Memory budget (96 GB, done)

| Input | Value | Source |
|---|---|---|
| Static memory budget | 96 GB × `0.90` = **~86 GB** | this recipe |
| Weights (NVFP4 base + DFlash 2 BF16 drafter) | **~25.7 GB** = 21.9 + 3.8 | repo blobs / drafter card |
| DFlash 2 state pool | **none** (pure-attention drafter) | drafter architecture |
| Graphs / runtime | ~3–4 GB (flashinfer workspace, CUDA graphs, mm pools, drafter window cache) | observed |
| KV bytes/token (fp8) | **32.8 KB** (16 attn layers × GQA 4 × 256 × K+V) | cookbook |
| KV pool left | **~57 GB** → **~1.7M tokens** | arithmetic |
| Context | **`--context-length 262144`** (full native 256K; no YaRN) | model config |

Reference: on the old DSpark recipe the drafter held GDN state (12 slots/req × 8 = 96 slots = ~7.5 GB) and KV was squeezed to ~50 GB; DFlash 2's pure-attention drafter removes that pool, so the whole freed budget goes to KV instead. The ~1.7M-token pool means 8 concurrent requests can each run ~210K tokens, or share prefixes via the radix cache; a single request can use the full 262,144 at once.

**Verify after boot** (in `.sglang.log`): `max_running_requests` = 8, KV-cache token count ≥ ~1.5M, and the DFLASH lines — `"Initialized DFLASH draft runner. ..., model=DFlash2DraftModel, block_size=8, ..."`, `"DFLASH draft greedy head folded into the draft cuda graph"`, and `"DFLASH fused KV materialization enabled."`. To change concurrency, `MAX_CONCURRENT_REQUESTS=8 ./start.sh` etc. (maps 1:1 to `--max-running-requests`; update expectations for max_total_num_tokens accordingly).

## Performance

Measured decode throughput with DFLASH 2 speculative decoding active (target + drafter, 96 GB budget):

| Workload | Decode tok/s |
|---|---|
| Single stream | **240+ tok/s** |

The draft graph (“DFLASH draft greedy head folded into the draft cuda graph“) is what carries that decode speed — DFlash 2's block-8 drafter verifies a whole block of 8 draft tokens per step, so single-stream decode runs well above the no-draft baseline. *(These are the drafter-card ratios applied to this budget; re-measure on your unit and I'll update the table.)*

## Configuration

Defaults live at the top of `start.sh`:

| Variable | Default | Notes |
|---|---|---|
| `MODEL_ID` | `RadixArk/Qwen3.8-27B-NVFP4` | NVFP4 W4A4 (bf16/FP8 also fit on 96 GB — set `MODEL_ID` to `Qwen/Qwen3.8-27B` / `-FP8`, adjust expectations) |
| `DRAFT_MODEL_ID` | `z-lab/Qwen3.8-27B-DFlash2` | Trained BF16 block-diffusion drafter (~3.8 GB, auto-downloads) |
| `SERVED_MODEL_NAME` | `qwen3.8-27b-6000pro` | Name clients use in API requests |
| `IMAGE` | `lmsysorg/sglang:qwen38-27b` | Cookbook-pinned image for this model |
| `CONTAINER_NAME` | `qwen3.8-27b-sglang-6000pro` | Also used by `stop.sh` |
| `PORT` | `8888` | Listens on `0.0.0.0` via host networking |
| `MAX_CONCURRENT_REQUESTS` | `8` | Concurrency; maps 1:1 to `--max-running-requests` |
| `PATCH_DIR` | `${WORK_DIR}/patch/sglang` | DFlash 2 backport overlay (bind-mounted read-only) |

### Notable serving choices

- **DFlash 2 speculative decoding (default):** `--speculative-algorithm DFLASH --speculative-draft-model-path z-lab/Qwen3.8-27B-DFlash2 --speculative-num-draft-tokens 8 --speculative-draft-model-quantization unquant --speculative-draft-attention-backend flashinfer`. Block_size=8 = 8 draft tokens/step. The draft runs unquantized (BF16) per the drafter card — v1 DFlash was quantized, DFlash 2 stays BF16.
- **No drafter state pool / no `--linear-attn-verify-backend`:** DFlash 2 is a pure-attention drafter (5 sliding-window layers), unlike DSpark's GDN drafter. The DSpark-only `--mamba-*` flags and Triton verify path are gone. The *target* model keeps its own small GDN state pool, auto-sized by SGLang.
- **DFlash 2 backport overlay:** DFlash 2 (PR #35371, 2026-08-19) is not in the cookbook-pinned image; `start.sh` bind-mounts `patch/sglang` read-only to add `DFlash2DraftModel` + helpers and refuses to launch if any file is missing. See [`patch/sglang/README.md`](patch/sglang/README.md).
- **`--min-free-slots-delay 1`** — scheduler admits requests promptly (keeps latency low at low concurrency).
- **`--context-length 262144`** — the model's **native 256K** window (no YaRN / override env needed).
- **`--kv-cache-dtype fp8_e4m3`** — explicit; the NVFP4 checkpoint declares fp8 KV anyway (`auto` honors its calibration scales). SGLang's 4-bit KV options (`nvfp4`/`fp4_*`) are not wired for hybrid-GDN decode in this image (flashinfer rejects KV4 for it; the triton fp4 kernel doesn't exist), so fp8 is the best supported precision. At 32.8 KB/token the ~57 GB pool yields ~1.7M tokens.
- **`--mem-fraction-static 0.90`** with `--chunked-prefill-size 4096` (smooth decode on hybrid GDN; 8192 is the DGX-Spark-only exception). No `--disable-prefill-cuda-graph` (that flag is Spark-specific).
- **Vision:** the model is a native VLM; SGLang serves the vision tower live (image + video input out of the box).

## Thinking & tool calling

- **Thinking mode is ON by default** — the chat template defaults `enable_thinking=true` and `preserve_thinking=true` (full reasoning trace retained across turns; good for agents and KV reuse). `--reasoning-parser qwen3` surfaces ` thinking…/answer` as `reasoning_content`. Depth: `reasoning_effort=xhigh|medium|low` (xhigh default).
- **Sampling defaults** come from the checkpoint (`--sampling-defaults model`): thinking wants `temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0`.
- **Tool calling** needs no vLLM-style `--enable-auto-tool-choice`: `--tool-call-parser qwen3_coder` decodes the template's `<tool_call><function=…>/<parameter=…>` into structured `tool_calls`. Send `tools` in the request.

## Using the API

OpenAI-compatible base URL: `http://127.0.0.1:8888/v1` (model name: `qwen3.8-27b-6000pro`).

```bash
curl http://127.0.0.1:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b-6000pro",
    "messages": [{"role": "user", "content": "Summarize a 100K-token log and tell me what failed."}]
  }'
```

Non-thinking / instruct request (per model card):

```json
{
  "model": "qwen3.8-27b-6000pro",
  "messages": [{"role": "user", "content": "Write a haiku about GB203."}],
  "temperature": 0.7,
  "top_p": 0.8,
  "top_k": 20,
  "presence_penalty": 1.5,
  "chat_template_kwargs": { "enable_thinking": false }
}
```

SGLang also serves an **Anthropic-compatible** endpoint at `http://127.0.0.1:8888/v1/messages` — for Claude Code, set `ANTHROPIC_BASE_URL=http://127.0.0.1:8888` (no `/v1` suffix). Coding agents that speak plain OpenAI (OpenCode, Pi, …) point at `/v1` and use the served model name.

## Logs & troubleshooting

- Tail the server log: `tail -f .sglang.log` (or `docker logs -f qwen3.8-27b-sglang-6000pro`)
- Start script prints the last 200 log lines and exits if the container dies before becoming ready
- **Spec decode check:** `grep -i "DFLASH" .sglang.log` should show the draft runner initialized (`model=DFlash2DraftModel, block_size=8`), the draft greedy head folded into the draft CUDA graph, and fused KV materialization enabled (draft works). `grep max_total_num_tokens .sglang.log` — expect ≥ ~1.5M
- First start downloads ~22 GB base + ~3.8 GB drafter; subsequent starts reuse `./.cache/huggingface`
- If the box also drives a display, lower `--mem-fraction-static` to ~0.85 so the desktop keeps VRAM headroom

## Repository layout

```
.
├── start.sh      # launch SGLang container (DFlash 2), wait for readiness
├── stop.sh       # stop the container, clean up pid file
├── patch/sglang  # DFlash 2 backport overlay for the cookbook image (git-tracked, mount read-only)
├── .gitignore    # excludes .cache/, logs, pid file
└── README.md
```

## Credits

- [SGLang cookbook — Qwen3.8-27B](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B) — serving recipe, DSpark/MTP guidance, GDN state-pool calculator
- [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) — trained DFlash 2 drafter (~3.8 GB, BF16, block-8 serving recipe)
- [Inco AI — DFlash 2 blog](https://inco.ai/blog/dflash2/) — drafter design, head-to-head numbers vs MTP/DSpark
- [Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B) — 256K native context, sampling defaults, thinking-mode behavior
- [SGLang](https://github.com/sgl-project/sglang) — inference engine and OpenAI/Anthropic-compatible server; DFlash 2 upstream PRs #35371 / #35496
- [MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark](https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark) — the 128 GB sibling recipe (DSpark path; this repo replaces DSpark with DFlash 2)
- [MiaAI-Lab/Qwen3.8-27B-SGLang-RTX-5090](https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-RTX-5090) — sibling repo where this DFlash 2 adaptation was first validated
