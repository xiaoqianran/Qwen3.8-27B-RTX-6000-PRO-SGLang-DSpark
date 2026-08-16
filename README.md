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

The serving recipe builds on the **[SGLang cookbook's RTX PRO 6000 cell](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B)** for the model, with the **DSpark** speculative-decoding recipe from the Qwen3.8-27B drafter card, and the GDN state-pool calculator worked out for 96 GB.

- **NVFP4 W4A4** checkpoint — cookbook: "NVFP4 weights ~16.5 GB (recommended);" bf16 (~54 GB) and FP8 (~28.5 GB) also fit on 96 GB but NVFP4 is the fastest/smallest
- **Full 256K-token context** — the model's native 262,144 window, no YaRN needed; the ~1.5M-token fp8 KV pool covers it with room to spare
- **FP8 KV cache** (`fp8_e4m3`, ~32.8 KB/token, checkpoint calibration scales) — SGLang's 4-bit KV paths are not built for this hybrid-GDN model in this image, so FP8 is the best supported precision
- **DSpark speculative decoding** (trained BF16 drafter `RadixArk/Qwen3.8-27B-DSpark`, block 7 = 8 draft tokens/step) — ***not*** the in-checkpoint MTP head: measured head-to-head, DSpark wins on speed/acceptance while MTP's bf16 module wastes ~3 GB and squeezes the KV pool
- **8 concurrent requests** with the GDN state pool sized from the cookbook formula
- **Thinking mode on by default** (`--reasoning-parser qwen3` → `reasoning_content`) and **tool calling** (`qwen3_coder` parser)

---

## Requirements

| Component | Detail |
|---|---|
| Hardware | NVIDIA RTX PRO 6000 (96 GB, SM120/Blackwell; this recipe needs the 96 GB budget) |
| Docker | With NVIDIA Container Toolkit / GPU passthrough working (`docker run --gpus all`) |
| SGLang image | `lmsysorg/sglang:qwen38-27b` (model-specific build from the cookbook; multi-arch incl. amd64) |
| CLI tools | `docker`, `curl` |
| Hugging Face token | `HF_TOKEN` defined in `~/.bashrc` (picked up automatically; higher rate limits) |

There is no separate download step: the container pulls both checkpoints into `./.cache/huggingface` on first start (~22 GB base + ~2.7 GB DSpark drafter).

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
| `start.sh` | Launches the SGLang container (`docker run -d`, host network, `--shm-size 32g`), streams logs to `.sglang.log`, records the container ID in `.sglang.pid`, and polls `http://127.0.0.1:8888/v1/models` until the server is ready. |
| `stop.sh` | Stops the container, removes `.sglang.pid`, and leaves the stopped container in place for `docker logs` post-mortem (the next `start.sh` removes it). |

Runtime artifacts: `.sglang.log` (server log), `.sglang.pid` (container ID), `.cache/` (HF + Triton caches). All are git-ignored.

## The calculator (96 GB, done)

The cookbook's [mamba ratio calculator](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B#mamba-ratio-calculator) for this model, evaluated for the RTX PRO 6000:

| Input | Value | Source |
|---|---|---|
| Static memory budget | 96 GB × `0.90` = **~86 GB** | this recipe |
| Weights (NVFP4 base + DSpark BF16 drafter) | **~24.5 GB** = 21.9 + 2.7 | repo blobs / drafter card |
| State slot size | **78.4 MB** (48 GDN layers × 48 heads × 128 × 128 bf16 + conv) | cookbook |
| Slots per request | S + D = **12** = 4 (`extra_buffer_lazy`) + 8 (DSpark block 7) | cookbook / drafter card |
| Concurrency | **8** → `--max-mamba-cache-size 96` → state pool **7.5 GB** | pinned |
| KV bytes/token (fp8) | **32.8 KB** (16 attn layers × GQA 4 × 256 × K+V) | cookbook |
| KV pool left | **~50 GB** → **~1.5M tokens** | arithmetic |
| Context | **`--context-length 262144`** (full native 256K; no YaRN) | model config |

Reference: the *balanced* `--mamba-full-memory-ratio` at an average request length of 32K tokens would be (12 × 78.4 MB) ÷ (32768 × 32.8 KB) ≈ **0.87**. This script pins the pool with `--max-mamba-cache-size`, which overrides the ratio. The ~1.5M-token pool means 8 concurrent requests can each run ~180K tokens, or share prefixes via the radix cache; a single request can use the full 262,144 at once.

**Verify after boot** (in `.sglang.log`): `max_running_requests` = 8, KV-cache token count ≥ ~1.3M, and the DSpark line "Draft proposal … folded into the draft cuda graph". To change concurrency, `MAX_CONCURRENT_REQUESTS=8 ./start.sh` etc. (mamba pool scales as N × 12).

## Configuration

Defaults live at the top of `start.sh`:

| Variable | Default | Notes |
|---|---|---|
| `MODEL_ID` | `RadixArk/Qwen3.8-27B-NVFP4` | NVFP4 W4A4 (bf16/FP8 also fit on 96 GB — set `MODEL_ID` to `Qwen/Qwen3.8-27B` / `-FP8`, adjust expectations) |
| `DRAFT_MODEL_ID` | `RadixArk/Qwen3.8-27B-DSpark` | Trained BF16 drafter (~2.7 GB, auto-downloads) |
| `SERVED_MODEL_NAME` | `qwen3.8-27b-6000pro` | Name clients use in API requests |
| `IMAGE` | `lmsysorg/sglang:qwen38-27b` | Cookbook-pinned image for this model |
| `CONTAINER_NAME` | `qwen3.8-27b-sglang-6000pro` | Also used by `stop.sh` |
| `PORT` | `8888` | Listens on `0.0.0.0` via host networking |
| `MAX_CONCURRENT_REQUESTS` | `8` | Concurrency; mamba pool = N × 12 slots |

### Notable serving choices

- **DSpark speculative decoding (default, not MTP):** `--speculative-algorithm DSPARK --speculative-draft-model-path RadixArk/Qwen3.8-27B-DSpark --speculative-dspark-block-size 7 --speculative-draft-model-quantization unquant --speculative-draft-attention-backend flashinfer`. Block 7 = 8 draft tokens/step (verify width 8 incl. the bonus token). The draft runs unquantized (BF16) per the drafter card — quantizing it to the target's NVFP4 did not raise acceptance while costing decode speed. **Why not MTP:** measured head-to-head, DSpark drafts faster (higher acceptance, cheaper steps), and the in-checkpoint MTP head's bf16 module eats ~3 GB that on a 32 GB card collapses the KV pool. With a 96 GB budget, DSpark is the strict speed win with no trade-off.
- **`--linear-attn-verify-backend triton`** — the GDN verify path for the draft; part of the validated DSpark recipe.
- **`--min-free-slots-delay 1`** — scheduler admits requests promptly against the pinned mamba pool (keeps latency low at low concurrency).
- **`--context-length 262144`** — the model's **native 256K** window (no YaRN / override env needed).
- **`--kv-cache-dtype fp8_e4m3`** — explicit; the NVFP4 checkpoint declares fp8 KV anyway (`auto` honors its calibration scales). SGLang's 4-bit KV options (`nvfp4`/`fp4_*`) are not wired for hybrid-GDN decode in this image (flashinfer rejects KV4 for it; the triton fp4 kernel doesn't exist), so fp8 is the best supported precision. At 32.8 KB/token the ~50 GB pool yields ~1.5M tokens.
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
- **Spec decode check:** `grep -i "DSpark draft proposal" .sglang.log` should show the draft proposal folded into the draft CUDA graph (draft works). `grep max_total_num_tokens .sglang.log` — expect ≥ ~1.3M
- First start downloads ~22 GB base + ~2.7 GB drafter; subsequent starts reuse `./.cache/huggingface`
- If the box also drives a display, lower `--mem-fraction-static` to ~0.85 so the desktop keeps VRAM headroom

## Repository layout

```
.
├── start.sh      # launch SGLang container, wait for readiness
├── stop.sh       # stop the container, clean up pid file
├── .gitignore    # excludes .cache/, logs, pid file
└── README.md
```

## Credits

- [SGLang cookbook — Qwen3.8-27B](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B) — serving recipe, DSpark/MTP guidance, GDN state-pool calculator
- [RadixArk/Qwen3.8-27B-DSpark](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark) — trained DSpark drafter (~2.7 GB, BF16, block-7 serving recipe)
- [Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B) — 256K native context, sampling defaults, thinking-mode behavior
- [SGLang](https://github.com/sgl-project/sglang) — inference engine and OpenAI/Anthropic-compatible server
- [MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark](https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark) — the 128 GB sibling recipe (same DSpark block-7 path)
