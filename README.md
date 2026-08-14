# Qwen3.8 27B on SGLang for RTX 5090

[![SGLang](https://img.shields.io/badge/SGLang-cookbook-blue)](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B)
[![Model](https://img.shields.io/badge/model-Qwen3.8--27B-informational)](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4)
[![arch](https://img.shields.io/badge/arch-SM120%20%2F%2032GB-lightgrey)](#)

Opinionated, ready-to-run scripts to serve **[Qwen3.8-27B](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4)** with **[SGLang](https://docs.sglang.io)** in Docker on an NVIDIA **RTX 5090 (32 GB, SM120)**. One script starts an OpenAI-compatible server, one stops it.

The serving recipe is the **[SGLang cookbook's RTX 5090 cell](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B)** — the model-specific, validated launch configuration — with the cookbook's speed options turned on and its GDN state-pool calculator worked out for 32 GB.

- **NVFP4 W4A4** checkpoint — cookbook: "NVFP4 weights ~16.5 GB (recommended for RTX 5090-class GPUs)"; FP8 (~28.5 GB) is "not serviceable beyond bs≤2" on 32 GB cards and bf16 (~54 GB) does not fit at all
- **100K-token context** — calculator-derived so a full-length request always fits the KV pool (1M YaRN is physically impossible here: one 1M request alone needs ~33 GB of KV)
- **FP8 KV cache** (`fp8_e4m3`, ~2× KV memory savings, checkpoint calibration scales)
- **MTP speculative decoding** (EAGLE 3/1/4 via the in-checkpoint MTP head) — faster decode
- **2 concurrent requests** with the GDN state pool sized from the cookbook formula
- **Thinking mode on by default** (`--reasoning-parser qwen3` → `reasoning_content`) and **tool calling** (`qwen3_coder` parser)

---

## Requirements

| Component | Detail |
|---|---|
| Hardware | NVIDIA RTX 5090 (32 GB, SM120/Blackwell) |
| Docker | With NVIDIA Container Toolkit / GPU passthrough working (`docker run --gpus all`) |
| SGLang image | `lmsysorg/sglang:qwen38-27b` (model-specific build from the cookbook; multi-arch incl. amd64) |
| CLI tools | `docker`, `curl` |
| Hugging Face token | `HF_TOKEN` defined in `~/.bashrc` (picked up automatically; higher rate limits) |

There is no separate download step: the container pulls the checkpoint into `./.cache/huggingface` on first start (~22 GB download; the cookbook cites ~16.5 GB for the NVFP4 LM weights alone, before the vision tower + MTP head).

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

## The calculator (32 GB, done)

The cookbook's [mamba ratio calculator](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B#mamba-ratio-calculator) for this model, evaluated for the 5090:

| Input | Value | Source |
|---|---|---|
| Static memory budget | 32 GB × `0.85` = **27.2 GB** | cookbook RTX 5090 cell |
| Weights (NVFP4 + vision + MTP) | **~17–22 GB** | cookbook (LM figure) / repo blobs (full) |
| State slot size | **78.4 MB** (48 GDN layers × 48 heads × 128 × 128 bf16 + conv) | cookbook |
| Slots per request | S + D = **8** = 4 (`extra_buffer_lazy`) + 4 (MTP draft) | cookbook |
| Concurrency | **2** → `--max-mamba-cache-size 16` → state pool **1.25 GB** | pinned |
| KV bytes/token (fp8) | **32.8 KB** (16 attn layers × GQA 4 × 256 × K+V) | cookbook |
| KV pool left | **~4.0–9.5 GB** → **~120K–290K tokens** | arithmetic |
| Context cap | **`--context-length 100000`** — a max request costs ~3.3 GB KV, fits even the worst-case pool | derived |

Reference: the *balanced* `--mamba-full-memory-ratio` at an average request length of 32K tokens would be (8 × 78.4 MB) ÷ (32768 × 32.8 KB) ≈ **0.58**. This script pins the pool with `--max-mamba-cache-size` instead, which overrides the ratio. On 32 GB the state pool bounds concurrency long before KV does (cookbook tip), hence the lazy strategy (S=4) and the small pinned pool.

**Verify after boot** (in `.sglang.log`): `max_running_requests` ≥ 2, and the KV-cache token count ≥ ~120K. To change concurrency, adjust `--max-mamba-cache-size` in multiples of 8 (2 requests per 16 slots).

## Configuration

Defaults live at the top of `start.sh`:

| Variable | Default | Notes |
|---|---|---|
| `MODEL_ID` | `RadixArk/Qwen3.8-27B-NVFP4` | NVFP4 only — bf16 (~54 GB) doesn't fit; FP8 (~28.5 GB) "not serviceable beyond bs≤2" per cookbook |
| `SERVED_MODEL_NAME` | `qwen3.8-27b-5090` | Name clients use in API requests |
| `IMAGE` | `lmsysorg/sglang:qwen38-27b` | Cookbook-pinned image for this model |
| `CONTAINER_NAME` | `qwen3.8-27b-sglang-5090` | Also used by `stop.sh` |
| `PORT` | `8888` | Listens on `0.0.0.0` via host networking |
| Concurrency | 2 requests | `--max-mamba-cache-size 16` (see calculator) |

### Notable serving choices

- **Recipe (cookbook, RTX 5090 cell):** `--mem-fraction-static 0.85`, `--attention-backend flashinfer` (`trtllm_mha` is SM100-only), `--chunked-prefill-size 2048`. The 2048-token chunks are deliberate: on hybrid GDN models, 8192-token chunks stall decode ~600 ms at a time; 2048 keeps inter-token latency smooth (8192 is the DGX Spark-only exception). No `--disable-prefill-cuda-graph` here — that flag is specific to the Spark cell.
- **If the 5090 drives a display**, lower `--mem-fraction-static` to ~0.80 so the desktop keeps VRAM headroom.
- **MTP speculative decoding:** `--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4` uses the checkpoint's own MTP head — the biggest decode-time win. MTP with FlashInfer needs a build newer than 0.6.15.post1 (the cookbook image is built for these recipes); if spec decode errors at boot, rerun with `--attention-backend triton`.
- **Context:** `--context-length 100000` — native is 262,144 but the 32 GB KV pool (worst case ~120K tokens) can't service it; 100K guarantees a full-length request always schedules and two ~50–60K sessions run concurrently. The 1M YaRN extension from the model card is not applied: one 1M-token request needs ~33 GB of KV alone.
- **KV cache:** explicit `--kv-cache-dtype fp8_e4m3`; the NVFP4 checkpoint declares FP8 KV anyway (`auto` would honor it with its calibration scales).
- **Vision:** the model is a native VLM and SGLang serves the vision tower live (image + video input supported out of the box).

## Thinking & tool calling

- **Thinking mode is ON by default** — the chat template defaults `enable_thinking=true` and `preserve_thinking=true` (the full reasoning trace is retained across turns; good for agents and KV reuse). `--reasoning-parser qwen3` surfaces `<think>…</think>` as `reasoning_content` instead of inline text. Depth is tunable per request with `reasoning_effort=xhigh|medium|low` (xhigh default).
- **Sampling defaults** come from the checkpoint's `generation_config.json` (`--sampling-defaults model`): thinking mode wants `temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0`.
- **Tool calling** needs no extra SGLang flag (unlike vLLM's `--enable-auto-tool-choice`): `--tool-call-parser qwen3_coder` decodes the template's `<tool_call><function=…>/<parameter=…>` payload into structured `tool_calls`. Just send `tools` in the request. (The hermes parser expects a different payload and would never parse.)

## Using the API

OpenAI-compatible base URL: `http://127.0.0.1:8888/v1` (model name: `qwen3.8-27b-5090`).

```bash
curl http://127.0.0.1:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b-5090",
    "messages": [{"role": "user", "content": "Explain YaRN in two sentences."}]
  }'
```

Non-thinking / instruct request (per the model card):

```json
{
  "model": "qwen3.8-27b-5090",
  "messages": [{"role": "user", "content": "Write a haiku about GB202."}],
  "temperature": 0.7,
  "top_p": 0.8,
  "top_k": 20,
  "presence_penalty": 1.5,
  "chat_template_kwargs": { "enable_thinking": false }
}
```

SGLang also serves an **Anthropic-compatible** endpoint at `http://127.0.0.1:8888/v1/messages` — for Claude Code, set `ANTHROPIC_BASE_URL=http://127.0.0.1:8888` (no `/v1` suffix; Claude Code appends it). The same parser flags apply there. Coding agents that speak plain OpenAI (OpenCode, Pi, …) point at `/v1` and use the served model name.

## Logs & troubleshooting

- Tail the server log: `tail -f .sglang.log` (or `docker logs -f qwen3.8-27b-sglang-5090`)
- `start.sh` prints the last 200 log lines and exits if the container dies before becoming ready
- Concurrency check: `grep max_running_requests .sglang.log` — should be ≥ 2
- If speculative decoding fails to start, switch `--attention-backend flashinfer` → `triton` (FlashInfer version caveat above)
- If startup OOMs on a card that also runs a display, lower `--mem-fraction-static` to 0.80
- First start downloads ~22 GB of weights; subsequent starts reuse `./.cache/huggingface`

## Repository layout

```
.
├── start.sh      # launch SGLang container, wait for readiness
├── stop.sh       # stop the container, clean up pid file
├── .gitignore    # excludes .cache/, logs, pid file
└── README.md
```

## Credits

- [SGLang cookbook — Qwen3.8-27B](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B) — the RTX 5090 serving recipe, MTP guidance, and the GDN state-pool calculator
- [Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B) — sampling recommendations and thinking-mode behavior
- [RadixArk/Qwen3.8-27B-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4) — NVFP4 W4A4 checkpoint (FP8 KV calibration scales)
- [SGLang](https://github.com/sgl-project/sglang) — inference engine and OpenAI/Anthropic-compatible server
