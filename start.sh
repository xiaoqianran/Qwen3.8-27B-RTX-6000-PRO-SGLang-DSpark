#!/usr/bin/env bash
set -euo pipefail

# Qwen3.8-27B on SGLang (RTX PRO 6000, 96GB VRAM, SM120/Blackwell)
#
# Recipe from the SGLang cookbook (RTX PRO 6000, nvfp4, balanced):
#   https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B
# and the DSpark serving recipe from the Qwen3.8-27B DSpark drafter card.
#
# What this config is:
#   - Full 256K context window (native 262,144 -- no YaRN needed; that is
#     max_position_embeddings on the Qwen3.8-27B config).
#   - DSpark speculative decoding (a separate ~2.7GB trained BF16 drafter:
#     RadixArk/Qwen3.8-27B-DSpark). DSpark, NOT the in-checkpoint MTP head:
#     measured head-to-head, DSpark drafts faster / higher acceptance while
#     the MTP module (bf16, in-checkpoint) wastes ~3GB and its KV-pool
#     squeeze makes it slower end-to-end on the 32GB card; on 96GB the
#     choice is purely speed -> DSpark. DSpark = 8 draft tokens per step
#     (block size 7 = verify width 8 incl. the bonus token).
#   - 8 concurrent requests with the GDN state pool pinned from the
#     cookbook calculator: S=4 (extra_buffer_lazy) + D=8 = 12 slots/req
#     -> --max-mamba-cache-size 96 (see calculator below).
#   - FlashInfer draft attention + the Triton linear-attention verify path
#     (--linear-attn-verify-backend triton) per the DSpark recipe.
#
# Hardware notes:
#   - --attention-backend flashinfer required on SM120 (trtllm_mha SM100-only).
#   - NVFP4 W4A4 checkpoint (~16.5GB LM / ~21.9GB repo incl. vision + MTP
#     head); FP8 (~28.5GB) fits fine on 96GB too but is slower + heavier.
#   - KV cache is FP8 (--kv-cache-dtype fp8_e4m3). SGLang's 4-bit KV paths
#     are not built for this hybrid-GDN model (flashinfer rejects KV4 for
#     it, and the triton fp4 decode kernel does not exist in this image),
#     so FP8 is the best supported precision here: 32.8 KB/token.
#
# GDN state pool calculator (done for 96GB):
#   state slot  = 78.4 MB (48 GDN layers x 48 heads x 128 x 128 bf16
#                 + bf16 conv state), pinned with --mamba-ssm-dtype bfloat16.
#   slots/req   = S + D = 4 (extra_buffer_lazy) + 8 (DSpark block 7) = 12
#   state pool  = 8 concurrent x 12 slots = --max-mamba-cache-size 96
#               = 96 x 78.4 MB = ~7.5 GB
#   budget      = 96 GB x 0.90 (mem-fraction-static) = ~86 GB
#   weights     = ~24.5 GB (21.9 base + 2.7 DSpark BF16 draft)
#   graphs/runtime ~3-4 GB (flashinfer workspace, CUDA graphs, mm pools)
#   KV pool     = 86 - 24.5 - 7.5 - 3.5 = ~50 GB -> ~1.5M tokens at 32.8KB
#   -> context  : --context-length 262144 (full native 256K window). The
#                 ~1.5M-token pool means 8 concurrent requests can each be
#                 ~180K, or share prefixes via radix cache; a single
#                 request can go the entire 262144 by itself.
#   The balanced --mamba-full-memory-ratio at L=32K would be
#   (12 x 78.4 MB) / (32768 x 32.8 KB) ~= 0.87; the pinned
#   --max-mamba-cache-size overrides it. On 96GB, KV no longer bounds
#   concurrency before the state pool does -> pin the pool explicitly.
#   After boot, check .sglang.log: max_running_requests = 8, and the KV
#   Cache #tokens (expect ~1.3-1.5M).
#
# Verify the DSpark path after boot (grep .sglang.log):
#   "DSpark draft proposal (greedy + sampling) folded into the draft cuda
#    graph." -> draft decode/verify graphs captured -> spec decode ON.
#   max_total_num_tokens ~= 1300000+ -> the KV budget.
#
# Thinking & tool calling (model card + cookbook section 3):
#   - Thinking mode is ON by default: the chat template defaults
#     enable_thinking=true and preserve_thinking=true. --reasoning-parser
#     qwen3 surfaces thinking as reasoning_content. Disable per request:
#     chat_template_kwargs {"enable_thinking": false} (then prefer
#     temperature=0.7, top_p=0.8, presence_penalty=1.5).
#   - --sampling-defaults model: recommended params from the checkpoint's
#     generation_config.json (thinking: temperature=1.0, top_p=0.95,
#     top_k=20, min_p=0.0, presence_penalty=0.0).
#   - Tool calling: --tool-call-parser qwen3_coder decodes the template's
#     <tool_call><function=...>/<parameter=...> payload; SGLang needs no
#     vLLM-style --enable-auto-tool-choice.

MODEL_ID="RadixArk/Qwen3.8-27B-NVFP4"
DRAFT_MODEL_ID="RadixArk/Qwen3.8-27B-DSpark"
SERVED_MODEL_NAME="qwen3.8-27b-6000pro"
IMAGE="lmsysorg/sglang:qwen38-27b"
CONTAINER_NAME="qwen3.8-27b-sglang-6000pro"
HOST="0.0.0.0"
PORT="8888"
PID_FILE=".sglang.pid"
LOG_FILE=".sglang.log"
WORK_DIR="$(pwd)"
HF_HOME="${WORK_DIR}/.cache/huggingface"
TRITON_CACHE_DIR="${WORK_DIR}/.cache/triton"
READY_URL="http://127.0.0.1:${PORT}/v1/models"

# Concurrency knob: max concurrent requests. Each request holds 12 GDN
# state slots (S=4 lazy + D=8 DSpark drafts) -> mamba cache = N x 12.
MAX_CONCURRENT_REQUESTS="${MAX_CONCURRENT_REQUESTS:-8}"
MAMBA_SLOTS_PER_REQ=12
if (( MAX_CONCURRENT_REQUESTS > 8 )); then
  echo "warning: MAX_CONCURRENT_REQUESTS=${MAX_CONCURRENT_REQUESTS} -> mamba pool ${MAMBA_SLOTS_PER_REQ}xN slots"
fi
MAMBA_CACHE_SIZE=$(( MAX_CONCURRENT_REQUESTS * MAMBA_SLOTS_PER_REQ ))

command -v docker >/dev/null 2>&1 || {
  echo "docker is not on PATH"
  exit 1
}

command -v curl >/dev/null 2>&1 || {
  echo "curl is not on PATH"
  exit 1
}

mkdir -p "${HF_HOME}" "${TRITON_CACHE_DIR}"

# Pick up HF_TOKEN from ~/.bashrc (defined without `export` there) so the
# container gets authenticated Hub access (higher rate limits, faster downloads).
if [[ -z "${HF_TOKEN:-}" && -f "${HOME}/.bashrc" ]]; then
  HF_TOKEN="$(sed -n 's/^HF_TOKEN=["'"'"']\?\([A-Za-z0-9_-]\+\).*/\1/p' "${HOME}/.bashrc" | head -1)"
fi
export HF_TOKEN

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Container ${CONTAINER_NAME} is already running"
    echo "Log: ${LOG_FILE}"
    exit 0
  fi
  docker rm "${CONTAINER_NAME}" >/dev/null
fi

echo "Starting SGLang container for ${MODEL_ID} (RTX PRO 6000, 96GB)"
echo "Image: ${IMAGE}"
echo "Draft model: ${DRAFT_MODEL_ID} (DSpark, BF16, auto-downloads into HF cache)"
echo "Served model name: ${SERVED_MODEL_NAME}"
echo "Context: 262144 (native 256K), concurrency: ${MAX_CONCURRENT_REQUESTS}"
echo "Listening on ${HOST}:${PORT}"
echo "Writing progress to ${LOG_FILE}"

cat >"${LOG_FILE}" <<EOF
[$(date -Is)] launching SGLang container (RTX PRO 6000)
EOF

docker run -d \
  --name "${CONTAINER_NAME}" \
  --network host \
  --ipc host \
  --privileged \
  --gpus all \
  --shm-size 32g \
  -e HF_HOME=/root/.cache/huggingface \
  -e TRITON_CACHE_DIR=/root/.triton \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -v "${HF_HOME}:/root/.cache/huggingface" \
  -v "${TRITON_CACHE_DIR}:/root/.triton" \
  "${IMAGE}" \
  python3 -m sglang.launch_server \
  --model-path "${MODEL_ID}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  --mem-fraction-static 0.90 \
  --attention-backend flashinfer \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 4096 \
  --kv-cache-dtype fp8_e4m3 \
  --mamba-ssm-dtype bfloat16 \
  --mamba-full-memory-ratio 4.21 \
  --context-length 262144 \
  --mamba-radix-cache-strategy extra_buffer_lazy \
  --max-mamba-cache-size "${MAMBA_CACHE_SIZE}" \
  --max-running-requests "${MAX_CONCURRENT_REQUESTS}" \
  --linear-attn-verify-backend triton \
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path "${DRAFT_MODEL_ID}" \
  --speculative-dspark-block-size 7 \
  --speculative-draft-model-quantization unquant \
  --speculative-draft-attention-backend flashinfer \
  --min-free-slots-delay 1 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --sampling-defaults model \
  --host "${HOST}" \
  --port "${PORT}" \
  >/dev/null

container_id="$(docker inspect -f '{{.Id}}' "${CONTAINER_NAME}")"
echo "${container_id}" > "${PID_FILE}"
echo "Spawned container ${CONTAINER_NAME} (${container_id})"

log_follow_pid=""
cleanup() {
  if [[ -n "${log_follow_pid}" ]] && kill -0 "${log_follow_pid}" 2>/dev/null; then
    kill "${log_follow_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# Stream the SGLang startup log to the terminal AND record it in .sglang.log.
# $! tracks tee (pipeline tail); killing it on exit SIGPIPEs docker logs.
docker logs -f "${CONTAINER_NAME}" 2>&1 | tee -a "${LOG_FILE}" &
log_follow_pid=$!

echo "Waiting for HTTP readiness at ${READY_URL}"
heartbeat=0
until curl -fsS "${READY_URL}" >/dev/null 2>&1; do
  if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "SGLang container exited before becoming ready"
    tail -n 200 "${LOG_FILE}" || true
    exit 1
  fi
  # The log itself is streaming above; only a light heartbeat every ~30s.
  if (( heartbeat % 6 == 0 )); then
    echo "  still starting..."
  fi
  heartbeat=$((heartbeat + 1))
  sleep 5
done

echo "SGLang is ready"
echo "OpenAI base URL: http://${HOST}:${PORT}/v1"
echo "Anthropic-compatible: http://${HOST}:${PORT}/v1/messages (no /v1 suffix in ANTHROPIC_BASE_URL)"
echo "Served model name: ${SERVED_MODEL_NAME}"
echo "Context 256K, DSpark spec-decode ON, ${MAX_CONCURRENT_REQUESTS} concurrent"
echo "Thinking: ON by default (disable per request: chat_template_kwargs {\"enable_thinking\": false})"

echo "SGLang is ready and responding; shell is now free."
