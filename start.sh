#!/usr/bin/env bash
set -euo pipefail

# Qwen3.8-27B on SGLang (RTX PRO 6000, 96GB VRAM, SM120/Blackwell)
#
# This is the repo's single start.sh and it runs DFlash 2 speculative
# decoding (the previous DSpark variant is replaced). The base model /
# context / KV / thinking / tool-calling setup is identical to the cookbook
# recipe; only the spec-decode path changed.
# Recipe from the SGLang cookbook (RTX PRO 6000, nvfp4, balanced):
#   https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B
# and DFlash 2 serving recipe from the drafter card / blog:
#   https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2
#   https://inco.ai/blog/dflash2/
#
# What this config is:
#   - Full 256K context window (native 262,144 -- no YaRN needed; that is
#     max_position_embeddings on the Qwen3.8-27B config).
#   - DFlash 2 speculative decoding (a ~3.8GB trained BF16 block-diffusion
#     drafter: z-lab/Qwen3.8-27B-DFlash2). DFlash 2, NOT the in-checkpoint
#     MTP head and NOT DSpark: measured head-to-head on the drafter card,
#     DFlash 2 beats both (e.g. GSM8K 3.43x vs 2.59x MTP / 2.69x DSpark
#     at concurrency 1, 2.84x vs 2.19x/2.23x at concurrency 8). DFlash 2
#     drafts a whole block of 8 tokens per step (config block_size=8).
#   - DFlash 2 is a pure-attention drafter (5 sliding-window layers), so
#     NO GDN/mamba state pool is needed for the drafter -- the DSPARK-only
#     --mamba-* flags and --linear-attn-verify-backend are gone. (The TARGET
#     model is still a hybrid-GDN model and keeps its own small state pool,
#     auto-sized by SGLang.)
#   - 8 concurrent requests mapped straight to --max-running-requests
#     (no drafter mamba multiplier; KV cache bounds concurrency now).
#   - FlashInfer attention for target and draft (SM120 requires flashinfer;
#     trtllm_mha is SM100-only).
#
# IMPORTANT: DFlash 2 landed in upstream SGLang on 2026-08-19 (PR #35371);
# the cookbook-pinned image lmsysorg/sglang:qwen38-27b predates it and only
# carries the DFlash 1 model class. This script therefore mounts a small
# backport overlay (./patch/sglang, read-only) that adds upstream's
# DFlash2DraftModel (two-tap conv + candidate selector) and the supporting
# helpers. See ./patch/sglang/README.md. This script (start.sh) refuses to
# launch if the overlay is missing.
#
# Hardware notes:
#   - --attention-backend flashinfer required on SM120.
#   - NVFP4 W4A4 checkpoint (~16.5GB LM / ~21.9GB repo incl. vision + MTP
#     head); FP8 (~28.5GB) fits fine on 96GB too but is slower + heavier.
#   - KV cache is FP8 (--kv-cache-dtype fp8_e4m3). SGLang's 4-bit KV paths
#     are not built for this hybrid-GDN model (flashinfer rejects KV4 for
#     it, and the triton fp4 decode kernel does not exist in this image),
#     so FP8 is the best supported precision here: 32.8 KB/token.
#
# Memory budget (done for 96GB, DFlash 2 -- no drafter state pool):
#   budget      = 96 GB x 0.90 (mem-fraction-static) = ~86 GB
#   weights     = ~25.7 GB (21.9 NVFP4 base + 3.8 DFlash2 BF16 draft)
#   graphs/runtime ~3-4 GB (flashinfer workspace, CUDA graphs, mm pools,
#                 drafter window cache, ..)
#   KV pool     = 86 - 25.7 - 3.5 = ~57 GB -> ~1.7M tokens at 32.8KB
#   -> context  : --context-length 262144 (full native 256K window). The
#                 ~1.7M-token pool means 8 concurrent requests can each be
#                 ~210K, or share prefixes via radix cache; a single
#                 request can go the entire 262144 by itself.
#   After boot, check .sglang.log: max_running_requests = 8, and the KV
#   Cache #tokens (expect ~1.5-1.7M).
#
# Verify the DFLASH path after boot (grep .sglang.log):
#   "Initialized DFLASH draft runner. attention_backend=flashinfer,
#    model=DFlash2DraftModel, block_size=8, ..." -> draft loaded + spec ON.
#   "DFLASH draft greedy head folded into the draft cuda graph" -> graph captured.
#   "DFLASH fused KV materialization enabled. n_layers=5, ..." -> ctx KV fused.
#   max_total_num_tokens ~= 1500000+ -> the KV budget.
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
DRAFT_MODEL_ID="z-lab/Qwen3.8-27B-DFlash2"
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
PATCH_DIR="${WORK_DIR}/patch/sglang"
READY_URL="http://127.0.0.1:${PORT}/v1/models"

# Concurrency knob: max concurrent requests maps straight to
# --max-running-requests (DFlash2 has no drafter state pool, so there is
# no mamba-multiplier like the DSpark recipe had).
MAX_CONCURRENT_REQUESTS="${MAX_CONCURRENT_REQUESTS:-8}"

command -v docker >/dev/null 2>&1 || {
  echo "docker is not on PATH"
  exit 1
}

command -v curl >/dev/null 2>&1 || {
  echo "curl is not on PATH"
  exit 1
}

mkdir -p "${HF_HOME}" "${TRITON_CACHE_DIR}"

# The DFlash2 backport overlay must be present (see patch/sglang/README.md).
# It is bind-mounted read-only into the container to add DFlash2DraftModel
# support to this image. Fail fast with a hint instead of letting the
# container die on "DFlash2DraftModel ... not a registered model".
REQUIRED_PATCH_FILES=(
  "srt/models/dflash.py"
  "kernels/ops/speculative/dflash.py"
  "srt/speculative/dflash_utils.py"
  "srt/speculative/dflash_worker_v2.py"
  "srt/speculative/dflash_info.py"
  "srt/speculative/dflash_info_v2.py"
  "srt/speculative/draft_worker_common.py"
  "srt/speculative/spec_utils.py"
  "srt/mem_cache/allocation_sizing.py"
  "srt/layers/moe/utils.py"
  "srt/layers/logprob_processor.py"
)
for f in "${REQUIRED_PATCH_FILES[@]}"; do
  if [[ ! -s "${PATCH_DIR}/${f}" ]]; then
    echo "error: missing DFlash2 patch file ${PATCH_DIR}/${f}"
    echo "       re-clone the repo (the files are committed) or restore the patch/ tree"
    exit 1
  fi
done

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
echo "Draft model: ${DRAFT_MODEL_ID} (DFlash 2, BF16, block 8, auto-downloads into HF cache)"
echo "DFlash2 backport overlay: ${PATCH_DIR} (mounted read-only)"
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
  -v "${PATCH_DIR}/srt/models/dflash.py:/sgl-workspace/sglang/python/sglang/srt/models/dflash.py:ro" \
  -v "${PATCH_DIR}/kernels/ops/speculative/dflash.py:/sgl-workspace/sglang/python/sglang/kernels/ops/speculative/dflash.py:ro" \
  -v "${PATCH_DIR}/srt/speculative/dflash_utils.py:/sgl-workspace/sglang/python/sglang/srt/speculative/dflash_utils.py:ro" \
  -v "${PATCH_DIR}/srt/speculative/dflash_worker_v2.py:/sgl-workspace/sglang/python/sglang/srt/speculative/dflash_worker_v2.py:ro" \
  -v "${PATCH_DIR}/srt/speculative/dflash_info.py:/sgl-workspace/sglang/python/sglang/srt/speculative/dflash_info.py:ro" \
  -v "${PATCH_DIR}/srt/speculative/dflash_info_v2.py:/sgl-workspace/sglang/python/sglang/srt/speculative/dflash_info_v2.py:ro" \
  -v "${PATCH_DIR}/srt/speculative/draft_worker_common.py:/sgl-workspace/sglang/python/sglang/srt/speculative/draft_worker_common.py:ro" \
  -v "${PATCH_DIR}/srt/speculative/spec_utils.py:/sgl-workspace/sglang/python/sglang/srt/speculative/spec_utils.py:ro" \
  -v "${PATCH_DIR}/srt/mem_cache/allocation_sizing.py:/sgl-workspace/sglang/python/sglang/srt/mem_cache/allocation_sizing.py:ro" \
  -v "${PATCH_DIR}/srt/layers/moe/utils.py:/sgl-workspace/sglang/python/sglang/srt/layers/moe/utils.py:ro" \
  -v "${PATCH_DIR}/srt/layers/logprob_processor.py:/sgl-workspace/sglang/python/sglang/srt/layers/logprob_processor.py:ro" \
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
  --context-length 262144 \
  --max-running-requests "${MAX_CONCURRENT_REQUESTS}" \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path "${DRAFT_MODEL_ID}" \
  --speculative-num-draft-tokens 8 \
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
echo "Context 256K, DFLASH spec-decode ON, ${MAX_CONCURRENT_REQUESTS} concurrent"
echo "Thinking: ON by default (disable per request: chat_template_kwargs {\"enable_thinking\": false})"

echo "SGLang is ready and responding; shell is now free."
