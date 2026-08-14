#!/usr/bin/env bash
set -euo pipefail

# Qwen3.8-27B on SGLang (RTX 5090, 32GB VRAM, SM120)
#
# Recipe from the SGLang cookbook, RTX 5090 cell (verified recipe):
#   https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B
#
# Notes:
#   - Dense hybrid GDN vision-language model (48 linear-attention + 16
#     full-attention layers); SGLang serves it through the Qwen3-VL path,
#     so the vision tower is live.
#   - RTX 5090 cell specifics vs DGX Spark: --mem-fraction-static 0.85
#     (not 0.95), --chunked-prefill-size 2048 (2048 keeps decode
#     inter-token latency smooth on hybrid GDN models; 8192 is the
#     DGX-Spark-only exception), and NO --disable-prefill-cuda-graph.
#   - --attention-backend flashinfer is required on SM120
#     (trtllm_mha is SM100-only).
#   - NVFP4-only (cookbook: "NVFP4 weights ~16.5GB (recommended for
#     RTX 5090-class GPUs)"; FP8 ~28.5GB is "not serviceable beyond
#     bs<=2" on 32GB cards, bf16 ~54GB does not fit at all).
#   - KV cache is explicitly FP8 (--kv-cache-dtype fp8_e4m3); the NVFP4
#     checkpoint declares kv_cache_quant_algo: FP8, so its calibration
#     scales are applied automatically. ~32.8 KB/token.
#
# Mamba/GDN state pool calculator (cookbook formula, done for 32GB):
#   state slot  = 78.4 MB (48 GDN layers x 48 heads x 128 x 128 at bf16
#                 + bf16 conv state)
#   slots/req   = S + D = 4 (extra_buffer_lazy) + 4 (MTP draft) = 8
#   state pool  = --max-mamba-cache-size 16 = 2 concurrent x 8 slots
#               = 16 x 78.4 MB = ~1.25 GB
#   budget      = 32 GB x 0.85 (mem-fraction-static) = 27.2 GB
#   weights     = ~17 GB (cookbook LM figure) to ~22 GB (full repo incl.
#                 vision tower + MTP head)
#   KV pool     = 27.2 - weights - 1.25 = ~4.0-9.5 GB
#               = ~120K-290K tokens at 32.8 KB/token (fp8)
#   -> context  : --context-length 100000 (a max-length request costs
#                 ~3.3 GB of KV and fits even the worst-case pool;
#                 2 x ~50-60K sessions run concurrently)
#   Reference: the balanced --mamba-full-memory-ratio at L=32K would be
#   (8 x 78.4 MB) / (32768 x 32.8 KB) ~= 0.58; we pin the pool with
#   --max-mamba-cache-size instead, which overrides the ratio.
#   On 32GB the state pool bounds concurrency long before KV does
#   (cookbook tip), hence lazy strategy (S=4) + a small pool.
#   After boot, check in .sglang.log: max_running_requests >= 2 and the
#   KV-cache token count >= ~120K.
#
# Speed: MTP speculative decoding is ON (in-checkpoint MTP head):
#   --speculative-algorithm EAGLE --speculative-num-steps 3 \
#   --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
#   MTP with FlashInfer needs a FlashInfer build newer than 0.6.15.post1
#   (prefill plan with uniform_q_len). The lmsysorg/sglang:qwen38-27b
#   image is built for these recipes; if spec decode errors at boot,
#   rerun with --attention-backend triton.
#
# Thinking & tool calling (model card + cookbook section 3):
#   - Thinking mode is ON by default: the chat template defaults
#     enable_thinking=true and preserve_thinking=true. --reasoning-parser
#     qwen3 surfaces <think> as reasoning_content. Disable per request:
#     chat_template_kwargs {"enable_thinking": false} (then prefer
#     temperature=0.7, top_p=0.8, presence_penalty=1.5). Reasoning depth:
#     reasoning_effort=xhigh|medium|low (xhigh default).
#   - --sampling-defaults model: recommended params from the checkpoint's
#     generation_config.json (thinking: temperature=1.0, top_p=0.95,
#     top_k=20, min_p=0.0, presence_penalty=0.0).
#   - Tool calling: --tool-call-parser qwen3_coder decodes the template's
#     <tool_call><function=...>/<parameter=...> payload; SGLang needs no
#     vLLM-style --enable-auto-tool-choice.
#
# If the 5090 also drives a display, lower --mem-fraction-static to
# ~0.80 so the desktop keeps VRAM headroom.

MODEL_ID="RadixArk/Qwen3.8-27B-NVFP4"
SERVED_MODEL_NAME="qwen3.8-27b-5090"
IMAGE="lmsysorg/sglang:qwen38-27b"
CONTAINER_NAME="qwen3.8-27b-sglang-5090"
HOST="0.0.0.0"
PORT="8888"
PID_FILE=".sglang.pid"
LOG_FILE=".sglang.log"
WORK_DIR="$(pwd)"
HF_HOME="${WORK_DIR}/.cache/huggingface"
TRITON_CACHE_DIR="${WORK_DIR}/.cache/triton"
READY_URL="http://127.0.0.1:${PORT}/v1/models"

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

echo "Starting SGLang container for ${MODEL_ID} (RTX 5090, 32GB)"
echo "Image: ${IMAGE}"
echo "Served model name: ${SERVED_MODEL_NAME}"
echo "Listening on ${HOST}:${PORT}"
echo "Writing progress to ${LOG_FILE}"

cat >"${LOG_FILE}" <<EOF
[$(date -Is)] launching SGLang container (RTX 5090)
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
  --mem-fraction-static 0.85 \
  --attention-backend flashinfer \
  --chunked-prefill-size 2048 \
  --kv-cache-dtype fp8_e4m3 \
  --context-length 100000 \
  --mamba-radix-cache-strategy extra_buffer_lazy \
  --max-mamba-cache-size 16 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
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
echo "Thinking: ON by default (disable per request: chat_template_kwargs {\"enable_thinking\": false})"

echo "SGLang is ready and responding; shell is now free."
