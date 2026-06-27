#!/usr/bin/env bash
# Restart the local rollout vLLM server with new weights (for on-policy sync).
# Usage: restart_rollout_vllm.sh <weights_dir> <served_name> <util> [port]
# Blocks until the server answers /v1/models, then exits 0.
set -uo pipefail
WEIGHTS="$1"; SERVED="$2"; UTIL="${3:-0.15}"; PORT="${4:-8000}"
PYBIN="${EAR_VLLM_PYBIN:-/home/dameng/miniconda3/envs/vllm-cuda/bin/python}"
LOG="${EAR_VLLM_LOG:-/tmp/rollout_vllm.log}"

# Only kill the vLLM server + its engine child. Do NOT kill arbitrary GPU procs
# here: during on-policy sync the trainer (actor) is also a GPU process and must
# survive.
pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
pkill -9 -f "EngineCore" 2>/dev/null || true
for i in $(seq 1 30); do ss -ltn 2>/dev/null | grep -q ":$PORT" || break; sleep 2; done
sleep 3

setsid "$PYBIN" -m vllm.entrypoints.openai.api_server \
  --model "$WEIGHTS" --served-model-name "$SERVED" --dtype bfloat16 --max-model-len 8192 \
  --gpu-memory-utilization "$UTIL" --enable-prefix-caching --port "$PORT" \
  > "$LOG" 2>&1 < /dev/null &

for i in $(seq 1 80); do
  curl -s "http://localhost:$PORT/v1/models" 2>/dev/null | grep -q "$(basename "$SERVED")" && exit 0
  sleep 5
done
echo "restart_rollout_vllm: server not ready" >&2
exit 1
