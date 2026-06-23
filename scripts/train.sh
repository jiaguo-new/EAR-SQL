#!/usr/bin/env bash
# EAR-SQL training entrypoint.
# Usage: bash scripts/train.sh configs/ear_sql.yaml
set -euo pipefail
CFG="${1:-configs/ear_sql.yaml}"

export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)"
echo "[EAR-SQL] config = ${CFG}"
echo "[EAR-SQL] PYTHONPATH = ${PYTHONPATH}"

# Recommended: launch rollout server (vLLM) here before training, e.g.
#   python -m vllm.entrypoints.openai.api_server --model "$MODEL" --port 8000 &

python -m src.grpo_trainer "${CFG}"
