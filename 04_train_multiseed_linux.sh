#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
"$PWD/.venv/bin/python" scripts/25_train_multiseed_gpu.py \
  --config config/model_v2_1_gpu_4070tis.yaml \
  --fail-fast -v 2>&1 | tee -a logs/gpu_multiseed_training.log
