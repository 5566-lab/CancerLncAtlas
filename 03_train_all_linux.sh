#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
"$PWD/.venv/bin/python" scripts/22_train_gpu.py \
  --config config/model_v2_1_gpu_4070tis.yaml -v 2>&1 | tee -a logs/gpu_full_training.log
