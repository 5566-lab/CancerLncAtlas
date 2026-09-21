#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${1:-cc_hhgt_v29}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"

if command -v mamba >/dev/null 2>&1; then SOLVER=mamba; else SOLVER=conda; fi
"$SOLVER" create -y -n "$ENV_NAME" "python=$PYTHON_VERSION" pip
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$ROOT/requirements/requirements-base.txt"
python -m pip install torch==2.12.0 --index-url "$TORCH_INDEX_URL"
python -m pip install torch-geometric==2.8.0
python -m pip install -e "$ROOT" --no-deps
python "$ROOT/scripts/00_verify_gpu_bundle.py" --require-gpu
python - <<'PY'
import torch
import torch_geometric
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
print('torch_geometric', torch_geometric.__version__)
if not torch.cuda.is_available():
    raise SystemExit('CUDA is unavailable; refuse formal V2.9 GPU training')
PY
