#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_CMD="${PYTHON_CMD:-python3}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"

"$PYTHON_CMD" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit(f'Python >=3.10 is required; observed {sys.version}')
PY
"$PYTHON_CMD" -m venv .venv
PYTHON_BIN="$PWD/.venv/bin/python"
"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel
"$PYTHON_BIN" -m pip install -r requirements/requirements-base.txt
"$PYTHON_BIN" -m pip install torch==2.12.0 --index-url "$TORCH_INDEX_URL"
"$PYTHON_BIN" -m pip install torch-geometric==2.8.0
"$PYTHON_BIN" -m pip install -e . --no-deps
"$PYTHON_BIN" scripts/00_verify_gpu_bundle.py --require-gpu

echo "GPU environment setup and verification completed."
