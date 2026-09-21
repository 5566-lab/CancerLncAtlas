#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-config/model_v3_0_clinical.yaml}"
JOBS="${JOBS:-1}"
exec "$PYTHON_BIN" scripts/69_run_v30_full_pipeline.py --config "$CONFIG" --python "$PYTHON_BIN" --jobs "$JOBS" --resume
