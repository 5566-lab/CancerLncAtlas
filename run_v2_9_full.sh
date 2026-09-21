#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-config/model_v2_9_state_graph.yaml}"
DIRECT_EVENT_GLOBS="${CC_HHGT_DIRECT_EVENT_GLOBS:-}"
"$PYTHON_BIN" scripts/50_run_v29_full_pipeline.py \
  --config "$CONFIG" \
  --python "$PYTHON_BIN" \
  --direct-event-globs "$DIRECT_EVENT_GLOBS" \
  "$@"
