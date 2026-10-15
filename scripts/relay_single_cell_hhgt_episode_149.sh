#!/usr/bin/env bash
set -euo pipefail

ROOT=/dsk2/user/dengsc/sc_metacell_formal_integration_20261005_r1
PYTHON=/dsk2/user/dengsc/miniconda3/envs/cc_hhgt/bin/python
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
BASE=/dell_2/DSC/CancerLncAtlas/rbp_encode_v33_20260921_r1/runtime/c_graph_global_binding_20260928_r1/results
SOURCE="$BASE/single_cell_metacell_v32-single-cell-metacell-donor-oof-20261005-r1"
GRAPH="$BASE/single_cell_hhgt_graph_assets_v32-single-cell-metacell-donor-oof-20261005-r1"
NESTED="$BASE/single_cell_hhgt_nested_assets_v32-single-cell-metacell-donor-oof-20261005-r1"
EPISODES="$BASE/single_cell_hhgt_episodes_v32-single-cell-metacell-donor-oof-20261005-r1"
STATUS="$NESTED/batch_supervisor/STATUS.json"
DEADLINE=$(( $(date +%s) + 345600 ))

[[ "$(hostname)" == 149 ]] || { echo "WRONG_HOST"; exit 2; }

while (( $(date +%s) < DEADLINE )); do
  if [[ -f "$STATUS" ]]; then
    # STATUS.json is atomically replaced by the supervisor; tolerate the brief
    # interval between the existence check and opening the new file.
    state=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["state"])' "$STATUS" 2>/dev/null || true)
    case "$state" in
      COMPLETE)
        successes=$(find "$NESTED/cancers" -mindepth 2 -maxdepth 2 -name SUCCESS.json -type f | wc -l)
        [[ "$successes" -eq 28 ]] || {
          echo "NESTED_COMPLETE_WITHOUT_28_CANCER_SUCCESSES:$successes"
          exit 3
        }
        cd "$ROOT"
        exec /usr/bin/timeout --signal=TERM --kill-after=300s 604800s \
          "$PYTHON" scripts/launch_single_cell_hhgt_episode_batch_149.py \
          --run-id v32-sc-hhgt-episodes-20261005-r1 \
          --source-result-root "$SOURCE" \
          --graph-asset-root "$GRAPH" \
          --nested-asset-root "$NESTED" \
          --output-root "$EPISODES" \
          --workers 2 \
          --threads-per-job 24 \
          --launch-stagger-seconds 180 \
          --per-cancer-timeout 259200 \
          --max-failures 1 \
          --lasso-max-genes 1000 \
          --lasso-bootstraps 8
        ;;
      FAILED|FAILED_FINALIZE)
        echo "NESTED_BATCH_FAILED"
        exit 4
        ;;
    esac
  fi
  printf '%s waiting_for_nested_assets\n' "$(date --iso-8601=seconds)"
  sleep 300
done

echo "EPISODE_RELAY_DEADLINE_EXPIRED"
exit 5
