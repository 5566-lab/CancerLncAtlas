#!/usr/bin/env bash
set -euo pipefail

ROOT=/dsk2/user/dengsc/sc_metacell_formal_integration_20261005_r1
PYTHON=/dsk2/user/dengsc/miniconda3/envs/cc_hhgt/bin/python
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
BASE=/dell_2/DSC/CancerLncAtlas/rbp_encode_v33_20260921_r1/runtime/c_graph_global_binding_20260928_r1/results
GRAPH="$BASE/single_cell_hhgt_graph_assets_v32-single-cell-metacell-donor-oof-20261005-r1"
EPISODES="$BASE/single_cell_hhgt_episodes_v32-single-cell-metacell-donor-oof-20261005-r1"
TRAINING="$BASE/single_cell_hhgt_v32-single-cell-metacell-donor-oof-20261005-r1"
STATUS="$EPISODES/batch_supervisor/STATUS.json"
MANIFEST="$EPISODES/SC_HHGT_EPISODE_MANIFEST.json"
DEADLINE=$(( $(date +%s) + 432000 ))

[[ "$(hostname)" == 149 ]] || { echo "WRONG_HOST"; exit 2; }

while (( $(date +%s) < DEADLINE )); do
  if [[ -f "$STATUS" ]]; then
    # STATUS.json is atomically replaced by the supervisor; tolerate the brief
    # interval between the existence check and opening the new file.
    state=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["state"])' "$STATUS" 2>/dev/null || true)
    case "$state" in
      COMPLETE)
        [[ -f "$MANIFEST" ]] || { echo "EPISODE_MANIFEST_MISSING"; exit 3; }
        successes=$(find "$EPISODES/cancers" -mindepth 2 -maxdepth 2 -name SUCCESS.json -type f | wc -l)
        [[ "$successes" -eq 28 ]] || {
          echo "EPISODES_COMPLETE_WITHOUT_28_CANCER_SUCCESSES:$successes"
          exit 4
        }
        [[ ! -e "$TRAINING" ]] || { echo "TRAINING_ROOT_ALREADY_EXISTS"; exit 5; }
        cd "$ROOT"
        "$PYTHON" scripts/train_single_cell_hhgt_149.py \
          --episode-root "$EPISODES" \
          --graph-assets-root "$GRAPH" \
          --output-root "$TRAINING" \
          --seed 20261005 \
          --epochs 40 \
          --patience 8 \
          --training-workers 5 \
          --threads-per-worker 8
        [[ -f "$TRAINING/SUCCESS.json" ]] || { echo "TRAINING_SUCCESS_MISSING"; exit 6; }
        "$PYTHON" scripts/validate_single_cell_hhgt_training_149.py \
          --training-root "$TRAINING"
        status=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
          "$TRAINING/INDEPENDENT_VALIDATION.json")
        [[ "$status" == PASS ]] || { echo "INDEPENDENT_VALIDATION_NOT_PASS:$status"; exit 7; }
        echo "TRAINING_AND_INDEPENDENT_VALIDATION_COMPLETE"
        exit 0
        ;;
      FAILED|FAILED_FINALIZE)
        echo "EPISODE_BATCH_FAILED"
        exit 8
        ;;
    esac
  fi
  printf '%s waiting_for_episode_assets\n' "$(date --iso-8601=seconds)"
  sleep 300
done

echo "TRAINING_RELAY_DEADLINE_EXPIRED"
exit 9
