#!/usr/bin/env bash
set -euo pipefail

ROOT=/dsk2/user/dengsc/sc_metacell_formal_integration_20261005_r1
EPISODE_ROOT=/dell_2/DSC/CancerLncAtlas/rbp_encode_v33_20260921_r1/runtime/c_graph_global_binding_20260928_r1/results/single_cell_hhgt_episodes_v32-single-cell-metacell-donor-oof-20261005-r1
STATUS="$EPISODE_ROOT/batch_supervisor/STATUS.json"
LOG_DIR="$ROOT/logs"

[[ "$(hostname)" == 149 ]] || { echo "WRONG_HOST"; exit 2; }
mkdir -p "$LOG_DIR"

# Direct recovery runs own their cancer episode paths.  Do not let the batch
# supervisor race any active per-cancer materializer, including recovery jobs
# added after this coordinator started.
while :; do
  manual_pids=$(pgrep -f \
    '[m]aterialize_single_cell_hhgt_training_episodes_149.py .*--cancer-id' || true)
  [[ -n "$manual_pids" ]] || break
  printf '%s waiting_for_manual_episode_runs\n' "$(date --iso-8601=seconds)"
  sleep 60
done

nohup "$ROOT/scripts/relay_single_cell_hhgt_episode_149.sh" \
  >"$LOG_DIR/relay_single_cell_hhgt_episode_149.log" 2>&1 &
episode_relay_pid=$!
echo "episode_relay_pid=$episode_relay_pid"

# Do not start the training relay while the stale failed supervisor receipt is
# still visible.  The new supervisor atomically replaces it with RUNNING.
for _ in $(seq 1 60); do
  state=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("state", ""))' \
    "$STATUS" 2>/dev/null || true)
  if [[ "$state" == RUNNING || "$state" == COMPLETE ]]; then
    break
  fi
  kill -0 "$episode_relay_pid" 2>/dev/null || {
    echo "EPISODE_RELAY_EXITED_BEFORE_RUNNING"
    exit 3
  }
  sleep 5
done
[[ "$state" == RUNNING || "$state" == COMPLETE ]] || {
  echo "EPISODE_SUPERVISOR_DID_NOT_START:$state"
  exit 4
}

nohup "$ROOT/scripts/relay_train_and_validate_single_cell_hhgt_149.sh" \
  >"$LOG_DIR/relay_train_and_validate_single_cell_hhgt_149.log" 2>&1 &
echo "training_relay_pid=$!"
nohup "$ROOT/scripts/relay_close_single_cell_hhgt_149.sh" \
  >"$LOG_DIR/relay_close_single_cell_hhgt_149.log" 2>&1 &
echo "closure_relay_pid=$!"
echo "PIPELINE_RELAYS_STARTED"
