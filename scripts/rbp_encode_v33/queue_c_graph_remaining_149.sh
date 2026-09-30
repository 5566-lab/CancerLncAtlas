#!/usr/bin/env bash
# Finish the necessary C graph sidecars after the two active CPU builders exit.
set -euo pipefail
test "$(hostname)" = 149

T=/dell_2/DSC/CancerLncAtlas/rbp_encode_v33_20260921_r1/runtime/c_graph_train_20260926_r1
test -f "$T/PREFLIGHT_FOLD_1.json"
test -f "$T/PREFLIGHT_FOLD_2.json"
test ! -e "$T/C_GRAPH_QUEUE_EXIT_CODE"
trap 'rc=$?; printf "%s\n" "$rc" > "$T/C_GRAPH_QUEUE_EXIT_CODE"' EXIT

printf '{"format":"C_GRAPH_QUEUE_V1","target_host":"149","wait_for_folds":[1,2],"launch_folds":[3,4],"paid_gpu_allowed":false}\n' > "$T/C_GRAPH_QUEUE_PREFLIGHT.json"

deadline=$((SECONDS + 10800))
for fold in 1 2; do
  while test ! -f "$T/BUILD_FOLD_${fold}_EXIT_CODE"; do
    if ((SECONDS >= deadline)); then
      echo "Timed out waiting for C graph fold $fold" >&2
      exit 124
    fi
    sleep 30
  done
  test "$(cat "$T/BUILD_FOLD_${fold}_EXIT_CODE")" = 0
  test -s "$T/overlays/C_G2_PATIENT_FOLD_${fold}.pt"
  test -s "$T/overlays/C_G2_PATIENT_FOLD_${fold}.json"
done

python3 - "$T" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for fold in (1, 2):
    receipt = json.loads((root / "overlays" / f"C_G2_PATIENT_FOLD_{fold}.json").read_text())
    path = root / "overlays" / f"C_G2_PATIENT_FOLD_{fold}.pt"
    assert receipt["host"] == "149" and receipt["fold"] == fold
    assert receipt["path"] == str(path) and receipt["bytes"] == path.stat().st_size
    assert receipt["active_eclip_edges"] > 0 and receipt["predicted_edges"] == 0
PY

for fold in 3 4; do
  test ! -e "$T/PREFLIGHT_FOLD_${fold}.json"
  test ! -e "$T/BUILD_FOLD_${fold}_EXIT_CODE"
  test ! -e "$T/overlays/C_G2_PATIENT_FOLD_${fold}.pt"
  test ! -e "$T/overlays/C_G2_PATIENT_FOLD_${fold}.json"
done

bash "$T/code/scripts/rbp_encode_v33/run_c_graph_remaining_149.sh" 3 > "$T/BUILD_FOLD_3.log" 2>&1 &
pid3=$!
bash "$T/code/scripts/rbp_encode_v33/run_c_graph_remaining_149.sh" 4 > "$T/BUILD_FOLD_4.log" 2>&1 &
pid4=$!
printf '{"format":"C_GRAPH_QUEUE_LAUNCHED_V1","target_host":"149","fold_3_pid":%s,"fold_4_pid":%s}\n' "$pid3" "$pid4" > "$T/C_GRAPH_QUEUE_LAUNCHED.json"

rc=0
wait "$pid3" || rc=1
wait "$pid4" || rc=1
exit "$rc"
