#!/usr/bin/env bash
# Detached CPU continuation; never starts paid compute.
set -euo pipefail
test "$(hostname)" = 149
NEW=/dell_2/DSC/CancerLncAtlas/rbp_encode_v33_20260921_r1/runtime/c_graph_global_binding_20260928_r1
builder_pid=3601891
for minute in $(seq 1 180); do
    if test -s "$NEW/G2_GLOBAL_FOLD_0.json"; then
        break
    fi
    if ! kill -0 "$builder_pid" 2>/dev/null; then
        echo "Graph builder exited before Fold 0 was complete" >&2
        exit 1
    fi
    sleep 60
done
test -s "$NEW/G2_GLOBAL_FOLD_0.json"
test ! -e "$NEW/CPU_MODEL_PREFLIGHT_FOLD_0.json"
test ! -e "$NEW/CLEANUP_SUPERSEDED_C_FOLD0.json"
python3 - "$NEW/CPU_PREFLIGHT_INPUT_FOLD_0.json" <<'PY'
import json, socket, sys
from pathlib import Path
assert socket.gethostname() == "149"
path = Path(sys.argv[1])
if path.exists(): raise FileExistsError(path)
path.write_text(json.dumps({"target_host":"149", "fold":0, "graph_variant":"G2",
                            "workload":"CPU model validation", "paid_gpu_allowed":False,
                            "file_hashes_computed":False}, indent=2)+"\n")
PY
R=/dell_2/DSC/CancerLncAtlas/rbp_encode_v33_20260921_r1
source "$R/env.sh"
export PYTHONPATH="$NEW/code:$PYTHONPATH"
timeout 7200 python3 "$NEW/validate_global_c_overlay_149.py" \
    --fold 0 --overlay-receipt "$NEW/G2_GLOBAL_FOLD_0.json" \
    --all-folds-sha-ready "$R/runtime/c_graph_train_20260926_r1/C_ALL_FOLDS_SHA_READY.json" \
    --output "$NEW/CPU_MODEL_PREFLIGHT_FOLD_0.json"
python3 "$NEW/cleanup_superseded_c_fold0_149.py" \
    --old-overlay "$R/runtime/c_graph_train_20260926_r1/overlays/C_G2_PATIENT_FOLD_0.pt" \
    --new-overlay "$NEW/G2_GLOBAL_FOLD_0.pt" \
    --new-receipt "$NEW/G2_GLOBAL_FOLD_0.json" \
    --cpu-receipt "$NEW/CPU_MODEL_PREFLIGHT_FOLD_0.json" \
    --cleanup-receipt "$NEW/CLEANUP_SUPERSEDED_C_FOLD0.json"
for minute in $(seq 1 720); do
    if test -s "$NEW/ALL_FIVE_G2_GLOBAL_GRAPH_READY.json"; then
        break
    fi
    if ! kill -0 "$builder_pid" 2>/dev/null; then
        echo "Graph builder exited before the five-fold ready receipt" >&2
        exit 1
    fi
    sleep 60
done
test -s "$NEW/ALL_FIVE_G2_GLOBAL_GRAPH_READY.json"
test ! -e "$NEW/ALL_FIVE_G2_GLOBAL_CPU_READY.json"
bash "$NEW/validate_global_c_all_folds_149.sh"
