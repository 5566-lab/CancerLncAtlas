#!/usr/bin/env bash
# Validate the final G2 graph and CPU model for each frozen patient fold.
set -euo pipefail
test "$(hostname)" = 149
R=/dell_2/DSC/CancerLncAtlas/rbp_encode_v33_20260921_r1
OLD="$R/runtime/c_graph_train_20260926_r1"
NEW="$R/runtime/c_graph_global_binding_20260928_r1"
source "$R/env.sh"
export PYTHONPATH="$NEW/code:$PYTHONPATH"
test -s "$NEW/ALL_FIVE_G2_GLOBAL_GRAPH_READY.json"
test -s "$NEW/CPU_MODEL_PREFLIGHT_FOLD_0.json"
test -s "$NEW/CLEANUP_SUPERSEDED_C_FOLD0.json"
python3 - "$NEW/CPU_MODEL_PREFLIGHT_FOLD_0.json" "$NEW/CLEANUP_SUPERSEDED_C_FOLD0.json" <<'PY'
import json,sys
from pathlib import Path
cpu=json.loads(Path(sys.argv[1]).read_text())
clean=json.loads(Path(sys.argv[2]).read_text())
if (cpu.get("status") != "PASS_G2_GLOBAL_C_CPU_MODEL_PREFLIGHT"
        or cpu.get("target_host") != "149" or cpu.get("fold") != 0
        or clean.get("status") != "SUPERSEDED_C_FOLD0_REMOVED"
        or clean.get("target_host") != "149"):
    raise RuntimeError("Existing Fold 0 validation or cleanup is invalid")
PY
pids=()
for fold in 1 2 3 4; do
    receipt="$NEW/G2_GLOBAL_FOLD_${fold}.json"
    output="$NEW/CPU_MODEL_PREFLIGHT_FOLD_${fold}.json"
    test -s "$receipt"
    test ! -e "$output"
    test ! -e "$NEW/CPU_PREFLIGHT_INPUT_FOLD_${fold}.json"
    (
    python3 - "$fold" "$NEW/CPU_PREFLIGHT_INPUT_FOLD_${fold}.json" <<'PY'
import json, socket, sys
from pathlib import Path
assert socket.gethostname() == "149"
fold = int(sys.argv[1]); path = Path(sys.argv[2])
if path.exists(): raise FileExistsError(path)
path.write_text(json.dumps({"target_host":"149", "fold":fold, "graph_variant":"G2",
                            "workload":"CPU model validation", "paid_gpu_allowed":False,
                            "file_hashes_computed":False}, indent=2)+"\n")
PY
    timeout 7200 python3 "$NEW/validate_global_c_overlay_149.py" \
        --fold "$fold" --overlay-receipt "$receipt" \
        --all-folds-sha-ready "$OLD/C_ALL_FOLDS_SHA_READY.json" \
        --output "$output"
    python3 "$NEW/cleanup_superseded_c_fold0_149.py" \
        --fold "$fold" \
        --old-overlay "$OLD/overlays/C_G2_PATIENT_FOLD_${fold}.pt" \
        --new-overlay "$NEW/G2_GLOBAL_FOLD_${fold}.pt" \
        --new-receipt "$receipt" --cpu-receipt "$output" \
        --cleanup-receipt "$NEW/CLEANUP_SUPERSEDED_C_FOLD_${fold}.json"
    ) > "$NEW/CPU_MODEL_PREFLIGHT_FOLD_${fold}.log" 2>&1 &
    pids+=("$!")
done
failed=0
for index in 0 1 2 3; do
    if ! wait "${pids[$index]}"; then
        echo "Fold $((index+1)) CPU model validation failed; inspect its log" >&2
        failed=1
    fi
done
test "$failed" = 0
python3 - "$NEW" <<'PY'
import json, socket, sys
from pathlib import Path
assert socket.gethostname() == "149"
root = Path(sys.argv[1]); folds = []
for fold in range(5):
    path = root / f"CPU_MODEL_PREFLIGHT_FOLD_{fold}.json"
    row = json.loads(path.read_text())
    if (row.get("status") != "PASS_G2_GLOBAL_C_CPU_MODEL_PREFLIGHT"
            or row.get("target_host") != "149" or row.get("fold") != fold
            or row.get("file_hashes_computed") is not False):
        raise RuntimeError(f"Fold {fold} has no final G2 CPU validation")
    folds.append({"fold":fold,"receipt":str(path),"graph_edges":row["graph_edges"]})
    if fold > 0:
        cleanup = json.loads((root / f"CLEANUP_SUPERSEDED_C_FOLD_{fold}.json").read_text())
        if (cleanup.get("status") != "SUPERSEDED_C_FOLD_REMOVED"
                or cleanup.get("target_host") != "149"
                or cleanup.get("fold") != fold):
            raise RuntimeError(f"Fold {fold} obsolete sidecar was not removed")
target = root / "ALL_FIVE_G2_GLOBAL_CPU_READY.json"
if target.exists(): raise FileExistsError(target)
target.write_text(json.dumps({"status":"PASS_ALL_FIVE_G2_GLOBAL_CPU_READY",
                              "target_host":"149", "paid_gpu_allowed":False,
                              "file_hashes_computed":False, "folds":folds}, indent=2)+"\n")
print(json.dumps({"status":"PASS_ALL_FIVE_G2_GLOBAL_CPU_READY","folds":folds}))
PY
