#!/usr/bin/env bash
# Run only folds 1..4 beside the already-running Fold 0 on CPU host 149.
set -euo pipefail
test "$(hostname)" = 149
R=/dell_2/DSC/CancerLncAtlas/rbp_encode_v33_20260921_r1
OLD="$R/runtime/c_graph_train_20260926_r1/overlays"
NEW="$R/runtime/c_graph_global_binding_20260928_r1"
source "$R/env.sh"
export PYTHONPATH="$NEW/code:$PYTHONPATH"
test -s "$NEW/PREFLIGHT.json"
test -s "$NEW/lnc_protein_binding_global.parquet"
test ! -e "$NEW/ALL_FIVE_G2_GLOBAL_GRAPH_READY.json"

build_one() {
    local fold="$1" old old_receipt target preflight
    old="$OLD/C_G2_PATIENT_FOLD_${fold}.pt"
    old_receipt="$OLD/C_G2_PATIENT_FOLD_${fold}.json"
    target="$NEW/G2_GLOBAL_FOLD_${fold}.pt"
    preflight="$NEW/PREFLIGHT_GLOBAL_G2_FOLD_${fold}.json"
    test -s "$old"
    test -s "$old_receipt"
    test ! -e "$target"
    test ! -e "${target%.pt}.json"
    python3 - "$fold" "$old" "$old_receipt" "$preflight" <<'PY'
import json, socket, sys
from pathlib import Path
assert socket.gethostname() == "149"
fold, old, receipt, target = int(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
prior = json.loads(receipt.read_text())
if prior.get("fold") != fold or prior.get("path") != str(old) or old.stat().st_size != prior.get("bytes"):
    raise RuntimeError("Prior C fold path, scope or byte size drift")
if target.exists(): raise FileExistsError(target)
target.write_text(json.dumps({"target_host":"149", "workload":"final_G2_global_binding_CPU",
                              "fold":fold, "paid_gpu_allowed":False,
                              "file_hashes_computed":False, "source_overlay":str(old),
                              "source_overlay_bytes":old.stat().st_size}, indent=2)+"\n")
PY
    timeout 7200 python3 "$NEW/build_global_c_overlay_149.py" \
        --old-overlay "$old" \
        --global-binding "$NEW/lnc_protein_binding_global.parquet" \
        --output "$target" --fold "$fold"
}

pids=()
for fold in 1 2 3 4; do
    build_one "$fold" > "$NEW/GRAPH_BUILD_FOLD_${fold}.log" 2>&1 &
    pids+=("$!")
done
failed=0
for index in 0 1 2 3; do
    if ! wait "${pids[$index]}"; then
        echo "Fold $((index+1)) graph build failed; inspect its per-fold log" >&2
        failed=1
    fi
done
test "$failed" = 0

for minute in $(seq 1 180); do
    test -s "$NEW/G2_GLOBAL_FOLD_0.json" && break
    sleep 60
done
test -s "$NEW/G2_GLOBAL_FOLD_0.json"
python3 - "$NEW" <<'PY'
import json, socket, sys
from pathlib import Path
assert socket.gethostname() == "149"
root = Path(sys.argv[1]); folds = []
for fold in range(5):
    path = root / f"G2_GLOBAL_FOLD_{fold}.pt"
    receipt = json.loads(path.with_suffix(".json").read_text())
    if (receipt.get("status") != "PASS_G2_GLOBAL_BINDING_SIDECAR"
            or receipt.get("binding_context_policy") != "GLOBAL_PHYSICAL_BINDING"
            or receipt.get("patient_fold") != fold
            or path.stat().st_size != receipt.get("bytes")):
        raise RuntimeError(f"Final G2 fold {fold} is not complete")
    folds.append({"fold":fold,"path":str(path),"bytes":path.stat().st_size,
                  "new_edges":receipt["new_edges"],"new_binding_edges":receipt["new_binding_edges"]})
target = root / "ALL_FIVE_G2_GLOBAL_GRAPH_READY.json"
if target.exists(): raise FileExistsError(target)
target.write_text(json.dumps({"status":"PASS_ALL_FIVE_G2_GLOBAL_GRAPH_READY",
                              "target_host":"149", "paid_gpu_allowed":False,
                              "file_hashes_computed":False,"folds":folds},indent=2)+"\n")
print(json.dumps({"status":"PASS_ALL_FIVE_G2_GLOBAL_GRAPH_READY","folds":folds}))
PY
