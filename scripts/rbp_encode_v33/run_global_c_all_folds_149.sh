#!/usr/bin/env bash
# Final G2 only. Reuse the old A folds and old C graph relations; replace binding.
set -euo pipefail

test "$(hostname)" = 149
R=/dell_2/DSC/CancerLncAtlas/rbp_encode_v33_20260921_r1
OLD="$R/runtime/c_graph_train_20260926_r1/overlays"
NEW="$R/runtime/c_graph_global_binding_20260928_r1"
source "$R/env.sh"
export PYTHONPATH="$NEW/code:$PYTHONPATH"

test -s "$NEW/PREFLIGHT.json"
test -s "$NEW/lnc_protein_binding_global.parquet"
test -s "$NEW/build_global_c_overlay_149.py"
for fold in 0 1 2 3 4; do
    old="$OLD/C_G2_PATIENT_FOLD_${fold}.pt"
    old_receipt="$OLD/C_G2_PATIENT_FOLD_${fold}.json"
    target="$NEW/G2_GLOBAL_FOLD_${fold}.pt"
    test -s "$old"
    test -s "$old_receipt"
    test ! -e "$target"
    test ! -e "${target%.pt}.json"
    python3 - "$fold" "$old" "$old_receipt" "$NEW/PREFLIGHT_GLOBAL_G2_FOLD_${fold}.json" <<'PY'
import json, socket, sys
from pathlib import Path
assert socket.gethostname() == "149"
fold, old, receipt, target = int(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
metadata = json.loads(receipt.read_text())
if metadata.get("fold") != fold or metadata.get("path") != str(old) or old.stat().st_size != metadata.get("bytes"):
    raise RuntimeError("Prior C fold path, scope or byte size drift")
if target.exists():
    existing = json.loads(target.read_text())
    if existing.get("target_host") != "149" or existing.get("fold") != fold:
        raise RuntimeError("Existing CPU preflight has wrong scope")
else:
    target.write_text(json.dumps({"target_host":"149", "workload":"final_G2_global_binding_CPU", "fold":fold,
                                  "paid_gpu_allowed":False, "file_hashes_computed":False,
                                  "source_overlay":str(old), "source_overlay_bytes":old.stat().st_size}, indent=2)+"\n")
PY
    timeout 7200 python3 "$NEW/build_global_c_overlay_149.py" \
        --old-overlay "$old" \
        --global-binding "$NEW/lnc_protein_binding_global.parquet" \
        --output "$target" --fold "$fold"
done
python3 - "$NEW" <<'PY'
import json, socket, sys
from pathlib import Path
assert socket.gethostname() == "149"
root = Path(sys.argv[1])
folds = []
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
