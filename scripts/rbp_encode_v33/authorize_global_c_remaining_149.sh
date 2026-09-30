#!/usr/bin/env bash
# Seal training authorization for final G2 folds 1..4 after five-fold CPU checks.
set -euo pipefail
test "$(hostname)" = 149
R=/dell_2/DSC/CancerLncAtlas/rbp_encode_v33_20260921_r1
OLD="$R/runtime/c_graph_train_20260926_r1"
NEW="$R/runtime/c_graph_global_binding_20260928_r1"
source "$R/env.sh"
export PYTHONPATH="$NEW/code:$PYTHONPATH"
test -s "$NEW/ALL_FIVE_G2_GLOBAL_CPU_READY.json"
test -s "$NEW/GLOBAL_G2_TRAIN_CODE_READY.json"
test -s "$NEW/auth_postpay_final_g2_fold0/AUTH_READY.json"
for fold in 1 2 3 4; do
    output="$NEW/auth_postpay_final_g2_fold${fold}"
    test ! -e "$output"
    test -s "$NEW/G2_GLOBAL_FOLD_${fold}.json"
    test -s "$NEW/CPU_MODEL_PREFLIGHT_FOLD_${fold}.json"
    python3 - "$fold" "$NEW/AUTH_GLOBAL_FOLD${fold}_PREFLIGHT.json" <<'PY'
import json, socket, sys
from pathlib import Path
assert socket.gethostname() == "149"
fold = int(sys.argv[1]); path = Path(sys.argv[2])
if path.exists(): raise FileExistsError(path)
path.write_text(json.dumps({"target_host":"149", "fold":fold,
                            "workload":"final_G2_training_authorization_CPU",
                            "paid_gpu_allowed":False,
                            "large_input_hashes_computed":False},indent=2)+"\n")
PY
    python3 "$NEW/authorize_global_c_fold_149.py" \
        --fold "$fold" \
        --template-auth "$OLD/auth_postpay_v100s_fold${fold}_v2" \
        --parent-ready "$OLD/C_ALL_FOLDS_SHA_READY.json" \
        --overlay-receipt "$NEW/G2_GLOBAL_FOLD_${fold}.json" \
        --cpu-receipt "$NEW/CPU_MODEL_PREFLIGHT_FOLD_${fold}.json" \
        --global-binding "$NEW/lnc_protein_binding_global.parquet" \
        --repo-root "$NEW/code" --output-root "$output" \
        --cloud-root "/root/CancerLncAtlas_C_GLOBAL_G2_20260928/fold_${fold}" \
        --code-archive "$NEW/GLOBAL_G2_TRAIN_CODE.tar.gz" \
        --wheel-archive "$OLD/CLOUD_WHEELHOUSE.tar" \
        --wheel-requirements "$OLD/cloud_runtime_wheels_20260926.txt" \
        --launcher "$NEW/launch_global_c_gpu.py" \
        --max-paid-hours 36 --max-cost-cny 210
done
python3 - "$NEW" <<'PY'
import json, socket, sys
from pathlib import Path
assert socket.gethostname() == "149"
root = Path(sys.argv[1]); folds = []
for fold in range(5):
    path = root / f"auth_postpay_final_g2_fold{fold}" / "AUTH_READY.json"
    row = json.loads(path.read_text())
    if (row.get("status") != "PASS_CPU_ONLY_GLOBAL_G2_FOLD_AUTHORIZATION"
            or row.get("host") != "149" or row.get("fold") != fold
            or row.get("gpu_started") is not False
            or row.get("large_files_rehashed") is not False):
        raise RuntimeError(f"Final G2 fold {fold} has no authorized CPU receipt")
    folds.append({"fold":fold,"path":str(path),
                  "graph_overlay_bytes":row["graph_overlay_bytes"]})
target = root / "ALL_FIVE_G2_GLOBAL_AUTH_READY.json"
if target.exists(): raise FileExistsError(target)
target.write_text(json.dumps({"status":"PASS_ALL_FIVE_G2_GLOBAL_AUTH_READY",
                              "target_host":"149", "gpu_started":False,
                              "large_files_rehashed":False,
                              "folds":folds},indent=2)+"\n")
print(json.dumps({"status":"PASS_ALL_FIVE_G2_GLOBAL_AUTH_READY","folds":folds}))
PY
