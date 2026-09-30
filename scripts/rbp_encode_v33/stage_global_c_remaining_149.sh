#!/usr/bin/env bash
# Stage final G2 folds 1..4 on the existing no-GPU CompShare instance.
set -euo pipefail
test "$(hostname)" = 149
R=/dell_2/DSC/CancerLncAtlas/rbp_encode_v33_20260921_r1
OLD="$R/runtime/c_graph_train_20260926_r1"
NEW="$R/runtime/c_graph_global_binding_20260928_r1"
INSTANCE=uhost-1vsbwarp9m3s
HOST=117.50.215.156
PORT=23
KEY="$OLD/keys/v100s_transfer_ed25519"
test -s "$NEW/NO_GPU_STAGING_PREFLIGHT_149.json"
test -s "$NEW/ALL_FIVE_G2_GLOBAL_AUTH_READY.json"
test -s "$NEW/G2_GLOBAL_FOLD_0_CLOUD_STAGED_NO_REHASH.json"
for fold in 1 2 3 4; do
    receipt="$NEW/G2_GLOBAL_FOLD_${fold}_CLOUD_STAGED_NO_REHASH.json"
    if test -e "$receipt"; then
        python3 - "$receipt" "$fold" "$INSTANCE" <<'PY'
import json, sys
from pathlib import Path
row=json.loads(Path(sys.argv[1]).read_text())
assert row['format']=='C_GLOBAL_G2_CLOUD_STAGED_NO_REHASH_V1'
assert row['fold']==int(sys.argv[2]) and row['instance_id']==sys.argv[3]
print(f"Fold {row['fold']} already staged", flush=True)
PY
        continue
    fi
    python3 "$NEW/stage_global_c_cloud_no_rehash_149.py" \
        --manifest "$NEW/auth_postpay_final_g2_fold${fold}/CLOUD_TRANSFER_MANIFEST.json" \
        --fold-authority "$OLD/C_ALL_FOLDS_SHA_READY.json" \
        --prior-transfer-manifest "$OLD/auth_postpay_v100s_fold${fold}_v2/CLOUD_TRANSFER_MANIFEST.json" \
        --overlay-receipt "$NEW/G2_GLOBAL_FOLD_${fold}.json" \
        --cpu-receipt "$NEW/CPU_MODEL_PREFLIGHT_FOLD_${fold}.json" \
        --instance-id "$INSTANCE" --ssh-host "$HOST" --ssh-port "$PORT" \
        --identity-file "$KEY" --receipt "$receipt" --user-directed-no-rehash
    echo "Fold $fold staged" >&2
done
python3 - "$NEW" "$INSTANCE" <<'PY'
import json, socket, sys
from pathlib import Path
assert socket.gethostname()=='149'
root=Path(sys.argv[1]); instance=sys.argv[2]; rows=[]
for fold in range(5):
    receipt=root/f'G2_GLOBAL_FOLD_{fold}_CLOUD_STAGED_NO_REHASH.json'
    row=json.loads(receipt.read_text())
    if (row.get('format')!='C_GLOBAL_G2_CLOUD_STAGED_NO_REHASH_V1'
            or row.get('fold')!=fold or row.get('instance_id')!=instance
            or row.get('preparation_host')!='149'
            or len(row.get('verified_roles',[]))!=12):
        raise RuntimeError(f'Fold {fold} cloud staging receipt is invalid')
    rows.append({'fold':fold,'receipt':str(receipt)})
target=root/'ALL_FIVE_G2_GLOBAL_CLOUD_STAGED_NO_REHASH.json'
if target.exists(): raise FileExistsError(target)
target.write_text(json.dumps({'status':'PASS_ALL_FIVE_G2_GLOBAL_CLOUD_STAGED_NO_REHASH',
    'target_host':'149','gpu_target_instance_id':instance,'large_files_rehashed':False,
    'folds':rows},indent=2)+'\n')
print(target,flush=True)
PY
