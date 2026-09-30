#!/usr/bin/env bash
# Return all five successful final G2 folds, then seal one size-only receipt.
set -euo pipefail
test "$(hostname)" = 149
R=/dell_2/DSC/CancerLncAtlas/rbp_encode_v33_20260921_r1
OLD="$R/runtime/c_graph_train_20260926_r1"
NEW="$R/runtime/c_graph_global_binding_20260928_r1"
INSTANCE=uhost-1vsbwarp9m3s
HOST=117.50.215.156
PORT=23
KEY="$OLD/keys/v100s_transfer_ed25519"
test -s "$NEW/ALL_FIVE_G2_GLOBAL_CPU_READY.json"
test -s "$NEW/ALL_FIVE_G2_GLOBAL_AUTH_READY.json"
test -s "$NEW/ALL_FIVE_G2_GLOBAL_CLOUD_STAGED_NO_REHASH.json"
for fold in 0 1 2 3 4; do
    destination="$NEW/results/fold_${fold}"
    receipt="$NEW/RESULT_RETURN_FOLD_${fold}.json"
    if test -e "$receipt"; then
        python3 - "$receipt" "$destination" "$fold" <<'PY'
import json, sys
from pathlib import Path
row=json.loads(Path(sys.argv[1]).read_text())
if (row.get('status')!='PASS_RESULT_RETURN_SIZE_VERIFIED_NO_REHASH'
        or row.get('destination')!=sys.argv[2]
        or row.get('fold')!=int(sys.argv[3])
        or row.get('file_count',0)<4
        or not Path(sys.argv[2]).is_dir()):
    raise RuntimeError('Existing result return receipt is invalid')
PY
        continue
    fi
    test ! -e "$destination"
    python3 "$NEW/collect_global_c_results_no_rehash_149.py" \
        --fold "$fold" --ssh-host "$HOST" --ssh-port "$PORT" \
        --identity-file "$KEY" \
        --cloud-root "/root/CancerLncAtlas_C_GLOBAL_G2_20260928/fold_${fold}" \
        --destination "$destination" --receipt "$receipt"
done
python3 - "$NEW" "$INSTANCE" <<'PY'
import json, socket, sys
from pathlib import Path
assert socket.gethostname()=='149'
root=Path(sys.argv[1]);instance=sys.argv[2]
rows=[]
for fold in range(5):
    path=root/f'RESULT_RETURN_FOLD_{fold}.json'
    row=json.loads(path.read_text())
    if (row.get('status')!='PASS_RESULT_RETURN_SIZE_VERIFIED_NO_REHASH'
            or row.get('fold')!=fold
            or row.get('destination')!=str(root/'results'/f'fold_{fold}')
            or row.get('success',{}).get('optimizer_steps',0)<=0):
        raise RuntimeError(f'Fold {fold} result is not verified')
    rows.append({'fold':fold,'receipt':str(path),'bytes':row['total_bytes'],
                 'success':row['success']})
target=root/'ALL_FIVE_G2_RESULTS_RETURNED_NO_REHASH.json'
if target.exists(): raise FileExistsError(target)
target.write_text(json.dumps({'status':'PASS_ALL_FIVE_G2_RESULTS_RETURNED_NO_REHASH',
    'target_host':'149','gpu_instance_id':instance,'large_files_rehashed':False,
    'folds':rows},indent=2)+'\n')
print(target,flush=True)
PY
