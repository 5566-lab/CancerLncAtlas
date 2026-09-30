#!/usr/bin/env bash
set -euo pipefail

test "$(hostname)" = 149
R=/dell_2/DSC/CancerLncAtlas/rbp_encode_v33_20260921_r1
T="$R/runtime/c_graph_train_20260926_r1"

python3 - "$T/C_ALL_FOLDS_SHA_READY.json" "$T/C_REMAINING_CPU_PREFLIGHT.json" <<'PY'
import datetime
import json
import sys
from pathlib import Path

ready = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if (ready.get("status") != "PASS_C_ALL_FOLDS_INPUT_SHA256"
        or ready.get("target_host") != "149"
        or [row.get("fold") for row in ready.get("folds", [])] != list(range(5))):
    raise RuntimeError("Five-fold C input SHA256 receipt is not ready on host 149")
preflight = {
    "status": "C_REMAINING_CPU_PREFLIGHT_STARTED",
    "target_host": "149",
    "folds": [1, 2, 3, 4],
    "paid_gpu_started": False,
    "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
with Path(sys.argv[2]).open("x", encoding="utf-8") as handle:
    json.dump(preflight, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

source "$R/env.sh"
export PYTHONPATH="$T/code:${PYTHONPATH:-}"

for fold in 1 2 3 4; do
    output="$T/CPU_OVERLAY_PREFLIGHT_FOLD_${fold}.json"
    exit_code="$T/CPU_OVERLAY_PREFLIGHT_FOLD_${fold}_EXIT_CODE"
    test ! -e "$output"
    test ! -e "$exit_code"
    set +e
    timeout 5400 python3 "$T/validate_c_overlay_fold0_149.py" \
        --fold "$fold" \
        --overlay-receipt "$T/overlays/C_G2_PATIENT_FOLD_${fold}.json" \
        --all-folds-sha-ready "$T/C_ALL_FOLDS_SHA_READY.json" \
        --output "$output"
    rc=$?
    set -e
    printf '%s\n' "$rc" > "$exit_code"
    if [ "$rc" -ne 0 ]; then
        exit "$rc"
    fi
done
