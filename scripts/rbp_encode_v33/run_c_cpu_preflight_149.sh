#!/usr/bin/env bash
set -euo pipefail
test "$(hostname)" = 149
R=/dell_2/DSC/CancerLncAtlas/rbp_encode_v33_20260921_r1
T="$R/runtime/c_graph_train_20260926_r1"
source "$R/env.sh"
export PYTHONPATH="$T/code:$PYTHONPATH"
set +e
timeout 5400 python3 "$T/code/scripts/rbp_encode_v33/validate_c_overlay_fold0_149.py" \
  --overlay-receipt "$T/overlays/C_G2_PATIENT_FOLD_0.json" \
  --parent-hash-receipt "$T/SOURCE_A_FOLD0.sha256" \
  --output "$T/CPU_OVERLAY_PREFLIGHT.json"
rc=$?
printf '%s\n' "$rc" > "$T/CPU_OVERLAY_PREFLIGHT_EXIT_CODE"
exit "$rc"
