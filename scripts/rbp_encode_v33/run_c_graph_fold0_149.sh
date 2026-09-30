#!/usr/bin/env bash
# One intentional graph-only artifact.  The 170 GB A payloads are read-only.
set -euo pipefail
test "$(hostname)" = 149
R=/dell_2/DSC/CancerLncAtlas/rbp_encode_v33_20260921_r1
T="$R/runtime/c_graph_train_20260926_r1"
test -f "$T/PREFLIGHT.json"
source "$R/env.sh"
export PYTHONPATH="$T/code:$PYTHONPATH"
set +e
timeout 3600 python3 "$T/code/scripts/rbp_encode_v33/build_c_graph_overlay.py" \
  --authority-root "$R/outputs/phase7_authority_v2" \
  --prepared-root /dell_2/DSC/CancerLncAtlas/inputs/v32_g012_local_cnv_formal_prepared_20260903_r1 \
  --source-static-auth /dell_2/DSC/CancerLncAtlas/runtime/tools/v32_g012_corrected_paid_gpu_20260905_r1/bootstrap/STATIC_AUTH_READY.json \
  --output-root "$T/overlays" \
  --fold 0
rc=$?
printf '%s\n' "$rc" > "$T/BUILD_FOLD0_EXIT_CODE"
exit "$rc"
