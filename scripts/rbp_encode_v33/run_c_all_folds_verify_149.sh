#!/usr/bin/env bash
# Read-only SHA256 audit of the five reused A inputs and five new C graph files.
set -euo pipefail
test "$(hostname)" = 149

T=/dell_2/DSC/CancerLncAtlas/rbp_encode_v33_20260921_r1/runtime/c_graph_train_20260926_r1
AUTH=/dell_2/DSC/CancerLncAtlas/runtime/tools/v32_g012_corrected_paid_gpu_20260905_r1/bootstrap/STATIC_AUTH_READY.json
test ! -e "$T/C_ALL_FOLDS_VERIFY_PREFLIGHT.json"
test ! -e "$T/C_ALL_FOLDS_SHA_READY.json"
test ! -e "$T/C_ALL_FOLDS_VERIFY_EXIT_CODE"
printf '{"format":"C_ALL_FOLDS_SHA_PREFLIGHT_V1","target_host":"149","workload":"CPU_ONLY_HASH_AUDIT","paid_gpu_allowed":false}\n' > "$T/C_ALL_FOLDS_VERIFY_PREFLIGHT.json"

set +e
timeout 14400 python3 "$T/verify_c_all_folds_149.py" --runtime-root "$T" --static-auth "$AUTH"
rc=$?
printf '%s\n' "$rc" > "$T/C_ALL_FOLDS_VERIFY_EXIT_CODE"
exit "$rc"
