#!/usr/bin/env bash
# Build exactly one graph-only C sidecar; reuse immutable A batches at training.
set -euo pipefail
fold="${1:?patient fold required}"
case "$fold" in 1|2|3|4) ;; *) echo "fold must be 1..4" >&2; exit 2 ;; esac
test "$(hostname)" = 149
R=/dell_2/DSC/CancerLncAtlas/rbp_encode_v33_20260921_r1
T="$R/runtime/c_graph_train_20260926_r1"
test -f "$T/C_FOLD0_CPU_READINESS.json"
preflight="$T/PREFLIGHT_FOLD_${fold}.json"
test ! -e "$preflight"
test ! -e "$T/overlays/C_G2_PATIENT_FOLD_${fold}.pt"
test ! -e "$T/overlays/C_G2_PATIENT_FOLD_${fold}.json"
test ! -e "$T/BUILD_FOLD_${fold}_EXIT_CODE"
printf '{"format":"C_GRAPH_FOLD_CPU_PREFLIGHT_V1","target_host":"149","patient_fold":%s,"workload":"C_GRAPH_ONLY","paid_gpu_allowed":false}\n' "$fold" > "$preflight"
source "$R/env.sh"
export PYTHONPATH="$T/code:$PYTHONPATH"
set +e
timeout 10800 python3 "$T/code/scripts/rbp_encode_v33/build_c_graph_overlay.py" \
  --authority-root "$R/outputs/phase7_authority_v2" \
  --prepared-root /dell_2/DSC/CancerLncAtlas/inputs/v32_g012_local_cnv_formal_prepared_20260903_r1 \
  --source-static-auth /dell_2/DSC/CancerLncAtlas/runtime/tools/v32_g012_corrected_paid_gpu_20260905_r1/bootstrap/STATIC_AUTH_READY.json \
  --output-root "$T/overlays" \
  --fold "$fold"
rc=$?
printf '%s\n' "$rc" > "$T/BUILD_FOLD_${fold}_EXIT_CODE"
exit "$rc"
