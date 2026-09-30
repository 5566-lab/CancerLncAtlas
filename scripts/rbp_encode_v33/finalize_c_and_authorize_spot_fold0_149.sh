#!/usr/bin/env bash
set -euo pipefail

test "$(hostname)" = 149
R=/dell_2/DSC/CancerLncAtlas/rbp_encode_v33_20260921_r1
T="$R/runtime/c_graph_train_20260926_r1"
RUNNER_PID=3326592
EXIT_RECEIPT="$T/C_FOLD0_SPOT_AUTH_PIPELINE_EXIT_CODE"

test ! -e "$EXIT_RECEIPT"
finish() {
    rc=$?
    printf '%s\n' "$rc" > "$EXIT_RECEIPT"
}
trap finish EXIT

test ! -e "$T/C_FOLD0_SPOT_AUTH_PIPELINE_PREFLIGHT.json"
printf '%s\n' '{"status":"WAITING_FOR_C_ALL_FOLDS_CPU_READY","target_host":"149","billing_mode":"Spot","fold":0,"paid_gpu_started":false}' \
    > "$T/C_FOLD0_SPOT_AUTH_PIPELINE_PREFLIGHT.json"

while ps -p "$RUNNER_PID" -o args= 2>/dev/null \
        | grep -Fq "$T/run_c_cpu_preflight_remaining_149.sh"; do
    sleep 30
done

for fold in 1 2 3 4; do
    test "$(cat "$T/CPU_OVERLAY_PREFLIGHT_FOLD_${fold}_EXIT_CODE")" = 0
done

python3 "$T/finalize_c_all_folds_cpu_149.py" --runtime-root "$T"

source "$R/env.sh"
export PYTHONPATH="$T/code:${PYTHONPATH:-}"

AUTH="$T/auth_spot_fold0"
python3 "$T/code/scripts/rbp_encode_v33/authorize_c_overlay_fold0.py" \
    --fold 0 \
    --old-config /dell_2/DSC/CancerLncAtlas/results/v32_g012_corrected_paid_gpu_training_20260905_r1/G2/config.yaml \
    --old-input-manifest /dell_2/DSC/CancerLncAtlas/results/v32_g012_corrected_paid_gpu_training_20260905_r1/G2/authorization/INPUT_MANIFEST.json \
    --overlay-receipt "$T/overlays/C_G2_PATIENT_FOLD_0.json" \
    --all-folds-sha-ready "$T/C_ALL_FOLDS_SHA_READY.json" \
    --output-root "$AUTH" \
    --cloud-root /root/CancerLncAtlas_C_20260926/fold_0 \
    --code-archive "$T/C_TRAIN_CODE.tar.gz" \
    --wheel-archive "$T/CLOUD_WHEELHOUSE.tar" \
    --wheel-requirements "$T/cloud_runtime_wheels_20260926.txt" \
    --launcher "$T/launch_c_gpu.py" \
    --billing-mode Spot \
    --hardware-class PAID_PREEMPTIBLE_GPU \
    --max-paid-hours 96 \
    --max-cost-cny 210

MANIFEST="$AUTH/CLOUD_TRANSFER_MANIFEST.json"
MANIFEST_SHA="$(sha256sum "$MANIFEST" | cut -d' ' -f1)"
python3 "$T/stage_c_cloud_149.py" preflight \
    --manifest "$MANIFEST" \
    --manifest-sha256 "$MANIFEST_SHA" \
    --receipt "$AUTH/SOURCE_READY.json"
