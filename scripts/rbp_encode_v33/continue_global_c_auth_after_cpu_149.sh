#!/usr/bin/env bash
# Authorize final G2 folds only after the independent five-fold CPU receipt.
set -euo pipefail
test "$(hostname)" = 149
NEW=/dell_2/DSC/CancerLncAtlas/rbp_encode_v33_20260921_r1/runtime/c_graph_global_binding_20260928_r1
cpu_controller_pid=3602562
for minute in $(seq 1 180); do
    test -s "$NEW/ALL_FIVE_G2_GLOBAL_CPU_READY.json" && break
    if ! kill -0 "$cpu_controller_pid" 2>/dev/null; then
        echo "Five-fold CPU validator exited without its readiness receipt" >&2
        exit 1
    fi
    sleep 60
done
test -s "$NEW/ALL_FIVE_G2_GLOBAL_CPU_READY.json"
bash "$NEW/authorize_global_c_remaining_149.sh"
