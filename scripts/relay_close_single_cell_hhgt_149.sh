#!/usr/bin/env bash
set -euo pipefail

ROOT=/dsk2/user/dengsc/sc_metacell_formal_integration_20261005_r1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
BASE=/dell_2/DSC/CancerLncAtlas/rbp_encode_v33_20260921_r1/runtime/c_graph_global_binding_20260928_r1/results
SOURCE="$BASE/single_cell_metacell_v32-single-cell-metacell-donor-oof-20261005-r1"
GRAPH="$BASE/single_cell_hhgt_graph_assets_v32-single-cell-metacell-donor-oof-20261005-r1"
NESTED="$BASE/single_cell_hhgt_nested_assets_v32-single-cell-metacell-donor-oof-20261005-r1"
EPISODES="$BASE/single_cell_hhgt_episodes_v32-single-cell-metacell-donor-oof-20261005-r1"
TRAINING="$BASE/single_cell_hhgt_v32-single-cell-metacell-donor-oof-20261005-r1"
AUDIT=/dsk2/user/dengsc/sc_hhgt_closure_20261005_r1
POINTER=/dell_2/DSC/CancerLncAtlas/rbp_encode_v33_20260921_r1/manifests/CURRENT_SINGLE_CELL_HEAD.json
VERSION_LOG=/dsk2/user/dengsc/CancerLncAtlas_v32_bugfix_20260906_r1/VERSION_CHANGE_LOG.jsonl
EXPECTED_POINTER=1dba49961b3491c247f3b5a7825f06dfc1be18ce1afbcefd501da3e266446bf2
EXPECTED_SOURCE=cdd6edb40061e99459eebfca53fa80d72a91aff4b4967b2c9a2086bc9d565958
EVENT_ID=V32-SINGLE-CELL-SC-HHGT-ACTIVATED-20261005-R1
DEADLINE=$(( $(date +%s) + 604800 ))

[[ "$(hostname)" == 149 ]] || { echo "WRONG_HOST"; exit 2; }

while (( $(date +%s) < DEADLINE )); do
  current_pointer=$(sha256sum "$POINTER" | awk '{print $1}')
  [[ "$current_pointer" == "$EXPECTED_POINTER" ]] || {
    echo "FORMAL_POINTER_CHANGED:$current_pointer"
    exit 3
  }
  if grep -Fq "\"event_id\":\"$EVENT_ID\"" "$VERSION_LOG"; then
    echo "ACTIVATION_EVENT_ALREADY_EXISTS"
    exit 4
  fi
  if [[ -f "$TRAINING/INDEPENDENT_VALIDATION.json" ]]; then
    status=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$TRAINING/INDEPENDENT_VALIDATION.json" 2>/dev/null || true)
    [[ -n "$status" ]] || {
      printf '%s waiting_for_stable_independent_validation\n' "$(date --iso-8601=seconds)"
      sleep 5
      continue
    }
    [[ "$status" == PASS ]] || { echo "INDEPENDENT_VALIDATION_NOT_PASS:$status"; exit 5; }
    cd "$ROOT"
    exec /dsk2/user/dengsc/miniconda3/envs/cc_hhgt/bin/python \
      scripts/close_and_promote_single_cell_hhgt_149.py \
      --source-root "$SOURCE" \
      --graph-root "$GRAPH" \
      --nested-root "$NESTED" \
      --episode-root "$EPISODES" \
      --training-root "$TRAINING" \
      --audit-root "$AUDIT" \
      --pointer "$POINTER" \
      --version-log "$VERSION_LOG" \
      --expected-source-validation-sha256 "$EXPECTED_SOURCE" \
      --expected-old-pointer-sha256 "$EXPECTED_POINTER" \
      --promote
  fi
  printf '%s waiting_for_independent_validation\n' "$(date --iso-8601=seconds)"
  sleep 300
done

echo "CLOSURE_RELAY_DEADLINE_EXPIRED"
exit 6
