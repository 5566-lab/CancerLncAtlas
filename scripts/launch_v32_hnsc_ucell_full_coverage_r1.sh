#!/usr/bin/env bash
set -euo pipefail

runtime=./data/CancerLncAtlas/runtime/code_v32_single_cell_20260826_r10_hnsc_ucell_full_coverage
python=${PRIVATE_WORK_ROOT}/miniconda3/bin/python
runner=${runtime}/scripts/run_v32_single_cell_cell_level_pilot.py
output=./data/CancerLncAtlas/results/model/v32_full_multitask/single_cell_cell_level_hnsc_ucell_pilot_20260826_r1
log=${runtime}/HNSC_FULL_COVERAGE_R1.stdout.log
pidfile=${runtime}/HNSC_FULL_COVERAGE_R1.pid

if [[ -e "${output}" ]]; then
  echo "Refusing output reuse: ${output}" >&2
  exit 41
fi
if [[ -e "${log}" || -e "${pidfile}" ]]; then
  echo "Refusing launcher state reuse" >&2
  exit 42
fi

nohup "${python}" "${runner}" \
  --mode full \
  --preflight-json ./data/CancerLncAtlas/results/model/v32_full_multitask/single_cell_cell_level_hnsc_preflight_20260826_r2_complete_signature/PREFLIGHT.json \
  --expected-preflight-sha256 02faa826366f5eba4b7ee610c1aac320205da116dc71cf71f6d93f46420a077a \
  --official-parity-json ./data/CancerLncAtlas/runtime/code_v32_single_cell_20260826_r10_hnsc_ucell_full_coverage/references/official_ucell/PARITY.json \
  --expected-official-parity-sha256 a1d33d7e3713f4cacb9ec7aa7aeab94cc518e586c8610163ed9799d59717880f \
  --small-parity-json ./data/CancerLncAtlas/results/model/v32_full_multitask/single_cell_cell_level_hnsc_ucell_small_20260826_r1/SMALL_CHUNK_PARITY.json \
  --expected-small-parity-sha256 aa0aa293d1cf01644958ea7d8c399f9d1cf1e8603899ee15be492a87c7634d8b \
  --output-root "${output}" \
  --max-rank 1500 \
  --chunk-size 64 \
  >"${log}" 2>&1 < /dev/null &

pid=$!
printf '%s\n' "${pid}" > "${pidfile}"
if ! kill -0 "${pid}" 2>/dev/null; then
  echo "Full HNSC process exited during launch; inspect ${log}" >&2
  exit 43
fi
echo "STARTED_PID=${pid}"
echo "LOG=${log}"
echo "OUTPUT=${output}"
