#!/usr/bin/env bash
# Read-only, fail-closed preflight for the 17 formal V3.2 single-cell cancers.
#
# This does NOT launch UCell computation or retraining.  It validates each raw
# H5, cell metadata, annotation, exact-pathway membership and barcode identity,
# and writes only small preflight JSON files under the declared DSC result root.
set -euo pipefail

runtime=./data/CancerLncAtlas/runtime/code_v32_single_cell_20260826_r10_hnsc_ucell_full_coverage
python=${PRIVATE_WORK_ROOT}/miniconda3/bin/python
runner=${runtime}/scripts/preflight_v32_single_cell_cell_level.py
r6_root=./data/CancerLncAtlas/results/model/v32_full_multitask/single_cell_h5_33c_partitions_20260826_r6_semantic_correction
output=./data/CancerLncAtlas/results/model/v32_full_multitask/single_cell_cell_level_17c_preflight_20260826_r1

run_status_sha=3a9ec28a6168f1e38d65521d9e2d7fff818becef6f6d74a3dd9846c1715c0423
handoff_sha=627fbe52df38e58692b4091d9d1ab870da6629c6eec56f9c6eb7aa5c11c03ad4
manifest_sha=bea2d20a261063dacdd05b3b18865f8de0fec3504086e08e6295b63fe0a4ffad

cancers=(ACC CHOL DLBC ESCA GBM HNSC KIRC LAML LGG LUSC MESO PCPG READ SARC SKCM THYM UCEC)

if [[ ! -x "${python}" || ! -f "${runner}" ]]; then
  echo "Missing pinned Python or preflight runner" >&2
  exit 51
fi
if [[ ! -d "${r6_root}" ]]; then
  echo "Missing immutable r6 authority: ${r6_root}" >&2
  exit 52
fi
if [[ -e "${output}" ]]; then
  echo "Refusing output reuse: ${output}" >&2
  exit 53
fi

mkdir -p "${output}"
for cancer in "${cancers[@]}"; do
  "${python}" "${runner}" \
    --r6-root "${r6_root}" \
    --cancer "${cancer}" \
    --expected-run-status-sha256 "${run_status_sha}" \
    --expected-handoff-sha256 "${handoff_sha}" \
    --expected-manifest-sha256 "${manifest_sha}" \
    --max-rank 1500 \
    --chunk-size 64 \
    --output-json "${output}/${cancer}.PREFLIGHT.json" \
    >"${output}/${cancer}.stdout.json"
done

count=$(find "${output}" -maxdepth 1 -type f -name '*.PREFLIGHT.json' | wc -l)
if [[ "${count}" -ne 17 ]]; then
  echo "Expected 17 completed preflights, observed ${count}" >&2
  exit 54
fi

printf '%s\n' \
  '{' \
  '  "status": "PASS_17_FORMAL_CANCER_PREFLIGHTS",' \
  '  "cell_level_ucell_started": false,' \
  '  "retraining_started": false,' \
  '  "historical_sc_trajectory_used": false,' \
  '  "next_gate": "GENERALIZE_AND_AUDIT_NON_HNSC_CELL_LEVEL_RUNNER"' \
  '}' >"${output}/BATCH_SUCCESS.json"

echo "PREFLIGHT_ONLY_COMPLETE=${output}"
