#!/usr/bin/env bash
# Hash-bound server launcher for the fresh V3.2 single-cell diagnostic run.
# Independent cancer stages may run concurrently; final aggregation remains
# fail-closed and starts only after every stage succeeds.
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 <config.json> <PREFLIGHT.json> <preflight-sha256> <workers>" >&2
  exit 2
fi
if [[ "${V32_SC_DIAGNOSTIC_EXECUTE:-}" != "YES_DIAGNOSTIC_ONLY" ]]; then
  echo "Set V32_SC_DIAGNOSTIC_EXECUTE=YES_DIAGNOSTIC_ONLY to execute on the server" >&2
  exit 61
fi

config=$1
preflight=$2
preflight_sha=$3
workers=$4
if [[ ! "$workers" =~ ^[1-3]$ ]]; then
  echo "workers must be 1, 2, or 3" >&2
  exit 66
fi

python=${V32_PYTHON:?V32_PYTHON must name the pinned server Python}
code_root=${V32_CODE_ROOT:?V32_CODE_ROOT must name the uploaded V3.2 code root}
configured_python=$("${python}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["runtime"]["python_executable"])' "${config}")
if [[ "$(readlink -f "${python}")" != "$(readlink -f "${configured_python}")" ]]; then
  echo "V32_PYTHON differs from the config-pinned Python" >&2
  exit 63
fi
rscript_path=$("${python}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["runtime_dependencies"]["rscript_executable"])' "${preflight}")
if [[ ! -x "${rscript_path}" ]]; then
  echo "Preflight-pinned Rscript is no longer executable: ${rscript_path}" >&2
  exit 64
fi
export PATH="$(dirname "${rscript_path}"):${PATH}"
r_library_paths=$("${python}" -c 'import json,sys; print(":".join(json.load(open(sys.argv[1]))["runtime_dependencies"]["r_library_paths"]))' "${preflight}")
if [[ -z "${r_library_paths}" ]]; then
  echo "Preflight did not pin any R library paths" >&2
  exit 65
fi
export R_LIBS_USER="${r_library_paths}"

cd "${code_root}"
"${python}" -m cc_hhgt.v32.single_cell_diagnostic_rerun prepare \
  --config "${config}" \
  --preflight "${preflight}" \
  --expected-preflight-sha256 "${preflight_sha}"

workspace=$("${python}" -c 'import json,sys; c=json.load(open(sys.argv[1])); print(c["compute_output_root"] + "/workspace")' "${config}")
mapfile -t cancers < <("${python}" -c 'import json,sys; p=json.load(open(sys.argv[1])); print("\n".join(p["run_cancers"]))' "${preflight}")
if [[ ${#cancers[@]} -eq 0 ]]; then
  echo "Passing preflight contains no runnable cancers" >&2
  exit 62
fi

export CANCERLNCATLAS_ROOT="${workspace}"
printf '%s\0' "${cancers[@]}" | \
  xargs -0 -n1 -P "${workers}" bash "${workspace}/scripts/run_sc_trajectory_one.sh"

for cancer in "${cancers[@]}"; do
  qa="${workspace}/results/sc_trajectory_staging/${cancer}/dorothea_signed_intersection_qa.tsv"
  if [[ ! -s "${qa}" ]] || ! tail -n 1 "${qa}" | grep -q $'\tPASS$'; then
    echo "${cancer} lacks a passing DoRothEA signed-intersection audit" >&2
    exit 67
  fi
done

"${python}" "${workspace}/python/61_sc_trajectory_pathway_stats.py" "${cancers[@]}"
"${python}" "${workspace}/python/63_sc_lnc_pathway_association_ssgsea.py" "${cancers[@]}"

workspace_manifest="${workspace%/workspace}/WORKSPACE_MANIFEST.json"
workspace_sha=$(sha256sum "${workspace_manifest}" | awk '{print $1}')
"${python}" -m cc_hhgt.v32.single_cell_diagnostic_rerun finalize \
  --config "${config}" \
  --expected-workspace-manifest-sha256 "${workspace_sha}"

sha256sum "${workspace%/workspace}/V32_DIAGNOSTIC_MANIFEST.json"
echo "V3.2 diagnostic rerun complete; model-fusion weight remains zero"
