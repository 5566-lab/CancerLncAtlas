#!/usr/bin/env bash
# Explicit server-only launcher.  No computation occurs without both a
# passing, SHA-pinned preflight and the diagnostic-only acknowledgement token.
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <config.json> <PREFLIGHT.json> <preflight-sha256>" >&2
  exit 2
fi
if [[ "${V32_SC_DIAGNOSTIC_EXECUTE:-}" != "YES_DIAGNOSTIC_ONLY" ]]; then
  echo "Set V32_SC_DIAGNOSTIC_EXECUTE=YES_DIAGNOSTIC_ONLY to execute on the server" >&2
  exit 61
fi

config=$1
preflight=$2
preflight_sha=$3
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
for cancer in "${cancers[@]}"; do
  bash "${workspace}/scripts/run_sc_trajectory_one.sh" "${cancer}"
done

"${python}" "${workspace}/python/61_sc_trajectory_pathway_stats.py" "${cancers[@]}"
"${python}" "${workspace}/python/63_sc_lnc_pathway_association_ssgsea.py" "${cancers[@]}"

workspace_manifest="${workspace%/workspace}/WORKSPACE_MANIFEST.json"
workspace_sha=$(sha256sum "${workspace_manifest}" | awk '{print $1}')
"${python}" -m cc_hhgt.v32.single_cell_diagnostic_rerun finalize \
  --config "${config}" \
  --expected-workspace-manifest-sha256 "${workspace_sha}"

echo "V3.2 diagnostic rerun complete; model-fusion weight remains zero"
