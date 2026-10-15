#!/usr/bin/env bash
set -euo pipefail

# Prepare fresh G0/G1/G2 fold payloads, including the fitted L1 baseline.
# No neural training is launched. Never run this wrapper for a code-only check.
# Patient authority validation refuses legacy/synthetic fold manifests.
test "$(hostname)" = 149 || { echo "FORMAL_PREPARATION_REQUIRES_HOST_149" >&2; exit 20; }
project_root=/dell_2/DSC/CancerLncAtlas
code_root="${V32_CODE_ROOT:-$project_root/runtime/CC_HHGT_v3_2_ranked_subtypes_dev}"
python_bin="${V32_PYTHON_BIN:-/dsk2/user/dengsc/miniconda3/envs/cc_hhgt/bin/python}"
authority_root="${V32_PATIENT_FOLD_AUTHORITY_ROOT:-$project_root/inputs/v32_routed_candidate_20260829_r1/patient_fold_authority_20260829_r1}"
patient_folds="${V32_PATIENT_FOLDS:-$authority_root/SAMPLE_PATIENT_FOLD_MAP.tsv}"
patient_fold_receipt="${V32_PATIENT_FOLD_AUTHORITY_RECEIPT:-$authority_root/PATIENT_FOLD_AUTHORITY_RECEIPT.json}"
hashes="${V32_G012_INPUT_HASHES:?V32_G012_INPUT_HASHES must name the frozen graph/input hash JSON}"
graph_receipt="${V32_G012_GRAPH_AUTHORITY_RECEIPT:?V32_G012_GRAPH_AUTHORITY_RECEIPT is required}"
graph_receipt_sha256="${V32_G012_GRAPH_AUTHORITY_RECEIPT_SHA256:?V32_G012_GRAPH_AUTHORITY_RECEIPT_SHA256 is required}"
graph_detection="${V32_G012_GRAPH_DETECTION:?V32_G012_GRAPH_DETECTION is required}"
graph_detection_sha256="${V32_G012_GRAPH_DETECTION_SHA256:?V32_G012_GRAPH_DETECTION_SHA256 is required}"
graph_membership="${V32_G012_GRAPH_SIGNED_MEMBERSHIP:?V32_G012_GRAPH_SIGNED_MEMBERSHIP is required}"
graph_membership_sha256="${V32_G012_GRAPH_SIGNED_MEMBERSHIP_SHA256:?V32_G012_GRAPH_SIGNED_MEMBERSHIP_SHA256 is required}"
graph_hierarchy="${V32_G012_GRAPH_PATHWAY_HIERARCHY:?V32_G012_GRAPH_PATHWAY_HIERARCHY is required}"
graph_hierarchy_sha256="${V32_G012_GRAPH_PATHWAY_HIERARCHY_SHA256:?V32_G012_GRAPH_PATHWAY_HIERARCHY_SHA256 is required}"
graph_binding="${V32_G012_GRAPH_LNC_PROTEIN_BINDING:?V32_G012_GRAPH_LNC_PROTEIN_BINDING is required}"
graph_binding_sha256="${V32_G012_GRAPH_LNC_PROTEIN_BINDING_SHA256:?V32_G012_GRAPH_LNC_PROTEIN_BINDING_SHA256 is required}"
graph_encoding="${V32_G012_GRAPH_PROTEIN_GENE_ENCODING:?V32_G012_GRAPH_PROTEIN_GENE_ENCODING is required}"
graph_encoding_sha256="${V32_G012_GRAPH_PROTEIN_GENE_ENCODING_SHA256:?V32_G012_GRAPH_PROTEIN_GENE_ENCODING_SHA256 is required}"
graph_ppi="${V32_G012_GRAPH_PPI:?V32_G012_GRAPH_PPI is required}"
graph_ppi_sha256="${V32_G012_GRAPH_PPI_SHA256:?V32_G012_GRAPH_PPI_SHA256 is required}"
graph_fold_expression_pattern="${V32_G012_GRAPH_FOLD_EXPRESSION_PATTERN:?fold expression pattern is required}"
graph_fold_coexpression_pattern="${V32_G012_GRAPH_FOLD_COEXPRESSION_PATTERN:?fold coexpression pattern is required}"
formal_activity_root="${V32_G012_FORMAL_ACTIVITY_ROOT:?formal activity root is required}"
association_mode="${V32_G012_ASSOCIATION_MODE:-local-cnv-adjusted}"
formal_candidates="${V32_G012_FORMAL_CANDIDATES:?formal candidate authority is required}"
formal_candidates_sha256="${V32_G012_FORMAL_CANDIDATES_SHA256:?formal candidate authority SHA256 is required}"
local_cnv_audit_root="${V32_G012_LOCAL_CNV_AUDIT_ROOT:?local-CNV audit root is required}"
local_cnv_audit_success_sha256="${V32_G012_LOCAL_CNV_AUDIT_SUCCESS_SHA256:?local-CNV audit SUCCESS SHA256 is required}"
local_cnv_validation_root="${V32_G012_LOCAL_CNV_VALIDATION_ROOT:?validation-A1 root is required}"
local_cnv_validation_success_sha256="${V32_G012_LOCAL_CNV_VALIDATION_SUCCESS_SHA256:?validation-A1 SUCCESS SHA256 is required}"
output_root="${V32_G012_PREPARED_OUTPUT:-$project_root/inputs/v32_g012_patient_first_20260829_r2/FORMAL_PREPARED_FOLDS}"

test "$(readlink -f "$project_root")" = /dell_2/DSC/CancerLncAtlas
test -x "$python_bin"
test -f "$hashes"
test -d "$formal_activity_root"
test ! -e "$output_root" || { echo "REFUSING_G012_OUTPUT_REUSE=$output_root" >&2; exit 21; }

"$python_bin" "$code_root/scripts/gate_v32_patient_fold_authority.py" \
  --repo-root "$code_root" \
  --patient-folds "$patient_folds" \
  --patient-fold-authority-receipt "$patient_fold_receipt" >/dev/null

"$python_bin" "$code_root/scripts/prepare_v32_formal.py" \
  --hashes "$hashes" \
  --patient-folds "$patient_folds" \
  --patient-fold-authority-receipt "$patient_fold_receipt" \
  --graph-authority-receipt "$graph_receipt" \
  --graph-authority-receipt-sha256 "$graph_receipt_sha256" \
  --graph-detection "$graph_detection" \
  --graph-detection-sha256 "$graph_detection_sha256" \
  --graph-signed-membership "$graph_membership" \
  --graph-signed-membership-sha256 "$graph_membership_sha256" \
  --graph-pathway-hierarchy "$graph_hierarchy" \
  --graph-pathway-hierarchy-sha256 "$graph_hierarchy_sha256" \
  --graph-lnc-protein-binding "$graph_binding" \
  --graph-lnc-protein-binding-sha256 "$graph_binding_sha256" \
  --graph-protein-gene-encoding "$graph_encoding" \
  --graph-protein-gene-encoding-sha256 "$graph_encoding_sha256" \
  --graph-ppi "$graph_ppi" \
  --graph-ppi-sha256 "$graph_ppi_sha256" \
  --graph-fold-expression-pattern "$graph_fold_expression_pattern" \
  --graph-fold-coexpression-pattern "$graph_fold_coexpression_pattern" \
  --formal-activity-root "$formal_activity_root" \
  --association-mode "$association_mode" \
  --formal-candidates "$formal_candidates" \
  --formal-candidates-sha256 "$formal_candidates_sha256" \
  --local-cnv-audit-root "$local_cnv_audit_root" \
  --local-cnv-audit-success-sha256 "$local_cnv_audit_success_sha256" \
  --local-cnv-validation-root "$local_cnv_validation_root" \
  --local-cnv-validation-success-sha256 "$local_cnv_validation_success_sha256" \
  --output "$output_root" \
  --seed 20260726 \
  --budget-per-cancer 100000 \
  --start-fold 0 \
  --fold-count 5

test -f "$output_root/PATIENT_FOLD_BINDING.json"
test ! -e "$output_root/PATIENT_FOLD_4.pt"
test -f "$output_root/G0/PATIENT_FOLD_4.pt"
test -f "$output_root/G1/PATIENT_FOLD_4.pt"
test -f "$output_root/G2/PATIENT_FOLD_4.pt"
echo "G012_PATIENT_FIRST_PREPARATION_COMPLETE_BASELINE_FITTED_NO_NEURAL_TRAINING=$output_root"
