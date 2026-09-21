#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
CONFIG="${1:-${ROOT}/config/model_v2.yaml}"
MODE="${2:-full}"
PYTHON="${PYTHON:-python}"

run() {
  echo "[$(date '+%F %T')] $*"
  "$@"
}

prepare() {
  run "$PYTHON" "$ROOT/scripts/00_audit_model_inputs.py" --config "$CONFIG"
  run "$PYTHON" "$ROOT/scripts/01_standardize_model_inputs.py" --config "$CONFIG"
  run "$PYTHON" "$ROOT/scripts/01b_standardize_external_validation.py" --config "$CONFIG"
  run "$PYTHON" "$ROOT/scripts/02_build_id_crosswalks.py" --config "$CONFIG"
  run "$PYTHON" "$ROOT/scripts/03_build_lnc_gene_coexpression.py" --config "$CONFIG"
  run "$PYTHON" "$ROOT/scripts/04_build_pathway_state_edges.py" --config "$CONFIG"
  run "$PYTHON" "$ROOT/scripts/07_build_ucell_support.py" --config "$CONFIG"
  run "$PYTHON" "$ROOT/scripts/05_build_pathway_similarity.py" --config "$CONFIG"
  run "$PYTHON" "$ROOT/scripts/06_build_pathway_families.py" --config "$CONFIG"
  run "$PYTHON" "$ROOT/scripts/08_build_pair_evidence.py" --config "$CONFIG"
  run "$PYTHON" "$ROOT/scripts/09_build_candidate_universe.py" --config "$CONFIG"
  run "$PYTHON" "$ROOT/scripts/10_build_fold_specific_graphs.py" --config "$CONFIG"
}

case "$MODE" in
  prepare)
    prepare
    ;;
  baseline)
    run "$PYTHON" "$ROOT/scripts/11_train_evidence_baselines.py" --config "$CONFIG"
    run "$PYTHON" "$ROOT/scripts/16_calibrate_predictions.py" --config "$CONFIG" --models logistic hist_gradient_boosting
    run "$PYTHON" "$ROOT/scripts/17_score_all_candidates.py" --config "$CONFIG"
    run "$PYTHON" "$ROOT/scripts/23_run_external_validation.py" --config "$CONFIG"
    run "$PYTHON" "$ROOT/scripts/18_generate_explanations.py" --config "$CONFIG"
    run "$PYTHON" "$ROOT/scripts/19_materialize_model_genesets.py" --config "$CONFIG"
    run "$PYTHON" "$ROOT/scripts/20_export_model_report.py" --config "$CONFIG"
    ;;
  full)
    prepare
    run "$PYTHON" "$ROOT/scripts/15_run_loco_evaluation.py" --config "$CONFIG" --models logistic hist_gradient_boosting rgcn hgt cc_hhgt
    run "$PYTHON" "$ROOT/scripts/16_calibrate_predictions.py" --config "$CONFIG"
    run "$PYTHON" "$ROOT/scripts/17_score_all_candidates.py" --config "$CONFIG"
    run "$PYTHON" "$ROOT/scripts/23_run_external_validation.py" --config "$CONFIG"
    run "$PYTHON" "$ROOT/scripts/18_generate_explanations.py" --config "$CONFIG"
    run "$PYTHON" "$ROOT/scripts/19_materialize_model_genesets.py" --config "$CONFIG"
    run "$PYTHON" "$ROOT/scripts/20_export_model_report.py" --config "$CONFIG"
    ;;
  report)
    run "$PYTHON" "$ROOT/scripts/20_export_model_report.py" --config "$CONFIG"
    ;;
  external)
    run "$PYTHON" "$ROOT/scripts/01b_standardize_external_validation.py" --config "$CONFIG"
    run "$PYTHON" "$ROOT/scripts/23_run_external_validation.py" --config "$CONFIG"
    run "$PYTHON" "$ROOT/scripts/20_export_model_report.py" --config "$CONFIG"
    ;;
  *)
    echo "Usage: $0 [config.yaml] {prepare|baseline|full|external|report}" >&2
    exit 2
    ;;
esac
