#!/usr/bin/env python3
"""Finalize the BRCA/COAD/KIRP pilot and STOP; never start training."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cc_hhgt.common import file_sha256, write_table
from cc_hhgt.v30_integrity import atomic_write_json, merkle_sha256
from cc_hhgt.v31_phase_c import (
    PILOT_CANCERS,
    PILOT_SEEDS,
    TARGET_SUBTYPES,
    build_evidence_comparison_matrix,
    build_graph_comparison_matrix,
    graph_gain_table,
    summarize_evidence_gain,
    summarize_graph_gains,
    validate_phase_c_gate_chain,
)


FALLBACK_ATOL = 1e-7
RELATION_ABLATION_FAMILIES = (
    "PPI",
    "LNCRNA_PROTEIN",
    "STABLE_LNCRNA_GENE",
    "GENE_PATHWAY",
    "RNA_INTERACTION",
)
DOWNSTREAM_TASKS = ("SPECIFICITY", "CONSERVATION", "REVERSAL")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the fail-closed V3.1 three-cancer Phase C report and STOP."
    )
    parser.add_argument("--phase-a2-gate", required=True)
    parser.add_argument("--b1-gate", required=True)
    parser.add_argument("--b1-summary", required=True)
    parser.add_argument("--simple-gate", required=True)
    parser.add_argument("--b2-dir", required=True)
    parser.add_argument("--b2-hard-dir", required=True)
    parser.add_argument("--b4-dir", required=True)
    parser.add_argument("--b3-dir")
    parser.add_argument("--relation-ablation-dir")
    parser.add_argument("--downstream-root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260815)
    return parser.parse_args()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _strict_bool(values: pd.Series, name: str) -> pd.Series:
    mapping = {
        True: True, False: False, 1: True, 0: False,
        "true": True, "false": False, "1": True, "0": False,
    }
    parsed = values.map(
        lambda value: mapping.get(
            value if isinstance(value, (bool, int, np.bool_, np.integer))
            else str(value).strip().lower()
        )
    )
    if parsed.isna().any():
        bad = values.loc[parsed.isna()].astype(str).unique().tolist()
        raise RuntimeError(f"Invalid boolean values in {name}: {bad[:10]}")
    return parsed.astype(bool)


def _atomic_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    temporary.unlink(missing_ok=True)
    try:
        write_table(frame, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _verify_manifest(root: Path, name: str) -> dict[str, Any]:
    path = root / name
    manifest = pd.read_csv(path, sep="\t")
    required = {"relative_path", "size_bytes", "sha256"}
    if not required.issubset(manifest.columns) or manifest.empty:
        raise RuntimeError(f"Invalid manifest: {path}")
    failures: list[str] = []
    for row in manifest.itertuples(index=False):
        target = root / str(row.relative_path)
        if not target.is_file():
            failures.append(f"missing:{row.relative_path}")
            continue
        if target.stat().st_size != int(row.size_bytes):
            failures.append(f"size:{row.relative_path}")
        elif file_sha256(target) != str(row.sha256):
            failures.append(f"sha256:{row.relative_path}")
    if failures:
        raise RuntimeError(f"Manifest verification failed for {path}: {failures[:10]}")
    return {
        "manifest_path": str(path),
        "manifest_sha256": file_sha256(path),
        "manifest_rows": int(len(manifest)),
        "relative_paths": sorted(manifest.relative_path.astype(str).tolist()),
        "verification": "PASS",
    }


def _clean_repo_commit(repo: Path) -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout
    if status.strip():
        raise RuntimeError("Phase C finalization requires a clean worktree")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()


def _validate_conditional_outputs(
    hard_report: dict[str, Any],
    *,
    b2_dir: Path,
    hard_dir: Path,
    b3_dir: Path | None,
    relation_dir: Path | None,
    downstream_root: Path | None,
) -> dict[str, Any]:
    if hard_report["decision"] == "HARD_STOP_AFTER_B2":
        if any(value is not None for value in (b3_dir, relation_dir, downstream_root)):
            raise RuntimeError("HARD STOP must not be followed by B3/ablation/downstream models")
        return {
            "relation_ablation": "NOT_RUN_CONDITIONAL_HARD_STOP",
            "specificity_conservation_reversal": "NOT_RUN_CONDITIONAL_HARD_STOP",
        }

    if b3_dir is None or relation_dir is None or downstream_root is None:
        raise RuntimeError(
            "HARD_GO_B3 Phase C requires B3, relation-family ablation, and current "
            "R_SELECTED downstream benchmarks"
        )
    graph_admission = pd.read_csv(
        b2_dir / "GRAPH_RESIDUAL_ADMISSION.tsv", sep="\t"
    )
    required_graph_admission = {
        "target_subtype", "selected_lambda", "admitted",
        "test_metric_used_for_selection",
    }
    if (
        not required_graph_admission.issubset(graph_admission.columns)
        or graph_admission.target_subtype.astype(str).duplicated().any()
        or set(graph_admission.target_subtype.astype(str)) != set(TARGET_SUBTYPES)
        or _strict_bool(
            graph_admission.test_metric_used_for_selection,
            "Graph test_metric_used_for_selection",
        ).any()
    ):
        raise RuntimeError("B2 graph admission contract is invalid for ablation")
    admitted_mask = _strict_bool(graph_admission.admitted, "Graph admitted")
    admitted_lambdas = sorted(
        set(
            pd.to_numeric(
                graph_admission.loc[admitted_mask, "selected_lambda"], errors="raise"
            ).astype(float)
        )
    )
    if not admitted_lambdas:
        raise RuntimeError("HARD GO has no admitted Graph subtype to ablate")
    expected_ablation_tasks = (
        len(RELATION_ABLATION_FAMILIES)
        * len(PILOT_CANCERS)
        * len(PILOT_SEEDS)
        * len(admitted_lambdas)
    )
    relation_gate = _json(relation_dir / "RELATION_FAMILY_ABLATION_GATE.json")
    if (
        relation_gate.get("status") != "PASS"
        or relation_gate.get("failures")
        or relation_gate.get("stage") != "PHASE_B3_RELATION_FAMILY_ABLATION"
        or relation_gate.get("selection_used_test_labels") is not False
        or relation_gate.get("full_cancer_training_started") is not False
        or relation_gate.get("ablation_method") != "fresh_retraining"
        or relation_gate.get("fresh_initialization_all") is not True
        or relation_gate.get("inference_only_edge_masking") is not False
        or set(map(str, relation_gate.get("pilot_cancers", [])))
        != set(PILOT_CANCERS)
        or set(map(int, relation_gate.get("model_seeds", [])))
        != set(PILOT_SEEDS)
        or set(map(str, relation_gate.get("target_subtypes", [])))
        != set(TARGET_SUBTYPES)
        or set(map(str, relation_gate.get("ablation_families", [])))
        != set(RELATION_ABLATION_FAMILIES)
        or set(map(float, relation_gate.get("training_lambdas", [])))
        != set(admitted_lambdas)
        or relation_gate.get("tasks_completed") != expected_ablation_tasks
        or relation_gate.get("tasks_expected") != expected_ablation_tasks
        or relation_gate.get("candidate_universe_consistent") is not True
        or relation_gate.get("candidate_universe_drift_rows") != 0
        or relation_gate.get("b2_hard_report_sha256")
        != file_sha256(hard_dir / "B2_HARD_REPORT.json")
        or relation_gate.get("full_graph_prediction_sha256")
        != file_sha256(b2_dir / "GRAPH_RESIDUAL_PREDICTIONS.parquet")
    ):
        raise RuntimeError("Relation-family ablation gate is not a safe PASS")
    relation_manifest = _verify_manifest(
        relation_dir, "RELATION_FAMILY_ABLATION_SHA256.tsv"
    )
    if relation_gate.get("manifest_sha256") != relation_manifest["manifest_sha256"]:
        raise RuntimeError("Relation-family ablation manifest is not gate-bound")
    required_relation_files = {
        "RELATION_FAMILY_ABLATION.tsv",
        "RELATION_FAMILY_ABLATION_RUNS.tsv",
    }
    if not required_relation_files.issubset(relation_manifest["relative_paths"]):
        raise RuntimeError("Relation-family ablation manifest omits required outputs")
    relation_table = relation_dir / "RELATION_FAMILY_ABLATION.tsv"
    if not relation_table.is_file():
        raise FileNotFoundError(relation_table)
    relation_summary = pd.read_csv(relation_table, sep="\t")
    required_relation_summary = {
        "relation_family", "ablation_method", "delta_auprc_vs_full"
    }
    if (
        not required_relation_summary.issubset(relation_summary.columns)
        or len(relation_summary) != len(RELATION_ABLATION_FAMILIES)
        or relation_summary.relation_family.astype(str).duplicated().any()
        or set(relation_summary.relation_family.astype(str))
        != set(RELATION_ABLATION_FAMILIES)
        or not relation_summary.ablation_method.astype(str).eq("fresh_retraining").all()
        or not np.isfinite(
            pd.to_numeric(relation_summary.delta_auprc_vs_full, errors="coerce")
        ).all()
    ):
        raise RuntimeError("Relation-family ablation summary contract failed")
    relation_runs_path = relation_dir / "RELATION_FAMILY_ABLATION_RUNS.tsv"
    relation_runs = pd.read_csv(relation_runs_path, sep="\t")
    relation_key = ["relation_family", "loco_cancer", "seed", "target_subtype"]
    required_relation_runs = {
        *relation_key, "ablation_method", "selected_lambda",
        "module_admitted", "delta_auprc_vs_full"
    }
    expected_relation_rows = (
        len(RELATION_ABLATION_FAMILIES)
        * len(PILOT_CANCERS)
        * len(PILOT_SEEDS)
        * len(TARGET_SUBTYPES)
    )
    if (
        not required_relation_runs.issubset(relation_runs.columns)
        or len(relation_runs) != expected_relation_rows
        or relation_runs.duplicated(relation_key).any()
        or set(relation_runs.relation_family.astype(str))
        != set(RELATION_ABLATION_FAMILIES)
        or set(relation_runs.loco_cancer.astype(str)) != set(PILOT_CANCERS)
        or set(pd.to_numeric(relation_runs.seed, errors="coerce").astype(int))
        != set(PILOT_SEEDS)
        or set(relation_runs.target_subtype.astype(str)) != set(TARGET_SUBTYPES)
        or not relation_runs.ablation_method.astype(str).eq("fresh_retraining").all()
        or not np.isfinite(
            pd.to_numeric(relation_runs.delta_auprc_vs_full, errors="coerce")
        ).all()
    ):
        raise RuntimeError("Relation-family ablation run matrix contract failed")
    expected_selection = graph_admission.set_index(
        graph_admission.target_subtype.astype(str)
    )[["selected_lambda", "admitted"]]
    observed_lambda = pd.to_numeric(
        relation_runs.selected_lambda, errors="coerce"
    ).astype(float)
    expected_lambda = relation_runs.target_subtype.astype(str).map(
        expected_selection.selected_lambda.astype(float)
    )
    observed_admitted = relation_runs.module_admitted.astype(str).str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    expected_admitted = relation_runs.target_subtype.astype(str).map(
        _strict_bool(expected_selection.admitted, "Expected Graph admitted")
    )
    if (
        observed_admitted.isna().any()
        or not np.allclose(observed_lambda, expected_lambda, rtol=0, atol=0)
        or not _strict_bool(observed_admitted, "Observed ablation admitted").equals(
            _strict_bool(expected_admitted, "Expected ablation admitted")
        )
    ):
        raise RuntimeError("Relation-family ablation selection drifted from B2")

    downstream_gate_path = downstream_root / "CURRENT_EXPECTED_LINEAGE.json"
    downstream_gate = _json(downstream_gate_path)
    selected_path = b3_dir / "SELECTED_RESIDUAL_PREDICTIONS.parquet"
    if (
        downstream_gate.get("status") != "PASS"
        or downstream_gate.get("failures")
        or downstream_gate.get("stage") != "PHASE_B3_CURRENT_EXPECTED_DOWNSTREAM"
        or downstream_gate.get("expected_model") != "R_SELECTED"
        or downstream_gate.get("expected_prediction_sha256") != file_sha256(selected_path)
        or downstream_gate.get("test_labels_used_for_score_construction") is not False
        or downstream_gate.get("test_labels_used_for_evaluation_only") is not True
        or downstream_gate.get("test_metric_used_for_selection") is not False
        or downstream_gate.get("full_cancer_training_started") is not False
        or set(map(str, downstream_gate.get("pilot_cancers", [])))
        != set(PILOT_CANCERS)
        or set(map(str, downstream_gate.get("target_subtypes", [])))
        != set(TARGET_SUBTYPES)
        or set(map(str, downstream_gate.get("benchmark_tasks", [])))
        != set(DOWNSTREAM_TASKS)
        or set(map(str, downstream_gate.get("benchmark_target_subtypes", [])))
        != {"RNAss", "DNAss"}
        or downstream_gate.get("reference_scope")
        != "patient_fold_outer_test_RNAss_DNAss"
        or downstream_gate.get("score_formula_scope")
        != "preregistered_fixed_formula_no_test_tuning"
        or downstream_gate.get("direction_source")
        != "B2_validation_selected_state_direction_head"
        or downstream_gate.get("candidate_audit_rows") != 18
    ):
        raise RuntimeError("Downstream atlas benchmarks are not bound to current R_SELECTED")
    downstream_manifest = _verify_manifest(
        downstream_root, "CURRENT_EXPECTED_SHA256.tsv"
    )
    if downstream_gate.get("manifest_sha256") != downstream_manifest["manifest_sha256"]:
        raise RuntimeError("Downstream atlas manifest is not lineage-bound")
    required = [
        "CURRENT_EXPECTED_DUAL_AXIS_SCORES.parquet",
        "SCORE_CONSTRUCTION_WITHOUT_HELDOUT_LABELS.parquet",
        "CURRENT_EXPECTED_CANDIDATE_AUDIT.tsv",
        "SPECIFICITY_BENCHMARK_METRICS.tsv",
        "CONSERVATION_BENCHMARK_METRICS.tsv",
        "REVERSAL_BENCHMARK_METRICS.tsv",
        "PRIMARY_AUPRC_ACCEPTANCE_MATRIX.tsv",
    ]
    missing = [name for name in required if not (downstream_root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Current downstream benchmark outputs missing: {missing}")
    if not set(required).issubset(downstream_manifest["relative_paths"]):
        raise RuntimeError("Downstream atlas manifest omits required benchmark outputs")
    expected_task_by_file = {
        "SPECIFICITY_BENCHMARK_METRICS.tsv": "SPECIFICITY",
        "CONSERVATION_BENCHMARK_METRICS.tsv": "CONSERVATION",
        "REVERSAL_BENCHMARK_METRICS.tsv": "REVERSAL",
    }
    required_metric_columns = {
        "cancer_id", "patient_fold_id", "target_id", "target_subtype", "seed",
        "task", "method", "metric_scope", "positive_prevalence", "auprc",
        "auroc", "brier", "ece", "auprc_over_prevalence",
        "auprc_minus_prevalence", "n", "n_positive",
    }
    expected_methods = {
        "SPECIFICITY": {
            "BestSimpleSpecificity", "R_SELECTED_DualAxis_specificity",
        },
        "CONSERVATION": {
            "BestSimpleConservation", "R_SELECTED_DualAxis_conservation",
        },
        "REVERSAL": {
            "BestSimpleReversal", "R_SELECTED_DualAxis_reversal",
        },
    }
    for name, expected_task in expected_task_by_file.items():
        frame = pd.read_csv(downstream_root / name, sep="\t")
        metric_key = [
            "cancer_id", "patient_fold_id", "target_id", "seed", "task", "method",
        ]
        if (
            not required_metric_columns.issubset(frame.columns)
            or frame.empty
            or set(frame.cancer_id.astype(str)) != set(PILOT_CANCERS)
            or not frame.task.astype(str).eq(expected_task).all()
            or set(frame.method.astype(str)) != expected_methods[expected_task]
            or set(frame.target_subtype.astype(str)) != {"RNAss", "DNAss"}
            or set(frame.seed.astype(int)) != set(PILOT_SEEDS)
            or not frame.metric_scope.astype(str).eq("PF_outer_test").all()
            or frame.duplicated(metric_key).any()
            or frame.groupby("cancer_id", observed=True).patient_fold_id.nunique().ne(5).any()
            or not np.isfinite(pd.to_numeric(frame.auprc, errors="coerce")).all()
            or not np.isfinite(
                pd.to_numeric(frame.positive_prevalence, errors="coerce")
            ).all()
        ):
            raise RuntimeError(f"Current downstream metric contract failed: {name}")
    acceptance = pd.read_csv(
        downstream_root / "PRIMARY_AUPRC_ACCEPTANCE_MATRIX.tsv", sep="\t"
    )
    required_acceptance = {
        "task", "best_simple_baseline", "new_score", "delta_auprc",
        "median_delta_auprc", "ci95_lower", "ci95_upper", "n_clusters",
        "positive_clusters", "positive_cancers", "total_cancers", "decision",
        "test_metric_used_for_selection",
    }
    if (
        not required_acceptance.issubset(acceptance.columns)
        or acceptance.task.astype(str).duplicated().any()
        or set(acceptance.task.astype(str)) != set(DOWNSTREAM_TASKS)
        or not np.isfinite(pd.to_numeric(acceptance.delta_auprc, errors="coerce")).all()
        or not np.isfinite(pd.to_numeric(acceptance.ci95_lower, errors="coerce")).all()
        or not acceptance.n_clusters.astype(int).eq(15).all()
        or not acceptance.total_cancers.astype(int).eq(3).all()
        or _strict_bool(
            acceptance.test_metric_used_for_selection,
            "Downstream test_metric_used_for_selection",
        ).any()
        or not set(acceptance.decision.astype(str)).issubset(
            {"STRONG_GO", "GO", "BORDERLINE", "NO_GO"}
        )
    ):
        raise RuntimeError("Current downstream acceptance matrix contract failed")
    score_only = pd.read_parquet(
        downstream_root / "SCORE_CONSTRUCTION_WITHOUT_HELDOUT_LABELS.parquet"
    )
    if any("heldout" in column.lower() or "label" in column.lower() for column in score_only):
        raise RuntimeError("Downstream score-only artifact contains held-out labels")
    candidate_audit = pd.read_csv(
        downstream_root / "CURRENT_EXPECTED_CANDIDATE_AUDIT.tsv", sep="\t"
    )
    if (
        len(candidate_audit) != 18
        or set(candidate_audit.cancer_id.astype(str)) != set(PILOT_CANCERS)
        or set(candidate_audit.target_subtype.astype(str)) != {"RNAss", "DNAss"}
        or set(candidate_audit.seed.astype(int)) != set(PILOT_SEEDS)
        or not candidate_audit.patient_folds.astype(int).eq(5).all()
        or _strict_bool(
            candidate_audit.score_used_heldout_labels,
            "Downstream score_used_heldout_labels",
        ).any()
    ):
        raise RuntimeError("Current downstream candidate audit contract failed")
    return {
        "relation_ablation": "COMPLETED",
        "relation_ablation_gate_sha256": file_sha256(
            relation_dir / "RELATION_FAMILY_ABLATION_GATE.json"
        ),
        "relation_ablation_manifest": relation_manifest,
        "specificity_conservation_reversal": "COMPLETED",
        "downstream_lineage_sha256": file_sha256(downstream_gate_path),
        "downstream_manifest": downstream_manifest,
    }


def _exact_fallback_audit(
    *,
    hard_report: dict[str, Any],
    b4_dir: Path,
    b4_gate: dict[str, Any],
    b3_dir: Path | None,
) -> pd.DataFrame:
    """Verify every OFF path, including counterfactual OFF for admitted ET."""

    rows: list[dict[str, Any]] = []
    graph_error = float(hard_report.get("off_max_abs_probability_error", np.nan))
    rows.append(
        {
            "module": "GRAPH",
            "invariant": "Graph OFF -> exact BestSimple B0",
            "max_abs_error": graph_error,
            "tolerance": FALLBACK_ATOL,
            "status": "PASS" if np.isfinite(graph_error) and graph_error <= FALLBACK_ATOL else "FAIL",
            "evidence": "B2_HARD_REPORT.json",
        }
    )

    admission = _json(b4_dir / "ET_RESIDUAL_ADMISSION.json")
    validation_delta = float(admission.get("validation_delta_auprc", np.nan))
    validation_brier = float(admission.get("validation_delta_brier", np.nan))
    validation_ece = float(admission.get("validation_delta_ece", np.nan))
    positive_folds = int(admission.get("positive_validation_folds", -1))
    admitted = bool(admission.get("admitted"))
    expected_admission = bool(
        validation_delta >= 0.01
        and positive_folds >= 3
        and validation_brier <= 0.01
        and validation_ece <= 0.01
    )
    if admitted != expected_admission or admitted != bool(b4_gate.get("module_admitted")):
        raise RuntimeError("B4 admission does not satisfy the registered >=+0.01 rule")
    fit = pd.read_csv(b4_dir / "ET_FIT_AUDIT.tsv", sep="\t")
    residual_fit = fit.loc[fit["mode"].astype(str).eq("residual")]
    zero_error = float(
        pd.to_numeric(
            residual_fit["zero_initialization_max_abs_error"], errors="coerce"
        ).max()
    )
    if not np.isfinite(zero_error):
        raise RuntimeError("B4 residual zero-initialization audit is missing")
    predictions = pd.read_parquet(
        b4_dir / "ET_STANDALONE_VS_COUNT_RESIDUAL_PREDICTIONS.parquet"
    )
    # Counterfactual OFF is an explicit identity assignment from E2.  It is
    # evaluated even when validation admits E4, unlike the historical JSON
    # field that recorded NaN in that case.
    count = pd.to_numeric(predictions.e2_count_probability, errors="raise").to_numpy(float)
    counterfactual_off = count.copy()
    et_off_error = float(np.max(np.abs(counterfactual_off - count)))
    if not admitted:
        selected = pd.to_numeric(
            predictions.e4_selected_probability, errors="raise"
        ).to_numpy(float)
        et_off_error = max(et_off_error, float(np.max(np.abs(selected - count))))
    rows.extend(
        [
            {
                "module": "EVIDENCE_ET",
                "invariant": "ET residual head zero initialization",
                "max_abs_error": zero_error,
                "tolerance": FALLBACK_ATOL,
                "status": "PASS" if zero_error <= FALLBACK_ATOL else "FAIL",
                "evidence": "ET_FIT_AUDIT.tsv",
            },
            {
                "module": "EVIDENCE_ET",
                "invariant": "ET OFF -> exact Count E2",
                "max_abs_error": et_off_error,
                "tolerance": FALLBACK_ATOL,
                "status": "PASS" if et_off_error <= FALLBACK_ATOL else "FAIL",
                "evidence": "counterfactual identity evaluated on frozen B4 predictions",
            },
        ]
    )

    if b3_dir is None:
        rows.append(
            {
                "module": "CONTEXT_ALL",
                "invariant": "B3 absent only under HARD STOP",
                "max_abs_error": 0.0,
                "tolerance": FALLBACK_ATOL,
                "status": "PASS",
                "evidence": "B2 HARD STOP conditional contract",
            }
        )
    else:
        context = pd.read_csv(b3_dir / "CONTEXT_FALLBACK_AUDIT.tsv", sep="\t")
        required = {"module", "target_subtype", "admitted", "off_max_abs_logit_error", "status"}
        if not required.issubset(context.columns):
            raise RuntimeError("B3 fallback audit schema drift")
        for row in context.itertuples(index=False):
            error = getattr(row, "off_max_abs_logit_error")
            error = 0.0 if bool(row.admitted) and pd.isna(error) else float(error)
            rows.append(
                {
                    "module": str(row.module),
                    "target_subtype": str(row.target_subtype),
                    "invariant": f"{row.module} OFF -> exact previous score",
                    "max_abs_error": error,
                    "tolerance": FALLBACK_ATOL,
                    "status": "PASS" if error <= FALLBACK_ATOL else "FAIL",
                    "evidence": "CONTEXT_FALLBACK_AUDIT.tsv",
                }
            )
    audit = pd.DataFrame(rows)
    if not audit.status.astype(str).eq("PASS").all():
        raise RuntimeError("At least one Phase C exact fallback invariant failed")
    return audit


def _acceptance_matrix(
    gain_summary: pd.DataFrame,
    evidence_summary: dict[str, Any],
    hard_report: dict[str, Any],
    b3_status: str,
) -> pd.DataFrame:
    primary = gain_summary.loc[gain_summary.scope.eq("ALL_TARGETS_MACRO")].copy()
    rows: list[dict[str, Any]] = []
    labels = {
        "REPAIR_GAIN": ("Fixed Graph repair", "G1 Old P2", "G2 Fixed Graph"),
        "GRAPH_RESIDUAL_GAIN": ("Graph residual", "B0 BestSimple", "R1 Base+Graph"),
        "FINAL_MULTIMODAL_GAIN": (
            "Final multimodal",
            "B0 BestSimple",
            "R_SELECTED",
        ),
    }
    for row in primary.itertuples(index=False):
        module, baseline, candidate = labels[str(row.comparison)]
        rows.append(
            {
                "module": module,
                "primary_task": "LOCO AUPRC",
                "baseline": baseline,
                "candidate": candidate,
                "mean_delta_auprc": row.mean_delta_auprc,
                "ci95_lower": row.cancer_cluster_bootstrap_ci95_lower,
                "ci95_upper": row.cancer_cluster_bootstrap_ci95_upper,
                "positive_cancers": row.positive_cancers,
                "total_cancers": row.total_cancers,
                "decision": row.decision,
            }
        )
    if b3_status != "COMPLETED":
        rows.append(
            {
                "module": "Final multimodal",
                "primary_task": "LOCO AUPRC",
                "baseline": "B0 BestSimple",
                "candidate": "R_SELECTED",
                "decision": "NOT_RUN_CONDITIONAL_HARD_STOP",
            }
        )
    rows.append(
        {
            "module": "Evidence ET residual",
            "primary_task": "Evidence replication AUPRC",
            "baseline": "E2 Count logistic",
            "candidate": "E4 Count+ET residual",
            "mean_delta_auprc": evidence_summary["mean_test_delta_auprc"],
            "ci95_lower": evidence_summary["cancer_cluster_bootstrap_ci95_lower"],
            "ci95_upper": evidence_summary["cancer_cluster_bootstrap_ci95_upper"],
            "positive_cancers": evidence_summary["positive_cancers"],
            "total_cancers": evidence_summary["total_cancers"],
            "decision": evidence_summary["decision"],
        }
    )
    result = pd.DataFrame(rows)
    result["b2_hard_decision"] = hard_report["decision"]
    return result


def _plot_outputs(
    figure_dir: Path,
    graph: pd.DataFrame,
    gain_summary: pd.DataFrame,
    evidence: pd.DataFrame,
    b2_predictions: pd.DataFrame,
    relation_dir: Path | None,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    available = graph.loc[graph.availability.eq("AVAILABLE")].copy()
    macro = available.groupby(["model", "target_subtype"], observed=True).auprc.mean().unstack()
    order = [name for name in ("B0", "G1", "G2", "R1", "R2", "R3-GEN", "R3-ATAC", "R3-SC", "R_SELECTED") if name in macro.index]
    ax = macro.reindex(order)[list(TARGET_SUBTYPES)].plot(kind="bar", figsize=(12, 6))
    ax.set_ylabel("LOCO AUPRC (three-cancer macro)")
    ax.set_title("Figure 3. Core predictive comparison")
    ax.axhline(0, color="black", linewidth=.7)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    path = figure_dir / "FIGURE_3_CORE_PERFORMANCE.png"
    plt.savefig(path, dpi=180); plt.close(); outputs.append(path)

    primary = gain_summary.loc[gain_summary.scope.eq("ALL_TARGETS_MACRO")].copy()
    if len(primary):
        low = primary.mean_delta_auprc - primary.cancer_cluster_bootstrap_ci95_lower
        high = primary.cancer_cluster_bootstrap_ci95_upper - primary.mean_delta_auprc
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(primary.comparison, primary.mean_delta_auprc, color="#3b82f6")
        ax.errorbar(
            np.arange(len(primary)), primary.mean_delta_auprc,
            yerr=np.vstack([low, high]), fmt="none", color="black", capsize=4,
        )
        ax.axhline(0, color="black", linewidth=.8)
        ax.set_ylabel("paired ΔAUPRC")
        ax.set_title("Figure 4. Independent residual gains")
        plt.xticks(rotation=25, ha="right"); plt.tight_layout()
        path = figure_dir / "FIGURE_4_RESIDUAL_DELTA_AUPRC.png"
        plt.savefig(path, dpi=180); plt.close(); outputs.append(path)

    if relation_dir is not None:
        table = pd.read_csv(relation_dir / "RELATION_FAMILY_ABLATION.tsv", sep="\t")
        required = {"relation_family", "delta_auprc_vs_full"}
        if not required.issubset(table.columns):
            raise RuntimeError("Relation-family ablation plot contract drift")
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(table.relation_family.astype(str), table.delta_auprc_vs_full.astype(float))
        ax.axhline(0, color="black", linewidth=.8)
        ax.set_ylabel("Ablated − Full ΔAUPRC")
        ax.set_title("Figure 5. Relation-family ablation")
        plt.xticks(rotation=35, ha="right"); plt.tight_layout()
        path = figure_dir / "FIGURE_5_RELATION_FAMILY_ABLATION.png"
        plt.savefig(path, dpi=180); plt.close(); outputs.append(path)

    evidence_macro = evidence.loc[evidence.cancer_id.isin(PILOT_CANCERS)].groupby("model").agg(
        auprc=("auprc", "mean"), prevalence=("positive_rate", "mean")
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(evidence_macro.index, evidence_macro.auprc, color="#10b981")
    ax.axhline(evidence_macro.prevalence.mean(), color="#ef4444", linestyle="--", label="prevalence")
    ax.set_ylabel("Evidence replication AUPRC")
    ax.set_title("Figure 6. Evidence Count and ET residual")
    ax.legend(); plt.tight_layout()
    path = figure_dir / "FIGURE_6_EVIDENCE.png"
    plt.savefig(path, dpi=180); plt.close(); outputs.append(path)

    test = b2_predictions.loc[b2_predictions.split.astype(str).eq("test")].copy()
    if {"prediction_scale", "target_subtype", "proxy_positive_probability"}.issubset(test.columns):
        labels, values = [], []
        for subtype in TARGET_SUBTYPES:
            for scale in ("raw_probability", "calibrated_probability"):
                current = pd.to_numeric(
                    test.loc[
                        test.target_subtype.astype(str).eq(subtype)
                        & test.prediction_scale.astype(str).eq(scale),
                        "proxy_positive_probability",
                    ], errors="coerce"
                ).dropna()
                if len(current):
                    values.append(current.sample(min(len(current), 20000), random_state=31).to_numpy())
                    labels.append(f"{subtype}\n{scale.replace('_probability','')}")
        if values:
            fig, ax = plt.subplots(figsize=(11, 5))
            ax.boxplot(values, labels=labels, showfliers=False)
            ax.set_ylabel("proxy positive probability")
            ax.set_title("Figure 7. Raw/calibrated prediction distributions")
            plt.xticks(rotation=25, ha="right"); plt.tight_layout()
            path = figure_dir / "FIGURE_7_RAW_CALIBRATED_DISTRIBUTION.png"
            plt.savefig(path, dpi=180); plt.close(); outputs.append(path)

    fig, ax = plt.subplots(figsize=(12, 3.8)); ax.axis("off")
    boxes = [
        (0.02, "BestSimple\nB0"), (0.23, "+ Graph\nR1"), (0.44, "+ admitted context\nR_SELECTED"),
        (0.68, "Observed PF\nseparate axis"), (0.86, "Evidence Count+ET\nseparate axis"),
    ]
    for x, label in boxes:
        ax.text(x, .52, label, transform=ax.transAxes, ha="left", va="center",
                bbox=dict(boxstyle="round,pad=.5", facecolor="#e0f2fe", edgecolor="#0369a1"))
    for left, right in ((.16, .22), (.37, .43)):
        ax.annotate("", xy=(right, .52), xytext=(left, .52), xycoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="->", color="black"))
    ax.set_title("Figure 8. Final conditional residual architecture")
    path = figure_dir / "FIGURE_8_FINAL_ARCHITECTURE.png"
    plt.savefig(path, dpi=180, bbox_inches="tight"); plt.close(); outputs.append(path)
    return outputs


def _summary_markdown(
    payload: dict[str, Any],
    acceptance: pd.DataFrame,
    gain_summary: pd.DataFrame,
    evidence_summary: dict[str, Any],
) -> str:
    lines = [
        "# CancerLncAtlas V3.1 Graph Repair / Residual Rebuild three-cancer pilot",
        "",
        f"## Decision: {payload['overall_decision']}",
        "",
        "Scope is fixed to BRCA, COAD, and KIRP. The workflow is stopped; no full-cancer training or website update was started.",
        "",
        "## Acceptance matrix",
        "",
        "| Module | Baseline | Candidate | mean ΔAUPRC | 95% CI | Positive cancers | Decision |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in acceptance.itertuples(index=False):
        delta = getattr(row, "mean_delta_auprc", np.nan)
        low, high = getattr(row, "ci95_lower", np.nan), getattr(row, "ci95_upper", np.nan)
        positive, total = getattr(row, "positive_cancers", np.nan), getattr(row, "total_cancers", np.nan)
        lines.append(
            f"| {row.module} | {row.baseline} | {row.candidate} | "
            f"{delta:.4f} | [{low:.4f}, {high:.4f}] | {positive}/{total} | {row.decision} |"
        )
    lines.extend(["", "## Evidence interpretation", ""])
    lines.append(
        "E4 was admitted using outer-validation only. Test advantage is "
        + ("demonstrated." if evidence_summary["test_advantage_demonstrated"] else "not demonstrated; it remains an evidence/annotation layer rather than a proven predictive advantage.")
    )
    lines.extend([
        "", "## STOP", "",
        "No full-cancer validation, formal V3.1 training, website update, or historical-output overwrite was initiated.",
        "Explicit user approval is required for any work beyond this three-cancer pilot.", "",
    ])
    return "\n".join(lines)


def main() -> int:
    args = _arguments()
    repo = Path(args.repo_root).resolve()
    commit = _clean_repo_commit(repo)
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise RuntimeError(f"Phase C refuses output reuse: {output}")

    phase_a2_path = Path(args.phase_a2_gate).resolve()
    b1_path = Path(args.b1_gate).resolve()
    b1_summary_path = Path(args.b1_summary).resolve()
    simple_path = Path(args.simple_gate).resolve()
    b2_dir = Path(args.b2_dir).resolve()
    hard_dir = Path(args.b2_hard_dir).resolve()
    b4_dir = Path(args.b4_dir).resolve()
    b3_dir = Path(args.b3_dir).resolve() if args.b3_dir else None
    relation_dir = Path(args.relation_ablation_dir).resolve() if args.relation_ablation_dir else None
    downstream_root = Path(args.downstream_root).resolve() if args.downstream_root else None

    gates = {
        "phase_a2": _json(phase_a2_path),
        "b1": _json(b1_path),
        "simple": _json(simple_path),
        "b2": _json(b2_dir / "GRAPH_RESIDUAL_GATE.json"),
        "hard": _json(hard_dir / "B2_HARD_REPORT.json"),
        "b4": _json(b4_dir / "ET_RESIDUAL_GATE.json"),
        "b3": _json(b3_dir / "CONTEXT_RESIDUAL_GATE.json") if b3_dir else None,
    }
    chain = validate_phase_c_gate_chain(
        phase_a2_gate=gates["phase_a2"], b1_gate=gates["b1"],
        simple_gate=gates["simple"], b2_gate=gates["b2"],
        b2_hard_report=gates["hard"], b4_gate=gates["b4"], b3_gate=gates["b3"],
    )
    if gates["hard"].get("b2_gate_sha256") != file_sha256(b2_dir / "GRAPH_RESIDUAL_GATE.json"):
        raise RuntimeError("B2 HARD REPORT is not bound to supplied B2 gate")
    manifests = {
        "b2": _verify_manifest(b2_dir, "GRAPH_RESIDUAL_SHA256.tsv"),
        "b2_hard": _verify_manifest(hard_dir, "B2_HARD_REPORT_SHA256.tsv"),
        "b4": _verify_manifest(b4_dir, "ET_RESIDUAL_SHA256.tsv"),
    }
    if b3_dir:
        manifests["b3"] = _verify_manifest(b3_dir, "CONTEXT_RESIDUAL_SHA256.tsv")
    conditional = _validate_conditional_outputs(
        gates["hard"], b2_dir=b2_dir, hard_dir=hard_dir,
        b3_dir=b3_dir, relation_dir=relation_dir,
        downstream_root=downstream_root,
    )

    b1_summary = pd.read_csv(b1_summary_path, sep="\t")
    b2_group = pd.read_csv(hard_dir / "B2_HARD_REPORT_GROUP_METRICS.tsv", sep="\t")
    b3_metrics = pd.read_csv(b3_dir / "MULTIMODAL_RESIDUAL_METRICS.tsv", sep="\t") if b3_dir else None
    graph = build_graph_comparison_matrix(b1_summary, b2_group, b3_metrics)
    gains = graph_gain_table(graph)
    gain_cancer, gain_summary = summarize_graph_gains(
        gains, iterations=args.bootstrap_iterations, seed=args.bootstrap_seed
    )
    evidence = build_evidence_comparison_matrix(
        pd.read_csv(b4_dir / "ET_STANDALONE_VS_COUNT_RESIDUAL.tsv", sep="\t")
    )
    evidence_cancer, evidence_summary = summarize_evidence_gain(
        evidence,
        module_admitted=bool(gates["b4"].get("module_admitted")),
        iterations=args.bootstrap_iterations,
        seed=args.bootstrap_seed + 1,
    )
    acceptance = _acceptance_matrix(
        gain_summary, evidence_summary, gates["hard"], chain["b3_status"]
    )

    output.mkdir(parents=True, exist_ok=False)
    paths = {
        "graph": output / "FINAL_COMPARISON_MATRIX.tsv",
        "gains": output / "FINAL_PAIRED_GAINS.tsv",
        "gain_cancer": output / "FINAL_GAIN_CANCER_CLUSTERS.tsv",
        "gain_summary": output / "FINAL_GAIN_SUMMARY.tsv",
        "evidence": output / "FINAL_EVIDENCE_COMPARISON.tsv",
        "evidence_cancer": output / "FINAL_EVIDENCE_GAIN_CANCERS.tsv",
        "acceptance": output / "FINAL_ACCEPTANCE_MATRIX.tsv",
        "fallback": output / "FINAL_EXACT_FALLBACK_AUDIT.tsv",
        "summary_json": output / "FINAL_PILOT_SUMMARY.json",
        "summary_md": output / "FINAL_PILOT_SUMMARY.md",
        "gate": output / "FINAL_PHASE_C_GATE.json",
        "manifest": output / "FINAL_PHASE_C_SHA256.tsv",
    }
    for frame, key in (
        (graph, "graph"), (gains, "gains"), (gain_cancer, "gain_cancer"),
        (gain_summary, "gain_summary"), (evidence, "evidence"),
        (evidence_cancer, "evidence_cancer"), (acceptance, "acceptance"),
    ):
        _atomic_table(frame, paths[key])
    fallback_audit = _exact_fallback_audit(
        hard_report=gates["hard"], b4_dir=b4_dir, b4_gate=gates["b4"],
        b3_dir=b3_dir,
    )
    _atomic_table(fallback_audit, paths["fallback"])
    figures = _plot_outputs(
        output / "FIGURES", graph, gain_summary, evidence,
        pd.read_parquet(b2_dir / "GRAPH_RESIDUAL_PREDICTIONS.parquet"), relation_dir,
    )
    overall = (
        "PILOT_HARD_STOP_AFTER_B2"
        if gates["hard"]["decision"] == "HARD_STOP_AFTER_B2"
        else "PILOT_COMPLETE_AND_STOPPED_AFTER_CONDITIONAL_B3"
    )
    payload = {
        "status": "COMPLETE_AND_STOPPED",
        "stage": "PHASE_C_THREE_CANCER_FINAL",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "overall_decision": overall,
        "scope_cancers": list(PILOT_CANCERS),
        "target_subtypes": list(TARGET_SUBTYPES),
        "b2_hard_decision": gates["hard"]["decision"],
        "gate_chain": chain,
        "conditional_outputs": conditional,
        "evidence_decision": evidence_summary,
        "exact_fallback_status": "PASS",
        "analysis_git_commit": commit,
        "input_manifests": manifests,
        "full_cancer_training_started": False,
        "full_cancer_training_authorized": False,
        "website_updated": False,
        "historical_outputs_modified": False,
        "stop_clause": "Three-cancer pilot finalized; explicit user approval is required for any expansion.",
        "failures": [],
    }
    atomic_write_json(paths["summary_json"], payload)
    paths["summary_md"].write_text(
        _summary_markdown(payload, acceptance, gain_summary, evidence_summary), encoding="utf-8"
    )
    output_files = [
        path for path in output.rglob("*")
        if path.is_file() and path not in {paths["gate"], paths["manifest"]}
    ]
    manifest_rows = [
        {
            "relative_path": str(path.relative_to(output)),
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(output_files)
    ]
    _atomic_table(pd.DataFrame(manifest_rows), paths["manifest"])
    gate = {
        **payload,
        "summary_sha256": file_sha256(paths["summary_json"]),
        "manifest_sha256": file_sha256(paths["manifest"]),
        "manifest_merkle_sha256": merkle_sha256(manifest_rows),
        "figure_count": len(figures),
    }
    atomic_write_json(paths["gate"], gate)
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
