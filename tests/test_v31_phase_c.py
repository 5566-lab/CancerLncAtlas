from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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


def test_phase_c_finalizer_is_report_only() -> None:
    path = Path(__file__).parents[1] / "scripts" / "96_finalize_v31_phase_c.py"
    source = path.read_text(encoding="utf-8")
    assert "83_run_v31_graph_residual" not in source
    assert "89_run_v31_context_residuals" not in source
    assert "subprocess.Popen" not in source
    spec = importlib.util.spec_from_file_location("v31_phase_c_finalizer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.main)


def test_phase_c_revalidates_strict_b4_admission_and_et_off(tmp_path: Path) -> None:
    path = Path(__file__).parents[1] / "scripts" / "96_finalize_v31_phase_c.py"
    spec = importlib.util.spec_from_file_location("v31_phase_c_finalizer_fallback", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    admission = {
        "status": "PASS",
        "admitted": True,
        "validation_delta_auprc": 0.02,
        "validation_delta_brier": 0.0,
        "validation_delta_ece": 0.0,
        "positive_validation_folds": 5,
    }
    (tmp_path / "ET_RESIDUAL_ADMISSION.json").write_text(
        json.dumps(admission), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "mode": ["residual", "residual"],
            "zero_initialization_max_abs_error": [0.0, 0.0],
        }
    ).to_csv(tmp_path / "ET_FIT_AUDIT.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "e2_count_probability": [0.2, 0.8],
            "e4_selected_probability": [0.25, 0.75],
        }
    ).to_parquet(tmp_path / "ET_STANDALONE_VS_COUNT_RESIDUAL_PREDICTIONS.parquet")
    audit = module._exact_fallback_audit(
        hard_report={"off_max_abs_probability_error": 0.0},
        b4_dir=tmp_path,
        b4_gate={"module_admitted": True},
        b3_dir=None,
    )
    assert audit.status.eq("PASS").all()
    assert audit.loc[audit["invariant"].eq("ET OFF -> exact Count E2"), "max_abs_error"].iloc[0] == 0


def test_phase_c_rejects_historical_b4_admission_below_point_01(tmp_path: Path) -> None:
    path = Path(__file__).parents[1] / "scripts" / "96_finalize_v31_phase_c.py"
    spec = importlib.util.spec_from_file_location("v31_phase_c_finalizer_threshold", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "ET_RESIDUAL_ADMISSION.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "admitted": True,
                "validation_delta_auprc": 0.005,
                "validation_delta_brier": 0.0,
                "validation_delta_ece": 0.0,
                "positive_validation_folds": 5,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {"mode": ["residual"], "zero_initialization_max_abs_error": [0.0]}
    ).to_csv(tmp_path / "ET_FIT_AUDIT.tsv", sep="\t", index=False)
    pd.DataFrame(
        {"e2_count_probability": [0.2], "e4_selected_probability": [0.2]}
    ).to_parquet(tmp_path / "ET_STANDALONE_VS_COUNT_RESIDUAL_PREDICTIONS.parquet")
    with pytest.raises(RuntimeError, match="registered >=\\+0.01 rule"):
        module._exact_fallback_audit(
            hard_report={"off_max_abs_probability_error": 0.0},
            b4_dir=tmp_path,
            b4_gate={"module_admitted": True},
            b3_dir=None,
        )


def _gate(stage: str) -> dict:
    return {
        "status": "PASS",
        "stage": stage,
        "failures": [],
        "full_cancer_training_started": False,
    }


def _hard(decision: str) -> dict:
    go = decision == "HARD_GO_B3"
    return {
        **_gate("B2_HARD_REPORT"),
        "decision": decision,
        "b3_authorized": go,
        "full_cancer_training_authorized": False,
        "gates": {
            "delta_auprc": go,
            "cluster_aware_ci": go,
            "cancer_direction": go,
            "calibration": True,
            "exact_fallback": True,
            "integrity": True,
        },
    }


def test_phase_c_gate_chain_enforces_conditional_b3() -> None:
    common = dict(
        phase_a2_gate={
            **_gate("PHASE_A2"),
            "pilot_cancers": list(PILOT_CANCERS),
            "seeds_allowed_next": list(PILOT_SEEDS),
            "checks": {"full_cancer_training_started": False},
            "full_cancer_training_authorized": False,
        },
        b1_gate={
            **_gate("PHASE_B1"),
            "pilot_cancers": list(PILOT_CANCERS),
            "primary_seeds": list(PILOT_SEEDS),
            "tasks_completed": 12,
            "tasks_expected": 12,
        },
        simple_gate={
            **_gate("BEST_SIMPLE"),
            "pilot_cancers": list(PILOT_CANCERS),
            "selection_used_test_labels": False,
        },
        b2_gate={
            **_gate("PHASE_B2"),
            "pilot_cancers": list(PILOT_CANCERS),
            "model_seeds": list(PILOT_SEEDS),
            "tasks_completed": 27,
            "tasks_expected": 27,
            "selection_used_test_labels": False,
        },
        b4_gate={
            **_gate("PHASE_B4_EVIDENCE_COUNT_PLUS_ET_RESIDUAL"),
            "cancers": list(PILOT_CANCERS),
            "selection_used_test_labels": False,
            "full_cancer_model_training_started": False,
        },
    )
    stopped = validate_phase_c_gate_chain(
        **common, b2_hard_report=_hard("HARD_STOP_AFTER_B2"), b3_gate=None
    )
    assert stopped["b3_status"] == "NOT_RUN_CONDITIONAL_HARD_STOP"
    b3 = {
        **_gate("PHASE_B3_MULTIMODAL_CONTEXT_RESIDUAL"),
        "b2_hard_report_decision": "HARD_GO_B3",
        "pilot_cancers": list(PILOT_CANCERS),
        "model_seeds": list(PILOT_SEEDS),
        "selection_used_test_labels": False,
    }
    go = validate_phase_c_gate_chain(
        **common, b2_hard_report=_hard("HARD_GO_B3"), b3_gate=b3
    )
    assert go["b3_status"] == "COMPLETED"
    with pytest.raises(RuntimeError, match="without Phase B3"):
        validate_phase_c_gate_chain(
            **common, b2_hard_report=_hard("HARD_GO_B3"), b3_gate=None
        )
    with pytest.raises(RuntimeError, match="despite HARD_STOP"):
        validate_phase_c_gate_chain(
            **common, b2_hard_report=_hard("HARD_STOP_AFTER_B2"), b3_gate=b3
        )
    unsafe_b4 = dict(common["b4_gate"])
    unsafe_b4["full_cancer_model_training_started"] = True
    with pytest.raises(RuntimeError, match="forbidden full-cancer"):
        validate_phase_c_gate_chain(
            **{**common, "b4_gate": unsafe_b4},
            b2_hard_report=_hard("HARD_STOP_AFTER_B2"),
            b3_gate=None,
        )


def _write_manifest(root: Path, name: str, files: list[str]) -> str:
    rows = []
    for relative in files:
        path = root / relative
        rows.append(
            {
                "relative_path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    path = root / name
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_phase_c_conditional_outputs_require_retrained_lineage_and_manifests(
    tmp_path: Path,
) -> None:
    script = Path(__file__).parents[1] / "scripts" / "96_finalize_v31_phase_c.py"
    spec = importlib.util.spec_from_file_location("v31_phase_c_conditional", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    b2 = tmp_path / "b2"
    hard = tmp_path / "hard"
    b3 = tmp_path / "b3"
    relation = tmp_path / "relation"
    downstream = tmp_path / "downstream"
    for root in (b2, hard, b3, relation, downstream):
        root.mkdir()
    (b2 / "GRAPH_RESIDUAL_PREDICTIONS.parquet").write_bytes(b"graph")
    pd.DataFrame(
        {
            "target_subtype": list(TARGET_SUBTYPES),
            "selected_lambda": 0.01,
            "admitted": True,
            "test_metric_used_for_selection": False,
        }
    ).to_csv(b2 / "GRAPH_RESIDUAL_ADMISSION.tsv", sep="\t", index=False)
    hard_payload = _hard("HARD_GO_B3")
    (hard / "B2_HARD_REPORT.json").write_text(
        json.dumps(hard_payload), encoding="utf-8"
    )
    (b3 / "SELECTED_RESIDUAL_PREDICTIONS.parquet").write_bytes(b"selected")

    pd.DataFrame(
        {
            "relation_family": list(module.RELATION_ABLATION_FAMILIES),
            "ablation_method": "fresh_retraining",
            "delta_auprc_vs_full": np.linspace(-0.01, 0.01, 5),
        }
    ).to_csv(relation / "RELATION_FAMILY_ABLATION.tsv", sep="\t", index=False)
    run_rows = []
    for family in module.RELATION_ABLATION_FAMILIES:
        for cancer in PILOT_CANCERS:
            for seed in PILOT_SEEDS:
                for subtype in TARGET_SUBTYPES:
                    run_rows.append(
                        {
                            "relation_family": family,
                            "loco_cancer": cancer,
                            "seed": seed,
                            "target_subtype": subtype,
                            "ablation_method": "fresh_retraining",
                            "selected_lambda": 0.01,
                            "module_admitted": True,
                            "delta_auprc_vs_full": 0.001,
                        }
                    )
    pd.DataFrame(run_rows).to_csv(
        relation / "RELATION_FAMILY_ABLATION_RUNS.tsv", sep="\t", index=False
    )
    relation_manifest_sha = _write_manifest(
        relation,
        "RELATION_FAMILY_ABLATION_SHA256.tsv",
        ["RELATION_FAMILY_ABLATION.tsv", "RELATION_FAMILY_ABLATION_RUNS.tsv"],
    )
    relation_gate = {
        **_gate("PHASE_B3_RELATION_FAMILY_ABLATION"),
        "selection_used_test_labels": False,
        "ablation_method": "fresh_retraining",
        "fresh_initialization_all": True,
        "inference_only_edge_masking": False,
        "pilot_cancers": list(PILOT_CANCERS),
        "model_seeds": list(PILOT_SEEDS),
        "target_subtypes": list(TARGET_SUBTYPES),
        "ablation_families": list(module.RELATION_ABLATION_FAMILIES),
        "training_lambdas": [0.01],
        "tasks_completed": 45,
        "tasks_expected": 45,
        "candidate_universe_consistent": True,
        "candidate_universe_drift_rows": 0,
        "b2_hard_report_sha256": hashlib.sha256(
            (hard / "B2_HARD_REPORT.json").read_bytes()
        ).hexdigest(),
        "full_graph_prediction_sha256": hashlib.sha256(
            (b2 / "GRAPH_RESIDUAL_PREDICTIONS.parquet").read_bytes()
        ).hexdigest(),
        "manifest_sha256": relation_manifest_sha,
    }
    (relation / "RELATION_FAMILY_ABLATION_GATE.json").write_text(
        json.dumps(relation_gate), encoding="utf-8"
    )

    metric_names = {
        "SPECIFICITY_BENCHMARK_METRICS.tsv": "SPECIFICITY",
        "CONSERVATION_BENCHMARK_METRICS.tsv": "CONSERVATION",
        "REVERSAL_BENCHMARK_METRICS.tsv": "REVERSAL",
    }
    method_pairs = {
        "SPECIFICITY": (
            "BestSimpleSpecificity", "R_SELECTED_DualAxis_specificity",
        ),
        "CONSERVATION": (
            "BestSimpleConservation", "R_SELECTED_DualAxis_conservation",
        ),
        "REVERSAL": (
            "BestSimpleReversal", "R_SELECTED_DualAxis_reversal",
        ),
    }
    for name, task in metric_names.items():
        rows = []
        for cancer in PILOT_CANCERS:
            for fold in range(5):
                for target, subtype in (
                    ("stemness_rna::RNAss", "RNAss"),
                    ("stemness_dna::DNAss", "DNAss"),
                ):
                    for seed in PILOT_SEEDS:
                        for method in method_pairs[task]:
                            rows.append(
                                {
                                    "cancer_id": cancer,
                                    "patient_fold_id": f"LOCO_{cancer}__PF{fold:02d}",
                                    "target_id": target,
                                    "target_subtype": subtype,
                                    "seed": seed,
                                    "task": task,
                                    "method": method,
                                    "metric_scope": "PF_outer_test",
                                    "positive_prevalence": 0.1,
                                    "auprc": 0.2,
                                    "auroc": 0.6,
                                    "brier": 0.1,
                                    "ece": 0.05,
                                    "auprc_over_prevalence": 2.0,
                                    "auprc_minus_prevalence": 0.1,
                                    "n": 100,
                                    "n_positive": 10,
                                }
                            )
        pd.DataFrame(rows).to_csv(downstream / name, sep="\t", index=False)
    pd.DataFrame(
        {
            "task": list(module.DOWNSTREAM_TASKS),
            "best_simple_baseline": "simple",
            "new_score": "R_SELECTED",
            "delta_auprc": 0.02,
            "median_delta_auprc": 0.02,
            "ci95_lower": 0.001,
            "ci95_upper": 0.04,
            "n_clusters": 15,
            "positive_clusters": 10,
            "positive_cancers": 3,
            "total_cancers": 3,
            "decision": "GO",
            "test_metric_used_for_selection": False,
        }
    ).to_csv(
        downstream / "PRIMARY_AUPRC_ACCEPTANCE_MATRIX.tsv", sep="\t", index=False
    )
    pd.DataFrame(
        {
            "cancer_id": ["BRCA"],
            "seed": [PILOT_SEEDS[0]],
            "specificity_index": [0.2],
        }
    ).to_parquet(downstream / "CURRENT_EXPECTED_DUAL_AXIS_SCORES.parquet")
    pd.DataFrame(
        {
            "cancer_id": ["BRCA"],
            "seed": [PILOT_SEEDS[0]],
            "specificity_index": [0.2],
        }
    ).to_parquet(
        downstream / "SCORE_CONSTRUCTION_WITHOUT_HELDOUT_LABELS.parquet"
    )
    audit_rows = []
    for cancer in PILOT_CANCERS:
        for subtype in ("RNAss", "DNAss"):
            for seed in PILOT_SEEDS:
                audit_rows.append(
                    {
                        "cancer_id": cancer,
                        "target_subtype": subtype,
                        "seed": seed,
                        "candidate_rows": 100,
                        "patient_folds": 5,
                        "unique_lncrnas": 20,
                        "score_used_heldout_labels": False,
                    }
                )
    pd.DataFrame(audit_rows).to_csv(
        downstream / "CURRENT_EXPECTED_CANDIDATE_AUDIT.tsv",
        sep="\t",
        index=False,
    )
    downstream_files = [
        *metric_names,
        "PRIMARY_AUPRC_ACCEPTANCE_MATRIX.tsv",
        "CURRENT_EXPECTED_DUAL_AXIS_SCORES.parquet",
        "SCORE_CONSTRUCTION_WITHOUT_HELDOUT_LABELS.parquet",
        "CURRENT_EXPECTED_CANDIDATE_AUDIT.tsv",
    ]
    downstream_manifest_sha = _write_manifest(
        downstream, "CURRENT_EXPECTED_SHA256.tsv", downstream_files
    )
    downstream_gate = {
        **_gate("PHASE_B3_CURRENT_EXPECTED_DOWNSTREAM"),
        "expected_model": "R_SELECTED",
        "expected_prediction_sha256": hashlib.sha256(
            (b3 / "SELECTED_RESIDUAL_PREDICTIONS.parquet").read_bytes()
        ).hexdigest(),
        "test_labels_used_for_score_construction": False,
        "test_labels_used_for_evaluation_only": True,
        "test_metric_used_for_selection": False,
        "pilot_cancers": list(PILOT_CANCERS),
        "target_subtypes": list(TARGET_SUBTYPES),
        "benchmark_target_subtypes": ["RNAss", "DNAss"],
        "benchmark_tasks": list(module.DOWNSTREAM_TASKS),
        "reference_scope": "patient_fold_outer_test_RNAss_DNAss",
        "score_formula_scope": "preregistered_fixed_formula_no_test_tuning",
        "direction_source": "B2_validation_selected_state_direction_head",
        "candidate_audit_rows": 18,
        "manifest_sha256": downstream_manifest_sha,
    }
    (downstream / "CURRENT_EXPECTED_LINEAGE.json").write_text(
        json.dumps(downstream_gate), encoding="utf-8"
    )

    result = module._validate_conditional_outputs(
        hard_payload,
        b2_dir=b2,
        hard_dir=hard,
        b3_dir=b3,
        relation_dir=relation,
        downstream_root=downstream,
    )
    assert result["relation_ablation"] == "COMPLETED"
    assert result["specificity_conservation_reversal"] == "COMPLETED"

    relation_gate["inference_only_edge_masking"] = True
    (relation / "RELATION_FAMILY_ABLATION_GATE.json").write_text(
        json.dumps(relation_gate), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="Relation-family ablation gate"):
        module._validate_conditional_outputs(
            hard_payload,
            b2_dir=b2,
            hard_dir=hard,
            b3_dir=b3,
            relation_dir=relation,
            downstream_root=downstream,
        )


def _b1() -> pd.DataFrame:
    rows = []
    for cancer_index, cancer in enumerate(PILOT_CANCERS):
        for subtype_index, subtype in enumerate(TARGET_SUBTYPES):
            for model_index, model in enumerate(("OLD_P0", "OLD_P2", "FIXED_G_T")):
                score = 0.1 + cancer_index * 0.01 + subtype_index * 0.02 + model_index * 0.005
                rows.append(
                    {
                        "summary_scope": "within_cancer_seed_macro",
                        "cancer_id": cancer,
                        "target_subtype": subtype,
                        "model_id": model,
                        "n_runs": 3,
                        "auprc_mean": score,
                        "auroc_mean": score + 0.3,
                        "brier_mean": 0.2,
                        "ece_mean": 0.1,
                        "positive_rate_mean": 0.05,
                        "auprc_over_prevalence_mean": score / 0.05,
                        "auprc_minus_prevalence_mean": score - 0.05,
                    }
                )
    return pd.DataFrame(rows)


def _b2() -> pd.DataFrame:
    rows = []
    for cancer in PILOT_CANCERS:
        for seed in PILOT_SEEDS:
            for subtype in TARGET_SUBTYPES:
                rows.append(
                    {
                        "loco_cancer": cancer,
                        "seed": seed,
                        "target_subtype": subtype,
                        "positive_rate": 0.1,
                        "base_auprc": 0.2,
                        "r1_auprc": 0.23,
                        "base_auroc": 0.6,
                        "r1_auroc": 0.62,
                        "base_brier": 0.2,
                        "r1_brier": 0.19,
                        "base_ece": 0.1,
                        "r1_ece": 0.09,
                    }
                )
    return pd.DataFrame(rows)


def _b3() -> pd.DataFrame:
    rows = []
    for cancer in PILOT_CANCERS:
        for seed in PILOT_SEEDS:
            for subtype in TARGET_SUBTYPES:
                for module, score in (
                    ("RNA", 0.24),
                    ("GENOMIC", 0.235),
                    ("ATAC", 0.23),
                    ("SINGLECELL", 0.23),
                    ("SELECTED", 0.245),
                ):
                    rows.append(
                        {
                            "module": module,
                            "loco_cancer": cancer,
                            "seed": seed,
                            "target_subtype": subtype,
                            "split": "test",
                            "model_variant": "R_SELECTED" if module == "SELECTED" else "M2",
                            "positive_rate": 0.1,
                            "auprc": score,
                            "auroc": 0.63,
                            "brier": 0.18,
                            "ece": 0.08,
                            "auprc_over_prevalence": score / 0.1,
                            "auprc_minus_prevalence": score - 0.1,
                        }
                    )
    return pd.DataFrame(rows)


def test_graph_comparison_has_exact_model_cancer_target_matrix() -> None:
    matrix = build_graph_comparison_matrix(_b1(), _b2(), _b3())
    assert len(matrix) == 10 * 3 * 3
    assert matrix.availability.eq("AVAILABLE").all()
    gains = graph_gain_table(matrix)
    assert set(gains.comparison) == {
        "REPAIR_GAIN",
        "GRAPH_RESIDUAL_GAIN",
        "FINAL_MULTIMODAL_GAIN",
    }
    assert np.allclose(
        gains.loc[gains.comparison.eq("GRAPH_RESIDUAL_GAIN"), "delta_auprc"],
        0.03,
    )
    cancer, summary = summarize_graph_gains(gains, iterations=200, seed=7)
    assert len(cancer) == 3 * 4 * 3
    primary = summary.loc[
        summary.comparison.eq("GRAPH_RESIDUAL_GAIN")
        & summary.scope.eq("ALL_TARGETS_MACRO")
    ].iloc[0]
    assert primary.decision == "STRONG GO"
    assert primary.positive_cancers == 3


def test_graph_comparison_hard_stop_has_explicit_b3_placeholders() -> None:
    matrix = build_graph_comparison_matrix(_b1(), _b2(), None)
    unavailable = matrix.loc[matrix.model.str.startswith("R") & ~matrix.model.eq("R1")]
    assert len(unavailable) == 5 * 3 * 3
    assert unavailable.availability.eq("NOT_RUN_CONDITIONAL_HARD_STOP").all()
    assert set(graph_gain_table(matrix).comparison) == {"REPAIR_GAIN", "GRAPH_RESIDUAL_GAIN"}


def test_evidence_comparison_keeps_same_estimand_and_all_scopes() -> None:
    rows = []
    methods = (
        "E0_PREVALENCE",
        "E1_EVENT_COUNT_RANK",
        "E2_COUNT_LOGISTIC",
        "E3_ET_STANDALONE_SELECTED",
        "E4_SELECTED_OR_COUNT",
    )
    for scope in (*PILOT_CANCERS, "THREE_CANCER_POOLED"):
        for method in methods:
            rows.append(
                {
                    "scope": scope,
                    "method": method,
                    "positive_rate": 0.2,
                    "auprc": 0.3,
                    "auroc": 0.6,
                    "brier": 0.15,
                    "ece": 0.05,
                    "auprc_over_prevalence": 1.5,
                    "auprc_minus_prevalence": 0.1,
                }
            )
    result = build_evidence_comparison_matrix(pd.DataFrame(rows))
    assert len(result) == 20
    assert set(result.model) == {"E0", "E1", "E2", "E3", "E4"}
    cancer, summary = summarize_evidence_gain(
        result.assign(auprc=result.auprc + result.model.eq("E4") * 0.03),
        module_admitted=True,
        iterations=200,
        seed=9,
    )
    assert len(cancer) == 3
    assert summary["test_advantage_demonstrated"] is True
