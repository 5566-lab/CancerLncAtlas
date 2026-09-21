"""Fail-closed Phase C comparison helpers for the V3.1 three-cancer pilot.

This module deliberately contains no training entry point.  It combines frozen
Phase A2/B1/B2/B3/B4 metric artifacts, enforces the conditional B3 contract,
and produces comparison tables for BRCA/COAD/KIRP only.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from .common import require_columns


PILOT_CANCERS = ("BRCA", "COAD", "KIRP")
PILOT_SEEDS = (20260726, 20261726, 20262726)
TARGET_SUBTYPES = ("Pathway", "RNAss", "DNAss")


def _clean_pass_gate(gate: Mapping[str, Any], stage: str) -> None:
    if gate.get("status") != "PASS" or gate.get("failures"):
        raise RuntimeError(f"{stage} gate is not a clean PASS")
    unsafe_flags = (
        "full_cancer_training_started",
        "full_cancer_model_training_started",
        "full_cancer_training_authorized",
        "full_cancer_model_training_authorized",
    )
    if any(gate.get(key) is True for key in unsafe_flags):
        raise RuntimeError(f"{stage} gate reports forbidden full-cancer training/authorization")


def validate_phase_c_gate_chain(
    *,
    phase_a2_gate: Mapping[str, Any],
    b1_gate: Mapping[str, Any],
    simple_gate: Mapping[str, Any],
    b2_gate: Mapping[str, Any],
    b2_hard_report: Mapping[str, Any],
    b4_gate: Mapping[str, Any],
    b3_gate: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate the complete pilot chain without authorizing further work."""

    for gate, stage in (
        (phase_a2_gate, "Phase A2"),
        (b1_gate, "Phase B1"),
        (simple_gate, "BestSimple"),
        (b2_gate, "Phase B2"),
        (b4_gate, "Phase B4"),
    ):
        _clean_pass_gate(gate, stage)
    _clean_pass_gate(b2_hard_report, "B2 HARD REPORT")

    exact_cancers = set(PILOT_CANCERS)
    exact_seeds = set(PILOT_SEEDS)
    for gate, stage, key in (
        (phase_a2_gate, "Phase A2", "pilot_cancers"),
        (b1_gate, "Phase B1", "pilot_cancers"),
        (simple_gate, "BestSimple", "pilot_cancers"),
        (b2_gate, "Phase B2", "pilot_cancers"),
        (b4_gate, "Phase B4", "cancers"),
    ):
        if set(map(str, gate.get(key, []))) != exact_cancers:
            raise RuntimeError(f"{stage} gate cancer scope drift")
    if set(map(int, phase_a2_gate.get("seeds_allowed_next", []))) != exact_seeds:
        raise RuntimeError("Phase A2 seed scope drift")
    if set(map(int, b1_gate.get("primary_seeds", []))) != exact_seeds:
        raise RuntimeError("Phase B1 seed scope drift")
    if set(map(int, b2_gate.get("model_seeds", []))) != exact_seeds:
        raise RuntimeError("Phase B2 seed scope drift")
    if b1_gate.get("tasks_completed") != 12 or b1_gate.get("tasks_expected") != 12:
        raise RuntimeError("Phase B1 is not exactly 12/12 tasks")
    if b2_gate.get("tasks_completed") != 27 or b2_gate.get("tasks_expected") != 27:
        raise RuntimeError("Phase B2 is not exactly 27/27 tasks")
    if simple_gate.get("selection_used_test_labels") is not False:
        raise RuntimeError("BestSimple does not prove validation-only selection")
    if b2_gate.get("selection_used_test_labels") is not False:
        raise RuntimeError("Phase B2 does not prove validation-only selection")
    if b4_gate.get("selection_used_test_labels") is not False:
        raise RuntimeError("Phase B4 does not prove validation-only selection")
    a2_checks = phase_a2_gate.get("checks", {})
    if a2_checks.get("full_cancer_training_started") is not False:
        raise RuntimeError("Phase A2 does not explicitly prove no full-cancer training")
    if phase_a2_gate.get("full_cancer_training_authorized") is not False:
        raise RuntimeError("Phase A2 does not explicitly forbid full-cancer training")

    if b2_hard_report.get("stage") != "B2_HARD_REPORT":
        raise RuntimeError("Phase C requires the canonical B2 HARD REPORT")
    decision = str(b2_hard_report.get("decision"))
    if decision not in {"HARD_GO_B3", "HARD_STOP_AFTER_B2"}:
        raise RuntimeError(f"Unknown B2 hard decision: {decision}")
    if b2_hard_report.get("full_cancer_training_authorized") is not False:
        raise RuntimeError("B2 HARD REPORT contains unsafe full-cancer authorization")

    required_hard_gates = {
        "delta_auprc",
        "cluster_aware_ci",
        "cancer_direction",
        "calibration",
        "exact_fallback",
        "integrity",
    }
    hard_gates = b2_hard_report.get("gates", {})
    if set(hard_gates) != required_hard_gates:
        raise RuntimeError("B2 HARD REPORT hard-gate schema drift")

    if decision == "HARD_GO_B3":
        if b2_hard_report.get("b3_authorized") is not True:
            raise RuntimeError("HARD_GO_B3 lacks explicit B3 authorization")
        if not all(hard_gates.values()):
            raise RuntimeError("HARD_GO_B3 contains a failed hard gate")
        if b3_gate is None:
            raise RuntimeError("Phase C cannot finalize HARD_GO_B3 without Phase B3")
        _clean_pass_gate(b3_gate, "Phase B3")
        if b3_gate.get("stage") != "PHASE_B3_MULTIMODAL_CONTEXT_RESIDUAL":
            raise RuntimeError("Unexpected Phase B3 gate stage")
        if b3_gate.get("b2_hard_report_decision") != "HARD_GO_B3":
            raise RuntimeError("Phase B3 is not bound to the HARD GO decision")
        if set(map(str, b3_gate.get("pilot_cancers", []))) != exact_cancers:
            raise RuntimeError("Phase B3 cancer scope drift")
        if set(map(int, b3_gate.get("model_seeds", []))) != exact_seeds:
            raise RuntimeError("Phase B3 seed scope drift")
        if b3_gate.get("selection_used_test_labels") is not False:
            raise RuntimeError("Phase B3 does not prove validation-only selection")
        b3_status = "COMPLETED"
    else:
        if b2_hard_report.get("b3_authorized") is not False:
            raise RuntimeError("HARD_STOP_AFTER_B2 incorrectly authorizes B3")
        if b3_gate is not None:
            raise RuntimeError("Phase B3 exists despite HARD_STOP_AFTER_B2")
        b3_status = "NOT_RUN_CONDITIONAL_HARD_STOP"

    return {
        "status": "PASS",
        "stage": "PHASE_C_GATE_CHAIN",
        "b2_hard_decision": decision,
        "b3_status": b3_status,
        "b4_status": "COMPLETED",
        "full_cancer_training_started": False,
        "full_cancer_training_authorized": False,
    }


def _b1_rows(b1_summary: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        b1_summary,
        [
            "summary_scope",
            "cancer_id",
            "target_subtype",
            "model_id",
            "n_runs",
            "auprc_mean",
            "auroc_mean",
            "brier_mean",
            "ece_mean",
            "positive_rate_mean",
            "auprc_over_prevalence_mean",
            "auprc_minus_prevalence_mean",
        ],
        "B1 comparison summary",
    )
    mapping = {"OLD_P0": "G0", "OLD_P2": "G1", "FIXED_G_T": "G2"}
    rows = b1_summary.loc[
        b1_summary.summary_scope.astype(str).eq("within_cancer_seed_macro")
        & b1_summary.model_id.astype(str).isin(mapping)
    ].copy()
    rows["model"] = rows.model_id.map(mapping)
    rows = rows.rename(
        columns={
            "auprc_mean": "auprc",
            "auroc_mean": "auroc",
            "brier_mean": "brier",
            "ece_mean": "ece",
            "positive_rate_mean": "positive_rate",
            "auprc_over_prevalence_mean": "auprc_over_prevalence",
            "auprc_minus_prevalence_mean": "auprc_minus_prevalence",
        }
    )
    rows["source_stage"] = "B1"
    rows["availability"] = "AVAILABLE"
    return rows[
        [
            "model",
            "cancer_id",
            "target_subtype",
            "n_runs",
            "positive_rate",
            "auprc",
            "auroc",
            "brier",
            "ece",
            "auprc_over_prevalence",
            "auprc_minus_prevalence",
            "source_stage",
            "availability",
        ]
    ]


def _b2_rows(b2_group_metrics: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        b2_group_metrics,
        [
            "loco_cancer",
            "seed",
            "target_subtype",
            "positive_rate",
            "base_auprc",
            "r1_auprc",
            "base_auroc",
            "r1_auroc",
            "base_brier",
            "r1_brier",
            "base_ece",
            "r1_ece",
        ],
        "B2 HARD REPORT group metrics",
    )
    if set(b2_group_metrics.loco_cancer.astype(str)) != set(PILOT_CANCERS):
        raise RuntimeError("B2 comparison cancer scope drift")
    if set(b2_group_metrics.seed.astype(int)) != set(PILOT_SEEDS):
        raise RuntimeError("B2 comparison seed scope drift")
    if set(b2_group_metrics.target_subtype.astype(str)) != set(TARGET_SUBTYPES):
        raise RuntimeError("B2 comparison target scope drift")
    if b2_group_metrics.duplicated(["loco_cancer", "seed", "target_subtype"]).any():
        raise RuntimeError("B2 comparison contains duplicate groups")

    rows: list[pd.DataFrame] = []
    for model, prefix in (("B0", "base"), ("R1", "r1")):
        current = (
            b2_group_metrics.groupby(
                ["loco_cancer", "target_subtype"], observed=True, sort=True
            )
            .agg(
                n_runs=("seed", "size"),
                positive_rate=("positive_rate", "mean"),
                auprc=(f"{prefix}_auprc", "mean"),
                auroc=(f"{prefix}_auroc", "mean"),
                brier=(f"{prefix}_brier", "mean"),
                ece=(f"{prefix}_ece", "mean"),
            )
            .reset_index()
            .rename(columns={"loco_cancer": "cancer_id"})
        )
        current["model"] = model
        current["auprc_over_prevalence"] = current.auprc / current.positive_rate
        current["auprc_minus_prevalence"] = current.auprc - current.positive_rate
        current["source_stage"] = "B2"
        current["availability"] = "AVAILABLE"
        rows.append(current)
    return pd.concat(rows, ignore_index=True, sort=False)


def _b3_rows(b3_metrics: pd.DataFrame | None) -> pd.DataFrame:
    columns = [
        "model",
        "cancer_id",
        "target_subtype",
        "n_runs",
        "positive_rate",
        "auprc",
        "auroc",
        "brier",
        "ece",
        "auprc_over_prevalence",
        "auprc_minus_prevalence",
        "source_stage",
        "availability",
    ]
    if b3_metrics is None:
        placeholders = []
        for model in ("R2", "R3-GEN", "R3-ATAC", "R3-SC", "R_SELECTED"):
            for cancer in PILOT_CANCERS:
                for subtype in TARGET_SUBTYPES:
                    placeholders.append(
                        {
                            "model": model,
                            "cancer_id": cancer,
                            "target_subtype": subtype,
                            "n_runs": 0,
                            "source_stage": "B3",
                            "availability": "NOT_RUN_CONDITIONAL_HARD_STOP",
                        }
                    )
        return pd.DataFrame(placeholders).reindex(columns=columns)

    require_columns(
        b3_metrics,
        [
            "module",
            "loco_cancer",
            "seed",
            "target_subtype",
            "split",
            "model_variant",
            "positive_rate",
            "auprc",
            "auroc",
            "brier",
            "ece",
            "auprc_over_prevalence",
            "auprc_minus_prevalence",
        ],
        "B3 context metrics",
    )
    test = b3_metrics.loc[b3_metrics.split.astype(str).eq("test")].copy()
    mapping = {
        "RNA": "R2",
        "GENOMIC": "R3-GEN",
        "ATAC": "R3-ATAC",
        "SINGLECELL": "R3-SC",
        "SELECTED": "R_SELECTED",
    }
    test = test.loc[test.module.astype(str).isin(mapping)].copy()
    test["model"] = test.module.map(mapping)
    current = (
        test.groupby(
            ["model", "loco_cancer", "target_subtype"],
            observed=True,
            sort=True,
        )
        .agg(
            n_runs=("seed", "size"),
            positive_rate=("positive_rate", "mean"),
            auprc=("auprc", "mean"),
            auroc=("auroc", "mean"),
            brier=("brier", "mean"),
            ece=("ece", "mean"),
            auprc_over_prevalence=("auprc_over_prevalence", "mean"),
            auprc_minus_prevalence=("auprc_minus_prevalence", "mean"),
        )
        .reset_index()
        .rename(columns={"loco_cancer": "cancer_id"})
    )
    current["source_stage"] = "B3"
    current["availability"] = "AVAILABLE"
    return current.reindex(columns=columns)


def build_graph_comparison_matrix(
    b1_summary: pd.DataFrame,
    b2_group_metrics: pd.DataFrame,
    b3_metrics: pd.DataFrame | None,
) -> pd.DataFrame:
    """Build the exact G0/G1/G2/B0/R1/R2/R3/R_SELECTED comparison."""

    result = pd.concat(
        [_b1_rows(b1_summary), _b2_rows(b2_group_metrics), _b3_rows(b3_metrics)],
        ignore_index=True,
        sort=False,
    )
    key = ["model", "cancer_id", "target_subtype"]
    if result.duplicated(key).any():
        raise RuntimeError("Phase C graph comparison contains duplicate keys")
    expected_models = {
        "G0",
        "G1",
        "G2",
        "B0",
        "R1",
        "R2",
        "R3-GEN",
        "R3-ATAC",
        "R3-SC",
        "R_SELECTED",
    }
    if set(result.model.astype(str)) != expected_models:
        raise RuntimeError("Phase C graph model set drift")
    for model in expected_models:
        rows = result.loc[result.model.astype(str).eq(model)]
        if set(rows.cancer_id.astype(str)) != set(PILOT_CANCERS):
            raise RuntimeError(f"{model} lacks the exact pilot cancer set")
        if set(rows.target_subtype.astype(str)) != set(TARGET_SUBTYPES):
            raise RuntimeError(f"{model} lacks the exact target subtype set")
    order = {name: index for index, name in enumerate(
        ["G0", "G1", "G2", "B0", "R1", "R2", "R3-GEN", "R3-ATAC", "R3-SC", "R_SELECTED"]
    )}
    result["_order"] = result.model.map(order)
    return result.sort_values(
        ["_order", "target_subtype", "cancer_id"], kind="stable"
    ).drop(columns="_order").reset_index(drop=True)


def build_evidence_comparison_matrix(evidence_metrics: pd.DataFrame) -> pd.DataFrame:
    """Normalize B4 E0-E4 metrics without reinterpreting its estimand."""

    require_columns(
        evidence_metrics,
        [
            "scope",
            "method",
            "positive_rate",
            "auprc",
            "auroc",
            "brier",
            "ece",
            "auprc_over_prevalence",
            "auprc_minus_prevalence",
        ],
        "B4 evidence metrics",
    )
    method_map = {
        "E0_PREVALENCE": "E0",
        "E1_EVENT_COUNT_RANK": "E1",
        "E2_COUNT_LOGISTIC": "E2",
        "E3_ET_STANDALONE_SELECTED": "E3",
        "E4_SELECTED_OR_COUNT": "E4",
    }
    result = evidence_metrics.loc[
        evidence_metrics.scope.astype(str).isin([*PILOT_CANCERS, "THREE_CANCER_POOLED"])
        & evidence_metrics.method.astype(str).isin(method_map)
    ].copy()
    result["model"] = result.method.map(method_map)
    result = result.rename(columns={"scope": "cancer_id"})
    if result.duplicated(["model", "cancer_id"]).any():
        raise RuntimeError("Phase C evidence comparison contains duplicate keys")
    for model in method_map.values():
        observed = set(result.loc[result.model.eq(model), "cancer_id"].astype(str))
        if observed != {*PILOT_CANCERS, "THREE_CANCER_POOLED"}:
            raise RuntimeError(f"{model} evidence scope drift")
    return result[
        [
            "model",
            "cancer_id",
            "positive_rate",
            "auprc",
            "auroc",
            "brier",
            "ece",
            "auprc_over_prevalence",
            "auprc_minus_prevalence",
        ]
    ].sort_values(["model", "cancer_id"], kind="stable").reset_index(drop=True)


def graph_gain_table(graph: pd.DataFrame) -> pd.DataFrame:
    """Report repair, graph-residual, and final-system gains separately."""

    available = graph.loc[graph.availability.astype(str).eq("AVAILABLE")].copy()
    index = available.set_index(["model", "cancer_id", "target_subtype"])
    comparisons = [("REPAIR_GAIN", "G2", "G1"), ("GRAPH_RESIDUAL_GAIN", "R1", "B0")]
    if "R_SELECTED" in set(available.model.astype(str)):
        comparisons.append(("FINAL_MULTIMODAL_GAIN", "R_SELECTED", "B0"))
    rows: list[dict[str, Any]] = []
    for name, candidate, baseline in comparisons:
        for cancer in PILOT_CANCERS:
            for subtype in TARGET_SUBTYPES:
                candidate_key = (candidate, cancer, subtype)
                baseline_key = (baseline, cancer, subtype)
                if candidate_key not in index.index or baseline_key not in index.index:
                    continue
                c = index.loc[candidate_key]
                b = index.loc[baseline_key]
                rows.append(
                    {
                        "comparison": name,
                        "candidate_model": candidate,
                        "baseline_model": baseline,
                        "cancer_id": cancer,
                        "target_subtype": subtype,
                        "delta_auprc": float(c.auprc - b.auprc),
                        "delta_auroc": float(c.auroc - b.auroc),
                        "delta_brier": float(c.brier - b.brier),
                        "delta_ece": float(c.ece - b.ece),
                    }
                )
    return pd.DataFrame(rows)


def _cluster_bootstrap_mean(
    values: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) != len(PILOT_CANCERS) or not np.isfinite(values).all():
        raise RuntimeError("Phase C requires exactly three finite cancer clusters")
    rng = np.random.default_rng(int(seed))
    sampled = rng.choice(values, size=(int(iterations), len(values)), replace=True)
    return tuple(float(value) for value in np.quantile(sampled.mean(axis=1), [0.025, 0.5, 0.975]))


def summarize_graph_gains(
    gains: pd.DataFrame,
    *,
    iterations: int = 10000,
    seed: int = 20260815,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize gains with cancer, not candidate rows, as the cluster."""

    require_columns(
        gains,
        [
            "comparison",
            "candidate_model",
            "baseline_model",
            "cancer_id",
            "target_subtype",
            "delta_auprc",
            "delta_auroc",
            "delta_brier",
            "delta_ece",
        ],
        "Phase C graph gain table",
    )
    cancer_parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    comparisons = list(dict.fromkeys(gains.comparison.astype(str)))
    scopes = [*TARGET_SUBTYPES, "ALL_TARGETS_MACRO"]
    for comparison_index, comparison in enumerate(comparisons):
        comparison_rows = gains.loc[gains.comparison.astype(str).eq(comparison)]
        for scope_index, scope in enumerate(scopes):
            current = comparison_rows if scope == "ALL_TARGETS_MACRO" else comparison_rows.loc[
                comparison_rows.target_subtype.astype(str).eq(scope)
            ]
            by_cancer = (
                current.groupby("cancer_id", observed=True, sort=True)
                .agg(
                    mean_delta_auprc=("delta_auprc", "mean"),
                    mean_delta_auroc=("delta_auroc", "mean"),
                    mean_delta_brier=("delta_brier", "mean"),
                    mean_delta_ece=("delta_ece", "mean"),
                    target_rows=("target_subtype", "size"),
                )
                .reset_index()
            )
            if tuple(by_cancer.cancer_id.astype(str)) != PILOT_CANCERS:
                raise RuntimeError(f"{comparison}/{scope} lacks the exact pilot cancers")
            by_cancer.insert(0, "scope", scope)
            by_cancer.insert(0, "comparison", comparison)
            cancer_parts.append(by_cancer)
            values = by_cancer.mean_delta_auprc.to_numpy(float)
            low, median, high = _cluster_bootstrap_mean(
                values,
                iterations=iterations,
                seed=seed + comparison_index * 10 + scope_index,
            )
            mean_delta = float(values.mean())
            positive = int(np.sum(values > 0))
            if comparison == "REPAIR_GAIN":
                decision = "GO" if mean_delta > 0 and positive >= 2 else "NO-GO"
            elif mean_delta >= 0.02 and low > 0 and positive >= 2:
                decision = "STRONG GO"
            elif mean_delta >= 0.02 and positive >= 2:
                decision = "PILOT GO"
            elif mean_delta > 0:
                decision = "BORDERLINE"
            else:
                decision = "NO-GO"
            summary_rows.append(
                {
                    "comparison": comparison,
                    "scope": scope,
                    "mean_delta_auprc": mean_delta,
                    "median_cancer_delta_auprc": float(np.median(values)),
                    "cancer_cluster_bootstrap_ci95_lower": low,
                    "cancer_cluster_bootstrap_median": median,
                    "cancer_cluster_bootstrap_ci95_upper": high,
                    "positive_cancers": positive,
                    "total_cancers": len(PILOT_CANCERS),
                    "decision": decision,
                    "bootstrap_iterations": int(iterations),
                    "bootstrap_seed": int(seed + comparison_index * 10 + scope_index),
                }
            )
    return pd.concat(cancer_parts, ignore_index=True), pd.DataFrame(summary_rows)


def summarize_evidence_gain(
    evidence: pd.DataFrame,
    *,
    module_admitted: bool,
    iterations: int = 10000,
    seed: int = 20260816,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compare E4 with E2 while preserving validation admission provenance."""

    pilot = evidence.loc[evidence.cancer_id.astype(str).isin(PILOT_CANCERS)].copy()
    pivot = pilot.pivot(index="cancer_id", columns="model", values="auprc")
    if tuple(pivot.index.astype(str)) != PILOT_CANCERS or not {"E2", "E4"}.issubset(pivot.columns):
        raise RuntimeError("B4 evidence comparison lacks E2/E4 pilot rows")
    rows = pivot.reset_index()[["cancer_id"]].copy()
    rows["e2_auprc"] = pivot.E2.to_numpy(float)
    rows["e4_auprc"] = pivot.E4.to_numpy(float)
    rows["delta_auprc"] = rows.e4_auprc - rows.e2_auprc
    values = rows.delta_auprc.to_numpy(float)
    low, median, high = _cluster_bootstrap_mean(values, iterations=iterations, seed=seed)
    mean_delta = float(values.mean())
    positive = int(np.sum(values > 0))
    test_advantage = bool(mean_delta > 0 and low > 0 and positive >= 2)
    if module_admitted and test_advantage:
        decision = "ADMITTED_AND_TEST_ADVANTAGE_DEMONSTRATED"
    elif module_admitted:
        decision = "ADMITTED_ON_VALIDATION_TEST_ADVANTAGE_NOT_DEMONSTRATED"
    else:
        decision = "OFF_VALIDATION_NO_GO"
    return rows, {
        "module": "EVIDENCE_COUNT_PLUS_ET_RESIDUAL",
        "validation_admitted": bool(module_admitted),
        "mean_test_delta_auprc": mean_delta,
        "cancer_cluster_bootstrap_ci95_lower": low,
        "cancer_cluster_bootstrap_median": median,
        "cancer_cluster_bootstrap_ci95_upper": high,
        "positive_cancers": positive,
        "total_cancers": len(PILOT_CANCERS),
        "test_advantage_demonstrated": test_advantage,
        "decision": decision,
        "bootstrap_iterations": int(iterations),
        "bootstrap_seed": int(seed),
    }
