"""Fail-closed HARD REPORT for the three-cancer B2 graph residual pilot."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .common import require_columns
from .metrics import binary_metrics
from .v31_residual import FALLBACK_ATOL


PILOT_CANCERS = ("BRCA", "COAD", "KIRP")
PILOT_SEEDS = (20260726, 20261726, 20262726)
TARGET_SUBTYPES = ("Pathway", "RNAss", "DNAss")
SHRINKAGE_GRID = (0.001, 0.01, 0.1)
HARD_MIN_MEAN_DELTA_AUPRC = 0.02


def _group_metrics(test: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups = ["loco_cancer", "seed", "target_subtype"]
    for keys, group in test.groupby(groups, observed=True, sort=True):
        base = binary_metrics(group.proxy_label, group.base_probability)
        final = binary_metrics(group.proxy_label, group.proxy_positive_probability)
        rows.append(
            {
                **dict(zip(groups, keys)),
                "candidate_rows": int(len(group)),
                "positive_rows": int(final["n_positive"]),
                "positive_rate": float(final["positive_rate"]),
                "base_auprc": float(base["auprc"]),
                "r1_auprc": float(final["auprc"]),
                "delta_auprc": float(final["auprc"] - base["auprc"]),
                "base_auprc_over_prevalence": float(
                    base["auprc"] / final["positive_rate"]
                ),
                "r1_auprc_over_prevalence": float(
                    final["auprc"] / final["positive_rate"]
                ),
                "base_auroc": float(base["auroc"]),
                "r1_auroc": float(final["auroc"]),
                "delta_auroc": float(final["auroc"] - base["auroc"]),
                "base_brier": float(base["brier"]),
                "r1_brier": float(final["brier"]),
                "delta_brier": float(final["brier"] - base["brier"]),
                "base_ece": float(base["ece"]),
                "r1_ece": float(final["ece"]),
                "delta_ece": float(final["ece"] - base["ece"]),
                "module_admitted": bool(group.module_admitted.astype(bool).all()),
            }
        )
    return pd.DataFrame(rows)


def _cancer_cluster_bootstrap(
    cancer_values: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> tuple[float, float, float]:
    """Bootstrap the macro mean with cancer as the independent cluster."""

    values = np.asarray(cancer_values, dtype=float)
    if len(values) != len(PILOT_CANCERS) or not np.isfinite(values).all():
        raise RuntimeError("HARD REPORT requires three finite cancer-level effects")
    rng = np.random.default_rng(seed)
    sampled = rng.choice(values, size=(int(iterations), len(values)), replace=True)
    means = sampled.mean(axis=1)
    return tuple(float(x) for x in np.quantile(means, [0.025, 0.5, 0.975]))


def _summaries(
    metrics: pd.DataFrame,
    *,
    iterations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cancer_parts: list[pd.DataFrame] = []
    scopes = [*TARGET_SUBTYPES, "ALL_TARGETS_MACRO"]
    for scope in scopes:
        current = metrics if scope == "ALL_TARGETS_MACRO" else metrics.loc[
            metrics.target_subtype.astype(str).eq(scope)
        ]
        by_cancer = (
            current.groupby("loco_cancer", observed=True, sort=True)
            .agg(
                best_simple_auprc=("base_auprc", "mean"),
                selected_b2_auprc=("r1_auprc", "mean"),
                mean_delta_auprc=("delta_auprc", "mean"),
                prevalence=("positive_rate", "mean"),
                best_simple_auprc_over_prevalence=(
                    "base_auprc_over_prevalence",
                    "mean",
                ),
                selected_b2_auprc_over_prevalence=(
                    "r1_auprc_over_prevalence",
                    "mean",
                ),
                best_simple_auroc=("base_auroc", "mean"),
                selected_b2_auroc=("r1_auroc", "mean"),
                mean_delta_auroc=("delta_auroc", "mean"),
                best_simple_brier=("base_brier", "mean"),
                selected_b2_brier=("r1_brier", "mean"),
                mean_delta_brier=("delta_brier", "mean"),
                best_simple_ece=("base_ece", "mean"),
                selected_b2_ece=("r1_ece", "mean"),
                mean_delta_ece=("delta_ece", "mean"),
                model_seed_rows=("seed", "size"),
            )
            .reset_index()
        )
        by_cancer.insert(0, "scope", scope)
        cancer_parts.append(by_cancer)
    cancer = pd.concat(cancer_parts, ignore_index=True)

    rows: list[dict[str, Any]] = []
    for offset, scope in enumerate(scopes):
        current = cancer.loc[cancer.scope.eq(scope)].sort_values("loco_cancer")
        if tuple(current.loco_cancer.astype(str)) != PILOT_CANCERS:
            raise RuntimeError(f"{scope} lacks the exact pilot cancer set")
        values = current.mean_delta_auprc.to_numpy(dtype=float)
        low, median, high = _cancer_cluster_bootstrap(
            values,
            iterations=iterations,
            seed=seed + offset,
        )
        mean_delta = float(values.mean())
        positive = int(np.sum(values > 0))
        rows.append(
            {
                "scope": scope,
                "best_simple_auprc": float(current.best_simple_auprc.mean()),
                "selected_b2_auprc": float(current.selected_b2_auprc.mean()),
                "mean_delta_auprc": mean_delta,
                "cancer_cluster_bootstrap_ci95_lower": low,
                "cancer_cluster_bootstrap_median": median,
                "cancer_cluster_bootstrap_ci95_upper": high,
                "positive_cancers": positive,
                "total_cancers": len(PILOT_CANCERS),
                "prevalence": float(current.prevalence.mean()),
                "best_simple_auprc_over_prevalence": float(
                    current.best_simple_auprc_over_prevalence.mean()
                ),
                "selected_b2_auprc_over_prevalence": float(
                    current.selected_b2_auprc_over_prevalence.mean()
                ),
                "best_simple_auroc": float(current.best_simple_auroc.mean()),
                "selected_b2_auroc": float(current.selected_b2_auroc.mean()),
                "mean_delta_auroc": float(current.mean_delta_auroc.mean()),
                "best_simple_brier": float(current.best_simple_brier.mean()),
                "selected_b2_brier": float(current.selected_b2_brier.mean()),
                "mean_delta_brier": float(current.mean_delta_brier.mean()),
                "best_simple_ece": float(current.best_simple_ece.mean()),
                "selected_b2_ece": float(current.selected_b2_ece.mean()),
                "mean_delta_ece": float(current.mean_delta_ece.mean()),
                "meets_delta_threshold": bool(
                    mean_delta >= HARD_MIN_MEAN_DELTA_AUPRC
                ),
                "meets_ci_threshold": bool(low > 0),
                "meets_cancer_direction_threshold": bool(positive >= 2),
            }
        )
    return cancer, pd.DataFrame(rows)


def build_b2_hard_report(
    predictions: pd.DataFrame,
    admission: pd.DataFrame,
    fallback: pd.DataFrame,
    b2_gate: dict[str, Any],
    task_audits: pd.DataFrame,
    *,
    iterations: int = 10000,
    seed: int = 20260815,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build metrics and the strict B2 -> B3 decision without test tuning."""

    require_columns(
        predictions,
        [
            "candidate_id",
            "loco_cancer",
            "seed",
            "target_subtype",
            "split",
            "prediction_scale",
            "proxy_label",
            "base_probability",
            "proxy_positive_probability",
            "module_admitted",
            "test_metric_used_for_selection",
        ],
        "B2 selected predictions",
    )
    require_columns(
        admission,
        [
            "target_subtype",
            "admitted",
            "validation_delta_auprc",
            "validation_delta_brier",
            "validation_delta_ece",
            "minimum_delta_auprc",
            "maximum_brier_worsening",
            "maximum_ece_worsening",
            "test_metric_used_for_selection",
        ],
        "B2 admission",
    )
    require_columns(
        task_audits,
        [
            "loco_cancer",
            "seed",
            "residual_shrinkage_lambda",
            "status",
            "residual_initialization_max_abs",
        ],
        "B2 task audits",
    )

    failures: list[str] = []
    if b2_gate.get("status") != "PASS":
        failures.append("B2 finalizer gate is not PASS")
    if b2_gate.get("tasks_completed") != 27 or b2_gate.get("tasks_expected") != 27:
        failures.append("B2 task matrix is not exactly 27/27")
    if b2_gate.get("selection_used_test_labels") is not False:
        failures.append("B2 gate does not prove test-independent selection")
    if not predictions.test_metric_used_for_selection.astype(bool).eq(False).all():
        failures.append("prediction rows claim test-driven selection")
    if admission.test_metric_used_for_selection.astype(bool).any():
        failures.append("admission used test metrics")

    test = predictions.loc[
        predictions.split.astype(str).eq("test")
        & predictions.prediction_scale.astype(str).eq("raw_probability")
    ].copy()
    key = ["loco_cancer", "seed", "target_subtype", "candidate_id"]
    if test.empty:
        failures.append("no raw test predictions")
    if test.duplicated(key).any():
        failures.append("duplicate raw test candidate keys")
    if set(test.loco_cancer.astype(str)) != set(PILOT_CANCERS):
        failures.append("test cancer set mismatch")
    if set(test.seed.astype(int)) != set(PILOT_SEEDS):
        failures.append("model seed set mismatch")
    if set(test.target_subtype.astype(str)) != set(TARGET_SUBTYPES):
        failures.append("target subtype set mismatch")
    for (cancer_name, subtype), group in test.groupby(
        ["loco_cancer", "target_subtype"], observed=True, sort=True
    ):
        candidate_audit = group.groupby("candidate_id", observed=True).agg(
            seed_count=("seed", "nunique"),
            label_count=("proxy_label", "nunique"),
        )
        if (
            not candidate_audit.seed_count.eq(len(PILOT_SEEDS)).all()
            or not candidate_audit.label_count.eq(1).all()
        ):
            failures.append(
                f"{cancer_name}/{subtype} candidate keys or labels drift across seeds"
            )
        if group.module_admitted.astype(bool).nunique() != 1:
            failures.append(
                f"{cancer_name}/{subtype} module admission drifts within test rows"
            )
    for column in ("base_probability", "proxy_positive_probability"):
        values = pd.to_numeric(test[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            failures.append(f"{column} contains invalid probabilities")
    if any(group.proxy_label.nunique() != 2 for _, group in test.groupby(key[:3])):
        failures.append("at least one test metric group lacks both label classes")

    admitted_subtypes = set(
        admission.loc[admission.admitted.astype(bool), "target_subtype"].astype(str)
    )
    calibrated = predictions.loc[
        predictions.split.astype(str).eq("test")
        & predictions.prediction_scale.astype(str).eq("calibrated_probability")
    ].copy()
    observed_calibrated_subtypes = set(calibrated.target_subtype.astype(str))
    if observed_calibrated_subtypes != admitted_subtypes:
        failures.append("calibrated test subtype set does not match admitted modules")
    calibration_rank_error = 0.0
    for subtype in sorted(admitted_subtypes):
        raw_subtype = test.loc[test.target_subtype.astype(str).eq(subtype)]
        calibrated_subtype = calibrated.loc[
            calibrated.target_subtype.astype(str).eq(subtype)
        ]
        signature_columns = [*key, "proxy_label"]
        raw_signature = raw_subtype[signature_columns].sort_values(key).reset_index(drop=True)
        calibrated_signature = (
            calibrated_subtype[signature_columns].sort_values(key).reset_index(drop=True)
        )
        if not raw_signature.equals(calibrated_signature):
            failures.append(f"{subtype} raw/calibrated candidate or label mismatch")
            continue
        for group_keys, raw_group in raw_subtype.groupby(key[:3], observed=True, sort=True):
            calibrated_group = calibrated_subtype
            for column, value in zip(key[:3], group_keys):
                calibrated_group = calibrated_group.loc[
                    calibrated_group[column].astype(str).eq(str(value))
                ]
            raw_group = raw_group.sort_values("candidate_id")
            calibrated_group = calibrated_group.sort_values("candidate_id")
            raw_metric = binary_metrics(
                raw_group.proxy_label, raw_group.proxy_positive_probability
            )
            calibrated_metric = binary_metrics(
                calibrated_group.proxy_label,
                calibrated_group.proxy_positive_probability,
            )
            calibration_rank_error = max(
                calibration_rank_error,
                abs(raw_metric["auprc"] - calibrated_metric["auprc"]),
                abs(raw_metric["auroc"] - calibrated_metric["auroc"]),
            )
    if calibration_rank_error > 1e-10:
        failures.append(
            "within-group calibration changed AUROC/AUPRC ranking metrics: "
            f"max_abs={calibration_rank_error}"
        )

    admission_pass = (
        (~admission.admitted.astype(bool))
        | (
            (admission.validation_delta_auprc >= admission.minimum_delta_auprc)
            & (
                admission.validation_delta_brier
                <= admission.maximum_brier_worsening
            )
            & (admission.validation_delta_ece <= admission.maximum_ece_worsening)
        )
    )
    if not admission_pass.all():
        failures.append("an admitted subtype violates validation calibration limits")

    off = test.loc[~test.module_admitted.astype(bool)]
    off_error = (
        float(
            np.max(
                np.abs(
                    off.base_probability.to_numpy(dtype=float)
                    - off.proxy_positive_probability.to_numpy(dtype=float)
                )
            )
        )
        if len(off)
        else 0.0
    )
    if off_error > FALLBACK_ATOL:
        failures.append("selected OFF rows do not exactly equal B0")
    if len(fallback):
        require_columns(
            fallback,
            ["status", "max_abs_probability_error", "tolerance"],
            "B2 fallback audit",
        )
        if not fallback.status.astype(str).eq("PASS").all():
            failures.append("finalizer fallback audit failed")
        if (fallback.max_abs_probability_error.astype(float) > FALLBACK_ATOL).any():
            failures.append("finalizer fallback error exceeds 1e-7")

    expected_task_rows = 27
    if len(task_audits) != expected_task_rows:
        failures.append("task audit table is not exactly 27 rows")
    task_key = ["loco_cancer", "seed", "residual_shrinkage_lambda"]
    if task_audits.duplicated(task_key).any():
        failures.append("task audit table contains duplicate matrix keys")
    if set(task_audits.loco_cancer.astype(str)) != set(PILOT_CANCERS):
        failures.append("task audit cancer set mismatch")
    if set(task_audits.seed.astype(int)) != set(PILOT_SEEDS):
        failures.append("task audit seed set mismatch")
    observed_lambdas = set(
        pd.to_numeric(
            task_audits.residual_shrinkage_lambda, errors="coerce"
        ).astype(float)
    )
    if observed_lambdas != set(SHRINKAGE_GRID):
        failures.append("task audit shrinkage grid mismatch")
    if not task_audits.status.astype(str).eq("PASS").all():
        failures.append("at least one task audit is not PASS")
    if (
        pd.to_numeric(task_audits.residual_initialization_max_abs, errors="coerce")
        > FALLBACK_ATOL
    ).any():
        failures.append("at least one residual head violates zero initialization")

    if failures:
        raise RuntimeError("B2 HARD REPORT integrity failure: " + "; ".join(failures))

    metrics = _group_metrics(test)
    cancer, summary = _summaries(metrics, iterations=iterations, seed=seed)
    # Pathway is the atlas discovery task and therefore controls B3 admission.
    # RNAss/DNAss remain separately reported diagnostics and cannot compensate
    # for an insufficient practical gain on the primary Pathway task.
    primary = summary.loc[summary.scope.eq("Pathway")].iloc[0]
    calibration_gate = bool(admission_pass.all())
    fallback_gate = bool(off_error <= FALLBACK_ATOL)
    integrity_gate = True
    hard_go = bool(
        primary.meets_delta_threshold
        and primary.meets_ci_threshold
        and primary.meets_cancer_direction_threshold
        and calibration_gate
        and fallback_gate
        and integrity_gate
    )
    payload = {
        "status": "PASS",
        "stage": "B2_HARD_REPORT",
        "decision": "HARD_GO_B3" if hard_go else "HARD_STOP_AFTER_B2",
        "b3_authorized": hard_go,
        "primary_scope": "Pathway",
        "primary_aggregation": (
            "macro mean of paired selected-B2 minus BestSimple Pathway AUPRC "
            "deltas across model seed and cancer; cancer is the bootstrap cluster"
        ),
        "primary_best_simple_auprc": float(primary.best_simple_auprc),
        "primary_selected_b2_auprc": float(primary.selected_b2_auprc),
        "primary_mean_delta_auprc": float(primary.mean_delta_auprc),
        "cluster_aware_ci95_lower": float(
            primary.cancer_cluster_bootstrap_ci95_lower
        ),
        "cluster_aware_ci95_upper": float(
            primary.cancer_cluster_bootstrap_ci95_upper
        ),
        "positive_cancers": int(primary.positive_cancers),
        "total_cancers": int(primary.total_cancers),
        "primary_prevalence": float(primary.prevalence),
        "primary_best_simple_auprc_over_prevalence": float(
            primary.best_simple_auprc_over_prevalence
        ),
        "primary_selected_b2_auprc_over_prevalence": float(
            primary.selected_b2_auprc_over_prevalence
        ),
        "primary_best_simple_auroc": float(primary.best_simple_auroc),
        "primary_selected_b2_auroc": float(primary.selected_b2_auroc),
        "primary_mean_delta_auroc": float(primary.mean_delta_auroc),
        "primary_best_simple_brier": float(primary.best_simple_brier),
        "primary_selected_b2_brier": float(primary.selected_b2_brier),
        "primary_mean_delta_brier": float(primary.mean_delta_brier),
        "primary_best_simple_ece": float(primary.best_simple_ece),
        "primary_selected_b2_ece": float(primary.selected_b2_ece),
        "primary_mean_delta_ece": float(primary.mean_delta_ece),
        "hard_thresholds": {
            "minimum_mean_delta_auprc": HARD_MIN_MEAN_DELTA_AUPRC,
            "ci95_lower_must_exceed": 0.0,
            "minimum_positive_cancers": 2,
        },
        "gates": {
            "delta_auprc": bool(primary.meets_delta_threshold),
            "cluster_aware_ci": bool(primary.meets_ci_threshold),
            "cancer_direction": bool(primary.meets_cancer_direction_threshold),
            "calibration": calibration_gate,
            "exact_fallback": fallback_gate,
            "integrity": integrity_gate,
        },
        "selection_scope": "pooled_outer_validation",
        "test_metrics_used_for_lambda_or_module_admission": False,
        "test_metrics_used_only_for_b3_hard_gate": True,
        "bootstrap_iterations": int(iterations),
        "bootstrap_seed": int(seed),
        "off_max_abs_probability_error": off_error,
        "raw_calibrated_rank_metric_max_abs_delta": calibration_rank_error,
        "full_cancer_training_authorized": False,
        "failures": [],
    }
    return metrics, cancer, summary, payload
