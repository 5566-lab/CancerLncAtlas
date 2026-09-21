#!/usr/bin/env python3
"""Compare train-fold LASSO coefficients with graph/patient scores on held-out-patient association labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score

from cc_hhgt.common import write_json, write_table


TARGET_STATES = ["stemness_rna::RNAss", "stemness_dna::DNAss"]
PREDICTION_COLUMNS = [
    "state_patient_probability",
    "state_rgcn_probability",
    "state_hgt_probability",
    "state_cc_hhgt_strict_probability",
]
SCORE_COLUMNS = {
    "lasso_abs_coefficient": "lasso_abs_coefficient",
    "train_abs_effect": "train_abs_effect",
    "patient_head": "state_patient_probability",
    "rgcn_head": "state_rgcn_probability",
    "hgt_head": "state_hgt_probability",
    "cc_hhgt_strict_head": "state_cc_hhgt_strict_probability",
    "graph_heads_equal": "graph_heads_equal_probability",
    "all_heads_equal": "all_heads_equal_probability",
}


def safe_binary_metrics(labels: np.ndarray, score: np.ndarray) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=float)
    score = np.asarray(score, dtype=float)
    valid = np.isfinite(labels) & np.isfinite(score) & np.isin(labels, [0.0, 1.0])
    labels = labels[valid]
    score = score[valid]
    result: dict[str, float | int] = {
        "n": int(len(labels)),
        "n_positive": int(labels.sum()),
        "positive_rate": float(labels.mean()) if len(labels) else float("nan"),
        "auroc": float("nan"),
        "auprc": float("nan"),
    }
    if len(labels) and np.unique(labels).size == 2:
        result["auroc"] = float(roc_auc_score(labels, score))
        result["auprc"] = float(average_precision_score(labels, score))
    return result


def completed_lasso_tasks(root: Path) -> set[tuple[str, str]]:
    tasks: set[tuple[str, str]] = set()
    for path in root.glob("*/*/SUCCESS.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") == "COMPLETED":
            tasks.add((str(payload["cancer_id"]), str(payload["state_id"])))
    return tasks


def collect_patient_predictions(root: Path) -> tuple[pd.DataFrame, list[str]]:
    paths = sorted(root.rglob("prediction.parquet"))
    if not paths:
        raise RuntimeError(f"No patient-fold prediction.parquet files under {root}")
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_parquet(
            path,
            columns=[
                "cancer_id",
                "patient_fold_id",
                "lncrna_id",
                "state_id",
                "seed",
                "train_effect",
                "test_membership_label",
                *PREDICTION_COLUMNS,
            ],
        )
        frame = frame.loc[frame.state_id.astype(str).isin(TARGET_STATES)]
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise RuntimeError("Patient prediction files contain no RNAss or DNAss rows")
    raw = pd.concat(frames, ignore_index=True)
    keys = ["cancer_id", "patient_fold_id", "lncrna_id", "state_id"]
    label_conflicts = (
        raw.groupby(keys, observed=True).test_membership_label.nunique(dropna=False).gt(1)
    )
    if label_conflicts.any():
        raise RuntimeError(f"Patient-fold membership labels disagree across seeds for {int(label_conflicts.sum())} rows")
    aggregated = (
        raw.groupby(keys, observed=True)
        .agg(
            test_membership_label=("test_membership_label", "first"),
            train_effect=("train_effect", "mean"),
            n_model_seeds=("seed", "nunique"),
            **{column: (column, "mean") for column in PREDICTION_COLUMNS},
        )
        .reset_index()
    )
    return aggregated, [str(path) for path in paths]


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    summary = (
        metrics.groupby(["model_name", "state_id"], observed=True)
        .agg(
            n_fold_units=("auroc", "size"),
            n_valid_fold_units=("auroc", "count"),
            mean_auroc=("auroc", "mean"),
            median_auroc=("auroc", "median"),
            std_auroc=("auroc", "std"),
            mean_auprc=("auprc", "mean"),
            median_auprc=("auprc", "median"),
        )
        .reset_index()
    )
    threshold = (
        metrics.assign(ge_060=metrics.auroc.ge(0.60).where(metrics.auroc.notna()))
        .groupby(["model_name", "state_id"], observed=True)
        .ge_060.mean()
        .rename("fraction_fold_units_auroc_ge_060")
        .reset_index()
    )
    return summary.merge(threshold, on=["model_name", "state_id"], how="left")


def paired_deltas(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = ["cancer_id", "patient_fold_id", "state_id"]
    pivot = metrics.pivot(index=index, columns="model_name", values="auroc")
    if "lasso_abs_coefficient" not in pivot.columns:
        raise RuntimeError("LASSO metric column is absent")
    rows: list[pd.DataFrame] = []
    tests: list[dict] = []
    for model_name in sorted(set(pivot.columns) - {"lasso_abs_coefficient"}):
        paired = pivot[["lasso_abs_coefficient", model_name]].dropna().copy()
        paired["delta_auroc_vs_lasso"] = paired[model_name] - paired["lasso_abs_coefficient"]
        paired = paired.reset_index()
        paired["model_name"] = model_name
        rows.append(paired[index + ["model_name", "lasso_abs_coefficient", model_name, "delta_auroc_vs_lasso"]].rename(columns={model_name: "model_auroc"}))
        for state_id, group in paired.groupby("state_id", observed=True):
            delta = group.delta_auroc_vs_lasso.to_numpy(float)
            nonzero = delta[np.isfinite(delta) & ~np.isclose(delta, 0.0)]
            p_value = float(wilcoxon(nonzero).pvalue) if len(nonzero) else float("nan")
            tests.append(
                {
                    "model_name": model_name,
                    "state_id": str(state_id),
                    "n_paired": int(len(group)),
                    "mean_delta_auroc_vs_lasso": float(group.delta_auroc_vs_lasso.mean()),
                    "median_delta_auroc_vs_lasso": float(group.delta_auroc_vs_lasso.median()),
                    "wilcoxon_p_value": p_value,
                }
            )
    return pd.concat(rows, ignore_index=True), pd.DataFrame(tests)


def summarize_patient_score_lasso(lasso_root: Path) -> pd.DataFrame:
    path = lasso_root / "lasso_fold_metrics.tsv"
    if not path.exists():
        raise RuntimeError(f"Missing {path}")
    frame = pd.read_csv(path, sep="\t")
    return (
        frame.groupby("state_id", observed=True)
        .agg(
            n_fold_units=("test_r2", "size"),
            median_test_r2=("test_r2", "median"),
            mean_test_r2=("test_r2", "mean"),
            median_test_pearson=("test_pearson", "median"),
            median_test_spearman=("test_spearman", "median"),
            median_test_high_low_auroc=("test_high_low_auroc", "median"),
            mean_test_high_low_auroc=("test_high_low_auroc", "mean"),
        )
        .reset_index()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Matched held-out-patient comparison of LASSO and state graph scores")
    parser.add_argument("--lasso-root", required=True)
    parser.add_argument("--patient-prediction-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-seeds", type=int, default=3)
    args = parser.parse_args()

    lasso_root = Path(args.lasso_root).resolve()
    patient_root = Path(args.patient_prediction_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    lasso_audit = json.loads((lasso_root / "LASSO_BASELINE_AUDIT.json").read_text(encoding="utf-8"))
    if lasso_audit.get("status") != "PASS":
        raise RuntimeError("LASSO baseline audit did not PASS")
    coefficients = pd.read_parquet(lasso_root / "lasso_coefficients.parquet")
    coefficients = coefficients.loc[coefficients.state_id.astype(str).isin(TARGET_STATES)].copy()
    completed = completed_lasso_tasks(lasso_root)
    patient, source_paths = collect_patient_predictions(patient_root)
    patient = patient.loc[
        [(str(cancer), str(state)) in completed for cancer, state in zip(patient.cancer_id, patient.state_id, strict=True)]
    ].copy()
    keys = ["cancer_id", "patient_fold_id", "lncrna_id", "state_id"]
    coefficient_columns = keys + ["absolute_lasso_coefficient", "standardized_lasso_coefficient", "selected_by_lasso"]
    duplicate_coefficients = int(coefficients.duplicated(keys).sum())
    if duplicate_coefficients:
        raise RuntimeError(f"Duplicate LASSO coefficient keys: {duplicate_coefficients}")
    matched = patient.merge(coefficients[coefficient_columns], on=keys, how="left")
    matched["lasso_abs_coefficient"] = matched.absolute_lasso_coefficient.fillna(0.0)
    matched["lasso_signed_coefficient"] = matched.standardized_lasso_coefficient.fillna(0.0)
    matched["selected_by_lasso"] = matched.selected_by_lasso.eq(True)
    matched["train_abs_effect"] = matched.train_effect.abs()
    matched["graph_heads_equal_probability"] = matched[
        ["state_rgcn_probability", "state_hgt_probability", "state_cc_hhgt_strict_probability"]
    ].mean(axis=1, skipna=False)
    matched["all_heads_equal_probability"] = matched[PREDICTION_COLUMNS].mean(axis=1, skipna=False)
    matched["common_score_complete"] = matched[list(SCORE_COLUMNS.values())].notna().all(axis=1)

    metric_rows: list[dict] = []
    for (cancer_id, fold_id, state_id), group in matched.groupby(
        ["cancer_id", "patient_fold_id", "state_id"], observed=True
    ):
        # Every score, including LASSO, is evaluated on the identical candidate rows.
        group = group.loc[group.common_score_complete]
        labels = group.test_membership_label.to_numpy(float)
        for model_name, score_column in SCORE_COLUMNS.items():
            metric_rows.append(
                {
                    "cancer_id": str(cancer_id),
                    "patient_fold_id": str(fold_id),
                    "state_id": str(state_id),
                    "model_name": model_name,
                    **safe_binary_metrics(labels, group[score_column].to_numpy(float)),
                }
            )
    metrics = pd.DataFrame(metric_rows)
    summary = summarize_metrics(metrics)
    deltas, paired_tests = paired_deltas(metrics)
    patient_score_summary = summarize_patient_score_lasso(lasso_root)

    write_table(matched, output_dir / "matched_patient_fold_association_prediction.parquet")
    metrics.to_csv(output_dir / "matched_patient_fold_association_metrics.tsv", sep="\t", index=False)
    summary.to_csv(output_dir / "matched_patient_fold_association_summary.tsv", sep="\t", index=False)
    deltas.to_csv(output_dir / "paired_auroc_deltas_vs_lasso.tsv", sep="\t", index=False)
    paired_tests.to_csv(output_dir / "paired_auroc_tests_vs_lasso.tsv", sep="\t", index=False)
    patient_score_summary.to_csv(output_dir / "lasso_patient_score_summary.tsv", sep="\t", index=False)

    incomplete_seeds = int(matched.n_model_seeds.ne(args.expected_seeds).sum())
    expected_units = len(completed) * 5
    observed_units = int(matched[["cancer_id", "patient_fold_id", "state_id"]].drop_duplicates().shape[0])
    missing_scores = {name: int(matched[column].isna().sum()) for name, column in SCORE_COLUMNS.items()}
    core_missing = sum(missing_scores[name] for name in ["lasso_abs_coefficient", "train_abs_effect", "patient_head"])
    common_complete_rows = int(matched.common_score_complete.sum())
    common_complete_fraction = float(matched.common_score_complete.mean()) if len(matched) else 0.0
    audit = {
        "status": "PASS"
        if not incomplete_seeds
        and observed_units == expected_units
        and core_missing == 0
        and common_complete_fraction >= 0.80
        else "FAIL",
        "evaluation_unit": "cancer x patient_fold x lncRNA x state association replication",
        "label": "association membership computed only from held-out patients",
        "lasso_score": "absolute standardized coefficient fit only on outer-train patients; unscreened features score zero",
        "patient_model_score": "three-seed mean from the requested patient-prediction root",
        "completed_lasso_tasks": len(completed),
        "expected_fold_state_units": expected_units,
        "observed_fold_state_units": observed_units,
        "matched_rows": int(len(matched)),
        "common_complete_rows": common_complete_rows,
        "common_complete_fraction": common_complete_fraction,
        "incomplete_model_seed_rows": incomplete_seeds,
        "missing_scores": missing_scores,
        "patient_prediction_root": str(patient_root),
        "patient_prediction_files": len(source_paths),
        "lasso_root": str(lasso_root),
    }
    write_json(audit, output_dir / "MATCHED_LASSO_GRAPH_AUDIT.json")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if audit["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
