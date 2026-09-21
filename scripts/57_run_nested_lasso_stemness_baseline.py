#!/usr/bin/env python3
"""Rebuild a leakage-controlled patient-level LASSO baseline for RNAss and DNAss."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from cc_hhgt.common import write_json, write_table


TARGETS = {
    "RNAss": "stemness_rna::RNAss",
    "DNAss": "stemness_dna::DNAss",
}


def finite_pearson(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3 or np.std(left[valid]) == 0 or np.std(right[valid]) == 0:
        return float("nan")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def finite_spearman(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3 or np.unique(left[valid]).size < 2 or np.unique(right[valid]).size < 2:
        return float("nan")
    return float(spearmanr(left[valid], right[valid]).statistic)


def screen_features(train: np.ndarray, target: np.ndarray, max_features: int) -> tuple[np.ndarray, np.ndarray]:
    # Spearman screening is performed inside each outer training fold only.
    ranked_train = np.apply_along_axis(rankdata, 0, train)
    ranked_target = rankdata(target)
    centered_x = ranked_train - ranked_train.mean(axis=0, keepdims=True)
    centered_y = ranked_target - ranked_target.mean()
    denominator = np.sqrt(np.square(centered_x).sum(axis=0) * np.square(centered_y).sum())
    correlation = np.divide(
        centered_x.T @ centered_y,
        denominator,
        out=np.zeros(train.shape[1], dtype=float),
        where=denominator > 0,
    )
    order = np.argsort(-np.abs(correlation), kind="stable")
    keep = order[: min(max_features, len(order))]
    return keep, correlation[keep]


def fold_assignments(manifest: pd.DataFrame, fold_id: str) -> pd.Series:
    fold = manifest.loc[manifest.patient_fold_id.astype(str).eq(str(fold_id)), ["patient_id", "split"]].copy()
    conflict = fold.groupby("patient_id", observed=True).split.nunique()
    if conflict.gt(1).any():
        raise RuntimeError(f"Conflicting split assignments in {fold_id}")
    return fold.drop_duplicates("patient_id").set_index("patient_id").split.astype(str)


def fit_outer_fold(
    cancer_id: str,
    state_id: str,
    fold_id: str,
    expression: pd.DataFrame,
    target: pd.Series,
    assignments: pd.Series,
    max_features: int,
    alphas: int,
    min_train: int,
    min_test: int,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    patients = expression.index.intersection(target.dropna().index).intersection(assignments.index)
    train_ids = patients[assignments.reindex(patients).eq("train")]
    test_ids = patients[assignments.reindex(patients).eq("test")]
    if len(train_ids) < min_train or len(test_ids) < min_test:
        raise RuntimeError(
            f"Insufficient patients for {cancer_id}/{state_id}/{fold_id}: "
            f"train={len(train_ids)}, test={len(test_ids)}"
        )
    x_train = expression.loc[train_ids].to_numpy(float)
    x_test = expression.loc[test_ids].to_numpy(float)
    y_train = target.loc[train_ids].to_numpy(float)
    y_test = target.loc[test_ids].to_numpy(float)

    medians = np.nanmedian(x_train, axis=0)
    medians[~np.isfinite(medians)] = 0.0
    train_missing = ~np.isfinite(x_train)
    test_missing = ~np.isfinite(x_test)
    if train_missing.any():
        x_train[train_missing] = np.take(medians, np.where(train_missing)[1])
    if test_missing.any():
        x_test[test_missing] = np.take(medians, np.where(test_missing)[1])
    variable = np.nanstd(x_train, axis=0) > 1e-8
    if not variable.any():
        raise RuntimeError(f"No variable lncRNA features for {cancer_id}/{state_id}/{fold_id}")
    variable_indices = np.flatnonzero(variable)
    screened_local, screening_correlation = screen_features(x_train[:, variable], y_train, max_features)
    selected_indices = variable_indices[screened_local]

    scaler = StandardScaler()
    z_train = scaler.fit_transform(x_train[:, selected_indices])
    z_test = scaler.transform(x_test[:, selected_indices])
    inner_splits = min(5, max(2, len(train_ids) // 10))
    inner_cv = KFold(n_splits=inner_splits, shuffle=True, random_state=20260810)
    model = LassoCV(
        alphas=alphas,
        cv=inner_cv,
        max_iter=20_000,
        tol=1e-5,
        n_jobs=1,
        selection="cyclic",
    )
    model.fit(z_train, y_train)
    prediction = model.predict(z_test)
    threshold = float(np.median(y_train))
    high_label = y_test >= threshold
    high_low_auroc = float(roc_auc_score(high_label, prediction)) if np.unique(high_label).size == 2 else float("nan")
    rmse = float(np.sqrt(mean_squared_error(y_test, prediction)))
    metrics = {
        "cancer_id": cancer_id,
        "state_id": state_id,
        "patient_fold_id": fold_id,
        "n_train": int(len(train_ids)),
        "n_test": int(len(test_ids)),
        "n_input_features": int(expression.shape[1]),
        "n_variable_features": int(variable.sum()),
        "n_screened_features": int(len(selected_indices)),
        "n_nonzero_features": int(np.count_nonzero(np.abs(model.coef_) > 1e-12)),
        "inner_cv_splits": int(inner_splits),
        "alpha": float(model.alpha_),
        "train_median_threshold": threshold,
        "test_r2": float(r2_score(y_test, prediction)),
        "test_rmse": rmse,
        "test_pearson": finite_pearson(y_test, prediction),
        "test_spearman": finite_spearman(y_test, prediction),
        "test_high_low_auroc": high_low_auroc,
    }
    predictions = pd.DataFrame(
        {
            "cancer_id": cancer_id,
            "state_id": state_id,
            "patient_fold_id": fold_id,
            "patient_id": test_ids.astype(str),
            "observed_score": y_test,
            "predicted_score": prediction,
            "train_median_threshold": threshold,
            "observed_high_label": high_label.astype(int),
        }
    )
    coefficients = pd.DataFrame(
        {
            "cancer_id": cancer_id,
            "state_id": state_id,
            "patient_fold_id": fold_id,
            "lncrna_id": expression.columns[selected_indices].astype(str),
            "screening_spearman": screening_correlation,
            "standardized_lasso_coefficient": model.coef_,
        }
    )
    coefficients["absolute_lasso_coefficient"] = coefficients.standardized_lasso_coefficient.abs()
    coefficients["selected_by_lasso"] = coefficients.absolute_lasso_coefficient.gt(1e-12)
    return metrics, predictions, coefficients


def write_skipped(output_dir: Path, cancer_id: str, state_id: str, reason: str, n_patients: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "SKIPPED",
        "cancer_id": cancer_id,
        "state_id": state_id,
        "reason": reason,
        "n_target_patients": int(n_patients),
    }
    write_json(payload, output_dir / "SKIPPED.json")
    return payload


def run_target(
    cancer_id: str,
    state_id: str,
    expression: pd.DataFrame,
    scores: pd.DataFrame,
    manifest: pd.DataFrame,
    result_root: Path,
    max_features: int,
    alphas: int,
    min_total: int,
    min_train: int,
    min_test: int,
    resume: bool,
) -> dict:
    output_dir = result_root / cancer_id / state_id.split("::")[-1]
    success_path = output_dir / "SUCCESS.json"
    skipped_path = output_dir / "SKIPPED.json"
    if resume and success_path.exists():
        return json.loads(success_path.read_text(encoding="utf-8"))
    if resume and skipped_path.exists():
        return json.loads(skipped_path.read_text(encoding="utf-8"))
    cancer_scores = scores.loc[scores.cancer_id.eq(cancer_id), ["patient_id", state_id]].dropna()
    target = cancer_scores.groupby("patient_id", observed=True)[state_id].median()
    target.index = target.index.astype(str)
    available = expression.index.intersection(target.index)
    if len(available) < min_total:
        return write_skipped(output_dir, cancer_id, state_id, f"fewer than {min_total} matched patients", len(available))
    target = target.reindex(available)
    cancer_manifest = manifest.loc[manifest.cancer_id.eq(cancer_id)].copy()
    folds = sorted(cancer_manifest.patient_fold_id.astype(str).unique())
    metric_rows: list[dict] = []
    predictions: list[pd.DataFrame] = []
    coefficients: list[pd.DataFrame] = []
    for fold_id in folds:
        assignment = fold_assignments(cancer_manifest, fold_id)
        metrics, fold_prediction, fold_coefficients = fit_outer_fold(
            cancer_id,
            state_id,
            fold_id,
            expression,
            target,
            assignment,
            max_features,
            alphas,
            min_train,
            min_test,
        )
        metric_rows.append(metrics)
        predictions.append(fold_prediction)
        coefficients.append(fold_coefficients)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(output_dir / "fold_metrics.tsv", sep="\t", index=False)
    write_table(pd.concat(predictions, ignore_index=True), output_dir / "patient_oof_prediction.parquet")
    write_table(pd.concat(coefficients, ignore_index=True), output_dir / "lasso_coefficients.parquet")
    payload = {
        "status": "COMPLETED",
        "cancer_id": cancer_id,
        "state_id": state_id,
        "n_target_patients": int(len(available)),
        "n_outer_folds": int(len(folds)),
        "screening_policy": "outer-train-only Spearman",
        "model_policy": "outer-train-only standardized LassoCV with shuffled inner KFold",
        "max_features": int(max_features),
        "alphas": int(alphas),
    }
    write_json(payload, success_path)
    return payload


def run_cancer(payload: dict) -> list[dict]:
    cancer_id = payload["cancer_id"]
    expression_path = Path(payload["expression_root"]) / f"cancer_id={cancer_id}" / "part-0.parquet"
    long = pd.read_parquet(expression_path, columns=["patient_id", "lncrna_id", "logcpm"])
    expression = long.pivot_table(index="patient_id", columns="lncrna_id", values="logcpm", aggfunc="mean")
    expression.index = expression.index.astype(str)
    expression.columns = expression.columns.astype(str)
    scores = pd.read_csv(
        payload["score_matrix"],
        sep="\t",
        usecols=["patient_id", "cancer_type", "is_tumor", *payload["state_ids"]],
    )
    scores = scores.loc[scores.is_tumor.eq(True)].rename(columns={"cancer_type": "cancer_id"})
    manifest = pd.read_csv(payload["fold_manifest"], sep="\t", usecols=["cancer_id", "patient_id", "patient_fold_id", "split"])
    results = []
    for state_id in payload["state_ids"]:
        results.append(
            run_target(
                cancer_id,
                state_id,
                expression,
                scores,
                manifest,
                Path(payload["result_root"]),
                payload["max_features"],
                payload["alphas"],
                payload["min_total"],
                payload["min_train"],
                payload["min_test"],
                payload["resume"],
            )
        )
    return results


def collect_outputs(result_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict], list[dict]]:
    metrics = [pd.read_csv(path, sep="\t") for path in result_root.glob("*/*/fold_metrics.tsv")]
    predictions = [pd.read_parquet(path) for path in result_root.glob("*/*/patient_oof_prediction.parquet")]
    coefficients = [pd.read_parquet(path) for path in result_root.glob("*/*/lasso_coefficients.parquet")]
    completed = [json.loads(path.read_text(encoding="utf-8")) for path in result_root.glob("*/*/SUCCESS.json")]
    skipped = [json.loads(path.read_text(encoding="utf-8")) for path in result_root.glob("*/*/SKIPPED.json")]
    return (
        pd.concat(metrics, ignore_index=True) if metrics else pd.DataFrame(),
        pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(),
        pd.concat(coefficients, ignore_index=True) if coefficients else pd.DataFrame(),
        completed,
        skipped,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Nested patient-fold LASSO baseline for RNAss and DNAss")
    parser.add_argument("--input-root", default="input_snapshot")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--cancers", default="all", help="Comma-separated TCGA codes or all")
    parser.add_argument("--targets", default="RNAss,DNAss", help="RNAss,DNAss or full state IDs")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--max-features", type=int, default=500)
    parser.add_argument("--alphas", type=int, default=60)
    parser.add_argument("--min-total", type=int, default=20)
    parser.add_argument("--min-train", type=int, default=15)
    parser.add_argument("--min-test", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    result_root = Path(args.output_root).resolve()
    result_root.mkdir(parents=True, exist_ok=True)
    expression_root = input_root / "parquet" / "bulk_lncRNA_expression"
    score_matrix = input_root / "TCGA_sample_scores" / "03_final_tables" / "TCGA_sample_score_matrix.tsv.gz"
    fold_manifest = input_root / "adapter" / "cancer_specific_patient_fold_manifest.tsv"
    discovered = sorted(path.parent.name.split("=", 1)[1] for path in expression_root.glob("cancer_id=*/part-0.parquet"))
    cancers = discovered if args.cancers.lower() == "all" else [item.strip() for item in args.cancers.split(",") if item.strip()]
    requested_targets = [item.strip() for item in args.targets.split(",") if item.strip()]
    state_ids = [TARGETS.get(item, item) for item in requested_targets]
    unknown = sorted(set(state_ids) - set(TARGETS.values()))
    if unknown:
        raise RuntimeError(f"Unsupported targets: {unknown}")
    missing_cancers = sorted(set(cancers) - set(discovered))
    if missing_cancers:
        raise RuntimeError(f"Missing expression inputs for cancers: {missing_cancers}")

    base_payload = {
        "expression_root": str(expression_root),
        "score_matrix": str(score_matrix),
        "fold_manifest": str(fold_manifest),
        "result_root": str(result_root),
        "state_ids": state_ids,
        "max_features": args.max_features,
        "alphas": args.alphas,
        "min_total": args.min_total,
        "min_train": args.min_train,
        "min_test": args.min_test,
        "resume": args.resume,
    }
    statuses: list[dict] = []
    failures: list[dict] = []
    if args.jobs == 1:
        for index, cancer_id in enumerate(cancers, start=1):
            print(f"[{index}/{len(cancers)}] {cancer_id}", flush=True)
            try:
                statuses.extend(run_cancer({**base_payload, "cancer_id": cancer_id}))
            except Exception:
                failures.append({"cancer_id": cancer_id, "error": traceback.format_exc()})
                print(failures[-1]["error"], flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = {
                executor.submit(run_cancer, {**base_payload, "cancer_id": cancer_id}): cancer_id for cancer_id in cancers
            }
            for index, future in enumerate(as_completed(futures), start=1):
                cancer_id = futures[future]
                try:
                    statuses.extend(future.result())
                    print(f"[{index}/{len(cancers)}] {cancer_id} completed", flush=True)
                except Exception:
                    failures.append({"cancer_id": cancer_id, "error": traceback.format_exc()})
                    print(f"[{index}/{len(cancers)}] {cancer_id} failed", flush=True)

    metrics, predictions, coefficients, completed, skipped = collect_outputs(result_root)
    if not metrics.empty:
        metrics.to_csv(result_root / "lasso_fold_metrics.tsv", sep="\t", index=False)
    if not predictions.empty:
        write_table(predictions, result_root / "lasso_patient_oof_prediction.parquet")
    if not coefficients.empty:
        write_table(coefficients, result_root / "lasso_coefficients.parquet")
    expected_tasks = len(cancers) * len(state_ids)
    audit = {
        "status": "PASS" if not failures and len(completed) + len(skipped) == expected_tasks else "FAIL",
        "expected_tasks": expected_tasks,
        "completed_tasks": len(completed),
        "skipped_tasks": len(skipped),
        "failures": failures,
        "cancers": cancers,
        "state_ids": state_ids,
        "evaluation_unit": "patient",
        "outer_validation": "existing patient_fold_id; all screening and fitting restricted to outer train patients",
    }
    write_json(audit, result_root / "LASSO_BASELINE_AUDIT.json")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if audit["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
