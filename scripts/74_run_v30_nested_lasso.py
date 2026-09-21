#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from cc_hhgt.common import load_config, read_table
from cc_hhgt.io import read_cancer_partition
from cc_hhgt.stats import bh_fdr, correlation_p_values
from cc_hhgt.v30_integrity import atomic_write_json, file_sha256, verify_file_manifest


TARGETS = ("stemness_rna::RNAss", "stemness_dna::DNAss")


def records(path: Path) -> list[dict[str, Any]]:
    frame = pd.read_csv(path, sep="\t")
    return frame.to_dict("records")


def exact_expression(input_root: Path, canonical: pd.DataFrame, cancer: str) -> pd.DataFrame:
    source = read_cancer_partition(input_root / "parquet" / "bulk_lncRNA_expression", cancer)
    mapping = canonical.loc[canonical.cancer_id.astype(str).eq(cancer), ["sample_id", "patient_id"]].copy()
    allowed = set(mapping.sample_id.astype(str))
    source = source.loc[source.sample_id.astype(str).isin(allowed), ["sample_id", "lncrna_id", "logcpm"]].copy()
    if source.duplicated(["sample_id", "lncrna_id"]).any():
        raise RuntimeError(f"Duplicate exact expression rows for {cancer}")
    matrix = source.pivot(index="sample_id", columns="lncrna_id", values="logcpm")
    mapping = mapping.drop_duplicates("sample_id").set_index(mapping.sample_id.astype(str))
    matrix.index = mapping.loc[matrix.index.astype(str), "patient_id"].astype(str).to_numpy()
    if matrix.index.duplicated().any():
        raise RuntimeError(f"Canonical sample contract produced multiple samples for a patient in {cancer}")
    return matrix.sort_index()


def covariate_design(
    train_cov: pd.DataFrame,
    test_cov: pd.DataFrame,
    columns: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    train_parts = [np.ones((len(train_cov), 1), dtype=float)]
    test_parts = [np.ones((len(test_cov), 1), dtype=float)]
    used: list[str] = []
    for column in columns:
        if column not in train_cov:
            continue
        train_col = train_cov[column]
        test_col = test_cov[column] if column in test_cov else pd.Series(index=test_cov.index, dtype=float)
        numeric = pd.to_numeric(train_col, errors="coerce")
        if numeric.notna().mean() >= 0.8:
            median = float(numeric.median()) if numeric.notna().any() else 0.0
            train_value = numeric.fillna(median).to_numpy(float)
            test_value = pd.to_numeric(test_col, errors="coerce").fillna(median).to_numpy(float)
            mean = float(train_value.mean())
            sd = float(train_value.std(ddof=0)) or 1.0
            train_parts.append(((train_value - mean) / sd)[:, None])
            test_parts.append(((test_value - mean) / sd)[:, None])
            used.append(column)
        else:
            text = train_col.astype("string").fillna("__MISSING__")
            levels = sorted(text.unique())
            if len(levels) <= 1:
                continue
            test_text = test_col.astype("string").fillna("__MISSING__")
            # Drop the first train level; unseen test levels map to all zero.
            for level in levels[1:]:
                train_parts.append(text.eq(level).to_numpy(float)[:, None])
                test_parts.append(test_text.eq(level).to_numpy(float)[:, None])
            used.append(column)
    train_design = np.concatenate(train_parts, axis=1)
    test_design = np.concatenate(test_parts, axis=1)
    rank = int(np.linalg.matrix_rank(train_design))
    return train_design, test_design, {"columns": used, "train_design_rank": rank}


def nested_associations(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    train_design: np.ndarray,
    test_design: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    # Expression imputation is train-fitted; state outcomes are never imputed.
    medians = np.nanmedian(x_train, axis=0)
    medians[~np.isfinite(medians)] = 0.0
    x_train = np.where(np.isfinite(x_train), x_train, medians)
    x_test = np.where(np.isfinite(x_test), x_test, medians)
    ranked_train = np.apply_along_axis(rankdata, 0, x_train)
    ranked_test = np.apply_along_axis(rankdata, 0, x_test)
    y_rank_train = rankdata(y_train)
    y_rank_test = rankdata(y_test)
    pinv = np.linalg.pinv(train_design)
    beta_x = pinv @ ranked_train
    beta_y = pinv @ y_rank_train
    residual_train_x = ranked_train - train_design @ beta_x
    residual_test_x = ranked_test - test_design @ beta_x
    residual_train_y = y_rank_train - train_design @ beta_y
    residual_test_y = y_rank_test - test_design @ beta_y

    def correlations(x: np.ndarray, y: np.ndarray, df: int) -> pd.DataFrame:
        x = x - x.mean(axis=0, keepdims=True)
        y = y - y.mean()
        x_sd = x.std(axis=0, ddof=1)
        y_sd = float(y.std(ddof=1))
        effect = np.divide(
            x.T @ y,
            max(len(y) - 1, 1) * x_sd * y_sd,
            out=np.zeros(x.shape[1], dtype=float),
            where=(x_sd > 0) & (y_sd > 0),
        )
        # correlation_p_values expects n and residual design rank.  For test
        # residuals the nuisance coefficients are frozen from outer train, so
        # no test degrees of freedom are spent estimating them.
        rank_for_formula = max(len(y) - df - 1, 0)
        p_value = correlation_p_values(effect, len(y), residual_design_rank=rank_for_formula)
        return pd.DataFrame({"effect": effect, "p_value": p_value, "fdr": bh_fdr(p_value, total_tests=len(p_value)), "df": df})

    train_rank = int(np.linalg.matrix_rank(train_design))
    train_df = len(y_train) - train_rank - 1
    test_df = len(y_test) - 2
    if train_df <= 0 or test_df <= 0:
        raise RuntimeError(f"Non-positive association df: train={train_df}, test={test_df}")
    return correlations(residual_train_x, residual_train_y, train_df), correlations(residual_test_x, residual_test_y, test_df), train_rank


def finite_spearman(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(spearmanr(y, prediction).statistic) if len(y) >= 3 and np.unique(y).size > 1 and np.unique(prediction).size > 1 else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description="Canonical outer-nested LASSO and association replication for RNAss/DNAss")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument(
        "--repo-root",
        help="Relocated frozen code root; contents must match CODE_MANIFEST.tsv",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-features", type=int, default=500)
    parser.add_argument("--alphas", type=int, default=60)
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    cfg = load_config(
        args.config,
        project_root_override=run_root,
        create_dirs=False,
    )
    input_root = Path(args.input_root).resolve()
    input_manifest = Path(args.input_manifest).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise RuntimeError(f"Refusing to reuse nested LASSO output: {output_root}")
    provenance = json.loads((run_root / "provenance" / "PROVENANCE.json").read_text(encoding="utf-8"))
    if provenance["run_id"] != args.run_id or input_manifest != (run_root / "provenance" / "INPUT_MANIFEST.tsv").resolve():
        raise RuntimeError("Nested LASSO run lineage mismatch")
    input_records = records(input_manifest)
    code_records = records(run_root / "provenance" / "CODE_MANIFEST.tsv")
    runtime_repo_root = Path(args.repo_root or provenance["repo_root"]).resolve()
    mismatches = verify_file_manifest(input_root, input_records) + verify_file_manifest(runtime_repo_root, code_records)
    if mismatches:
        raise RuntimeError(f"Frozen assets changed before nested LASSO: {mismatches[:20]}")

    canonical_root = run_root / "canonical"
    asset_tables = run_root / "assets" / "results" / "tables"
    canonical = read_table(canonical_root / "canonical_samples.parquet")
    state_cases = read_table(canonical_root / "canonical_state_complete_cases.parquet")
    patient_folds = read_table(canonical_root / "canonical_patient_fold_manifest.parquet")
    eligibility = read_table(asset_tables / "state_model_eligibility.tsv")
    covariates = read_table(input_root / "processed" / "tcga_association_covariates.parquet")
    covariates = covariates.merge(canonical[["cancer_id", "sample_id", "patient_id"]], on=["cancer_id", "sample_id", "patient_id"], how="inner", validate="one_to_one")
    covariates = covariates.set_index("patient_id")
    formal_cancers = sorted(eligibility.loc[eligibility.cancer_role.eq("FORMAL"), "cancer_id"].astype(str).unique())
    temporary = output_root.parent / f".{output_root.name}.tmp"
    temporary.mkdir(parents=True)
    metric_rows: list[dict] = []
    prediction_rows: list[pd.DataFrame] = []
    coefficient_rows: list[pd.DataFrame] = []
    association_rows: list[pd.DataFrame] = []
    availability_rows: list[dict] = []
    cov_columns = list(cfg["state_graph"].get("covariates", []))

    for cancer in formal_cancers:
        expression = exact_expression(input_root, canonical, cancer)
        cancer_folds = patient_folds.loc[patient_folds.cancer_id.astype(str).eq(cancer)].copy()
        for state_id in TARGETS:
            eligible = eligibility.loc[eligibility.cancer_id.astype(str).eq(cancer) & eligibility.state_id.astype(str).eq(state_id)]
            if len(eligible) != 1:
                raise RuntimeError(f"Missing eligibility {cancer}/{state_id}")
            if str(eligible.iloc[0].evaluation_eligibility) != "ELIGIBLE":
                availability_rows.append({"cancer_id": cancer, "state_id": state_id, "patient_fold_id": None, "status": "UNAVAILABLE", "reason": str(eligible.iloc[0].evaluation_unavailable_reason)})
                continue
            target = state_cases.loc[state_cases.cancer_id.astype(str).eq(cancer) & state_cases.state_id.astype(str).eq(state_id), ["patient_id", "state_value"]]
            if target.duplicated("patient_id").any() or target.state_value.isna().any():
                raise RuntimeError(f"Invalid canonical target {cancer}/{state_id}")
            target = target.set_index("patient_id").state_value.astype(float)
            for fold_id in sorted(cancer_folds.patient_fold_id.astype(str).unique()):
                assignment = cancer_folds.loc[cancer_folds.patient_fold_id.astype(str).eq(fold_id), ["patient_id", "split"]].drop_duplicates()
                if assignment.groupby("patient_id").split.nunique().gt(1).any():
                    raise RuntimeError(f"Conflicting patient split {fold_id}")
                assignment = assignment.set_index("patient_id").split.astype(str)
                patients = expression.index.intersection(target.index).intersection(assignment.index)
                train_ids = patients[assignment.reindex(patients).eq("train")]
                test_ids = patients[assignment.reindex(patients).eq("test")]
                if len(train_ids) < 15 or len(test_ids) < 3:
                    availability_rows.append({"cancer_id": cancer, "state_id": state_id, "patient_fold_id": fold_id, "status": "UNAVAILABLE", "reason": f"PATIENT_COUNT train={len(train_ids)} test={len(test_ids)}"})
                    continue
                raw_train = expression.loc[train_ids].to_numpy(float)
                raw_test = expression.loc[test_ids].to_numpy(float)
                y_train = target.loc[train_ids].to_numpy(float)
                y_test = target.loc[test_ids].to_numpy(float)
                detection = np.nanmean(raw_train > 0, axis=0)
                variance = np.nanvar(raw_train, axis=0, ddof=1)
                eligible_feature = (detection >= float(cfg["state_graph"]["candidate_min_detection"])) & np.isfinite(variance) & (variance > 1e-8)
                indices = np.flatnonzero(eligible_feature)
                if len(indices) > int(cfg["state_graph"]["candidate_max_lncRNAs"]):
                    order = np.argsort(-variance[indices], kind="stable")
                    indices = indices[order[: int(cfg["state_graph"]["candidate_max_lncRNAs"])]]
                feature_ids = expression.columns[indices].astype(str)
                x_train = raw_train[:, indices]
                x_test = raw_test[:, indices]
                train_cov = covariates.reindex(train_ids)
                test_cov = covariates.reindex(test_ids)
                train_design, test_design, design_audit = covariate_design(train_cov, test_cov, cov_columns)
                train_assoc, test_assoc, design_rank = nested_associations(x_train.copy(), x_test.copy(), y_train, y_test, train_design, test_design)
                train_assoc["lncrna_id"] = feature_ids
                test_assoc["lncrna_id"] = feature_ids

                medians = np.nanmedian(x_train, axis=0); medians[~np.isfinite(medians)] = 0.0
                x_train = np.where(np.isfinite(x_train), x_train, medians)
                x_test = np.where(np.isfinite(x_test), x_test, medians)
                ranked_x = np.apply_along_axis(rankdata, 0, x_train)
                ranked_y = rankdata(y_train)
                centered_x = ranked_x - ranked_x.mean(axis=0)
                centered_y = ranked_y - ranked_y.mean()
                denominator = np.sqrt(np.square(centered_x).sum(axis=0) * np.square(centered_y).sum())
                screening = np.divide(centered_x.T @ centered_y, denominator, out=np.zeros(len(feature_ids)), where=denominator > 0)
                selected_local = np.argsort(-np.abs(screening), kind="stable")[: min(args.max_features, len(screening))]
                scaler = StandardScaler().fit(x_train[:, selected_local])
                inner_splits = min(5, max(2, len(train_ids) // 10))
                lasso = LassoCV(alphas=args.alphas, cv=KFold(inner_splits, shuffle=True, random_state=20260810), max_iter=20_000, tol=1e-5, n_jobs=1).fit(scaler.transform(x_train[:, selected_local]), y_train)
                prediction = lasso.predict(scaler.transform(x_test[:, selected_local]))
                threshold = float(np.median(y_train))
                high = y_test >= threshold
                metric_rows.append({"cancer_id": cancer, "state_id": state_id, "patient_fold_id": fold_id, "n_train": len(train_ids), "n_test": len(test_ids), "n_eligible_features": len(feature_ids), "n_screened_features": len(selected_local), "n_nonzero_features": int(np.count_nonzero(np.abs(lasso.coef_) > 1e-12)), "alpha": float(lasso.alpha_), "test_r2": float(r2_score(y_test, prediction)), "test_rmse": float(np.sqrt(mean_squared_error(y_test, prediction))), "test_spearman": finite_spearman(y_test, prediction), "test_high_low_auroc": float(roc_auc_score(high, prediction)) if np.unique(high).size == 2 else np.nan, "outer_test_patients_used_upstream": 0})
                prediction_rows.append(pd.DataFrame({"cancer_id": cancer, "state_id": state_id, "patient_fold_id": fold_id, "patient_id": test_ids.astype(str), "observed_score": y_test, "predicted_score": prediction, "train_median_threshold": threshold, "observed_high_label": high.astype(int)}))
                all_coef = np.zeros(len(feature_ids)); all_coef[selected_local] = lasso.coef_
                coefficients = pd.DataFrame({"cancer_id": cancer, "state_id": state_id, "patient_fold_id": fold_id, "lncrna_id": feature_ids, "screening_spearman": screening, "standardized_lasso_coefficient": all_coef})
                coefficients["absolute_lasso_coefficient"] = coefficients.standardized_lasso_coefficient.abs()
                coefficients["selected_by_lasso"] = coefficients.absolute_lasso_coefficient.gt(1e-12)
                coefficient_rows.append(coefficients)
                assoc = train_assoc.rename(columns={"effect": "train_effect", "p_value": "train_p_value", "fdr": "train_fdr", "df": "train_df"}).merge(test_assoc.rename(columns={"effect": "test_effect", "p_value": "test_p_value", "fdr": "test_fdr", "df": "test_df"}), on="lncrna_id", validate="one_to_one").merge(coefficients, on=["lncrna_id"], how="left", suffixes=("", "_lasso"))
                strong = assoc.test_effect.abs().ge(float(cfg["state_graph"]["candidate_strong_abs_effect"])) & assoc.test_fdr.le(float(cfg["state_graph"]["candidate_strong_fdr"]))
                weak = assoc.test_effect.abs().ge(float(cfg["state_graph"]["candidate_weak_abs_effect"])) & assoc.test_fdr.le(float(cfg["state_graph"]["candidate_weak_fdr"])) & ~strong
                assoc["test_label_class"] = np.select([strong, weak], ["strong_positive", "weak_positive"], default="unlabeled")
                assoc["test_membership_label"] = assoc.test_label_class.ne("unlabeled").astype(int)
                assoc["cancer_id"] = cancer; assoc["state_id"] = state_id; assoc["patient_fold_id"] = fold_id
                assoc["n_train"] = len(train_ids); assoc["n_test"] = len(test_ids); assoc["train_design_rank"] = design_rank
                assoc["fdr_family_size"] = len(assoc); assoc["outer_test_patients_used_upstream"] = 0
                association_rows.append(assoc)
                availability_rows.append({"cancer_id": cancer, "state_id": state_id, "patient_fold_id": fold_id, "status": "AVAILABLE", "reason": None})

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    coefficients = pd.concat(coefficient_rows, ignore_index=True) if coefficient_rows else pd.DataFrame()
    associations = pd.concat(association_rows, ignore_index=True) if association_rows else pd.DataFrame()
    availability = pd.DataFrame(availability_rows)
    metrics.to_csv(temporary / "patient_score_fold_metrics.tsv", sep="\t", index=False)
    predictions.to_parquet(temporary / "patient_score_oof.parquet", index=False, compression="zstd")
    coefficients.to_parquet(temporary / "lasso_coefficients.parquet", index=False, compression="zstd")
    associations.to_parquet(temporary / "association_replication.parquet", index=False, compression="zstd")
    availability.to_csv(temporary / "eligibility.tsv", sep="\t", index=False)
    if not metrics.empty and int(metrics.outer_test_patients_used_upstream.sum()) != 0:
        raise RuntimeError("Outer-test patients appeared in an upstream LASSO fit")
    mismatches = verify_file_manifest(input_root, input_records) + verify_file_manifest(Path(provenance["repo_root"]), code_records)
    if mismatches:
        raise RuntimeError(f"Frozen assets changed during nested LASSO: {mismatches[:20]}")
    success = {"status": "PASS", "run_id": args.run_id, "patient_score_folds": len(metrics), "association_rows": len(associations), "outer_test_patients_used_in_any_upstream_fit": 0, "state_outcome_imputation": "NONE", "sample_join": "EXACT_CANONICAL_ALIQUOT", "evaluation_unit_patient_score": "patient", "evaluation_unit_association_replication": "cancer x patient_fold x lncRNA x state", "input_manifest_sha256": file_sha256(input_manifest)}
    atomic_write_json(temporary / "SUCCESS.json", success)
    os.replace(temporary, output_root)
    print(json.dumps(success, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
