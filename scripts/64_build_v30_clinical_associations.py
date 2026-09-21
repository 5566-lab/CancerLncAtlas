#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from cc_hhgt.clinical_v30 import clean_na_columns, fit_phreg, harrell_c_index, make_preprocessor, random_effects_log_hazard
from cc_hhgt.common import load_config, write_json, write_table
from cc_hhgt.stats import bh_fdr


def zscore_train(train: pd.Series, other: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    x = pd.to_numeric(train, errors="coerce").to_numpy(float)
    y = pd.to_numeric(other, errors="coerce").to_numpy(float)
    mean = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd < 1e-8:
        sd = 1.0
    return (x - mean) / sd, (y - mean) / sd


def build_design(train: pd.DataFrame, test: pd.DataFrame, subject_type: str, subject: str, modifier: str | None, cfg: dict):
    settings = cfg["clinical"]
    numeric_cov = [c for c in settings.get("clinical_covariates", {}).get("numeric", []) if c in train]
    categorical_cov = [c for c in settings.get("clinical_covariates", {}).get("categorical", []) if c in train]
    pre = make_preprocessor(numeric_cov, categorical_cov)
    cov_train = np.asarray(pre.fit_transform(train), float)
    cov_test = np.asarray(pre.transform(test), float)
    cov_names = list(pre.get_feature_names_out())
    subject_train, subject_test = zscore_train(train[subject], test[subject])
    if modifier is None:
        x_train = np.column_stack([subject_train, cov_train])
        x_test = np.column_stack([subject_test, cov_test])
        names = ["subject", *cov_names]
        target = "subject"
    else:
        modifier_train, modifier_test = zscore_train(train[modifier], test[modifier])
        interaction_train = subject_train * modifier_train
        interaction_test = subject_test * modifier_test
        x_train = np.column_stack([subject_train, modifier_train, interaction_train, cov_train])
        x_test = np.column_stack([subject_test, modifier_test, interaction_test, cov_test])
        names = ["subject", "modifier", "interaction", *cov_names]
        target = "interaction"
    return x_train, x_test, names, target


def fit_record(cancer: str, fold: str, endpoint: str, subject_type: str, subject_id: str, subject_col: str, modifier_id: str | None, modifier_col: str | None, frame: pd.DataFrame, cfg: dict) -> dict:
    train = clean_na_columns(frame.loc[(frame.patient_fold_id.astype(str).eq(fold)) & frame.split.eq("train")].copy())
    test = clean_na_columns(frame.loc[(frame.patient_fold_id.astype(str).eq(fold)) & frame.split.eq("test")].copy())
    time_col, event_col = f"time_days::{endpoint}", f"event::{endpoint}"
    needed = [time_col, event_col, subject_col] + ([modifier_col] if modifier_col else [])
    train = train.dropna(subset=needed)
    test = test.dropna(subset=needed)
    if len(train) < int(cfg["clinical"].get("min_patients", 50)) or train[event_col].sum() < int(cfg["clinical"].get("min_events", 12)):
        return {"status": "INSUFFICIENT", "cancer_id": cancer, "patient_fold_id": fold, "endpoint": endpoint, "subject_type": subject_type, "subject_id": subject_id, "modifier_id": modifier_id}
    if len(test) < 10 or test[event_col].sum() < int(cfg["clinical"].get("min_test_events", 2)):
        return {"status": "INSUFFICIENT_TEST", "cancer_id": cancer, "patient_fold_id": fold, "endpoint": endpoint, "subject_type": subject_type, "subject_id": subject_id, "modifier_id": modifier_id}
    try:
        x_train, x_test, names, target = build_design(train, test, subject_type, subject_col, modifier_col, cfg)
        # Preserve the biological target terms and keep only the highest-variance
        # covariate columns supported by the train-fold event count. This avoids
        # singular Cox fits in small cancers without using outcome information
        # from validation or test patients.
        target_count = 1 if modifier_col is None else 3
        event_count = int(pd.to_numeric(train[event_col], errors="coerce").fillna(0).sum())
        max_columns = max(target_count, min(x_train.shape[1], event_count - 3, max(target_count, len(train) // 8)))
        if x_train.shape[1] > max_columns:
            covariate_indices = np.arange(target_count, x_train.shape[1])
            variances = np.nanvar(x_train[:, covariate_indices], axis=0)
            order = covariate_indices[np.argsort(-np.nan_to_num(variances, nan=-1.0))]
            keep = np.concatenate([np.arange(target_count), order[: max_columns - target_count]])
            x_train = x_train[:, keep]
            x_test = x_test[:, keep]
            names = [names[int(i)] for i in keep]
    except Exception as exc:
        return {"status": "DESIGN_FAILED", "error": repr(exc), "cancer_id": cancer, "patient_fold_id": fold, "endpoint": endpoint, "subject_type": subject_type, "subject_id": subject_id, "modifier_id": modifier_id}
    train_fit = fit_phreg(
        pd.to_numeric(train[time_col], errors="coerce").to_numpy(float),
        pd.to_numeric(train[event_col], errors="coerce").to_numpy(float),
        x_train, names, ties=cfg["clinical"].get("cox_ties", "breslow"), maxiter=int(cfg["clinical"].get("cox_maxiter", 100)),
    )
    test_fit = fit_phreg(
        pd.to_numeric(test[time_col], errors="coerce").to_numpy(float),
        pd.to_numeric(test[event_col], errors="coerce").to_numpy(float),
        x_test, names, ties=cfg["clinical"].get("cox_ties", "breslow"), maxiter=int(cfg["clinical"].get("cox_maxiter", 100)),
    )
    if train_fit.get("status") != "PASS":
        return {"status": "TRAIN_" + train_fit.get("status", "FAILED"), "cancer_id": cancer, "patient_fold_id": fold, "endpoint": endpoint, "subject_type": subject_type, "subject_id": subject_id, "modifier_id": modifier_id}
    train_beta = train_fit["params"][target]
    train_se = train_fit["bse"][target]
    train_p = train_fit["pvalues"][target]
    test_beta = test_fit.get("params", {}).get(target, math.nan)
    test_se = test_fit.get("bse", {}).get(target, math.nan)
    test_p = test_fit.get("pvalues", {}).get(target, math.nan)
    risk = x_test @ np.array([train_fit["params"][n] for n in names])
    test_cindex = harrell_c_index(test[time_col].to_numpy(float), test[event_col].to_numpy(float), risk)
    return {
        "status": "PASS",
        "cancer_id": cancer,
        "patient_fold_id": fold,
        "endpoint": endpoint,
        "subject_type": subject_type,
        "subject_id": subject_id,
        "modifier_id": modifier_id,
        "target_term": target,
        "train_beta": train_beta,
        "train_se": train_se,
        "train_p_value": train_p,
        "train_hazard_ratio": float(np.exp(np.clip(train_beta, -20, 20))),
        "test_beta": test_beta,
        "test_se": test_se,
        "test_p_value": test_p,
        "test_hazard_ratio": float(np.exp(np.clip(test_beta, -20, 20))) if np.isfinite(test_beta) else math.nan,
        "test_c_index": test_cindex,
        "n_train": train_fit.get("n"),
        "events_train": train_fit.get("events"),
        "n_test": test_fit.get("n", len(test)),
        "events_test": test_fit.get("events", int(test[event_col].sum())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build cross-fitted lncRNA and lncRNA-pathway/state clinical associations")
    parser.add_argument("--config", default="config/model_v3_0_clinical.yaml")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--cancer", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = cfg["_results"] / "clinical_data"
    lnc_candidates = pd.read_parquet(root / "clinical_lncRNA_candidates.parquet")
    pair_candidates = pd.read_parquet(root / "clinical_pair_candidates.parquet")
    state_candidates = pd.read_parquet(root / "clinical_state_candidates.parquet")
    tasks = []
    for path in sorted((root / "patients").rglob("part-0.parquet")):
        cancer = path.parent.name.split("=", 1)[-1]
        if args.cancer and cancer != args.cancer:
            continue
        frame = pd.read_parquet(path)
        folds = sorted(frame.patient_fold_id.astype(str).unique())
        endpoints = [e for e in [*cfg["clinical"].get("required_endpoints", []), *cfg["clinical"].get("optional_endpoints", [])] if f"time_days::{e}" in frame]
        for fold in folds:
            for endpoint in endpoints:
                lnc_list = lnc_candidates.loc[lnc_candidates.cancer_id.eq(cancer)].sort_values("functional_probability", ascending=False).head(int(cfg["clinical"].get("max_clinical_lncRNAs_per_cancer", 96)))["lncrna_id"].astype(str)
                for lnc in lnc_list:
                    col = f"lnc::{lnc}"
                    if col in frame:
                        tasks.append((cancer, fold, endpoint, "lncRNA", lnc, col, None, None, frame))
                if cfg["clinical"].get("pair_clinical_model_enabled", True):
                    pairs = pair_candidates.loc[pair_candidates.cancer_id.eq(cancer)].sort_values("functional_probability", ascending=False).head(int(cfg["clinical"].get("max_clinical_pair_candidates_per_cancer", 64)))
                    for row in pairs.itertuples(index=False):
                        lnc_col, pf_col = f"lnc::{row.lncrna_id}", f"pf::{row.pathway_family_id}"
                        if lnc_col in frame and pf_col in frame:
                            tasks.append((cancer, fold, endpoint, "lncRNA_pathway", str(row.lncrna_id), lnc_col, str(row.pathway_family_id), pf_col, frame))
                if cfg["clinical"].get("state_clinical_model_enabled", True) and not state_candidates.empty:
                    state_score_col = "lncrna_state_probability" if "lncrna_state_probability" in state_candidates else ("state_patient_probability" if "state_patient_probability" in state_candidates else None)
                    states = state_candidates.loc[state_candidates.cancer_id.eq(cancer)]
                    if state_score_col:
                        states = states.sort_values(state_score_col, ascending=False)
                    states = states.head(int(cfg["clinical"].get("max_clinical_state_candidates_per_cancer", 48)))
                    for row in states.itertuples(index=False):
                        lnc_col, state_col = f"lnc::{row.lncrna_id}", str(row.state_id)
                        if lnc_col in frame and state_col in frame:
                            tasks.append((cancer, fold, endpoint, "lncRNA_state", str(row.lncrna_id), lnc_col, state_col, state_col, frame))
    if not tasks:
        raise RuntimeError("No V3.0 clinical association tasks")
    results = Parallel(n_jobs=args.jobs, verbose=10)(delayed(fit_record)(*task, cfg) for task in tasks)
    records = pd.DataFrame(results)
    release = cfg["_results"] / "clinical_association_release"
    write_table(records, release / "clinical_association_fold_records.parquet")
    passed = records.loc[records.status.eq("PASS")].copy()
    if passed.empty:
        raise RuntimeError("No clinical association Cox model passed")
    rows = []
    group_cols = ["cancer_id", "endpoint", "subject_type", "subject_id", "modifier_id"]
    for keys, group in passed.groupby(group_cols, observed=True, dropna=False):
        meta = random_effects_log_hazard(group.test_beta.to_numpy(float), group.test_se.to_numpy(float))
        signs = np.sign(group.test_beta.to_numpy(float))
        signs = signs[np.isfinite(signs) & (signs != 0)]
        same = int(max((signs > 0).sum(), (signs < 0).sum())) if len(signs) else 0
        rows.append(dict(zip(group_cols, keys, strict=True)) | {
            "meta_beta": meta["beta"],
            "hazard_ratio": float(np.exp(np.clip(meta["beta"], -20, 20))) if np.isfinite(meta["beta"]) else math.nan,
            "ci_lower": float(np.exp(np.clip(meta["ci_lower"], -20, 20))) if np.isfinite(meta["ci_lower"]) else math.nan,
            "ci_upper": float(np.exp(np.clip(meta["ci_upper"], -20, 20))) if np.isfinite(meta["ci_upper"]) else math.nan,
            "p_value": meta["p_value"],
            "tau2": meta["tau2"],
            "i2": meta["i2"],
            "n_folds_available": meta["k"],
            "n_folds_same_direction": same,
            "mean_test_c_index": float(pd.to_numeric(group.test_c_index, errors="coerce").mean()),
            "n_patients_test_total": int(pd.to_numeric(group.n_test, errors="coerce").fillna(0).sum()),
            "n_events_test_total": int(pd.to_numeric(group.events_test, errors="coerce").fillna(0).sum()),
        })
    summary = pd.DataFrame(rows)
    summary["fdr"] = summary.groupby(["cancer_id", "endpoint", "subject_type"], observed=True).p_value.transform(lambda x: bh_fdr(pd.to_numeric(x, errors="coerce").to_numpy(float)))
    summary["replication_tier"] = np.select([summary.n_folds_same_direction.ge(4), summary.n_folds_same_direction.ge(3)], ["robust_core", "replicated"], default="exploratory")
    summary["direction"] = np.where(summary.meta_beta.ge(0), "risk", "protective")
    write_table(summary, release / "clinical_association_summary.parquet")
    payload = {"status": "COMPLETED", "tasks": len(records), "passed_tasks": int(len(passed)), "summary_rows": int(len(summary)), "cancers": int(summary.cancer_id.nunique()), "endpoints": sorted(summary.endpoint.unique())}
    write_json(payload, release / "SUCCESS.json")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
