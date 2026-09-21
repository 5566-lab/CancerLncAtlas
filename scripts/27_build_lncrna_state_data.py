#!/usr/bin/env python3
"""Build fold-safe cancer x lncRNA x tumor-state training data."""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cc_hhgt_v26.adapter_data import (  # noqa: E402
    CATEGORICAL_COVARIATES,
    NUMERIC_COVARIATES,
    bh_fdr,
    correlation_for_pairs,
    residualize_train_fitted,
)

INPUT = ROOT / "input_snapshot"
RESULT_BASE = Path(os.getenv("CC_HHGT_RESULT_ROOT", str(ROOT / "results")))
OUT = Path(os.getenv("CC_HHGT_STATE_DATA_ROOT", str(RESULT_BASE / "lncrna_state_data")))
FOLD_PATH = INPUT / "adapter" / "cancer_specific_patient_fold_manifest.tsv"
STATE_PATH = INPUT / "TCGA_sample_scores" / "03_final_tables" / "TCGA_sample_score_matrix.tsv.gz"
COV_PATH = INPUT / "processed" / "tcga_association_covariates.parquet"
STATE_REGEX = os.getenv(
    "CC_HHGT_STATE_TARGET_REGEX",
    r"(?i)(extend|telomerase|stemness|rnass|dnass|ereg[._-]?expss)",
)
MIN_DETECTION = float(os.getenv("CC_HHGT_STATE_MIN_DETECTION", "0.05"))
MIN_ABS_CORR = float(os.getenv("CC_HHGT_STATE_MIN_ABS_CORR", "0.12"))
TOP_LNC_PER_STATE = int(os.getenv("CC_HHGT_STATE_TOP_LNC_PER_STATE", "3000"))
MAX_MODEL_CANDIDATES = int(os.getenv("CC_HHGT_STATE_MAX_MODEL_CANDIDATES_PER_FOLD", "100000"))
MAX_EVALUATION_PAIRS = int(os.getenv("CC_HHGT_STATE_MAX_EVALUATION_PAIRS_PER_FOLD", "0"))
MIN_PATIENTS = int(os.getenv("CC_HHGT_STATE_MIN_PATIENTS", "20"))
KEYS = ["cancer_id", "patient_fold_id", "lncrna_id", "state_id"]


def sample_key(value: object) -> str:
    return str(value)[:15]


def select_state_columns() -> list[str]:
    header = pd.read_csv(STATE_PATH, sep="\t", nrows=0).columns.tolist()
    exact = [x.strip() for x in os.getenv("CC_HHGT_STATE_TARGETS", "").split(",") if x.strip()]
    if exact:
        missing = sorted(set(exact) - set(header))
        if missing:
            raise RuntimeError(f"requested tumor-state columns missing: {missing}")
        return exact
    pattern = re.compile(STATE_REGEX)
    selected = [column for column in header if pattern.search(column)]
    if not selected:
        raise RuntimeError(f"no tumor-state columns matched regex: {STATE_REGEX}")
    return selected


def load_inputs(cancer: str, state_columns: list[str], fold: pd.DataFrame):
    expr_path = INPUT / "parquet" / "bulk_lncRNA_expression" / f"cancer_id={cancer}" / "part-0.parquet"
    expr = pd.read_parquet(expr_path, columns=["sample_id", "lncrna_id", "logcpm", "tpm"])
    expr["sample_id"] = expr["sample_id"].map(sample_key)
    expression = expr.pivot_table(index="sample_id", columns="lncrna_id", values="logcpm", aggfunc="mean")
    detection = (
        expr.assign(detected=expr["tpm"].gt(0.1).astype(float))
        .pivot_table(index="sample_id", columns="lncrna_id", values="detected", aggfunc="max")
        .fillna(0.0)
    )

    usecols = ["sample_id", *state_columns]
    header = pd.read_csv(STATE_PATH, sep="\t", nrows=0).columns
    if "cancer_type" in header:
        usecols.append("cancer_type")
    state = pd.read_csv(STATE_PATH, sep="\t", usecols=lambda c: c in set(usecols))
    if "cancer_type" in state:
        state = state[state["cancer_type"].astype(str).eq(cancer)]
    state["sample_id"] = state["sample_id"].map(sample_key)
    state = state.groupby("sample_id", observed=True)[state_columns].mean()

    cov = pd.read_parquet(COV_PATH)
    cov = cov[cov["cancer_id"].astype(str).eq(cancer)].copy()
    cov["sample_id"] = cov["sample_id"].map(sample_key)
    cov = cov.drop_duplicates("sample_id").set_index("sample_id")

    fold = fold.copy()
    fold["sample_id"] = fold["sample_id"].map(sample_key)
    samples = sorted(set(fold["sample_id"]) & set(expression.index) & set(state.index) & set(cov.index))
    if len(samples) < MIN_PATIENTS:
        raise RuntimeError(f"{cancer}: only {len(samples)} aligned samples for lncRNA-state model")
    return expression.reindex(samples), detection.reindex(samples).fillna(0), state.reindex(samples), cov.reindex(samples), fold


def build_pair_universe(
    cancer: str,
    x: np.ndarray,
    y: np.ndarray,
    lnc: list[str],
    states: list[str],
    detection_rate: np.ndarray,
) -> pd.DataFrame:
    """Build a complete held-out evaluation universe and a train-only model mask.

    Every lncRNA passing the train-patient detection threshold is evaluated
    against every requested state.  A smaller train-only subset is marked by
    ``selected_for_model`` for efficient neural training.  This separation
    allows transcriptome-wide cancer/state FDR while preserving train-only
    candidate selection.
    """
    if x.shape[0] < 6:
        return pd.DataFrame(
            columns=[
                "cancer_id", "lncrna_id", "state_id",
                "selection_score", "selected_for_model",
            ]
        )
    xz = x - np.nanmean(x, axis=0, keepdims=True)
    yz = y - np.nanmean(y, axis=0, keepdims=True)
    denom = (
        np.sqrt(np.nansum(xz * xz, axis=0))[:, None]
        * np.sqrt(np.nansum(yz * yz, axis=0))[None, :]
    )
    corr = np.divide(
        xz.T @ yz,
        denom,
        out=np.zeros((x.shape[1], y.shape[1])),
        where=denom > 1e-10,
    )
    eligible_index = np.flatnonzero(detection_rate >= MIN_DETECTION)
    if not len(eligible_index):
        return pd.DataFrame()
    rows: list[pd.DataFrame] = []
    for state_index, state_id in enumerate(states):
        score = np.abs(corr[:, state_index])
        ordered = eligible_index[np.argsort(score[eligible_index])[::-1]]
        selected = set(
            eligible_index[score[eligible_index] >= MIN_ABS_CORR].tolist()
        )
        selected.update(ordered[: min(TOP_LNC_PER_STATE, len(ordered))].tolist())
        part = pd.DataFrame(
            {
                "cancer_id": cancer,
                "lncrna_id": [lnc[i] for i in eligible_index],
                "state_id": state_id,
                "selection_score": score[eligible_index].astype("float32"),
                "selected_for_model": np.asarray(
                    [i in selected for i in eligible_index], dtype="int8"
                ),
            }
        )
        rows.append(part)
    result = pd.concat(rows, ignore_index=True)
    selected_rows = result.loc[result.selected_for_model.eq(1)]
    if len(selected_rows) > MAX_MODEL_CANDIDATES:
        keep = set(
            selected_rows.nlargest(MAX_MODEL_CANDIDATES, "selection_score").index
        )
        result["selected_for_model"] = result.index.to_series().isin(keep).astype("int8")
    if MAX_EVALUATION_PAIRS > 0 and len(result) > MAX_EVALUATION_PAIRS:
        raise RuntimeError(
            f"{cancer}: evaluation universe has {len(result):,} pairs, exceeding "
            f"CC_HHGT_STATE_MAX_EVALUATION_PAIRS_PER_FOLD={MAX_EVALUATION_PAIRS}. "
            "Raise the limit rather than silently truncating transcriptome-wide FDR."
        )
    return result


def build_cancer(cancer: str, fold_all: pd.DataFrame, state_columns: list[str]):
    fold = fold_all[fold_all["cancer_id"].astype(str).eq(cancer)].copy()
    expression, detection, state, cov, fold = load_inputs(cancer, state_columns, fold)
    sample_to_index = {sample: index for index, sample in enumerate(expression.index)}
    lnc = list(map(str, expression.columns))
    states = list(map(str, state.columns))
    lnc_index = {value: index for index, value in enumerate(lnc)}
    state_index = {value: index for index, value in enumerate(states)}
    rows = []
    provenance = []
    fold_column = "fold_id" if "fold_id" in fold else "patient_fold_id"
    for fold_id in sorted(fold[fold_column].unique()):
        current = fold[fold[fold_column].eq(fold_id)]
        split_samples = {
            split: [sample for sample in current.loc[current["split"].eq(split), "sample_id"] if sample in sample_to_index]
            for split in ["train", "validation", "test"]
        }
        expr_resid, state_resid, design = residualize_train_fitted(expression, state, cov, split_samples["train"])
        train_idx = np.array([sample_to_index[s] for s in split_samples["train"]], dtype=int)
        detection_rate = detection.to_numpy()[train_idx].mean(axis=0)
        candidate = build_pair_universe(
            cancer,
            expr_resid.to_numpy()[train_idx],
            state_resid.to_numpy()[train_idx],
            lnc,
            states,
            detection_rate,
        )
        if candidate.empty:
            raise RuntimeError(f"{cancer} {fold_id}: no lncRNA-state candidates")
        candidate["patient_fold_id"] = str(fold_id)
        left = candidate["lncrna_id"].map(lnc_index).astype(int).to_numpy()
        right = candidate["state_id"].map(state_index).astype(int).to_numpy()
        for split in ["train", "validation", "test"]:
            idx = np.array([sample_to_index[s] for s in split_samples[split]], dtype=int)
            corr, pvalue = correlation_for_pairs(
                expr_resid.to_numpy()[idx],
                state_resid.to_numpy()[idx],
                left,
                right,
            )
            candidate[f"{split}_effect"] = corr.astype("float32")
            candidate[f"{split}_pvalue"] = pvalue.astype("float32")
            candidate[f"{split}_fdr"] = (
                candidate.groupby("state_id", observed=True)[f"{split}_pvalue"]
                .transform(lambda values: bh_fdr(values.to_numpy(float)))
                .astype("float32")
            )
            candidate[f"{split}_n_patients"] = len(idx)

        threshold_by_state = (
            candidate.groupby("state_id", observed=True)["train_effect"]
            .apply(lambda values: float(max(0.20, values.abs().dropna().quantile(0.80))))
            .to_dict()
        )
        candidate["membership_threshold"] = candidate["state_id"].map(threshold_by_state).astype("float32")
        for split in ["train", "validation", "test"]:
            observed = candidate[f"{split}_effect"].notna()
            candidate[f"{split}_membership_label"] = np.where(
                observed,
                (
                    candidate[f"{split}_effect"].abs().ge(candidate["membership_threshold"])
                    & candidate[f"{split}_fdr"].le(0.20)
                ).astype("float32"),
                np.nan,
            )
            positive_membership = candidate[f"{split}_membership_label"].eq(1.0)
            candidate[f"{split}_direction_label"] = np.where(
                observed & positive_membership,
                candidate[f"{split}_effect"].ge(0).astype("float32"),
                np.nan,
            )
        detection_map = dict(zip(lnc, detection_rate, strict=True))
        train_state = state.loc[split_samples["train"]]
        candidate["train_detection_rate"] = candidate["lncrna_id"].map(detection_map)
        candidate["train_state_mean"] = candidate["state_id"].map(train_state.mean())
        candidate["train_state_sd"] = candidate["state_id"].map(train_state.std())
        candidate["train_state_missing_rate"] = candidate["state_id"].map(train_state.isna().mean())
        candidate["lncrna_available"] = 1
        candidate["state_available"] = 1
        candidate["pair_available"] = candidate["train_effect"].notna().astype("int8")
        rows.append(candidate)
        provenance.append({
            "cancer_id": cancer,
            "patient_fold_id": str(fold_id),
            **design,
            "n_states": len(states),
            "state_ids": "|".join(states),
            "n_evaluation_pairs": len(candidate),
            "n_model_candidates": int(candidate.selected_for_model.sum()),
            "membership_threshold_by_state": json.dumps(threshold_by_state, ensure_ascii=False),
            "candidate_selection_scope": "train_patients_only_per_fold",
            "residualizer_fit_scope": "train_patients_only",
            "direction_supervision_scope": "positive_membership_pairs_only",
        })
    return pd.concat(rows, ignore_index=True), provenance


def main() -> int:
    state_columns = select_state_columns()
    fold_all = pd.read_csv(FOLD_PATH, sep="\t")
    cancers = sorted(fold_all["cancer_id"].astype(str).unique())
    OUT.mkdir(parents=True, exist_ok=True)
    provenance = []
    for index, cancer in enumerate(cancers, 1):
        path = OUT / f"cancer_id={cancer}" / "part-0.parquet"
        if path.exists():
            observed = set(pd.read_parquet(path, columns=["state_id"])["state_id"].astype(str))
            if set(state_columns).issubset(observed):
                print(f"[skip] {cancer} ({index}/{len(cancers)})", flush=True)
                continue
        frame, rows = build_cancer(cancer, fold_all, state_columns)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False, compression="zstd")
        provenance.extend(rows)
        print(f"[done] {cancer}: rows={len(frame):,} states={frame['state_id'].nunique()} ({index}/{len(cancers)})", flush=True)
    if provenance:
        pd.DataFrame(provenance).to_parquet(OUT / "fold_provenance.parquet", index=False, compression="zstd")
    contract = {
        "status": "COMPLETED",
        "completed_at": datetime.now().isoformat(),
        "task": "cancer_x_lncrna_x_tumor_state",
        "state_columns": state_columns,
        "state_regex": STATE_REGEX,
        "candidate_selection": "train-only residual correlations; complete detectable-lncRNA evaluation universe",
        "multiple_testing_scope": "all detectable lncRNAs within each cancer x state x held-out fold",
        "state_semantics": "phenotype/state, not pathway membership",
        "direction_supervision_scope": "positive_membership_pairs_only",
        "n_cancers": len(cancers),
    }
    (OUT / "SUCCESS.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(contract, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
