"""Leakage-controlled patient-fold data construction for the cancer adapter.

The original implementation inherited its complete candidate universe from the
strict LOCO predictions.  That design is safe for cross-cancer evaluation but
cannot recover a lncRNA-pathway pair that is only detectable in one cancer and
therefore never entered the strict OOF table.

This module builds a fold-local candidate union:

    strict candidates
    UNION patient-native candidates selected from train patients only
    UNION optional evidence-derived candidates

Patient-native candidate selection is performed independently in every patient
fold.  Validation/test patients are never used for residualisation, candidate
selection, thresholds, scaling, or feature construction.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "input_snapshot"
RESULT = Path(os.getenv("CC_HHGT_RESULT_ROOT", str(ROOT / "results")))
OOF = Path(os.getenv("CC_HHGT_STRICT_OOF_PATH", str(RESULT / "strict_release" / "strict_cross_cancer_oof_prediction.parquet")))
FAMILY_MEMBER = RESULT / "static_pathway" / "static_pathway_family_member.parquet"
PATIENT_FOLD = INPUT / "adapter" / "cancer_specific_patient_fold_manifest.tsv"
ADAPTER_DATA = Path(os.getenv("CC_HHGT_ADAPTER_DATA_ROOT", str(RESULT / "adapter_data")))

PATIENT_NATIVE_TOP_K = int(os.getenv("CC_HHGT_PATIENT_NATIVE_TOP_K", "16"))
PATIENT_NATIVE_MIN_ABS_CORR = float(
    os.getenv("CC_HHGT_PATIENT_NATIVE_MIN_ABS_CORR", "0.15")
)
PATIENT_NATIVE_MIN_DETECTION = float(
    os.getenv("CC_HHGT_PATIENT_NATIVE_MIN_DETECTION", "0.05")
)
PATIENT_NATIVE_MAX_CANDIDATES = int(
    os.getenv("CC_HHGT_PATIENT_NATIVE_MAX_CANDIDATES", "100000")
)
PATIENT_NATIVE_BLOCK_SIZE = int(
    os.getenv("CC_HHGT_PATIENT_NATIVE_BLOCK_SIZE", "512")
)

OPTIONAL_EVIDENCE_CANDIDATE_PATHS = [
    INPUT / "adapter" / "evidence_candidate_pairs.parquet",
    INPUT / "adapter" / "direct_evidence_candidates.parquet",
    RESULT / "evidence_fusion" / "evidence_candidate_pairs.parquet",
    RESULT / "evidence_fusion" / "direct_evidence_candidates.parquet",
    RESULT / "evidence_events" / "evidence_candidate_pairs.parquet",
]
OPTIONAL_FOLD_EVIDENCE_PATHS = [
    INPUT / "adapter" / "cancer_specific_fold_pair_evidence.parquet",
    INPUT / "adapter" / "fold_pair_evidence.parquet",
    ROOT / "cancer_specific_fold_pair_evidence.parquet",
]

NUMERIC_COVARIATES = [
    "age_years",
    "purity",
    "leukocyte_fraction",
    "absolute_purity",
    "absolute_ploidy",
    "paper_purity",
    "cell_fraction_T.cells.CD8",
    "cell_fraction_T.cells.CD4.memory.resting",
    "cell_fraction_T.cells.regulatory..Tregs.",
    "cell_fraction_Macrophages.M0",
    "cell_fraction_Macrophages.M1",
    "cell_fraction_Macrophages.M2",
]
CATEGORICAL_COVARIATES = [
    "sex",
    "stage",
    "molecular_subtype",
    "clinical_subtype",
    "technical_batch",
]
STATE_COLUMNS = [
    "CYT::mean_log_GZMA_PRF1",
    "ESTIMATE::ESTIMATEScore",
    "ESTIMATE::ImmuneScore",
    "ESTIMATE::StromalScore",
    "ESTIMATE::TumorPurity",
    "EXTEND::published_score",
    "stemness_rna::RNAss",
    "stemness_rna::EREG.EXPss",
    "stemness_dna::DNAss",
    "mutation_load::nonsilent_rate",
    "mutation_load::silent_rate",
]
PAIR_KEYS = ["cancer_id", "lncrna_id", "pathway_family_id"]
FOLD_PAIR_KEYS = ["cancer_id", "patient_fold_id", "lncrna_id", "pathway_family_id"]


def bh_fdr(pvalues: np.ndarray) -> np.ndarray:
    result = np.full(len(pvalues), np.nan, dtype=np.float64)
    finite = np.isfinite(pvalues)
    values = pvalues[finite]
    if not len(values):
        return result
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.minimum(adjusted, 1.0)
    result[finite] = restored
    return result


def correlation_for_pairs(
    left: np.ndarray,
    right: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n = left.shape[0]
    corr = np.full(len(left_index), np.nan, dtype=np.float64)
    pvalue = np.full(len(left_index), np.nan, dtype=np.float64)
    if n < 6 or not len(left_index):
        return corr, pvalue
    lx = left[:, left_index]
    ry = right[:, right_index]
    lx = lx - np.nanmean(lx, axis=0, keepdims=True)
    ry = ry - np.nanmean(ry, axis=0, keepdims=True)
    lnorm = np.sqrt(np.nansum(lx * lx, axis=0))
    rnorm = np.sqrt(np.nansum(ry * ry, axis=0))
    denominator = lnorm * rnorm
    valid = denominator > 1e-10
    corr[valid] = np.nansum(lx[:, valid] * ry[:, valid], axis=0) / denominator[valid]
    corr = np.clip(corr, -0.999999, 0.999999)
    tvalue = np.abs(corr) * np.sqrt((n - 2) / np.maximum(1 - corr * corr, 1e-8))
    pvalue[np.isfinite(corr)] = 2 * stats.t.sf(
        tvalue[np.isfinite(corr)], df=n - 2
    )
    return corr, pvalue


def residualize_train_fitted(
    expression: pd.DataFrame,
    activity: pd.DataFrame,
    covariates: pd.DataFrame,
    train_samples: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    numeric = [column for column in NUMERIC_COVARIATES if column in covariates]
    categorical = [
        column for column in CATEGORICAL_COVARIATES if column in covariates
    ]
    transformers = []
    if numeric:
        transformers.append(
            (
                "numeric",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    StandardScaler(),
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
                categorical,
            )
        )
    transformer = ColumnTransformer(
        transformers,
        remainder="drop",
        sparse_threshold=0.0,
    )
    train_mask = covariates.index.isin(train_samples)
    if train_mask.sum() < 6:
        raise RuntimeError("fewer than six train patients after sample alignment")
    if transformers:
        transformer.fit(covariates.loc[train_mask])
        design = np.asarray(transformer.transform(covariates), dtype=np.float64)
    else:
        design = np.empty((len(covariates), 0), dtype=np.float64)
    design = np.column_stack([np.ones(len(design)), design])

    def _residualize(frame: pd.DataFrame) -> pd.DataFrame:
        values = frame.to_numpy(dtype=np.float64)
        train_values = values[train_mask]
        medians = np.nanmedian(train_values, axis=0)
        medians[~np.isfinite(medians)] = 0.0
        values = np.where(np.isfinite(values), values, medians[None, :])
        ridge = Ridge(alpha=10.0, fit_intercept=False)
        ridge.fit(design[train_mask], values[train_mask])
        residual = values - ridge.predict(design)
        return pd.DataFrame(residual, index=frame.index, columns=frame.columns)

    return (
        _residualize(expression),
        _residualize(activity),
        {
            "n_train_samples": int(train_mask.sum()),
            "n_design_columns": int(design.shape[1]),
        },
    )


def _state_context(state: pd.DataFrame, train_samples: list[str]) -> dict[str, float]:
    subset = state[state["sample_id"].isin(train_samples)]
    result: dict[str, float] = {}
    for column in STATE_COLUMNS:
        if column not in subset:
            result[f"state_mean__{column}"] = np.nan
            result[f"state_available__{column}"] = 0.0
            continue
        values = pd.to_numeric(subset[column], errors="coerce")
        result[f"state_mean__{column}"] = float(values.mean())
        result[f"state_available__{column}"] = float(values.notna().any())
    return result


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _candidate_env_path() -> Path | None:
    value = os.getenv("CC_HHGT_EVIDENCE_CANDIDATES")
    return Path(value) if value else None


def _fold_evidence_env_path() -> Path | None:
    value = os.getenv("CC_HHGT_FOLD_PAIR_EVIDENCE")
    return Path(value) if value else None


def _normalise_candidate_columns(frame: pd.DataFrame, cancer: str) -> pd.DataFrame:
    aliases = {
        "lncRNA_id": "lncrna_id",
        "lnc_id": "lncrna_id",
        "pathway_family": "pathway_family_id",
        "family_id": "pathway_family_id",
        "cancer": "cancer_id",
    }
    frame = frame.rename(columns={key: value for key, value in aliases.items() if key in frame})
    if "cancer_id" not in frame:
        frame["cancer_id"] = cancer
    frame = frame[
        frame["cancer_id"].astype(str).isin([cancer, "PAN", "PANCAN", "*"])
    ].copy()
    frame["cancer_id"] = cancer
    missing = sorted(set(PAIR_KEYS) - set(frame.columns))
    if missing:
        raise RuntimeError(f"evidence candidate table lacks keys: {missing}")
    return frame.dropna(subset=PAIR_KEYS).drop_duplicates(PAIR_KEYS)


def load_strict_candidates(cancer: str) -> pd.DataFrame:
    if not OOF.exists():
        raise FileNotFoundError(OOF)
    frame = pd.read_parquet(OOF, filters=[("cancer_id", "=", cancer)])
    if frame.empty:
        # Some parquet engines do not push down filters for a single file.
        frame = pd.read_parquet(OOF)
        frame = frame[frame["cancer_id"].eq(cancer)].copy()
    keep = [
        *PAIR_KEYS,
        "cross_cancer_probability",
        "cross_cancer_probability_sd",
        "direction_probability",
        "label_class",
    ]
    cold_column = "cold_start" if "cold_start" in frame else "label_source"
    if cold_column in frame:
        keep.append(cold_column)
    keep = [column for column in keep if column in frame]
    frame = frame[keep].drop_duplicates(PAIR_KEYS)
    if cold_column in frame and cold_column != "cold_start":
        frame = frame.rename(columns={cold_column: "cold_start"})
    frame["candidate_from_strict"] = 1
    return frame


def _existing_unique_paths(paths: Iterable[Path | None]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        if path is None:
            continue
        path = Path(path)
        key = str(path.resolve()) if path.exists() else str(path)
        if path.exists() and key not in seen:
            seen.add(key)
            result.append(path)
    return result


def load_evidence_candidates(cancer: str) -> pd.DataFrame:
    """Load and union every available evidence-candidate table.

    Candidate provenance is retained, but direct target evidence is not used as
    a patient-native numeric feature. Multiple files are unioned rather than
    silently selecting only the first existing path.
    """
    paths = _existing_unique_paths(
        [_candidate_env_path(), *OPTIONAL_EVIDENCE_CANDIDATE_PATHS]
    )
    if not paths:
        return pd.DataFrame(columns=[*PAIR_KEYS, "candidate_from_evidence"])
    frames: list[pd.DataFrame] = []
    for path in paths:
        try:
            frame = pd.read_parquet(path, filters=[("cancer_id", "=", cancer)])
        except Exception:
            frame = pd.read_parquet(path)
        frame = _normalise_candidate_columns(frame, cancer)
        frame["candidate_from_evidence"] = 1
        frame["evidence_candidate_source_file"] = str(path)
        frames.append(frame)
    merged = pd.concat(frames, ignore_index=True, sort=False)
    # Preserve one row per candidate while retaining all contributing sources.
    source = (
        merged.groupby(PAIR_KEYS, observed=True)["evidence_candidate_source_file"]
        .agg(lambda x: "|".join(sorted(set(map(str, x)))))
        .rename("evidence_candidate_source_files")
        .reset_index()
    )
    optional_numeric = [
        column
        for column in merged.columns
        if column not in {*PAIR_KEYS, "candidate_from_evidence", "evidence_candidate_source_file"}
        and pd.api.types.is_numeric_dtype(merged[column])
    ]
    if optional_numeric:
        numeric = (
            merged.groupby(PAIR_KEYS, observed=True)[optional_numeric]
            .max()
            .reset_index()
        )
        result = source.merge(numeric, on=PAIR_KEYS, how="left")
    else:
        result = source
    result["candidate_from_evidence"] = 1
    return result[[*PAIR_KEYS, "candidate_from_evidence", *[c for c in result.columns if c not in {*PAIR_KEYS, "candidate_from_evidence"}]]]


def load_fold_pair_evidence(cancer: str) -> pd.DataFrame:
    """Load and union every available fold-safe evidence table."""
    paths = _existing_unique_paths(
        [_fold_evidence_env_path(), *OPTIONAL_FOLD_EVIDENCE_PATHS]
    )
    if not paths:
        return pd.DataFrame(columns=FOLD_PAIR_KEYS)
    frames: list[pd.DataFrame] = []
    for path in paths:
        try:
            frame = pd.read_parquet(path, filters=[("cancer_id", "=", cancer)])
        except Exception:
            frame = pd.read_parquet(path)
        frame = frame[frame["cancer_id"].eq(cancer)].copy()
        missing = sorted(set(FOLD_PAIR_KEYS) - set(frame.columns))
        if missing:
            raise RuntimeError(f"fold pair evidence lacks keys {missing}: {path}")
        frame["fold_evidence_source_file"] = str(path)
        frames.append(frame)
    merged = pd.concat(frames, ignore_index=True, sort=False)
    numeric = [
        column
        for column in merged.columns
        if column not in {*FOLD_PAIR_KEYS, "fold_evidence_source_file"}
        and pd.api.types.is_numeric_dtype(merged[column])
    ]
    result = merged[FOLD_PAIR_KEYS].drop_duplicates().copy()
    if numeric:
        agg = (
            merged.groupby(FOLD_PAIR_KEYS, observed=True)[numeric]
            .max()
            .reset_index()
        )
        result = result.merge(agg, on=FOLD_PAIR_KEYS, how="left")
    source = (
        merged.groupby(FOLD_PAIR_KEYS, observed=True)["fold_evidence_source_file"]
        .agg(lambda x: "|".join(sorted(set(map(str, x)))))
        .rename("fold_evidence_source_files")
        .reset_index()
    )
    return result.merge(source, on=FOLD_PAIR_KEYS, how="left")


@dataclass
class CancerMatrices:
    sample_order: list[str]
    expression: pd.DataFrame
    detection: pd.DataFrame
    activity: pd.DataFrame
    covariates: pd.DataFrame
    state: pd.DataFrame


def load_cancer_matrices(
    cancer: str,
    candidate_lnc: list[str] | None,
    candidate_family: list[str] | None,
    family_member: pd.DataFrame,
    covariates_all: pd.DataFrame,
    state_all: pd.DataFrame,
    fold_manifest: pd.DataFrame,
) -> CancerMatrices:
    expression_path = (
        INPUT
        / "parquet"
        / "bulk_lncRNA_expression"
        / f"cancer_id={cancer}"
        / "part-0.parquet"
    )
    activity_path = (
        INPUT
        / "parquet"
        / "bulk_pathway_activity"
        / f"cancer_id={cancer}"
        / "part-0.parquet"
    )
    expression_kwargs: dict[str, object] = {
        "columns": ["sample_id", "lncrna_id", "logcpm", "tpm"]
    }
    if candidate_lnc:
        expression_kwargs["filters"] = [("lncrna_id", "in", candidate_lnc)]
    expression_long = pd.read_parquet(expression_path, **expression_kwargs)
    if candidate_lnc is None:
        candidate_lnc = sorted(expression_long["lncrna_id"].dropna().unique())

    if candidate_family is None:
        candidate_family = sorted(family_member["pathway_family_id"].dropna().unique())
    required_members = family_member[
        family_member["pathway_family_id"].isin(candidate_family)
    ].copy()
    activity_long = pd.read_parquet(
        activity_path,
        filters=[("pathway_id", "in", required_members["pathway_id"].tolist())],
        columns=["sample_id", "pathway_id", "activity_score", "quality_flag"],
    )
    activity_long = activity_long[
        activity_long["quality_flag"].fillna("pass").eq("pass")
    ].merge(
        required_members[["pathway_family_id", "pathway_id", "membership_weight"]],
        on="pathway_id",
        how="inner",
    )
    activity_long["weight"] = (
        activity_long["membership_weight"].abs().fillna(1.0).clip(lower=1e-6)
    )
    activity_long["weighted_score"] = (
        activity_long["activity_score"] * activity_long["weight"]
    )
    numerator = activity_long.groupby(
        ["sample_id", "pathway_family_id"], observed=True
    )["weighted_score"].sum()
    denominator = activity_long.groupby(
        ["sample_id", "pathway_family_id"], observed=True
    )["weight"].sum()
    family_activity = (numerator / denominator).rename("activity").reset_index()

    available_samples = set(fold_manifest["sample_id"])
    available_samples &= set(expression_long["sample_id"])
    available_samples &= set(family_activity["sample_id"])
    available_samples &= set(
        covariates_all.loc[covariates_all["cancer_id"].eq(cancer), "sample_id"]
    )
    sample_order = sorted(available_samples)
    if len(sample_order) < 20:
        raise RuntimeError(f"{cancer}: only {len(sample_order)} aligned patient samples")
    expression = (
        expression_long.pivot_table(
            index="sample_id", columns="lncrna_id", values="logcpm", aggfunc="mean"
        )
        .reindex(index=sample_order, columns=candidate_lnc)
        .astype("float32")
    )
    detection = (
        expression_long.assign(detected=expression_long["tpm"].gt(0.1).astype(float))
        .pivot_table(
            index="sample_id", columns="lncrna_id", values="detected", aggfunc="max"
        )
        .reindex(index=sample_order, columns=candidate_lnc)
        .fillna(0.0)
        .astype("float32")
    )
    activity = (
        family_activity.pivot_table(
            index="sample_id",
            columns="pathway_family_id",
            values="activity",
            aggfunc="mean",
        )
        .reindex(index=sample_order, columns=candidate_family)
        .astype("float32")
    )
    covariates = (
        covariates_all[covariates_all["cancer_id"].eq(cancer)]
        .drop_duplicates("sample_id")
        .set_index("sample_id")
        .reindex(sample_order)
    )
    state = state_all[state_all["sample_id"].isin(sample_order)].drop_duplicates(
        "sample_id"
    )
    return CancerMatrices(sample_order, expression, detection, activity, covariates, state)


def select_patient_native_candidates(
    cancer: str,
    expression_train: np.ndarray,
    activity_train: np.ndarray,
    lnc_ids: list[str],
    family_ids: list[str],
    detection_rate: np.ndarray,
) -> pd.DataFrame:
    """Select top train-only lncRNA-pathway pairs without strict-model input."""
    if expression_train.shape[0] < 6 or not len(lnc_ids) or not len(family_ids):
        return pd.DataFrame(columns=[*PAIR_KEYS, "patient_native_selection_score"])

    x = np.asarray(expression_train, dtype=np.float64)
    y = np.asarray(activity_train, dtype=np.float64)
    x_median = np.nanmedian(x, axis=0)
    y_median = np.nanmedian(y, axis=0)
    x_median[~np.isfinite(x_median)] = 0.0
    y_median[~np.isfinite(y_median)] = 0.0
    x = np.where(np.isfinite(x), x, x_median[None, :])
    y = np.where(np.isfinite(y), y, y_median[None, :])
    x -= x.mean(axis=0, keepdims=True)
    y -= y.mean(axis=0, keepdims=True)
    x_norm = np.sqrt(np.sum(x * x, axis=0))
    y_norm = np.sqrt(np.sum(y * y, axis=0))
    y_valid = y_norm > 1e-10
    eligible = np.flatnonzero(
        (np.asarray(detection_rate) >= PATIENT_NATIVE_MIN_DETECTION)
        & (x_norm > 1e-10)
    )
    if not len(eligible) or not y_valid.any():
        return pd.DataFrame(columns=[*PAIR_KEYS, "patient_native_selection_score"])

    rows: list[tuple[str, str, str, float]] = []
    top_k = min(PATIENT_NATIVE_TOP_K, len(family_ids))
    for start in range(0, len(eligible), PATIENT_NATIVE_BLOCK_SIZE):
        index = eligible[start : start + PATIENT_NATIVE_BLOCK_SIZE]
        denominator = x_norm[index, None] * y_norm[None, :]
        corr = np.divide(
            x[:, index].T @ y,
            denominator,
            out=np.zeros((len(index), y.shape[1]), dtype=np.float64),
            where=denominator > 1e-10,
        )
        corr[:, ~y_valid] = np.nan
        absolute = np.abs(corr)
        if top_k < absolute.shape[1]:
            chosen = np.argpartition(absolute, -top_k, axis=1)[:, -top_k:]
        else:
            chosen = np.tile(np.arange(absolute.shape[1]), (len(index), 1))
        for local_row, lnc_index in enumerate(index):
            family_index = chosen[local_row]
            scores = absolute[local_row, family_index]
            valid = np.isfinite(scores) & (scores >= PATIENT_NATIVE_MIN_ABS_CORR)
            for right_index, score in zip(family_index[valid], scores[valid], strict=True):
                rows.append(
                    (cancer, str(lnc_ids[lnc_index]), str(family_ids[right_index]), float(score))
                )
    if not rows:
        return pd.DataFrame(columns=[*PAIR_KEYS, "patient_native_selection_score"])
    frame = pd.DataFrame(rows, columns=[*PAIR_KEYS, "patient_native_selection_score"])
    frame = (
        frame.sort_values("patient_native_selection_score", ascending=False)
        .drop_duplicates(PAIR_KEYS)
        .head(PATIENT_NATIVE_MAX_CANDIDATES)
        .reset_index(drop=True)
    )
    frame["candidate_from_patient"] = 1
    return frame


def _combine_candidate_sources(
    strict: pd.DataFrame,
    patient: pd.DataFrame,
    evidence: pd.DataFrame,
) -> pd.DataFrame:
    source_frames = []
    for frame, flag in [
        (strict, "candidate_from_strict"),
        (patient, "candidate_from_patient"),
        (evidence, "candidate_from_evidence"),
    ]:
        if frame.empty:
            continue
        value = frame.copy()
        if flag not in value:
            value[flag] = 1
        source_frames.append(value)
    if not source_frames:
        return pd.DataFrame(columns=PAIR_KEYS)
    union = pd.concat(source_frames, ignore_index=True, sort=False)
    metadata_columns = [
        "cross_cancer_probability",
        "cross_cancer_probability_sd",
        "direction_probability",
        "label_class",
        "cold_start",
        "patient_native_selection_score",
    ]
    aggregations: dict[str, str] = {
        "candidate_from_strict": "max",
        "candidate_from_patient": "max",
        "candidate_from_evidence": "max",
    }
    for column in metadata_columns:
        if column in union:
            aggregations[column] = "max" if pd.api.types.is_numeric_dtype(union[column]) else "first"
    for flag in ["candidate_from_strict", "candidate_from_patient", "candidate_from_evidence"]:
        if flag not in union:
            union[flag] = 0
    result = union.groupby(PAIR_KEYS, observed=True, as_index=False).agg(aggregations)
    for flag in ["candidate_from_strict", "candidate_from_patient", "candidate_from_evidence"]:
        result[flag] = result[flag].fillna(0).astype("int8")
    result["candidate_source"] = result.apply(
        lambda row: "|".join(
            source
            for source, flag in [
                ("strict", "candidate_from_strict"),
                ("patient_native", "candidate_from_patient"),
                ("evidence", "candidate_from_evidence"),
            ]
            if int(row[flag]) == 1
        ),
        axis=1,
    )
    result["strict_available"] = result["cross_cancer_probability"].notna().astype("int8")
    if "label_class" not in result:
        result["label_class"] = "patient_native_unlabeled"
    else:
        result["label_class"] = result["label_class"].fillna("patient_native_unlabeled")
    if "cold_start" not in result:
        result["cold_start"] = "patient_native_or_evidence"
    else:
        result["cold_start"] = result["cold_start"].fillna("patient_native_or_evidence")
    return result


def _merge_fold_evidence(frame: pd.DataFrame, evidence: pd.DataFrame) -> pd.DataFrame:
    if evidence.empty:
        return frame
    join_keys = FOLD_PAIR_KEYS if "patient_fold_id" in evidence else PAIR_KEYS
    selected = evidence.copy()
    overlapping = [column for column in selected if column in frame and column not in join_keys]
    merged = frame.merge(selected, on=join_keys, how="left", suffixes=("", "__external"))
    for column in overlapping:
        external = f"{column}__external"
        if external in merged:
            merged[column] = merged[column].combine_first(merged[external])
            merged = merged.drop(columns=external)
    return merged


def build_cancer(cancer: str) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    strict_candidates = load_strict_candidates(cancer)
    evidence_candidates = load_evidence_candidates(cancer)
    family_member = pd.read_parquet(FAMILY_MEMBER)
    fold_all = pd.read_csv(PATIENT_FOLD, sep="\t")
    fold_manifest = fold_all[fold_all["cancer_id"].eq(cancer)].copy()
    if fold_manifest.empty:
        raise RuntimeError(f"{cancer}: patient-fold manifest is empty")
    covariates_all = pd.read_parquet(
        INPUT / "processed" / "tcga_association_covariates.parquet"
    )
    state_all = pd.read_csv(
        INPUT / "TCGA_sample_scores" / "03_final_tables" / "TCGA_sample_score_matrix.tsv.gz",
        sep="\t",
        usecols=lambda column: column
        in {"sample_id", "patient_id", "cancer_type", *STATE_COLUMNS},
    )
    fold_pair_evidence = load_fold_pair_evidence(cancer)

    # Load the complete cancer-local lncRNA and pathway-family matrices.  This
    # is required to discover cancer-native candidates that were absent from
    # the strict OOF table.
    matrices = load_cancer_matrices(
        cancer,
        None,
        None,
        family_member,
        covariates_all,
        state_all,
        fold_manifest,
    )
    lnc = list(map(str, matrices.expression.columns))
    family = list(map(str, matrices.activity.columns))
    lnc_index = {value: index for index, value in enumerate(lnc)}
    family_index = {value: index for index, value in enumerate(family)}
    rows: list[pd.DataFrame] = []
    provenance: list[dict[str, object]] = []
    sample_to_index = {
        sample: index for index, sample in enumerate(matrices.sample_order)
    }

    for fold_id in sorted(fold_manifest["fold_id"].unique()):
        current = fold_manifest[fold_manifest["fold_id"].eq(fold_id)]
        split_samples = {
            split: current.loc[current["split"].eq(split), "sample_id"].tolist()
            for split in ["train", "validation", "test"]
        }
        expression_residual, activity_residual, design_info = residualize_train_fitted(
            matrices.expression,
            matrices.activity,
            matrices.covariates,
            split_samples["train"],
        )
        train_indices = np.array(
            [
                sample_to_index[sample]
                for sample in split_samples["train"]
                if sample in sample_to_index
            ],
            dtype=np.int64,
        )
        detection_rate = matrices.detection.to_numpy()[train_indices].mean(axis=0)
        patient_candidates = select_patient_native_candidates(
            cancer,
            expression_residual.to_numpy()[train_indices],
            activity_residual.to_numpy()[train_indices],
            lnc,
            family,
            detection_rate,
        )
        frame = _combine_candidate_sources(
            strict_candidates,
            patient_candidates,
            evidence_candidates,
        )
        if frame.empty:
            raise RuntimeError(f"{cancer} {fold_id}: candidate union is empty")
        frame["patient_fold_id"] = fold_id

        # Pairs from external evidence may contain identifiers that are absent
        # from the cancer-local expression/activity matrices.  They remain in
        # the evidence database, but the patient expert marks them unavailable.
        left_index = frame["lncrna_id"].map(lnc_index)
        right_index = frame["pathway_family_id"].map(family_index)
        pair_index_available = left_index.notna() & right_index.notna()
        left_safe = left_index.fillna(0).astype(int).to_numpy()
        right_safe = right_index.fillna(0).astype(int).to_numpy()

        for split in ["train", "validation", "test"]:
            indices = np.array(
                [
                    sample_to_index[sample]
                    for sample in split_samples[split]
                    if sample in sample_to_index
                ],
                dtype=np.int64,
            )
            corr = np.full(len(frame), np.nan, dtype=np.float64)
            pvalue = np.full(len(frame), np.nan, dtype=np.float64)
            valid_rows = np.flatnonzero(pair_index_available.to_numpy())
            valid_corr, valid_pvalue = correlation_for_pairs(
                expression_residual.to_numpy()[indices],
                activity_residual.to_numpy()[indices],
                left_safe[valid_rows],
                right_safe[valid_rows],
            )
            corr[valid_rows] = valid_corr
            pvalue[valid_rows] = valid_pvalue
            frame[f"{split}_effect"] = corr.astype("float32")
            frame[f"{split}_pvalue"] = pvalue.astype("float32")
            frame[f"{split}_fdr"] = bh_fdr(pvalue).astype("float32")
            frame[f"{split}_n_patients"] = len(indices)

        train_abs = frame["train_effect"].abs()
        available_train = train_abs[np.isfinite(train_abs)]
        threshold = float(max(0.20, available_train.quantile(0.80)))
        for split in ["train", "validation", "test"]:
            available = frame[f"{split}_effect"].notna()
            frame[f"{split}_membership_label"] = np.where(
                available,
                (
                    frame[f"{split}_effect"].abs().ge(threshold)
                    & frame[f"{split}_fdr"].le(0.20)
                ).astype("float32"),
                np.nan,
            )
            frame[f"{split}_direction_label"] = np.where(
                available,
                frame[f"{split}_effect"].ge(0).astype("float32"),
                np.nan,
            )
        detection_map = dict(zip(lnc, detection_rate, strict=True))
        frame["train_detection_rate"] = frame["lncrna_id"].map(detection_map)
        frame["lncrna_available"] = frame["lncrna_id"].isin(lnc_index).astype("int8")
        frame["pathway_available"] = frame["pathway_family_id"].isin(family_index).astype("int8")
        frame["pair_available"] = frame["train_effect"].notna().astype("int8")
        frame["patient_available"] = frame["pair_available"]
        frame["strict_candidate_available"] = frame["strict_available"].astype("int8")
        state_context = _state_context(matrices.state, split_samples["train"])
        for column, value in state_context.items():
            frame[column] = value
        frame = _merge_fold_evidence(frame, fold_pair_evidence)
        rows.append(frame)
        provenance.append(
            {
                "cancer_id": cancer,
                "patient_fold_id": fold_id,
                **design_info,
                "n_validation_samples": int(len(split_samples["validation"])),
                "n_test_samples": int(len(split_samples["test"])),
                "membership_threshold": threshold,
                "n_strict_candidates": int(frame["candidate_from_strict"].sum()),
                "n_patient_native_candidates": int(frame["candidate_from_patient"].sum()),
                "n_evidence_candidates": int(frame["candidate_from_evidence"].sum()),
                "n_candidate_union": int(len(frame)),
                "n_strict_unavailable_patient_candidates": int(
                    ((frame["candidate_from_patient"] == 1) & (frame["strict_available"] == 0)).sum()
                ),
                "scaler_fit_scope": "train_patients_only",
                "residualizer_fit_scope": "train_patients_only",
                "candidate_selection_scope": "train_patients_only_per_fold",
                "static_family_contract": "SPF_exact_pathway_weighted_activity",
            }
        )
    return pd.concat(rows, ignore_index=True), provenance


def write_contract() -> None:
    contract = {
        "contract_version": "cancer_native_candidate_union_v1",
        "candidate_sources": {
            "strict": str(OOF),
            "patient_native": (
                "train-only top-k residual lncRNA-pathway associations, "
                f"top_k={PATIENT_NATIVE_TOP_K}, min_abs_corr={PATIENT_NATIVE_MIN_ABS_CORR}, "
                f"min_detection={PATIENT_NATIVE_MIN_DETECTION}, "
                f"max_per_fold={PATIENT_NATIVE_MAX_CANDIDATES}"
            ),
            "evidence_optional": [str(path) for path in OPTIONAL_EVIDENCE_CANDIDATE_PATHS],
        },
        "family_source": str(FAMILY_MEMBER),
        "patient_fold_source": str(PATIENT_FOLD),
        "pair_family_contract": "SPF; old PF inputs are explicitly unused",
        "feature_fit_scope": "train patients only for every patient fold",
        "validation_test_use": "labels and evaluation only",
        "missingness": "availability mask; unavailable is never zero evidence",
        "strict_role": "optional cross-cancer expert; no longer the sole candidate gateway",
        "label_definition": (
            "absolute residual correlation above train-derived 80th percentile "
            "(minimum 0.20) and BH FDR <= 0.20"
        ),
    }
    ADAPTER_DATA.mkdir(parents=True, exist_ok=True)
    (ADAPTER_DATA / "data_contract.json").write_text(
        json.dumps(contract, indent=2), encoding="utf-8"
    )
