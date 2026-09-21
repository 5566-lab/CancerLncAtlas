from __future__ import annotations

import hashlib

import pandas as pd

from .contracts import dataframe_sha256


def _patient_order(seed: int, cancer_id: str, patient_id: str) -> str:
    return hashlib.sha256(f"{seed}|{cancer_id}|{patient_id}".encode("utf-8")).hexdigest()


def build_patient_fold_manifest(
    canonical_samples: pd.DataFrame,
    *,
    n_folds: int = 5,
    seed: int = 20260726,
) -> pd.DataFrame:
    required = {"cancer_id", "sample_id", "patient_id"}
    if missing := sorted(required - set(canonical_samples.columns)):
        raise ValueError(f"Canonical sample table lacks columns: {missing}")
    if n_folds < 3:
        raise ValueError("At least three folds are needed for train/validation/test")
    samples = canonical_samples[["cancer_id", "sample_id", "patient_id"]].copy()
    if samples.isna().any().any():
        raise RuntimeError("Canonical sample table contains null cancer/sample/patient IDs")
    for column in ("cancer_id", "sample_id", "patient_id"):
        samples[column] = samples[column].astype(str).str.strip()
        if samples[column].eq("").any():
            raise RuntimeError(f"Canonical sample table contains empty {column}")
    samples = samples.drop_duplicates()
    if samples.sample_id.duplicated().any():
        raise RuntimeError("A sample maps to multiple patients or cancers")
    patient_cancers = samples[["cancer_id", "patient_id"]].drop_duplicates()
    if patient_cancers.patient_id.duplicated().any():
        raise RuntimeError("A patient cannot appear in multiple cancers")
    counts = patient_cancers.groupby("cancer_id", observed=True).patient_id.nunique()
    if (counts < n_folds).any():
        bad = counts[counts < n_folds].to_dict()
        raise RuntimeError(f"Cancers with fewer patients than folds: {bad}")
    patient_cancers["_order"] = [
        _patient_order(seed, cancer, patient)
        for cancer, patient in patient_cancers[["cancer_id", "patient_id"]].itertuples(index=False, name=None)
    ]
    patient_cancers = patient_cancers.sort_values(
        ["cancer_id", "_order", "patient_id"], kind="stable"
    )
    patient_cancers["patient_fold_id"] = (
        patient_cancers.groupby("cancer_id", observed=True).cumcount() % int(n_folds)
    ).astype(int)
    patient_cancers["fold_seed"] = int(seed)
    result = samples.merge(
        patient_cancers.drop(columns="_order"),
        on=["cancer_id", "patient_id"],
        how="left",
        validate="many_to_one",
    ).sort_values(
        ["cancer_id", "patient_fold_id", "patient_id", "sample_id"], kind="stable"
    ).reset_index(drop=True)
    if result.sample_id.duplicated().any():
        raise RuntimeError("Patient fold manifest contains duplicate samples")
    if result.groupby(["cancer_id", "patient_id"], observed=True).patient_fold_id.nunique().gt(1).any():
        raise RuntimeError("A patient's samples cross patient folds")
    result.attrs["patient_manifest_sha256"] = dataframe_sha256(
        result[["cancer_id", "patient_id", "patient_fold_id", "fold_seed"]].drop_duplicates(),
        ["cancer_id", "patient_id"],
    )
    result.attrs["sample_manifest_sha256"] = dataframe_sha256(
        result, ["cancer_id", "patient_id", "sample_id"]
    )
    return result


def assign_outer_split(
    fold_manifest: pd.DataFrame,
    outer_fold: int,
    *,
    n_folds: int = 5,
    validation_offset: int = 1,
) -> pd.DataFrame:
    required = {"cancer_id", "sample_id", "patient_id", "patient_fold_id"}
    if missing := sorted(required - set(fold_manifest.columns)):
        raise ValueError(f"Fold manifest lacks columns: {missing}")
    if not 0 <= int(outer_fold) < int(n_folds):
        raise ValueError("outer_fold is outside the registered fold range")
    validation_fold = (int(outer_fold) + int(validation_offset)) % int(n_folds)
    result = fold_manifest.copy()
    result["outer_fold"] = int(outer_fold)
    result["split"] = "train"
    result.loc[result.patient_fold_id.astype(int).eq(validation_fold), "split"] = "validation"
    result.loc[result.patient_fold_id.astype(int).eq(int(outer_fold)), "split"] = "test"
    observed = set(result.split.astype(str))
    if observed != {"train", "validation", "test"}:
        raise RuntimeError(f"Outer split lacks a registered partition: {sorted(observed)}")
    if result.sample_id.astype(str).duplicated().any():
        raise RuntimeError("Outer split duplicates patient/sample IDs")
    patient_partitions = result.groupby(
        ["cancer_id", "patient_id"], observed=True
    ).agg(folds=("patient_fold_id", "nunique"), splits=("split", "nunique"))
    if patient_partitions[["folds", "splits"]].gt(1).any().any():
        raise RuntimeError("A patient's samples cross folds or outer splits")
    result.attrs["manifest_sha256"] = dataframe_sha256(
        result, ["cancer_id", "patient_id", "sample_id", "outer_fold"]
    )
    return result


def build_all_outer_splits(
    fold_manifest: pd.DataFrame, *, n_folds: int = 5, validation_offset: int = 1
) -> pd.DataFrame:
    return pd.concat(
        [
            assign_outer_split(
                fold_manifest,
                fold,
                n_folds=n_folds,
                validation_offset=validation_offset,
            )
            for fold in range(int(n_folds))
        ],
        ignore_index=True,
    )
