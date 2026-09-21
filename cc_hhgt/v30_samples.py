from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np
import pandas as pd


CANONICAL_KEYS = ["cancer_id", "patient_id", "sample_id"]


def tcga_sample_barcode(sample_id: pd.Series) -> pd.Series:
    return sample_id.astype("string").str.extract(r"^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-[0-9]{2})", expand=False)


def tcga_sample_type_code(sample_id: pd.Series) -> pd.Series:
    barcode = tcga_sample_barcode(sample_id)
    return pd.to_numeric(barcode.str.rsplit("-", n=1).str[-1], errors="coerce").astype("Int64")


def sample_universe_sha256(frame: pd.DataFrame) -> str:
    columns = ["cancer_id", "patient_id", "sample_id", "sample_barcode", "sample_type_code"]
    lines = (
        frame[columns]
        .sort_values(columns, kind="stable")
        .astype(str)
        .agg("\t".join, axis=1)
        .str.cat(sep="\n")
        .encode("utf-8")
    )
    return hashlib.sha256(lines).hexdigest()


def build_canonical_sample_outputs(
    fold_manifest: pd.DataFrame,
    state_scores: pd.DataFrame,
    *,
    target_states: Iterable[str],
    reference_only: Iterable[str] = (),
    minimum_observed: int = 30,
    allowed_tumor_codes: Iterable[int] = range(1, 10),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    target_states = [str(state) for state in target_states]
    required_fold = {*CANONICAL_KEYS, "patient_fold_id", "split"}
    required_score = {"sample_id", "patient_id", "cancer_type", "sample_type_code", "is_tumor", *target_states}
    missing_fold = sorted(required_fold - set(fold_manifest.columns))
    missing_score = sorted(required_score - set(state_scores.columns))
    if missing_fold:
        raise ValueError(f"fold manifest missing columns: {missing_fold}")
    if missing_score:
        raise ValueError(f"state score table missing columns: {missing_score}")

    manifest = fold_manifest.copy()
    canonical = manifest[CANONICAL_KEYS].drop_duplicates().reset_index(drop=True)
    patient_sample_counts = canonical.groupby(["cancer_id", "patient_id"], observed=True).sample_id.nunique()
    if not patient_sample_counts.eq(1).all():
        raise RuntimeError(f"Canonical manifest maps patients to multiple samples: {int(patient_sample_counts.gt(1).sum())}")
    canonical["sample_barcode"] = tcga_sample_barcode(canonical.sample_id)
    canonical["sample_type_code"] = tcga_sample_type_code(canonical.sample_id)
    if canonical.sample_barcode.isna().any() or canonical.sample_type_code.isna().any():
        raise RuntimeError("Canonical manifest contains malformed TCGA sample IDs")
    allowed = set(map(int, allowed_tumor_codes))
    invalid_type = ~canonical.sample_type_code.astype(int).isin(allowed)
    if invalid_type.any():
        observed = canonical.loc[invalid_type, "sample_type_code"].value_counts().to_dict()
        raise RuntimeError(f"Canonical manifest includes non-tumor sample types: {observed}")

    score = state_scores.copy()
    source_score_rows = int(len(score))
    source_score_non_tumor_rows = int(score.is_tumor.ne(True).sum())
    source_score_normal_rows = int(pd.to_numeric(score.sample_type_code, errors="coerce").eq(11).sum())
    score["sample_barcode"] = tcga_sample_barcode(score.sample_id)
    source_score_invalid_barcode_rows = int(score.sample_barcode.isna().sum())
    # Invalid legacy identifiers cannot participate in an exact canonical
    # sample join.  Record and exclude them before duplicate-key checks so
    # multiple NULL barcodes are not mistaken for a biological sample key.
    score = score.loc[score.sample_barcode.notna()].copy()
    score["score_sample_type_code"] = pd.to_numeric(score.sample_type_code, errors="coerce").astype("Int64")
    score_duplicates = score.duplicated(["patient_id", "sample_barcode"], keep=False)
    if score_duplicates.any():
        raise RuntimeError(f"State score table has duplicate patient/sample barcodes: {int(score_duplicates.sum())}")
    selected_columns = [
        "patient_id",
        "sample_barcode",
        "cancer_type",
        "score_sample_type_code",
        "is_tumor",
        *target_states,
    ]
    canonical = canonical.merge(score[selected_columns], on=["patient_id", "sample_barcode"], how="left", validate="one_to_one", indicator=True)
    if not canonical._merge.eq("both").all():
        raise RuntimeError(f"Canonical samples missing from state score table: {int(canonical._merge.ne('both').sum())}")
    cancer_mismatch = canonical.cancer_type.notna() & canonical.cancer_id.ne(canonical.cancer_type)
    if cancer_mismatch.any():
        raise RuntimeError(f"Canonical state-score cancer mismatches: {int(cancer_mismatch.sum())}")
    if not canonical.is_tumor.eq(True).all():
        raise RuntimeError(f"Non-tumor score rows entered canonical table: {int(canonical.is_tumor.ne(True).sum())}")
    if not canonical.score_sample_type_code.astype(int).isin(allowed).all():
        raise RuntimeError("State score sample type disagrees with tumor-only canonical contract")
    canonical = canonical.drop(columns=["_merge"])
    universe_hash = sample_universe_sha256(canonical)
    canonical["sample_universe_sha256"] = universe_hash

    expanded = manifest.merge(
        canonical[[*CANONICAL_KEYS, "sample_barcode", "sample_type_code", "sample_universe_sha256"]],
        on=CANONICAL_KEYS,
        how="inner",
        validate="many_to_one",
    )
    if len(expanded) != len(manifest):
        raise RuntimeError("Canonical fold expansion dropped manifest rows")

    value_rows: list[pd.DataFrame] = []
    eligibility_rows: list[dict] = []
    reference = set(map(str, reference_only))
    for cancer_id, cancer in canonical.groupby("cancer_id", observed=True):
        for state_id in target_states:
            values = pd.to_numeric(cancer[state_id], errors="coerce")
            observed = values.notna()
            n_observed = int(observed.sum())
            n_missing = int((~observed).sum())
            is_reference = str(cancer_id) in reference
            eligible = n_observed >= minimum_observed and not is_reference
            if is_reference:
                reason = "REFERENCE_ONLY"
            elif n_observed < minimum_observed:
                reason = f"INSUFFICIENT_STATE_COMPLETE_CASES:{n_observed}<{minimum_observed}"
            else:
                reason = ""
            if observed.any():
                part = cancer.loc[observed, CANONICAL_KEYS + ["sample_barcode", "sample_type_code", "sample_universe_sha256"]].copy()
                part["state_id"] = state_id
                part["state_value"] = values.loc[observed].to_numpy(float)
                part["state_observed"] = True
                value_rows.append(part)
            eligibility_rows.append(
                {
                    "cancer_id": str(cancer_id),
                    "state_id": state_id,
                    "n_canonical_samples": int(len(cancer)),
                    "n_observed": n_observed,
                    "n_missing": n_missing,
                    "missing_rate": n_missing / len(cancer) if len(cancer) else np.nan,
                    "eligibility": "ELIGIBLE" if eligible else "UNAVAILABLE",
                    "unavailable_reason": reason,
                    "cancer_role": "REFERENCE_ONLY" if is_reference else "FORMAL",
                    "sample_universe_sha256": universe_hash,
                    "state_outcome_imputation": "NONE",
                    "evaluation_policy": "STATE_SPECIFIC_COMPLETE_CASE",
                }
            )
    state_values = pd.concat(value_rows, ignore_index=True) if value_rows else pd.DataFrame()
    eligibility = pd.DataFrame(eligibility_rows)
    audit = {
        "status": "PASS",
        "canonical_samples": int(len(canonical)),
        "canonical_patients": int(canonical[["cancer_id", "patient_id"]].drop_duplicates().shape[0]),
        "canonical_cancers": int(canonical.cancer_id.nunique()),
        "fold_rows": int(len(expanded)),
        "normal_samples": int((~canonical.sample_type_code.astype(int).isin(allowed)).sum()),
        "non_tumor_score_rows": int(canonical.is_tumor.ne(True).sum()),
        "state_value_rows": int(len(state_values)),
        "state_nan_rows": int(state_values.state_value.isna().sum()) if len(state_values) else 0,
        "source_score_rows": source_score_rows,
        "source_score_invalid_barcode_rows": source_score_invalid_barcode_rows,
        "source_score_non_tumor_rows": source_score_non_tumor_rows,
        "source_score_type11_rows": source_score_normal_rows,
        "sample_universe_sha256": universe_hash,
        "reference_only": sorted(reference),
        "minimum_observed": int(minimum_observed),
    }
    return canonical, expanded, state_values, eligibility, audit
