"""Typed exact-universe adapter for the V3.2 single-cell expert.

The native single-cell head is sparse and keyed by dataset/cell type.  Fusion
is keyed by cancer/lncRNA/exact pathway and requires one typed row for every
primary candidate.  This adapter performs only the declared aggregation/left
join; it never turns absence into a numeric score.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
TARGET_KEYS = ("cancer_id", "lncrna_id", "pathway_id")
PROBABILITY_COLUMN = "single_cell_replication_probability"
AVAILABILITY_COLUMN = "single_cell_available"
REASON_COLUMN = "single_cell_unavailable_reason"
ADAPTER_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_EXACT_FUSION_EXPERT_V1"


class SingleCellFusionAdapterError(RuntimeError):
    """Raised when sparse single-cell results cannot be typed safely."""


def _table(value: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    path = Path(value)
    if not path.exists():
        raise SingleCellFusionAdapterError(f"Input is missing: {path}")
    return pd.read_parquet(path)


def _keys(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    missing = sorted(set(TARGET_KEYS) - set(frame.columns))
    if missing:
        raise SingleCellFusionAdapterError(f"{label} lacks keys: {missing}")
    result = frame.copy()
    for column in TARGET_KEYS:
        result[column] = result[column].astype("string").str.strip()
        if result[column].isna().any() or result[column].eq("").any():
            raise SingleCellFusionAdapterError(f"{label}.{column} contains empty IDs")
    result["cancer_id"] = result.cancer_id.str.upper()
    if result.duplicated(list(TARGET_KEYS)).any():
        raise SingleCellFusionAdapterError(f"{label} duplicates exact target keys")
    return result


def build_single_cell_exact_fusion_expert(
    candidates: pd.DataFrame | str | Path,
    single_cell_exact: pd.DataFrame | str | Path,
) -> pd.DataFrame:
    """Return a full candidate-universe, null-aware single-cell expert."""

    primary = _keys(_table(candidates), "candidates")[list(TARGET_KEYS)]
    sparse = _keys(_table(single_cell_exact), "single_cell_exact")
    required = {PROBABILITY_COLUMN, AVAILABILITY_COLUMN, REASON_COLUMN, "analysis_version"}
    missing = sorted(required - set(sparse.columns))
    if missing:
        raise SingleCellFusionAdapterError(f"single_cell_exact lacks columns: {missing}")
    if not sparse.analysis_version.astype(str).eq(ANALYSIS_VERSION).all():
        raise SingleCellFusionAdapterError("single_cell_exact is not current V3.2")
    if sparse[AVAILABILITY_COLUMN].isna().any():
        raise SingleCellFusionAdapterError("single-cell availability contains nulls")
    available = sparse[AVAILABILITY_COLUMN].astype(bool)
    probability = pd.to_numeric(sparse[PROBABILITY_COLUMN], errors="coerce")
    if probability.loc[available].isna().any():
        raise SingleCellFusionAdapterError("available single-cell rows lack probability")
    if not probability.loc[available].between(0.0, 1.0).all():
        raise SingleCellFusionAdapterError("single-cell probability is outside [0,1]")
    if probability.loc[~available].notna().any():
        raise SingleCellFusionAdapterError("unavailable single-cell rows contain a score")
    if sparse.loc[~available, REASON_COLUMN].isna().any():
        raise SingleCellFusionAdapterError("unavailable single-cell rows lack a reason")

    extras = sparse[list(TARGET_KEYS)].merge(
        primary,
        on=list(TARGET_KEYS),
        how="left",
        indicator=True,
        validate="one_to_one",
    )
    if extras._merge.ne("both").any():
        raise SingleCellFusionAdapterError("single-cell exact rows escape primary candidates")

    keep = sparse[list(TARGET_KEYS) + [PROBABILITY_COLUMN, AVAILABILITY_COLUMN, REASON_COLUMN]].copy()
    keep[PROBABILITY_COLUMN] = probability.astype(float)
    result = primary.merge(keep, on=list(TARGET_KEYS), how="left", validate="one_to_one")
    absent = result[AVAILABILITY_COLUMN].isna()
    result.loc[absent, AVAILABILITY_COLUMN] = False
    result[AVAILABILITY_COLUMN] = result[AVAILABILITY_COLUMN].astype(bool)
    result.loc[absent, PROBABILITY_COLUMN] = np.nan
    result.loc[absent, REASON_COLUMN] = "NO_SINGLE_CELL_EXACT_PAIR_MEASUREMENT"

    unavailable = ~result[AVAILABILITY_COLUMN]
    if result.loc[unavailable, PROBABILITY_COLUMN].notna().any():
        raise SingleCellFusionAdapterError("typed unavailable rows contain a score")
    if result.loc[unavailable, REASON_COLUMN].isna().any():
        raise SingleCellFusionAdapterError("typed unavailable rows lack a reason")
    if len(result) != len(primary) or result.duplicated(list(TARGET_KEYS)).any():
        raise SingleCellFusionAdapterError("adapter changed the exact candidate universe")

    result["analysis_version"] = ANALYSIS_VERSION
    result["module_id"] = "single_cell"
    result["adapter_format"] = ADAPTER_FORMAT
    result["target_level"] = "cancer_x_lncrna_x_exact_pathway"
    result["source_target_level"] = "dataset_x_celltype_x_lncrna_x_exact_pathway"
    result["aggregation"] = "MEAN_OF_DONOR_OR_DATASET_BLOCKED_OOF_PREDICTIONS"
    result["direct_target_evidence"] = False
    result["family_to_exact_broadcast"] = False
    result["changes_primary_ranking"] = False
    result["availability_encoding"] = "null_with_reason"
    return result.sort_values(list(TARGET_KEYS), kind="stable").reset_index(drop=True)


__all__ = [
    "ADAPTER_FORMAT",
    "ANALYSIS_VERSION",
    "AVAILABILITY_COLUMN",
    "PROBABILITY_COLUMN",
    "REASON_COLUMN",
    "SingleCellFusionAdapterError",
    "TARGET_KEYS",
    "build_single_cell_exact_fusion_expert",
]
