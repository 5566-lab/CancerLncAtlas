"""Context-aware, leakage-safe V3.2 single-cell training extension.

This is an additive V2 implementation.  It does not mutate the formal V1
single-cell binding or the immutable exact-pathway primary score.  The V1
trainer collapsed pathway activity to ``dataset_id x pathway_id``.  V2 keeps
the primary cell population in the feature key and exposes both the local
activity and its dataset-global contrast.

Observed signed association effects and FDR values are labels/evidence.  They
are materialised in a separate sidecar and are never admitted to the feature
matrix.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .input_lineage import artifact_sha256
from .single_cell_training import (
    ANALYSIS_VERSION,
    N_FOLDS,
    SingleCellTrainingConfig,
    SingleCellTrainingError,
    _FORMAL_SOURCE_TIERS,
    _assert_source_path,
    _normalise_activity,
    _normalise_lnc_celltype,
    _normalise_token,
    _predict,
    _read_table,
    _sample_indices,
    _source_audit,
    build_blocked_folds,
    candidate_core,
    candidate_core_availability,
    exact_candidate_join,
    fit_private_head,
    load_fold_core_embeddings,
    normalise_candidates,
    normalise_dataset_manifest,
    normalise_single_cell_associations,
    split_block_ids,
    validate_core_manifest,
)


MODULE_ID = "single_cell_context_v2"
FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_CONTEXT_AWARE_TRAINING_V2"
PREFLIGHT_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_CONTEXT_V2_PREFLIGHT_V1"
RUN_PLAN_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_CONTEXT_V2_RUN_PLAN_V1"
LINEAGE_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_CONTEXT_V2_LINEAGE_V1"
CHECKPOINT_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_CONTEXT_PRIVATE_HEAD_V2"
PREDICTION_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_CONTEXT_PREDICTIONS_V2"
EVIDENCE_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_SIGNED_EVIDENCE_SIDECAR_V1"
DONOR_FOLD_MANIFEST_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_CONTEXT_DONOR_FOLDS_V1"
FOLD_LOCAL_FEATURE_MANIFEST_FORMAT = (
    "CC_HHGT_V3_2_SINGLE_CELL_CONTEXT_FOLD_LOCAL_FEATURES_V1"
)
ROOT_APPROVAL_TOKEN = "ROOT_APPROVED_SINGLE_CELL_CONTEXT_V2_FULL_TRAINING"

CONTEXT_KEYS = ("dataset_id", "cancer_id", "cell_key", "pathway_id")
CONTEXT_ID_KEYS = ("dataset_id", "cancer_id", "cell_key")
GLOBAL_KEYS = ("dataset_id", "cancer_id", "pathway_id")

CONTEXT_FEATURES = (
    "lnc_detection_rate",
    "lnc_mean_log_expression",
    "lnc_specificity_tau",
    "pathway_activity_context",
    "pathway_activity_global",
    "pathway_activity_delta",
    "pathway_ucell_context",
    "pathway_ucell_global",
    "pathway_ucell_delta",
    "pathway_pseudotime_context",
    "pathway_pseudotime_global",
    "pathway_pseudotime_delta",
    "log1p_context_n_cells",
    "log1p_global_n_cells",
    "lnc_feature_available",
    "pathway_context_available",
    "pathway_global_available",
    "compartment_malignant",
    "compartment_immune",
    "compartment_stromal",
    "compartment_other",
)

_IMMUNE_KEYS = frozenset(
    {
        "b_cell", "t_cell", "nk_cell", "myeloid", "plasma_cell",
        "hematopoietic_other", "hematopoietic_progenitor", "lymphoid",
        "dendritic", "dendritic_cell", "mast", "mast_cell", "monocyte",
        "macrophage", "neutrophil", "immune",
    }
)
_STROMAL_KEYS = frozenset(
    {
        "endothelial", "fibroblast_stromal", "fibroblast", "stromal",
        "pericyte", "smooth_muscle", "smooth_muscle_cell",
    }
)
_MALIGNANT_KEYS = frozenset({"malignant", "malignant_candidate"})
_FORBIDDEN_FEATURE_TOKENS = frozenset(
    {
        "rho", "fdr", "q_value", "p_value", "padj", "association",
        "evidence", "target", "label", "direction", "correlation", "beta",
    }
)
_FORBIDDEN_HISTORICAL_BASENAMES = frozenset(
    {
        "sc_pseudotime_pathway",
        "standardized_pseudotime",
        "standardised_pseudotime",
        "ucell_aggregate",
    }
)

# These are historical audit facts, not inputs.  No historical path is accepted
# by this module and the preflight deliberately does not read either table.
FORBIDDEN_HISTORICAL_INVENTORY = (
    {
        "generation": "V3.0",
        "logical_table": "standardized_pseudotime",
        "numeric_rows": 454_198,
        "cancers": 33,
        "status": "DERIVED_STAGING_FORBIDDEN_NOT_READ",
    },
    {
        "generation": "V3.0",
        "logical_table": "ucell_aggregate",
        "numeric_rows": 6_483_828,
        "cancers": 33,
        "status": "DERIVED_STAGING_FORBIDDEN_NOT_READ",
    },
)


class SingleCellContextV2Error(SingleCellTrainingError):
    """Raised when a context-V2 lineage, feature, split, or gate is invalid."""


def assert_not_historical_source(path: str | Path) -> None:
    """Reject known V3.0 derived/staging assets before opening the path."""

    source = Path(path)
    stem = _normalise_token(source.stem)
    parts = {_normalise_token(part) for part in source.parts}
    if stem in _FORBIDDEN_HISTORICAL_BASENAMES or parts.intersection(
        {"v3_0", "v30", "cc_hhgt_v3_0", "derived_staging"}
    ):
        raise SingleCellContextV2Error(
            f"Historical V3.0 derived/staging single-cell source is forbidden: {source}"
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    temporary.unlink(missing_ok=True)
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def compartment_from_cell_key(value: Any) -> str:
    """Map a primary cell population to the four explicit V2 compartments."""

    key = _normalise_token(str(value).split("::", 1)[0])
    if key in _MALIGNANT_KEYS:
        return "malignant"
    if key in _IMMUNE_KEYS or any(
        token in key
        for token in ("immune", "myeloid", "lymph", "macroph", "monocyte")
    ):
        return "immune"
    if key in _STROMAL_KEYS or any(
        token in key for token in ("stromal", "fibroblast", "endothelial", "pericyte")
    ):
        return "stromal"
    return "other"


def assert_leakage_free_feature_schema(columns: Sequence[str]) -> None:
    """Fail if an association label/evidence field is proposed as a feature."""

    bad: list[str] = []
    for original in columns:
        key = _normalise_token(original)
        if any(token == key or token in key.split("_") for token in _FORBIDDEN_FEATURE_TOKENS):
            bad.append(str(original))
    if bad:
        raise SingleCellContextV2Error(
            "Association rho/FDR/evidence fields cannot enter context-V2 features: "
            f"{sorted(set(bad))}"
        )


def _weighted_activity_group(frame: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    local = frame.copy()
    local = local.loc[local.activity_value.notna()].copy()
    if local.empty:
        return pd.DataFrame(columns=[*keys, "activity_kind", "activity_value", "n_cells"])
    weight = pd.to_numeric(local.n_cells, errors="coerce").fillna(1.0).clip(lower=1.0)
    local["_weight"] = weight
    local["_weighted_value"] = pd.to_numeric(
        local.activity_value, errors="coerce"
    ) * weight
    grouped = local.groupby([*keys, "activity_kind"], observed=True, as_index=False).agg(
        _weighted_value=("_weighted_value", "sum"),
        _weight=("_weight", "sum"),
        n_cells=("n_cells", "sum"),
    )
    grouped["activity_value"] = grouped._weighted_value / grouped._weight
    return grouped[[*keys, "activity_kind", "activity_value", "n_cells"]]


def _activity_wide(frame: pd.DataFrame, keys: Sequence[str], suffix: str) -> pd.DataFrame:
    grouped = _weighted_activity_group(frame, keys)
    if grouped.empty:
        return pd.DataFrame(columns=list(keys))
    values = grouped.pivot_table(
        index=list(keys), columns="activity_kind", values="activity_value", aggfunc="mean"
    ).reset_index()
    values.columns.name = None
    rename = {
        "activity": f"pathway_activity_{suffix}",
        "ucell": f"pathway_ucell_{suffix}",
        "pseudotime": f"pathway_pseudotime_{suffix}",
    }
    values = values.rename(columns=rename)
    cells = grouped.groupby(list(keys), observed=True, as_index=False).agg(
        **{f"{suffix}_n_cells": ("n_cells", "sum")}
    )
    return values.merge(cells, on=list(keys), how="outer", validate="one_to_one")


def _association_cell_key(frame: pd.DataFrame) -> pd.Series:
    if "cell_type_major" in frame:
        source = frame.cell_type_major
    elif "cell_key" in frame:
        source = frame.cell_key
    elif "cell_type" in frame:
        source = frame.cell_type.astype(str).str.split("::", n=1).str[0]
    else:
        raise SingleCellContextV2Error("Association rows lack a primary cell population")
    return source.astype(str).map(_normalise_token)


def build_context_domain_feature_frame(
    associations: pd.DataFrame,
    lnc_celltype: pd.DataFrame,
    activity: pd.DataFrame,
) -> pd.DataFrame:
    """Build leakage-free context and context-minus-global domain features.

    The pathway join is literal on
    ``dataset_id,cancer_id,cell_key,pathway_id``.  Dataset-global pathway
    values are retained in separate columns; they never replace the context
    value.
    """

    assert_leakage_free_feature_schema(CONTEXT_FEATURES)
    required = {"dataset_id", "cancer_id", "lncrna_id", "pathway_id"}
    if missing := sorted(required - set(associations.columns)):
        raise SingleCellContextV2Error(f"Context associations lack keys: {missing}")
    base = associations.reset_index(drop=True).copy()
    base["_row_id"] = np.arange(len(base), dtype=np.int64)
    base["cell_key"] = _association_cell_key(base)
    if base.cell_key.eq("").any():
        raise SingleCellContextV2Error("Empty cell_key is forbidden for context-V2 training")

    lnc = lnc_celltype.copy()
    if "cell_key" not in lnc:
        source = "cell_type" if "cell_type" in lnc else "cell_type_major"
        lnc["cell_key"] = lnc[source].astype(str).map(_normalise_token)
    lnc_keys = ["dataset_id", "cancer_id", "lncrna_id", "cell_key"]
    lnc_values = [
        "lnc_detection_rate", "lnc_mean_log_expression", "lnc_specificity_tau",
        "lnc_n_cells",
    ]
    for column in lnc_values:
        if column not in lnc:
            lnc[column] = np.nan
    lnc = lnc.groupby(lnc_keys, observed=True, as_index=False).agg(
        lnc_detection_rate=("lnc_detection_rate", "mean"),
        lnc_mean_log_expression=("lnc_mean_log_expression", "mean"),
        lnc_specificity_tau=("lnc_specificity_tau", "mean"),
        lnc_n_cells=("lnc_n_cells", "max"),
    )
    merged = base.merge(lnc, on=lnc_keys, how="left", validate="many_to_one")

    local = activity.copy()
    if not local.empty:
        if "cell_key" not in local:
            source = "cell_type" if "cell_type" in local else "cell_type_major"
            local["cell_key"] = local[source].astype(str).map(_normalise_token)
        for column in ("n_cells", "activity_kind"):
            if column not in local:
                local[column] = np.nan if column == "n_cells" else "activity"
        context_wide = _activity_wide(
            local.loc[local.cell_key.astype(str).ne("")], CONTEXT_KEYS, "context"
        )
        global_wide = _activity_wide(local, GLOBAL_KEYS, "global")
    else:
        context_wide = pd.DataFrame(columns=list(CONTEXT_KEYS))
        global_wide = pd.DataFrame(columns=list(GLOBAL_KEYS))
    merged = merged.merge(
        context_wide, on=list(CONTEXT_KEYS), how="left", validate="many_to_one"
    )
    merged = merged.merge(
        global_wide, on=list(GLOBAL_KEYS), how="left", validate="many_to_one"
    )
    for column in ("lnc_n_cells", "context_n_cells", "global_n_cells"):
        if column not in merged:
            merged[column] = np.nan

    for kind in ("activity", "ucell", "pseudotime"):
        context = f"pathway_{kind}_context"
        global_name = f"pathway_{kind}_global"
        delta = f"pathway_{kind}_delta"
        if context not in merged:
            merged[context] = np.nan
        if global_name not in merged:
            merged[global_name] = np.nan
        merged[delta] = pd.to_numeric(merged[context], errors="coerce") - pd.to_numeric(
            merged[global_name], errors="coerce"
        )
    merged["lnc_feature_available"] = merged[
        ["lnc_detection_rate", "lnc_mean_log_expression", "lnc_specificity_tau"]
    ].notna().any(axis=1).astype(float)
    merged["pathway_context_available"] = merged[
        [
            "pathway_activity_context", "pathway_ucell_context",
            "pathway_pseudotime_context",
        ]
    ].notna().any(axis=1).astype(float)
    merged["pathway_global_available"] = merged[
        [
            "pathway_activity_global", "pathway_ucell_global",
            "pathway_pseudotime_global",
        ]
    ].notna().any(axis=1).astype(float)
    merged["log1p_context_n_cells"] = np.log1p(
        pd.concat(
            [
                pd.to_numeric(merged.get("lnc_n_cells"), errors="coerce"),
                pd.to_numeric(merged.get("context_n_cells"), errors="coerce"),
            ],
            axis=1,
        ).max(axis=1, skipna=True).fillna(0.0).clip(lower=0.0)
    )
    merged["log1p_global_n_cells"] = np.log1p(
        pd.to_numeric(merged.get("global_n_cells"), errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
    )
    merged["compartment"] = merged.cell_key.map(compartment_from_cell_key)
    for compartment in ("malignant", "immune", "stromal", "other"):
        merged[f"compartment_{compartment}"] = merged.compartment.eq(compartment).astype(float)
    if not merged[
        [
            "compartment_malignant", "compartment_immune",
            "compartment_stromal", "compartment_other",
        ]
    ].sum(axis=1).eq(1.0).all():
        raise SingleCellContextV2Error("Compartment encoding is not one-hot")
    for column in CONTEXT_FEATURES:
        merged[column] = pd.to_numeric(merged[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0).astype(np.float32)
    return merged.sort_values("_row_id", kind="stable").reset_index(drop=True)


def context_feature_matrix(feature_frame: pd.DataFrame) -> np.ndarray:
    assert_leakage_free_feature_schema(CONTEXT_FEATURES)
    if missing := sorted(set(CONTEXT_FEATURES) - set(feature_frame.columns)):
        raise SingleCellContextV2Error(f"Context feature frame lacks {missing}")
    return feature_frame.loc[:, CONTEXT_FEATURES].to_numpy(np.float32, copy=True)


def build_signed_evidence_sidecar(associations: pd.DataFrame) -> pd.DataFrame:
    """Materialise observed signed rho/FDR separately from model features."""

    required = {
        "dataset_id", "cancer_id", "lncrna_id", "pathway_id",
        "association_effect", "association_fdr",
    }
    if missing := sorted(required - set(associations.columns)):
        raise SingleCellContextV2Error(f"Signed evidence input lacks {missing}")
    result = pd.DataFrame(index=associations.index)
    for column in (
        "dataset_id", "cancer_id", "donor_id", "cell_type", "cell_type_major",
        "cell_state", "lncrna_id", "pathway_id", "n_observations",
    ):
        result[column] = associations[column] if column in associations else pd.NA
    result["cell_key"] = _association_cell_key(associations)
    result["observed_signed_rho"] = pd.to_numeric(
        associations.association_effect, errors="coerce"
    )
    result["observed_fdr"] = pd.to_numeric(
        associations.association_fdr, errors="coerce"
    ).clip(0.0, 1.0)
    result["observed_signed_rho_x_one_minus_fdr"] = (
        result.observed_signed_rho * (1.0 - result.observed_fdr)
    )
    result["observed_direction"] = np.select(
        [result.observed_signed_rho.gt(0), result.observed_signed_rho.lt(0)],
        ["positive", "negative"],
        default="zero",
    )
    result["direction_is_observed_label_not_feature"] = True
    key_columns = [
        "dataset_id", "cancer_id", "donor_id", "cell_key", "lncrna_id", "pathway_id"
    ]
    result["evidence_row_id"] = [
        hashlib.sha256("|".join(map(str, row)).encode("utf-8")).hexdigest()
        for row in result[key_columns].itertuples(index=False, name=None)
    ]
    return result.reset_index(drop=True)


def _column_by_alias(
    frame: pd.DataFrame, aliases: Sequence[str]
) -> str | None:
    lookup = {_normalise_token(column): str(column) for column in frame.columns}
    for alias in aliases:
        if _normalise_token(alias) in lookup:
            return lookup[_normalise_token(alias)]
    return None


def _truth_series(frame: pd.DataFrame, column: str | None) -> pd.Series:
    if column is None:
        return pd.Series(False, index=frame.index, dtype=bool)
    values = frame[column]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    return values.fillna("").astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes"}
    )


def _v32_series(frame: pd.DataFrame, column: str | None) -> pd.Series:
    if column is None:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame[column].fillna("").astype(str).str.contains(
        r"(?<!\d)v?3[._-]?2(?!\d)", case=False, regex=True
    )


def audit_donor_resolved_association_source(frame: pd.DataFrame) -> dict[str, Any]:
    """Prove that targets are donor-resolved, not aggregate rho broadcasts.

    A donor identifier alone is never sufficient.  The source must explicitly
    declare donor observation/target semantics and a unique fresh row lineage.
    Cross-donor pseudobulk Spearman (including the current formal V1 method) is
    an aggregate context target and is forbidden as a donor-level target.
    """

    donor_column = _column_by_alias(frame, ("donor_id", "patient_id", "subject_id"))
    method_column = _column_by_alias(frame, ("association_method", "method"))
    unit_column = _column_by_alias(
        frame, ("association_observation_unit", "observation_unit", "target_unit")
    )
    target_level_column = _column_by_alias(
        frame, ("association_target_level", "target_level")
    )
    resolved_column = _column_by_alias(
        frame, ("donor_resolved_target", "is_donor_resolved_target")
    )
    row_id_column = _column_by_alias(
        frame, ("donor_target_row_id", "association_row_id", "source_row_id")
    )
    generation_column = _column_by_alias(
        frame, ("generation", "source_generation")
    )
    donor = (
        frame[donor_column].fillna("").astype(str).str.strip()
        if donor_column is not None
        else pd.Series("", index=frame.index, dtype=str)
    )
    methods = (
        frame[method_column].fillna("").astype(str).map(_normalise_token)
        if method_column is not None
        else pd.Series("", index=frame.index, dtype=str)
    )
    aggregate_method = methods.map(
        lambda value: bool(
            any(
                token in value
                for token in (
                    "across_donor", "cross_donor", "pooled_donor", "aggregate",
                    "donor_pseudobulk_spearman", "pseudobulk_spearman",
                )
            )
            or ("spearman" in value and "within_donor" not in value)
        )
    )
    unit = (
        frame[unit_column].fillna("").astype(str).map(_normalise_token)
        if unit_column is not None
        else pd.Series("", index=frame.index, dtype=str)
    )
    target_level = (
        frame[target_level_column].fillna("").astype(str).map(_normalise_token)
        if target_level_column is not None
        else pd.Series("", index=frame.index, dtype=str)
    )
    valid_units = {"donor", "within_donor", "donor_cell", "donor_resolved"}
    valid_levels = {
        "donor_x_celltype_x_lncrna_x_exact_pathway",
        "dataset_x_donor_x_celltype_x_lncrna_x_exact_pathway",
    }
    explicit_resolved = _truth_series(frame, resolved_column)
    fresh_generation = _v32_series(frame, generation_column)
    row_id = (
        frame[row_id_column].fillna("").astype(str).str.strip()
        if row_id_column is not None
        else pd.Series("", index=frame.index, dtype=str)
    )
    row_id_valid = bool(
        len(frame) > 0
        and row_id.ne("").all()
        and int(row_id.nunique()) == int(len(frame))
    )
    broadcast_groups = 0
    effect_column = _column_by_alias(
        frame, ("rho", "effect", "association_effect", "correlation", "beta")
    )
    fdr_column = _column_by_alias(frame, ("fdr", "q_value", "padj", "p_value"))
    identity_aliases = {
        "dataset_id": ("dataset_id",),
        "cancer_id": ("cancer_id",),
        "cell_type": ("cell_type", "cell_type_major", "cell_population"),
        "lncrna_id": ("lncrna_id", "lncRNA_id"),
        "pathway_id": ("pathway_id",),
    }
    identity_columns = [
        _column_by_alias(frame, aliases) for aliases in identity_aliases.values()
    ]
    if (
        donor_column is not None
        and effect_column is not None
        and fdr_column is not None
        and all(column is not None for column in identity_columns)
        and not aggregate_method.any()
        and explicit_resolved.all()
        and row_id_valid
        and len(frame)
    ):
        signature = frame[
            [column for column in identity_columns if column is not None]
            + [effect_column, fdr_column, donor_column]
        ].copy()
        grouped = signature.groupby(
            [column for column in identity_columns if column is not None]
            + [effect_column, fdr_column],
            observed=True,
        )[donor_column].nunique()
        broadcast_groups = int(grouped.gt(1).sum())
    reasons: list[str] = []
    if len(frame) == 0:
        reasons.append("EMPTY_ASSOCIATION_SOURCE")
    if donor_column is None or donor.eq("").any():
        reasons.append("DONOR_ID_MISSING")
    if method_column is None:
        reasons.append("ASSOCIATION_METHOD_MISSING")
    elif aggregate_method.any():
        reasons.append("CROSS_DONOR_AGGREGATE_METHOD_FORBIDDEN")
    if unit_column is None or not unit.isin(valid_units).all():
        reasons.append("DONOR_OBSERVATION_UNIT_NOT_PROVEN")
    if target_level_column is None or not target_level.isin(valid_levels).all():
        reasons.append("DONOR_TARGET_LEVEL_NOT_PROVEN")
    if resolved_column is None or not explicit_resolved.all():
        reasons.append("DONOR_RESOLVED_TARGET_ATTESTATION_MISSING")
    if generation_column is None or not fresh_generation.all():
        reasons.append("FRESH_V32_TARGET_GENERATION_NOT_PROVEN")
    if not row_id_valid:
        reasons.append("UNIQUE_DONOR_TARGET_ROW_LINEAGE_MISSING")
    if broadcast_groups:
        reasons.append("AGGREGATE_TARGET_BROADCAST_ACROSS_DONORS")
    return {
        "policy": "FRESH_DONOR_RESOLVED_TARGET_NO_AGGREGATE_RHO_BROADCAST",
        "rows": int(len(frame)),
        "donor_column": donor_column,
        "rows_with_donor_id": int(donor.ne("").sum()),
        "rows_without_donor_id": int(donor.eq("").sum()),
        "distinct_donors": int(donor.loc[donor.ne("")].nunique()),
        "association_methods": sorted(set(methods.astype(str)))[:50],
        "aggregate_method_rows": int(aggregate_method.sum()),
        "broadcast_signature_groups": broadcast_groups,
        "unique_donor_target_row_lineage": row_id_valid,
        "ready": not reasons,
        "reasons": reasons,
    }


def audit_donor_resolved_lnc_source(frame: pd.DataFrame) -> dict[str, Any]:
    """Require fresh donor-resolved non-predictive lncRNA measurements."""

    donor_column = _column_by_alias(frame, ("donor_id", "patient_id", "subject_id"))
    resolved_column = _column_by_alias(
        frame, ("donor_resolved_feature", "is_donor_resolved_feature")
    )
    row_id_column = _column_by_alias(
        frame, ("donor_feature_row_id", "source_row_id")
    )
    generation_column = _column_by_alias(
        frame, ("generation", "source_generation")
    )
    donor = (
        frame[donor_column].fillna("").astype(str).str.strip()
        if donor_column is not None
        else pd.Series("", index=frame.index, dtype=str)
    )
    row_id = (
        frame[row_id_column].fillna("").astype(str).str.strip()
        if row_id_column is not None
        else pd.Series("", index=frame.index, dtype=str)
    )
    reasons: list[str] = []
    if len(frame) == 0:
        reasons.append("EMPTY_LNCRNA_FEATURE_SOURCE")
    if donor_column is None or donor.eq("").any():
        reasons.append("DONOR_RESOLVED_LNCRNA_FEATURES_MISSING")
    if resolved_column is None or not _truth_series(frame, resolved_column).all():
        reasons.append("DONOR_RESOLVED_LNCRNA_ATTESTATION_MISSING")
    if generation_column is None or not _v32_series(frame, generation_column).all():
        reasons.append("FRESH_V32_LNCRNA_GENERATION_NOT_PROVEN")
    unique_lineage = bool(
        len(frame) > 0
        and row_id.ne("").all()
        and int(row_id.nunique()) == int(len(frame))
    )
    if not unique_lineage:
        reasons.append("UNIQUE_DONOR_LNCRNA_ROW_LINEAGE_MISSING")
    return {
        "policy": "FRESH_DONOR_RESOLVED_NONPREDICTIVE_LNCRNA_FEATURES",
        "rows": int(len(frame)),
        "donor_column": donor_column,
        "rows_with_donor_id": int(donor.ne("").sum()),
        "distinct_donors": int(donor.loc[donor.ne("")].nunique()),
        "unique_donor_feature_row_lineage": unique_lineage,
        "ready": not reasons,
        "reasons": reasons,
    }


def require_donor_blocked_targets(
    associations: pd.DataFrame,
    *,
    source_semantic_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit donor IDs and their source semantics; IDs alone never unlock."""

    donor = (
        associations.donor_id.fillna("").astype(str).str.strip()
        if "donor_id" in associations
        else pd.Series("", index=associations.index, dtype=str)
    )
    semantic_ready = bool(
        source_semantic_audit is not None
        and source_semantic_audit.get("ready") is True
    )
    missing_rows = int(donor.eq("").sum())
    distinct = int(donor.loc[donor.ne("")].nunique())
    return {
        "policy": "STRICT_DONOR_BLOCKED_FIVE_FOLD_NO_DATASET_FALLBACK",
        "rows": int(len(associations)),
        "rows_with_donor_id": int(donor.ne("").sum()),
        "rows_without_donor_id": missing_rows,
        "distinct_donors": distinct,
        "source_semantics_ready": semantic_ready,
        "source_semantic_reasons": (
            list(source_semantic_audit.get("reasons", []))
            if source_semantic_audit is not None else ["SOURCE_SEMANTICS_NOT_AUDITED"]
        ),
        "ready": bool(
            len(associations) > 0
            and missing_rows == 0
            and distinct >= N_FOLDS
            and semantic_ready
        ),
        "fallback_used": False,
    }


def _donor_block_id(frame: pd.DataFrame) -> pd.Series:
    return (
        frame.dataset_id.astype(str).str.strip()
        + "|donor:"
        + frame.donor_id.fillna("").astype(str).str.strip()
    )


def validate_context_donor_folds(frame: pd.DataFrame) -> dict[str, Any]:
    """Require every cancer x primary-cell context in every one of five folds."""

    required = {
        "dataset_id", "cancer_id", "donor_id", "cell_key", "block_id",
        "single_cell_fold_id",
    }
    if missing := sorted(required - set(frame.columns)):
        raise SingleCellContextV2Error(f"Donor fold table lacks {missing}")
    donor = frame.donor_id.fillna("").astype(str).str.strip()
    block = frame.block_id.fillna("").astype(str)
    folds = pd.to_numeric(frame.single_cell_fold_id, errors="raise").astype(int)
    donor_cross = int(
        pd.DataFrame({"block_id": block, "fold": folds})
        .groupby("block_id", observed=True).fold.nunique().gt(1).sum()
    )
    dataset_fallback_rows = int(block.str.contains("|dataset:", regex=False).sum())
    invalid_fold_rows = int((~folds.isin(range(N_FOLDS))).sum())
    local = frame.assign(
        donor_id=donor,
        cell_key=frame.cell_key.astype(str).map(_normalise_token),
        single_cell_fold_id=folds,
    )
    context = local.groupby(
        ["cancer_id", "cell_key"], observed=True, as_index=False
    ).agg(
        distinct_donors=("block_id", "nunique"),
        distinct_folds=("single_cell_fold_id", "nunique"),
    )
    context["all_five_folds"] = context.distinct_folds.eq(N_FOLDS)
    context["at_least_five_donors"] = context.distinct_donors.ge(N_FOLDS)
    failures = context.loc[
        ~(context.all_five_folds & context.at_least_five_donors)
    ]
    ready = bool(
        len(frame) > 0
        and donor.ne("").all()
        and donor_cross == 0
        and dataset_fallback_rows == 0
        and invalid_fold_rows == 0
        and len(context) > 0
        and failures.empty
    )
    return {
        "policy": "EVERY_CANCER_X_CELLTYPE_CONTEXT_PRESENT_IN_ALL_FIVE_DONOR_FOLDS",
        "rows": int(len(frame)),
        "contexts": int(len(context)),
        "contexts_passing": int(len(context) - len(failures)),
        "contexts_failing": int(len(failures)),
        "minimum_donors_per_context": (
            int(context.distinct_donors.min()) if len(context) else 0
        ),
        "minimum_folds_per_context": (
            int(context.distinct_folds.min()) if len(context) else 0
        ),
        "donor_blocks_crossing_folds": donor_cross,
        "dataset_fallback_rows": dataset_fallback_rows,
        "invalid_fold_rows": invalid_fold_rows,
        "failure_examples": failures.head(50).to_dict(orient="records"),
        "ready": ready,
    }


def build_strict_donor_folds(
    associations: pd.DataFrame,
    *,
    source_semantic_audit: Mapping[str, Any],
    seed: int = 20260825,
) -> pd.DataFrame:
    gate = require_donor_blocked_targets(
        associations, source_semantic_audit=source_semantic_audit
    )
    if not gate["ready"]:
        raise SingleCellContextV2Error(
            "Formal context-V2 training requires proven donor-resolved targets; "
            f"audit={gate}"
        )
    result = associations.reset_index(drop=True).copy()
    result["cell_key"] = _association_cell_key(result)
    result["block_id"] = _donor_block_id(result)
    donor_context = result[
        ["block_id", "cancer_id", "cell_key"]
    ].drop_duplicates()
    context_count = donor_context.groupby("block_id", observed=True).size().to_dict()
    blocks = sorted(
        donor_context.block_id.unique(),
        key=lambda value: (
            -int(context_count[str(value)]),
            hashlib.sha256(f"{int(seed)}|{value}".encode("utf-8")).hexdigest(),
            str(value),
        ),
    )
    context_fold_counts: dict[tuple[str, str, int], int] = {}
    fold_sizes = [0] * N_FOLDS
    mapping: dict[str, int] = {}
    by_block = {
        str(block): list(
            local[["cancer_id", "cell_key"]].itertuples(index=False, name=None)
        )
        for block, local in donor_context.groupby("block_id", observed=True)
    }
    for block in blocks:
        contexts = by_block[str(block)]
        scored: list[tuple[int, int, str, int]] = []
        for fold in range(N_FOLDS):
            new_coverage = sum(
                context_fold_counts.get((str(cancer), str(cell), fold), 0) == 0
                for cancer, cell in contexts
            )
            tie = hashlib.sha256(
                f"{int(seed)}|{block}|fold:{fold}".encode("utf-8")
            ).hexdigest()
            scored.append((-new_coverage, fold_sizes[fold], tie, fold))
        selected = min(scored)[3]
        mapping[str(block)] = selected
        fold_sizes[selected] += 1
        for cancer, cell in contexts:
            key = (str(cancer), str(cell), selected)
            context_fold_counts[key] = context_fold_counts.get(key, 0) + 1
    result["single_cell_fold_id"] = result.block_id.map(mapping).astype(int)
    coverage = validate_context_donor_folds(result)
    if not coverage["ready"]:
        raise SingleCellContextV2Error(
            "Every cancer x celltype context must cover all five held-out donor folds; "
            f"audit={coverage}"
        )
    return result


def _formal_activity(
    activity: pd.DataFrame, dataset_manifest: pd.DataFrame
) -> pd.DataFrame:
    local = activity.merge(
        dataset_manifest[["dataset_id", "cancer_id", "qualified", "source_tier"]].rename(
            columns={"source_tier": "dataset_source_tier"}
        ),
        on=["dataset_id", "cancer_id"], how="left", validate="many_to_one",
    )
    local["effective_source_tier"] = local.activity_source_tier.where(
        local.activity_source_tier.ne("undeclared"), local.dataset_source_tier
    )
    return local.loc[
        local.qualified.fillna(False)
        & local.effective_source_tier.isin(_FORMAL_SOURCE_TIERS)
    ].drop(columns=["qualified", "dataset_source_tier"])


def _prepare_inputs(
    *,
    candidates_path: str | Path,
    dataset_manifest_path: str | Path,
    association_path: str | Path,
    lnc_celltype_path: str | Path,
    activity_paths: Sequence[tuple[str, str | Path]],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    dict[str, Any],
]:
    candidates = normalise_candidates(_read_table(candidates_path))
    manifest = normalise_dataset_manifest(_read_table(dataset_manifest_path))
    association_raw = _read_table(association_path)
    association_semantic_audit = audit_donor_resolved_association_source(
        association_raw
    )
    associations_all = normalise_single_cell_associations(
        association_raw, manifest
    )
    associations = exact_candidate_join(
        associations_all.loc[associations_all.formal_row].copy(), candidates
    )
    lnc_raw = _read_table(lnc_celltype_path)
    lnc_semantic_audit = audit_donor_resolved_lnc_source(lnc_raw)
    lnc = _normalise_lnc_celltype(lnc_raw).merge(
        manifest[["dataset_id", "cancer_id", "qualified"]],
        on=["dataset_id", "cancer_id"], how="left", validate="many_to_one",
    )
    lnc = lnc.loc[lnc.qualified.fillna(False)].drop(columns="qualified")
    parts = [
        _normalise_activity(_read_table(path), kind=kind, dataset_manifest=manifest)
        for kind, path in activity_paths
    ]
    activity = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=[
            "dataset_id", "cancer_id", "donor_id", "cell_type", "cell_key",
            "pathway_id", "activity_kind", "activity_source_tier",
            "activity_value", "mean_pseudotime", "n_cells",
        ]
    )
    activity = _formal_activity(activity, manifest) if len(activity) else activity
    return (
        candidates,
        manifest,
        associations,
        lnc,
        activity,
        association_semantic_audit,
        lnc_semantic_audit,
    )


def _context_sets(
    associations: pd.DataFrame, activity: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assoc = associations.copy()
    assoc["cell_key"] = _association_cell_key(assoc)
    assoc_context = assoc.loc[:, CONTEXT_ID_KEYS].drop_duplicates().sort_values(
        list(CONTEXT_ID_KEYS), kind="stable"
    ).reset_index(drop=True)
    activity_context = activity.loc[
        activity.cell_key.astype(str).ne(""), CONTEXT_ID_KEYS
    ].drop_duplicates().sort_values(list(CONTEXT_ID_KEYS), kind="stable").reset_index(drop=True)
    coverage = assoc_context.merge(
        activity_context.assign(activity_context_available=True),
        on=list(CONTEXT_ID_KEYS), how="left", validate="one_to_one",
    )
    coverage["activity_context_available"] = coverage.activity_context_available.fillna(
        False
    ).astype(bool)
    return assoc_context, activity_context, coverage


def _assert_disjoint_output(output: Path, protected: Sequence[Path]) -> None:
    resolved = output.resolve()
    for item in protected:
        source = item.resolve()
        if resolved == source or resolved == source.parent or source.is_relative_to(resolved):
            raise SingleCellContextV2Error(
                f"V2 output overlaps a protected V1/input artifact: {resolved} vs {source}"
            )


def _execution_code_binding() -> tuple[dict[str, dict[str, str]], str]:
    module_path = Path(__file__).resolve()
    script_root = module_path.parents[2] / "scripts"
    paths = {
        "single_cell_context_v2.py": module_path,
        "preflight_v32_single_cell_context_v2.py": (
            script_root / "preflight_v32_single_cell_context_v2.py"
        ),
        "run_v32_single_cell_context_v2.py": (
            script_root / "run_v32_single_cell_context_v2.py"
        ),
    }
    binding = {
        name: {"path": str(path), "sha256": artifact_sha256(path)}
        for name, path in paths.items() if path.is_file()
    }
    if set(binding) != set(paths):
        raise SingleCellContextV2Error("Context-V2 execution code set is incomplete")
    return binding, _canonical_sha256(binding)


def _input_artifact_bindings(
    input_audit: Mapping[str, Any],
    *,
    core_manifest: Path,
    formal_v1_binding: Path,
) -> dict[str, dict[str, str]]:
    bindings = {
        str(row["artifact_id"]): {
            "path": str(Path(row["path"]).resolve()),
            "sha256": str(row["sha256"]),
        }
        for row in input_audit.get("artifacts", [])
    }
    bindings["fresh_v32_core_manifest"] = {
        "path": str(core_manifest.resolve()),
        "sha256": artifact_sha256(core_manifest),
    }
    bindings["protected_formal_v1_binding"] = {
        "path": str(formal_v1_binding.resolve()),
        "sha256": artifact_sha256(formal_v1_binding),
    }
    return bindings


def _write_donor_fold_manifest(
    *,
    output: Path,
    associations: pd.DataFrame,
    association_semantic_audit: Mapping[str, Any],
    association_input_sha256: str,
    seed: int,
) -> tuple[dict[str, Any], Path, str]:
    donor_gate = require_donor_blocked_targets(
        associations, source_semantic_audit=association_semantic_audit
    )
    assignments_path: Path | None = None
    assignment_sha: str | None = None
    coverage: dict[str, Any] = {
        "policy": "EVERY_CANCER_X_CELLTYPE_CONTEXT_PRESENT_IN_ALL_FIVE_DONOR_FOLDS",
        "ready": False,
        "reason": "DONOR_TARGET_GATE_NOT_READY",
    }
    error: str | None = None
    if donor_gate["ready"]:
        try:
            folded = build_strict_donor_folds(
                associations,
                source_semantic_audit=association_semantic_audit,
                seed=seed,
            )
            coverage = validate_context_donor_folds(folded)
            assignments = folded[
                [
                    "dataset_id", "cancer_id", "donor_id", "cell_key",
                    "block_id", "single_cell_fold_id",
                ]
            ].drop_duplicates().sort_values(
                ["cancer_id", "cell_key", "block_id"], kind="stable"
            )
            assignments_path = output / "DONOR_FOLD_ASSIGNMENTS.parquet"
            _atomic_parquet(assignments, assignments_path)
            assignment_sha = artifact_sha256(assignments_path)
        except SingleCellContextV2Error as exc:
            error = str(exc)
    ready = bool(
        donor_gate["ready"]
        and coverage.get("ready") is True
        and assignments_path is not None
        and assignment_sha is not None
        and error is None
    )
    payload = {
        "format": DONOR_FOLD_MANIFEST_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "module_id": MODULE_ID,
        "status": "READY" if ready else "BLOCKED",
        "ready": ready,
        "seed": int(seed),
        "folds": N_FOLDS,
        "association_input_sha256": association_input_sha256,
        "association_semantic_audit": dict(association_semantic_audit),
        "donor_gate": donor_gate,
        "context_fold_coverage": coverage,
        "dataset_fallback_used": False,
        "assignment_path": str(assignments_path) if assignments_path else None,
        "assignment_sha256": assignment_sha,
        "error": error,
    }
    path = output / "DONOR_FOLD_MANIFEST.json"
    _atomic_json(path, payload)
    return payload, path, artifact_sha256(path)


def validate_donor_fold_manifest(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_association_sha256: str,
) -> dict[str, Any]:
    source = Path(path).resolve()
    observed = artifact_sha256(source)
    if observed != str(expected_sha256):
        raise SingleCellContextV2Error("Donor fold manifest SHA256 drift")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("format") != DONOR_FOLD_MANIFEST_FORMAT:
        raise SingleCellContextV2Error("Donor fold manifest format mismatch")
    if payload.get("ready") is not True or payload.get("status") != "READY":
        raise SingleCellContextV2Error("Donor fold manifest is not READY")
    if payload.get("dataset_fallback_used") is not False:
        raise SingleCellContextV2Error("Dataset fallback is forbidden")
    if str(payload.get("association_input_sha256")) != str(
        expected_association_sha256
    ):
        raise SingleCellContextV2Error("Fold manifest association binding drift")
    assignment_path = Path(str(payload.get("assignment_path", ""))).resolve()
    assignment_sha = str(payload.get("assignment_sha256", ""))
    if artifact_sha256(assignment_path) != assignment_sha:
        raise SingleCellContextV2Error("Donor fold assignment SHA256 drift")
    assignments = pd.read_parquet(assignment_path)
    audit = validate_context_donor_folds(assignments)
    if not audit["ready"]:
        raise SingleCellContextV2Error(
            f"Donor fold assignment violates held-out donor isolation: {audit}"
        )
    return payload


def _validate_fold_local_feature_manifest(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_fold_manifest_sha256: str,
    expected_input_binding_sha256: str,
) -> dict[str, Any]:
    source = Path(path).resolve()
    if artifact_sha256(source) != str(expected_sha256):
        raise SingleCellContextV2Error("Fold-local feature manifest SHA256 drift")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("format") != FOLD_LOCAL_FEATURE_MANIFEST_FORMAT:
        raise SingleCellContextV2Error("Fold-local feature manifest format mismatch")
    if payload.get("ready") is not True or payload.get("status") != "READY":
        raise SingleCellContextV2Error("Fold-local features are not READY")
    if payload.get("training_donors_only_materialization") is not True:
        raise SingleCellContextV2Error("Fold-local features used non-training donors")
    if int(payload.get("held_out_donor_rows_used", -1)) != 0:
        raise SingleCellContextV2Error("Held-out donor rows entered feature materialization")
    if str(payload.get("donor_fold_manifest_sha256")) != str(
        expected_fold_manifest_sha256
    ):
        raise SingleCellContextV2Error("Fold-local feature/fold binding drift")
    if str(payload.get("input_binding_sha256")) != str(
        expected_input_binding_sha256
    ):
        raise SingleCellContextV2Error("Fold-local feature/input binding drift")
    records = payload.get("folds")
    if not isinstance(records, Mapping) or set(map(str, records)) != set(
        map(str, range(N_FOLDS))
    ):
        raise SingleCellContextV2Error("Fold-local feature manifest requires folds 0..4")
    for fold in range(N_FOLDS):
        record = records[str(fold)]
        if record.get("training_donors_only") is not True:
            raise SingleCellContextV2Error(
                f"Fold {fold} lacks training-donor-only attestation"
            )
        if int(record.get("held_out_donor_rows_used", -1)) != 0:
            raise SingleCellContextV2Error(
                f"Fold {fold} used held-out donor rows"
            )
        feature_path = Path(str(record.get("path", ""))).resolve()
        if artifact_sha256(feature_path) != str(record.get("sha256", "")):
            raise SingleCellContextV2Error(
                f"Fold {fold} feature artifact SHA256 drift"
            )
    return payload


def run_context_v2_preflight(
    *,
    candidates_path: str | Path,
    dataset_manifest_path: str | Path,
    association_path: str | Path,
    lnc_celltype_path: str | Path,
    activity_path: str | Path,
    core_embedding_manifest_path: str | Path,
    formal_v1_binding_path: str | Path,
    output_root: str | Path,
    pseudotime_path: str | Path | None = None,
    ucell_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run a bounded read-only-input preflight and emit a retraining plan.

    Context counts are recomputed from the supplied rows.  No expected count is
    embedded in code.  The current formal inputs happen to yield 129
    association contexts, 137 activity contexts, 129 matches, and zero missing.
    """

    paths = {
        "candidates": Path(candidates_path).resolve(),
        "dataset_manifest": Path(dataset_manifest_path).resolve(),
        "association": Path(association_path).resolve(),
        "lnc_celltype": Path(lnc_celltype_path).resolve(),
        "activity": Path(activity_path).resolve(),
        "core_manifest": Path(core_embedding_manifest_path).resolve(),
        "formal_v1_binding": Path(formal_v1_binding_path).resolve(),
    }
    optional = {
        "pseudotime": Path(pseudotime_path).resolve() if pseudotime_path else None,
        "ucell": Path(ucell_path).resolve() if ucell_path else None,
    }
    output = Path(output_root).resolve()
    if output.exists():
        raise SingleCellContextV2Error(f"Preflight refuses output reuse: {output}")
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(path)
        if name not in {"core_manifest", "formal_v1_binding"}:
            assert_not_historical_source(path)
            _assert_source_path(path)
    for path in optional.values():
        if path is not None:
            assert_not_historical_source(path)
            _assert_source_path(path)
    _assert_disjoint_output(output, list(paths.values()))
    v1_hash_before = artifact_sha256(paths["formal_v1_binding"])

    activities: list[tuple[str, Path, str]] = [
        (
            "activity", paths["activity"],
            "V3.2_FRESH_FROM_RAW_H5_PROTEIN_ONLY_EXACT_PATHWAY",
        )
    ]
    if optional["pseudotime"] is not None:
        activities.append(
            (
                "pseudotime", optional["pseudotime"],
                "V3.2_FRESH_NONPREDICTIVE_PSEUDOTIME",
            )
        )
    if optional["ucell"] is not None:
        activities.append(
            ("ucell", optional["ucell"], "V3.2_FRESH_NONPREDICTIVE_UCELL")
        )
    input_audit = _source_audit(
        candidates=paths["candidates"],
        dataset_manifest=paths["dataset_manifest"],
        association=paths["association"],
        association_generation="V3.2_FRESH_FROM_RAW_H5_DONOR_PSEUDOBULK",
        lnc_celltype=paths["lnc_celltype"],
        lnc_celltype_generation="V3.2_FRESH_FROM_RAW_H5_EXPRESSION_FACTS",
        activities=activities,
    )
    _core, core_sha, core_parameter_sha = validate_core_manifest(paths["core_manifest"])
    (
        candidates,
        manifest,
        associations,
        lnc,
        activity,
        association_semantic_audit,
        lnc_semantic_audit,
    ) = _prepare_inputs(
        candidates_path=paths["candidates"],
        dataset_manifest_path=paths["dataset_manifest"],
        association_path=paths["association"],
        lnc_celltype_path=paths["lnc_celltype"],
        activity_paths=[(kind, path) for kind, path, _generation in activities],
    )
    association_context, activity_context, coverage = _context_sets(
        associations, activity
    )
    missing = coverage.loc[~coverage.activity_context_available].copy()
    counts = {
        "association_distinct_contexts": int(len(association_context)),
        "activity_available_distinct_contexts": int(len(activity_context)),
        "matched_association_contexts": int(coverage.activity_context_available.sum()),
        "missing_association_contexts": int(len(missing)),
    }
    context_gate = bool(
        counts["association_distinct_contexts"] > 0
        and counts["matched_association_contexts"]
        == counts["association_distinct_contexts"]
        and counts["missing_association_contexts"] == 0
    )
    donor_gate = require_donor_blocked_targets(
        associations, source_semantic_audit=association_semantic_audit
    )
    feature_schema_gate = True
    assert_leakage_free_feature_schema(CONTEXT_FEATURES)
    feature_probe_rows = min(512, len(associations))
    probe_associations = associations.iloc[:feature_probe_rows].copy()
    probe_associations["cell_key"] = _association_cell_key(probe_associations)
    probe_context = probe_associations[
        ["dataset_id", "cancer_id", "cell_key", "pathway_id"]
    ].drop_duplicates()
    probe_global = probe_associations[
        ["dataset_id", "cancer_id", "pathway_id"]
    ].drop_duplicates()
    probe_activity = activity.merge(
        probe_global, on=["dataset_id", "cancer_id", "pathway_id"],
        how="inner", validate="many_to_one",
    )
    probe_lnc_keys = probe_associations[
        ["dataset_id", "cancer_id", "lncrna_id", "cell_key"]
    ].drop_duplicates()
    probe_lnc = lnc.merge(
        probe_lnc_keys,
        on=["dataset_id", "cancer_id", "lncrna_id", "cell_key"],
        how="inner", validate="many_to_one",
    )
    # The context subset is retained for a direct availability sanity check;
    # global rows for the same exact pathway stay available to compute delta.
    observed_probe_context = probe_activity[
        ["dataset_id", "cancer_id", "cell_key", "pathway_id"]
    ].drop_duplicates().merge(
        probe_context, on=["dataset_id", "cancer_id", "cell_key", "pathway_id"],
        how="inner",
    )
    feature_probe = build_context_domain_feature_frame(
        probe_associations, probe_lnc, probe_activity
    )
    if feature_probe_rows and context_feature_matrix(feature_probe).shape != (
        feature_probe_rows, len(CONTEXT_FEATURES)
    ):
        feature_schema_gate = False
    sidecar_probe = build_signed_evidence_sidecar(
        associations.iloc[:feature_probe_rows]
    )
    if any(column in CONTEXT_FEATURES for column in sidecar_probe.columns):
        raise SingleCellContextV2Error("Evidence sidecar overlaps the feature schema")

    output.mkdir(parents=True)
    execution_code, execution_code_sha = _execution_code_binding()
    input_bindings = _input_artifact_bindings(
        input_audit,
        core_manifest=paths["core_manifest"],
        formal_v1_binding=paths["formal_v1_binding"],
    )
    input_binding_sha = _canonical_sha256(input_bindings)
    association_binding = input_bindings.get("single_cell_association_target")
    if association_binding is None:
        raise SingleCellContextV2Error("Association input binding is missing")
    fold_manifest, fold_manifest_path, fold_manifest_sha = _write_donor_fold_manifest(
        output=output,
        associations=associations,
        association_semantic_audit=association_semantic_audit,
        association_input_sha256=association_binding["sha256"],
        seed=20260825,
    )
    fold_local_feature_gate = {
        "policy": (
            "PER_FOLD_FEATURES_MATERIALIZED_FROM_TRAINING_DONORS_ONLY_"
            "NO_HELD_OUT_DONOR_ROWS"
        ),
        "fresh_donor_resolved_lnc_source": lnc_semantic_audit["ready"],
        "donor_fold_manifest_ready": fold_manifest["ready"],
        "materialization_manifest_provided": False,
        "executor_implemented": False,
        "held_out_donor_rows_used": None,
        "ready": False,
        "reasons": [
            reason
            for condition, reason in (
                (
                    not lnc_semantic_audit["ready"],
                    "FRESH_DONOR_RESOLVED_LNCRNA_SOURCE_MISSING",
                ),
                (
                    not fold_manifest["ready"],
                    "STRICT_DONOR_FOLD_MANIFEST_NOT_READY",
                ),
                (True, "FOLD_LOCAL_FEATURE_MATERIALIZATION_NOT_PROVIDED"),
                (True, "FOLD_LOCAL_FEATURE_EXECUTOR_NOT_ENABLED"),
            )
            if condition
        ],
    }
    full_training_ready = bool(
        context_gate
        and donor_gate["ready"]
        and association_semantic_audit["ready"]
        and lnc_semantic_audit["ready"]
        and fold_manifest["ready"]
        and fold_local_feature_gate["ready"]
        and feature_schema_gate
    )
    if len(missing):
        _atomic_parquet(missing, output / "MISSING_CONTEXTS.parquet")
    run_plan = {
        "format": RUN_PLAN_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "module_id": MODULE_ID,
        "created_at_utc": _utc_now(),
        "training_requested": False,
        "long_training_started": False,
        "root_approval_required": True,
        "full_training_ready": full_training_ready,
        "root_approval_token_name": ROOT_APPROVAL_TOKEN,
        "exact_primary": {
            "modified": False,
            "write_paths": [],
            "score_used_as_feature": False,
            "ranking_used_as_output": False,
        },
        "feature_key": list(CONTEXT_KEYS),
        "context_normalization": "single_cell_training._normalise_token",
        "feature_schema": list(CONTEXT_FEATURES),
        "activity_policy": (
            "CELL_CONTEXT_WEIGHTED_MEAN_PLUS_DATASET_GLOBAL_AND_CONTEXT_MINUS_GLOBAL"
        ),
        "compartment_policy": {
            "malignant": ["Malignant", "Malignant_candidate"],
            "immune": "EXPLICIT_IMMUNE_TYPES",
            "stromal": "EXPLICIT_STROMAL_TYPES",
            "other": "FAIL_SAFE_REMAINDER",
        },
        "target_policy": "ABS_RHO_X_ONE_MINUS_FDR_RELEVANCE_LABEL_ONLY",
        "signed_evidence_policy": (
            "OBSERVED_SIGNED_RHO_AND_FDR_IN_SEPARATE_SIDECAR_NEVER_A_FEATURE"
        ),
        "split_policy": donor_gate["policy"],
        "association_target_semantics": association_semantic_audit,
        "lncrna_feature_source_semantics": lnc_semantic_audit,
        "donor_fold_manifest_path": str(fold_manifest_path),
        "donor_fold_manifest_sha256": fold_manifest_sha,
        "fold_local_feature_gate": fold_local_feature_gate,
        "core_policy": "FROZEN_FRESH_V32_CORE_PRIVATE_HEAD_FROM_SCRATCH",
        "old_outputs_policy": "FORBIDDEN_NO_CHECKPOINT_OR_PREDICTION_LOADER",
        "historical_inventory": list(FORBIDDEN_HISTORICAL_INVENTORY),
        "historical_inventory_read": False,
        "formal_v1_binding_policy": "READ_ONLY_HASH_BEFORE_AND_AFTER_NO_MUTATION",
        "execution_code_sha256": execution_code_sha,
        "input_binding_sha256": input_binding_sha,
        "formal_train_gate": {
            "context_coverage": context_gate,
            "fresh_donor_resolved_targets": association_semantic_audit["ready"],
            "fresh_donor_resolved_lncrna_features": lnc_semantic_audit["ready"],
            "donor_block_ids": donor_gate["ready"],
            "every_context_all_five_folds": fold_manifest["ready"],
            "fold_local_feature_materialization": fold_local_feature_gate["ready"],
            "fresh_v32_core": True,
            "leakage_free_features": feature_schema_gate,
            "root_approved": False,
            "ready": full_training_ready,
        },
    }
    _atomic_json(output / "RETRAINING_RUN_PLAN.json", run_plan)
    _atomic_json(output / "INPUT_LINEAGE_AUDIT.json", input_audit)
    v1_hash_after = artifact_sha256(paths["formal_v1_binding"])
    if v1_hash_after != v1_hash_before:
        raise SingleCellContextV2Error("Formal V1 binding changed during V2 preflight")
    lineage_contract = {
        "format": LINEAGE_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "module_id": MODULE_ID,
        "created_at_utc": _utc_now(),
        "status": "PREFLIGHT_ONLY_NOT_TRAINED",
        "release_ready": False,
        "production_deployed": False,
        "exact_primary_modified": False,
        "formal_v1_binding": {
            "path": str(paths["formal_v1_binding"]),
            "sha256_before": v1_hash_before,
            "sha256_after": v1_hash_after,
            "unchanged": True,
            "use_role": "read_only_protected_reference",
        },
        "fresh_v32_core": {
            "manifest_path": str(paths["core_manifest"]),
            "manifest_sha256": core_sha,
            "parameter_composite_sha256": core_parameter_sha,
            "historical_checkpoint_loaded": False,
            "historical_prediction_loaded": False,
        },
        "input_lineage_sha256": input_audit["lineage_sha256"],
        "input_artifact_bindings": input_bindings,
        "input_binding_sha256": input_binding_sha,
        "execution_code": execution_code,
        "execution_code_sha256": execution_code_sha,
        "input_paths": {key: str(value) for key, value in paths.items()},
        "optional_input_paths": {
            key: str(value) if value is not None else None
            for key, value in optional.items()
        },
        "context_counts": counts,
        "context_normalization": "single_cell_training._normalise_token",
        "formal_gate_policy": (
            "normalise_dataset_manifest.qualified_AND_ROW_SOURCE_TIER_IN_FORMAL_AUTHORITY"
        ),
        "context_gate_pass": context_gate,
        "association_target_semantic_gate": association_semantic_audit,
        "lncrna_feature_semantic_gate": lnc_semantic_audit,
        "donor_split_gate": donor_gate,
        "donor_fold_manifest": {
            "path": str(fold_manifest_path),
            "sha256": fold_manifest_sha,
            "ready": fold_manifest["ready"],
            "context_fold_coverage": fold_manifest["context_fold_coverage"],
        },
        "fold_local_feature_gate": fold_local_feature_gate,
        "candidate_rows": int(len(candidates)),
        "formal_association_rows": int(len(associations)),
        "formal_activity_rows": int(len(activity)),
        "formal_lnc_celltype_rows": int(len(lnc)),
        "feature_probe_rows": feature_probe_rows,
        "feature_probe_contexts_with_activity": int(len(observed_probe_context)),
        "feature_schema": list(CONTEXT_FEATURES),
        "feature_schema_sha256": _canonical_sha256(CONTEXT_FEATURES),
        "signed_evidence_columns": list(sidecar_probe.columns),
        "signed_evidence_is_feature": False,
        "abs_only_claimed_as_direction": False,
        "historical_inventory": list(FORBIDDEN_HISTORICAL_INVENTORY),
        "historical_inventory_read": False,
        "long_training_started": False,
        "root_approval_required": True,
        "full_training_ready": full_training_ready,
    }
    _atomic_json(output / "LINEAGE_CONTRACT.json", lineage_contract)
    preflight = {
        "format": PREFLIGHT_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "module_id": MODULE_ID,
        "status": (
            "ROOT_REVIEW_REQUIRED"
            if context_gate
            else "FAIL_CLOSED_CONTEXT_COVERAGE"
        ),
        "context_counts": counts,
        "context_gate_pass": context_gate,
        "association_target_semantic_gate": association_semantic_audit,
        "lncrna_feature_semantic_gate": lnc_semantic_audit,
        "donor_split_gate": donor_gate,
        "donor_fold_manifest_path": str(fold_manifest_path),
        "donor_fold_manifest_sha256": fold_manifest_sha,
        "donor_fold_manifest_ready": fold_manifest["ready"],
        "fold_local_feature_manifest_path": None,
        "fold_local_feature_manifest_sha256": None,
        "fold_local_feature_gate": fold_local_feature_gate,
        "input_binding_sha256": input_binding_sha,
        "execution_code_sha256": execution_code_sha,
        "full_training_ready": full_training_ready,
        "long_training_started": False,
        "root_approval_required": True,
        "lineage_contract_path": str(output / "LINEAGE_CONTRACT.json"),
        "lineage_contract_sha256": artifact_sha256(output / "LINEAGE_CONTRACT.json"),
        "run_plan_path": str(output / "RETRAINING_RUN_PLAN.json"),
        "run_plan_sha256": artifact_sha256(output / "RETRAINING_RUN_PLAN.json"),
        "input_lineage_audit_path": str(output / "INPUT_LINEAGE_AUDIT.json"),
        "input_lineage_audit_sha256": artifact_sha256(output / "INPUT_LINEAGE_AUDIT.json"),
        "formal_v1_binding_unchanged": True,
        "historical_tables_read": False,
    }
    _atomic_json(output / "PREFLIGHT.json", preflight)
    return preflight


def validate_training_preflight_contract(
    *,
    preflight_path: str | Path,
    expected_preflight_sha256: str,
    expected_execution_code_sha256: str,
    donor_fold_manifest_path: str | Path,
    expected_donor_fold_manifest_sha256: str,
    fold_local_feature_manifest_path: str | Path,
    expected_fold_local_feature_manifest_sha256: str,
    supplied_input_paths: Mapping[str, str | Path],
) -> dict[str, Any]:
    """Verify every immutable training dependency before any heavy work."""

    source = Path(preflight_path).resolve()
    if artifact_sha256(source) != str(expected_preflight_sha256):
        raise SingleCellContextV2Error("Preflight SHA256 drift")
    preflight = json.loads(source.read_text(encoding="utf-8"))
    if preflight.get("format") != PREFLIGHT_FORMAT:
        raise SingleCellContextV2Error("Training requires a context-V2 preflight")
    if preflight.get("full_training_ready") is not True:
        raise SingleCellContextV2Error("Preflight full_training_ready is not true")
    lineage_path = Path(str(preflight.get("lineage_contract_path", ""))).resolve()
    if artifact_sha256(lineage_path) != str(
        preflight.get("lineage_contract_sha256", "")
    ):
        raise SingleCellContextV2Error("Lineage contract SHA256 drift")
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    if lineage.get("full_training_ready") is not True:
        raise SingleCellContextV2Error("Lineage full_training_ready is not true")
    execution_code, observed_code_sha = _execution_code_binding()
    if (
        observed_code_sha != str(expected_execution_code_sha256)
        or observed_code_sha != str(preflight.get("execution_code_sha256", ""))
        or observed_code_sha != str(lineage.get("execution_code_sha256", ""))
        or execution_code != lineage.get("execution_code")
    ):
        raise SingleCellContextV2Error("Execution code SHA256 drift")
    bindings = lineage.get("input_artifact_bindings")
    if not isinstance(bindings, Mapping) or not bindings:
        raise SingleCellContextV2Error("Input artifact bindings are missing")
    if _canonical_sha256(bindings) != str(lineage.get("input_binding_sha256", "")):
        raise SingleCellContextV2Error("Input binding composite SHA256 drift")
    if str(lineage.get("input_binding_sha256")) != str(
        preflight.get("input_binding_sha256")
    ):
        raise SingleCellContextV2Error("Preflight/input binding mismatch")
    if set(supplied_input_paths) != set(bindings):
        raise SingleCellContextV2Error("Supplied training input set differs from preflight")
    for artifact_id, declaration in bindings.items():
        supplied = Path(supplied_input_paths[artifact_id]).resolve()
        declared = Path(str(declaration.get("path", ""))).resolve()
        if supplied != declared:
            raise SingleCellContextV2Error(
                f"Training input path substitution is forbidden: {artifact_id}"
            )
        if artifact_sha256(supplied) != str(declaration.get("sha256", "")):
            raise SingleCellContextV2Error(
                f"Training input SHA256 drift: {artifact_id}"
            )
    declared_fold_path = Path(
        str(preflight.get("donor_fold_manifest_path", ""))
    ).resolve()
    if Path(donor_fold_manifest_path).resolve() != declared_fold_path:
        raise SingleCellContextV2Error("Donor fold manifest path substitution")
    if str(preflight.get("donor_fold_manifest_sha256")) != str(
        expected_donor_fold_manifest_sha256
    ):
        raise SingleCellContextV2Error("Expected donor fold manifest SHA mismatch")
    association_sha = str(
        bindings["single_cell_association_target"]["sha256"]
    )
    validate_donor_fold_manifest(
        declared_fold_path,
        expected_sha256=expected_donor_fold_manifest_sha256,
        expected_association_sha256=association_sha,
    )
    declared_feature_path = Path(
        str(preflight.get("fold_local_feature_manifest_path", ""))
    ).resolve()
    if Path(fold_local_feature_manifest_path).resolve() != declared_feature_path:
        raise SingleCellContextV2Error("Fold-local feature manifest path substitution")
    if str(preflight.get("fold_local_feature_manifest_sha256")) != str(
        expected_fold_local_feature_manifest_sha256
    ):
        raise SingleCellContextV2Error(
            "Expected fold-local feature manifest SHA mismatch"
        )
    _validate_fold_local_feature_manifest(
        declared_feature_path,
        expected_sha256=expected_fold_local_feature_manifest_sha256,
        expected_fold_manifest_sha256=expected_donor_fold_manifest_sha256,
        expected_input_binding_sha256=str(lineage["input_binding_sha256"]),
    )
    return {
        "status": "PASS",
        "preflight_sha256": expected_preflight_sha256,
        "execution_code_sha256": observed_code_sha,
        "input_binding_sha256": lineage["input_binding_sha256"],
        "donor_fold_manifest_sha256": expected_donor_fold_manifest_sha256,
        "fold_local_feature_manifest_sha256": (
            expected_fold_local_feature_manifest_sha256
        ),
    }


def run_context_v2_training(
    *,
    candidates_path: str | Path,
    dataset_manifest_path: str | Path,
    association_path: str | Path,
    lnc_celltype_path: str | Path,
    activity_path: str | Path,
    core_embedding_manifest_path: str | Path,
    formal_v1_binding_path: str | Path,
    preflight_path: str | Path,
    expected_preflight_sha256: str,
    expected_execution_code_sha256: str,
    donor_fold_manifest_path: str | Path,
    expected_donor_fold_manifest_sha256: str,
    fold_local_feature_manifest_path: str | Path,
    expected_fold_local_feature_manifest_sha256: str,
    output_root: str | Path,
    training_run_id: str,
    root_approval_token: str,
    pseudotime_path: str | Path | None = None,
    ucell_path: str | Path | None = None,
    config: SingleCellTrainingConfig | None = None,
) -> dict[str, Any]:
    """Train donor-blocked V2 private heads after explicit root approval.

    This entry point is intentionally unreachable with the current aggregate
    association table because that table has no donor IDs.  It becomes
    trainable when a fresh donor-resolved V3.2 association source passes the
    same preflight and the caller supplies the exact approval token.
    """

    if root_approval_token != ROOT_APPROVAL_TOKEN:
        raise SingleCellContextV2Error("Explicit root approval token is required")
    supplied_inputs: dict[str, Path] = {
        "v32_candidates": Path(candidates_path),
        "single_cell_dataset_manifest": Path(dataset_manifest_path),
        "single_cell_association_target": Path(association_path),
        "single_cell_lnc_celltype": Path(lnc_celltype_path),
        "single_cell_activity": Path(activity_path),
        "fresh_v32_core_manifest": Path(core_embedding_manifest_path),
        "protected_formal_v1_binding": Path(formal_v1_binding_path),
    }
    if pseudotime_path:
        supplied_inputs["single_cell_pseudotime"] = Path(pseudotime_path)
    if ucell_path:
        supplied_inputs["single_cell_ucell"] = Path(ucell_path)
    validate_training_preflight_contract(
        preflight_path=preflight_path,
        expected_preflight_sha256=expected_preflight_sha256,
        expected_execution_code_sha256=expected_execution_code_sha256,
        donor_fold_manifest_path=donor_fold_manifest_path,
        expected_donor_fold_manifest_sha256=expected_donor_fold_manifest_sha256,
        fold_local_feature_manifest_path=fold_local_feature_manifest_path,
        expected_fold_local_feature_manifest_sha256=(
            expected_fold_local_feature_manifest_sha256
        ),
        supplied_input_paths=supplied_inputs,
    )
    # The current code deliberately has no fold-local feature executor.  This
    # stops the legacy all-donor aggregation below from ever being reached.
    raise SingleCellContextV2Error(
        "FOLD_LOCAL_FEATURE_EXECUTOR_NOT_ENABLED: full training remains fail-closed"
    )

    settings = config or SingleCellTrainingConfig()  # pragma: no cover
    settings.validate()
    output = Path(output_root).resolve()
    core_manifest, core_manifest_sha, core_parameter_sha = validate_core_manifest(
        core_embedding_manifest_path
    )
    output.mkdir(parents=True)
    checkpoint_root = output / "checkpoints"
    checkpoint_root.mkdir()
    prediction_sum = np.zeros(len(folds), dtype=np.float64)
    prediction_count = np.zeros(len(folds), dtype=np.int16)
    checkpoint_records: list[dict[str, Any]] = []
    for fold in range(N_FOLDS):
        core = load_fold_core_embeddings(
            core_embedding_manifest_path, core_manifest, fold
        )
        available = candidate_core_availability(folds, core)
        split = split_block_ids(folds, fold)
        train_index = np.flatnonzero(
            folds.block_id.isin(split["train"]).to_numpy() & available
        )
        validation_index = np.flatnonzero(
            folds.block_id.isin(split["validation"]).to_numpy() & available
        )
        test_index = np.flatnonzero(
            folds.block_id.isin(split["test"]).to_numpy() & available
        )
        labels = folds.association_target.to_numpy(np.float32)
        train_index = train_index[
            _sample_indices(
                labels[train_index], settings.max_train_rows, settings.seed + fold
            )
        ]
        validation_index = validation_index[
            _sample_indices(
                labels[validation_index], settings.max_validation_rows,
                settings.seed + 10_000 + fold,
            )
        ]
        if len(train_index) < 2 or len(validation_index) < 1 or len(test_index) < 1:
            raise SingleCellContextV2Error(f"Donor fold {fold} is not trainable")
        train_core, train_available = candidate_core(folds.iloc[train_index], core)
        validation_core, validation_available = candidate_core(
            folds.iloc[validation_index], core
        )
        if not train_available.all() or not validation_available.all():
            raise SingleCellContextV2Error("Core availability changed during materialisation")
        head, metadata, mean, scale, history = fit_private_head(
            train_core, domain[train_index], labels[train_index],
            validation_core, domain[validation_index], labels[validation_index],
            fold=fold, config=settings,
        )
        import torch

        checkpoint = checkpoint_root / f"single_cell_context_v2_fold_{fold}.pt"
        torch.save(
            {
                "checkpoint_format": CHECKPOINT_FORMAT,
                "analysis_version": ANALYSIS_VERSION,
                "module_id": MODULE_ID,
                "fold": fold,
                "model_state": head.state_dict(),
                "domain_features": list(CONTEXT_FEATURES),
                "domain_mean": mean,
                "domain_scale": scale,
                "history": history,
                "initialization": metadata,
                "core_checkpoint_sha256": core.checkpoint_sha256,
                "core_parameter_sha256": core.parameter_sha256,
                "old_checkpoint_loaded": False,
                "old_predictions_used": False,
            },
            checkpoint,
        )
        checkpoint_records.append(
            {
                "fold": fold,
                "path": str(checkpoint),
                "sha256": artifact_sha256(checkpoint),
                "train_rows": int(len(train_index)),
                "validation_rows": int(len(validation_index)),
                "test_rows": int(len(test_index)),
                "fresh_random_initialization": True,
                "old_checkpoint_loaded": False,
            }
        )
        for start in range(0, len(test_index), settings.prediction_batch_size):
            selected = test_index[start : start + settings.prediction_batch_size]
            test_core, test_available = candidate_core(folds.iloc[selected], core)
            if not test_available.all():
                raise SingleCellContextV2Error("Test core coverage changed")
            prediction_sum[selected] += _predict(
                head, test_core, domain[selected], mean, scale,
                settings.prediction_batch_size,
            )
            prediction_count[selected] += 1
    if not np.all(prediction_count == 1):
        raise SingleCellContextV2Error("OOF donor predictions are not exactly once per row")
    predictions = folds[
        [
            "dataset_id", "cancer_id", "donor_id", "cell_type",
            "cell_type_major", "lncrna_id", "pathway_id", "block_id",
            "single_cell_fold_id",
        ]
    ].copy()
    predictions["cell_key"] = _association_cell_key(folds)
    predictions["single_cell_context_probability"] = (
        prediction_sum / prediction_count
    ).astype(np.float32)
    predictions["prediction_format"] = PREDICTION_FORMAT
    prediction_path = output / "single_cell_context_v2_oof_predictions.parquet"
    evidence_path = output / "signed_association_evidence_sidecar.parquet"
    _atomic_parquet(predictions, prediction_path)
    _atomic_parquet(evidence, evidence_path)
    checkpoint_manifest = {
        "format": CHECKPOINT_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "module_id": MODULE_ID,
        "records": checkpoint_records,
        "all_five_folds_trained": len(checkpoint_records) == N_FOLDS,
        "all_private_heads_fresh": True,
        "core_frozen": True,
    }
    _atomic_json(output / "CHECKPOINT_MANIFEST.json", checkpoint_manifest)
    lineage = {
        "format": LINEAGE_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "module_id": MODULE_ID,
        "training_run_id": training_run_id,
        "status": "SUCCESS",
        "release_ready": False,
        "production_deployed": False,
        "root_approval_token_verified": True,
        "exact_primary_modified": False,
        "split_policy": "STRICT_DONOR_BLOCKED_FIVE_FOLD_NO_DATASET_FALLBACK",
        "donor_split_audit": require_donor_blocked_targets(folds),
        "feature_key": list(CONTEXT_KEYS),
        "feature_schema": list(CONTEXT_FEATURES),
        "signed_evidence_is_feature": False,
        "signed_evidence_path": str(evidence_path),
        "signed_evidence_sha256": artifact_sha256(evidence_path),
        "prediction_path": str(prediction_path),
        "prediction_sha256": artifact_sha256(prediction_path),
        "core_manifest_sha256": core_manifest_sha,
        "core_parameter_composite_sha256": core_parameter_sha,
        "old_checkpoint_loaded": False,
        "old_predictions_used_as_features": False,
        "old_rankings_used_as_outputs": False,
        "historical_inventory_read": False,
        "historical_inventory": list(FORBIDDEN_HISTORICAL_INVENTORY),
        "training_config": asdict(settings),
    }
    _atomic_json(output / "LINEAGE.json", lineage)
    success = {
        "format": FORMAT,
        "status": "SUCCESS",
        "module_id": MODULE_ID,
        "training_run_id": training_run_id,
        "trained_folds": len(checkpoint_records),
        "prediction_rows": int(len(predictions)),
        "exact_primary_modified": False,
        "release_ready": False,
        "production_deployed": False,
        "lineage_sha256": artifact_sha256(output / "LINEAGE.json"),
        "checkpoint_manifest_sha256": artifact_sha256(
            output / "CHECKPOINT_MANIFEST.json"
        ),
    }
    _atomic_json(output / "SUCCESS.json", success)
    return success
