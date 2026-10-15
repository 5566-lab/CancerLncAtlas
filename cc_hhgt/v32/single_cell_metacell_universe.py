"""Fold-local candidate construction for donor-blocked metacell training.

The bulk arm is a frozen exact-pair universe used for re-scoring.  The
single-cell discovery arm is selected only from outer-training-donor LASSO
evidence.  Query-donor association targets are attached after candidate
selection and can therefore never create their own candidate rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


CONTEXT_KEYS = ("dataset_id", "cancer_id", "cell_key")
PAIR_KEYS = ("lncrna_id", "pathway_id")
UNIVERSE_KEYS = CONTEXT_KEYS + PAIR_KEYS
TARGET_KEYS = CONTEXT_KEYS + ("query_donor_id",) + PAIR_KEYS

SELECTION_POLICY = (
    "UNION_OF_FROZEN_BULK_EXACT_PAIRS_AND_OUTER_TRAIN_DONOR_LASSO_TOPK"
)
TARGET_POLICY = "ABS_WITHIN_DONOR_CELLTYPE_METACELL_SPEARMAN_RHO"

_FORBIDDEN_SELECTION_COLUMNS = {
    "association_target",
    "association_effect",
    "association_fdr",
    "effect",
    "fdr",
    "p_value",
    "padj",
    "q_value",
    "rho",
    "target",
}


class MetacellUniverseError(RuntimeError):
    """Raised when a discovery-universe leakage or identity gate fails."""


@dataclass(frozen=True)
class DiscoveryPolicy:
    """Pre-registered, label-free selection policy for one outer fold."""

    top_k_pathways_per_lncrna: int = 25
    min_training_donors: int = 4
    min_pathway_stable_support: float = 0.0
    min_inner_relative_mse_gain: float = 0.0
    require_null_calibration: bool = True
    allow_core_limited_discovery: bool = False

    def validate(self) -> None:
        if self.top_k_pathways_per_lncrna <= 0:
            raise MetacellUniverseError("top_k_pathways_per_lncrna must be positive")
        if self.min_training_donors < 4:
            raise MetacellUniverseError("min_training_donors must be at least four")
        if self.min_pathway_stable_support < 0:
            raise MetacellUniverseError("min_pathway_stable_support cannot be negative")


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], context: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise MetacellUniverseError(f"{context} lacks columns {missing}")


def _clean_id(value: Any) -> str:
    return str(value).strip()


def _normalise_ids(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[column] = result[column].map(_clean_id)
        if result[column].eq("").any():
            raise MetacellUniverseError(f"blank identifier in {column}")
    return result


def _strict_bool(series: pd.Series, context: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        if series.isna().any():
            raise MetacellUniverseError(f"missing boolean in {context}")
        return series.astype(bool)
    values = series.fillna("").astype(str).str.strip().str.lower()
    if not values.isin({"true", "false", "1", "0"}).all():
        raise MetacellUniverseError(f"invalid boolean in {context}")
    return values.isin({"true", "1"})


def _as_id_set(values: Iterable[Any], context: str) -> frozenset[str]:
    result = frozenset(_clean_id(value) for value in values)
    if "" in result:
        raise MetacellUniverseError(f"blank identifier in {context}")
    return result


def _validate_context(context: Mapping[str, Any]) -> dict[str, str]:
    missing = sorted(set(CONTEXT_KEYS) - set(context))
    if missing:
        raise MetacellUniverseError(f"context lacks keys {missing}")
    result = {key: _clean_id(context[key]) for key in CONTEXT_KEYS}
    if not all(result.values()):
        raise MetacellUniverseError("context identifiers must be non-empty")
    return result


def audit_node_coverage(
    *,
    measured_lncrna_ids: Iterable[Any],
    available_pathway_ids: Iterable[Any],
    core_lncrna_ids: Iterable[Any],
    core_pathway_ids: Iterable[Any],
) -> dict[str, Any]:
    """Report every measured node that a frozen-core head cannot score."""

    measured_lncrnas = _as_id_set(measured_lncrna_ids, "measured lncRNAs")
    available_pathways = _as_id_set(available_pathway_ids, "available pathways")
    core_lncrnas = _as_id_set(core_lncrna_ids, "core lncRNAs")
    core_pathways = _as_id_set(core_pathway_ids, "core pathways")
    missing_lncrnas = sorted(measured_lncrnas - core_lncrnas)
    missing_pathways = sorted(available_pathways - core_pathways)
    return {
        "policy": "EXPLICIT_CORE_NODE_INTERSECTION_NO_SILENT_OOV_DROP",
        "measured_lncrnas": len(measured_lncrnas),
        "core_covered_lncrnas": len(measured_lncrnas & core_lncrnas),
        "missing_core_lncrnas": missing_lncrnas,
        "available_pathways": len(available_pathways),
        "core_covered_pathways": len(available_pathways & core_pathways),
        "missing_core_pathways": missing_pathways,
        "complete": not missing_lncrnas and not missing_pathways,
    }


def _normalise_bulk_candidates(
    frame: pd.DataFrame,
    *,
    cancer_id: str,
    measured_lncrnas: frozenset[str],
    available_pathways: frozenset[str],
    core_lncrnas: frozenset[str],
    core_pathways: frozenset[str],
) -> pd.DataFrame:
    _require_columns(frame, ("cancer_id",) + PAIR_KEYS, "bulk candidates")
    columns = ("cancer_id",) + PAIR_KEYS
    local = _normalise_ids(frame.loc[:, list(columns)], columns)
    local = local.loc[
        local.cancer_id.eq(cancer_id)
        & local.lncrna_id.isin(measured_lncrnas & core_lncrnas)
        & local.pathway_id.isin(available_pathways & core_pathways)
    ]
    return local.loc[:, PAIR_KEYS].drop_duplicates().reset_index(drop=True)


def _normalise_lasso_evidence(
    frame: pd.DataFrame,
    *,
    context: Mapping[str, str],
    outer_fold_id: int,
    measured_lncrnas: frozenset[str],
    available_pathways: frozenset[str],
    core_lncrnas: frozenset[str],
    core_pathways: frozenset[str],
    policy: DiscoveryPolicy,
) -> pd.DataFrame:
    forbidden = sorted(_FORBIDDEN_SELECTION_COLUMNS & set(frame.columns))
    if forbidden:
        raise MetacellUniverseError(
            f"LASSO candidate selection contains target-derived columns {forbidden}"
        )
    required = CONTEXT_KEYS + PAIR_KEYS + (
        "outer_fold_id",
        "lasso_available",
        "pathway_stable_support",
        "pathway_null_z",
        "inner_relative_mse_gain",
        "pathway_calibration_available",
        "pathway_selected_member_count",
        "training_donor_count",
        "held_out_donor_rows_used",
        "source_donor_hash",
    )
    _require_columns(frame, required, "fold-local LASSO evidence")
    local = _normalise_ids(frame.loc[:, list(required)], CONTEXT_KEYS + PAIR_KEYS)
    mask = np.ones(len(local), dtype=bool)
    for key in CONTEXT_KEYS:
        mask &= local[key].eq(context[key]).to_numpy()
    local = local.loc[mask].copy()
    if local.empty:
        return local

    folds = pd.to_numeric(local.outer_fold_id, errors="raise").astype(int)
    if not folds.eq(int(outer_fold_id)).all():
        raise MetacellUniverseError("LASSO evidence mixes outer folds")
    held_out = pd.to_numeric(local.held_out_donor_rows_used, errors="raise").astype(int)
    if not held_out.eq(0).all():
        raise MetacellUniverseError("held-out donor rows entered LASSO selection")
    donor_hash = local.source_donor_hash.fillna("").astype(str).str.strip()
    if donor_hash.eq("").any() or donor_hash.nunique() != 1:
        raise MetacellUniverseError("one non-empty source donor hash is required per partition")

    numeric = (
        "pathway_stable_support",
        "pathway_null_z",
        "inner_relative_mse_gain",
        "pathway_selected_member_count",
        "training_donor_count",
    )
    for column in numeric:
        local[column] = pd.to_numeric(local[column], errors="raise")
    if not np.isfinite(local.loc[:, numeric].to_numpy(dtype=float)).all():
        raise MetacellUniverseError("non-finite LASSO selection evidence")

    available = _strict_bool(local.lasso_available, "lasso_available")
    calibrated = _strict_bool(
        local.pathway_calibration_available, "pathway_calibration_available"
    )
    eligible = (
        available
        & local.training_donor_count.ge(policy.min_training_donors)
        & local.pathway_selected_member_count.gt(0)
        & local.pathway_stable_support.gt(policy.min_pathway_stable_support)
        & local.inner_relative_mse_gain.ge(policy.min_inner_relative_mse_gain)
        & local.lncrna_id.isin(measured_lncrnas & core_lncrnas)
        & local.pathway_id.isin(available_pathways & core_pathways)
    )
    if policy.require_null_calibration:
        eligible &= calibrated
    local = local.loc[eligible].copy()
    if local.empty:
        return local

    duplicate = local.duplicated(PAIR_KEYS, keep=False)
    if duplicate.any():
        raise MetacellUniverseError("duplicate exact pairs in LASSO evidence partition")
    local = local.sort_values(
        ["lncrna_id", "pathway_stable_support", "pathway_null_z", "inner_relative_mse_gain", "pathway_id"],
        ascending=[True, False, False, False, True],
        kind="mergesort",
    )
    return local.groupby("lncrna_id", observed=True, sort=False).head(
        policy.top_k_pathways_per_lncrna
    ).reset_index(drop=True)


def build_dual_universe_partition(
    *,
    context: Mapping[str, Any],
    outer_fold_id: int,
    bulk_candidates: pd.DataFrame,
    lasso_evidence: pd.DataFrame,
    measured_lncrna_ids: Iterable[Any],
    available_pathway_ids: Iterable[Any],
    core_lncrna_ids: Iterable[Any],
    core_pathway_ids: Iterable[Any],
    policy: DiscoveryPolicy | None = None,
) -> pd.DataFrame:
    """Build one context/fold universe before any query-donor target is read."""

    settings = policy or DiscoveryPolicy()
    settings.validate()
    clean_context = _validate_context(context)
    fold = int(outer_fold_id)
    if fold not in range(5):
        raise MetacellUniverseError("outer_fold_id must be in 0..4")

    measured_lncrnas = _as_id_set(measured_lncrna_ids, "measured lncRNAs")
    available_pathways = _as_id_set(available_pathway_ids, "available pathways")
    core_lncrnas = _as_id_set(core_lncrna_ids, "core lncRNAs")
    core_pathways = _as_id_set(core_pathway_ids, "core pathways")
    coverage = audit_node_coverage(
        measured_lncrna_ids=measured_lncrnas,
        available_pathway_ids=available_pathways,
        core_lncrna_ids=core_lncrnas,
        core_pathway_ids=core_pathways,
    )
    if not coverage["complete"] and not settings.allow_core_limited_discovery:
        raise MetacellUniverseError(
            "single-cell discovery contains nodes outside frozen core; "
            "write an explicit OOV audit and enable core-limited discovery or "
            "supply a cold-start representation"
        )
    bulk = _normalise_bulk_candidates(
        bulk_candidates,
        cancer_id=clean_context["cancer_id"],
        measured_lncrnas=measured_lncrnas,
        available_pathways=available_pathways,
        core_lncrnas=core_lncrnas,
        core_pathways=core_pathways,
    )
    discovery = _normalise_lasso_evidence(
        lasso_evidence,
        context=clean_context,
        outer_fold_id=fold,
        measured_lncrnas=measured_lncrnas,
        available_pathways=available_pathways,
        core_lncrnas=core_lncrnas,
        core_pathways=core_pathways,
        policy=settings,
    )

    evidence_columns = [
        "pathway_stable_support",
        "pathway_null_z",
        "inner_relative_mse_gain",
        "pathway_calibration_available",
        "pathway_selected_member_count",
        "training_donor_count",
        "source_donor_hash",
    ]
    discovery_pairs = discovery.loc[:, list(PAIR_KEYS) + evidence_columns].copy()
    union = bulk.assign(is_bulk_candidate=True).merge(
        discovery_pairs.assign(is_sc_discovery_candidate=True),
        on=list(PAIR_KEYS),
        how="outer",
        validate="one_to_one",
    )
    if union.empty:
        columns = list(UNIVERSE_KEYS) + [
            "outer_fold_id", "is_bulk_candidate", "is_sc_discovery_candidate",
            "universe_role", "selection_policy", "candidate_selection_target_used",
            "held_out_donor_rows_used", "discovery_node_scope",
        ] + evidence_columns
        return pd.DataFrame(columns=columns)

    union["is_bulk_candidate"] = union.is_bulk_candidate.fillna(False).astype(bool)
    union["is_sc_discovery_candidate"] = (
        union.is_sc_discovery_candidate.fillna(False).astype(bool)
    )
    union["universe_role"] = np.select(
        [
            union.is_bulk_candidate & union.is_sc_discovery_candidate,
            union.is_bulk_candidate,
        ],
        ["bulk_and_sc_discovery", "bulk_rescore"],
        default="sc_discovery",
    )
    for key, value in clean_context.items():
        union[key] = value
    union["outer_fold_id"] = fold
    union["selection_policy"] = SELECTION_POLICY
    union["candidate_selection_target_used"] = False
    union["held_out_donor_rows_used"] = 0
    union["discovery_node_scope"] = (
        "FULL_MEASURED_CORE_COVERAGE"
        if coverage["complete"] else "EXPLICIT_FROZEN_CORE_INTERSECTION"
    )
    order = list(UNIVERSE_KEYS) + [
        "outer_fold_id", "is_bulk_candidate", "is_sc_discovery_candidate",
        "universe_role", "selection_policy", "candidate_selection_target_used",
        "held_out_donor_rows_used", "discovery_node_scope",
    ] + evidence_columns
    return union.loc[:, order].sort_values(list(UNIVERSE_KEYS), kind="mergesort").reset_index(drop=True)


def attach_query_donor_targets(
    universe: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    expected_outer_fold_id: int,
) -> pd.DataFrame:
    """Attach held-out donor metacell targets after universe construction."""

    _require_columns(
        universe,
        UNIVERSE_KEYS + (
            "outer_fold_id", "candidate_selection_target_used",
            "held_out_donor_rows_used",
        ),
        "dual universe",
    )
    target_required = TARGET_KEYS + (
        "rho", "association_target", "n_metacells", "association_method",
        "association_observation_unit", "association_target_level",
        "donor_resolved_target", "donor_target_row_id",
    )
    _require_columns(targets, target_required, "query-donor metacell targets")
    if _strict_bool(
        universe.candidate_selection_target_used,
        "candidate_selection_target_used",
    ).any():
        raise MetacellUniverseError("target-derived candidate selection is forbidden")
    if not pd.to_numeric(universe.held_out_donor_rows_used, errors="raise").eq(0).all():
        raise MetacellUniverseError("universe used held-out donor rows")
    folds = pd.to_numeric(universe.outer_fold_id, errors="raise").astype(int)
    if len(universe) and not folds.eq(int(expected_outer_fold_id)).all():
        raise MetacellUniverseError("universe outer fold mismatch")

    local = _normalise_ids(targets.loc[:, list(target_required)], TARGET_KEYS)
    if not local.association_method.eq("WITHIN_DONOR_METACELL_SPEARMAN").all():
        raise MetacellUniverseError("query targets use the wrong association method")
    if not local.association_observation_unit.eq("within_donor_metacell").all():
        raise MetacellUniverseError("query targets are not metacell-resolved")
    if not local.association_target_level.eq(
        "donor_x_celltype_x_lncrna_x_exact_pathway"
    ).all():
        raise MetacellUniverseError("query targets use the wrong target level")
    if not _strict_bool(local.donor_resolved_target, "donor_resolved_target").all():
        raise MetacellUniverseError("query targets lack donor-resolved attestation")
    row_ids = local.donor_target_row_id.fillna("").astype(str).str.strip()
    if row_ids.eq("").any() or row_ids.nunique() != len(local):
        raise MetacellUniverseError("query target row lineage is not unique")
    if local.duplicated(TARGET_KEYS).any():
        raise MetacellUniverseError("duplicate query-donor exact-pair targets")

    local["rho"] = pd.to_numeric(local.rho, errors="raise")
    local["association_target"] = pd.to_numeric(local.association_target, errors="raise")
    local["n_metacells"] = pd.to_numeric(local.n_metacells, errors="raise").astype(int)
    if not np.isfinite(local[["rho", "association_target"]].to_numpy(dtype=float)).all():
        raise MetacellUniverseError("non-finite query target")
    if not np.allclose(local.association_target, local.rho.abs(), rtol=0.0, atol=1e-7):
        raise MetacellUniverseError("query target is not absolute within-donor rho")
    if local.n_metacells.lt(20).any():
        raise MetacellUniverseError("query target has fewer than 20 metacells")

    result = universe.merge(local, on=list(UNIVERSE_KEYS), how="inner", validate="one_to_many")
    result["target_policy"] = TARGET_POLICY
    return result.sort_values(list(TARGET_KEYS), kind="mergesort").reset_index(drop=True)
