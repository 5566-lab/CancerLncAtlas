"""Nested donor isolation for SC-HHGT episode preparation.

The outer donor folds are immutable. Donor-rich cancer/fold strata reserve a
strict validation fold. Strata that cannot retain three fit-source donors use
leave-one-target fit episodes instead; they never supply model-selection rows.
This keeps low-donor cancers in optimization without exposing an outer query.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse

from .single_cell_metacells import (
    LocalSpace,
    aggregate_measurements,
    donor_weights,
    local_disjoint_groups,
)


NESTED_ASSET_FORMAT = "CANCERLNCATLAS_V32_SC_HHGT_NESTED_ASSET_V1"
NESTED_PREFLIGHT_FORMAT = "CANCERLNCATLAS_V32_SC_HHGT_NESTED_PREFLIGHT_V1"
VALIDATION_GEOMETRY_ISOLATION = "outer_and_inner_target_fold"
FIT_GEOMETRY_ISOLATION = "outer_and_inner_target_and_validation_fold"
SUPPORT_LIMITED_FIT_GEOMETRY_ISOLATION = "outer_and_inner_target_fold"


@dataclass(frozen=True)
class NestedFoldPlan:
    cancer_id: str
    outer_fold_id: int
    inner_target_fold_id: int
    inner_validation_fold_id: int
    role: str
    outer_train_donors: tuple[str, ...]
    outer_query_donors: tuple[str, ...]
    inner_target_donors: tuple[str, ...]
    inner_validation_donors: tuple[str, ...]
    source_donors: tuple[str, ...]
    excluded_donors: tuple[str, ...]
    geometry_isolation: str
    validation_provider: bool
    support_limited_fit: bool
    minimum_source_donors: int

    @property
    def available(self) -> bool:
        return len(self.source_donors) >= self.minimum_source_donors

    def as_dict(self) -> dict[str, object]:
        return {
            "cancer_id": self.cancer_id,
            "outer_fold_id": self.outer_fold_id,
            "inner_target_fold_id": self.inner_target_fold_id,
            "inner_validation_fold_id": self.inner_validation_fold_id,
            "role": self.role,
            "outer_train_donors": list(self.outer_train_donors),
            "outer_query_donors": list(self.outer_query_donors),
            "inner_target_donors": list(self.inner_target_donors),
            "inner_validation_donors": list(self.inner_validation_donors),
            "source_donors": list(self.source_donors),
            "excluded_donors": list(self.excluded_donors),
            "geometry_isolation": self.geometry_isolation,
            "validation_provider": self.validation_provider,
            "support_limited_fit": self.support_limited_fit,
            "minimum_source_donors": self.minimum_source_donors,
            "available": self.available,
            "unavailable_reason": (
                None
                if self.available
                else f"INSUFFICIENT_SOURCE_DONORS_LT{self.minimum_source_donors}"
            ),
        }


def _donor_tuple(values: Iterable[object]) -> tuple[str, ...]:
    donors = tuple(sorted({str(value).strip() for value in values}))
    if any(not value for value in donors):
        raise ValueError("EMPTY_DONOR_ID")
    return donors


def nested_fold_plans(
    donor_folds: pd.DataFrame,
    *,
    cancer_id: str,
    outer_fold_id: int,
) -> list[NestedFoldPlan]:
    """Derive the globally strict fit/validation plan from frozen folds."""

    required = {"cancer_id", "donor_id", "outer_fold_id"}
    if missing := sorted(required - set(donor_folds.columns)):
        raise ValueError(f"DONOR_FOLDS_MISSING_COLUMNS:{missing}")
    frame = donor_folds.loc[
        donor_folds.cancer_id.astype(str).str.upper().eq(str(cancer_id).upper())
    ].copy()
    if frame.empty or frame.donor_id.astype(str).duplicated().any():
        raise ValueError("DONOR_FOLD_IDENTITY_INVALID")
    frame["donor_id"] = frame.donor_id.astype(str)
    frame["outer_fold_id"] = pd.to_numeric(
        frame.outer_fold_id, errors="raise"
    ).astype(int)
    active = sorted(frame.outer_fold_id.unique().tolist())
    if active != list(range(len(active))) or len(active) < 3:
        raise ValueError("ACTIVE_OUTER_FOLDS_MUST_BE_CONTIGUOUS_AND_AT_LEAST_THREE")
    outer = int(outer_fold_id)
    if outer not in active:
        raise ValueError("OUTER_FOLD_NOT_ACTIVE")
    validation = (outer + 1) % len(active)
    by_fold = {
        fold: _donor_tuple(frame.loc[frame.outer_fold_id.eq(fold), "donor_id"])
        for fold in active
    }
    all_donors = set(frame.donor_id)
    outer_query = by_fold[outer]
    outer_train = _donor_tuple(all_donors - set(outer_query))
    validation_donors = by_fold[validation]
    strict_fit_targets = [fold for fold in active if fold not in {outer, validation}]
    strict_validation_source = all_donors - set(outer_query) - set(validation_donors)
    strict_fit_sources = [
        all_donors
        - set(outer_query)
        - set(validation_donors)
        - set(by_fold[target])
        for target in strict_fit_targets
    ]
    validation_provider = (
        len(strict_validation_source) >= 3
        and bool(strict_fit_sources)
        and all(len(source) >= 3 for source in strict_fit_sources)
    )
    plans: list[NestedFoldPlan] = []
    if not validation_provider:
        for target in [fold for fold in active if fold != outer]:
            target_donors = by_fold[target]
            excluded = set(outer_query) | set(target_donors)
            plan = NestedFoldPlan(
                cancer_id=str(cancer_id).upper(),
                outer_fold_id=outer,
                inner_target_fold_id=target,
                inner_validation_fold_id=validation,
                role="fit",
                outer_train_donors=outer_train,
                outer_query_donors=outer_query,
                inner_target_donors=target_donors,
                inner_validation_donors=(),
                source_donors=_donor_tuple(all_donors - excluded),
                excluded_donors=_donor_tuple(excluded),
                geometry_isolation=SUPPORT_LIMITED_FIT_GEOMETRY_ISOLATION,
                validation_provider=False,
                support_limited_fit=True,
                minimum_source_donors=2,
            )
            _validate_plan(plan, all_donors)
            plans.append(plan)
        return plans
    for target in [validation] + [
        fold for fold in active if fold not in {outer, validation}
    ]:
        role = "validation" if target == validation else "fit"
        target_donors = by_fold[target]
        excluded = set(outer_query) | set(target_donors)
        isolation = VALIDATION_GEOMETRY_ISOLATION
        if role == "fit":
            excluded |= set(validation_donors)
            isolation = FIT_GEOMETRY_ISOLATION
        source = _donor_tuple(all_donors - excluded)
        plan = NestedFoldPlan(
            cancer_id=str(cancer_id).upper(),
            outer_fold_id=outer,
            inner_target_fold_id=target,
            inner_validation_fold_id=validation,
            role=role,
            outer_train_donors=outer_train,
            outer_query_donors=outer_query,
            inner_target_donors=target_donors,
            inner_validation_donors=validation_donors,
            source_donors=source,
            excluded_donors=_donor_tuple(excluded),
            geometry_isolation=isolation,
            validation_provider=True,
            support_limited_fit=False,
            minimum_source_donors=3,
        )
        _validate_plan(plan, all_donors)
        plans.append(plan)
    return plans


def _validate_plan(plan: NestedFoldPlan, all_donors: Iterable[str]) -> None:
    all_set = set(map(str, all_donors))
    source = set(plan.source_donors)
    target = set(plan.inner_target_donors)
    validation = set(plan.inner_validation_donors)
    outer_query = set(plan.outer_query_donors)
    excluded = set(plan.excluded_donors)
    if source & excluded or target & source or outer_query & set(plan.outer_train_donors):
        raise ValueError("NESTED_DONOR_PARTITION_LEAK")
    if source | excluded != all_set:
        raise ValueError("NESTED_DONOR_PARTITION_INCOMPLETE")
    if plan.role == "validation":
        if not plan.validation_provider or plan.support_limited_fit:
            raise ValueError("NESTED_VALIDATION_PROVIDER_DRIFT")
        if target != validation or excluded != outer_query | target:
            raise ValueError("NESTED_VALIDATION_PARTITION_DRIFT")
        if plan.geometry_isolation != VALIDATION_GEOMETRY_ISOLATION:
            raise ValueError("NESTED_VALIDATION_GEOMETRY_TOKEN_DRIFT")
    elif plan.role == "fit":
        if plan.support_limited_fit:
            if plan.validation_provider or validation or excluded != outer_query | target:
                raise ValueError("NESTED_SUPPORT_LIMITED_FIT_PARTITION_DRIFT")
            if plan.geometry_isolation != SUPPORT_LIMITED_FIT_GEOMETRY_ISOLATION:
                raise ValueError("NESTED_SUPPORT_LIMITED_FIT_GEOMETRY_TOKEN_DRIFT")
            if plan.minimum_source_donors != 2:
                raise ValueError("NESTED_SUPPORT_LIMITED_SOURCE_GATE_DRIFT")
        else:
            if not plan.validation_provider or target & validation or excluded != outer_query | target | validation:
                raise ValueError("NESTED_FIT_PARTITION_DRIFT")
            if plan.geometry_isolation != FIT_GEOMETRY_ISOLATION:
                raise ValueError("NESTED_FIT_GEOMETRY_TOKEN_DRIFT")
    else:
        raise ValueError("NESTED_ROLE_INVALID")


def fit_nested_metacells(
    coding_log: sparse.spmatrix,
    coding: sparse.spmatrix,
    lncrna: sparse.spmatrix,
    all_gene_library: np.ndarray,
    metadata: pd.DataFrame,
    plan: NestedFoldPlan,
    *,
    measurement_scale: str,
    max_genes: int = 1500,
    components: int = 20,
    max_cells_per_donor: int = 2000,
    target_size: int = 5,
    min_size: int = 3,
    radius_quantile: float = 0.9,
    seed: int = 1729,
) -> dict[str, object]:
    """Refit PCA/radius and aggregate metacells using only plan sources."""

    if not plan.available:
        raise ValueError(
            f"INSUFFICIENT_SOURCE_DONORS_LT{plan.minimum_source_donors}"
        )
    required = {"cell_id", "donor_id", "cell_type", "state_id"}
    if missing := sorted(required - set(metadata.columns)):
        raise ValueError(f"CELL_METADATA_MISSING_COLUMNS:{missing}")
    meta = metadata.reset_index(drop=True).copy()
    donors = meta.donor_id.astype(str)
    if set(donors) != set(plan.source_donors) | set(plan.excluded_donors):
        raise ValueError("NESTED_METADATA_DONOR_SCOPE_DRIFT")
    if not (len(meta) == coding.shape[0] == lncrna.shape[0] == coding_log.shape[0]):
        raise ValueError("NESTED_CELL_MATRIX_ALIGNMENT_DRIFT")
    # Persist only source plus the episode's label-bearing target fold.  Outer
    # query and reserved-validation donors are not merely masked: they are
    # physically absent from every episode matrix and membership table.
    retained_donors = set(plan.source_donors) | set(plan.inner_target_donors)
    keep = donors.isin(retained_donors).to_numpy()
    if set(donors[keep]) != retained_donors:
        raise ValueError("NESTED_RETAINED_DONORS_MISSING")
    meta = meta.loc[keep].reset_index(drop=True)
    donors = meta.donor_id.astype(str)
    coding_log = sparse.csr_matrix(coding_log)[keep]
    coding = sparse.csr_matrix(coding)[keep]
    lncrna = sparse.csr_matrix(lncrna)[keep]
    all_gene_library = np.asarray(all_gene_library)[keep]
    space = LocalSpace.fit(
        coding_log,
        donors,
        plan.source_donors,
        max_genes=max_genes,
        components=components,
        max_cells_per_donor=max_cells_per_donor,
        seed=seed,
    )
    if tuple(space.training_donors) != plan.source_donors:
        raise ValueError("NESTED_PCA_SOURCE_DONOR_DRIFT")
    coordinates = space.transform(coding_log)
    assignments, geometry = local_disjoint_groups(
        coordinates,
        meta[["donor_id", "cell_type", "state_id"]],
        target_size=target_size,
        min_size=min_size,
        radius_quantile=radius_quantile,
        allowed_training_donors=plan.source_donors,
        seed=seed + 1,
    )
    if tuple(geometry["training_donors"]) != plan.source_donors:
        raise ValueError("NESTED_RADIUS_SOURCE_DONOR_DRIFT")
    coding_mc, group_ids, aggregation_policy = aggregate_measurements(
        sparse.csr_matrix(coding),
        assignments,
        scale=measurement_scale,
        total_library=np.asarray(all_gene_library),
    )
    lnc_mc, lnc_group_ids, lnc_policy = aggregate_measurements(
        sparse.csr_matrix(lncrna),
        assignments,
        scale=measurement_scale,
        total_library=np.asarray(all_gene_library),
    )
    if not np.array_equal(group_ids, lnc_group_ids) or aggregation_policy != lnc_policy:
        raise ValueError("NESTED_AGGREGATION_ORDER_DRIFT")
    first_rows = np.asarray(
        [np.flatnonzero(assignments == group)[0] for group in group_ids],
        dtype=np.int64,
    )
    columns = [
        column
        for column in ("dataset_id", "cancer_id", "donor_id", "cell_type", "state_id")
        if column in meta
    ]
    metacell_metadata = meta.iloc[first_rows][columns].reset_index(drop=True)
    metacell_metadata["metacell_id"] = group_ids.astype(np.int64)
    sizes = np.bincount(assignments[assignments >= 0])
    metacell_metadata["n_cells"] = sizes[group_ids]
    roles = donor_roles(plan)
    metacell_metadata["observation_role"] = (
        metacell_metadata.donor_id.astype(str).map(roles)
    )
    if metacell_metadata.observation_role.isna().any():
        raise ValueError("NESTED_METACELL_DONOR_ROLE_MISSING")
    metacell_metadata["lasso_input_allowed"] = metacell_metadata.observation_role.eq(
        "source"
    )
    membership = meta[["cell_id", "donor_id", "cell_type", "state_id"]].copy()
    membership["metacell_id"] = assignments
    membership["observation_role"] = donors.map(roles)
    membership["geometry_fit_allowed"] = membership.observation_role.eq("source")
    geometry = {
        **geometry,
        "geometry_isolation": plan.geometry_isolation,
        "role": plan.role,
        "outer_fold_id": plan.outer_fold_id,
        "inner_target_fold_id": plan.inner_target_fold_id,
        "inner_validation_fold_id": plan.inner_validation_fold_id,
        "source_donors": list(plan.source_donors),
        "excluded_donors": list(plan.excluded_donors),
        "outer_query_donors": list(plan.outer_query_donors),
        "inner_target_donors": list(plan.inner_target_donors),
        "inner_validation_donors": list(plan.inner_validation_donors),
        "pca_training_donors": list(space.training_donors),
        "radius_training_donors": list(geometry["training_donors"]),
    }
    from .single_cell_geometry_replay import capture
    return {
        "local_space_arrays": capture(space, coordinates),
        "membership": membership,
        "metadata": metacell_metadata,
        "coding": coding_mc.astype(np.float32),
        "lncrna": lnc_mc.astype(np.float32),
        "metacell_ids": group_ids.astype(np.int64),
        "aggregation_policy": aggregation_policy,
        "geometry_audit": geometry,
    }


def donor_roles(plan: NestedFoldPlan) -> dict[str, str]:
    roles = {donor: "source" for donor in plan.source_donors}
    roles.update({donor: "inner_target" for donor in plan.inner_target_donors})
    if len(roles) != len(set(plan.source_donors) | set(plan.inner_target_donors)):
        raise ValueError("NESTED_DONOR_ROLE_OVERLAP")
    return roles


def lasso_observations(metadata: pd.DataFrame, plan: NestedFoldPlan) -> pd.DataFrame:
    required = {
        "metacell_id", "donor_id", "cell_type", "state_id", "n_cells",
        "observation_role", "lasso_input_allowed",
    }
    if missing := sorted(required - set(metadata.columns)):
        raise ValueError(f"NESTED_METACELL_METADATA_MISSING_COLUMNS:{missing}")
    result = metadata.copy().reset_index(drop=True)
    result.insert(0, "metacell_row", np.arange(len(result), dtype=np.int64))
    result = result.rename(columns={"cell_type": "cell_key"})
    donors = result.donor_id.astype(str)
    allowed = set(donors[result.lasso_input_allowed.astype(bool)])
    forbidden = set(donors[~result.lasso_input_allowed.astype(bool)])
    # A planned donor can legitimately contribute no metacell after the fixed
    # donor x cell-type x state grouping (for example, every stratum has fewer
    # than ``min_size`` cells).  Keep the planned sets as the fail-closed donor
    # allowlists, while preserving the actually observed subset for LASSO.
    if (
        not allowed.issubset(set(plan.source_donors))
        or not forbidden.issubset(set(plan.inner_target_donors))
        or forbidden - set(plan.excluded_donors)
        or allowed & set(plan.excluded_donors)
    ):
        raise ValueError("NESTED_LASSO_DONOR_SCOPE_DRIFT")
    columns = [
        "metacell_row", "metacell_id", "donor_id", "cell_key", "state_id",
        "n_cells", "observation_role", "lasso_input_allowed",
    ]
    for optional in ("dataset_id", "cancer_id"):
        if optional in result:
            columns.append(optional)
    return result.loc[:, columns]


def _weighted_summary(values, donors: np.ndarray) -> np.ndarray:
    matrix = values.toarray() if sparse.issparse(values) else np.asarray(values)
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2 or len(matrix) != len(donors) or not np.isfinite(matrix).all():
        raise ValueError("NESTED_SUMMARY_INPUT_INVALID")
    weights = donor_weights(donors).astype(np.float64)
    weights /= weights.sum()
    mean = weights @ matrix
    variance = weights @ ((matrix - mean) ** 2)
    detected = weights @ (matrix > 0)
    return np.column_stack([mean, np.sqrt(np.maximum(variance, 0)), detected]).astype(
        np.float32
    )


def source_node_features(
    coding,
    lncrna,
    pathway_activity: np.ndarray,
    metadata: pd.DataFrame,
    plan: NestedFoldPlan,
    pathway_family_membership: sparse.spmatrix,
    *,
    target_rows: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Build context-indexed node features from source metacells only."""

    activity = np.asarray(pathway_activity, dtype=np.float32)
    if not (len(metadata) == coding.shape[0] == lncrna.shape[0] == len(activity)):
        raise ValueError("NESTED_NODE_FEATURE_ALIGNMENT_DRIFT")
    family = sparse.csr_matrix(pathway_family_membership, dtype=np.float32)
    if family.shape[0] != activity.shape[1]:
        raise ValueError("NESTED_PATHWAY_FAMILY_SHAPE_DRIFT")
    denominator = np.asarray(family.sum(axis=0)).ravel()
    if len(denominator) == 0 or (denominator <= 0).any():
        raise ValueError("NESTED_PATHWAY_FAMILY_EMPTY")
    source_set = set(plan.source_donors)
    target_set = set(plan.inner_target_donors)
    rows: list[dict[str, object]] = []
    features: dict[str, list[np.ndarray]] = {
        "lncRNA": [], "gene": [], "pathway": [], "pathway_family": [],
        "cell_context": [],
    }
    target_contexts = set(target_rows.cell_key.astype(str)) if len(target_rows) else set()
    for cell_key in sorted(target_contexts):
        cell = metadata.cell_type.astype(str).eq(cell_key)
        source = cell & metadata.donor_id.astype(str).isin(source_set)
        target = cell & metadata.donor_id.astype(str).isin(target_set)
        source_donors = metadata.loc[source, "donor_id"].astype(str).to_numpy()
        if len(set(source_donors)) < plan.minimum_source_donors or not target.any():
            continue
        gene = _weighted_summary(coding[source.to_numpy()], source_donors)
        lnc = _weighted_summary(lncrna[source.to_numpy()], source_donors)
        pathway = _weighted_summary(activity[source.to_numpy()], source_donors)
        family_values = np.asarray(family.transpose() @ pathway, dtype=np.float32)
        family_values /= denominator[:, None]
        coding_local = coding[source.to_numpy()]
        lnc_local = lncrna[source.to_numpy()]
        coding_detection = float(coding_local.getnnz() / np.prod(coding_local.shape))
        lnc_detection = float(lnc_local.getnnz() / np.prod(lnc_local.shape))
        context = np.asarray(
            [[
                np.log1p(len(set(source_donors))),
                np.log1p(source.sum()),
                coding_detection,
                lnc_detection,
            ]],
            dtype=np.float32,
        )
        features["gene"].append(gene)
        features["lncRNA"].append(lnc)
        features["pathway"].append(pathway)
        features["pathway_family"].append(family_values)
        features["cell_context"].append(context)
        rows.append(
            {
                "context_index": len(rows),
                "cell_key": cell_key,
                "source_donor_count": len(set(source_donors)),
                "source_metacell_count": int(source.sum()),
                "target_donor_count": int(
                    metadata.loc[target, "donor_id"].astype(str).nunique()
                ),
                "target_metacell_count": int(target.sum()),
                "target_row_count": int(target_rows.cell_key.astype(str).eq(cell_key).sum()),
            }
        )
    if not rows:
        return {}, pd.DataFrame(
            columns=[
                "context_index", "cell_key", "source_donor_count",
                "source_metacell_count", "target_donor_count",
                "target_metacell_count", "target_row_count",
            ]
        )
    stacked = {key: np.stack(value).astype(np.float32) for key, value in features.items()}
    return stacked, pd.DataFrame(rows)


def validate_nested_isolation(
    plan: NestedFoldPlan,
    geometry_audit: dict[str, object],
    observations: pd.DataFrame,
    universe: pd.DataFrame,
    target_rows: pd.DataFrame,
) -> None:
    """Fail closed if any target/validation/query donor reached graph inputs."""

    source = set(plan.source_donors)
    excluded = set(plan.excluded_donors)
    for field in ("training_donors", "pca_training_donors", "radius_training_donors"):
        if set(map(str, geometry_audit.get(field, []))) != source:
            raise ValueError(f"NESTED_{field.upper()}_DRIFT")
    allowed = observations.loc[
        observations.lasso_input_allowed.astype(bool), "donor_id"
    ].astype(str)
    if not set(allowed).issubset(source) or set(allowed) & excluded:
        raise ValueError("NESTED_LASSO_SOURCE_LEAK")
    forbidden_flags = [
        "candidate_selection_query_donor_used",
        "candidate_selection_target_donor_used",
        "candidate_selection_validation_donor_used",
    ]
    for column in forbidden_flags:
        if column not in universe or universe[column].astype(bool).any():
            raise ValueError(f"NESTED_UNIVERSE_LEAK_FLAG:{column}")
    if len(target_rows):
        labels = set(target_rows.query_donor_id.astype(str))
        if not labels or not labels.issubset(set(plan.inner_target_donors)):
            raise ValueError("NESTED_TARGET_DONOR_SCOPE_DRIFT")
        if labels & source:
            raise ValueError("NESTED_TARGET_SOURCE_DONOR_LEAK")


__all__ = [
    "FIT_GEOMETRY_ISOLATION",
    "NESTED_ASSET_FORMAT",
    "NESTED_PREFLIGHT_FORMAT",
    "NestedFoldPlan",
    "SUPPORT_LIMITED_FIT_GEOMETRY_ISOLATION",
    "VALIDATION_GEOMETRY_ISOLATION",
    "donor_roles",
    "fit_nested_metacells",
    "lasso_observations",
    "nested_fold_plans",
    "source_node_features",
    "validate_nested_isolation",
]
