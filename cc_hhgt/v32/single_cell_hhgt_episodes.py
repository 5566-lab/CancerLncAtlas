"""Compact, donor-isolated episode assembly for the formal SC-HHGT trainer."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy import sparse
import torch

from .single_cell_batched_lasso import (
    BatchedLassoPolicy,
    BatchedLassoResult,
    fit_batched_lasso,
)


EPISODE_FORMAT = "CANCERLNCATLAS_V32_SC_HHGT_EPISODE_V1"
STATIC_FORMAT = "CANCERLNCATLAS_V32_SC_HHGT_EPISODE_STATIC_V1"
PAIR_FEATURE_NAMES = (
    "source_meta_rho",
    "source_direction_consistency",
    "log1p_source_donor_count",
    "lasso_pathway_stable_support",
    "lasso_inner_relative_mse_gain",
    "lasso_available",
)
LASSO_DERIVED_PAIR_FEATURE_MASK = (False, False, False, True, True, True)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def sparse_coo_payload(matrix: sparse.spmatrix) -> dict[str, Any]:
    value = sparse.coo_matrix(matrix, dtype=np.float32)
    value.sum_duplicates()
    return {
        "row_index": torch.as_tensor(value.row.copy(), dtype=torch.long),
        "column_index": torch.as_tensor(value.col.copy(), dtype=torch.long),
        "weight": torch.as_tensor(value.data.copy(), dtype=torch.float32),
        "shape": [int(value.shape[0]), int(value.shape[1])],
    }


def _donor_tuple(values: Iterable[object]) -> tuple[str, ...]:
    result = tuple(sorted({str(value).strip() for value in values}))
    if "" in result:
        raise ValueError("SC_EPISODE_EMPTY_DONOR")
    return result


def _weighted_summary(
    values: Any, donors: np.ndarray, *, minimum_donors: int = 3
) -> np.ndarray:
    matrix = values.toarray() if sparse.issparse(values) else np.asarray(values)
    matrix = np.asarray(matrix, dtype=np.float32)
    donors = np.asarray(donors, dtype=str)
    if matrix.ndim != 2 or matrix.shape[0] != len(donors) or not len(donors):
        raise ValueError("SC_EPISODE_SUMMARY_ALIGNMENT")
    unique, inverse, counts = np.unique(donors, return_inverse=True, return_counts=True)
    if minimum_donors < 2:
        raise ValueError("SC_EPISODE_MINIMUM_DONORS_LT2")
    if len(unique) < minimum_donors:
        raise ValueError(f"SC_EPISODE_SOURCE_DONORS_LT{minimum_donors}")
    weights = 1.0 / counts[inverse].astype(np.float64)
    weights /= weights.sum()
    mean = weights @ matrix
    variance = weights @ ((matrix - mean) ** 2)
    detected = weights @ (matrix > 0)
    return np.column_stack((mean, np.sqrt(np.maximum(variance, 0.0)), detected)).astype(
        np.float32
    )


def source_context_features(
    coding: sparse.spmatrix,
    lncrna: sparse.spmatrix,
    pathway_activity: np.ndarray,
    observations: pd.DataFrame,
    *,
    cell_key: str,
    source_donors: Iterable[object],
    pathway_family_membership: sparse.spmatrix,
    minimum_context_donors: int = 3,
) -> dict[str, np.ndarray]:
    """Summarize one context using source donors only and donor-balanced weights."""

    required = {"metacell_row", "donor_id", "cell_key"}
    if missing := sorted(required - set(observations)):
        raise ValueError(f"SC_EPISODE_OBSERVATION_COLUMNS:{missing}")
    coding = sparse.csr_matrix(coding, dtype=np.float32)
    lncrna = sparse.csr_matrix(lncrna, dtype=np.float32)
    activity = np.asarray(pathway_activity, dtype=np.float32)
    if not (coding.shape[0] == lncrna.shape[0] == activity.shape[0] == len(observations)):
        raise ValueError("SC_EPISODE_MATRIX_ALIGNMENT")
    source = set(_donor_tuple(source_donors))
    selected = observations.cell_key.astype(str).eq(str(cell_key)) & observations.donor_id.astype(
        str
    ).isin(source)
    rows = observations.loc[selected, "metacell_row"].to_numpy(np.int64)
    donors = observations.loc[selected, "donor_id"].astype(str).to_numpy()
    if minimum_context_donors < 2:
        raise ValueError("SC_EPISODE_CONTEXT_MINIMUM_DONORS_LT2")
    if len(set(donors)) < minimum_context_donors:
        raise ValueError(
            f"SC_EPISODE_CONTEXT_SOURCE_DONORS_LT{minimum_context_donors}"
        )
    if not np.array_equal(rows, np.sort(rows)) or rows.min(initial=0) < 0:
        raise ValueError("SC_EPISODE_METACELL_ROW_INVALID")
    family = sparse.csr_matrix(pathway_family_membership, dtype=np.float32)
    if family.shape[0] != activity.shape[1]:
        raise ValueError("SC_EPISODE_PATHWAY_FAMILY_SHAPE")
    denominator = np.asarray(family.sum(axis=0)).ravel()
    if not len(denominator) or np.any(denominator <= 0):
        raise ValueError("SC_EPISODE_EMPTY_PATHWAY_FAMILY")
    gene = _weighted_summary(
        coding[rows], donors, minimum_donors=minimum_context_donors
    )
    lnc = _weighted_summary(
        lncrna[rows], donors, minimum_donors=minimum_context_donors
    )
    pathway = _weighted_summary(
        activity[rows], donors, minimum_donors=minimum_context_donors
    )
    pathway_family = np.asarray(family.T @ pathway, dtype=np.float32)
    pathway_family /= denominator[:, None]
    coding_local, lnc_local = coding[rows], lncrna[rows]
    context = np.asarray(
        [[
            np.log1p(len(set(donors))),
            np.log1p(len(rows)),
            coding_local.getnnz() / float(np.prod(coding_local.shape)),
            lnc_local.getnnz() / float(np.prod(lnc_local.shape)),
        ]],
        dtype=np.float32,
    )
    return {
        "lncRNA": lnc,
        "gene": gene,
        "pathway": pathway,
        "pathway_family": pathway_family,
        "cell_context": context,
    }


def fit_context_lasso(
    coding: sparse.spmatrix,
    lncrna: sparse.spmatrix,
    observations: pd.DataFrame,
    *,
    cell_key: str,
    source_donors: Iterable[object],
    excluded_donors: Iterable[object],
    response_global_indices: np.ndarray,
    lncrna_ids: list[str],
    gene_ids: list[str],
    policy: BatchedLassoPolicy,
) -> BatchedLassoResult:
    selected = observations.cell_key.astype(str).eq(str(cell_key))
    rows = observations.loc[selected, "metacell_row"].to_numpy(np.int64)
    donors = observations.loc[selected, "donor_id"].astype(str).to_numpy()
    response_global_indices = np.asarray(response_global_indices, dtype=np.int64)
    if not len(rows) or not len(response_global_indices):
        raise ValueError("SC_EPISODE_EMPTY_LASSO_CONTEXT")
    return fit_batched_lasso(
        sparse.csr_matrix(coding)[rows],
        sparse.csr_matrix(lncrna)[rows][:, response_global_indices],
        donors,
        source_donors=_donor_tuple(source_donors),
        excluded_donors=_donor_tuple(excluded_donors),
        response_ids=[lncrna_ids[index] for index in response_global_indices],
        gene_ids=gene_ids,
        policy=policy,
    )


def lasso_pair_features(
    result: BatchedLassoResult,
    gene_pathway_membership: sparse.spmatrix,
    row_response_indices: np.ndarray,
    row_pathway_global_indices: np.ndarray,
    selected_pathway_global_indices: np.ndarray,
    *,
    stability_threshold: float,
) -> np.ndarray:
    """Return support, CV gain, and availability for candidate rows."""

    response = np.asarray(row_response_indices, dtype=np.int64)
    pathway_global = np.asarray(row_pathway_global_indices, dtype=np.int64)
    selected_pathway_global = np.asarray(selected_pathway_global_indices, dtype=np.int64)
    pathway_lookup = {int(value): index for index, value in enumerate(selected_pathway_global)}
    try:
        pathway_local = np.asarray([pathway_lookup[int(value)] for value in pathway_global])
    except KeyError as exc:
        raise ValueError("SC_EPISODE_PATHWAY_LOCALIZATION") from exc
    stable = result.stability >= float(stability_threshold)
    evidence = sparse.coo_matrix(
        (
            np.abs(result.coefficient[stable]) * result.stability[stable],
            (result.response_index[stable], result.gene_index[stable]),
        ),
        shape=(result.response_count, result.gene_count),
    ).tocsr()
    membership = sparse.csr_matrix(gene_pathway_membership, dtype=np.float32)[
        :, selected_pathway_global
    ]
    support = (evidence @ membership).tocsr()
    pathway_size = np.sqrt(np.maximum(np.asarray(membership.getnnz(axis=0)).ravel(), 1))
    support_value = np.asarray(support[response, pathway_local]).ravel() / pathway_size[
        pathway_local
    ]
    null = np.asarray(result.inner_null_mse, dtype=np.float64)
    cv = np.asarray(result.inner_cv_mse, dtype=np.float64)
    gain = np.zeros(result.response_count, dtype=np.float32)
    valid = result.available & np.isfinite(null) & np.isfinite(cv) & (null > 0)
    gain[valid] = np.clip(1.0 - cv[valid] / null[valid], -1.0, 1.0)
    return np.column_stack(
        (
            np.log1p(np.maximum(support_value, 0.0)),
            gain[response],
            result.available[response].astype(np.float32),
        )
    ).astype(np.float32)


def local_node_indices(
    *,
    lnc_global: np.ndarray,
    pathway_global: np.ndarray,
    lasso_result: BatchedLassoResult,
    gene_pathway_membership: sparse.spmatrix,
    pathway_family_membership: sparse.spmatrix,
    external_edges: pd.DataFrame,
    stability_threshold: float,
) -> dict[str, np.ndarray]:
    lnc_global = np.unique(np.asarray(lnc_global, dtype=np.int64))
    pathway_global = np.unique(np.asarray(pathway_global, dtype=np.int64))
    membership = sparse.csr_matrix(gene_pathway_membership)
    family = sparse.csr_matrix(pathway_family_membership)
    gene_from_pathway = membership[:, pathway_global].nonzero()[0]
    stable = lasso_result.stability >= float(stability_threshold)
    gene_from_lasso = lasso_result.gene_index[stable]
    external_gene = np.empty(0, dtype=np.int64)
    if len(external_edges):
        selected = external_edges.source_index.astype(np.int64).isin(lnc_global)
        external_gene = external_edges.loc[selected, "target_index"].to_numpy(np.int64)
    family_global = family[pathway_global].nonzero()[1]
    result = {
        "lncRNA": lnc_global,
        "gene": np.unique(np.concatenate((gene_from_pathway, gene_from_lasso, external_gene))),
        "pathway": pathway_global,
        "pathway_family": np.unique(family_global),
    }
    if any(not len(value) for value in result.values()):
        raise ValueError("SC_EPISODE_EMPTY_LOCAL_NODE_TYPE")
    return result


def localized_lasso_edges(
    result: BatchedLassoResult,
    response_global_indices: np.ndarray,
    node_global_indices: Mapping[str, np.ndarray],
) -> dict[str, torch.Tensor]:
    lnc_position = {
        int(value): index for index, value in enumerate(node_global_indices["lncRNA"])
    }
    gene_position = {
        int(value): index for index, value in enumerate(node_global_indices["gene"])
    }
    response_global_indices = np.asarray(response_global_indices, dtype=np.int64)
    keep, lnc_local, gene_local = [], [], []
    for edge, (response_index, gene_index) in enumerate(
        zip(result.response_index, result.gene_index, strict=True)
    ):
        lnc_global = int(response_global_indices[int(response_index)])
        if lnc_global in lnc_position and int(gene_index) in gene_position:
            keep.append(edge)
            lnc_local.append(lnc_position[lnc_global])
            gene_local.append(gene_position[int(gene_index)])
    indices = np.asarray(keep, dtype=np.int64)
    return {
        "lnc_index": torch.as_tensor(lnc_local, dtype=torch.long),
        "gene_index": torch.as_tensor(gene_local, dtype=torch.long),
        "coefficient": torch.as_tensor(result.coefficient[indices], dtype=torch.float32),
        "stability": torch.as_tensor(result.stability[indices], dtype=torch.float32),
    }


def episode_pair_features(rows: pd.DataFrame, lasso_features: np.ndarray) -> dict[str, torch.Tensor]:
    required = {"source_meta_rho", "source_direction_consistency", "source_donor_count"}
    if missing := sorted(required - set(rows)):
        raise ValueError(f"SC_EPISODE_PAIR_FEATURE_COLUMNS:{missing}")
    base = np.column_stack(
        (
            pd.to_numeric(rows.source_meta_rho, errors="raise").to_numpy(np.float32),
            pd.to_numeric(rows.source_direction_consistency, errors="raise").to_numpy(
                np.float32
            ),
            np.log1p(
                pd.to_numeric(rows.source_donor_count, errors="raise").to_numpy(np.float32)
            ),
        )
    )
    with_lasso = np.column_stack((base, np.asarray(lasso_features, dtype=np.float32)))
    if with_lasso.shape != (len(rows), len(PAIR_FEATURE_NAMES)) or not np.isfinite(
        with_lasso
    ).all():
        raise ValueError("SC_EPISODE_PAIR_FEATURE_VALUES")
    without_lasso = with_lasso.copy()
    without_lasso[:, np.asarray(LASSO_DERIVED_PAIR_FEATURE_MASK, dtype=bool)] = 0.0
    return {
        "with_lasso": torch.as_tensor(with_lasso, dtype=torch.float32),
        "without_lasso": torch.as_tensor(without_lasso, dtype=torch.float32),
    }


def lasso_policy_receipt(policy: BatchedLassoPolicy) -> dict[str, Any]:
    return {"policy": asdict(policy), "policy_sha256": canonical_hash(asdict(policy))}


__all__ = [
    "EPISODE_FORMAT",
    "LASSO_DERIVED_PAIR_FEATURE_MASK",
    "PAIR_FEATURE_NAMES",
    "STATIC_FORMAT",
    "canonical_hash",
    "episode_pair_features",
    "fit_context_lasso",
    "lasso_pair_features",
    "lasso_policy_receipt",
    "local_node_indices",
    "localized_lasso_edges",
    "source_context_features",
    "sparse_coo_payload",
]
