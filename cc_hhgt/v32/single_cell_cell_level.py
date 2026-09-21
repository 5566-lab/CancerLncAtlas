"""Fail-closed cell-level UCell primitives for the V3.2 single-cell pilot.

The implementation follows the current UCell ``u_stat`` contract:

* ranks are descending and ties use the average rank;
* ranks at or beyond the tail cutoff are capped at ``max_rank``;
* the denominator uses the *complete* exact signature length; and
* members absent from the matrix contribute ``max_rank`` to the rank sum.

The last rule is UCell's defined tail treatment.  It is not expression-data
imputation, and callers must retain present/missing member counts in a sidecar.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata


UCELL_ALGORITHM = "UCELL_MANN_WHITNEY_U_UPDATED_NORMALIZATION"
UCELL_FORMULA = "1-(rank_sum-n*(n+1)/2)/(n*maxRank-n*(n+1)/2)"
UCELL_MISSING_MEMBER_POLICY = (
    "MATRIX_MISSING_MEMBERS_CONTRIBUTE_MAX_RANK_BY_UCELL_TAIL_DEFINITION"
)
UCELL_TIES_METHOD = "average"

EMPTY_EXACT_SIGNATURE = "EMPTY_EXACT_SIGNATURE"
NO_MEMBER_PROTEIN_IN_MATRIX = "NO_MEMBER_PROTEIN_IN_MATRIX"
SIGNATURE_AT_OR_ABOVE_MAX_RANK = (
    "FULL_SIGNATURE_SIZE_AT_OR_EXCEEDS_UCELL_MAX_RANK_DOMAIN"
)


class UCellContractError(RuntimeError):
    """Raised when a score request violates the pinned UCell contract."""


@dataclass(frozen=True)
class UCellSignatureDomain:
    """Availability decision for one exact pathway signature."""

    available: bool
    unavailable_reason: str | None
    signature_gene_count: int
    present_gene_count: int
    missing_gene_count: int
    max_rank: int


@dataclass(frozen=True)
class UCellPathwayContract:
    """All-pathway availability sidecar plus available sparse signatures."""

    availability: pd.DataFrame
    available_pathway_ids: tuple[str, ...]
    signature_matrix: sparse.csr_matrix
    signature_gene_counts: np.ndarray
    present_gene_counts: np.ndarray
    missing_gene_counts: np.ndarray


def evaluate_ucell_signature_domain(
    *,
    signature_gene_count: int,
    present_gene_count: int,
    max_rank: int,
) -> UCellSignatureDomain:
    """Return the fail-closed UCell domain decision for one signature.

    V3.2 deliberately uses a stricter gate than UCell's public wrapper:
    complete signatures at or above ``max_rank`` are typed unavailable.  This
    keeps the complete-signature denominator inside the declared pilot domain
    and prevents a large pathway from being silently shortened to its detected
    subset.
    """

    n = int(signature_gene_count)
    present = int(present_gene_count)
    cutoff = int(max_rank)
    if cutoff <= 0:
        raise UCellContractError("max_rank must be positive")
    if n < 0 or present < 0 or present > n:
        raise UCellContractError(
            "signature counts must satisfy 0 <= present <= complete signature"
        )
    if n == 0:
        reason = EMPTY_EXACT_SIGNATURE
    elif present == 0:
        reason = NO_MEMBER_PROTEIN_IN_MATRIX
    elif n >= cutoff:
        reason = SIGNATURE_AT_OR_ABOVE_MAX_RANK
    else:
        reason = None
    return UCellSignatureDomain(
        available=reason is None,
        unavailable_reason=reason,
        signature_gene_count=n,
        present_gene_count=present,
        missing_gene_count=n - present,
        max_rank=cutoff,
    )


def rank_expression_ucell(
    expression_gene_by_cell: np.ndarray,
    *,
    max_rank: int,
) -> np.ndarray:
    """Return UCell-compatible capped ranks for a dense gene-by-cell block.

    This mirrors ``data.table::frankv(..., order=-1, ties.method='average')``
    followed by the current UCell tail cap.  The returned array is float64 so
    half ranks from even-sized ties are not truncated.
    """

    values = np.asarray(expression_gene_by_cell)
    if values.ndim != 2:
        raise UCellContractError("expression must be a 2-D gene-by-cell array")
    cutoff = int(max_rank)
    if cutoff <= 0:
        raise UCellContractError("max_rank must be positive")
    if not np.isfinite(values).all():
        raise UCellContractError("expression contains non-finite values")
    if (values < 0).any():
        raise UCellContractError("expression contains negative values")

    ranks = rankdata(-values, method=UCELL_TIES_METHOD, axis=0)
    return np.minimum(np.asarray(ranks, dtype=np.float64), float(cutoff))


def score_ucell_from_capped_ranks(
    capped_ranks_gene_by_cell: np.ndarray,
    *,
    present_gene_indices: Iterable[int],
    signature_gene_count: int,
    max_rank: int,
) -> np.ndarray:
    """Score a complete exact signature from its present matrix members.

    Matrix-missing signature members are represented only by their count and
    each contributes ``max_rank`` to ``rank_sum``, exactly as current UCell's
    ``u_stat``/``_calculate_U`` implementation does.
    """

    ranks = np.asarray(capped_ranks_gene_by_cell, dtype=np.float64)
    if ranks.ndim != 2:
        raise UCellContractError("capped ranks must be a 2-D gene-by-cell array")
    if not np.isfinite(ranks).all():
        raise UCellContractError("capped ranks contain non-finite values")

    indices = np.asarray(list(present_gene_indices), dtype=np.int64)
    if indices.ndim != 1:
        raise UCellContractError("present_gene_indices must be one-dimensional")
    if len(indices) and (indices.min() < 0 or indices.max() >= ranks.shape[0]):
        raise UCellContractError("present gene index is outside the rank matrix")
    if len(indices) != len(np.unique(indices)):
        raise UCellContractError("present signature gene indices are duplicated")

    domain = evaluate_ucell_signature_domain(
        signature_gene_count=int(signature_gene_count),
        present_gene_count=int(len(indices)),
        max_rank=int(max_rank),
    )
    if not domain.available:
        raise UCellContractError(
            f"signature is outside the numeric UCell domain: "
            f"{domain.unavailable_reason}"
        )

    present_ranks = np.minimum(ranks[indices, :], float(domain.max_rank))
    rank_sum = present_ranks.sum(axis=0, dtype=np.float64)
    rank_sum += float(domain.missing_gene_count * domain.max_rank)

    n = float(domain.signature_gene_count)
    rank_sum_min = n * (n + 1.0) / 2.0
    denominator = n * float(domain.max_rank) - rank_sum_min
    if denominator <= 0:
        raise UCellContractError("UCell denominator is non-positive")
    scores = 1.0 - (rank_sum - rank_sum_min) / denominator
    tolerance = 1e-12
    if (scores < -tolerance).any() or (scores > 1.0 + tolerance).any():
        raise UCellContractError("UCell score fell outside [0, 1]")
    return np.clip(scores, 0.0, 1.0)


def score_ucell_expression_block(
    expression_gene_by_cell: np.ndarray,
    *,
    present_gene_indices: Iterable[int],
    signature_gene_count: int,
    max_rank: int,
) -> np.ndarray:
    """Rank and score one complete signature for a cell block."""

    ranks = rank_expression_ucell(expression_gene_by_cell, max_rank=max_rank)
    return score_ucell_from_capped_ranks(
        ranks,
        present_gene_indices=present_gene_indices,
        signature_gene_count=signature_gene_count,
        max_rank=max_rank,
    )


def build_ucell_pathway_contract(
    membership: pd.DataFrame,
    *,
    protein_gene_ids: Iterable[str],
    max_rank: int,
) -> UCellPathwayContract:
    """Build a complete-signature contract without shortening any pathway."""

    required = {"pathway_id", "gene_id"}
    if missing := sorted(required - set(membership.columns)):
        raise UCellContractError(f"membership lacks columns: {missing}")
    pairs = membership.loc[:, ["pathway_id", "gene_id"]].copy()
    pairs["pathway_id"] = pairs.pathway_id.astype(str)
    pairs["gene_id"] = pairs.gene_id.astype(str)
    pairs = pairs.drop_duplicates().sort_values(
        ["pathway_id", "gene_id"], kind="mergesort"
    )
    proteins = tuple(map(str, protein_gene_ids))
    if len(proteins) != len(set(proteins)):
        raise UCellContractError("protein rank universe contains duplicate gene IDs")
    protein_index = {gene_id: index for index, gene_id in enumerate(proteins)}

    availability_rows: list[dict[str, object]] = []
    available_ids: list[str] = []
    signature_counts: list[int] = []
    present_counts: list[int] = []
    missing_counts: list[int] = []
    matrix_rows: list[int] = []
    matrix_columns: list[int] = []
    for pathway_id, group in pairs.groupby("pathway_id", sort=True, observed=True):
        complete_genes = tuple(group.gene_id.astype(str))
        present = [protein_index[gene] for gene in complete_genes if gene in protein_index]
        domain = evaluate_ucell_signature_domain(
            signature_gene_count=len(complete_genes),
            present_gene_count=len(present),
            max_rank=max_rank,
        )
        availability_rows.append(
            {
                "pathway_id": str(pathway_id),
                "signature_gene_count": domain.signature_gene_count,
                "present_gene_count": domain.present_gene_count,
                "missing_gene_count": domain.missing_gene_count,
                "max_rank": domain.max_rank,
                "ucell_available": domain.available,
                "unavailable_reason": domain.unavailable_reason,
                "complete_signature_length_used_in_denominator": True,
                "matrix_missing_member_policy": UCELL_MISSING_MEMBER_POLICY,
                "matrix_missing_member_is_expression_data_imputation": False,
                "smoothing_used": False,
            }
        )
        if domain.available:
            row_index = len(available_ids)
            available_ids.append(str(pathway_id))
            signature_counts.append(domain.signature_gene_count)
            present_counts.append(domain.present_gene_count)
            missing_counts.append(domain.missing_gene_count)
            matrix_rows.extend([row_index] * len(present))
            matrix_columns.extend(present)

    availability = pd.DataFrame(availability_rows)
    matrix = sparse.csr_matrix(
        (
            np.ones(len(matrix_rows), dtype=np.float64),
            (
                np.asarray(matrix_rows, dtype=np.int64),
                np.asarray(matrix_columns, dtype=np.int64),
            ),
        ),
        shape=(len(available_ids), len(proteins)),
        dtype=np.float64,
    )
    return UCellPathwayContract(
        availability=availability,
        available_pathway_ids=tuple(available_ids),
        signature_matrix=matrix,
        signature_gene_counts=np.asarray(signature_counts, dtype=np.int64),
        present_gene_counts=np.asarray(present_counts, dtype=np.int64),
        missing_gene_counts=np.asarray(missing_counts, dtype=np.int64),
    )


def score_ucell_pathway_block(
    capped_ranks_gene_by_cell: np.ndarray,
    *,
    signature_matrix: sparse.csr_matrix,
    signature_gene_counts: np.ndarray,
    missing_gene_counts: np.ndarray,
    max_rank: int,
) -> np.ndarray:
    """Vectorized UCell scores for available pathways by cells."""

    ranks = np.asarray(capped_ranks_gene_by_cell, dtype=np.float64)
    if ranks.ndim != 2:
        raise UCellContractError("capped ranks must be a 2-D gene-by-cell array")
    if signature_matrix.shape[1] != ranks.shape[0]:
        raise UCellContractError("signature matrix and rank universe disagree")
    n = np.asarray(signature_gene_counts, dtype=np.float64)
    missing = np.asarray(missing_gene_counts, dtype=np.float64)
    if n.ndim != 1 or missing.ndim != 1 or len(n) != signature_matrix.shape[0]:
        raise UCellContractError("signature count vectors disagree with matrix rows")
    if len(missing) != len(n):
        raise UCellContractError("missing count vector disagrees with signatures")
    if (n <= 0).any() or (n >= int(max_rank)).any():
        raise UCellContractError("numeric signature is outside the declared domain")

    rank_sum = np.asarray(signature_matrix @ ranks, dtype=np.float64)
    rank_sum += missing[:, None] * float(max_rank)
    rank_sum_min = n * (n + 1.0) / 2.0
    denominator = n * float(max_rank) - rank_sum_min
    scores = 1.0 - (rank_sum - rank_sum_min[:, None]) / denominator[:, None]
    tolerance = 1e-12
    if (scores < -tolerance).any() or (scores > 1.0 + tolerance).any():
        raise UCellContractError("UCell pathway block fell outside [0, 1]")
    return np.clip(scores, 0.0, 1.0)
