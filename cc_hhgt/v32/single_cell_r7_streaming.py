"""Bounded-memory primitives for fresh V3.2 r7 single-cell computation.

The module deliberately stores only sufficient statistics.  A cell block may
temporarily hold ``protein_genes x chunk_cells`` ranks and
``available_pathways x chunk_cells`` scores, but no caller is allowed to
materialize or persist a complete ``cells x 2,135 pathways`` matrix.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata, t as student_t


INSUFFICIENT_DONOR_REPLICATION = "INSUFFICIENT_DONOR_REPLICATION"
NO_VARIABLE_LNCRNA = "NO_VARIABLE_LNCRNA_AT_DONOR_LEVEL"
NO_VARIABLE_PATHWAY = "NO_VARIABLE_PATHWAY_AT_DONOR_LEVEL"
ASSOCIATION_AVAILABLE = "AVAILABLE"
TYPED_UNAVAILABLE = "TYPED_UNAVAILABLE"
BIOLOGICAL_UNIT = "DONOR"
BH_CONSERVATIVE_METHOD = (
    "BH_ON_RETAINED_LOW_P_PREFIX_WITH_GLOBAL_TEST_DENOMINATOR_"
    "CONSERVATIVE_UPPER_BOUND"
)
RUNTIME_BASELINE_BUDGET_BYTES = 352 * 1024**2
DUCKDB_EXTERNAL_SORT_BUDGET_BYTES = 64 * 1024**2
PARQUET_BATCH_BUDGET_BYTES = 16 * 1024**2

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IMMUNE = frozenset(
    {
        "b_cell",
        "b",
        "t_cell",
        "t",
        "nk_cell",
        "myeloid",
        "plasma_cell",
        "plasma",
        "lymphocyte",
        "lymphoid",
        "hematopoietic_progenitor",
        "hematopoietic_other",
    }
)
_STROMAL = frozenset(
    {"fibroblast_stromal", "fibroblast", "endothelial", "endothelium"}
)
_MALIGNANT = frozenset({"malignant", "malignant_candidate"})


class StreamingContractError(RuntimeError):
    """Raised when a streaming, lineage, or publication invariant fails."""


@dataclass(frozen=True)
class AssociationResult:
    evidence: pd.DataFrame
    tested_universe: pd.DataFrame
    availability: dict[str, Any]


@dataclass(frozen=True)
class AssociationEvidenceChunk:
    """One bounded numeric evidence block emitted before BH adjustment."""

    lncrna_rows: np.ndarray
    pathway_rows: np.ndarray
    spearman_rho: np.ndarray
    nominal_p: np.ndarray


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def classify_compartment(cell_type_major: object) -> str:
    """Map only explicitly supported cell types to three formal compartments."""

    value = str(cell_type_major).strip().lower()
    if value in _MALIGNANT:
        return "malignant"
    if value in _IMMUNE:
        return "immune"
    if value in _STROMAL:
        return "stromal"
    return "other_unresolved"


def assert_fresh_r7_source(path: Path, *, role: str) -> None:
    """Reject historical derived products as inputs to a fresh r7 run."""

    normalized = str(path).replace("\\", "/").lower()
    raw_roles = {"raw_h5", "metadata"}
    authority_roles = {"annotation", "membership", "annotation_provenance"}
    if role in raw_roles and "/processed/sc_tool_input/" in normalized:
        return
    if role in authority_roles and (
        "/single_cell_cell_level_r7_portable_supersession_20260829/authority/"
        in normalized
    ):
        return
    derived_tokens = (
        "/results/sc_trajectory/",
        "/cell_level_ucell/",
        "prediction",
        "ranking",
        "checkpoint",
        "ucell_r5",
        "ucell_r6",
        "_r5/",
        "_r6/",
    )
    if any(token in normalized for token in derived_tokens):
        raise StreamingContractError(
            f"historical derived source cannot be relabelled fresh: {path}"
        )
    raise StreamingContractError(f"source is outside the pinned r7 contract: {role} {path}")


def read_csc_column_block(
    *,
    data: Any,
    indices: Any,
    indptr: Any,
    shape: tuple[int, int],
    start: int,
    stop: int,
) -> sparse.csc_matrix:
    """Read one contiguous CSC cell block without reading the complete payload."""

    rows, columns = map(int, shape)
    begin = int(start)
    end = int(stop)
    if rows <= 0 or columns <= 0 or not (0 <= begin < end <= columns):
        raise StreamingContractError(
            f"invalid CSC block [{begin}, {end}) for shape {(rows, columns)}"
        )
    pointers = np.asarray(indptr[begin : end + 1], dtype=np.int64)
    if len(pointers) != end - begin + 1 or (np.diff(pointers) < 0).any():
        raise StreamingContractError("CSC indptr block is invalid")
    offset = int(pointers[0])
    limit = int(pointers[-1])
    values = np.asarray(data[offset:limit])
    row_indices = np.asarray(indices[offset:limit], dtype=np.int64)
    if len(values) != len(row_indices):
        raise StreamingContractError("CSC data/indices block lengths disagree")
    local = pointers - offset
    matrix = sparse.csc_matrix(
        (values, row_indices, local), shape=(rows, end - begin)
    )
    if matrix.nnz and (
        not np.isfinite(matrix.data).all() or (matrix.data < 0).any()
    ):
        raise StreamingContractError("expression block contains invalid values")
    return matrix


def estimate_streaming_resources(
    *,
    cells: int,
    features: int,
    protein_genes: int,
    lncrnas: int,
    pathways: int,
    donor_celltype_groups: int,
    nnz: int,
    chunk_cells: int,
    association_pathway_block: int = 32,
    duckdb_external_sort_budget_bytes: int = DUCKDB_EXTERNAL_SORT_BUDGET_BYTES,
) -> dict[str, Any]:
    """Return a conservative RAM/disk estimate for a sufficient-statistics run."""

    values = {
        "cells": cells,
        "features": features,
        "protein_genes": protein_genes,
        "lncrnas": lncrnas,
        "pathways": pathways,
        "donor_celltype_groups": donor_celltype_groups,
        "nnz": nnz,
        "chunk_cells": chunk_cells,
    }
    if any(int(value) <= 0 for value in values.values()):
        raise StreamingContractError(f"resource dimensions must be positive: {values}")
    block = int(chunk_cells)
    average_nnz_per_cell = int(np.ceil(int(nnz) / int(cells)))
    sparse_block_bytes = average_nnz_per_cell * block * 12 + (block + 1) * 8
    expression_and_rank_bytes = int(protein_genes) * block * 8 * 2
    pathway_block_bytes = int(pathways) * block * 8
    accumulator_bytes = (
        int(lncrnas) * int(donor_celltype_groups) * (8 + 8)
        + int(pathways) * int(donor_celltype_groups) * 8
        + int(donor_celltype_groups) * 8
    )
    association_elements = int(lncrnas) * min(
        int(pathways), int(association_pathway_block)
    )
    # One rho matrix plus bounded candidate indices/statistics.  The previous
    # estimate counted rho alone even though denominator/statistic/p-value
    # temporaries and pandas evidence rows coexisted.
    association_block_bytes = association_elements * 49
    ingest_phase_bytes = (
        sparse_block_bytes
        + expression_and_rank_bytes
        + pathway_block_bytes
        + accumulator_bytes
    )
    association_phase_bytes = (
        accumulator_bytes * 2 + association_block_bytes
    )
    duckdb_budget = int(duckdb_external_sort_budget_bytes)
    if duckdb_budget < DUCKDB_EXTERNAL_SORT_BUDGET_BYTES:
        raise StreamingContractError(
            "DuckDB external-sort budget cannot be below the frozen 64 MiB baseline"
        )
    external_sort_phase_bytes = (
        accumulator_bytes
        + duckdb_budget
        + PARQUET_BATCH_BUDGET_BYTES
    )
    subtotal = max(
        ingest_phase_bytes,
        association_phase_bytes,
        external_sort_phase_bytes,
    )
    estimated_peak = int(
        np.ceil(subtotal * 1.10 + RUNTIME_BASELINE_BUDGET_BYTES)
    )
    checkpoint_bytes = int(accumulator_bytes + 8 * 1024**2)
    return {
        **{key: int(value) for key, value in values.items()},
        "forbidden_cell_by_pathway_elements": int(cells) * int(pathways),
        "forbidden_dense_float32_bytes": int(cells) * int(pathways) * 4,
        "largest_dense_block_elements": max(
            int(protein_genes) * block, int(pathways) * block
        ),
        "dense_block_cell_bound": block,
        "estimated_sparse_input_block_bytes": int(sparse_block_bytes),
        "ingest_phase_bytes": int(ingest_phase_bytes),
        "association_phase_bytes": int(association_phase_bytes),
        "external_sort_phase_bytes": int(external_sort_phase_bytes),
        "association_workspace_bytes": int(association_block_bytes),
        "algorithmic_subtotal_bytes": int(subtotal),
        "runtime_baseline_budget_bytes": RUNTIME_BASELINE_BUDGET_BYTES,
        "duckdb_external_sort_budget_bytes": duckdb_budget,
        "parquet_batch_budget_bytes": PARQUET_BATCH_BUDGET_BYTES,
        "estimate_includes_interpreter_runtime_baseline": True,
        "estimated_peak_ram_bytes": estimated_peak,
        "checkpoint_bytes": checkpoint_bytes,
        "cell_level_pathway_matrix_persisted": False,
        "cell_level_pathway_rows_persisted": 0,
        "resume_state_is_sufficient_statistics_only": True,
        "association_evidence_accumulated_in_python": False,
        "association_bh_external_spill_enabled": True,
    }


class StreamingGroupAccumulator:
    """Online lncRNA/pathway sufficient statistics by donor-celltype group."""

    def __init__(self, *, lncrna_count: int, pathway_count: int, group_count: int) -> None:
        if min(lncrna_count, pathway_count, group_count) <= 0:
            raise StreamingContractError("accumulator dimensions must be positive")
        self.cell_counts = np.zeros(group_count, dtype=np.int64)
        self.lncrna_detect_counts = np.zeros(
            (lncrna_count, group_count), dtype=np.int64
        )
        self.lncrna_expression_sums = np.zeros(
            (lncrna_count, group_count), dtype=np.float64
        )
        self.pathway_activity_sums = np.zeros(
            (pathway_count, group_count), dtype=np.float64
        )

    def update(
        self,
        lncrna_expression: sparse.spmatrix,
        pathway_activity: np.ndarray,
        group_codes: np.ndarray,
    ) -> None:
        lnc = sparse.csc_matrix(lncrna_expression)
        scores = np.asarray(pathway_activity, dtype=np.float64)
        codes = np.asarray(group_codes, dtype=np.int64)
        cells = lnc.shape[1]
        if scores.ndim != 2 or scores.shape[1] != cells or codes.shape != (cells,):
            raise StreamingContractError("block cells disagree across accumulator inputs")
        if lnc.shape[0] != self.lncrna_expression_sums.shape[0]:
            raise StreamingContractError("lncRNA block row count drift")
        if scores.shape[0] != self.pathway_activity_sums.shape[0]:
            raise StreamingContractError("pathway block row count drift")
        if cells == 0:
            return
        if codes.min() < 0 or codes.max() >= len(self.cell_counts):
            raise StreamingContractError("group code outside accumulator")
        if lnc.nnz and (not np.isfinite(lnc.data).all() or (lnc.data < 0).any()):
            raise StreamingContractError("lncRNA block contains invalid expression")
        if not np.isfinite(scores).all():
            raise StreamingContractError("pathway block contains non-finite activity")
        for code in np.unique(codes):
            mask = codes == code
            count = int(mask.sum())
            block = lnc[:, mask]
            detected = block.copy()
            detected.eliminate_zeros()
            self.cell_counts[code] += count
            self.lncrna_detect_counts[:, code] += np.asarray(
                detected.getnnz(axis=1), dtype=np.int64
            )
            self.lncrna_expression_sums[:, code] += np.asarray(
                block.sum(axis=1), dtype=np.float64
            ).reshape(-1)
            self.pathway_activity_sums[:, code] += scores[:, mask].sum(axis=1)

    def finalize(self) -> dict[str, np.ndarray]:
        denominator = self.cell_counts.astype(np.float64)
        valid = denominator > 0
        lnc_means = np.full_like(self.lncrna_expression_sums, np.nan)
        pathway_means = np.full_like(self.pathway_activity_sums, np.nan)
        detect_rates = np.full_like(self.lncrna_expression_sums, np.nan)
        lnc_means[:, valid] = self.lncrna_expression_sums[:, valid] / denominator[valid]
        pathway_means[:, valid] = (
            self.pathway_activity_sums[:, valid] / denominator[valid]
        )
        detect_rates[:, valid] = self.lncrna_detect_counts[:, valid] / denominator[valid]
        return {
            "cell_counts": self.cell_counts.copy(),
            "lncrna_detect_counts": self.lncrna_detect_counts.copy(),
            "lncrna_expression_sums": self.lncrna_expression_sums.copy(),
            "lncrna_expression_means": lnc_means,
            "lncrna_detect_rates": detect_rates,
            "pathway_activity_sums": self.pathway_activity_sums.copy(),
            "pathway_activity_means": pathway_means,
        }

    def state_arrays(self) -> dict[str, np.ndarray]:
        return {
            "cell_counts": self.cell_counts,
            "lncrna_detect_counts": self.lncrna_detect_counts,
            "lncrna_expression_sums": self.lncrna_expression_sums,
            "pathway_activity_sums": self.pathway_activity_sums,
        }

    def restore(self, arrays: Mapping[str, np.ndarray]) -> None:
        for name, target in self.state_arrays().items():
            if name not in arrays:
                raise StreamingContractError(f"checkpoint lacks accumulator array: {name}")
            source = np.asarray(arrays[name])
            if source.shape != target.shape or source.dtype != target.dtype:
                raise StreamingContractError(f"checkpoint array schema drift: {name}")
            target[...] = source


def _empty_evidence() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "compartment",
            "lncrna_id",
            "pathway_id",
            "n_donors",
            "spearman_rho",
            "nominal_p",
            "bh_q_global_tests",
            "fdr_0_10_pass",
            "multiple_testing_adjustment",
            "bh_q_is_conservative_upper_bound",
            "biological_unit",
            "cell_as_independent_replicate",
            "evidence_scope",
        ]
    )


def _bh_with_total_tests(pvalues: np.ndarray, total_tests: int) -> np.ndarray:
    if len(pvalues) == 0:
        return np.asarray([], dtype=np.float64)
    order = np.argsort(pvalues, kind="mergesort")
    ranked = np.asarray(pvalues, dtype=np.float64)[order]
    adjusted = ranked * float(total_tests) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def compute_donor_associations(
    lncrna_means: pd.DataFrame,
    pathway_means: pd.DataFrame,
    *,
    compartment: str,
    min_donors: int = 5,
    min_abs_rho: float = 0.5,
    max_nominal_p: float = 0.05,
    pathway_block: int = 128,
) -> AssociationResult:
    """Screen lncRNA-pathway associations across donor-level context means."""

    if not lncrna_means.index.is_unique or not pathway_means.index.is_unique:
        raise StreamingContractError("donor index must be unique")
    if list(map(str, lncrna_means.index)) != list(map(str, pathway_means.index)):
        raise StreamingContractError("lncRNA/pathway donor order disagrees")
    if not lncrna_means.columns.is_unique or not pathway_means.columns.is_unique:
        raise StreamingContractError("lncRNA/pathway identifiers must be unique")
    n_donors = len(lncrna_means)
    availability = {
        "compartment": str(compartment),
        "biological_unit": BIOLOGICAL_UNIT,
        "cell_as_independent_replicate": False,
        "donor_count": int(n_donors),
        "minimum_donors": int(min_donors),
        "status": ASSOCIATION_AVAILABLE,
        "unavailable_reason": None,
    }
    tested = pd.DataFrame(
        {
            "compartment": str(compartment),
            "lncrna_id": lncrna_means.columns.astype(str),
            "donor_count": int(n_donors),
            "biological_unit": BIOLOGICAL_UNIT,
        }
    )
    if n_donors < int(min_donors):
        availability.update(
            status=TYPED_UNAVAILABLE,
            unavailable_reason=INSUFFICIENT_DONOR_REPLICATION,
            tested_pair_count=0,
        )
        tested["test_status"] = TYPED_UNAVAILABLE
        tested["unavailable_reason"] = INSUFFICIENT_DONOR_REPLICATION
        return AssociationResult(_empty_evidence(), tested, availability)

    x = lncrna_means.to_numpy(dtype=np.float64)
    y = pathway_means.to_numpy(dtype=np.float64)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise StreamingContractError("donor-level means contain non-finite values")
    variable_lnc = np.ptp(x, axis=0) > 0
    variable_pathway = np.ptp(y, axis=0) > 0
    tested["test_status"] = np.where(variable_lnc, "TESTED", TYPED_UNAVAILABLE)
    tested["unavailable_reason"] = np.where(variable_lnc, None, NO_VARIABLE_LNCRNA)
    if not variable_lnc.any() or not variable_pathway.any():
        reason = NO_VARIABLE_LNCRNA if not variable_lnc.any() else NO_VARIABLE_PATHWAY
        availability.update(
            status=TYPED_UNAVAILABLE,
            unavailable_reason=reason,
            tested_pair_count=0,
        )
        return AssociationResult(_empty_evidence(), tested, availability)

    x_ids = lncrna_means.columns.astype(str).to_numpy()[variable_lnc]
    y_ids = pathway_means.columns.astype(str).to_numpy()[variable_pathway]
    x_rank = rankdata(x[:, variable_lnc], method="average", axis=0)
    y_rank = rankdata(y[:, variable_pathway], method="average", axis=0)
    x_centered = x_rank - x_rank.mean(axis=0, keepdims=True)
    y_centered = y_rank - y_rank.mean(axis=0, keepdims=True)
    x_norm = np.sqrt((x_centered**2).sum(axis=0))
    y_norm = np.sqrt((y_centered**2).sum(axis=0))
    total_tests = int(len(x_ids) * len(y_ids))
    rows: list[pd.DataFrame] = []
    for start in range(0, len(y_ids), int(pathway_block)):
        stop = min(start + int(pathway_block), len(y_ids))
        rho = (x_centered.T @ y_centered[:, start:stop]) / (
            x_norm[:, None] * y_norm[None, start:stop]
        )
        rho = np.clip(rho, -1.0, 1.0)
        denominator = np.maximum(1.0 - rho**2, np.finfo(np.float64).eps)
        statistic = np.abs(rho) * np.sqrt((n_donors - 2.0) / denominator)
        statistic[np.isclose(np.abs(rho), 1.0, rtol=0.0, atol=4 * np.finfo(float).eps)] = np.inf
        pvalue = 2.0 * student_t.sf(statistic, df=n_donors - 2)
        selected = (np.abs(rho) >= float(min_abs_rho)) & (
            pvalue <= float(max_nominal_p)
        )
        lnc_rows, pathway_rows = np.nonzero(selected)
        if len(lnc_rows):
            rows.append(
                pd.DataFrame(
                    {
                        "compartment": str(compartment),
                        "lncrna_id": x_ids[lnc_rows],
                        "pathway_id": y_ids[start:stop][pathway_rows],
                        "n_donors": int(n_donors),
                        "spearman_rho": rho[lnc_rows, pathway_rows],
                        "nominal_p": pvalue[lnc_rows, pathway_rows],
                    }
                )
            )
    evidence = pd.concat(rows, ignore_index=True) if rows else _empty_evidence()
    if len(evidence):
        evidence["bh_q_global_tests"] = _bh_with_total_tests(
            evidence.nominal_p.to_numpy(dtype=np.float64), total_tests
        )
        evidence["fdr_0_10_pass"] = evidence.bh_q_global_tests.le(0.10)
        evidence["multiple_testing_adjustment"] = BH_CONSERVATIVE_METHOD
        evidence["bh_q_is_conservative_upper_bound"] = True
        evidence["biological_unit"] = BIOLOGICAL_UNIT
        evidence["cell_as_independent_replicate"] = False
        evidence["evidence_scope"] = "EXPLORATORY_DONOR_LEVEL_SPEARMAN_SCREEN"
        evidence = evidence.sort_values(
            ["bh_q_global_tests", "nominal_p", "lncrna_id", "pathway_id"],
            kind="mergesort",
        ).reset_index(drop=True)
    availability.update(
        variable_lncrnas=int(variable_lnc.sum()),
        variable_pathways=int(variable_pathway.sum()),
        tested_pair_count=total_tests,
        retained_evidence_count=int(len(evidence)),
        multiple_testing_denominator=total_tests,
        multiple_testing_adjustment=BH_CONSERVATIVE_METHOD,
        bh_q_is_conservative_upper_bound=True,
        inferential_scope="EXPLORATORY_NOT_CAUSAL",
    )
    return AssociationResult(evidence, tested, availability)


def stream_donor_association_chunks(
    lncrna_means: pd.DataFrame,
    pathway_means: pd.DataFrame,
    *,
    compartment: str,
    evidence_sink: Callable[[AssociationEvidenceChunk], None],
    min_donors: int = 5,
    min_abs_rho: float = 0.5,
    max_nominal_p: float = 0.05,
    pathway_block: int = 32,
) -> dict[str, Any]:
    """Emit selected donor-level associations without retaining string rows.

    The sink receives at most one pathway block at a time.  Multiple-testing
    adjustment is deliberately deferred to a bounded external-sort stage so a
    cancer with many lncRNAs cannot accumulate millions of pandas object rows
    in the Python process.
    """

    if not callable(evidence_sink):
        raise StreamingContractError("association evidence sink is not callable")
    if int(pathway_block) <= 0:
        raise StreamingContractError("association pathway block must be positive")
    if not lncrna_means.index.is_unique or not pathway_means.index.is_unique:
        raise StreamingContractError("donor index must be unique")
    if list(map(str, lncrna_means.index)) != list(map(str, pathway_means.index)):
        raise StreamingContractError("lncRNA/pathway donor order disagrees")
    if not lncrna_means.columns.is_unique or not pathway_means.columns.is_unique:
        raise StreamingContractError("lncRNA/pathway identifiers must be unique")

    n_donors = len(lncrna_means)
    availability = {
        "compartment": str(compartment),
        "biological_unit": BIOLOGICAL_UNIT,
        "cell_as_independent_replicate": False,
        "donor_count": int(n_donors),
        "minimum_donors": int(min_donors),
        "status": ASSOCIATION_AVAILABLE,
        "unavailable_reason": None,
    }
    if n_donors < int(min_donors):
        availability.update(
            status=TYPED_UNAVAILABLE,
            unavailable_reason=INSUFFICIENT_DONOR_REPLICATION,
            tested_pair_count=0,
            retained_evidence_count=0,
        )
        return availability

    x = lncrna_means.to_numpy(dtype=np.float64, copy=False)
    y = pathway_means.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise StreamingContractError("donor-level means contain non-finite values")
    variable_lnc = np.ptp(x, axis=0) > 0
    variable_pathway = np.ptp(y, axis=0) > 0
    if not variable_lnc.any() or not variable_pathway.any():
        reason = NO_VARIABLE_LNCRNA if not variable_lnc.any() else NO_VARIABLE_PATHWAY
        availability.update(
            status=TYPED_UNAVAILABLE,
            unavailable_reason=reason,
            tested_pair_count=0,
            retained_evidence_count=0,
        )
        return availability

    x_original_rows = np.flatnonzero(variable_lnc).astype(np.int32, copy=False)
    y_original_rows = np.flatnonzero(variable_pathway).astype(np.int32, copy=False)
    x_rank = rankdata(x[:, variable_lnc], method="average", axis=0)
    y_rank = rankdata(y[:, variable_pathway], method="average", axis=0)
    x_rank -= x_rank.mean(axis=0, keepdims=True)
    y_rank -= y_rank.mean(axis=0, keepdims=True)
    x_norm = np.sqrt((x_rank**2).sum(axis=0))
    y_norm = np.sqrt((y_rank**2).sum(axis=0))
    total_tests = int(len(x_original_rows) * len(y_original_rows))
    retained = 0

    for start in range(0, len(y_original_rows), int(pathway_block)):
        stop = min(start + int(pathway_block), len(y_original_rows))
        rho = x_rank.T @ y_rank[:, start:stop]
        rho /= x_norm[:, None]
        rho /= y_norm[None, start:stop]
        np.clip(rho, -1.0, 1.0, out=rho)

        candidate_lnc, candidate_pathway = np.nonzero(
            np.abs(rho) >= float(min_abs_rho)
        )
        if len(candidate_lnc):
            selected_rho = rho[candidate_lnc, candidate_pathway]
            denominator = np.maximum(
                1.0 - selected_rho**2, np.finfo(np.float64).eps
            )
            statistic = np.abs(selected_rho) * np.sqrt(
                (n_donors - 2.0) / denominator
            )
            statistic[
                np.isclose(
                    np.abs(selected_rho),
                    1.0,
                    rtol=0.0,
                    atol=4 * np.finfo(float).eps,
                )
            ] = np.inf
            pvalue = 2.0 * student_t.sf(statistic, df=n_donors - 2)
            keep = pvalue <= float(max_nominal_p)
            if keep.any():
                chunk = AssociationEvidenceChunk(
                    lncrna_rows=x_original_rows[candidate_lnc[keep]],
                    pathway_rows=y_original_rows[start:stop][candidate_pathway[keep]],
                    spearman_rho=np.asarray(selected_rho[keep], dtype=np.float64),
                    nominal_p=np.asarray(pvalue[keep], dtype=np.float64),
                )
                evidence_sink(chunk)
                retained += int(keep.sum())
            del denominator, statistic, pvalue, keep, selected_rho
        del candidate_lnc, candidate_pathway, rho

    availability.update(
        variable_lncrnas=int(variable_lnc.sum()),
        variable_pathways=int(variable_pathway.sum()),
        tested_pair_count=total_tests,
        retained_evidence_count=retained,
        multiple_testing_denominator=total_tests,
        multiple_testing_adjustment=BH_CONSERVATIVE_METHOD,
        bh_q_is_conservative_upper_bound=True,
        inferential_scope="EXPLORATORY_NOT_CAUSAL",
    )
    return availability


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise StreamingContractError(f"unsafe checkpoint temporary exists: {temporary}")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_checkpoint(
    root: Path,
    *,
    contract_sha256: str,
    next_cell: int,
    total_cells: int,
    arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Atomically replace a resumable sufficient-statistics checkpoint."""

    contract = str(contract_sha256).lower()
    if not _SHA256.fullmatch(contract):
        raise StreamingContractError("checkpoint contract SHA is malformed")
    if not (0 <= int(next_cell) <= int(total_cells)):
        raise StreamingContractError("checkpoint next_cell is outside total_cells")
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise StreamingContractError(f"checkpoint root is unsafe: {root}")
    npz_path = root / "CHECKPOINT.npz"
    temporary = root / f".CHECKPOINT.npz.{os.getpid()}.tmp"
    if temporary.exists() or temporary.is_symlink():
        raise StreamingContractError(f"checkpoint temporary already exists: {temporary}")
    normalized: dict[str, np.ndarray] = {}
    for name, value in arrays.items():
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", str(name)):
            raise StreamingContractError(f"unsafe checkpoint array name: {name}")
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise StreamingContractError(f"object checkpoint array is forbidden: {name}")
        normalized[str(name)] = array
    with temporary.open("xb") as stream:
        np.savez(stream, **normalized)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, npz_path)
    state = {
        "format": "CC_HHGT_V3_2_SINGLE_CELL_R7_STREAM_CHECKPOINT_V1",
        "contract_sha256": contract,
        "next_cell": int(next_cell),
        "total_cells": int(total_cells),
        "checkpoint_npz": npz_path.name,
        "checkpoint_npz_sha256": _sha256_file(npz_path),
        "arrays": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in sorted(normalized.items())
        },
        "cell_level_pathway_matrix_persisted": False,
        "historical_derived_results_used": False,
    }
    _atomic_json(root / "CHECKPOINT_STATE.json", state)
    return state


def load_checkpoint(
    root: Path, *, expected_contract_sha256: str
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    state_path = root / "CHECKPOINT_STATE.json"
    if state_path.is_symlink() or not state_path.is_file():
        raise StreamingContractError("checkpoint state is absent or unsafe")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("format") != "CC_HHGT_V3_2_SINGLE_CELL_R7_STREAM_CHECKPOINT_V1":
        raise StreamingContractError("checkpoint format drift")
    if state.get("contract_sha256") != str(expected_contract_sha256).lower():
        raise StreamingContractError("checkpoint contract SHA mismatch")
    npz_path = root / str(state.get("checkpoint_npz", ""))
    if npz_path.parent != root or npz_path.is_symlink() or not npz_path.is_file():
        raise StreamingContractError("checkpoint payload is absent or unsafe")
    observed = _sha256_file(npz_path)
    if observed != state.get("checkpoint_npz_sha256"):
        raise StreamingContractError("checkpoint payload SHA mismatch")
    declared = state.get("arrays", {})
    arrays: dict[str, np.ndarray] = {}
    with np.load(npz_path, allow_pickle=False) as payload:
        if set(payload.files) != set(declared):
            raise StreamingContractError("checkpoint array set drift")
        for name in payload.files:
            value = np.asarray(payload[name])
            record = declared[name]
            if list(value.shape) != record.get("shape") or str(value.dtype) != record.get(
                "dtype"
            ):
                raise StreamingContractError(f"checkpoint array schema drift: {name}")
            arrays[name] = value
    next_cell = int(state.get("next_cell", -1))
    total_cells = int(state.get("total_cells", -1))
    if not (0 <= next_cell <= total_cells):
        raise StreamingContractError("checkpoint position is invalid")
    return state, arrays


def atomic_publish_directory(staging: Path, final: Path) -> None:
    """Publish one completed cancer directory with a same-filesystem rename."""

    if staging.is_symlink() or not staging.is_dir():
        raise StreamingContractError(f"staging directory is absent or unsafe: {staging}")
    if final.exists() or final.is_symlink():
        raise StreamingContractError(f"final cancer result already exists: {final}")
    if staging.parent.resolve() != final.parent.resolve():
        raise StreamingContractError("atomic publish requires a shared parent directory")
    os.rename(staging, final)


def select_smallest_formal_cancer(preflight: Mapping[str, Any]) -> dict[str, Any]:
    rows = list(preflight.get("formal_cancers", []))
    if not rows:
        raise StreamingContractError("preflight contains no formal cancers")
    candidates = []
    for row in rows:
        cancer = str(row.get("cancer_id", "")).upper()
        cells = int(row.get("matrix_header", {}).get("cells", 0))
        if not cancer or cells <= 0:
            raise StreamingContractError("formal cancer row lacks cancer/cell count")
        candidates.append((cells, cancer))
    cells, cancer = min(candidates)
    return {"cancer_id": cancer, "cells": cells}
