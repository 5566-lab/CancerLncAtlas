"""Contracts for rebuilding SC-HHGT inputs from frozen metacell membership.

This module does not fit metacells or LASSO models.  It aligns the already
materialized cell-to-metacell assignment with freshly read measurements and
emits fold-local, donor-labelled observations for a later episode builder.
"""
from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy import sparse

from .single_cell_metacells import aggregate_measurements


ASSET_FORMAT = "CANCERLNCATLAS_V32_SC_HHGT_GRAPH_ASSETS_V1"
MEMBERSHIP_ORIENTATION = "gene_by_pathway"
UNASSIGNED_PATHWAY_FAMILY_ID = "__UNASSIGNED_PATHWAY_FAMILY__"
ALLOWED_EXTERNAL_NODE_TYPES = frozenset(
    {"lncRNA", "gene", "pathway", "pathway_family"}
)
_PREFIX = re.compile(r"^(?:LNCRNA|LNC|GENE):", flags=re.IGNORECASE)
_ENSEMBL_VERSION = re.compile(r"^(ENS[A-Z]*G\d+)\.\d+$", flags=re.IGNORECASE)
_RELATION_TOKEN = re.compile(r"[^A-Za-z0-9_]+")


def canonical_biological_id(value: object) -> str:
    """Normalize only known namespace wrappers and Ensembl versions."""

    text = _PREFIX.sub("", str(value).strip())
    match = _ENSEMBL_VERSION.fullmatch(text)
    if match:
        text = match.group(1)
    return text.upper()


def align_frozen_membership(
    cell_metadata: pd.DataFrame,
    frozen_membership: pd.DataFrame,
) -> np.ndarray:
    """Return frozen metacell IDs in measurement-row order.

    Cell identity and donor/context labels must match exactly.  No geometry is
    recomputed and no unassigned cell is silently dropped.
    """

    identity = ["cell_id", "donor_id", "cell_type", "state_id"]
    required = set(identity + ["metacell_id"])
    if missing := sorted(set(identity) - set(cell_metadata.columns)):
        raise ValueError(f"CELL_METADATA_MISSING_COLUMNS:{missing}")
    if missing := sorted(required - set(frozen_membership.columns)):
        raise ValueError(f"FROZEN_MEMBERSHIP_MISSING_COLUMNS:{missing}")
    metadata = cell_metadata.loc[:, identity].copy()
    membership = frozen_membership.loc[:, identity + ["metacell_id"]].copy()
    for frame in (metadata, membership):
        for column in identity:
            frame[column] = frame[column].astype(str)
    if metadata.cell_id.duplicated().any() or membership.cell_id.duplicated().any():
        raise ValueError("CELL_ID_NOT_UNIQUE")
    if set(metadata.cell_id) != set(membership.cell_id):
        raise ValueError("FROZEN_MEMBERSHIP_CELL_SET_DRIFT")
    aligned = metadata[["cell_id"]].merge(
        membership,
        on="cell_id",
        how="left",
        sort=False,
        validate="one_to_one",
    )
    for column in identity[1:]:
        if not aligned[column].equals(metadata[column]):
            raise ValueError(f"FROZEN_MEMBERSHIP_{column.upper()}_DRIFT")
    numeric = pd.to_numeric(aligned.metacell_id, errors="coerce").to_numpy(float)
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError("FROZEN_MEMBERSHIP_ID_INVALID")
    assignments = numeric.astype(np.int64)
    if (assignments < -1).any():
        raise ValueError("FROZEN_MEMBERSHIP_ID_INVALID")
    return assignments


def aggregate_frozen_measurements(
    coding: sparse.spmatrix,
    lncrna: sparse.spmatrix,
    assignments: np.ndarray,
    *,
    measurement_scale: str,
    all_gene_library: np.ndarray,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, np.ndarray, str]:
    """Aggregate two aligned matrices using exactly the frozen assignments."""

    labels = np.asarray(assignments, dtype=np.int64)
    if coding.shape[0] != len(labels) or lncrna.shape[0] != len(labels):
        raise ValueError("MEASUREMENT_MEMBERSHIP_ROW_DRIFT")
    coding_mc, metacell_ids, policy = aggregate_measurements(
        sparse.csr_matrix(coding),
        labels,
        scale=measurement_scale,
        total_library=np.asarray(all_gene_library),
    )
    lnc_mc, lnc_ids, lnc_policy = aggregate_measurements(
        sparse.csr_matrix(lncrna),
        labels,
        scale=measurement_scale,
        total_library=np.asarray(all_gene_library),
    )
    if not np.array_equal(metacell_ids, lnc_ids) or policy != lnc_policy:
        raise ValueError("FROZEN_AGGREGATION_ORDER_DRIFT")
    return (
        coding_mc.astype(np.float32),
        lnc_mc.astype(np.float32),
        metacell_ids.astype(np.int64),
        policy,
    )


def lasso_observations(
    frozen_metadata: pd.DataFrame,
    metacell_ids: np.ndarray,
    *,
    source_donors: Iterable[object],
    query_donors: Iterable[object],
) -> pd.DataFrame:
    """Build the row authority used by source-only LASSO episode fitting."""

    required = {
        "metacell_id", "donor_id", "cell_type", "state_id", "n_cells",
    }
    if missing := sorted(required - set(frozen_metadata.columns)):
        raise ValueError(f"FROZEN_METACELL_METADATA_MISSING_COLUMNS:{missing}")
    metadata = frozen_metadata.copy()
    numeric = pd.to_numeric(metadata.metacell_id, errors="coerce")
    if numeric.isna().any() or numeric.duplicated().any():
        raise ValueError("FROZEN_METACELL_ID_INVALID")
    metadata["metacell_id"] = numeric.astype(np.int64)
    metadata = metadata.set_index("metacell_id")
    if not metadata.index.is_unique:
        raise ValueError("FROZEN_METACELL_ID_INVALID")
    requested = np.asarray(metacell_ids, dtype=np.int64)
    if set(metadata.index) != set(requested):
        raise ValueError("FROZEN_METACELL_SET_DRIFT")
    metadata = metadata.loc[requested].reset_index()
    source = {str(value) for value in source_donors}
    query = {str(value) for value in query_donors}
    if not source or not query or source.intersection(query):
        raise ValueError("SC_GRAPH_DONOR_LEAK")
    donors = metadata.donor_id.astype(str)
    unknown = set(donors) - source - query
    if unknown:
        raise ValueError(f"FROZEN_METACELL_UNKNOWN_DONORS:{sorted(unknown)}")
    metadata.insert(0, "metacell_row", np.arange(len(metadata), dtype=np.int64))
    metadata = metadata.rename(columns={"cell_type": "cell_key"})
    metadata["donor_id"] = donors
    metadata["observation_role"] = np.where(donors.isin(source), "source", "query")
    metadata["lasso_input_allowed"] = metadata.observation_role.eq("source")
    columns = [
        "metacell_row", "metacell_id", "donor_id", "cell_key", "state_id",
        "n_cells", "observation_role", "lasso_input_allowed",
    ]
    for optional in ("dataset_id", "cancer_id"):
        if optional in metadata:
            columns.append(optional)
    return metadata.loc[:, columns].reset_index(drop=True)


def gene_by_pathway_membership(
    pathway_by_gene: sparse.spmatrix,
    *,
    gene_count: int,
    pathway_count: int,
) -> sparse.csr_matrix:
    """Validate the UCell signature matrix and transpose it explicitly."""

    matrix = sparse.csr_matrix(pathway_by_gene)
    if matrix.shape != (int(pathway_count), int(gene_count)):
        raise ValueError("PATHWAY_GENE_MEMBERSHIP_SHAPE")
    if matrix.nnz and (
        not np.isfinite(matrix.data).all() or not np.equal(matrix.data, 1).all()
    ):
        raise ValueError("PATHWAY_GENE_MEMBERSHIP_VALUES")
    result = matrix.transpose().tocsr().astype(np.uint8)
    result.sum_duplicates()
    if result.nnz and not np.equal(result.data, 1).all():
        raise ValueError("GENE_PATHWAY_MEMBERSHIP_DUPLICATE_EDGE")
    return result


def normalize_pathway_families(
    hierarchy: pd.DataFrame,
    pathway_ids: Iterable[object],
) -> tuple[sparse.csr_matrix, pd.DataFrame, pd.DataFrame]:
    """Return the complete many-to-many pathway-by-family hierarchy."""

    required = {"pathway_id", "pathway_family_id", "is_representative"}
    if missing := sorted(required - set(hierarchy.columns)):
        raise ValueError(f"PATHWAY_FAMILY_MISSING_COLUMNS:{missing}")
    wanted = tuple(map(str, pathway_ids))
    if len(wanted) != len(set(wanted)):
        raise ValueError("PATHWAY_ID_DUPLICATE")
    frame = hierarchy.copy()
    frame["pathway_id"] = frame.pathway_id.astype(str)
    frame["pathway_family_id"] = frame.pathway_family_id.astype(str)
    frame = frame.loc[frame.pathway_id.isin(set(wanted))].copy()
    if "membership_weight" not in frame:
        frame["membership_weight"] = 1.0
    frame["membership_weight"] = pd.to_numeric(
        frame.membership_weight, errors="raise"
    )
    if (
        not np.isfinite(frame.membership_weight.to_numpy(float)).all()
        or frame.membership_weight.lt(0).any()
    ):
        raise ValueError("PATHWAY_FAMILY_MEMBERSHIP_WEIGHT_INVALID")
    # Match the formal hierarchy materializer: zero-weight records are not
    # edges, while negative/non-finite records invalidate the authority.
    frame = frame.loc[frame.membership_weight.gt(0)].copy()
    frame = frame.sort_values(
        ["pathway_id", "pathway_family_id", "membership_weight"],
        ascending=[True, True, False],
        kind="mergesort",
    ).drop_duplicates(["pathway_id", "pathway_family_id"], keep="first")
    missing_pathways = sorted(set(wanted) - set(frame.pathway_id))
    if missing_pathways:
        if UNASSIGNED_PATHWAY_FAMILY_ID in set(frame.pathway_family_id):
            raise ValueError("PATHWAY_FAMILY_UNASSIGNED_ID_COLLISION")
        frame = pd.concat(
            [
                frame,
                pd.DataFrame(
                    {
                        "pathway_id": missing_pathways,
                        "pathway_family_id": UNASSIGNED_PATHWAY_FAMILY_ID,
                        "membership_weight": np.ones(len(missing_pathways), dtype=float),
                        "is_representative": np.zeros(len(missing_pathways), dtype=bool),
                    }
                ),
            ],
            ignore_index=True,
        )
    family_ids = tuple(sorted(frame.pathway_family_id.unique()))
    family_index = {value: index for index, value in enumerate(family_ids)}
    pathway_index = {value: index for index, value in enumerate(wanted)}
    full = frame[
        ["pathway_id", "pathway_family_id", "membership_weight", "is_representative"]
    ].copy()
    full["pathway_index"] = full.pathway_id.map(pathway_index).astype(np.int64)
    full["family_index"] = full.pathway_family_id.map(family_index).astype(np.int64)
    full = full.sort_values(
        ["pathway_index", "family_index"], kind="mergesort"
    ).reset_index(drop=True)
    membership = sparse.csr_matrix(
        (
            full.membership_weight.to_numpy(np.float32),
            (
                full.pathway_index.to_numpy(np.int64),
                full.family_index.to_numpy(np.int64),
            ),
        ),
        shape=(len(wanted), len(family_ids)),
        dtype=np.float32,
    )
    if np.any(np.asarray(membership.getnnz(axis=1)).ravel() == 0):
        raise ValueError("PATHWAY_FAMILY_ROW_COVERAGE")
    families = pd.DataFrame(
        {
            "node_type": "pathway_family",
            "global_index": np.arange(len(family_ids), dtype=np.int64),
            "canonical_id": family_ids,
        }
    )
    return membership, full, families


def context_feature_scope(
    universe: pd.DataFrame,
    lncrna_ids: Iterable[object],
    pathway_ids: Iterable[object],
) -> pd.DataFrame:
    """Index the exact lncRNA/pathway nodes used by each cell context."""

    required = {"cell_key", "lncrna_id", "pathway_id"}
    if missing := sorted(required - set(universe.columns)):
        raise ValueError(f"UNIVERSE_MISSING_COLUMNS:{missing}")
    indices = {
        "lncRNA": {str(value): i for i, value in enumerate(lncrna_ids)},
        "pathway": {str(value): i for i, value in enumerate(pathway_ids)},
    }
    rows: list[pd.DataFrame] = []
    for node_type, column in (("lncRNA", "lncrna_id"), ("pathway", "pathway_id")):
        selected = universe[["cell_key", column]].copy()
        selected[column] = selected[column].astype(str)
        selected = selected.drop_duplicates().rename(columns={column: "canonical_id"})
        selected["global_index"] = selected.canonical_id.map(indices[node_type])
        if selected.global_index.isna().any():
            raise ValueError(f"UNIVERSE_{node_type.upper()}_IDENTITY_DRIFT")
        selected["global_index"] = selected.global_index.astype(np.int64)
        selected["node_type"] = node_type
        selected = selected.sort_values(
            ["cell_key", "global_index"], kind="mergesort"
        )
        selected["context_index"] = selected.groupby("cell_key", sort=False).cumcount()
        rows.append(selected)
    return pd.concat(rows, ignore_index=True)[
        ["cell_key", "node_type", "canonical_id", "global_index", "context_index"]
    ]


def _relation_name(values: pd.DataFrame) -> pd.Series:
    components = (
        values[["relation_type", "direction", "experiment_family"]]
        .fillna("unspecified")
        .astype(str)
    )
    result = components.agg("__".join, axis=1).map(
        lambda value: _RELATION_TOKEN.sub("_", value).strip("_").lower()
    )
    return "external_" + result


def normalize_external_evidence(
    raw: pd.DataFrame,
    *,
    lncrna_ids: Iterable[object],
    gene_ids: Iterable[object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter experimental non-predicted lncRNA-gene evidence.

    Duplicate source records become one typed edge.  Edge weight is one;
    record and PMID counts remain audit features rather than silently becoming
    an uncalibrated confidence score.
    """

    required = {
        "lncrna_id", "partner_id", "partner_type", "relation_type", "direction",
        "experiment_family", "is_experimental", "is_predicted", "pmid",
        "source_database", "mapping_status",
    }
    if missing := sorted(required - set(raw.columns)):
        raise ValueError(f"EXTERNAL_EVIDENCE_MISSING_COLUMNS:{missing}")
    values = raw.loc[:, sorted(required)].copy()
    if values.is_experimental.isna().any() or values.is_predicted.isna().any():
        raise ValueError("EXTERNAL_EVIDENCE_BOOLEAN_NULL")
    values = values.loc[
        values.mapping_status.astype(str).eq("lncrna_and_partner_mapped")
        & values.is_experimental.astype(bool)
        & ~values.is_predicted.astype(bool)
    ].copy()
    values["relation"] = _relation_name(values)
    schema = pd.DataFrame(
        {
            "source_type": "lncRNA",
            "relation": sorted(values.relation.unique()),
            "target_type": "gene",
        }
    )
    values["source_id"] = values.lncrna_id.map(canonical_biological_id)
    values["target_id"] = values.partner_id.map(canonical_biological_id)
    lnc_index = {
        canonical_biological_id(value): index for index, value in enumerate(lncrna_ids)
    }
    gene_index = {
        canonical_biological_id(value): index for index, value in enumerate(gene_ids)
    }
    values = values.loc[
        values.source_id.isin(lnc_index) & values.target_id.isin(gene_index)
    ].copy()
    keys = [
        "source_id", "target_id", "relation", "relation_type", "direction",
        "experiment_family",
    ]
    if values.empty:
        edges = pd.DataFrame(
            columns=[
                "source_type", "source_id", "source_index", "relation",
                "target_type", "target_id", "target_index", "edge_weight",
                "evidence_record_count", "distinct_pmid_count", "source_databases",
            ]
        )
        return edges, schema
    grouped = values.groupby(keys, observed=True, sort=True, dropna=False).agg(
        evidence_record_count=("lncrna_id", "size"),
        distinct_pmid_count=("pmid", lambda item: item.dropna().astype(str).nunique()),
        source_databases=(
            "source_database",
            lambda item: "|".join(sorted(set(item.dropna().astype(str)))),
        ),
    ).reset_index()
    grouped.insert(0, "source_type", "lncRNA")
    grouped["source_index"] = grouped.source_id.map(lnc_index).astype(np.int64)
    grouped.insert(4, "target_type", "gene")
    grouped["target_index"] = grouped.target_id.map(gene_index).astype(np.int64)
    grouped["edge_weight"] = np.float32(1.0)
    edges = grouped[
        [
            "source_type", "source_id", "source_index", "relation", "target_type",
            "target_id", "target_index", "edge_weight", "evidence_record_count",
            "distinct_pmid_count", "source_databases", "relation_type", "direction",
            "experiment_family",
        ]
    ].sort_values(
        ["relation", "source_index", "target_index"], kind="mergesort"
    ).reset_index(drop=True)
    return edges, schema
