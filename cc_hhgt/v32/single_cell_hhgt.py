"""Donor-isolated, weighted single-cell HHGT components.

The graph contains biological entities plus exactly one ``cell_context`` node.
Continuous metacell-derived node features and signed edge weights both enter the
encoder.  HGTConv supplies relation-aware topology messages while a parallel
weighted path preserves the magnitude and direction of LASSO/external evidence.

This module intentionally does not import ``cc_hhgt.gnn``.  Historical installs
can resolve that name to an unweighted backend, so the single-cell encoder is
self-contained and its semantics do not depend on ``PYTHONPATH`` ordering.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv


NodeType = str
EdgeType = tuple[str, str, str]

NODE_TYPES: tuple[NodeType, ...] = (
    "lncRNA",
    "gene",
    "pathway",
    "pathway_family",
    "cell_context",
)

LASSO_EDGE_TYPES: tuple[EdgeType, ...] = (
    ("lncRNA", "lasso_positive", "gene"),
    ("gene", "lasso_positive_reverse", "lncRNA"),
    ("lncRNA", "lasso_negative", "gene"),
    ("gene", "lasso_negative_reverse", "lncRNA"),
)

BASE_EDGE_TYPES: tuple[EdgeType, ...] = (
    ("gene", "member_of", "pathway"),
    ("pathway", "has_member", "gene"),
    ("pathway", "belongs_to", "pathway_family"),
    ("pathway_family", "contains", "pathway"),
    ("cell_context", "context_of_lncRNA", "lncRNA"),
    ("lncRNA", "in_context", "cell_context"),
    ("cell_context", "context_of_gene", "gene"),
    ("gene", "in_context", "cell_context"),
    ("cell_context", "context_of_pathway", "pathway"),
    ("pathway", "in_context", "cell_context"),
    ("cell_context", "context_of_pathway_family", "pathway_family"),
    ("pathway_family", "in_context", "cell_context"),
) + LASSO_EDGE_TYPES

_EXTERNAL_RELATION_PATTERN = re.compile(r"[^A-Za-z0-9_]+")
_ALLOWED_EXTERNAL_NODE_TYPES = frozenset(NODE_TYPES) - {"cell_context"}


@dataclass(frozen=True)
class ContextGraph:
    """A graph and the donor provenance required to trust it."""

    data: HeteroData
    source_donors: tuple[str, ...]
    excluded_donors: tuple[str, ...]
    external_edge_audit: tuple[dict[str, Any], ...]
    edge_schema: tuple[EdgeType, ...]

    def clone(self) -> "ContextGraph":
        return ContextGraph(
            data=self.data.clone(),
            source_donors=self.source_donors,
            excluded_donors=self.excluded_donors,
            external_edge_audit=tuple(dict(row) for row in self.external_edge_audit),
            edge_schema=self.edge_schema,
        )

    def to(self, device: torch.device | str) -> "ContextGraph":
        return ContextGraph(
            data=self.data.clone().to(device),
            source_donors=self.source_donors,
            excluded_donors=self.excluded_donors,
            external_edge_audit=tuple(dict(row) for row in self.external_edge_audit),
            edge_schema=self.edge_schema,
        )


def _donor_set(values: Iterable[Any], *, label: str) -> set[str]:
    result: set[str] = set()
    for value in values:
        if value is None or not str(value).strip():
            raise ValueError(f"{label}_HAS_EMPTY_DONOR")
        result.add(str(value).strip())
    return result


def _float_matrix(value: Any, *, label: str) -> Tensor:
    try:
        tensor = torch.as_tensor(value, dtype=torch.float32, device="cpu")
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(label) from exc
    if tensor.ndim != 2 or tensor.shape[0] == 0 or tensor.shape[1] == 0:
        raise ValueError(label)
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(label)
    return tensor.detach().clone().contiguous()


def _canonical_external_edge_type(spec: Sequence[Any]) -> EdgeType:
    if len(spec) != 3:
        raise ValueError("EXTERNAL_EDGE_SCHEMA")
    source_type, relation, target_type = map(str, spec)
    if (
        source_type not in _ALLOWED_EXTERNAL_NODE_TYPES
        or target_type not in _ALLOWED_EXTERNAL_NODE_TYPES
    ):
        raise ValueError("EXTERNAL_EDGE_NODE_TYPE")
    relation = _EXTERNAL_RELATION_PATTERN.sub("_", relation.strip()).strip("_")
    if not relation:
        raise ValueError("EXTERNAL_EDGE_RELATION")
    if not relation.startswith("external_"):
        relation = f"external_{relation}"
    return source_type, relation, target_type


def _membership_edges(
    membership: Any,
    *,
    row_count: int,
    column_count: int,
    orientation: str | None,
    required_orientation: str,
    label: str,
    binary: bool,
    require_every_row: bool,
    require_every_column: bool,
) -> tuple[Tensor, Tensor]:
    if orientation != required_orientation:
        raise ValueError(f"{label}_ORIENTATION_REQUIRED")
    expected = (row_count, column_count)
    transposed = (column_count, row_count)
    if isinstance(membership, Mapping):
        if set(membership) != {"row_index", "column_index", "weight", "shape"}:
            raise ValueError(f"{label}_COO_SCHEMA")
        shape = tuple(int(value) for value in membership["shape"])
        if shape != expected:
            if expected != transposed and shape == transposed:
                raise ValueError(f"{label}_TRANSPOSED")
            raise ValueError(f"{label}_SHAPE")
        try:
            row = torch.as_tensor(
                membership["row_index"], dtype=torch.long, device="cpu"
            )
            column = torch.as_tensor(
                membership["column_index"], dtype=torch.long, device="cpu"
            )
            weight = torch.as_tensor(
                membership["weight"], dtype=torch.float32, device="cpu"
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(f"{label}_COO_VALUES") from exc
        if any(
            value.ndim != 1 or value.numel() != weight.numel()
            for value in (row, column, weight)
        ):
            raise ValueError(f"{label}_COO_SHAPE")
    elif hasattr(membership, "tocoo"):
        coo = membership.tocoo(copy=False)
        shape = tuple(int(value) for value in coo.shape)
        if shape != expected:
            if expected != transposed and shape == transposed:
                raise ValueError(f"{label}_TRANSPOSED")
            raise ValueError(f"{label}_SHAPE")
        row = torch.as_tensor(np.asarray(coo.row).copy(), dtype=torch.long)
        column = torch.as_tensor(np.asarray(coo.col).copy(), dtype=torch.long)
        weight = torch.as_tensor(np.asarray(coo.data).copy(), dtype=torch.float32)
    else:
        try:
            raw = torch.as_tensor(membership, dtype=torch.float32, device="cpu")
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(f"{label}_SHAPE") from exc
        if tuple(raw.shape) != expected:
            if expected != transposed and tuple(raw.shape) == transposed:
                raise ValueError(f"{label}_TRANSPOSED")
            raise ValueError(f"{label}_SHAPE")
        index = torch.nonzero(raw != 0, as_tuple=False).T.contiguous()
        row, column = index
        weight = raw[row, column]
    if not bool(torch.isfinite(weight).all()) or bool((weight <= 0).any()):
        raise ValueError(f"{label}_VALUES")
    if binary and not bool((weight == 1).all()):
        raise ValueError(f"{label}_VALUES")
    if weight.numel() and (
        bool((row < 0).any())
        or int(row.max()) >= row_count
        or bool((column < 0).any())
        or int(column.max()) >= column_count
    ):
        raise ValueError(f"{label}_IDENTITY")
    if weight.numel():
        linear = row.to(torch.int64) * column_count + column
        if torch.unique(linear).numel() != weight.numel():
            raise ValueError(f"{label}_DUPLICATE_EDGE")
    row_degree = torch.zeros(row_count, dtype=torch.long)
    column_degree = torch.zeros(column_count, dtype=torch.long)
    if weight.numel():
        row_degree.index_add_(0, row, torch.ones_like(row))
        column_degree.index_add_(0, column, torch.ones_like(column))
    if require_every_row and bool((row_degree == 0).any()):
        raise ValueError(f"{label}_ROW_COVERAGE")
    if require_every_column and bool((column_degree == 0).any()):
        raise ValueError(f"{label}_COLUMN_COVERAGE")
    return torch.stack((row, column), dim=0).contiguous(), weight.contiguous()


def _lasso_tensors(
    coefficients: Any | None,
    stability: Any | None,
    *,
    lnc_count: int,
    gene_count: int,
) -> tuple[Tensor, Tensor]:
    shape = (lnc_count, gene_count)
    if coefficients is None and stability is None:
        return torch.zeros(shape), torch.zeros(shape)
    if coefficients is None or stability is None:
        raise ValueError("LASSO_COEFFICIENT_STABILITY_PAIR_REQUIRED")
    try:
        coefficient_tensor = torch.as_tensor(
            coefficients, dtype=torch.float32, device="cpu"
        )
        stability_tensor = torch.as_tensor(
            stability, dtype=torch.float32, device="cpu"
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("LASSO_GENE_IDENTITY") from exc
    if tuple(coefficient_tensor.shape) != shape or tuple(stability_tensor.shape) != shape:
        raise ValueError("LASSO_GENE_IDENTITY")
    if not bool(torch.isfinite(coefficient_tensor).all()) or not bool(
        torch.isfinite(stability_tensor).all()
    ):
        raise ValueError("LASSO_VALUES")
    if bool((stability_tensor < 0).any()) or bool((stability_tensor > 1).any()):
        raise ValueError("LASSO_STABILITY_RANGE")
    return coefficient_tensor.contiguous(), stability_tensor.contiguous()


def _sparse_lasso_tensors(
    value: Mapping[str, Any] | None,
    *,
    lnc_count: int,
    gene_count: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if value is None:
        empty_index = torch.empty(0, dtype=torch.long)
        empty_value = torch.empty(0, dtype=torch.float32)
        return empty_index, empty_index.clone(), empty_value, empty_value.clone()
    if not isinstance(value, Mapping):
        raise ValueError("SPARSE_LASSO_RECORD")
    required = {"lnc_index", "gene_index", "coefficient", "stability"}
    if set(value) != required:
        raise ValueError("SPARSE_LASSO_SCHEMA")
    try:
        lnc_index = torch.as_tensor(value["lnc_index"], dtype=torch.long, device="cpu")
        gene_index = torch.as_tensor(value["gene_index"], dtype=torch.long, device="cpu")
        coefficient = torch.as_tensor(
            value["coefficient"], dtype=torch.float32, device="cpu"
        )
        stability = torch.as_tensor(
            value["stability"], dtype=torch.float32, device="cpu"
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("SPARSE_LASSO_VALUES") from exc
    size = coefficient.numel()
    if any(
        tensor.ndim != 1 or tensor.numel() != size
        for tensor in (lnc_index, gene_index, coefficient, stability)
    ):
        raise ValueError("SPARSE_LASSO_SHAPE")
    if not bool(torch.isfinite(coefficient).all()) or not bool(
        torch.isfinite(stability).all()
    ):
        raise ValueError("LASSO_VALUES")
    if bool((stability < 0).any()) or bool((stability > 1).any()):
        raise ValueError("LASSO_STABILITY_RANGE")
    if size and (
        bool((lnc_index < 0).any())
        or int(lnc_index.max()) >= lnc_count
        or bool((gene_index < 0).any())
        or int(gene_index.max()) >= gene_count
    ):
        raise ValueError("LASSO_GENE_IDENTITY")
    if bool((coefficient == 0).any()):
        raise ValueError("SPARSE_LASSO_ZERO_COEFFICIENT")
    if size:
        linear = lnc_index.to(torch.int64) * gene_count + gene_index
        if torch.unique(linear).numel() != size:
            raise ValueError("SPARSE_LASSO_DUPLICATE_EDGE")
    return (
        lnc_index.contiguous(),
        gene_index.contiguous(),
        coefficient.contiguous(),
        stability.contiguous(),
    )


def build_context_graph(
    features: Mapping[str, Any],
    gene_pathway_membership: Any,
    pathway_family_membership: Any,
    *,
    membership_orientation: str | None = None,
    pathway_family_orientation: str | None = None,
    lasso_coefficients: Any | None = None,
    lasso_stability: Any | None = None,
    sparse_lasso_edges: Mapping[str, Any] | None = None,
    lasso_stability_threshold: float = 0.5,
    external_edges: Iterable[Mapping[str, Any]] | None = None,
    external_edge_batches: Iterable[Mapping[str, Any]] | None = None,
    external_relation_types: Iterable[Sequence[Any]] | None = None,
    source_donors: Iterable[Any],
    excluded_donors: Iterable[Any],
) -> ContextGraph:
    """Build one donor-isolated graph for a cell-type context.

    Both membership matrices require explicitly declared orientations. LASSO
    relations are materialized even when no LASSO evidence is supplied,
    preserving identical graph metadata for with/without-LASSO comparisons.
    """

    source = _donor_set(source_donors, label="SOURCE")
    excluded = _donor_set(excluded_donors, label="EXCLUDED")
    if not source or source.intersection(excluded):
        raise ValueError("SC_GRAPH_DONOR_LEAK")
    if set(features) != set(NODE_TYPES):
        raise ValueError("SC_NODE_SCHEMA")

    continuous = {
        node_type: _float_matrix(features[node_type], label="INVALID_SC_NODE_FEATURES")
        for node_type in NODE_TYPES
    }
    if continuous["cell_context"].shape[0] != 1:
        raise ValueError("SC_CELL_CONTEXT_COUNT")

    node_count = {node_type: value.shape[0] for node_type, value in continuous.items()}
    membership_index, membership_weight = _membership_edges(
        gene_pathway_membership,
        row_count=node_count["gene"],
        column_count=node_count["pathway"],
        orientation=membership_orientation,
        required_orientation="gene_by_pathway",
        label="GENE_PATHWAY_MEMBERSHIP",
        binary=True,
        require_every_row=False,
        require_every_column=True,
    )
    family_index, family_weight = _membership_edges(
        pathway_family_membership,
        row_count=node_count["pathway"],
        column_count=node_count["pathway_family"],
        orientation=pathway_family_orientation,
        required_orientation="pathway_by_family",
        label="PATHWAY_FAMILY_MEMBERSHIP",
        binary=False,
        require_every_row=True,
        require_every_column=True,
    )
    if sparse_lasso_edges is not None and (
        lasso_coefficients is not None or lasso_stability is not None
    ):
        raise ValueError("DENSE_AND_SPARSE_LASSO_ARE_MUTUALLY_EXCLUSIVE")
    if sparse_lasso_edges is None:
        coefficient, stability = _lasso_tensors(
            lasso_coefficients,
            lasso_stability,
            lnc_count=node_count["lncRNA"],
            gene_count=node_count["gene"],
        )
        dense_nonzero = torch.nonzero(coefficient != 0, as_tuple=False)
        sparse_lnc = dense_nonzero[:, 0]
        sparse_gene = dense_nonzero[:, 1]
        sparse_coefficient = coefficient[sparse_lnc, sparse_gene]
        sparse_stability = stability[sparse_lnc, sparse_gene]
    else:
        sparse_lnc, sparse_gene, sparse_coefficient, sparse_stability = (
            _sparse_lasso_tensors(
                sparse_lasso_edges,
                lnc_count=node_count["lncRNA"],
                gene_count=node_count["gene"],
            )
        )
    if not 0 <= float(lasso_stability_threshold) <= 1:
        raise ValueError("LASSO_STABILITY_THRESHOLD_RANGE")

    if external_edges is not None and external_edge_batches is not None:
        raise ValueError("EXTERNAL_RECORDS_AND_BATCHES_ARE_MUTUALLY_EXCLUSIVE")
    external_records = list(external_edges or ())
    external_batches = list(external_edge_batches or ())
    registered_external: set[EdgeType] = set()
    if external_relation_types is not None:
        for spec in external_relation_types:
            registered_external.add(_canonical_external_edge_type(spec))
    else:
        for record in (*external_records, *external_batches):
            if not isinstance(record, Mapping):
                raise ValueError("EXTERNAL_EDGE_RECORD")
            registered_external.add(
                _canonical_external_edge_type(
                    (
                        record.get("source_type", ""),
                        record.get("relation", "evidence"),
                        record.get("target_type", ""),
                    )
                )
            )
    if registered_external.intersection(BASE_EDGE_TYPES):
        raise ValueError("EXTERNAL_EDGE_RELATION_COLLISION")
    external_schema = tuple(sorted(registered_external))
    edge_schema = BASE_EDGE_TYPES + external_schema

    buffers: dict[EdgeType, list[tuple[Tensor, Tensor]]] = {
        edge_type: [] for edge_type in edge_schema
    }
    degree = {
        node_type: torch.zeros(count, dtype=torch.float32)
        for node_type, count in node_count.items()
    }

    def append_relation(edge_type: EdgeType, edge_index: Any, edge_weight: Any) -> None:
        source_type, _, target_type = edge_type
        if edge_type not in buffers:
            raise ValueError("SC_EDGE_SCHEMA")
        try:
            index = torch.as_tensor(edge_index, dtype=torch.long, device="cpu")
            weight = torch.as_tensor(edge_weight, dtype=torch.float32, device="cpu")
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError("SC_EDGE_SHAPE") from exc
        if tuple(index.shape) != (2, weight.numel()) or weight.ndim != 1:
            raise ValueError("SC_EDGE_SHAPE")
        if not bool(torch.isfinite(weight).all()):
            raise ValueError("SC_EDGE_WEIGHT")
        if index.numel() and (
            bool((index < 0).any())
            or int(index[0].max()) >= node_count[source_type]
            or int(index[1].max()) >= node_count[target_type]
        ):
            raise ValueError("SC_EDGE_IDENTITY")
        index = index.contiguous()
        weight = weight.contiguous()
        buffers[edge_type].append((index, weight))
        if weight.numel():
            ones = torch.ones_like(weight)
            degree[source_type].index_add_(0, index[0], ones)
            degree[target_type].index_add_(0, index[1], ones)

    append_relation(
        ("gene", "member_of", "pathway"), membership_index, membership_weight
    )
    append_relation(
        ("pathway", "has_member", "gene"),
        membership_index.flip(0),
        membership_weight,
    )

    append_relation(
        ("pathway", "belongs_to", "pathway_family"), family_index, family_weight
    )
    append_relation(
        ("pathway_family", "contains", "pathway"),
        family_index.flip(0),
        family_weight,
    )

    for node_type in NODE_TYPES[:-1]:
        context_index = torch.stack(
            (
                torch.zeros(node_count[node_type], dtype=torch.long),
                torch.arange(node_count[node_type], dtype=torch.long),
            )
        )
        context_weight = torch.ones(node_count[node_type], dtype=torch.float32)
        append_relation(
            ("cell_context", f"context_of_{node_type}", node_type),
            context_index,
            context_weight,
        )
        append_relation(
            (node_type, "in_context", "cell_context"),
            context_index.flip(0),
            context_weight,
        )

    stable = sparse_stability >= float(lasso_stability_threshold)
    for sign, keep in (
        ("positive", (sparse_coefficient > 0) & stable),
        ("negative", (sparse_coefficient < 0) & stable),
    ):
        index = torch.stack((sparse_lnc[keep], sparse_gene[keep]), dim=0)
        weight = sparse_coefficient[keep] * sparse_stability[keep]
        append_relation(("lncRNA", f"lasso_{sign}", "gene"), index, weight)
        append_relation(
            ("gene", f"lasso_{sign}_reverse", "lncRNA"),
            index.flip(0),
            weight,
        )

    external_audit: list[dict[str, Any]] = []
    for record_index, record in enumerate(external_records):
        if not isinstance(record, Mapping):
            raise ValueError("EXTERNAL_EDGE_RECORD")
        edge_type = _canonical_external_edge_type(
            (
                record.get("source_type", ""),
                record.get("relation", "evidence"),
                record.get("target_type", ""),
            )
        )
        if edge_type not in registered_external:
            raise ValueError("EXTERNAL_EDGE_NOT_IN_SCHEMA")
        edge_source_donors = _donor_set(
            record.get("source_donors", ()), label="EXTERNAL_SOURCE"
        )
        if edge_source_donors.intersection(excluded):
            raise ValueError("EXTERNAL_EDGE_HELDOUT_DONOR_LEAK")
        if edge_source_donors.difference(source):
            raise ValueError("EXTERNAL_EDGE_SOURCE_DONOR_SCOPE")

        source_hash = str(record.get("source_hash", "")).strip()
        provenance = str(record.get("provenance", "")).strip()
        available = bool(record.get("available", True))
        if available and not (source_hash and provenance):
            raise ValueError("EXTERNAL_EDGE_PROVENANCE")
        try:
            confidence = float(record.get("confidence", 0.0))
            sign = float(record.get("sign", 1.0))
            source_index = int(record.get("source_index", -1))
            target_index = int(record.get("target_index", -1))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("EXTERNAL_EDGE_VALUES") from exc
        if not torch.isfinite(torch.tensor(confidence)) or confidence < 0:
            raise ValueError("EXTERNAL_EDGE_CONFIDENCE")
        if not torch.isfinite(torch.tensor(sign)):
            raise ValueError("EXTERNAL_EDGE_SIGN")
        if available and (confidence <= 0 or sign not in (-1.0, 1.0)):
            raise ValueError("EXTERNAL_EDGE_NONZERO_SIGNED_WEIGHT_REQUIRED")
        valid_identity = (
            0 <= source_index < node_count[edge_type[0]]
            and 0 <= target_index < node_count[edge_type[2]]
        )
        if available and not valid_identity:
            raise ValueError("EXTERNAL_EDGE_IDENTITY")
        if available:
            append_relation(
                edge_type,
                [[source_index], [target_index]],
                [confidence * sign],
            )
        external_audit.append(
            {
                "record_index": record_index,
                "source_type": edge_type[0],
                "relation": edge_type[1],
                "target_type": edge_type[2],
                "source_index": source_index,
                "target_index": target_index,
                "confidence": confidence,
                "sign": sign,
                "available": available,
                "source_hash": source_hash,
                "provenance": provenance,
                "source_donors": sorted(edge_source_donors),
            }
        )

    for batch_index, batch in enumerate(external_batches):
        if not isinstance(batch, Mapping):
            raise ValueError("EXTERNAL_EDGE_BATCH_RECORD")
        edge_type = _canonical_external_edge_type(
            (
                batch.get("source_type", ""),
                batch.get("relation", "evidence"),
                batch.get("target_type", ""),
            )
        )
        if edge_type not in registered_external:
            raise ValueError("EXTERNAL_EDGE_NOT_IN_SCHEMA")
        edge_source_donors = _donor_set(
            batch.get("source_donors", ()), label="EXTERNAL_SOURCE"
        )
        if edge_source_donors.intersection(excluded):
            raise ValueError("EXTERNAL_EDGE_HELDOUT_DONOR_LEAK")
        if edge_source_donors.difference(source):
            raise ValueError("EXTERNAL_EDGE_SOURCE_DONOR_SCOPE")
        source_hash = str(batch.get("source_hash", "")).strip()
        provenance = str(batch.get("provenance", "")).strip()
        available = bool(batch.get("available", True))
        if available and not (source_hash and provenance):
            raise ValueError("EXTERNAL_EDGE_PROVENANCE")
        try:
            index = torch.as_tensor(
                batch.get("edge_index"), dtype=torch.long, device="cpu"
            )
            weight = torch.as_tensor(
                batch.get("edge_weight"), dtype=torch.float32, device="cpu"
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError("EXTERNAL_EDGE_BATCH_VALUES") from exc
        if index.ndim != 2 or index.shape[0] != 2 or weight.ndim != 1:
            raise ValueError("EXTERNAL_EDGE_SHAPE")
        if index.shape[1] != weight.numel():
            raise ValueError("EXTERNAL_EDGE_SHAPE")
        if not bool(torch.isfinite(weight).all()):
            raise ValueError("EXTERNAL_EDGE_CONFIDENCE")
        if available and bool((weight == 0).any()):
            raise ValueError("EXTERNAL_EDGE_NONZERO_SIGNED_WEIGHT_REQUIRED")
        if available:
            append_relation(edge_type, index, weight)
        external_audit.append(
            {
                "batch_index": batch_index,
                "source_type": edge_type[0],
                "relation": edge_type[1],
                "target_type": edge_type[2],
                "edge_count": int(weight.numel()) if available else 0,
                "available": available,
                "source_hash": source_hash,
                "provenance": provenance,
                "source_donors": sorted(edge_source_donors),
                "minimum_edge_weight": float(weight.min()) if weight.numel() else None,
                "maximum_edge_weight": float(weight.max()) if weight.numel() else None,
            }
        )

    data = HeteroData()
    for node_type in NODE_TYPES:
        data[node_type].continuous_x = continuous[node_type]
        data[node_type].x = torch.stack(
            (torch.log1p(degree[node_type]), torch.ones_like(degree[node_type])), dim=1
        )
        data[node_type].num_nodes = node_count[node_type]
    for edge_type in edge_schema:
        pieces = buffers[edge_type]
        if pieces:
            edge_index = torch.cat([piece[0] for piece in pieces], dim=1)
            edge_weight = torch.cat([piece[1] for piece in pieces], dim=0)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_weight = torch.empty((0,), dtype=torch.float32)
        data[edge_type].edge_index = edge_index
        data[edge_type].edge_weight = edge_weight

    return ContextGraph(
        data=data,
        source_donors=tuple(sorted(source)),
        excluded_donors=tuple(sorted(excluded)),
        external_edge_audit=tuple(external_audit),
        edge_schema=edge_schema,
    )


# Preserve the Sep-10 handoff name while making the preferred API explicit.
context_graph = build_context_graph


def _validate_context_graph(graph: ContextGraph) -> None:
    if not isinstance(graph, ContextGraph):
        raise TypeError("SC_CONTEXT_GRAPH_REQUIRED")
    if not graph.source_donors or set(graph.source_donors).intersection(
        graph.excluded_donors
    ):
        raise ValueError("SC_GRAPH_DONOR_LEAK")
    data = graph.data
    if tuple(data.node_types) != NODE_TYPES:
        raise ValueError("SC_NODE_SCHEMA")
    if tuple(data.edge_types) != graph.edge_schema:
        raise ValueError("SC_EDGE_SCHEMA")
    if data["cell_context"].num_nodes != 1:
        raise ValueError("SC_CELL_CONTEXT_COUNT")
    for node_type in NODE_TYPES:
        structural = data[node_type].x
        continuous = data[node_type].continuous_x
        if structural.ndim != 2 or structural.shape != (data[node_type].num_nodes, 2):
            raise ValueError("INVALID_SC_STRUCTURAL_FEATURES")
        if continuous.ndim != 2 or continuous.shape[0] != data[node_type].num_nodes:
            raise ValueError("INVALID_SC_NODE_FEATURES")
        if not bool(torch.isfinite(structural).all()) or not bool(
            torch.isfinite(continuous).all()
        ):
            raise ValueError("INVALID_SC_NODE_FEATURES")
    for edge_type in graph.edge_schema:
        edge_index = data[edge_type].edge_index
        edge_weight = data[edge_type].edge_weight
        if edge_index.dtype != torch.long or tuple(edge_index.shape) != (
            2,
            edge_weight.numel(),
        ):
            raise ValueError("SC_EDGE_SHAPE")
        if edge_weight.ndim != 1 or not bool(torch.isfinite(edge_weight).all()):
            raise ValueError("SC_EDGE_WEIGHT")


class WeightedHGTEncoder(nn.Module):
    """HGT topology encoder with an explicit signed edge-weight path."""

    def __init__(
        self,
        example_graph: ContextGraph,
        *,
        hidden_channels: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        _validate_context_graph(example_graph)
        if hidden_channels <= 0 or num_layers <= 0 or num_heads <= 0:
            raise ValueError("SC_HGT_DIMENSIONS")
        if hidden_channels % num_heads:
            raise ValueError("HEAD_DIMENSION")
        if not 0 <= dropout < 1:
            raise ValueError("SC_HGT_DROPOUT")

        self.node_types = NODE_TYPES
        self.edge_types = example_graph.edge_schema
        self.continuous_feature_dims = {
            node_type: int(example_graph.data[node_type].continuous_x.shape[1])
            for node_type in NODE_TYPES
        }
        metadata = (list(self.node_types), list(self.edge_types))
        self.structural_projection = nn.ModuleDict(
            {node_type: nn.Linear(2, hidden_channels) for node_type in NODE_TYPES}
        )
        self.continuous_projection = nn.ModuleDict(
            {
                node_type: nn.Linear(self.continuous_feature_dims[node_type], hidden_channels)
                for node_type in NODE_TYPES
            }
        )
        self.layers = nn.ModuleList(
            [
                HGTConv(
                    hidden_channels,
                    hidden_channels,
                    metadata,
                    heads=num_heads,
                )
                for _ in range(num_layers)
            ]
        )
        self.weighted_projection = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        node_type: nn.Linear(hidden_channels, hidden_channels, bias=False)
                        for node_type in NODE_TYPES
                    }
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = float(dropout)

    def _validate_for_model(self, graph: ContextGraph) -> None:
        _validate_context_graph(graph)
        if graph.edge_schema != self.edge_types:
            raise ValueError("SC_HGT_EDGE_SCHEMA_DRIFT")
        parameter_device = next(self.parameters()).device
        for node_type in NODE_TYPES:
            continuous = graph.data[node_type].continuous_x
            if continuous.shape[1] != self.continuous_feature_dims[node_type]:
                raise ValueError("SC_CONTINUOUS_FEATURE_SCHEMA_DRIFT")
            if continuous.device != parameter_device or graph.data[node_type].x.device != parameter_device:
                raise ValueError("SC_GRAPH_MODEL_DEVICE_MISMATCH")
        for edge_type in self.edge_types:
            if (
                graph.data[edge_type].edge_index.device != parameter_device
                or graph.data[edge_type].edge_weight.device != parameter_device
            ):
                raise ValueError("SC_GRAPH_MODEL_DEVICE_MISMATCH")

    @staticmethod
    def _weighted_aggregate(
        hidden: Mapping[str, Tensor],
        data: HeteroData,
        projection: nn.ModuleDict,
    ) -> dict[str, Tensor]:
        aggregate = {node_type: torch.zeros_like(value) for node_type, value in hidden.items()}
        denominator = {
            node_type: value.new_zeros((value.shape[0], 1))
            for node_type, value in hidden.items()
        }
        for edge_type in data.edge_types:
            source_type, _, target_type = edge_type
            edge_index = data[edge_type].edge_index
            edge_weight = data[edge_type].edge_weight.to(dtype=hidden[source_type].dtype)
            source_index, target_index = edge_index
            if edge_weight.numel() == 0:
                continue
            aggregate[target_type].index_add_(
                0,
                target_index,
                hidden[source_type][source_index] * edge_weight.unsqueeze(1),
            )
            denominator[target_type].index_add_(
                0, target_index, edge_weight.abs().unsqueeze(1)
            )
        return {
            node_type: projection[node_type](
                aggregate[node_type] / denominator[node_type].clamp_min(1e-8)
            )
            for node_type in NODE_TYPES
        }

    def encode(
        self, graph: ContextGraph, *, message_passing: bool = True
    ) -> dict[str, Tensor]:
        self._validate_for_model(graph)
        data = graph.data
        hidden = {
            node_type: F.gelu(
                self.structural_projection[node_type](data[node_type].x)
                + self.continuous_projection[node_type](data[node_type].continuous_x)
            )
            for node_type in NODE_TYPES
        }
        if not message_passing:
            return hidden
        for convolution, projection in zip(
            self.layers, self.weighted_projection, strict=True
        ):
            topology = convolution(hidden, data.edge_index_dict)
            weighted = self._weighted_aggregate(hidden, data, projection)
            hidden = {
                node_type: F.dropout(
                    F.gelu(topology.get(node_type, hidden[node_type]) + weighted[node_type]),
                    p=self.dropout,
                    training=self.training,
                )
                for node_type in NODE_TYPES
            }
        return hidden


class SingleCellHHGT(nn.Module):
    """Predict signed within-donor metacell rho from one context graph."""

    def __init__(
        self,
        example_graph: ContextGraph,
        *,
        hidden_channels: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        pair_feature_count: int = 5,
    ) -> None:
        super().__init__()
        if pair_feature_count < 0:
            raise ValueError("PAIR_FEATURE_COUNT")
        self.pair_feature_count = int(pair_feature_count)
        self.encoder = WeightedHGTEncoder(
            example_graph,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_channels * 3 + self.pair_feature_count, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def encode_graph(
        self, graph: ContextGraph, *, message_passing: bool = True
    ) -> dict[str, Tensor]:
        return self.encoder.encode(graph, message_passing=message_passing)

    def decode(
        self,
        encoded: Mapping[str, Tensor],
        lnc_index: Any,
        pathway_index: Any,
        pair_features: Any,
    ) -> Tensor:
        device = next(self.parameters()).device
        if set(encoded) != set(NODE_TYPES):
            raise ValueError("SC_ENCODED_NODE_SCHEMA")
        if any(value.ndim != 2 or value.device != device for value in encoded.values()):
            raise ValueError("SC_ENCODED_NODE_VALUES")
        lnc_index = torch.as_tensor(lnc_index, dtype=torch.long, device=device)
        pathway_index = torch.as_tensor(pathway_index, dtype=torch.long, device=device)
        pair_features = torch.as_tensor(
            pair_features, dtype=torch.float32, device=device
        )
        if lnc_index.ndim != 1 or pathway_index.ndim != 1:
            raise ValueError("SC_QUERY_INDEX_SHAPE")
        batch_size = lnc_index.numel()
        if pathway_index.numel() != batch_size or tuple(pair_features.shape) != (
            batch_size,
            self.pair_feature_count,
        ):
            raise ValueError("SC_QUERY_BATCH_SHAPE")
        if not bool(torch.isfinite(pair_features).all()):
            raise ValueError("SC_QUERY_FEATURE_VALUES")
        if batch_size and (
            bool((lnc_index < 0).any())
            or int(lnc_index.max()) >= encoded["lncRNA"].shape[0]
            or bool((pathway_index < 0).any())
            or int(pathway_index.max()) >= encoded["pathway"].shape[0]
        ):
            raise ValueError("SC_QUERY_IDENTITY")

        context = encoded["cell_context"][0].expand(batch_size, -1)
        joined = torch.cat(
            (
                encoded["lncRNA"][lnc_index],
                encoded["pathway"][pathway_index],
                context,
                pair_features,
            ),
            dim=1,
        )
        return torch.tanh(self.decoder(joined).squeeze(1))

    def forward(
        self,
        graph: ContextGraph,
        lnc_index: Any,
        pathway_index: Any,
        pair_features: Any,
        *,
        message_passing: bool = True,
    ) -> Tensor:
        encoded = self.encode_graph(graph, message_passing=message_passing)
        return self.decode(encoded, lnc_index, pathway_index, pair_features)


def _gradient_l1(module: nn.Module) -> float:
    values = [
        parameter.grad.detach().abs().sum()
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    if not values:
        return 0.0
    total = torch.stack(values).sum()
    if not bool(torch.isfinite(total)):
        raise RuntimeError("SC_NONFINITE_GRADIENT")
    return float(total)


def optimizer_step(
    model: SingleCellHHGT,
    optimizer: torch.optim.Optimizer,
    graph: ContextGraph,
    lnc_index: Any,
    pathway_index: Any,
    pair_features: Any,
    target_rho: Any,
    *,
    message_passing: bool = True,
    max_gradient_norm: float = 5.0,
) -> dict[str, float | int]:
    """Run one signed-rho optimization step and enforce encoder gradients."""

    if max_gradient_norm <= 0:
        raise ValueError("SC_GRADIENT_CLIP")
    device = next(model.parameters()).device
    target = torch.as_tensor(target_rho, dtype=torch.float32, device=device)
    if target.ndim != 1 or not bool(torch.isfinite(target).all()):
        raise ValueError("SC_TARGET_SHAPE_OR_VALUES")
    if bool((target < -1).any()) or bool((target > 1).any()):
        raise ValueError("SC_TARGET_RHO_RANGE")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    prediction = model(
        graph,
        lnc_index,
        pathway_index,
        pair_features,
        message_passing=message_passing,
    )
    if prediction.shape != target.shape:
        raise ValueError("SC_TARGET_BATCH_SHAPE")
    loss = F.smooth_l1_loss(prediction, target, beta=0.1)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("SC_NONFINITE_LOSS")
    loss.backward()

    gradients = {
        "hgt_gradient_l1": _gradient_l1(model.encoder.layers),
        "weighted_projection_gradient_l1": _gradient_l1(
            model.encoder.weighted_projection
        ),
        "continuous_projection_gradient_l1": _gradient_l1(
            model.encoder.continuous_projection
        ),
        "structural_projection_gradient_l1": _gradient_l1(
            model.encoder.structural_projection
        ),
        "decoder_gradient_l1": _gradient_l1(model.decoder),
    }
    if message_passing and gradients["hgt_gradient_l1"] <= 0:
        raise RuntimeError("SC_HGT_ENCODER_HAS_NO_GRADIENT")
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
    optimizer.step()
    return {
        "loss": float(loss.detach()),
        **gradients,
        "optimizer_steps": 1,
    }


__all__ = [
    "BASE_EDGE_TYPES",
    "ContextGraph",
    "LASSO_EDGE_TYPES",
    "NODE_TYPES",
    "SingleCellHHGT",
    "WeightedHGTEncoder",
    "build_context_graph",
    "context_graph",
    "optimizer_step",
]
