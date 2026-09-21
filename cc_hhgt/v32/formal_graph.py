"""Fresh, typed G0/G1/G2 graph materialization for V3.2.

This module deliberately does not read historical graph parquets.  It turns
fresh, already-authorized source tables into the exact role/relation contract
accepted by :mod:`cc_hhgt.v32.safe_graph`.  Weights are positive magnitudes;
signed biology is represented both by a typed relation and by
``relation_polarity``.  The PPI builder canonicalizes source pairs and then
emits exactly two reciprocal rows under one relation type.

The returned master authority contains the G2 union.  G0/G1 are true edge
ablations over that same node universe and relation schema.  Their node
features are always computed from the common G0 base, so a protein-layer
degree cannot leak an arm identity into ``node_x``.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from ..relation_sampling import (
    RuntimeSamplingPolicy,
    runtime_chunk_positions,
    schedule_runtime_edges,
)
from .safe_graph import EDGE_KEYS, SafeGraph, build_safe_graph


FORMAL_GRAPH_FORMAT = "CANCERLNCATLAS_V32_FRESH_G012_GRAPH_V1"
VARIANTS = ("G0", "G1", "G2")
G1_ROLES = frozenset(
    {"static_global_lnc_protein_binding", "static_protein_gene_encoding"}
)
G2_ONLY_ROLES = frozenset({"static_symmetric_ppi"})
COEXPRESSION_ROLES = frozenset(
    {"fold_train_coexpression_positive", "fold_train_coexpression_negative"}
)
FORMAL_RELATION_SCHEMA = (
    ("lncRNA", "expressed_in", "cancer", "transductive_expression_eligibility", False),
    ("lncRNA", "coexpressed_positive", "gene", "fold_train_coexpression_positive", False),
    ("lncRNA", "coexpressed_negative", "gene", "fold_train_coexpression_negative", False),
    ("gene", "member_of_positive", "pathway", "static_signed_pathway_membership_positive", False),
    ("gene", "member_of_negative", "pathway", "static_signed_pathway_membership_negative", False),
    ("pathway", "member_of_family", "pathway_family", "static_pathway_hierarchy", False),
    ("lncRNA", "binds_protein", "protein", "static_global_lnc_protein_binding", False),
    ("protein", "encoded_by", "gene", "static_protein_gene_encoding", False),
    ("protein", "physical_interaction", "protein", "static_symmetric_ppi", True),
)


@dataclass(frozen=True)
class FormalGraphAuthority:
    nodes: pd.DataFrame
    edges: pd.DataFrame
    safe_graph: SafeGraph
    variant_masks: Mapping[str, pd.Series]
    relation_schema: pd.DataFrame
    manifest: dict


def _text(values: Iterable[object]) -> pd.Series:
    output = pd.Series(values, dtype="string").str.strip()
    return output.mask(output.isna() | output.eq(""))


def _numeric(values: Iterable[object], *, name: str) -> pd.Series:
    output = pd.to_numeric(pd.Series(values).reset_index(drop=True), errors="coerce")
    if output.isna().any() or not np.isfinite(output.to_numpy(float)).all():
        raise RuntimeError(f"Fresh graph {name} contains non-finite values")
    return output.astype(float)


def _edge_id(row: pd.Series) -> str:
    payload = "\x1f".join(str(row[column]) for column in EDGE_KEYS)
    return "V32EDGE:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _typed_edges(
    source_id: Iterable[object],
    target_id: Iterable[object],
    *,
    source_type: str,
    relation_type: str,
    target_type: str,
    edge_role: str,
    source_split: str,
    weight: Iterable[object],
    relation_polarity: float,
    cancer_id: Iterable[object] | None = None,
    is_context_specific: bool = False,
    edge_outer_fold: int | None = None,
    source_database: Iterable[object] | str | None = None,
) -> pd.DataFrame:
    source = _text(source_id).reset_index(drop=True)
    target = _text(target_id).reset_index(drop=True)
    if len(source) != len(target):
        raise ValueError("Fresh graph endpoint columns have different lengths")
    count = len(source)
    magnitude = _numeric(weight, name="weight")
    if len(magnitude) != count:
        raise ValueError("Fresh graph weight column has the wrong length")
    if magnitude.le(0).any():
        raise RuntimeError("Fresh graph magnitudes must be strictly positive")
    if cancer_id is None:
        cancer = pd.Series(pd.NA, index=range(count), dtype="string")
    else:
        cancer = _text(cancer_id).reset_index(drop=True)
        if len(cancer) != count:
            raise ValueError("Fresh graph cancer column has the wrong length")
    if source_database is None or np.isscalar(source_database):
        database = pd.Series(source_database, index=range(count), dtype="string")
    else:
        database = _text(source_database).reset_index(drop=True)
        if len(database) != count:
            raise ValueError("Fresh graph source_database has the wrong length")
    frame = pd.DataFrame(
        {
            "source_type": str(source_type),
            "source_id": source,
            "relation_type": str(relation_type),
            "target_type": str(target_type),
            "target_id": target,
            "edge_role": str(edge_role),
            "source_split": str(source_split),
            "outcome_derived": False,
            "observed": True,
            "weight": magnitude,
            "relation_polarity": float(relation_polarity),
            # The signed source statistic is consumed during transformation;
            # no raw fold statistic is allowed to survive into the graph.
            "raw_effect": pd.Series(pd.NA, index=range(count), dtype="Float64"),
            "requires_fold_localization": False,
            "edge_outer_fold": (
                pd.Series(pd.NA, index=range(count), dtype="Int64")
                if edge_outer_fold is None
                else pd.Series(int(edge_outer_fold), index=range(count), dtype="Int64")
            ),
            "cancer_id": cancer,
            "is_context_specific": bool(is_context_specific),
            "source_database": database,
        }
    )
    if frame[["source_id", "target_id"]].isna().any().any():
        raise RuntimeError("Fresh graph endpoints contain null/empty identifiers")
    return frame


def _max_magnitude(frame: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate a typed relation deterministically by maximum magnitude."""

    if frame.empty:
        return frame.reset_index(drop=True)
    ordering = [*EDGE_KEYS, "weight", "source_database"]
    existing = [column for column in ordering if column in frame]
    ordered = frame.sort_values(
        existing,
        ascending=[True] * len(EDGE_KEYS) + [False] + ([True] if "source_database" in existing else []),
        kind="stable",
        na_position="last",
    )
    return ordered.drop_duplicates(list(EDGE_KEYS), keep="first").reset_index(drop=True)


def materialize_expressed_in(
    detection: pd.DataFrame,
    candidate_pairs: pd.DataFrame,
    *,
    minimum_detection_rate: float = 0.10,
) -> pd.DataFrame:
    required = {"cancer_id", "lncrna_id", "detection_rate"}
    if missing := sorted(required - set(detection)):
        raise ValueError(f"Detection table lacks columns: {missing}")
    if missing := sorted({"cancer_id", "lncrna_id"} - set(candidate_pairs)):
        raise ValueError(f"Candidate table lacks columns: {missing}")
    values = detection.loc[:, ["cancer_id", "lncrna_id", "detection_rate"]].copy().reset_index(drop=True)
    values["detection_rate"] = _numeric(values.detection_rate, name="detection_rate")
    values = values.loc[values.detection_rate.ge(float(minimum_detection_rate))]
    keys = candidate_pairs.loc[:, ["cancer_id", "lncrna_id"]].drop_duplicates().reset_index(drop=True)
    values = values.merge(keys, on=["cancer_id", "lncrna_id"], how="inner", validate="one_to_one")
    result = _typed_edges(
        values.lncrna_id,
        values.cancer_id,
        source_type="lncRNA",
        relation_type="expressed_in",
        target_type="cancer",
        edge_role="transductive_expression_eligibility",
        source_split="all_samples_pre_outcome",
        weight=values.detection_rate,
        relation_polarity=1.0,
        cancer_id=values.cancer_id,
        is_context_specific=True,
        source_database="fresh_pre_outcome_expression_detection",
    )
    expected = keys.merge(values[["cancer_id", "lncrna_id"]], how="left", indicator=True)
    if expected._merge.ne("both").any():
        examples = expected.loc[expected._merge.ne("both"), ["cancer_id", "lncrna_id"]].head(10)
        raise RuntimeError(
            "Candidate pairs include lncRNAs below the declared expressed_in threshold: "
            f"{examples.to_dict('records')}"
        )
    return _max_magnitude(result)


def materialize_signed_coexpression(
    coexpression: pd.DataFrame,
    *,
    outer_fold: int,
) -> pd.DataFrame:
    required = {"cancer_id", "lncrna_id", "gene_id", "rho", "source_split", "edge_outer_fold"}
    if missing := sorted(required - set(coexpression)):
        raise ValueError(f"Coexpression table lacks columns: {missing}")
    values = coexpression.copy().reset_index(drop=True)
    if not values.source_split.astype(str).eq("train").all():
        raise RuntimeError("Fold-local coexpression contains a non-train source split")
    folds = pd.to_numeric(values.edge_outer_fold, errors="coerce")
    if folds.isna().any() or not folds.eq(int(outer_fold)).all():
        raise RuntimeError("Fold-local coexpression outer-fold provenance drift")
    rho = _numeric(values.rho, name="coexpression rho")
    if rho.eq(0).any():
        raise RuntimeError("Zero-effect coexpression cannot be assigned a signed relation")
    frames: list[pd.DataFrame] = []
    for positive, relation, role, polarity in (
        (True, "coexpressed_positive", "fold_train_coexpression_positive", 1.0),
        (False, "coexpressed_negative", "fold_train_coexpression_negative", -1.0),
    ):
        selected = values.loc[rho.gt(0) if positive else rho.lt(0)].copy()
        selected_rho = rho.loc[selected.index]
        if selected.empty:
            continue
        frames.append(
            _typed_edges(
                selected.lncrna_id,
                selected.gene_id,
                source_type="lncRNA",
                relation_type=relation,
                target_type="gene",
                edge_role=role,
                source_split="train",
                weight=selected_rho.abs(),
                relation_polarity=polarity,
                cancer_id=selected.cancer_id,
                is_context_specific=True,
                edge_outer_fold=int(outer_fold),
                source_database="fresh_outer_train_coexpression",
            )
        )
    if not frames:
        return _empty_edges()
    return _max_magnitude(pd.concat(frames, ignore_index=True))


def materialize_signed_membership(
    membership: pd.DataFrame,
    *,
    candidate_pathways: Iterable[str] | None = None,
) -> pd.DataFrame:
    required = {"gene_id", "pathway_id", "weight"}
    if missing := sorted(required - set(membership)):
        raise ValueError(f"Pathway membership table lacks columns: {missing}")
    values = membership.copy().reset_index(drop=True)
    if candidate_pathways is not None:
        values = values.loc[values.pathway_id.astype(str).isin(set(map(str, candidate_pathways)))]
    signed = _numeric(values.weight, name="pathway membership weight")
    values = values.loc[signed.ne(0)].copy()
    signed = signed.loc[values.index]
    frames = []
    for positive, relation, role, polarity in (
        (True, "member_of_positive", "static_signed_pathway_membership_positive", 1.0),
        (False, "member_of_negative", "static_signed_pathway_membership_negative", -1.0),
    ):
        selected = values.loc[signed.gt(0) if positive else signed.lt(0)]
        selected_weight = signed.loc[selected.index]
        if selected.empty:
            continue
        frames.append(
            _typed_edges(
                selected.gene_id,
                selected.pathway_id,
                source_type="gene",
                relation_type=relation,
                target_type="pathway",
                edge_role=role,
                source_split="static",
                weight=selected_weight.abs(),
                relation_polarity=polarity,
                source_database=selected.get("source_database", "static_pathway_annotation"),
            )
        )
    return _max_magnitude(pd.concat(frames, ignore_index=True)) if frames else _empty_edges()


def materialize_pathway_hierarchy(hierarchy: pd.DataFrame) -> pd.DataFrame:
    required = {"pathway_id", "pathway_family_id", "weight"}
    if missing := sorted(required - set(hierarchy)):
        raise ValueError(f"Pathway hierarchy table lacks columns: {missing}")
    hierarchy = hierarchy.copy().reset_index(drop=True)
    weight = _numeric(hierarchy.weight, name="hierarchy weight")
    if weight.lt(0).any():
        raise RuntimeError("Pathway hierarchy weights cannot be negative")
    selected = hierarchy.loc[weight.gt(0)].copy()
    weight = weight.loc[selected.index]
    return _max_magnitude(
        _typed_edges(
            selected.pathway_id,
            selected.pathway_family_id,
            source_type="pathway",
            relation_type="member_of_family",
            target_type="pathway_family",
            edge_role="static_pathway_hierarchy",
            source_split="static",
            weight=weight,
            relation_polarity=1.0,
            source_database=selected.get("source_database", "static_pathway_hierarchy"),
        )
    )


def materialize_global_lnc_protein_binding(
    binding: pd.DataFrame,
    *,
    candidate_lncrnas: Iterable[str] | None = None,
) -> pd.DataFrame:
    required = {"lncrna_id", "protein_id", "weight", "cancer_id", "is_context_specific"}
    if missing := sorted(required - set(binding)):
        raise ValueError(f"lncRNA-protein table lacks columns: {missing}")
    binding = binding.copy().reset_index(drop=True)
    context = binding.is_context_specific
    if not pd.api.types.is_bool_dtype(context.dropna().dtype) or context.isna().any():
        raise RuntimeError("lncRNA-protein context flag must be a non-null boolean")
    selected = binding.loc[binding.cancer_id.isna() & ~context].copy()
    if candidate_lncrnas is not None:
        selected = selected.loc[selected.lncrna_id.astype(str).isin(set(map(str, candidate_lncrnas)))]
    return _max_magnitude(
        _typed_edges(
            selected.lncrna_id,
            selected.protein_id,
            source_type="lncRNA",
            relation_type="binds_protein",
            target_type="protein",
            edge_role="static_global_lnc_protein_binding",
            source_split="static",
            weight=selected.weight,
            relation_polarity=1.0,
            source_database=selected.get("source_database", "static_global_binding"),
        )
    )


def materialize_protein_gene_encoding(protein_gene: pd.DataFrame) -> pd.DataFrame:
    required = {"protein_id", "gene_id", "weight"}
    if missing := sorted(required - set(protein_gene)):
        raise ValueError(f"Protein-gene table lacks columns: {missing}")
    protein_gene = protein_gene.copy().reset_index(drop=True)
    return _max_magnitude(
        _typed_edges(
            protein_gene.protein_id,
            protein_gene.gene_id,
            source_type="protein",
            relation_type="encoded_by",
            target_type="gene",
            edge_role="static_protein_gene_encoding",
            source_split="static",
            weight=protein_gene.weight,
            relation_polarity=1.0,
            source_database=protein_gene.get("source_database", "static_uniprot_mapping"),
        )
    )


def materialize_symmetric_ppi(ppi: pd.DataFrame) -> pd.DataFrame:
    required = {"protein_id_a", "protein_id_b", "weight"}
    if missing := sorted(required - set(ppi)):
        raise ValueError(f"PPI table lacks columns: {missing}")
    values = ppi.copy().reset_index(drop=True)
    left_raw = _text(values.protein_id_a)
    right_raw = _text(values.protein_id_b)
    if left_raw.isna().any() or right_raw.isna().any():
        raise RuntimeError("PPI contains null/empty endpoints")
    if left_raw.eq(right_raw).any():
        raise RuntimeError("PPI self-loops are forbidden")
    values["left"] = np.where(left_raw.lt(right_raw), left_raw, right_raw)
    values["right"] = np.where(left_raw.lt(right_raw), right_raw, left_raw)
    values["weight"] = _numeric(values.weight, name="PPI weight")
    values = values.sort_values(
        ["left", "right", "weight"], ascending=[True, True, False], kind="stable"
    ).drop_duplicates(["left", "right"], keep="first")
    forward = _typed_edges(
        values.left,
        values.right,
        source_type="protein",
        relation_type="physical_interaction",
        target_type="protein",
        edge_role="static_symmetric_ppi",
        source_split="static",
        weight=values.weight,
        relation_polarity=1.0,
        source_database=values.get("source_database", "static_string_ppi"),
    )
    reverse = forward.copy()
    reverse[["source_id", "target_id"]] = forward[["target_id", "source_id"]].to_numpy()
    result = pd.concat([forward, reverse], ignore_index=True)
    if len(result) != 2 * len(values):
        raise AssertionError("PPI reciprocal expansion changed row count")
    return result.sort_values(list(EDGE_KEYS), kind="stable").reset_index(drop=True)


def _empty_edges() -> pd.DataFrame:
    return _typed_edges(
        [], [], source_type="lncRNA", relation_type="coexpressed_positive",
        target_type="gene", edge_role="fold_train_coexpression_positive",
        source_split="train", weight=[], relation_polarity=1.0,
        cancer_id=[], is_context_specific=True, edge_outer_fold=0,
    ).iloc[0:0]


def _variant_mask(edges: pd.DataFrame, variant: str) -> pd.Series:
    arm = str(variant).upper()
    if arm not in VARIANTS:
        raise ValueError(f"Unknown graph variant: {variant}")
    role = edges.edge_role.astype(str)
    if arm == "G0":
        return ~(role.isin(G1_ROLES | G2_ONLY_ROLES))
    if arm == "G1":
        return ~role.isin(G2_ONLY_ROLES)
    return pd.Series(True, index=edges.index)


def _build_nodes(edges: pd.DataFrame, required_nodes: pd.DataFrame | None) -> pd.DataFrame:
    source = edges[["source_type", "source_id"]].rename(
        columns={"source_type": "node_type", "source_id": "canonical_id"}
    )
    target = edges[["target_type", "target_id"]].rename(
        columns={"target_type": "node_type", "target_id": "canonical_id"}
    )
    frames = [source, target]
    if required_nodes is not None:
        if missing := sorted({"node_type", "canonical_id"} - set(required_nodes)):
            raise ValueError(f"Required node inventory lacks columns: {missing}")
        frames.append(required_nodes[["node_type", "canonical_id"]])
    nodes = pd.concat(frames, ignore_index=True).dropna().astype(str).drop_duplicates()
    nodes = nodes.sort_values(["node_type", "canonical_id"], kind="stable").reset_index(drop=True)
    nodes["node_index_within_type"] = nodes.groupby("node_type", sort=False).cumcount().astype(np.int64)
    nodes["node_id"] = [
        "V32NODE:" + hashlib.sha256(f"{kind}\x1f{identifier}".encode()).hexdigest()
        for kind, identifier in nodes[["node_type", "canonical_id"]].itertuples(index=False, name=None)
    ]
    return nodes[["node_id", "node_type", "canonical_id", "node_index_within_type"]]


def build_formal_graph_authority(
    relations: Iterable[pd.DataFrame],
    *,
    outer_fold: int,
    required_nodes: pd.DataFrame | None = None,
) -> FormalGraphAuthority:
    frames = [frame.copy() for frame in relations if frame is not None and not frame.empty]
    if not frames:
        raise RuntimeError("Fresh formal graph has no relations")
    edges = pd.concat(frames, ignore_index=True)
    safe = build_safe_graph(edges, outer_fold=int(outer_fold))
    edges = safe.edges.copy()
    nodes = _build_nodes(edges, required_nodes)
    lookup = {
        (row.node_type, row.canonical_id): row.node_id
        for row in nodes.itertuples(index=False)
    }
    edges["source_canonical_id"] = edges.source_id.astype(str)
    edges["target_canonical_id"] = edges.target_id.astype(str)
    edges["source_node_id"] = [
        lookup.get((kind, identifier))
        for kind, identifier in edges[["source_type", "source_id"]].itertuples(index=False, name=None)
    ]
    edges["target_node_id"] = [
        lookup.get((kind, identifier))
        for kind, identifier in edges[["target_type", "target_id"]].itertuples(index=False, name=None)
    ]
    if edges[["source_node_id", "target_node_id"]].isna().any().any():
        raise RuntimeError("Fresh graph endpoint is absent from the frozen node inventory")
    edges["edge_id"] = edges.apply(_edge_id, axis=1)
    edges["relation_provenance"] = np.where(
        edges.is_context_specific, "cancer_unlabeled_context", "global_static"
    )
    edges["symmetric_same_relation"] = edges.edge_role.eq("static_symmetric_ppi")
    edges = edges.sort_values(list(EDGE_KEYS), kind="stable").reset_index(drop=True)
    masks = {arm: _variant_mask(edges, arm) for arm in VARIANTS}
    relation_schema = pd.DataFrame(
        FORMAL_RELATION_SCHEMA,
        columns=[
            "source_type",
            "relation_type",
            "target_type",
            "edge_role",
            "symmetric_same_relation",
        ],
    ).sort_values(
        ["source_type", "relation_type", "target_type"], kind="stable"
    ).reset_index(drop=True)
    observed_schema = set(
        edges[
            ["source_type", "relation_type", "target_type", "edge_role", "symmetric_same_relation"]
        ].drop_duplicates().itertuples(index=False, name=None)
    )
    registered_schema = set(relation_schema.itertuples(index=False, name=None))
    if not observed_schema.issubset(registered_schema):
        raise RuntimeError(
            f"Fresh graph emitted an unregistered typed relation: {observed_schema - registered_schema}"
        )
    manifest = {
        "format": FORMAL_GRAPH_FORMAT,
        "outer_fold": int(outer_fold),
        "node_counts": nodes.groupby("node_type", observed=True).size().sort_index().to_dict(),
        "g2_edge_count": int(len(edges)),
        "variant_edge_counts": {arm: int(mask.sum()) for arm, mask in masks.items()},
        "relation_counts": edges.groupby("relation_type", observed=True).size().sort_index().to_dict(),
        "ppi_directed_rows": int(edges.edge_role.eq("static_symmetric_ppi").sum()),
        "ppi_relation_types": int(
            edges.loc[edges.edge_role.eq("static_symmetric_ppi"), "relation_type"].nunique()
        ),
        "common_g0_node_features_required": True,
        "same_relation_schema_all_variants": True,
        "historical_graph_rows_used": False,
    }
    return FormalGraphAuthority(nodes, edges, safe, masks, relation_schema, manifest)


def variant_edges(authority: FormalGraphAuthority, variant: str) -> pd.DataFrame:
    arm = str(variant).upper()
    if arm not in authority.variant_masks:
        raise ValueError(f"Unknown graph variant: {variant}")
    return authority.edges.loc[authority.variant_masks[arm]].copy().reset_index(drop=True)


def build_variant_runtime_bundle(
    authority: FormalGraphAuthority,
    variant: str,
    *,
    edge_chunk_size: int = 250_000,
):
    """Build one arm with a resident backbone and rotating coexpression.

    The master G2 schedule determines chunk count/order for every arm.  Masked
    relation rows are removed from the active edge index (not assigned zero
    weights), while the complete G2 schema remains present as empty tensors.
    """

    from ..gnn import _materialize_graph_bundle

    master_schedule = schedule_runtime_edges(
        authority.edges,
        RuntimeSamplingPolicy(
            edge_chunk_size=int(edge_chunk_size),
            resident_backbone=True,
            variable_relation_types=("coexpressed_positive", "coexpressed_negative"),
        ),
    )
    allowed_roles = set(variant_edges(authority, variant).edge_role.astype(str))
    arm_schedule = master_schedule.loc[
        master_schedule.edge_role.astype(str).isin(allowed_roles)
    ].copy().reset_index(drop=True)
    arm_positions = runtime_chunk_positions(arm_schedule)
    chunks = sorted(arm_positions)
    expected = list(range(len(runtime_chunk_positions(master_schedule))))
    if chunks != expected:
        raise RuntimeError(f"Variant {variant} lost a runtime chunk: {chunks} != {expected}")
    active = arm_schedule.iloc[arm_positions[0]].copy()
    common_g0 = variant_edges(authority, "G0")
    return _materialize_graph_bundle(
        authority.nodes,
        active,
        feature_edges=common_g0,
        canonical_edges=variant_edges(authority, variant),
        runtime_schedule=arm_schedule,
        relation_schema_edges=authority.relation_schema,
        runtime_chunk_positions=arm_positions,
        active_runtime_chunk=0,
        graph_contract="CONTRACT-S",
    )
