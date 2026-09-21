from __future__ import annotations

from collections import defaultdict, deque
import sys
import types

import pandas as pd
import pytest

from cc_hhgt.relation_sampling import (
    RuntimeSamplingPolicy,
    iter_runtime_chunks,
    runtime_chunk_positions,
    schedule_runtime_edges,
)
from cc_hhgt.v32.formal_graph import (
    build_formal_graph_authority,
    build_variant_runtime_bundle,
    materialize_expressed_in,
    materialize_global_lnc_protein_binding,
    materialize_pathway_hierarchy,
    materialize_protein_gene_encoding,
    materialize_signed_coexpression,
    materialize_signed_membership,
    materialize_symmetric_ppi,
    variant_edges,
)
from cc_hhgt.v32.safe_graph import build_safe_graph


def _sources():
    candidate = pd.DataFrame(
        {
            "cancer_id": ["BRCA"] * 5,
            "lncrna_id": ["L0a", "L0b", "L1", "L2", "L3"],
        }
    )
    detection = candidate.assign(detection_rate=[0.4, 0.5, 0.6, 0.7, 0.8])
    coexpression = pd.DataFrame(
        {
            "cancer_id": ["BRCA"] * 5,
            "lncrna_id": ["L0a", "L0b", "L3", "L3", "L3"],
            "gene_id": ["C0", "C0", "C1", "C2", "C3"],
            "rho": [0.8, -0.7, 0.6, -0.5, 0.4],
            "source_split": ["train"] * 5,
            "edge_outer_fold": [2] * 5,
        }
    )
    membership = pd.DataFrame(
        {
            "gene_id": ["C0", "C1", "C2", "C3", "G1", "G2"],
            "pathway_id": ["P0", "PC1", "PC2", "PC3", "P1", "P2"],
            "weight": [1.0, 1.0, -0.75, 0.5, 1.0, -1.0],
        }
    )
    hierarchy = pd.DataFrame(
        {
            "pathway_id": ["P0", "PC1", "PC2", "PC3", "P1", "P2", "ZERO"],
            "pathway_family_id": ["F0", "F1", "F2", "F3", "F1", "F2", "FZ"],
            "weight": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        }
    )
    binding = pd.DataFrame(
        {
            "lncrna_id": ["L1", "L2", "L2"],
            "protein_id": ["PR1", "PRA", "BAD_CONTEXT"],
            "weight": [0.9, 0.8, 1.0],
            "cancer_id": [pd.NA, pd.NA, "BRCA"],
            "is_context_specific": [False, False, True],
        }
    )
    encoding = pd.DataFrame(
        {
            "protein_id": ["PR1", "PRA", "PRB"],
            "gene_id": ["G1", "G1", "G2"],
            "weight": [1.0, 1.0, 1.0],
        }
    )
    # Both input directions and a duplicate are intentionally present.  The
    # canonicalizer must still emit exactly two messages for the one pair.
    ppi = pd.DataFrame(
        {
            "protein_id_a": ["PRB", "PRA", "PRA"],
            "protein_id_b": ["PRA", "PRB", "PRB"],
            "weight": [0.7, 0.6, 0.65],
        }
    )
    relations = [
        materialize_expressed_in(detection, candidate),
        materialize_signed_coexpression(coexpression, outer_fold=2),
        materialize_signed_membership(membership),
        materialize_pathway_hierarchy(hierarchy),
        materialize_global_lnc_protein_binding(binding, candidate_lncrnas=candidate.lncrna_id),
        materialize_protein_gene_encoding(encoding),
        materialize_symmetric_ppi(ppi),
    ]
    return candidate, relations


def _authority():
    _, relations = _sources()
    return build_formal_graph_authority(relations, outer_fold=2)


def _distance(frame: pd.DataFrame, start: str, target: str) -> int | None:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for row in frame.itertuples(index=False):
        adjacency[str(row.source_id)].add(str(row.target_id))
        # Every formal non-PPI source row is materialized with its reverse
        # message; PPI already contains both directions.
        if str(row.relation_type) != "physical_interaction":
            adjacency[str(row.target_id)].add(str(row.source_id))
    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        node, depth = queue.popleft()
        if node == target:
            return depth
        for neighbour in adjacency[node]:
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append((neighbour, depth + 1))
    return None


def test_fresh_materializers_preserve_context_sign_and_all_static_layers() -> None:
    authority = _authority()
    edges = authority.edges
    assert set(edges.loc[edges.relation_type.str.startswith("coexpressed"), "relation_type"]) == {
        "coexpressed_positive",
        "coexpressed_negative",
    }
    negative_membership = edges.loc[edges.relation_type.eq("member_of_negative")]
    assert set(negative_membership.target_id) == {"PC2", "P2"}
    assert negative_membership.relation_polarity.eq(-1.0).all()
    assert negative_membership.weight.gt(0).all()
    assert "ZERO" not in set(edges.source_id) | set(edges.target_id)
    assert set(edges.loc[edges.relation_type.eq("binds_protein"), "target_id"]) == {
        "PR1",
        "PRA",
    }
    assert "BAD_CONTEXT" not in set(edges.target_id)
    assert edges.relation_type.eq("encoded_by").sum() == 3
    ppi = edges.loc[edges.relation_type.eq("physical_interaction")]
    assert len(ppi) == 2
    assert set(zip(ppi.source_id, ppi.target_id, strict=True)) == {
        ("PRA", "PRB"),
        ("PRB", "PRA"),
    }
    assert ppi.weight.nunique() == 1
    assert ppi.symmetric_same_relation.all()


def test_g0_g1_g2_are_true_masks_over_one_node_and_relation_schema() -> None:
    authority = _authority()
    g0, g1, g2 = (variant_edges(authority, arm) for arm in ("G0", "G1", "G2"))
    assert not g0.edge_role.isin(
        ["static_global_lnc_protein_binding", "static_protein_gene_encoding", "static_symmetric_ppi"]
    ).any()
    assert g1.edge_role.isin(
        ["static_global_lnc_protein_binding", "static_protein_gene_encoding"]
    ).any()
    assert not g1.edge_role.eq("static_symmetric_ppi").any()
    assert g2.edge_role.eq("static_symmetric_ppi").sum() == 2
    assert authority.manifest["same_relation_schema_all_variants"] is True
    assert authority.manifest["historical_graph_rows_used"] is False


def test_resident_backbone_keeps_two_three_four_hop_paths_in_every_chunk() -> None:
    authority = _authority()
    schedule = schedule_runtime_edges(
        authority.edges,
        RuntimeSamplingPolicy(
            edge_chunk_size=2,
            resident_backbone=True,
            variable_relation_types=("coexpressed_positive", "coexpressed_negative"),
        ),
    )
    assert set(schedule.runtime_chunk) == {-1, 0, 1, 2}
    resident = schedule.loc[schedule.runtime_residency.eq("resident_backbone")]
    assert resident.canonical_edge_id.is_unique
    assert resident.runtime_chunk.eq(-1).all()
    assert len(runtime_chunk_positions(schedule)) == 3
    rotating = schedule.loc[schedule.runtime_residency.eq("rotating_variable")]
    assert rotating.canonical_edge_id.nunique() == len(rotating) == 5
    assert rotating.groupby("runtime_chunk").size().max() - rotating.groupby("runtime_chunk").size().min() <= 1
    for chunk in iter_runtime_chunks(schedule):
        assert _distance(chunk, "L1", "P1") == 3
        assert _distance(chunk, "L2", "P2") == 4
        owned = chunk.loc[chunk.runtime_residency.eq("rotating_variable")]
        for edge in owned.itertuples(index=False):
            expected_pathway = {
                "C0": "P0",
                "C1": "PC1",
                "C2": "PC2",
                "C3": "PC3",
            }[str(edge.target_id)]
            assert _distance(chunk, str(edge.source_id), expected_pathway) == 2


def test_runtime_schedule_is_invariant_to_input_row_order() -> None:
    authority = _authority()
    policy = RuntimeSamplingPolicy(
        edge_chunk_size=2,
        resident_backbone=True,
        variable_relation_types=("coexpressed_positive", "coexpressed_negative"),
    )
    first = schedule_runtime_edges(authority.edges, policy)
    second = schedule_runtime_edges(authority.edges.sample(frac=1, random_state=7), policy)
    columns = ["runtime_chunk", "runtime_order", "canonical_edge_id", "runtime_residency"]
    assert first[columns].equals(second[columns])


def test_safe_graph_rejects_context_erasure_and_non_train_coexpression() -> None:
    authority = _authority()
    expressed = authority.safe_graph.edges.loc[
        authority.safe_graph.edges.relation_type.eq("expressed_in")
    ].copy()
    expressed["is_context_specific"] = False
    with pytest.raises(RuntimeError, match="Unsafe edges"):
        build_safe_graph(expressed, outer_fold=2)

    _, relations = _sources()
    coexpression = next(
        frame for frame in relations if frame.relation_type.astype(str).str.startswith("coexpressed").any()
    ).copy()
    coexpression["source_split"] = "validation"
    with pytest.raises(RuntimeError, match="Unsafe edges"):
        build_safe_graph(coexpression, outer_fold=2)


def test_ppi_self_loop_is_a_hard_failure() -> None:
    with pytest.raises(RuntimeError, match="self-loops"):
        materialize_symmetric_ppi(
            pd.DataFrame({"protein_id_a": ["P"], "protein_id_b": ["P"], "weight": [1.0]})
        )


def test_tensor_materializer_uses_one_ppi_relation_and_signed_runtime_weights(monkeypatch) -> None:
    torch = pytest.importorskip("torch")

    class _Store:
        pass

    class _HeteroData:
        def __init__(self):
            self._stores = {}

        def __getitem__(self, key):
            return self._stores.setdefault(key, _Store())

        @property
        def edge_types(self):
            return [key for key in self._stores if isinstance(key, tuple)]

        @property
        def node_types(self):
            return [key for key in self._stores if isinstance(key, str)]

    package = types.ModuleType("torch_geometric")
    data_module = types.ModuleType("torch_geometric.data")
    data_module.HeteroData = _HeteroData
    package.data = data_module
    monkeypatch.setitem(sys.modules, "torch_geometric", package)
    monkeypatch.setitem(sys.modules, "torch_geometric.data", data_module)
    if "yaml" not in sys.modules:
        yaml_module = types.ModuleType("yaml")
        yaml_module.safe_load = lambda value: value
        yaml_module.safe_dump = lambda value, **kwargs: str(value)
        monkeypatch.setitem(sys.modules, "yaml", yaml_module)
    if "sklearn" not in sys.modules:
        sklearn_module = types.ModuleType("sklearn")
        metrics_module = types.ModuleType("sklearn.metrics")
        for name in (
            "average_precision_score",
            "brier_score_loss",
            "log_loss",
            "roc_auc_score",
        ):
            setattr(metrics_module, name, lambda *args, **kwargs: 0.0)
        sklearn_module.metrics = metrics_module
        monkeypatch.setitem(sys.modules, "sklearn", sklearn_module)
        monkeypatch.setitem(sys.modules, "sklearn.metrics", metrics_module)
    if "cc_hhgt.metrics" not in sys.modules:
        project_metrics = types.ModuleType("cc_hhgt.metrics")
        project_metrics.proxy_binary_metrics = lambda *args, **kwargs: {}
        monkeypatch.setitem(sys.modules, "cc_hhgt.metrics", project_metrics)
    monkeypatch.setattr("cc_hhgt.gnn.require_torch_geometric", lambda: torch)
    authority = _authority()
    bundle = build_variant_runtime_bundle(authority, "G2", edge_chunk_size=2)
    ppi_type = ("protein", "physical_interaction", "protein")
    assert ppi_type in bundle.hetero_data.edge_types
    assert ("protein", "rev_physical_interaction", "protein") not in bundle.hetero_data.edge_types
    assert bundle.hetero_data[ppi_type].edge_index.shape[1] == 2
    negative_type = ("gene", "member_of_negative", "pathway")
    assert bundle.hetero_data[negative_type].edge_weight.lt(0).all()
    # G0/G1 keep the same schema but masked relations have empty edge indices.
    g0 = build_variant_runtime_bundle(authority, "G0", edge_chunk_size=2)
    assert set(g0.hetero_data.edge_types) == set(bundle.hetero_data.edge_types)
    assert g0.hetero_data[ppi_type].edge_index.shape[1] == 0
    assert g0.hetero_data[("lncRNA", "binds_protein", "protein")].edge_index.shape[1] == 0
    for node_type in bundle.hetero_data.node_types:
        assert g0.hetero_data[node_type].x.equal(bundle.hetero_data[node_type].x)
