from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch


pytest.importorskip("torch_geometric")
from torch_geometric.nn import HGTConv

from cc_hhgt.v32 import single_cell_hhgt
from cc_hhgt.v32.single_cell_hhgt import (
    LASSO_EDGE_TYPES,
    NODE_TYPES,
    SingleCellHHGT,
    build_context_graph,
    optimizer_step,
)


EXTERNAL_SCHEMA = (("gene", "functional prior", "pathway"),)
EXTERNAL_EDGE_TYPE = ("gene", "external_functional_prior", "pathway")


def _features() -> dict[str, np.ndarray]:
    return {
        "lncRNA": np.asarray([[1.0, 0.2], [0.1, 1.2], [-0.4, 0.7]], np.float32),
        "gene": np.asarray([[0.3, 1.0], [1.4, -0.2], [-0.5, 0.8]], np.float32),
        "pathway": np.asarray([[0.7, -0.3], [-0.2, 1.1]], np.float32),
        "pathway_family": np.asarray([[0.5, 0.9]], np.float32),
        "cell_context": np.asarray([[0.25, -0.75, 1.5]], np.float32),
    }


def _external_edges() -> list[dict]:
    return [
        {
            "source_type": "gene",
            "relation": "functional prior",
            "target_type": "pathway",
            "source_index": 0,
            "target_index": 0,
            "confidence": 0.75,
            "sign": 1,
            "available": True,
            "source_hash": "source-a-sha256",
            "provenance": "source-a:test",
            "source_donors": ["d1"],
        },
        {
            "source_type": "gene",
            "relation": "functional prior",
            "target_type": "pathway",
            "source_index": 1,
            "target_index": 0,
            "confidence": 0.25,
            "sign": -1,
            "available": True,
            "source_hash": "source-b-sha256",
            "provenance": "source-b:test",
            "source_donors": ["d2"],
        },
    ]


def _graph(*, with_lasso: bool = False, external: bool = True):
    coefficient = None
    stability = None
    if with_lasso:
        coefficient = np.asarray(
            [[0.8, 0.0, 0.0], [0.0, -0.6, 0.0], [0.0, 0.0, 0.4]],
            np.float32,
        )
        stability = np.asarray(
            [[0.9, 0.0, 0.0], [0.0, 0.8, 0.0], [0.0, 0.0, 0.7]],
            np.float32,
        )
    return build_context_graph(
        _features(),
        np.asarray([[1, 0], [1, 1], [0, 1]], dtype=np.int8),
        np.asarray([[1], [1]], dtype=np.float32),
        membership_orientation="gene_by_pathway",
        pathway_family_orientation="pathway_by_family",
        lasso_coefficients=coefficient,
        lasso_stability=stability,
        external_edges=_external_edges() if external else [],
        external_relation_types=EXTERNAL_SCHEMA,
        source_donors=["d1", "d2"],
        excluded_donors=["heldout"],
    )


def _model(graph):
    torch.manual_seed(17)
    return SingleCellHHGT(
        graph,
        hidden_channels=16,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
        pair_feature_count=3,
    )


def _query(model, graph):
    return model(
        graph,
        torch.tensor([0, 1, 2]),
        torch.tensor([0, 0, 1]),
        torch.tensor([[0.1, 0.2, 0.3], [-0.2, 0.4, 0.8], [0.9, -0.1, 0.2]]),
    )


def test_fixed_five_node_schema_uses_self_contained_hgtconv():
    graph = _graph()
    model = _model(graph)
    assert tuple(graph.data.node_types) == NODE_TYPES
    assert graph.data["cell_context"].num_nodes == 1
    assert all(isinstance(layer, HGTConv) for layer in model.encoder.layers)
    source = inspect.getsource(single_cell_hhgt)
    assert "from cc_hhgt.gnn" not in source
    assert "import cc_hhgt.gnn" not in source


def test_membership_orientation_and_donor_leak_fail_closed():
    kwargs = {
        "pathway_family_membership": np.asarray([[1], [1]]),
        "membership_orientation": "gene_by_pathway",
        "pathway_family_orientation": "pathway_by_family",
        "source_donors": ["d1", "d2"],
        "excluded_donors": ["heldout"],
    }
    membership = np.asarray([[1, 0], [1, 1], [0, 1]])
    with pytest.raises(ValueError, match="GENE_PATHWAY_MEMBERSHIP_TRANSPOSED"):
        build_context_graph(_features(), membership.T, **kwargs)
    with pytest.raises(ValueError, match="ORIENTATION_REQUIRED"):
        build_context_graph(
            _features(),
            membership,
            kwargs["pathway_family_membership"],
            pathway_family_orientation="pathway_by_family",
            source_donors=["d1"],
            excluded_donors=["heldout"],
        )
    with pytest.raises(ValueError, match="SC_GRAPH_DONOR_LEAK"):
        build_context_graph(
            _features(),
            membership,
            membership_orientation="gene_by_pathway",
            pathway_family_membership=kwargs["pathway_family_membership"],
            pathway_family_orientation="pathway_by_family",
            source_donors=["d1", "heldout"],
            excluded_donors=["heldout"],
        )

    leaking = _external_edges()
    leaking[0] = dict(leaking[0], source_donors=["heldout"])
    with pytest.raises(ValueError, match="EXTERNAL_EDGE_HELDOUT_DONOR_LEAK"):
        build_context_graph(
            _features(),
            membership,
            kwargs["pathway_family_membership"],
            membership_orientation="gene_by_pathway",
            pathway_family_orientation="pathway_by_family",
            external_edges=leaking,
            external_relation_types=EXTERNAL_SCHEMA,
            source_donors=["d1", "d2"],
            excluded_donors=["heldout"],
        )


def test_with_and_without_lasso_have_identical_schema():
    without_lasso = _graph(with_lasso=False)
    with_lasso = _graph(with_lasso=True)
    assert without_lasso.edge_schema == with_lasso.edge_schema
    assert tuple(without_lasso.data.edge_types) == tuple(with_lasso.data.edge_types)
    assert all(
        without_lasso.data[edge_type].edge_index.shape[1] == 0
        for edge_type in LASSO_EDGE_TYPES
    )
    assert sum(
        with_lasso.data[edge_type].edge_index.shape[1]
        for edge_type in LASSO_EDGE_TYPES
    ) == 6


def test_sparse_lasso_coo_matches_dense_edges_without_dense_allocation():
    dense = _graph(with_lasso=True, external=False)
    coefficient = np.asarray(
        [[0.8, 0.0, 0.0], [0.0, -0.6, 0.0], [0.0, 0.0, 0.4]], np.float32
    )
    stability = np.asarray(
        [[0.9, 0.0, 0.0], [0.0, 0.8, 0.0], [0.0, 0.0, 0.7]], np.float32
    )
    row, column = np.nonzero(coefficient)
    sparse_graph = build_context_graph(
        _features(),
        np.asarray([[1, 0], [1, 1], [0, 1]], dtype=np.int8),
        np.asarray([[1], [1]], dtype=np.float32),
        membership_orientation="gene_by_pathway",
        pathway_family_orientation="pathway_by_family",
        sparse_lasso_edges={
            "lnc_index": row,
            "gene_index": column,
            "coefficient": coefficient[row, column],
            "stability": stability[row, column],
        },
        external_relation_types=EXTERNAL_SCHEMA,
        source_donors=["d1", "d2"],
        excluded_donors=["heldout"],
    )
    for edge_type in LASSO_EDGE_TYPES:
        torch.testing.assert_close(
            sparse_graph.data[edge_type].edge_index,
            dense.data[edge_type].edge_index,
        )
        torch.testing.assert_close(
            sparse_graph.data[edge_type].edge_weight,
            dense.data[edge_type].edge_weight,
        )


def test_many_to_many_pathway_family_membership_preserves_all_edges():
    features = _features()
    features["pathway_family"] = np.asarray(
        [[0.5, 0.9], [-0.1, 0.4], [0.2, -0.6]], np.float32
    )
    hierarchy = np.asarray([[1.0, 0.5, 0.0], [0.0, 1.0, 0.25]], np.float32)
    graph = build_context_graph(
        features,
        np.asarray([[1, 0], [1, 1], [0, 1]], dtype=np.int8),
        hierarchy,
        membership_orientation="gene_by_pathway",
        pathway_family_orientation="pathway_by_family",
        source_donors=["d1", "d2"],
        excluded_donors=["heldout"],
    )
    edge_type = ("pathway", "belongs_to", "pathway_family")
    assert graph.data[edge_type].edge_index.tolist() == [[0, 0, 1, 1], [0, 1, 1, 2]]
    torch.testing.assert_close(
        graph.data[edge_type].edge_weight, torch.tensor([1.0, 0.5, 1.0, 0.25])
    )
    with pytest.raises(ValueError, match="PATHWAY_FAMILY_MEMBERSHIP_TRANSPOSED"):
        build_context_graph(
            features,
            np.asarray([[1, 0], [1, 1], [0, 1]], dtype=np.int8),
            hierarchy.T,
            membership_orientation="gene_by_pathway",
            pathway_family_orientation="pathway_by_family",
            source_donors=["d1", "d2"],
            excluded_donors=["heldout"],
        )


def test_multiple_external_edges_on_same_relation_are_accumulated():
    graph = _graph()
    edge_index = graph.data[EXTERNAL_EDGE_TYPE].edge_index
    edge_weight = graph.data[EXTERNAL_EDGE_TYPE].edge_weight
    assert edge_index.tolist() == [[0, 1], [0, 0]]
    torch.testing.assert_close(edge_weight, torch.tensor([0.75, -0.25]))
    assert len(graph.external_edge_audit) == 2


def test_columnar_external_batch_matches_record_edges():
    records = _graph()
    batched = build_context_graph(
        _features(),
        np.asarray([[1, 0], [1, 1], [0, 1]], dtype=np.int8),
        np.asarray([[1], [1]], dtype=np.float32),
        membership_orientation="gene_by_pathway",
        pathway_family_orientation="pathway_by_family",
        external_edge_batches=[
            {
                "source_type": "gene",
                "relation": "functional prior",
                "target_type": "pathway",
                "edge_index": [[0, 1], [0, 0]],
                "edge_weight": [0.75, -0.25],
                "source_hash": "columnar-source-sha256",
                "provenance": "columnar:test",
                "source_donors": ["d1", "d2"],
            }
        ],
        external_relation_types=EXTERNAL_SCHEMA,
        source_donors=["d1", "d2"],
        excluded_donors=["heldout"],
    )
    torch.testing.assert_close(
        batched.data[EXTERNAL_EDGE_TYPE].edge_index,
        records.data[EXTERNAL_EDGE_TYPE].edge_index,
    )
    torch.testing.assert_close(
        batched.data[EXTERNAL_EDGE_TYPE].edge_weight,
        records.data[EXTERNAL_EDGE_TYPE].edge_weight,
    )
    assert batched.external_edge_audit[0]["edge_count"] == 2


def test_continuous_feature_intervention_changes_encoding_and_prediction():
    graph = _graph()
    model = _model(graph).eval()
    changed = graph.clone()
    changed.data["cell_context"].continuous_x[0] += torch.tensor([4.0, -3.0, 2.0])
    with torch.no_grad():
        before_embedding = model.encoder.encode(graph)["cell_context"].clone()
        after_embedding = model.encoder.encode(changed)["cell_context"].clone()
        before_prediction = _query(model, graph)
        after_prediction = _query(model, changed)
    assert torch.max(torch.abs(before_embedding - after_embedding)).item() > 1e-6
    assert torch.max(torch.abs(before_prediction - after_prediction)).item() > 1e-6


def test_edge_weight_intervention_changes_encoding_without_topology_change():
    graph = _graph()
    model = _model(graph).eval()
    changed = graph.clone()
    original_index = graph.data[EXTERNAL_EDGE_TYPE].edge_index.clone()
    changed.data[EXTERNAL_EDGE_TYPE].edge_weight = torch.tensor([-0.10, 0.90])
    with torch.no_grad():
        before_embedding = model.encoder.encode(graph)["pathway"].clone()
        after_embedding = model.encoder.encode(changed)["pathway"].clone()
        before_prediction = _query(model, graph)
        after_prediction = _query(model, changed)
    torch.testing.assert_close(changed.data[EXTERNAL_EDGE_TYPE].edge_index, original_index)
    assert torch.max(torch.abs(before_embedding - after_embedding)).item() > 1e-6
    assert torch.max(torch.abs(before_prediction - after_prediction)).item() > 1e-6


def test_signed_tanh_output_and_optimizer_step_update_hgt_parameters():
    graph = _graph(with_lasso=True)
    signed_model = _model(graph).eval()
    with torch.no_grad():
        for parameter in signed_model.decoder.parameters():
            parameter.zero_()
        signed_model.decoder[-1].bias.fill_(-0.75)
        signed = _query(signed_model, graph)
    assert bool((signed < 0).all())
    assert bool((signed >= -1).all() and (signed <= 1).all())

    model = _model(graph)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    before = [parameter.detach().clone() for parameter in model.encoder.layers.parameters()]
    report = optimizer_step(
        model,
        optimizer,
        graph,
        [0, 1, 2],
        [0, 0, 1],
        [[0.1, 0.2, 0.3], [-0.2, 0.4, 0.8], [0.9, -0.1, 0.2]],
        [-0.8, 0.6, -0.4],
    )
    after = list(model.encoder.layers.parameters())
    assert report["optimizer_steps"] == 1
    assert report["hgt_gradient_l1"] > 0
    assert report["weighted_projection_gradient_l1"] > 0
    assert report["continuous_projection_gradient_l1"] > 0
    assert report["decoder_gradient_l1"] > 0
    assert any(not torch.equal(old, new.detach()) for old, new in zip(before, after))
