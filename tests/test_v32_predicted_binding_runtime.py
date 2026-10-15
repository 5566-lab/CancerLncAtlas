"""No optimizer: exercise real graph tensors, HGT/RGCN forwards and dropout."""
import copy
import numpy as np
import pandas as pd
import pytest
import torch

import cc_hhgt.gnn as gnn
from cc_hhgt.v32.formal_graph import build_formal_graph_authority, materialize_typed_lnc_protein_binding, variant_edges
from cc_hhgt.v32.training import _runtime_graph_for_chunk


def fixture(predicted_weight=.25):
    binding = pd.DataFrame({"lncrna_id": ["LNC:L1", "LNC:L1"], "protein_id": ["UNIPROT:P1", "UNIPROT:P2"],
        "cancer_id": [None, None], "is_context_specific": [False, False], "weight": [1., predicted_weight],
        "graph_assay_class": ["eclip", "predicted"], "source_database": ["ENCODE_eCLIP", "RBNS_SEQUENCE"]})
    edges = materialize_typed_lnc_protein_binding(binding, include_predicted=True, require_global_binding=True)
    authority = build_formal_graph_authority([edges], outer_fold=0)
    schema = authority.relation_schema.loc[authority.relation_schema.relation_type.isin(edges.relation_type)].copy()
    bundle = gnn._materialize_graph_bundle(authority.nodes, authority.edges, feature_edges=authority.edges.iloc[:0],
        canonical_edges=authority.edges, runtime_schedule=authority.edges, relation_schema_edges=schema,
        runtime_chunk_positions={0: np.arange(len(authority.edges))}, active_runtime_chunk=0)
    bundle.relation_edge_dropout = {"binds_protein_predicted": .5}
    bundle.binding_evidence_arm = "experimental_predicted"
    return bundle


def config():
    return {"training": {"hidden_channels": 8, "num_layers": 1, "num_heads": 2, "dropout": 0., "pair_hidden_channels": 4}}


def test_real_hgt_forward_excludes_predicted_topology_but_uses_scaled_messages():
    bundle = fixture(); model = gnn.build_model("cc_hhgt", bundle, 2, config()).eval()
    seen = []
    model.layers[0].register_forward_pre_hook(lambda module, args: seen.append(set(args[1])))
    with torch.inference_mode():
        output = model.encode(bundle.hetero_data)
        states = {key: torch.ones((len(ids), 8)) for key, ids in bundle.node_maps.items()}
        projection = {key: torch.nn.Identity() for key in states}
        messages = model.weighted_aggregate(states, bundle.hetero_data, projection)
    assert all("predicted" not in edge[1] for edge in seen[0])
    assert torch.isfinite(output["protein"]).all()
    assert messages["protein"][bundle.node_maps["protein"]["UNIPROT:P2"]].tolist() == pytest.approx([.25]*8)
    assert messages["protein"][bundle.node_maps["protein"]["UNIPROT:P1"]].tolist() == pytest.approx([1.]*8)


def test_real_rgcn_forward_has_only_measured_topology_and_preserved_predicted_weight():
    bundle = fixture(); model = gnn.build_model("rgcn", bundle, 2, config()).eval()
    graph = gnn.move_graph(bundle, "cpu", "rgcn")
    observed = []
    model.layers[0].register_forward_pre_hook(lambda module, args: observed.append(args[1].shape[1]))
    with torch.inference_mode(): result = model.encode(graph)
    assert observed == [2] and graph["edge_index"].shape[1] == 4
    assert graph["weighted_only_edge_mask"].sum() == 2
    states = torch.ones((len(bundle.nodes), 1))
    messages = gnn._weighted_homogeneous_aggregate(states, graph)
    protein_index = bundle.nodes.index[bundle.nodes.canonical_id.eq("UNIPROT:P2")][0]
    assert messages[protein_index, 0] == pytest.approx(.25)
    assert torch.isfinite(result).all()


def test_dropout_is_repeatable_order_invariant_and_validation_keeps_all_edges():
    bundle = fixture(); original = bundle.edges.copy()
    first = gnn.runtime_bundle_for_step(bundle, 0, apply_relation_edge_dropout=True,
        relation_edge_dropout_seed=7, relation_edge_dropout_step=12)
    second = gnn.runtime_bundle_for_step(bundle, 0, apply_relation_edge_dropout=True,
        relation_edge_dropout_seed=7, relation_edge_dropout_step=12)
    assert first.edges.edge_id.tolist() == second.edges.edge_id.tolist()
    reordered = copy.deepcopy(bundle)
    reordered.runtime_schedule = reordered.runtime_schedule.iloc[::-1].reset_index(drop=True)
    third = gnn.runtime_bundle_for_step(reordered, 0, apply_relation_edge_dropout=True,
        relation_edge_dropout_seed=7, relation_edge_dropout_step=12)
    assert set(first.edges.edge_id) == set(third.edges.edge_id)
    assert first.edges.relation_type.eq("binds_protein_eclip").sum() == 1
    assert len(gnn.runtime_bundle_for_step(bundle, 0).edges) == 2
    pd.testing.assert_frame_equal(original, bundle.edges)


def test_one_predicted_edge_can_really_drop_and_training_bridge_preserves_eval_default():
    bundle = fixture(); bundle.relation_edge_dropout = {"binds_protein_predicted": .999999}
    dropped = _runtime_graph_for_chunk(bundle, 0, device="cpu", apply_relation_edge_dropout=True,
                                       relation_edge_dropout_seed=7, relation_edge_dropout_step=0)
    evaluated = _runtime_graph_for_chunk(bundle, 0, device="cpu")
    key = ("lncRNA", "binds_protein_predicted", "protein")
    reverse = ("protein", "rev_binds_protein_predicted", "lncRNA")
    assert dropped[key].edge_index.shape[1] == dropped[reverse].edge_index.shape[1] == 0
    assert evaluated[key].edge_index.shape[1] == evaluated[reverse].edge_index.shape[1] == 1
    assert torch.equal(dropped["protein"].x, evaluated["protein"].x)


def test_empty_active_relations_keep_the_full_relation_schema():
    bundle = fixture()
    empty = gnn._materialize_graph_bundle(bundle.nodes, bundle.edges.iloc[:0], feature_edges=bundle.edges.iloc[:0],
                                          relation_schema_edges=bundle.relation_schema_edges)
    assert empty.homogeneous["num_relations"] == bundle.homogeneous["num_relations"] == 4
    assert set(empty.hetero_data.edge_types) == set(bundle.hetero_data.edge_types)
    assert empty.homogeneous["edge_index"].shape == (2, 0)


def test_existing_measured_hgt_forward_is_unchanged_when_predicted_relation_is_empty(monkeypatch):
    bundle = fixture(); bundle.relation_edge_dropout = {"binds_protein_predicted": .999999}
    graph = _runtime_graph_for_chunk(bundle, 0, device="cpu", apply_relation_edge_dropout=True,
                                    relation_edge_dropout_seed=7, relation_edge_dropout_step=0)
    model = gnn.build_model("cc_hhgt", bundle, 2, config()).eval()
    with torch.inference_mode(): actual = model.encode(graph)
    monkeypatch.setattr(gnn, "_topology_edge_index_dict", lambda value: value)
    with torch.inference_mode(): historical = model.encode(graph)
    for key in actual: torch.testing.assert_close(actual[key], historical[key], rtol=0, atol=0)


def test_measured_relation_dropout_or_unmasked_old_predicted_graph_is_rejected():
    bundle = fixture(); bundle.relation_edge_dropout = {"binds_protein_eclip": .5}
    with pytest.raises(RuntimeError, match="Only predicted"):
        gnn.runtime_bundle_for_step(bundle, 0, apply_relation_edge_dropout=True)
    bundle = fixture(); del bundle.homogeneous["topology_edge_index"]
    with pytest.raises(RuntimeError, match="weighted-only topology"):
        gnn.move_graph(bundle, "cpu", "rgcn")
