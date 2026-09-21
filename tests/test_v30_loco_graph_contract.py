from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cc_hhgt.gnn import build_model, filter_fold_edges, fold_node_features


def test_heldout_cancer_keeps_only_expression_and_reference_context_is_removed() -> None:
    edges = pd.DataFrame(
        {
            "source_type": ["lncRNA"] * 6,
            "target_type": ["cancer", "cancer", "state", "cancer", "state", "cancer"],
            "source_canonical_id": ["L1", "L2", "ACC", "L1", "REF_A", "L3"],
            "target_canonical_id": ["ACC", "ACC", "S1", "BRCA", "S1", "REF_A"],
            "relation_type": ["expressed_in", "expressed_in", "state_enriched", "expressed_in", "state_enriched", "expressed_in"],
            "cancer_id": ["ACC", "ACC", "ACC", "BRCA", "REF_A", "REF_A"],
            "is_context_specific": [True] * 6,
            "target_independent": [True, True, False, True, False, True],
        }
    )
    filtered = filter_fold_edges(edges, heldout_cancers={"ACC"}, reference_only={"REF_A", "REF_B"})
    assert set(filtered.loc[filtered.cancer_id.eq("ACC"), "relation_type"]) == {"expressed_in"}
    assert not filtered.cancer_id.eq("REF_A").any()
    assert filtered.cancer_id.eq("BRCA").any()


def test_heldout_cancer_features_are_deterministic_and_distinguishable() -> None:
    cancers = pd.DataFrame({"canonical_id": ["ACC", "BRCA"]})
    edges = pd.DataFrame(
        {
            "source_type": ["lncRNA", "lncRNA", "lncRNA"],
            "target_type": ["cancer", "cancer", "cancer"],
            "source_canonical_id": ["L1", "L2", "L3"],
            "target_canonical_id": ["ACC", "ACC", "BRCA"],
        }
    )
    first = fold_node_features(cancers, edges, "cancer")
    second = fold_node_features(cancers, edges, "cancer")
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first[0], first[1])


def test_rgcn_logits_change_when_only_edge_weights_change() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    torch.manual_seed(7)
    bundle = SimpleNamespace(
        homogeneous={"num_nodes": 4, "num_relations": 1},
    )
    cfg = {
        "training": {
            "hidden_channels": 8,
            "num_layers": 1,
            "num_heads": 1,
            "dropout": 0.0,
            "pair_hidden_channels": 4,
        }
    }
    model = build_model("rgcn", bundle, feature_dim=0, cfg=cfg).eval()
    base = {
        "edge_index": torch.tensor([[0, 1, 2], [3, 3, 3]], dtype=torch.long),
        "edge_type": torch.zeros(3, dtype=torch.long),
        "edge_weight": torch.tensor([1.0, 1.0, 1.0]),
        "node_x": torch.tensor([[0.0, 1.0], [1.0, 1.0], [3.0, 1.0], [0.0, 1.0]]),
    }
    changed = {**base, "edge_weight": torch.tensor([1.0, 1.0, 8.0])}
    batch = {"l": torch.tensor([0]), "p": torch.tensor([1]), "c": torch.tensor([3]), "x": torch.empty((1, 0))}
    with torch.no_grad():
        first = model(base, batch)[0]
        second = model(changed, batch)[0]
    assert not torch.allclose(first, second)


@pytest.mark.parametrize("kind", ["hgt", "cc_hhgt"])
def test_heterogeneous_logits_change_when_only_edge_weights_change(kind: str) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from torch_geometric.data import HeteroData

    torch.manual_seed(11)
    data = HeteroData()
    data["lncRNA"].x = torch.tensor([[0.0, 1.0], [1.0, 1.0], [3.0, 1.0]])
    data["pathway_family"].x = torch.tensor([[0.5, 1.0]])
    data["cancer"].x = torch.tensor([[0.2, 1.0]])
    relation = ("lncRNA", "expressed_in", "cancer")
    data[relation].edge_index = torch.tensor([[0, 1, 2], [0, 0, 0]], dtype=torch.long)
    data[relation].edge_weight = torch.tensor([1.0, 1.0, 1.0])
    bundle = SimpleNamespace(hetero_data=data)
    cfg = {
        "training": {
            "hidden_channels": 8,
            "num_layers": 1,
            "num_heads": 2,
            "dropout": 0.0,
            "pair_hidden_channels": 4,
        }
    }
    model = build_model(kind, bundle, feature_dim=2, cfg=cfg).eval()
    changed = data.clone()
    changed[relation].edge_weight = torch.tensor([1.0, 1.0, 8.0])
    batch = {
        "l": torch.tensor([0]),
        "p": torch.tensor([0]),
        "c": torch.tensor([0]),
        "x": torch.tensor([[0.1, 0.2]]),
    }
    with torch.no_grad():
        first = model(data, batch)[0]
        second = model(changed, batch)[0]
    assert not torch.allclose(first, second)
