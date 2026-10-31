from __future__ import annotations

import copy

import numpy as np
import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from cla_engine.model import ARCHITECTURE, JointHGT, loss_one, to_graph
from cc_hhgt.v32.joint_model import TASK_WEIGHTS


def _fixture():
    torch.manual_seed(19)
    nodes = {
        "lncRNA": ["L0", "L1"],
        "pathway": ["P0", "P1"],
        "cancer": ["C0"],
        "pathway_family": ["F0"],
        "gene": ["G0", "G1"],
        "protein": ["R0", "R1"],
    }
    relations = [
        ("lncRNA", "targets", "pathway"),
        ("pathway", "rev_targets", "lncRNA"),
        ("cancer", "conditions", "lncRNA"),
        ("lncRNA", "rev_conditions", "cancer"),
        ("gene", "gene_context", "lncRNA"),
        ("protein", "protein_context", "pathway"),
    ]
    spec = {
        "architecture": ARCHITECTURE,
        "nodes": nodes,
        "relations": relations,
        "cancers": ["C0"],
        "states": ["RNAss", "ImmuneScore"],
        "state_families": {"RNAss": "stemness", "ImmuneScore": "immune"},
        "feature_dims": {kind: 6 for kind in nodes},
        "sc_dim": 1,
        "singlecell_role": "interpretation_only",
        "genomics_enabled": True,
        "numeric_features": 8,
    }
    config = {
        "task_weights": {"pathway": 1.0, "state": 0.5},
        "hidden": 8,
        "heads": 2,
        "layers": 2,
        "dropout": 0.0,
        "residual_bound": 1.0,
        "gate_penalty": 0.0,
    }
    arrays = {}
    for offset, (kind, ids) in enumerate(nodes.items(), start=1):
        values = np.arange(len(ids) * 6, dtype=np.float32).reshape(len(ids), 6)
        arrays[f"x::{kind}"] = (values + offset) / 17.0
    edge_values = {
        relations[0]: np.array([[0, 1], [0, 1]], dtype=np.int64),
        relations[1]: np.array([[0, 1], [0, 1]], dtype=np.int64),
        relations[2]: np.array([[0, 0], [0, 1]], dtype=np.int64),
        relations[3]: np.array([[0, 1], [0, 0]], dtype=np.int64),
        relations[4]: np.array([[0, 1], [0, 1]], dtype=np.int64),
        relations[5]: np.array([[0, 1], [0, 1]], dtype=np.int64),
    }
    for relation, edge_index in edge_values.items():
        key = "|".join(relation)
        arrays[f"edge::{key}"] = edge_index
        arrays[f"weight::{key}"] = np.ones(edge_index.shape[1], dtype=np.float32)
    model = JointHGT(spec, config)
    graph = to_graph(arrays, "cpu")
    batches = {
        "pathway": {
            "lnc": torch.tensor([0, 1]),
            "target": torch.tensor([0, 1]),
            "numeric": torch.tensor([[0.2, 0.3, 0.4, 0.5, 0, 0, 0, 0],
                                     [0.5, 0.4, 0.3, 0.2, 0, 0, 0, 0]]),
            "base": torch.tensor([-0.3, 0.4]),
            "label": torch.tensor([1.0, 0.0]),
            "available": torch.tensor([True, True]),
            "weight": torch.tensor([1.0, 1.0]),
            "direction_label": torch.zeros(2),
            "direction_available": torch.zeros(2, dtype=torch.bool),
        },
        "state": {
            "lnc": torch.tensor([0, 1]),
            "target": torch.tensor([0, 1]),
            "numeric": torch.tensor([[0.4, 0.2, 0.6, 0.3, 0, 0, 0, 0],
                                     [0.3, 0.6, 0.2, 0.4, 0, 0, 0, 0]]),
            "base": torch.tensor([0.2, -0.2]),
            "label": torch.tensor([0.0, 1.0]),
            "available": torch.tensor([True, True]),
            "weight": torch.tensor([1.0, 1.0]),
            "direction_label": torch.zeros(2),
            "direction_available": torch.zeros(2, dtype=torch.bool),
        },
    }
    return model, graph, batches


def _losses(model, graph, batches):
    encoded = model.encode(graph, 0)
    outputs = {task: model.forward_task(task, encoded, batch) for task, batch in batches.items()}
    return {
        task: loss_one(outputs[task], batches[task], direction_weight=0.0)
        for task in ("pathway", "state")
    }


def _maximum_norm(gradients):
    values = [gradient.detach().float().norm() for gradient in gradients if gradient is not None]
    assert values
    assert all(torch.isfinite(value) for value in values)
    return max(float(value) for value in values)


def _changed(before, parameters):
    return any(not torch.equal(old, new.detach()) for old, new in zip(before, parameters, strict=True))


def test_joint_bce_losses_reach_one_shared_hgt_with_frozen_weights():
    model, graph, batches = _fixture()
    assert TASK_WEIGHTS == {"pathway": 1.0, "state": 0.5}

    # Both bounded residual outputs intentionally start at zero to reproduce
    # the exact LASSO fallback.  The first update unlocks those output maps;
    # subsequent task losses must then reach the live shared HGT.
    warmup = torch.optim.SGD(model.parameters(), lr=0.1)
    first = _losses(model, graph, batches)
    warmup.zero_grad(set_to_none=True)
    (first["pathway"] + 0.5 * first["state"]).backward()
    warmup.step()

    losses = _losses(model, graph, batches)
    shared = list(model.layers[0].parameters())
    pathway_gradient = torch.autograd.grad(losses["pathway"], shared, retain_graph=True, allow_unused=True)
    state_gradient = torch.autograd.grad(losses["state"], shared, retain_graph=True, allow_unused=True)
    combined_gradient = torch.autograd.grad(
        losses["pathway"] + TASK_WEIGHTS["state"] * losses["state"],
        shared,
        retain_graph=True,
        allow_unused=True,
    )
    assert _maximum_norm(pathway_gradient) > 0.0
    assert _maximum_norm(state_gradient) > 0.0
    assert _maximum_norm(combined_gradient) > 0.0
    for pathway_value, state_value, combined_value in zip(
        pathway_gradient, state_gradient, combined_gradient, strict=True
    ):
        if combined_value is None:
            assert pathway_value is None and state_value is None
            continue
        expected = torch.zeros_like(combined_value)
        if pathway_value is not None:
            expected = expected + pathway_value
        if state_value is not None:
            expected = expected + 0.5 * state_value
        torch.testing.assert_close(combined_value, expected, rtol=1e-5, atol=1e-7)

    pathway_head = list(model.joint.pathway_model.residual_map.parameters()) + list(
        model.joint.pathway_model.residual_output.parameters()
    )
    state_head = list(model.joint.state_residual_map.parameters()) + list(
        model.joint.state_residual_output.parameters()
    )
    before_shared = [value.detach().clone() for value in shared]
    before_pathway = [value.detach().clone() for value in pathway_head]
    before_state = [value.detach().clone() for value in state_head]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    combined = _losses(model, graph, batches)
    (combined["pathway"] + 0.5 * combined["state"]).backward()
    optimizer.step()
    assert _changed(before_shared, shared)
    assert _changed(before_pathway, pathway_head)
    assert _changed(before_state, state_head)


def test_joint_context_keeps_six_channels_and_rejects_invalid_context():
    model, graph, _ = _fixture()
    for node_type, projection in model.context_projection.items():
        assert projection.in_features == 6
        assert graph["features"][node_type].shape[1] == 6
        # The public GNN still owns its historical degree channels; the six
        # current context channels are additive rather than replacing x.
        assert graph["data"][node_type].x.shape[1] == 2

    valid = model.encode(graph, 0)
    assert all(torch.isfinite(value).all() for value in valid[0].values())

    wrong_shape = copy.deepcopy(graph["data"])
    wrong_shape["lncRNA"].r4_context_embedding = torch.zeros(2, 9)
    with pytest.raises(RuntimeError, match="R4 context shape mismatch: lncRNA"):
        model.encoder.encode(wrong_shape)

    nonfinite = copy.deepcopy(graph["data"])
    nonfinite["lncRNA"].r4_context_embedding = torch.zeros(2, 8)
    nonfinite["lncRNA"].r4_context_embedding[0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="R4 context is nonfinite: lncRNA"):
        model.encoder.encode(nonfinite)
