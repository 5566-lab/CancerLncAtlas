"""Self-contained regression for the current real-PyG joint backend.

The check never loads a historical source snapshot. It verifies that the
current additive context hook preserves the current encoder when context is
absent or exactly zero, then exercises the active joint BCE path far enough to
prove that both membership losses reach the same live HGT and that a guarded
optimizer step produces a representable parameter update.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .common import atomic_json, fail, status
from .model import ARCHITECTURE, JointHGT, loss_one, to_graph


def _graph_fixture(device):
    ids = {
        "lncRNA": ["L0", "L1", "L2"],
        "gene": ["G0", "G1", "G2"],
        "protein": ["P0", "P1", "P2"],
        "pathway": ["W0", "W1"],
        "pathway_family": ["F0"],
        "cancer": ["C0"],
    }
    relations = [
        ("lncRNA", "binds", "protein"),
        ("protein", "rev_binds", "lncRNA"),
        ("protein", "PPI", "protein"),
        ("protein", "encoded_by", "gene"),
        ("gene", "rev_encoded_by", "protein"),
        ("gene", "member", "pathway"),
        ("pathway", "rev_member", "gene"),
        ("pathway", "member_of_family", "pathway_family"),
        ("pathway_family", "rev_member_of_family", "pathway"),
        ("lncRNA", "measured_in", "cancer"),
        ("cancer", "rev_measured_in", "lncRNA"),
    ]
    rng = np.random.default_rng(4)
    arrays = {
        f"x::{node_type}": rng.normal(size=(len(values), 6)).astype("float32")
        for node_type, values in ids.items()
    }
    for source_type, relation, target_type in relations:
        count = max(len(ids[source_type]), len(ids[target_type]))
        index = np.vstack(
            [
                np.arange(count) % len(ids[source_type]),
                np.arange(count) % len(ids[target_type]),
            ]
        )
        key = "|".join((source_type, relation, target_type))
        arrays["edge::" + key] = index.astype("int64")
        arrays["weight::" + key] = np.linspace(0.5, 1.0, count, dtype="float32")
    arrays.update(
        singlecell=rng.normal(size=(3, 3)).astype("float32"),
        sc_available=np.ones(3, "float32"),
        mutation=rng.normal(size=(3, 3)).astype("float32"),
        mutation_available=np.ones(3, "float32"),
    )
    return ids, relations, arrays, to_graph(arrays, device)


def _current_context_equivalence(graph, *, device, hgtconv):
    from cc_hhgt.gnn import build_model

    bundle = SimpleNamespace(hetero_data=graph["data"])
    checks = []
    for layers in (2, 3):
        config = {
            "training": {
                "hidden_channels": 8,
                "num_layers": layers,
                "num_heads": 2,
                "dropout": 0.0,
                "pair_hidden_channels": 8,
            },
            "task_contract": {"pathway": {"target_level": "exact_pathway"}},
        }
        torch.manual_seed(19)
        model = build_model("cc_hhgt", bundle, 5, config).to(device)
        model.eval()
        if not all(isinstance(layer, hgtconv) for layer in model.layers):
            fail("BACKBONE_SUBSTITUTED", "Current encoder must use real torch_geometric HGTConv")

        plain = graph["data"].clone()
        zero = graph["data"].clone()
        for node_type, values in plain.x_dict.items():
            zero[node_type].r4_context_embedding = torch.zeros(
                (values.shape[0], 8), dtype=values.dtype, device=values.device
            )
        plain_encoded = model.encode(plain)
        zero_encoded = model.encode(zero)
        for node_type in plain_encoded:
            torch.testing.assert_close(
                plain_encoded[node_type], zero_encoded[node_type], rtol=0.0, atol=0.0
            )

        model.zero_grad(set_to_none=True)
        sum(value.square().sum() for value in plain_encoded.values()).backward()
        plain_gradients = {
            name: None if parameter.grad is None else parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
        }
        model.zero_grad(set_to_none=True)
        zero_encoded = model.encode(zero)
        sum(value.square().sum() for value in zero_encoded.values()).backward()
        zero_gradients = {
            name: None if parameter.grad is None else parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
        }
        for name in plain_gradients:
            left, right = plain_gradients[name], zero_gradients[name]
            if left is None or right is None:
                if left is not None or right is not None:
                    fail("ZERO_CONTEXT_GRADIENT_DRIFT", name)
            else:
                torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
        checks.append(
            {
                "layers": layers,
                "forward_exact": True,
                "gradient_exact": True,
                "comparison": "CURRENT_ENCODER_NO_CONTEXT_VS_EXACT_ZERO_CONTEXT",
            }
        )
    return checks


def _joint_batch(device):
    return {
        "lnc": torch.tensor([0, 1, 2, 0], device=device),
        "target": torch.tensor([0, 1, 0, 1], device=device),
        "numeric": torch.ones(4, 5, device=device),
        "base": torch.zeros(4, device=device),
        "label": torch.tensor([0.0, 1.0, 0.0, 1.0], device=device),
        "available": torch.ones(4, dtype=torch.bool, device=device),
        "weight": torch.ones(4, device=device),
        "direction_label": torch.tensor([0.0, 1.0, 0.0, 1.0], device=device),
        "direction_available": torch.ones(4, dtype=torch.bool, device=device),
    }


def _joint_losses(model, graph, batch):
    encoded = model.encode(graph, 0)
    return {
        task: loss_one(model.forward_task(task, encoded, batch), batch, direction_weight=0.0)
        for task in ("pathway", "state")
    }


def run(output, device="cpu"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    try:
        import torch_geometric
        from torch_geometric.nn import HGTConv
    except ImportError:
        fail(
            "PYG_REQUIRED_NO_FALLBACK",
            "Actual torch_geometric is required; not replaced by an MLP or native implementation",
        )
    if device.startswith("cuda") and not torch.cuda.is_available():
        fail("CUDA_UNAVAILABLE", "Requested CUDA backend is unavailable")

    from cc_hhgt.v32.optimizer_guard_v2 import optimizer_step_with_global_delta_guard

    ids, relations, _arrays, graph = _graph_fixture(device)
    equivalence = _current_context_equivalence(graph, device=device, hgtconv=HGTConv)
    spec = {
        "architecture": ARCHITECTURE,
        "nodes": ids,
        "relations": relations,
        "cancers": ["C0"],
        "states": ["S0", "S1"],
        "numeric_features": 5,
        "feature_dims": {key: 6 for key in ids},
        "sc_dim": 3,
        "singlecell_role": "input_context",
        "genomics_enabled": True,
    }
    config = {
        "hidden": 8,
        "heads": 2,
        "layers": 3,
        "dropout": 0.0,
        "task_weights": {"pathway": 1.0, "state": 0.5},
        "residual_bound": 1.0,
        "gate_penalty": 1e-5,
    }
    torch.manual_seed(7)
    model = JointHGT(spec, config).to(device)
    model.train()
    batch = _joint_batch(device)

    encoded = model.encode(graph, 0)
    for task in ("pathway", "state"):
        output_value = model.forward_task(task, encoded, batch)
        torch.testing.assert_close(output_value["logit"], batch["base"], rtol=0.0, atol=0.0)

    # The bounded residual output maps deliberately start at zero. One real
    # BCE update unlocks them; the next losses must reach the shared HGT.
    warmup = torch.optim.SGD(model.parameters(), lr=0.1)
    warmup_losses = _joint_losses(model, graph, batch)
    warmup.zero_grad(set_to_none=True)
    (warmup_losses["pathway"] + 0.5 * warmup_losses["state"]).backward()
    warmup.step()

    losses = _joint_losses(model, graph, batch)
    shared_parameters = list(model.layers[0].parameters())
    gradients = {}
    for task, value in losses.items():
        task_gradients = torch.autograd.grad(
            value, shared_parameters, retain_graph=True, allow_unused=True
        )
        gradients[task] = sum(
            float(gradient.detach().abs().sum().cpu())
            for gradient in task_gradients
            if gradient is not None
        )
    if min(gradients.values()) <= 0:
        fail(
            "JOINT_GRADIENT_FAILURE",
            "Both BCE membership losses must reach the same current HGTConv",
            gradients=gradients,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    optimizer.zero_grad(set_to_none=True)
    losses = _joint_losses(model, graph, batch)
    objective = losses["pathway"] + 0.5 * losses["state"]
    objective.backward()
    guard = optimizer_step_with_global_delta_guard(
        model, optimizer, torch=torch, objective_value=float(objective.detach().cpu())
    )
    if not all(isinstance(layer, HGTConv) for layer in model.layers):
        fail("BACKBONE_SUBSTITUTED", "Actual current HGTConv required")

    result = {
        "state": "REAL_PYG_BACKEND_CHECK_PASS_NOT_TRAINED",
        "backend_contract": "CURRENT_CC_HHGT_PYG_JOINT_CONTEXT_V1",
        "torch": str(torch.__version__),
        "pyg": torch_geometric.__version__,
        "device": device,
        "current_encoder_zero_context_equivalence": equivalence,
        "shared_membership_gradients": gradients,
        "task_weights": {"pathway": 1.0, "state": 0.5},
        "global_representable_delta_guard": guard,
        "six_channel_context_features_checked": True,
        "cancer_is_graph_node": True,
        "historical_reference_loaded": False,
        "full_r1_kernel_equivalence_claimed": False,
        "native_hgt_fallback": False,
        "production_release_allowed": False,
    }
    atomic_json(output / "BACKEND_TEST.json", result)
    status(output, result["state"])
    return result
