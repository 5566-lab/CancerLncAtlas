from __future__ import annotations

import numpy as np
import torch

from cc_hhgt_v26.moe_model import AvailabilityMaskedMoE
from cc_hhgt_v26.state_model import StateExpertNetwork


def test_state_network_patient_and_graph_heads():
    model = StateExpertNetwork(patient_dim=7, graph_dims={"rgcn": 4, "cc_hhgt_strict": 4}, hidden=16)
    patient = torch.randn(6, 7)
    graph = {
        "rgcn": torch.randn(6, 20),
        "cc_hhgt_strict": torch.randn(6, 20),
    }
    out = model(patient, graph)
    assert out["patient_logit"].shape == (6,)
    assert out["rgcn_logit"].shape == (6,)
    assert out["cc_hhgt_strict_logit"].shape == (6,)
    assert out["direction_logit"].shape == (6,)


def test_state_moe_masks_unavailable_graph_experts():
    torch.manual_seed(1)
    model = AvailabilityMaskedMoE(n_experts=4, quality_dim=3, hidden=8)
    logits = torch.randn(5, 4)
    availability = torch.tensor(
        [
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 0, 0, 1],
        ],
        dtype=torch.float32,
    )
    quality = torch.randn(5, 3)
    _, weights = model(logits, availability, quality)
    assert torch.allclose(weights[availability == 0], torch.zeros_like(weights[availability == 0]), atol=1e-7)
    assert np.allclose(weights.detach().sum(dim=1).numpy(), 1.0, atol=1e-6)
