from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from cc_hhgt_v26.adapter_data import (
    _combine_candidate_sources,
    select_patient_native_candidates,
)
from cc_hhgt_v26.adapter_model import CancerAdapter
from cc_hhgt_v26.moe_model import AvailabilityMaskedMoE


def test_patient_adapter_masks_joint_branch_when_strict_missing() -> None:
    torch.manual_seed(1)
    model = CancerAdapter(embedding_dim=4, patient_dim=5, strict_dim=3, hidden=12)
    model.eval()
    batch = 6
    lnc = torch.randn(batch, 4)
    pathway = torch.randn(batch, 4)
    patient = torch.randn(batch, 5)
    strict = torch.randn(batch, 3)
    cancer = torch.randn(batch, 4)
    unavailable = torch.zeros(batch, dtype=torch.bool)

    first = model(lnc, pathway, patient, strict, cancer, unavailable)
    second = model(
        lnc * 100,
        pathway * -100,
        patient,
        strict * 1000,
        cancer * -100,
        unavailable,
    )
    assert torch.allclose(first["joint_weight"], torch.zeros(batch), atol=1e-7)
    assert torch.allclose(first["native_weight"], torch.ones(batch), atol=1e-7)
    assert torch.allclose(
        first["membership_logit"], first["native_membership_logit"], atol=1e-6
    )
    # With strict unavailable, changing graph/strict inputs cannot change output.
    assert torch.allclose(
        first["membership_logit"], second["membership_logit"], atol=1e-6
    )


def test_top_level_moe_assigns_zero_weight_to_unavailable_expert() -> None:
    torch.manual_seed(2)
    model = AvailabilityMaskedMoE(n_experts=3, quality_dim=2, hidden=8)
    expert_logits = torch.tensor([[1.0, 2.0, -1.0], [0.5, -0.5, 3.0]])
    availability = torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.float32)
    quality = torch.zeros((2, 2))
    _, weights = model(expert_logits, availability, quality)
    assert weights[0, 1].item() == 0.0
    assert weights[1, 0].item() == 0.0
    assert torch.allclose(weights.sum(dim=1), torch.ones(2), atol=1e-6)


def test_patient_native_candidate_selection_recovers_cancer_local_pair(monkeypatch) -> None:
    import cc_hhgt_v26.adapter_data as module

    monkeypatch.setattr(module, "PATIENT_NATIVE_TOP_K", 1)
    monkeypatch.setattr(module, "PATIENT_NATIVE_MIN_ABS_CORR", 0.5)
    monkeypatch.setattr(module, "PATIENT_NATIVE_MAX_CANDIDATES", 10)
    n = 20
    signal = np.linspace(-2, 2, n)
    expression = np.column_stack([signal, np.random.default_rng(3).normal(size=n)])
    activity = np.column_stack([signal * 2, np.random.default_rng(4).normal(size=n)])
    selected = select_patient_native_candidates(
        "LUAD",
        expression,
        activity,
        ["LNC:LOCAL", "LNC:NOISE"],
        ["PF:MAPK", "PF:NOISE"],
        np.array([1.0, 1.0]),
    )
    hit = selected[
        selected["lncrna_id"].eq("LNC:LOCAL")
        & selected["pathway_family_id"].eq("PF:MAPK")
    ]
    assert len(hit) == 1
    assert int(hit.iloc[0]["candidate_from_patient"]) == 1


def test_candidate_union_retains_strict_missing_patient_pair() -> None:
    strict = pd.DataFrame(
        {
            "cancer_id": ["LUAD"],
            "lncrna_id": ["LNC:SHARED"],
            "pathway_family_id": ["PF:A"],
            "cross_cancer_probability": [0.8],
            "candidate_from_strict": [1],
        }
    )
    patient = pd.DataFrame(
        {
            "cancer_id": ["LUAD"],
            "lncrna_id": ["LNC:LOCAL"],
            "pathway_family_id": ["PF:B"],
            "patient_native_selection_score": [0.7],
            "candidate_from_patient": [1],
        }
    )
    evidence = pd.DataFrame(columns=["cancer_id", "lncrna_id", "pathway_family_id"])
    union = _combine_candidate_sources(strict, patient, evidence)
    local = union[union["lncrna_id"].eq("LNC:LOCAL")].iloc[0]
    assert int(local["strict_available"]) == 0
    assert int(local["candidate_from_patient"]) == 1
    assert local["candidate_source"] == "patient_native"
