from __future__ import annotations

import pytest

from cc_hhgt.v32.fusion_expert_policy import (
    EXACT_PATHWAY_FUSION_EXPERTS,
    experts_for_endpoint,
    validate_exact_pathway_fusion_expert_policy,
)
from cc_hhgt.v32.multimodal_fusion import MultimodalFusionError


def test_formal_expert_policy_routes_single_cell_but_not_drug() -> None:
    validate_exact_pathway_fusion_expert_policy()
    assert [spec.expert_id for spec in experts_for_endpoint("discovery")] == [
        "genomic",
        "single_cell",
    ]
    assert [spec.expert_id for spec in experts_for_endpoint("confidence")] == [
        "genomic",
        "single_cell",
        "evidence_transformer",
    ]
    assert "drug" not in {spec.expert_id for spec in EXACT_PATHWAY_FUSION_EXPERTS}
    evidence = EXACT_PATHWAY_FUSION_EXPERTS[-1]
    assert evidence.direct_target_evidence is True
    assert evidence.endpoint_roles == ("confidence",)


def test_native_actionability_is_not_an_exact_pathway_endpoint() -> None:
    with pytest.raises(MultimodalFusionError, match="Unsupported exact-pathway"):
        experts_for_endpoint("actionability")
