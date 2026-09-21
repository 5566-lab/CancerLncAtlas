from __future__ import annotations

from dataclasses import replace

import pytest

import cc_hhgt.v32.score_architecture as architecture


def test_score_architecture_keeps_native_estimands_separate() -> None:
    architecture.validate_score_architecture()
    manifest = architecture.score_architecture_manifest()
    assert manifest["primary_score_immutable"] is True
    assert manifest["single_total_score_forbidden"] is True
    assert manifest["direct_target_evidence_in_discovery"] is False

    routes = manifest["module_routes"]
    assert routes["drug"]["affects_endpoints"] == ("drug_actionability",)
    assert routes["evidence_transformer"]["affects_endpoints"] == (
        "exact_pathway_confidence",
    )
    assert set(routes["single_cell"]["affects_endpoints"]) == {
        "single_cell_context",
        "exact_pathway_secondary_discovery",
        "exact_pathway_confidence",
    }
    assert routes["single_cell"]["exact_pathway_aggregation"]


def test_score_architecture_rejects_drug_routing_into_pathway_primary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad = dict(architecture.MODULE_ROUTES)
    bad["drug"] = replace(
        bad["drug"],
        affects_endpoints=("exact_pathway_primary",),
        primary_mutation_allowed=True,
    )
    monkeypatch.setattr(architecture, "MODULE_ROUTES", bad)
    with pytest.raises(architecture.ScoreArchitectureError, match="Only the exact-pathway"):
        architecture.validate_score_architecture()


def test_score_architecture_rejects_direct_evidence_in_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad = dict(architecture.MODULE_ROUTES)
    bad["evidence_transformer"] = replace(
        bad["evidence_transformer"],
        affects_endpoints=("exact_pathway_secondary_discovery",),
    )
    monkeypatch.setattr(architecture, "MODULE_ROUTES", bad)
    with pytest.raises(architecture.ScoreArchitectureError, match="confidence-only"):
        architecture.validate_score_architecture()
