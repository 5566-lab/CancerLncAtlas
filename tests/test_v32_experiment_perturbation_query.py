from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from cc_hhgt.v32.experiment_perturbation import (
    ANALYSIS_VERSION,
    ARTIFACT_FILENAMES,
    BINDING_FORMAT,
    FORMAL_ASSAY_DETAIL_SHA256,
    FORMAL_CANDIDATE_UNIVERSE_SHA256,
    FORMAL_EVIDENCE_EVENT_SHA256,
    FORMAL_PATHWAY_MEMBERS_SHA256,
    PENDING_CONFIDENCE_REASON,
    file_sha256,
)
from cc_hhgt.v32.experiment_perturbation_query import (
    ExperimentPerturbationQueryAssetError,
    ExperimentPerturbationQueryInputError,
    ExperimentPerturbationReleaseQuery,
)


def _binding(tmp_path: Path) -> tuple[Path, str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    exact = tmp_path / ARTIFACT_FILENAMES["v32_perturbation_exact_pathway"]
    pd.DataFrame(
        [
            {
                "perturbation_event_id": "PERT:1",
                "source_event_id": "EVENT:1",
                "cancer_id": "BRCA",
                "lncrna_id": "LNC:ENSG000001",
                "pathway_id": "PATH:P1",
                "partner_id": "ENSG000101",
                "assay_family": "functional_perturbation",
                "assay_detail": "SIRNA_KNOCKDOWN;WESTERN_BLOT",
                "assay_detail_available": True,
                "assay_detail_unavailable_reason": pd.NA,
                "perturbation_methods": "SIRNA_KNOCKDOWN",
                "perturbation_method_available": True,
                "readout_assays": "WESTERN_BLOT",
                "readout_assay_available": True,
                "assay_detail_match_route": "PMID_LNCRNA_GENE_EXACT",
                "assay_detail_evidence_strength": "HIGH",
                "assay_detail_manual_review_required": False,
                "direction_class": "positive",
                "evidence_fact_available": True,
                "family_broadcast_used": False,
                "independent_probability_generated": False,
                "confidence_impact_status": PENDING_CONFIDENCE_REASON,
                "changes_primary_ranking": False,
                "changes_discovery_ranking": False,
                "release_ready": False,
            }
        ]
    ).to_parquet(exact, index=False)
    paths: dict[str, Path] = {"v32_perturbation_exact_pathway": exact}
    for role, filename in ARTIFACT_FILENAMES.items():
        if role == "v32_perturbation_exact_pathway":
            continue
        path = tmp_path / filename
        pd.DataFrame({"fixture": pd.Series(dtype="string")}).to_parquet(
            path, index=False
        )
        paths[role] = path

    manifest = tmp_path / "EXPERIMENT_PERTURBATION_MANIFEST.json"
    manifest.write_text(
        json.dumps(
            {
                "analysis_version": ANALYSIS_VERSION,
                "status": "MATERIALISATION_SUCCESS",
                "release_ready": False,
                "confidence_impact_policy": {
                    "endpoint": "confidence_only",
                    "status": PENDING_CONFIDENCE_REASON,
                    "independent_probability_generated": False,
                    "changes_discovery_ranking": False,
                    "changes_primary_ranking": False,
                },
                "assay_detail_contract": {
                    "assay_detail_available_rows": 1,
                    "manual_review_required_rows": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    artifacts = {
        role: {
            "path": str(path),
            "sha256": file_sha256(path),
            "rows": 1 if role == "v32_perturbation_exact_pathway" else 0,
        }
        for role, path in paths.items()
    }
    binding = tmp_path / "EXPERIMENT_PERTURBATION_BINDING.json"
    binding.write_text(
        json.dumps(
            {
                "analysis_version": ANALYSIS_VERSION,
                "module_id": "experiment_perturbation",
                "binding_format": BINDING_FORMAT,
                "status": "MATERIALISATION_SUCCESS_RELEASE_PENDING",
                "release_ready": False,
                "release_blockers": [PENDING_CONFIDENCE_REASON],
                "manifest": {
                    "path": str(manifest),
                    "sha256": file_sha256(manifest),
                },
                "artifacts": artifacts,
                "input_sha256": {
                    "assay_detail": FORMAL_ASSAY_DETAIL_SHA256,
                    "candidate_universe": FORMAL_CANDIDATE_UNIVERSE_SHA256,
                    "evidence_event": FORMAL_EVIDENCE_EVENT_SHA256,
                    "pathway_members": FORMAL_PATHWAY_MEMBERS_SHA256,
                },
                "invariants": {
                    "functional_perturbation_only": True,
                    "partner_literal_exact_member_mapping_only": True,
                    "family_broadcast_used": False,
                    "pan_cancer_explicit": True,
                    "old_results_used": False,
                    "independent_probability_generated": False,
                    "changes_primary_ranking": False,
                },
            }
        ),
        encoding="utf-8",
    )
    return binding, file_sha256(binding), exact


def test_fresh_exact_fact_query_is_confidence_only(tmp_path: Path) -> None:
    binding, digest, _ = _binding(tmp_path)
    query = ExperimentPerturbationReleaseQuery(
        binding, expected_binding_sha256=digest
    )
    result = query.query_exact_pathway_facts(
        cancer_id="brca",
        lncrna_id="ENSG000001.2",
        partner_id="GENE:ENSG000101",
        direction_class="positive",
    )
    assert result["returned_rows"] == 1
    assert result["confidence_only"] is True
    assert result["independent_probability_generated"] is False
    assert result["affects_discovery"] is False
    assert result["changes_primary_ranking"] is False
    capability = query.capability_status()
    assert capability["fresh_v32_facts"] is True
    assert capability["confidence_integration_status"] == PENDING_CONFIDENCE_REASON


def test_query_fails_closed_on_hash_semantic_and_input_drift(tmp_path: Path) -> None:
    binding, digest, exact = _binding(tmp_path)
    with pytest.raises(ExperimentPerturbationQueryAssetError, match="requires"):
        ExperimentPerturbationReleaseQuery(binding, expected_binding_sha256=None)
    with pytest.raises(ExperimentPerturbationQueryAssetError, match="mismatch"):
        ExperimentPerturbationReleaseQuery(
            binding, expected_binding_sha256="0" * 64
        )
    query = ExperimentPerturbationReleaseQuery(
        binding, expected_binding_sha256=digest
    )
    with pytest.raises(ExperimentPerturbationQueryInputError, match="direction_class"):
        query.query_exact_pathway_facts(direction_class="unknown")
    frame = pd.read_parquet(exact)
    frame.loc[0, "changes_primary_ranking"] = True
    frame.to_parquet(exact, index=False)
    with pytest.raises(ExperimentPerturbationQueryAssetError, match="SHA/path drift"):
        ExperimentPerturbationReleaseQuery(binding, expected_binding_sha256=digest)
