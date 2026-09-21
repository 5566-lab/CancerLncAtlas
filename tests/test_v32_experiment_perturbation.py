from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from cc_hhgt.v32.experiment_perturbation import (
    ARTIFACT_FILENAMES,
    PENDING_CONFIDENCE_REASON,
    ExperimentPerturbationContractError,
    file_sha256,
    materialise_experiment_perturbation,
    validate_experiment_perturbation_binding,
)


def _fixture_inputs(root: Path, *, derived_column: bool = False) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    evidence = pd.DataFrame(
        [
            {
                "evidence_event_id": "EVID:mapped",
                "pmid": "1001",
                "lncrna_id": "LNC:ENSG000001",
                "partner_id": "GENE:ENSG000101",
                "relation_type": "regulation",
                "experiment_family": "functional_perturbation",
                "cancer_id": "",
                "cell_line": "CL-A",
                "tissue": "Liver",
                "drug_id": "",
                "direction": "positively-E",
                "independent_event_hash": "hash-mapped",
                "manual_review_status": "unreviewed",
                "pathway_family_id": "",
                "pathway_id": "",
            },
            {
                "evidence_event_id": "EVID:family-only",
                "pmid": "1002",
                "lncrna_id": "LNC:ENSG000001",
                "partner_id": "",
                "relation_type": "association",
                "experiment_family": "functional_perturbation",
                "cancer_id": "",
                "cell_line": "",
                "tissue": "",
                "drug_id": "",
                "direction": "",
                "independent_event_hash": "hash-family",
                "manual_review_status": "unreviewed",
                "pathway_family_id": "PF:TEST",
                "pathway_id": "",
            },
            {
                "evidence_event_id": "EVID:missing-lnc",
                "pmid": "1003",
                "lncrna_id": "",
                "partner_id": "GENE:ENSG000101",
                "relation_type": "regulation",
                "experiment_family": "functional_perturbation",
                "cancer_id": "",
                "cell_line": "CL-B",
                "tissue": "Breast",
                "drug_id": "",
                "direction": "negatively-E",
                "independent_event_hash": "hash-missing",
                "manual_review_status": "unreviewed",
                "pathway_family_id": "",
                "pathway_id": "",
            },
            {
                "evidence_event_id": "EVID:invalid-partner",
                "pmid": "1004",
                "lncrna_id": "LNC:ENSG000001",
                "partner_id": "GENE:ENSG999999",
                "relation_type": "regulation",
                "experiment_family": "CRISPR perturbation",
                "cancer_id": "PAN_CANCER",
                "cell_line": "",
                "tissue": "",
                "drug_id": "",
                "direction": "",
                "independent_event_hash": "hash-invalid",
                "manual_review_status": "unreviewed",
                "pathway_family_id": "",
                "pathway_id": "",
            },
            {
                "evidence_event_id": "EVID:direct-ignored",
                "pmid": "1005",
                "lncrna_id": "LNC:ENSG000001",
                "partner_id": "",
                "relation_type": "association",
                "experiment_family": "functional_perturbation",
                "cancer_id": "",
                "cell_line": "",
                "tissue": "",
                "drug_id": "",
                "direction": "",
                "independent_event_hash": "hash-direct",
                "manual_review_status": "unreviewed",
                "pathway_family_id": "",
                "pathway_id": "PATH:P2",
            },
            {
                "evidence_event_id": "EVID:not-perturbation",
                "pmid": "1006",
                "lncrna_id": "LNC:ENSG000001",
                "partner_id": "GENE:ENSG000102",
                "relation_type": "binding_or_interaction",
                "experiment_family": "physical_binding",
                "cancer_id": "",
                "cell_line": "CL-C",
                "tissue": "Lung",
                "drug_id": "",
                "direction": "",
                "independent_event_hash": "hash-physical",
                "manual_review_status": "unreviewed",
                "pathway_family_id": "",
                "pathway_id": "",
            },
        ]
    )
    if derived_column:
        evidence["old_confidence"] = 0.9
    candidates = pd.DataFrame(
        [
            {"cancer_id": "AAA", "lncrna_id": "LNC:ENSG000001", "pathway_id": "PATH:P1"},
            {"cancer_id": "BBB", "lncrna_id": "LNC:ENSG000001", "pathway_id": "PATH:P1"},
            {"cancer_id": "AAA", "lncrna_id": "LNC:ENSG000001", "pathway_id": "PATH:P2"},
        ]
    )
    members = pd.DataFrame(
        [
            {"pathway_id": "PATH:P1", "gene_id": "ENSG000101"},
            {"pathway_id": "PATH:P2", "gene_id": "ENSG000102"},
        ]
    )
    paths = {
        "evidence_event": root / "evidence_event.parquet",
        "candidate_universe": root / "FORMAL_CANDIDATE_UNIVERSE.parquet",
        "pathway_members": root / "exact_pathway_gene_membership_ensembl.parquet",
    }
    evidence.to_parquet(paths["evidence_event"], index=False)
    candidates.to_parquet(paths["candidate_universe"], index=False)
    members.to_parquet(paths["pathway_members"], index=False)
    return paths


def _build(root: Path, *, derived_column: bool = False):
    paths = _fixture_inputs(root / "inputs", derived_column=derived_column)
    hashes = {key: file_sha256(path) for key, path in paths.items()}
    result = materialise_experiment_perturbation(
        evidence_event_path=paths["evidence_event"],
        candidate_universe_path=paths["candidate_universe"],
        pathway_members_path=paths["pathway_members"],
        output_dir=root / "output",
        expected_hashes=hashes,
        formal=False,
        runner_path=Path(__file__),
    )
    return paths, result


def test_literal_partner_mapping_pan_cancer_and_typed_missing(tmp_path: Path) -> None:
    _, result = _build(tmp_path)
    source = pd.read_parquet(result.artifact_paths["v32_experiment_events"])
    exact = pd.read_parquet(result.artifact_paths["v32_perturbation_exact_pathway"])
    provenance = pd.read_parquet(result.artifact_paths["v32_experiment_provenance"])
    rejected = pd.read_parquet(result.artifact_paths["v32_experiment_rejected"])

    assert len(source) == 5
    assert source.cancer_id.eq("PAN_CANCER").all()
    assert int(source.mapping_available.sum()) == 1
    assert set(exact.cancer_id) == {"AAA", "BBB"}
    assert set(exact.pathway_id) == {"PATH:P1"}
    assert set(exact.mapping_route) == {"PARTNER_EXACT_MEMBER"}
    assert set(exact.lncrna_id) == {"LNC:ENSG000001"}
    assert set(exact.evidence_lncrna_id) == {"ENSG000001"}
    assert exact.direction_available.all()
    assert set(exact.direction_class) == {"positive"}
    assert exact.assay_detail.isna().all()
    assert not exact.assay_detail_available.any()
    assert not exact.family_broadcast_used.any()
    assert not exact.independent_probability_generated.any()
    assert not exact.release_ready.any()

    reasons = set(rejected.rejection_reason.dropna().astype(str))
    assert "FAMILY_ONLY_NOT_BROADCAST_TO_EXACT" in reasons
    assert "LNC_ID_UNMAPPED" in reasons
    assert "NO_EXACT_PATHWAY_OR_STATIC_MEMBER_MAPPING" in reasons
    assert len(rejected) == 4
    missing = source.loc[source.experiment_event_id.eq("EVID:missing-lnc")].iloc[0]
    assert not bool(missing.lncrna_id_available)
    assert pd.isna(missing.lncrna_id)
    assert missing.rejection_reason == "LNC_ID_UNMAPPED"
    family = source.loc[source.experiment_event_id.eq("EVID:family-only")].iloc[0]
    assert family.rejection_reason == "FAMILY_ONLY_NOT_BROADCAST_TO_EXACT"
    direct = source.loc[source.experiment_event_id.eq("EVID:direct-ignored")].iloc[0]
    assert not bool(direct.mapping_available)
    assert result.manifest["mapping_contract"]["direct_exact_assertions_ignored"] == 1
    assert not provenance.family_broadcast_used.any()
    assert set(provenance.mapping_available.astype(bool)) == {False, True}


def test_hash_bound_manifest_and_no_predictive_payload(tmp_path: Path) -> None:
    paths, result = _build(tmp_path)
    binding = validate_experiment_perturbation_binding(result.output_dir)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert binding["manifest"]["sha256"] == file_sha256(result.manifest_path)
    assert binding["release_ready"] is False
    assert binding["release_blockers"] == [PENDING_CONFIDENCE_REASON]
    assert manifest["lineage_policy"]["old_predictions_used"] is False
    assert manifest["lineage_policy"]["training_performed"] is False
    assert manifest["lineage_policy"]["probability_or_score_generated"] is False
    assert manifest["confidence_impact_policy"]["endpoint"] == "confidence_only"
    assert manifest["row_audit"]["source_row_accounting_pass"] is True
    assert manifest["input_artifacts"]["evidence_event"]["sha256"] == file_sha256(
        paths["evidence_event"]
    )
    assert set(binding["artifacts"]) == set(ARTIFACT_FILENAMES)
    for artifact_id, info in binding["artifacts"].items():
        frame = pd.read_parquet(info["path"])
        assert len(frame) == info["rows"]
        forbidden = {
            column
            for column in frame.columns
            if any(token in column.lower() for token in ("score", "probability", "logit"))
            and column
            not in {
                "independent_probability_generated",
                "changes_primary_ranking",
                "changes_discovery_ranking",
            }
        }
        assert not forbidden, (artifact_id, forbidden)


def test_binding_verifier_detects_artifact_tampering(tmp_path: Path) -> None:
    _, result = _build(tmp_path)
    path = result.artifact_paths["v32_experiment_rejected"]
    tampered = pd.read_parquet(path)
    tampered.loc[0, "rejection_reason"] = "TAMPERED"
    tampered.to_parquet(path, index=False)
    with pytest.raises(ExperimentPerturbationContractError, match="Artifact hash binding failed"):
        validate_experiment_perturbation_binding(result.output_dir)


def test_hash_bound_assay_detail_is_joined_without_changing_rankings(tmp_path: Path) -> None:
    paths = _fixture_inputs(tmp_path / "inputs")
    evidence = pd.read_parquet(paths["evidence_event"])
    selected_ids = evidence.loc[
        evidence.experiment_family.str.contains("perturbation", case=False),
        "evidence_event_id",
    ].tolist()
    detail_rows = []
    for event_id in selected_ids:
        available = event_id == "EVID:mapped"
        detail_rows.append(
            {
                "evidence_event_id": event_id,
                "assay_detail": "SIRNA_KNOCKDOWN;WESTERN_BLOT" if available else pd.NA,
                "assay_detail_available": available,
                "assay_detail_unavailable_reason": (
                    pd.NA if available else "NO_SUPPORTED_DETAIL_MATCH"
                ),
                "perturbation_methods": "SIRNA_KNOCKDOWN" if available else pd.NA,
                "perturbation_method_available": available,
                "readout_assays": "WESTERN_BLOT" if available else pd.NA,
                "readout_assay_available": available,
                "match_route": "PMID_LNCRNA_GENE_EXACT" if available else "UNAVAILABLE_TYPED",
                "evidence_strength": "HIGH" if available else "NONE",
                "manual_review_required": not available,
                "source_database": "NcPath" if available else pd.NA,
                "source_record_ids": "NCPATH:1" if available else pd.NA,
                "source_context": "fixture" if available else pd.NA,
                "changes_primary_ranking": False,
                "changes_discovery_ranking": False,
            }
        )
    detail_path = tmp_path / "inputs" / "v32_experiment_assay_detail.parquet"
    pd.DataFrame(detail_rows).to_parquet(detail_path, index=False)
    hashes = {key: file_sha256(path) for key, path in paths.items()}
    hashes["assay_detail"] = file_sha256(detail_path)
    result = materialise_experiment_perturbation(
        evidence_event_path=paths["evidence_event"],
        candidate_universe_path=paths["candidate_universe"],
        pathway_members_path=paths["pathway_members"],
        assay_detail_path=detail_path,
        output_dir=tmp_path / "output",
        expected_hashes=hashes,
        formal=False,
    )

    source = pd.read_parquet(result.artifact_paths["v32_experiment_events"])
    exact = pd.read_parquet(result.artifact_paths["v32_perturbation_exact_pathway"])
    mapped = source.loc[source.source_record_id.eq("EVID:mapped")].iloc[0]
    assert mapped.assay_detail == "SIRNA_KNOCKDOWN;WESTERN_BLOT"
    assert bool(mapped.assay_detail_available)
    assert mapped.assay_detail_match_route == "PMID_LNCRNA_GENE_EXACT"
    assert set(exact.assay_detail) == {"SIRNA_KNOCKDOWN;WESTERN_BLOT"}
    assert exact.assay_detail_available.all()
    assert not exact.changes_primary_ranking.any()
    assert not exact.changes_discovery_ranking.any()
    assert result.manifest["assay_detail_contract"]["one_row_per_selected_event"] is True
    assert result.manifest["input_artifacts"]["assay_detail"]["sha256"] == file_sha256(
        detail_path
    )


def test_assay_detail_rejects_incomplete_event_coverage(tmp_path: Path) -> None:
    paths = _fixture_inputs(tmp_path / "inputs")
    detail = pd.DataFrame(
        [
            {
                "evidence_event_id": "EVID:mapped",
                "assay_detail": "SIRNA_KNOCKDOWN",
                "assay_detail_available": True,
                "assay_detail_unavailable_reason": pd.NA,
                "perturbation_methods": "SIRNA_KNOCKDOWN",
                "perturbation_method_available": True,
                "readout_assays": pd.NA,
                "readout_assay_available": False,
                "match_route": "PMID_LNCRNA_GENE_EXACT",
                "evidence_strength": "HIGH",
                "manual_review_required": False,
                "source_database": "NcPath",
                "source_record_ids": "NCPATH:1",
                "source_context": pd.NA,
                "changes_primary_ranking": False,
                "changes_discovery_ranking": False,
            }
        ]
    )
    detail_path = tmp_path / "inputs" / "v32_experiment_assay_detail.parquet"
    detail.to_parquet(detail_path, index=False)
    hashes = {key: file_sha256(path) for key, path in paths.items()}
    hashes["assay_detail"] = file_sha256(detail_path)
    with pytest.raises(ExperimentPerturbationContractError, match="cover exactly"):
        materialise_experiment_perturbation(
            evidence_event_path=paths["evidence_event"],
            candidate_universe_path=paths["candidate_universe"],
            pathway_members_path=paths["pathway_members"],
            assay_detail_path=detail_path,
            output_dir=tmp_path / "output",
            expected_hashes=hashes,
            formal=False,
        )


def test_forbids_historical_prediction_path_even_when_hash_is_supplied(tmp_path: Path) -> None:
    paths = _fixture_inputs(tmp_path / "legacy_predictions")
    hashes = {key: file_sha256(path) for key, path in paths.items()}
    with pytest.raises(ExperimentPerturbationContractError, match="forbidden historical/derived"):
        materialise_experiment_perturbation(
            evidence_event_path=paths["evidence_event"],
            candidate_universe_path=paths["candidate_universe"],
            pathway_members_path=paths["pathway_members"],
            output_dir=tmp_path / "output",
            expected_hashes=hashes,
            formal=False,
        )


def test_forbids_derived_score_columns_and_hash_mismatch(tmp_path: Path) -> None:
    paths = _fixture_inputs(tmp_path / "derived", derived_column=True)
    hashes = {key: file_sha256(path) for key, path in paths.items()}
    with pytest.raises(ExperimentPerturbationContractError, match="forbidden learned/historical"):
        materialise_experiment_perturbation(
            evidence_event_path=paths["evidence_event"],
            candidate_universe_path=paths["candidate_universe"],
            pathway_members_path=paths["pathway_members"],
            output_dir=tmp_path / "output-derived",
            expected_hashes=hashes,
            formal=False,
        )

    clean = _fixture_inputs(tmp_path / "clean")
    wrong = {key: file_sha256(path) for key, path in clean.items()}
    wrong["evidence_event"] = "0" * 64
    with pytest.raises(ExperimentPerturbationContractError, match="SHA-256 mismatch"):
        materialise_experiment_perturbation(
            evidence_event_path=clean["evidence_event"],
            candidate_universe_path=clean["candidate_universe"],
            pathway_members_path=clean["pathway_members"],
            output_dir=tmp_path / "output-wrong-hash",
            expected_hashes=wrong,
            formal=False,
        )
