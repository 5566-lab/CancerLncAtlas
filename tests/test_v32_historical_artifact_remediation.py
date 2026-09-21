from __future__ import annotations

import json
from pathlib import Path

import pytest

from cc_hhgt.v32.historical_artifact_remediation import (
    FORMAT,
    HistoricalArtifactQuery,
    HistoricalArtifactRemediationError,
    sha256_file,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FORMAL_RELEASE = (
    REPO_ROOT
    / "artifacts"
    / "v32_historical_artifact_remediation_20260826_r3_evidence_direction_closed"
)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_query_loader_fails_closed_on_bound_file_drift(tmp_path: Path) -> None:
    root = tmp_path / "release"
    root.mkdir()
    artifact = root / "artifact.json"
    _write_json(artifact, {"status": "READY"})
    binding = {
        "format": FORMAT,
        "capabilities": {},
        "files": {
            artifact.name: {
                "sha256": sha256_file(artifact),
                "bytes": artifact.stat().st_size,
            }
        },
    }
    binding_path = root / "HISTORICAL_ARTIFACT_REMEDIATION_BINDING.json"
    _write_json(binding_path, binding)
    _write_json(
        root / "SUCCESS.json",
        {
            "status": "SUCCESS_MATERIALIZATION_WITH_GAPS_PRESERVED",
            "binding_sha256": sha256_file(binding_path),
        },
    )
    binding_sha256 = sha256_file(binding_path)
    HistoricalArtifactQuery(root, expected_binding_sha256=binding_sha256)
    with pytest.raises(
        HistoricalArtifactRemediationError, match="external binding hash mismatch"
    ):
        HistoricalArtifactQuery(root, expected_binding_sha256="0" * 64)
    _write_json(artifact, {"status": "TAMPERED"})
    with pytest.raises(HistoricalArtifactRemediationError, match="hash drift"):
        HistoricalArtifactQuery(root)


def test_query_loader_requires_hash_bound_independent_audit(tmp_path: Path) -> None:
    root = tmp_path / "release"
    audit_root = tmp_path / "audit"
    root.mkdir()
    audit_root.mkdir()
    binding_path = root / "HISTORICAL_ARTIFACT_REMEDIATION_BINDING.json"
    _write_json(binding_path, {"format": FORMAT, "capabilities": {}, "files": {}})
    binding_sha256 = sha256_file(binding_path)
    _write_json(
        root / "SUCCESS.json",
        {
            "status": "SUCCESS_MATERIALIZATION_WITH_GAPS_PRESERVED",
            "binding_sha256": binding_sha256,
        },
    )
    report_path = audit_root / "INDEPENDENT_AUDIT.json"
    _write_json(report_path, {"status": "PASS", "checks": 3})
    audit_path = audit_root / "INDEPENDENT_AUDIT_BINDING.json"
    _write_json(
        audit_path,
        {
            "status": "PASS_HASH_BOUND",
            "checks": 3,
            "failed_checks": 0,
            "release_binding_sha256": binding_sha256,
            "report_path": str(report_path),
            "report_sha256": sha256_file(report_path),
        },
    )
    HistoricalArtifactQuery(
        root,
        expected_binding_sha256=binding_sha256,
        independent_audit_binding_path=audit_path,
        expected_independent_audit_sha256=sha256_file(audit_path),
    )
    with pytest.raises(
        HistoricalArtifactRemediationError,
        match="path and expected SHA256 must be supplied together",
    ):
        HistoricalArtifactQuery(root, independent_audit_binding_path=audit_path)
    _write_json(report_path, {"status": "TAMPERED"})
    with pytest.raises(HistoricalArtifactRemediationError, match="report hash drift"):
        HistoricalArtifactQuery(
            root,
            expected_binding_sha256=binding_sha256,
            independent_audit_binding_path=audit_path,
            expected_independent_audit_sha256=sha256_file(audit_path),
        )


@pytest.mark.skipif(not FORMAL_RELEASE.is_dir(), reason="formal remediation release not materialized")
def test_formal_release_queries_and_typed_gaps() -> None:
    query = HistoricalArtifactQuery(FORMAL_RELEASE)
    priority = query.clinical_priority(
        cancer_id="BRCA", endpoint="OS", subject_type="lncRNA", limit=5
    )
    assert priority
    assert all(row["availability"] is True for row in priority)
    assert all(row["changes_primary_ranking"] is False for row in priority)

    mutation = query.mutation_subgroup(cancer_id="BRCA", limit=5)
    assert len(mutation) == 5
    assert all(row["target_level"] == "exact_pathway" for row in mutation)
    assert all(row["family_broadcast"] is False for row in mutation)

    cnv = query.cnv_coverage()
    assert len(cnv) == 33
    brca = next(row for row in cnv if row["cancer_id"] == "BRCA")
    acc = next(row for row in cnv if row["cancer_id"] == "ACC")
    assert brca["cnv_available_rows"] > 0
    assert acc["cnv_available_rows"] == 0
    assert acc["coverage_unavailable_reason"] == "CNV_NOT_AVAILABLE_FOR_CANCER"

    single_cell = query.capability_status("single_cell")
    assert single_cell["status"] == "PARTIAL_WITH_TYPED_GAPS"
    assert single_cell["artifacts"]["v32_sc_ucell"]["scope"] == "HNSC_ONLY_PILOT"
    assert single_cell["artifacts"]["v32_sc_pseudotime"]["present"] is False
    assert single_cell["artifacts"]["v32_sc_figure_manifest"]["available_figure_files"] == 0

    evidence = query.capability_status("evidence_transformer")
    assert evidence["artifacts"]["v32_fused_confidence_probability"]["rows"] == 3_300_000
    direction = evidence["artifacts"]["v32_evidence_direction_probability"]
    assert direction["present"] is True
    assert direction["rows"] == 3_300_000
    assert direction["available_rows"] == 825_753
    assert direction["typed_null_rows"] == 2_474_247
    assert evidence["fused_confidence_semantics"]["evidence_transformer_weight"] == 0.0

    mixed = query.capability_status("mixed_lncrna_protein_pathway_query")
    assert {
        "v32_mixed_set_encoder",
        "v32_custom_gene_set_encoder",
        "v32_protein_set_encoder",
    }.issubset(mixed["artifacts"])


@pytest.mark.skipif(not FORMAL_RELEASE.is_dir(), reason="formal remediation release not materialized")
def test_formal_query_bounds_are_fail_closed() -> None:
    query = HistoricalArtifactQuery(FORMAL_RELEASE)
    with pytest.raises(ValueError, match="1..1000"):
        query.clinical_priority(cancer_id="BRCA", limit=1001)
    with pytest.raises(ValueError, match="1..1000"):
        query.mutation_subgroup(cancer_id="BRCA", limit=0)
