from __future__ import annotations

import json
from pathlib import Path

from cc_hhgt.v32.download_catalog import sha256_file
from cc_hhgt.v32.download_catalog_r7 import materialize_download_catalog_r7


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def test_r7_closes_pseudotime_gap_with_independent_server_authority(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    # The catalog intentionally allows only authorities/output under the repo.
    tmp_path.resolve().relative_to(root)
    authority = tmp_path / "authority"
    deployment_path = authority / "SERVER_DEPLOYMENT_BINDING.json"
    deployment = {
        "format": "CC_HHGT_V3_2_SINGLE_CELL_DIAGNOSTIC_SERVER_DEPLOYMENT_BINDING_V1",
        "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
        "status": "SERVER_MOUNT_READY_HASH_PINNED",
        "cancer_count": 17,
        "numeric_pseudotime_rows": 17,
        "exact_pathway_association_rows": 34,
        "figure_count": 153,
        "contract_artifacts": {
            "v32_sc_pseudotime": {"rows": 17},
            "v32_sc_figure_manifest": {"rows": 153},
        },
        "real_query_smoke": {
            "status": "PASS",
            "real_numeric_pathway_query_exercised": True,
            "real_numeric_cell_query_exercised": True,
            "real_figure_manifest_exercised": True,
            "real_figure_file_rehash_exercised": True,
            "real_download_manifest_exercised": True,
            "request_time_payload_rehash_exercised": True,
            "path_traversal_rejected": True,
        },
        "semantics": {
            "root_provenance": "INFERRED_CYTOTRACE2_UCELL_CONSENSUS",
            "root_is_explicit": False,
            "diagnostic_only": True,
            "model_fusion_permitted": False,
            "primary_score_weight": 0,
            "secondary_score_weight": 0,
            "historical_outputs_used": False,
            "changes_primary_ranking": False,
        },
        "production_deployed": False,
    }
    _write(deployment_path, deployment)
    deployment_sha = sha256_file(deployment_path)
    report_path = authority / "INDEPENDENT_AUDIT_REPORT.json"
    report = {
        "status": "PASS",
        "failure_count": 0,
        "accepted_for_staging_integration": True,
        "deployment_binding": {"sha256": deployment_sha},
        "production_deployed": False,
    }
    _write(report_path, report)
    audit_path = authority / "INDEPENDENT_AUDIT_BINDING.json"
    audit = {
        "format": "CC_HHGT_V3_2_SINGLE_CELL_DIAGNOSTIC_SERVER_INDEPENDENT_AUDIT_BINDING_V1",
        "status": "PASS_HASH_BOUND",
        "deployment_binding_sha256": deployment_sha,
        "failed_checks": 0,
        "accepted_for_staging_integration": True,
        "production_deployed": False,
        "report": {"path": str(report_path.resolve()), "sha256": sha256_file(report_path)},
    }
    _write(audit_path, audit)
    audit_sha = sha256_file(audit_path)

    output = tmp_path / "catalog"
    materialize_download_catalog_r7(
        repo_root=root,
        output_root=output,
        deployment_relative_path=deployment_path.relative_to(root).as_posix(),
        deployment_sha256=deployment_sha,
        audit_relative_path=audit_path.relative_to(root).as_posix(),
        audit_sha256=audit_sha,
    )
    catalog = json.loads(
        (output / "V32_STAGING_DOWNLOAD_CATALOG.json").read_text(encoding="utf-8")
    )
    downloads = {row["download_id"]: row for row in catalog["downloads"]}
    assert downloads["single_cell_pseudotime"]["status"] == "SERVER_HASH_PINNED_DOWNLOAD"
    assert downloads["single_cell_pseudotime"]["contract_complete"] is True
    assert catalog["known_capability_gaps"] == []
    assert catalog["status_counts"] == {
        "DYNAMIC_QUERY_EXPORT": 4,
        "READY_FILE": 39,
        "READY_PARTS": 8,
        "SERVER_HASH_PINNED_DOWNLOAD": 3,
    }
