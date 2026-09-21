"""Independent verifier for the complete strict V3.2 r5 audit."""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import integrated_completeness_strict_independent as helper
from . import integrated_completeness_strict_r4_independent as base


RELEASE_FORMAT = "CANCERLNCATLAS_V32_INTEGRATED_COMPLETENESS_STRICT_BINDING_V5"
REPORT_FORMAT = "CANCERLNCATLAS_V32_INTEGRATED_COMPLETENESS_STRICT_AUDIT_V5"
AUDIT_REPORT_FORMAT = (
    "CANCERLNCATLAS_V32_INTEGRATED_COMPLETENESS_STRICT_R5_INDEPENDENT_AUDIT_V1"
)
AUDIT_BINDING_FORMAT = (
    "CANCERLNCATLAS_V32_INTEGRATED_COMPLETENESS_STRICT_R5_INDEPENDENT_BINDING_V1"
)


class StrictR5IndependentAuditError(RuntimeError):
    """Raised when the independent r5 verifier cannot proceed safely."""


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise StrictR5IndependentAuditError(f"JSON object required: {path}")
    return value


def independent_audit_strict_integrated_completeness_r5(
    *,
    repo_root: str | Path,
    binding_path: str | Path,
    expected_binding_sha256: str,
    output_root: str | Path,
    auditor_code_path: str | Path | None = None,
    runner_code_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    old_values = {
        "RELEASE_FORMAT": base.RELEASE_FORMAT,
        "REPORT_FORMAT": base.REPORT_FORMAT,
        "AUDIT_REPORT_FORMAT": base.AUDIT_REPORT_FORMAT,
        "AUDIT_BINDING_FORMAT": base.AUDIT_BINDING_FORMAT,
        "AUDIT_REVISION": base.AUDIT_REVISION,
        "DOWNLOAD_CATALOG_VERSION": base.DOWNLOAD_CATALOG_VERSION,
        "AUDIT_REPORT_FILENAME": base.AUDIT_REPORT_FILENAME,
        "AUDIT_BINDING_FILENAME": base.AUDIT_BINDING_FILENAME,
        "DEFAULT_RUNNER": base.DEFAULT_RUNNER,
    }
    base.RELEASE_FORMAT = RELEASE_FORMAT
    base.REPORT_FORMAT = REPORT_FORMAT
    base.AUDIT_REPORT_FORMAT = AUDIT_REPORT_FORMAT
    base.AUDIT_BINDING_FORMAT = AUDIT_BINDING_FORMAT
    base.AUDIT_REVISION = "r5"
    base.DOWNLOAD_CATALOG_VERSION = "v3.2-20260827-r7"
    base.AUDIT_REPORT_FILENAME = "INDEPENDENT_AUDIT_R5_REPORT.json"
    base.AUDIT_BINDING_FILENAME = "INDEPENDENT_AUDIT_R5_BINDING.json"
    base.DEFAULT_RUNNER = (
        "scripts/audit_v32_integrated_completeness_strict_r5_independent.py"
    )
    try:
        result = base.independent_audit_strict_integrated_completeness_r4(
            repo_root=root,
            binding_path=binding_path,
            expected_binding_sha256=expected_binding_sha256,
            output_root=output_root,
            auditor_code_path=auditor_code_path or Path(__file__).resolve(),
            runner_code_path=runner_code_path,
        )
    finally:
        for key, value in old_values.items():
            setattr(base, key, value)

    release_path, release = helper._read_json(binding_path, root, "strict r5 binding")
    report_path = Path(result["report_path"]).resolve()
    audit_report = _read(report_path)
    checks = audit_report.get("checks", [])
    if not isinstance(checks, list):
        raise StrictR5IndependentAuditError("Base independent checks are invalid")

    def check(check_id: str, passed: bool, detail: Any = None) -> None:
        checks.append({"check_id": check_id, "passed": bool(passed), "detail": detail})

    inputs = release.get("inputs", {})

    def input_payload(input_id: str) -> tuple[Path, dict[str, Any]]:
        record = inputs.get(input_id, {})
        path = helper._record_path(record, root, input_id)
        return path, _read(path)

    deployment_path, deployment = input_payload(
        "single_cell_diagnostic_server_deployment_binding"
    )
    audit_path, sc_audit = input_payload(
        "single_cell_diagnostic_server_independent_audit_binding"
    )
    sc_report_path, sc_report = input_payload(
        "single_cell_diagnostic_server_independent_audit_report"
    )
    ucell_path, ucell = input_payload(
        "single_cell_ucell_17c_server_deployment_binding"
    )
    smoke = deployment.get("real_query_smoke", {})
    semantics = deployment.get("semantics", {})
    artifacts = deployment.get("contract_artifacts", {})
    check(
        "single_cell_deployment_contract",
        (
            deployment.get("format")
            == "CC_HHGT_V3_2_SINGLE_CELL_DIAGNOSTIC_SERVER_DEPLOYMENT_BINDING_V1"
            and deployment.get("status") == "SERVER_MOUNT_READY_HASH_PINNED"
            and deployment.get("cancer_count") == 17
            and deployment.get("numeric_pseudotime_rows", 0) > 0
            and deployment.get("exact_pathway_association_rows", 0) > 0
            and deployment.get("figure_count") == 153
            and artifacts.get("v32_sc_pseudotime", {}).get("rows", 0) > 0
            and artifacts.get("v32_sc_figure_manifest", {}).get("rows") == 153
            and deployment.get("production_deployed") is False
        ),
        {"path": str(deployment_path)},
    )
    check(
        "single_cell_real_server_smoke",
        (
            smoke.get("status") == "PASS"
            and smoke.get("real_numeric_pathway_query_exercised") is True
            and smoke.get("real_numeric_cell_query_exercised") is True
            and smoke.get("real_figure_manifest_exercised") is True
            and smoke.get("real_figure_file_rehash_exercised") is True
            and smoke.get("real_download_manifest_exercised") is True
            and smoke.get("request_time_payload_rehash_exercised") is True
            and smoke.get("path_traversal_rejected") is True
        ),
    )
    check(
        "single_cell_zero_weight_semantics",
        (
            semantics.get("root_provenance")
            == "INFERRED_CYTOTRACE2_UCELL_CONSENSUS"
            and semantics.get("root_is_explicit") is False
            and semantics.get("diagnostic_only") is True
            and semantics.get("model_fusion_permitted") is False
            and semantics.get("primary_score_weight") == 0
            and semantics.get("secondary_score_weight") == 0
            and semantics.get("historical_outputs_used") is False
            and semantics.get("changes_primary_ranking") is False
        ),
    )
    sc_report_decl = sc_audit.get("report", {})
    check(
        "single_cell_independent_audit_chain",
        (
            sc_audit.get("format")
            == "CC_HHGT_V3_2_SINGLE_CELL_DIAGNOSTIC_SERVER_INDEPENDENT_AUDIT_BINDING_V1"
            and sc_audit.get("status") == "PASS_HASH_BOUND"
            and sc_audit.get("failed_checks") == 0
            and sc_audit.get("accepted_for_staging_integration") is True
            and sc_audit.get("deployment_binding_sha256")
            == helper.sha256_file(deployment_path)
            and sc_report_decl.get("sha256") == helper.sha256_file(sc_report_path)
            and sc_report.get("status") == "PASS"
            and sc_report.get("failure_count") == 0
            and sc_report.get("accepted_for_staging_integration") is True
            and sc_report.get("production_deployed") is False
            and sc_audit.get("production_deployed") is False
        ),
        {"audit_path": str(audit_path), "report_path": str(sc_report_path)},
    )
    ucell_query_smoke = ucell.get("real_query_smoke", {})
    ucell_download_smoke = ucell.get("real_download_smoke", {})
    ucell_totals = ucell.get("audited_totals", {})
    check(
        "single_cell_ucell_17c_server_contract",
        (
            ucell.get("format")
            == "CC_HHGT_V3_2_SINGLE_CELL_UCELL_17C_SERVER_DEPLOYMENT_BINDING_V1"
            and ucell.get("analysis_version")
            == "CancerLncAtlas_V3.2_FULL_MULTITASK"
            and ucell.get("status") == "SERVER_MOUNT_READY_HASH_PINNED"
            and ucell.get("cancer_count") == 17
            and ucell.get("pathways_per_cancer") == 2135
            and ucell_totals.get("cells") == 1_057_271
            and ucell_totals.get("coverage_rows") == 2_257_273_585
            and ucell_totals.get("numeric_rows", 0) > 0
            and ucell_totals.get("typed_unavailable_rows", 0) > 0
            and ucell_totals.get("donor_celltype_groups") == 2024
            and ucell.get("typed_unavailable_preserved") is True
            and ucell.get("unavailable_values_filled_with_zero_or_half") is False
            and ucell.get("historical_outputs_used") is False
            and ucell.get("ucell_capability_release_ready") is True
            and ucell.get("production_deployed") is False
        ),
        {"path": str(ucell_path)},
    )
    check(
        "single_cell_ucell_17c_real_smokes",
        (
            ucell_query_smoke.get("status") == "PASS"
            and ucell_query_smoke.get("runtime_tree_rehash_performed") is True
            and ucell_query_smoke.get("both_source_runs_queried") is True
            and ucell_query_smoke.get("typed_unavailable_null_preserved") is True
            and ucell_download_smoke.get("status") == "PASS"
            and ucell_download_smoke.get(
                "request_time_payload_rehash_exercised"
            )
            is True
            and ucell_download_smoke.get("path_traversal_rejected") is True
            and ucell_download_smoke.get("pseudotime_artifacts_excluded") is True
        ),
    )
    complete_rows = audit_report.get("recomputed_blocking_capability_ids", []) == []
    check("all_25_recomputed_complete", complete_rows)

    fail_count = sum(not bool(row.get("passed")) for row in checks)
    audit_report.update(
        {
            "format": AUDIT_REPORT_FORMAT,
            "audit_revision": "r5",
            "status": "PASS" if fail_count == 0 else "FAIL",
            "accepted_as_complete": fail_count == 0 and complete_rows,
            "accepted_as_truthful_staging_audit": fail_count == 0,
            "independent_of_r5_evaluator_implementation": True,
            "r5_evaluator_imported": False,
            "check_count": len(checks),
            "pass_count": len(checks) - fail_count,
            "fail_count": fail_count,
            "checks": checks,
        }
    )
    helper._atomic_json(report_path, audit_report)

    audit_binding_path = Path(result["binding_path"]).resolve()
    audit_binding = _read(audit_binding_path)
    audit_binding.update(
        {
            "format": AUDIT_BINDING_FORMAT,
            "audit_revision": "r5",
            "status": audit_report["status"],
            "accepted_as_complete": audit_report["accepted_as_complete"],
            "accepted_as_truthful_staging_audit": audit_report[
                "accepted_as_truthful_staging_audit"
            ],
            "fail_count": fail_count,
            "report": {
                "path": report_path.relative_to(root).as_posix(),
                "sha256": helper.sha256_file(report_path),
                "bytes": report_path.stat().st_size,
            },
        }
    )
    helper._atomic_json(audit_binding_path, audit_binding)
    return {
        "binding_path": str(audit_binding_path),
        "binding_sha256": helper.sha256_file(audit_binding_path),
        "report_path": str(report_path),
        "report_sha256": helper.sha256_file(report_path),
        "status": audit_report["status"],
        "audited_release_status": audit_report.get("audited_release_status"),
        "accepted_as_complete": audit_report["accepted_as_complete"],
        "accepted_as_truthful_staging_audit": audit_report[
            "accepted_as_truthful_staging_audit"
        ],
        "fail_count": fail_count,
        "check_count": len(checks),
    }


__all__ = [
    "AUDIT_BINDING_FORMAT",
    "AUDIT_REPORT_FORMAT",
    "StrictR5IndependentAuditError",
    "independent_audit_strict_integrated_completeness_r5",
]
