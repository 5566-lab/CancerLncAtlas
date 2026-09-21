"""Strict V3.2 completeness audit r5 with server single-cell publication."""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import integrated_completeness_strict as r2
from . import integrated_completeness_strict_r4 as r4


REPORT_FORMAT = "CANCERLNCATLAS_V32_INTEGRATED_COMPLETENESS_STRICT_AUDIT_V5"
BINDING_FORMAT = "CANCERLNCATLAS_V32_INTEGRATED_COMPLETENESS_STRICT_BINDING_V5"
AUDIT_REVISION = "r5"
CATALOG_VERSION = "v3.2-20260827-r7"
R4_BINDING_SHA256 = (
    "1e343714f3c191f6ce74a0c9a795cc2776c28ee4721c642331c4a1fbf0ad5de0"
)
SC_DEPLOYMENT_FORMAT = (
    "CC_HHGT_V3_2_SINGLE_CELL_DIAGNOSTIC_SERVER_DEPLOYMENT_BINDING_V1"
)
SC_AUDIT_FORMAT = (
    "CC_HHGT_V3_2_SINGLE_CELL_DIAGNOSTIC_SERVER_INDEPENDENT_AUDIT_BINDING_V1"
)
SC_UCELL_DEPLOYMENT_FORMAT = (
    "CC_HHGT_V3_2_SINGLE_CELL_UCELL_17C_SERVER_DEPLOYMENT_BINDING_V1"
)
SC_UCELL_DEPLOYMENT_SHA256 = (
    "4925433dde88bbb8c93ee07fcc668ae6cc043a135d7b1c1c3a8ace342e265c75"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class StrictCompletenessR5Error(RuntimeError):
    """Raised when r5 cannot issue a fail-closed result."""


def _read_json(path: Path, label: str) -> dict[str, Any]:
    source = path.resolve()
    if not source.is_file() or source.is_symlink():
        raise StrictCompletenessR5Error(f"{label} is missing or unsafe: {source}")
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise StrictCompletenessR5Error(f"{label} must be a JSON object")
    return value


def _resolve_unified_server_binding(
    *, root: Path, unified: Mapping[str, Any], capability_id: str, expected_sha: str
) -> tuple[Path, dict[str, Any]]:
    entry = unified.get("bindings", {}).get(capability_id, {})
    if (
        not isinstance(entry, Mapping)
        or entry.get("status") != "SERVER_MOUNT_READY_HASH_PINNED"
        or entry.get("production_deployed") is not False
        or entry.get("deployment_binding_sha256") != expected_sha
    ):
        raise StrictCompletenessR5Error(
            f"Unified binding {capability_id} is not the expected server authority"
        )
    path = (root / str(entry.get("deployment_binding_path", ""))).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise StrictCompletenessR5Error(
            f"Unified binding {capability_id} escapes repository"
        ) from exc
    if r2.sha256_file(path) != expected_sha:
        raise StrictCompletenessR5Error(f"Unified binding {capability_id} SHA drift")
    return path, _read_json(path, capability_id)


def _single_cell_server_evidence(
    *,
    root: Path,
    unified_path: Path,
    deployment_sha256: str,
    audit_sha256: str,
) -> tuple[dict[str, Any], dict[str, Path]]:
    unified = _read_json(unified_path, "unified staging bindings")
    deployment_path, deployment = _resolve_unified_server_binding(
        root=root,
        unified=unified,
        capability_id="single_cell_diagnostic",
        expected_sha=deployment_sha256,
    )
    audit_path, audit = _resolve_unified_server_binding(
        root=root,
        unified=unified,
        capability_id="single_cell_diagnostic_independent_audit",
        expected_sha=audit_sha256,
    )
    smoke = deployment.get("real_query_smoke", {})
    semantics = deployment.get("semantics", {})
    artifacts = deployment.get("contract_artifacts", {})
    report_decl = audit.get("report", {})
    report_path = Path(str(report_decl.get("path", ""))).resolve()
    try:
        report_path.relative_to(root)
    except ValueError as exc:
        raise StrictCompletenessR5Error(
            "Single-cell independent-audit report escapes repository"
        ) from exc
    report = _read_json(report_path, "single-cell independent-audit report")
    report_sha = str(report_decl.get("sha256", "")).lower()
    ready = bool(
        deployment.get("format") == SC_DEPLOYMENT_FORMAT
        and deployment.get("status") == "SERVER_MOUNT_READY_HASH_PINNED"
        and deployment.get("cancer_count") == 17
        and deployment.get("numeric_pseudotime_rows", 0) > 0
        and deployment.get("exact_pathway_association_rows", 0) > 0
        and deployment.get("figure_count") == 153
        and artifacts.get("v32_sc_pseudotime", {}).get("rows", 0) > 0
        and artifacts.get("v32_sc_figure_manifest", {}).get("rows") == 153
        and smoke.get("status") == "PASS"
        and smoke.get("real_numeric_pathway_query_exercised") is True
        and smoke.get("real_numeric_cell_query_exercised") is True
        and smoke.get("real_figure_manifest_exercised") is True
        and smoke.get("real_figure_file_rehash_exercised") is True
        and smoke.get("real_download_manifest_exercised") is True
        and smoke.get("request_time_payload_rehash_exercised") is True
        and smoke.get("path_traversal_rejected") is True
        and semantics.get("root_provenance")
        == "INFERRED_CYTOTRACE2_UCELL_CONSENSUS"
        and semantics.get("root_is_explicit") is False
        and semantics.get("diagnostic_only") is True
        and semantics.get("model_fusion_permitted") is False
        and semantics.get("primary_score_weight") == 0
        and semantics.get("secondary_score_weight") == 0
        and semantics.get("historical_outputs_used") is False
        and semantics.get("changes_primary_ranking") is False
        and deployment.get("production_deployed") is False
        and audit.get("format") == SC_AUDIT_FORMAT
        and audit.get("status") == "PASS_HASH_BOUND"
        and audit.get("deployment_binding_sha256") == deployment_sha256
        and audit.get("failed_checks") == 0
        and audit.get("accepted_for_staging_integration") is True
        and audit.get("production_deployed") is False
        and _SHA256.fullmatch(report_sha) is not None
        and r2.sha256_file(report_path) == report_sha
        and report.get("status") == "PASS"
        and report.get("failure_count") == 0
        and report.get("accepted_for_staging_integration") is True
        and report.get("deployment_binding", {}).get("sha256")
        == deployment_sha256
        and report.get("production_deployed") is False
    )
    evidence = {
        "status": "PASS" if ready else "FAIL",
        "deployment_binding_sha256": deployment_sha256,
        "independent_audit_binding_sha256": audit_sha256,
        "independent_audit_report_sha256": report_sha,
        "cancer_count": deployment.get("cancer_count"),
        "numeric_pseudotime_rows": deployment.get("numeric_pseudotime_rows"),
        "exact_pathway_association_rows": deployment.get(
            "exact_pathway_association_rows"
        ),
        "figure_count": deployment.get("figure_count"),
        "real_query_smoke": smoke,
        "root_provenance": semantics.get("root_provenance"),
        "root_is_explicit": semantics.get("root_is_explicit"),
        "diagnostic_only": semantics.get("diagnostic_only"),
        "primary_score_weight": semantics.get("primary_score_weight"),
        "secondary_score_weight": semantics.get("secondary_score_weight"),
        "historical_outputs_used": semantics.get("historical_outputs_used"),
        "changes_primary_ranking": semantics.get("changes_primary_ranking"),
    }
    if not ready:
        raise StrictCompletenessR5Error(
            "Single-cell server runtime/query/download evidence did not pass"
        )
    return evidence, {
        "single_cell_diagnostic_server_deployment_binding": deployment_path,
        "single_cell_diagnostic_server_independent_audit_binding": audit_path,
        "single_cell_diagnostic_server_independent_audit_report": report_path,
    }


def _single_cell_ucell_server_evidence(
    *, root: Path, unified_path: Path
) -> tuple[dict[str, Any], Path]:
    unified = _read_json(unified_path, "unified staging bindings")
    deployment_path, deployment = _resolve_unified_server_binding(
        root=root,
        unified=unified,
        capability_id="single_cell_ucell_17c",
        expected_sha=SC_UCELL_DEPLOYMENT_SHA256,
    )
    query_smoke = deployment.get("real_query_smoke", {})
    download_smoke = deployment.get("real_download_smoke", {})
    totals = deployment.get("audited_totals", {})
    remote_binding = deployment.get("remote_binding", {})
    ready = bool(
        deployment.get("format") == SC_UCELL_DEPLOYMENT_FORMAT
        and deployment.get("analysis_version")
        == "CancerLncAtlas_V3.2_FULL_MULTITASK"
        and deployment.get("status") == "SERVER_MOUNT_READY_HASH_PINNED"
        and deployment.get("capability_id") == "single_cell_ucell_17c"
        and deployment.get("scope")
        == "FORMAL_17_CANCER_CELL_LEVEL_EXACT_PATHWAY_UCELL"
        and deployment.get("cancer_count") == 17
        and deployment.get("pathways_per_cancer") == 2135
        and totals.get("cells") == 1_057_271
        and totals.get("coverage_rows") == 2_257_273_585
        and totals.get("numeric_rows", 0) > 0
        and totals.get("typed_unavailable_rows", 0) > 0
        and totals.get("donor_celltype_groups") == 2024
        and remote_binding.get("sha256")
        == "584a16864153feb17a0214c96da233bef8e7ecf0043f2628835e82abafd63614"
        and query_smoke.get("status") == "PASS"
        and query_smoke.get("runtime_tree_rehash_performed") is True
        and query_smoke.get("both_source_runs_queried") is True
        and query_smoke.get("typed_unavailable_null_preserved") is True
        and download_smoke.get("status") == "PASS"
        and download_smoke.get("request_time_payload_rehash_exercised") is True
        and download_smoke.get("path_traversal_rejected") is True
        and download_smoke.get("pseudotime_artifacts_excluded") is True
        and deployment.get("typed_unavailable_preserved") is True
        and deployment.get("unavailable_values_filled_with_zero_or_half") is False
        and deployment.get("historical_outputs_used") is False
        and deployment.get("pseudotime_is_separate_diagnostic_scope") is True
        and deployment.get("ucell_capability_release_ready") is True
        and deployment.get("production_deployed") is False
    )
    evidence = {
        "status": "PASS" if ready else "FAIL",
        "deployment_binding_sha256": SC_UCELL_DEPLOYMENT_SHA256,
        "cancer_count": deployment.get("cancer_count"),
        "pathways_per_cancer": deployment.get("pathways_per_cancer"),
        "audited_totals": totals,
        "real_query_smoke": query_smoke,
        "real_download_smoke": download_smoke,
        "typed_unavailable_preserved": deployment.get(
            "typed_unavailable_preserved"
        ),
        "historical_outputs_used": deployment.get("historical_outputs_used"),
    }
    if not ready:
        raise StrictCompletenessR5Error(
            "Single-cell 17-cancer UCell server evidence did not pass"
        )
    return evidence, deployment_path


def evaluate_strict_integrated_completeness_r5(
    *,
    repo_root: str | Path,
    parity_contract_path: str | Path,
    unified_bindings_path: str | Path,
    web_catalog_path: str | Path,
    frontend_html_path: str | Path,
    frontend_javascript_path: str | Path,
    main_app_path: str | Path,
    download_binding_sha256: str,
    download_audit_sha256: str,
    single_cell_deployment_sha256: str,
    single_cell_audit_sha256: str,
    evaluator_code_path: str | Path | None = None,
    runner_code_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    digests = (
        download_binding_sha256,
        download_audit_sha256,
        single_cell_deployment_sha256,
        single_cell_audit_sha256,
    )
    if any(_SHA256.fullmatch(value.lower()) is None for value in digests):
        raise StrictCompletenessR5Error("Invalid r5 input SHA256")
    root = Path(repo_root).resolve()
    unified_path = Path(unified_bindings_path).resolve()
    ucell_evidence, ucell_path = _single_cell_ucell_server_evidence(
        root=root, unified_path=unified_path
    )
    sc_evidence, sc_inputs = _single_cell_server_evidence(
        root=root,
        unified_path=unified_path,
        deployment_sha256=single_cell_deployment_sha256.lower(),
        audit_sha256=single_cell_audit_sha256.lower(),
    )

    old_catalog_version = r4.CATALOG_VERSION
    old_binding_sha = r4.R6_DOWNLOAD_BINDING_SHA256
    old_audit_sha = r4.R6_DOWNLOAD_AUDIT_SHA256
    old_resolver = r2._resolve_artifact

    def resolve_artifact_r5(artifact_id: str, **kwargs: Any) -> dict[str, Any]:
        if artifact_id == "v32_sc_ucell":
            totals = ucell_evidence["audited_totals"]
            return {
                "artifact_id": artifact_id,
                "state": "HASH_BOUND",
                "resolution": "SERVER_DEPLOYMENT_REAL_QUERY_AND_DOWNLOAD_SMOKE",
                "authority_id": "single_cell_ucell_17c",
                "deployment_binding_sha256": SC_UCELL_DEPLOYMENT_SHA256,
                "rows": totals["coverage_rows"],
                "numeric_rows": totals["numeric_rows"],
                "typed_unavailable_rows": totals["typed_unavailable_rows"],
                "cancer_count": ucell_evidence["cancer_count"],
                "pathways_per_cancer": ucell_evidence["pathways_per_cancer"],
                "request_time_payload_rehash": True,
                "typed_unavailable_preserved": True,
                "historical_outputs_used": False,
            }
        if artifact_id in {"v32_sc_pseudotime", "v32_sc_figure_manifest"}:
            rows = (
                sc_evidence["numeric_pseudotime_rows"]
                if artifact_id == "v32_sc_pseudotime"
                else sc_evidence["figure_count"]
            )
            return {
                "artifact_id": artifact_id,
                "state": "HASH_BOUND",
                "resolution": "SERVER_DEPLOYMENT_AND_INDEPENDENT_AUDIT",
                "authority_id": "single_cell_diagnostic",
                "deployment_binding_sha256": single_cell_deployment_sha256.lower(),
                "independent_audit_binding_sha256": single_cell_audit_sha256.lower(),
                "rows": rows,
                "request_time_payload_rehash": True,
                "root_provenance": sc_evidence["root_provenance"],
                "root_is_explicit": False,
                "diagnostic_only": True,
                "primary_score_weight": 0,
                "secondary_score_weight": 0,
                "historical_outputs_used": False,
                "changes_primary_ranking": False,
            }
        return old_resolver(artifact_id, **kwargs)

    r4.CATALOG_VERSION = CATALOG_VERSION
    r4.R6_DOWNLOAD_BINDING_SHA256 = download_binding_sha256.lower()
    r4.R6_DOWNLOAD_AUDIT_SHA256 = download_audit_sha256.lower()
    r2._resolve_artifact = resolve_artifact_r5
    try:
        report, inputs = r4.evaluate_strict_integrated_completeness_r4(
            repo_root=root,
            parity_contract_path=parity_contract_path,
            unified_bindings_path=unified_path,
            web_catalog_path=web_catalog_path,
            frontend_html_path=frontend_html_path,
            frontend_javascript_path=frontend_javascript_path,
            main_app_path=main_app_path,
            evaluator_code_path=evaluator_code_path or Path(__file__).resolve(),
            runner_code_path=runner_code_path,
        )
    finally:
        r4.CATALOG_VERSION = old_catalog_version
        r4.R6_DOWNLOAD_BINDING_SHA256 = old_binding_sha
        r4.R6_DOWNLOAD_AUDIT_SHA256 = old_audit_sha
        r2._resolve_artifact = old_resolver

    for input_id, path in sc_inputs.items():
        inputs[input_id] = r2._source_record(path, root)
    inputs["single_cell_ucell_17c_server_deployment_binding"] = r2._source_record(
        ucell_path, root
    )
    report["format"] = REPORT_FORMAT
    report["audit_revision"] = AUDIT_REVISION
    report["supersedes_strict_r4_binding_sha256"] = R4_BINDING_SHA256
    report["download_catalog_version"] = CATALOG_VERSION
    report["implementation_evidence"]["single_cell_server_runtime"] = sc_evidence
    report["implementation_evidence"]["single_cell_ucell_17c_server_runtime"] = (
        ucell_evidence
    )
    report["server_single_cell_acceptance_policy"] = {
        "independent_audit_binding_sha256": single_cell_audit_sha256.lower(),
        "request_time_payload_rehash_required": True,
        "numeric_pseudotime_required": True,
        "formal_figure_count_required": 153,
        "root_provenance": "INFERRED_CYTOTRACE2_UCELL_CONSENSUS",
        "root_is_explicit": False,
        "primary_score_weight": 0,
        "secondary_score_weight": 0,
        "historical_outputs_used": False,
        "production_deployed_required": False,
    }
    report["input_sha256"] = inputs
    return report, inputs


def materialize_strict_integrated_completeness_audit_r5(
    *, output_root: str | Path, **kwargs: Any
) -> dict[str, Any]:
    root = Path(kwargs["repo_root"]).resolve()
    output = Path(output_root).resolve()
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise StrictCompletenessR5Error("Audit output must remain inside repository") from exc
    if output.exists() and any(output.iterdir()):
        raise StrictCompletenessR5Error(f"Refusing non-empty output reuse: {output}")
    output.mkdir(parents=True, exist_ok=True)
    report, inputs = evaluate_strict_integrated_completeness_r5(**kwargs)
    report_path = output / "INTEGRATED_COMPLETENESS_STRICT_R5_REPORT.json"
    r2._atomic_json(report_path, report)
    binding = {
        "format": BINDING_FORMAT,
        "analysis_version": report["analysis_version"],
        "model_version": "V3.2",
        "environment": "staging",
        "audit_revision": AUDIT_REVISION,
        "status": report["status"],
        "all_25_capabilities_four_gate_complete": report[
            "all_25_capabilities_four_gate_complete"
        ],
        "release_ready": report["release_ready"],
        "production_deployed": False,
        "fail_closed": True,
        "typed_unavailable_counts_as_artifact_complete": False,
        "typed_unavailable_counts_as_download_complete": False,
        "server_hash_pinned_downloads_count_as_deliverable": True,
        "supersedes_strict_r4_binding_sha256": R4_BINDING_SHA256,
        "download_catalog_version": CATALOG_VERSION,
        "download_catalog_binding_sha256": kwargs["download_binding_sha256"].lower(),
        "download_catalog_audit_binding_sha256": kwargs[
            "download_audit_sha256"
        ].lower(),
        "single_cell_deployment_binding_sha256": kwargs[
            "single_cell_deployment_sha256"
        ].lower(),
        "single_cell_independent_audit_binding_sha256": kwargs[
            "single_cell_audit_sha256"
        ].lower(),
        "required_capability_count": report["required_capability_count"],
        "complete_capability_count": report["complete_capability_count"],
        "partial_capability_count": report["partial_capability_count"],
        "blocking_capability_ids": report["blocking_capability_ids"],
        "gate_counts": report["gate_counts"],
        "report": r2._source_record(report_path, root),
        "inputs": inputs,
    }
    binding_path = output / "INTEGRATED_COMPLETENESS_STRICT_R5_BINDING.json"
    r2._atomic_json(binding_path, binding)
    return {
        "binding_path": str(binding_path),
        "binding_sha256": r2.sha256_file(binding_path),
        "report_path": str(report_path),
        "report_sha256": r2.sha256_file(report_path),
        "status": report["status"],
        "complete_capability_count": report["complete_capability_count"],
        "partial_capability_count": report["partial_capability_count"],
        "blocking_capability_ids": report["blocking_capability_ids"],
        "gate_counts": report["gate_counts"],
    }


__all__ = [
    "AUDIT_REVISION",
    "BINDING_FORMAT",
    "CATALOG_VERSION",
    "REPORT_FORMAT",
    "StrictCompletenessR5Error",
    "evaluate_strict_integrated_completeness_r5",
    "materialize_strict_integrated_completeness_audit_r5",
]
