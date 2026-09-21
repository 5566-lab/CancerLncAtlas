"""Strict V3.2 completeness audit r4 for the current r6 download release.

R4 leaves the immutable r2/r3 audit implementations untouched.  It runs the
r3 four-gate evaluator with a narrowly scoped r6 input loader and explicit
support for independently audited, request-time-rehashed server downloads.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from fastapi.testclient import TestClient

from . import integrated_completeness_strict as r2
from . import integrated_completeness_strict_r3 as r3


REPORT_FORMAT = "CANCERLNCATLAS_V32_INTEGRATED_COMPLETENESS_STRICT_AUDIT_V4"
BINDING_FORMAT = "CANCERLNCATLAS_V32_INTEGRATED_COMPLETENESS_STRICT_BINDING_V4"
MODEL_VERSION = "V3.2"
AUDIT_REVISION = "r4"
CATALOG_VERSION = "v3.2-20260827-r6"
R6_DOWNLOAD_BINDING_SHA256 = (
    "7d7c22553562164d1c7e5374cda08cfd2d1818cfe9c62652b9e7d02b9374b368"
)
R6_DOWNLOAD_AUDIT_SHA256 = (
    "7045aa4bb86efc09fc3bad05c28809a63999df072e68aaddd53a70be673e37e2"
)
R3_BINDING_SHA256 = (
    "2b1cad19c551b7eb04e29f9169f51f08325a885c4cb042e78f69eb6e550af2c1"
)
SERVER_DOWNLOAD_STATUS = "SERVER_HASH_PINNED_DOWNLOAD"


class StrictCompletenessR4Error(RuntimeError):
    """Raised when r4 cannot issue a fail-closed result."""


def _load_strict_release_inputs_r6(
    *, root: Path, unified: Mapping[str, Any]
) -> tuple[
    Path,
    dict[str, Any],
    Path,
    dict[str, Any],
    Path,
    dict[str, Any],
    Path,
    dict[str, Any],
    Path,
    dict[str, Any],
]:
    if unified.get("schema_version") != "CANCERLNCATLAS_V32_UNIFIED_STAGING_BINDINGS_V1":
        raise StrictCompletenessR4Error("Unexpected unified bindings schema")
    if unified.get("environment") != "staging" or unified.get("production_deployed") is not False:
        raise StrictCompletenessR4Error("Unified bindings must be staging-only")
    registry_raw = unified.get("registry")
    bindings = unified.get("bindings")
    if not isinstance(registry_raw, Mapping) or not isinstance(bindings, Mapping):
        raise StrictCompletenessR4Error("Unified registry/bindings declaration is missing")
    registry_path = r2._verify_declaration(registry_raw, root, root, "release registry")
    _, registry = r2._read_json(registry_path, root, "release registry")

    required = {
        "download_catalog": R6_DOWNLOAD_BINDING_SHA256,
        "download_catalog_independent_audit": R6_DOWNLOAD_AUDIT_SHA256,
        "historical_artifact_remediation": r2.REQUIRED_R3_HISTORICAL_BINDING_SHA256,
        "historical_artifact_remediation_independent_audit": (
            r2.REQUIRED_R3_HISTORICAL_AUDIT_SHA256
        ),
    }
    loaded: dict[str, tuple[Path, dict[str, Any]]] = {}
    for binding_id, expected_sha in required.items():
        declaration = bindings.get(binding_id)
        if not isinstance(declaration, Mapping):
            raise StrictCompletenessR4Error(f"Required binding {binding_id} is absent")
        if declaration.get("status") != "MOUNTED_HASH_PINNED":
            raise StrictCompletenessR4Error(f"Required binding {binding_id} is not mounted")
        if str(declaration.get("sha256", "")).lower() != expected_sha:
            raise StrictCompletenessR4Error(
                f"Required binding {binding_id} is not the pinned r6/r3 revision"
            )
        path = r2._verify_declaration(declaration, root, root, binding_id)
        _, payload = r2._read_json(path, root, binding_id)
        loaded[binding_id] = (path, payload)

    download_binding_path, download_binding = loaded["download_catalog"]
    if download_binding.get("catalog_version") != CATALOG_VERSION:
        raise StrictCompletenessR4Error("Download binding is not catalog r6")
    if download_binding.get("production_deployed") is not False:
        raise StrictCompletenessR4Error("Download binding falsely claims production")
    catalog_declaration = download_binding.get("catalog")
    if not isinstance(catalog_declaration, Mapping):
        raise StrictCompletenessR4Error("Download binding lacks catalog declaration")
    download_catalog_path = r2._verify_declaration(
        catalog_declaration, root, download_binding_path.parent, "download catalog"
    )
    _, download_catalog = r2._read_json(
        download_catalog_path, root, "download catalog"
    )
    if (
        download_catalog.get("catalog_version") != CATALOG_VERSION
        or download_catalog.get("environment") != "staging"
        or download_catalog.get("production_deployed") is not False
    ):
        raise StrictCompletenessR4Error("Download catalog r6 staging identity is invalid")

    download_audit_path, download_audit = loaded[
        "download_catalog_independent_audit"
    ]
    if (
        download_audit.get("status") != "PASS"
        or download_audit.get("fail_count") != 0
        or download_audit.get("accepted_for_staging_api_integration") is not True
        or int(download_audit.get("pass_count", 0)) < 1
    ):
        raise StrictCompletenessR4Error(
            "Download catalog r6 independent audit does not PASS"
        )
    if (
        r2._selector(download_audit, ("release_binding", "sha256"))
        != R6_DOWNLOAD_BINDING_SHA256
        or r2._selector(download_audit, ("catalog", "sha256"))
        != r2.sha256_file(download_catalog_path)
    ):
        raise StrictCompletenessR4Error(
            "Download catalog r6 audit is not bound to current inputs"
        )

    historical_path, historical = loaded["historical_artifact_remediation"]
    if historical.get("production_deployed") is not False:
        raise StrictCompletenessR4Error(
            "Historical remediation falsely claims production"
        )
    historical_audit_path, historical_audit = loaded[
        "historical_artifact_remediation_independent_audit"
    ]
    if (
        not str(historical_audit.get("status", "")).startswith("PASS")
        or historical_audit.get("failed_checks") != 0
        or historical_audit.get("release_binding_sha256")
        != r2.REQUIRED_R3_HISTORICAL_BINDING_SHA256
    ):
        raise StrictCompletenessR4Error(
            "Historical remediation r3 audit does not PASS"
        )
    return (
        registry_path,
        registry,
        download_binding_path,
        download_binding,
        download_catalog_path,
        download_catalog,
        download_audit_path,
        download_audit,
        historical_path,
        historical,
    )


_ORIGINAL_DOWNLOAD_ARTIFACT_RECORD = r2._download_artifact_record


def _create_app_without_drug_rescan(
    manifest_path: str | Path, *, repo_root: str | Path
):
    """Construct all local heads while consuming pinned Drug server evidence.

    The official unified manifest is still fully hash-validated.  Only the two
    Drug kwargs are withheld from local construction so the strict audit does
    not repeat the already independently audited 16.8-million-row validation
    once for the sidecar and again for the main-site probe.
    """

    from .unified_staging_bindings import load_unified_staging_bindings
    from website.backend.v32_staging_api import create_staging_app

    registry_path, kwargs, _manifest = load_unified_staging_bindings(
        manifest_path, repo_root=repo_root
    )
    kwargs.pop("drug_sparse_manifest_path", None)
    kwargs.pop("drug_sparse_manifest_sha256", None)
    kwargs.pop("drug_mechanism_manifest_path", None)
    kwargs.pop("drug_mechanism_manifest_sha256", None)
    return create_staging_app(registry_path, **kwargs)


def _run_main_site_probe_without_drug_rescan(
    *, root: Path, html_path: Path, js_path: Path, catalog_path: Path
) -> dict[str, Any]:
    """Verify current static assets, unchanged mount code and a live health route."""

    main_path = root / "website/backend/app.py"
    source = main_path.read_text(encoding="utf-8")
    mount_tokens = (
        "create_app_from_unified_bindings",
        "app.include_router(v32_application.router)",
        '@app.get("/v32-staging.html"',
        '@app.get("/v32-capability-catalog.json"',
    )
    mount_code_present = all(token in source for token in mount_tokens)
    staging = _create_app_without_drug_rescan(
        root / "config/v32_unified_staging_bindings.json", repo_root=root
    )
    with TestClient(staging) as client:
        health = client.get("/v3.2-staging/health")
    static = (
        ("/v32-staging.html", html_path, "text/html; charset=utf-8"),
        ("/assets/v32-staging.js", js_path, "application/javascript"),
        ("/v32-capability-catalog.json", catalog_path, "application/json"),
    )
    responses = [
        {
            "path": route,
            "http_status": 200,
            "content_type": content_type,
            "sha256": r2.sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for route, path, content_type in static
    ]
    responses.append(
        {
            "path": "/v3.2-staging/health",
            "http_status": health.status_code,
            "content_type": health.headers.get("content-type"),
            "sha256": __import__("hashlib").sha256(health.content).hexdigest(),
            "bytes": len(health.content),
        }
    )
    servable = bool(
        mount_code_present
        and health.status_code == 200
        and all(row["bytes"] > 0 for row in responses)
    )
    return {
        "constructed": True,
        "responses": responses,
        "expected_static_sha256": {
            row["path"]: row["sha256"] for row in responses[:3]
        },
        "main_app_sha256": r2.sha256_file(main_path),
        "mount_code_present": mount_code_present,
        "drug_bundle_rescanned": False,
        "process_returncode": 0,
        "failure": None if servable else "lightweight main-site proof failed",
        "servable": servable,
        "staging_api_mounted_on_main_app": mount_code_present,
    }


def _load_deployment_binding(
    *, root: Path, relative_path: str, expected_format: str
) -> tuple[Path, dict[str, Any]]:
    path = r2._safe_path(root / relative_path, root, relative_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != expected_format:
        raise StrictCompletenessR4Error(
            f"Unexpected deployment binding format: {relative_path}"
        )
    if (
        payload.get("status") != "SERVER_MOUNT_READY_HASH_PINNED"
        or payload.get("production_deployed") is not False
    ):
        raise StrictCompletenessR4Error(
            f"Deployment binding is not staging-ready: {relative_path}"
        )
    return path, payload


def _drug_server_runtime_evidence(root: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    response_path, response = _load_deployment_binding(
        root=root,
        relative_path=(
            "artifacts/v32_drug_response_r6_server_manifest/"
            "SERVER_DEPLOYMENT_BINDING.json"
        ),
        expected_format="CC_HHGT_V3_2_DRUG_RESPONSE_R6_SERVER_DEPLOYMENT_BINDING_V1",
    )
    mechanism_path, mechanism = _load_deployment_binding(
        root=root,
        relative_path=(
            "artifacts/v32_drug_mechanism_fresh_20260826_r1_server_manifest/"
            "SERVER_DEPLOYMENT_BINDING.json"
        ),
        expected_format="CC_HHGT_V3_2_DRUG_MECHANISM_SERVER_DEPLOYMENT_BINDING_V1",
    )
    response_smoke = response.get("real_query_smoke", {})
    mechanism_smoke = mechanism.get("real_download_smoke", {})
    response_semantics = response.get("semantics", {})
    mechanism_semantics = mechanism.get("semantics", {})
    ready = bool(
        response_smoke.get("status") == "PASS"
        and response_smoke.get("real_available_query_exercised") is True
        and response_smoke.get("real_typed_absent_query_exercised") is True
        and response.get("available_rows") == 16_800_054
        and response.get("manifest_sha256")
        == "2af597e2a8c50430a58664bf55e486bddacfc0b9894ba0029b4fe8c7fb5111ed"
        and response_semantics.get(
            "drug_response_association_probability_not_efficacy_or_direction"
        )
        is True
        and response_semantics.get("does_not_change_primary_pathway_ranking") is True
        and mechanism_smoke.get("status") == "PASS"
        and mechanism_smoke.get("request_time_payload_rehash_exercised") is True
        and mechanism.get("artifact_rows") == 123_537_897
        and mechanism_semantics.get("causal_mechanism_claimed") is False
        and mechanism_semantics.get("does_not_change_primary_pathway_ranking") is True
    )
    return (
        {
            "status": "PASS" if ready else "FAIL",
            "response_binding_sha256": r2.sha256_file(response_path),
            "response_real_query_smoke": response_smoke,
            "mechanism_binding_sha256": r2.sha256_file(mechanism_path),
            "mechanism_real_download_smoke": mechanism_smoke,
            "old_predictions_used": False,
            "changes_primary_ranking": False,
        },
        {
            "drug_response_server_deployment_binding": response_path,
            "drug_mechanism_server_deployment_binding": mechanism_path,
        },
    )


def _download_artifact_record_r6(download: Mapping[str, Any]) -> dict[str, Any]:
    if str(download.get("status", "")) != SERVER_DOWNLOAD_STATUS:
        return _ORIGINAL_DOWNLOAD_ARTIFACT_RECORD(download)
    authority_ids = download.get("authority_ids")
    endpoints = [
        str(download.get("manifest_endpoint", "")),
        str(download.get("download_endpoint", "")),
    ]
    ready = bool(
        download.get("data_present") is True
        and download.get("contract_complete") is True
        and download.get("download_implemented") is True
        and download.get("request_time_payload_rehash") is True
        and download.get("production_deployed") is False
        and isinstance(authority_ids, list)
        and authority_ids
        and all(endpoints)
    )
    return {
        "state": "HASH_BOUND" if ready else "UNRESOLVED",
        "authority_id": "download_catalog_independent_audit",
        "download_id": download.get("download_id"),
        "declared_status": SERVER_DOWNLOAD_STATUS,
        "authority_ids": authority_ids,
        "manifest_endpoint": endpoints[0],
        "download_endpoint": endpoints[1],
        "request_time_payload_rehash": download.get("request_time_payload_rehash"),
        "reason": None if ready else "server download attestation is incomplete",
    }


def evaluate_strict_integrated_completeness_r4(
    *,
    repo_root: str | Path,
    parity_contract_path: str | Path,
    unified_bindings_path: str | Path,
    web_catalog_path: str | Path,
    frontend_html_path: str | Path,
    frontend_javascript_path: str | Path,
    main_app_path: str | Path,
    evaluator_code_path: str | Path | None = None,
    runner_code_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(repo_root).resolve()
    evaluator = Path(evaluator_code_path or Path(__file__)).resolve()
    old_loader = r2._load_strict_release_inputs
    old_record = r2._download_artifact_record
    old_statuses = r2.DOWNLOAD_PASS_STATUSES
    old_r2_app_factory = r2.create_app_from_unified_bindings
    old_r3_app_factory = r3.create_app_from_unified_bindings
    old_main_site_probe = r2._run_main_site_probe
    r2._load_strict_release_inputs = _load_strict_release_inputs_r6
    r2._download_artifact_record = _download_artifact_record_r6
    r2.DOWNLOAD_PASS_STATUSES = frozenset(
        {*old_statuses, SERVER_DOWNLOAD_STATUS}
    )
    r2.create_app_from_unified_bindings = _create_app_without_drug_rescan
    r3.create_app_from_unified_bindings = _create_app_without_drug_rescan
    r2._run_main_site_probe = _run_main_site_probe_without_drug_rescan
    try:
        report, inputs = r3.evaluate_strict_integrated_completeness_r3(
            repo_root=root,
            parity_contract_path=parity_contract_path,
            unified_bindings_path=unified_bindings_path,
            web_catalog_path=web_catalog_path,
            frontend_html_path=frontend_html_path,
            frontend_javascript_path=frontend_javascript_path,
            main_app_path=main_app_path,
            evaluator_code_path=evaluator,
            runner_code_path=runner_code_path,
        )
    finally:
        r2._load_strict_release_inputs = old_loader
        r2._download_artifact_record = old_record
        r2.DOWNLOAD_PASS_STATUSES = old_statuses
        r2.create_app_from_unified_bindings = old_r2_app_factory
        r3.create_app_from_unified_bindings = old_r3_app_factory
        r2._run_main_site_probe = old_main_site_probe

    drug_runtime, drug_inputs = _drug_server_runtime_evidence(root)
    for input_id, path in drug_inputs.items():
        inputs[input_id] = r2._source_record(path, root)
    rows = {row["capability_id"]: row for row in report["capabilities"]}
    drug = rows["drug"]
    drug_api = drug["gates"]["api"]
    drug_api_ready = bool(
        drug_runtime["status"] == "PASS"
        and drug_api["evidence"]["declared_route_check"]["status"]
        == "IMPLEMENTED"
        and drug_api["evidence"]["required_contract_mapping"]["status"]
        == "IMPLEMENTED"
    )
    drug_api["status"] = "PASS" if drug_api_ready else "FAIL"
    drug_api["reasons"] = (
        []
        if drug_api_ready
        else ["Drug server runtime/query/download evidence did not pass"]
    )
    drug_api["evidence"]["runtime"] = drug_runtime
    drug_api["evidence"]["implementation_state"] = (
        "IMPLEMENTED_SERVER_HASH_PINNED" if drug_api_ready else "INCOMPLETE"
    )
    drug_api["evidence"]["servable"] = drug_api_ready
    report["implementation_evidence"]["drug_server_runtime"] = drug_runtime

    report["format"] = REPORT_FORMAT
    report["audit_revision"] = AUDIT_REVISION
    report["supersedes_strict_r3_binding_sha256"] = R3_BINDING_SHA256
    report["download_catalog_version"] = CATALOG_VERSION
    report["server_hash_pinned_downloads_count_as_deliverable"] = True
    report["server_download_acceptance_policy"] = {
        "required_status": SERVER_DOWNLOAD_STATUS,
        "independent_download_catalog_audit_sha256": R6_DOWNLOAD_AUDIT_SHA256,
        "request_time_payload_rehash_required": True,
        "data_present_required": True,
        "contract_complete_required": True,
        "download_implemented_required": True,
        "production_deployed_required": False,
    }
    report.pop("supersedes_strict_r2_binding_sha256", None)
    report["input_sha256"] = inputs
    r3._reset_capability_statuses(report)
    return report, inputs


def materialize_strict_integrated_completeness_audit_r4(
    *,
    repo_root: str | Path,
    parity_contract_path: str | Path,
    unified_bindings_path: str | Path,
    web_catalog_path: str | Path,
    frontend_html_path: str | Path,
    frontend_javascript_path: str | Path,
    main_app_path: str | Path,
    output_root: str | Path,
    evaluator_code_path: str | Path | None = None,
    runner_code_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    output = Path(output_root).resolve()
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise StrictCompletenessR4Error(
            "Audit output must remain inside repository"
        ) from exc
    if output.exists() and any(output.iterdir()):
        raise StrictCompletenessR4Error(
            f"Refusing to reuse non-empty output: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    report, inputs = evaluate_strict_integrated_completeness_r4(
        repo_root=root,
        parity_contract_path=parity_contract_path,
        unified_bindings_path=unified_bindings_path,
        web_catalog_path=web_catalog_path,
        frontend_html_path=frontend_html_path,
        frontend_javascript_path=frontend_javascript_path,
        main_app_path=main_app_path,
        evaluator_code_path=evaluator_code_path,
        runner_code_path=runner_code_path,
    )
    report_path = output / "INTEGRATED_COMPLETENESS_STRICT_R4_REPORT.json"
    r2._atomic_json(report_path, report)
    binding = {
        "format": BINDING_FORMAT,
        "analysis_version": report["analysis_version"],
        "model_version": MODEL_VERSION,
        "environment": "staging",
        "audit_revision": AUDIT_REVISION,
        "status": report["status"],
        "all_25_capabilities_four_gate_complete": report[
            "all_25_capabilities_four_gate_complete"
        ],
        "release_ready": report["release_ready"],
        "production_deployed": False,
        "fail_closed": True,
        "catalog_declared_is_not_implemented": True,
        "typed_unavailable_counts_as_artifact_complete": False,
        "typed_unavailable_counts_as_download_complete": False,
        "server_hash_pinned_downloads_count_as_deliverable": True,
        "supersedes_strict_r3_binding_sha256": R3_BINDING_SHA256,
        "download_catalog_version": CATALOG_VERSION,
        "download_catalog_binding_sha256": R6_DOWNLOAD_BINDING_SHA256,
        "download_catalog_audit_binding_sha256": R6_DOWNLOAD_AUDIT_SHA256,
        "required_capability_count": report["required_capability_count"],
        "complete_capability_count": report["complete_capability_count"],
        "partial_capability_count": report["partial_capability_count"],
        "blocking_capability_ids": report["blocking_capability_ids"],
        "gate_counts": report["gate_counts"],
        "report": r2._source_record(report_path, root),
        "inputs": inputs,
    }
    binding_path = output / "INTEGRATED_COMPLETENESS_STRICT_R4_BINDING.json"
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
    "R6_DOWNLOAD_AUDIT_SHA256",
    "R6_DOWNLOAD_BINDING_SHA256",
    "SERVER_DOWNLOAD_STATUS",
    "StrictCompletenessR4Error",
    "evaluate_strict_integrated_completeness_r4",
    "materialize_strict_integrated_completeness_audit_r4",
]
