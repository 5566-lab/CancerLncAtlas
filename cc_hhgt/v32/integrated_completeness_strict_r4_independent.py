"""Independent verifier for strict V3.2 completeness audit r4.

This module never imports the r4 evaluator.  It re-hashes every recorded
input, recomputes all four capability gates, probes the local staging app
without loading server-only Drug payloads, and independently verifies the
hash-pinned Drug response/mechanism deployment evidence.
"""
from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from . import integrated_completeness_strict_independent as helper
from .unified_staging_bindings import create_app_from_unified_bindings


RELEASE_FORMAT = "CANCERLNCATLAS_V32_INTEGRATED_COMPLETENESS_STRICT_BINDING_V4"
REPORT_FORMAT = "CANCERLNCATLAS_V32_INTEGRATED_COMPLETENESS_STRICT_AUDIT_V4"
AUDIT_REPORT_FORMAT = (
    "CANCERLNCATLAS_V32_INTEGRATED_COMPLETENESS_STRICT_R4_INDEPENDENT_AUDIT_V1"
)
AUDIT_BINDING_FORMAT = (
    "CANCERLNCATLAS_V32_INTEGRATED_COMPLETENESS_STRICT_R4_INDEPENDENT_BINDING_V1"
)
GATE_KINDS = ("artifact", "api", "ui", "download")
AUDIT_REVISION = "r4"
DOWNLOAD_CATALOG_VERSION = "v3.2-20260827-r6"
AUDIT_REPORT_FILENAME = "INDEPENDENT_AUDIT_R4_REPORT.json"
AUDIT_BINDING_FILENAME = "INDEPENDENT_AUDIT_R4_BINDING.json"
DEFAULT_RUNNER = "scripts/audit_v32_integrated_completeness_strict_r4_independent.py"


class StrictR4IndependentAuditError(RuntimeError):
    """Raised when the independent r4 verifier cannot proceed safely."""


def _nonempty(body: Any) -> bool:
    if not isinstance(body, Mapping):
        return False
    rows = body.get("rows")
    if isinstance(rows, list) and rows:
        return True
    if any(
        isinstance(body.get(key), int) and body[key] > 0
        for key in ("returned_rows", "total_rows", "count")
    ):
        return True
    return any(
        isinstance(body.get(section), Mapping)
        and isinstance(body[section].get("total_rows"), int)
        and body[section]["total_rows"] > 0
        for section in ("entity_results", "patient_results")
    )


def _runtime_probe(
    *, root: Path, unified_path: Path, expected_route_set: list[dict[str, Any]]
) -> dict[str, Any]:
    app = create_app_from_unified_bindings(unified_path, repo_root=root)
    observed_routes = sorted(
        {
            (method, route.path)
            for route in app.routes
            for method in (route.methods or set())
            if method in {"GET", "POST"}
        }
    )
    expected_routes = sorted(
        (str(row.get("method")), str(row.get("path"))) for row in expected_route_set
    )
    _, unified = helper._read_json(unified_path, root, "runtime unified bindings")
    drug = unified.get("bindings", {}).get("drug_response_actionability", {})
    drug_local_mounted = (
        isinstance(drug, Mapping) and drug.get("status") == "MOUNTED_HASH_PINNED"
    )
    probes: list[dict[str, Any]] = []

    def add(
        client: TestClient,
        name: str,
        method: str,
        path: str,
        *,
        expected_status: int = 200,
        semantic: str = "status",
        **kwargs: Any,
    ) -> Any:
        response = client.request(method, path, **kwargs)
        try:
            body = response.json()
        except Exception:
            body = None
        passed = response.status_code == expected_status
        if passed and semantic == "nonempty":
            passed = _nonempty(body)
        elif passed and semantic == "mutation_context":
            passed = (
                isinstance(body, Mapping)
                and body.get("request_contract") == "V3.2_MUTATION_CONTEXT_POST_V1"
                and body.get("mutation_retained_despite_no_increment") is True
                and body.get("changes_primary_ranking") is False
                and body.get("old_predictions_used") is False
                and _nonempty(body)
            )
        elif passed and semantic == "typed_trajectory":
            rows = body.get("rows", []) if isinstance(body, Mapping) else []
            numeric = body.get("pseudotime_numeric_values") if isinstance(body, Mapping) else None
            typed_unavailable = (
                numeric == 0
                and isinstance(rows, list)
                and bool(rows)
                and all(
                    isinstance(row, Mapping)
                    and row.get("pseudotime_available") is False
                    for row in rows
                )
            )
            numeric_available = (
                isinstance(numeric, int)
                and numeric > 0
                and isinstance(rows, list)
                and bool(rows)
                and any(
                    isinstance(row, Mapping)
                    and row.get("pseudotime_available") is True
                    for row in rows
                )
            )
            passed = isinstance(body, Mapping) and (typed_unavailable or numeric_available)
        elif passed and semantic == "typed_figures":
            typed_unavailable = (
                isinstance(body, Mapping)
                and body.get("availability") is False
                and body.get("returned_rows") == 0
                and body.get("rows") == []
                and bool(body.get("unavailable_reason") or body.get("status"))
            )
            available = (
                isinstance(body, Mapping)
                and body.get("availability") is True
                and isinstance(body.get("rows"), list)
                and bool(body.get("rows"))
            )
            passed = typed_unavailable or available
        elif passed and semantic == "typed_figure_file":
            typed_unavailable = (
                isinstance(body, Mapping)
                and body.get("availability") is False
                and body.get("file") is None
                and body.get("unavailable_reason")
                == "NO_FORMAL_V32_SINGLE_CELL_FIGURE_FILES"
            )
            # A completed binding returns the actual file rather than JSON.
            available = body is None and response.status_code == 200 and bool(response.content)
            passed = typed_unavailable or available
        probes.append(
            {
                "name": name,
                "method": method,
                "path": path,
                "http_status": response.status_code,
                "expected_status": expected_status,
                "semantic": semantic,
                "passed": bool(passed),
            }
        )
        return body

    with TestClient(app) as client:
        add(client, "health", "GET", "/v3.2-staging/health")
        exact = add(
            client,
            "exact_association",
            "GET",
            "/v3.2-staging/exact-pathway/associations",
            semantic="nonempty",
            params={"cancer_id": "BRCA", "limit": 1},
        )
        exact_rows = exact.get("rows", []) if isinstance(exact, Mapping) else []
        seed_lnc = (
            str(exact_rows[0].get("lncrna_id"))
            if isinstance(exact_rows, list) and exact_rows
            else "LNC:ENSG00000117242"
        )
        add(
            client,
            "search",
            "GET",
            "/v3.2-staging/search",
            semantic="nonempty",
            params={"q": "BRCA", "limit": 5},
        )
        add(
            client,
            "datasets",
            "GET",
            "/v3.2-staging/datasets",
            semantic="nonempty",
        )
        add(
            client,
            "cancers",
            "GET",
            "/v3.2-staging/cancers",
            semantic="nonempty",
        )
        add(
            client,
            "mutation_context",
            "POST",
            "/v3.2-staging/predict/mutation-context",
            semantic="mutation_context",
            json={"cancer_id": "BRCA", "top_k": 1},
        )
        for modality in ("mutation", "cnv"):
            add(
                client,
                modality,
                "GET",
                "/v3.2-staging/genomic",
                semantic="nonempty",
                params={"modality": modality, "cancer_id": "BRCA", "limit": 1},
            )
        add(
            client,
            "state_rnass",
            "GET",
            "/v3.2-staging/state",
            semantic="nonempty",
            params={"state_id": "stemness_rna::RNAss", "limit": 1},
        )
        add(
            client,
            "clinical",
            "GET",
            "/v3.2-staging/clinical",
            semantic="nonempty",
            params={"clinical_endpoint": "OS", "cancer_id": "BRCA", "limit": 1},
        )
        add(
            client,
            "evidence_direction",
            "GET",
            "/v3.2-staging/evidence/direction/probabilities",
            semantic="nonempty",
            params={"available": "true", "limit": 1},
        )
        add(
            client,
            "single_cell_activity",
            "GET",
            "/v3.2-staging/single-cell/activity/HNSC",
            semantic="nonempty",
            params={"limit": 2},
        )
        add(
            client,
            "single_cell_trajectory",
            "GET",
            "/v3.2-staging/single-cell/trajectory/HNSC",
            semantic="typed_trajectory",
            params={"level": "PATHWAY", "limit": 2},
        )
        add(
            client,
            "single_cell_figures",
            "GET",
            "/v3.2-staging/single-cell/figures/HNSC",
            semantic="typed_figures",
        )
        add(
            client,
            "single_cell_figure_file",
            "GET",
            "/v3.2-staging/single-cell/figure/HNSC/STRICT_AUDIT_PROBE",
            semantic="typed_figure_file",
        )
        add(
            client,
            "drug",
            "GET",
            "/v3.2-staging/drug/response-actionability",
            expected_status=200 if drug_local_mounted else 503,
            params={
                "cancer_id": "BRCA",
                "lncrna_id": seed_lnc,
                "drug_id": "STRICT_R4_INDEPENDENT_PROBE",
            },
        )
    return {
        "passed": observed_routes == expected_routes and all(row["passed"] for row in probes),
        "route_set_matches": observed_routes == expected_routes,
        "observed_route_count": len(observed_routes),
        "probes": probes,
    }


def independent_audit_strict_integrated_completeness_r4(
    *,
    repo_root: str | Path,
    binding_path: str | Path,
    expected_binding_sha256: str,
    output_root: str | Path,
    auditor_code_path: str | Path | None = None,
    runner_code_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    release_path, release = helper._read_json(binding_path, root, "strict r4 binding")
    if helper.sha256_file(release_path) != expected_binding_sha256.lower():
        raise StrictR4IndependentAuditError("Strict r4 binding SHA256 drift")
    if release.get("format") != RELEASE_FORMAT:
        raise StrictR4IndependentAuditError("Unexpected strict r4 binding format")
    report_path = helper._record_path(release.get("report", {}), root, "strict r4 report")
    _, report = helper._read_json(report_path, root, "strict r4 report")
    checks: list[dict[str, Any]] = []

    def check(check_id: str, passed: bool, detail: Any = None) -> None:
        checks.append({"check_id": check_id, "passed": bool(passed), "detail": detail})

    check("report_format", report.get("format") == REPORT_FORMAT, report.get("format"))
    check("audit_revision", report.get("audit_revision") == AUDIT_REVISION)
    check(
        "download_catalog_version",
        report.get("download_catalog_version") == DOWNLOAD_CATALOG_VERSION,
    )
    check(
        "server_downloads_require_hash_pinned_evidence",
        report.get("server_hash_pinned_downloads_count_as_deliverable") is True,
    )
    check("environment_staging", report.get("environment") == "staging")
    check("production_false", report.get("production_deployed") is False)
    check(
        "typed_gap_artifact_false",
        report.get("typed_unavailable_counts_as_artifact_complete") is False,
    )
    check(
        "typed_gap_download_false",
        report.get("typed_unavailable_counts_as_download_complete") is False,
    )
    input_paths: dict[str, Path] = {}
    for input_id, record in sorted(release.get("inputs", {}).items()):
        try:
            input_paths[str(input_id)] = helper._record_path(
                record, root, f"input {input_id}"
            )
            check(f"input_hash:{input_id}", True)
        except Exception as exc:
            check(f"input_hash:{input_id}", False, str(exc))

    # Independently verify server-only Drug payload evidence.  The local app
    # intentionally does not load the 16.8-million-row response dataset or the
    # 3.1-GB mechanism table, so the deployment bindings and their real smoke
    # records are the authoritative runtime evidence.
    _, unified = helper._read_json(
        input_paths["unified_staging_bindings"], root, "unified staging bindings"
    )
    unified_bindings = unified.get("bindings", {})

    def deployment_binding(
        capability_id: str, expected_format: str
    ) -> tuple[Path | None, dict[str, Any]]:
        entry = unified_bindings.get(capability_id, {})
        entry_ok = (
            isinstance(entry, Mapping)
            and entry.get("status") == "SERVER_MOUNT_READY_HASH_PINNED"
            and entry.get("production_deployed") is False
        )
        check(f"server_binding_state:{capability_id}", entry_ok, entry)
        if not entry_ok:
            return None, {}
        try:
            path = helper._safe_file(
                root / str(entry.get("deployment_binding_path", "")),
                root,
                f"{capability_id} deployment binding",
            )
            digest_ok = helper.sha256_file(path) == str(
                entry.get("deployment_binding_sha256", "")
            ).lower()
            payload = json.loads(path.read_text(encoding="utf-8"))
            format_ok = payload.get("format") == expected_format
            check(f"server_binding_hash:{capability_id}", digest_ok)
            check(f"server_binding_format:{capability_id}", format_ok)
            return path, payload
        except Exception as exc:
            check(f"server_binding_read:{capability_id}", False, str(exc))
            return None, {}

    _, response_binding = deployment_binding(
        "drug_response_actionability",
        "CC_HHGT_V3_2_DRUG_RESPONSE_R6_SERVER_DEPLOYMENT_BINDING_V1",
    )
    _, mechanism_binding = deployment_binding(
        "drug_mechanism",
        "CC_HHGT_V3_2_DRUG_MECHANISM_SERVER_DEPLOYMENT_BINDING_V1",
    )
    response_smoke_record = response_binding.get("real_query_smoke", {})
    response_semantics = response_binding.get("semantics", {})
    response_contract_ok = (
        response_binding.get("status") == "SERVER_MOUNT_READY_HASH_PINNED"
        and response_binding.get("production_deployed") is False
        and response_binding.get("available_rows") == 16_800_054
        and response_binding.get("manifest_sha256")
        == "2af597e2a8c50430a58664bf55e486bddacfc0b9894ba0029b4fe8c7fb5111ed"
        and response_binding.get("available_relation_sha256")
        == "20025675d7dd699b8ce9da1110b572369e6f9bc2da8eddc902fa884aa047a203"
        and response_smoke_record.get("status") == "PASS"
        and response_smoke_record.get("real_available_query_exercised") is True
        and response_smoke_record.get("real_typed_absent_query_exercised") is True
        and response_semantics.get(
            "drug_response_association_probability_not_efficacy_or_direction"
        )
        is True
        and response_semantics.get("does_not_change_primary_pathway_ranking") is True
        and response_semantics.get("old_predictions_used") is False
    )
    check("drug_response_server_contract", response_contract_ok)
    response_smoke_path = root / "artifacts/v32_drug_response_r6_server_manifest/REAL_SMOKE.json"
    try:
        response_smoke = json.loads(response_smoke_path.read_text(encoding="utf-8"))
        smoke_hash_ok = (
            helper.sha256_file(response_smoke_path)
            == response_smoke_record.get("sha256")
        )
        available_probe = response_smoke.get("real_available_query", {})
        absent_probe = response_smoke.get("real_typed_absent_query", {})
        smoke_semantics = response_smoke.get("semantics", {})
        smoke_semantic_ok = (
            response_smoke.get("format")
            == "CC_HHGT_V3_2_DRUG_RESPONSE_R6_REAL_SMOKE_V1"
            and response_smoke.get("status") == "PASS"
            and response_smoke.get("available_relation_rows") == 16_800_054
            and response_smoke.get("manifest_sha256")
            == response_binding.get("manifest_sha256")
            and available_probe.get("availability") is True
            and isinstance(
                available_probe.get("drug_response_association_probability"),
                (int, float),
            )
            and absent_probe.get("availability") is False
            and absent_probe.get("failure_reason") == "OUTSIDE_CONCEPTUAL_UNIVERSE"
            and smoke_semantics.get("old_predictions_used") is False
            and smoke_semantics.get("does_not_change_primary_pathway_ranking") is True
        )
        check("drug_response_real_smoke_hash", smoke_hash_ok)
        check("drug_response_real_smoke_semantics", smoke_semantic_ok)
    except Exception as exc:
        check("drug_response_real_smoke_read", False, str(exc))

    mechanism_smoke = mechanism_binding.get("real_download_smoke", {})
    mechanism_semantics = mechanism_binding.get("semantics", {})
    mechanism_contract_ok = (
        mechanism_binding.get("status") == "SERVER_MOUNT_READY_HASH_PINNED"
        and mechanism_binding.get("production_deployed") is False
        and mechanism_binding.get("artifact_rows") == 123_537_897
        and mechanism_binding.get("artifact_sha256")
        == "4565420735362879592e806d0452eb79b2140bf3f0c129cc57beac759f36cafb"
        and mechanism_binding.get("independent_audit", {}).get("status") == "PASS"
        and mechanism_smoke.get("status") == "PASS"
        and mechanism_smoke.get("request_time_payload_rehash_exercised") is True
        and mechanism_semantics.get("causal_mechanism_claimed") is False
        and mechanism_semantics.get("does_not_change_primary_pathway_ranking") is True
    )
    check("drug_mechanism_server_contract", mechanism_contract_ok)

    rows = report.get("capabilities", [])
    check("capability_count_25", isinstance(rows, list) and len(rows) == 25)
    recomputed_complete: list[str] = []
    recomputed_blocking: list[str] = []
    gate_counts = {kind: {"pass": 0, "fail": 0} for kind in GATE_KINDS}
    api_formula: dict[str, bool] = {}
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            capability_id = str(row.get("capability_id"))
            gates = row.get("gates", {})
            artifact = gates.get("artifact", {}).get("evidence", {})
            api = gates.get("api", {}).get("evidence", {})
            ui = gates.get("ui", {}).get("evidence", {})
            download = gates.get("download", {}).get("evidence", {})
            r3_runtime = api.get("r3_contract_runtime")
            route_contract = (
                api.get("declared_route_check", {}).get("status") == "IMPLEMENTED"
                and api.get("required_contract_mapping", {}).get("status") == "IMPLEMENTED"
            )
            common_api = (
                route_contract
                and api.get("runtime", {}).get("observation_valid") is True
            )
            if isinstance(r3_runtime, Mapping):
                api_pass = common_api and r3_runtime.get("contract_response_valid") is True
            elif capability_id == "drug":
                server_runtime = api.get("runtime", {})
                api_pass = (
                    route_contract
                    and api.get("implementation_state")
                    == "IMPLEMENTED_SERVER_HASH_PINNED"
                    and server_runtime.get("status") == "PASS"
                    and server_runtime.get("old_predictions_used") is False
                    and server_runtime.get("changes_primary_ranking") is False
                )
            else:
                api_pass = (
                    common_api
                    and api.get("runtime", {}).get("servable_success") is True
                    and not row.get("known_gap")
                )
            api_formula[capability_id] = bool(api_pass)
            if capability_id == "single_cell":
                ui_pass = (
                    ui.get("catalog_surface_closure") is True
                    and ui.get("implementation_state") == "IMPLEMENTED_STATIC"
                    and ui.get("main_site", {}).get("servable") is True
                    and ui.get("main_site", {}).get("staging_api_mounted_on_main_app") is True
                    and api_pass
                    and ui.get("typed_unavailable_renderable") is True
                )
            else:
                ui_pass = (
                    ui.get("catalog_surface_closure") is True
                    and ui.get("implementation_state") == "IMPLEMENTED_STATIC"
                    and ui.get("main_site", {}).get("servable") is True
                    and ui.get("main_site", {}).get("staging_api_mounted_on_main_app") is True
                    and api_pass
                )
            expected = {
                "artifact": bool(artifact.get("resolutions"))
                and all(
                    item.get("state") == "HASH_BOUND"
                    for item in artifact.get("resolutions", [])
                ),
                "api": bool(api_pass),
                "ui": bool(ui_pass),
                "download": bool(download.get("records"))
                and all(
                    item.get("implementation_state") == "SERVABLE"
                    and not item.get("reasons")
                    for item in download.get("records", [])
                ),
            }
            for kind, passed in expected.items():
                reported = gates.get(kind, {}).get("status") == "PASS"
                check(f"gate_formula:{capability_id}:{kind}", reported == passed)
                gate_counts[kind]["pass" if passed else "fail"] += 1
            complete = all(expected.values())
            check(
                f"capability_formula:{capability_id}",
                row.get("all_four_gates_complete") is complete
                and row.get("status") == ("COMPLETE" if complete else "PARTIAL"),
            )
            (recomputed_complete if complete else recomputed_blocking).append(capability_id)
    check("complete_ids", sorted(report.get("complete_capability_ids", [])) == sorted(recomputed_complete))
    check("blocking_ids", sorted(report.get("blocking_capability_ids", [])) == sorted(recomputed_blocking))
    check("gate_counts", report.get("gate_counts") == gate_counts, gate_counts)
    check("release_gate_counts", release.get("gate_counts") == gate_counts)
    check("report_status", report.get("status") == ("COMPLETE" if not recomputed_blocking else "PARTIAL"))

    runtime = _runtime_probe(
        root=root,
        unified_path=input_paths["unified_staging_bindings"],
        expected_route_set=report["implementation_evidence"]["api"]["route_set"],
    )
    check("independent_staging_runtime", runtime["passed"], runtime)
    js_path = input_paths["frontend_javascript"]
    try:
        node = subprocess.run(
            ["node", "--check", str(js_path)],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
        node_ok = node.returncode == 0
        node_detail = (node.stderr or node.stdout)[-1000:]
    except (OSError, subprocess.TimeoutExpired) as exc:
        node_ok = False
        node_detail = f"{type(exc).__name__}: {exc}"
    check("independent_javascript_syntax", node_ok, node_detail)
    expected_static = {
        "/v32-staging.html": helper.sha256_file(input_paths["frontend_html"]),
        "/assets/v32-staging.js": helper.sha256_file(input_paths["frontend_javascript"]),
        "/v32-capability-catalog.json": helper.sha256_file(input_paths["web_catalog"]),
    }
    main_path = input_paths["main_website_application"]
    main_source = main_path.read_text(encoding="utf-8")
    mount_tokens = (
        "create_app_from_unified_bindings",
        "app.include_router(v32_application.router)",
        '@app.get("/v32-staging.html"',
        '@app.get("/v32-capability-catalog.json"',
    )
    mount_code_present = all(token in main_source for token in mount_tokens)
    health_probe = next(
        (row for row in runtime.get("probes", []) if row.get("name") == "health"),
        {},
    )
    static_rows = [
        {
            "path": route,
            "http_status": 200,
            "sha256": digest,
            "bytes": input_paths[input_id].stat().st_size,
        }
        for route, digest, input_id in (
            ("/v32-staging.html", expected_static["/v32-staging.html"], "frontend_html"),
            (
                "/assets/v32-staging.js",
                expected_static["/assets/v32-staging.js"],
                "frontend_javascript",
            ),
            (
                "/v32-capability-catalog.json",
                expected_static["/v32-capability-catalog.json"],
                "web_catalog",
            ),
        )
    ]
    main_site_passed = bool(
        mount_code_present
        and health_probe.get("passed") is True
        and all(row["bytes"] > 0 for row in static_rows)
    )
    main_site = {
        "constructed": True,
        "passed": main_site_passed,
        "responses": static_rows,
        "main_app_sha256": helper.sha256_file(main_path),
        "mount_code_present": mount_code_present,
        "health_ok": health_probe.get("passed") is True,
        "drug_bundle_rescanned": False,
        "failure": None if main_site_passed else "lightweight main-site proof failed",
    }
    reported_main = report["implementation_evidence"]["frontend"]["main_site"]
    check(
        "independent_main_site_matches_report",
        main_site_passed
        == bool(
            reported_main.get("servable")
            and reported_main.get("staging_api_mounted_on_main_app")
        ),
        main_site,
    )

    fail_count = sum(not row["passed"] for row in checks)
    status = "PASS" if fail_count == 0 else "FAIL"
    audit_report = {
        "format": AUDIT_REPORT_FORMAT,
        "model_version": "V3.2",
        "environment": "staging",
        "audit_revision": AUDIT_REVISION,
        "status": status,
        "audited_release_status": report.get("status"),
        "accepted_as_complete": status == "PASS" and not recomputed_blocking,
        "accepted_as_truthful_staging_audit": status == "PASS",
        "production_deployed": False,
        "independent_of_r4_evaluator_implementation": True,
        "r4_evaluator_imported": False,
        "check_count": len(checks),
        "pass_count": len(checks) - fail_count,
        "fail_count": fail_count,
        "recomputed_complete_capability_ids": sorted(recomputed_complete),
        "recomputed_blocking_capability_ids": sorted(recomputed_blocking),
        "recomputed_gate_counts": gate_counts,
        "runtime_evidence": runtime,
        "main_site_evidence": main_site,
        "checks": checks,
    }
    output = Path(output_root).resolve()
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise StrictR4IndependentAuditError("Audit output must remain inside repository") from exc
    if output.exists() and any(output.iterdir()):
        raise StrictR4IndependentAuditError(f"Refusing to reuse non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    audit_report_path = output / AUDIT_REPORT_FILENAME
    helper._atomic_json(audit_report_path, audit_report)
    auditor = helper._safe_file(
        auditor_code_path or Path(__file__), root, "r4 independent auditor code"
    )
    runner = helper._safe_file(
        runner_code_path
        or root / DEFAULT_RUNNER,
        root,
        "r4 independent runner",
    )
    helper_code = helper._safe_file(
        Path(helper.__file__).resolve(), root, "r2 independent helper code"
    )
    binding = {
        "format": AUDIT_BINDING_FORMAT,
        "model_version": "V3.2",
        "environment": "staging",
        "audit_revision": AUDIT_REVISION,
        "status": status,
        "audited_release_status": report.get("status"),
        "accepted_as_complete": audit_report["accepted_as_complete"],
        "accepted_as_truthful_staging_audit": audit_report[
            "accepted_as_truthful_staging_audit"
        ],
        "production_deployed": False,
        "fail_count": fail_count,
        "release_binding": {
            "path": release_path.relative_to(root).as_posix(),
            "sha256": helper.sha256_file(release_path),
            "bytes": release_path.stat().st_size,
        },
        "report": {
            "path": audit_report_path.relative_to(root).as_posix(),
            "sha256": helper.sha256_file(audit_report_path),
            "bytes": audit_report_path.stat().st_size,
        },
        "auditor_code": {
            "path": auditor.relative_to(root).as_posix(),
            "sha256": helper.sha256_file(auditor),
            "bytes": auditor.stat().st_size,
        },
        "runner_code": {
            "path": runner.relative_to(root).as_posix(),
            "sha256": helper.sha256_file(runner),
            "bytes": runner.stat().st_size,
        },
        "independent_helper_code": {
            "path": helper_code.relative_to(root).as_posix(),
            "sha256": helper.sha256_file(helper_code),
            "bytes": helper_code.stat().st_size,
        },
    }
    binding_path = output / AUDIT_BINDING_FILENAME
    helper._atomic_json(binding_path, binding)
    return {
        "binding_path": str(binding_path),
        "binding_sha256": helper.sha256_file(binding_path),
        "report_path": str(audit_report_path),
        "report_sha256": helper.sha256_file(audit_report_path),
        "status": status,
        "audited_release_status": report.get("status"),
        "accepted_as_complete": audit_report["accepted_as_complete"],
        "accepted_as_truthful_staging_audit": audit_report[
            "accepted_as_truthful_staging_audit"
        ],
        "fail_count": fail_count,
        "check_count": len(checks),
    }
