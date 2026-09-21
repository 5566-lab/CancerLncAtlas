"""Independent verifier for the V3.2 implementation-level strict audit."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from .unified_staging_bindings import create_app_from_unified_bindings


RELEASE_FORMAT = "CANCERLNCATLAS_V32_INTEGRATED_COMPLETENESS_STRICT_BINDING_V2"
REPORT_FORMAT = "CANCERLNCATLAS_V32_INTEGRATED_COMPLETENESS_STRICT_AUDIT_V2"
AUDIT_REPORT_FORMAT = (
    "CANCERLNCATLAS_V32_INTEGRATED_COMPLETENESS_STRICT_INDEPENDENT_AUDIT_V1"
)
AUDIT_BINDING_FORMAT = (
    "CANCERLNCATLAS_V32_INTEGRATED_COMPLETENESS_STRICT_INDEPENDENT_BINDING_V1"
)
GATE_KINDS = ("artifact", "api", "ui", "download")


class StrictIndependentAuditError(RuntimeError):
    """Raised when an independent audit cannot be performed safely."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _safe_file(path: str | Path, root: Path, label: str) -> Path:
    source = Path(path)
    resolved = (source if source.is_absolute() else root / source).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise StrictIndependentAuditError(f"{label} escapes repository") from exc
    if source.is_symlink() or not resolved.is_file() or resolved.stat().st_size <= 0:
        raise StrictIndependentAuditError(f"{label} is missing/empty/symlink: {resolved}")
    return resolved


def _read_json(path: str | Path, root: Path, label: str) -> tuple[Path, dict[str, Any]]:
    source = _safe_file(path, root, label)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StrictIndependentAuditError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise StrictIndependentAuditError(f"{label} must be an object")
    return source, value


def _record_path(record: Mapping[str, Any], root: Path, label: str) -> Path:
    source = _safe_file(str(record.get("path", "")), root, label)
    if sha256_file(source) != str(record.get("sha256", "")).lower():
        raise StrictIndependentAuditError(f"{label} SHA256 drift")
    if record.get("bytes") not in (None, source.stat().st_size):
        raise StrictIndependentAuditError(f"{label} byte-count drift")
    return source


def _main_site_probe(root: Path, expected: Mapping[str, str]) -> dict[str, Any]:
    program = r'''
import hashlib
import json
from fastapi.testclient import TestClient
from website.backend import app as main_module
paths = [
    "/v32-staging.html",
    "/assets/v32-staging.js",
    "/v32-capability-catalog.json",
    "/v3.2-staging/health",
]
rows = []
with TestClient(main_module.app) as client:
    for path in paths:
        response = client.get(path)
        rows.append({
            "path": path,
            "http_status": response.status_code,
            "content_type": response.headers.get("content-type"),
            "sha256": hashlib.sha256(response.content).hexdigest(),
        })
print(json.dumps({"constructed": True, "responses": rows}, sort_keys=True))
'''
    environment = dict(os.environ)
    environment["CANCERLNCATLAS_REQUIRE_V31_EXACT_PATHWAY"] = "0"
    environment["CANCERLNCATLAS_ENABLE_V32_STAGING"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(root), str(root / "src"), environment.get("PYTHONPATH", "")]
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", program],
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=90,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"passed": False, "failure": f"{type(exc).__name__}: {exc}"}
    payload: dict[str, Any] = {}
    if completed.returncode == 0:
        try:
            parsed = json.loads(completed.stdout.strip().splitlines()[-1])
            if isinstance(parsed, dict):
                payload = parsed
        except (IndexError, json.JSONDecodeError):
            pass
    rows = payload.get("responses", []) if isinstance(payload, Mapping) else []
    by_path = {row.get("path"): row for row in rows if isinstance(row, Mapping)}
    static_ok = all(
        by_path.get(path, {}).get("http_status") == 200
        and by_path.get(path, {}).get("sha256") == digest
        for path, digest in expected.items()
    )
    health = by_path.get("/v3.2-staging/health", {})
    health_ok = (
        health.get("http_status") == 200
        and str(health.get("content_type", "")).startswith("application/json")
    )
    return {
        "passed": bool(completed.returncode == 0 and static_ok and health_ok),
        "constructed": payload.get("constructed") is True,
        "static_ok": static_ok,
        "health_ok": health_ok,
        "responses": rows,
        "returncode": completed.returncode,
        "failure": None if completed.returncode == 0 else completed.stderr[-2000:],
    }


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
    _, unified = _read_json(unified_path, root, "runtime unified bindings")
    drug = unified.get("bindings", {}).get("drug_response_actionability", {})
    drug_mounted = isinstance(drug, Mapping) and drug.get("status") == "MOUNTED_HASH_PINNED"
    probes: list[dict[str, Any]] = []

    def request(client: TestClient, name: str, method: str, path: str, expected_status: int, **kwargs: Any) -> Any:
        response = client.request(method, path, **kwargs)
        probes.append(
            {
                "name": name,
                "method": method,
                "path": path,
                "http_status": response.status_code,
                "expected_status": expected_status,
                "passed": response.status_code == expected_status,
            }
        )
        return response

    with TestClient(app) as client:
        request(client, "health", "GET", "/v3.2-staging/health", 200)
        exact = request(
            client,
            "exact",
            "GET",
            "/v3.2-staging/exact-pathway/associations",
            200,
            params={"cancer_id": "BRCA", "limit": 1},
        )
        exact_body = exact.json() if exact.status_code == 200 else {}
        rows = exact_body.get("rows", []) if isinstance(exact_body, Mapping) else []
        seed = str(rows[0].get("lncrna_id")) if rows else "LNC:ENSG00000117242"
        request(
            client,
            "state_rnass",
            "GET",
            "/v3.2-staging/state",
            200,
            params={"state_id": "stemness_rna::RNAss", "limit": 1},
        )
        request(
            client,
            "clinical",
            "GET",
            "/v3.2-staging/clinical",
            200,
            params={"clinical_endpoint": "OS", "cancer_id": "BRCA", "limit": 1},
        )
        for modality in ("mutation", "cnv"):
            request(
                client,
                modality,
                "GET",
                "/v3.2-staging/genomic",
                200,
                params={"modality": modality, "cancer_id": "BRCA", "limit": 1},
            )
        request(
            client,
            "evidence_direction",
            "GET",
            "/v3.2-staging/evidence/direction/probabilities",
            200,
            params={"available": "true", "limit": 1},
        )
        request(
            client,
            "single_cell_gap",
            "GET",
            "/v3.2-staging/single-cell/audit/gaps",
            200,
        )
        pseudo = request(
            client,
            "pseudotime_gap",
            "GET",
            "/v3.2-staging/single-cell/hnsc/pseudotime-availability",
            200,
            params={"level": "CELL", "limit": 1},
        )
        if pseudo.status_code == 200:
            pseudo_body = pseudo.json()
            probes[-1]["passed"] = (
                isinstance(pseudo_body, Mapping)
                and pseudo_body.get("pseudotime_numeric_values") == 0
            )
        request(
            client,
            "drug",
            "GET",
            "/v3.2-staging/drug/response-actionability",
            200 if drug_mounted else 503,
            params={
                "cancer_id": "BRCA",
                "lncrna_id": seed,
                "drug_id": "STRICT_INDEPENDENT_PROBE",
            },
        )
    return {
        "passed": observed_routes == expected_routes and all(row["passed"] for row in probes),
        "route_set_matches": observed_routes == expected_routes,
        "observed_route_count": len(observed_routes),
        "probes": probes,
    }


def independent_audit_strict_integrated_completeness(
    *,
    repo_root: str | Path,
    binding_path: str | Path,
    expected_binding_sha256: str,
    output_root: str | Path,
    auditor_code_path: str | Path | None = None,
    runner_code_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    release_path, release = _read_json(binding_path, root, "strict release binding")
    if sha256_file(release_path) != expected_binding_sha256.lower():
        raise StrictIndependentAuditError("Strict release binding SHA256 drift")
    if release.get("format") != RELEASE_FORMAT:
        raise StrictIndependentAuditError("Unexpected strict release binding format")
    report_path = _record_path(release.get("report", {}), root, "strict report")
    _, report = _read_json(report_path, root, "strict report")

    checks: list[dict[str, Any]] = []

    def check(check_id: str, passed: bool, detail: Any = None) -> None:
        checks.append({"check_id": check_id, "passed": bool(passed), "detail": detail})

    check("report_format", report.get("format") == REPORT_FORMAT, report.get("format"))
    check("environment_staging", report.get("environment") == "staging")
    check("production_false", report.get("production_deployed") is False)
    check("catalog_not_implementation", report.get("catalog_declared_is_not_implemented") is True)
    input_paths: dict[str, Path] = {}
    for input_id, record in sorted(release.get("inputs", {}).items()):
        try:
            input_paths[str(input_id)] = _record_path(record, root, f"input {input_id}")
            check(f"input_hash:{input_id}", True)
        except StrictIndependentAuditError as exc:
            check(f"input_hash:{input_id}", False, str(exc))

    rows = report.get("capabilities", [])
    check("capability_count_25", isinstance(rows, list) and len(rows) == 25)
    recomputed_complete: list[str] = []
    recomputed_blocking: list[str] = []
    gate_counts = {kind: {"pass": 0, "fail": 0} for kind in GATE_KINDS}
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            capability_id = str(row.get("capability_id"))
            gates = row.get("gates", {})
            artifact_evidence = gates.get("artifact", {}).get("evidence", {})
            api_evidence = gates.get("api", {}).get("evidence", {})
            ui_evidence = gates.get("ui", {}).get("evidence", {})
            download_evidence = gates.get("download", {}).get("evidence", {})
            expected_pass = {
                "artifact": bool(artifact_evidence.get("resolutions"))
                and all(
                    item.get("state") == "HASH_BOUND"
                    for item in artifact_evidence.get("resolutions", [])
                ),
                "api": (
                    api_evidence.get("declared_route_check", {}).get("status") == "IMPLEMENTED"
                    and api_evidence.get("required_contract_mapping", {}).get("status") == "IMPLEMENTED"
                    and api_evidence.get("runtime", {}).get("observation_valid") is True
                    and api_evidence.get("runtime", {}).get("servable_success") is True
                    and not row.get("known_gap")
                ),
                "ui": (
                    ui_evidence.get("catalog_surface_closure") is True
                    and ui_evidence.get("implementation_state") == "IMPLEMENTED_STATIC"
                    and ui_evidence.get("main_site", {}).get("servable") is True
                    and ui_evidence.get("main_site", {}).get("staging_api_mounted_on_main_app") is True
                    and not row.get("known_gap")
                ),
                "download": bool(download_evidence.get("records"))
                and all(
                    item.get("implementation_state") == "SERVABLE"
                    and not item.get("reasons")
                    for item in download_evidence.get("records", [])
                ),
            }
            for kind, passed in expected_pass.items():
                reported = gates.get(kind, {}).get("status") == "PASS"
                check(f"gate_formula:{capability_id}:{kind}", reported == passed)
                gate_counts[kind]["pass" if passed else "fail"] += 1
            complete = all(expected_pass.values())
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
        "/v32-staging.html": sha256_file(input_paths["frontend_html"]),
        "/assets/v32-staging.js": sha256_file(input_paths["frontend_javascript"]),
        "/v32-capability-catalog.json": sha256_file(input_paths["web_catalog"]),
    }
    main_site = _main_site_probe(root, expected_static)
    reported_main = report["implementation_evidence"]["frontend"]["main_site"]
    check(
        "independent_main_site_matches_report",
        main_site.get("passed")
        == bool(
            reported_main.get("servable")
            and reported_main.get("staging_api_mounted_on_main_app")
        ),
        main_site,
    )

    fail_count = sum(not row["passed"] for row in checks)
    audit_status = "PASS" if fail_count == 0 else "FAIL"
    audit_report = {
        "format": AUDIT_REPORT_FORMAT,
        "model_version": "V3.2",
        "environment": "staging",
        "status": audit_status,
        "audited_release_status": report.get("status"),
        "accepted_as_complete": audit_status == "PASS" and not recomputed_blocking,
        "accepted_as_truthful_staging_audit": audit_status == "PASS",
        "production_deployed": False,
        "independent_of_evaluator_implementation": True,
        "evaluator_imported": False,
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
        raise StrictIndependentAuditError("Audit output must remain inside repository") from exc
    if output.exists() and any(output.iterdir()):
        raise StrictIndependentAuditError(f"Refusing to reuse non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    audit_report_path = output / "INDEPENDENT_AUDIT_REPORT.json"
    _atomic_json(audit_report_path, audit_report)
    auditor = _safe_file(
        auditor_code_path or Path(__file__), root, "independent auditor code"
    )
    runner = _safe_file(
        runner_code_path
        or root / "scripts/audit_v32_integrated_completeness_strict_independent.py",
        root,
        "independent auditor runner",
    )
    audit_binding = {
        "format": AUDIT_BINDING_FORMAT,
        "model_version": "V3.2",
        "environment": "staging",
        "status": audit_status,
        "audited_release_status": report.get("status"),
        "accepted_as_complete": audit_report["accepted_as_complete"],
        "accepted_as_truthful_staging_audit": audit_report[
            "accepted_as_truthful_staging_audit"
        ],
        "production_deployed": False,
        "fail_count": fail_count,
        "release_binding": {
            "path": release_path.relative_to(root).as_posix(),
            "sha256": sha256_file(release_path),
            "bytes": release_path.stat().st_size,
        },
        "report": {
            "path": audit_report_path.relative_to(root).as_posix(),
            "sha256": sha256_file(audit_report_path),
            "bytes": audit_report_path.stat().st_size,
        },
        "auditor_code": {
            "path": auditor.relative_to(root).as_posix(),
            "sha256": sha256_file(auditor),
            "bytes": auditor.stat().st_size,
        },
        "runner_code": {
            "path": runner.relative_to(root).as_posix(),
            "sha256": sha256_file(runner),
            "bytes": runner.stat().st_size,
        },
    }
    audit_binding_path = output / "INDEPENDENT_AUDIT_BINDING.json"
    _atomic_json(audit_binding_path, audit_binding)
    return {
        "binding_path": str(audit_binding_path),
        "binding_sha256": sha256_file(audit_binding_path),
        "report_path": str(audit_report_path),
        "report_sha256": sha256_file(audit_report_path),
        "status": audit_status,
        "audited_release_status": report.get("status"),
        "accepted_as_complete": audit_report["accepted_as_complete"],
        "accepted_as_truthful_staging_audit": audit_report[
            "accepted_as_truthful_staging_audit"
        ],
        "fail_count": fail_count,
    }
