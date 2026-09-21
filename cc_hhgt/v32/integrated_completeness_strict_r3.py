"""Strict V3.2 completeness audit r3 with runtime-verified compatibility routes.

R3 supersedes, but does not mutate, the immutable r2 audit.  It distinguishes
an implemented API that truthfully returns typed unavailability from a
materialized artifact/download.  Typed absence may satisfy API/UI contract
handling, but can never satisfy artifact or download completeness.
"""
from __future__ import annotations

import copy
import gc
from pathlib import Path
from typing import Any, Mapping

from fastapi.testclient import TestClient

from . import integrated_completeness_strict as r2
from .unified_staging_bindings import create_app_from_unified_bindings


REPORT_FORMAT = "CANCERLNCATLAS_V32_INTEGRATED_COMPLETENESS_STRICT_AUDIT_V3"
BINDING_FORMAT = "CANCERLNCATLAS_V32_INTEGRATED_COMPLETENESS_STRICT_BINDING_V3"
MODEL_VERSION = "V3.2"
R2_BINDING_SHA256 = "8c6b772b466e183f31c3acdbd880b56c7758d615fc32e73b77323ded9f1ae653"


API_EQUIVALENTS = copy.deepcopy(r2.API_EQUIVALENTS)
API_EQUIVALENTS["exact_pathway"].update(
    {
        "GET /api/site/search": ("GET", "/v3.2-staging/search"),
        "GET /api/site/datasets": ("GET", "/v3.2-staging/datasets"),
        "GET /api/site/cancers": ("GET", "/v3.2-staging/cancers"),
    }
)
API_EQUIVALENTS["mutation"]["POST /api/site/predict/mutation-context"] = (
    "POST",
    "/v3.2-staging/predict/mutation-context",
)
API_EQUIVALENTS["single_cell"].update(
    {
        "GET /api/site/sc-activity/{cancer}": (
            "GET",
            "/v3.2-staging/single-cell/activity/{cancer_id}",
        ),
        "GET /api/site/sc-trajectory/{cancer}": (
            "GET",
            "/v3.2-staging/single-cell/trajectory/{cancer_id}",
        ),
        "GET /api/site/sc-figures/{cancer}": (
            "GET",
            "/v3.2-staging/single-cell/figures/{cancer_id}",
        ),
        "GET /api/site/sc-figure/{cancer}/{figure}": (
            "GET",
            "/v3.2-staging/single-cell/figure/{cancer_id}/{figure_id}",
        ),
    }
)

CAPABILITY_UI_ACTIONS = copy.deepcopy(r2.CAPABILITY_UI_ACTIONS)
CAPABILITY_UI_ACTIONS["single_cell"] = (
    "sc",
    "sc-expression",
    "sc-audit",
    "sc-gaps",
    "ucell",
    "sc-activity",
    "pseudotime",
    "sc-figures",
)


class StrictCompletenessR3Error(RuntimeError):
    """Raised when r3 cannot issue a fail-closed result."""


def _body_rows(body: Any) -> list[Any]:
    if not isinstance(body, Mapping):
        return []
    rows = body.get("rows", [])
    return rows if isinstance(rows, list) else []


def _probe_record(
    *,
    name: str,
    method: str,
    path: str,
    response: Any,
    semantic: str,
) -> dict[str, Any]:
    try:
        body = response.json()
    except Exception:
        body = None
    rows = _body_rows(body)
    data_available: bool | None = None
    if semantic == "nonempty":
        valid = response.status_code == 200 and r2._body_has_payload(body)
        data_available = bool(valid)
    elif semantic == "mutation_context":
        valid = (
            response.status_code == 200
            and isinstance(body, Mapping)
            and body.get("request_contract") == "V3.2_MUTATION_CONTEXT_POST_V1"
            and body.get("mutation_retained_despite_no_increment") is True
            and body.get("changes_primary_ranking") is False
            and body.get("old_predictions_used") is False
            and r2._body_has_payload(body)
        )
        data_available = bool(valid)
    elif semantic == "typed_trajectory":
        valid = (
            response.status_code == 200
            and isinstance(body, Mapping)
            and body.get("pseudotime_numeric_values") == 0
            and all(
                isinstance(row, Mapping)
                and row.get("pseudotime_available") is False
                for row in rows
            )
        )
        data_available = False
    elif semantic == "typed_figures":
        valid = (
            response.status_code == 200
            and isinstance(body, Mapping)
            and body.get("availability") is False
            and body.get("returned_rows") == 0
            and not rows
            and bool(body.get("unavailable_reason") or body.get("status"))
        )
        data_available = False
    elif semantic == "typed_figure_file":
        valid = (
            response.status_code == 200
            and isinstance(body, Mapping)
            and body.get("availability") is False
            and body.get("file") is None
            and body.get("unavailable_reason")
            == "NO_FORMAL_V32_SINGLE_CELL_FIGURE_FILES"
        )
        data_available = False
    else:
        raise StrictCompletenessR3Error(f"Unknown r3 runtime semantic: {semantic}")
    summary: dict[str, Any] = {}
    if isinstance(body, Mapping):
        for key in (
            "module",
            "query_kind",
            "returned_rows",
            "availability",
            "unavailable_reason",
            "pseudotime_numeric_values",
            "request_contract",
            "mutation_retained_despite_no_increment",
            "changes_primary_ranking",
            "old_predictions_used",
        ):
            if key in body:
                summary[key] = body.get(key)
    return {
        "name": name,
        "method": method,
        "path": path,
        "http_status": response.status_code,
        "semantic": semantic,
        "contract_response_valid": bool(valid),
        "data_available": data_available,
        "body_summary": summary,
    }


def _run_r3_contract_probes(
    *, root: Path, unified_path: Path
) -> dict[str, dict[str, Any]]:
    gc.collect()
    app = create_app_from_unified_bindings(unified_path, repo_root=root)
    results: dict[str, list[dict[str, Any]]] = {
        "exact_pathway": [],
        "mutation": [],
        "single_cell": [],
    }

    def add(
        client: TestClient,
        capability_id: str,
        name: str,
        method: str,
        path: str,
        semantic: str,
        **kwargs: Any,
    ) -> Any:
        response = client.request(method, path, **kwargs)
        results[capability_id].append(
            _probe_record(
                name=name,
                method=method,
                path=path,
                response=response,
                semantic=semantic,
            )
        )
        return response

    with TestClient(app) as client:
        add(
            client,
            "exact_pathway",
            "search",
            "GET",
            "/v3.2-staging/search",
            "nonempty",
            params={"q": "BRCA", "limit": 5},
        )
        add(
            client,
            "exact_pathway",
            "datasets",
            "GET",
            "/v3.2-staging/datasets",
            "nonempty",
        )
        add(
            client,
            "exact_pathway",
            "cancers",
            "GET",
            "/v3.2-staging/cancers",
            "nonempty",
        )
        add(
            client,
            "mutation",
            "mutation_context",
            "POST",
            "/v3.2-staging/predict/mutation-context",
            "mutation_context",
            json={"cancer_id": "BRCA", "top_k": 1},
        )
        add(
            client,
            "single_cell",
            "activity_hnsc",
            "GET",
            "/v3.2-staging/single-cell/activity/HNSC",
            "nonempty",
            params={"limit": 2},
        )
        add(
            client,
            "single_cell",
            "trajectory_hnsc",
            "GET",
            "/v3.2-staging/single-cell/trajectory/HNSC",
            "typed_trajectory",
            params={"level": "PATHWAY", "limit": 2},
        )
        add(
            client,
            "single_cell",
            "figures_hnsc",
            "GET",
            "/v3.2-staging/single-cell/figures/HNSC",
            "typed_figures",
        )
        add(
            client,
            "single_cell",
            "figure_file_hnsc",
            "GET",
            "/v3.2-staging/single-cell/figure/HNSC/STRICT_AUDIT_PROBE",
            "typed_figure_file",
        )
    return {
        capability_id: {
            "contract_response_valid": bool(rows)
            and all(row["contract_response_valid"] for row in rows),
            "all_data_available": bool(rows)
            and all(row["data_available"] is True for row in rows),
            "typed_unavailable_observed": any(
                row["contract_response_valid"]
                and row["data_available"] is False
                for row in rows
            ),
            "probes": rows,
        }
        for capability_id, rows in results.items()
    }


def _reset_capability_statuses(report: dict[str, Any]) -> None:
    complete: list[str] = []
    blocking: list[str] = []
    for row in report["capabilities"]:
        all_complete = all(
            row["gates"][kind]["status"] == "PASS" for kind in r2.GATE_KINDS
        )
        row["all_four_gates_complete"] = all_complete
        row["status"] = "COMPLETE" if all_complete else "PARTIAL"
        (complete if all_complete else blocking).append(row["capability_id"])
    report["complete_capability_ids"] = complete
    report["blocking_capability_ids"] = blocking
    report["complete_capability_count"] = len(complete)
    report["partial_capability_count"] = len(blocking)
    report["status"] = "COMPLETE" if not blocking else "PARTIAL"
    report["all_25_capabilities_four_gate_complete"] = not blocking
    report["release_ready"] = not blocking
    report["gate_counts"] = {
        kind: {
            "pass": sum(
                row["gates"][kind]["status"] == "PASS"
                for row in report["capabilities"]
            ),
            "fail": sum(
                row["gates"][kind]["status"] == "FAIL"
                for row in report["capabilities"]
            ),
        }
        for kind in r2.GATE_KINDS
    }


def evaluate_strict_integrated_completeness_r3(
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
    old_api = r2.API_EQUIVALENTS
    old_ui = r2.CAPABILITY_UI_ACTIONS
    r2.API_EQUIVALENTS = API_EQUIVALENTS
    r2.CAPABILITY_UI_ACTIONS = CAPABILITY_UI_ACTIONS
    try:
        report, inputs = r2.evaluate_strict_integrated_completeness(
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
        r2.API_EQUIVALENTS = old_api
        r2.CAPABILITY_UI_ACTIONS = old_ui

    unified_path = Path(unified_bindings_path).resolve()
    r3_runtime = _run_r3_contract_probes(root=root, unified_path=unified_path)
    rows = {row["capability_id"]: row for row in report["capabilities"]}
    for capability_id in ("exact_pathway", "mutation", "single_cell"):
        row = rows[capability_id]
        api = row["gates"]["api"]
        evidence = api["evidence"]
        runtime = r3_runtime[capability_id]
        implementation_pass = (
            evidence["declared_route_check"]["status"] == "IMPLEMENTED"
            and evidence["required_contract_mapping"]["status"] == "IMPLEMENTED"
            and evidence["runtime"]["observation_valid"] is True
            and runtime["contract_response_valid"] is True
        )
        api["status"] = "PASS" if implementation_pass else "FAIL"
        api["reasons"] = (
            []
            if implementation_pass
            else ["one or more r3 compatibility routes failed runtime validation"]
        )
        evidence["implementation_state"] = (
            "IMPLEMENTED" if implementation_pass else "INCOMPLETE"
        )
        evidence["servable"] = implementation_pass
        evidence["r3_contract_runtime"] = runtime
        evidence["typed_unavailable_satisfies_artifact_or_download"] = False

    sc = rows["single_cell"]
    sc_ui = sc["gates"]["ui"]
    sc_ui_evidence = sc_ui["evidence"]
    sc_ui_pass = (
        sc_ui_evidence["catalog_surface_closure"] is True
        and sc_ui_evidence["implementation_state"] == "IMPLEMENTED_STATIC"
        and sc_ui_evidence["main_site"]["servable"] is True
        and sc_ui_evidence["main_site"]["staging_api_mounted_on_main_app"] is True
        and sc["gates"]["api"]["status"] == "PASS"
    )
    sc_ui["status"] = "PASS" if sc_ui_pass else "FAIL"
    sc_ui["reasons"] = (
        []
        if sc_ui_pass
        else ["single-cell typed-unavailable UI contract is not fully implemented"]
    )
    sc_ui_evidence["typed_unavailable_renderable"] = sc_ui_pass
    sc_ui_evidence["typed_unavailable_satisfies_artifact_or_download"] = False

    report["format"] = REPORT_FORMAT
    report["audit_revision"] = "r3"
    report["supersedes_strict_r2_binding_sha256"] = R2_BINDING_SHA256
    report["typed_unavailable_may_satisfy_api_contract"] = True
    report["typed_unavailable_may_satisfy_ui_rendering_contract"] = True
    report["typed_unavailable_counts_as_artifact_complete"] = False
    report["typed_unavailable_counts_as_download_complete"] = False
    report["implementation_evidence"]["r3_contract_runtime"] = r3_runtime

    r2_core = r2._safe_path(
        Path(r2.__file__).resolve(), root, "strict r2 core evaluator code"
    )
    inputs["strict_r2_core_evaluator_code"] = r2._source_record(r2_core, root)
    inputs["evaluator_code"] = r2._source_record(evaluator, root)
    report["input_sha256"] = inputs
    _reset_capability_statuses(report)
    return report, inputs


def materialize_strict_integrated_completeness_audit_r3(
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
        raise StrictCompletenessR3Error("Audit output must remain inside repository") from exc
    if output.exists() and any(output.iterdir()):
        raise StrictCompletenessR3Error(f"Refusing to reuse non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    report, inputs = evaluate_strict_integrated_completeness_r3(
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
    report_path = output / "INTEGRATED_COMPLETENESS_STRICT_R3_REPORT.json"
    r2._atomic_json(report_path, report)
    binding = {
        "format": BINDING_FORMAT,
        "analysis_version": report["analysis_version"],
        "model_version": MODEL_VERSION,
        "environment": "staging",
        "audit_revision": "r3",
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
        "supersedes_strict_r2_binding_sha256": R2_BINDING_SHA256,
        "required_capability_count": report["required_capability_count"],
        "complete_capability_count": report["complete_capability_count"],
        "partial_capability_count": report["partial_capability_count"],
        "blocking_capability_ids": report["blocking_capability_ids"],
        "gate_counts": report["gate_counts"],
        "report": r2._source_record(report_path, root),
        "inputs": inputs,
    }
    binding_path = output / "INTEGRATED_COMPLETENESS_STRICT_R3_BINDING.json"
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
