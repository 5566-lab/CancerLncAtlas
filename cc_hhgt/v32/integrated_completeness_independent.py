"""Independent verifier for a V3.2 integrated-completeness binding.

This module deliberately does not import the materializer.  It re-hashes the
four authorities and their transitive binding files, reconstructs the
capability/download closure, and verifies that gaps or pending records were
not reported as complete.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


REPORT_FORMAT = "CANCERLNCATLAS_V32_INTEGRATED_COMPLETENESS_AUDIT_V1"
BINDING_FORMAT = "CANCERLNCATLAS_V32_INTEGRATED_COMPLETENESS_BINDING_V1"
INDEPENDENT_REPORT_FORMAT = (
    "CANCERLNCATLAS_V32_INTEGRATED_COMPLETENESS_INDEPENDENT_AUDIT_V1"
)
INDEPENDENT_BINDING_FORMAT = (
    "CANCERLNCATLAS_V32_INTEGRATED_COMPLETENESS_INDEPENDENT_BINDING_V1"
)
MODEL_VERSION = "V3.2"
GATE_KINDS = ("artifact", "api", "ui", "download")
DOWNLOAD_PASS_STATUSES = frozenset(
    {"READY_FILE", "READY_PARTS", "DYNAMIC_QUERY_EXPORT"}
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

CAPABILITY_BINDING_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "exact_pathway": (),
    "gene_set": ("gene_set_ranked_subtype", "gene_set_ranked_subtype_independent_audit"),
    "ranked_subtype": ("gene_set_ranked_subtype", "gene_set_ranked_subtype_independent_audit"),
    "network": ("unified_network", "unified_network_independent_audit"),
    "state_rnass": ("state_gene_sets",),
    "state_dnass": ("state_gene_sets",),
    "state_extend": ("state_gene_sets",),
    "state_ereg_expss": ("state_gene_sets",),
    "state_dmpss": ("state_gene_sets",),
    "state_enhss": ("state_gene_sets",),
    "state_ereg_methss": ("state_gene_sets",),
    "clinical": ("historical_artifact_remediation", "historical_artifact_remediation_independent_audit"),
    "expression_landscape": ("bulk_expression",),
    "survival_kaplan_meier": ("clinical_kaplan_meier",),
    "bulk_coexpression": ("bulk_coexpression",),
    "external_validation": ("external_validation",),
    "continuous_pathway_activity": ("continuous_pathway_activity",),
    "single_cell": (
        "single_cell_exact_pathway",
        "single_cell_expression",
        "single_cell_hnsc_ucell",
        "single_cell_gap_audit",
        "single_cell_gap_independent_audit",
    ),
    "mutation": ("historical_artifact_remediation", "historical_artifact_remediation_independent_audit"),
    "cnv": ("historical_artifact_remediation", "historical_artifact_remediation_independent_audit"),
    "drug": ("drug_response_actionability",),
    "physical_interaction": ("physical_interaction",),
    "experiment_perturbation": (
        "experiment_raw_facts",
        "experiment_evidence_bridge",
        "experiment_evidence_bridge_audit",
    ),
    "evidence_transformer": (
        "evidence_transformer",
        "evidence_direction_probabilities",
        "evidence_direction_probabilities_independent_audit",
    ),
    "mixed_lncrna_protein_pathway_query": (
        "historical_artifact_remediation",
        "historical_artifact_remediation_independent_audit",
    ),
}

CAPABILITY_REGISTRY_MODULES: dict[str, str] = {
    "exact_pathway": "exact_pathway",
    "state_rnass": "state",
    "state_dnass": "state",
    "state_extend": "state",
    "state_ereg_expss": "state",
    "state_dmpss": "state",
    "state_enhss": "state",
    "state_ereg_methss": "state",
    "clinical": "clinical",
    "mutation": "mutation_cnv",
    "cnv": "mutation_cnv",
}


class IndependentCompletenessAuditError(RuntimeError):
    """Raised when the completeness binding or its evidence is inconsistent."""


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
    requested = Path(path)
    if requested.is_symlink():
        raise IndependentCompletenessAuditError(f"{label} may not be a symlink")
    resolved = requested.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise IndependentCompletenessAuditError(f"{label} escapes repository") from exc
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise IndependentCompletenessAuditError(f"{label} is missing or empty: {resolved}")
    return resolved


def _read_json(path: str | Path, root: Path, label: str) -> tuple[Path, dict[str, Any]]:
    source = _safe_file(path, root, label)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IndependentCompletenessAuditError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise IndependentCompletenessAuditError(f"{label} must be a JSON object")
    return source, value


def _path_from_record(record: Mapping[str, Any], root: Path, label: str) -> Path:
    raw = Path(str(record.get("path", "")))
    path = _safe_file(raw if raw.is_absolute() else root / raw, root, label)
    expected = str(record.get("sha256", "")).lower()
    if not _SHA256_RE.fullmatch(expected) or sha256_file(path) != expected:
        raise IndependentCompletenessAuditError(f"{label} SHA256 drift")
    if record.get("bytes") != path.stat().st_size:
        raise IndependentCompletenessAuditError(f"{label} byte-count drift")
    return path


def _declared_path(declaration: Mapping[str, Any], root: Path, label: str) -> Path:
    raw = Path(str(declaration.get("path", "")))
    path = _safe_file(raw if raw.is_absolute() else root / raw, root, label)
    expected = str(declaration.get("sha256", "")).lower()
    if not _SHA256_RE.fullmatch(expected) or sha256_file(path) != expected:
        raise IndependentCompletenessAuditError(f"{label} declaration SHA256 drift")
    return path


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise IndependentCompletenessAuditError("PyYAML is required") from exc
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise IndependentCompletenessAuditError("Parity contract must be an object")
    return value


def _add_check(
    checks: list[dict[str, Any]], name: str, observed: Any, expected: Any
) -> None:
    checks.append(
        {
            "name": name,
            "observed": observed,
            "expected": expected,
            "status": "PASS" if observed == expected else "FAIL",
        }
    )


def independent_audit_integrated_completeness(
    *,
    repo_root: str | Path,
    binding_path: str | Path,
    expected_binding_sha256: str,
    output_root: str | Path,
    auditor_code_path: str | Path | None = None,
    runner_code_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    release_binding_path, binding = _read_json(
        binding_path, root, "integrated completeness binding"
    )
    expected_binding_sha = str(expected_binding_sha256).lower()
    if not _SHA256_RE.fullmatch(expected_binding_sha):
        raise IndependentCompletenessAuditError("Expected binding SHA256 is invalid")
    if sha256_file(release_binding_path) != expected_binding_sha:
        raise IndependentCompletenessAuditError("Integrated binding SHA256 drift")

    output = Path(output_root).resolve()
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise IndependentCompletenessAuditError("Audit output must remain in repository") from exc
    if output.exists() and any(output.iterdir()):
        raise IndependentCompletenessAuditError(f"Refusing non-empty output reuse: {output}")
    output.mkdir(parents=True, exist_ok=True)

    report_record = binding.get("report")
    inputs = binding.get("inputs")
    if not isinstance(report_record, Mapping) or not isinstance(inputs, Mapping):
        raise IndependentCompletenessAuditError("Binding lacks report or inputs")
    report_path = _path_from_record(report_record, root, "completeness report")
    _, report = _read_json(report_path, root, "completeness report")
    resolved_inputs: dict[str, Path] = {}
    for input_id, record in sorted(inputs.items()):
        if not isinstance(record, Mapping):
            raise IndependentCompletenessAuditError(f"Input {input_id} is not a record")
        resolved_inputs[str(input_id)] = _path_from_record(
            record, root, f"input {input_id}"
        )
    required_inputs = {
        "parity_contract",
        "unified_staging_bindings",
        "web_catalog",
        "release_registry",
        "download_catalog_binding",
        "download_catalog",
        "download_catalog_independent_audit_binding",
        "evaluator_code",
        "runner_code",
    }
    if not required_inputs.issubset(resolved_inputs):
        raise IndependentCompletenessAuditError("Binding omits a required reproducibility input")
    if report.get("input_sha256") != inputs:
        raise IndependentCompletenessAuditError("Report/binding input records differ")

    contract = _load_yaml(resolved_inputs["parity_contract"])
    _, unified = _read_json(
        resolved_inputs["unified_staging_bindings"], root, "unified bindings"
    )
    _, web = _read_json(resolved_inputs["web_catalog"], root, "web catalog")
    registry_path, registry = _read_json(
        resolved_inputs["release_registry"], root, "release registry"
    )
    _, download_catalog = _read_json(
        resolved_inputs["download_catalog"], root, "download catalog"
    )
    contract_caps = contract.get("capabilities")
    if not isinstance(contract_caps, Mapping) or len(contract_caps) != 25:
        raise IndependentCompletenessAuditError("Contract is not a 25-capability closure")
    capability_ids = sorted(str(value) for value in contract_caps)

    web_values = web.get("capabilities")
    if not isinstance(web_values, Sequence) or isinstance(web_values, (str, bytes)):
        raise IndependentCompletenessAuditError("Web capability rows are invalid")
    web_rows = {
        str(row.get("capability_id")): row
        for row in web_values
        if isinstance(row, Mapping)
    }
    if sorted(web_rows) != capability_ids or len(web_rows) != len(web_values):
        raise IndependentCompletenessAuditError("Web capability closure drift")

    download_values = download_catalog.get("downloads")
    if not isinstance(download_values, Sequence) or isinstance(
        download_values, (str, bytes)
    ):
        raise IndependentCompletenessAuditError("Download rows are invalid")
    download_rows = {
        str(row.get("download_id")): row
        for row in download_values
        if isinstance(row, Mapping)
    }
    if len(download_rows) != len(download_values):
        raise IndependentCompletenessAuditError("Download rows are duplicated")

    capability_values = report.get("capabilities")
    if not isinstance(capability_values, Sequence) or isinstance(
        capability_values, (str, bytes)
    ):
        raise IndependentCompletenessAuditError("Report capability rows are invalid")
    report_rows = {
        str(row.get("capability_id")): row
        for row in capability_values
        if isinstance(row, Mapping)
    }
    if sorted(report_rows) != capability_ids or len(report_rows) != len(capability_values):
        raise IndependentCompletenessAuditError("Report capability closure drift")

    unified_bindings = unified.get("bindings")
    if not isinstance(unified_bindings, Mapping):
        raise IndependentCompletenessAuditError("Unified binding map is invalid")
    checks: list[dict[str, Any]] = []
    _add_check(checks, "release_binding_format", binding.get("format"), BINDING_FORMAT)
    _add_check(checks, "release_binding_model", binding.get("model_version"), MODEL_VERSION)
    _add_check(checks, "release_binding_fail_closed", binding.get("fail_closed"), True)
    _add_check(checks, "report_format", report.get("format"), REPORT_FORMAT)
    _add_check(checks, "report_model", report.get("model_version"), MODEL_VERSION)
    _add_check(checks, "report_fail_closed", report.get("fail_closed"), True)
    _add_check(checks, "report_typed_gaps_not_complete", report.get("typed_gaps_count_as_complete"), False)
    _add_check(checks, "report_pending_not_complete", report.get("pending_training_counts_as_complete"), False)
    _add_check(checks, "capability_count", report.get("required_capability_count"), 25)

    complete_ids: list[str] = []
    blocking_ids: list[str] = []
    for capability_id in capability_ids:
        row = report_rows[capability_id]
        web_row = web_rows[capability_id]
        gates = row.get("gates")
        if not isinstance(gates, Mapping) or set(gates) != set(GATE_KINDS):
            raise IndependentCompletenessAuditError(
                f"{capability_id} gate closure is invalid"
            )
        gate_statuses = {
            kind: str(gates[kind].get("status", ""))
            if isinstance(gates[kind], Mapping)
            else "INVALID"
            for kind in GATE_KINDS
        }
        if any(value not in {"PASS", "FAIL"} for value in gate_statuses.values()):
            raise IndependentCompletenessAuditError(
                f"{capability_id} has an invalid gate status"
            )
        all_complete = all(value == "PASS" for value in gate_statuses.values())
        _add_check(
            checks,
            f"{capability_id}.all_four_recomputed",
            row.get("all_four_gates_complete"),
            all_complete,
        )
        _add_check(
            checks,
            f"{capability_id}.status_recomputed",
            row.get("status"),
            "COMPLETE" if all_complete else "PARTIAL",
        )

        known_gap = web_row.get("known_gap")
        web_queryable = web_row.get("staging_status") == "QUERYABLE_STAGING"
        if known_gap or not web_queryable:
            _add_check(
                checks,
                f"{capability_id}.web_gap_cannot_complete",
                all_complete,
                False,
            )
            _add_check(
                checks,
                f"{capability_id}.gap_blocks_artifact_api_ui",
                [gate_statuses[kind] for kind in ("artifact", "api", "ui")],
                ["FAIL", "FAIL", "FAIL"],
            )

        required_api = [
            str(value) for value in contract_caps[capability_id]["gates"]["api"]["ids"]
        ]
        declared_api = [str(value) for value in web_row.get("required_api_contract", [])]
        staging_endpoints = [str(value) for value in web_row.get("staging_endpoints", [])]
        api_expected = (
            sorted(required_api) == sorted(declared_api)
            and bool(staging_endpoints)
            and web_queryable
            and not bool(known_gap)
        )
        _add_check(
            checks,
            f"{capability_id}.api_recomputed",
            gate_statuses["api"],
            "PASS" if api_expected else "FAIL",
        )
        required_ui = [
            str(value) for value in contract_caps[capability_id]["gates"]["ui"]["ids"]
        ]
        declared_ui = [str(value) for value in web_row.get("ui_surface_ids", [])]
        ui_expected = (
            sorted(required_ui) == sorted(declared_ui)
            and web_queryable
            and not bool(known_gap)
        )
        _add_check(
            checks,
            f"{capability_id}.ui_recomputed",
            gate_statuses["ui"],
            "PASS" if ui_expected else "FAIL",
        )

        artifact_evidence = gates["artifact"].get("evidence", {})
        authority_keys = artifact_evidence.get("authority_binding_keys", [])
        if not isinstance(authority_keys, Sequence) or isinstance(
            authority_keys, (str, bytes)
        ):
            raise IndependentCompletenessAuditError(
                f"{capability_id} authority list is invalid"
            )
        expected_authority_keys = list(CAPABILITY_BINDING_REQUIREMENTS[capability_id])
        _add_check(
            checks,
            f"{capability_id}.authority_map",
            list(authority_keys),
            expected_authority_keys,
        )
        authority_ok = True
        for key_value in expected_authority_keys:
            key = str(key_value)
            declaration = unified_bindings.get(key)
            if not isinstance(declaration, Mapping) or declaration.get("status") != "MOUNTED_HASH_PINNED":
                authority_ok = False
                continue
            input_key = f"unified_binding:{key}"
            if input_key not in resolved_inputs:
                authority_ok = False
                continue
            try:
                declared = _declared_path(declaration, root, f"binding {key}")
            except IndependentCompletenessAuditError:
                authority_ok = False
                continue
            if declared != resolved_inputs[input_key]:
                authority_ok = False

        registry_check = artifact_evidence.get("registry_check")
        registry_ok = True
        expected_registry_module = CAPABILITY_REGISTRY_MODULES.get(capability_id)
        if capability_id == "mixed_lncrna_protein_pathway_query":
            mixed = registry.get("capabilities", {}).get("mixed_exact_pathway_query", {})
            registry_ok = isinstance(mixed, Mapping) and mixed.get("enabled") is True
            expected_registry_label = "registry.capabilities.mixed_exact_pathway_query"
        elif expected_registry_module is not None:
            expected_registry_label = expected_registry_module
            module = registry.get("modules", {}).get(expected_registry_module, {})
            registry_ok = isinstance(module, Mapping)
            if registry_ok:
                registry_ok = (
                    module.get("status") == "SUCCESS_NEWLY_TRAINED"
                    and module.get("new_training_attestation") is True
                    and module.get("old_checkpoint_loaded") is False
                    and module.get("old_predictions_used_as_features") is False
                    and module.get("old_rankings_used_as_outputs") is False
                )
            if registry_ok:
                lineage = registry_path.parent / str(module.get("lineage_path", ""))
                try:
                    lineage = _safe_file(
                        lineage, root, f"registry module {expected_registry_module} lineage"
                    )
                    registry_ok = sha256_file(lineage) == str(
                        module.get("lineage_sha256", "")
                    ).lower()
                except IndependentCompletenessAuditError:
                    registry_ok = False
        else:
            expected_registry_label = None
            registry_ok = not isinstance(registry_check, Mapping)
        if isinstance(registry_check, Mapping):
            observed_registry_label = registry_check.get("module_id")
            _add_check(
                checks,
                f"{capability_id}.registry_label",
                observed_registry_label,
                expected_registry_label,
            )
            _add_check(
                checks,
                f"{capability_id}.registry_recomputed",
                registry_check.get("status"),
                "PASS" if registry_ok else "FAIL",
            )
        elif expected_registry_label is not None:
            registry_ok = False
        artifact_expected = authority_ok and registry_ok and not bool(known_gap)
        _add_check(
            checks,
            f"{capability_id}.artifact_recomputed",
            gate_statuses["artifact"],
            "PASS" if artifact_expected else "FAIL",
        )

        required_downloads = [
            str(value)
            for value in contract_caps[capability_id]["gates"]["download"]["ids"]
        ]
        download_expected = True
        for download_id in required_downloads:
            download = download_rows.get(download_id)
            if not isinstance(download, Mapping):
                download_expected = False
                continue
            status = download.get("status")
            row_ok = (
                status in DOWNLOAD_PASS_STATUSES
                and download.get("contract_complete") is True
                and download.get("download_implemented") is True
            )
            if status in {"READY_FILE", "READY_PARTS"}:
                row_ok = row_ok and download.get("data_present") is True
            if status == "DYNAMIC_QUERY_EXPORT":
                row_ok = (
                    row_ok
                    and bool(str(download.get("export_endpoint", "")).strip())
                    and bool(download.get("export_formats"))
                )
            download_expected = download_expected and row_ok
        _add_check(
            checks,
            f"{capability_id}.download_recomputed",
            gate_statuses["download"],
            "PASS" if download_expected else "FAIL",
        )

        if all_complete:
            complete_ids.append(capability_id)
        else:
            blocking_ids.append(capability_id)

    expected_status = "COMPLETE" if not blocking_ids else "PARTIAL"
    _add_check(checks, "overall_status_recomputed", report.get("status"), expected_status)
    _add_check(checks, "binding_status_matches_report", binding.get("status"), expected_status)
    _add_check(
        checks,
        "overall_boolean_recomputed",
        report.get("all_25_capabilities_four_gate_complete"),
        expected_status == "COMPLETE",
    )
    _add_check(checks, "complete_ids_recomputed", report.get("complete_capability_ids"), complete_ids)
    _add_check(checks, "blocking_ids_recomputed", report.get("blocking_capability_ids"), blocking_ids)
    _add_check(checks, "complete_count_recomputed", report.get("complete_capability_count"), len(complete_ids))
    _add_check(checks, "partial_count_recomputed", report.get("partial_capability_count"), len(blocking_ids))

    fail_count = sum(check["status"] != "PASS" for check in checks)
    independent_report = {
        "format": INDEPENDENT_REPORT_FORMAT,
        "model_version": MODEL_VERSION,
        "environment": "staging",
        "status": "PASS" if fail_count == 0 else "FAIL",
        "audited_completeness_status": expected_status,
        "accepted_as_complete": fail_count == 0 and expected_status == "COMPLETE",
        "accepted_as_truthful_staging_audit": fail_count == 0,
        "independent_of_materializer_implementation": True,
        "materializer_imported": False,
        "pass_count": len(checks) - fail_count,
        "fail_count": fail_count,
        "blocking_capability_ids": blocking_ids,
        "checks": checks,
    }
    independent_report_path = output / "INDEPENDENT_AUDIT_REPORT.json"
    _atomic_json(independent_report_path, independent_report)

    auditor = _safe_file(
        auditor_code_path or Path(__file__), root, "independent auditor code"
    )
    runner = _safe_file(
        runner_code_path
        or root / "scripts/audit_v32_integrated_completeness_independent.py",
        root,
        "independent auditor runner",
    )
    independent_binding = {
        "format": INDEPENDENT_BINDING_FORMAT,
        "model_version": MODEL_VERSION,
        "environment": "staging",
        "status": independent_report["status"],
        "audited_completeness_status": expected_status,
        "accepted_as_complete": independent_report["accepted_as_complete"],
        "accepted_as_truthful_staging_audit": independent_report[
            "accepted_as_truthful_staging_audit"
        ],
        "production_deployed": False,
        "independent_of_materializer_implementation": True,
        "materializer_imported": False,
        "pass_count": independent_report["pass_count"],
        "fail_count": fail_count,
        "release_binding": {
            "path": release_binding_path.relative_to(root).as_posix(),
            "sha256": sha256_file(release_binding_path),
            "bytes": release_binding_path.stat().st_size,
        },
        "release_report": {
            "path": report_path.relative_to(root).as_posix(),
            "sha256": sha256_file(report_path),
            "bytes": report_path.stat().st_size,
        },
        "report": {
            "path": independent_report_path.relative_to(root).as_posix(),
            "sha256": sha256_file(independent_report_path),
            "bytes": independent_report_path.stat().st_size,
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
    independent_binding_path = output / "INDEPENDENT_AUDIT_BINDING.json"
    _atomic_json(independent_binding_path, independent_binding)
    if fail_count:
        raise IndependentCompletenessAuditError(
            f"Independent completeness audit failed {fail_count} checks"
        )
    return {
        "binding_path": str(independent_binding_path),
        "binding_sha256": sha256_file(independent_binding_path),
        "report_path": str(independent_report_path),
        "report_sha256": sha256_file(independent_report_path),
        "status": independent_report["status"],
        "audited_completeness_status": expected_status,
        "accepted_as_complete": independent_report["accepted_as_complete"],
        "accepted_as_truthful_staging_audit": True,
        "pass_count": independent_report["pass_count"],
        "fail_count": 0,
    }
