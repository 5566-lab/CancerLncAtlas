"""Fail-closed, cross-surface completeness audit for V3.2 staging.

This audit intentionally answers a narrower question than a smoke test: does
every capability in the historical parity contract have all four publication
gates (artifact, API, UI, and download) at the same time?  Typed gaps and
pending training are valid staging states, but they can never satisfy a gate.

The evaluator is data driven.  In particular, ``drug_response_actionability``
may move from ``PENDING_FORMAL_SUCCESS`` to ``MOUNTED_HASH_PINNED`` and the
same evaluator can be rerun without a code change.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .capability_parity import EXPECTED_CAPABILITY_IDS, load_parity_config


REPORT_FORMAT = "CANCERLNCATLAS_V32_INTEGRATED_COMPLETENESS_AUDIT_V1"
BINDING_FORMAT = "CANCERLNCATLAS_V32_INTEGRATED_COMPLETENESS_BINDING_V1"
UNIFIED_FORMAT = "CANCERLNCATLAS_V32_UNIFIED_STAGING_BINDINGS_V1"
WEB_FORMAT = "CANCERLNCATLAS_V32_STAGING_WEB_CATALOG_V1"
DOWNLOAD_BINDING_FORMAT = "CANCERLNCATLAS_V32_STAGING_DOWNLOAD_CATALOG_BINDING_V1"
DOWNLOAD_FORMAT = "CANCERLNCATLAS_V32_STAGING_DOWNLOAD_CATALOG_V1"
MODEL_VERSION = "V3.2"
GATE_KINDS = ("artifact", "api", "ui", "download")
DOWNLOAD_PASS_STATUSES = frozenset(
    {"READY_FILE", "READY_PARTS", "DYNAMIC_QUERY_EXPORT"}
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class IntegratedCompletenessError(RuntimeError):
    """Raised when an input cannot be trusted enough to issue an audit."""


# A binding is an authority envelope, not a substitute for the contract's
# artifact IDs.  The report retains both lists so reviewers can see exactly
# which authority was used for every required artifact set.
CAPABILITY_BINDING_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "exact_pathway": (),
    "gene_set": (
        "gene_set_ranked_subtype",
        "gene_set_ranked_subtype_independent_audit",
    ),
    "ranked_subtype": (
        "gene_set_ranked_subtype",
        "gene_set_ranked_subtype_independent_audit",
    ),
    "network": ("unified_network", "unified_network_independent_audit"),
    "state_rnass": ("state_gene_sets",),
    "state_dnass": ("state_gene_sets",),
    "state_extend": ("state_gene_sets",),
    "state_ereg_expss": ("state_gene_sets",),
    "state_dmpss": ("state_gene_sets",),
    "state_enhss": ("state_gene_sets",),
    "state_ereg_methss": ("state_gene_sets",),
    "clinical": (
        "historical_artifact_remediation",
        "historical_artifact_remediation_independent_audit",
    ),
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
    "mutation": (
        "historical_artifact_remediation",
        "historical_artifact_remediation_independent_audit",
    ),
    "cnv": (
        "historical_artifact_remediation",
        "historical_artifact_remediation_independent_audit",
    ),
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

AUDIT_BINDING_KEYS = frozenset(
    {
        "gene_set_ranked_subtype_independent_audit",
        "historical_artifact_remediation_independent_audit",
        "unified_network_independent_audit",
        "single_cell_gap_independent_audit",
        "experiment_evidence_bridge_audit",
        "evidence_direction_probabilities_independent_audit",
        "download_catalog_independent_audit",
    }
)


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
        raise IntegratedCompletenessError(f"{label} may not be a symlink: {requested}")
    resolved = requested.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise IntegratedCompletenessError(f"{label} escapes repository root: {resolved}") from exc
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise IntegratedCompletenessError(f"{label} is missing or empty: {resolved}")
    return resolved


def _resolve_declared_path(value: Any, root: Path, label: str) -> Path:
    raw = Path(str(value or ""))
    return _safe_file(raw if raw.is_absolute() else root / raw, root, label)


def _read_json(path: str | Path, root: Path, label: str) -> tuple[Path, dict[str, Any]]:
    source = _safe_file(path, root, label)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegratedCompletenessError(f"{label} is invalid JSON: {source}") from exc
    if not isinstance(value, dict):
        raise IntegratedCompletenessError(f"{label} must be a JSON object")
    return source, value


def _source_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _require_sha(value: Any, label: str) -> str:
    normalized = str(value or "").lower()
    if not _SHA256_RE.fullmatch(normalized):
        raise IntegratedCompletenessError(f"{label} is not a lowercase SHA256")
    return normalized


def _verify_declaration(
    declaration: Mapping[str, Any], root: Path, label: str
) -> tuple[Path, dict[str, Any] | None]:
    source = _resolve_declared_path(declaration.get("path"), root, label)
    expected = _require_sha(declaration.get("sha256"), f"{label}.sha256")
    observed = sha256_file(source)
    if observed != expected:
        raise IntegratedCompletenessError(
            f"{label} SHA256 drift: observed {observed}, expected {expected}"
        )
    if "bytes" in declaration and declaration.get("bytes") != source.stat().st_size:
        raise IntegratedCompletenessError(f"{label} byte-count drift")
    payload: dict[str, Any] | None = None
    if source.suffix.lower() == ".json":
        _, payload = _read_json(source, root, label)
    return source, payload


def _audit_payload_passes(payload: Mapping[str, Any]) -> bool:
    status = str(payload.get("status", "")).upper()
    if not status.startswith("PASS"):
        return False
    for key in ("fail_count", "failed_checks"):
        if key in payload and payload.get(key) != 0:
            return False
    return True


def _validate_root_documents(
    contract: Mapping[str, Any],
    unified: Mapping[str, Any],
    web: Mapping[str, Any],
) -> None:
    if set(contract["capabilities"]) != set(EXPECTED_CAPABILITY_IDS):
        raise IntegratedCompletenessError("Parity contract is not the exact 25-capability closure")
    if set(CAPABILITY_BINDING_REQUIREMENTS) != set(EXPECTED_CAPABILITY_IDS):
        raise IntegratedCompletenessError("Internal capability authority map is incomplete")
    if unified.get("schema_version") != UNIFIED_FORMAT:
        raise IntegratedCompletenessError("Unified staging binding schema mismatch")
    if unified.get("environment") != "staging" or unified.get("production_deployed") is not False:
        raise IntegratedCompletenessError("Unified bindings must describe non-production staging")
    if web.get("schema_version") != WEB_FORMAT:
        raise IntegratedCompletenessError("Web catalog schema mismatch")
    if web.get("model_version") != MODEL_VERSION or web.get("environment") != "staging":
        raise IntegratedCompletenessError("Web catalog must describe V3.2 staging")
    if web.get("production_deployed") is not False:
        raise IntegratedCompletenessError("Web catalog unexpectedly claims production deployment")


def _validate_registry(
    unified: Mapping[str, Any], root: Path
) -> tuple[Path, dict[str, Any], dict[str, dict[str, Any]]]:
    declaration = unified.get("registry")
    if not isinstance(declaration, Mapping):
        raise IntegratedCompletenessError("Unified bindings lack the release registry")
    registry_path, registry = _verify_declaration(declaration, root, "release registry")
    if registry is None:
        raise IntegratedCompletenessError("Release registry must be JSON")
    modules = registry.get("modules")
    if not isinstance(modules, Mapping):
        raise IntegratedCompletenessError("Release registry lacks modules")
    checks: dict[str, dict[str, Any]] = {}
    for capability_id, module_id in sorted(CAPABILITY_REGISTRY_MODULES.items()):
        module = modules.get(module_id)
        reasons: list[str] = []
        if not isinstance(module, Mapping):
            reasons.append(f"registry module {module_id} is missing")
        else:
            if module.get("status") != "SUCCESS_NEWLY_TRAINED":
                reasons.append(f"registry module {module_id} is {module.get('status')}")
            if module.get("new_training_attestation") is not True:
                reasons.append(f"registry module {module_id} lacks new-training attestation")
            for flag in (
                "old_checkpoint_loaded",
                "old_predictions_used_as_features",
                "old_rankings_used_as_outputs",
            ):
                if module.get(flag) is not False:
                    reasons.append(f"registry module {module_id}.{flag} is not false")
            lineage_raw = module.get("lineage_path")
            lineage_sha = module.get("lineage_sha256")
            try:
                lineage_path = _safe_file(
                    registry_path.parent / str(lineage_raw or ""),
                    root,
                    f"registry module {module_id} lineage",
                )
                if sha256_file(lineage_path) != _require_sha(
                    lineage_sha, f"registry module {module_id}.lineage_sha256"
                ):
                    raise IntegratedCompletenessError(
                        f"registry module {module_id} lineage SHA256 drift"
                    )
            except IntegratedCompletenessError as exc:
                reasons.append(str(exc))
        checks[capability_id] = {
            "module_id": module_id,
            "status": "PASS" if not reasons else "FAIL",
            "reasons": reasons,
        }
    mixed = registry.get("capabilities", {}).get("mixed_exact_pathway_query", {})
    if not isinstance(mixed, Mapping) or mixed.get("enabled") is not True:
        checks["mixed_lncrna_protein_pathway_query"] = {
            "module_id": "registry.capabilities.mixed_exact_pathway_query",
            "status": "FAIL",
            "reasons": ["mixed exact-pathway registry capability is not enabled"],
        }
    else:
        checks["mixed_lncrna_protein_pathway_query"] = {
            "module_id": "registry.capabilities.mixed_exact_pathway_query",
            "status": "PASS",
            "reasons": [],
        }
    return registry_path, registry, checks


def _validate_unified_bindings(
    unified: Mapping[str, Any], root: Path
) -> tuple[dict[str, dict[str, Any]], dict[str, Path]]:
    bindings = unified.get("bindings")
    if not isinstance(bindings, Mapping):
        raise IntegratedCompletenessError("Unified staging document lacks bindings")
    results: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}
    for key, declaration_value in sorted(bindings.items()):
        if not isinstance(declaration_value, Mapping):
            raise IntegratedCompletenessError(f"Binding {key} must be an object")
        declaration = dict(declaration_value)
        status = str(declaration.get("status", ""))
        reasons: list[str] = []
        if status == "MOUNTED_HASH_PINNED":
            source, payload = _verify_declaration(declaration, root, f"binding {key}")
            paths[str(key)] = source
            if key in AUDIT_BINDING_KEYS and (
                payload is None or not _audit_payload_passes(payload)
            ):
                reasons.append(f"independent audit binding {key} does not report PASS")
        else:
            reasons.append(f"binding status is {status or 'MISSING'}")
        results[str(key)] = {
            "declared_status": status or None,
            "status": "PASS" if not reasons else "FAIL",
            "reasons": reasons,
            "path": (
                paths[str(key)].relative_to(root).as_posix()
                if str(key) in paths
                else None
            ),
            "sha256": declaration.get("sha256"),
        }
    return results, paths


def _web_rows(
    web: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    rows_value = web.get("capabilities")
    if not isinstance(rows_value, Sequence) or isinstance(rows_value, (str, bytes)):
        raise IntegratedCompletenessError("Web catalog capabilities must be an array")
    rows = {
        str(row.get("capability_id")): dict(row)
        for row in rows_value
        if isinstance(row, Mapping)
    }
    if len(rows) != len(rows_value):
        raise IntegratedCompletenessError("Web catalog has duplicate or invalid capability rows")
    if set(rows) != set(contract["capabilities"]):
        raise IntegratedCompletenessError("Web catalog capability closure differs from contract")
    if web.get("capability_count") != len(rows):
        raise IntegratedCompletenessError("Web catalog capability_count drift")
    return rows


def _load_download_catalog(
    unified: Mapping[str, Any],
    root: Path,
    binding_paths: Mapping[str, Path],
) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    binding_path = binding_paths.get("download_catalog")
    if binding_path is None:
        raise IntegratedCompletenessError("Download catalog binding is not hash-mounted")
    _, binding = _read_json(binding_path, root, "download catalog binding")
    if binding.get("format") != DOWNLOAD_BINDING_FORMAT:
        raise IntegratedCompletenessError("Download catalog binding format mismatch")
    if binding.get("model_version") != MODEL_VERSION:
        raise IntegratedCompletenessError("Download catalog binding model version mismatch")
    declaration = binding.get("catalog")
    if not isinstance(declaration, Mapping):
        raise IntegratedCompletenessError("Download catalog binding lacks catalog declaration")
    catalog_path, catalog = _verify_declaration(
        declaration, root, "download catalog payload"
    )
    if catalog is None:
        raise IntegratedCompletenessError("Download catalog payload must be JSON")
    if catalog.get("format") != DOWNLOAD_FORMAT:
        raise IntegratedCompletenessError("Download catalog payload format mismatch")
    if catalog.get("model_version") != MODEL_VERSION or catalog.get("environment") != "staging":
        raise IntegratedCompletenessError("Download catalog must describe V3.2 staging")
    if catalog.get("production_deployed") is not False:
        raise IntegratedCompletenessError("Download catalog unexpectedly claims production")

    audit_path = binding_paths.get("download_catalog_independent_audit")
    if audit_path is None:
        raise IntegratedCompletenessError("Download catalog independent audit is not mounted")
    _, audit = _read_json(audit_path, root, "download catalog independent audit")
    if not _audit_payload_passes(audit):
        raise IntegratedCompletenessError("Download catalog independent audit is not PASS")
    release_declaration = audit.get("release_binding")
    catalog_declaration = audit.get("catalog")
    if not isinstance(release_declaration, Mapping) or not isinstance(
        catalog_declaration, Mapping
    ):
        raise IntegratedCompletenessError("Download independent audit lacks hash-bound inputs")
    if release_declaration.get("sha256") != sha256_file(binding_path):
        raise IntegratedCompletenessError("Download independent audit release-binding drift")
    if catalog_declaration.get("sha256") != sha256_file(catalog_path):
        raise IntegratedCompletenessError("Download independent audit catalog drift")
    return binding_path, binding, catalog_path, catalog


def _download_rows(
    catalog: Mapping[str, Any], contract: Mapping[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    expected: dict[str, list[str]] = {}
    for capability_id, capability in contract["capabilities"].items():
        for download_id in capability["gates"]["download"]["ids"]:
            expected.setdefault(str(download_id), []).append(str(capability_id))
    expected = {key: sorted(value) for key, value in expected.items()}
    rows_value = catalog.get("downloads")
    if not isinstance(rows_value, Sequence) or isinstance(rows_value, (str, bytes)):
        raise IntegratedCompletenessError("Download catalog downloads must be an array")
    rows = {
        str(row.get("download_id")): dict(row)
        for row in rows_value
        if isinstance(row, Mapping)
    }
    if len(rows) != len(rows_value):
        raise IntegratedCompletenessError("Download catalog has duplicate or invalid rows")
    if set(rows) != set(expected):
        raise IntegratedCompletenessError("Download ID closure differs from parity contract")
    if catalog.get("download_count") != len(rows):
        raise IntegratedCompletenessError("Download catalog download_count drift")
    for download_id, row in rows.items():
        observed_capabilities = sorted(str(value) for value in row.get("capability_ids", []))
        if observed_capabilities != expected[download_id]:
            raise IntegratedCompletenessError(
                f"Download {download_id} capability membership drift"
            )
    return rows, expected


def _gate(status: bool, reasons: list[str], evidence: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": "PASS" if status and not reasons else "FAIL",
        "reasons": reasons,
        "evidence": dict(evidence),
    }


def evaluate_integrated_completeness(
    *,
    repo_root: str | Path,
    parity_contract_path: str | Path,
    unified_bindings_path: str | Path,
    web_catalog_path: str | Path,
    evaluator_code_path: str | Path | None = None,
    runner_code_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate all 25 capabilities and return a report plus its source records."""

    root = Path(repo_root).resolve()
    contract_path = _safe_file(parity_contract_path, root, "parity contract")
    unified_path, unified = _read_json(unified_bindings_path, root, "unified bindings")
    web_path, web = _read_json(web_catalog_path, root, "web catalog")
    contract = load_parity_config(contract_path)
    _validate_root_documents(contract, unified, web)
    registry_path, _, registry_checks = _validate_registry(unified, root)
    binding_checks, binding_paths = _validate_unified_bindings(unified, root)
    download_binding_path, _, download_catalog_path, download_catalog = (
        _load_download_catalog(unified, root, binding_paths)
    )
    web_by_id = _web_rows(web, contract)
    downloads, _ = _download_rows(download_catalog, contract)

    capability_rows: list[dict[str, Any]] = []
    for capability_id in sorted(contract["capabilities"]):
        required = contract["capabilities"][capability_id]
        web_row = web_by_id[capability_id]
        known_gap = web_row.get("known_gap")

        authority_keys = CAPABILITY_BINDING_REQUIREMENTS[capability_id]
        artifact_reasons: list[str] = []
        for key in authority_keys:
            record = binding_checks.get(key)
            if record is None:
                artifact_reasons.append(f"required authority binding {key} is absent")
            elif record["status"] != "PASS":
                artifact_reasons.extend(
                    f"{key}: {reason}" for reason in record.get("reasons", [])
                )
        registry_check = registry_checks.get(capability_id)
        if registry_check is not None and registry_check["status"] != "PASS":
            artifact_reasons.extend(registry_check["reasons"])
        if known_gap:
            artifact_reasons.append(f"catalogued typed/pending gap: {known_gap}")
        artifact_gate = _gate(
            not artifact_reasons,
            artifact_reasons,
            {
                "required_artifact_ids": required["gates"]["artifact"]["ids"],
                "authority_binding_keys": list(authority_keys),
                "registry_check": registry_check,
            },
        )

        required_api = [str(value) for value in required["gates"]["api"]["ids"]]
        declared_api = [str(value) for value in web_row.get("required_api_contract", [])]
        endpoints = [str(value) for value in web_row.get("staging_endpoints", [])]
        api_reasons: list[str] = []
        if sorted(declared_api) != sorted(required_api):
            api_reasons.append("required API contract IDs do not match parity contract")
        if not endpoints:
            api_reasons.append("no staging API endpoint is declared")
        if web_row.get("staging_status") != "QUERYABLE_STAGING":
            api_reasons.append(f"staging status is {web_row.get('staging_status')}")
        if known_gap:
            api_reasons.append("known gap prevents API parity")
        api_gate = _gate(
            not api_reasons,
            api_reasons,
            {"required_contract_ids": required_api, "staging_endpoints": endpoints},
        )

        required_ui = [str(value) for value in required["gates"]["ui"]["ids"]]
        declared_ui = [str(value) for value in web_row.get("ui_surface_ids", [])]
        ui_reasons: list[str] = []
        if sorted(declared_ui) != sorted(required_ui):
            ui_reasons.append("UI surface IDs do not match parity contract")
        if web_row.get("staging_status") != "QUERYABLE_STAGING":
            ui_reasons.append(f"staging status is {web_row.get('staging_status')}")
        if known_gap:
            ui_reasons.append("known gap prevents UI parity")
        ui_gate = _gate(
            not ui_reasons,
            ui_reasons,
            {"required_surface_ids": required_ui, "declared_surface_ids": declared_ui},
        )

        required_downloads = [
            str(value) for value in required["gates"]["download"]["ids"]
        ]
        download_reasons: list[str] = []
        download_evidence: list[dict[str, Any]] = []
        for download_id in required_downloads:
            record = downloads[download_id]
            status = str(record.get("status", ""))
            record_reasons: list[str] = []
            if status not in DOWNLOAD_PASS_STATUSES:
                record_reasons.append(f"non-deliverable status {status or 'MISSING'}")
            if record.get("contract_complete") is not True:
                record_reasons.append("contract_complete is not true")
            if record.get("download_implemented") is not True:
                record_reasons.append("download_implemented is not true")
            if status in {"READY_FILE", "READY_PARTS"} and record.get("data_present") is not True:
                record_reasons.append("ready download does not attest data_present")
            if status == "DYNAMIC_QUERY_EXPORT":
                if not str(record.get("export_endpoint", "")).strip():
                    record_reasons.append("dynamic export lacks endpoint")
                if not record.get("export_formats"):
                    record_reasons.append("dynamic export lacks formats")
            download_evidence.append(
                {"download_id": download_id, "status": status, "reasons": record_reasons}
            )
            download_reasons.extend(
                f"{download_id}: {reason}" for reason in record_reasons
            )
        download_gate = _gate(
            not download_reasons,
            download_reasons,
            {"required_download_ids": required_downloads, "records": download_evidence},
        )

        gates = {
            "artifact": artifact_gate,
            "api": api_gate,
            "ui": ui_gate,
            "download": download_gate,
        }
        all_complete = all(gates[kind]["status"] == "PASS" for kind in GATE_KINDS)
        if web_row.get("all_four_parity_gates_pass") is True and not all_complete:
            raise IntegratedCompletenessError(
                f"Web catalog falsely claims four-gate parity for {capability_id}"
            )
        capability_rows.append(
            {
                "capability_id": capability_id,
                "status": "COMPLETE" if all_complete else "PARTIAL",
                "all_four_gates_complete": all_complete,
                "target_level": required.get("target_level"),
                "staging_status": web_row.get("staging_status"),
                "known_gap": known_gap,
                "gates": gates,
            }
        )

    complete = [row["capability_id"] for row in capability_rows if row["all_four_gates_complete"]]
    blocking = [row["capability_id"] for row in capability_rows if not row["all_four_gates_complete"]]
    status = "COMPLETE" if not blocking else "PARTIAL"

    evaluator = _safe_file(
        evaluator_code_path or Path(__file__), root, "integrated evaluator code"
    )
    runner = _safe_file(
        runner_code_path
        or root / "scripts" / "audit_v32_integrated_completeness.py",
        root,
        "integrated evaluator runner",
    )
    input_paths: dict[str, Path] = {
        "parity_contract": contract_path,
        "unified_staging_bindings": unified_path,
        "web_catalog": web_path,
        "release_registry": registry_path,
        "download_catalog_binding": download_binding_path,
        "download_catalog": download_catalog_path,
        "download_catalog_independent_audit_binding": binding_paths[
            "download_catalog_independent_audit"
        ],
        "evaluator_code": evaluator,
        "runner_code": runner,
    }
    for binding_id, path in sorted(binding_paths.items()):
        input_paths[f"unified_binding:{binding_id}"] = path
    inputs = {key: _source_record(path, root) for key, path in sorted(input_paths.items())}
    report = {
        "format": REPORT_FORMAT,
        "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
        "model_version": MODEL_VERSION,
        "environment": "staging",
        "status": status,
        "all_25_capabilities_four_gate_complete": status == "COMPLETE",
        "release_ready": status == "COMPLETE",
        "production_deployed": False,
        "fail_closed": True,
        "typed_gaps_count_as_complete": False,
        "pending_training_counts_as_complete": False,
        "required_capability_count": len(capability_rows),
        "complete_capability_count": len(complete),
        "partial_capability_count": len(blocking),
        "complete_capability_ids": complete,
        "blocking_capability_ids": blocking,
        "gate_kinds": list(GATE_KINDS),
        "capabilities": capability_rows,
        "input_sha256": inputs,
    }
    return report, inputs


def materialize_integrated_completeness_audit(
    *,
    repo_root: str | Path,
    parity_contract_path: str | Path,
    unified_bindings_path: str | Path,
    web_catalog_path: str | Path,
    output_root: str | Path,
    evaluator_code_path: str | Path | None = None,
    runner_code_path: str | Path | None = None,
) -> dict[str, Any]:
    """Write a new immutable audit directory and a hash-bound binding."""

    root = Path(repo_root).resolve()
    output = Path(output_root).resolve()
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise IntegratedCompletenessError("Audit output must remain inside repository") from exc
    if output.exists() and any(output.iterdir()):
        raise IntegratedCompletenessError(f"Refusing to reuse non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    report, inputs = evaluate_integrated_completeness(
        repo_root=root,
        parity_contract_path=parity_contract_path,
        unified_bindings_path=unified_bindings_path,
        web_catalog_path=web_catalog_path,
        evaluator_code_path=evaluator_code_path,
        runner_code_path=runner_code_path,
    )
    report_path = output / "INTEGRATED_COMPLETENESS_REPORT.json"
    _atomic_json(report_path, report)
    report_record = _source_record(report_path, root)
    binding = {
        "format": BINDING_FORMAT,
        "analysis_version": report["analysis_version"],
        "model_version": MODEL_VERSION,
        "environment": "staging",
        "status": report["status"],
        "all_25_capabilities_four_gate_complete": report[
            "all_25_capabilities_four_gate_complete"
        ],
        "release_ready": report["release_ready"],
        "production_deployed": False,
        "fail_closed": True,
        "deterministic_for_pinned_inputs": True,
        "required_capability_count": report["required_capability_count"],
        "complete_capability_count": report["complete_capability_count"],
        "partial_capability_count": report["partial_capability_count"],
        "blocking_capability_ids": report["blocking_capability_ids"],
        "report": report_record,
        "inputs": inputs,
    }
    binding_path = output / "INTEGRATED_COMPLETENESS_BINDING.json"
    _atomic_json(binding_path, binding)
    return {
        "binding_path": str(binding_path),
        "binding_sha256": sha256_file(binding_path),
        "report_path": str(report_path),
        "report_sha256": sha256_file(report_path),
        "status": report["status"],
        "complete_capability_count": report["complete_capability_count"],
        "partial_capability_count": report["partial_capability_count"],
        "blocking_capability_ids": report["blocking_capability_ids"],
    }
