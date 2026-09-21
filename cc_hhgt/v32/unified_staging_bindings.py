"""Hash-checked launcher manifest for the unified, non-production V3.2 API."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from .release_registry import artifact_sha256


SCHEMA_VERSION = "CANCERLNCATLAS_V32_UNIFIED_STAGING_BINDINGS_V1"
MOUNTED_STATUS = "MOUNTED_HASH_PINNED"
PENDING_STATUS = "PENDING_FORMAL_SUCCESS"
SERVER_MOUNT_READY_STATUS = "SERVER_MOUNT_READY_HASH_PINNED"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_ARGUMENT_PAIRS = {
    ("network_manifest_path", "network_manifest_sha256"),
    ("network_audit_binding_path", "network_audit_binding_sha256"),
    ("interaction_manifest_path", "interaction_manifest_sha256"),
    ("drug_mechanism_manifest_path", "drug_mechanism_manifest_sha256"),
    ("drug_sparse_manifest_path", "drug_sparse_manifest_sha256"),
    ("evidence_binding_path", "evidence_binding_sha256"),
    ("evidence_direction_binding_path", "evidence_direction_binding_sha256"),
    (
        "evidence_direction_audit_binding_path",
        "evidence_direction_audit_binding_sha256",
    ),
    ("experiment_perturbation_binding_path", "experiment_perturbation_binding_sha256"),
    (
        "experiment_perturbation_bridge_binding_path",
        "experiment_perturbation_bridge_binding_sha256",
    ),
    (
        "experiment_perturbation_bridge_audit_binding_path",
        "experiment_perturbation_bridge_audit_binding_sha256",
    ),
    ("single_cell_fusion_binding_path", "single_cell_fusion_binding_sha256"),
    ("single_cell_expression_binding_path", "single_cell_expression_binding_sha256"),
    ("single_cell_gap_binding_path", "single_cell_gap_binding_sha256"),
    (
        "single_cell_gap_audit_binding_path",
        "single_cell_gap_audit_binding_sha256",
    ),
    (
        "single_cell_formal_context_binding_path",
        "single_cell_formal_context_binding_sha256",
    ),
    (
        "single_cell_formal_context_audit_binding_path",
        "single_cell_formal_context_audit_binding_sha256",
    ),
    ("bulk_expression_binding_path", "bulk_expression_binding_sha256"),
    ("hnsc_ucell_binding_path", "hnsc_ucell_binding_sha256"),
    (
        "single_cell_ucell_17c_binding_path",
        "single_cell_ucell_17c_binding_sha256",
    ),
    (
        "single_cell_diagnostic_binding_path",
        "single_cell_diagnostic_binding_sha256",
    ),
    ("state_gene_set_binding_path", "state_gene_set_binding_sha256"),
    ("clinical_km_binding_path", "clinical_km_binding_sha256"),
    ("bulk_coexpression_binding_path", "bulk_coexpression_binding_sha256"),
    ("external_validation_binding_path", "external_validation_binding_sha256"),
    ("multimodal_fusion_binding_path", "multimodal_fusion_binding_sha256"),
    ("continuous_activity_binding_path", "continuous_activity_binding_sha256"),
    ("gene_set_subtype_binding_path", "gene_set_subtype_binding_sha256"),
    (
        "gene_set_subtype_audit_binding_path",
        "gene_set_subtype_audit_binding_sha256",
    ),
    (
        "exact_pathway_report_binding_path",
        "exact_pathway_report_binding_sha256",
    ),
    (
        "exact_pathway_report_audit_binding_path",
        "exact_pathway_report_audit_binding_sha256",
    ),
    (
        "historical_remediation_binding_path",
        "historical_remediation_binding_sha256",
    ),
    (
        "historical_remediation_audit_binding_path",
        "historical_remediation_audit_binding_sha256",
    ),
    ("download_catalog_binding_path", "download_catalog_binding_sha256"),
    (
        "download_catalog_audit_binding_path",
        "download_catalog_audit_binding_sha256",
    ),
    # These two independent auxiliary overlays were added after the original
    # unified manifest schema.  Keep them in the same hash-pinned launcher
    # contract so a server staging app cannot silently omit the fresh formal23
    # single-cell or directional-CNV heads.
    (
        "single_cell_formal23_success_path",
        "single_cell_formal23_success_sha256",
    ),
    ("directional_cnv_binding_path", "directional_cnv_binding_sha256"),
    (
        "directional_cnv_audit_binding_path",
        "directional_cnv_audit_binding_sha256",
    ),
}


class UnifiedStagingBindingError(RuntimeError):
    """Raised when a unified staging launcher manifest is stale or unsafe."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise UnifiedStagingBindingError(f"{label} must be an object")
    return value


def _resolve(repo_root: Path, value: Any, label: str) -> Path:
    raw = Path(str(value))
    path = raw.resolve() if raw.is_absolute() else (repo_root / raw).resolve()
    try:
        path.relative_to(repo_root)
    except ValueError as exc:
        raise UnifiedStagingBindingError(f"{label} escapes the repository root") from exc
    if not path.is_file() or path.is_symlink():
        raise UnifiedStagingBindingError(f"{label} is missing or unsafe: {path}")
    return path


def load_unified_staging_bindings(
    manifest_path: str | Path,
    *,
    repo_root: str | Path,
) -> tuple[Path, dict[str, str], dict[str, Any]]:
    """Return the validated core registry, ``create_staging_app`` kwargs and manifest."""

    root = Path(repo_root).resolve()
    source = Path(manifest_path).resolve()
    try:
        manifest = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UnifiedStagingBindingError("Cannot read unified staging manifest") from exc
    manifest = dict(_mapping(manifest, "manifest"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise UnifiedStagingBindingError("Unified staging manifest schema mismatch")
    if manifest.get("environment") != "staging":
        raise UnifiedStagingBindingError("Unified launcher accepts environment=staging only")
    if manifest.get("production_deployed") is not False:
        raise UnifiedStagingBindingError("Unified staging manifest may not claim production deployment")
    if manifest.get("release_ready") is not False:
        raise UnifiedStagingBindingError("Current unified staging candidate must remain release_ready=false")

    registry = _mapping(manifest.get("registry"), "registry")
    registry_path = _resolve(root, registry.get("path"), "registry.path")
    registry_sha = str(registry.get("sha256", "")).lower()
    if not _SHA256.fullmatch(registry_sha) or artifact_sha256(registry_path) != registry_sha:
        raise UnifiedStagingBindingError("Core registry hash mismatch")

    bindings = _mapping(manifest.get("bindings"), "bindings")
    kwargs: dict[str, str] = {}
    seen_arguments: set[str] = set()
    for capability_id, raw_entry in bindings.items():
        entry = _mapping(raw_entry, f"bindings.{capability_id}")
        status = entry.get("status")
        if status == PENDING_STATUS:
            if not str(entry.get("reason", "")).strip():
                raise UnifiedStagingBindingError(
                    f"Pending binding {capability_id} lacks a reason"
                )
            continue
        if status == SERVER_MOUNT_READY_STATUS:
            if not str(entry.get("reason", "")).strip():
                raise UnifiedStagingBindingError(
                    f"Server-ready binding {capability_id} lacks a reason"
                )
            if entry.get("production_deployed") is not False:
                raise UnifiedStagingBindingError(
                    f"Server-ready binding {capability_id} may not claim deployment"
                )
            path = _resolve(
                root,
                entry.get("deployment_binding_path"),
                f"bindings.{capability_id}.deployment_binding_path",
            )
            digest = str(entry.get("deployment_binding_sha256", "")).lower()
            if not _SHA256.fullmatch(digest) or artifact_sha256(path) != digest:
                raise UnifiedStagingBindingError(
                    f"Server deployment binding hash mismatch: {capability_id}"
                )
            continue
        if status != MOUNTED_STATUS:
            raise UnifiedStagingBindingError(
                f"Binding {capability_id} has unsupported status {status!r}"
            )
        pair = (str(entry.get("path_argument", "")), str(entry.get("sha_argument", "")))
        if pair not in _ALLOWED_ARGUMENT_PAIRS:
            raise UnifiedStagingBindingError(
                f"Binding {capability_id} uses an unknown API argument pair"
            )
        if pair[0] in seen_arguments or pair[1] in seen_arguments:
            raise UnifiedStagingBindingError(f"Duplicate launcher argument for {capability_id}")
        path = _resolve(root, entry.get("path"), f"bindings.{capability_id}.path")
        digest = str(entry.get("sha256", "")).lower()
        if not _SHA256.fullmatch(digest) or artifact_sha256(path) != digest:
            raise UnifiedStagingBindingError(f"Binding hash mismatch: {capability_id}")
        kwargs[pair[0]] = str(path)
        kwargs[pair[1]] = digest
        seen_arguments.update(pair)
    return registry_path, kwargs, manifest


def create_app_from_unified_bindings(
    manifest_path: str | Path,
    *,
    repo_root: str | Path,
):
    """Build the isolated FastAPI app after validating every mounted hash."""

    from website.backend.v32_staging_api import create_staging_app

    registry_path, kwargs, _ = load_unified_staging_bindings(
        manifest_path,
        repo_root=repo_root,
    )
    return create_staging_app(registry_path, **kwargs)


__all__ = [
    "MOUNTED_STATUS",
    "PENDING_STATUS",
    "SERVER_MOUNT_READY_STATUS",
    "SCHEMA_VERSION",
    "UnifiedStagingBindingError",
    "create_app_from_unified_bindings",
    "load_unified_staging_bindings",
]
