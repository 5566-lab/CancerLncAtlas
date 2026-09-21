"""R7 download catalog: closes single-cell pseudotime with server evidence."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from . import download_catalog as base


CATALOG_VERSION = "v3.2-20260827-r7"
DEPLOYMENT_FORMAT = (
    "CC_HHGT_V3_2_SINGLE_CELL_DIAGNOSTIC_SERVER_DEPLOYMENT_BINDING_V1"
)
AUDIT_FORMAT = (
    "CC_HHGT_V3_2_SINGLE_CELL_DIAGNOSTIC_SERVER_INDEPENDENT_AUDIT_BINDING_V1"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class DownloadCatalogR7Error(RuntimeError):
    """Raised when the new single-cell authority is not fail-closed."""


def _load_json(path: Path, label: str) -> dict[str, Any]:
    source = path.resolve()
    if not source.is_file() or source.is_symlink():
        raise DownloadCatalogR7Error(f"{label} is missing or unsafe: {source}")
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DownloadCatalogR7Error(f"{label} must be a JSON object")
    return value


def _validate_diagnostic_authorities(
    *,
    repository: Path,
    deployment_relative_path: str,
    deployment_sha256: str,
    audit_relative_path: str,
    audit_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    deployment_path = (repository / deployment_relative_path).resolve()
    audit_path = (repository / audit_relative_path).resolve()
    try:
        deployment_path.relative_to(repository)
        audit_path.relative_to(repository)
    except ValueError as exc:
        raise DownloadCatalogR7Error("Single-cell authority escapes repository") from exc
    deployment = _load_json(deployment_path, "single-cell diagnostic deployment")
    audit = _load_json(audit_path, "single-cell diagnostic independent audit")
    if base.sha256_file(deployment_path) != deployment_sha256:
        raise DownloadCatalogR7Error("Single-cell diagnostic deployment SHA drift")
    if base.sha256_file(audit_path) != audit_sha256:
        raise DownloadCatalogR7Error("Single-cell diagnostic audit SHA drift")
    smoke = deployment.get("real_query_smoke", {})
    semantics = deployment.get("semantics", {})
    artifacts = deployment.get("contract_artifacts", {})
    if (
        deployment.get("format") != DEPLOYMENT_FORMAT
        or deployment.get("analysis_version") != base.ANALYSIS_VERSION
        or deployment.get("status") != "SERVER_MOUNT_READY_HASH_PINNED"
        or deployment.get("cancer_count") != 17
        or deployment.get("numeric_pseudotime_rows", 0) <= 0
        or deployment.get("exact_pathway_association_rows", 0) <= 0
        or deployment.get("figure_count") != 153
        or artifacts.get("v32_sc_pseudotime", {}).get("rows", 0) <= 0
        or artifacts.get("v32_sc_figure_manifest", {}).get("rows") != 153
        or smoke.get("status") != "PASS"
        or smoke.get("real_numeric_pathway_query_exercised") is not True
        or smoke.get("real_numeric_cell_query_exercised") is not True
        or smoke.get("real_figure_manifest_exercised") is not True
        or smoke.get("real_figure_file_rehash_exercised") is not True
        or smoke.get("real_download_manifest_exercised") is not True
        or smoke.get("request_time_payload_rehash_exercised") is not True
        or smoke.get("path_traversal_rejected") is not True
        or semantics.get("root_provenance")
        != "INFERRED_CYTOTRACE2_UCELL_CONSENSUS"
        or semantics.get("root_is_explicit") is not False
        or semantics.get("diagnostic_only") is not True
        or semantics.get("model_fusion_permitted") is not False
        or semantics.get("primary_score_weight") != 0
        or semantics.get("secondary_score_weight") != 0
        or semantics.get("historical_outputs_used") is not False
        or semantics.get("changes_primary_ranking") is not False
        or deployment.get("production_deployed") is not False
    ):
        raise DownloadCatalogR7Error(
            "Single-cell diagnostic deployment violates the r7 policy"
        )
    if (
        audit.get("format") != AUDIT_FORMAT
        or audit.get("status") != "PASS_HASH_BOUND"
        or audit.get("deployment_binding_sha256") != deployment_sha256
        or audit.get("failed_checks") != 0
        or audit.get("accepted_for_staging_integration") is not True
        or audit.get("production_deployed") is not False
    ):
        raise DownloadCatalogR7Error(
            "Single-cell diagnostic independent audit did not pass"
        )
    report_decl = audit.get("report")
    if not isinstance(report_decl, Mapping):
        raise DownloadCatalogR7Error("Single-cell audit lacks a report declaration")
    report_path = Path(str(report_decl.get("path", ""))).resolve()
    try:
        report_path.relative_to(repository)
    except ValueError as exc:
        raise DownloadCatalogR7Error("Single-cell audit report escapes repository") from exc
    report_sha = str(report_decl.get("sha256", "")).lower()
    report = _load_json(report_path, "single-cell diagnostic audit report")
    if (
        _SHA256.fullmatch(report_sha) is None
        or base.sha256_file(report_path) != report_sha
        or report.get("status") != "PASS"
        or report.get("failure_count") != 0
        or report.get("accepted_for_staging_integration") is not True
        or report.get("deployment_binding", {}).get("sha256")
        != deployment_sha256
        or report.get("production_deployed") is not False
    ):
        raise DownloadCatalogR7Error(
            "Single-cell diagnostic independent-audit report drift"
        )
    return deployment, audit


def materialize_download_catalog_r7(
    *,
    repo_root: str | Path,
    output_root: str | Path,
    deployment_relative_path: str,
    deployment_sha256: str,
    audit_relative_path: str,
    audit_sha256: str,
) -> dict[str, Any]:
    repository = Path(repo_root).resolve()
    deployment_digest = deployment_sha256.lower()
    audit_digest = audit_sha256.lower()
    if _SHA256.fullmatch(deployment_digest) is None or _SHA256.fullmatch(audit_digest) is None:
        raise DownloadCatalogR7Error("Invalid diagnostic authority SHA256")
    deployment, _audit = _validate_diagnostic_authorities(
        repository=repository,
        deployment_relative_path=deployment_relative_path,
        deployment_sha256=deployment_digest,
        audit_relative_path=audit_relative_path,
        audit_sha256=audit_digest,
    )

    old_version = base.CATALOG_VERSION
    old_specs = base.AUTHORITY_SPECS
    old_build_entries = base._build_entries
    old_validate = base._validate_authority_semantics
    old_gap_metadata = base._single_cell_gap_audit_metadata

    def build_entries(
        repo_root_value: Path, capability_map: Mapping[str, list[str]]
    ) -> dict[str, dict[str, Any]]:
        entries = old_build_entries(repo_root_value, capability_map)
        entries["single_cell_pseudotime"] = base._server_download(
            "single_cell_pseudotime",
            capability_map["single_cell_pseudotime"],
            manifest_endpoint=(
                "GET /v3.2-staging/single-cell/diagnostic/{cancer_id}/download-manifest"
            ),
            download_endpoint=(
                "GET /v3.2-staging/single-cell/diagnostic/{cancer_id}/download/"
                "{relative_path:path}"
            ),
            export_formats=["tsv.gz", "pdf", "json"],
            authorities=[
                "single_cell_diagnostic_server",
                "single_cell_diagnostic_server_independent",
                "single_cell_formal_context",
                "single_cell_formal_context_independent",
                "single_cell_ucell_17c_server",
                "single_cell_gap_audit",
                "single_cell_gap_audit_independent",
            ],
            scope=(
                "FORMAL_17_OF_17_CANCERS_NUMERIC_INFERRED_ROOT_PSEUDOTIME_"
                "EXACT_PATHWAY_AND_FIGURES_ZERO_MODEL_WEIGHT"
            ),
        )
        return entries

    def validate(repo_root_value: Path) -> None:
        old_validate(repo_root_value)
        _validate_diagnostic_authorities(
            repository=repo_root_value,
            deployment_relative_path=deployment_relative_path,
            deployment_sha256=deployment_digest,
            audit_relative_path=audit_relative_path,
            audit_sha256=audit_digest,
        )

    def gap_metadata(repo_root_value: Path) -> dict[str, Any]:
        metadata = dict(old_gap_metadata(repo_root_value))
        status = dict(metadata.get("capability_status", {}))
        status["pseudotime"] = "SERVER_HASH_PINNED_DIAGNOSTIC_READY_17_CANCERS"
        status["figures"] = "SERVER_HASH_PINNED_FIGURES_READY_17_CANCERS"
        metadata.update(
            {
                "capability_status": status,
                "pseudotime_numeric_rows": deployment["numeric_pseudotime_rows"],
                "current_v32_figure_files": deployment["figure_count"],
                "diagnostic_formal_cancers_covered": 17,
                "diagnostic_formal_cancers_total": 17,
                "diagnostic_server_binding_sha256": deployment_digest,
                "diagnostic_server_independent_audit_sha256": audit_digest,
                "diagnostic_root_provenance": (
                    "INFERRED_CYTOTRACE2_UCELL_CONSENSUS"
                ),
                "diagnostic_root_is_explicit": False,
                "diagnostic_primary_score_weight": 0,
                "diagnostic_secondary_score_weight": 0,
            }
        )
        return metadata

    base.CATALOG_VERSION = CATALOG_VERSION
    base.AUTHORITY_SPECS = {
        **old_specs,
        "single_cell_diagnostic_server": (
            deployment_relative_path,
            deployment_digest,
        ),
        "single_cell_diagnostic_server_independent": (
            audit_relative_path,
            audit_digest,
        ),
    }
    base._build_entries = build_entries
    base._validate_authority_semantics = validate
    base._single_cell_gap_audit_metadata = gap_metadata
    try:
        result = base.materialize_download_catalog(
            repo_root=repository, output_root=output_root
        )
        output = Path(output_root).resolve()
        catalog_path = output / "V32_STAGING_DOWNLOAD_CATALOG.json"
        catalog = _load_json(catalog_path, "r7 download catalog")
        catalog["known_capability_gaps"] = []
        base._atomic_json(catalog_path, catalog)
        binding_path = output / "DOWNLOAD_CATALOG_BINDING.json"
        binding = _load_json(binding_path, "r7 download binding")
        binding["status"] = "MATERIALIZED_COMPLETE_HASH_PINNED_SERVER_EXPORTS"
        binding["catalog"] = {
            "path": str(catalog_path),
            "sha256": base.sha256_file(catalog_path),
            "bytes": catalog_path.stat().st_size,
        }
        base._atomic_json(binding_path, binding)
        success_path = output / "SUCCESS.json"
        success = _load_json(success_path, "r7 download success")
        success.update(
            {
                "status": binding["status"],
                "binding_sha256": base.sha256_file(binding_path),
                "catalog_sha256": base.sha256_file(catalog_path),
            }
        )
        base._atomic_json(success_path, success)
        return binding
    finally:
        base.CATALOG_VERSION = old_version
        base.AUTHORITY_SPECS = old_specs
        base._build_entries = old_build_entries
        base._validate_authority_semantics = old_validate
        base._single_cell_gap_audit_metadata = old_gap_metadata


__all__ = [
    "CATALOG_VERSION",
    "DownloadCatalogR7Error",
    "materialize_download_catalog_r7",
]
