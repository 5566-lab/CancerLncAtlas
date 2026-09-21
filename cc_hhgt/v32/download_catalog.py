"""Strict, versioned V3.2 staging download catalog.

The catalog is deliberately separate from the production website.  It binds
only current V3.2 result files (plus the static identifier/pathway crosswalk
used by dynamic queries), records every file hash, and preserves known gaps
instead of manufacturing downloadable placeholders.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from cc_hhgt.v32.capability_parity import load_parity_config


ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
MODEL_VERSION = "V3.2"
CATALOG_VERSION = "v3.2-20260827-r6"
CATALOG_FORMAT = "CANCERLNCATLAS_V32_STAGING_DOWNLOAD_CATALOG_V1"
BINDING_FORMAT = "CANCERLNCATLAS_V32_STAGING_DOWNLOAD_CATALOG_BINDING_V1"

READY_FILE = "READY_FILE"
READY_PARTS = "READY_PARTS"
PARTIAL_READY_PARTS = "PARTIAL_READY_PARTS"
DYNAMIC_QUERY_EXPORT = "DYNAMIC_QUERY_EXPORT"
PENDING_FORMAL_SUCCESS = "PENDING_FORMAL_SUCCESS"
GAP_TYPED_UNAVAILABLE = "GAP_TYPED_UNAVAILABLE"
DATA_PRESENT_DOWNLOAD_NOT_IMPLEMENTED = "DATA_PRESENT_DOWNLOAD_NOT_IMPLEMENTED"
SERVER_HASH_PINNED_DOWNLOAD = "SERVER_HASH_PINNED_DOWNLOAD"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_DOWNLOAD_ID = re.compile(r"^[a-z0-9][a-z0-9_]*$")
_LEGACY_DERIVED_PATH = re.compile(
    r"(?:^|[/\\])(?:v?2[._-][0-9]+|v?3[._-][01])(?:[/\\]|$)",
    re.IGNORECASE,
)
_INDEPENDENT_AUDIT_PASS_COUNTS = {
    "v3.2-20260826-r2": 695,
    "v3.2-20260826-r3": 714,
    "v3.2-20260826-r4": 714,
    # Filled after the independent r5 audit is materialized.  The audit count
    # is version-bound so a later catalog cannot silently reuse this mount.
    "v3.2-20260826-r5": 731,
    # Filled after the independent r6 audit is materialized.
    "v3.2-20260827-r6": 745,
    # R7 closes inferred-root pseudotime/figures with independently audited
    # request-time-rehashed COMPUTE_HOST publication evidence.
    "v3.2-20260827-r7": 754,
}


class DownloadCatalogError(RuntimeError):
    """Raised when a staging download cannot be bound fail-closed."""


# These are immutable release/binding authorities already independently
# audited in this V3.2 worktree.  Changing any authority requires a new catalog
# version rather than silently regenerating this one.
AUTHORITY_SPECS: dict[str, tuple[str, str]] = {
    "parity_contract": (
        "config/v32_historical_capability_parity.yaml",
        "",  # Filled from the canonical file and pinned in the catalog itself.
    ),
    "historical_remediation": (
        "artifacts/v32_historical_artifact_remediation_20260826_r3_evidence_direction_closed/"
        "HISTORICAL_ARTIFACT_REMEDIATION_BINDING.json",
        "b128681bff1856ab09df21c387013b431e0da956bce6f32243b1361661e0c4d8",
    ),
    "historical_remediation_audit": (
        "artifacts/v32_historical_artifact_remediation_20260826_r3_evidence_direction_closed_independent_audit/"
        "INDEPENDENT_AUDIT_BINDING.json",
        "b652fcb0cdecddfd698983322c6b485025a4a656f0b86f7856b52ef20840e3ae",
    ),
    "unified_network": (
        "artifacts/v32_unified_network_release_20260826_r1_local/"
        "NETWORK_RELEASE_MANIFEST.json",
        "ee5b954f1aa4b162a363efe2fa16296ee6bbfdd6cdb374cc5d4e0243837e7907",
    ),
    "unified_network_audit": (
        "artifacts/v32_unified_network_release_20260826_r1_local_independent_audit/"
        "NETWORK_INDEPENDENT_AUDIT_BINDING.json",
        "290c228b261f8116a8873b77912e2c49ce83a0c44484b6ddad6109108eea185d",
    ),
    "gene_set_release": (
        "artifacts/v32_gene_set_parity_release_20260826_r1/GENE_SET_RELEASE_MANIFEST.json",
        "34d8d59be3a995e86c07e3a0eaee9db63448f90a0ef983acddf2392c9947c725",
    ),
    "gene_set_audit": (
        "artifacts/v32_gene_set_parity_release_20260826_r1_independent_audit/"
        "INDEPENDENT_AUDIT_BINDING.json",
        "5ead03cb90d02e9e78417996fa490150c4b7a5005234ef22310818df18e06da7",
    ),
    "formal_model_audit": (
        "artifacts/formal_release_report_1seed/FORMAL_RELEASE_AUDIT.json",
        "1dee9c3750bb3c903884cb8d8c57da4a0bb11aa58c1b1a6b484fb203e5c1ee17",
    ),
    "formal_materialization": (
        "artifacts/formal_release_materialized_1seed/MATERIALIZATION_SUCCESS.json",
        "893aa1d2c26ef6e0d4d53965e2768bab03bca83365cf7ef9e57e6ed14c72e08c",
    ),
    "formal_materialization_audit": (
        "artifacts/formal_release_materialized_1seed/MATERIALIZATION_COVERAGE_AUDIT.json",
        "f44d6ae5c2f31c8d1f52df98c632a865c01202106750296dffad029433088639",
    ),
    "exact_lineage": (
        "artifacts/v32_full_multitask/exact_pathway_release_r2/MODULE_LINEAGE.json",
        "892f13c7a0109d644f93806fefc5b254117cef0329285ad53c8d98107459ca08",
    ),
    "state_release": (
        "artifacts/v32_state_gene_sets_20260826_r2_code_bound/STATE_GENE_SET_BINDING.json",
        "6f7a61c70f2460283100642150c0b0ed99c2fd9e016c1910d03a48b621ceefb9",
    ),
    "clinical_km": (
        "artifacts/v32_clinical_km_fresh_20260826_r1/CLINICAL_KM_BINDING.json",
        "c1905f2669dbb670bcad675c7d0ef82503cd15951cf41c8e0645b3a04c0ea6e9",
    ),
    "bulk_expression": (
        "artifacts/v32_bulk_expression_release_20260826_r2_nullable_counts/"
        "BULK_EXPRESSION_BINDING.json",
        "68d287921af53d234b6dc09e7cdf9c6c6261e0d658cf2b59358384cd06e282d9",
    ),
    "bulk_coexpression": (
        "artifacts/v32_bulk_coexpression_release_20260826_r1/"
        "BULK_COEXPRESSION_BINDING.json",
        "80de460d24c59ff13e6bf7cd3a65b96a16418d90457ef2243ba80ddfd6a753ab",
    ),
    "external_validation": (
        "artifacts/v32_external_validation_fresh_20260826_r1/"
        "EXTERNAL_VALIDATION_BINDING.json",
        "62d617fbe2fcf3d6a8afa33e969587d885498eba1d10a96ed1f9821331937713",
    ),
    "continuous_activity": (
        "artifacts/v32_continuous_pathway_activity_fresh_20260826_r1/OUTPUT_BINDING.json",
        "42ed326b81ae23de2cf1ea7423013d0a6511dbce1b72e4ebd1359ef8d3c2d143",
    ),
    "hnsc_ucell": (
        "artifacts/single_cell_cell_level_hnsc_ucell_query_binding_20260826_r1/"
        "HNSC_UCELL_QUERY_BINDING.json",
        "436891cea4795262b9f423e47d90c46a6a85389b207bd11164db3cb7fb47ea5b",
    ),
    "single_cell_ucell_17c_server": (
        "artifacts/v32_single_cell_ucell_17c_query_binding_20260827_r1_server_manifest/"
        "SERVER_DEPLOYMENT_BINDING.json",
        "4925433dde88bbb8c93ee07fcc668ae6cc043a135d7b1c1c3a8ace342e265c75",
    ),
    "drug_r6_sparse": (
        "artifacts/v32_drug_formal_r6_20260826_local_mirror/"
        "DRUG_SPARSE_QUERY_MANIFEST.json",
        "2af597e2a8c50430a58664bf55e486bddacfc0b9894ba0029b4fe8c7fb5111ed",
    ),
    "drug_mechanism_server": (
        "artifacts/v32_drug_mechanism_fresh_20260826_r1_server_manifest/"
        "SERVER_DEPLOYMENT_BINDING.json",
        "5ea2c59ef01f43c05fd9e750d08b1c0b193c10b1054dca1c0960efe0d962bf5e",
    ),
    "single_cell_gap_audit": (
        "artifacts/v32_single_cell_gap_audit_20260826_r2_codebound/"
        "SINGLE_CELL_GAP_AUDIT_BINDING.json",
        "26e94b0e0a9aa7c217d50494c2b92ac477f4c01fdcb2a2147d61dc89eab76ccf",
    ),
    "single_cell_gap_audit_independent": (
        "artifacts/v32_single_cell_gap_audit_20260826_r2_codebound_independent_audit/"
        "INDEPENDENT_AUDIT_BINDING.json",
        "1e909bc81a2180f1ab301233bf57d77911305120627f6443f7acafce7cbc1784",
    ),
    "single_cell_formal_context": (
        "artifacts/v32_single_cell_formal_context_20260826_r2_celltype_gap_explicit/"
        "SINGLE_CELL_FORMAL_CONTEXT_BINDING.json",
        "798df5cd58278e93daba3d6b7c20e47a394ce65a6c0ee3df06693c4df3c8cc4c",
    ),
    "single_cell_formal_context_independent": (
        "artifacts/v32_single_cell_formal_context_20260826_r2_celltype_gap_explicit_independent_audit/"
        "INDEPENDENT_AUDIT_BINDING.json",
        "612bfeb3a74384c706d6af98dc871535f462664580ed3dd478be90555fe1410c",
    ),
    "experiment": (
        "artifacts/v32_experiment_perturbation_fresh_20260826_r1/"
        "EXPERIMENT_PERTURBATION_BINDING.json",
        "55962a2483272308eb63667a42e2dba53f9be7eae3d99cbc3132d8ff45260f95",
    ),
    "evidence": (
        "artifacts/v32_evidence_output_binding_20260826_r2_local/EVIDENCE_OUTPUT_BINDING.json",
        "444b5d779615ff94c93890d56ab8e807fd1b9ab1592a2bba9e610a0e44b06748",
    ),
    "evidence_direction": (
        "artifacts/v32_evidence_direction_probabilities_20260826_r1_fixed_seed_reinference/"
        "EVIDENCE_DIRECTION_PROBABILITY_BINDING.json",
        "2577e382bd72e27850b10fa30d7b8af253b338adf7995031c901146618871da1",
    ),
    "evidence_direction_audit": (
        "artifacts/v32_evidence_direction_probabilities_20260826_r3_independent_audit/"
        "INDEPENDENT_AUDIT_BINDING.json",
        "b9dca24cbf30fe0e48bf9791d6f1ca1d12ad7ef767ed844221962a1fbe4e3880",
    ),
    "physical_interaction": (
        "artifacts/v32_physical_interaction_release_20260826_r2_local/"
        "INTERACTION_RELEASE_MANIFEST.json",
        "30dbb6bb27c9f58ba21472d42635db0eeb95ed55c45fc8ae24e481e917f59f5c",
    ),
    "physical_interaction_audit": (
        "artifacts/v32_physical_interaction_release_20260826_r2_local_independent_audit/"
        "PHYSICAL_INTERACTION_INDEPENDENT_AUDIT.json",
        "90d1ab6221a8594084e248289910bce7c78d04d6c22731b9b73f6c27c73fe858",
    ),
}


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


def _within_root(path: str | Path, root: Path, label: str) -> Path:
    requested = Path(path)
    if requested.is_symlink():
        raise DownloadCatalogError(f"{label} may not be a symlink")
    resolved = requested.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise DownloadCatalogError(f"{label} escapes repository root: {resolved}") from exc
    return resolved


def _safe_file(path: str | Path, root: Path, label: str) -> Path:
    resolved = _within_root(path, root, label)
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise DownloadCatalogError(f"{label} is missing or empty: {resolved}")
    return resolved


def _safe_relative_name(value: str) -> str:
    candidate = PurePosixPath(value.replace("\\", "/"))
    if (
        candidate.is_absolute()
        or not candidate.parts
        or any(part in {"", ".", ".."} for part in candidate.parts)
        or ":" in candidate.parts[0]
    ):
        raise DownloadCatalogError(f"Unsafe download-relative name: {value!r}")
    return candidate.as_posix()


def _file_record(
    repo_root: Path,
    source: str | Path,
    relative_name: str,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    path = _safe_file(source, repo_root, relative_name)
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise DownloadCatalogError(
            f"SHA256 drift for {relative_name}: {digest} != {expected_sha256}"
        )
    relative = _safe_relative_name(relative_name)
    record = {
        "source_path": str(path),
        "source_repo_relative_path": path.relative_to(repo_root).as_posix(),
        "relative_name": relative,
        "sha256": digest,
        "bytes": path.stat().st_size,
    }
    if _LEGACY_DERIVED_PATH.search(record["source_repo_relative_path"]):
        raise DownloadCatalogError(
            f"Legacy derived output may not be downloaded: {record['source_repo_relative_path']}"
        )
    return record


def _tree_sha256(parts: Iterable[Mapping[str, Any]]) -> str:
    rows = [
        f"{part['relative_name']}\t{part['sha256']}\t{part['bytes']}"
        for part in sorted(parts, key=lambda value: str(value["relative_name"]))
    ]
    return hashlib.sha256(("\n".join(rows) + "\n").encode("utf-8")).hexdigest()


def _directory_parts(
    repo_root: Path,
    source_root: str | Path,
    *,
    prefix: str = "",
    suffixes: tuple[str, ...] = (".parquet",),
) -> list[dict[str, Any]]:
    root = _within_root(source_root, repo_root, f"partition root {source_root}")
    if not root.is_dir():
        raise DownloadCatalogError(f"Partition root is missing: {root}")
    if any(path.is_symlink() for path in root.rglob("*")):
        raise DownloadCatalogError(f"Partition root contains a symlink: {root}")
    records = []
    for path in sorted(
        (candidate for candidate in root.rglob("*") if candidate.is_file()),
        key=lambda value: value.relative_to(root).as_posix(),
    ):
        if suffixes and path.suffix.lower() not in suffixes:
            continue
        relative = path.relative_to(root).as_posix()
        name = f"{prefix.rstrip('/')}/{relative}" if prefix else relative
        records.append(_file_record(repo_root, path, name))
    if not records:
        raise DownloadCatalogError(f"No partition files found: {root}")
    return records


def _base_entry(
    download_id: str,
    capability_ids: list[str],
    status: str,
    *,
    data_present: bool,
    download_implemented: bool,
    contract_complete: bool,
    authority_ids: list[str],
) -> dict[str, Any]:
    if not _SAFE_DOWNLOAD_ID.fullmatch(download_id):
        raise DownloadCatalogError(f"Invalid download ID: {download_id}")
    return {
        "download_id": download_id,
        "capability_ids": sorted(capability_ids),
        "status": status,
        "data_present": data_present,
        "download_implemented": download_implemented,
        "contract_complete": contract_complete,
        "authority_ids": sorted(set(authority_ids)),
        "model_version": MODEL_VERSION,
        "availability_encoding": "null_with_reason",
        "unavailable_fill_value": None,
        "family_to_exact_broadcast": False,
        "production_deployed": False,
    }


def _ready_file(
    repo_root: Path,
    download_id: str,
    capabilities: list[str],
    source: str,
    relative_name: str,
    expected_sha256: str,
    authorities: list[str],
) -> dict[str, Any]:
    entry = _base_entry(
        download_id,
        capabilities,
        READY_FILE,
        data_present=True,
        download_implemented=True,
        contract_complete=True,
        authority_ids=authorities,
    )
    entry["file"] = _file_record(
        repo_root, repo_root / source, relative_name, expected_sha256=expected_sha256
    )
    return entry


def _ready_parts(
    download_id: str,
    capabilities: list[str],
    parts: list[dict[str, Any]],
    authorities: list[str],
    *,
    partial: bool = False,
    known_gap: str | None = None,
    missing_components: list[str] | None = None,
) -> dict[str, Any]:
    entry = _base_entry(
        download_id,
        capabilities,
        PARTIAL_READY_PARTS if partial else READY_PARTS,
        data_present=True,
        download_implemented=True,
        contract_complete=not partial,
        authority_ids=authorities,
    )
    relative_names = [str(part["relative_name"]) for part in parts]
    if len(relative_names) != len(set(relative_names)):
        raise DownloadCatalogError(f"Duplicate part names for {download_id}")
    entry.update(
        {
            "parts": sorted(parts, key=lambda value: str(value["relative_name"])),
            "part_count": len(parts),
            "total_bytes": sum(int(part["bytes"]) for part in parts),
            "sha256_tree": _tree_sha256(parts),
        }
    )
    if known_gap:
        entry["known_gap"] = known_gap
    if missing_components:
        entry["missing_components"] = list(missing_components)
    return entry


def _dynamic(
    download_id: str,
    capabilities: list[str],
    endpoint: str,
    authorities: list[str],
) -> dict[str, Any]:
    entry = _base_entry(
        download_id,
        capabilities,
        DYNAMIC_QUERY_EXPORT,
        data_present=False,
        download_implemented=True,
        contract_complete=True,
        authority_ids=authorities,
    )
    entry.update(
        {
            "export_endpoint": endpoint,
            "result_materialization": "ON_REQUEST_NO_PLACEHOLDER_FILE",
            "export_formats": ["json", "tsv"],
        }
    )
    return entry


def _server_download(
    download_id: str,
    capabilities: list[str],
    *,
    manifest_endpoint: str,
    download_endpoint: str,
    export_formats: list[str],
    authorities: list[str],
    scope: str,
) -> dict[str, Any]:
    entry = _base_entry(
        download_id,
        capabilities,
        SERVER_HASH_PINNED_DOWNLOAD,
        data_present=True,
        download_implemented=True,
        contract_complete=True,
        authority_ids=authorities,
    )
    entry.update(
        {
            "server_location": "COMPUTE_HOST",
            "manifest_endpoint": manifest_endpoint,
            "download_endpoint": download_endpoint,
            "export_formats": list(export_formats),
            "result_materialization": "SERVER_SIDE_IMMUTABLE_HASH_PINNED_ARTIFACT",
            "request_time_payload_rehash": True,
            "scope": scope,
        }
    )
    return entry


def _pending_drug(download_id: str, capabilities: list[str]) -> dict[str, Any]:
    entry = _base_entry(
        download_id,
        capabilities,
        PENDING_FORMAL_SUCCESS,
        data_present=False,
        download_implemented=False,
        contract_complete=False,
        authority_ids=[],
    )
    entry["unavailable_reason"] = (
        "DRUG_R6_FORMAL_SUCCESS_LINEAGE_BINDING_AND_INDEPENDENT_AUDIT_NOT_YET_AVAILABLE"
    )
    return entry


def _gap(
    download_id: str,
    capabilities: list[str],
    reason: str,
    authorities: list[str],
    *,
    evidence_file: dict[str, Any] | None = None,
) -> dict[str, Any]:
    entry = _base_entry(
        download_id,
        capabilities,
        GAP_TYPED_UNAVAILABLE,
        data_present=False,
        download_implemented=False,
        contract_complete=False,
        authority_ids=authorities,
    )
    entry["unavailable_reason"] = reason
    if evidence_file is not None:
        entry["typed_availability_evidence"] = evidence_file
    return entry


def _load_json(path: Path, label: str) -> dict[str, Any]:
    source = path
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DownloadCatalogError(f"Invalid JSON {label}: {source}") from exc
    if not isinstance(value, dict):
        raise DownloadCatalogError(f"{label} must be a JSON object")
    return value


def _pin_authorities(repo_root: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for authority_id, (relative, expected) in AUTHORITY_SPECS.items():
        path = _safe_file(repo_root / relative, repo_root, f"authority {authority_id}")
        observed = sha256_file(path)
        if expected and observed != expected:
            raise DownloadCatalogError(
                f"Authority {authority_id} drifted: {observed} != {expected}"
            )
        records[authority_id] = {
            "path": str(path),
            "repo_relative_path": path.relative_to(repo_root).as_posix(),
            "sha256": observed,
            "bytes": path.stat().st_size,
        }
    return records


def _download_capabilities(contract: Mapping[str, Any]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for capability_id, capability in contract["capabilities"].items():
        for download_id in capability["gates"]["download"]["ids"]:
            result.setdefault(str(download_id), []).append(str(capability_id))
    if len(result) != 54:
        raise DownloadCatalogError(
            f"Historical download contract drifted: expected 54 IDs, observed {len(result)}"
        )
    return result


def _declared_gene_set_parts(repo_root: Path) -> list[dict[str, Any]]:
    manifest = _load_json(
        repo_root / AUTHORITY_SPECS["gene_set_release"][0], "Gene Set release manifest"
    )
    record = manifest.get("required_artifact_ids", {}).get("v32_gene_set_member_matrix", {})
    declarations = record.get("part_artifacts")
    if not isinstance(declarations, list) or len(declarations) != 33:
        raise DownloadCatalogError("Gene Set release does not declare 33 member parts")
    parts = []
    for declaration in declarations:
        path = Path(str(declaration.get("path", "")))
        part = _file_record(
            repo_root,
            path,
            f"gene_set_members/{path.name}",
            expected_sha256=str(declaration.get("sha256", "")),
        )
        parts.append(part)
    # Preserve and independently recompute the upstream tree contract too.
    upstream_tree = hashlib.sha256(
        "\n".join(
            f"{Path(part['source_path']).name}\t{part['sha256']}"
            for part in parts
        ).encode("utf-8")
    ).hexdigest()
    if upstream_tree != record.get("sha256_tree"):
        raise DownloadCatalogError("Gene Set member upstream tree SHA256 drift")
    return parts


def _build_entries(
    repo_root: Path, capabilities: dict[str, list[str]]
) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}

    def caps(download_id: str) -> list[str]:
        return capabilities[download_id]

    def add_file(
        download_id: str,
        source: str,
        name: str,
        digest: str,
        authorities: list[str],
    ) -> None:
        entries[download_id] = _ready_file(
            repo_root, download_id, caps(download_id), source, name, digest, authorities
        )

    add_file(
        "exact_pathway_predictions",
        "artifacts/v32_full_multitask/exact_pathway_release_r2/"
        "exact_pathway_five_fold_ensemble.parquet",
        "exact_pathway_predictions.parquet",
        "4259ecc7453f0c636200087e8393db59a73824414f64c78a9fbc8e3bcb47c3a1",
        ["exact_lineage"],
    )
    add_file(
        "v32_model_card",
        "artifacts/formal_release_report_1seed/MODEL_REPORT.md",
        "V3.2_MODEL_REPORT.md",
        "0616c58b92c08ec1210d7e8c5ff2a60cb88eaaeb705b667ea1727e3b0cfe13ad",
        ["formal_model_audit", "exact_lineage"],
    )

    add_file(
        "gene_set_catalog",
        "artifacts/formal_release_materialized_1seed/GENESET_MASTER.parquet",
        "gene_set_catalog.parquet",
        "c030068de9ca6f0f8568463459d9aa142e4d50f297bb0de2ca01408032358879",
        ["gene_set_release", "gene_set_audit"],
    )
    entries["gene_set_members"] = _ready_parts(
        "gene_set_members",
        caps("gene_set_members"),
        _declared_gene_set_parts(repo_root),
        ["gene_set_release", "gene_set_audit"],
    )
    add_file(
        "gene_set_enrichment",
        "artifacts/v32_gene_set_parity_release_20260826_r1/gene_set_enrichment.parquet",
        "gene_set_enrichment.parquet",
        "ae04c73576ae46c352a524e2bef4a9986f201567b2051df0a23f7859b58326d6",
        ["gene_set_release", "gene_set_audit"],
    )
    add_file(
        "gene_set_gmt",
        "artifacts/formal_release_materialized_1seed/"
        "CancerLncAtlas_V3_2_exact_pathway_ranked.gmt",
        "CancerLncAtlas_V3.2_exact_pathway_ranked.gmt",
        "3828fbb463a88133fc6c5376e07df96e8eba4a85d61fd6e927fa49abb4149318",
        ["gene_set_release", "gene_set_audit"],
    )
    add_file(
        "gene_set_report",
        "artifacts/v32_gene_set_parity_release_20260826_r1/GENE_SET_REPORT_MANIFEST.json",
        "gene_set_report.json",
        "8a274c01284899ce14d8702378e05fb3cc3e67c9f6a2f779ffbdf83fb9ce0722",
        ["gene_set_release", "gene_set_audit"],
    )

    add_file(
        "ranked_subtypes",
        "artifacts/formal_release_materialized_1seed/PATHWAY_CONTEXT_SUBTYPE.parquet",
        "ranked_exact_pathway_subtypes.parquet",
        "4b1121ba1307bd8d0b70c7888c0791f8eb2515f32fb7ada890fe4f4586841724",
        ["formal_materialization", "formal_materialization_audit"],
    )
    add_file(
        "subtype_stability",
        "artifacts/formal_release_materialized_1seed/PATHWAY_PAIRWISE_SIMILARITY.parquet",
        "ranked_subtype_stability.parquet",
        "9acf4e25c0bc30cb6f572cdf226c45e44f26a9d584bbdcb51753b6ad54093e60",
        ["formal_materialization", "formal_materialization_audit"],
    )

    add_file(
        "network_nodes",
        "artifacts/v32_unified_network_release_20260826_r1_local/v32_network_nodes.parquet",
        "v32_network_nodes.parquet",
        "5a63c4bcf701463a2e78d07ff17c176bf2394fefbaf0b1c594ede1888910eda2",
        ["unified_network", "unified_network_audit"],
    )
    add_file(
        "network_edges",
        "artifacts/v32_unified_network_release_20260826_r1_local/v32_network_edges.parquet",
        "v32_network_edges.parquet",
        "ddc4d2f85a6ba0701eeb5ea0fe1f529a64489d9a64a8e2b84fafea208176f481",
        ["unified_network", "unified_network_audit"],
    )

    add_file(
        "lncrna_state_associations",
        "artifacts/v32_full_multitask/state_release_complete/state_typed_predictions.parquet",
        "lncrna_state_associations.parquet",
        "6f1a12e1d927e06972d7e4ffaf67eed4bd7ecc978d875341bdd19d6cb88e8029",
        ["state_release"],
    )
    add_file(
        "state_gene_set_gmt",
        "artifacts/v32_state_gene_sets_20260826_r2_code_bound/state_gene_sets.gmt",
        "state_gene_sets.gmt",
        "1d3db3da6e9b049a442012e60e6ad5c717b29b5812fdd5974176293f95049cf5",
        ["state_release"],
    )
    add_file(
        "state_report",
        "artifacts/v32_state_gene_sets_20260826_r2_code_bound/STATE_GENE_SET_REPORT.md",
        "V3.2_STATE_REPORT.md",
        "3798e6993471e604c791c917e8c0877e04e8f44f54f68383307a3f3de29e1eba",
        ["state_release"],
    )

    add_file(
        "clinical_associations",
        "artifacts/v32_full_multitask/clinical_entity/entity_clinical_associations.parquet",
        "clinical_associations.parquet",
        "19d3d5c182fa07929a6b437596243a9c4c2812d371cb6c32f611cf017f361db6",
        ["historical_remediation", "historical_remediation_audit"],
    )
    add_file(
        "patient_risk_oof",
        "artifacts/v32_full_multitask/clinical/clinical_patient_risk.parquet",
        "patient_risk_oof.parquet",
        "15f6ecbb52f46f59f0b96bd3308456c14b287249304560f091d166e73faa629b",
        ["historical_remediation", "historical_remediation_audit"],
    )
    add_file(
        "translational_priority",
        "artifacts/v32_historical_artifact_remediation_20260826_r3_evidence_direction_closed/"
        "clinical_translational_priority.parquet",
        "clinical_translational_priority.parquet",
        "7b73bcf2da924bba554b7bddf65add81a53cc3f22c4e1dbf8db61b5f8af23d84",
        ["historical_remediation", "historical_remediation_audit"],
    )
    add_file(
        "clinical_report",
        "artifacts/v32_historical_artifact_remediation_20260826_r3_evidence_direction_closed/"
        "CLINICAL_RELEASE_MANIFEST.json",
        "V3.2_CLINICAL_RELEASE_REPORT.json",
        "ade9ddadbe70c6998c42a5692d7248f26a46010a56da624dda69fcf63d25d2e6",
        ["historical_remediation", "historical_remediation_audit"],
    )

    expression_source = (
        "artifacts/v32_bulk_expression_release_20260826_r2_nullable_counts/"
        "bulk_lncrna_expression_summary.parquet"
    )
    add_file(
        "bulk_lncrna_expression_summary",
        expression_source,
        "bulk_lncrna_expression_summary.parquet",
        "da44a3c781389111d84c4d210b8aded82dd71659983fadea1029021be8ce2a46",
        ["bulk_expression"],
    )
    add_file(
        "lncrna_model_coverage",
        expression_source,
        "lncrna_model_coverage.parquet",
        "da44a3c781389111d84c4d210b8aded82dd71659983fadea1029021be8ce2a46",
        ["bulk_expression"],
    )

    km_parts = _directory_parts(
        repo_root,
        repo_root / "artifacts/v32_clinical_km_fresh_20260826_r1/clinical_km_curves",
        prefix="kaplan_meier_curves",
    )
    entries["kaplan_meier_curves"] = _ready_parts(
        "kaplan_meier_curves", caps("kaplan_meier_curves"), km_parts, ["clinical_km"]
    )
    add_file(
        "kaplan_meier_statistics",
        "artifacts/v32_clinical_km_fresh_20260826_r1/clinical_km_statistics.parquet",
        "kaplan_meier_statistics.parquet",
        "831cf584e6a11a7dfd2a945b8cd293dfcc412f019b65157520cea3c7d9798052",
        ["clinical_km"],
    )

    coex_root = repo_root / "artifacts/v32_bulk_coexpression_release_20260826_r1"
    entries["tumor_lncrna_gene_coexpression"] = _ready_parts(
        "tumor_lncrna_gene_coexpression",
        caps("tumor_lncrna_gene_coexpression"),
        _directory_parts(
            repo_root,
            coex_root / "tumor_lncrna_gene_coexpression",
            prefix="tumor_lncrna_gene_coexpression",
        ),
        ["bulk_coexpression"],
    )
    cluster_parts = _directory_parts(
        repo_root,
        coex_root / "coexpression_cluster_summary",
        prefix="summary",
    ) + _directory_parts(
        repo_root,
        coex_root / "coexpression_cluster_membership",
        prefix="membership",
    )
    entries["coexpression_clusters"] = _ready_parts(
        "coexpression_clusters",
        caps("coexpression_clusters"),
        cluster_parts,
        ["bulk_coexpression"],
    )

    for download_id, filename, digest in (
        (
            "external_validation_metrics",
            "external_validation_metrics.parquet",
            "8c4ee059481c20b1367639d5ff2fe00c704b5e6d1f4b06badf0f4a47de027e53",
        ),
        (
            "external_validation_details",
            "external_validation_cohort_details.parquet",
            "fa08527b98764244d11e04cf32f94ba5920b56f728ce25cdc7ca7d74f0b6da6d",
        ),
        (
            "external_validation_overlap_audit",
            "external_validation_overlap_audit.parquet",
            "88a3ce37f7494da1e3769c3cca87cb5b2556b535803bab5d389620abcaeeea07",
        ),
    ):
        add_file(
            download_id,
            f"artifacts/v32_external_validation_fresh_20260826_r1/{filename}",
            filename,
            digest,
            ["external_validation"],
        )

    for download_id, filename, digest in (
        (
            "continuous_pathway_activity_oof",
            "continuous_pathway_activity_oof.parquet",
            "93aafb0b36b8c4cf13cf57b8b62026332e22a544a9519de70e77e46670594934",
        ),
        (
            "continuous_pathway_activity_metrics",
            "continuous_pathway_activity_metrics.parquet",
            "cb040af606f78fd531e5ad52323c0dd2524deb9de3babddfb0733d04953dd52e",
        ),
        (
            "continuous_pathway_activity_attributions",
            "continuous_pathway_activity_attributions.parquet",
            "895a15260857f5de1068942976d39cfca57db126f21e433189ea8895bb01abb6",
        ),
    ):
        add_file(
            download_id,
            f"artifacts/v32_continuous_pathway_activity_fresh_20260826_r1/{filename}",
            filename,
            digest,
            ["continuous_activity"],
        )

    sc_root = repo_root / "artifacts/v32_single_cell_fresh_20260826_r1"
    sc_association_parts = [
        _file_record(
            repo_root,
            sc_root / "lnc_celltype_summary.parquet",
            "single_cell_associations/lncrna_celltype.parquet",
            expected_sha256="28d131e3b67432d7cb8fadffa2dc35ba7d899315c521a0e83c89c190ebc9317a",
        ),
        _file_record(
            repo_root,
            sc_root / "lnc_exact_pathway.parquet",
            "single_cell_associations/lncrna_exact_pathway.parquet",
            expected_sha256="83e25646ef604b9d1a76f5872c73421ed0041c2fe097549e7c8838848ede7112",
        ),
        _file_record(
            repo_root,
            sc_root / "single_cell_typed_predictions.parquet",
            "single_cell_associations/celltype_typed_predictions.parquet",
            expected_sha256="228ae23cf8b5d320a66ab2893d66cd017671bcd804d7a17b109da7c548bfd65c",
        ),
    ]
    entries["single_cell_associations"] = _ready_parts(
        "single_cell_associations",
        caps("single_cell_associations"),
        sc_association_parts,
        ["single_cell_formal_context", "single_cell_formal_context_independent"],
    )
    add_file(
        "single_cell_activity",
        "artifacts/v32_single_cell_fresh_20260826_r1/activity.parquet",
        "single_cell_activity.parquet",
        "fcd1a2252c181841b8b51de016d65fac2599b697d051ea4eac4649e0a4ed10f1",
        ["single_cell_formal_context", "single_cell_formal_context_independent"],
    )
    ucell_root = (
        repo_root
        / "artifacts/single_cell_cell_level_hnsc_ucell_pilot_20260826_r1_query_snapshot"
    )
    entries["single_cell_ucell"] = _server_download(
        "single_cell_ucell",
        caps("single_cell_ucell"),
        manifest_endpoint=(
            "GET /v3.2-staging/single-cell/ucell/{cancer_id}/download-manifest"
        ),
        download_endpoint=(
            "GET /v3.2-staging/single-cell/ucell/{cancer_id}/download/"
            "{relative_path:path}"
        ),
        export_formats=["parquet", "json"],
        authorities=[
            "single_cell_ucell_17c_server",
            "single_cell_gap_audit",
            "single_cell_gap_audit_independent",
            "single_cell_formal_context",
            "single_cell_formal_context_independent",
        ],
        scope="FORMAL_17_OF_17_CANCERS_CELL_LEVEL_EXACT_PATHWAY_UCELL",
    )
    pseudo_evidence = _file_record(
        repo_root,
        ucell_root / "pseudotime_pathway_availability.parquet",
        "typed_availability/pseudotime_pathway_availability.parquet",
    )
    entries["single_cell_pseudotime"] = _gap(
        "single_cell_pseudotime",
        caps("single_cell_pseudotime"),
        "NO_EXPLICIT_TRAJECTORY_ROOT_OR_ORDERED_SOURCE_STATE; NUMERIC_ROWS_ZERO",
        [
            "hnsc_ucell",
            "historical_remediation",
            "historical_remediation_audit",
            "single_cell_gap_audit",
            "single_cell_gap_audit_independent",
            "single_cell_formal_context",
            "single_cell_formal_context_independent",
        ],
        evidence_file=pseudo_evidence,
    )

    genomic_source = (
        "artifacts/v32_full_multitask/genomic_fresh_rerun1/"
        "mutation_cnv_typed_predictions.parquet"
    )
    genomic_sha = "a0b4d4bfe1399dc7d34968dced86ed62d76ea76b5638236ad11694277a3a24a5"
    for download_id in ("mutation_context", "cnv_context", "cnv_associations"):
        add_file(
            download_id,
            genomic_source,
            f"{download_id}.parquet",
            genomic_sha,
            ["historical_remediation", "historical_remediation_audit"],
        )
    add_file(
        "mutation_subgroup_predictions",
        "artifacts/v32_historical_artifact_remediation_20260826_r3_evidence_direction_closed/"
        "mutation_subgroup_predictions.parquet",
        "mutation_subgroup_predictions.parquet",
        "905f97d52cb0769acba39f8f5bc8aadece2c990cfb3750d8b98f1803c28679e9",
        ["historical_remediation", "historical_remediation_audit"],
    )
    add_file(
        "cnv_coverage",
        "artifacts/v32_historical_artifact_remediation_20260826_r3_evidence_direction_closed/"
        "cnv_coverage.parquet",
        "cnv_coverage.parquet",
        "62a1c2ffceb4a76331ec01ce9050411d03ebf926b12ab3ed542af1ec989ac758",
        ["historical_remediation", "historical_remediation_audit"],
    )

    drug_root = repo_root / "artifacts/v32_drug_formal_r6_20260826_local_mirror"
    response_parts = _directory_parts(
        repo_root,
        drug_root / "drug_response_association",
        prefix="drug_response_predictions",
    )
    entries["drug_response_predictions"] = _ready_parts(
        "drug_response_predictions",
        caps("drug_response_predictions"),
        response_parts,
        ["drug_r6_sparse"],
    )
    drug_evidence_parts = _directory_parts(
        repo_root,
        drug_root / "drug_sparse_query_factors",
        prefix="drug_evidence/factors",
    )
    for name in (
        "CHECKPOINT_MANIFEST.json",
        "CONTEXT_CACHE_AUDIT.json",
        "CORE_TARGET_PREFLIGHT_AUDIT.json",
        "EXECUTION_BUDGET_AUDIT.json",
        "FACTORED_RELATION_AUDIT.json",
        "INPUT_LINEAGE_AUDIT.json",
        "MODULE_LINEAGE.json",
        "RUN_CONFIG.json",
        "SPARSE_QUERY_VALIDATION_AUDIT.json",
        "STAGING_REVALIDATION_AUDIT.json",
        "SUCCESS.json",
    ):
        drug_evidence_parts.append(
            _file_record(
                repo_root,
                drug_root / name,
                f"drug_evidence/{name}",
            )
        )
    entries["drug_evidence"] = _ready_parts(
        "drug_evidence",
        caps("drug_evidence"),
        drug_evidence_parts,
        ["drug_r6_sparse"],
    )
    entries["drug_mechanisms"] = _server_download(
        "drug_mechanisms",
        caps("drug_mechanisms"),
        manifest_endpoint=(
            "GET /v3.2-staging/drug/structural-mechanisms/download-manifest"
        ),
        download_endpoint="GET /v3.2-staging/drug/structural-mechanisms/download",
        export_formats=["parquet"],
        authorities=["drug_r6_sparse", "drug_mechanism_server"],
        scope="FRESH_V3_2_STRUCTURAL_MECHANISM_HYPOTHESES",
    )

    for download_id, filename, digest in (
        (
            "physical_interactions",
            "physical_interaction_relationships.parquet",
            "33e8f705477471fd4ed1f303b22d7e6a25d0d82fc9cff769e3e7fe178c7e3528",
        ),
        (
            "interaction_provenance",
            "relationship_evidence.parquet",
            "dae720e60f77eec43a0e6e016121fd1ba99bff68b3b3b8cd336d6d48b6f0584a",
        ),
        (
            "interaction_pathway_enrichment",
            "interaction_exact_pathway_enrichment.parquet",
            "2fa0e1524abbe7ebc3543a6d513489056337e585f9709caa3ba164cccf9fcd4f",
        ),
    ):
        add_file(
            download_id,
            f"artifacts/v32_physical_interaction_release_20260826_r2_local/{filename}",
            filename,
            digest,
            ["physical_interaction", "physical_interaction_audit"],
        )

    add_file(
        "experiment_events",
        "artifacts/v32_experiment_perturbation_fresh_20260826_r1/"
        "v32_experiment_events.parquet",
        "experiment_events.parquet",
        "e729928a000d4dcb4d2f18b22284730f047f18db78ba5a803309e1f68e718065",
        ["experiment"],
    )
    add_file(
        "perturbation_associations",
        "artifacts/v32_experiment_perturbation_fresh_20260826_r1/"
        "v32_perturbation_exact_pathway.parquet",
        "perturbation_exact_pathway_associations.parquet",
        "24ed60e6647d1db7443e7d1f506f80ced9999f2d27eaf53eac8a4cccb15edfa8",
        ["experiment"],
    )

    evidence_probability_parts = [
        _file_record(
            repo_root,
            repo_root
            / "artifacts/v32_evidence_fusion_adapter_20260826_r1/"
            "evidence_exact_fusion_expert.parquet",
            "evidence_probabilities/neural_evidence_probability.parquet",
            expected_sha256="af5a892ffc02bc9b5ef1915d211cead6f9ea8d776be9ec8e925052de75b7c0d3",
        ),
        _file_record(
            repo_root,
            repo_root
            / "artifacts/v32_multimodal_fusion_formal_20260826_r2_transparent/"
            "multimodal_secondary_scores.parquet",
            "evidence_probabilities/fused_confidence_probability.parquet",
            expected_sha256="23690313401adaaea84a1f32d5d16590d5b7bae68b8f7ae0a436899afbecee0c",
        ),
        _file_record(
            repo_root,
            repo_root
            / "artifacts/v32_evidence_direction_probabilities_20260826_r1_fixed_seed_reinference/"
            "evidence_direction_probabilities.parquet",
            "evidence_probabilities/evidence_direction_probabilities.parquet",
            expected_sha256="a2b41290275cb57c339abe76df248a3f3f2f69ef08e8a33f16b83ae8255f40a9",
        ),
    ]
    entries["evidence_probabilities"] = _ready_parts(
        "evidence_probabilities",
        caps("evidence_probabilities"),
        evidence_probability_parts,
        [
            "evidence",
            "evidence_direction",
            "evidence_direction_audit",
            "historical_remediation",
            "historical_remediation_audit",
        ],
    )
    add_file(
        "evidence_provenance",
        "artifacts/evidence_pairblocked_fresh_20260826_r2_pinned_local/"
        "event_lineage.parquet",
        "evidence_event_lineage.parquet",
        "8fc3c592678b5ef279a6b06a27ccbe43ffc996893879477e690955fae9d51528",
        ["evidence", "historical_remediation", "historical_remediation_audit"],
    )

    dynamic_routes = {
        "mixed_query_results": "POST /v3.2-staging/enrichment/mixed-exact-pathway",
        "mixed_query_mapping_report": "POST /v3.2-staging/enrichment/mixed-exact-pathway",
        # The mounted mixed endpoint dispatches all three deterministic encoder
        # contracts; separate unmounted URLs must not be advertised.
        "custom_gene_set_results": "POST /v3.2-staging/enrichment/mixed-exact-pathway",
        "protein_set_results": "POST /v3.2-staging/enrichment/mixed-exact-pathway",
    }
    for download_id, endpoint in dynamic_routes.items():
        entries[download_id] = _dynamic(
            download_id,
            caps(download_id),
            endpoint,
            ["historical_remediation", "historical_remediation_audit"],
        )

    # dataset_catalog is generated after all statuses are known, so it is added
    # by materialize_download_catalog without a circular self-hash.
    return entries


def _validate_authority_semantics(repo_root: Path) -> None:
    remediation = _load_json(
        repo_root / AUTHORITY_SPECS["historical_remediation"][0], "historical remediation"
    )
    if (
        remediation.get("model_version") != MODEL_VERSION
        or remediation.get("family_broadcast") is not False
        or remediation.get("legacy_checkpoint_or_prediction_used") is not False
        or remediation.get("production_deployed") is not False
    ):
        raise DownloadCatalogError("Historical remediation authority violates V3.2 policy")
    network = _load_json(
        repo_root / AUTHORITY_SPECS["unified_network"][0], "unified network"
    )
    if (
        network.get("status") != "SUCCESS_HASH_BOUND_CURRENT_V32_NETWORK"
        or network.get("analysis_version") != ANALYSIS_VERSION
        or network.get("family_to_exact_broadcast") is not False
        or network.get("production_deployed") is not False
    ):
        raise DownloadCatalogError("Unified network authority violates V3.2 policy")
    gene_set = _load_json(
        repo_root / AUTHORITY_SPECS["gene_set_release"][0], "Gene Set release"
    )
    if (
        gene_set.get("status") != "SUCCESS_NEWLY_MATERIALIZED_V32"
        or gene_set.get("analysis_version") != ANALYSIS_VERSION
        or gene_set.get("family_to_exact_broadcast") is not False
        or gene_set.get("old_predictions_used") is not False
        or gene_set.get("production_deployed") is not False
    ):
        raise DownloadCatalogError("Gene Set authority violates V3.2 policy")
    evidence_direction = _load_json(
        repo_root / AUTHORITY_SPECS["evidence_direction"][0],
        "Evidence direction probability release",
    )
    evidence_direction_audit = _load_json(
        repo_root / AUTHORITY_SPECS["evidence_direction_audit"][0],
        "Evidence direction probability independent audit",
    )
    direction_artifact = evidence_direction.get("artifact")
    if (
        evidence_direction.get("status") != "SUCCESS_V32_CHECKPOINT_REINFERENCE"
        or evidence_direction.get("analysis_version") != ANALYSIS_VERSION
        or evidence_direction.get("source_checkpoints_newly_trained_v32") is not True
        or evidence_direction.get("old_checkpoint_loaded") is not False
        or evidence_direction.get("old_predictions_used_as_features") is not False
        or evidence_direction.get("historical_rankings_used") is not False
        or evidence_direction.get("used_for_fusion") is not False
        or evidence_direction.get("production_deployed") is not False
        or not isinstance(direction_artifact, Mapping)
        or direction_artifact.get("rows") != 3_300_000
        or direction_artifact.get("sha256")
        != "a2b41290275cb57c339abe76df248a3f3f2f69ef08e8a33f16b83ae8255f40a9"
        or evidence_direction_audit.get("status")
        != "PASS_INDEPENDENT_V32_EVIDENCE_DIRECTION_AUDIT"
        or evidence_direction_audit.get("release_binding", {}).get("sha256")
        != AUTHORITY_SPECS["evidence_direction"][1]
        or evidence_direction_audit.get("three_class_probabilities_valid") is not True
        or evidence_direction_audit.get("typed_nulls_valid") is not True
        or evidence_direction_audit.get("production_deployed") is not False
    ):
        raise DownloadCatalogError(
            "Evidence direction authorities violate V3.2 probability policy"
        )
    ucell_server = _load_json(
        repo_root / AUTHORITY_SPECS["single_cell_ucell_17c_server"][0],
        "17-cancer UCell server deployment binding",
    )
    ucell_smoke = ucell_server.get("real_download_smoke")
    if (
        ucell_server.get("format")
        != "CC_HHGT_V3_2_SINGLE_CELL_UCELL_17C_SERVER_DEPLOYMENT_BINDING_V1"
        or ucell_server.get("analysis_version") != ANALYSIS_VERSION
        or ucell_server.get("status") != "SERVER_MOUNT_READY_HASH_PINNED"
        or ucell_server.get("cancer_count") != 17
        or ucell_server.get("ucell_capability_release_ready") is not True
        or ucell_server.get("typed_unavailable_preserved") is not True
        or ucell_server.get("historical_outputs_used") is not False
        or ucell_server.get("production_deployed") is not False
        or not isinstance(ucell_smoke, Mapping)
        or ucell_smoke.get("status") != "PASS"
        or ucell_smoke.get("request_time_payload_rehash_exercised") is not True
        or ucell_smoke.get("path_traversal_rejected") is not True
        or ucell_smoke.get("pseudotime_artifacts_excluded") is not True
    ):
        raise DownloadCatalogError(
            "17-cancer UCell server authority violates V3.2 download policy"
        )
    drug_sparse = _load_json(
        repo_root / AUTHORITY_SPECS["drug_r6_sparse"][0],
        "Drug R6 sparse query authority",
    )
    if (
        drug_sparse.get("manifest_format")
        != "CC_HHGT_V3_2_DRUG_SPARSE_QUERY_BUNDLE_V1"
        or drug_sparse.get("analysis_version") != ANALYSIS_VERSION
        or drug_sparse.get("model_run_status") != "SUCCESS"
        or drug_sparse.get("available_rows") != 16_800_054
        or drug_sparse.get("old_association_tables_read") is not False
        or drug_sparse.get("old_checkpoints_used") is not False
        or drug_sparse.get("old_predictions_used") is not False
        or drug_sparse.get("scientific_status") != "diagnostic_only"
        or drug_sparse.get("tcga_patient_response_claimed") is not False
        or drug_sparse.get("validation_audit", {}).get("status") != "PASS"
    ):
        raise DownloadCatalogError("Drug R6 sparse authority violates V3.2 policy")
    drug_server = _load_json(
        repo_root / AUTHORITY_SPECS["drug_mechanism_server"][0],
        "Drug mechanism server deployment binding",
    )
    drug_smoke = drug_server.get("real_download_smoke")
    if (
        drug_server.get("format")
        != "CC_HHGT_V3_2_DRUG_MECHANISM_SERVER_DEPLOYMENT_BINDING_V1"
        or drug_server.get("analysis_version") != ANALYSIS_VERSION
        or drug_server.get("status") != "SERVER_MOUNT_READY_HASH_PINNED"
        or drug_server.get("artifact_rows") != 123_537_897
        or drug_server.get("artifact_sha256")
        != "4565420735362879592e806d0452eb79b2140bf3f0c129cc57beac759f36cafb"
        or drug_server.get("independent_audit", {}).get("status") != "PASS"
        or drug_server.get("website_api_mount_ready") is not True
        or drug_server.get("production_deployed") is not False
        or not isinstance(drug_smoke, Mapping)
        or drug_smoke.get("status") != "PASS"
        or drug_smoke.get("request_time_payload_rehash_exercised") is not True
        or drug_server.get("semantics", {}).get("causal_mechanism_claimed") is not False
        or drug_server.get("semantics", {}).get("target_contribution_claimed") is not False
        or drug_server.get("semantics", {}).get("does_not_change_primary_pathway_ranking")
        is not True
    ):
        raise DownloadCatalogError(
            "Drug mechanism server authority violates V3.2 download policy"
        )


def _single_cell_gap_audit_metadata(repo_root: Path) -> dict[str, Any]:
    """Bind the formal single-cell gap audit without inventing a download ID."""

    release = _load_json(
        repo_root / AUTHORITY_SPECS["single_cell_gap_audit"][0],
        "single-cell formal gap-audit binding",
    )
    independent = _load_json(
        repo_root / AUTHORITY_SPECS["single_cell_gap_audit_independent"][0],
        "single-cell gap independent-audit binding",
    )
    report_record = release.get("outputs", {}).get("SINGLE_CELL_GAP_AUDIT.json")
    if not isinstance(report_record, Mapping):
        raise DownloadCatalogError("Single-cell gap binding lacks its audit report")
    report_path = _safe_file(
        report_record.get("path", ""), repo_root, "single-cell formal gap-audit report"
    )
    if (
        sha256_file(report_path) != report_record.get("sha256")
        or report_path.stat().st_size != report_record.get("bytes")
    ):
        raise DownloadCatalogError("Single-cell gap-audit report declaration drift")
    report = _load_json(report_path, "single-cell formal gap-audit report")
    independent_report_path = _safe_file(
        independent.get("report_path", ""),
        repo_root,
        "single-cell gap independent-audit report",
    )
    if sha256_file(independent_report_path) != independent.get("report_sha256"):
        raise DownloadCatalogError("Single-cell independent audit report drift")
    independent_report = _load_json(
        independent_report_path, "single-cell gap independent-audit report"
    )
    capability_status = release.get("capability_status")
    detection_counts = report.get("lncrna_detection", {}).get("counts")
    ucell = report.get("ucell")
    pseudotime = report.get("pseudotime")
    figures = report.get("figures")
    if (
        release.get("status") != "SUCCESS_HASH_BOUND_TYPED_GAP_AUDIT"
        or release.get("analysis_version") != ANALYSIS_VERSION
        or release.get("production_deployed") is not False
        or release.get("release_ready") is not False
        or not isinstance(capability_status, Mapping)
        or capability_status.get("pseudotime")
        != "GAP_TYPED_UNAVAILABLE_CURRENT_V32"
        or capability_status.get("ucell") != "PARTIAL_SCOPE_HNSC_ONLY"
        or capability_status.get("figures")
        != "GAP_TYPED_UNAVAILABLE_CURRENT_V32"
        or independent.get("status") != "PASS_HASH_BOUND"
        or independent.get("checks") != 68
        or independent.get("failed_checks") != 0
        or independent.get("release_binding_sha256")
        != AUTHORITY_SPECS["single_cell_gap_audit"][1]
        or independent.get("production_deployed") is not False
        or independent_report.get("status") != "PASS"
        or independent_report.get("failed_checks") != 0
        or not isinstance(detection_counts, Mapping)
        or detection_counts.get("cancers") != 33
        or detection_counts.get("cells") != 1_919_578
        or detection_counts.get("detected_union") != 15_879
        or not isinstance(ucell, Mapping)
        or ucell.get("formal_cancers_covered") != 1
        or ucell.get("formal_cancers_total") != 33
        or not isinstance(pseudotime, Mapping)
        or pseudotime.get("numeric_rows") != 0
        or not isinstance(figures, Mapping)
        or figures.get("current_v32_figure_files") != 0
        or report.get("historical_predictions_used") is not False
        or report.get("historical_rankings_used") is not False
        or report.get("numeric_values_invented") is not False
        or report.get("production_deployed") is not False
    ):
        raise DownloadCatalogError("Single-cell formal gap audit violates V3.2 policy")
    return {
        "kind": "FORMAL_AUDIT_METADATA_REFERENCE_NO_NEW_DOWNLOAD_CONTRACT_ID",
        "download_contract_id_created": False,
        "authority_ids": [
            "single_cell_gap_audit",
            "single_cell_gap_audit_independent",
        ],
        "release_binding_sha256": AUTHORITY_SPECS["single_cell_gap_audit"][1],
        "independent_audit_binding_sha256": AUTHORITY_SPECS[
            "single_cell_gap_audit_independent"
        ][1],
        "independent_audit_report_sha256": independent["report_sha256"],
        "independent_audit_checks": independent["checks"],
        "independent_audit_failed_checks": independent["failed_checks"],
        "capability_status": dict(capability_status),
        "lncrna_detection_counts": dict(detection_counts),
        "pseudotime_numeric_rows": pseudotime["numeric_rows"],
        "ucell_formal_cancers_covered": ucell["formal_cancers_covered"],
        "ucell_formal_cancers_total": ucell["formal_cancers_total"],
        "current_v32_figure_files": figures["current_v32_figure_files"],
        "historical_results_promoted": False,
        "production_deployed": False,
        "release_ready": False,
    }


def _single_cell_formal_context_metadata(repo_root: Path) -> dict[str, Any]:
    """Validate and summarize the independently audited 17-cancer context layer.

    This is deliberately separate from the 33-cancer raw-H5 audit and from the
    later 17-cancer cell-level UCell server release.  It binds the four
    formal-context tables without promoting pseudotime or figure gaps.
    """

    binding_path = repo_root / AUTHORITY_SPECS["single_cell_formal_context"][0]
    audit_binding_path = (
        repo_root / AUTHORITY_SPECS["single_cell_formal_context_independent"][0]
    )
    binding = _load_json(binding_path, "single-cell formal-context binding")
    audit = _load_json(
        audit_binding_path, "single-cell formal-context independent-audit binding"
    )
    audit_report_path = _safe_file(
        audit.get("audit_report_path", ""),
        repo_root,
        "single-cell formal-context independent-audit report",
    )
    if sha256_file(audit_report_path) != audit.get("audit_report_sha256"):
        raise DownloadCatalogError("Single-cell formal-context audit report drift")
    audit_report = _load_json(
        audit_report_path, "single-cell formal-context independent-audit report"
    )

    artifact_expectations: dict[str, tuple[str, int, int]] = {
        "lncrna_celltype": (
            "28d131e3b67432d7cb8fadffa2dc35ba7d899315c521a0e83c89c190ebc9317a",
            229_104,
            194_082,
        ),
        "pathway_activity": (
            "fcd1a2252c181841b8b51de016d65fac2599b697d051ea4eac4649e0a4ed10f1",
            4_354_871,
            4_320_711,
        ),
        "typed_predictions": (
            "228ae23cf8b5d320a66ab2893d66cd017671bcd804d7a17b109da7c548bfd65c",
            7_814_014,
            6_214_014,
        ),
        "exact_association": (
            "83e25646ef604b9d1a76f5872c73421ed0041c2fe097549e7c8838848ede7112",
            2_554_541,
            954_541,
        ),
    }
    artifacts = binding.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise DownloadCatalogError("Single-cell formal-context binding lacks artifacts")
    summaries: dict[str, dict[str, Any]] = {}
    for artifact_id, (expected_sha, expected_rows, expected_available) in (
        artifact_expectations.items()
    ):
        record = artifacts.get(artifact_id)
        if not isinstance(record, Mapping):
            raise DownloadCatalogError(
                f"Single-cell formal-context binding lacks {artifact_id}"
            )
        source = _safe_file(
            record.get("path", ""), repo_root, f"single-cell {artifact_id}"
        )
        if (
            record.get("sha256") != expected_sha
            or sha256_file(source) != expected_sha
            or source.stat().st_size != record.get("bytes")
            or record.get("rows") != expected_rows
            or record.get("available_rows") != expected_available
        ):
            raise DownloadCatalogError(
                f"Single-cell formal-context {artifact_id} declaration drift"
            )
        summaries[artifact_id] = {
            "sha256": expected_sha,
            "bytes": source.stat().st_size,
            "rows": expected_rows,
            "available_rows": expected_available,
        }

    cell_level = binding.get("cell_level_ucell")
    fusion = binding.get("fusion")
    if (
        binding.get("format")
        != "CC_HHGT_V3_2_SINGLE_CELL_FORMAL_CONTEXT_BINDING_V1"
        or binding.get("analysis_version") != ANALYSIS_VERSION
        or binding.get("status")
        != "FORMAL_CONTEXT_READY_WITH_CELL_LEVEL_UCELL_PARTIAL"
        or binding.get("formal_context_cancer_count") != 17
        or binding.get("raw_h5_cancer_count") != 33
        or binding.get("historical_checkpoints_used") is not False
        or binding.get("historical_predictions_used") is not False
        or binding.get("historical_rankings_used") is not False
        or binding.get("exact_primary_modified") is not False
        or binding.get("production_deployed") is not False
        or binding.get("release_ready") is not False
        or not isinstance(cell_level, Mapping)
        or cell_level.get("covered_cancers") != ["HNSC"]
        or cell_level.get("formal_cancers_total") != 17
        or cell_level.get("other_formal_cancers_require_remote_compute") != 16
        or not isinstance(fusion, Mapping)
        or fusion.get("single_cell_currently_changes_secondary_score") is not False
        or fusion.get("changes_exact_primary_score") is not False
        or audit.get("format")
        != "CC_HHGT_V3_2_SINGLE_CELL_FORMAL_CONTEXT_INDEPENDENT_AUDIT_BINDING_V1"
        or audit.get("status") != "PASS"
        or audit.get("check_count") != 25
        or audit.get("failure_count") != 0
        or audit.get("source_binding_sha256")
        != AUTHORITY_SPECS["single_cell_formal_context"][1]
        or Path(str(audit.get("source_binding_path", ""))).resolve()
        != binding_path.resolve()
        or audit.get("production_deployed") is not False
        or audit.get("release_ready") is not False
        or audit_report.get("format")
        != "CC_HHGT_V3_2_SINGLE_CELL_FORMAL_CONTEXT_INDEPENDENT_AUDIT_V1"
        or audit_report.get("status") != "PASS"
        or audit_report.get("check_count") != 25
        or audit_report.get("failure_count") != 0
        or audit_report.get("source_binding_sha256")
        != AUTHORITY_SPECS["single_cell_formal_context"][1]
    ):
        raise DownloadCatalogError(
            "Single-cell formal-context authorities violate V3.2 scope policy"
        )

    return {
        "formal_context_status": binding["status"],
        "formal_context_authority_ids": [
            "single_cell_formal_context",
            "single_cell_formal_context_independent",
        ],
        "formal_context_binding_sha256": AUTHORITY_SPECS[
            "single_cell_formal_context"
        ][1],
        "formal_context_independent_audit_binding_sha256": AUTHORITY_SPECS[
            "single_cell_formal_context_independent"
        ][1],
        "formal_context_independent_audit_report_sha256": audit[
            "audit_report_sha256"
        ],
        "formal_context_independent_audit_checks": audit["check_count"],
        "formal_context_independent_audit_failures": audit["failure_count"],
        "raw_h5_cancers_audited": binding["raw_h5_cancer_count"],
        "formal_context_cancers_available": binding["formal_context_cancer_count"],
        "formal_context_cancers_total": binding["raw_h5_cancer_count"],
        "formal_context_artifacts": summaries,
        "formal_context_download_ids": [
            "single_cell_associations",
            "single_cell_activity",
        ],
        "typed_predictions_downloaded": True,
        "single_cell_fusion_weight_zero": True,
        "exact_primary_modified": False,
        "cell_level_ucell_formal_cancers_covered": 1,
        "cell_level_ucell_formal_cancers_total": 17,
        "cell_level_ucell_remaining_cancers": 16,
        "production_deployed": False,
        "release_ready": False,
    }


def materialize_download_catalog(
    *, repo_root: str | Path, output_root: str | Path
) -> dict[str, Any]:
    """Materialize a non-production, hash-bound catalog for all 54 IDs."""

    repository = Path(repo_root).resolve()
    if not repository.is_dir():
        raise DownloadCatalogError(f"Repository root is missing: {repository}")
    output = Path(output_root).resolve()
    try:
        output.relative_to(repository)
    except ValueError as exc:
        raise DownloadCatalogError("Output must remain inside the repository") from exc
    if output.exists() and any(output.iterdir()):
        raise DownloadCatalogError(f"Refusing non-empty output reuse: {output}")
    output.mkdir(parents=True, exist_ok=True)

    authorities = _pin_authorities(repository)
    _validate_authority_semantics(repository)
    single_cell_gap_metadata = _single_cell_gap_audit_metadata(repository)
    single_cell_formal_context_metadata = _single_cell_formal_context_metadata(
        repository
    )
    single_cell_metadata = {
        **single_cell_gap_metadata,
        **single_cell_formal_context_metadata,
        "ucell_formal_cancers_covered": 17,
        "ucell_formal_cancers_total": 17,
        "cell_level_ucell_formal_cancers_covered": 17,
        "cell_level_ucell_formal_cancers_total": 17,
        "cell_level_ucell_remaining_cancers": 0,
        "cell_level_ucell_server_binding_sha256": AUTHORITY_SPECS[
            "single_cell_ucell_17c_server"
        ][1],
    }
    parity_path = repository / "config/v32_historical_capability_parity.yaml"
    contract = load_parity_config(parity_path)
    capability_map = _download_capabilities(contract)
    entries = _build_entries(repository, capability_map)
    if set(entries) != set(capability_map) - {"dataset_catalog"}:
        raise DownloadCatalogError(
            "Download implementation map does not match the 54-ID parity contract"
        )

    dataset_index = {
        "format": "CANCERLNCATLAS_V32_DATASET_CATALOG_V1",
        "catalog_version": CATALOG_VERSION,
        "model_version": MODEL_VERSION,
        "environment": "staging",
        "production_deployed": False,
        "release_ready": False,
        "download_count": len(capability_map),
        "datasets": [
            {
                "download_id": download_id,
                "capability_ids": capability_map[download_id],
                "status": READY_FILE if download_id == "dataset_catalog" else entries[download_id]["status"],
                "download_implemented": True
                if download_id == "dataset_catalog"
                else entries[download_id]["download_implemented"],
            }
            for download_id in sorted(capability_map)
        ],
    }
    dataset_path = output / "V32_DATASET_CATALOG.json"
    _atomic_json(dataset_path, dataset_index)
    entries["dataset_catalog"] = _ready_file(
        repository,
        "dataset_catalog",
        capability_map["dataset_catalog"],
        str(dataset_path.relative_to(repository)),
        "V3.2_DATASET_CATALOG.json",
        sha256_file(dataset_path),
        ["parity_contract"],
    )

    status_counts: dict[str, int] = {}
    for entry in entries.values():
        status_counts[entry["status"]] = status_counts.get(entry["status"], 0) + 1
    catalog = {
        "format": CATALOG_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "catalog_version": CATALOG_VERSION,
        "model_version": MODEL_VERSION,
        "environment": "staging",
        "production_deployed": False,
        "release_ready": False,
        "family_to_exact_broadcast": False,
        "historical_prediction_ranking_or_checkpoint_downloaded": False,
        "download_count": len(entries),
        "status_counts": dict(sorted(status_counts.items())),
        "authorities": authorities,
        "capability_audit_metadata": {
            "single_cell": single_cell_metadata,
        },
        "known_capability_gaps": [
            {
                "capability_id": "single_cell",
                "artifact_id": "v32_sc_pseudotime",
                "status": "GAP_TYPED_UNAVAILABLE",
                "reason": "NO_NUMERIC_PSEUDOTIME_VALUES",
                "audit_authority_ids": single_cell_gap_metadata["authority_ids"],
            },
            {
                "capability_id": "single_cell",
                "artifact_id": "v32_sc_figure_manifest",
                "status": "TYPED_UNAVAILABLE_NO_FIGURE_FILES",
                "reason": "FORMAL_FIGURE_FILES_ZERO",
                "audit_authority_ids": single_cell_gap_metadata["authority_ids"],
            },
        ],
        "downloads": [entries[download_id] for download_id in sorted(entries)],
    }
    catalog_path = output / "V32_STAGING_DOWNLOAD_CATALOG.json"
    _atomic_json(catalog_path, catalog)
    catalog_sha = sha256_file(catalog_path)
    binding = {
        "format": BINDING_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "catalog_version": CATALOG_VERSION,
        "model_version": MODEL_VERSION,
        "status": "MATERIALIZED_WITH_TYPED_GAPS_AND_HASH_PINNED_SERVER_EXPORTS",
        "catalog": {
            "path": str(catalog_path),
            "sha256": catalog_sha,
            "bytes": catalog_path.stat().st_size,
        },
        "dataset_catalog": {
            "path": str(dataset_path),
            "sha256": sha256_file(dataset_path),
            "bytes": dataset_path.stat().st_size,
        },
        "download_count": len(entries),
        "status_counts": catalog["status_counts"],
        "all_contract_ids_accounted_for": set(entries) == set(capability_map),
        "production_deployed": False,
        "release_ready": False,
    }
    binding_path = output / "DOWNLOAD_CATALOG_BINDING.json"
    _atomic_json(binding_path, binding)
    _atomic_json(
        output / "SUCCESS.json",
        {
            "status": binding["status"],
            "binding": binding_path.name,
            "binding_sha256": sha256_file(binding_path),
            "catalog": catalog_path.name,
            "catalog_sha256": catalog_sha,
            "download_count": len(entries),
            "production_deployed": False,
            "release_ready": False,
        },
    )
    return binding


def load_download_catalog(
    catalog_path: str | Path, *, expected_sha256: str
) -> dict[str, Any]:
    """Load and hash-verify a catalog before an API mounts it."""

    source = Path(catalog_path).resolve()
    if not _SHA256.fullmatch(str(expected_sha256).lower()):
        raise DownloadCatalogError("A pinned catalog SHA256 is required")
    if not source.is_file() or sha256_file(source) != str(expected_sha256).lower():
        raise DownloadCatalogError("Catalog SHA256 drift")
    value = _load_json(source, "download catalog")
    if (
        value.get("format") != CATALOG_FORMAT
        or value.get("model_version") != MODEL_VERSION
        or value.get("production_deployed") is not False
        or value.get("release_ready") is not False
        or value.get("download_count") != 54
    ):
        raise DownloadCatalogError("Catalog header is invalid")
    return value


def catalog_entry(catalog: Mapping[str, Any], download_id: str) -> dict[str, Any]:
    """Return one unambiguous logical download declaration."""

    matches = [
        value
        for value in catalog.get("downloads", [])
        if isinstance(value, Mapping) and value.get("download_id") == download_id
    ]
    if len(matches) != 1:
        raise DownloadCatalogError(f"Unknown or duplicate download ID: {download_id}")
    return dict(matches[0])


def _verify_catalog_file_record(
    record: Mapping[str, Any], repository_root: Path
) -> Path:
    source = Path(str(record.get("source_path", "")))
    if source.is_symlink():
        raise DownloadCatalogError(f"Download source may not be a symlink: {source}")
    resolved = source.resolve()
    try:
        repo_relative = resolved.relative_to(repository_root).as_posix()
    except ValueError as exc:
        raise DownloadCatalogError(f"Download source escapes repository: {resolved}") from exc
    if repo_relative != record.get("source_repo_relative_path"):
        raise DownloadCatalogError(f"Download source relative path drift: {resolved}")
    if not resolved.is_file() or resolved.stat().st_size != record.get("bytes"):
        raise DownloadCatalogError(f"Download source missing or byte count drifted: {resolved}")
    expected = str(record.get("sha256", ""))
    if not _SHA256.fullmatch(expected) or sha256_file(resolved) != expected:
        raise DownloadCatalogError(f"Download source SHA256 drift: {resolved}")
    _safe_relative_name(str(record.get("relative_name", "")))
    return resolved


def resolve_download_payload(
    catalog: Mapping[str, Any],
    download_id: str,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Resolve one request and recheck its hashes at request time.

    Static files and every partition are re-hashed before their paths are
    returned.  Dynamic exports return only their mounted V3.2 route.  Pending,
    partial-gap-only and typed-unavailable entries fail closed.
    """

    repository = Path(repository_root).resolve()
    entry = catalog_entry(catalog, download_id)
    status = entry.get("status")
    if status == READY_FILE:
        record = entry.get("file")
        if not isinstance(record, Mapping):
            raise DownloadCatalogError(f"READY_FILE lacks a file: {download_id}")
        path = _verify_catalog_file_record(record, repository)
        return {
            "kind": "file",
            "download_id": download_id,
            "path": str(path),
            "relative_name": record["relative_name"],
            "sha256": record["sha256"],
            "bytes": record["bytes"],
        }
    if status in {READY_PARTS, PARTIAL_READY_PARTS}:
        declared = entry.get("parts")
        if not isinstance(declared, list) or not declared:
            raise DownloadCatalogError(f"Partition download lacks parts: {download_id}")
        verified = []
        for value in declared:
            if not isinstance(value, Mapping):
                raise DownloadCatalogError(f"Invalid partition declaration: {download_id}")
            path = _verify_catalog_file_record(value, repository)
            verified.append(
                {
                    "path": str(path),
                    "relative_name": value["relative_name"],
                    "sha256": value["sha256"],
                    "bytes": value["bytes"],
                }
            )
        if _tree_sha256(declared) != entry.get("sha256_tree"):
            raise DownloadCatalogError(f"Partition tree SHA256 drift: {download_id}")
        return {
            "kind": "parts",
            "download_id": download_id,
            "status": status,
            "contract_complete": entry.get("contract_complete"),
            "sha256_tree": entry["sha256_tree"],
            "parts": verified,
            "known_gap": entry.get("known_gap"),
        }
    if status == DYNAMIC_QUERY_EXPORT:
        if entry.get("download_implemented") is not True:
            raise DownloadCatalogError(f"Dynamic export is not implemented: {download_id}")
        return {
            "kind": "dynamic_query_export",
            "download_id": download_id,
            "endpoint": entry["export_endpoint"],
            "export_formats": entry["export_formats"],
        }
    if status == SERVER_HASH_PINNED_DOWNLOAD:
        if (
            entry.get("download_implemented") is not True
            or entry.get("request_time_payload_rehash") is not True
        ):
            raise DownloadCatalogError(
                f"Server hash-pinned download is not implemented: {download_id}"
            )
        return {
            "kind": "server_hash_pinned_download",
            "download_id": download_id,
            "manifest_endpoint": entry["manifest_endpoint"],
            "download_endpoint": entry["download_endpoint"],
            "export_formats": entry["export_formats"],
            "scope": entry["scope"],
        }
    reason = entry.get("unavailable_reason", entry.get("known_gap", status))
    raise DownloadCatalogError(f"Download {download_id} is unavailable: {reason}")


def resolve_download_part(
    catalog: Mapping[str, Any],
    download_id: str,
    part_index: int,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Resolve and hash only the partition requested by the client.

    The independently audited catalog already pins every part.  At request
    time we cheaply recheck the complete declaration tree, then re-hash the
    selected file bytes.  Re-hashing every sibling for every part request made
    an N-part download perform O(N^2) I/O without strengthening the integrity
    check on the bytes actually returned by that request.
    """

    repository = Path(repository_root).resolve()
    entry = catalog_entry(catalog, download_id)
    status = entry.get("status")
    if status not in {READY_PARTS, PARTIAL_READY_PARTS}:
        if status in {
            READY_FILE,
            DYNAMIC_QUERY_EXPORT,
            SERVER_HASH_PINNED_DOWNLOAD,
        }:
            raise DownloadCatalogError(
                f"Download {download_id} is not a partitioned payload"
            )
        reason = entry.get("unavailable_reason", entry.get("known_gap", status))
        raise DownloadCatalogError(
            f"Download {download_id} is unavailable: {reason}"
        )
    declared = entry.get("parts")
    if not isinstance(declared, list) or not declared:
        raise DownloadCatalogError(f"Partition download lacks parts: {download_id}")
    if any(not isinstance(value, Mapping) for value in declared):
        raise DownloadCatalogError(f"Invalid partition declaration: {download_id}")
    if entry.get("part_count", len(declared)) != len(declared):
        raise DownloadCatalogError(f"Partition count drift: {download_id}")
    try:
        observed_tree = _tree_sha256(declared)
    except (KeyError, TypeError, ValueError) as exc:
        raise DownloadCatalogError(
            f"Invalid partition declaration: {download_id}"
        ) from exc
    if observed_tree != entry.get("sha256_tree"):
        raise DownloadCatalogError(f"Partition tree SHA256 drift: {download_id}")
    if isinstance(part_index, bool) or not isinstance(part_index, int):
        raise DownloadCatalogError("Unknown download part index")
    if part_index < 0 or part_index >= len(declared):
        raise DownloadCatalogError("Unknown download part index")
    record = declared[part_index]
    path = _verify_catalog_file_record(record, repository)
    return {
        "kind": "part",
        "download_id": download_id,
        "status": status,
        "contract_complete": entry.get("contract_complete"),
        "sha256_tree": entry["sha256_tree"],
        "part_index": part_index,
        "path": str(path),
        "relative_name": record["relative_name"],
        "sha256": record["sha256"],
        "bytes": record["bytes"],
        "known_gap": entry.get("known_gap"),
    }


def _public_value(value: Any) -> Any:
    """Remove server-local source paths before returning catalog JSON."""

    if isinstance(value, Mapping):
        return {
            str(key): _public_value(item)
            for key, item in value.items()
            if key not in {"source_path", "source_repo_relative_path", "path"}
        }
    if isinstance(value, list):
        return [_public_value(item) for item in value]
    return value


class AuditedDownloadCatalog:
    """Four-hash, independently audited staging download mount."""

    def __init__(
        self,
        binding_path: str | Path,
        *,
        expected_binding_sha256: str,
        audit_binding_path: str | Path,
        expected_audit_binding_sha256: str,
        repository_root: str | Path,
    ) -> None:
        self.repository_root = Path(repository_root).resolve()
        binding_source = _safe_file(
            binding_path, self.repository_root, "download catalog binding"
        )
        if sha256_file(binding_source) != str(expected_binding_sha256).lower():
            raise DownloadCatalogError("Download catalog binding SHA256 drift")
        binding = _load_json(binding_source, "download catalog binding")
        catalog_version = str(binding.get("catalog_version", ""))
        expected_audit_pass_count = _INDEPENDENT_AUDIT_PASS_COUNTS.get(
            catalog_version
        )
        if (
            binding.get("format") != BINDING_FORMAT
            or binding.get("model_version") != MODEL_VERSION
            or expected_audit_pass_count is None
            or binding.get("download_count") != 54
            or binding.get("production_deployed") is not False
            or binding.get("release_ready") is not False
        ):
            raise DownloadCatalogError("Download catalog binding header is invalid")
        catalog_record = binding.get("catalog")
        if not isinstance(catalog_record, Mapping):
            raise DownloadCatalogError("Download catalog binding lacks catalog declaration")
        catalog_path = _safe_file(
            catalog_record.get("path", ""), self.repository_root, "download catalog"
        )
        catalog_sha256 = str(catalog_record.get("sha256", "")).lower()
        self.catalog = load_download_catalog(
            catalog_path, expected_sha256=catalog_sha256
        )

        audit_source = _safe_file(
            audit_binding_path,
            self.repository_root,
            "download catalog independent-audit binding",
        )
        if sha256_file(audit_source) != str(expected_audit_binding_sha256).lower():
            raise DownloadCatalogError(
                "Download catalog independent-audit binding SHA256 drift"
            )
        audit = _load_json(audit_source, "download catalog independent-audit binding")
        audit_release = audit.get("release_binding")
        audit_catalog = audit.get("catalog")
        report_record = audit.get("report")
        if (
            audit.get("format")
            != "CANCERLNCATLAS_V32_STAGING_DOWNLOAD_CATALOG_INDEPENDENT_AUDIT_BINDING_V1"
            or audit.get("status") != "PASS"
            or audit.get("pass_count") != expected_audit_pass_count
            or audit.get("fail_count") != 0
            or audit.get("download_count") != 54
            or audit.get("accepted_for_staging_api_integration") is not True
            or audit.get("independent_of_materializer_implementation") is not True
            or audit.get("materializer_imported") is not False
            or audit.get("production_deployed") is not False
            or audit.get("release_ready") is not False
            or not isinstance(audit_release, Mapping)
            or Path(str(audit_release.get("path", ""))).resolve() != binding_source
            or audit_release.get("sha256") != str(expected_binding_sha256).lower()
            or not isinstance(audit_catalog, Mapping)
            or Path(str(audit_catalog.get("path", ""))).resolve() != catalog_path
            or audit_catalog.get("sha256") != catalog_sha256
            or not isinstance(report_record, Mapping)
        ):
            raise DownloadCatalogError(
                "Download catalog independent audit did not approve this binding"
            )
        report_path = _safe_file(
            report_record.get("path", ""),
            self.repository_root,
            "download catalog independent-audit report",
        )
        if sha256_file(report_path) != report_record.get("sha256"):
            raise DownloadCatalogError("Download catalog audit report SHA256 drift")
        report = _load_json(report_path, "download catalog independent-audit report")
        checks = report.get("checks")
        if (
            report.get("format")
            != "CANCERLNCATLAS_V32_STAGING_DOWNLOAD_CATALOG_INDEPENDENT_AUDIT_V1"
            or report.get("status") != "PASS"
            or report.get("pass_count") != expected_audit_pass_count
            or report.get("fail_count") != 0
            or report.get("accepted_for_staging_api_integration") is not True
            or report.get("all_file_hashes_recomputed") is not True
            or report.get("all_partition_tree_hashes_recomputed") is not True
            or report.get("materializer_imported") is not False
            or not isinstance(checks, list)
            or len(checks) != expected_audit_pass_count
            or any(
                not isinstance(check, Mapping) or check.get("status") != "PASS"
                for check in checks
            )
        ):
            raise DownloadCatalogError(
                "Download catalog independent-audit report failed the "
                f"{expected_audit_pass_count}/{expected_audit_pass_count} gate"
            )
        self.binding = binding
        self.audit = audit
        self.binding_path = binding_source
        self.audit_binding_path = audit_source
        self.catalog_path = catalog_path

    def entry(self, download_id: str) -> dict[str, Any]:
        return _public_value(catalog_entry(self.catalog, download_id))

    def public_catalog(self) -> dict[str, Any]:
        return _public_value(self.catalog)

    def resolve(self, download_id: str) -> dict[str, Any]:
        return resolve_download_payload(
            self.catalog, download_id, repository_root=self.repository_root
        )

    def resolve_part(self, download_id: str, part_index: int) -> dict[str, Any]:
        return resolve_download_part(
            self.catalog,
            download_id,
            part_index,
            repository_root=self.repository_root,
        )


__all__ = [
    "ANALYSIS_VERSION",
    "MODEL_VERSION",
    "CATALOG_VERSION",
    "CATALOG_FORMAT",
    "BINDING_FORMAT",
    "READY_FILE",
    "READY_PARTS",
    "PARTIAL_READY_PARTS",
    "DYNAMIC_QUERY_EXPORT",
    "PENDING_FORMAL_SUCCESS",
    "GAP_TYPED_UNAVAILABLE",
    "DATA_PRESENT_DOWNLOAD_NOT_IMPLEMENTED",
    "SERVER_HASH_PINNED_DOWNLOAD",
    "DownloadCatalogError",
    "AuditedDownloadCatalog",
    "sha256_file",
    "materialize_download_catalog",
    "load_download_catalog",
    "catalog_entry",
    "resolve_download_payload",
    "resolve_download_part",
]
