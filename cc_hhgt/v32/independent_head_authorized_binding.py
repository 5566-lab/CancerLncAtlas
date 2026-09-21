"""Build a fail-closed authorized binding for V3.2 auxiliary functional heads.

The portable web candidate separates rewritten JSON declarations from the
large immutable payload transfer.  This module closes that publication gap
without changing model scores or scientific claims: it selects the exact
payload records already present in the portable copy manifest, verifies the
local bytes, and emits a server-native binding for an independent audit.

The output deliberately remains a *candidate* binding.  In particular Drug
keeps ``diagnostic_only`` semantics, Evidence keeps its partial status, and no
component is marked release-ready or production-deployed.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping


FORMAT = "CANCERLNCATLAS_V32_INDEPENDENT_HEAD_AUTHORIZED_BINDING_V1"
ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
ALLOWED_SERVER_ROOTS = (
    "./data/CancerLncAtlas",
    "./data/CancerLncAtlas",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")


class IndependentHeadBindingError(RuntimeError):
    """Raised when a candidate auxiliary-head bundle cannot be trusted."""


@dataclass(frozen=True)
class ComponentSpec:
    payload_directories: tuple[str, ...]
    expected_payload_files: int
    binding_relative_paths: tuple[str, ...]
    capability_status: str
    scientific_status: str
    query_payload_complete: bool
    training_archive_complete: bool
    blockers: tuple[str, ...] = ()


COMPONENTS: Mapping[str, ComponentSpec] = {
    "drug": ComponentSpec(
        payload_directories=("v32_drug_formal_r6_20260826_local_mirror",),
        expected_payload_files=699,
        binding_relative_paths=(
            "artifacts/v32_drug_formal_r6_20260826_local_mirror/DRUG_SPARSE_QUERY_MANIFEST.json",
            "artifacts/v32_drug_formal_r6_20260826_local_mirror/SPARSE_QUERY_VALIDATION_AUDIT.json",
        ),
        capability_status="AUTHORIZED_QUERY_PAYLOAD_HASH_AUDIT_PENDING",
        scientific_status="diagnostic_only",
        query_payload_complete=True,
        training_archive_complete=False,
        blockers=(
            "PRIVATE_HEAD_CHECKPOINT_BYTES_NOT_PRESENT_IN_LOCAL_QUERY_MIRROR",
            "CELL_LINE_ASSOCIATION_NOT_EFFICACY_DIRECTION_OR_TCGA_PATIENT_RESPONSE",
        ),
    ),
    "evidence": ComponentSpec(
        payload_directories=(
            "evidence_pairblocked_fresh_20260826_r2_pinned_local",
            "v32_evidence_fusion_adapter_20260826_r1",
            "v32_evidence_direction_probabilities_20260826_r1_fixed_seed_reinference",
        ),
        expected_payload_files=21,
        binding_relative_paths=(
            "artifacts/v32_evidence_output_binding_20260826_r2_local/EVIDENCE_OUTPUT_BINDING.json",
            "artifacts/v32_evidence_fusion_adapter_20260826_r1/EVIDENCE_FUSION_BINDING.json",
            "artifacts/v32_evidence_direction_probabilities_20260826_r1_fixed_seed_reinference/EVIDENCE_DIRECTION_PROBABILITY_BINDING.json",
        ),
        capability_status="AUTHORIZED_AUXILIARY_PARTIAL_HASH_AUDIT_PENDING",
        scientific_status="partial_not_publishable",
        query_payload_complete=True,
        training_archive_complete=False,
        blockers=("EVIDENCE_HEAD_REMAINS_AUXILIARY_PARTIAL_NOT_PUBLISHABLE",),
    ),
    "clinical": ComponentSpec(
        payload_directories=("v32_clinical_km_fresh_20260826_r1",),
        expected_payload_files=34,
        binding_relative_paths=(
            "artifacts/v32_clinical_km_fresh_20260826_r1/CLINICAL_KM_BINDING.json",
        ),
        capability_status="AUTHORIZED_SECONDARY_RESULT_HASH_AUDIT_PENDING",
        scientific_status="secondary_fresh_lncrna_survival_statistics",
        query_payload_complete=True,
        training_archive_complete=False,
    ),
    "state_gene_set": ComponentSpec(
        payload_directories=("v32_state_gene_sets_20260826_r2_code_bound",),
        expected_payload_files=4,
        binding_relative_paths=(
            "artifacts/v32_state_gene_sets_20260826_r2_code_bound/STATE_GENE_SET_BINDING.json",
        ),
        capability_status="AUTHORIZED_SECONDARY_RESULT_HASH_AUDIT_PENDING",
        scientific_status="secondary_state_gene_set_and_report",
        query_payload_complete=True,
        training_archive_complete=False,
    ),
    "gene_set_ranked_subtype": ComponentSpec(
        payload_directories=(
            "formal_release_materialized_1seed",
            "v32_gene_set_parity_release_20260826_r1",
        ),
        expected_payload_files=40,
        binding_relative_paths=(
            "artifacts/v32_gene_set_parity_release_20260826_r1/GENE_SET_RELEASE_MANIFEST.json",
            "artifacts/v32_gene_set_ranked_subtype_staging_20260826_r1/GENE_SET_RANKED_SUBTYPE_RELEASE_BINDING.json",
        ),
        capability_status="AUTHORIZED_DIRECT_PAYLOAD_HASH_AUDIT_PENDING_TRANSITIVE_METADATA",
        scientific_status="deterministic_v32_exact_pathway_derived_functional_head",
        query_payload_complete=True,
        training_archive_complete=False,
        blockers=(
            "PORTABLE_BINDING_HAS_24_GENE_SET_AND_72_RANKED_SUBTYPE_TRANSITIVE_METADATA_DEPENDENCIES",
            "TYPED_UNAVAILABLE_CHOL_UCS",
        ),
    ),
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _write_immutable(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.is_file() and path.read_bytes() == payload:
            return
        raise IndependentHeadBindingError(f"Refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)


def _under_allowed_root(path: str) -> bool:
    return any(path == root or path.startswith(f"{root}/") for root in ALLOWED_SERVER_ROOTS)


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)
    elif isinstance(value, str):
        yield value


def _payload_directory(target_path: str) -> str | None:
    parts = PurePosixPath(target_path).parts
    try:
        index = parts.index("artifacts")
    except ValueError:
        return None
    return parts[index + 1] if len(parts) > index + 1 else None


def _verify_binding_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise IndependentHeadBindingError(f"Binding JSON is missing or a symlink: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    forbidden: list[str] = []
    for value in _walk_strings(payload):
        if _WINDOWS_ABSOLUTE.match(value) or value.startswith("${DATA_ROOT}/"):
            forbidden.append(value)
        elif value.startswith("/") and value.startswith(
            ("/public", "/dell_", "/dsk", "/mnt", "/srv")
        ) and not _under_allowed_root(value):
            forbidden.append(value)
    if forbidden:
        raise IndependentHeadBindingError(
            f"Operational binding contains out-of-authority paths: {path}: {forbidden[:3]}"
        )
    return {
        "source_repo_relative_path": path.as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def build_authorized_binding(
    *,
    repo_root: str | Path,
    portable_overlay_root: str | Path,
    copy_manifest_path: str | Path,
    output_root: str | Path,
    publication_root: str,
    drug_materialized_root: str,
) -> dict[str, Any]:
    """Build an immutable, server-native candidate binding and audit input."""

    repository = Path(repo_root).resolve()
    overlay = Path(portable_overlay_root).resolve()
    manifest_source = Path(copy_manifest_path).resolve()
    output = Path(output_root).resolve()
    if not _under_allowed_root(publication_root) or not _under_allowed_root(
        drug_materialized_root
    ):
        raise IndependentHeadBindingError("Publication path is outside authorized roots")
    if output.exists() and (output.is_symlink() or any(output.iterdir())):
        raise IndependentHeadBindingError("Output directory must be absent or empty")
    output.mkdir(parents=True, exist_ok=True)

    audit_tool_source = repository / "scripts/audit_v32_independent_head_authorized_binding_server.py"
    if audit_tool_source.is_symlink() or not audit_tool_source.is_file():
        raise IndependentHeadBindingError("Standalone server audit tool is missing")
    audit_tool_sha = sha256_file(audit_tool_source)
    audit_tool_bytes = audit_tool_source.stat().st_size
    _write_immutable(output / "audit_server.py", audit_tool_source.read_bytes())

    copy_manifest = json.loads(manifest_source.read_text(encoding="utf-8"))
    if copy_manifest.get("format") != "CANCERLNCATLAS_V32_PORTABLE_COPY_MANIFEST_V1":
        raise IndependentHeadBindingError("Portable copy manifest format mismatch")
    candidate_root = str(copy_manifest.get("target_root", ""))
    if not _under_allowed_root(candidate_root):
        raise IndependentHeadBindingError("Candidate root is outside authorized roots")
    entries = copy_manifest.get("entries")
    if not isinstance(entries, list):
        raise IndependentHeadBindingError("Portable copy manifest has no entries")

    by_directory: dict[str, list[dict[str, Any]]] = {}
    for raw in entries:
        target = str(raw.get("target_path", ""))
        digest = str(raw.get("sha256", ""))
        size = raw.get("bytes")
        source = Path(str(raw.get("source_path", "")))
        directory = _payload_directory(target)
        if (
            not directory
            or not _under_allowed_root(target)
            or not _SHA256.fullmatch(digest)
            or not isinstance(size, int)
            or size < 0
        ):
            continue
        if source.is_symlink() or not source.is_file():
            raise IndependentHeadBindingError(f"Local payload is missing: {source}")
        if source.stat().st_size != size or sha256_file(source) != digest:
            raise IndependentHeadBindingError(f"Local payload mismatch: {source}")
        by_directory.setdefault(directory, []).append(
            {"path": target, "sha256": digest, "bytes": size}
        )

    components: dict[str, Any] = {}
    total_files = 0
    total_bytes = 0
    for component_id, spec in COMPONENTS.items():
        payloads: list[dict[str, Any]] = []
        for directory in spec.payload_directories:
            payloads.extend(by_directory.get(directory, []))
        payloads.sort(key=lambda item: item["path"])
        if len(payloads) != spec.expected_payload_files:
            raise IndependentHeadBindingError(
                f"{component_id} payload count mismatch: "
                f"{len(payloads)} != {spec.expected_payload_files}"
            )
        binding_snapshots: list[dict[str, Any]] = []
        for relative in spec.binding_relative_paths:
            source = overlay / PurePosixPath(relative)
            record = _verify_binding_json(source)
            record["source_repo_relative_path"] = relative
            record["published_snapshot_path"] = f"{publication_root}/bindings/{relative}"
            binding_snapshots.append(record)
            _write_immutable(output / "bindings" / PurePosixPath(relative), source.read_bytes())

        component_bytes = sum(int(item["bytes"]) for item in payloads)
        total_files += len(payloads)
        total_bytes += component_bytes
        components[component_id] = {
            "analysis_version": ANALYSIS_VERSION,
            "status": spec.capability_status,
            "scientific_status": spec.scientific_status,
            "payloads": payloads,
            "payload_file_count": len(payloads),
            "payload_bytes": component_bytes,
            "binding_snapshots": binding_snapshots,
            "query_payload_complete": spec.query_payload_complete,
            "training_archive_complete": spec.training_archive_complete,
            "blockers": list(spec.blockers),
            "changes_primary_ranking": False,
            "main_ranking_modified": False,
            "production_deployed": False,
            "release_ready": False,
        }

    drug_manifest_relative = COMPONENTS["drug"].binding_relative_paths[0]
    drug_audit_relative = COMPONENTS["drug"].binding_relative_paths[1]
    binding: dict[str, Any] = {
        "format": FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "status": "LOCAL_VERIFIED_SERVER_AUDIT_PENDING",
        "publication_root": publication_root,
        "candidate_payload_root": candidate_root,
        "drug_materialized_bundle": {
            "root": drug_materialized_root,
            "manifest_path": f"{drug_materialized_root}/DRUG_SPARSE_QUERY_MANIFEST.json",
            "manifest_sha256": sha256_file(overlay / drug_manifest_relative),
            "validation_audit_path": f"{drug_materialized_root}/SPARSE_QUERY_VALIDATION_AUDIT.json",
            "validation_audit_sha256": sha256_file(overlay / drug_audit_relative),
            "materialization_policy": "NEW_DIRECTORY_HARDLINK_FROM_HASH_VERIFIED_CANDIDATE_NO_SYMLINK_NO_OVERWRITE",
        },
        "source_copy_manifest": {
            "sha256": sha256_file(manifest_source),
            "entries": len(entries),
        },
        "server_audit_tool": {
            "path": f"{publication_root}/audit_server.py",
            "sha256": audit_tool_sha,
            "bytes": audit_tool_bytes,
        },
        "components": components,
        "totals": {"payload_files": total_files, "payload_bytes": total_bytes},
        "allowed_server_roots": list(ALLOWED_SERVER_ROOTS),
        "scope_guards": {
            "public0_read": False,
            "public0_write": False,
            "production_port_8260_touched": False,
            "main_score_changed": False,
            "diagnostic_promoted_to_formal": False,
            "partial_promoted_to_publishable": False,
        },
        "production_deployed": False,
        "release_ready": False,
    }
    encoded = _canonical_json(binding)
    binding_path = output / "AUTHORIZED_BINDING.json"
    _write_immutable(binding_path, encoded)
    receipt = {
        "format": "CANCERLNCATLAS_V32_INDEPENDENT_HEAD_LOCAL_RECEIPT_V1",
        "binding": {
            "path": f"{publication_root}/AUTHORIZED_BINDING.json",
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "bytes": len(encoded),
        },
        "payload_files": total_files,
        "payload_bytes": total_bytes,
        "production_deployed": False,
        "release_ready": False,
    }
    _write_immutable(output / "LOCAL_RECEIPT.json", _canonical_json(receipt))
    return binding


__all__ = [
    "ALLOWED_SERVER_ROOTS",
    "ANALYSIS_VERSION",
    "COMPONENTS",
    "FORMAT",
    "IndependentHeadBindingError",
    "build_authorized_binding",
    "sha256_file",
]
