#!/usr/bin/env python3
"""Build and independently validate a runnable Gene Set metadata closure.

The portable rebinder conservatively followed historical JSON inventory
mentions and reported 24/72 transitive blockers.  This closure does not waive
hash checks: it validates every runtime-dereferenced record, regenerates the
two self-location-sensitive metadata leaves (audit binding and SUCCESS), then
runs the real query loader and representative 33-cancer probes.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from cc_hhgt.v32.gene_set_subtype_query import GeneSetSubtypeReleaseQuery
from cc_hhgt.v32.gene_set_subtype_release import TYPED_UNAVAILABLE
from materialize_v32_gene_set_runtime_artifact_root import (
    CORRECTION_REASON,
    EXPECTED_BYTE_CORRECTIONS,
    FORMAT as ARTIFACT_MIRROR_FORMAT,
)


FORMAT = "CANCERLNCATLAS_V32_GENE_SET_RUNTIME_METADATA_CLOSURE_V1"
ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
AUTHORIZED_BINDING_FORMAT = (
    "CANCERLNCATLAS_V32_INDEPENDENT_HEAD_AUTHORIZED_BINDING_V1"
)
RELEASE_NAME = "GENE_SET_RANKED_SUBTYPE_RELEASE_BINDING.json"
AUDIT_NAME = "INDEPENDENT_AUDIT_BINDING.json"
REPORT_NAME = "INDEPENDENT_AUDIT_REPORT.json"
SUCCESS_NAME = "SUCCESS.json"
EXPECTED_RELEASE_RECORDS = 47
EXPECTED_UPSTREAM_GENE_SET_RECORDS = 47
EXPECTED_UPSTREAM_GENE_SET_UNIQUE_RECORDS = 44
EXPECTED_LEGACY_GENE_SET_UNRESOLVED = 24
EXPECTED_LEGACY_RANKED_UNRESOLVED = 72
MAX_PEAK_RSS_KB = 2 * 1024 * 1024
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class GeneSetClosureError(RuntimeError):
    """A metadata pin, exact query, or resource gate failed closed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GeneSetClosureError(message)


def _load_pinned(path: Path, expected_sha256: str, role: str) -> dict[str, Any]:
    expected = str(expected_sha256).lower()
    _require(SHA256.fullmatch(expected) is not None, f"{role} SHA256 is invalid")
    _require(path.is_file() and not path.is_symlink(), f"{role} is missing or symlinked")
    _require(sha256_file(path) == expected, f"{role} SHA256 mismatch")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GeneSetClosureError(f"{role} is invalid JSON") from exc
    _require(isinstance(payload, dict), f"{role} must be a JSON object")
    return payload


def _encoded(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _write(path: Path, payload: bytes) -> None:
    _require(not path.exists(), f"Refusing to overwrite closure artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)


def _records(value: Any) -> list[Mapping[str, Any]]:
    observed: list[Mapping[str, Any]] = []
    if isinstance(value, Mapping):
        if "path" in value and "sha256" in value:
            observed.append(value)
        for item in value.values():
            observed.extend(_records(item))
    elif isinstance(value, list):
        for item in value:
            observed.extend(_records(item))
    return observed


def _validate_file_records(
    records: Iterable[Mapping[str, Any]],
    *,
    manifest_root: Path,
    artifact_root: Path,
    allowed_roots: tuple[Path, ...],
    origin: str,
    label: str,
) -> list[dict[str, Any]]:
    validated = []
    for index, record in enumerate(records):
        expected = str(record.get("sha256", "")).lower()
        _require(SHA256.fullmatch(expected) is not None, f"{label}[{index}] SHA invalid")
        declared = Path(str(record.get("path", "")))
        _require(declared.is_absolute(), f"{label}[{index}] declared path is not absolute")
        try:
            relative = declared.relative_to(manifest_root)
        except ValueError as exc:
            raise GeneSetClosureError(
                f"{label}[{index}] declared path escapes manifest_root: {declared}"
            ) from exc
        _require(
            relative.parts and ".." not in relative.parts,
            f"{label}[{index}] declared path contains an unsafe relative component",
        )
        unresolved = artifact_root / relative
        _require(
            not unresolved.is_symlink(),
            f"{label}[{index}] artifact is symlinked",
        )
        path = unresolved.resolve(strict=True)
        _require(
            path.is_file()
            and not path.is_symlink()
            and (path == artifact_root or artifact_root in path.parents)
            and any(path == root or root in path.parents for root in allowed_roots),
            f"{label}[{index}] path is unsafe or out of scope",
        )
        observed = sha256_file(path)
        _require(observed == expected, f"{label}[{index}] SHA drifted: {path}")
        size = path.stat().st_size
        declared_bytes = (
            int(record["bytes"]) if record.get("bytes") is not None else None
        )
        bytes_corrected = False
        if declared_bytes is not None and size != declared_bytes:
            bytes_corrected = (
                origin,
                relative.as_posix(),
                declared_bytes,
                size,
                expected,
            ) in EXPECTED_BYTE_CORRECTIONS
            _require(
                bytes_corrected,
                f"{label}[{index}] bytes drifted outside the single authorized correction",
            )
        validated.append(
            {
                "declared_path": str(declared),
                "runtime_path": str(path),
                "relative_path": relative.as_posix(),
                "sha256": observed,
                "bytes": size,
                "declared_bytes": declared_bytes,
                "bytes_metadata_corrected": bytes_corrected,
            }
        )
    return validated


def _rewrite_file_records(
    value: Any,
    *,
    manifest_root: Path,
    artifact_root: Path,
    allowed_roots: tuple[Path, ...],
    origin: str,
    label: str,
) -> Any:
    """Rewrite only hash-pinned file records after validating the mirror."""
    if isinstance(value, Mapping):
        rewritten = {key: copy.deepcopy(item) for key, item in value.items()}
        if "path" in value and "sha256" in value:
            validated = _validate_file_records(
                [value],
                manifest_root=manifest_root,
                artifact_root=artifact_root,
                allowed_roots=allowed_roots,
                origin=origin,
                label=label,
            )[0]
            rewritten["path"] = validated["runtime_path"]
            if value.get("bytes") is not None:
                rewritten["bytes"] = validated["bytes"]
        for key, item in list(rewritten.items()):
            if key == "path" and "sha256" in value:
                continue
            rewritten[key] = _rewrite_file_records(
                item,
                manifest_root=manifest_root,
                artifact_root=artifact_root,
                allowed_roots=allowed_roots,
                origin=origin,
                label=label,
            )
        return rewritten
    if isinstance(value, list):
        return [
            _rewrite_file_records(
                item,
                manifest_root=manifest_root,
                artifact_root=artifact_root,
                allowed_roots=allowed_roots,
                origin=origin,
                label=label,
            )
            for item in value
        ]
    return copy.deepcopy(value)


def _kernel_peak_rss_kb() -> int | None:
    status = Path("/proc/self/status")
    if not status.is_file():
        return None
    for line in status.read_text(encoding="ascii", errors="replace").splitlines():
        if line.startswith("VmHWM:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1])
    return None


def build_closure(
    *,
    source_binding_path: Path,
    source_binding_sha256: str,
    source_audit_path: Path,
    source_audit_sha256: str,
    source_audit_report_path: Path | None = None,
    source_audit_report_sha256: str | None = None,
    authorized_binding_path: Path,
    authorized_binding_sha256: str,
    portable_lineage_path: Path,
    portable_lineage_sha256: str,
    manifest_root: Path,
    artifact_root: Path,
    artifact_mirror_receipt_path: Path,
    artifact_mirror_receipt_sha256: str,
    output_root: Path,
    server_root: str,
) -> dict[str, Any]:
    output = output_root.resolve()
    _require(
        not output.exists(),
        f"Gene Set closure output must be absent: {output}",
    )
    _require(
        str(output).replace("\\", "/") == server_root,
        "server_root must be the exact POSIX output_root",
    )
    authorized = _load_pinned(
        authorized_binding_path.resolve(strict=True),
        authorized_binding_sha256,
        "authorized binding",
    )
    release = _load_pinned(
        source_binding_path.resolve(strict=True),
        source_binding_sha256,
        "portable Gene Set release binding",
    )
    source_audit = _load_pinned(
        source_audit_path.resolve(strict=True),
        source_audit_sha256,
        "portable Gene Set audit binding",
    )
    lineage = _load_pinned(
        portable_lineage_path.resolve(strict=True),
        portable_lineage_sha256,
        "portable rebinding lineage",
    )
    mirror_receipt = _load_pinned(
        artifact_mirror_receipt_path.resolve(strict=True),
        artifact_mirror_receipt_sha256,
        "Gene Set artifact mirror receipt",
    )
    _require(authorized.get("format") == AUTHORIZED_BINDING_FORMAT, "Authorized binding format drifted")
    _require(authorized.get("analysis_version") == ANALYSIS_VERSION, "Authorized binding analysis drifted")
    _require(release.get("analysis_version") == ANALYSIS_VERSION, "Gene Set analysis drifted")
    snapshots = authorized.get("components", {}).get(
        "gene_set_ranked_subtype", {}
    ).get("binding_snapshots", [])
    matching = [
        item
        for item in snapshots
        if Path(str(item.get("published_snapshot_path", ""))).name == RELEASE_NAME
        and item.get("sha256") == str(source_binding_sha256).lower()
    ]
    _require(len(matching) == 1, "Gene Set release SHA is not authorized")

    allowed_roots = tuple(
        Path(str(item)).resolve()
        for item in authorized.get("allowed_server_roots", [])
    )
    _require(allowed_roots, "Authorized binding has no allowed roots")
    requested_manifest_root = Path(manifest_root)
    requested_artifact_root = Path(artifact_root)
    _require(
        requested_manifest_root.is_absolute()
        and ".." not in requested_manifest_root.parts,
        "manifest_root must be an absolute path without parent traversal",
    )
    _require(
        requested_artifact_root.is_absolute()
        and ".." not in requested_artifact_root.parts,
        "artifact_root must be an absolute path without parent traversal",
    )
    manifest_root = requested_manifest_root.resolve(strict=True)
    artifact_root = requested_artifact_root.resolve(strict=True)
    _require(
        manifest_root.is_dir()
        and not manifest_root.is_symlink()
        and any(
            manifest_root == root or root in manifest_root.parents
            for root in allowed_roots
        ),
        "manifest_root is unsafe or out of authorized scope",
    )
    _require(
        artifact_root.is_dir()
        and not artifact_root.is_symlink()
        and any(
            artifact_root == root or root in artifact_root.parents
            for root in allowed_roots
        ),
        "artifact_root is unsafe or out of authorized scope",
    )
    _require(
        manifest_root != artifact_root,
        "manifest_root and artifact_root must be explicitly separated",
    )
    expected_corrections = sorted(
        (
            {
                "origin": origin,
                "relative_path": relative,
                "declared_bytes": declared,
                "actual_bytes": actual,
                "sha256": sha,
                "logical_id": f"{origin}:{relative}",
                "sha_match": True,
                "correction_reason": CORRECTION_REASON,
            }
            for origin, relative, declared, actual, sha in EXPECTED_BYTE_CORRECTIONS
        ),
        key=lambda item: item["logical_id"],
    )
    expected_correction_unique_files = len(
        {item["relative_path"] for item in expected_corrections}
    )
    expected_correction_manifest_sha = hashlib.sha256(
        (
            json.dumps(
                expected_corrections,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()
    _require(
        mirror_receipt.get("format") == ARTIFACT_MIRROR_FORMAT
        and mirror_receipt.get("status") == "PASS"
        and mirror_receipt.get("binding", {}).get("sha256")
        == str(source_binding_sha256).lower()
        and mirror_receipt.get("portable_lineage", {}).get("sha256")
        == str(portable_lineage_sha256).lower()
        and mirror_receipt.get("server_artifact_root") == str(artifact_root)
        and mirror_receipt.get("metadata_byte_corrections") == expected_corrections
        and mirror_receipt.get("metadata_byte_correction_declarations")
        == len(EXPECTED_BYTE_CORRECTIONS)
        and mirror_receipt.get("metadata_byte_correction_unique_files")
        == expected_correction_unique_files
        and mirror_receipt.get("metadata_byte_correction_manifest_sha256")
        == expected_correction_manifest_sha
        and mirror_receipt.get("negative_gates", {}).get(
            "original_manifest_modified"
        )
        is False
        and mirror_receipt.get("negative_gates", {}).get(
            "expected_sha256_changed"
        )
        is False,
        "Gene Set artifact mirror correction authority drifted",
    )
    release_records = _records(release)
    _require(len(release_records) == EXPECTED_RELEASE_RECORDS, "Gene Set release record count drifted")
    validated_release_records = _validate_file_records(
        release_records,
        manifest_root=manifest_root,
        artifact_root=artifact_root,
        allowed_roots=allowed_roots,
        origin="release",
        label="release_record",
    )
    release_byte_corrections = [
        item for item in validated_release_records if item["bytes_metadata_corrected"]
    ]
    _require(
        len(release_byte_corrections)
        == sum(origin == "release" for origin, *_ in EXPECTED_BYTE_CORRECTIONS),
        "Release byte corrections differ from the frozen correction manifest",
    )
    runtime_release = _rewrite_file_records(
        release,
        manifest_root=manifest_root,
        artifact_root=artifact_root,
        allowed_roots=allowed_roots,
        origin="release",
        label="release_record",
    )
    upstream_path = Path(
        str(runtime_release["sources"]["audited_gene_set_release_manifest"]["path"])
    ).resolve(strict=True)
    upstream = json.loads(upstream_path.read_text(encoding="utf-8"))
    _require(isinstance(upstream, dict), "Upstream Gene Set manifest is malformed")
    upstream_records = _records(upstream)
    _require(
        len(upstream_records) == EXPECTED_UPSTREAM_GENE_SET_RECORDS
        and len({(item["path"], item["sha256"]) for item in upstream_records})
        == EXPECTED_UPSTREAM_GENE_SET_UNIQUE_RECORDS,
        "Upstream Gene Set path-record closure drifted",
    )
    validated_upstream_records = _validate_file_records(
        upstream_records,
        manifest_root=manifest_root,
        artifact_root=artifact_root,
        allowed_roots=allowed_roots,
        origin="upstream",
        label="upstream_record",
    )
    _require(
        sum(item["bytes_metadata_corrected"] for item in validated_upstream_records)
        == sum(origin == "upstream" for origin, *_ in EXPECTED_BYTE_CORRECTIONS),
        "Upstream byte corrections differ from the frozen correction manifest",
    )

    rewrites = lineage.get("json_rewrites")
    _require(isinstance(rewrites, list), "Portable lineage lacks JSON rewrite records")
    gene_records = [
        item
        for item in rewrites
        if Path(str(item.get("source_path", "")).replace("\\", "/")).name
        == "GENE_SET_RELEASE_MANIFEST.json"
    ]
    ranked_records = [
        item
        for item in rewrites
        if Path(str(item.get("source_path", "")).replace("\\", "/")).name
        == RELEASE_NAME
    ]
    _require(
        len(gene_records) == 1
        and gene_records[0].get("unresolved_count") == 0
        and gene_records[0].get("transitive_unresolved_count")
        == EXPECTED_LEGACY_GENE_SET_UNRESOLVED,
        "Legacy Gene Set pending count drifted",
    )
    _require(
        len(ranked_records) == 1
        and ranked_records[0].get("unresolved_count") == 0
        and ranked_records[0].get("transitive_unresolved_count")
        == EXPECTED_LEGACY_RANKED_UNRESOLVED,
        "Legacy ranked-subtype pending count drifted",
    )
    cycles = [
        item
        for item in lineage.get("unresolved", [])
        if item.get("kind") == "json_dependency_cycle"
        and str(item.get("json_source", "")).replace("\\", "/").endswith(
            "/v32_evidence_fusion_adapter_20260826_r1_independent_post_audit/SUCCESS.json"
        )
    ]
    _require(len(cycles) == 2, "Legacy provenance-cycle root cause drifted")

    report_record = source_audit.get("report")
    _require(isinstance(report_record, dict), "Gene Set audit report record is missing")
    declared_report_path = str(report_record.get("path", ""))
    declared_report_sha = str(report_record.get("sha256", "")).lower()
    if source_audit_report_path is None:
        report_path = Path(declared_report_path).resolve(strict=True)
        report_sha = declared_report_sha
    else:
        _require(
            source_audit_report_sha256 is not None,
            "Explicit Gene Set audit report requires a SHA256",
        )
        report_path = Path(source_audit_report_path).resolve(strict=True)
        report_sha = str(source_audit_report_sha256).lower()
        _require(
            report_sha == declared_report_sha,
            "Explicit Gene Set audit report SHA differs from the audit declaration",
        )
    _require(sha256_file(report_path) == report_sha, "Gene Set audit report SHA drifted")

    release_target = output / "release" / RELEASE_NAME
    report_target = output / "audit" / REPORT_NAME
    audit_target = output / "audit" / AUDIT_NAME
    success_target = output / "release" / SUCCESS_NAME
    _write(release_target, _encoded(runtime_release))
    _write(report_target, report_path.read_bytes())
    runtime_audit = dict(source_audit)
    runtime_audit["release_binding"] = {
        **dict(source_audit["release_binding"]),
        "path": f"{server_root}/release/{RELEASE_NAME}",
        "sha256": sha256_file(release_target),
    }
    runtime_audit["report"] = {
        **dict(source_audit["report"]),
        "path": f"{server_root}/audit/{REPORT_NAME}",
        "sha256": sha256_file(report_target),
    }
    _write(audit_target, _encoded(runtime_audit))
    success = {
        "format": "CC_HHGT_V3_2_GENE_SET_RANKED_SUBTYPE_RUNTIME_SUCCESS_V1",
        "status": "PASS_RUNTIME_METADATA_CLOSED",
        "analysis_version": ANALYSIS_VERSION,
        "binding": RELEASE_NAME,
        "binding_sha256": sha256_file(release_target),
        "trained_model": False,
        "transitive_metadata_closed": True,
        "release_ready": False,
        "production_deployed": False,
    }
    _write(success_target, _encoded(success))

    query = GeneSetSubtypeReleaseQuery(
        release_target,
        expected_binding_sha256=sha256_file(release_target),
        audit_binding_path=audit_target,
        expected_audit_binding_sha256=sha256_file(audit_target),
    )
    catalog = query.query_gene_set_catalog(cancer_id="BRCA", limit=1)
    rows = catalog.get("rows")
    _require(
        catalog.get("returned_rows") == 1
        and isinstance(rows, list)
        and len(rows) == 1,
        "BRCA representative Gene Set probe is empty",
    )
    gene_set_id = str(rows[0]["geneset_id"])
    detail = query.query_gene_set_detail(gene_set_id=gene_set_id, member_limit=1)
    _require(
        detail.get("member_returned_rows") == 1
        and detail.get("gene_set", {}).get("pathway_target_level")
        == "exact_pathway",
        "Representative Gene Set member/detail probe failed",
    )
    overview = query.query_subtype_overview()
    overview_rows = overview.get("rows")
    _require(
        overview.get("returned_rows") == 33
        and isinstance(overview_rows, list)
        and len(overview_rows) == 33,
        "Gene Set overview is not 33-cancer complete",
    )
    unavailable = {
        str(row["cancer_id"]): str(row["unavailable_reason"])
        for row in overview_rows
        if row.get("available") is False
    }
    _require(
        unavailable == TYPED_UNAVAILABLE
        and sum(row.get("available") is True for row in overview_rows) == 31,
        "Gene Set typed availability drifted",
    )
    peak = _kernel_peak_rss_kb()
    _require(peak is not None and 0 < peak <= MAX_PEAK_RSS_KB, "Gene Set closure peak RSS gate failed")
    return {
        "format": FORMAT,
        "status": "PASS",
        "analysis_version": ANALYSIS_VERSION,
        "inputs": {
            "authorized_binding": {
                "path": str(authorized_binding_path.resolve()),
                "sha256": str(authorized_binding_sha256).lower(),
            },
            "portable_release_binding": {
                "path": str(source_binding_path.resolve()),
                "sha256": str(source_binding_sha256).lower(),
            },
            "portable_audit_binding": {
                "path": str(source_audit_path.resolve()),
                "sha256": str(source_audit_sha256).lower(),
            },
            "portable_audit_report": {
                "path": str(report_path),
                "sha256": report_sha,
                "declared_path": declared_report_path,
                "same_content_sha_different_runtime_path": (
                    str(report_path) != declared_report_path
                ),
            },
            "portable_rebinding_lineage": {
                "path": str(portable_lineage_path.resolve()),
                "sha256": str(portable_lineage_sha256).lower(),
            },
            "manifest_root": {
                "path": str(manifest_root),
                "role": "DECLARED_PATH_AUTHORITY_ONLY",
            },
            "artifact_root": {
                "path": str(artifact_root),
                "role": "HASH_PINNED_RUNTIME_PAYLOAD_MIRROR",
            },
            "artifact_mirror_receipt": {
                "path": str(artifact_mirror_receipt_path.resolve()),
                "sha256": str(artifact_mirror_receipt_sha256).lower(),
                "correction_manifest_sha256": mirror_receipt.get(
                    "metadata_byte_correction_manifest_sha256"
                ),
            },
        },
        "runtime_binding_declaration": {
            "binding_path": f"{server_root}/release/{RELEASE_NAME}",
            "binding_sha256": sha256_file(release_target),
            "audit_binding_path": f"{server_root}/audit/{AUDIT_NAME}",
            "audit_binding_sha256": sha256_file(audit_target),
            "scientific_status": "deterministic_v32_exact_pathway_derived_functional_head",
            "production_deployed": False,
            "release_ready": False,
        },
        "metadata_closure": {
            "transitive_metadata_closed": True,
            "release_records_hash_validated": len(validated_release_records),
            "release_records_exact_bytes": len(validated_release_records)
            - len(release_byte_corrections),
            "release_records_explicit_bytes_metadata_corrections": len(
                release_byte_corrections
            ),
            "upstream_gene_set_records_hash_validated": len(validated_upstream_records),
            "upstream_gene_set_unique_records": EXPECTED_UPSTREAM_GENE_SET_UNIQUE_RECORDS,
            "upstream_records_exact_bytes": len(validated_upstream_records)
            - sum(
                item["bytes_metadata_corrected"]
                for item in validated_upstream_records
            ),
            "upstream_records_explicit_bytes_metadata_corrections": sum(
                item["bytes_metadata_corrected"]
                for item in validated_upstream_records
            ),
            "legacy_gene_set_transitive_unresolved": EXPECTED_LEGACY_GENE_SET_UNRESOLVED,
            "legacy_ranked_subtype_transitive_unresolved": EXPECTED_LEGACY_RANKED_UNRESOLVED,
            "legacy_local_unresolved": 0,
            "legacy_false_negative_root_cause": (
                "PROVENANCE_INVENTORY_JSON_MENTIONS_CLASSIFIED_AS_LIVE_"
                "DEPENDENCIES_CREATED_TWO_ARTIFICIAL_SUCCESS_BACK_EDGES"
            ),
            "portable_rebinder_pending_was_runtime_false_negative": True,
            "manifest_root_and_artifact_root_separately_bound": True,
            "all_runtime_paths_contained_by_artifact_root": True,
            "all_runtime_payload_hashes_and_bytes_validated": True,
            "metadata_byte_correction_declarations": len(
                EXPECTED_BYTE_CORRECTIONS
            ),
            "metadata_byte_correction_unique_files": expected_correction_unique_files,
            "metadata_byte_correction_manifest_sha256": mirror_receipt.get(
                "metadata_byte_correction_manifest_sha256"
            ),
            "sha256_authority_preserved": True,
            "expected_sha256_changed": False,
            "silent_relaxation": False,
        },
        "exact_probes": {
            "representative_cancer_id": "BRCA",
            "representative_gene_set_id": gene_set_id,
            "representative_member_rows": 1,
            "cancers_typed": 33,
            "available_cancers": 31,
            "typed_unavailable": unavailable,
        },
        "resource_gate": {
            "kernel_peak_rss_kB": peak,
            "max_peak_rss_kB": MAX_PEAK_RSS_KB,
            "passed": True,
        },
        "family_to_exact_broadcast": False,
        "changes_primary_ranking": False,
        "production_port_8260_touched": False,
        "production_deployed": False,
        "release_ready": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-binding", required=True)
    parser.add_argument("--source-binding-sha256", required=True)
    parser.add_argument("--source-audit", required=True)
    parser.add_argument("--source-audit-sha256", required=True)
    parser.add_argument("--source-audit-report", required=True)
    parser.add_argument("--source-audit-report-sha256", required=True)
    parser.add_argument("--authorized-binding", required=True)
    parser.add_argument("--authorized-binding-sha256", required=True)
    parser.add_argument("--portable-lineage", required=True)
    parser.add_argument("--portable-lineage-sha256", required=True)
    parser.add_argument("--manifest-root", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--artifact-mirror-receipt", required=True)
    parser.add_argument("--artifact-mirror-receipt-sha256", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--server-root", required=True)
    args = parser.parse_args()
    output = Path(args.output_root).resolve()
    payload = build_closure(
        source_binding_path=Path(args.source_binding),
        source_binding_sha256=args.source_binding_sha256,
        source_audit_path=Path(args.source_audit),
        source_audit_sha256=args.source_audit_sha256,
        source_audit_report_path=Path(args.source_audit_report),
        source_audit_report_sha256=args.source_audit_report_sha256,
        authorized_binding_path=Path(args.authorized_binding),
        authorized_binding_sha256=args.authorized_binding_sha256,
        portable_lineage_path=Path(args.portable_lineage),
        portable_lineage_sha256=args.portable_lineage_sha256,
        manifest_root=Path(args.manifest_root),
        artifact_root=Path(args.artifact_root),
        artifact_mirror_receipt_path=Path(args.artifact_mirror_receipt),
        artifact_mirror_receipt_sha256=args.artifact_mirror_receipt_sha256,
        output_root=output,
        server_root=args.server_root,
    )
    receipt = output / "GENE_SET_RUNTIME_CLOSURE_RECEIPT.json"
    _write(receipt, _encoded(payload))
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
