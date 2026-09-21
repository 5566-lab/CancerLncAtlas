#!/usr/bin/env python3
"""Materialize a complete, immutable Gene Set runtime payload mirror.

The source binding keeps absolute paths from the candidate manifest root.
This tool maps those paths to a local source checkout, copies only hash-pinned
files into a new artifact root, and emits a deterministic audit receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


FORMAT = "CANCERLNCATLAS_V32_GENE_SET_RUNTIME_ARTIFACT_MIRROR_V1"
EXPECTED_RELEASE_RECORDS = 47
EXPECTED_UPSTREAM_RECORDS = 47
EXPECTED_UPSTREAM_UNIQUE_RECORDS = 44
SHA256 = re.compile(r"^[0-9a-f]{64}$")
STALE_BYTES_RELATIVE_PATH = (
    "artifacts/v32_gene_set_parity_release_20260826_r1/"
    "GENE_SET_RELEASE_MANIFEST.json"
)
STALE_BYTES_DECLARED = 16_913
STALE_BYTES_ACTUAL = 14_568
STALE_BYTES_SHA256 = "1b17edbaa3fa3a1f4be8f99799302972a1574f14632277323d987c1676df5174"
CORRECTION_REASON = (
    "PORTABLE_JSON_REBINDING_REENCODED_JSON_AND_UPDATED_CONTENT_SHA256_"
    "WHILE_PARENT_BYTES_FIELD_RETAINED_PRE_REBIND_SIZE"
)
EXPECTED_BYTE_CORRECTIONS = (
    ("release", "artifacts/formal_release_materialized_1seed/MATERIALIZATION_COVERAGE_AUDIT.json", 636, 505, "eb371aed94f1b68a9288a2a63cf4ddafc9c17e974e86bcf74ec7142035e1b15d"),
    ("release", "artifacts/formal_release_materialized_1seed/MATERIALIZATION_SUCCESS.json", 541, 458, "10095b8fb0f35aea2307ae1177e9397fbb77301181bd8ad7684c65192a22c52c"),
    ("release", "artifacts/v32_gene_set_parity_release_20260826_r1/GENE_SET_RELEASE_MANIFEST.json", 16_913, 14_568, "1b17edbaa3fa3a1f4be8f99799302972a1574f14632277323d987c1676df5174"),
    ("release", "artifacts/v32_gene_set_parity_release_20260826_r1/GENE_SET_REPORT_MANIFEST.json", 1_692, 1_358, "c75f2099a60c46f6d970bb1d2b119edd25a337cac873b8c0a2e2a12c931bf226"),
    ("release", "artifacts/v32_gene_set_parity_release_20260826_r1_independent_audit/INDEPENDENT_AUDIT_BINDING.json", 876, 828, "f77431e1a5126a00ba33a386e4543fddf5962e6ce279ee32a30abf7323400393"),
    ("upstream", "artifacts/formal_release_materialized_1seed/MATERIALIZATION_SUCCESS.json", 541, 458, "10095b8fb0f35aea2307ae1177e9397fbb77301181bd8ad7684c65192a22c52c"),
    ("upstream", "artifacts/v32_gene_set_parity_release_20260826_r1/GENE_SET_REPORT_MANIFEST.json", 1_692, 1_358, "c75f2099a60c46f6d970bb1d2b119edd25a337cac873b8c0a2e2a12c931bf226"),
    ("upstream", "artifacts/v32_multimodal_fusion_formal_20260826_r2_transparent/MULTIMODAL_FUSION_BINDING.json", 8_927, 8_133, "960d026fbfd50986b2f422efa08a0cce53c8d04c7518e1ce9e7aa9449b3568cb"),
    ("upstream", "artifacts/v32_multimodal_fusion_formal_20260826_r2_transparent_independent_post_audit/INDEPENDENT_POST_AUDIT_BINDING.json", 1_859, 1_746, "3152e3fab3e9858eafe6109b71ff6b292ce7c76111a631ee2220bce1233e0c0f"),
    ("upstream", "artifacts/v32_physical_interaction_release_20260826_r2_local/INTERACTION_RELEASE_MANIFEST.json", 5_447, 4_979, "ad8ddab76c0d2605f549308f25dcf9f1fa705dfc4744b49359ec406551ac332f"),
    ("upstream", "artifacts/v32_physical_interaction_release_20260826_r2_local_independent_audit/PHYSICAL_INTERACTION_INDEPENDENT_AUDIT.json", 30_285, 23_322, "85d81ee20136e6f110e062bed1b4d0d7abdd9728e218c55f418403f662f5d17c"),
)


class GeneSetMirrorError(RuntimeError):
    """The runtime mirror could not be proven byte-identical and contained."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GeneSetMirrorError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_pinned(path: Path, expected_sha256: str) -> dict[str, Any]:
    expected = str(expected_sha256).lower()
    _require(SHA256.fullmatch(expected) is not None, "Binding SHA256 is invalid")
    _require(path.is_file() and not path.is_symlink(), "Binding is missing or symlinked")
    _require(_sha256(path) == expected, "Binding SHA256 mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "Binding must be a JSON object")
    return payload


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


def _relative(record: Mapping[str, Any], manifest_root: PurePosixPath) -> PurePosixPath:
    raw = PurePosixPath(str(record.get("path", "")))
    _require(raw.is_absolute(), f"Declared payload path is not absolute: {raw}")
    try:
        relative = raw.relative_to(manifest_root)
    except ValueError as exc:
        raise GeneSetMirrorError(f"Declared payload escapes manifest root: {raw}") from exc
    _require(relative.parts and ".." not in relative.parts, f"Unsafe payload path: {raw}")
    return relative


def _source(
    record: Mapping[str, Any],
    *,
    manifest_root: PurePosixPath,
    source_roots: tuple[Path, ...],
) -> tuple[PurePosixPath, Path, str, int, int | None, bool]:
    relative = _relative(record, manifest_root)
    expected = str(record.get("sha256", "")).lower()
    _require(SHA256.fullmatch(expected) is not None, f"Invalid payload SHA: {relative}")
    candidates: list[tuple[Path, str, int]] = []
    for source_root in source_roots:
        candidate = source_root.joinpath(*relative.parts)
        if not candidate.exists():
            continue
        _require(not candidate.is_symlink(), f"Source payload is symlinked: {candidate}")
        resolved = candidate.resolve(strict=True)
        _require(
            resolved.is_file() and (resolved == source_root or source_root in resolved.parents),
            f"Source payload escapes source root: {candidate}",
        )
        observed = _sha256(resolved)
        size = resolved.stat().st_size
        if observed == expected:
            candidates.append((resolved, observed, size))
    _require(candidates, f"No source root contains the pinned payload: {relative}")
    resolved, observed, size = candidates[0]
    declared_bytes = int(record["bytes"]) if record.get("bytes") is not None else None
    bytes_corrected = declared_bytes is not None and size != declared_bytes
    return relative, resolved, observed, size, declared_bytes, bytes_corrected


def materialize(
    *,
    binding_path: Path,
    binding_sha256: str,
    portable_lineage_path: Path,
    portable_lineage_sha256: str,
    manifest_root_text: str,
    source_root_paths: list[Path],
    output_root_path: Path,
    server_artifact_root: str,
) -> dict[str, Any]:
    binding = _load_pinned(binding_path.resolve(strict=True), binding_sha256)
    lineage = _load_pinned(
        portable_lineage_path.resolve(strict=True), portable_lineage_sha256
    )
    manifest_root = PurePosixPath(manifest_root_text)
    _require(manifest_root.is_absolute() and ".." not in manifest_root.parts, "Unsafe manifest root")
    _require(source_root_paths, "At least one source root is required")
    source_roots = tuple(path.resolve(strict=True) for path in source_root_paths)
    _require(
        len(set(source_roots)) == len(source_roots)
        and all(path.is_dir() and not path.is_symlink() for path in source_roots),
        "Source roots are unsafe or duplicated",
    )
    output_root = output_root_path.resolve()
    _require(not output_root.exists(), f"Output root already exists: {output_root}")
    server_root = PurePosixPath(server_artifact_root)
    _require(server_root.is_absolute() and ".." not in server_root.parts, "Unsafe server artifact root")

    release_records = _records(binding)
    _require(len(release_records) == EXPECTED_RELEASE_RECORDS, "Release record count drifted")
    upstream_record = binding.get("sources", {}).get("audited_gene_set_release_manifest")
    _require(isinstance(upstream_record, Mapping), "Upstream manifest record is missing")
    _, upstream_source, _, _, _, upstream_source_bytes_corrected = _source(
        upstream_record,
        manifest_root=manifest_root,
        source_roots=source_roots,
    )
    _require(
        upstream_source_bytes_corrected,
        "Expected the single explicit upstream-manifest bytes correction",
    )
    upstream = json.loads(upstream_source.read_text(encoding="utf-8"))
    _require(isinstance(upstream, dict), "Upstream manifest is malformed")
    upstream_records = _records(upstream)
    _require(len(upstream_records) == EXPECTED_UPSTREAM_RECORDS, "Upstream record count drifted")
    _require(
        len({(str(item.get("path")), str(item.get("sha256"))) for item in upstream_records})
        == EXPECTED_UPSTREAM_UNIQUE_RECORDS,
        "Upstream unique record count drifted",
    )

    inventory: dict[str, dict[str, Any]] = {}
    byte_corrections: list[dict[str, Any]] = []
    for origin, records in (("release", release_records), ("upstream", upstream_records)):
        for record in records:
            relative, source, observed, size, declared_bytes, bytes_corrected = _source(
                record,
                manifest_root=manifest_root,
                source_roots=source_roots,
            )
            key = relative.as_posix()
            if bytes_corrected:
                byte_corrections.append(
                    {
                        "origin": origin,
                        "relative_path": key,
                        "declared_bytes": declared_bytes,
                        "actual_bytes": size,
                        "sha256": observed,
                    }
                )
            prior = inventory.get(key)
            if prior is not None:
                _require(
                    prior["sha256"] == observed and prior["bytes"] == size,
                    f"Conflicting duplicate payload declaration: {key}",
                )
                prior["origins"] = sorted(set(prior["origins"] + [origin]))
                continue
            inventory[key] = {
                "source": source,
                "sha256": observed,
                "bytes": size,
                "origins": [origin],
            }

    byte_corrections = sorted(
        (
            {
                **item,
                "logical_id": f"{item['origin']}:{item['relative_path']}",
                "sha_match": True,
                "correction_reason": CORRECTION_REASON,
            }
            for item in byte_corrections
        ),
        key=lambda item: item["logical_id"],
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
    observed_correction_core = tuple(
        sorted(
            (
                item["origin"],
                item["relative_path"],
                item["declared_bytes"],
                item["actual_bytes"],
                item["sha256"],
            )
            for item in byte_corrections
        )
    )
    observed_core_json = json.dumps(
        observed_correction_core, ensure_ascii=False, separators=(",", ":")
    )
    expected_core_json = json.dumps(
        tuple(sorted(EXPECTED_BYTE_CORRECTIONS)),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    _require(
        hashlib.sha256(observed_core_json.encode("utf-8")).hexdigest()
        == hashlib.sha256(expected_core_json.encode("utf-8")).hexdigest(),
        "Mirror bytes corrections differ from the frozen explicit manifest: "
        + json.dumps(
            {
                "observed_only": [
                    repr(item)
                    for item in set(observed_correction_core)
                    - set(EXPECTED_BYTE_CORRECTIONS)
                ],
                "expected_only": [
                    repr(item)
                    for item in set(EXPECTED_BYTE_CORRECTIONS)
                    - set(observed_correction_core)
                ],
                "observed_sha": hashlib.sha256(
                    observed_core_json.encode("utf-8")
                ).hexdigest(),
                "expected_sha": hashlib.sha256(
                    expected_core_json.encode("utf-8")
                ).hexdigest(),
                "observed_length": len(observed_core_json),
                "expected_length": len(expected_core_json),
            },
            sort_keys=True,
        ),
    )
    rewrite_records = lineage.get("json_rewrites")
    _require(isinstance(rewrite_records, list), "Portable lineage lacks json_rewrites")
    rewrite_evidence = []
    for relative in sorted({item["relative_path"] for item in byte_corrections}):
        target = str(manifest_root.joinpath(*PurePosixPath(relative).parts))
        expected_sha = next(
            item["sha256"] for item in byte_corrections if item["relative_path"] == relative
        )
        matched = [
            item
            for item in rewrite_records
            if item.get("target_path") == target
            and item.get("rebound_sha256") == expected_sha
            and item.get("source_sha256") != expected_sha
        ]
        _require(len(matched) == 1, f"Portable lineage did not explain byte correction: {relative}")
        rewrite_evidence.append(
            {
                "relative_path": relative,
                "target_path": target,
                "source_sha256": matched[0]["source_sha256"],
                "rebound_sha256": matched[0]["rebound_sha256"],
                "rewrite_count": matched[0].get("rewrite_count"),
            }
        )

    output_root.mkdir(parents=True, exist_ok=False)
    try:
        for relative, declaration in sorted(inventory.items()):
            target = output_root.joinpath(*PurePosixPath(relative).parts)
            _require(not target.exists(), f"Refusing to overwrite mirror payload: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(declaration["source"], target, follow_symlinks=False)
            _require(
                not target.is_symlink()
                and target.resolve().is_file()
                and output_root in target.resolve().parents,
                f"Materialized payload escaped output root: {target}",
            )
            _require(
                _sha256(target) == declaration["sha256"]
                and target.stat().st_size == declaration["bytes"],
                f"Materialized payload verification failed: {relative}",
            )
    except Exception:
        shutil.rmtree(output_root, ignore_errors=True)
        raise

    rows = [
        {
            "relative_path": relative,
            "server_path": str(server_root.joinpath(*PurePosixPath(relative).parts)),
            "sha256": declaration["sha256"],
            "bytes": declaration["bytes"],
            "origins": declaration["origins"],
        }
        for relative, declaration in sorted(inventory.items())
    ]
    inventory_sha = hashlib.sha256(
        "\n".join(
            f"{item['relative_path']}\t{item['sha256']}\t{item['bytes']}"
            for item in rows
        ).encode("utf-8")
    ).hexdigest()
    correction_manifest_sha = hashlib.sha256(
        (
            json.dumps(
                byte_corrections,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()
    return {
        "format": FORMAT,
        "status": "PASS",
        "binding": {
            "path": str(binding_path.resolve()),
            "sha256": str(binding_sha256).lower(),
        },
        "portable_lineage": {
            "path": str(portable_lineage_path.resolve()),
            "sha256": str(portable_lineage_sha256).lower(),
        },
        "manifest_root": str(manifest_root),
        "source_roots": [str(path) for path in source_roots],
        "local_artifact_root": str(output_root),
        "server_artifact_root": str(server_root),
        "release_records_validated": len(release_records),
        "upstream_records_validated": len(upstream_records),
        "upstream_unique_records": EXPECTED_UPSTREAM_UNIQUE_RECORDS,
        "unique_payload_files": len(rows),
        "payload_bytes": sum(item["bytes"] for item in rows),
        "payload_inventory_sha256": inventory_sha,
        "metadata_byte_corrections": byte_corrections,
        "metadata_byte_correction_declarations": len(byte_corrections),
        "metadata_byte_correction_unique_files": len(rewrite_evidence),
        "metadata_byte_correction_manifest_sha256": correction_manifest_sha,
        "portable_rebinding_lineage_evidence": rewrite_evidence,
        "release_records_exact_bytes": EXPECTED_RELEASE_RECORDS - 5,
        "release_records_explicit_bytes_metadata_corrections": 5,
        "upstream_records_exact_bytes": EXPECTED_UPSTREAM_RECORDS - 6,
        "upstream_records_explicit_bytes_metadata_corrections": 6,
        "payloads": rows,
        "manifest_root_and_artifact_root_separately_bound": True,
        "containment_gate_passed": True,
        "sha256_gate_passed": True,
        "bytes_gate_passed_after_frozen_explicit_metadata_correction_manifest": True,
        "expected_sha256_changed": False,
        "silent_relaxation": False,
        "negative_gates": {
            "hash_mismatch_accepted_as_bytes_correction": False,
            "unexpected_bytes_mismatch_accepted": False,
            "path_escape_accepted": False,
            "symlink_accepted": False,
            "original_manifest_modified": False,
            "expected_sha256_changed": False,
        },
        "symlinks_permitted": False,
        "overwrite_permitted": False,
        "scientific_table_rows_read": 0,
        "scientific_tables_read": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--binding-sha256", required=True)
    parser.add_argument("--portable-lineage", type=Path, required=True)
    parser.add_argument("--portable-lineage-sha256", required=True)
    parser.add_argument("--manifest-root", required=True)
    parser.add_argument("--source-root", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--server-artifact-root", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.receipt.exists(), f"Receipt already exists: {args.receipt}")
    payload = materialize(
        binding_path=args.binding,
        binding_sha256=args.binding_sha256,
        portable_lineage_path=args.portable_lineage,
        portable_lineage_sha256=args.portable_lineage_sha256,
        manifest_root_text=args.manifest_root,
        source_root_paths=args.source_root,
        output_root_path=args.output_root,
        server_artifact_root=args.server_artifact_root,
    )
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
