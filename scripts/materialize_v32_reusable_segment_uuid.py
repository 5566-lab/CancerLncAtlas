#!/usr/bin/env python3
"""Materialize hash-verified reusable GDC segment files into UUID layout.

This is a raw-input layout bridge.  It never reads predictions, rankings, or
checkpoints.  Every source is constrained to an explicitly authorized root,
verified against the frozen GDC manifest, and published with a no-replace hard
link from a same-directory temporary file.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


class ReusableSegmentError(RuntimeError):
    """Raised when reusable raw-segment materialization is unsafe."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _md5_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.md5()  # noqa: S324 - GDC manifests use MD5.
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return digest.hexdigest(), size


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _assert_real_path(path: Path, root: Path, *, file_required: bool) -> Path:
    if path.is_symlink():
        raise ReusableSegmentError(f"Symlink is forbidden: {path}")
    resolved = path.resolve(strict=True)
    if not _inside(resolved, root):
        raise ReusableSegmentError(f"Path escapes authorized root: {path}")
    cursor = resolved
    while cursor != root:
        if cursor.is_symlink():
            raise ReusableSegmentError(f"Symlink ancestor is forbidden: {cursor}")
        cursor = cursor.parent
    if file_required and not resolved.is_file():
        raise ReusableSegmentError(f"Expected a regular file: {resolved}")
    if not file_required and not resolved.is_dir():
        raise ReusableSegmentError(f"Expected a real directory: {resolved}")
    return resolved


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    data = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _validate_inputs(
    manifest_path: Path,
    retry_path: Path,
    allowed_source_root: Path,
) -> list[dict[str, Any]]:
    manifest = _read_tsv(manifest_path)
    retry = _read_tsv(retry_path)
    retry_ids = [str(row.get("id", "")) for row in retry]
    if not retry_ids or len(retry_ids) != len(set(retry_ids)):
        raise ReusableSegmentError("Retry manifest is empty or contains duplicate UUIDs")
    reusable = {
        str(row.get("file_id", "")): row
        for row in manifest
        if str(row.get("selected_for_patient", "")).lower() == "true"
        and str(row.get("existing_reusable", "")).lower() == "true"
    }
    if set(retry_ids) != set(reusable):
        missing = sorted(set(reusable) - set(retry_ids))[:5]
        extra = sorted(set(retry_ids) - set(reusable))[:5]
        raise ReusableSegmentError(
            f"Retry UUIDs are not exactly the frozen reusable set; missing={missing}, extra={extra}"
        )
    retry_by_id = {str(row["id"]): row for row in retry}
    records: list[dict[str, Any]] = []
    for file_id in retry_ids:
        if not UUID_RE.fullmatch(file_id):
            raise ReusableSegmentError(f"Invalid GDC UUID: {file_id}")
        row = reusable[file_id]
        retry_row = retry_by_id[file_id]
        file_name = str(row.get("file_name", ""))
        if not file_name or Path(file_name).name != file_name or "\\" in file_name:
            raise ReusableSegmentError(f"Unsafe GDC file name: {file_name!r}")
        if str(retry_row.get("filename", "")) != file_name:
            raise ReusableSegmentError(f"Retry filename drift for {file_id}")
        expected_md5 = str(row.get("md5sum", "")).lower()
        expected_size = int(row.get("file_size", "-1"))
        if str(retry_row.get("md5", "")).lower() != expected_md5:
            raise ReusableSegmentError(f"Retry MD5 drift for {file_id}")
        if int(retry_row.get("size", "-1")) != expected_size:
            raise ReusableSegmentError(f"Retry size drift for {file_id}")
        source_raw = str(row.get("existing_path", ""))
        source_candidate = Path(source_raw)
        if os.name == "nt":
            if not source_candidate.is_absolute():
                raise ReusableSegmentError(f"Reusable source is not absolute: {file_id}")
        elif not source_raw.startswith("/"):
            raise ReusableSegmentError(f"Reusable source is not an absolute POSIX path: {file_id}")
        source = _assert_real_path(source_candidate, allowed_source_root, file_required=True)
        observed_md5, observed_size = _md5_and_size(source)
        if observed_size != expected_size or observed_md5 != expected_md5:
            raise ReusableSegmentError(f"Reusable source size/MD5 mismatch: {file_id}")
        records.append(
            {
                "file_id": file_id,
                "cancer_id": str(row.get("cancer_id", "")).upper(),
                "file_name": file_name,
                "source_path": str(source),
                "size": expected_size,
                "md5": expected_md5,
            }
        )
    return records


def materialize_reusable_segments(
    *,
    manifest_path: str | Path,
    retry_path: str | Path,
    allowed_source_root: str | Path,
    download_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    manifest_file = Path(manifest_path).resolve(strict=True)
    retry_file = Path(retry_path).resolve(strict=True)
    source_root = Path(allowed_source_root).resolve(strict=True)
    if source_root.is_symlink() or not source_root.is_dir():
        raise ReusableSegmentError("Authorized source root must be a real directory")
    target_root = Path(download_root).resolve(strict=True)
    if target_root.is_symlink() or not target_root.is_dir():
        raise ReusableSegmentError("Download root must be a real existing directory")
    audit_root = Path(output_root).resolve()
    if audit_root.exists():
        raise ReusableSegmentError(f"Audit output root already exists: {audit_root}")

    # Validate the complete set before writing a single raw target.
    records = _validate_inputs(manifest_file, retry_file, source_root)
    audit_root.mkdir(parents=True)
    published = 0
    already_verified = 0
    completed: list[dict[str, Any]] = []
    try:
        for record in records:
            directory = target_root / record["file_id"]
            if directory.exists():
                if directory.is_symlink() or not directory.is_dir():
                    raise ReusableSegmentError(f"Unsafe UUID target directory: {directory}")
            else:
                directory.mkdir()
            target = directory / record["file_name"]
            if target.exists() or target.is_symlink():
                if target.is_symlink() or not target.is_file():
                    raise ReusableSegmentError(f"Unsafe existing target: {target}")
                observed_md5, observed_size = _md5_and_size(target)
                if observed_size != record["size"] or observed_md5 != record["md5"]:
                    raise ReusableSegmentError(f"Refusing mismatched existing target: {target}")
                status = "ALREADY_VERIFIED"
                already_verified += 1
            else:
                partial = directory / f".{record['file_name']}.reuse-{uuid.uuid4().hex}.partial"
                digest = hashlib.md5()  # noqa: S324 - GDC manifests use MD5.
                size = 0
                try:
                    with Path(record["source_path"]).open("rb") as source, partial.open("xb") as sink:
                        for chunk in iter(lambda: source.read(1024 * 1024), b""):
                            sink.write(chunk)
                            size += len(chunk)
                            digest.update(chunk)
                        sink.flush()
                        os.fsync(sink.fileno())
                    if size != record["size"] or digest.hexdigest() != record["md5"]:
                        raise ReusableSegmentError(f"Copied bytes failed size/MD5: {record['file_id']}")
                    try:
                        os.link(partial, target)
                        status = "PUBLISHED_NO_REPLACE"
                        published += 1
                    except FileExistsError:
                        observed_md5, observed_size = _md5_and_size(target)
                        if observed_size != record["size"] or observed_md5 != record["md5"]:
                            raise ReusableSegmentError(f"Concurrent target conflict: {target}")
                        status = "CONCURRENT_ALREADY_VERIFIED"
                        already_verified += 1
                finally:
                    partial.unlink(missing_ok=True)
            completed.append({**record, "target_path": str(target), "status": status})
    except Exception as exc:
        _write_json(
            audit_root / "INCOMPLETE.json",
            {
                "status": "INCOMPLETE",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "records_validated_before_write": len(records),
                "records_completed": len(completed),
                "published": published,
                "already_verified": already_verified,
            },
        )
        raise

    inventory = hashlib.sha256()
    for row in sorted(completed, key=lambda item: item["file_id"]):
        inventory.update(row["file_id"].encode("ascii"))
        inventory.update(b"\0")
        inventory.update(row["md5"].encode("ascii"))
        inventory.update(b"\0")
        inventory.update(str(row["size"]).encode("ascii"))
        inventory.update(b"\n")
    payload = {
        "status": "PASS",
        "format": "CC_HHGT_V3_2_REUSABLE_GDC_SEGMENT_UUID_MATERIALIZATION_V1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_file),
        "manifest_sha256": _sha256(manifest_file),
        "retry_manifest_path": str(retry_file),
        "retry_manifest_sha256": _sha256(retry_file),
        "allowed_source_root": str(source_root),
        "download_root": str(target_root),
        "records": len(completed),
        "published": published,
        "already_verified": already_verified,
        "total_bytes": sum(int(row["size"]) for row in completed),
        "inventory_sha256": inventory.hexdigest(),
        "all_sources_verified_before_write": True,
        "no_replace_publish": True,
        "symlinks_permitted": False,
        "old_predictions_read": False,
        "old_rankings_read": False,
        "old_checkpoints_read": False,
        "records_detail": completed,
    }
    _write_json(audit_root / "SUCCESS.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-tsv", required=True)
    parser.add_argument("--retry-manifest", required=True)
    parser.add_argument("--allowed-source-root", required=True)
    parser.add_argument("--download-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    result = materialize_reusable_segments(
        manifest_path=args.manifest_tsv,
        retry_path=args.retry_manifest,
        allowed_source_root=args.allowed_source_root,
        download_root=args.download_root,
        output_root=args.output_root,
    )
    print(
        json.dumps(
            {key: result[key] for key in ("status", "records", "published", "already_verified", "total_bytes")},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
