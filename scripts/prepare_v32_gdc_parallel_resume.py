#!/usr/bin/env python3
"""Build disjoint GDC resume manifests or verify their size-complete result."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED = ("id", "filename", "md5", "size", "state")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    if any(column not in fields for column in REQUIRED):
        raise RuntimeError(f"GDC manifest lacks required columns: {REQUIRED}")
    if not rows:
        raise RuntimeError("GDC manifest is empty")
    ids = [row["id"] for row in rows]
    targets = [(row["id"], row["filename"]) for row in rows]
    if len(ids) != len(set(ids)) or len(targets) != len(set(targets)):
        raise RuntimeError("GDC manifest IDs or targets are duplicated")
    for row in rows:
        size = int(row["size"])
        if size <= 0 or not row["filename"]:
            raise RuntimeError("GDC manifest contains an invalid target or size")
    return fields, rows


def target_path(root: Path, row: dict[str, str]) -> Path:
    return root / row["id"] / row["filename"]


def size_complete(root: Path, row: dict[str, str]) -> bool:
    path = target_path(root, row)
    return path.is_file() and not path.is_symlink() and path.stat().st_size == int(row["size"])


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare(args: argparse.Namespace) -> int:
    manifest = args.manifest.resolve(strict=True)
    download_root = args.download_root.resolve(strict=True)
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"Parallel resume manifest output reuse refused: {output}")
    if not 2 <= args.shards <= 32:
        raise ValueError("--shards must be in 2..32")
    fields, rows = read_manifest(manifest)
    existing_ids = {
        entry.name
        for entry in os.scandir(download_root)
        if entry.is_dir(follow_symlinks=False)
    }
    complete = []
    remaining = []
    for row in rows:
        is_complete = row["id"] in existing_ids and size_complete(download_root, row)
        (complete if is_complete else remaining).append(row)
    if not remaining:
        raise RuntimeError("No incomplete GDC files remain; parallel resume is unnecessary")

    # Greedy byte balancing keeps the shard processes close in wall time while
    # preserving strict target disjointness.
    bins: list[list[dict[str, str]]] = [[] for _ in range(args.shards)]
    byte_totals = [0] * args.shards
    for row in sorted(remaining, key=lambda value: int(value["size"]), reverse=True):
        shard = min(range(args.shards), key=lambda index: (byte_totals[index], index))
        bins[shard].append(row)
        byte_totals[shard] += int(row["size"])

    output.mkdir(parents=True)
    shard_records = []
    assigned: set[tuple[str, str]] = set()
    for index, shard_rows in enumerate(bins):
        path = output / f"shard_{index:02d}.gdc_manifest.tsv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(shard_rows)
        keys = {(row["id"], row["filename"]) for row in shard_rows}
        if assigned & keys:
            raise RuntimeError("Parallel GDC shards overlap")
        assigned.update(keys)
        shard_records.append(
            {
                "shard": index,
                "path": str(path),
                "rows": len(shard_rows),
                "bytes": byte_totals[index],
                "sha256": sha256(path),
            }
        )
    expected = {(row["id"], row["filename"]) for row in remaining}
    if assigned != expected:
        raise RuntimeError("Parallel GDC shards do not exactly cover remaining targets")
    audit = {
        "format": "CANCERLNCATLAS_V32_GDC_METHYLATION_PARALLEL_RESUME_V1",
        "status": "READY",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "download_root": str(download_root),
        "all_rows": len(rows),
        "all_bytes": sum(int(row["size"]) for row in rows),
        "already_size_complete_rows": len(complete),
        "remaining_rows": len(remaining),
        "remaining_bytes": sum(int(row["size"]) for row in remaining),
        "shards_are_target_disjoint": True,
        "size_complete_is_not_md5_complete": True,
        "full_md5_deferred_to_feature_materializer": True,
        "shards": shard_records,
    }
    write_json(audit, output / "AUDIT.json")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


def verify(args: argparse.Namespace) -> int:
    manifest = args.manifest.resolve(strict=True)
    download_root = args.download_root.resolve(strict=True)
    receipt = args.receipt.resolve()
    if receipt.exists():
        raise FileExistsError(f"GDC size receipt reuse refused: {receipt}")
    _, rows = read_manifest(manifest)
    failures = []
    for row in rows:
        path = target_path(download_root, row)
        observed = path.stat().st_size if path.is_file() and not path.is_symlink() else None
        expected = int(row["size"])
        if observed != expected:
            failures.append(
                {
                    "id": row["id"],
                    "filename": row["filename"],
                    "expected_size": expected,
                    "observed_size": observed,
                }
            )
    if failures:
        raise RuntimeError(
            f"GDC parallel resume left {len(failures)} size-invalid files; preview={failures[:10]}"
        )
    payload = {
        "format": "CANCERLNCATLAS_V32_GDC_METHYLATION_SIZE_COMPLETE_V1",
        "status": "PASS_SIZE_COMPLETE_PENDING_FULL_MD5",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "download_root": str(download_root),
        "files": len(rows),
        "bytes": sum(int(row["size"]) for row in rows),
        "full_md5_deferred_to_feature_materializer": True,
    }
    receipt.parent.mkdir(parents=True, exist_ok=True)
    write_json(payload, receipt)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    verify_parser = subparsers.add_parser("verify")
    for child in (prepare_parser, verify_parser):
        child.add_argument("--manifest", type=Path, required=True)
        child.add_argument("--download-root", type=Path, required=True)
    prepare_parser.add_argument("--output-root", type=Path, required=True)
    prepare_parser.add_argument("--shards", type=int, default=8)
    verify_parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    return prepare(args) if args.command == "prepare" else verify(args)


if __name__ == "__main__":
    raise SystemExit(main())
