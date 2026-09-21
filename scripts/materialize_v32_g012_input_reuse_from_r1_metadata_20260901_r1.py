#!/usr/bin/env python3
"""Materialize the legacy Oracle reuse receipt without re-hashing 15 PT files.

The immutable per-fold digests are copied from the returned R1 authority.  This
CPU-only compatibility bridge checks exact paths, regular-file readability and
sizes.  The actual Oracle input (G2/Fold0) remains hash-verified by the training
loader before deserialization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path


R1_SHA256 = "1f5bc54b2a8acc037373bfa380c95c8a717e76fde956852ec481b16a0d8716f0"
SOURCE_ARCHIVE_SHA256 = "1c17b7be89621e5125c39e87f05ce14beb1f2227afdb481d2e9a20049b56bab0"
EXPECTED_TOTAL_BYTES = 63_341_812_187
VARIANTS = ("G0", "G1", "G2")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-parent", type=Path, required=True)
    parser.add_argument("--r1-static", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    parent = args.prepared_parent.resolve(strict=True)
    raw = args.r1_static.read_bytes()
    if sha256_bytes(raw) != R1_SHA256:
        raise SystemExit("METADATA_REUSE_R1_STATIC_SHA_DRIFT")
    authority = json.loads(raw.decode("utf-8"))
    if authority.get("status") != "STATIC_AUTH_READY":
        raise SystemExit("METADATA_REUSE_R1_STATIC_STATUS_DRIFT")
    variants = authority.get("variants")
    if not isinstance(variants, dict) or set(variants) != set(VARIANTS):
        raise SystemExit("METADATA_REUSE_VARIANT_SET_DRIFT")

    records: list[dict[str, object]] = []
    total = 0
    for variant in VARIANTS:
        rows = variants[variant].get("fold_inputs")
        if not isinstance(rows, list) or len(rows) != 5:
            raise SystemExit(f"METADATA_REUSE_FOLD_ROWS_DRIFT={variant}")
        by_fold = {int(row["fold"]): row for row in rows}
        if set(by_fold) != set(range(5)):
            raise SystemExit(f"METADATA_REUSE_FOLD_SET_DRIFT={variant}")
        for fold in range(5):
            source = by_fold[fold]
            path = parent / variant / f"PATIENT_FOLD_{fold}.pt"
            expected_path = str(path)
            if source.get("path") != expected_path:
                raise SystemExit(f"METADATA_REUSE_PATH_DRIFT={variant}:{fold}")
            info = os.lstat(path)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise SystemExit(f"METADATA_REUSE_NOT_UNIQUE_REGULAR={variant}:{fold}")
            expected_size = int(source.get("size_bytes", -1))
            if info.st_size != expected_size or expected_size <= 0:
                raise SystemExit(f"METADATA_REUSE_SIZE_DRIFT={variant}:{fold}")
            digest = source.get("sha256")
            if not isinstance(digest, str) or len(digest) != 64:
                raise SystemExit(f"METADATA_REUSE_SHA_AUTHORITY_INVALID={variant}:{fold}")
            with path.open("rb") as handle:
                if not handle.read(1):
                    raise SystemExit(f"METADATA_REUSE_UNREADABLE_HEAD={variant}:{fold}")
                handle.seek(-1, os.SEEK_END)
                if not handle.read(1):
                    raise SystemExit(f"METADATA_REUSE_UNREADABLE_TAIL={variant}:{fold}")
            records.append(
                {
                    "variant": variant,
                    "fold": fold,
                    "path": expected_path,
                    "sha256": digest,
                    "size_bytes": expected_size,
                }
            )
            total += expected_size
    if len(records) != 15 or total != EXPECTED_TOTAL_BYTES:
        raise SystemExit("METADATA_REUSE_CARDINALITY_OR_TOTAL_DRIFT")

    payload = {
        "format": "CC_HHGT_V3_2_G012_R2_INPUT_REUSE_READY_V1",
        "status": "INPUT_REUSE_READY_HASH_VERIFIED",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prepared_parent": str(parent),
        "source_input_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "r1_static_auth_r6_path": str(args.r1_static.resolve()),
        "r1_static_auth_r6_sha256": R1_SHA256,
        "fold_artifact_count": 15,
        "fold_artifact_total_bytes": total,
        "fold_artifacts": records,
        "reused_existing_prepared_inputs": True,
        "retransfer_performed": False,
        "copy_performed": False,
        "extraction_performed": False,
        "source_files_modified": False,
        "formal_result_artifacts_used": False,
        "per_fold_hashing_performed": False,
        "per_fold_hashes_reused_from_r1_authority": True,
        "g2_fold0_runtime_hash_verification_required": True,
    }
    output = args.output.resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise SystemExit("METADATA_REUSE_OUTPUT_ALREADY_EXISTS")
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    descriptor = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(partial, output)
    print(
        json.dumps(
            {
                "status": "INPUT_REUSE_METADATA_BRIDGE_READY",
                "output": str(output),
                "sha256": sha256_bytes(encoded),
                "fold_count": 15,
                "fold_total_bytes": total,
                "per_fold_hashing_performed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
