#!/usr/bin/env python3
"""Build one compressed G0/G1/G2 patient-fold transfer bundle.

The corrected prepared payloads are intentionally monolithic PyTorch files.
This utility does not rewrite their serialization or delete the authority.
It packages exactly one patient fold (the three graph variants) with zstd so
that a GPU host can stage/train one fold at a time instead of requiring the
full ~170-GiB tree on its boot disk.

This is a server-149-only operation.  The resulting archive is a transport
artifact; training must still validate the member list and hashes before use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


VARIANTS = ("G0", "G1", "G2")
FORMAT = "CANCERLNCATLAS_V32_G012_FOLD_ZSTD_BUNDLE_V1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--host", default="149")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    actual_host = socket.gethostname()
    if actual_host != args.host:
        raise SystemExit(
            f"refusing non-server execution: expected host {args.host!r}, "
            f"observed {actual_host!r}"
        )
    source = args.prepared_root.resolve()
    output = args.output_root.resolve()
    if not source.is_dir():
        raise SystemExit(f"prepared root is not a directory: {source}")
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f"G012_PATIENT_FOLD_{args.fold}.tar.zst"
    manifest = output / f"G012_PATIENT_FOLD_{args.fold}.json"
    if archive.exists() or manifest.exists():
        raise SystemExit("refusing to overwrite an existing bundle or manifest")

    members: list[dict[str, object]] = []
    relative_members: list[str] = []
    for variant in VARIANTS:
        relative = f"{variant}/PATIENT_FOLD_{args.fold}.pt"
        path = source / relative
        if not path.is_file() or path.is_symlink():
            raise SystemExit(f"source member must be a regular non-symlink file: {path}")
        relative_members.append(relative)
        members.append({
            "variant": variant,
            "relative_path": relative,
            "size_bytes": path.stat().st_size,
        })

    partial = output / f".{archive.name}.partial"
    if partial.exists():
        raise SystemExit(f"refusing to reuse stale partial archive: {partial}")
    command = [
        "tar",
        "--zstd",
        "-cf",
        str(partial),
        "-C",
        str(source),
        *relative_members,
    ]
    try:
        subprocess.run(command, check=True)
        listing = subprocess.run(
            ["tar", "--zstd", "-tf", str(partial)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        if sorted(listing) != sorted(relative_members):
            raise RuntimeError(f"archive member mismatch: {listing!r}")
        os.replace(partial, archive)
    except Exception:
        partial.unlink(missing_ok=True)
        raise

    payload = {
        "format": FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "host": actual_host,
        "prepared_root": str(source),
        "fold": args.fold,
        "variants": list(VARIANTS),
        "archive": archive.name,
        "archive_sha256": sha256_file(archive),
        "archive_size_bytes": archive.stat().st_size,
        "members": members,
        "source_is_read_only": True,
        "source_deleted": False,
        "sealed_test_read": False,
        "training_started": False,
    }
    temporary = Path(tempfile.mkstemp(prefix=f".{manifest.name}.", dir=output)[1])
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, manifest)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

