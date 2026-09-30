#!/usr/bin/env python3
"""Verify cloud code archive matches the authorized 149 training code tree."""
from __future__ import annotations

import argparse
import hashlib
import json
import socket
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.training_guard import code_tree_sha256  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if socket.gethostname() != "149" or args.output.exists():
        raise RuntimeError("Code archive verification host or output scope mismatch")
    archive = args.archive.resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="c_code_archive_verify_") as temp:
        extract = Path(temp)
        with tarfile.open(archive, "r:gz") as handle:
            handle.extractall(extract)
        actual = code_tree_sha256(extract)
        expected = code_tree_sha256(ROOT)
        if actual != expected:
            raise RuntimeError("Cloud code archive differs from 149 authorization code")
    receipt = {
        "status": "PASS_C_CODE_ARCHIVE_149",
        "host": "149", "code_archive_sha256": sha256(archive),
        "code_tree_sha256": actual,
        "archive_bytes": archive.stat().st_size,
    }
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
