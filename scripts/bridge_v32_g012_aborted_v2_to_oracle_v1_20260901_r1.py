#!/usr/bin/env python3
"""Preserve the sealed V2 abort receipt and publish a V1 compatibility view."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path("./data/CancerLncAtlas/runtime/bootstrap/v32_g012_paid_gpu_20260831_r1")
PUBLIC = ROOT / "ABORTED.json"
EXPECTED_V2_SHA = "82cb30819badd548c4962e7b03e3d646dce66939a7486250d62bb41661a85da3"
ARCHIVE = ROOT / f"ABORTED.V2_SOURCE_{EXPECTED_V2_SHA}.json"


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def main() -> int:
    raw = PUBLIC.read_bytes()
    if digest(raw) != EXPECTED_V2_SHA:
        raise SystemExit("ABORT_COMPAT_SOURCE_SHA_DRIFT")
    payload = json.loads(raw.decode("utf-8"))
    if payload.get("format") != "CC_HHGT_V3_2_G012_ABORTED_RUN_V2":
        raise SystemExit("ABORT_COMPAT_SOURCE_FORMAT_DRIFT")
    if ARCHIVE.exists():
        raise SystemExit("ABORT_COMPAT_ARCHIVE_ALREADY_EXISTS")
    payload["format"] = "CC_HHGT_V3_2_G012_ABORTED_RUN_V1"
    payload["compatibility_bridge"] = {
        "reason": "ORACLE_R2_VALIDATOR_EXPECTS_ABORT_V1_WHILE_SEALER_EMITTED_V2",
        "source_format": "CC_HHGT_V3_2_G012_ABORTED_RUN_V2",
        "source_sha256": EXPECTED_V2_SHA,
        "source_preserved_path": str(ARCHIVE),
        "source_preserved": True,
    }
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    partial = ROOT / ".ABORTED.v1.compat.partial"
    if partial.exists():
        raise SystemExit("ABORT_COMPAT_PARTIAL_PRESENT")
    descriptor = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(PUBLIC, ARCHIVE)
    os.replace(partial, PUBLIC)
    print(
        json.dumps(
            {
                "status": "ABORT_V1_COMPATIBILITY_VIEW_READY",
                "source_v2_sha256": EXPECTED_V2_SHA,
                "source_preserved_path": str(ARCHIVE),
                "compatibility_sha256": digest(encoded),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
