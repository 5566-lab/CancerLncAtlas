#!/usr/bin/env python3
"""Delete only the pinned raw HM450 download after formal closure passes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_PARENT = Path(
    "./data/CancerLncAtlas/inputs/v32_distal_regulatory_mutation_20260830_r1"
)
EXPECTED_NAME = "gdc_methylation_beta_download_20260830_r1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--closure-audit", type=Path, required=True)
    parser.add_argument("--methylation-audit", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    closure_path = args.closure_audit.resolve(strict=True)
    methylation_path = args.methylation_audit.resolve(strict=True)
    closure = json.loads(closure_path.read_text(encoding="utf-8"))
    methylation = json.loads(methylation_path.read_text(encoding="utf-8"))
    if closure.get("status") != "PASS" or closure.get("raw_methylation_delete_authorized") is not True:
        raise RuntimeError("Closure does not authorize raw methylation deletion")
    if methylation.get("status") != "PASS_PATIENT_LNCRNA_METHYLATION_FEATURES":
        raise RuntimeError("Methylation feature audit is not PASS")
    if methylation.get("raw_download_delete_ready") is not True:
        raise RuntimeError("Methylation feature audit does not permit deletion")

    raw = args.raw_root.resolve(strict=True)
    expected_parent = EXPECTED_PARENT.resolve(strict=True)
    if raw.parent != expected_parent or raw.name != EXPECTED_NAME:
        raise RuntimeError(f"Raw deletion target outside pinned scope: {raw}")
    if raw.is_symlink() or not raw.is_dir():
        raise RuntimeError("Raw deletion target is not a real directory")
    declared = {
        str(Path(str(closure.get("raw_methylation_delete_target", ""))).resolve()),
        str(Path(str(methylation.get("raw_download_delete_target", ""))).resolve()),
    }
    if declared != {str(raw)}:
        raise RuntimeError(f"Raw deletion target declaration drift: {declared} != {raw}")
    receipt = args.receipt.resolve()
    if receipt.exists():
        raise FileExistsError(f"Deletion receipt reuse refused: {receipt}")
    receipt.parent.mkdir(parents=True, exist_ok=True)

    files = 0
    bytes_deleted = 0
    for root, _, names in os.walk(raw):
        for name in names:
            path = Path(root) / name
            if path.is_symlink():
                raise RuntimeError(f"Symlink inside raw deletion tree: {path}")
            files += 1
            bytes_deleted += path.stat().st_size
    expected_files = int(methylation.get("download_files_md5_verified", -1))
    if files != expected_files:
        raise RuntimeError(f"Raw inventory count drift: {files} != {expected_files}")
    expected_bytes = int(methylation.get("download_bytes_md5_verified", -1))
    if bytes_deleted != expected_bytes:
        raise RuntimeError(f"Raw inventory bytes drift: {bytes_deleted} != {expected_bytes}")

    shutil.rmtree(raw)
    if raw.exists():
        raise RuntimeError("Pinned raw methylation directory still exists after deletion")
    payload = {
        "format": "CANCERLNCATLAS_V32_RAW_METHYLATION_DELETION_RECEIPT_V1",
        "status": "PASS_DELETED",
        "deleted_at_utc": datetime.now(timezone.utc).isoformat(),
        "deleted_target": str(raw),
        "deleted_files": files,
        "deleted_bytes": bytes_deleted,
        "recoverable": False,
        "closure_audit": {"path": str(closure_path), "sha256": sha256(closure_path)},
        "methylation_audit": {"path": str(methylation_path), "sha256": sha256(methylation_path)},
        "retained_feature_root": str(methylation_path.parent),
    }
    temporary = receipt.with_name(f".{receipt.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, receipt)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
