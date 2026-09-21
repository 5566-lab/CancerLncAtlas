#!/usr/bin/env python3
"""Rebind an unchanged directional-CNV audit package into one staging root.

The source release binding and independent-audit report are copied byte for
byte.  Only the JSON paths inside the independent binding are changed from a
superseded staging root to the current isolated root; the resulting binding and
unified launcher manifest are hash-pinned before the receipt is written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(raw)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--release-source", type=Path, required=True)
    parser.add_argument("--audit-source", type=Path, required=True)
    parser.add_argument("--report-source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    root = args.staging_root.resolve()
    release_source = args.release_source.resolve()
    audit_source = args.audit_source.resolve()
    report_source = args.report_source.resolve()
    manifest_path = args.manifest.resolve()
    receipt_path = args.receipt.resolve()
    for source in (release_source, audit_source, report_source, manifest_path):
        if source.is_symlink() or not source.is_file():
            raise SystemExit(f"missing_or_unsafe_source={source}")
    if not root.is_dir() or root.is_symlink():
        raise SystemExit(f"missing_or_unsafe_staging_root={root}")

    release_dst = root / "code/artifacts/v32_server_bindings/DIRECTIONAL_CNV_WEBSITE_BINDING_PROVENANCE_20260904_r2.json"
    audit_dst = root / "code/artifacts/v32_server_bindings/INDEPENDENT_AUDIT_BINDING_PROVENANCE_20260904_r2.json"
    report_dst = root / "audit/directional_cnv_independent_20260904_r3/AUDIT.json"
    manifest_before = sha256(manifest_path)
    release_bytes = release_source.read_bytes()
    report_bytes = report_source.read_bytes()
    atomic_write(release_dst, release_bytes)
    atomic_write(report_dst, report_bytes)

    binding = load_json(audit_source)
    release_record = binding.get("release_binding")
    report_record = binding.get("report")
    if not isinstance(release_record, dict) or not isinstance(report_record, dict):
        raise SystemExit("audit_binding_missing_release_or_report")
    binding["release_binding"] = {
        **release_record,
        "path": str(release_dst),
        "sha256": sha256(release_dst),
    }
    binding["report"] = {
        **report_record,
        "path": str(report_dst),
        "sha256": sha256(report_dst),
    }
    audit_bytes = (json.dumps(binding, indent=2, sort_keys=True) + "\n").encode("utf-8")
    atomic_write(audit_dst, audit_bytes)
    audit_sha = sha256(audit_dst)

    manifest = load_json(manifest_path)
    entries = manifest.get("bindings")
    if not isinstance(entries, dict) or not isinstance(entries.get("directional_cnv_independent_audit"), dict):
        raise SystemExit("manifest_missing_directional_cnv_independent_audit")
    if not isinstance(entries.get("directional_cnv"), dict):
        raise SystemExit("manifest_missing_directional_cnv")
    release_entry = dict(entries["directional_cnv"])
    release_entry["path"] = str(release_dst.relative_to(root / "code")).replace("\\", "/")
    release_entry["sha256"] = sha256(release_dst)
    entries["directional_cnv"] = release_entry
    entry = dict(entries["directional_cnv_independent_audit"])
    entry["path"] = str(audit_dst.relative_to(root / "code")).replace("\\", "/")
    entry["sha256"] = audit_sha
    entries["directional_cnv_independent_audit"] = entry
    manifest["bindings"] = entries
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    atomic_write(manifest_path, manifest_bytes)
    manifest_after = sha256(manifest_path)

    receipt_payload = {
        "schema": "cancerlncatlas.directional_cnv_staging_rebind.v1",
        "status": "PASS_PATH_REBIND_ONLY",
        "host_expected": "149",
        "staging_root": str(root),
        "release_binding": {
            "path": str(release_dst),
            "sha256": sha256(release_dst),
            "source_path": str(release_source),
            "source_sha256": sha256(release_source),
            "bytes_unchanged": release_bytes == release_dst.read_bytes(),
        },
        "audit_binding": {
            "path": str(audit_dst),
            "sha256": audit_sha,
            "source_path": str(audit_source),
        },
        "report": {
            "path": str(report_dst),
            "sha256": sha256(report_dst),
            "source_path": str(report_source),
            "bytes_unchanged": report_bytes == report_dst.read_bytes(),
        },
        "manifest_sha256_before": manifest_before,
        "manifest_sha256_after": manifest_after,
        "data_or_prediction_recomputed": False,
        "training_started": False,
        "production_deployed": False,
    }
    atomic_write(
        receipt_path,
        (json.dumps(receipt_payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(
        json.dumps(
            {
                "audit_sha256": audit_sha,
                "manifest_sha256": manifest_after,
                "receipt": str(receipt_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
