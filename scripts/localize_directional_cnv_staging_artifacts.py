#!/usr/bin/env python3
"""Localize unchanged directional-CNV sidecars into an isolated staging root.

The source parquet/JSON bytes are copied only after their pinned SHA-256 values
are checked.  Release and independent-audit bindings are then regenerated as
path-rebound staging metadata; no prediction or model computation is changed.
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
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def load_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"missing_or_unsafe={path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"json_object_required={path}")
    return value


def dump_object(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


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
    if not root.is_dir() or root.is_symlink():
        raise SystemExit(f"missing_or_unsafe_staging_root={root}")
    release = load_object(release_source)
    audit_binding = load_object(audit_source)
    report = load_object(report_source)
    manifest = load_object(manifest_path)
    artifacts = release.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise SystemExit("release_binding_missing_artifacts")

    localized: dict[str, dict[str, Any]] = {}
    artifact_root = root / "cnv_artifacts"
    for name, record in artifacts.items():
        if not isinstance(record, dict):
            raise SystemExit(f"invalid_artifact_record={name}")
        source = Path(str(record.get("path", ""))).resolve()
        expected = str(record.get("sha256", "")).lower()
        if source.is_symlink() or not source.is_file() or len(expected) != 64:
            raise SystemExit(f"missing_or_invalid_artifact={name}")
        observed = sha256(source)
        if observed != expected:
            raise SystemExit(f"source_artifact_hash_mismatch={name}")
        destination = artifact_root / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        copied = sha256(destination)
        if copied != expected:
            raise SystemExit(f"localized_artifact_hash_mismatch={name}")
        updated = dict(record)
        updated["path"] = str(destination)
        artifacts[name] = updated
        localized[name] = {
            "source_path": str(source),
            "destination_path": str(destination),
            "sha256": expected,
            "bytes": destination.stat().st_size,
        }
    release["artifacts"] = artifacts
    release["staging_revision"] = "LOCALIZED_ARTIFACTS_20260904_R1"
    release_bytes = dump_object(release)
    release_destination = root / "code/artifacts/v32_server_bindings/DIRECTIONAL_CNV_WEBSITE_BINDING_PROVENANCE_20260904_r3.json"
    atomic_write(release_destination, release_bytes)
    release_sha = sha256(release_destination)

    report["release_binding_sha256"] = release_sha
    checks = report.get("checks")
    if not isinstance(checks, dict):
        raise SystemExit("independent_report_missing_checks")
    checks["artifact_paths_localized"] = True
    checks["localized_artifact_hashes_recomputed"] = True
    report["checks"] = checks
    report["staging_rebind_only"] = True
    report["source_report_sha256"] = sha256(report_source)
    report_bytes = dump_object(report)
    report_destination = root / "audit/directional_cnv_independent_20260904_r4/AUDIT.json"
    atomic_write(report_destination, report_bytes)
    report_sha = sha256(report_destination)

    release_record = audit_binding.get("release_binding")
    report_record = audit_binding.get("report")
    if not isinstance(release_record, dict) or not isinstance(report_record, dict):
        raise SystemExit("audit_binding_missing_release_or_report")
    audit_binding["release_binding"] = {
        **release_record,
        "path": str(release_destination),
        "sha256": release_sha,
    }
    audit_binding["report"] = {
        **report_record,
        "path": str(report_destination),
        "sha256": report_sha,
    }
    audit_bytes = dump_object(audit_binding)
    audit_destination = root / "code/artifacts/v32_server_bindings/INDEPENDENT_AUDIT_BINDING_PROVENANCE_20260904_r3.json"
    atomic_write(audit_destination, audit_bytes)
    audit_sha = sha256(audit_destination)

    entries = manifest.get("bindings")
    if not isinstance(entries, dict):
        raise SystemExit("manifest_missing_bindings")
    for key in ("directional_cnv", "directional_cnv_independent_audit"):
        if not isinstance(entries.get(key), dict):
            raise SystemExit(f"manifest_missing_{key}")
    release_entry = dict(entries["directional_cnv"])
    release_entry["path"] = str(release_destination.relative_to(root / "code")).replace("\\", "/")
    release_entry["sha256"] = release_sha
    entries["directional_cnv"] = release_entry
    audit_entry = dict(entries["directional_cnv_independent_audit"])
    audit_entry["path"] = str(audit_destination.relative_to(root / "code")).replace("\\", "/")
    audit_entry["sha256"] = audit_sha
    entries["directional_cnv_independent_audit"] = audit_entry
    manifest["bindings"] = entries
    manifest["staging_revision"] = "LOCALIZED_CNV_ARTIFACTS_20260904_R1"
    manifest_before = sha256(manifest_path)
    atomic_write(manifest_path, dump_object(manifest))
    manifest_after = sha256(manifest_path)

    receipt = {
        "schema": "cancerlncatlas.directional_cnv_artifact_localization.v1",
        "status": "PASS_UNCHANGED_BYTES_REBOUND_PATHS",
        "host_expected": "149",
        "staging_root": str(root),
        "localized_artifacts": localized,
        "release_binding": {"path": str(release_destination), "sha256": release_sha},
        "audit_binding": {"path": str(audit_destination), "sha256": audit_sha},
        "report": {"path": str(report_destination), "sha256": report_sha},
        "manifest_sha256_before": manifest_before,
        "manifest_sha256_after": manifest_after,
        "prediction_or_model_recomputed": False,
        "training_started": False,
        "production_deployed": False,
    }
    atomic_write(receipt_path, dump_object(receipt))
    print(json.dumps({"release_sha256": release_sha, "audit_sha256": audit_sha, "manifest_sha256": manifest_after}, sort_keys=True))


if __name__ == "__main__":
    main()
