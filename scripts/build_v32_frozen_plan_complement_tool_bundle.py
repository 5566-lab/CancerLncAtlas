#!/usr/bin/env python3
"""Build the immutable, content-addressed server bundle for complement audit."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "v32_frozen_plan_complement_audit_tool_bundle_20260829_r1"
REMOTE_TOOLS = PurePosixPath("./data/CancerLncAtlas/runtime/tools")
REMOTE_AUDIT = PurePosixPath(
    "./data/CancerLncAtlas/runtime/audits/"
    "v32_frozen_plan_complement_20260829_r1/AUDIT.json"
)
FULL_PLAN = PurePosixPath(
    "./data/CancerLncAtlas/runtime/web_candidates/v32_20260829_r3/v32/"
    "_transfer/control/"
    "09f2ae4c810a392b0718f0e2d3e02bc6e5487783a9a0d2644a240141d1705669/"
    "df99598d986e49b289801d93b2301ada/"
    "plan.2b57f4116f6d7915a9aa0ac616b8420a2ebc65e4db3415b9540dc5aa26b0db58.json"
)
REMAINING_PLAN = PurePosixPath(
    "./data/CancerLncAtlas/runtime/web_candidates/v32_20260829_r3/v32/"
    "_transfer/control/"
    "c291558fe128d7c7d651072570eef96592f45b9fc45eceae0896d26b12daefaa/"
    "27aba79c8cb94014b940b9f5d69ea0e6/"
    "plan.4d87aebcb33a6976bf8ff99212e78bb940b908d32059d8378c6a3512dede6897.json"
)
TARGET_ROOT = PurePosixPath(
    "./data/CancerLncAtlas/runtime/web_candidates/v32_20260829_r3/v32"
)
ALLOWED_ROOT = PurePosixPath("./data/CancerLncAtlas")


def canonical(payload: dict) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def record(path: Path, *, relative: str) -> dict:
    return {"relative_path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)}


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite bundle: {OUTPUT}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{OUTPUT.name}.", dir=OUTPUT.parent))
    try:
        payload_dir = staging / "payload"
        payload_dir.mkdir()
        tool_name = "v32_frozen_plan_complement_audit.py"
        test_name = "test_v32_frozen_plan_complement_audit.py"
        shutil.copyfile(ROOT / "scripts" / tool_name, payload_dir / tool_name)
        shutil.copyfile(ROOT / "tests" / test_name, payload_dir / test_name)

        run_argv = [
            "python3",
            "${BUNDLE_ROOT}/" + tool_name,
            "--full-plan",
            str(FULL_PLAN),
            "--full-plan-sha256",
            "2b57f4116f6d7915a9aa0ac616b8420a2ebc65e4db3415b9540dc5aa26b0db58",
            "--remaining-plan",
            str(REMAINING_PLAN),
            "--remaining-plan-sha256",
            "4d87aebcb33a6976bf8ff99212e78bb940b908d32059d8378c6a3512dede6897",
            "--target-root",
            str(TARGET_ROOT),
            "--allowed-root",
            str(ALLOWED_ROOT),
            "--expected-full-count",
            "1562",
            "--expected-full-bytes",
            "11365150641",
            "--expected-remaining-count",
            "759",
            "--expected-remaining-bytes",
            "11282384395",
            "--expected-complement-count",
            "803",
            "--expected-complement-bytes",
            "82766246",
            "--output",
            str(REMOTE_AUDIT),
        ]
        run_spec = {
            "format": "CANCERLNCATLAS_V32_FROZEN_COMPLEMENT_RUN_SPEC_V1",
            "argv": run_argv,
            "expected": {
                "status": "PASS",
                "complement_count": 803,
                "complement_bytes": 82766246,
                "remaining_targets_rehashed_count": 0,
                "conflicts": 0,
            },
            "constraints": {
                "server": "COMPUTE_HOST",
                "standard_library_only": True,
                "atomic_no_replace_audit": True,
                "production_port_8260_touched": False,
                "services_started_or_restarted": False,
            },
        }
        write_new(payload_dir / "RUN_SPEC.json", canonical(run_spec))
        primary = [
            record(payload_dir / tool_name, relative=tool_name),
            record(payload_dir / test_name, relative=test_name),
            record(payload_dir / "RUN_SPEC.json", relative="RUN_SPEC.json"),
        ]
        content_sha256 = hashlib.sha256(canonical({"entries": primary})).hexdigest()
        bundle_id = f"v32_frozen_plan_complement_audit_20260829_r1_{content_sha256[:16]}"
        remote_root = REMOTE_TOOLS / bundle_id
        bundle_manifest = {
            "format": "CANCERLNCATLAS_V32_IMMUTABLE_TOOL_BUNDLE_V1",
            "bundle_id": bundle_id,
            "bundle_content_sha256": content_sha256,
            "immutable": True,
            "overwrite_permitted": False,
            "entries": primary,
            "remote_target_root": str(remote_root),
        }
        write_new(payload_dir / "BUNDLE_MANIFEST.json", canonical(bundle_manifest))

        upload_rows = []
        for name in (tool_name, test_name, "RUN_SPEC.json", "BUNDLE_MANIFEST.json"):
            source = payload_dir / name
            upload_rows.append(
                {
                    "kind": "file",
                    # The directory is atomically renamed after construction;
                    # record the final path, never the temporary staging path.
                    "source_path": str((OUTPUT / "payload" / name).resolve(strict=False)),
                    "target_path": str(remote_root / name),
                    "target_relative_path": name,
                    "bytes": source.stat().st_size,
                    "sha256": sha256(source),
                    "exists": True,
                    "file_count": 1,
                }
            )
        copy_manifest = {
            "format": "CANCERLNCATLAS_V32_PORTABLE_COPY_MANIFEST_V1",
            # Keep transfer-control metadata in runtime/tools/_transfer rather
            # than inside the immutable content-addressed bundle directory.
            "target_root": str(REMOTE_TOOLS),
            "overwrite_permitted": False,
            "symlinks_permitted": False,
            "payloads_are_byte_identical": True,
            "entries": upload_rows,
        }
        copy_path = staging / "REMOTE_COPY_MANIFEST.json"
        write_new(copy_path, canonical(copy_manifest))
        success = {
            "format": "CANCERLNCATLAS_V32_FROZEN_COMPLEMENT_TOOL_BUNDLE_BUILD_V1",
            "status": "SUCCESS",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "bundle_id": bundle_id,
            "bundle_content_sha256": content_sha256,
            "bundle_manifest_sha256": sha256(payload_dir / "BUNDLE_MANIFEST.json"),
            "copy_manifest_sha256": sha256(copy_path),
            "file_count": len(upload_rows),
            "total_bytes": sum(row["bytes"] for row in upload_rows),
            "remote_target_root": str(remote_root),
            "remote_audit_output": str(REMOTE_AUDIT),
        }
        write_new(staging / "SUCCESS.json", canonical(success))
        staging.rename(OUTPUT)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(success, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
