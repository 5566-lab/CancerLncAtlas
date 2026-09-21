#!/usr/bin/env python3
"""Freeze the one-file 17-cancer r7 supervisor transfer payload."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from scripts.transfer_v32_payloads_to_local_gpu import (  # noqa: E402
    COPY_FORMAT,
    _canonical_json,
    build_plan,
    load_copy_manifest,
    sha256_file,
)


TARGET_ROOT = (
    "./data/CancerLncAtlas/runtime/tools/"
    "single_cell_r7_17c_supervisor_20260829_r1"
)
RELATIVE = "scripts/run_v32_single_cell_r7_17c_cohort.py"
EXPECTED_BYTES = 29_487
EXPECTED_SHA256 = (
    "c50bebed211306f9a17115720e067133b85d09ad3bcc0dfa29a0dfe4edae7996"
)
FROZEN_RUNNER_MANIFEST_SHA256 = (
    "06090c75a4c3a722c5be99c3ba6f3ae205c0905dbf023a68201c80d162fbf8d1"
)


class SupervisorBridgeError(RuntimeError):
    pass


def _exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def materialize(output_root: Path) -> dict[str, object]:
    output = output_root.resolve()
    if output.exists() or output.is_symlink():
        raise SupervisorBridgeError(f"bridge reuse is forbidden: {output}")
    source = (ROOT / RELATIVE).resolve()
    if (
        source.is_symlink()
        or not source.is_file()
        or source.stat().st_size != EXPECTED_BYTES
        or sha256_file(source) != EXPECTED_SHA256
    ):
        raise SupervisorBridgeError("supervisor source drift")
    manifest = {
        "format": COPY_FORMAT,
        "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
        "run_id": "v32-single-cell-r7-17c-supervisor-20260829-r1",
        "target_root": TARGET_ROOT,
        "payloads_are_byte_identical": True,
        "symlinks_permitted": False,
        "overwrite_permitted": False,
        "entry_count": 1,
        "total_bytes": EXPECTED_BYTES,
        "entries": [
            {
                "kind": "file",
                "source_path": str(source),
                "target_path": f"{TARGET_ROOT}/{RELATIVE}",
                "bytes": EXPECTED_BYTES,
                "sha256": EXPECTED_SHA256,
            }
        ],
        "upload_started": False,
        "production_deployed": False,
    }
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise SupervisorBridgeError(f"unsafe temporary exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        manifest_path = temporary / "COPY_MANIFEST.json"
        _exclusive(manifest_path, _canonical_json(manifest))
        loaded, manifest_sha = load_copy_manifest(manifest_path)
        plan = build_plan(loaded, manifest_sha256=manifest_sha)
        if plan["entry_count"] != 1 or plan["total_bytes"] != EXPECTED_BYTES:
            raise SupervisorBridgeError("transfer plan drift")
        validation = {
            "format": "CC_HHGT_V3_2_SINGLE_CELL_R7_17C_SUPERVISOR_BRIDGE_V1",
            "copy_manifest_sha256": manifest_sha,
            "target_root": TARGET_ROOT,
            "source_sha256": EXPECTED_SHA256,
            "frozen_runner_manifest_sha256": FROZEN_RUNNER_MANIFEST_SHA256,
            "formal_cancers": 17,
            "serial_execution": True,
            "memory_limit_bytes": 512 * 1024**2,
            "raw_h5_full_sha_first": True,
            "typed_failure_stops_cohort": True,
            "production_deployed": False,
            "port_8260_touched": False,
        }
        _exclusive(temporary / "VALIDATION.json", _canonical_json(validation))
        os.rename(temporary, output)
        return validation
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    print(
        json.dumps(
            materialize(parser.parse_args().output_root),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
