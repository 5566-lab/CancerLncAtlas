#!/usr/bin/env python3
"""Freeze the one-file SARC r6 memory-gate supervisor."""
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
    "single_cell_r7_r6_sarc_gate_20260829_r2"
)
RELATIVE = "scripts/run_v32_single_cell_r7_r6_sarc_gate.py"
EXPECTED_BYTES = 23_740
EXPECTED_SHA256 = (
    "81c8c7b4ba85f23da1da09768dffbce3234b6f7a692dbcce39d6193ce76672e8"
)
FROZEN_RUNNER_MANIFEST_SHA256 = (
    "910c52e615df271672a6bf4360c5acfbcdc2ba4d66604e768b62e68fafbf913a"
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
        raise SupervisorBridgeError("SARC gate supervisor source drift")
    manifest = {
        "format": COPY_FORMAT,
        "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
        "run_id": "v32-single-cell-r7-r6-sarc-gate-20260829-r2",
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
            raise SupervisorBridgeError("SARC gate transfer plan drift")
        validation = {
            "format": "CC_HHGT_V3_2_SINGLE_CELL_R7_R6_SARC_GATE_BRIDGE_V2",
            "copy_manifest_sha256": manifest_sha,
            "target_root": TARGET_ROOT,
            "source_sha256": EXPECTED_SHA256,
            "frozen_runner_manifest_sha256": FROZEN_RUNNER_MANIFEST_SHA256,
            "cancer_id": "SARC",
            "remaining_cancers_started": False,
            "serial_execution": True,
            "memory_limit_bytes": 512 * 1024**2,
            "memory_limit_increased": False,
            "memory_metric": "MAX_CHILD_PID_VMRSS_OR_VMHWM_NOT_PROCESS_GROUP_SUM",
            "poll_seconds": 0.25,
            "pinned_python_symlink_allowed": True,
            "pinned_python_resolved_target": (
                "${PRIVATE_WORK_ROOT}/miniconda3/bin/python3.13"
            ),
            "pinned_python_sha256": (
                "dcb43d1acbc001b6ca88ed41f47273d53b4658766875852ac1ede3003bba9329"
            ),
            "pinned_python_stat_bound": True,
            "raw_h5_full_sha_first": True,
            "typed_failure_stops_run": True,
            "hnsc_overwrite_permitted": False,
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
