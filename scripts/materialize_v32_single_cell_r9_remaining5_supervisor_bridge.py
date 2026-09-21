#!/usr/bin/env python3
"""Freeze the r9 remaining-five supervisor and its read-only binding smoke."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
from typing import Any


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
    "single_cell_r7_streaming_r9_supervisor_20260829_r1"
)
EXPECTED = {
    "scripts/run_v32_single_cell_r7_r8_remaining15_cohort.py": (
        "scripts/run_v32_single_cell_r9_remaining5_cohort.py",
        37_916,
        "a9755e128b8feab5ed81ae50762d1eb2ca2556c4a65e231226e180e9e61cb71e",
    ),
    "scripts/smoke_v32_single_cell_r9_remaining5_bindings.py": (
        "scripts/smoke_v32_single_cell_r9_remaining5_bindings.py",
        1_770,
        "1ef2f4ec5f33f39970b307420d39c2f4ed0d1d55289dfc9713462d56b6d7672b",
    ),
}


class SupervisorBridgeError(RuntimeError):
    """Raised when a supervisor bridge leaf drifts."""


def exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def materialize(output_root: Path) -> dict[str, Any]:
    output = output_root.resolve()
    if output.exists() or output.is_symlink():
        raise SupervisorBridgeError(f"bridge output reuse forbidden: {output}")
    entries: list[dict[str, Any]] = []
    for relative, (target_relative, expected_bytes, expected_sha) in EXPECTED.items():
        source = (ROOT / relative).resolve()
        if source.is_symlink() or not source.is_file():
            raise SupervisorBridgeError(f"absent/unsafe source: {source}")
        if source.stat().st_size != expected_bytes or sha256_file(source) != expected_sha:
            raise SupervisorBridgeError(f"source drift: {source}")
        entries.append(
            {
                "kind": "file",
                "source_path": str(source),
                "target_path": f"{TARGET_ROOT}/{target_relative}",
                "bytes": expected_bytes,
                "sha256": expected_sha,
            }
        )
    manifest = {
        "format": COPY_FORMAT,
        "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
        "run_id": "v32-single-cell-r9-remaining5-supervisor-20260829-r1",
        "target_root": TARGET_ROOT,
        "payloads_are_byte_identical": True,
        "symlinks_permitted": False,
        "overwrite_permitted": False,
        "entry_count": len(entries),
        "total_bytes": sum(int(row["bytes"]) for row in entries),
        "entries": entries,
        "upload_started": False,
        "production_deployed": False,
    }
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise SupervisorBridgeError(f"temporary path exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        manifest_path = temporary / "COPY_MANIFEST.json"
        exclusive(manifest_path, _canonical_json(manifest))
        loaded, manifest_sha = load_copy_manifest(manifest_path)
        plan = build_plan(loaded, manifest_sha256=manifest_sha)
        if plan["entry_count"] != 2 or plan["total_bytes"] != 39_686:
            raise SupervisorBridgeError("supervisor transfer plan lost a leaf")
        validation = {
            "format": "CANCERLNCATLAS_V32_SINGLE_CELL_R9_REMAINING5_SUPERVISOR_VALIDATION_V1",
            "copy_manifest_sha256": manifest_sha,
            "target_root": TARGET_ROOT,
            "entry_count": 2,
            "total_bytes": 39_686,
            "formal_runner_tool_manifest_sha256": (
                "7b226f9fa09627ceead790718186d489e483cf20ad75d77fe2d5b18e7c37e572"
            ),
            "runtime_binding_sha256": (
                "0287998ee55e3a4a1d1381f4e1994aca75ea58d0c1bd2e810645fb523b3a6a92"
            ),
            "runtime_binding_contract_sha256": (
                "291759551ec7d2788cf2fab671f6b4a9fe92917b0c13a47d45541728115ff99a"
            ),
            "remaining_execution_order": ["ACC", "UCEC", "READ", "GBM", "PCPG"],
            "candidate_equivalence_outputs_reused": False,
            "process_group_rss_measured": True,
            "member_vmhwm_guard_measured": True,
            "thread_pins_required": True,
            "formal_run_started": False,
            "upload_started": False,
            "production_deployed": False,
        }
        exclusive(temporary / "VALIDATION.json", _canonical_json(validation))
        os.rename(temporary, output)
        return validation
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    result = materialize(parser.parse_args().output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
