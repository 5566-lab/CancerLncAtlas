#!/usr/bin/env python3
"""Freeze the SARC r4 supervisor bound to the r8 BH runner."""
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
    "single_cell_r7_r8_sarc_gate_20260829_r4"
)
EXPECTED = {
    "scripts/run_v32_single_cell_r7_r6_sarc_gate.py": (
        23_740,
        "81c8c7b4ba85f23da1da09768dffbce3234b6f7a692dbcce39d6193ce76672e8",
    ),
    "scripts/run_v32_single_cell_r7_r8_sarc_gate.py": (
        3_513,
        "252008337167935c68f95689d476214d8241c3b08df39be1cc33e99a0c55002b",
    ),
}
FROZEN_RUNNER_MANIFEST_SHA256 = (
    "89360f86e1c96f36c69fcf347ed752ba657bb388ebfe62fdd0500a5a852ab55f"
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
    entries = []
    for relative, (expected_bytes, expected_sha) in EXPECTED.items():
        source = (ROOT / relative).resolve()
        if (
            source.is_symlink()
            or not source.is_file()
            or source.stat().st_size != expected_bytes
            or sha256_file(source) != expected_sha
        ):
            raise SupervisorBridgeError(f"SARC r4 supervisor source drift: {source}")
        entries.append(
            {
                "kind": "file",
                "source_path": str(source),
                "target_path": f"{TARGET_ROOT}/{relative}",
                "bytes": expected_bytes,
                "sha256": expected_sha,
            }
        )
    manifest = {
        "format": COPY_FORMAT,
        "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
        "run_id": "v32-single-cell-r7-r8-sarc-gate-20260829-r4",
        "target_root": TARGET_ROOT,
        "payloads_are_byte_identical": True,
        "symlinks_permitted": False,
        "overwrite_permitted": False,
        "entry_count": len(entries),
        "total_bytes": sum(row["bytes"] for row in entries),
        "entries": entries,
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
        if plan["entry_count"] != 2 or plan["total_bytes"] != 27_253:
            raise SupervisorBridgeError("SARC r4 gate transfer plan drift")
        validation = {
            "format": "CC_HHGT_V3_2_SINGLE_CELL_R7_R8_SARC_GATE_BRIDGE_V1",
            "copy_manifest_sha256": manifest_sha,
            "target_root": TARGET_ROOT,
            "entry_count": 2,
            "total_bytes": 27_253,
            "frozen_runner_manifest_sha256": FROZEN_RUNNER_MANIFEST_SHA256,
            "cancer_id": "SARC",
            "output_root": (
                "./data/CancerLncAtlas/results/model/"
                "v32_single_cell_r7_fresh_streaming_r8_sarc_20260829_r4"
            ),
            "audit_root": (
                "./data/CancerLncAtlas/runtime/audits/"
                "single_cell_r7_r8_sarc_20260829_r4"
            ),
            "r3_partial_reuse_permitted": False,
            "remaining_cancers_started": False,
            "serial_execution": True,
            "memory_limit_bytes": 512 * 1024**2,
            "memory_limit_increased": False,
            "memory_metric": "MAX_CHILD_PID_VMRSS_OR_VMHWM_NOT_PROCESS_GROUP_SUM",
            "poll_seconds": 0.25,
            "duckdb_internal_memory_limit": "64MB",
            "duckdb_internal_memory_limit_increased": False,
            "association_bh_family": "COMPARTMENT_ORDER",
            "association_family_scope_changed": False,
            "association_total_tests_denominator_changed": False,
            "association_exact_key_unique_checked": True,
            "parquet_physical_rowid_join_used": False,
            "local_selected_tests": "38_PASSED",
            "large_spool_dual_gate_smoke": "PASS",
            "large_spool_formal_gate_value_bytes": 346_189_824,
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
            "isolated_import_and_schema_smoke_required_before_start": True,
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
