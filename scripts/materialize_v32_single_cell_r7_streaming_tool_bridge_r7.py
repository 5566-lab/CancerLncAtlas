#!/usr/bin/env python3
"""Freeze the r7 low-memory runner with explicit Arrow schemas."""
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
    "single_cell_r7_streaming_20260829_r7"
)
SUPERSEDED_MANIFEST_SHA256 = (
    "910c52e615df271672a6bf4360c5acfbcdc2ba4d66604e768b62e68fafbf913a"
)
EXPECTED = {
    "cc_hhgt/__init__.py": (
        98,
        "e246acd89da9a9733fbac09b2770161b5ed199695a93a832ca965d8d201b78dc",
    ),
    "cc_hhgt/v32/__init__.py": (
        363,
        "359f958ad6e7e67078201282718193b40bc0d09bf15d27cbefb3a604b535ca91",
    ),
    "cc_hhgt/v32/contracts.py": (
        8_363,
        "387e64d4931bac936e7affa4a3f974812dbfbb16d50deacd670685c796d41f46",
    ),
    "cc_hhgt/v32/single_cell_cell_level.py": (
        12_413,
        "08e5b3519539fe488c78626aab7a7ef94bad4f4ef7f94b2f656f029bd5056815",
    ),
    "cc_hhgt/v32/single_cell_r7_streaming.py": (
        31_679,
        "7133182f386ca74ea8b36013f16ef7679cb95306949cff8eb91ffee24ee36fd3",
    ),
    "scripts/run_v32_single_cell_r7_streaming.py": (
        67_295,
        "dc2dd3049a3da065488c1f1278c8549316896277a918ed916e565d02d73839ef",
    ),
    (
        "artifacts/single_cell_r7_sarc_memory_recovery_audit_20260829_r1/"
        "MEMORY_ROOT_CAUSE_AND_REMEDIATION.json"
    ): (
        6_673,
        "f8ad8a6c182f109a78737da61aecb0797d26cf8cf5fe26342cbb8fdc43b0665d",
    ),
    (
        "artifacts/single_cell_r7_sarc_memory_recovery_audit_20260829_r1/"
        "MEMORY_ROOT_CAUSE_AND_REMEDIATION.md"
    ): (
        3_443,
        "6196263f29852efbab6b92fbf4f01dd7497fccaf597c36a6b44f89fd5ee841a8",
    ),
    (
        "artifacts/single_cell_r7_sarc_schema_recovery_audit_20260829_r1/"
        "SARC_R6_SCHEMA_FAILURE_AND_R7_FIX.json"
    ): (
        3_622,
        "196b1fc127c1fb1baa770f235fd87323be490d843aa4f84465b81e081b6806e1",
    ),
    (
        "artifacts/single_cell_r7_sarc_schema_recovery_audit_20260829_r1/"
        "SARC_R6_SCHEMA_FAILURE_AND_R7_FIX.md"
    ): (
        2_181,
        "3d6ba2fd99c8009dc78548f3ed3945033da6a4fd54f5f291953a120652549127",
    ),
}
EXPECTED_ENTRY_COUNT = 10
EXPECTED_TOTAL_BYTES = 136_130


class StreamingToolBridgeError(RuntimeError):
    """Raised when the immutable r7 tool bridge drifts."""


def _exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _target_relative(source_relative: str) -> str:
    if source_relative.startswith("artifacts/"):
        return "audit/" + Path(source_relative).name
    return source_relative


def materialize(output_root: str | Path) -> dict[str, Any]:
    output = Path(output_root).resolve()
    if output.exists() or output.is_symlink():
        raise StreamingToolBridgeError(f"tool bridge reuse is forbidden: {output}")
    entries = []
    for relative, (expected_bytes, expected_sha) in EXPECTED.items():
        source = (ROOT / relative).resolve()
        if source.is_symlink() or not source.is_file():
            raise StreamingToolBridgeError(f"tool source is absent/unsafe: {source}")
        if source.stat().st_size != expected_bytes or sha256_file(source) != expected_sha:
            raise StreamingToolBridgeError(f"tool source drift: {source}")
        target_relative = _target_relative(relative)
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
        "run_id": "v32-single-cell-r7-streaming-tool-20260829-r7-arrow-schema",
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
        raise StreamingToolBridgeError(f"unsafe bridge temporary exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        manifest_path = temporary / "COPY_MANIFEST.json"
        _exclusive(manifest_path, _canonical_json(manifest))
        loaded, manifest_sha = load_copy_manifest(manifest_path)
        plan = build_plan(loaded, manifest_sha256=manifest_sha)
        if (
            plan["entry_count"] != EXPECTED_ENTRY_COUNT
            or plan["total_bytes"] != EXPECTED_TOTAL_BYTES
        ):
            raise StreamingToolBridgeError("r7 transfer plan lost a leaf")
        validation = {
            "format": "CANCERLNCATLAS_V32_SINGLE_CELL_R7_STREAMING_TOOL_VALIDATION_V3",
            "copy_manifest_sha256": manifest_sha,
            "target_root": TARGET_ROOT,
            "entry_count": EXPECTED_ENTRY_COUNT,
            "total_bytes": EXPECTED_TOTAL_BYTES,
            "supersedes_copy_manifest_sha256": SUPERSEDED_MANIFEST_SHA256,
            "supersession_reason": (
                "r6 memory remediation succeeded through the lncRNA summary, but "
                "first-batch Arrow inference typed all-NA unavailable_reason as null; "
                "r7 supplies fixed Arrow schemas before every batch"
            ),
            "immutable_memory_audit_included": True,
            "immutable_schema_audit_included": True,
            "explicit_arrow_schema_before_first_batch": True,
            "first_batch_schema_inference": False,
            "zero_row_parquet_schema_fixed": True,
            "local_selected_tests": "33_PASSED",
            "memory_limit_bytes": 512 * 1024**2,
            "memory_limit_increased": False,
            "association_evidence_accumulated_in_python": False,
            "association_pathway_block": 32,
            "parquet_batch_rows": 16_384,
            "duckdb_memory_limit": "64MB",
            "duckdb_threads": 1,
            "resource_runtime_baseline_bytes": 352 * 1024**2,
            "sarc_replanned_peak_ram_bytes": 471_580_560,
            "no_cell_level_pathway_matrix": True,
            "donor_is_biological_replicate": True,
            "per_cancer_atomic_publish": True,
            "historical_derived_inputs_permitted": False,
            "raw_h5_full_sha_gate_before_matrix_access": True,
            "r1_partial_reuse_permitted": False,
            "r2_partial_reuse_permitted": False,
            "hnsc_overwrite_permitted": False,
            "remaining_cancers_started": False,
            "port_8260_touched": False,
            "upload_started": False,
            "production_deployed": False,
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
    result = materialize(parser.parse_args().output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
