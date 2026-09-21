#!/usr/bin/env python3
"""Freeze the complete r7 streaming runner as a production copy manifest."""
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
    "single_cell_r7_streaming_20260829_r5"
)
SUPERSEDED_MANIFEST_SHA256 = (
    "ad173a711effadaf8313ebcf101b2914c7328d00e44b2e3574ed28502f2aa991"
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
        24_761,
        "d3fdea24178d9da0b72b5f7f8c17905f621abc5d349bc7f344af95604b7309d8",
    ),
    "scripts/run_v32_single_cell_r7_streaming.py": (
        51_877,
        "8d34a2202c1be25dff3295969b2dd2f823b8fe57f52b8b717852cc8030fa4e6b",
    ),
}


class StreamingToolBridgeError(RuntimeError):
    """Raised when the immutable streaming tool bridge drifts."""


def _exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


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
        "run_id": "v32-single-cell-r7-streaming-tool-20260829-r5",
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
        if plan["entry_count"] != 6 or plan["total_bytes"] != 97_875:
            raise StreamingToolBridgeError("production transfer plan lost a tool leaf")
        validation = {
            "format": "CANCERLNCATLAS_V32_SINGLE_CELL_R7_STREAMING_TOOL_VALIDATION_V1",
            "copy_manifest_sha256": manifest_sha,
            "target_root": TARGET_ROOT,
            "entry_count": 6,
            "total_bytes": 97_875,
            "supersedes_copy_manifest_sha256": SUPERSEDED_MANIFEST_SHA256,
            "supersession_reason": (
                "r4 HNSC pilot succeeded and its numeric/hash audit passed; r5 adds "
                "warning-free perfect-correlation statistics, explicit conservative-"
                "BH semantics, testability source lineage, raw-count integrality, "
                "dataset-ID gates, and an interpreter-inclusive memory budget"
            ),
            "production_loader": "load_copy_manifest",
            "production_planner": "build_plan",
            "no_cell_level_pathway_matrix": True,
            "donor_is_biological_replicate": True,
            "per_cancer_atomic_publish": True,
            "resume_checkpoint_is_sufficient_statistics_only": True,
            "historical_r5_r6_derived_inputs_permitted": False,
            "raw_h5_full_sha_gate_before_matrix_access": True,
            "resource_estimate_includes_runtime_baseline": True,
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
