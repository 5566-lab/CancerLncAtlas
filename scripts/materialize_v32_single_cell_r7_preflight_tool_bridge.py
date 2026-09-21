#!/usr/bin/env python3
"""Create the two-file production transfer manifest for the r7 preflight tool."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
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
    "single_cell_r7_preflight_20260829_r3"
)
PREFLIGHT_RELATIVE = Path("scripts/preflight_v32_single_cell_r7_portable_server.py")
BRIDGE_RELATIVE = Path(
    "artifacts/single_cell_cell_level_r7_transfer_bridge_20260829_r2/"
    "COPY_MANIFEST.json"
)
EXPECTED = {
    "preflight_v32_single_cell_r7_portable_server.py": {
        "sha256": "78782cf439f77eb1409ada774a989a8600bd48e583409db6956e70139d50e441",
        "bytes": 27_271,
    },
    "COPY_MANIFEST.json": {
        "sha256": "b6a08c197317d48e090d80d6e201a61c0dc0eb0f4d29a88e10535f433f3c9ecd",
        "bytes": 6_169,
    },
}


class ToolBridgeError(RuntimeError):
    """Raised when the standalone-tool bridge cannot prove byte identity."""


def _exclusive(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def materialize_tool_bridge(output_root: str | Path) -> dict[str, Any]:
    output = Path(output_root).resolve()
    if output.exists() or output.is_symlink():
        raise ToolBridgeError(f"Tool bridge reuse/overwrite is forbidden: {output}")
    source_pairs = (
        (
            ROOT / PREFLIGHT_RELATIVE,
            "preflight_v32_single_cell_r7_portable_server.py",
        ),
        (ROOT / BRIDGE_RELATIVE, "COPY_MANIFEST.json"),
    )
    entries: list[dict[str, Any]] = []
    for source, target_name in source_pairs:
        source = source.resolve()
        expected = EXPECTED[target_name]
        if source.is_symlink() or not source.is_file():
            raise ToolBridgeError(f"Tool source is missing/unsafe: {source}")
        if source.stat().st_size != expected["bytes"]:
            raise ToolBridgeError(f"Tool source byte-size drift: {source}")
        observed_sha = sha256_file(source)
        if observed_sha != expected["sha256"]:
            raise ToolBridgeError(f"Tool source SHA drift: {source}")
        entries.append(
            {
                "kind": "file",
                "source_path": str(source),
                "target_path": f"{TARGET_ROOT}/{target_name}",
                "bytes": expected["bytes"],
                "sha256": expected["sha256"],
            }
        )
    manifest = {
        "format": COPY_FORMAT,
        "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
        "run_id": "v32-single-cell-r7-preflight-tool-20260829-r3",
        "target_root": TARGET_ROOT,
        "payloads_are_byte_identical": True,
        "symlinks_permitted": False,
        "overwrite_permitted": False,
        "entry_count": len(entries),
        "total_bytes": sum(entry["bytes"] for entry in entries),
        "entries": entries,
        "upload_started": False,
        "production_deployed": False,
    }
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise ToolBridgeError(f"Unsafe pre-existing temporary: {temporary}")
    temporary.mkdir(parents=True)
    try:
        manifest_path = temporary / "COPY_MANIFEST.json"
        _exclusive(manifest_path, _canonical_json(manifest))
        loaded, manifest_sha = load_copy_manifest(manifest_path)
        plan = build_plan(loaded, manifest_sha256=manifest_sha)
        if plan["entry_count"] != 2 or plan["total_bytes"] != 33_440:
            raise ToolBridgeError("Production transfer plan lost a tool payload")
        validation = {
            "format": "CANCERLNCATLAS_V32_SINGLE_CELL_R7_PREFLIGHT_TOOL_BRIDGE_VALIDATION_V1",
            "copy_manifest_sha256": manifest_sha,
            "production_loader": "load_copy_manifest",
            "production_planner": "build_plan",
            "target_root": TARGET_ROOT,
            "entry_count": plan["entry_count"],
            "total_bytes": plan["total_bytes"],
            "standalone_preflight": True,
            "remote_r7_bridge_manifest_included": True,
            "supersedes_tool_bridge_copy_manifest_sha256": (
                "212923c8959f9b06bb2fe143011b90bbdf8af261892eb5ff07e6b2eaf9d7a701"
            ),
            "supersession_reason": (
                "R2_REQUIRED_TRANSFER_RESIDUE_INODE_IDENTITY;R3_VERIFIES_EXACT_NAME_"
                "SIZE_SHA_AND_REPORTS_INDEPENDENT_STORAGE"
            ),
            "payloads_are_byte_identical": True,
            "symlinks_permitted": False,
            "overwrite_permitted": False,
            "upload_started": False,
            "production_deployed": False,
        }
        _exclusive(
            temporary / "TOOL_BRIDGE_VALIDATION.json", _canonical_json(validation)
        )
        os.rename(temporary, output)
        return validation
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main() -> int:
    result = materialize_tool_bridge(build_parser().parse_args().output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
