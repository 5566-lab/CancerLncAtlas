#!/usr/bin/env python3
"""Create the immutable copy contract for the portable single-cell r7 bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from cc_hhgt.v32.input_lineage import artifact_sha256  # noqa: E402


DESTINATION_ROOT = (
    "./data/CancerLncAtlas/runtime/"
    "single_cell_cell_level_r7_portable_supersession_20260829"
)
MANIFEST_NAME = "COPY_MANIFEST.json"


class CopyManifestError(RuntimeError):
    """Raised when a copy contract would be ambiguous or overwrite state."""


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _assert_no_symlinks(root: Path) -> None:
    if root.is_symlink():
        raise CopyManifestError(f"Bundle root cannot be a symlink: {root}")
    for directory, names, files in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in [*names, *files]:
            path = base / name
            if path.is_symlink():
                raise CopyManifestError(f"Symlink is forbidden in bundle: {path}")


def materialize_copy_manifest(bundle_root: str | Path) -> dict[str, Any]:
    root = Path(bundle_root).resolve()
    output = root / MANIFEST_NAME
    if not root.is_dir():
        raise CopyManifestError(f"Bundle root is absent: {root}")
    if output.exists() or output.is_symlink():
        raise CopyManifestError(f"Copy manifest reuse/overwrite is forbidden: {output}")
    _assert_no_symlinks(root)

    run_path = root / "RUN_STATUS.json"
    if not run_path.is_file():
        raise CopyManifestError("RUN_STATUS.json is missing")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if run.get("run_id") != "v32-single-cell-cell-level-r7-portable-20260829":
        raise CopyManifestError(f"Unexpected r7 run_id: {run.get('run_id')}")

    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != MANIFEST_NAME
    )
    if not files:
        raise CopyManifestError("Portable bundle contains no files")
    entries: list[dict[str, Any]] = []
    for path in files:
        if path.is_symlink():
            raise CopyManifestError(f"Symlink is forbidden in bundle: {path}")
        relative = path.relative_to(root).as_posix()
        if relative.startswith("../") or relative == "..":
            raise CopyManifestError(f"Escaping relative path: {relative}")
        destination = f"{DESTINATION_ROOT}/{relative}"
        if not destination.startswith(DESTINATION_ROOT + "/"):
            raise CopyManifestError(f"Escaping destination path: {destination}")
        entries.append(
            {
                "relative_path": relative,
                "source_path": str(path),
                "destination_path": destination,
                "bytes": int(path.stat().st_size),
                "sha256": artifact_sha256(path),
                "source_is_symlink": False,
                "destination_must_not_exist": True,
                "overwrite_allowed": False,
                "copy_mode": "CREATE_NEW_THEN_SHA_VERIFY_THEN_RENAME_IF_ABSENT",
            }
        )

    payload = {
        "format": "CC_HHGT_V3_2_SINGLE_CELL_R7_PORTABLE_COPY_MANIFEST_V1",
        "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
        "run_id": run["run_id"],
        "source_root": str(root),
        "destination_root": DESTINATION_ROOT,
        "file_count": len(entries),
        "total_bytes": sum(entry["bytes"] for entry in entries),
        "files": entries,
        "policy": {
            "symlinks_allowed": False,
            "destination_overwrite_allowed": False,
            "existing_destination_action": "FAIL_CLOSED",
            "partial_suffix": ".partial",
            "partial_file_must_be_created_exclusively": True,
            "verify_bytes_before_rename": True,
            "verify_sha256_before_rename": True,
            "rename_requires_destination_absent": True,
            "copy_manifest_self_included": False,
            "copy_manifest_self_hash_must_be_supplied_out_of_band": True,
        },
        "historical_assets_relabelled_fresh": False,
        "heavy_recompute_started": False,
        "release_ready": False,
        "production_deployed": False,
    }
    payload["manifest_contract_sha256"] = _canonical_sha256(payload)
    # Exclusive creation is intentional: this manifest is never overwritten.
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", required=True, type=Path)
    return parser


def main() -> int:
    result = materialize_copy_manifest(build_parser().parse_args().bundle_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
