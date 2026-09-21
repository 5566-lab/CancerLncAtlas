#!/usr/bin/env python3
"""Materialize a portable payload manifest without overwrites or symlinks."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path, PurePosixPath

from cc_hhgt.v32.portable_rebinding import PortableRebindingError, payload_summary


def _target(record: dict, declared_root: PurePosixPath, actual_root: Path) -> Path:
    raw = PurePosixPath(record["target_path"])
    try:
        relative = raw.relative_to(declared_root)
    except ValueError as exc:
        raise PortableRebindingError(f"Copy target escapes candidate root: {raw}") from exc
    return actual_root.joinpath(*relative.parts)


def _copy_file(source: Path, target: Path, expected_sha: str) -> None:
    if source.is_symlink() or not source.is_file():
        raise PortableRebindingError(f"Copy source is not a regular file: {source}")
    if target.exists():
        summary = payload_summary(target)
        if summary.get("sha256") != expected_sha:
            raise PortableRebindingError(f"Refusing to overwrite mismatched target: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target, follow_symlinks=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--copy-manifest", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.copy_manifest.read_text(encoding="utf-8"))
    if manifest.get("format") != "CANCERLNCATLAS_V32_PORTABLE_COPY_MANIFEST_V1":
        raise PortableRebindingError("Copy manifest format mismatch")
    if manifest.get("symlinks_permitted") is not False or manifest.get("overwrite_permitted") is not False:
        raise PortableRebindingError("Unsafe copy policy")
    declared_root = PurePosixPath(manifest["target_root"])
    actual_root = args.target_root.resolve()
    actual_root.mkdir(parents=True, exist_ok=True)
    for record in manifest.get("entries", []):
        source = Path(record["source_path"])
        target = _target(record, declared_root, actual_root)
        if record["kind"] == "file":
            _copy_file(source, target, record["sha256"])
        else:
            if source.is_symlink() or not source.is_dir():
                raise PortableRebindingError(f"Copy source is not a real directory: {source}")
            for item in sorted(source.rglob("*"), key=lambda value: value.as_posix()):
                if item.is_symlink():
                    raise PortableRebindingError(f"Source tree contains a symlink: {item}")
                if item.is_file():
                    _copy_file(item, target / item.relative_to(source), payload_summary(item)["sha256"])
        observed = payload_summary(target)
        expected_key = "sha256" if record["kind"] == "file" else "tree_sha256"
        if observed.get(expected_key) != record.get(expected_key):
            raise PortableRebindingError(f"Post-copy hash mismatch: {target}")
    print(json.dumps({"status": "PASS", "entries": len(manifest.get("entries", []))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
