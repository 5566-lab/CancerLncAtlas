#!/usr/bin/env python3
"""Extract only compiled Pandas wheel members for a zipimport overlay."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


FORMAT = "CANCERLNCATLAS_V32_PANDAS_WHEEL_EXTENSION_OVERLAY_V1"
ALLOWED_ROOT = Path("./data/CancerLncAtlas")
WHEEL_NAME = "pandas-2.2.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
WHEEL_SHA256 = "86976a1c5b25ae3f8ccae3a5306e443569ee3c3faf444dfd0f41cda24667ad57"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists(), f"Refusing to overwrite: {path}")
    encoded = (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(encoded)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--overlay-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    wheel = args.wheel.resolve(strict=True)
    overlay = args.overlay_root.absolute()
    receipt = args.receipt.absolute()
    require(ALLOWED_ROOT in wheel.parents and ALLOWED_ROOT in overlay.parents and ALLOWED_ROOT in receipt.parents, "Path escaped ${PRIVATE_WORK_ROOT} authority")
    require(wheel.name == WHEEL_NAME and wheel.is_file() and not wheel.is_symlink(), "Unexpected Pandas wheel")
    require(sha256_file(wheel) == WHEEL_SHA256, "Pandas wheel SHA drifted")
    require(not overlay.exists() and not receipt.exists(), "Overlay or receipt already exists")

    with zipfile.ZipFile(wheel) as archive:
        selected = []
        seen: set[str] = set()
        for info in archive.infolist():
            name = info.filename
            include = (
                (name.startswith("pandas/") and name.endswith(".so"))
                or (name.startswith("pandas.libs/") and not info.is_dir())
            )
            if not include:
                continue
            pure = PurePosixPath(name)
            require(not pure.is_absolute() and ".." not in pure.parts, f"Unsafe wheel member: {name}")
            require(name not in seen, f"Duplicate selected wheel member: {name}")
            seen.add(name)
            selected.append(info)
        require(
            selected
            and all(info.filename.startswith("pandas/") for info in selected)
            and all(info.filename.endswith(".so") for info in selected),
            "Compiled overlay selection is incomplete",
        )

        overlay.mkdir(parents=True, exist_ok=False)
        records = []
        for info in selected:
            destination = overlay.joinpath(*PurePosixPath(info.filename).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            require(not destination.exists(), f"Destination collision: {destination}")
            digest = hashlib.sha256()
            total = 0
            with archive.open(info) as source, destination.open("xb") as target:
                while chunk := source.read(8 * 1024 * 1024):
                    target.write(chunk)
                    digest.update(chunk)
                    total += len(chunk)
            require(total == info.file_size, f"Extracted byte mismatch: {info.filename}")
            records.append({"member": info.filename, "path": str(destination), "bytes": total, "sha256": digest.hexdigest()})

    inventory_lines = [f"{row['member']}\t{row['bytes']}\t{row['sha256']}" for row in sorted(records, key=lambda item: item["member"])]
    payload = {
        "format": FORMAT,
        "status": "PASS",
        "wheel": {"path": str(wheel), "bytes": wheel.stat().st_size, "sha256": WHEEL_SHA256},
        "overlay_root": str(overlay),
        "selected_member_count": len(records),
        "selected_bytes": sum(row["bytes"] for row in records),
        "inventory_sha256": hashlib.sha256(("\n".join(inventory_lines) + "\n").encode("utf-8")).hexdigest(),
        "records": records,
        "pure_python_loaded_directly_from_hash_pinned_wheel": True,
        "main_score_changed": False,
        "production_port_8260_touched": False,
        "production_deployed": False,
        "release_ready": False,
    }
    write_exclusive(receipt, payload)
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
