#!/usr/bin/env python3
"""Compute directory-hash variants in one physical read pass.

This is a diagnostic only: it never changes the declared hash or any payload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _aggregate(
    records: list[dict[str, object]],
    *,
    order: str,
    name_style: str,
    prefix: str = "",
) -> str:
    if order == "posix":
        ordered = sorted(records, key=lambda item: str(item["relative_path"]))
    elif order == "casefold":
        ordered = sorted(
            records, key=lambda item: str(item["relative_path"]).casefold()
        )
    else:
        raise ValueError(order)
    digest = hashlib.sha256()
    for item in ordered:
        name = prefix + str(item["relative_path"])
        if name_style == "windows":
            name = name.replace("/", "\\")
        elif name_style == "lower_posix":
            name = name.casefold()
        elif name_style != "posix":
            raise ValueError(name_style)
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(item["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"Unsafe directory: {root}")
    files = [item for item in root.rglob("*") if item.is_file()]
    records = [
        {
            "relative_path": item.relative_to(root).as_posix(),
            "sha256": _sha256(item),
            "bytes": item.stat().st_size,
        }
        for item in files
    ]
    names = [str(item["relative_path"]) for item in records]
    if len(set(name.casefold() for name in names)) != len(names):
        raise RuntimeError("Case-colliding paths")
    variants: dict[str, str] = {}
    for order in ("posix", "casefold"):
        for style in ("posix", "windows", "lower_posix"):
            variants[f"{order}_order__{style}_names"] = _aggregate(
                records, order=order, name_style=style
            )
            variants[f"{order}_order__prefixed_{style}_names"] = _aggregate(
                records,
                order=order,
                name_style=style,
                prefix=root.name + "/",
            )
    expected = str(args.expected_sha256).lower()
    print(
        json.dumps(
            {
                "format": "CANCERLNCATLAS_V32_DRUG_TREE_HASH_VARIANTS_V1",
                "root": str(root),
                "files": len(records),
                "bytes": sum(int(item["bytes"]) for item in records),
                "expected_sha256": expected,
                "matching_variants": [
                    name for name, value in variants.items() if value == expected
                ],
                "variants": variants,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
