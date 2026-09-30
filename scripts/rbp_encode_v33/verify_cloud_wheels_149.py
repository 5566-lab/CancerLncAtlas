#!/usr/bin/env python3
"""Check staged cloud wheels byte-for-byte against official PyPI digests."""
from __future__ import annotations

import argparse
import hashlib
import json
import socket
import urllib.request
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wheel-dir", type=Path, required=True)
    p.add_argument("--requirements", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    args = p.parse_args()
    if socket.gethostname() != "149":
        raise RuntimeError("Cloud wheel verification must run on host 149")
    if args.manifest.exists():
        raise FileExistsError("Wheel verification manifest already exists")
    wheels = sorted(args.wheel_dir.glob("*.whl"))
    if not wheels:
        raise RuntimeError("Cloud wheel directory is empty")
    required = {
        line.split("==", 1)[0].strip().lower().replace("-", "_")
        for line in args.requirements.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    required.add("torch_geometric")
    present = {wheel.name.split("-", 1)[0].lower() for wheel in wheels}
    if missing := sorted(required - present):
        raise RuntimeError(f"Cloud wheelhouse is incomplete: {missing}")
    if "torch" in present or any(name.startswith("nvidia_") for name in present):
        raise RuntimeError("Cloud wheelhouse must not replace image Torch/CUDA")
    cache: dict[tuple[str, str], dict] = {}
    rows = []
    for wheel in wheels:
        name, version = wheel.name.split("-", 2)[:2]
        key = (name, version)
        if key not in cache:
            url = f"https://pypi.org/pypi/{name.replace('_', '-')}/{version}/json"
            with urllib.request.urlopen(url, timeout=30) as response:
                cache[key] = json.load(response)
        published = next(
            (row["digests"]["sha256"] for row in cache[key]["urls"]
             if row["filename"] == wheel.name), None,
        )
        actual = sha256(wheel)
        if published is None or actual != published:
            raise RuntimeError(f"Official PyPI SHA256 mismatch: {wheel.name}")
        rows.append((wheel.name, actual, wheel.stat().st_size))
    args.manifest.write_text(
        "filename\tsha256\tbytes\n" +
        "".join(f"{name}\t{digest}\t{size}\n" for name, digest, size in rows),
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS_OFFICIAL_PYPI_WHEEL_DIGESTS",
                      "host": "149", "wheel_count": len(rows),
                      "total_bytes": sum(row[2] for row in rows),
                      "manifest_sha256": sha256(args.manifest)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
