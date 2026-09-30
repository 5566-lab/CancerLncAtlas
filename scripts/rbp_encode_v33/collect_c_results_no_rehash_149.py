#!/usr/bin/env python3
"""Return one C-fold result tree to host 149 using exact file sizes only."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import stat
import subprocess
from pathlib import Path, PurePosixPath


def ssh_args(host: str, port: int, key: Path) -> list[str]:
    return [
        "ssh", "-i", str(key), "-p", str(port), "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=20",
        "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3",
        f"root@{host}",
    ]


def safe_endpoint(value: str) -> bool:
    parts = value.split(".")
    return len(parts) == 4 and all(
        part.isdigit() and 0 <= int(part) <= 255 for part in parts
    )


def remote(host: str, port: int, key: Path, command: str) -> str:
    done = subprocess.run(
        [*ssh_args(host, port, key), command], capture_output=True, text=True,
        timeout=300, check=True,
    )
    return done.stdout


def parse_manifest(text: str) -> dict[str, int]:
    rows: dict[str, int] = {}
    for line in text.splitlines():
        if not line:
            continue
        relative, raw_size = line.split("\t", 1)
        if (not relative or relative.startswith("/") or ".." in PurePosixPath(relative).parts
                or not re.fullmatch(r"[A-Za-z0-9_.|/-]+", relative)):
            raise RuntimeError(f"Unsafe result path: {relative!r}")
        if relative in rows:
            raise RuntimeError("Duplicate result path")
        rows[relative] = int(raw_size)
    if not rows:
        raise RuntimeError("Cloud result tree is empty")
    return rows


def local_manifest(root: Path) -> dict[str, int]:
    return {
        path.relative_to(root).as_posix(): path.stat().st_size
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--ssh-host", required=True)
    parser.add_argument("--ssh-port", type=int, required=True)
    parser.add_argument("--identity-file", type=Path, required=True)
    parser.add_argument("--cloud-root", type=PurePosixPath, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    if socket.gethostname() != "149":
        raise RuntimeError("Result return must execute on host 149")
    if not safe_endpoint(args.ssh_host) or not 1 <= args.ssh_port <= 65535:
        raise RuntimeError("Invalid CompShare SSH endpoint")
    key = args.identity_file.resolve(strict=True)
    if stat.S_IMODE(key.stat().st_mode) & 0o077:
        raise RuntimeError("Dedicated SSH identity file is not private")
    expected_root = PurePosixPath(
        f"/root/CancerLncAtlas_C_V100S_20260927/fold_{args.fold}"
    )
    if args.cloud_root != expected_root:
        raise RuntimeError("Cloud root differs from the authorized fold root")
    destination = args.destination.resolve()
    receipt = args.receipt.resolve()
    if destination.exists() or receipt.exists():
        raise FileExistsError("Refusing to overwrite returned results or receipt")
    destination.parent.mkdir(parents=True, exist_ok=True)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    transfer = destination.parent / f".{destination.name}.transfer"
    if transfer.exists():
        raise FileExistsError(f"Incomplete transfer directory exists: {transfer}")

    cloud_results = args.cloud_root / "results"
    command = (
        f"test -d {cloud_results} && cd {cloud_results} && "
        "find . -type f -printf '%P\\t%s\\n' | sort"
    )
    expected = parse_manifest(remote(args.ssh_host, args.ssh_port, key, command))
    transfer.mkdir()
    try:
        subprocess.run([
            "scp", "-r", "-i", str(key), "-P", str(args.ssh_port),
            "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=3",
            f"root@{args.ssh_host}:{cloud_results}", str(transfer),
        ], timeout=7200, check=True)
        received = transfer / "results"
        observed = local_manifest(received)
        if observed != expected:
            raise RuntimeError("Returned result file paths or byte sizes differ")
        os.replace(received, destination)
    finally:
        if transfer.exists():
            shutil.rmtree(transfer)

    payload = {
        "status": "PASS_RESULT_RETURN_SIZE_VERIFIED_NO_REHASH",
        "integrity_policy": "USER_DIRECTED_PATH_AND_BYTE_SIZE_ONLY",
        "fold": args.fold,
        "source_host": remote(args.ssh_host, args.ssh_port, key, "hostname").strip(),
        "destination_host": "149",
        "destination": str(destination),
        "file_count": len(expected),
        "total_bytes": sum(expected.values()),
        "files": expected,
    }
    temporary = receipt.with_name(f".{receipt.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, receipt)
    print(json.dumps({key: payload[key] for key in (
        "status", "fold", "file_count", "total_bytes"
    )}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
