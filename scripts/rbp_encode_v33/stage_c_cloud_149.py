#!/usr/bin/env python3
"""Preflight C fold inputs on 149, then transfer them directly to one GPU host.

The preflight phase is CPU-only and must finish before a paid instance exists.
The transfer phase needs an already authorized instance and a dedicated SSH key
installed on it.  This script never creates or starts an instance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import socket
import stat
import subprocess
from pathlib import Path, PurePosixPath


ROLES = {
    "code_archive", "wheel_archive", "wheel_requirements", "launcher",
    "source_prepared", "c_graph_overlay", "config.yaml",
    "INPUT_MANIFEST.json", "TASK_MANIFEST.tsv", "TRAINING_APPROVAL.json",
    "AUTH_READY.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_rows(path: Path, expected_sha: str) -> tuple[dict, PurePosixPath, list[dict]]:
    if sha256(path) != expected_sha:
        raise RuntimeError("Cloud transfer manifest SHA256 mismatch")
    data = json.loads(path.read_text(encoding="utf-8"))
    if (data.get("format") != "CANCERLNCATLAS_C_FOLD_CLOUD_TRANSFER_V2"
            or data.get("preparation_host") != "149"
            or type(data.get("fold")) is not int
            or data["fold"] not in range(5)):
        raise RuntimeError("Transfer manifest has the wrong preparation scope")
    root = PurePosixPath(data["gpu_target_root"])
    if not root.is_absolute() or ".." in root.parts or str(root) in {"/", "/root"}:
        raise RuntimeError("GPU target root must be a dedicated absolute directory")
    rows = data.get("files")
    if not isinstance(rows, list) or len(rows) != len(ROLES):
        raise RuntimeError("Transfer manifest has the wrong file count")
    if {row.get("role") for row in rows} != ROLES:
        raise RuntimeError("Transfer manifest has missing or duplicate roles")
    destinations = set()
    for row in rows:
        source = Path(row["source"])
        destination = PurePosixPath(row["destination"])
        if not source.is_absolute() or not source.is_file():
            raise RuntimeError(f"Missing source: {row['role']}")
        if (not destination.is_absolute() or ".." in destination.parts
                or not destination.is_relative_to(root)):
            raise RuntimeError(f"Destination escapes cloud root: {row['role']}")
        if not re.fullmatch(r"/[A-Za-z0-9_./-]+", str(destination)):
            raise RuntimeError(f"Unsafe cloud destination: {row['role']}")
        if destination in destinations:
            raise RuntimeError("Duplicate cloud destination")
        destinations.add(destination)
        if source.stat().st_size != row["bytes"]:
            raise RuntimeError(f"Source size drift: {row['role']}")
        if not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise RuntimeError(f"Malformed source SHA256: {row['role']}")
    return data, root, rows


def write_once(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def preflight(path: Path, expected_sha: str, receipt: Path) -> None:
    if receipt.exists():
        raise FileExistsError(f"Refusing to repeat an existing source preflight: {receipt}")
    data, root, rows = manifest_rows(path, expected_sha)
    sources = {}
    for row in rows:
        source = Path(row["source"])
        if sha256(source) != row["sha256"]:
            raise RuntimeError(f"Source SHA256 mismatch: {row['role']}")
        sources[row["role"]] = {
            "path": str(source), "sha256": row["sha256"],
            "bytes": row["bytes"], "mtime_ns": source.stat().st_mtime_ns,
        }
        print(json.dumps({"status": "SOURCE_VERIFIED", "role": row["role"]}), flush=True)
    write_once(receipt, {
        "format": "C_GRAPH_CLOUD_SOURCE_PREFLIGHT_V1", "target_host": "149",
        "gpu_target_root": str(root), "manifest_sha256": expected_sha,
        "billing_mode": data["billing_mode"], "fold": data["fold"],
        "files": sources, "paid_gpu_started": False,
    })


def remote_args(host: str, port: int, key: Path) -> list[str]:
    return [
        "ssh", "-i", str(key), "-p", str(port), "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=15",
        "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3",
        f"root@{host}",
    ]


def remote(host: str, port: int, key: Path, command: str) -> str:
    result = subprocess.run(
        [*remote_args(host, port, key), command], capture_output=True, text=True,
        timeout=7200, check=True,
    )
    return result.stdout.strip()


def remote_hash(host: str, port: int, key: Path, path: PurePosixPath) -> str:
    output = remote(host, port, key, f"sha256sum -- {shlex.quote(str(path))}")
    return output.split()[0]


def transfer_one(host: str, port: int, key: Path, source: Path,
                 destination: PurePosixPath, expected_sha: str) -> None:
    parent = destination.parent
    temporary = destination.with_name(destination.name + ".partial")
    remote(host, port, key, f"mkdir -p -- {shlex.quote(str(parent))}")
    exists = remote(host, port, key, f"test -e {shlex.quote(str(destination))} && echo yes || echo no")
    if exists == "yes":
        if remote_hash(host, port, key, destination) != expected_sha:
            raise RuntimeError(f"Existing cloud file has wrong SHA256: {destination}")
        return
    remote(host, port, key, f"rm -f -- {shlex.quote(str(temporary))}")
    argv = [
        "scp", "-i", str(key), "-P", str(port), "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=15",
        "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3",
        str(source), f"root@{host}:{temporary}",
    ]
    try:
        subprocess.run(argv, timeout=7200, check=True)
        if remote_hash(host, port, key, temporary) != expected_sha:
            raise RuntimeError(f"Transferred SHA256 mismatch: {destination}")
        remote(host, port, key,
               f"test ! -e {shlex.quote(str(destination))} && "
               f"mv -- {shlex.quote(str(temporary))} {shlex.quote(str(destination))}")
    except Exception:
        remote(host, port, key, f"rm -f -- {shlex.quote(str(temporary))}")
        raise


def transfer(args: argparse.Namespace) -> None:
    if args.receipt.exists():
        raise FileExistsError(f"Refusing to repeat an existing cloud transfer: {args.receipt}")
    data, root, rows = manifest_rows(args.manifest, args.manifest_sha256)
    ready = json.loads(args.source_ready_receipt.read_text(encoding="utf-8"))
    if (ready.get("format") != "C_GRAPH_CLOUD_SOURCE_PREFLIGHT_V1"
            or ready.get("target_host") != "149"
            or ready.get("manifest_sha256") != args.manifest_sha256
            or ready.get("gpu_target_root") != str(root)
            or ready.get("billing_mode") != data["billing_mode"]
            or ready.get("fold") != data["fold"]):
        raise RuntimeError("CPU source preflight does not match this transfer")
    for row in rows:
        source = Path(row["source"])
        prior = ready["files"].get(row["role"])
        if (prior is None or prior["path"] != str(source)
                or prior["sha256"] != row["sha256"]
                or prior["bytes"] != source.stat().st_size
                or prior["mtime_ns"] != source.stat().st_mtime_ns):
            raise RuntimeError(f"CPU source changed after preflight: {row['role']}")
    if (not re.fullmatch(r"cpod-[a-z0-9-]+\.podtcp\.compshare\.cn", args.ssh_host)
            or not 1 <= args.ssh_port <= 65535):
        raise RuntimeError("Invalid CompShare SSH endpoint")
    if not args.identity_file.is_file() or stat.S_IMODE(args.identity_file.stat().st_mode) & 0o077:
        raise RuntimeError("Dedicated SSH identity file is absent or not private")
    gpu_hostname = remote(args.ssh_host, args.ssh_port, args.identity_file, "hostname")
    if not gpu_hostname or gpu_hostname == "149":
        raise RuntimeError("CompShare SSH endpoint did not identify a separate GPU host")
    available = int(remote(args.ssh_host, args.ssh_port, args.identity_file,
                           "df -B1 --output=avail /root | tail -n 1"))
    required = sum(row["bytes"] for row in rows) + args.manifest.stat().st_size + 10 * 1024**3
    if available < required:
        raise RuntimeError("CompShare disk lacks transfer bytes plus 10 GiB headroom")
    for row in rows:
        transfer_one(args.ssh_host, args.ssh_port, args.identity_file,
                     Path(row["source"]), PurePosixPath(row["destination"]), row["sha256"])
        print(json.dumps({"status": "CLOUD_FILE_VERIFIED", "role": row["role"]}), flush=True)
    transfer_one(args.ssh_host, args.ssh_port, args.identity_file, args.manifest,
                 root / "auth" / "CLOUD_TRANSFER_MANIFEST.json", args.manifest_sha256)
    write_once(args.receipt, {
        "format": "C_GRAPH_CLOUD_STAGED_V1", "preparation_host": "149",
        "gpu_host": gpu_hostname, "instance_id": args.instance_id,
        "billing_mode": data["billing_mode"], "fold": data["fold"],
        "gpu_target_root": str(root),
        "manifest_sha256": args.manifest_sha256,
        "verified_roles": sorted(ROLES),
    })


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("preflight", "transfer"))
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--receipt", type=Path, required=True)
    p.add_argument("--source-ready-receipt", type=Path)
    p.add_argument("--instance-id")
    p.add_argument("--ssh-host")
    p.add_argument("--ssh-port", type=int)
    p.add_argument("--identity-file", type=Path)
    args = p.parse_args()
    if socket.gethostname() != "149":
        raise RuntimeError("C cloud staging must be coordinated on host 149")
    if args.mode == "preflight":
        preflight(args.manifest, args.manifest_sha256, args.receipt)
    else:
        if not all((args.source_ready_receipt, args.instance_id, args.ssh_host,
                    args.ssh_port, args.identity_file)):
            raise RuntimeError("Transfer requires a preflight receipt and one SSH endpoint")
        transfer(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
