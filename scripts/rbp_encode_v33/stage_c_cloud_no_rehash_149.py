#!/usr/bin/env python3
"""Stage one verified C fold without recomputing large-file hashes.

This path is for the user-directed single-writer workflow.  It reuses the
existing five-fold SHA receipt as authority, checks paths and byte sizes, and
relies on SSH transport integrity.  It never computes a file digest.
"""
from __future__ import annotations

import argparse
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


def ssh_args(host: str, port: int, key: Path) -> list[str]:
    return [
        "ssh", "-i", str(key), "-p", str(port), "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=15",
        "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3",
        f"root@{host}",
    ]


def remote(host: str, port: int, key: Path, command: str, *, timeout: int = 120) -> str:
    done = subprocess.run(
        [*ssh_args(host, port, key), command], capture_output=True, text=True,
        timeout=timeout, check=True,
    )
    return done.stdout.strip()


def safe_endpoint(value: str) -> bool:
    if re.fullmatch(r"cpod-[a-z0-9-]+\.podtcp\.compshare\.cn", value):
        return True
    parts = value.split(".")
    return len(parts) == 4 and all(part.isdigit() and 0 <= int(part) <= 255 for part in parts)


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fold-authority", type=Path, required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--ssh-host", required=True)
    parser.add_argument("--ssh-port", type=int, required=True)
    parser.add_argument("--identity-file", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--user-directed-no-rehash", action="store_true", required=True)
    args = parser.parse_args()

    if socket.gethostname() != "149":
        raise RuntimeError("C cloud staging must run on host 149")
    if not safe_endpoint(args.ssh_host) or not 1 <= args.ssh_port <= 65535:
        raise RuntimeError("Invalid CompShare SSH endpoint")
    key = args.identity_file.resolve(strict=True)
    if stat.S_IMODE(key.stat().st_mode) & 0o077:
        raise RuntimeError("Dedicated SSH identity file is not private")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if (manifest.get("format") != "CANCERLNCATLAS_C_FOLD_CLOUD_TRANSFER_V2"
            or manifest.get("preparation_host") != "149"
            or manifest.get("billing_mode") != "Postpay"):
        raise RuntimeError("Transfer manifest is not the authorized Postpay C-fold manifest")
    fold = manifest.get("fold")
    if type(fold) is not int or fold not in range(5):
        raise RuntimeError("Invalid fold")
    root = PurePosixPath(manifest["gpu_target_root"])
    if not root.is_absolute() or ".." in root.parts or str(root) in {"/", "/root"}:
        raise RuntimeError("Unsafe GPU target root")
    rows = manifest.get("files")
    if not isinstance(rows, list) or {row.get("role") for row in rows} != ROLES:
        raise RuntimeError("Transfer manifest roles differ from the authorized set")

    authority = json.loads(args.fold_authority.read_text(encoding="utf-8"))
    if (authority.get("status") != "PASS_C_ALL_FOLDS_INPUT_SHA256"
            or authority.get("target_host") != "149"):
        raise RuntimeError("Existing five-fold authority is invalid")
    matches = [row for row in authority.get("folds", []) if row.get("fold") == fold]
    if len(matches) != 1:
        raise RuntimeError("Existing five-fold authority lacks this fold")
    trusted = matches[0]
    by_role = {row["role"]: row for row in rows}
    for role, path_key, size_key, digest_key in (
        ("source_prepared", "source_path", "source_bytes", "source_sha256"),
        ("c_graph_overlay", "overlay_path", "overlay_bytes", "overlay_sha256"),
    ):
        row = by_role[role]
        if (row.get("source") != trusted.get(path_key)
                or row.get("bytes") != trusted.get(size_key)
                or row.get("sha256") != trusted.get(digest_key)):
            raise RuntimeError(f"{role} differs from the existing verified authority")

    destinations: set[PurePosixPath] = set()
    for row in rows:
        source = Path(row["source"])
        destination = PurePosixPath(row["destination"])
        if not source.is_absolute() or not source.is_file() or source.stat().st_size != row["bytes"]:
            raise RuntimeError(f"Source path or byte size changed: {row['role']}")
        if (not destination.is_absolute() or ".." in destination.parts
                or not destination.is_relative_to(root)
                or not re.fullmatch(r"/[A-Za-z0-9_./-]+", str(destination))):
            raise RuntimeError(f"Unsafe destination: {row['role']}")
        if destination in destinations:
            raise RuntimeError("Duplicate cloud destination")
        destinations.add(destination)

    gpu_hostname = remote(args.ssh_host, args.ssh_port, key, "hostname")
    if not gpu_hostname or gpu_hostname == "149":
        raise RuntimeError("SSH endpoint is not the separate CompShare host")
    available = int(remote(
        args.ssh_host, args.ssh_port, key,
        "df -B1 --output=avail /root | tail -n 1",
    ))
    required = sum(row["bytes"] for row in rows) + args.manifest.stat().st_size + 10 * 1024**3
    if available < required:
        raise RuntimeError("Cloud disk lacks input bytes plus 10 GiB headroom")

    completed = []
    for row in rows:
        source = Path(row["source"])
        destination = PurePosixPath(row["destination"])
        temporary = destination.with_name(destination.name + ".partial")
        remote(args.ssh_host, args.ssh_port, key,
               f"mkdir -p -- {shlex.quote(str(destination.parent))}")
        existing = remote(
            args.ssh_host, args.ssh_port, key,
            f"test -e {shlex.quote(str(destination))} && stat -c %s -- "
            f"{shlex.quote(str(destination))} || echo missing",
        )
        if existing != str(row["bytes"]):
            remote(args.ssh_host, args.ssh_port, key,
                   f"rm -f -- {shlex.quote(str(temporary))}")
            subprocess.run([
                "scp", "-i", str(key), "-P", str(args.ssh_port),
                "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
                "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30",
                "-o", "ServerAliveCountMax=3", str(source),
                f"root@{args.ssh_host}:{temporary}",
            ], timeout=7200, check=True)
            observed = remote(
                args.ssh_host, args.ssh_port, key,
                f"stat -c %s -- {shlex.quote(str(temporary))}",
            )
            if observed != str(row["bytes"]):
                remote(args.ssh_host, args.ssh_port, key,
                       f"rm -f -- {shlex.quote(str(temporary))}")
                raise RuntimeError(f"Transferred byte size mismatch: {row['role']}")
            remote(args.ssh_host, args.ssh_port, key,
                   f"mv -- {shlex.quote(str(temporary))} {shlex.quote(str(destination))}")
        completed.append(row["role"])
        print(json.dumps({"status": "CLOUD_FILE_SIZE_VERIFIED", "role": row["role"]}), flush=True)

    manifest_destination = root / "auth" / "CLOUD_TRANSFER_MANIFEST.json"
    subprocess.run([
        "scp", "-i", str(key), "-P", str(args.ssh_port), "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new", str(args.manifest),
        f"root@{args.ssh_host}:{manifest_destination}",
    ], timeout=300, check=True)
    write_once(args.receipt, {
        "format": "C_GRAPH_CLOUD_STAGED_NO_REHASH_V1",
        "integrity_policy": "USER_DIRECTED_REUSE_VERIFIED_RECEIPTS_SIZE_ONLY",
        "preparation_host": "149", "gpu_host": gpu_hostname,
        "instance_id": args.instance_id, "billing_mode": "Postpay", "fold": fold,
        "gpu_target_root": str(root), "verified_roles": sorted(completed),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
