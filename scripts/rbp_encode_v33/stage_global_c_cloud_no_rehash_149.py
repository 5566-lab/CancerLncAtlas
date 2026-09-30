#!/usr/bin/env python3
"""Stage one CPU-validated final G2 fold without recomputing large-file hashes.

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
    "source_prepared", "c_graph_overlay", "global_binding_source", "config.yaml",
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
    parser.add_argument("--prior-transfer-manifest", type=Path, required=True)
    parser.add_argument("--overlay-receipt", type=Path, required=True)
    parser.add_argument("--cpu-receipt", type=Path, required=True)
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
    if (manifest.get("format") != "CANCERLNCATLAS_C_GLOBAL_G2_FOLD_CLOUD_TRANSFER_V1"
            or manifest.get("preparation_host") != "149"
            or manifest.get("billing_mode") != "Postpay"):
        raise RuntimeError("Transfer manifest is not the authorized Postpay C-fold manifest")
    fold = manifest.get("fold")
    if type(fold) is not int or fold not in range(5):
        raise RuntimeError("Invalid fold")
    root = PurePosixPath(manifest["gpu_target_root"])
    expected_root = PurePosixPath(
        f"/root/CancerLncAtlas_C_GLOBAL_G2_20260928/fold_{fold}"
    )
    if root != expected_root:
        raise RuntimeError("Unsafe GPU target root")
    rows = manifest.get("files")
    if not isinstance(rows, list) or len(rows) != len(ROLES) or {row.get("role") for row in rows} != ROLES:
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
    source_row = by_role["source_prepared"]
    if (source_row.get("source") != trusted.get("source_path")
            or source_row.get("bytes") != trusted.get("source_bytes")
            or source_row.get("sha256_reused") != trusted.get("source_sha256")):
        raise RuntimeError("A parent differs from the existing verified authority")
    prior_transfer = json.loads(args.prior_transfer_manifest.read_text(encoding="utf-8"))
    prior_sources = [row for row in prior_transfer.get("files", [])
                     if row.get("role") == "source_prepared"]
    reuse_path = PurePosixPath(str(source_row.get("reuse_cloud_source", "")))
    if (prior_transfer.get("format") != "CANCERLNCATLAS_C_FOLD_CLOUD_TRANSFER_V2"
            or prior_transfer.get("fold") != fold or len(prior_sources) != 1
            or prior_sources[0].get("source") != trusted["source_path"]
            or prior_sources[0].get("bytes") != trusted["source_bytes"]
            or prior_sources[0].get("sha256") != trusted["source_sha256"]
            or prior_sources[0].get("destination") != str(reuse_path)
            or not reuse_path.is_absolute() or ".." in reuse_path.parts
            or not re.fullmatch(r"/[A-Za-z0-9_./-]+", str(reuse_path))):
        raise RuntimeError("Cloud A reuse path differs from its historical transfer receipt")
    sidecar = json.loads(args.overlay_receipt.read_text(encoding="utf-8"))
    cpu = json.loads(args.cpu_receipt.read_text(encoding="utf-8"))
    overlay_row = by_role["c_graph_overlay"]
    binding_row = by_role["global_binding_source"]
    if (sidecar.get("status") != "PASS_G2_GLOBAL_BINDING_SIDECAR"
            or sidecar.get("target_host") != "149"
            or sidecar.get("patient_fold") != fold
            or sidecar.get("binding_context_policy") != "GLOBAL_PHYSICAL_BINDING"
            or overlay_row.get("source") != sidecar.get("path")
            or overlay_row.get("bytes") != sidecar.get("bytes")
            or binding_row.get("source") != sidecar.get("global_binding")
            or cpu.get("status") != "PASS_G2_GLOBAL_C_CPU_MODEL_PREFLIGHT"
            or cpu.get("target_host") != "149" or cpu.get("fold") != fold
            or cpu.get("graph_overlay_path") != sidecar.get("path")
            or cpu.get("graph_overlay_bytes") != sidecar.get("bytes")):
        raise RuntimeError("Final G2 graph or CPU authority differs from transfer manifest")

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
    reuse_size = remote(
        args.ssh_host, args.ssh_port, key,
        f"test -f {shlex.quote(str(reuse_path))} && stat -c %s -- "
        f"{shlex.quote(str(reuse_path))} || echo missing",
    )
    reuse_parent = reuse_size == str(source_row["bytes"])
    required = (sum(row["bytes"] for row in rows
                    if row["role"] != "source_prepared" or not reuse_parent)
                + args.manifest.stat().st_size + 10 * 1024**3)
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
        if row["role"] == "source_prepared" and reuse_parent and existing == str(row["bytes"]):
            paired = remote(
                args.ssh_host, args.ssh_port, key,
                f"stat -c '%d:%i %s' -- {shlex.quote(str(reuse_path))} "
                f"{shlex.quote(str(destination))}",
            ).splitlines()
            if len(paired) != 2 or paired[0] != paired[1]:
                raise RuntimeError("Existing new A is not the verified cloud hard link")
        if existing != str(row["bytes"]):
            if row["role"] == "source_prepared" and reuse_parent:
                if existing != "missing":
                    raise RuntimeError("Existing new A destination has an unexpected size")
                remote(args.ssh_host, args.ssh_port, key,
                       f"ln -T -- {shlex.quote(str(reuse_path))} "
                       f"{shlex.quote(str(destination))}")
                paired = remote(args.ssh_host, args.ssh_port, key,
                                f"stat -c '%d:%i %s' -- {shlex.quote(str(reuse_path))} "
                                f"{shlex.quote(str(destination))}").splitlines()
                if len(paired) != 2 or paired[0] != paired[1] or not paired[0].endswith(
                    f" {row['bytes']}"
                ):
                    raise RuntimeError("Cloud A hard link did not reuse the existing inode")
                completed.append(row["role"])
                print(json.dumps({"status": "CLOUD_A_HARD_LINK_REUSED", "role": row["role"]}), flush=True)
                continue
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
        "format": "C_GLOBAL_G2_CLOUD_STAGED_NO_REHASH_V1",
        "integrity_policy": "USER_DIRECTED_SIZE_AND_GRAPH_SEMANTICS_NO_REHASH",
        "preparation_host": "149", "gpu_host": gpu_hostname,
        "instance_id": args.instance_id, "billing_mode": "Postpay", "fold": fold,
        "gpu_target_root": str(root), "verified_roles": sorted(completed),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
