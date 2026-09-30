#!/usr/bin/env python3
"""Return one final G2 global-C fold result tree to host 149 by path and size."""
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


def safe_endpoint(value: str) -> bool:
    if re.fullmatch(r"[0-9]{1,3}(?:\.[0-9]{1,3}){3}", value):
        return all(int(part) <= 255 for part in value.split("."))
    return bool(re.fullmatch(r"cpod-[a-z0-9-]+\.podtcp\.compshare\.cn", value))


def ssh_args(host: str, port: int, key: Path) -> list[str]:
    return [
        "ssh", "-i", str(key), "-p", str(port), "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=20",
        "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3",
        f"root@{host}",
    ]


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
        if (not relative or relative.startswith("/")
                or ".." in PurePosixPath(relative).parts
                or not re.fullmatch(r"[A-Za-z0-9_.|/-]+", relative)):
            raise RuntimeError(f"Unsafe result path: {relative!r}")
        if relative in rows:
            raise RuntimeError("Duplicate result path")
        size = int(raw_size)
        if size < 0:
            raise RuntimeError("Negative result size")
        rows[relative] = size
    if not rows:
        raise RuntimeError("Cloud result tree is empty")
    return rows


def local_manifest(root: Path) -> dict[str, int]:
    return {
        path.relative_to(root).as_posix(): path.stat().st_size
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def required_result_paths(fold: int) -> tuple[str, dict[str, str]]:
    task_dir = (
        "v32-g012-g2-c-global-binding-20260928-r1"
        f"__PATIENT_FOLD_{fold}__CC-HHGT__20260726"
    )
    files = {
        name: f"{task_dir}/{name}"
        for name in (
            "SUCCESS.json", "best_model_state.pt", "last_training_state.pt",
            "training_log.jsonl",
        )
    }
    return task_dir, files


def validate_success(root: Path, fold: int, manifest: dict[str, int]) -> dict:
    task_dir, required = required_result_paths(fold)
    if any(manifest.get(relative, 0) <= 0 for relative in required.values()):
        raise RuntimeError("Final G2 result tree lacks required nonempty files")
    success = json.loads((root / required["SUCCESS.json"]).read_text(encoding="utf-8"))
    expected_task_id = (
        "v32-g012-g2-c-global-binding-20260928-r1"
        f"|PATIENT_FOLD_{fold}|CC-HHGT|20260726"
    )
    if (success.get("status") != "SUCCESS"
            or success.get("patient_fold") != fold
            or success.get("task_id") != expected_task_id
            or success.get("graph_variant") != "G2"
            or success.get("did_not_hit_hard_cap") is not True
            or not isinstance(success.get("optimizer_steps"), int)
            or success["optimizer_steps"] <= 0
            or success.get("best_model_state_size_bytes")
            != manifest[required["best_model_state.pt"]]):
        raise RuntimeError(f"Final G2 fold {fold} SUCCESS receipt is invalid")
    return {
        "task_dir": task_dir,
        "task_id": expected_task_id,
        "stop_reason": success["stop_reason"],
        "completed_cycles": success["completed_cycles"],
        "optimizer_steps": success["optimizer_steps"],
        "best_validation_loss": success["best_validation_loss"],
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
        f"/root/CancerLncAtlas_C_GLOBAL_G2_20260928/fold_{args.fold}"
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
    source_host = remote(args.ssh_host, args.ssh_port, key, "hostname").strip()
    if not source_host:
        raise RuntimeError("CompShare host did not report a hostname")
    expected = parse_manifest(remote(args.ssh_host, args.ssh_port, key, command))
    _, required = required_result_paths(args.fold)
    if any(expected.get(relative, 0) <= 0 for relative in required.values()):
        raise RuntimeError("Cloud result tree has no complete final G2 fold")
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
        success_summary = validate_success(received, args.fold, observed)
        os.replace(received, destination)
    finally:
        if transfer.exists():
            shutil.rmtree(transfer)

    payload = {
        "status": "PASS_RESULT_RETURN_SIZE_VERIFIED_NO_REHASH",
        "integrity_policy": "USER_DIRECTED_PATH_AND_BYTE_SIZE_ONLY",
        "fold": args.fold,
        "source_host": source_host,
        "destination_host": "149",
        "destination": str(destination),
        "file_count": len(expected),
        "total_bytes": sum(expected.values()),
        "files": expected,
        "success": success_summary,
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
