#!/usr/bin/env python3
"""Capture kernel high-water memory for one strict Drug runtime validation."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def process_snapshot(pid: int, required_token: str) -> dict[str, object] | None:
    process_root = Path(f"/proc/{pid}")
    try:
        cmdline = process_root.joinpath("cmdline").read_bytes().replace(b"\0", b" ").decode(
            "utf-8", errors="replace"
        ).strip()
        if required_token not in cmdline:
            return None
        fields: dict[str, int] = {}
        for line in process_root.joinpath("status").read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition(":")
            if name in {"VmPeak", "VmSize", "VmHWM", "VmRSS"}:
                fields[name + "_kB"] = int(value.strip().split()[0])
        environ = process_root.joinpath("environ").read_bytes().split(b"\0")
        selected_environment = {}
        for entry in environ:
            key, sep, value = entry.partition(b"=")
            if sep and key.decode(errors="ignore") in {
                "CC_HHGT_DRUG_SPARSE_DUCKDB_MEMORY_LIMIT",
                "CC_HHGT_DRUG_SPARSE_DUCKDB_THREADS",
                "CC_HHGT_DRUG_SPARSE_DUCKDB_TEMP_DIRECTORY",
            }:
                selected_environment[key.decode()] = value.decode(errors="replace")
        return {
            "cmdline": cmdline,
            "memory": fields,
            "environment": selected_environment,
        }
    except (FileNotFoundError, ProcessLookupError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--required-token", required=True)
    parser.add_argument("--runtime-receipt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--interval-seconds", type=float, default=2.0)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite resource receipt: {output}")
    started = utc_now()
    samples = 0
    max_hwm_kb = 0
    max_rss_kb = 0
    first_snapshot = None
    last_snapshot = None
    while True:
        snapshot = process_snapshot(args.pid, args.required_token)
        if snapshot is None:
            break
        if first_snapshot is None:
            first_snapshot = snapshot
        last_snapshot = snapshot
        memory = snapshot["memory"]
        max_hwm_kb = max(max_hwm_kb, int(memory.get("VmHWM_kB", 0)))
        max_rss_kb = max(max_rss_kb, int(memory.get("VmRSS_kB", 0)))
        samples += 1
        time.sleep(max(args.interval_seconds, 0.25))
    receipt_path = Path(args.runtime_receipt).resolve()
    receipt_status = "MISSING"
    receipt_sha = None
    if receipt_path.is_file():
        try:
            receipt_status = json.loads(receipt_path.read_text(encoding="utf-8")).get(
                "status", "INVALID"
            )
        except Exception:
            receipt_status = "INVALID"
        receipt_sha = sha256_file(receipt_path)
    payload = {
        "format": "CANCERLNCATLAS_V32_DRUG_STRICT_RUNTIME_RESOURCE_RECEIPT_V1",
        "status": "PASS" if receipt_status == "PASS" else "PROCESS_EXITED_WITHOUT_PASS_RECEIPT",
        "pid": args.pid,
        "required_token": args.required_token,
        "monitor_started_at_utc": started,
        "monitor_finished_at_utc": utc_now(),
        "samples": samples,
        "kernel_peak_rss_kB": max_hwm_kb,
        "max_sampled_rss_kB": max_rss_kb,
        "first_snapshot": first_snapshot,
        "last_snapshot": last_snapshot,
        "runtime_receipt": {
            "path": str(receipt_path),
            "status": receipt_status,
            "sha256": receipt_sha,
        },
        "semantic_gate_relaxed": False,
        "production_deployed": False,
        "release_ready": False,
        "main_score_changed": False,
        "production_port_8260_touched": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    with output.open("xb") as stream:
        stream.write(encoded)
    print(payload["status"])
    return 0 if receipt_status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
