#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from cc_hhgt.v30_integrity import (
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_sha256,
    file_sha256,
    merkle_sha256,
    verify_file_manifest,
)


EXCLUDED_PARTS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "results",
    "input_snapshot",
    "cancerlncatlas_cc_hhgt.egg-info",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".tmp", ".log"}


def source_files(repo_root: Path) -> list[Path]:
    files: list[Path] = []
    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(repo_root)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if path.suffix.lower() in EXCLUDED_SUFFIXES:
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(repo_root).as_posix())


def hash_paths(root: Path, paths: list[Path], label: str) -> list[dict]:
    records: list[dict] = []
    total = len(paths)
    for index, path in enumerate(paths, start=1):
        stat = path.stat()
        records.append(
            {
                "relative_path": path.resolve().relative_to(root.resolve()).as_posix(),
                "size_bytes": int(stat.st_size),
                "sha256": file_sha256(path),
            }
        )
        if index == total or index % 10 == 0:
            print(f"[{label}] hashed {index}/{total}", flush=True)
    return records


def dataframe_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(sep="\t", index=False, lineterminator="\n").encode("utf-8")


def command_output(command: list[str]) -> dict:
    try:
        result = subprocess.run(command, capture_output=True, text=True)
    except OSError as exc:
        # CPU-only build hosts legitimately have no nvidia-smi.  Preserve the
        # absence in frozen provenance instead of making environment capture
        # itself GPU-dependent.  Required commands (notably Python) are still
        # rejected by their callers when this nonzero return code is observed.
        return {
            "command": command,
            "returncode": 127,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
            "availability": "UNAVAILABLE",
        }
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "availability": "AVAILABLE",
    }


def environment_payload(python: str) -> dict:
    details = command_output(
        [
            python,
            "-c",
            (
                "import json,platform,sys; d={'python':sys.version,'executable':sys.executable,"
                "'platform':platform.platform()}; "
                "\ntry:\n import torch; d.update({'torch':torch.__version__,'cuda':torch.version.cuda,"
                "'cuda_available':torch.cuda.is_available(),'cudnn':torch.backends.cudnn.version()})"
                "\nexcept Exception as e: d['torch_error']=repr(e)"
                "\nprint(json.dumps(d,sort_keys=True))"
            ),
        ]
    )
    if details["returncode"]:
        raise RuntimeError(f"Unable to capture Python environment: {details['stderr']}")
    payload = json.loads(details["stdout"])
    payload["pip_freeze"] = command_output([python, "-m", "pip", "freeze"])
    payload["nvidia_smi"] = command_output(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ]
    )
    payload["host_platform"] = platform.platform()
    return payload


def deterministic_code_zip(repo_root: Path, paths: list[Path], destination: Path) -> None:
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in paths:
            relative = path.relative_to(repo_root).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze code/config/input/environment for one V3.0 formal run")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--protocol", default="V3_STATE_FORMAL_RETRAINING_PROTOCOL.md")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output-parent", required=True)
    parser.add_argument("--label", default="state_formal")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    input_root = Path(args.input_root).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (repo_root / config_path).resolve()
    protocol_path = Path(args.protocol)
    if not protocol_path.is_absolute():
        protocol_path = (repo_root / protocol_path).resolve()
    output_parent = Path(args.output_parent).resolve()
    if not repo_root.is_dir() or not input_root.is_dir() or not config_path.is_file() or not protocol_path.is_file():
        raise RuntimeError("repo, input, config, or protocol path is missing")

    input_paths = sorted((path for path in input_root.rglob("*") if path.is_file()), key=lambda item: item.as_posix())
    code_paths = source_files(repo_root)
    config_paths = [path for path in code_paths if path == config_path or "config" in path.relative_to(repo_root).parts]
    schema_paths = [path for path in code_paths if "schemas" in path.relative_to(repo_root).parts]
    input_records = hash_paths(input_root, input_paths, "input")
    code_records = hash_paths(repo_root, code_paths, "code")
    config_records = hash_paths(repo_root, config_paths, "config")
    schema_records = hash_paths(repo_root, schema_paths, "schema")
    environment = environment_payload(args.python)
    roots = {
        "input_merkle_sha256": merkle_sha256(input_records),
        "code_merkle_sha256": merkle_sha256(code_records),
        "config_merkle_sha256": merkle_sha256(config_records),
        "schema_merkle_sha256": merkle_sha256(schema_records),
        "environment_sha256": canonical_json_sha256(environment),
        "protocol_sha256": file_sha256(protocol_path),
    }
    content_sha256 = canonical_json_sha256(roots)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"V3STATE-{args.label}-{timestamp}-{content_sha256[:12]}"
    output_parent.mkdir(parents=True, exist_ok=True)
    temporary_root = output_parent / f".{run_id}.tmp"
    final_root = output_parent / run_id
    if temporary_root.exists() or final_root.exists():
        raise RuntimeError(f"Refusing to reuse run root: {temporary_root} or {final_root}")
    temporary_root.mkdir()
    provenance_root = temporary_root / "provenance"
    provenance_root.mkdir()
    atomic_write_bytes(provenance_root / "INPUT_MANIFEST.tsv", dataframe_bytes(pd.DataFrame(input_records)))
    atomic_write_bytes(provenance_root / "CODE_MANIFEST.tsv", dataframe_bytes(pd.DataFrame(code_records)))
    atomic_write_bytes(provenance_root / "CONFIG_MANIFEST.tsv", dataframe_bytes(pd.DataFrame(config_records)))
    atomic_write_bytes(provenance_root / "SCHEMA_MANIFEST.tsv", dataframe_bytes(pd.DataFrame(schema_records)))
    atomic_write_json(provenance_root / "ENVIRONMENT.json", environment)
    deterministic_code_zip(repo_root, code_paths, provenance_root / "CODE_SNAPSHOT.zip")
    snapshot_sha256 = file_sha256(provenance_root / "CODE_SNAPSHOT.zip")
    mismatches = verify_file_manifest(repo_root, code_records) + verify_file_manifest(input_root, input_records)
    if mismatches:
        raise RuntimeError(f"Assets changed while the snapshot was being created: {mismatches[:20]}")
    provenance = {
        "status": "FROZEN",
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "input_root": str(input_root),
        "config_path": str(config_path),
        "protocol_path": str(protocol_path),
        "input_files": len(input_records),
        "code_files": len(code_records),
        "config_files": len(config_records),
        "schema_files": len(schema_records),
        "code_snapshot_sha256": snapshot_sha256,
        "content_sha256": content_sha256,
        **roots,
    }
    atomic_write_json(provenance_root / "PROVENANCE.json", provenance)
    atomic_write_json(
        temporary_root / "RUN_LOCK.json",
        {
            "status": "LOCKED",
            "run_id": run_id,
            "release_eligible": False,
            "formal_gate": "PENDING",
            **roots,
        },
    )
    os.replace(temporary_root, final_root)
    print(json.dumps({**provenance, "run_root": str(final_root)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

