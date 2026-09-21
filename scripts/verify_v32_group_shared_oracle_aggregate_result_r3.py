"""Verify a returned R3 result tar using one aggregate SHA256 only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, Sequence


SHA256 = re.compile(r"[0-9a-f]{64}")
MEMBERS = {"scientific_result.json", "execution.stdout.jsonl"}


def _archive_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(
    archive_path: str | Path,
    *,
    expected_result_sha256: str,
    expected_input_sha256: str,
    expected_code_sha256: str,
) -> dict[str, Any]:
    archive = Path(archive_path).resolve()
    for label, value in (
        ("RESULT", expected_result_sha256),
        ("INPUT", expected_input_sha256),
        ("CODE", expected_code_sha256),
    ):
        if SHA256.fullmatch(value) is None:
            raise ValueError(f"ORACLE_R3_EXPECTED_{label}_SHA256_INVALID")
    if not archive.is_file() or archive.is_symlink():
        raise ValueError(f"ORACLE_R3_RESULT_ARCHIVE_NOT_REGULAR={archive}")
    observed = _archive_sha256(archive)
    if observed != expected_result_sha256:
        raise ValueError(
            "ORACLE_R3_RESULT_ARCHIVE_SHA256_DRIFT="
            f"expected={expected_result_sha256},observed={observed}"
        )

    with tarfile.open(archive, "r:*") as bundle:
        members = bundle.getmembers()
        names = {member.name for member in members}
        if names != MEMBERS:
            raise ValueError(f"ORACLE_R3_RESULT_MEMBER_SCOPE_DRIFT={sorted(names)}")
        for member in members:
            pure = PurePosixPath(member.name)
            if (
                pure.is_absolute()
                or ".." in pure.parts
                or not member.isfile()
                or member.issym()
                or member.islnk()
            ):
                raise ValueError(f"ORACLE_R3_UNSAFE_RESULT_MEMBER={member.name}")
        result_handle = bundle.extractfile("scientific_result.json")
        log_handle = bundle.extractfile("execution.stdout.jsonl")
        if result_handle is None or log_handle is None:
            raise ValueError("ORACLE_R3_RESULT_MEMBER_UNREADABLE")
        scientific = json.loads(result_handle.read().decode("utf-8"))
        log_text = log_handle.read().decode("utf-8")

    if (
        scientific.get("scientific_pass") is not True
        or scientific.get("status")
        != "PASS_REAL_DATA_ORACLE_COMPARISON_ONLY_NOT_AUTHORIZED_FOR_FORMAL_TRAINING"
        or scientific.get("aggregate_input_archive_sha256")
        != expected_input_sha256
        or scientific.get("aggregate_code_archive_sha256")
        != expected_code_sha256
        or scientific.get("integrity_scope")
        != "INPUT_CODE_RESULT_ARCHIVES_ONLY"
        or scientific.get("per_file_hashing_performed") is not False
        or scientific.get("per_fold_hashing_performed") is not False
    ):
        raise ValueError("ORACLE_R3_SCIENTIFIC_RESULT_CONTRACT_DRIFT")

    events: list[dict[str, Any]] = []
    for line in log_text.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append(value)
    telemetry_valid = any(
        event.get("format")
        == "CANCERLNCATLAS_V32_ORACLE_AGGREGATE_RUN_R3_V1"
        and event.get("status") == "ORACLE_R3_NVIDIA_SMI_TELEMETRY"
        and event.get("gpu_count") == 1
        and "RTX 4090" in str(event.get("gpu_name", ""))
        and bool(event.get("gpu_uuid"))
        and event.get("telemetry_source") == "nvidia-smi"
        for event in events
    )
    optimizer_valid = any(
        event.get("format") == "CC_HHGT_V3_2_GROUP_SHARED_ORACLE_COMPARISON_V1"
        and event.get("status") == "ORACLE_R3_FIRST_OPTIMIZER_STEP"
        and event.get("run_id")
        == "v32-g012-g2-group-shared-oracle-paid-gpu-20260901-r3"
        and event.get("task_id")
        == "v32-g012-g2-group-shared-oracle-paid-gpu-20260901-r3|PATIENT_FOLD_0|CC-HHGT|20260726"
        and event.get("optimizer_step") == 1
        and event.get("grad_finite") is True
        and event.get("parameters_finite") is True
        and event.get("parameter_delta_positive") is True
        and event.get("aggregate_input_archive_sha256")
        == expected_input_sha256
        and event.get("aggregate_code_archive_sha256")
        == expected_code_sha256
        and event.get("per_file_hashing_performed") is False
        and event.get("per_fold_hashing_performed") is False
        for event in events
    )
    if not telemetry_valid or not optimizer_valid:
        missing = []
        if not telemetry_valid:
            missing.append("VALID_NVIDIA_SMI_TELEMETRY")
        if not optimizer_valid:
            missing.append("VALID_FIRST_OPTIMIZER_STEP")
        raise ValueError(f"ORACLE_R3_TRAINING_START_EVIDENCE_MISSING={missing}")
    return {
        "format": "CANCERLNCATLAS_V32_ORACLE_AGGREGATE_RESULT_RETURN_R3_V1",
        "status": "ORACLE_R3_RETURNED_RESULT_PASS",
        "result_archive_path": str(archive),
        "result_archive_sha256": observed,
        "result_archive_size_bytes": archive.stat().st_size,
        "input_archive_sha256": expected_input_sha256,
        "code_archive_sha256": expected_code_sha256,
        "nvidia_smi_telemetry_observed": True,
        "first_optimizer_step_observed": True,
        "training_start_confirmed": True,
        "integrity_scope": "INPUT_CODE_RESULT_ARCHIVES_ONLY",
        "per_file_hashing_performed": False,
        "per_fold_hashing_performed": False,
    }


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    partial = path.with_name(path.name + ".partial")
    if partial.exists() or path.exists():
        raise ValueError(f"ORACLE_R3_VERIFICATION_RECEIPT_EXISTS={path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    partial.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--expected-result-sha256", required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--expected-code-sha256", required=True)
    parser.add_argument("--receipt")
    args = parser.parse_args(argv)
    value = verify(
        args.archive,
        expected_result_sha256=args.expected_result_sha256,
        expected_input_sha256=args.expected_input_sha256,
        expected_code_sha256=args.expected_code_sha256,
    )
    if args.receipt:
        _atomic_json(Path(args.receipt).resolve(), value)
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
