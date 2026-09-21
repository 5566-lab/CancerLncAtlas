#!/usr/bin/env python3
"""Fail-closed serial supervisor for the frozen r7 single-cell runner.

Each cancer is planned and executed as an independent immutable partition.
The supervisor enforces a 512 MiB planned and observed process-RSS ceiling,
and stops the cohort with a typed failure when any cancer crosses the gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
from typing import Any


FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_R7_17C_COHORT_SUPERVISOR_V1"
SUCCESS_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_R7_17C_COHORT_SUCCESS_V1"
FAILURE_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_R7_17C_TYPED_FAILURE_V1"
TOOL_MANIFEST_SHA256 = (
    "06090c75a4c3a722c5be99c3ba6f3ae205c0905dbf023a68201c80d162fbf8d1"
)
SERVER_PREFLIGHT_SHA256 = (
    "a1565276f010e3d2a184aa27baccf31da7fb2b16cfa20690af81553c7acbb670"
)
RUN_STATUS_SHA256 = (
    "963268d03448ec514646dc199e3c4648faa1d47884f62faacfdd72aa23d6e2be"
)
TOOL_ROOT = Path(
    "./data/CancerLncAtlas/runtime/tools/"
    "single_cell_r7_streaming_20260829_r5"
)
PYTHON = Path("${PRIVATE_WORK_ROOT}/miniconda3/bin/python")
RUNNER = TOOL_ROOT / "scripts/run_v32_single_cell_r7_streaming.py"
R7_ROOT = Path(
    "./data/CancerLncAtlas/runtime/"
    "single_cell_cell_level_r7_portable_supersession_20260829"
)
PREFLIGHT = Path(
    "./data/CancerLncAtlas/runtime/audits/"
    "single_cell_r7_server_preflight_20260829_r2.json"
)
FORMAL_CANCERS = frozenset(
    {
        "ACC",
        "CHOL",
        "DLBC",
        "ESCA",
        "GBM",
        "HNSC",
        "KIRC",
        "LAML",
        "LGG",
        "LUSC",
        "MESO",
        "PCPG",
        "READ",
        "SARC",
        "SKCM",
        "THYM",
        "UCEC",
    }
)
MEMORY_LIMIT_BYTES = 512 * 1024**2
TOOL_SHAS = {
    "cc_hhgt/__init__.py": (
        98,
        "e246acd89da9a9733fbac09b2770161b5ed199695a93a832ca965d8d201b78dc",
    ),
    "cc_hhgt/v32/__init__.py": (
        363,
        "359f958ad6e7e67078201282718193b40bc0d09bf15d27cbefb3a604b535ca91",
    ),
    "cc_hhgt/v32/contracts.py": (
        8_363,
        "387e64d4931bac936e7affa4a3f974812dbfbb16d50deacd670685c796d41f46",
    ),
    "cc_hhgt/v32/single_cell_cell_level.py": (
        12_413,
        "08e5b3519539fe488c78626aab7a7ef94bad4f4ef7f94b2f656f029bd5056815",
    ),
    "cc_hhgt/v32/single_cell_r7_streaming.py": (
        24_761,
        "d3fdea24178d9da0b72b5f7f8c17905f621abc5d349bc7f344af95604b7309d8",
    ),
    "scripts/run_v32_single_cell_r7_streaming.py": (
        51_877,
        "8d34a2202c1be25dff3295969b2dd2f823b8fe57f52b8b717852cc8030fa4e6b",
    ),
}


class CohortError(RuntimeError):
    """Raised when a cohort invariant cannot be proven."""


def sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise CohortError(f"cannot hash absent/unsafe file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CohortError(f"JSON is absent/unsafe: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise CohortError(f"unsafe status temporary exists: {temporary}")
    exclusive_json(temporary, payload)
    os.replace(temporary, path)


def authorized(path: Path) -> None:
    normalized = str(path.resolve()).replace("\\", "/")
    if not normalized.startswith("./data/CancerLncAtlas/"):
        raise CohortError(f"path escapes authorized root: {path}")


def validate_frozen_tool() -> dict[str, Any]:
    observed = {}
    for relative, (expected_bytes, expected_sha) in TOOL_SHAS.items():
        path = TOOL_ROOT / relative
        size = path.stat().st_size if path.is_file() and not path.is_symlink() else -1
        digest = sha256_file(path)
        if size != expected_bytes or digest != expected_sha:
            raise CohortError(f"frozen r5 tool drift: {relative}")
        observed[relative] = {"bytes": size, "sha256": digest}
    if sha256_file(PREFLIGHT) != SERVER_PREFLIGHT_SHA256:
        raise CohortError("server preflight SHA drift")
    run_status = R7_ROOT / "RUN_STATUS.json"
    if sha256_file(run_status) != RUN_STATUS_SHA256:
        raise CohortError("r7 RUN_STATUS SHA drift")
    return observed


def ordered_formal_cancers() -> list[dict[str, Any]]:
    preflight = load_json(PREFLIGHT)
    rows = []
    for record in preflight.get("formal_cancers", []):
        cancer = str(record.get("cancer_id", "")).upper()
        header = record.get("matrix_header", {})
        rows.append({"cancer_id": cancer, "cells": int(header.get("cells", 0))})
    if {row["cancer_id"] for row in rows} != FORMAL_CANCERS:
        raise CohortError("preflight formal-cancer set drift")
    if any(row["cells"] <= 0 for row in rows):
        raise CohortError("preflight has a non-positive formal cell count")
    return sorted(rows, key=lambda row: (row["cells"], row["cancer_id"]))


def process_memory_bytes(process_id: int) -> tuple[int, int]:
    status = Path(f"/proc/{int(process_id)}/status")
    if not status.is_file():
        return 0, 0
    rss = 0
    high_water = 0
    for line in status.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("VmRSS:"):
            rss = int(line.split()[1]) * 1024
        elif line.startswith("VmHWM:"):
            high_water = int(line.split()[1]) * 1024
    return rss, high_water


def terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)


def run_monitored(
    argv: list[str], *, log_path: Path, memory_limit_bytes: int
) -> dict[str, Any]:
    if log_path.exists() or log_path.is_symlink():
        raise CohortError(f"log reuse is forbidden: {log_path}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(TOOL_ROOT)
    started = time.time()
    maximum_rss = 0
    maximum_hwm = 0
    memory_exceeded = False
    with log_path.open("xb") as log:
        process = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )
        while process.poll() is None:
            rss, high_water = process_memory_bytes(process.pid)
            maximum_rss = max(maximum_rss, rss)
            maximum_hwm = max(maximum_hwm, high_water)
            if max(rss, high_water) > int(memory_limit_bytes):
                memory_exceeded = True
                terminate_process_group(process)
                break
            time.sleep(1.0)
        return_code = process.wait()
        log.flush()
        os.fsync(log.fileno())
    return {
        "argv": argv,
        "return_code": int(return_code),
        "elapsed_seconds": round(time.time() - started, 3),
        "maximum_rss_bytes_observed": int(maximum_rss),
        "maximum_high_water_bytes_observed": int(maximum_hwm),
        "memory_limit_bytes": int(memory_limit_bytes),
        "memory_exceeded": memory_exceeded,
        "log_path": str(log_path),
        "log_sha256": sha256_file(log_path),
    }


def runner_args(
    *, mode: str, cancer: str, plan_path: Path | None, output_root: Path | None,
    resume: bool = False,
) -> list[str]:
    argv = [
        str(PYTHON),
        str(RUNNER),
        "--mode",
        mode,
        "--r7-root",
        str(R7_ROOT),
        "--server-preflight-json",
        str(PREFLIGHT),
        "--expected-server-preflight-sha256",
        SERVER_PREFLIGHT_SHA256,
        "--expected-run-status-sha256",
        RUN_STATUS_SHA256,
        "--cancer-id",
        cancer,
        "--chunk-cells",
        "64",
        "--checkpoint-every-chunks",
        "25",
        "--min-donors",
        "5",
        "--min-cells-per-donor-context",
        "20",
        "--min-lncrna-detect-rate",
        "0.01",
        "--min-lncrna-detecting-donors",
        "3",
        "--min-abs-rho",
        "0.5",
        "--max-nominal-p",
        "0.05",
        "--association-pathway-block",
        "128",
    ]
    if mode == "plan" and plan_path is not None:
        argv.extend(("--plan-output-json", str(plan_path)))
    elif mode == "run" and output_root is not None:
        argv.extend(("--output-parent", str(output_root)))
        if resume:
            argv.append("--resume")
    else:
        raise CohortError("runner argument mode/path mismatch")
    return argv


def audit_partition(
    *, final: Path, plan: dict[str, Any], cancer: str, output_root: Path
) -> dict[str, Any]:
    import pandas as pd

    success_path = final / "SUCCESS.json"
    lineage_path = final / "LINEAGE.json"
    manifest_path = final / "FILE_MANIFEST.parquet"
    success = load_json(success_path)
    lineage = load_json(lineage_path)
    if success.get("status") != "SUCCESS" or success.get("cancer_id") != cancer:
        raise CohortError(f"{cancer} SUCCESS contract drift")
    if success.get("contract_sha256") != plan.get("contract_sha256"):
        raise CohortError(f"{cancer} plan/SUCCESS contract mismatch")
    if lineage.get("contract_sha256") != success.get("contract_sha256"):
        raise CohortError(f"{cancer} lineage contract mismatch")
    for payload in (success, lineage):
        if payload.get("raw_h5_full_sha256_verified") is not True:
            raise CohortError(f"{cancer} lacks full raw-H5 SHA gate")
        if payload.get("full_input_sha256_verified_before_matrix_access") is not True:
            raise CohortError(f"{cancer} input SHA ordering gate is absent")
    if success.get("historical_derived_results_used") is not False:
        raise CohortError(f"{cancer} used historical derived results")
    for flag in (
        "historical_checkpoints_used",
        "historical_rankings_used",
        "historical_predictions_used",
        "historical_sc_trajectory_used",
    ):
        if lineage.get(flag) is not False:
            raise CohortError(f"{cancer} historical lineage flag is not false: {flag}")
    if success.get("cell_as_independent_replicate") is not False:
        raise CohortError(f"{cancer} treats cells as biological replicates")
    if success.get("cell_level_pathway_matrix_persisted") is not False:
        raise CohortError(f"{cancer} persisted a cell-by-pathway matrix")
    if int(success.get("cell_level_pathway_rows_written", -1)) != 0:
        raise CohortError(f"{cancer} wrote cell-level pathway rows")
    if sha256_file(lineage_path) != success.get("lineage_sha256"):
        raise CohortError(f"{cancer} SUCCESS does not bind LINEAGE")
    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != success.get("file_manifest_sha256") or manifest_sha != lineage.get(
        "file_manifest_sha256"
    ):
        raise CohortError(f"{cancer} manifest binding mismatch")
    code_shas = lineage.get("code_shas", {})
    expected_code = {
        "runner_sha256": TOOL_SHAS["scripts/run_v32_single_cell_r7_streaming.py"][1],
        "streaming_core_sha256": TOOL_SHAS[
            "cc_hhgt/v32/single_cell_r7_streaming.py"
        ][1],
        "ucell_core_sha256": TOOL_SHAS[
            "cc_hhgt/v32/single_cell_cell_level.py"
        ][1],
    }
    if code_shas != expected_code:
        raise CohortError(f"{cancer} was not produced by frozen r5 code")
    manifest = pd.read_parquet(manifest_path)
    expected_manifest_files = {
        "lncrna_donor_celltype_summary.parquet",
        "pathway_availability.parquet",
        "pathway_donor_celltype_summary.parquet",
        "association_evidence.parquet",
        "association_lncrna_testability.parquet",
        "association_context_availability.parquet",
        "RESOURCE_ESTIMATE.json",
    }
    if set(manifest.relative_path.astype(str)) != expected_manifest_files:
        raise CohortError(f"{cancer} manifest leaf set drift")
    for record in manifest.itertuples(index=False):
        relative = Path(str(record.relative_path))
        if relative.is_absolute() or ".." in relative.parts:
            raise CohortError(f"{cancer} unsafe manifest leaf: {relative}")
        payload = final / relative
        if payload.stat().st_size != int(record.bytes):
            raise CohortError(f"{cancer} manifest byte mismatch: {relative}")
        if sha256_file(payload) != str(record.sha256):
            raise CohortError(f"{cancer} manifest SHA mismatch: {relative}")
    work = (
        output_root
        / "_work"
        / f"cancer_id={cancer}"
        / str(success["contract_sha256"])
    )
    if (work / "RUNNING.json").exists():
        raise CohortError(f"{cancer} retained a RUNNING lock after SUCCESS")
    checkpoint_state = work / "CHECKPOINT_STATE.json"
    if sha256_file(checkpoint_state) != lineage.get("checkpoint_state_sha256"):
        raise CohortError(f"{cancer} checkpoint state binding mismatch")
    state = load_json(checkpoint_state)
    if int(state.get("next_cell", -1)) != int(state.get("total_cells", -2)):
        raise CohortError(f"{cancer} checkpoint is not terminal")
    return {
        "cancer_id": cancer,
        "contract_sha256": success["contract_sha256"],
        "success_sha256": sha256_file(success_path),
        "lineage_sha256": sha256_file(lineage_path),
        "file_manifest_sha256": manifest_sha,
        "cells": int(success["cells"]),
        "donors": int(success["donors"]),
        "lncrnas": int(success["lncrnas"]),
        "pathways_total": int(success["pathways_total"]),
        "pathways_available": int(success["pathways_available"]),
        "pathways_typed_unavailable": int(success["pathways_typed_unavailable"]),
        "association_evidence_rows": int(success["association_evidence_rows"]),
        "output_root": str(final),
        "fresh_full_sha_verified": True,
        "donor_aware": True,
        "typed_null_enforced": True,
        "historical_derived_results_used": False,
    }


def read_log_tail(path: Path, limit: int = 4000) -> str:
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-limit:]


def next_log_path(parent: Path, stem: str) -> Path:
    first = parent / f"{stem}.log"
    if not first.exists() and not first.is_symlink():
        return first
    attempt = 2
    while True:
        candidate = parent / f"{stem}.attempt{attempt}.log"
        if not candidate.exists() and not candidate.is_symlink():
            return candidate
        attempt += 1


def failure_payload(
    *, cancer: str, stage: str, reason: str, detail: dict[str, Any],
    output_root: Path, audit_root: Path,
) -> dict[str, Any]:
    payload = {
        "format": FAILURE_FORMAT,
        "status": "TYPED_FAILURE_STOPPED",
        "cancer_id": cancer,
        "stage": stage,
        "reason": reason,
        "detail": detail,
        "memory_limit_bytes": MEMORY_LIMIT_BYTES,
        "tool_manifest_sha256": TOOL_MANIFEST_SHA256,
        "output_root": str(output_root),
        "audit_root": str(audit_root),
        "remaining_cancers_not_started": True,
        "production_deployed": False,
        "port_8260_touched": False,
        "timestamp_unix": time.time(),
    }
    payload["failure_contract_sha256"] = canonical_sha256(payload)
    return payload


def acquire_lock(audit_root: Path) -> Path:
    lock = audit_root / "COHORT_RUNNING.json"
    if lock.exists():
        previous = load_json(lock)
        active = False
        if previous.get("hostname") == socket.gethostname():
            process_id = int(previous.get("process_id", -1))
            if process_id > 0:
                try:
                    os.kill(process_id, 0)
                    active = True
                except OSError:
                    active = False
        if active:
            raise CohortError(f"cohort supervisor already active: {previous}")
        stale = audit_root / f"STALE_COHORT_LOCK.{sha256_file(lock)}.json"
        if stale.exists():
            raise CohortError(f"stale cohort lock archive exists: {stale}")
        os.rename(lock, stale)
    exclusive_json(
        lock,
        {
            "format": FORMAT,
            "hostname": socket.gethostname(),
            "process_id": os.getpid(),
            "timestamp_unix": time.time(),
        },
    )
    return lock


def execute(args: argparse.Namespace) -> int:
    output_root = args.output_root.resolve()
    audit_root = args.audit_root.resolve()
    authorized(output_root)
    authorized(audit_root)
    if args.resume:
        if not output_root.is_dir() or not audit_root.is_dir():
            raise CohortError("resume requires existing output and audit roots")
    else:
        if output_root.exists() or output_root.is_symlink():
            raise CohortError(f"new cohort output root already exists: {output_root}")
        if audit_root.exists() or audit_root.is_symlink():
            raise CohortError(f"new cohort audit root already exists: {audit_root}")
        output_root.mkdir(parents=True)
        audit_root.mkdir(parents=True)
    if (audit_root / "COHORT_SUCCESS.json").exists():
        raise CohortError("cohort is already complete")
    if (audit_root / "TYPED_FAILURE.json").exists():
        raise CohortError("cohort has a typed terminal failure")
    lock = acquire_lock(audit_root)
    try:
        tool_files = validate_frozen_tool()
        ordered = ordered_formal_cancers()
        plans = audit_root / "plans"
        logs = audit_root / "logs"
        records_root = audit_root / "records"
        plans.mkdir(exist_ok=True)
        logs.mkdir(exist_ok=True)
        records_root.mkdir(exist_ok=True)
        completed: list[dict[str, Any]] = []
        for ordinal, scope in enumerate(ordered, start=1):
            cancer = scope["cancer_id"]
            record_path = records_root / f"{ordinal:02d}_{cancer}.json"
            plan_path = plans / f"{ordinal:02d}_{cancer}.json"
            if plan_path.exists():
                plan = load_json(plan_path)
            else:
                plan_log = logs / f"{ordinal:02d}_{cancer}.plan.log"
                plan_process = run_monitored(
                    runner_args(
                        mode="plan",
                        cancer=cancer,
                        plan_path=plan_path,
                        output_root=None,
                    ),
                    log_path=plan_log,
                    memory_limit_bytes=MEMORY_LIMIT_BYTES,
                )
                if plan_process["memory_exceeded"]:
                    failure = failure_payload(
                        cancer=cancer,
                        stage="PLAN",
                        reason="OBSERVED_PROCESS_MEMORY_EXCEEDED_512_MIB",
                        detail=plan_process,
                        output_root=output_root,
                        audit_root=audit_root,
                    )
                    exclusive_json(audit_root / "TYPED_FAILURE.json", failure)
                    return 2
                if plan_process["return_code"] != 0 or not plan_path.is_file():
                    plan_process["log_tail"] = read_log_tail(plan_log)
                    failure = failure_payload(
                        cancer=cancer,
                        stage="PLAN",
                        reason="PLAN_PROCESS_FAILED",
                        detail=plan_process,
                        output_root=output_root,
                        audit_root=audit_root,
                    )
                    exclusive_json(audit_root / "TYPED_FAILURE.json", failure)
                    return 2
                plan = load_json(plan_path)
                plan["supervisor_process_audit"] = plan_process
                plan_audit = plans / f"{ordinal:02d}_{cancer}.process.json"
                exclusive_json(plan_audit, plan_process)
            if plan.get("cancer_id") != cancer or plan.get(
                "raw_h5_full_sha256_verified"
            ) is not True:
                raise CohortError(f"{cancer} plan identity/full-SHA gate drift")
            planned_peak = int(plan.get("resource_estimate", {}).get(
                "estimated_peak_ram_bytes", MEMORY_LIMIT_BYTES + 1
            ))
            if plan.get("safe_to_start_pilot") is not True or planned_peak > MEMORY_LIMIT_BYTES:
                failure = failure_payload(
                    cancer=cancer,
                    stage="PLAN_GATE",
                    reason="PLANNED_PROCESS_MEMORY_EXCEEDED_512_MIB",
                    detail={
                        "estimated_peak_ram_bytes": planned_peak,
                        "plan_path": str(plan_path),
                        "plan_sha256": sha256_file(plan_path),
                    },
                    output_root=output_root,
                    audit_root=audit_root,
                )
                exclusive_json(audit_root / "TYPED_FAILURE.json", failure)
                return 2
            final = output_root / f"cancer_id={cancer}"
            if final.is_dir():
                partition = audit_partition(
                    final=final, plan=plan, cancer=cancer, output_root=output_root
                )
                run_process = {
                    "resumed_existing_verified_partition": True,
                    "maximum_high_water_bytes_observed": None,
                }
            else:
                work_parent = output_root / "_work" / f"cancer_id={cancer}"
                resume_partition = work_parent.exists()
                suffix = ".resume" if resume_partition else ""
                run_log = next_log_path(
                    logs, f"{ordinal:02d}_{cancer}.run{suffix}"
                )
                run_process = run_monitored(
                    runner_args(
                        mode="run",
                        cancer=cancer,
                        plan_path=None,
                        output_root=output_root,
                        resume=resume_partition,
                    ),
                    log_path=run_log,
                    memory_limit_bytes=MEMORY_LIMIT_BYTES,
                )
                if run_process["memory_exceeded"]:
                    failure = failure_payload(
                        cancer=cancer,
                        stage="RUN",
                        reason="OBSERVED_PROCESS_MEMORY_EXCEEDED_512_MIB",
                        detail=run_process,
                        output_root=output_root,
                        audit_root=audit_root,
                    )
                    exclusive_json(audit_root / "TYPED_FAILURE.json", failure)
                    return 2
                if run_process["return_code"] != 0 or not final.is_dir():
                    run_process["log_tail"] = read_log_tail(run_log)
                    failure = failure_payload(
                        cancer=cancer,
                        stage="RUN",
                        reason="RUN_PROCESS_FAILED",
                        detail=run_process,
                        output_root=output_root,
                        audit_root=audit_root,
                    )
                    exclusive_json(audit_root / "TYPED_FAILURE.json", failure)
                    return 2
                partition = audit_partition(
                    final=final, plan=plan, cancer=cancer, output_root=output_root
                )
            record = {
                "format": FORMAT,
                "status": "SUCCESS_AUDITED",
                "ordinal": ordinal,
                "preflight_cells": int(scope["cells"]),
                "plan_path": str(plan_path),
                "plan_sha256": sha256_file(plan_path),
                "planned_peak_ram_bytes": planned_peak,
                "run_process": run_process,
                **partition,
                "production_deployed": False,
                "port_8260_touched": False,
            }
            record["record_contract_sha256"] = canonical_sha256(record)
            if record_path.exists():
                existing = load_json(record_path)
                contract_payload = dict(existing)
                observed_contract = contract_payload.pop(
                    "record_contract_sha256", None
                )
                if observed_contract != canonical_sha256(contract_payload):
                    raise CohortError(f"{cancer} stored record hash drift")
                stable_keys = (
                    "cancer_id",
                    "contract_sha256",
                    "success_sha256",
                    "lineage_sha256",
                    "file_manifest_sha256",
                    "cells",
                    "donors",
                    "lncrnas",
                    "pathways_total",
                    "pathways_available",
                    "pathways_typed_unavailable",
                    "association_evidence_rows",
                )
                if any(existing.get(key) != record.get(key) for key in stable_keys):
                    raise CohortError(f"{cancer} immutable cohort record drift")
                record = existing
            else:
                exclusive_json(record_path, record)
            completed.append(record)
            status = {
                "format": FORMAT,
                "status": "RUNNING",
                "tool_manifest_sha256": TOOL_MANIFEST_SHA256,
                "output_root": str(output_root),
                "audit_root": str(audit_root),
                "ordered_cancers": [row["cancer_id"] for row in ordered],
                "completed_count": len(completed),
                "completed_cancers": [row["cancer_id"] for row in completed],
                "current_or_next_cancer": (
                    ordered[len(completed)]["cancer_id"]
                    if len(completed) < len(ordered)
                    else None
                ),
                "memory_limit_bytes": MEMORY_LIMIT_BYTES,
                "production_deployed": False,
                "port_8260_touched": False,
                "timestamp_unix": time.time(),
            }
            atomic_json(audit_root / "COHORT_STATUS.json", status)
        cohort = {
            "format": SUCCESS_FORMAT,
            "status": "SUCCESS_17_OF_17_AUDITED",
            "tool_manifest_sha256": TOOL_MANIFEST_SHA256,
            "tool_files": tool_files,
            "server_preflight_sha256": SERVER_PREFLIGHT_SHA256,
            "run_status_sha256": RUN_STATUS_SHA256,
            "output_root": str(output_root),
            "audit_root": str(audit_root),
            "memory_limit_bytes": MEMORY_LIMIT_BYTES,
            "execution_order": [row["cancer_id"] for row in completed],
            "cancer_count": len(completed),
            "total_cells": sum(row["cells"] for row in completed),
            "total_association_evidence_rows": sum(
                row["association_evidence_rows"] for row in completed
            ),
            "per_cancer": completed,
            "all_raw_h5_full_sha_verified": True,
            "all_donor_aware": True,
            "all_typed_null_enforced": True,
            "historical_derived_results_used": False,
            "production_deployed": False,
            "port_8260_touched": False,
            "timestamp_unix": time.time(),
        }
        cohort["cohort_contract_sha256"] = canonical_sha256(cohort)
        exclusive_json(audit_root / "COHORT_SUCCESS.json", cohort)
        atomic_json(
            audit_root / "COHORT_STATUS.json",
            {
                **cohort,
                "format": FORMAT,
                "status": "SUCCESS_17_OF_17_AUDITED",
            },
        )
        print(json.dumps(cohort, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    finally:
        if lock.exists() and not lock.is_symlink():
            lock.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--audit-root", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(execute(build_parser().parse_args()))
