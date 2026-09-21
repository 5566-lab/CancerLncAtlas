#!/usr/bin/env python3
"""Run only SARC r6 under the unchanged 512 MiB child-process gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_R7_R6_SARC_GATE_V1"
FAILURE_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_R7_R6_SARC_TYPED_FAILURE_V1"
ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
SOURCE_GENERATION = "V3.2_R7_FRESH_FROM_RAW_H5"
CANCER = "SARC"
MEMORY_LIMIT_BYTES = 512 * 1024**2
POLL_SECONDS = 0.25
PYTHON = Path("${PRIVATE_WORK_ROOT}/miniconda3/bin/python")
PYTHON_RESOLVED = Path("${PRIVATE_WORK_ROOT}/miniconda3/bin/python3.13")
PYTHON_TRUST_ROOT = Path("${PRIVATE_WORK_ROOT}/miniconda3")
PYTHON_SHA256 = "dcb43d1acbc001b6ca88ed41f47273d53b4658766875852ac1ede3003bba9329"
PYTHON_STAT = {
    "size": 35_604_512,
    "mode": 0o100775,
    "uid": 1001,
    "gid": 1001,
    "dev": 2081,
    "inode": 34_359_900_582,
}
TOOL_ROOT = Path(
    "./data/CancerLncAtlas/runtime/tools/"
    "single_cell_r7_streaming_20260829_r6"
)
RUNNER = TOOL_ROOT / "scripts/run_v32_single_cell_r7_streaming.py"
R7_ROOT = Path(
    "./data/CancerLncAtlas/runtime/"
    "single_cell_cell_level_r7_portable_supersession_20260829"
)
PREFLIGHT = Path(
    "./data/CancerLncAtlas/runtime/audits/"
    "single_cell_r7_server_preflight_20260829_r2.json"
)
PREFLIGHT_SHA256 = (
    "a1565276f010e3d2a184aa27baccf31da7fb2b16cfa20690af81553c7acbb670"
)
RUN_STATUS = R7_ROOT / "RUN_STATUS.json"
RUN_STATUS_SHA256 = (
    "963268d03448ec514646dc199e3c4648faa1d47884f62faacfdd72aa23d6e2be"
)
TOOL_MANIFEST_SHA256 = (
    "910c52e615df271672a6bf4360c5acfbcdc2ba4d66604e768b62e68fafbf913a"
)
TOOL_SHAS = {
    "cc_hhgt/__init__.py": "e246acd89da9a9733fbac09b2770161b5ed199695a93a832ca965d8d201b78dc",
    "cc_hhgt/v32/__init__.py": "359f958ad6e7e67078201282718193b40bc0d09bf15d27cbefb3a604b535ca91",
    "cc_hhgt/v32/contracts.py": "387e64d4931bac936e7affa4a3f974812dbfbb16d50deacd670685c796d41f46",
    "cc_hhgt/v32/single_cell_cell_level.py": "08e5b3519539fe488c78626aab7a7ef94bad4f4ef7f94b2f656f029bd5056815",
    "cc_hhgt/v32/single_cell_r7_streaming.py": "7133182f386ca74ea8b36013f16ef7679cb95306949cff8eb91ffee24ee36fd3",
    "scripts/run_v32_single_cell_r7_streaming.py": "a271a19c1282741f3c6406de6c707ee06201ab30b53cad5e2486dd2f85cec967",
    "audit/MEMORY_ROOT_CAUSE_AND_REMEDIATION.json": "f8ad8a6c182f109a78737da61aecb0797d26cf8cf5fe26342cbb8fdc43b0665d",
    "audit/MEMORY_ROOT_CAUSE_AND_REMEDIATION.md": "6196263f29852efbab6b92fbf4f01dd7497fccaf597c36a6b44f89fd5ee841a8",
}
EXPECTED_OUTPUTS = {
    "RESOURCE_ESTIMATE.json",
    "association_context_availability.parquet",
    "association_evidence.parquet",
    "association_lncrna_testability.parquet",
    "lncrna_donor_celltype_summary.parquet",
    "pathway_availability.parquet",
    "pathway_donor_celltype_summary.parquet",
}


class SarcGateError(RuntimeError):
    """Raised when the immutable SARC gate contract fails."""


def sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise SarcGateError(f"cannot hash absent/unsafe file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SarcGateError(f"JSON is absent/unsafe: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def exclusive_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise SarcGateError(f"unsafe JSON temporary exists: {temporary}")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def verify_python_identity() -> dict[str, Any]:
    if str(PYTHON) != "${PRIVATE_WORK_ROOT}/miniconda3/bin/python":
        raise SarcGateError("pinned Python link path drift")
    if not PYTHON.is_symlink() or not PYTHON.is_file():
        raise SarcGateError("pinned Python must be the expected executable symlink")
    resolved = PYTHON.resolve(strict=True)
    trust_root = PYTHON_TRUST_ROOT.resolve(strict=True)
    if resolved != PYTHON_RESOLVED or trust_root not in resolved.parents:
        raise SarcGateError("pinned Python resolves outside the trusted conda root")
    observed = resolved.stat()
    identity = {
        "link_path": str(PYTHON),
        "link_target": os.readlink(PYTHON),
        "resolved_target": str(resolved),
        "trusted_conda_root": str(trust_root),
        "sha256": sha256_file(resolved),
        "size": int(observed.st_size),
        "mode": int(observed.st_mode),
        "uid": int(observed.st_uid),
        "gid": int(observed.st_gid),
        "dev": int(observed.st_dev),
        "inode": int(observed.st_ino),
        "regular_file": stat.S_ISREG(observed.st_mode),
        "owner_executable": bool(observed.st_mode & stat.S_IXUSR),
    }
    for key, expected in PYTHON_STAT.items():
        if identity[key] != expected:
            raise SarcGateError(f"pinned Python stat drift: {key}")
    if identity["sha256"] != PYTHON_SHA256:
        raise SarcGateError("pinned Python SHA drift")
    if identity["regular_file"] is not True or identity["owner_executable"] is not True:
        raise SarcGateError("pinned Python target is not a regular executable")
    identity["identity_contract_sha256"] = canonical_sha256(identity)
    return identity


def verify_frozen_inputs() -> dict[str, Any]:
    python_identity = verify_python_identity()
    for relative, expected in TOOL_SHAS.items():
        observed = sha256_file(TOOL_ROOT / relative)
        if observed != expected:
            raise SarcGateError(f"r6 tool SHA drift: {relative}")
    if sha256_file(PREFLIGHT) != PREFLIGHT_SHA256:
        raise SarcGateError("server preflight SHA drift")
    if sha256_file(RUN_STATUS) != RUN_STATUS_SHA256:
        raise SarcGateError("r7 RUN_STATUS SHA drift")
    return python_identity


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


def run_monitored(argv: list[str], *, log_path: Path) -> dict[str, Any]:
    if log_path.exists() or log_path.is_symlink():
        raise SarcGateError(f"log reuse is forbidden: {log_path}")
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
            if max(rss, high_water) > MEMORY_LIMIT_BYTES:
                memory_exceeded = True
                terminate_process_group(process)
                break
            time.sleep(POLL_SECONDS)
        return_code = process.wait()
        log.flush()
        os.fsync(log.fileno())
    return {
        "argv": argv,
        "return_code": int(return_code),
        "elapsed_seconds": round(time.time() - started, 3),
        "maximum_rss_bytes_observed": int(maximum_rss),
        "maximum_high_water_bytes_observed": int(maximum_hwm),
        "formal_gate_value_bytes": int(max(maximum_rss, maximum_hwm)),
        "memory_limit_bytes": MEMORY_LIMIT_BYTES,
        "memory_exceeded": memory_exceeded,
        "memory_metric": "MAX_CHILD_PID_VMRSS_OR_VMHWM_NOT_PROCESS_GROUP_SUM",
        "process_group_rss_sum_measured": False,
        "poll_seconds": POLL_SECONDS,
        "log_path": str(log_path),
        "log_sha256": sha256_file(log_path),
    }


def runner_args(
    *, mode: str, plan_path: Path | None, output_root: Path | None
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
        PREFLIGHT_SHA256,
        "--expected-run-status-sha256",
        RUN_STATUS_SHA256,
        "--cancer-id",
        CANCER,
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
        "32",
    ]
    if mode == "plan":
        if plan_path is None or output_root is not None:
            raise SarcGateError("invalid plan argv request")
        argv.extend(["--plan-output-json", str(plan_path)])
    elif mode == "run":
        if output_root is None or plan_path is not None:
            raise SarcGateError("invalid run argv request")
        argv.extend(["--output-parent", str(output_root)])
    else:
        raise SarcGateError(f"unsupported mode: {mode}")
    return argv


def typed_failure(
    *, output_root: Path, audit_root: Path, stage: str, reason: str, detail: Any
) -> dict[str, Any]:
    failure = {
        "format": FAILURE_FORMAT,
        "status": "TYPED_FAILURE_STOPPED",
        "analysis_version": ANALYSIS_VERSION,
        "cancer_id": CANCER,
        "stage": stage,
        "reason": reason,
        "detail": detail,
        "output_root": str(output_root),
        "audit_root": str(audit_root),
        "memory_limit_bytes": MEMORY_LIMIT_BYTES,
        "memory_limit_increased": False,
        "remaining_cancers_started": False,
        "hnsc_overwritten": False,
        "port_8260_touched": False,
        "production_deployed": False,
        "tool_manifest_sha256": TOOL_MANIFEST_SHA256,
        "timestamp_unix": time.time(),
    }
    failure["failure_contract_sha256"] = canonical_sha256(failure)
    return failure


def validate_plan(plan: dict[str, Any]) -> None:
    estimate = plan.get("resource_estimate", {})
    gates = {
        "cancer_id": plan.get("cancer_id") == CANCER,
        "source_generation": plan.get("source_generation") == SOURCE_GENERATION,
        "full_sha": plan.get("raw_h5_full_sha256_verified") is True,
        "donor": plan.get("donor_is_biological_replicate") is True,
        "no_cell_matrix": plan.get("no_cell_level_pathway_output") is True,
        "external_bh": estimate.get("association_bh_external_spill_enabled") is True,
        "no_python_evidence": estimate.get("association_evidence_accumulated_in_python") is False,
        "runtime_baseline": estimate.get("runtime_baseline_budget_bytes") == 352 * 1024**2,
        "planned_under_gate": 0 < int(estimate.get("estimated_peak_ram_bytes", 0)) <= MEMORY_LIMIT_BYTES,
    }
    failed = sorted(key for key, passed in gates.items() if not passed)
    if failed:
        raise SarcGateError(f"SARC r6 plan gate failed: {failed}")


def parquet_rows(path: Path) -> int:
    if path.is_symlink() or not path.is_file():
        raise SarcGateError(f"Parquet is absent/unsafe: {path}")
    return int(pq.ParquetFile(path).metadata.num_rows)


def audit_output(output_root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    cancer_root = output_root / f"cancer_id={CANCER}"
    success_path = cancer_root / "SUCCESS.json"
    lineage_path = cancer_root / "LINEAGE.json"
    manifest_path = cancer_root / "FILE_MANIFEST.parquet"
    success = load_json(success_path)
    lineage = load_json(lineage_path)
    if success.get("status") != "SUCCESS" or success.get("cancer_id") != CANCER:
        raise SarcGateError("SARC SUCCESS contract drift")
    required_true = {
        "full_input_sha256_verified_before_matrix_access",
        "raw_h5_full_sha256_verified",
    }
    if any(success.get(key) is not True for key in required_true):
        raise SarcGateError("SARC fresh full-SHA gate is absent")
    if success.get("historical_derived_results_used") is not False:
        raise SarcGateError("SARC used historical derived results")
    if success.get("cell_as_independent_replicate") is not False:
        raise SarcGateError("SARC used cells as replicates")
    if success.get("association_engine") != "DUCKDB_EXTERNAL_BH_SORT_V1":
        raise SarcGateError("SARC did not use the bounded external BH engine")
    if success.get("association_duckdb_memory_limit") != "64MB":
        raise SarcGateError("SARC DuckDB memory limit drift")
    if int(success.get("parquet_batch_rows", 0)) != 16_384:
        raise SarcGateError("SARC Parquet batch drift")
    manifest_sha = sha256_file(manifest_path)
    if success.get("file_manifest_sha256") != manifest_sha:
        raise SarcGateError("SUCCESS/manifest SHA drift")
    if lineage.get("file_manifest_sha256") != manifest_sha:
        raise SarcGateError("LINEAGE/manifest SHA drift")
    if success.get("lineage_sha256") != sha256_file(lineage_path):
        raise SarcGateError("SUCCESS/LINEAGE SHA drift")
    manifest = pd.read_parquet(manifest_path)
    if set(manifest.relative_path.astype(str)) != EXPECTED_OUTPUTS:
        raise SarcGateError("SARC FILE_MANIFEST membership drift")
    for row in manifest.itertuples(index=False):
        path = cancer_root / str(row.relative_path)
        if int(row.bytes) != path.stat().st_size or str(row.sha256) != sha256_file(path):
            raise SarcGateError(f"SARC output hash drift: {row.relative_path}")
    observed = {
        "association_evidence": parquet_rows(
            cancer_root / "association_evidence.parquet"
        ),
        "association_testability": parquet_rows(
            cancer_root / "association_lncrna_testability.parquet"
        ),
        "lncrna_summary": parquet_rows(
            cancer_root / "lncrna_donor_celltype_summary.parquet"
        ),
        "pathway_summary": parquet_rows(
            cancer_root / "pathway_donor_celltype_summary.parquet"
        ),
    }
    if observed["association_evidence"] != int(success["association_evidence_rows"]):
        raise SarcGateError("SARC evidence row count drift")
    if observed["association_testability"] != int(success["lncrnas"]) * 3:
        raise SarcGateError("SARC testability row count drift")
    if int(success["cells"]) != int(plan["kept_cells"]):
        raise SarcGateError("SARC cell count differs from plan")
    return {
        "cancer_id": CANCER,
        "status": "SUCCESS_AUDITED",
        "output_root": str(cancer_root),
        "cells": int(success["cells"]),
        "donors": int(success["donors"]),
        "lncrnas": int(success["lncrnas"]),
        "pathways_total": int(success["pathways_total"]),
        "pathways_available": int(success["pathways_available"]),
        "pathways_typed_unavailable": int(success["pathways_typed_unavailable"]),
        "association_evidence_rows": int(success["association_evidence_rows"]),
        "contract_sha256": str(success["contract_sha256"]),
        "success_sha256": sha256_file(success_path),
        "lineage_sha256": sha256_file(lineage_path),
        "file_manifest_sha256": manifest_sha,
        "parquet_rows": observed,
        "fresh_full_sha_verified": True,
        "historical_derived_results_used": False,
        "donor_aware": True,
        "typed_null_enforced": True,
        "association_engine": success["association_engine"],
        "production_deployed": False,
        "port_8260_touched": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--audit-root", required=True, type=Path)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    audit_root = args.audit_root.resolve()
    authorized = Path("./data/CancerLncAtlas")
    if authorized not in output_root.parents or authorized not in audit_root.parents:
        raise SarcGateError("SARC gate paths leave the authorized ${PRIVATE_WORK_ROOT} tree")
    if output_root.exists() or output_root.is_symlink():
        raise SarcGateError(f"new immutable output root already exists: {output_root}")
    if audit_root.exists() or audit_root.is_symlink():
        raise SarcGateError(f"new immutable audit root already exists: {audit_root}")
    audit_root.mkdir(parents=True)
    (audit_root / "logs").mkdir()
    (audit_root / "plans").mkdir()
    exclusive_json(
        audit_root / "SARC_GATE_RUNNING.json",
        {
            "format": FORMAT,
            "status": "RUNNING",
            "cancer_id": CANCER,
            "memory_limit_bytes": MEMORY_LIMIT_BYTES,
            "memory_limit_increased": False,
            "memory_metric": "MAX_CHILD_PID_VMRSS_OR_VMHWM_NOT_PROCESS_GROUP_SUM",
            "process_group_rss_sum_measured": False,
            "tool_manifest_sha256": TOOL_MANIFEST_SHA256,
            "hnsc_overwritten": False,
            "remaining_cancers_started": False,
            "port_8260_touched": False,
            "production_deployed": False,
            "timestamp_unix": time.time(),
        },
    )
    stage = "VERIFY"
    try:
        python_identity = verify_frozen_inputs()
        exclusive_json(
            audit_root / "PINNED_PYTHON_IDENTITY.json", python_identity
        )
        stage = "PLAN"
        plan_path = audit_root / "plans/SARC.json"
        plan_process = run_monitored(
            runner_args(mode="plan", plan_path=plan_path, output_root=None),
            log_path=audit_root / "logs/SARC.plan.log",
        )
        if plan_process["memory_exceeded"]:
            failure = typed_failure(
                output_root=output_root,
                audit_root=audit_root,
                stage=stage,
                reason="OBSERVED_PROCESS_MEMORY_EXCEEDED_512_MIB",
                detail=plan_process,
            )
            exclusive_json(audit_root / "TYPED_FAILURE.json", failure)
            print(json.dumps(failure, ensure_ascii=False, indent=2, sort_keys=True))
            return 2
        if plan_process["return_code"] != 0:
            raise SarcGateError(f"SARC plan returned {plan_process['return_code']}")
        plan = load_json(plan_path)
        validate_plan(plan)
        atomic_json(
            audit_root / "SARC_GATE_STATUS.json",
            {
                "format": FORMAT,
                "status": "PLAN_PASSED_RUN_STARTING",
                "cancer_id": CANCER,
                "planned_peak_ram_bytes": int(
                    plan["resource_estimate"]["estimated_peak_ram_bytes"]
                ),
                "memory_limit_bytes": MEMORY_LIMIT_BYTES,
                "plan_sha256": sha256_file(plan_path),
                "plan_process": plan_process,
                "pinned_python_identity": python_identity,
                "timestamp_unix": time.time(),
            },
        )
        stage = "RUN"
        run_process = run_monitored(
            runner_args(mode="run", plan_path=None, output_root=output_root),
            log_path=audit_root / "logs/SARC.run.log",
        )
        if run_process["memory_exceeded"]:
            failure = typed_failure(
                output_root=output_root,
                audit_root=audit_root,
                stage=stage,
                reason="OBSERVED_PROCESS_MEMORY_EXCEEDED_512_MIB",
                detail=run_process,
            )
            exclusive_json(audit_root / "TYPED_FAILURE.json", failure)
            print(json.dumps(failure, ensure_ascii=False, indent=2, sort_keys=True))
            return 2
        if run_process["return_code"] != 0:
            raise SarcGateError(f"SARC run returned {run_process['return_code']}")
        stage = "OUTPUT_AUDIT"
        record = audit_output(output_root, plan)
        success = {
            "format": FORMAT,
            "status": "SUCCESS_AUDITED",
            "analysis_version": ANALYSIS_VERSION,
            "cancer_id": CANCER,
            "output_root": str(output_root),
            "audit_root": str(audit_root),
            "memory_limit_bytes": MEMORY_LIMIT_BYTES,
            "memory_limit_increased": False,
            "memory_metric": "MAX_CHILD_PID_VMRSS_OR_VMHWM_NOT_PROCESS_GROUP_SUM",
            "process_group_rss_sum_measured": False,
            "planned_peak_ram_bytes": int(
                plan["resource_estimate"]["estimated_peak_ram_bytes"]
            ),
            "plan_sha256": sha256_file(plan_path),
            "plan_process": plan_process,
            "run_process": run_process,
            "pinned_python_identity": python_identity,
            "record": record,
            "tool_manifest_sha256": TOOL_MANIFEST_SHA256,
            "hnsc_overwritten": False,
            "remaining_cancers_started": False,
            "port_8260_touched": False,
            "production_deployed": False,
            "timestamp_unix": time.time(),
        }
        success["sarc_gate_contract_sha256"] = canonical_sha256(success)
        exclusive_json(audit_root / "SARC_GATE_SUCCESS.json", success)
        atomic_json(audit_root / "SARC_GATE_STATUS.json", success)
        print(json.dumps(success, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        failure_path = audit_root / "TYPED_FAILURE.json"
        if not failure_path.exists():
            failure = typed_failure(
                output_root=output_root,
                audit_root=audit_root,
                stage=stage,
                reason="SUPERVISOR_OR_RUNNER_CONTRACT_ERROR",
                detail={"error_type": type(exc).__name__, "message": str(exc)},
            )
            exclusive_json(failure_path, failure)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
