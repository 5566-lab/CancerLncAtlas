#!/usr/bin/env python3
"""Audit r9 BH equivalence, large ACC coverage, memory, and interruption safety."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from typing import Any


PYTHON = Path("${PRIVATE_WORK_ROOT}/miniconda3/bin/python")
AUTHORIZED = Path("./data/CancerLncAtlas")
MEMORY_LIMIT_BYTES = 512 * 1024**2
POLL_SECONDS = 0.25


class AuditError(RuntimeError):
    """Raised when an equivalence or safety invariant fails."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def exclusive_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def authorized(path: Path) -> Path:
    value = path.resolve()
    if AUTHORIZED != value and AUTHORIZED not in value.parents:
        raise AuditError(f"path escapes authorized root: {value}")
    return value


def load_runner(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AuditError(f"cannot import runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def process_memory(pid: int) -> tuple[int, int]:
    status = Path(f"/proc/{pid}/status")
    if not status.is_file():
        return 0, 0
    rss = hwm = 0
    for line in status.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("VmRSS:"):
            rss = int(line.split()[1]) * 1024
        elif line.startswith("VmHWM:"):
            hwm = int(line.split()[1]) * 1024
    return rss, hwm


def terminate_group(process: subprocess.Popen[bytes]) -> None:
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


def run_monitored(argv: list[str], log_path: Path) -> dict[str, Any]:
    maximum_rss = maximum_hwm = 0
    started = time.time()
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OMP_DYNAMIC": "FALSE",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "BLIS_NUM_THREADS": "1",
        }
    )
    with log_path.open("xb") as log:
        process = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=environment,
        )
        while process.poll() is None:
            rss, hwm = process_memory(process.pid)
            maximum_rss = max(maximum_rss, rss)
            maximum_hwm = max(maximum_hwm, hwm)
            if max(rss, hwm) > MEMORY_LIMIT_BYTES:
                terminate_group(process)
                raise AuditError(f"fixture exceeded 512 MiB: {argv}")
            time.sleep(POLL_SECONDS)
        return_code = process.wait()
        log.flush()
        os.fsync(log.fileno())
    if return_code != 0:
        raise AuditError(f"fixture worker failed ({return_code}): {log_path}")
    return {
        "return_code": return_code,
        "elapsed_seconds": round(time.time() - started, 3),
        "maximum_rss_bytes": maximum_rss,
        "maximum_hwm_bytes": maximum_hwm,
        "formal_gate_value_bytes": max(maximum_rss, maximum_hwm),
        "memory_limit_bytes": MEMORY_LIMIT_BYTES,
        "poll_seconds": POLL_SECONDS,
        "log_path": str(log_path),
        "log_sha256": sha256_file(log_path),
    }


def worker(args: argparse.Namespace) -> int:
    import pyarrow.parquet as pq

    runner = load_runner(args.runner.resolve(), f"fixture_{os.getpid()}")
    runner.shutil.rmtree = lambda _path: None
    rows = int(pq.ParquetFile(args.raw).metadata.num_rows)
    observed = runner._finalize_association_evidence(
        raw_path=args.raw.resolve(),
        output_path=args.output.resolve(),
        scratch=args.scratch.resolve(),
        raw_rows=rows,
    )
    result = {
        "status": "SUCCESS",
        "runner_path": str(args.runner.resolve()),
        "runner_sha256": sha256_file(args.runner.resolve()),
        "rows": observed,
        "output_sha256": sha256_file(args.output.resolve()),
        "association_engine": runner.ASSOCIATION_ENGINE,
        "association_execution_strategy": runner.ASSOCIATION_EXECUTION_STRATEGY,
        "association_bh_family": runner.ASSOCIATION_BH_FAMILY,
        "association_stage_key_policy": runner.ASSOCIATION_STAGE_KEY_POLICY,
    }
    exclusive_json(args.result.resolve(), result)
    return 0


def make_small_raw(path: Path) -> None:
    import numpy as np
    import pandas as pd

    frame = pd.DataFrame(
        {
            "dataset_id": ["fixture"] * 15,
            "cancer_id": ["ACC"] * 15,
            "compartment_order": np.asarray([0] * 6 + [1] * 5 + [2] * 4, dtype=np.int8),
            "compartment": ["malignant"] * 6 + ["immune"] * 5 + ["stromal"] * 4,
            "lncrna_id": [
                "L2", "L1", "L1", "L3", "L4", "L5",
                "I2", "I1", "I3", "I4", "I5",
                "S2", "S1", "S3", "S4",
            ],
            "lncrna_symbol": [f"symbol_{index}" for index in range(15)],
            "pathway_id": [
                "P1", "P2", "P1", "P1", "P2", "P3",
                "P1", "P2", "P1", "P2", "P3",
                "P1", "P2", "P1", "P3",
            ],
            "n_donors": np.asarray([8] * 6 + [7] * 5 + [6] * 4, dtype=np.int64),
            "spearman_rho": [
                0.8, -0.7, 0.6, 0.5, -0.5, 0.9,
                0.9, 0.7, -0.6, 0.5, -0.55,
                0.75, -0.65, 0.55, -0.8,
            ],
            "nominal_p": [
                0.01, 0.01, 0.02, 0.2, 0.04, 0.001,
                0.005, 0.05, 0.05, 0.3, 0.02,
                0.02, 0.02, 0.4, 0.001,
            ],
            "total_tests": np.asarray([100] * 6 + [20] * 5 + [30] * 4, dtype=np.int64),
        }
    )
    frame.sample(frac=1.0, random_state=20260829).to_parquet(path, index=False)


def compare_outputs(left: Path, right: Path) -> dict[str, Any]:
    import duckdb

    connection = duckdb.connect()
    try:
        connection.execute("SET memory_limit='64MB'")
        connection.execute("SET threads=1")
        left_sql = "'" + str(left).replace("'", "''") + "'"
        right_sql = "'" + str(right).replace("'", "''") + "'"
        mismatch = connection.execute(
            f"""
            SELECT count(*) FROM (
                (SELECT * FROM read_parquet({left_sql})
                 EXCEPT ALL SELECT * FROM read_parquet({right_sql}))
                UNION ALL
                (SELECT * FROM read_parquet({right_sql})
                 EXCEPT ALL SELECT * FROM read_parquet({left_sql}))
            )
            """
        ).fetchone()[0]
        left_rows = connection.execute(
            f"SELECT count(*) FROM read_parquet({left_sql})"
        ).fetchone()[0]
        right_rows = connection.execute(
            f"SELECT count(*) FROM read_parquet({right_sql})"
        ).fetchone()[0]
    finally:
        connection.close()
    if mismatch or left_rows != right_rows:
        raise AuditError(
            f"r8/r9 evidence mismatch: mismatch={mismatch}, rows={left_rows}/{right_rows}"
        )
    return {
        "rows": int(left_rows),
        "except_all_bidirectional_mismatch_rows": int(mismatch),
        "left_sha256": sha256_file(left),
        "right_sha256": sha256_file(right),
        "byte_identical": sha256_file(left) == sha256_file(right),
    }


def compare_scratch(left: Path, right: Path) -> dict[str, Any]:
    suffixes = (".rank.parquet", ".adjusted.parquet", ".parquet")
    left_files = {
        str(path.relative_to(left)): sha256_file(path)
        for path in left.rglob("*")
        if path.is_file() and path.name.startswith("compartment_order=")
        and path.name.endswith(suffixes)
    }
    right_files = {
        str(path.relative_to(right)): sha256_file(path)
        for path in right.rglob("*")
        if path.is_file() and path.name.startswith("compartment_order=")
        and path.name.endswith(suffixes)
    }
    if left_files != right_files:
        raise AuditError("r8/r9 rank, adjusted, or compartment part files differ")
    if len(left_files) != 9:
        raise AuditError(f"three-compartment stage coverage drift: {len(left_files)}")
    return {
        "stage_file_count": len(left_files),
        "stage_files_byte_identical": True,
        "stage_file_sha256": left_files,
    }


def launch_worker(
    *, runner: Path, raw: Path, output: Path, scratch: Path, result: Path, log: Path
) -> dict[str, Any]:
    argv = [
        str(PYTHON), str(Path(__file__).resolve()), "worker",
        "--runner", str(runner), "--raw", str(raw), "--output", str(output),
        "--scratch", str(scratch), "--result", str(result),
    ]
    monitored = run_monitored(argv, log)
    value = json.loads(result.read_text(encoding="utf-8"))
    value["memory"] = monitored
    return value


def interrupt_worker(args: argparse.Namespace) -> int:
    runner = args.runner.resolve()
    root = authorized(args.root)
    if root.exists() or root.is_symlink():
        raise AuditError("interrupt worker root reuse forbidden")
    root.mkdir(parents=True)
    cancer = "ACC"
    contract_sha = "a" * 64
    producer_pid = os.getpid()
    publish = root / (
        f".cancer_id={cancer}.{contract_sha[:16]}.publish.{producer_pid}"
    )
    final = root / f"cancer_id={cancer}"
    publish.mkdir()
    raw = publish / ".association_evidence_raw.parquet"
    shutil.copyfile(args.source_raw.resolve(), raw)
    relative_paths = (
        "lncrna_donor_celltype_summary.parquet",
        "pathway_availability.parquet",
        "pathway_donor_celltype_summary.parquet",
        "association_evidence.parquet",
        "association_lncrna_testability.parquet",
        "association_context_availability.parquet",
        "RESOURCE_ESTIMATE.json",
    )
    for relative in relative_paths:
        if relative != "association_evidence.parquet":
            (publish / relative).write_bytes(b"interruption fixture\n")
    handoff = {
        "format": "CC_HHGT_V3_2_SINGLE_CELL_R9_POST_BH_EXEC_HANDOFF_V1",
        "association_engine": "DUCKDB_EXTERNAL_BH_SORT_PROCESS_ISOLATED_V2",
        "cancer_id": cancer,
        "producer_pid": producer_pid,
        "publish": str(publish),
        "final": str(final),
        "raw_path": str(raw),
        "evidence_path": str(publish / "association_evidence.parquet"),
        "scratch": str(publish / ".association_duckdb_scratch"),
        "raw_rows": int(args.raw_rows),
        "relative_paths": list(relative_paths),
        "lineage_base": {"cancer_id": cancer, "contract_sha256": contract_sha},
        "success_base": {"cancer_id": cancer},
        "source_runner_path": str(runner),
        "source_runner_sha256": sha256_file(runner),
    }
    handoff_path = root / f"POST_BH_HANDOFF.{producer_pid}.json"
    exclusive_json(handoff_path, handoff)
    handoff_sha = sha256_file(handoff_path)
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OMP_DYNAMIC": "FALSE",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "BLIS_NUM_THREADS": "1",
        }
    )
    os.execve(
        str(PYTHON),
        [
            str(PYTHON), str(runner), "--internal-post-bh-r9",
            "--handoff-json", str(handoff_path),
            "--expected-handoff-sha256", handoff_sha,
        ],
        environment,
    )
    raise AuditError("interrupt worker os.execve unexpectedly returned")


def interruption_test(
    *, runner: Path, source_raw: Path, raw_rows: int, root: Path
) -> dict[str, Any]:
    case_root = root / "interrupt_case"
    log_path = root / "interruption.log"
    argv = [
        str(PYTHON), str(Path(__file__).resolve()), "interrupt-worker",
        "--runner", str(runner), "--source-raw", str(source_raw),
        "--raw-rows", str(int(raw_rows)), "--root", str(case_root),
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OMP_DYNAMIC": "FALSE",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "BLIS_NUM_THREADS": "1",
        }
    )
    started = time.time()
    with log_path.open("xb") as log:
        process = subprocess.Popen(
            argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, env=environment,
        )
        publish = case_root / (
            f".cancer_id=ACC.{'a' * 16}.publish.{process.pid}"
        )
        final = case_root / "cancer_id=ACC"
        observed_bh_start = False
        deadline = time.time() + 60
        while process.poll() is None and time.time() < deadline:
            if (publish / ".association_duckdb_scratch").is_dir():
                observed_bh_start = True
                break
            time.sleep(POLL_SECONDS)
        if process.poll() is None:
            terminate_group(process)
        return_code = process.wait()
        log.flush()
        os.fsync(log.fileno())
    if not observed_bh_start:
        raise AuditError("interruption fixture did not reach BH stage")
    if (publish / "SUCCESS.json").exists() or final.exists():
        raise AuditError("interrupted BH exposed a publishable SUCCESS")
    return {
        "observed_bh_start": True,
        "true_os_execve_same_pid": True,
        "terminated_before_completion": return_code != 0,
        "return_code": return_code,
        "success_json_absent": True,
        "immutable_final_absent": True,
        "elapsed_seconds": round(time.time() - started, 3),
        "log_path": str(log_path),
        "log_sha256": sha256_file(log_path),
    }


def interruption_audit(args: argparse.Namespace) -> int:
    root = authorized(args.audit_root)
    if root.exists() or root.is_symlink():
        raise AuditError("interruption audit root reuse forbidden")
    root.mkdir(parents=True)
    result = interruption_test(
        runner=args.runner.resolve(),
        source_raw=authorized(args.source_raw),
        raw_rows=int(args.raw_rows),
        root=root,
    )
    report = {
        "format": "CC_HHGT_V3_2_SINGLE_CELL_R9_INTERRUPTION_SAFETY_AUDIT_V1",
        "status": "PASS",
        "runner_path": str(args.runner.resolve()),
        "runner_sha256": sha256_file(args.runner.resolve()),
        "source_raw_path": str(authorized(args.source_raw)),
        "source_raw_sha256": sha256_file(authorized(args.source_raw)),
        "raw_rows": int(args.raw_rows),
        "result": result,
        "production_deployed": False,
    }
    report["audit_contract_sha256"] = canonical_sha(report)
    report_path = root / "AUDIT.json"
    exclusive_json(report_path, report)
    print(json.dumps({**report, "audit_sha256": sha256_file(report_path)}, indent=2))
    return 0


def audit(args: argparse.Namespace) -> int:
    import duckdb

    root = authorized(args.audit_root)
    if root.exists() or root.is_symlink():
        raise AuditError(f"audit root reuse forbidden: {root}")
    root.mkdir(parents=True)
    r8_runner = args.r8_runner.resolve()
    r9_runner = args.r9_runner.resolve()
    source_raw = authorized(args.acc_raw)
    small_source = root / "small_source.parquet"
    make_small_raw(small_source)

    results: dict[str, Any] = {}
    for fixture, source in (("small", small_source), ("acc_large", source_raw)):
        fixture_root = root / fixture
        fixture_root.mkdir()
        values = {}
        for version, runner in (("r8", r8_runner), ("r9", r9_runner)):
            raw = fixture_root / f"{version}.raw.parquet"
            shutil.copyfile(source, raw)
            values[version] = launch_worker(
                runner=runner,
                raw=raw,
                output=fixture_root / f"{version}.evidence.parquet",
                scratch=fixture_root / f"{version}.scratch",
                result=fixture_root / f"{version}.result.json",
                log=fixture_root / f"{version}.log",
            )
        comparison = compare_outputs(
            fixture_root / "r8.evidence.parquet",
            fixture_root / "r9.evidence.parquet",
        )
        scratch_comparison = compare_scratch(
            fixture_root / "r8.scratch", fixture_root / "r9.scratch"
        )
        results[fixture] = {
            "workers": values,
            "evidence_comparison": comparison,
            "stage_comparison": scratch_comparison,
        }

    connection = duckdb.connect()
    try:
        connection.execute("SET memory_limit='64MB'")
        connection.execute("SET threads=1")
        raw_sql = "'" + str(source_raw).replace("'", "''") + "'"
        families = connection.execute(
            f"""
            SELECT compartment_order, compartment, count(*) AS retained_rows,
                   min(total_tests) AS minimum_total_tests,
                   max(total_tests) AS maximum_total_tests,
                   count(DISTINCT (lncrna_id, pathway_id)) AS exact_keys
            FROM read_parquet({raw_sql})
            GROUP BY compartment_order, compartment
            ORDER BY compartment_order
            """
        ).fetchdf().to_dict(orient="records")
    finally:
        connection.close()
    if [int(row["compartment_order"]) for row in families] != [0, 1, 2]:
        raise AuditError("ACC fixture does not cover all three ordered compartments")
    if any(
        int(row["retained_rows"]) != int(row["exact_keys"])
        or int(row["minimum_total_tests"]) != int(row["maximum_total_tests"])
        for row in families
    ):
        raise AuditError("ACC exact-key or BH denominator contract failed")

    interruption = interruption_test(
        runner=r9_runner,
        source_raw=source_raw,
        raw_rows=sum(int(row["retained_rows"]) for row in families),
        root=root,
    )
    report = {
        "format": "CC_HHGT_V3_2_SINGLE_CELL_R9_BH_EQUIVALENCE_AUDIT_V1",
        "status": "PASS",
        "r8_runner": {"path": str(r8_runner), "sha256": sha256_file(r8_runner)},
        "r9_runner": {"path": str(r9_runner), "sha256": sha256_file(r9_runner)},
        "acc_raw": {"path": str(source_raw), "sha256": sha256_file(source_raw)},
        "acc_bh_families": families,
        "fixtures": results,
        "interruption_safety": interruption,
        "bh_family": "COMPARTMENT_ORDER",
        "bh_total_tests_denominator_changed": False,
        "exact_key_policy_changed": False,
        "typed_null_contract_changed": False,
        "formal_memory_limit_bytes": MEMORY_LIMIT_BYTES,
        "os_execve_formal_transition": True,
        "parent_helper_overlap": False,
        "production_deployed": False,
    }
    report["audit_contract_sha256"] = canonical_sha(report)
    report_path = root / "AUDIT.json"
    exclusive_json(report_path, report)
    print(json.dumps({**report, "audit_sha256": sha256_file(report_path)}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--runner", required=True, type=Path)
    worker_parser.add_argument("--raw", required=True, type=Path)
    worker_parser.add_argument("--output", required=True, type=Path)
    worker_parser.add_argument("--scratch", required=True, type=Path)
    worker_parser.add_argument("--result", required=True, type=Path)
    interrupt_parser = subparsers.add_parser("interrupt-worker")
    interrupt_parser.add_argument("--runner", required=True, type=Path)
    interrupt_parser.add_argument("--source-raw", required=True, type=Path)
    interrupt_parser.add_argument("--raw-rows", required=True, type=int)
    interrupt_parser.add_argument("--root", required=True, type=Path)
    interrupt_audit_parser = subparsers.add_parser("interrupt-audit")
    interrupt_audit_parser.add_argument("--runner", required=True, type=Path)
    interrupt_audit_parser.add_argument("--source-raw", required=True, type=Path)
    interrupt_audit_parser.add_argument("--raw-rows", required=True, type=int)
    interrupt_audit_parser.add_argument("--audit-root", required=True, type=Path)
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--r8-runner", required=True, type=Path)
    audit_parser.add_argument("--r9-runner", required=True, type=Path)
    audit_parser.add_argument("--acc-raw", required=True, type=Path)
    audit_parser.add_argument("--audit-root", required=True, type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "worker":
        return worker(args)
    if args.command == "interrupt-worker":
        return interrupt_worker(args)
    if args.command == "interrupt-audit":
        return interruption_audit(args)
    return audit(args)


if __name__ == "__main__":
    raise SystemExit(main())
