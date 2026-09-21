#!/usr/bin/env python3
"""Prove the ${PRIVATE_WORK_ROOT} standalone runtime reproduces frozen R9 bytes.

The audit replays both the 15-row edge-case fixture and the 1,733,282-row ACC
fixture through the final R9 BH function.  Each replay runs in a fresh process
under the standalone interpreter, with temporary state inside the new audit
root.  The produced Parquet must be byte-identical and EXCEPT-ALL identical to
the frozen R9 output.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping


AUTHORIZED_ROOT = Path("./data/CancerLncAtlas")
FORMAT = "CANCERLNCATLAS_V32_SINGLE_CELL_R9_STANDALONE_EQUIVALENCE_V1"
FAILURE_FORMAT = (
    "CANCERLNCATLAS_V32_SINGLE_CELL_R9_STANDALONE_EQUIVALENCE_FAILURE_V1"
)
MEMORY_LIMIT_BYTES = 512 * 1024**2
POLL_SECONDS = 0.25
RUNNER = Path(
    "./data/CancerLncAtlas/runtime/tools/"
    "single_cell_r7_streaming_20260829_r9/scripts/"
    "run_v32_single_cell_r7_streaming.py"
)
RUNNER_SHA256 = "903658231bfe31387d12cab56b9de45ec88258e386f9be0e3ba442b61495932c"
FIXTURES = {
    "small": {
        "raw": (
            "./data/CancerLncAtlas/runtime/audits/"
            "single_cell_r9_equivalence_20260829_r1/small_source.parquet"
        ),
        "expected": (
            "./data/CancerLncAtlas/runtime/audits/"
            "single_cell_r9_equivalence_20260829_r1/small/r9.evidence.parquet"
        ),
        "rows": 15,
        "raw_sha256": (
            "588d4b019344ee3cace06e5e303242643938249710fe225221266ae632566172"
        ),
        "expected_output_sha256": (
            "0b96cf5da9e5bafeb85ae082bbd3dcfb83e54b4f00d8939139eff97482a19c61"
        ),
    },
    "acc_large": {
        "raw": (
            "./data/CancerLncAtlas/results/model/"
            "v32_single_cell_r7_fresh_streaming_r8_remaining15_20260829_r1/"
            ".cancer_id=ACC.7e5c2ff7735cca76.publish.1349364/"
            ".association_evidence_raw.parquet"
        ),
        "raw_sha256": (
            "0a9a0f7b8c355e58bf6affdf6019e4fb53de1fdf34f994f7448715ed5870d77d"
        ),
        "expected": (
            "./data/CancerLncAtlas/runtime/audits/"
            "single_cell_r9_equivalence_20260829_r1/acc_large/r9.evidence.parquet"
        ),
        "rows": 1_733_282,
        "expected_output_sha256": (
            "9863fe9d9d2dd7031548a471523c41a12fa395dac9f262ce8267b28a23cd96fc"
        ),
    },
}


class EquivalenceError(RuntimeError):
    """Raised when the standalone runtime changes a frozen R9 result."""


def sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise EquivalenceError(f"absent/unsafe file: {path}")
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


def exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise EquivalenceError(f"immutable JSON reuse forbidden: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def require_authorized(path: Path, *, exists: bool = True) -> Path:
    value = path.resolve(strict=exists)
    root = AUTHORIZED_ROOT.resolve(strict=True)
    try:
        value.relative_to(root)
    except ValueError as exc:
        raise EquivalenceError(f"path leaves authorized ${PRIVATE_WORK_ROOT} tree: {value}") from exc
    if value == root or "/dsk2" in str(value) or "/tmp" in str(value):
        raise EquivalenceError(f"forbidden equivalence path: {value}")
    return value


def load_runner(path: Path):
    spec = importlib.util.spec_from_file_location(
        f"v32_r9_standalone_fixture_{os.getpid()}", path
    )
    if spec is None or spec.loader is None:
        raise EquivalenceError(f"cannot load R9 runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def forbidden_loaded_paths() -> list[str]:
    values = {str(Path(value).resolve()) for value in sys.path if value}
    for module in tuple(sys.modules.values()):
        raw = getattr(module, "__file__", None)
        if raw:
            try:
                values.add(str(Path(raw).resolve()))
            except OSError:
                pass
    maps = Path("/proc/self/maps").read_text(
        encoding="utf-8", errors="replace"
    )
    values.update(
        field
        for line in maps.splitlines()
        for field in line.split()
        if field.startswith("/")
    )
    return sorted(
        value
        for value in values
        if "/dsk2" in value or value == "/tmp" or value.startswith("/tmp/")
    )


def worker(args: argparse.Namespace) -> int:
    import pyarrow.parquet as pq

    runner = require_authorized(args.runner)
    if sha256_file(runner) != RUNNER_SHA256:
        raise EquivalenceError("final R9 runner SHA drift")
    raw = require_authorized(args.raw)
    output = require_authorized(args.output, exists=False)
    scratch = require_authorized(args.scratch, exists=False)
    result_path = require_authorized(args.result, exists=False)
    if output.exists() or scratch.exists() or result_path.exists():
        raise EquivalenceError("standalone equivalence worker would reuse output")
    rows = int(pq.ParquetFile(raw).metadata.num_rows)
    if rows != int(args.expected_rows):
        raise EquivalenceError(f"fixture row drift: {rows} != {args.expected_rows}")
    module = load_runner(runner)
    observed = module._finalize_association_evidence(
        raw_path=raw,
        output_path=output,
        scratch=scratch,
        raw_rows=rows,
    )
    if int(observed) != rows:
        raise EquivalenceError("R9 finalizer returned a different row count")
    forbidden = forbidden_loaded_paths()
    if forbidden:
        raise EquivalenceError("standalone worker loaded forbidden paths: " + repr(forbidden))
    value = {
        "status": "SUCCESS",
        "rows": rows,
        "output_path": str(output),
        "output_bytes": output.stat().st_size,
        "output_sha256": sha256_file(output),
        "runner_path": str(runner),
        "runner_sha256": RUNNER_SHA256,
        "python_executable": str(Path(sys.executable).resolve()),
        "forbidden_loaded_paths": [],
        "association_engine": module.ASSOCIATION_ENGINE,
        "association_execution_strategy": module.ASSOCIATION_EXECUTION_STRATEGY,
        "association_bh_family": module.ASSOCIATION_BH_FAMILY,
        "association_stage_key_policy": module.ASSOCIATION_STAGE_KEY_POLICY,
    }
    exclusive_json(result_path, value)
    return 0


def process_memory(pid: int) -> tuple[int, int]:
    path = Path(f"/proc/{pid}/status")
    if not path.is_file():
        return 0, 0
    rss = hwm = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("VmRSS:"):
            rss = int(line.split()[1]) * 1024
        elif line.startswith("VmHWM:"):
            hwm = int(line.split()[1]) * 1024
    return rss, hwm


def terminate(process: subprocess.Popen[bytes]) -> None:
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


def worker_environment(temp_root: Path, tool_root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    for key in (
        "PYTHONPATH",
        "PYTHONHOME",
        "LD_LIBRARY_PATH",
        "CONDA_PREFIX",
        "CONDA_PYTHON_EXE",
        "_CE_CONDA",
        "_CE_M",
    ):
        environment.pop(key, None)
    environment.update(
        {
            "PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin",
            "PYTHONPATH": str(tool_root),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "TMPDIR": str(temp_root),
            "TMP": str(temp_root),
            "TEMP": str(temp_root),
            "PIP_CONFIG_FILE": os.devnull,
            "OMP_NUM_THREADS": "1",
            "OMP_DYNAMIC": "FALSE",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "BLIS_NUM_THREADS": "1",
        }
    )
    return environment


def run_worker(
    *, name: str, raw: Path, rows: int, root: Path, tool_root: Path
) -> dict[str, Any]:
    fixture_root = root / name
    fixture_root.mkdir(parents=True, exist_ok=False)
    temporary = fixture_root / "offline_temp"
    temporary.mkdir()
    disposable_raw = fixture_root / "source.copy.parquet"
    source_sha256 = sha256_file(raw)
    shutil.copyfile(raw, disposable_raw)
    if sha256_file(disposable_raw) != source_sha256:
        raise EquivalenceError(f"{name} disposable raw copy SHA drift")
    output = fixture_root / "standalone.evidence.parquet"
    scratch = fixture_root / "scratch"
    result = fixture_root / "worker.json"
    log_path = fixture_root / "worker.log"
    argv = [
        sys.executable,
        str(Path(__file__).resolve()),
        "worker",
        "--runner",
        str(RUNNER),
        "--raw",
        str(disposable_raw),
        "--output",
        str(output),
        "--scratch",
        str(scratch),
        "--result",
        str(result),
        "--expected-rows",
        str(rows),
    ]
    maximum_rss = maximum_hwm = 0
    started = time.time()
    with log_path.open("xb") as log:
        process = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=worker_environment(temporary, tool_root),
            start_new_session=True,
        )
        while process.poll() is None:
            rss, hwm = process_memory(process.pid)
            maximum_rss = max(maximum_rss, rss)
            maximum_hwm = max(maximum_hwm, hwm)
            if max(rss, hwm) > MEMORY_LIMIT_BYTES:
                terminate(process)
                raise EquivalenceError(f"{name} exceeded the frozen 512 MiB gate")
            time.sleep(POLL_SECONDS)
        return_code = process.wait()
        log.flush()
        os.fsync(log.fileno())
    if return_code != 0 or not result.is_file():
        raise EquivalenceError(
            f"{name} standalone worker failed ({return_code}); log={log_path}"
        )
    if disposable_raw.exists() or disposable_raw.is_symlink():
        raise EquivalenceError(f"{name} consuming finalizer left disposable raw behind")
    value = json.loads(result.read_text(encoding="utf-8"))
    value["frozen_source_path"] = str(raw)
    value["frozen_source_sha256"] = source_sha256
    value["disposable_raw_path"] = str(disposable_raw)
    value["disposable_raw_consumed"] = True
    value["memory"] = {
        "maximum_rss_bytes": maximum_rss,
        "maximum_hwm_bytes": maximum_hwm,
        "formal_gate_value_bytes": max(maximum_rss, maximum_hwm),
        "memory_limit_bytes": MEMORY_LIMIT_BYTES,
        "elapsed_seconds": round(time.time() - started, 3),
        "log_path": str(log_path),
        "log_sha256": sha256_file(log_path),
    }
    return value


def compare_parquet(
    left: Path, right: Path, expected_rows: int, scratch: Path
) -> dict[str, Any]:
    import duckdb
    import pyarrow.parquet as pq

    left_sha = sha256_file(left)
    right_sha = sha256_file(right)
    left_meta = pq.ParquetFile(left).metadata
    right_meta = pq.ParquetFile(right).metadata
    if int(left_meta.num_rows) != expected_rows or int(right_meta.num_rows) != expected_rows:
        raise EquivalenceError("standalone/frozen Parquet row count drift")
    scratch = require_authorized(scratch, exists=False)
    if scratch.exists() or scratch.is_symlink():
        raise EquivalenceError(f"DuckDB scratch reuse forbidden: {scratch}")
    scratch.mkdir(parents=True, exist_ok=False)
    scratch_sql = "'" + str(scratch).replace("'", "''") + "'"
    connection = duckdb.connect()
    try:
        connection.execute("SET memory_limit='64MB'")
        connection.execute("SET threads=1")
        connection.execute(f"SET temp_directory={scratch_sql}")
        left_sql = "'" + str(left).replace("'", "''") + "'"
        right_sql = "'" + str(right).replace("'", "''") + "'"
        mismatch = int(
            connection.execute(
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
        )
    finally:
        connection.close()
    if mismatch != 0 or left_sha != right_sha or left.stat().st_size != right.stat().st_size:
        raise EquivalenceError(
            f"standalone/frozen output mismatch: rows={mismatch}, sha={left_sha}/{right_sha}"
        )
    return {
        "rows": expected_rows,
        "except_all_bidirectional_mismatch_rows": mismatch,
        "byte_identical": True,
        "bytes": left.stat().st_size,
        "sha256": left_sha,
        "row_groups": int(left_meta.num_row_groups),
        "duckdb_temp_directory": str(scratch),
        "duckdb_temp_files_after_close": sorted(
            str(path.relative_to(scratch)) for path in scratch.rglob("*")
        ),
    }


def audit(args: argparse.Namespace) -> int:
    root = require_authorized(args.audit_root, exists=False)
    tool_root = require_authorized(args.tool_root)
    if root.exists() or root.is_symlink():
        raise EquivalenceError(f"equivalence audit reuse forbidden: {root}")
    python = Path(sys.executable).resolve()
    if not python.is_file() or "/dsk2" in str(python) or not python.is_relative_to(tool_root):
        raise EquivalenceError("equivalence audit is not running under the bound standalone runtime")
    if sha256_file(RUNNER) != RUNNER_SHA256:
        raise EquivalenceError("final R9 runner SHA drift")
    root.mkdir(parents=True, exist_ok=False)
    results: dict[str, Any] = {}
    for name, raw_spec in FIXTURES.items():
        raw = require_authorized(Path(str(raw_spec["raw"])))
        expected = require_authorized(Path(str(raw_spec["expected"])))
        if raw_spec.get("raw_sha256") and sha256_file(raw) != raw_spec["raw_sha256"]:
            raise EquivalenceError(f"{name} raw fixture SHA drift")
        expected_sha = sha256_file(expected)
        if expected_sha != raw_spec["expected_output_sha256"]:
            raise EquivalenceError(f"{name} frozen expected output SHA drift")
        worker_result = run_worker(
            name=name,
            raw=raw,
            rows=int(raw_spec["rows"]),
            root=root,
            tool_root=tool_root,
        )
        produced = Path(worker_result["output_path"])
        comparison = compare_parquet(
            produced,
            expected,
            int(raw_spec["rows"]),
            root / name / "duckdb_temp",
        )
        results[name] = {
            "raw_path": str(raw),
            "raw_sha256": sha256_file(raw),
            "frozen_expected_path": str(expected),
            "frozen_expected_sha256": expected_sha,
            "worker": worker_result,
            "comparison": comparison,
        }
    report: dict[str, Any] = {
        "format": FORMAT,
        "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
        "status": "PASS_SMALL_AND_ACC_LARGE_BYTE_EXACT",
        "standalone_python_path": str(python),
        "standalone_python_sha256": sha256_file(python),
        "tool_root": str(tool_root),
        "runner_path": str(RUNNER),
        "runner_sha256": RUNNER_SHA256,
        "fixtures": results,
        "fixture_count": 2,
        "small_byte_identical": True,
        "acc_large_byte_identical": True,
        "reads_dsk2": False,
        "writes_dsk2": False,
        "writes_tmp": False,
        "production_deployed": False,
        "port_8260_touched": False,
        "timestamp_unix": time.time(),
    }
    report["audit_contract_sha256"] = canonical_sha256(report)
    report_path = root / "AUDIT.json"
    exclusive_json(report_path, report)
    print(
        json.dumps(
            {**report, "audit_path": str(report_path), "audit_sha256": sha256_file(report_path)},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
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
    worker_parser.add_argument("--expected-rows", required=True, type=int)
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--audit-root", required=True, type=Path)
    audit_parser.add_argument("--tool-root", required=True, type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "worker":
        return worker(args)
    audit_root = args.audit_root.resolve()
    try:
        return audit(args)
    except Exception as exc:
        try:
            authorized = AUTHORIZED_ROOT.resolve(strict=True)
            if authorized in audit_root.parents:
                audit_root.mkdir(parents=True, exist_ok=True)
                failure: dict[str, Any] = {
                    "format": FAILURE_FORMAT,
                    "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
                    "status": "TYPED_FAILURE_PRESERVED",
                    "reason": f"{type(exc).__name__}:{exc}",
                    "audit_root": str(audit_root),
                    "standalone_python_path": str(Path(sys.executable).resolve()),
                    "runner_path": str(RUNNER),
                    "runner_expected_sha256": RUNNER_SHA256,
                    "reads_dsk2": False,
                    "writes_dsk2": False,
                    "writes_tmp": False,
                    "production_deployed": False,
                    "port_8260_touched": False,
                    "timestamp_unix": time.time(),
                }
                failure["failure_contract_sha256"] = canonical_sha256(failure)
                exclusive_json(audit_root / "TYPED_FAILURE.json", failure)
        except Exception:
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
