#!/usr/bin/env python3
"""Resume-safe host-149 batch launcher for strict nested SC-HHGT assets."""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time


ROOT = Path(__file__).resolve().parents[1]
PYTHON_PACKAGES = Path(
    "/public0/DSC/CancerLncAtlas/runtime/python_packages_v32_drug_20260825"
)
FORMAT = "CANCERLNCATLAS_V32_SC_HHGT_NESTED_BATCH_V1"
PREFLIGHT_FORMAT = "CANCERLNCATLAS_V32_SC_HHGT_NESTED_PREFLIGHT_V1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def initialize_preflight(args) -> dict[str, object]:
    source_validation = args.source_result_root / "MATERIALIZATION_VALIDATION.json"
    graph_preflight = args.graph_asset_root / "LAUNCH_PREFLIGHT.json"
    for path in (source_validation, graph_preflight, args.candidates):
        if not path.is_file():
            raise RuntimeError(f"PREFLIGHT_INPUT_MISSING:{path}")
    source = json.loads(source_validation.read_text(encoding="utf-8"))
    if (
        source.get("target_host") != "149"
        or source.get("status") != "PASS"
        or source.get("closure_status")
        != "PASS_FULL_28_CANCER_MATERIALIZATION_CLOSURE"
    ):
        raise RuntimeError("SOURCE_MATERIALIZATION_NOT_FORMALLY_VALIDATED")
    receipt = {
        "format": PREFLIGHT_FORMAT,
        "run_id": args.run_id,
        "target_host": "149",
        "cpu_only": True,
        "paid_compute": False,
        "formal_pointer_update_authorized": False,
        "source_metacell_result_root": str(args.source_result_root.resolve()),
        "graph_asset_root": str(args.graph_asset_root.resolve()),
        "candidate_path": str(args.candidates.resolve()),
        "geometry_policy": "STRICT_VALIDATION_WHERE_FEASIBLE_WITH_SUPPORT_LIMITED_LEAVE_ONE_TARGET_FIT",
        "validation_fold_policy": "v=(outer_fold_id+1)%active_fold_count",
        "fit_geometry_isolation": "outer_and_inner_target_and_validation_fold",
        "support_limited_fit_geometry_isolation": "outer_and_inner_target_fold",
        "support_limited_fit_minimum_source_donors": 2,
        "support_limited_strata_supply_validation": False,
        "validation_geometry_isolation": "outer_and_inner_target_fold",
        "bindings": {
            "source_materialization_validation": file_record(source_validation),
            "graph_asset_preflight": file_record(graph_preflight),
            "candidate": file_record(args.candidates),
        },
        "materializer": file_record(args.materializer),
        "nested_module": file_record(
            ROOT / "cc_hhgt" / "v32" / "single_cell_nested_episodes.py"
        ),
        "created_unix": time.time(),
    }
    destination = args.output_root / "LAUNCH_PREFLIGHT.json"
    if destination.is_file():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        comparable = dict(existing)
        comparable.pop("created_unix", None)
        expected = dict(receipt)
        expected.pop("created_unix", None)
        if comparable != expected:
            raise RuntimeError("EXISTING_NESTED_PREFLIGHT_DRIFT")
        return existing
    atomic_json(destination, receipt)
    return receipt


def _memory_available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024**2
    raise RuntimeError("MemAvailable missing from /proc/meminfo")


def _disk_available_gib(path: Path) -> float:
    stats = os.statvfs(path)
    return stats.f_bavail * stats.f_frsize / 1024**3


class Supervisor:
    def __init__(self, args, cancers: list[str]):
        self.args = args
        self.cancers = cancers
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.processes: dict[str, subprocess.Popen] = {}
        self.started = time.time()
        self.states = {cancer: {"state": "PENDING"} for cancer in cancers}
        self.failures = 0
        self.status = args.output_root / "batch_supervisor" / "STATUS.json"
        self.logs = args.output_root / "batch_supervisor" / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)

    def write(self, state: str) -> None:
        atomic_json(
            self.status,
            {
                "format": FORMAT,
                "target_host": "149",
                "cpu_only": True,
                "paid_compute": False,
                "state": state,
                "started_unix": self.started,
                "updated_unix": time.time(),
                "workers": self.args.workers,
                "threads_per_job": self.args.threads_per_job,
                "per_cancer_timeout_seconds": self.args.per_cancer_timeout,
                "materializer_sha256": sha256(self.args.materializer),
                "cancers": self.cancers,
                "states": self.states,
                "failures": self.failures,
                "formal_pointer_update_authorized": False,
            },
        )

    def terminate(self) -> None:
        self.stop.set()
        with self.lock:
            processes = list(self.processes.values())
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass

    def _resource_gate(self) -> None:
        memory = _memory_available_gib()
        scratch = _disk_available_gib(self.args.scratch_root)
        output = _disk_available_gib(self.args.output_root)
        if (
            memory < self.args.min_memory_gib
            or scratch < self.args.min_scratch_gib
            or output < self.args.min_output_gib
        ):
            raise RuntimeError(
                "NESTED_RESOURCE_GATE_FAILED:"
                f"memory={memory:.1f},scratch={scratch:.1f},output={output:.1f}"
            )

    def run_one(self, cancer: str) -> dict[str, object]:
        success = self.args.output_root / "cancers" / cancer / "SUCCESS.json"
        if success.is_file():
            receipt = json.loads(success.read_text(encoding="utf-8"))
            if receipt.get("state") == "CANCER_NESTED_ASSETS_COMPLETE":
                return {"state": "ALREADY_COMPLETE", "success_sha256": sha256(success)}
        self._resource_gate()
        started = time.time()
        deadline = started + self.args.per_cancer_timeout - 300
        command = [
            self.args.timeout_binary,
            "--signal=TERM",
            "--kill-after=300s",
            f"{self.args.per_cancer_timeout}s",
            sys.executable,
            str(self.args.materializer),
            "--cancer-id", cancer,
            "--outer-fold", "all",
            "--inner-target-fold", "all",
            "--source-result-root", str(self.args.source_result_root),
            "--graph-asset-root", str(self.args.graph_asset_root),
            "--output-root", str(self.args.output_root),
            "--scratch-root", str(self.args.scratch_root),
            "--repair-root", str(self.args.repair_root),
            "--candidates", str(self.args.candidates),
            "--deadline", str(deadline),
            "--delete-verified-scratch",
        ]
        log_path = self.logs / f"{cancer}.log"
        with log_path.open("w", encoding="utf-8") as handle:
            handle.write("COMMAND " + " ".join(command) + "\n")
            handle.flush()
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=self.args.command_environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            with self.lock:
                self.processes[cancer] = process
            try:
                return_code = process.wait()
            finally:
                with self.lock:
                    self.processes.pop(cancer, None)
        if return_code != 0:
            return {
                "state": "FAILED",
                "return_code": return_code,
                "elapsed_seconds": time.time() - started,
                "log_path": str(log_path),
            }
        if not success.is_file():
            return {
                "state": "FAILED_MISSING_CANCER_SUCCESS",
                "elapsed_seconds": time.time() - started,
                "log_path": str(log_path),
            }
        receipt = json.loads(success.read_text(encoding="utf-8"))
        if receipt.get("state") != "CANCER_NESTED_ASSETS_COMPLETE":
            return {
                "state": "FAILED_CANCER_RECEIPT",
                "elapsed_seconds": time.time() - started,
                "log_path": str(log_path),
            }
        return {
            "state": "COMPLETE",
            "elapsed_seconds": time.time() - started,
            "log_path": str(log_path),
            "success_sha256": sha256(success),
        }

    def run(self) -> None:
        self.write("RUNNING")
        pending = list(self.cancers)
        active = {}
        with ThreadPoolExecutor(max_workers=self.args.workers) as pool:
            while pending or active:
                while pending and len(active) < self.args.workers and not self.stop.is_set():
                    cancer = pending.pop(0)
                    self.states[cancer] = {"state": "RUNNING", "started_unix": time.time()}
                    active[pool.submit(self.run_one, cancer)] = cancer
                    self.write("RUNNING")
                if not active:
                    break
                completed, _ = wait(active, timeout=30, return_when=FIRST_COMPLETED)
                if not completed:
                    self.write("RUNNING")
                    continue
                for future in completed:
                    cancer = active.pop(future)
                    try:
                        result = future.result()
                    except Exception as error:  # supervisor must preserve the failure receipt
                        result = {"state": "FAILED", "error": repr(error)}
                    self.states[cancer] = result
                    if str(result.get("state", "")).startswith("FAILED"):
                        self.failures += 1
                        if self.failures >= self.args.max_failures:
                            self.terminate()
                self.write("RUNNING")
        state = "COMPLETE" if not self.failures and not self.stop.is_set() else "FAILED"
        self.write(state)
        if state != "COMPLETE":
            raise SystemExit("nested batch did not complete")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--run-id", required=True)
    result.add_argument("--source-result-root", required=True, type=Path)
    result.add_argument("--graph-asset-root", required=True, type=Path)
    result.add_argument("--output-root", required=True, type=Path)
    result.add_argument("--scratch-root", required=True, type=Path)
    result.add_argument(
        "--repair-root",
        type=Path,
        default=Path("/dsk2/user/dengsc/sc_preprocessing_repair_20261004_r1"),
    )
    result.add_argument("--candidates", required=True, type=Path)
    result.add_argument("--cancers", nargs="*", default=[])
    result.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=4)
    result.add_argument("--threads-per-job", type=int, default=12)
    result.add_argument("--per-cancer-timeout", type=int, default=86400)
    result.add_argument("--max-failures", type=int, default=1)
    result.add_argument("--min-memory-gib", type=float, default=24.0)
    result.add_argument("--min-scratch-gib", type=float, default=40.0)
    result.add_argument("--min-output-gib", type=float, default=100.0)
    result.add_argument("--initialize-preflight", action="store_true")
    result.add_argument(
        "--materializer",
        type=Path,
        default=ROOT / "scripts" / "materialize_single_cell_nested_hhgt_episodes_149.py",
    )
    result.add_argument("--timeout-binary", default="/usr/bin/timeout")
    return result


def main() -> None:
    args = parser().parse_args()
    if socket.gethostname() != "149":
        raise SystemExit("WRONG_HOST")
    for path in (args.output_root, args.scratch_root):
        path.mkdir(parents=True, exist_ok=True)
    if args.initialize_preflight:
        initialize_preflight(args)
    preflight = args.output_root / "LAUNCH_PREFLIGHT.json"
    if not preflight.is_file():
        raise SystemExit("NESTED_LAUNCH_PREFLIGHT_MISSING")
    source = json.loads(
        (args.source_result_root / "MATERIALIZATION_VALIDATION.json").read_text(
            encoding="utf-8"
        )
    )
    authorized = list(map(str, source.get("validated_cancers", [])))
    cancers = [value.upper() for value in args.cancers] if args.cancers else authorized
    if not cancers or not set(cancers).issubset(set(authorized)):
        raise SystemExit("REQUESTED_CANCERS_OUTSIDE_VALIDATED_SOURCE")
    if args.workers * args.threads_per_job > (os.cpu_count() or 1):
        raise SystemExit("worker/thread allocation exceeds online CPUs")
    environment = os.environ.copy()
    python_path = [str(ROOT), str(PYTHON_PACKAGES)]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join(python_path),
            "OMP_NUM_THREADS": str(args.threads_per_job),
            "MKL_NUM_THREADS": str(args.threads_per_job),
            "OPENBLAS_NUM_THREADS": str(args.threads_per_job),
        }
    )
    args.command_environment = environment
    supervisor = Supervisor(args, cancers)
    signal.signal(signal.SIGTERM, lambda *_: supervisor.terminate())
    signal.signal(signal.SIGINT, lambda *_: supervisor.terminate())
    supervisor.run()


if __name__ == "__main__":
    main()
