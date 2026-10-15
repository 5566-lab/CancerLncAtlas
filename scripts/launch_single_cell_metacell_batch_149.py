#!/usr/bin/env python3
"""Resume-safe CPU batch supervisor for fold-local metacell materialization."""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
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
DEFAULT_OUTPUT_ROOT = Path(
    "/dell_2/DSC/CancerLncAtlas/rbp_encode_v33_20260921_r1/runtime/"
    "c_graph_global_binding_20260928_r1/results/"
    "single_cell_metacell_v32-single-cell-metacell-donor-oof-20261005-r1"
)
DEFAULT_SCRATCH_ROOT = Path("/tmp/cla_sc_metacell_20261005_r1")
DEFAULT_REPAIR_ROOT = Path("/dsk2/user/dengsc/sc_preprocessing_repair_20261004_r1")
DEFAULT_ACTIVE_STATUS = Path(
    "/dell_2/DSC/CancerLncAtlas/runtime/"
    "single_cell_cell_level_r11_rescued_20260903_r1/RUN_STATUS.json"
)
DEFAULT_CANDIDATES = Path(
    "/dell_2/DSC/CancerLncAtlas/inputs/"
    "v32_g012_local_cnv_formal_prepared_20260903_r1/"
    "FORMAL_CANDIDATE_UNIVERSE.parquet"
)
DEFAULT_CORE_ROOT = Path(
    "/dell_2/DSC/CancerLncAtlas/rbp_encode_v33_20260921_r1/runtime/"
    "c_graph_global_binding_20260928_r1/core_embeddings"
)
FORMAT = "CANCERLNCATLAS_V32_SC_METACELL_BATCH_SUPERVISOR_V1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def gib_available(path: Path) -> float:
    stats = os.statvfs(path)
    return stats.f_bavail * stats.f_frsize / 1024**3


def memory_available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024**2
    raise RuntimeError("MemAvailable missing from /proc/meminfo")


def external_materializers() -> list[dict[str, object]]:
    found = []
    for cmdline_path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            command = cmdline_path.read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if "materialize_single_cell_metacell_149.py" in command:
            found.append({"pid": int(cmdline_path.parent.name), "command": command})
    return found


def interleave_large_small(cancers: list[str], sizes: dict[str, int]) -> list[str]:
    ranked = sorted(cancers, key=lambda cancer: (-sizes[cancer], cancer))
    result = []
    left, right = 0, len(ranked) - 1
    while left <= right:
        result.append(ranked[left])
        left += 1
        if left <= right:
            result.append(ranked[right])
            right -= 1
    return result


def validation_is_current(output_root: Path, cancer: str) -> bool:
    success = output_root / "cancers" / cancer / "SUCCESS.json"
    validation = output_root / f"MATERIALIZATION_VALIDATION.{cancer}.json"
    if not success.is_file() or not validation.is_file():
        return False
    try:
        report = load_json(validation)
        cancers = report.get("cancers", [])
        return (
            report.get("target_host") == "149"
            and report.get("status") == "PASS"
            and len(cancers) == 1
            and cancers[0].get("cancer_id") == cancer
            and validation.stat().st_mtime >= success.stat().st_mtime
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


class Supervisor:
    def __init__(self, args, cancers: list[str]):
        self.args = args
        self.cancers = cancers
        self.stop = threading.Event()
        self.process_lock = threading.Lock()
        self.processes: dict[str, subprocess.Popen] = {}
        self.status_path = args.output_root / "batch_supervisor" / "STATUS.json"
        self.log_root = args.output_root / "batch_supervisor" / "logs"
        self.log_root.mkdir(parents=True, exist_ok=True)
        self.started = time.time()
        self.states: dict[str, dict[str, object]] = {
            cancer: {"state": "PENDING"} for cancer in cancers
        }
        self.failures = 0
        self.resource_stop_reason: str | None = None

    def receipt(self, state: str) -> dict[str, object]:
        return {
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
            "validation_timeout_seconds": self.args.validation_timeout,
            "max_failures": self.args.max_failures,
            "materializer_sha256": sha256(self.args.materializer),
            "validator_sha256": sha256(self.args.validator),
            "cancers": self.cancers,
            "states": self.states,
            "failures": self.failures,
            "resource_stop_reason": self.resource_stop_reason,
            "pointer_promotion_attempted": False,
        }

    def write_status(self, state: str) -> None:
        atomic_json(self.status_path, self.receipt(state))

    def resource_gate(self) -> tuple[bool, str]:
        available_memory = memory_available_gib()
        available_scratch = gib_available(self.args.scratch_root)
        available_output = gib_available(self.args.output_root)
        details = (
            f"MemAvailable={available_memory:.1f}GiB, "
            f"scratch_free={available_scratch:.1f}GiB, "
            f"output_free={available_output:.1f}GiB"
        )
        okay = (
            available_memory >= self.args.min_memory_gib
            and available_scratch >= self.args.min_scratch_gib
            and available_output >= self.args.min_output_gib
        )
        return okay, details

    def terminate(self) -> None:
        self.stop.set()
        with self.process_lock:
            processes = list(self.processes.values())
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass

    def run_command(
        self, cancer: str, command: list[str], log_handle, timeout_label: str
    ) -> int:
        if self.stop.is_set():
            return 130
        log_handle.write("COMMAND " + " ".join(command) + "\n")
        log_handle.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=self.args.command_environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        with self.process_lock:
            self.processes[f"{cancer}:{timeout_label}"] = process
        try:
            return process.wait()
        finally:
            with self.process_lock:
                self.processes.pop(f"{cancer}:{timeout_label}", None)

    def run_one(self, cancer: str) -> dict[str, object]:
        started = time.time()
        log_path = self.log_root / f"{cancer}.log"
        success_path = self.args.output_root / "cancers" / cancer / "SUCCESS.json"
        with log_path.open("w", encoding="utf-8") as log_handle:
            if not success_path.is_file():
                deadline = time.time() + self.args.per_cancer_timeout - 300
                command = [
                    self.args.timeout_binary,
                    "--signal=TERM",
                    "--kill-after=300s",
                    f"{self.args.per_cancer_timeout}s",
                    sys.executable,
                    str(self.args.materializer),
                    "--cancer-id",
                    cancer,
                    "--output-root",
                    str(self.args.output_root),
                    "--scratch-root",
                    str(self.args.scratch_root),
                    "--repair-root",
                    str(self.args.repair_root),
                    "--candidates",
                    str(self.args.candidates),
                    "--deadline",
                    str(deadline),
                    "--delete-verified-scratch",
                ]
                return_code = self.run_command(cancer, command, log_handle, "materialize")
                if return_code != 0:
                    return {
                        "state": "FAILED_MATERIALIZATION",
                        "return_code": return_code,
                        "elapsed_seconds": time.time() - started,
                        "log_path": str(log_path),
                    }
                if not success_path.is_file():
                    return {
                        "state": "FAILED_MISSING_SUCCESS_RECEIPT",
                        "elapsed_seconds": time.time() - started,
                        "log_path": str(log_path),
                    }
            validation_command = [
                self.args.timeout_binary,
                "--signal=TERM",
                "--kill-after=60s",
                f"{self.args.validation_timeout}s",
                sys.executable,
                str(self.args.validator),
                "--output-root",
                str(self.args.output_root),
                "--core-root",
                str(self.args.core_root),
                "--cancer",
                cancer,
            ]
            return_code = self.run_command(
                cancer, validation_command, log_handle, "validate"
            )
            if return_code != 0 or not validation_is_current(
                self.args.output_root, cancer
            ):
                return {
                    "state": "FAILED_VALIDATION",
                    "return_code": return_code,
                    "elapsed_seconds": time.time() - started,
                    "log_path": str(log_path),
                }
        return {
            "state": "VALIDATED_COMPLETE",
            "elapsed_seconds": time.time() - started,
            "success_sha256": sha256(success_path),
            "validation_sha256": sha256(
                self.args.output_root
                / f"MATERIALIZATION_VALIDATION.{cancer}.json"
            ),
            "log_path": str(log_path),
        }

    def run(self) -> int:
        pending = [
            cancer
            for cancer in self.cancers
            if not validation_is_current(self.args.output_root, cancer)
        ]
        for cancer in self.cancers:
            if cancer not in pending:
                self.states[cancer] = {"state": "VALIDATED_COMPLETE_PREEXISTING"}
        self.write_status("RUNNING")
        active: dict[Future, str] = {}
        with ThreadPoolExecutor(max_workers=self.args.workers) as executor:
            while pending or active:
                while (
                    pending
                    and len(active) < self.args.workers
                    and self.failures < self.args.max_failures
                    and not self.stop.is_set()
                ):
                    okay, resources = self.resource_gate()
                    if not okay:
                        self.resource_stop_reason = resources
                        break
                    self.resource_stop_reason = None
                    active_large = any(
                        active_cancer in self.args.large_cancers
                        for active_cancer in active.values()
                    )
                    dispatch_index = next(
                        (
                            index
                            for index, candidate in enumerate(pending)
                            if not (
                                active_large
                                and candidate in self.args.large_cancers
                            )
                        ),
                        None,
                    )
                    if dispatch_index is None:
                        break
                    cancer = pending.pop(dispatch_index)
                    self.states[cancer] = {
                        "state": "RUNNING",
                        "started_unix": time.time(),
                        "resource_preflight": resources,
                    }
                    future = executor.submit(self.run_one, cancer)
                    active[future] = cancer
                    self.write_status("RUNNING")
                if not active:
                    break
                finished, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in finished:
                    cancer = active.pop(future)
                    try:
                        result = future.result()
                    except Exception as error:  # preserved in the supervisor receipt
                        result = {
                            "state": "FAILED_SUPERVISOR_EXCEPTION",
                            "error_type": type(error).__name__,
                            "error": str(error),
                        }
                    self.states[cancer] = result
                    if not str(result["state"]).startswith("VALIDATED_COMPLETE"):
                        self.failures += 1
                    self.write_status("RUNNING")
        for cancer in pending:
            self.states[cancer] = {
                "state": "NOT_STARTED",
                "reason": (
                    "RESOURCE_GATE" if self.resource_stop_reason else "FAILURE_BUDGET"
                ),
            }
        complete = all(
            str(record["state"]).startswith("VALIDATED_COMPLETE")
            for record in self.states.values()
        )
        final_state = "COMPLETE" if complete else "INCOMPLETE"
        self.write_status(final_state)
        return 0 if complete else 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--cancers", nargs="*")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--threads-per-job", type=int, default=24)
    parser.add_argument("--per-cancer-timeout", type=int, default=21600)
    parser.add_argument("--validation-timeout", type=int, default=3600)
    parser.add_argument("--max-failures", type=int, default=1)
    parser.add_argument("--min-memory-gib", type=float, default=96.0)
    parser.add_argument("--min-scratch-gib", type=float, default=120.0)
    parser.add_argument("--min-output-gib", type=float, default=100.0)
    parser.add_argument("--large-h5-gib", type=float, default=0.75)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--scratch-root", type=Path, default=DEFAULT_SCRATCH_ROOT)
    parser.add_argument("--repair-root", type=Path, default=DEFAULT_REPAIR_ROOT)
    parser.add_argument("--active-status", type=Path, default=DEFAULT_ACTIVE_STATUS)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--core-root", type=Path, default=DEFAULT_CORE_ROOT)
    parser.add_argument(
        "--materializer", type=Path,
        default=ROOT / "scripts" / "materialize_single_cell_metacell_149.py",
    )
    parser.add_argument(
        "--validator", type=Path,
        default=ROOT / "scripts" / "validate_metacell_materialization_149.py",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if socket.gethostname() != "149":
        raise SystemExit("WRONG_HOST")
    required = [
        args.output_root / "LAUNCH_PREFLIGHT.json",
        args.active_status,
        args.candidates,
        args.materializer,
        args.validator,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("MISSING_REQUIRED_INPUTS: " + ", ".join(missing))
    preflight = load_json(args.output_root / "LAUNCH_PREFLIGHT.json")
    if (
        preflight.get("target_host") != "149"
        or preflight.get("cpu_only") is not True
        or preflight.get("paid_compute") is not False
    ):
        raise SystemExit("INVALID_CPU_LAUNCH_PREFLIGHT")
    if args.max_failures < 1:
        raise SystemExit("max-failures must be positive")
    cpu_count = os.cpu_count() or 1
    if args.threads_per_job < 1 or args.workers * args.threads_per_job > cpu_count:
        raise SystemExit("worker/thread allocation exceeds online CPUs")
    timeout_binary = shutil.which("timeout")
    if not timeout_binary:
        raise SystemExit("GNU timeout is required")
    args.timeout_binary = timeout_binary
    args.scratch_root.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    status = load_json(args.active_status)
    jobs = sorted(
        path.name
        for path in (args.repair_root / "jobs").iterdir()
        if path.is_dir()
    )
    authorized = [
        cancer
        for cancer in jobs
        if status.get("per_cancer", {}).get(cancer, {}).get("formal_eligible") is True
    ]
    if len(authorized) != 28:
        raise SystemExit(f"EXPECTED_28_AUTHORIZED_JOBS_FOUND_{len(authorized)}")
    requested = [cancer.upper() for cancer in args.cancers] if args.cancers else authorized
    unknown = sorted(set(requested) - set(authorized))
    if unknown:
        raise SystemExit("UNAUTHORIZED_CANCERS: " + ", ".join(unknown))
    h5_sizes = {
        cancer: Path(status["per_cancer"][cancer]["h5_path"]).stat().st_size
        for cancer in requested
    }
    args.large_cancers = {
        cancer
        for cancer, size in h5_sizes.items()
        if size >= args.large_h5_gib * 1024**3
    }
    cancers = interleave_large_small(sorted(set(requested)), h5_sizes)
    completed = [
        cancer for cancer in cancers if validation_is_current(args.output_root, cancer)
    ]
    pending = [cancer for cancer in cancers if cancer not in completed]
    dry_run = {
        "format": FORMAT,
        "target_host": "149",
        "cpu_only": True,
        "paid_compute": False,
        "execute": args.execute,
        "authorized_cancer_count": len(authorized),
        "selected_cancer_count": len(cancers),
        "validated_complete": completed,
        "pending_order": pending,
        "large_cancers_serialized": sorted(args.large_cancers),
        "large_h5_threshold_gib": args.large_h5_gib,
        "workers": args.workers,
        "threads_per_job": args.threads_per_job,
        "per_cancer_timeout_seconds": args.per_cancer_timeout,
        "materializer_sha256": sha256(args.materializer),
        "validator_sha256": sha256(args.validator),
        "external_materializers": external_materializers(),
        "resources": {
            "memory_available_gib": memory_available_gib(),
            "scratch_available_gib": gib_available(args.scratch_root),
            "output_available_gib": gib_available(args.output_root),
        },
    }
    print(json.dumps(dry_run, indent=2, ensure_ascii=False), flush=True)
    if not args.execute:
        return
    if dry_run["external_materializers"]:
        raise SystemExit("EXTERNAL_MATERIALIZER_ALREADY_RUNNING")
    command_environment = os.environ.copy()
    python_path = [str(ROOT), str(PYTHON_PACKAGES)]
    if command_environment.get("PYTHONPATH"):
        python_path.append(command_environment["PYTHONPATH"])
    command_environment.update(
        {
            "PYTHONPATH": os.pathsep.join(python_path),
            "OPENBLAS_NUM_THREADS": str(args.threads_per_job),
            "OMP_NUM_THREADS": str(args.threads_per_job),
            "MKL_NUM_THREADS": str(args.threads_per_job),
        }
    )
    args.command_environment = command_environment
    lock_path = args.output_root / "batch_supervisor" / "LOCK"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("BATCH_SUPERVISOR_ALREADY_RUNNING")
        supervisor = Supervisor(args, cancers)

        def stop_handler(signum, frame):
            del signum, frame
            supervisor.terminate()

        signal.signal(signal.SIGINT, stop_handler)
        signal.signal(signal.SIGTERM, stop_handler)
        raise SystemExit(supervisor.run())


if __name__ == "__main__":
    main()
