#!/usr/bin/env python3
"""Resume-safe parallel SC-HHGT LASSO/episode assembly on host 149."""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
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
FORMAT = "CANCERLNCATLAS_V32_SC_HHGT_EPISODE_BATCH_V1"


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    temporary = path.with_name(
        f"{path.name}.partial.{os.getpid()}.{threading.get_ident()}"
    )
    for attempt in range(6):
        try:
            temporary.write_text(payload, encoding="utf-8")
            os.replace(temporary, path)
            return
        except PermissionError:
            temporary.unlink(missing_ok=True)
            if attempt == 5:
                raise
            time.sleep(0.2 * (attempt + 1))


def _available_gib(path: Path) -> float:
    value = os.statvfs(path)
    return value.f_bavail * value.f_frsize / 1024**3


def _memory_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024**2
    raise RuntimeError("MEMAVAILABLE_MISSING")


class Supervisor:
    def __init__(self, args, cancers: list[str]):
        self.args, self.cancers = args, cancers
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.processes: dict[str, subprocess.Popen] = {}
        self.states = {cancer: {"state": "PENDING"} for cancer in cancers}
        self.failures = 0
        self.started = time.time()
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
                "launch_stagger_seconds": self.args.launch_stagger_seconds,
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

    def resource_gate(self) -> None:
        memory, output = _memory_gib(), _available_gib(self.args.output_root)
        if memory < self.args.min_memory_gib or output < self.args.min_output_gib:
            raise RuntimeError(
                f"EPISODE_RESOURCE_GATE_FAILED:memory={memory:.1f},output={output:.1f}"
            )

    def common_command(self) -> list[str]:
        return [
            "--run-id", self.args.run_id,
            "--source-result-root", str(self.args.source_result_root),
            "--graph-asset-root", str(self.args.graph_asset_root),
            "--nested-asset-root", str(self.args.nested_asset_root),
            "--output-root", str(self.args.output_root),
            "--lasso-alphas", self.args.lasso_alphas,
            "--lasso-max-genes", str(self.args.lasso_max_genes),
            "--lasso-min-donors", str(self.args.lasso_min_donors),
            "--lasso-inner-folds", str(self.args.lasso_inner_folds),
            "--lasso-bootstraps", str(self.args.lasso_bootstraps),
            "--lasso-response-batch-size", str(self.args.lasso_response_batch_size),
            "--lasso-max-iter", str(self.args.lasso_max_iter),
            "--lasso-tolerance", str(self.args.lasso_tolerance),
            "--lasso-stability-threshold", str(self.args.lasso_stability_threshold),
        ]

    def run_one(self, cancer: str) -> dict[str, object]:
        success = self.args.output_root / "cancers" / cancer / "SUCCESS.json"
        if success.is_file():
            value = json.loads(success.read_text(encoding="utf-8"))
            if value.get("state") == "CANCER_EPISODES_COMPLETE":
                return {
                    "state": "ALREADY_COMPLETE",
                    "episode_count": int(value.get("episode_count", 0)),
                    "row_count": int(value.get("row_count", 0)),
                }
        self.resource_gate()
        started = time.time()
        deadline = started + self.args.per_cancer_timeout - 300
        command = [
            self.args.timeout_binary, "--signal=TERM", "--kill-after=300s",
            f"{self.args.per_cancer_timeout}s", sys.executable, str(self.args.materializer),
            *self.common_command(), "--cancer-id", cancer, "--deadline", str(deadline),
        ]
        log = self.logs / f"{cancer}.log"
        handle = None
        process = None
        try:
            with self.lock:
                # sshfs can return transient EPERM for concurrent file creation.
                for attempt in range(6):
                    try:
                        handle = log.open("w", encoding="utf-8")
                        break
                    except PermissionError:
                        if attempt == 5:
                            raise
                        time.sleep(0.2 * (attempt + 1))
                assert handle is not None
                handle.write("COMMAND " + " ".join(command) + "\n")
                handle.flush()
                # Concurrent setsid-backed Popen calls can race under the host's
                # process supervisor and return EPERM.  Serialize only process
                # creation; the cancer jobs still execute concurrently.
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=self.args.command_environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
                self.processes[cancer] = process
            code = process.wait()
        finally:
            if process is not None:
                with self.lock:
                    self.processes.pop(cancer, None)
            if handle is not None:
                handle.close()
        if code or not success.is_file():
            return {
                "state": "FAILED",
                "return_code": code,
                "elapsed_seconds": time.time() - started,
                "log": str(log),
            }
        value = json.loads(success.read_text(encoding="utf-8"))
        if value.get("state") != "CANCER_EPISODES_COMPLETE":
            return {"state": "FAILED_RECEIPT", "log": str(log)}
        return {
            "state": "COMPLETE",
            "elapsed_seconds": time.time() - started,
            "episode_count": int(value.get("episode_count", 0)),
            "row_count": int(value.get("row_count", 0)),
            "log": str(log),
        }

    def cached_success(self, cancer: str) -> dict[str, object] | None:
        success = self.args.output_root / "cancers" / cancer / "SUCCESS.json"
        if not success.is_file():
            return None
        value = json.loads(success.read_text(encoding="utf-8"))
        if value.get("state") != "CANCER_EPISODES_COMPLETE":
            return None
        return {
            "state": "ALREADY_COMPLETE",
            "episode_count": int(value.get("episode_count", 0)),
            "row_count": int(value.get("row_count", 0)),
        }

    def run(self) -> None:
        pending = []
        for cancer in self.cancers:
            cached = self.cached_success(cancer)
            if cached is None:
                pending.append(cancer)
            else:
                self.states[cancer] = cached
        self.write("RUNNING")
        active = {}
        with ThreadPoolExecutor(max_workers=self.args.workers) as pool:
            while pending or active:
                while pending and len(active) < self.args.workers and not self.stop.is_set():
                    cancer = pending.pop(0)
                    self.states[cancer] = {"state": "RUNNING", "started_unix": time.time()}
                    active[pool.submit(self.run_one, cancer)] = cancer
                    self.write("RUNNING")
                    if pending and len(active) < self.args.workers:
                        time.sleep(self.args.launch_stagger_seconds)
                if not active:
                    break
                completed, _ = wait(active, timeout=30, return_when=FIRST_COMPLETED)
                if not completed:
                    self.write("RUNNING")
                    continue
                for future in completed:
                    cancer = active.pop(future)
                    try:
                        value = future.result()
                    except Exception as error:
                        value = {"state": "FAILED", "error": repr(error)}
                    self.states[cancer] = value
                    if str(value.get("state", "")).startswith("FAILED"):
                        self.failures += 1
                        if self.failures >= self.args.max_failures:
                            self.terminate()
                self.write("RUNNING")
        if self.failures or self.stop.is_set():
            self.write("FAILED")
            raise SystemExit("episode batch did not complete")
        finalize_log = self.logs / "FINALIZE.log"
        command = [
            self.args.timeout_binary, "--signal=TERM", "--kill-after=120s", "3600s",
            sys.executable, str(self.args.materializer), *self.common_command(),
            "--finalize", "--deadline", str(time.time() + 3300),
        ]
        with finalize_log.open("w", encoding="utf-8") as handle:
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=self.args.command_environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
        if result.returncode or not (
            self.args.output_root / "SC_HHGT_EPISODE_MANIFEST.json"
        ).is_file():
            self.write("FAILED_FINALIZE")
            raise SystemExit("episode manifest finalization failed")
        self.write("COMPLETE")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run-id", required=True)
    result.add_argument("--source-result-root", required=True, type=Path)
    result.add_argument("--graph-asset-root", required=True, type=Path)
    result.add_argument("--nested-asset-root", required=True, type=Path)
    result.add_argument("--output-root", required=True, type=Path)
    result.add_argument("--cancers", nargs="*", default=[])
    result.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=4)
    result.add_argument("--threads-per-job", type=int, default=12)
    result.add_argument("--launch-stagger-seconds", type=float, default=5.0)
    result.add_argument("--per-cancer-timeout", type=int, default=86400)
    result.add_argument("--max-failures", type=int, default=1)
    result.add_argument("--min-memory-gib", type=float, default=48.0)
    result.add_argument("--min-output-gib", type=float, default=300.0)
    result.add_argument("--lasso-alphas", default="0.3,0.1,0.03,0.01")
    result.add_argument("--lasso-max-genes", type=int, default=1000)
    result.add_argument("--lasso-min-donors", type=int, default=4)
    result.add_argument("--lasso-inner-folds", type=int, default=5)
    result.add_argument("--lasso-bootstraps", type=int, default=8)
    result.add_argument("--lasso-response-batch-size", type=int, default=256)
    result.add_argument("--lasso-max-iter", type=int, default=2000)
    result.add_argument("--lasso-tolerance", type=float, default=1e-6)
    result.add_argument("--lasso-stability-threshold", type=float, default=0.5)
    result.add_argument("--timeout-binary", default="/usr/bin/timeout")
    result.add_argument(
        "--materializer",
        type=Path,
        default=ROOT / "scripts" / "materialize_single_cell_hhgt_training_episodes_149.py",
    )
    return result


def main() -> None:
    args = parser().parse_args()
    if socket.gethostname() != "149":
        raise SystemExit("WRONG_HOST")
    args.output_root.mkdir(parents=True, exist_ok=True)
    source = json.loads(
        (args.source_result_root / "MATERIALIZATION_VALIDATION.json").read_text(encoding="utf-8")
    )
    authorized = sorted(map(str, source.get("validated_cancers", ())))
    cancers = [value.upper() for value in args.cancers] if args.cancers else authorized
    if len(authorized) != 28 or not cancers or not set(cancers).issubset(authorized):
        raise SystemExit("REQUESTED_CANCERS_OUTSIDE_VALIDATED_28")
    if args.workers * args.threads_per_job > (os.cpu_count() or 1):
        raise SystemExit("THREAD_ALLOCATION_EXCEEDS_CPUS")
    if args.launch_stagger_seconds < 0:
        raise SystemExit("INVALID_LAUNCH_STAGGER")
    from cc_hhgt.v32.single_cell_batched_lasso import BatchedLassoPolicy
    from scripts.materialize_single_cell_hhgt_training_episodes_149 import (
        _initialize_preflight,
    )

    args.lasso_policy = BatchedLassoPolicy(
        alphas=tuple(float(value) for value in args.lasso_alphas.split(",")),
        max_genes=args.lasso_max_genes,
        min_donors=args.lasso_min_donors,
        inner_folds=args.lasso_inner_folds,
        bootstraps=args.lasso_bootstraps,
        response_batch_size=args.lasso_response_batch_size,
        max_iter=args.lasso_max_iter,
        tolerance=args.lasso_tolerance,
    ).validated()
    _initialize_preflight(args)
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(ROOT),
            "SC_LASSO_SOLVER_THREADS": str(args.threads_per_job),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    args.command_environment = environment
    supervisor = Supervisor(args, cancers)
    signal.signal(signal.SIGTERM, lambda *_: supervisor.terminate())
    signal.signal(signal.SIGINT, lambda *_: supervisor.terminate())
    supervisor.run()


if __name__ == "__main__":
    main()
