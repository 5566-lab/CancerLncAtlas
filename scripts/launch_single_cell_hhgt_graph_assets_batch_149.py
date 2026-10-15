#!/usr/bin/env python3
"""Resume-safe batch materialization of outer-fold SC-HHGT graph assets on 149."""
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
FORMAT = "CANCERLNCATLAS_V32_SC_HHGT_GRAPH_ASSET_BATCH_V1"
PREFLIGHT_FORMAT = "CANCERLNCATLAS_V32_SC_HHGT_GRAPH_ASSET_PREFLIGHT_V1"
CANCER_SUCCESS_FORMAT = "CANCERLNCATLAS_V32_SC_HHGT_CANCER_SUCCESS_V1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, object]:
    value = path.resolve()
    return {"path": str(value), "bytes": value.stat().st_size, "sha256": sha256(value)}


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def initialize_preflight(args) -> dict[str, object]:
    source_validation = args.source_result_root / "MATERIALIZATION_VALIDATION.json"
    for path in (source_validation, args.pathway_family, args.external_evidence):
        if not path.is_file():
            raise RuntimeError(f"GRAPH_PREFLIGHT_INPUT_MISSING:{path}")
    source = json.loads(source_validation.read_text(encoding="utf-8"))
    if (
        source.get("target_host") != "149"
        or source.get("status") != "PASS"
        or source.get("closure_status") != "PASS_FULL_28_CANCER_MATERIALIZATION_CLOSURE"
        or int(source.get("validated_cancer_count", -1)) != 28
    ):
        raise RuntimeError("GRAPH_SOURCE_MATERIALIZATION_NOT_VALIDATED")
    value = {
        "format": PREFLIGHT_FORMAT,
        "run_id": args.run_id,
        "target_host": "149",
        "cpu_only": True,
        "paid_compute": False,
        "formal_pointer_update_authorized": False,
        "source_metacell_result_root": str(args.source_result_root.resolve()),
        "source_validation_sha256": sha256(source_validation),
        "pathway_family_path": str(args.pathway_family.resolve()),
        "pathway_family_sha256": sha256(args.pathway_family),
        "external_evidence_path": str(args.external_evidence.resolve()),
        "external_evidence_sha256": sha256(args.external_evidence),
        "materializer": file_record(args.materializer),
        "launcher": file_record(Path(__file__).resolve()),
        "graph_asset_module": file_record(
            ROOT / "cc_hhgt" / "v32" / "single_cell_graph_assets.py"
        ),
        "frozen_metacell_materializer": file_record(
            ROOT / "scripts" / "materialize_single_cell_metacell_149.py"
        ),
        "created_unix": time.time(),
    }
    destination = args.output_root / "LAUNCH_PREFLIGHT.json"
    if destination.is_file():
        old = json.loads(destination.read_text(encoding="utf-8"))
        left, right = dict(old), dict(value)
        for field in ("created_unix", "lineage_binding_augmented_unix"):
            left.pop(field, None)
            right.pop(field, None)
        newly_bound = (
            "launcher", "graph_asset_module", "frozen_metacell_materializer",
        )
        missing = [field for field in newly_bound if field not in left]
        if missing:
            legacy_right = dict(right)
            for field in newly_bound:
                legacy_right.pop(field, None)
            if left != legacy_right or set(missing) != set(newly_bound):
                raise RuntimeError("GRAPH_PREFLIGHT_DRIFT")
            old.update({field: value[field] for field in newly_bound})
            old["lineage_binding_augmented_unix"] = time.time()
            atomic_json(destination, old)
            return old
        if left != right:
            raise RuntimeError("GRAPH_PREFLIGHT_DRIFT")
        return old
    atomic_json(destination, value)
    return value


def _available_gib(path: Path) -> float:
    stat = os.statvfs(path)
    return stat.f_bavail * stat.f_frsize / 1024**3


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
        memory = _memory_gib()
        scratch = _available_gib(self.args.scratch_root)
        output = _available_gib(self.args.output_root)
        if memory < self.args.min_memory_gib or scratch < self.args.min_scratch_gib or output < self.args.min_output_gib:
            raise RuntimeError(
                f"GRAPH_RESOURCE_GATE_FAILED:memory={memory:.1f},scratch={scratch:.1f},output={output:.1f}"
            )

    def write_cancer_success(
        self, cancer: str, source_folds: list[int]
    ) -> dict[str, object]:
        cancer_root = self.args.output_root / "cancers" / cancer
        source_success = self.args.source_result_root / "cancers" / cancer / "SUCCESS.json"
        static_success = cancer_root / "STATIC" / "SUCCESS.json"
        fold_successes = [
            cancer_root / f"outer_fold={fold}" / "SUCCESS.json"
            for fold in source_folds
        ]
        if (
            not source_folds or not source_success.is_file()
            or not static_success.is_file()
            or any(not path.is_file() for path in fold_successes)
        ):
            raise RuntimeError(f"CANCER_GRAPH_SUCCESS_INPUTS_INCOMPLETE:{cancer}")
        static_receipt = json.loads(static_success.read_text(encoding="utf-8"))
        if (
            static_receipt.get("state") != "CANCER_STATIC_GRAPH_ASSETS_READY"
            or static_receipt.get("cancer_id") != cancer
            or int(static_receipt.get("pathways_without_family_edges", -1)) != 0
            or int(static_receipt.get("pathway_family_edges", -1))
            != int(static_receipt.get("pathways", -2))
        ):
            raise RuntimeError(f"CANCER_STATIC_GRAPH_COVERAGE_FAILED:{cancer}")
        for fold, path in zip(source_folds, fold_successes, strict=True):
            receipt = json.loads(path.read_text(encoding="utf-8"))
            if (
                receipt.get("state") != "FOLD_GRAPH_ASSETS_READY"
                or receipt.get("cancer_id") != cancer
                or int(receipt.get("outer_fold_id", -1)) != fold
            ):
                raise RuntimeError(f"CANCER_FOLD_GRAPH_RECEIPT_FAILED:{cancer}:{fold}")
        destination = cancer_root / "SUCCESS.json"
        value = {
            "format": CANCER_SUCCESS_FORMAT,
            "state": "CANCER_GRAPH_ASSETS_COMPLETE",
            "target_host": "149",
            "cancer_id": cancer,
            "source_cancer_success": file_record(source_success),
            "cancer_static_success": file_record(static_success),
            "outer_fold_ids": source_folds,
            "outer_fold_successes": [file_record(path) for path in fold_successes],
            "launch_preflight": file_record(
                self.args.output_root / "LAUNCH_PREFLIGHT.json"
            ),
            "formal_pointer_update_authorized": False,
            "finished_unix": time.time(),
        }
        if destination.is_file():
            old = json.loads(destination.read_text(encoding="utf-8"))
            if (
                old.get("state") != "CANCER_GRAPH_ASSETS_COMPLETE"
                or old.get("cancer_id") != cancer
                or old.get("outer_fold_ids") != source_folds
            ):
                raise RuntimeError(f"CANCER_GRAPH_SUCCESS_DRIFT:{cancer}")
            for record in old.get("outer_fold_successes", []):
                path = Path(str(record.get("path", "")))
                if not path.is_file() or file_record(path) != record:
                    raise RuntimeError(f"CANCER_GRAPH_SUCCESS_HASH_DRIFT:{cancer}")
            return file_record(destination)
        atomic_json(destination, value)
        return file_record(destination)

    def run_one(self, cancer: str) -> dict[str, object]:
        cancer_root = self.args.output_root / "cancers" / cancer
        source_root = self.args.source_result_root / "cancers" / cancer
        source_folds = sorted(
            int(path.name.split("=", 1)[1])
            for path in source_root.glob("outer_fold=*")
            if (path / "SUCCESS.json").is_file()
        )
        if source_folds and all(
            (cancer_root / f"outer_fold={fold}" / "SUCCESS.json").is_file()
            for fold in source_folds
        ):
            success = self.write_cancer_success(cancer, source_folds)
            return {
                "state": "ALREADY_COMPLETE", "folds": source_folds,
                "cancer_success": success,
            }
        self.resource_gate()
        started = time.time()
        deadline = started + self.args.per_cancer_timeout - 300
        command = [
            self.args.timeout_binary, "--signal=TERM", "--kill-after=300s",
            f"{self.args.per_cancer_timeout}s", sys.executable, str(self.args.materializer),
            "--cancer-id", cancer, "--outer-fold", "all",
            "--source-result-root", str(self.args.source_result_root),
            "--output-root", str(self.args.output_root),
            "--scratch-root", str(self.args.scratch_root),
            "--repair-root", str(self.args.repair_root),
            "--pathway-family", str(self.args.pathway_family),
            "--external-evidence", str(self.args.external_evidence),
            "--deadline", str(deadline), "--delete-verified-scratch",
        ]
        log = self.logs / f"{cancer}.log"
        with log.open("w", encoding="utf-8") as handle:
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
                code = process.wait()
            finally:
                with self.lock:
                    self.processes.pop(cancer, None)
        complete = source_folds and all(
            (cancer_root / f"outer_fold={fold}" / "SUCCESS.json").is_file()
            for fold in source_folds
        )
        if code or not complete:
            return {
                "state": "FAILED",
                "return_code": code,
                "complete": bool(complete),
                "elapsed_seconds": time.time() - started,
                "log": str(log),
            }
        success = self.write_cancer_success(cancer, source_folds)
        return {
            "state": "COMPLETE",
            "folds": source_folds,
            "cancer_success": success,
            "elapsed_seconds": time.time() - started,
            "log": str(log),
        }

    def run(self) -> None:
        self.write("RUNNING")
        pending, active = list(self.cancers), {}
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
                        value = future.result()
                    except Exception as error:
                        value = {"state": "FAILED", "error": repr(error)}
                    self.states[cancer] = value
                    if str(value.get("state", "")).startswith("FAILED"):
                        self.failures += 1
                        if self.failures >= self.args.max_failures:
                            self.terminate()
                self.write("RUNNING")
        state = "COMPLETE" if not self.failures and not self.stop.is_set() else "FAILED"
        self.write(state)
        if state != "COMPLETE":
            raise SystemExit("graph asset batch did not complete")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run-id", required=True)
    result.add_argument("--source-result-root", required=True, type=Path)
    result.add_argument("--output-root", required=True, type=Path)
    result.add_argument("--scratch-root", required=True, type=Path)
    result.add_argument("--repair-root", required=True, type=Path)
    result.add_argument("--pathway-family", required=True, type=Path)
    result.add_argument("--external-evidence", required=True, type=Path)
    result.add_argument("--cancers", nargs="*", default=[])
    result.add_argument("--workers", type=int, choices=(1, 2), default=2)
    result.add_argument("--threads-per-job", type=int, default=24)
    result.add_argument("--per-cancer-timeout", type=int, default=21600)
    result.add_argument("--max-failures", type=int, default=1)
    result.add_argument("--min-memory-gib", type=float, default=48.0)
    result.add_argument("--min-scratch-gib", type=float, default=80.0)
    result.add_argument("--min-output-gib", type=float, default=300.0)
    result.add_argument("--timeout-binary", default="/usr/bin/timeout")
    result.add_argument(
        "--materializer",
        type=Path,
        default=ROOT / "scripts" / "materialize_single_cell_hhgt_assets_149.py",
    )
    return result


def main() -> None:
    args = parser().parse_args()
    if socket.gethostname() != "149":
        raise SystemExit("WRONG_HOST")
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.scratch_root.mkdir(parents=True, exist_ok=True)
    source = json.loads(
        (args.source_result_root / "MATERIALIZATION_VALIDATION.json").read_text(encoding="utf-8")
    )
    authorized = sorted(map(str, source.get("validated_cancers", ())))
    cancers = [value.upper() for value in args.cancers] if args.cancers else authorized
    if len(authorized) != 28 or not cancers or not set(cancers).issubset(authorized):
        raise SystemExit("REQUESTED_CANCERS_OUTSIDE_VALIDATED_28")
    if args.workers * args.threads_per_job > (os.cpu_count() or 1):
        raise SystemExit("THREAD_ALLOCATION_EXCEEDS_CPUS")
    initialize_preflight(args)
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join(
                [str(ROOT), "/public0/DSC/CancerLncAtlas/runtime/python_packages_v32_drug_20260825"]
            ),
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
