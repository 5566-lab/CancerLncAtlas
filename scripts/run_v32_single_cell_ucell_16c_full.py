#!/usr/bin/env python3
"""Run full fresh cell-level UCell for the 16 formal cancers missing HNSC."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from cc_hhgt.v32.input_lineage import artifact_sha256  # noqa: E402


SMALL_BATCH_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_UCELL_16C_SMALL_PARITY_BATCH_V1"
FULL_BATCH_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_UCELL_16C_FULL_BATCH_V1"
MISSING_CANCERS = (
    "ACC", "CHOL", "DLBC", "ESCA", "GBM", "KIRC", "LAML", "LGG", "LUSC",
    "MESO", "PCPG", "READ", "SARC", "SKCM", "THYM", "UCEC",
)


class FullUCellBatchError(RuntimeError):
    """Raised when full per-cancer UCell materialization fails closed."""


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FullUCellBatchError(f"JSON object required: {path}")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run_batch(args: argparse.Namespace) -> dict[str, Any]:
    batch_runner = Path(__file__).resolve()
    if artifact_sha256(batch_runner) != args.expected_batch_runner_sha256.lower():
        raise FullUCellBatchError("Full UCell batch runner SHA drift")
    small_path = args.small_parity_binding.resolve()
    if artifact_sha256(small_path) != args.expected_small_parity_binding_sha256.lower():
        raise FullUCellBatchError("Small-parity batch binding SHA drift")
    small = _load(small_path)
    if (
        small.get("format") != SMALL_BATCH_FORMAT
        or small.get("status") != "PASS_16_MISSING_FORMAL_CANCER_SMALL_PARITY"
        or tuple(small.get("cancers", ())) != MISSING_CANCERS
        or small.get("full_cell_level_ucell_started") is not False
        or small.get("existing_hnsc_result_reused_or_modified") is not False
    ):
        raise FullUCellBatchError("Small-parity batch semantics drifted")
    records = {row["cancer_id"]: row for row in small.get("records", [])}
    if set(records) != set(MISSING_CANCERS):
        raise FullUCellBatchError("Small-parity batch lacks the exact 16 cancers")

    runtime = args.runtime_root.resolve()
    runner = runtime / "scripts" / "run_v32_single_cell_cell_level_general.py"
    implementation = runtime / "scripts" / "run_v32_single_cell_cell_level_pilot.py"
    official_parity = runtime / "references" / "official_ucell" / "PARITY.json"
    for path, expected, role in (
        (runner, args.expected_runner_sha256, "generalized runner"),
        (implementation, args.expected_implementation_sha256, "UCell implementation"),
        (official_parity, args.expected_official_parity_sha256, "official parity"),
    ):
        if artifact_sha256(path) != str(expected).lower():
            raise FullUCellBatchError(f"{role} SHA drift")
    output = args.output_root.resolve()
    if output.exists():
        raise FullUCellBatchError(f"Refusing output reuse: {output}")
    output.mkdir(parents=True)
    logs = output / "logs"
    logs.mkdir()

    def execute(cancer: str) -> dict[str, Any]:
        small_record = records[cancer]
        small_root = Path(small_record["output_root"]).resolve()
        if artifact_sha256(small_root) != small_record["output_sha256"]:
            raise FullUCellBatchError(f"Small output SHA drift: {cancer}")
        status = _load(Path(small_record["status_path"]).resolve())
        preflight = Path(status["preflight_path"]).resolve()
        preflight_sha = str(status["preflight_sha256"])
        if artifact_sha256(preflight) != preflight_sha:
            raise FullUCellBatchError(f"Preflight SHA drift: {cancer}")
        parity = Path(small_record["parity_path"]).resolve()
        parity_sha = str(small_record["parity_sha256"])
        if artifact_sha256(parity) != parity_sha:
            raise FullUCellBatchError(f"Small parity SHA drift: {cancer}")
        cancer_output = output / f"cancer_id={cancer}"
        command = [
            str(args.python_executable.resolve()), str(runner),
            "--mode", "full",
            "--preflight-json", str(preflight),
            "--expected-preflight-sha256", preflight_sha,
            "--official-parity-json", str(official_parity),
            "--expected-official-parity-sha256", args.expected_official_parity_sha256,
            "--output-root", str(cancer_output),
            "--max-rank", "1500",
            "--chunk-size", str(args.chunk_size),
            "--small-parity-json", str(parity),
            "--expected-small-parity-sha256", parity_sha,
        ]
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(runtime)
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            check=False,
        )
        (logs / f"{cancer}.stdout.log").write_text(completed.stdout, encoding="utf-8")
        (logs / f"{cancer}.stderr.log").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise FullUCellBatchError(
                f"Full UCell failed for {cancer} with exit {completed.returncode}"
            )
        success_path = cancer_output / "SUCCESS.json"
        result = _load(success_path)
        if (
            result.get("status") != "SUCCESS"
            or result.get("cancer_id") != cancer
            or result.get("full_cell_level_completed") is not True
            or result.get("unavailable_values_filled_with_zero_or_half") is not False
            or result.get("single_cell_module_complete") is not False
        ):
            raise FullUCellBatchError(f"Full UCell success semantics drift: {cancer}")
        if (
            int(result.get("cell_level_coverage_rows", -1))
            != int(result.get("cells", -1)) * int(result.get("pathways_total_coverage", -1))
        ):
            raise FullUCellBatchError(f"Coverage row count drift: {cancer}")
        return {
            "cancer_id": cancer,
            "output_root": str(cancer_output),
            "output_sha256": artifact_sha256(cancer_output),
            "success_path": str(success_path),
            "success_sha256": artifact_sha256(success_path),
            "cells": int(result["cells"]),
            "pathways_total": int(result["pathways_total_coverage"]),
            "pathways_available": int(result["pathways_available"]),
            "pathways_typed_unavailable": int(result["pathways_typed_unavailable"]),
            "coverage_rows": int(result["cell_level_coverage_rows"]),
            "numeric_rows": int(result["cell_level_numeric_score_rows"]),
            "typed_unavailable_rows": int(result["cell_level_typed_unavailable_rows"]),
            "pseudotime_status": result.get("pseudotime_status"),
        }

    results: list[dict[str, Any]] = []
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(execute, cancer): cancer for cancer in MISSING_CANCERS}
        for future in as_completed(futures):
            cancer = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:  # noqa: BLE001 - aggregate all fail-closed results
                failures[cancer] = f"{type(exc).__name__}: {exc}"
    if failures:
        failure = {
            "status": "FAIL",
            "fail_closed": True,
            "failures": dict(sorted(failures.items())),
            "completed_cancers": sorted(row["cancer_id"] for row in results),
            "existing_hnsc_result_reused_or_modified": False,
            "retraining_started": False,
        }
        _write(output / "FAILURE.json", failure)
        raise FullUCellBatchError(json.dumps(failure, sort_keys=True))

    results.sort(key=lambda row: row["cancer_id"])
    batch = {
        "format": FULL_BATCH_FORMAT,
        "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
        "status": "SUCCESS_16_MISSING_FORMAL_CANCERS_FULL_UCELL",
        "small_parity_binding": {
            "path": str(small_path),
            "sha256": args.expected_small_parity_binding_sha256.lower(),
        },
        "generalized_runner": {"path": str(runner), "sha256": args.expected_runner_sha256},
        "ucell_implementation": {
            "path": str(implementation),
            "sha256": args.expected_implementation_sha256,
        },
        "batch_runner": {
            "path": str(batch_runner),
            "sha256": args.expected_batch_runner_sha256.lower(),
        },
        "records": results,
        "cancers": list(MISSING_CANCERS),
        "cancer_count": 16,
        "cells": sum(row["cells"] for row in results),
        "coverage_rows": sum(row["coverage_rows"] for row in results),
        "numeric_rows": sum(row["numeric_rows"] for row in results),
        "typed_unavailable_rows": sum(row["typed_unavailable_rows"] for row in results),
        "existing_hnsc_result_reused_or_modified": False,
        "historical_sc_trajectory_used": False,
        "historical_predictions_used": False,
        "historical_rankings_used": False,
        "historical_checkpoints_used": False,
        "retraining_started": False,
        "release_ready": False,
        "production_deployed": False,
    }
    binding_path = output / "FULL_UCELL_16C_BINDING.json"
    _write(binding_path, batch)
    success = {
        "status": batch["status"],
        "binding_path": str(binding_path),
        "binding_sha256": artifact_sha256(binding_path),
        "cancer_count": 16,
        "cells": batch["cells"],
        "coverage_rows": batch["coverage_rows"],
        "existing_hnsc_result_reused_or_modified": False,
        "retraining_started": False,
        "release_ready": False,
        "production_deployed": False,
    }
    _write(output / "SUCCESS.json", success)
    return success


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--small-parity-binding", required=True, type=Path)
    parser.add_argument("--expected-batch-runner-sha256", required=True)
    parser.add_argument("--expected-small-parity-binding-sha256", required=True)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--expected-runner-sha256", required=True)
    parser.add_argument("--expected-implementation-sha256", required=True)
    parser.add_argument("--python-executable", required=True, type=Path)
    parser.add_argument("--expected-official-parity-sha256", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=3, choices=range(1, 7))
    parser.add_argument("--chunk-size", type=int, default=64, choices=(32, 64, 128))
    return parser


def main() -> int:
    result = run_batch(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
