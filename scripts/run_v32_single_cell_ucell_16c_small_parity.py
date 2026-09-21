#!/usr/bin/env python3
"""Run hash-bound official UCell small-parity gates for the 16 missing cancers."""
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


BINDING_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_UCELL_17C_PREFLIGHT_BINDING_V1"
SMALL_BATCH_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_UCELL_16C_SMALL_PARITY_BATCH_V1"
FORMAL_CANCERS = (
    "ACC", "CHOL", "DLBC", "ESCA", "GBM", "HNSC", "KIRC", "LAML", "LGG",
    "LUSC", "MESO", "PCPG", "READ", "SARC", "SKCM", "THYM", "UCEC",
)


class SmallParityBatchError(RuntimeError):
    """Raised when a per-cancer small parity gate fails closed."""


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SmallParityBatchError(f"JSON object required: {path}")
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
        raise SmallParityBatchError("Small-parity batch runner SHA drift")
    binding_path = args.preflight_binding.resolve()
    if artifact_sha256(binding_path) != args.expected_preflight_binding_sha256.lower():
        raise SmallParityBatchError("Preflight binding SHA drift")
    binding = _load(binding_path)
    if (
        binding.get("format") != BINDING_FORMAT
        or binding.get("status") != "PASS_HASH_BOUND_17_FORMAL_PREFLIGHTS"
        or tuple(binding.get("formal_cancers", ())) != FORMAL_CANCERS
        or binding.get("cell_level_ucell_started") is not False
    ):
        raise SmallParityBatchError("Preflight binding semantics drifted")
    runtime = args.runtime_root.resolve()
    runner = runtime / "scripts" / "run_v32_single_cell_cell_level_general.py"
    implementation = runtime / "scripts" / "run_v32_single_cell_cell_level_pilot.py"
    official_parity = runtime / "references" / "official_ucell" / "PARITY.json"
    pyucell_root = runtime / "references" / "official_ucell" / "pyucell-0.7.3-wheel"
    for path, expected, role in (
        (runner, args.expected_runner_sha256, "generalized runner"),
        (implementation, args.expected_implementation_sha256, "UCell implementation"),
        (official_parity, args.expected_official_parity_sha256, "official parity"),
        (pyucell_root / "pyucell" / "scoring.py", args.expected_scoring_sha256, "scoring.py"),
        (pyucell_root / "pyucell" / "ranks.py", args.expected_ranks_sha256, "ranks.py"),
    ):
        if artifact_sha256(path) != str(expected).lower():
            raise SmallParityBatchError(f"{role} SHA drift")
    output = args.output_root.resolve()
    if output.exists():
        raise SmallParityBatchError(f"Refusing output reuse: {output}")
    output.mkdir(parents=True)
    logs = output / "logs"
    logs.mkdir()
    records = {row["cancer_id"]: row for row in binding.get("records", [])}
    if set(records) != set(FORMAL_CANCERS):
        raise SmallParityBatchError("Preflight binding does not contain exactly 17 records")
    cancers = tuple(cancer for cancer in FORMAL_CANCERS if cancer != "HNSC")

    def execute(cancer: str) -> dict[str, Any]:
        declaration = records[cancer]
        preflight = Path(declaration["path"]).resolve()
        if artifact_sha256(preflight) != declaration["sha256"]:
            raise SmallParityBatchError(f"Preflight SHA drift: {cancer}")
        cancer_output = output / f"cancer_id={cancer}"
        command = [
            str(args.python_executable.resolve()), str(runner),
            "--mode", "small",
            "--preflight-json", str(preflight),
            "--expected-preflight-sha256", declaration["sha256"],
            "--official-parity-json", str(official_parity),
            "--expected-official-parity-sha256", args.expected_official_parity_sha256,
            "--output-root", str(cancer_output),
            "--max-rank", "1500",
            "--chunk-size", "64",
            "--small-cells", "16",
            "--small-pathways", "32",
            "--official-pyucell-root", str(pyucell_root),
            "--expected-pyucell-scoring-sha256", args.expected_scoring_sha256,
            "--expected-pyucell-ranks-sha256", args.expected_ranks_sha256,
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
            raise SmallParityBatchError(
                f"Small parity failed for {cancer} with exit {completed.returncode}"
            )
        parity_path = cancer_output / "SMALL_CHUNK_PARITY.json"
        status_path = cancer_output / "STATUS.json"
        parity = _load(parity_path)
        status = _load(status_path)
        if (
            parity.get("status") != "PASS"
            or parity.get("present_missing_count_audit_pass") is not True
            or parity.get("v32_vs_pinned_helper_functions_r_pass") is not True
            or parity.get("official_pyucell_gate_pass") is not True
            or status.get("cancer_id") != cancer
            or status.get("historical_sc_trajectory_used") is not False
        ):
            raise SmallParityBatchError(f"Small parity semantics failed: {cancer}")
        return {
            "cancer_id": cancer,
            "output_root": str(cancer_output),
            "output_sha256": artifact_sha256(cancer_output),
            "parity_path": str(parity_path),
            "parity_sha256": artifact_sha256(parity_path),
            "status_path": str(status_path),
            "status_sha256": artifact_sha256(status_path),
            "cells_tested": int(parity.get("cells", -1)),
            "pathways_tested": int(parity.get("pathways", -1)),
            "r_max_abs_diff": float(
                parity.get("v32_vs_pinned_helper_functions_r_max_abs_diff", float("nan"))
            ),
            "pyucell_max_abs_diff": float(
                parity.get("v32_vs_official_pyucell_0_7_3_max_abs_diff", float("nan"))
            ),
        }

    results: list[dict[str, Any]] = []
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(execute, cancer): cancer for cancer in cancers}
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
            "passed_cancers": sorted(row["cancer_id"] for row in results),
            "full_cell_level_ucell_started": False,
        }
        _write(output / "FAILURE.json", failure)
        raise SmallParityBatchError(json.dumps(failure, sort_keys=True))
    results.sort(key=lambda row: row["cancer_id"])
    batch = {
        "format": SMALL_BATCH_FORMAT,
        "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
        "status": "PASS_16_MISSING_FORMAL_CANCER_SMALL_PARITY",
        "preflight_binding": {
            "path": str(binding_path),
            "sha256": args.expected_preflight_binding_sha256.lower(),
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
        "cancers": list(cancers),
        "cancer_count": 16,
        "existing_hnsc_result_reused_or_modified": False,
        "historical_sc_trajectory_used": False,
        "historical_predictions_used": False,
        "historical_rankings_used": False,
        "historical_checkpoints_used": False,
        "full_cell_level_ucell_started": False,
        "retraining_started": False,
        "release_ready": False,
        "production_deployed": False,
    }
    batch_path = output / "SMALL_PARITY_BATCH_BINDING.json"
    _write(batch_path, batch)
    success = {
        "status": batch["status"],
        "binding_path": str(batch_path),
        "binding_sha256": artifact_sha256(batch_path),
        "cancer_count": 16,
        "full_cell_level_ucell_started": False,
        "release_ready": False,
        "production_deployed": False,
    }
    _write(output / "SUCCESS.json", success)
    return success


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-binding", required=True, type=Path)
    parser.add_argument("--expected-batch-runner-sha256", required=True)
    parser.add_argument("--expected-preflight-binding-sha256", required=True)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--expected-runner-sha256", required=True)
    parser.add_argument("--expected-implementation-sha256", required=True)
    parser.add_argument("--python-executable", required=True, type=Path)
    parser.add_argument("--expected-official-parity-sha256", required=True)
    parser.add_argument("--expected-scoring-sha256", required=True)
    parser.add_argument("--expected-ranks-sha256", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 9))
    return parser


def main() -> int:
    result = run_batch(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
