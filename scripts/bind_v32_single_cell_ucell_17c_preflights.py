#!/usr/bin/env python3
"""Hash-bind the 17 formal V3.2 cell-level UCell preflight reports."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from cc_hhgt.v32.input_lineage import artifact_sha256  # noqa: E402


FORMAL_CANCERS = (
    "ACC", "CHOL", "DLBC", "ESCA", "GBM", "HNSC", "KIRC", "LAML", "LGG",
    "LUSC", "MESO", "PCPG", "READ", "SARC", "SKCM", "THYM", "UCEC",
)
PREFLIGHT_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_CELL_LEVEL_PREFLIGHT_V2"
BINDING_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_UCELL_17C_PREFLIGHT_BINDING_V1"


class PreflightBindingError(RuntimeError):
    """Raised when a batch preflight cannot be bound without weakening it."""


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightBindingError(f"Invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise PreflightBindingError(f"JSON object required: {path}")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def bind_preflights(
    *, preflight_root: str | Path, expected_batch_success_sha256: str,
    output_root: str | Path,
) -> dict[str, Any]:
    source = Path(preflight_root).resolve()
    output = Path(output_root).resolve()
    if not source.is_dir() or source.is_symlink():
        raise PreflightBindingError(f"Unsafe preflight root: {source}")
    if output.exists():
        raise PreflightBindingError(f"Refusing output reuse: {output}")
    batch_path = source / "BATCH_SUCCESS.json"
    batch_sha = artifact_sha256(batch_path)
    if batch_sha != str(expected_batch_success_sha256).lower():
        raise PreflightBindingError("BATCH_SUCCESS SHA drift")
    batch = _load(batch_path)
    if (
        batch.get("status") != "PASS_17_FORMAL_CANCER_PREFLIGHTS"
        or batch.get("cell_level_ucell_started") is not False
        or batch.get("retraining_started") is not False
        or batch.get("historical_sc_trajectory_used") is not False
    ):
        raise PreflightBindingError("Batch preflight completion semantics drifted")

    records: list[dict[str, Any]] = []
    for cancer in FORMAL_CANCERS:
        path = source / f"{cancer}.PREFLIGHT.json"
        payload = _load(path)
        if payload.get("format") != PREFLIGHT_FORMAT:
            raise PreflightBindingError(f"Wrong preflight format: {cancer}")
        if payload.get("cancer_id") != cancer:
            raise PreflightBindingError(f"Cancer identity drift: {cancer}")
        if payload.get("pilot_safe_to_start") is not True:
            raise PreflightBindingError(f"Preflight is not safe to start: {cancer}")
        if payload.get("inputs", {}).get("exact_pathways") != 2135:
            raise PreflightBindingError(f"Exact-pathway coverage drift: {cancer}")
        ucell = payload.get("ucell", {})
        if (
            ucell.get("numeric_output_permitted") is not True
            or ucell.get("complete_exact_signature_length_used_in_denominator") is not True
            or ucell.get("unavailable_values_filled_with_zero_or_half") is not False
        ):
            raise PreflightBindingError(f"UCell contract drift: {cancer}")
        if any(
            payload.get(key) is not False
            for key in (
                "historical_predictions_used", "historical_rankings_used",
                "historical_checkpoints_used", "historical_sc_trajectory_used",
                "training_started", "production_deployed",
            )
        ):
            raise PreflightBindingError(f"Freshness/release gate drift: {cancer}")
        records.append(
            {
                "cancer_id": cancer,
                "path": str(path),
                "sha256": artifact_sha256(path),
                "cells": int(payload.get("metadata_audit", {}).get("cells", -1)),
                "pathways_available": int(ucell.get("pathways_available", -1)),
                "pathways_unavailable": int(ucell.get("pathways_unavailable", -1)),
                "expected_numeric_rows": int(
                    ucell.get("cell_level_output_rows_expected", -1)
                ),
                "pseudotime_numeric_permitted": bool(
                    payload.get("pseudotime", {}).get("numeric_output_permitted")
                ),
            }
        )

    observed = sorted(path.stem.split(".")[0] for path in source.glob("*.PREFLIGHT.json"))
    if observed != sorted(FORMAL_CANCERS):
        raise PreflightBindingError("Preflight root is not exactly the 17 formal cancers")
    output.mkdir(parents=True)
    binding = {
        "format": BINDING_FORMAT,
        "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
        "status": "PASS_HASH_BOUND_17_FORMAL_PREFLIGHTS",
        "batch_success": {"path": str(batch_path), "sha256": batch_sha},
        "formal_cancers": list(FORMAL_CANCERS),
        "formal_cancer_count": 17,
        "formal_cells": sum(record["cells"] for record in records),
        "records": records,
        "historical_sc_trajectory_used": False,
        "historical_predictions_used": False,
        "historical_rankings_used": False,
        "historical_checkpoints_used": False,
        "cell_level_ucell_started": False,
        "retraining_started": False,
        "release_ready": False,
        "production_deployed": False,
    }
    binding_path = output / "PREFLIGHT_BATCH_BINDING.json"
    _write(binding_path, binding)
    binding_sha = artifact_sha256(binding_path)
    success = {
        "status": binding["status"],
        "binding_path": str(binding_path),
        "binding_sha256": binding_sha,
        "formal_cancer_count": 17,
        "cell_level_ucell_started": False,
        "retraining_started": False,
        "release_ready": False,
        "production_deployed": False,
    }
    _write(output / "SUCCESS.json", success)
    return success


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-root", required=True, type=Path)
    parser.add_argument("--expected-batch-success-sha256", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    result = bind_preflights(
        preflight_root=args.preflight_root,
        expected_batch_success_sha256=args.expected_batch_success_sha256,
        output_root=args.output_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
