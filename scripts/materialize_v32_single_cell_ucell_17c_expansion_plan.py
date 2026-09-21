#!/usr/bin/env python3
"""Bind the fail-closed 17-cancer UCell preflight and remaining compute gates."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.single_cell_formal_context_query import (  # noqa: E402
    FORMAL_CANCERS,
    SingleCellFormalContextQuery,
    artifact_sha256,
)


FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_UCELL_17C_EXPANSION_PLAN_V1"
EXPECTED_R6 = {
    "RUN_STATUS.json": "3a9ec28a6168f1e38d65521d9e2d7fff818becef6f6d74a3dd9846c1715c0423",
    "TRAINING_HANDOFF.json": "627fbe52df38e58692b4091d9d1ab870da6629c6eec56f9c6eb7aa5c11c03ad4",
    "dataset_manifest_33c.parquet": "bea2d20a261063dacdd05b3b18865f8de0fec3504086e08e6295b63fe0a4ffad",
}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-binding", required=True, type=Path)
    parser.add_argument("--expected-context-binding-sha256", required=True)
    parser.add_argument("--r6-control-root", required=True, type=Path)
    parser.add_argument("--preflight-runner", required=True, type=Path)
    parser.add_argument("--batch-preflight-launcher", required=True, type=Path)
    parser.add_argument("--hnsc-only-runner", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    context_path = args.context_binding.resolve()
    context = SingleCellFormalContextQuery(
        context_path,
        expected_binding_sha256=args.expected_context_binding_sha256,
    )
    r6_root = args.r6_control_root.resolve()
    if not r6_root.is_dir() or r6_root.is_symlink():
        raise RuntimeError(f"Invalid r6 control root: {r6_root}")
    controls = {}
    for name, expected in EXPECTED_R6.items():
        path = (r6_root / name).resolve()
        observed = artifact_sha256(path)
        if observed != expected:
            raise RuntimeError(f"r6 control SHA drift for {name}: {observed}")
        controls[name] = {"path": str(path), "sha256": observed}
    run = load_json(r6_root / "RUN_STATUS.json")
    if (
        run.get("blocked_fresh_direct_id_counts_typed_unavailable") is not True
        or set(run.get("formal_eligible_cancers", [])) != set(FORMAL_CANCERS)
        or run.get("fresh_ucell_status") != "PENDING_CELL_LEVEL_FRESH_RECOMPUTE"
        or any(
            run.get("per_cancer", {}).get(cancer, {}).get(
                "historical_sc_trajectory_used"
            )
            is not False
            for cancer in FORMAL_CANCERS
        )
    ):
        raise RuntimeError("r6 control semantics are not the formal corrected authority")

    code = {}
    for role, raw_path in {
        "preflight_runner": args.preflight_runner,
        "batch_preflight_launcher": args.batch_preflight_launcher,
        "hnsc_only_runner": args.hnsc_only_runner,
    }.items():
        path = raw_path.resolve()
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Missing source file: {role}")
        code[role] = {"path": str(path), "sha256": artifact_sha256(path)}
    preflight_source = args.preflight_runner.read_text(encoding="utf-8")
    launcher_source = args.batch_preflight_launcher.read_text(encoding="utf-8")
    hnsc_source = args.hnsc_only_runner.read_text(encoding="utf-8")
    if "--r6-root" not in preflight_source or "pilot_safe_to_start" not in preflight_source:
        raise RuntimeError("Preflight runner lacks the formal fail-closed gates")
    if "cell_level_ucell_started\": false" not in launcher_source or "nohup" in launcher_source:
        raise RuntimeError("Batch preflight launcher may not start cell-level computation")
    if "This restricted pilot is HNSC-only" not in hnsc_source:
        raise RuntimeError("Existing full runner is not explicitly HNSC-only")

    expansion = context.binding.get("ucell_expansion_inputs", [])
    if len(expansion) != 17 or {row.get("cancer_id") for row in expansion} != set(FORMAL_CANCERS):
        raise RuntimeError("Context binding lacks 17 unique UCell expansion inputs")
    if sum(row.get("cell_level_ucell_status") == "FRESH_HASH_BOUND_COMPLETE" for row in expansion) != 1:
        raise RuntimeError("Exactly one formal cancer must currently have fresh cell-level UCell")

    destination = args.output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    plan_path = destination / "UCELL_17C_EXPANSION_PLAN.json"
    success_path = destination / "SUCCESS.json"
    if plan_path.exists() or success_path.exists():
        raise RuntimeError(f"Refusing to overwrite expansion plan: {destination}")
    plan = {
        "format": FORMAT,
        "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PREFLIGHT_LAUNCH_READY_FULL_COMPUTE_RUNNER_NOT_READY",
        "context_binding": {
            "path": str(context_path),
            "sha256": args.expected_context_binding_sha256,
        },
        "r6_controls": controls,
        "code": code,
        "formal_cancers": sorted(FORMAL_CANCERS),
        "formal_cancer_count": 17,
        "formal_cells_scanned": sum(int(row.get("cells_scanned", 0)) for row in expansion),
        "expansion_inputs": expansion,
        "current_cell_level_ucell": {
            "covered_cancers": ["HNSC"],
            "covered_count": 1,
            "missing_formal_count": 16,
        },
        "execution_state": {
            "preflight_started": False,
            "cell_level_ucell_started": False,
            "retraining_started": False,
            "duplicate_run_started": False,
        },
        "gates": [
            {
                "stage": 1,
                "name": "RUN_17C_READ_ONLY_PREFLIGHT",
                "status": "READY_NOT_EXECUTED",
                "launcher": str(args.batch_preflight_launcher.resolve()),
            },
            {
                "stage": 2,
                "name": "GENERALIZE_NON_HNSC_CELL_LEVEL_RUNNER",
                "status": "BLOCKED_BY_HNSC_ONLY_IMPLEMENTATION",
                "requirement": "NEW_RUNNER_MUST_NOT_MUTATE_OR_REUSE_HNSC_PILOT_OUTPUT",
            },
            {
                "stage": 3,
                "name": "PER_CANCER_SMALL_OFFICIAL_UCELL_PARITY",
                "status": "PENDING_STAGE_2",
            },
            {
                "stage": 4,
                "name": "REMOTE_CELL_LEVEL_UCELL_16_CANCERS",
                "status": "PENDING_STAGES_1_TO_3",
            },
            {
                "stage": 5,
                "name": "INDEPENDENT_OUTPUT_AUDIT_AND_QUERY_BINDING",
                "status": "PENDING_STAGE_4",
            },
            {
                "stage": 6,
                "name": "PRIVATE_HEAD_V2_CELLTYPE_PATHWAY_FEATURE_RETRAIN",
                "status": "PENDING_STAGE_5",
                "reason": (
                    "CURRENT_PRIVATE_HEAD_COLLAPSES_PATHWAY_ACTIVITY_TO_"
                    "DATASET_X_PATHWAY_MEAN"
                ),
            },
        ],
        "historical_sc_trajectory_used": False,
        "historical_predictions_used": False,
        "historical_rankings_used": False,
        "changes_exact_primary_score": False,
        "release_ready": False,
        "production_deployed": False,
    }
    atomic_json(plan_path, plan)
    plan_sha = artifact_sha256(plan_path)
    atomic_json(
        success_path,
        {
            "status": plan["status"],
            "plan": plan_path.name,
            "plan_sha256": plan_sha,
            "preflight_started": False,
            "cell_level_ucell_started": False,
            "retraining_started": False,
            "release_ready": False,
            "production_deployed": False,
        },
    )
    print(
        json.dumps(
            {
                "status": plan["status"],
                "plan_path": str(plan_path),
                "plan_sha256": plan_sha,
                "success_path": str(success_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
