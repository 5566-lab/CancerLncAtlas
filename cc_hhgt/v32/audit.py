from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import canonical_json_sha256, validate_contract
from .orchestration import read_task_manifest
from .training_guard import load_structured_mapping


TRAINING_ARTIFACT_PATTERNS = ("*.pt", "*.pth", "*.ckpt")


def scan_training_artifacts(root: str | Path) -> list[str]:
    root = Path(root)
    paths: set[Path] = set()
    for pattern in TRAINING_ARTIFACT_PATTERNS:
        paths.update(path for path in root.rglob(pattern) if path.is_file())
    return sorted(str(path.resolve()) for path in paths)


def build_code_only_readiness(
    *,
    repo_root: str | Path,
    config_path: str | Path,
    task_manifest_path: str | Path,
    dry_run_report_path: str | Path,
    test_summary: Mapping[str, Any],
    process_matches: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    config = load_structured_mapping(config_path)
    validate_contract(config)
    tasks = read_task_manifest(task_manifest_path)
    dry_run = json.loads(Path(dry_run_report_path).read_text(encoding="utf-8"))
    artifacts = scan_training_artifacts(root)
    approval = root / "TRAINING_APPROVAL.json"
    checks = {
        "execution_mode_code_only": config["execution_control"]["execution_mode"] == "CODE_ONLY",
        "training_not_authorized": config["execution_control"]["training_authorized"] is False,
        "paid_disabled": config["execution_control"]["paid_enabled"] is False,
        "paid_cost_cap_zero": float(config["execution_control"]["max_cost_cny"]) == 0.0,
        "five_tasks": len(tasks) == 5,
        "all_tasks_blocked": all(row["status"] == "BLOCKED" for row in tasks),
        "all_tasks_local4070": all(row["owner"] == "local4070" for row in tasks),
        "zero_paid_tasks": all(row["paid_task"].lower() == "false" for row in tasks),
        "dry_run_attests_no_training": dry_run["safety_attestation"]["training_started"] is False,
        "training_approval_absent": not approval.exists(),
        "real_training_artifacts_absent": not artifacts,
        "local_training_processes_absent": len(process_matches) == 0,
        "tests_passed": bool(test_summary.get("passed", False)),
    }
    payload = {
        "report_format": "CC_HHGT_V3_2_IMPLEMENTATION_READY_NOT_TRAINED_V1",
        "analysis_version": config.get("analysis_version"),
        "status": "IMPLEMENTATION_READY_NOT_TRAINED" if all(checks.values()) else "NOT_READY",
        "training_authorized": False,
        "training_started": False,
        "optimizer_step_executed": False,
        "cuda_training_initialized": False,
        "server149_contacted": False,
        "paid_instance_started": False,
        "paid_cost_cny": 0.0,
        "website_replaced": False,
        "checks": checks,
        "training_artifacts": artifacts,
        "training_process_matches": list(process_matches),
        "test_summary": dict(test_summary),
    }
    payload["report_sha256"] = canonical_json_sha256(payload)
    return payload


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return destination
