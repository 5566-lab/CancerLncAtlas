from __future__ import annotations

import json
from pathlib import Path

import pytest

from cc_hhgt.v32.orchestration import (
    DEFAULT_SEED,
    ExecutionPolicy,
    OrchestrationContractError,
    build_dry_run_report,
    build_task_manifest,
    build_virtual_checkpoint,
    endpoint_registry,
    task_manifest_sha256,
    task_manifest_tsv,
    validate_virtual_resume,
    write_dry_run_report,
    write_task_manifest,
)


def _hashes() -> dict[str, str]:
    return {
        "code_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "input_manifest_sha256": "c" * 64,
        "task_manifest_sha256": "d" * 64,
    }


def test_default_policy_creates_exactly_five_blocked_local_tasks() -> None:
    policy = ExecutionPolicy()
    tasks = build_task_manifest(run_id="review", policy=policy)

    assert policy.execution_mode == "CODE_ONLY"
    assert policy.training_authorized is False
    assert policy.paid_enabled is False
    assert policy.max_paid_hours == 0
    assert policy.max_cost_cny == 0
    assert len(tasks) == 5
    assert {row["patient_fold"] for row in tasks} == set(range(5))
    assert {row["seed"] for row in tasks} == {DEFAULT_SEED}
    assert {row["owner"] for row in tasks} == {"local4070"}
    assert {row["status"] for row in tasks} == {"BLOCKED"}
    assert not any(row["paid_task"] for row in tasks)
    assert all("CODE_ONLY" in row["blocked_reason"] for row in tasks)


def test_endpoint_registry_is_descriptive_and_paid_is_disabled() -> None:
    endpoints = {row["endpoint_id"]: row for row in endpoint_registry()}
    assert endpoints["server149"]["remote_contacted"] is False
    assert endpoints["server149"]["may_train_cc_hhgt"] is False
    assert endpoints["local4070"]["may_train_cc_hhgt"] is False
    assert endpoints["paid_gpu"]["enabled"] is False
    assert endpoints["paid_gpu"]["max_cost_cny"] == 0


def test_manifest_serialization_is_deterministic(tmp_path: Path) -> None:
    tasks = build_task_manifest(run_id="deterministic")
    first = task_manifest_tsv(tasks)
    second = task_manifest_tsv(tasks)
    assert first == second
    assert task_manifest_sha256(tasks) == task_manifest_sha256(tasks)
    path = write_task_manifest(tmp_path / "TASK_MANIFEST.tsv", tasks)
    assert path.read_text(encoding="utf-8") == first
    assert first.count("\n") == 6


def test_virtual_checkpoint_validates_same_hardware_and_hashes() -> None:
    task = build_task_manifest(run_id="virtual")[0]
    checkpoint = build_virtual_checkpoint(task, artifact_hashes=_hashes())
    validate_virtual_resume(
        checkpoint,
        task,
        artifact_hashes=_hashes(),
        hardware_class="LOCAL_RTX_4070_TI_SUPER_16GB",
    )
    assert checkpoint["virtual_only"] is True
    assert checkpoint["contains_model_state"] is False
    assert checkpoint["optimizer_steps"] == 0

    with pytest.raises(OrchestrationContractError, match="Cross-hardware"):
        validate_virtual_resume(
            checkpoint,
            task,
            artifact_hashes=_hashes(),
            hardware_class="PAID_PREEMPTIBLE_GPU",
        )


def test_virtual_checkpoint_rejects_artifact_drift() -> None:
    task = build_task_manifest(run_id="drift")[0]
    checkpoint = build_virtual_checkpoint(task, artifact_hashes=_hashes())
    changed = {**_hashes(), "config_sha256": "e" * 64}
    with pytest.raises(OrchestrationContractError, match="hash mismatch"):
        validate_virtual_resume(
            checkpoint,
            task,
            artifact_hashes=changed,
            hardware_class=task["hardware_class"],
        )


def test_dry_run_attests_no_training_cuda_remote_or_cost(tmp_path: Path) -> None:
    report = build_dry_run_report(run_id="dry")
    assert report["task_summary"] == {
        "total": 5,
        "blocked": 5,
        "local4070": 5,
        "server149_training": 0,
        "paid": 0,
    }
    assert set(report["safety_attestation"].values()) <= {False, 0.0}
    path = write_dry_run_report(tmp_path / "DRY_RUN_REPORT.json", report)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["safety_attestation"]["training_started"] is False


def test_paid_policy_cannot_be_enabled_with_zero_caps() -> None:
    with pytest.raises(OrchestrationContractError, match="positive"):
        ExecutionPolicy.from_mapping(
            {
                "execution_mode": "TRAINING",
                "training_authorized": True,
                "paid_enabled": True,
                "max_paid_hours": 0,
                "max_cost_cny": 0,
            }
        )


def test_explicitly_capped_paid_gpu_emits_five_paid_pending_tasks() -> None:
    policy = ExecutionPolicy.from_mapping(
        {
            "execution_mode": "TRAINING",
            "training_authorized": True,
            "paid_enabled": True,
            "max_paid_hours": 96,
            "max_cost_cny": 200,
        }
    )
    tasks = build_task_manifest(
        run_id="fresh-paid",
        policy=policy,
        owner="paid_gpu",
        hardware_class="PAID_PREEMPTIBLE_GPU",
        paid_task=True,
    )
    assert len(tasks) == 5
    assert {row["owner"] for row in tasks} == {"paid_gpu"}
    assert {row["hardware_class"] for row in tasks} == {"PAID_PREEMPTIBLE_GPU"}
    assert {row["status"] for row in tasks} == {"PENDING"}
    assert all(row["paid_task"] is True for row in tasks)

    with pytest.raises(OrchestrationContractError, match="explicitly capped"):
        build_task_manifest(
            run_id="uncapped-paid",
            policy=ExecutionPolicy(
                execution_mode="TRAINING", training_authorized=True
            ),
            owner="paid_gpu",
            hardware_class="PAID_PREEMPTIBLE_GPU",
            paid_task=True,
        )
