from __future__ import annotations

import copy

import pytest

from cc_hhgt.v32.full_orchestration import (
    ALL_COMPONENTS,
    AUXILIARY_COMPONENTS,
    CORE_COMPONENT,
    FULL_TASK_COUNT,
    build_full_task_dag,
    ready_task_ids,
    validate_full_task_dag,
)
from cc_hhgt.v32.orchestration import N_PATIENT_FOLDS, OrchestrationContractError


def _task_by_component_and_fold(tasks: list[dict[str, object]]) -> dict[tuple[str, int], dict[str, object]]:
    return {
        (str(task["component"]), int(task["patient_fold"])): task
        for task in tasks
    }


def test_builds_complete_unique_core_first_dag() -> None:
    tasks = build_full_task_dag(run_id="v32-test", seed=17)

    assert len(tasks) == FULL_TASK_COUNT == len(ALL_COMPONENTS) * N_PATIENT_FOLDS
    assert len({task["task_id"] for task in tasks}) == FULL_TASK_COUNT
    assert [task["component"] for task in tasks[:N_PATIENT_FOLDS]] == [
        CORE_COMPONENT
    ] * N_PATIENT_FOLDS
    assert {task["component"] for task in tasks[N_PATIENT_FOLDS:]} == set(
        AUXILIARY_COMPONENTS
    )
    validate_full_task_dag(tasks)


def test_all_tasks_are_fresh_and_ban_historical_artifacts() -> None:
    tasks = build_full_task_dag(run_id="v32-test", seed=17)

    for task in tasks:
        assert task["fresh_init"] is True
        assert task["historical_checkpoint_allowed"] is False
        assert task["historical_result_allowed"] is False
        assert task["historical_artifact_policy"] == "FORBIDDEN"


def test_each_auxiliary_task_uses_frozen_same_fold_current_core() -> None:
    tasks = build_full_task_dag(run_id="v32-test", seed=17)
    indexed = _task_by_component_and_fold(tasks)

    for fold in range(N_PATIENT_FOLDS):
        core = indexed[(CORE_COMPONENT, fold)]
        assert core["depends_on"] == ()
        assert core["core_checkpoint_task_id"] is None
        assert core["core_frozen"] is False
        assert core["core_parameters_trainable"] is True
        assert core["core_gradient_policy"] == "TRAIN_FROM_SCRATCH"

        for component in AUXILIARY_COMPONENTS:
            aux = indexed[(component, fold)]
            assert aux["depends_on"] == (core["task_id"],)
            assert aux["core_checkpoint_task_id"] == core["task_id"]
            assert aux["checkpoint_input_policy"] == "CURRENT_RUN_MATCHING_FOLD_CORE_ONLY"
            assert aux["core_frozen"] is True
            assert aux["core_parameters_trainable"] is False
            assert aux["core_gradient_policy"] == "DETACH_NO_GRAD"


def test_ready_scheduler_enforces_global_core_phase_barrier() -> None:
    tasks = build_full_task_dag(run_id="v32-test", seed=17)
    core_ids = [str(task["task_id"]) for task in tasks if task["component"] == CORE_COMPONENT]

    assert ready_task_ids(tasks, completed_task_ids=[]) == core_ids
    assert ready_task_ids(tasks, completed_task_ids=[core_ids[0]]) == core_ids[1:]

    ready_after_core = ready_task_ids(tasks, completed_task_ids=core_ids)
    assert len(ready_after_core) == len(AUXILIARY_COMPONENTS) * N_PATIENT_FOLDS
    assert all(task_id not in core_ids for task_id in ready_after_core)


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda tasks: tasks.pop(), "Expected"),
        (lambda tasks: tasks.append(copy.deepcopy(tasks[0])), "unique"),
        (lambda tasks: tasks.__setitem__(0, copy.deepcopy(tasks[5])), "unique"),
    ],
)
def test_rejects_incomplete_duplicate_or_replaced_tasks(mutate, match: str) -> None:
    tasks = build_full_task_dag(run_id="v32-test", seed=17)
    mutate(tasks)

    with pytest.raises(OrchestrationContractError, match=match):
        validate_full_task_dag(tasks)


def test_rejects_cross_fold_dependency() -> None:
    tasks = build_full_task_dag(run_id="v32-test", seed=17)
    indexed = _task_by_component_and_fold(tasks)
    indexed[("state", 0)]["depends_on"] = (indexed[(CORE_COMPONENT, 1)]["task_id"],)

    with pytest.raises(OrchestrationContractError, match="matching-fold|depend"):
        validate_full_task_dag(tasks)


def test_rejects_auxiliary_task_before_core_phase() -> None:
    tasks = build_full_task_dag(run_id="v32-test", seed=17)
    tasks[0], tasks[5] = tasks[5], tasks[0]

    with pytest.raises(OrchestrationContractError, match="precede|phase"):
        validate_full_task_dag(tasks)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("fresh_init", False, "fresh_init"),
        ("historical_checkpoint_allowed", True, "historical checkpoints"),
        ("historical_result_allowed", True, "historical model results"),
        ("historical_artifact_policy", "ALLOW", "historical learned artifacts"),
        ("core_frozen", False, "core_frozen"),
        ("core_parameters_trainable", True, "core_parameters_trainable"),
        ("core_gradient_policy", "TRAIN", "core_gradient_policy"),
    ],
)
def test_rejects_auxiliary_policy_drift(
    field: str, value: object, match: str
) -> None:
    tasks = build_full_task_dag(run_id="v32-test", seed=17)
    auxiliary = next(task for task in tasks if task["component"] == "state")
    auxiliary[field] = value

    with pytest.raises(OrchestrationContractError, match=match):
        validate_full_task_dag(tasks)


@pytest.mark.parametrize(
    "field,value",
    [
        ("core_frozen", True),
        ("core_parameters_trainable", False),
        ("core_gradient_policy", "DETACH_NO_GRAD"),
    ],
)
def test_rejects_core_policy_drift(field: str, value: object) -> None:
    tasks = build_full_task_dag(run_id="v32-test", seed=17)
    core = next(task for task in tasks if task["component"] == CORE_COMPONENT)
    core[field] = value

    with pytest.raises(OrchestrationContractError, match=field):
        validate_full_task_dag(tasks)


def test_rejects_unknown_completed_task() -> None:
    tasks = build_full_task_dag(run_id="v32-test", seed=17)

    with pytest.raises(OrchestrationContractError, match="not present"):
        ready_task_ids(tasks, completed_task_ids=["not-in-this-dag"])
