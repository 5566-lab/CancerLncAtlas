"""Declarative orchestration contract for the complete V3.2 model bundle.

The historical :mod:`cc_hhgt.v32.orchestration` module intentionally remains
the five-task exact-pathway pilot contract.  This module describes the larger
fresh-training DAG without changing that legacy behaviour and never starts a
process, imports a training framework, or loads a checkpoint.

All five exact-pathway core folds form phase zero.  Auxiliary heads form phase
one, use only the matching fold's core checkpoint from the *same* V3.2 run,
and must keep that core frozen.  Historical checkpoints and historical model
results are forbidden for every task; reusable raw/static data are outside
the scope of this task graph.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .orchestration import (
    DEFAULT_SEED,
    N_PATIENT_FOLDS,
    OrchestrationContractError,
)


CORE_COMPONENT = "exact_pathway"
AUXILIARY_COMPONENTS = (
    "state",
    "clinical",
    "single_cell",
    "mutation_cnv",
    "drug",
    "interaction",
    "evidence",
)
ALL_COMPONENTS = (CORE_COMPONENT, *AUXILIARY_COMPONENTS)
CORE_PHASE = 0
AUXILIARY_PHASE = 1
FULL_TASK_COUNT = N_PATIENT_FOLDS * len(ALL_COMPONENTS)

TASK_FIELDS = frozenset(
    {
        "task_id",
        "run_id",
        "component",
        "task_type",
        "patient_fold",
        "seed",
        "phase",
        "depends_on",
        "fresh_init",
        "fresh_init_scope",
        "historical_checkpoint_allowed",
        "historical_result_allowed",
        "historical_artifact_policy",
        "checkpoint_input_policy",
        "core_checkpoint_task_id",
        "core_frozen",
        "core_parameters_trainable",
        "core_gradient_policy",
    }
)


def _validate_run_seed(run_id: str, seed: int) -> tuple[str, int]:
    normalized_run = str(run_id).strip()
    if not normalized_run:
        raise OrchestrationContractError("run_id cannot be empty")
    if "|" in normalized_run:
        raise OrchestrationContractError("run_id cannot contain the task-id delimiter '|'")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise OrchestrationContractError("seed must be a non-negative integer")
    return normalized_run, seed


def _task_id(run_id: str, component: str, patient_fold: int, seed: int) -> str:
    return f"{run_id}|{component}|PATIENT_FOLD_{patient_fold}|SEED_{seed}"


def _core_task_id(run_id: str, patient_fold: int, seed: int) -> str:
    return _task_id(run_id, CORE_COMPONENT, patient_fold, seed)


def _core_task(run_id: str, patient_fold: int, seed: int) -> dict[str, Any]:
    task_id = _core_task_id(run_id, patient_fold, seed)
    return {
        "task_id": task_id,
        "run_id": run_id,
        "component": CORE_COMPONENT,
        "task_type": "TRAIN_EXACT_PATHWAY_CORE",
        "patient_fold": patient_fold,
        "seed": seed,
        "phase": CORE_PHASE,
        "depends_on": (),
        "fresh_init": True,
        "fresh_init_scope": "FULL_COMPONENT",
        "historical_checkpoint_allowed": False,
        "historical_result_allowed": False,
        "historical_artifact_policy": "FORBIDDEN",
        "checkpoint_input_policy": "NONE",
        "core_checkpoint_task_id": None,
        "core_frozen": False,
        "core_parameters_trainable": True,
        "core_gradient_policy": "TRAIN_FROM_SCRATCH",
    }


def _auxiliary_task(
    run_id: str, component: str, patient_fold: int, seed: int
) -> dict[str, Any]:
    core_task_id = _core_task_id(run_id, patient_fold, seed)
    return {
        "task_id": _task_id(run_id, component, patient_fold, seed),
        "run_id": run_id,
        "component": component,
        "task_type": f"TRAIN_{component.upper()}_HEAD",
        "patient_fold": patient_fold,
        "seed": seed,
        "phase": AUXILIARY_PHASE,
        "depends_on": (core_task_id,),
        "fresh_init": True,
        "fresh_init_scope": "AUXILIARY_PRIVATE_PARAMETERS",
        "historical_checkpoint_allowed": False,
        "historical_result_allowed": False,
        "historical_artifact_policy": "FORBIDDEN",
        "checkpoint_input_policy": "CURRENT_RUN_MATCHING_FOLD_CORE_ONLY",
        "core_checkpoint_task_id": core_task_id,
        "core_frozen": True,
        "core_parameters_trainable": False,
        "core_gradient_policy": "DETACH_NO_GRAD",
    }


def build_full_task_dag(
    *, run_id: str = "V3_2_FULL_INTEGRATED", seed: int = DEFAULT_SEED
) -> list[dict[str, Any]]:
    """Build the deterministic forty-task complete V3.2 training DAG.

    The returned manifest is topologically ordered with all five core folds
    first.  Every auxiliary task depends on the current run's matching-fold
    core task and declares a newly initialized private head plus frozen core.
    """

    normalized_run, normalized_seed = _validate_run_seed(run_id, seed)
    tasks = [
        _core_task(normalized_run, fold, normalized_seed)
        for fold in range(N_PATIENT_FOLDS)
    ]
    tasks.extend(
        _auxiliary_task(normalized_run, component, fold, normalized_seed)
        for component in AUXILIARY_COMPONENTS
        for fold in range(N_PATIENT_FOLDS)
    )
    validate_full_task_dag(tasks)
    return tasks


def _strict_int(task: Mapping[str, Any], field: str) -> int:
    value = task.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise OrchestrationContractError(f"{field} must be an integer")
    return value


def _strict_bool(task: Mapping[str, Any], field: str) -> bool:
    value = task.get(field)
    if not isinstance(value, bool):
        raise OrchestrationContractError(f"{field} must be a boolean")
    return value


def _dependencies(task: Mapping[str, Any]) -> tuple[str, ...]:
    value = task.get("depends_on")
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise OrchestrationContractError("depends_on must be a sequence of task IDs")
    dependencies = tuple(str(item) for item in value)
    if any(not item for item in dependencies) or len(dependencies) != len(set(dependencies)):
        raise OrchestrationContractError(
            "depends_on task IDs must be unique and non-empty"
        )
    return dependencies


def validate_full_task_dag(tasks: Sequence[Mapping[str, Any]]) -> None:
    """Fail closed on incomplete, reordered, cross-fold, or legacy-fed DAGs."""

    if isinstance(tasks, (str, bytes)) or not isinstance(tasks, Sequence):
        raise OrchestrationContractError("tasks must be a sequence")
    for index, task in enumerate(tasks):
        if not isinstance(task, Mapping):
            raise OrchestrationContractError(f"Task {index} must be a mapping")
        missing = sorted(TASK_FIELDS - set(task))
        if missing:
            raise OrchestrationContractError(f"Task {index} lacks fields: {missing}")

    ids = [str(task["task_id"]) for task in tasks]
    if any(not task_id for task_id in ids) or len(ids) != len(set(ids)):
        raise OrchestrationContractError("Task IDs must be unique and non-empty")
    if len(tasks) != FULL_TASK_COUNT:
        raise OrchestrationContractError(
            f"Expected {FULL_TASK_COUNT} full V3.2 tasks, observed {len(tasks)}"
        )
    id_to_index = {task_id: index for index, task_id in enumerate(ids)}

    run_ids = {str(task["run_id"]).strip() for task in tasks}
    seeds = {_strict_int(task, "seed") for task in tasks}
    if len(run_ids) != 1 or "" in run_ids:
        raise OrchestrationContractError("All tasks must belong to one non-empty run_id")
    if len(seeds) != 1:
        raise OrchestrationContractError("All tasks must use one seed")
    run_id, seed = _validate_run_seed(next(iter(run_ids)), next(iter(seeds)))

    observed_pairs: list[tuple[str, int]] = []
    for task in tasks:
        component = str(task["component"])
        if component not in ALL_COMPONENTS:
            raise OrchestrationContractError(f"Unknown V3.2 component: {component!r}")
        fold = _strict_int(task, "patient_fold")
        if fold not in range(N_PATIENT_FOLDS):
            raise OrchestrationContractError(f"patient_fold must be 0..4, got {fold}")
        observed_pairs.append((component, fold))
    expected_pairs = {
        (component, fold)
        for component in ALL_COMPONENTS
        for fold in range(N_PATIENT_FOLDS)
    }
    if set(observed_pairs) != expected_pairs or len(observed_pairs) != len(
        expected_pairs
    ):
        raise OrchestrationContractError(
            "Full V3.2 DAG must contain every component/fold pair exactly once"
        )

    core_indices = [
        index for index, task in enumerate(tasks) if task["component"] == CORE_COMPONENT
    ]
    auxiliary_indices = [
        index for index, task in enumerate(tasks) if task["component"] != CORE_COMPONENT
    ]
    if max(core_indices) >= min(auxiliary_indices):
        raise OrchestrationContractError(
            "All exact-pathway core folds must precede every auxiliary task"
        )

    for index, task in enumerate(tasks):
        component = str(task["component"])
        fold = _strict_int(task, "patient_fold")
        task_id = str(task["task_id"])
        expected_task_id = _task_id(run_id, component, fold, seed)
        if task_id != expected_task_id:
            raise OrchestrationContractError(
                f"Non-canonical task_id for {component}/fold {fold}: {task_id!r}"
            )
        dependencies = _dependencies(task)
        for dependency in dependencies:
            if dependency not in id_to_index:
                raise OrchestrationContractError(
                    f"Task {task_id} has unknown dependency {dependency!r}"
                )
            if id_to_index[dependency] >= index:
                raise OrchestrationContractError(
                    f"Task {task_id} appears before dependency {dependency!r}"
                )

        if _strict_bool(task, "fresh_init") is not True:
            raise OrchestrationContractError(f"Task {task_id} must use fresh_init")
        if _strict_bool(task, "historical_checkpoint_allowed") is not False:
            raise OrchestrationContractError(
                f"Task {task_id} cannot allow historical checkpoints"
            )
        if _strict_bool(task, "historical_result_allowed") is not False:
            raise OrchestrationContractError(
                f"Task {task_id} cannot allow historical model results"
            )
        if task["historical_artifact_policy"] != "FORBIDDEN":
            raise OrchestrationContractError(
                f"Task {task_id} must forbid historical learned artifacts"
            )

        expected_core_id = _core_task_id(run_id, fold, seed)
        if component == CORE_COMPONENT:
            expected_values = {
                "task_type": "TRAIN_EXACT_PATHWAY_CORE",
                "phase": CORE_PHASE,
                "fresh_init_scope": "FULL_COMPONENT",
                "checkpoint_input_policy": "NONE",
                "core_checkpoint_task_id": None,
                "core_frozen": False,
                "core_parameters_trainable": True,
                "core_gradient_policy": "TRAIN_FROM_SCRATCH",
            }
            if dependencies:
                raise OrchestrationContractError("Core tasks cannot have dependencies")
        else:
            expected_values = {
                "task_type": f"TRAIN_{component.upper()}_HEAD",
                "phase": AUXILIARY_PHASE,
                "fresh_init_scope": "AUXILIARY_PRIVATE_PARAMETERS",
                "checkpoint_input_policy": "CURRENT_RUN_MATCHING_FOLD_CORE_ONLY",
                "core_checkpoint_task_id": expected_core_id,
                "core_frozen": True,
                "core_parameters_trainable": False,
                "core_gradient_policy": "DETACH_NO_GRAD",
            }
            if dependencies != (expected_core_id,):
                raise OrchestrationContractError(
                    f"Auxiliary task {task_id} must depend only on matching-fold core"
                )
        for field, expected in expected_values.items():
            observed = task[field]
            if observed != expected or type(observed) is not type(expected):
                raise OrchestrationContractError(
                    f"Task {task_id} has invalid {field}: {observed!r}; "
                    f"expected {expected!r}"
                )


def ready_task_ids(
    tasks: Sequence[Mapping[str, Any]],
    *,
    completed_task_ids: Sequence[str] = (),
) -> list[str]:
    """Return tasks eligible to start under the mandatory two-phase barrier.

    Even when one fold's core is complete, no auxiliary task becomes ready
    until all five core folds are complete.  This makes "core first" an
    executable scheduler rule in addition to a manifest ordering rule.
    """

    validate_full_task_dag(tasks)
    known = {str(task["task_id"]) for task in tasks}
    completed = {str(task_id) for task_id in completed_task_ids}
    unknown = sorted(completed - known)
    if unknown:
        raise OrchestrationContractError(
            f"Completed task IDs are not present in the DAG: {unknown}"
        )
    core_ids = {
        str(task["task_id"])
        for task in tasks
        if task["component"] == CORE_COMPONENT
    }
    active_phase = CORE_PHASE if not core_ids.issubset(completed) else AUXILIARY_PHASE
    return [
        str(task["task_id"])
        for task in tasks
        if task["task_id"] not in completed
        and task["phase"] == active_phase
        and set(_dependencies(task)).issubset(completed)
    ]
