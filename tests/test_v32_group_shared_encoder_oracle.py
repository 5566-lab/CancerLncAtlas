from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

from cc_hhgt.v32.group_shared_encoder_oracle import (
    ARTIFACT_CLASS,
    COMPONENT_NAMES,
    EXPECTED_RUNTIME_CHUNKS,
    PASS_STATUS,
    RECEIPT_FORMAT,
    SCIENTIFIC_FAIL_STATUS,
    ScheduleEvaluation,
    _base_receipt,
    _global_group_loss_components,
    _gradient_sha256,
    _group_zero_assignments,
    _paired_bootstrap_ci,
    _require_oracle_task,
    compare_schedule_evaluations,
    run_authorized_oracle_comparison,
)


def _gradients(*, gate_zero: bool = False) -> dict[str, np.ndarray]:
    return {
        "encoder.layer.weight": np.asarray([1.0, -0.25], dtype=np.float64),
        "residual_map.0.weight": np.asarray([0.5, 0.25], dtype=np.float64),
        "residual_output.weight": np.asarray([0.125], dtype=np.float64),
        "gate.0.weight": np.asarray(
            [0.0 if gate_zero else 0.75], dtype=np.float64
        ),
        "direction_output.0.weight": np.asarray([0.33], dtype=np.float64),
    }


def _evaluation(
    *,
    components: dict[str, tuple[float, ...]] | None = None,
    branches: tuple[bool, ...] | None = None,
    gradients: dict[str, np.ndarray] | None = None,
) -> ScheduleEvaluation:
    values = components or {
        name: tuple(1.0 + index * 1e-4 for index in range(EXPECTED_RUNTIME_CHUNKS))
        for name in COMPONENT_NAMES
    }
    mean_gradients = gradients or _gradients()
    return ScheduleEvaluation(
        component_values=values,
        branches=branches or (True,) * EXPECTED_RUNTIME_CHUNKS,
        mean_gradients=mean_gradients,
        mean_gradient_sha256=_gradient_sha256(mean_gradients),
        assignment_count=EXPECTED_RUNTIME_CHUNKS,
        encoder_forward_calls=EXPECTED_RUNTIME_CHUNKS,
        decoder_forward_calls=4 * EXPECTED_RUNTIME_CHUNKS,
        global_loss_calls=EXPECTED_RUNTIME_CHUNKS,
        backward_calls=EXPECTED_RUNTIME_CHUNKS,
    )


def test_identical_schedules_pass_every_frozen_identity_gate() -> None:
    reference = _evaluation()
    proposed = _evaluation()
    result = compare_schedule_evaluations(
        reference,
        proposed,
        theta_name="theta_0",
        bootstrap_replicates=500,
    )
    assert result["pass"] is True
    assert result["scientific_fail_reasons"] == []
    assert result["branch_gate"] == {
        "evaluations": 58,
        "active_count": 58,
        "inactive_count": 0,
        "constant_across_all_58": True,
        "paired_disagreement_count": 0,
        "pass": True,
    }
    assert result["complete_mean_gradient"]["relative_l2"] == 0.0
    assert result["complete_mean_gradient"]["cosine"] == pytest.approx(1.0)
    assert all(item["pass"] for item in result["components"].values())
    assert all(item["pass"] for item in result["module_mean_gradients"].values())


def test_any_branch_nonconstancy_across_58_is_scientific_fail() -> None:
    reference = _evaluation()
    proposed = _evaluation(branches=(True,) * 28 + (False,))
    result = compare_schedule_evaluations(
        reference,
        proposed,
        theta_name="theta_1",
        bootstrap_replicates=500,
    )
    assert result["pass"] is False
    assert "NNPU_BRANCH_NONCONSTANT_ACROSS_58_EVALUATIONS" in result[
        "scientific_fail_reasons"
    ]
    assert "NNPU_PAIRED_BRANCH_DISAGREEMENT" in result[
        "scientific_fail_reasons"
    ]
    assert result["branch_gate"]["active_count"] == 57
    assert result["branch_gate"]["constant_across_all_58"] is False


def test_scalar_identity_uses_one_ppm_with_absolute_floor() -> None:
    reference = _evaluation()
    changed = {
        name: tuple(reference.component_values[name]) for name in COMPONENT_NAMES
    }
    changed["total_objective"] = tuple(
        value + 2e-6 for value in changed["total_objective"]
    )
    proposed = _evaluation(components=changed)
    result = compare_schedule_evaluations(
        reference,
        proposed,
        theta_name="theta_0",
        bootstrap_replicates=500,
    )
    total = result["components"]["total_objective"]
    assert total["absolute_mean_difference"] > total["identity_tolerance"]
    assert total["identity_pass"] is False
    assert result["pass"] is False


def test_complete_gradient_relative_l2_gate_is_conjunctive() -> None:
    reference = _evaluation()
    changed = {name: value.copy() for name, value in reference.mean_gradients.items()}
    changed["encoder.layer.weight"] = changed["encoder.layer.weight"] + np.asarray(
        [0.0, 0.02]
    )
    proposed = _evaluation(gradients=changed)
    result = compare_schedule_evaluations(
        reference,
        proposed,
        theta_name="theta_0",
        bootstrap_replicates=500,
    )
    assert result["complete_mean_gradient"]["relative_l2"] > 1e-3
    assert result["complete_mean_gradient"]["pass"] is False
    assert "COMPLETE_MEAN_GRADIENT_GATE_FAILED" in result[
        "scientific_fail_reasons"
    ]


def test_module_both_zero_is_typed_na_but_one_sided_zero_fails() -> None:
    both_zero = _evaluation(gradients=_gradients(gate_zero=True))
    passed = compare_schedule_evaluations(
        both_zero,
        both_zero,
        theta_name="theta_0",
        bootstrap_replicates=500,
    )
    gate = passed["module_mean_gradients"]["gate"]
    assert gate["status"] == "TYPED_NA_BOTH_ZERO"
    assert gate["pass"] is True

    one_sided = _evaluation(gradients=_gradients(gate_zero=False))
    failed = compare_schedule_evaluations(
        both_zero,
        one_sided,
        theta_name="theta_0",
        bootstrap_replicates=500,
    )
    gate = failed["module_mean_gradients"]["gate"]
    assert gate["status"] == "ZERO_NONZERO_MISMATCH"
    assert gate["pass"] is False
    assert failed["pass"] is False


def test_group_zero_assignments_cover_same_b4_x_k29_pairs() -> None:
    permutation = tuple(range(EXPECTED_RUNTIME_CHUNKS - 1, -1, -1))
    legacy, proposed = _group_zero_assignments(permutation)
    assert len(legacy) == len(proposed) == EXPECTED_RUNTIME_CHUNKS
    assert all(len(assignment) == 4 for assignment in legacy)
    assert all(len({chunk for _, chunk in assignment}) == 4 for assignment in legacy)
    assert all(len({chunk for _, chunk in assignment}) == 1 for assignment in proposed)
    assert {pair for assignment in legacy for pair in assignment} == {
        pair for assignment in proposed for pair in assignment
    }


def test_bootstrap_is_paired_and_deterministic() -> None:
    reference = tuple(float(index) for index in range(29))
    proposed = tuple(value + 0.1 for value in reference)
    first = _paired_bootstrap_ci(reference, proposed, seed=7, replicates=500)
    second = _paired_bootstrap_ci(reference, proposed, seed=7, replicates=500)
    assert first == second
    assert first[0] == pytest.approx(0.1)
    assert first[1] == pytest.approx(0.1)


def test_oracle_task_is_strictly_g2_fold0_seed20260726_pending_paid() -> None:
    task = {
        "task_type": "CC_HHGT_PATIENT_FOLD",
        "model": "CC-HHGT",
        "status": "PENDING",
        "blocked_reason": "",
        "paid_task": "true",
        "patient_fold": "0",
        "seed": "20260726",
    }
    _require_oracle_task(task)
    for key, value in (("patient_fold", "1"), ("seed", "7"), ("status", "RUNNING")):
        changed = dict(task)
        changed[key] = value
        with pytest.raises(RuntimeError, match="TASK_CONTRACT_DRIFT"):
            _require_oracle_task(changed)


def test_terminal_receipts_are_comparison_only_and_never_formal() -> None:
    for status in (PASS_STATUS, SCIENTIFIC_FAIL_STATUS):
        receipt = _base_receipt(status)
        assert receipt["format"] == RECEIPT_FORMAT
        assert receipt["artifact_class"] == ARTIFACT_CLASS
        assert receipt["formal_training_authorized"] is False
        assert receipt["formal_artifacts_written"] == 0
        assert receipt["checkpoint_written"] is False
        assert receipt["success_json_written"] is False
        assert receipt["failure_json_written"] is False
        assert receipt["prediction_written"] is False
        assert receipt["winner_selection_input"] is False


def test_public_runner_revalidates_before_torch_and_has_no_file_writer() -> None:
    source = inspect.getsource(run_authorized_oracle_comparison)
    assert "TrainingAuthorizationContext" in source
    assert "_reverify_authorization_context(context)" in source
    assert source.index("context.authorized_trainer") < source.index(
        "from .gpu_backward_probe"
    )
    assert "torch.save" not in source
    assert "write_text" not in source
    assert "output_root" not in source
    assert "SUCCESS.json" not in source
    assert "FAILURE.json" not in source


@pytest.mark.parametrize(
    "authorized_trainer",
    [
        None,
        "cc_hhgt.v32.training:train_one_shard",
    ],
)
def test_public_runner_rejects_none_or_swapped_trainer_before_torch_import(
    authorized_trainer: str | None,
) -> None:
    from cc_hhgt.v32.training_guard import (
        TrainingAuthorizationContext,
        TrainingAuthorizationError,
    )

    context = TrainingAuthorizationContext(
        run_id="irrelevant",
        task_id="irrelevant",
        endpoint_id="paid_gpu",
        hardware_class="PAID_PREEMPTIBLE_GPU",
        repo_root="irrelevant",
        config_path="irrelevant",
        input_manifest_path="irrelevant",
        task_manifest_path="irrelevant",
        approval_path="irrelevant",
        artifact_hashes={},
        policy={},
        authorized_trainer=authorized_trainer,
    )
    before = sys.modules.get("torch")
    with pytest.raises(TrainingAuthorizationError, match="AUTHORIZED_TRAINER_DRIFT"):
        run_authorized_oracle_comparison(context)
    assert sys.modules.get("torch") is before


def test_preregistration_freezes_strict_branch_and_gradient_gates() -> None:
    text = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "v32_group_shared_encoder_estimator_preregistration_20260901.md"
    ).read_text(encoding="utf-8")
    assert "constant across all 58 evaluations" in text
    assert "max(1e-7, 1e-6 * abs(reference_mean))" in text
    assert "relative L2 error of the complete mean gradient is at most `1e-3`" in text
    assert "cosine similarity is at least `0.9999`" in text
    assert "within `[0.999, 1.001]`" in text
    assert "similarity is at least `0.999`" in text
    assert "formal_artifacts_written=0" in text


def test_component_decomposition_matches_production_global_loss() -> None:
    torch = pytest.importorskip("torch")
    from cc_hhgt.v32 import training

    outputs = []
    batches = []
    for index in range(4):
        rows = 7 + index
        logits = torch.linspace(-1.0, 1.0, rows) + 0.03 * index
        outputs.append(
            {
                "final_logit": logits.clone().requires_grad_(True),
                "direction_logit": (0.5 * logits).clone().requires_grad_(True),
                "raw_graph_residual": (0.1 * logits).clone().requires_grad_(True),
            }
        )
        batches.append(
            {
                "proxy_label": (torch.arange(rows) % 3 == 0).float(),
                "weak_positive": torch.arange(rows) % 2 == 0,
                "direction_label": (torch.arange(rows) % 2).float(),
                "direction_available": torch.ones(rows, dtype=torch.bool),
            }
        )
    observed = _global_group_loss_components(
        outputs,
        batches,
        torch=torch,
        direction_loss_weight=0.25,
        shrinkage=1e-4,
    )
    reference = training._global_loss_from_outputs(
        outputs,
        batches,
        torch=torch,
        direction_loss_weight=0.25,
        shrinkage=1e-4,
        reduction_dtype=torch.float32,
    )
    assert torch.allclose(
        observed.components["total_objective"], reference, atol=1e-7, rtol=1e-7
    )
