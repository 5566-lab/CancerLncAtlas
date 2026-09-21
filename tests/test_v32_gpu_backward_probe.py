from __future__ import annotations

import inspect
import builtins

import pytest

from cc_hhgt.v32 import training
from cc_hhgt.v32 import gpu_backward_probe as probe_module
from cc_hhgt.v32.gpu_backward_probe import (
    _PEAK_RESERVED_LIMIT_BYTES,
    _enforce_peak_reserved_limit,
    _require_authorized_pending_task,
    _resolve_gpu_identity,
    _select_exact_probe_group,
    _validate_graph_variant_binding,
    resolve_probe_profile,
    run_authorized_probe,
)
from cc_hhgt.v32.training_guard import TrainingAuthorizationContext


def _direct_context(trainer: str | None) -> TrainingAuthorizationContext:
    return TrainingAuthorizationContext(
        run_id="r",
        task_id="t",
        endpoint_id="paid_gpu",
        hardware_class="PAID_PREEMPTIBLE_GPU",
        repo_root="repo",
        config_path="config",
        input_manifest_path="input",
        task_manifest_path="tasks",
        approval_path="approval",
        artifact_hashes={},
        policy={},
        authorized_trainer=trainer,
    )

def _batch(rows: int):
    torch = pytest.importorskip("torch")
    return {
        "candidate_batch": {
            "l": torch.zeros(rows, dtype=torch.long),
            "p": torch.zeros(rows, dtype=torch.long),
            "c": torch.zeros(rows, dtype=torch.long),
        },
        "base_logit": torch.zeros(rows),
        "conservation_context": torch.zeros((rows, 4)),
        "graph_available": torch.ones(rows, dtype=torch.bool),
        "proxy_label": torch.zeros(rows),
        "weak_positive": torch.zeros(rows, dtype=torch.bool),
        "direction_label": torch.zeros(rows),
        "direction_available": torch.ones(rows, dtype=torch.bool),
    }


def test_probe_profiles_are_exact_and_fail_closed() -> None:
    environment = resolve_probe_profile({})
    assert (environment.microbatch, environment.accumulation) == (4096, 1)
    assert environment.group_row_capacity == 4096
    stress = resolve_probe_profile(
        {"V32_PROBE_MICROBATCH": "8192", "V32_PROBE_ACCUMULATION": "4"}
    )
    assert stress.name == "PRIMARY_GROUP_STRESS_GATE_8192_X_4"
    assert stress.group_row_capacity == 32768
    with pytest.raises(RuntimeError, match="allowed profiles"):
        resolve_probe_profile(
            {"V32_PROBE_MICROBATCH": "4096", "V32_PROBE_ACCUMULATION": "4"}
        )
    with pytest.raises(RuntimeError, match="must be integers"):
        resolve_probe_profile(
            {"V32_PROBE_MICROBATCH": "large", "V32_PROBE_ACCUMULATION": "1"}
        )


def test_probe_requires_one_pending_paid_formal_task() -> None:
    task = {
        "task_type": "CC_HHGT_PATIENT_FOLD",
        "model": "CC-HHGT",
        "status": "PENDING",
        "blocked_reason": "",
        "paid_task": "true",
    }
    _require_authorized_pending_task(task)
    task["status"] = "BLOCKED"
    with pytest.raises(RuntimeError, match="task contract drift"):
        _require_authorized_pending_task(task)


def test_probe_binds_graph_variant_across_run_config_path_and_payload(tmp_path) -> None:
    prepared = tmp_path / "G2" / "PATIENT_FOLD_0.pt"
    prepared.parent.mkdir()
    assert (
        _validate_graph_variant_binding(
            run_id="v32-g012-g2-paid-gpu-20260831-r1",
            config={"task_contract": {"graph_variant": "G2"}},
            payload={"formal_graph_variant": "G2"},
            prepared_path=prepared,
        )
        == "G2"
    )
    with pytest.raises(RuntimeError, match="VARIANT_BINDING_DRIFT"):
        _validate_graph_variant_binding(
            run_id="v32-g012-g2-paid-gpu-20260831-r1",
            config={"task_contract": {"graph_variant": "G2"}},
            payload={"formal_graph_variant": "G1"},
            prepared_path=prepared,
        )
    with pytest.raises(RuntimeError, match="VARIANT_BINDING_DRIFT"):
        _validate_graph_variant_binding(
            run_id="v32-g012-g0-paid-gpu-20260831-r1",
            config={"task_contract": {"graph_variant": "G2"}},
            payload={"formal_graph_variant": "G2"},
            prepared_path=prepared,
        )


def test_probe_peak_reserved_gate_keeps_fixed_two_gib_margin() -> None:
    observed = _enforce_peak_reserved_limit(_PEAK_RESERVED_LIMIT_BYTES - 1)
    assert observed == {
        "peak_reserved_within_limit": True,
        "peak_reserved_limit_bytes": 22 * 1024**3,
        "peak_reserved_headroom_bytes": 1,
    }
    with pytest.raises(RuntimeError, match="LIMIT_EXCEEDED"):
        _enforce_peak_reserved_limit(_PEAK_RESERVED_LIMIT_BYTES)
    with pytest.raises(RuntimeError, match="NONPOSITIVE"):
        _enforce_peak_reserved_limit(0)
    with pytest.raises(RuntimeError, match="NONPOSITIVE"):
        _enforce_peak_reserved_limit(-1)


def test_probe_gpu_identity_is_fixed_to_paid_rtx4090() -> None:
    class _Properties:
        name = "NVIDIA GeForce RTX 4090"
        total_memory = 24 * 1024**3

    class _Cuda:
        @staticmethod
        def current_device():
            return 0

        @staticmethod
        def get_device_properties(index):
            assert index == 0
            return _Properties()

    class _Torch:
        cuda = _Cuda()

    assert _resolve_gpu_identity(_Torch) == {
        "cuda_device_index": 0,
        "cuda_device_name": "NVIDIA GeForce RTX 4090",
        "cuda_total_memory_bytes": 24 * 1024**3,
    }
    _Properties.name = "NVIDIA GeForce RTX 3090"
    with pytest.raises(RuntimeError, match="GPU_IDENTITY_MISMATCH"):
        _resolve_gpu_identity(_Torch)
    _Properties.name = "NVIDIA GeForce RTX 4090"
    _Properties.total_memory = _PEAK_RESERVED_LIMIT_BYTES
    with pytest.raises(RuntimeError, match="MEMORY_CAPACITY_TOO_SMALL"):
        _resolve_gpu_identity(_Torch)


@pytest.mark.parametrize(
    ("environment", "expected_rows"),
    [
        ({}, [4096]),
        (
            {"V32_PROBE_MICROBATCH": "8192", "V32_PROBE_ACCUMULATION": "4"},
            [8192, 8192, 8192, 8192],
        ),
    ],
)
def test_probe_selects_one_exact_full_capacity_group(environment, expected_rows) -> None:
    profile = resolve_probe_profile(environment)
    batches = [_batch(8192) for _ in range(5)]
    pairs = tuple((index, index % 2) for index in range(len(batches)))
    group, rows = _select_exact_probe_group(
        training,
        train_batches=batches,
        pass_pairs=pairs,
        profile=profile,
    )
    assert len(group) == profile.accumulation
    assert rows == expected_rows
    assert sum(rows) == profile.group_row_capacity


def test_probe_refuses_partial_pressure_group() -> None:
    profile = resolve_probe_profile(
        {"V32_PROBE_MICROBATCH": "8192", "V32_PROBE_ACCUMULATION": "4"}
    )
    batches = [_batch(8192), _batch(8192), _batch(8192), _batch(7000)]
    pairs = tuple((index, 0) for index in range(len(batches)))
    with pytest.raises(RuntimeError, match="partial optimizer group"):
        _select_exact_probe_group(
            training,
            train_batches=batches,
            pass_pairs=pairs,
            profile=profile,
        )


def test_probe_delegates_production_backward_and_writes_no_formal_artifact() -> None:
    source = inspect.getsource(run_authorized_probe)
    assert "TrainingAuthorizationContext" in source
    assert "_reverify_authorization_context(context)" in source
    assert source.index("TrainingAuthorizationContext") < source.index("import torch")
    assert "load_prepared_artifact_from_authorized_handle(" in source
    assert "torch.load(prepared_path" not in source
    assert "training.streaming_group_backward(" in source
    assert "training.shared_encoder_conventional_group_backward(" in source
    assert "training.optimizer_step_with_guards(" in source
    assert "REAL_BACKWARD_PROBE_CALL_CONTRACT_DRIFT" in source
    assert '"global_loss_calls"' in source
    assert 'bool(update["grad_finite"])' in source
    assert 'bool(update["parameters_finite"])' in source
    assert 'bool(update["parameter_delta_positive"])' in source
    assert "optimizer_steps\": 1" in source
    assert "SUCCESS.json" not in source
    assert "checkpoint" not in source.lower()
    assert "torch.save" not in source


@pytest.mark.parametrize(
    "trainer",
    [None, "cc_hhgt.v32.training:run_authorized_task", "attacker:callable"],
)
def test_probe_rejects_none_or_swapped_trainer_before_reverify_or_torch_import(
    trainer: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_reverify(context):  # pragma: no cover - must remain unreachable
        raise AssertionError("authorization reverify was reached")

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "torch":
            raise AssertionError("torch import was reached")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(probe_module, "_reverify_authorization_context", forbidden_reverify)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(RuntimeError, match="AUTHORIZED_TRAINER_MISMATCH"):
        run_authorized_probe(_direct_context(trainer))


def test_probe_rejects_context_subclass_before_reverify_or_torch_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ForgedContext(TrainingAuthorizationContext):
        pass

    base = _direct_context(
        "cc_hhgt.v32.gpu_backward_probe:run_authorized_probe"
    )
    forged = ForgedContext(**base.__dict__)
    monkeypatch.setattr(
        probe_module,
        "_reverify_authorization_context",
        lambda context: (_ for _ in ()).throw(AssertionError("reverify reached")),
    )
    with pytest.raises(RuntimeError, match="REQUIRES_GUARDED"):
        run_authorized_probe(forged)
