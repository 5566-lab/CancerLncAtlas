from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

from cc_hhgt.v32 import training  # noqa: E402
from cc_hhgt.v32.training_guard import TrainingAuthorizationContext  # noqa: E402


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="formal shared-loop E2E requires CUDA"
)


class _TinyEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.25))

    def encode(self, graph):
        return graph.reshape(()) * self.scale


class _TinyFormalModel(torch.nn.Module):
    def __init__(self, encoder: _TinyEncoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.membership_scale = torch.nn.Parameter(torch.tensor(0.40))
        self.direction_scale = torch.nn.Parameter(torch.tensor(-0.15))

    def forward(
        self,
        graph,
        candidate_batch,
        base_logit,
        conservation_context,
        graph_available,
        *,
        admitted,
        encoded=None,
    ):
        assert admitted is True
        if encoded is None:
            encoded = self.encoder.encode(graph)
        feature = candidate_batch["feature"].reshape(-1).to(dtype=torch.float32)
        context = conservation_context[:, 0].to(dtype=torch.float32)
        residual = self.membership_scale * feature + encoded + 0.01 * context
        available = graph_available.to(dtype=torch.bool)
        residual = torch.where(available, residual, torch.zeros_like(residual))
        return {
            "final_logit": base_logit.reshape(-1) + residual,
            "direction_logit": self.direction_scale * feature + 0.10 * encoded,
            "raw_graph_residual": residual,
        }


def _batch(offset: int) -> dict[str, object]:
    feature = torch.tensor([-0.5 + 0.1 * offset, 0.7 + 0.1 * offset])
    proxy = torch.tensor([1.0, 0.0])
    return {
        "candidate_batch": {"feature": feature},
        "base_logit": torch.tensor([-0.1, 0.2]),
        "conservation_context": torch.stack(
            (feature, feature.square(), -feature, torch.ones_like(feature)), dim=1
        ),
        "graph_available": torch.ones(2, dtype=torch.bool),
        "proxy_label": proxy,
        "weak_positive": torch.tensor([offset % 2 == 0, False]),
        "direction_label": torch.tensor([0.0, 1.0]),
        "direction_available": torch.ones(2, dtype=torch.bool),
    }


def _install_tiny_authorized_runtime(
    monkeypatch,
    tmp_path: Path,
    *,
    max_cycles: int,
) -> tuple[TrainingAuthorizationContext, Path]:
    import cc_hhgt.gnn as gnn
    import cc_hhgt.v32.model as v32_model
    import cc_hhgt.v32.training_guard as training_guard

    prepared_path = tmp_path / "G2" / "PATIENT_FOLD_0.pt"
    output_root = tmp_path / "formal_output"
    config = {
        "task_contract": {"graph_variant": "G2"},
        "primary_model": {
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "direction_loss_weight": 0.25,
            "residual_shrinkage": 1e-4,
            "evidence_integration": {"mode": "external_router", "modalities": []},
        },
        "runtime_profile": {
            "mixed_precision": False,
            "candidate_microbatch_size": 2,
            "gradient_accumulation": 2,
            "oom_fallback_microbatch_size": 1,
            "oom_fallback_gradient_accumulation": 4,
            "candidate_chunk_schedule_mode": (
                training.CANDIDATE_CHUNK_SCHEDULE_BALANCED_GROUP_SHARED_ENCODER
            ),
            "allow_partial_candidate_chunk_rotation": True,
            "max_coverage_cycles": max_cycles,
            "patience_coverage_cycles": max_cycles + 2,
            "validation_min_delta": 1e-4,
            "require_early_stopping_before_cap": False,
        },
        "training_io": {
            "prepared_fold_pattern": str(prepared_path),
            "output_root": str(output_root),
        },
    }
    bundle = SimpleNamespace(runtime_schedule=object())
    payload = {
        "formal_graph_variant": "G2",
        "legacy_model_config": {},
        "feature_dim": 1,
        "conservation_context_features": 4,
        "bundle": bundle,
        "train_batches": [_batch(index) for index in range(4)],
        "validation_batches": [_batch(10), _batch(11)],
    }
    context = TrainingAuthorizationContext(
        run_id="v32-g012-g2-paid-gpu-20260901-r2-e2e",
        task_id="G2|PATIENT_FOLD_0|SEED_20260726",
        endpoint_id="e2e-local-cuda",
        hardware_class="LOCAL_E2E_CUDA",
        repo_root=str(tmp_path),
        config_path=str(tmp_path / "config.yaml"),
        input_manifest_path=str(tmp_path / "INPUT_MANIFEST.json"),
        task_manifest_path=str(tmp_path / "TASK_MANIFEST.tsv"),
        approval_path=str(tmp_path / "TRAINING_APPROVAL.json"),
        artifact_hashes={
            "code_sha256": "1" * 64,
            "config_sha256": "2" * 64,
            "input_manifest_sha256": "3" * 64,
            "task_manifest_sha256": "4" * 64,
        },
        policy={"training_authorized": True},
        authorized_trainer=training.AUTHORIZED_FORMAL_TRAINER_SPECIFICATION,
    )

    monkeypatch.setattr(training_guard, "guard_training_entry", lambda **_: context)
    monkeypatch.setattr(training, "_load_mapping", lambda _: config)
    monkeypatch.setattr(
        training,
        "_task_row",
        lambda *_: {
            "patient_fold": "0",
            "seed": "20260726",
        },
    )
    monkeypatch.setattr(
        training,
        "load_prepared_artifact_from_authorized_handle",
        lambda *_args, **_kwargs: (payload, "5" * 64),
    )
    monkeypatch.setattr(training, "_validate_prepared", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        training,
        "_prepared_input_authority_hashes",
        lambda _: {"tiny_input_sha256": "6" * 64},
    )
    monkeypatch.setattr(
        training,
        "_runtime_chunk_contract",
        lambda _: ((0, 1), {0: 0.5, 1: 0.5}),
    )
    monkeypatch.setattr(
        training,
        "_runtime_graph_for_chunk",
        lambda _bundle, chunk, *, device: torch.tensor(
            float(int(chunk) + 1), device=device
        ),
    )
    monkeypatch.setattr(gnn, "build_model", lambda *_args, **_kwargs: _TinyEncoder())
    monkeypatch.setattr(
        v32_model,
        "build_v32_cc_hhgt_residual",
        lambda encoder, **_kwargs: _TinyFormalModel(encoder),
    )
    return context, output_root / context.task_id.replace("|", "__")


def test_formal_public_runner_rejects_none_or_swapped_trainer_before_execution(
    monkeypatch, tmp_path
) -> None:
    context, task_root = _install_tiny_authorized_runtime(
        monkeypatch, tmp_path, max_cycles=1
    )
    for observed in (
        None,
        pilot_trainer := (
            "cc_hhgt.v32.equal_step_candidate_pilot:"
            "run_authorized_equal_step_candidate_pilot"
        ),
    ):
        drifted = replace(context, authorized_trainer=observed)
        with pytest.raises(
            RuntimeError,
            match="FORMAL_TRAINER_AUTHORIZATION_DRIFT",
        ):
            training.run_authorized_task(drifted)
    assert pilot_trainer != training.AUTHORIZED_FORMAL_TRAINER_SPECIFICATION
    assert not task_root.exists()


def test_formal_shared_loop_checkpoint_resume_and_success_identity(
    monkeypatch, tmp_path
) -> None:
    context, task_root = _install_tiny_authorized_runtime(
        monkeypatch, tmp_path, max_cycles=2
    )
    original_jsonl = training._atomic_jsonl_write
    crash = {"raised": False}

    def crash_after_first_complete_cycle(rows, path):
        if path.name == "training_log.jsonl" and len(rows) == 1 and not crash["raised"]:
            crash["raised"] = True
            raise RuntimeError("SIMULATED_POST_CHECKPOINT_CRASH")
        return original_jsonl(rows, path)

    monkeypatch.setattr(training, "_atomic_jsonl_write", crash_after_first_complete_cycle)
    with pytest.raises(RuntimeError, match="SIMULATED_POST_CHECKPOINT_CRASH"):
        training.run_authorized_task(context)
    checkpoint_path = task_root / "last_training_state.pt"
    assert checkpoint_path.is_file()
    assert not (task_root / "SUCCESS.json").exists()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["checkpoint_format"] == training.CHECKPOINT_FORMAT
    assert checkpoint["cycle"] == 0
    assert checkpoint["graph_variant"] == "G2"
    assert checkpoint["optimizer_steps"] == 2

    # The old V3 string is a typed rejection, not an ambiguous missing-field error.
    legacy = dict(checkpoint)
    legacy["checkpoint_format"] = training.LEGACY_CHECKPOINT_FORMAT_V3
    torch.save(legacy, checkpoint_path)
    monkeypatch.setattr(training, "_atomic_jsonl_write", original_jsonl)
    with pytest.raises(RuntimeError, match="V3_TYPED_REJECT_REQUIRES_V4"):
        training.run_authorized_task(context)

    # Counter drift is rejected before any resumed optimizer work.
    drifted = dict(checkpoint)
    drifted["encoder_forward_calls"] += 1
    torch.save(drifted, checkpoint_path)
    with pytest.raises(RuntimeError, match="STATE_HISTORY_COUNTER_DRIFT"):
        training.run_authorized_task(context)

    torch.save(checkpoint, checkpoint_path)
    assert training.run_authorized_task(context) == 0
    success = json.loads((task_root / "SUCCESS.json").read_text(encoding="utf-8"))
    assert success["status"] == "SUCCESS"
    assert success["graph_variant"] == "G2"
    assert success["checkpoint_scope"] == (
        "COMPLETE_BALANCED_GROUP_SHARED_ENCODER_CYCLE"
    )
    assert success["completed_cycles"] == 2
    assert success["optimizer_steps"] == 4
    assert success["training_optimizer_steps"] == 4
    assert success["training_encoder_forward_calls"] == 4
    assert success["training_decoder_forward_calls"] == 8
    assert success["training_global_loss_calls"] == 4
    assert success["training_backward_calls"] == 4
    assert success["training_call_telemetry_scope"] == (
        "OPTIMIZER_GROUPS_ONLY_EXCLUDES_VALIDATION"
    )
    assert success["validation_calls_included_in_training_telemetry"] is False
    assert success["allow_partial_candidate_chunk_rotation"] is True
    assert success["full_pair_rotation_completed"] is True
    assert success["partial_candidate_chunk_rotation_used"] is False


def test_formal_shared_loop_training_oom_is_fail_closed_without_fallback(
    monkeypatch, tmp_path
) -> None:
    context, task_root = _install_tiny_authorized_runtime(
        monkeypatch, tmp_path, max_cycles=1
    )
    streaming_calls = {"count": 0}

    def forbidden_streaming(*_args, **_kwargs):
        streaming_calls["count"] += 1
        raise AssertionError("shared OOM must never enter streaming fallback")

    def shared_oom(*_args, **_kwargs):
        raise torch.OutOfMemoryError("CUDA out of memory: injected shared E2E")

    monkeypatch.setattr(training, "streaming_group_backward", forbidden_streaming)
    monkeypatch.setattr(
        training, "shared_encoder_conventional_group_backward", shared_oom
    )
    with pytest.raises(RuntimeError, match="SHARED_ENCODER_GROUP_FAIL_CLOSED"):
        training.run_authorized_task(context)
    assert streaming_calls["count"] == 0
    assert not (task_root / "last_training_state.pt").exists()
    assert not (task_root / "SUCCESS.json").exists()


def test_formal_shared_loop_validation_oom_is_fail_closed_without_retry(
    monkeypatch, tmp_path
) -> None:
    context, task_root = _install_tiny_authorized_runtime(
        monkeypatch, tmp_path, max_cycles=1
    )
    validation_calls = {"count": 0}

    def validation_oom(*_args, **_kwargs):
        validation_calls["count"] += 1
        raise torch.OutOfMemoryError("CUDA out of memory: injected validation E2E")

    monkeypatch.setattr(training, "_evaluate_full_runtime_coverage", validation_oom)
    with pytest.raises(RuntimeError, match="SHARED_ENCODER_MODE_VALIDATION_FAIL_CLOSED"):
        training.run_authorized_task(context)
    assert validation_calls["count"] == 1
    assert not (task_root / "last_training_state.pt").exists()
    assert not (task_root / "SUCCESS.json").exists()
