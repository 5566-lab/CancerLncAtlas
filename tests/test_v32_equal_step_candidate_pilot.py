from __future__ import annotations

import json
import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from cc_hhgt.v32 import equal_step_candidate_pilot as pilot  # noqa: E402
from cc_hhgt.v32 import training  # noqa: E402
from cc_hhgt.v32.group_shared_encoder_oracle import (  # noqa: E402
    _runtime_value_sha256,
)
from cc_hhgt.v32.training_guard import TrainingAuthorizationContext  # noqa: E402


class _TinyEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.08))

    def encode(self, graph):
        return graph.reshape(()) * self.scale


class _TinyPilotModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _TinyEncoder()
        self.membership = torch.nn.Parameter(torch.tensor(0.30))
        self.direction = torch.nn.Parameter(torch.tensor(-0.12))

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
        feature = conservation_context[:, 0].to(dtype=torch.float32)
        residual = self.membership * feature + encoded
        if graph_available is not None:
            residual = torch.where(
                graph_available.to(dtype=torch.bool),
                residual,
                torch.zeros_like(residual),
            )
        return {
            "final_logit": base_logit.reshape(-1) + residual,
            "direction_logit": self.direction * feature + 0.05 * encoded,
            "raw_graph_residual": residual,
        }


def _batch(index: int, *, rows: int = 2) -> dict[str, object]:
    assert rows in {1, 2}
    feature = torch.tensor(
        [-0.7 + 0.08 * index, 0.6 + 0.05 * index], dtype=torch.float32
    )[:rows]
    proxy = torch.tensor([1.0, 0.0], dtype=torch.float32)[:rows]
    l_index = torch.tensor([0, 1], dtype=torch.long)[:rows]
    return {
        "candidate_batch": {
            "l": l_index,
            "p": torch.zeros(rows, dtype=torch.long),
            "c": torch.zeros(rows, dtype=torch.long),
        },
        "base_logit": torch.tensor([-0.2, 0.15], dtype=torch.float32)[:rows],
        "conservation_context": torch.stack(
            (feature, feature.square(), -feature, torch.ones_like(feature)), dim=1
        ),
        "graph_available": torch.ones(rows, dtype=torch.bool),
        "proxy_label": proxy,
        "weak_positive": torch.tensor([index % 2 == 0, False], dtype=torch.bool)[
            :rows
        ],
        "direction_label": torch.tensor([0.0, 1.0], dtype=torch.float32)[:rows],
        "direction_available": torch.ones(rows, dtype=torch.bool),
    }


def _tiny_payload() -> dict[str, object]:
    bundle = SimpleNamespace(
        runtime_schedule=object(),
        node_maps={
            "lncRNA": {"L0": 0, "L1": 1},
            "pathway": {"P0": 0},
            "cancer": {"C0": 0},
        },
    )
    return {
        "bundle": bundle,
        "train_batches": [_batch(index) for index in range(5)],
        "validation_batches": [_batch(20), _batch(21)],
    }


def _run_tiny_pair(monkeypatch, device: str):
    payload = _tiny_payload()
    bundle = payload["bundle"]
    model = _TinyPilotModel().to(device)
    training._seed_everything(torch, 20260726)
    initial_state = pilot._clone_state(model)
    initial_sha = _runtime_value_sha256(initial_state)
    initial_rng = training._capture_torch_rng_state(torch)
    initial_rng_sha = _runtime_value_sha256(initial_rng)
    identity = pilot._validation_identity(
        payload, runtime_value_sha256=_runtime_value_sha256
    )
    chunks = (0, 1)
    weights = {0: 0.5, 1: 0.5}
    permutation = training._frozen_chunk_permutation(
        chunks, seed=20260726, fold=0
    )
    offsets = training._frozen_pass_offset_permutation(
        chunks, seed=20260726, fold=0
    )
    plan = SimpleNamespace(candidate_microbatch_size=2, gradient_accumulation=2)
    precision = {"active": "fp32", "autocast_enabled": False}
    primary = {
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "direction_loss_weight": 0.25,
        "residual_shrinkage": 1e-4,
    }

    def graph_for_chunk(_chunk: int):
        # Equal graphs make the two schedule estimators scientifically equal in
        # this implementation E2E while still traversing both registered paths.
        return torch.tensor(1.0, device=device)

    monkeypatch.setattr(
        training,
        "_runtime_graph_for_chunk",
        lambda _bundle, _chunk, *, device: torch.tensor(1.0, device=device),
    )
    common = dict(
        pilot_id="tiny-pilot",
        authorization_run_id="tiny-run",
        authorization_task_id="tiny-task",
        variant="G2",
        model=model,
        initial_model_state=initial_state,
        initial_model_sha256=initial_sha,
        initial_rng_state=initial_rng,
        initial_rng_sha256=initial_rng_sha,
        train_batches=payload["train_batches"],
        validation_batches=payload["validation_batches"],
        validation_identity=identity,
        bundle=bundle,
        runtime_chunks=chunks,
        runtime_chunk_weights=weights,
        runtime_permutation=permutation,
        pass_offset_permutation=offsets,
        runtime_plan=plan,
        precision=precision,
        primary=primary,
        graph_for_chunk=graph_for_chunk,
        torch=torch,
        training=training,
        runtime_value_sha256=_runtime_value_sha256,
        device=device,
    )
    reference = pilot._execute_arm(
        estimator=pilot.REFERENCE_ESTIMATOR, **common
    )
    proposed = pilot._execute_arm(
        estimator=pilot.PROPOSED_ESTIMATOR, **common
    )
    return reference, proposed


def test_frozen_real_geometry_is_403_batches_101_steps_and_29_chunks() -> None:
    chunks = tuple(range(29))
    permutation = training._frozen_chunk_permutation(
        chunks, seed=20260726, fold=0
    )
    offsets = training._frozen_pass_offset_permutation(
        chunks, seed=20260726, fold=0
    )
    for mode in (pilot.REFERENCE_ESTIMATOR, pilot.PROPOSED_ESTIMATOR):
        schedules = training._candidate_chunk_cycle_schedules(
            batch_count=403,
            chunks=chunks,
            permutation=permutation,
            pass_offset_permutation=offsets,
            max_cycles=1,
            mode=mode,
            gradient_accumulation=4,
        )
        pairs = schedules[0][0]
        assert len(pairs) == 403
        groups = training._canonical_optimizer_batch_groups(403, 4)
        assert len(groups) == 101
        assert groups[-1] == (400, 401, 402)
        if mode == pilot.PROPOSED_ESTIMATOR:
            assert all(len({pairs[index][1] for index in group}) == 1 for group in groups)
        else:
            assert all(len({pairs[index][1] for index in group}) == len(group) for group in groups)


def test_candidate_comparison_applies_all_frozen_thresholds() -> None:
    logits = torch.tensor([-2.0, -0.5, 0.2, 1.8], dtype=torch.float64)
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float64)
    common = {
        "ordered_candidate_key_sha256": "1" * 64,
        "ordered_label_sha256": "2" * 64,
        "ordered_identity_sha256": "3" * 64,
        "ordered_model_input_sha256": "5" * 64,
        "ordered_mask_weight_sha256": "6" * 64,
        "validation_loss_payload_sha256": "7" * 64,
        "validation_loss_contract_sha256": "8" * 64,
        "precision_contract_sha256": "9" * 64,
        "fold_role": pilot._fold_role(),
        "runtime_chunk_weights": {"0": 0.5, "1": 0.5},
        "runtime_chunk_weights_sha256": "4" * 64,
        "candidate_rows": 4,
        "proxy_label": labels,
    }
    reference = {
        **common,
        "mean_final_logit": logits,
        "validation_logloss": 0.3000,
    }
    proposed = {
        **common,
        "mean_final_logit": logits + torch.tensor(
            [0.001, -0.001, 0.001, -0.001], dtype=torch.float64
        ),
        "validation_logloss": 0.3019,
    }
    result = pilot.compare_equal_step_arms(reference, proposed)
    assert result["pass"] is True
    assert result["validation_logloss_absolute_difference"] == pytest.approx(0.0019)
    assert result["mean_logit_pearson"] >= 0.995
    assert result["mean_logit_spearman"] >= 0.995

    failed = dict(proposed)
    failed["validation_logloss"] = 0.30201
    result = pilot.compare_equal_step_arms(reference, failed)
    assert result["pass"] is False
    assert result["gates"]["validation_logloss_absolute_difference"] is False


def test_cpu_synthetic_equal_step_e2e_uses_production_helpers_and_writes_nothing(
    monkeypatch, tmp_path: Path
) -> None:
    artifact_root = tmp_path / "comparison_only"
    artifact_root.mkdir()
    monkeypatch.chdir(artifact_root)
    progress = []
    monkeypatch.setattr(pilot, "_emit_receipt", lambda receipt: progress.append(receipt))
    reference, proposed = _run_tiny_pair(monkeypatch, "cpu")
    assert reference["training_call_telemetry"] == {
        "optimizer_steps": 3,
        "encoder_forward_calls": 10,
        "decoder_forward_calls": 10,
        "global_loss_calls": 6,
        "backward_calls": 5,
    }
    assert proposed["training_call_telemetry"] == {
        "optimizer_steps": 3,
        "encoder_forward_calls": 3,
        "decoder_forward_calls": 5,
        "global_loss_calls": 3,
        "backward_calls": 3,
    }
    assert reference["validation_runtime_chunks"] == 2
    assert proposed["validation_runtime_chunks"] == 2
    comparison = pilot.compare_equal_step_arms(reference, proposed)
    assert comparison["pass"] is True
    optimizer_progress = [
        item for item in progress if item["status"] == "EQUAL_STEP_OPTIMIZER_PROGRESS"
    ]
    validation_progress = [
        item
        for item in progress
        if item["status"] == "EQUAL_STEP_VALIDATION_CHUNK_PROGRESS"
    ]
    assert len(optimizer_progress) == 6
    assert len(validation_progress) == 4
    assert all(item["formal_artifacts_written"] == 0 for item in progress)
    assert all(item["output_channel"] == "STDOUT_JSON_ONLY" for item in progress)
    assert all(item["pilot_id"] == "tiny-pilot" for item in progress)
    assert all(item["authorization_run_id"] == "tiny-run" for item in progress)
    assert all(item["authorization_task_id"] == "tiny-task" for item in progress)
    assert all(item["graph_variant"] == "G2" for item in progress)
    assert len({item["progress_event_id"] for item in progress}) == len(progress)
    assert {
        (item["estimator"], item["optimizer_step"])
        for item in optimizer_progress
    } == {
        (estimator, step)
        for estimator in (pilot.REFERENCE_ESTIMATOR, pilot.PROPOSED_ESTIMATOR)
        for step in (1, 2, 3)
    }
    assert all(
        item["validation_chunks_completed"] in {1, 2}
        for item in validation_progress
    )
    for arm in ("reference", "proposed"):
        arm_steps = [
            item["completed"] for item in optimizer_progress if item["arm"] == arm
        ]
        arm_chunks = [
            item["completed"] for item in validation_progress if item["arm"] == arm
        ]
        assert arm_steps == [1, 2, 3]
        assert arm_chunks == [1, 2]
        assert all(item["phase"] == "OPTIMIZER" for item in optimizer_progress)
        assert all(item["phase"] == "VALIDATION" for item in validation_progress)
        assert all(item["completed"] <= item["total"] for item in progress)
    assert list(artifact_root.iterdir()) == []


def test_validation_identity_binds_every_loss_label_weight_mask_and_model_input() -> None:
    baseline_payload = _tiny_payload()
    baseline = pilot._validation_identity(
        baseline_payload, runtime_value_sha256=_runtime_value_sha256
    )

    def changed(field: str, mutate) -> dict[str, object]:
        payload = _tiny_payload()
        mutate(payload["validation_batches"][0])
        observed = pilot._validation_identity(
            payload, runtime_value_sha256=_runtime_value_sha256
        )
        assert observed[field] != baseline[field]
        assert (
            observed["validation_loss_payload_sha256"]
            != baseline["validation_loss_payload_sha256"]
        )
        assert observed["ordered_identity_sha256"] != baseline["ordered_identity_sha256"]
        return observed

    changed(
        "ordered_model_input_sha256",
        lambda batch: batch["base_logit"].add_(0.125),
    )
    changed(
        "ordered_model_input_sha256",
        lambda batch: batch["conservation_context"].add_(0.25),
    )
    graph_mask = changed(
        "ordered_mask_weight_sha256",
        lambda batch: batch["graph_available"].logical_not_(),
    )
    assert (
        graph_mask["ordered_model_input_sha256"]
        != baseline["ordered_model_input_sha256"]
    )
    changed(
        "ordered_mask_weight_sha256",
        lambda batch: batch["weak_positive"].logical_not_(),
    )
    changed(
        "ordered_mask_weight_sha256",
        lambda batch: batch["direction_available"].logical_not_(),
    )
    changed(
        "ordered_label_sha256",
        lambda batch: batch["direction_label"].logical_not_(),
    )
    changed(
        "ordered_label_sha256",
        lambda batch: batch["proxy_label"].copy_(1 - batch["proxy_label"]),
    )


def test_training_identity_binds_every_ordered_optimizer_input_and_loss_payload() -> None:
    baseline = pilot._training_identity(
        _tiny_payload(), runtime_value_sha256=_runtime_value_sha256
    )

    def changed(field: str, mutate) -> None:
        payload = _tiny_payload()
        mutate(payload["train_batches"][0])
        observed = pilot._training_identity(
            payload, runtime_value_sha256=_runtime_value_sha256
        )
        assert observed["batch_row_counts"] == baseline["batch_row_counts"]
        assert observed[field] != baseline[field]
        assert (
            observed["training_loss_payload_sha256"]
            != baseline["training_loss_payload_sha256"]
        )
        assert observed["ordered_identity_sha256"] != baseline["ordered_identity_sha256"]

    for key in ("l", "p", "c"):
        changed(
            "ordered_candidate_key_sha256",
            lambda batch, key=key: batch["candidate_batch"][key].add_(1),
        )
    changed("ordered_model_input_sha256", lambda batch: batch["base_logit"].add_(0.125))
    changed(
        "ordered_model_input_sha256",
        lambda batch: batch["conservation_context"].add_(0.25),
    )
    changed(
        "ordered_mask_weight_sha256",
        lambda batch: batch["graph_available"].logical_not_(),
    )
    changed(
        "ordered_mask_weight_sha256",
        lambda batch: batch["weak_positive"].logical_not_(),
    )
    changed(
        "ordered_mask_weight_sha256",
        lambda batch: batch["direction_available"].logical_not_(),
    )
    changed(
        "ordered_label_sha256",
        lambda batch: batch["direction_label"].logical_not_(),
    )
    changed(
        "ordered_label_sha256",
        lambda batch: batch["proxy_label"].copy_(1 - batch["proxy_label"]),
    )


def _cross_variant_stub() -> dict[str, dict[str, object]]:
    base = {
        "architecture_id": "HHGT_FORMAL_CORE_EXTERNAL_ROUTER",
        "initial_model_sha256": "1" * 64,
        "initial_rng_sha256": "2" * 64,
        "candidate_batch_row_counts_sha256": "3" * 64,
        "candidate_rows": 3300000,
        "ordered_train_candidate_key_sha256": "1" * 64,
        "ordered_train_label_sha256": "2" * 64,
        "ordered_train_model_input_sha256": "3" * 64,
        "ordered_train_mask_weight_sha256": "4" * 64,
        "training_loss_payload_sha256": "5" * 64,
        "ordered_train_identity_sha256": "6" * 64,
        "runtime_chunks": list(range(29)),
        "runtime_chunk_permutation_sha256": "4" * 64,
        "precision_contract_sha256": "5" * 64,
        "ordered_validation_candidate_key_sha256": "6" * 64,
        "ordered_validation_label_sha256": "7" * 64,
        "ordered_validation_model_input_sha256": "8" * 64,
        "ordered_validation_mask_weight_sha256": "9" * 64,
        "validation_loss_payload_sha256": "a" * 64,
        "validation_loss_contract_sha256": "b" * 64,
        "ordered_validation_identity_sha256": "c" * 64,
        "validation_fold_role": pilot._fold_role(),
        "validation_candidate_rows": 1000,
        "optimization_contract_sha256": "d" * 64,
        "reference": {
            "schedule_sha256": "e" * 64,
            "runtime_chunk_weights_sha256": "f" * 64,
        },
        "proposed": {
            "schedule_sha256": "0" * 64,
            "runtime_chunk_weights_sha256": "f" * 64,
        },
    }
    return {variant: copy.deepcopy(base) for variant in pilot.EXPECTED_VARIANTS}


def test_cross_variant_gates_explicitly_fail_closed_on_fairness_drift() -> None:
    baseline = pilot._cross_variant_comparison_gates(_cross_variant_stub())
    assert baseline and all(baseline.values())
    for field in (
        "initial_model_sha256",
        "initial_rng_sha256",
        "candidate_batch_row_counts_sha256",
        "ordered_train_candidate_key_sha256",
        "ordered_train_label_sha256",
        "ordered_train_model_input_sha256",
        "ordered_train_mask_weight_sha256",
        "training_loss_payload_sha256",
        "ordered_train_identity_sha256",
        "runtime_chunk_permutation_sha256",
        "precision_contract_sha256",
    ):
        variants = _cross_variant_stub()
        variants["G2"][field] = "DRIFT"
        observed = pilot._cross_variant_comparison_gates(variants)
        assert observed[field] is False


def test_observed_timing_projection_is_materialized_not_none() -> None:
    variants: dict[str, dict[str, object]] = {}
    for ordinal, variant in enumerate(pilot.EXPECTED_VARIANTS, start=1):
        reference_training = 10.0 + ordinal
        proposed_training = 12.0 + ordinal
        reference_validation = 20.0 + ordinal
        proposed_validation = 22.0 + ordinal
        variants[variant] = {
            "variant_wall_seconds": (
                reference_training
                + proposed_training
                + reference_validation
                + proposed_validation
                + 5.0
            ),
            "reference": {
                "training_wall_seconds": reference_training,
                "validation_wall_seconds": reference_validation,
            },
            "proposed": {
                "training_wall_seconds": proposed_training,
                "validation_wall_seconds": proposed_validation,
                "training_step_timing": {
                    "mean_seconds": 0.5 * ordinal,
                    "p95_seconds": 0.75 * ordinal,
                },
            },
        }
    observed = pilot._projection_from_observed_timing(variants)
    assert observed["source"] == "OBSERVED_EQUAL_STEP_PILOT_TIMING_INFORMATIONAL_ONLY"
    assert observed["authoritative_paid_time_gate"] is False
    assert observed["still_requires_separate_bounded_probe_receipt"] is True
    assert observed["g2_exact_validation_seconds_upper"] == 25.0
    assert observed["setup_seconds_by_variant"] == {
        "G0": 5.0,
        "G1": 5.0,
        "G2": 5.0,
    }
    assert observed["upper_seconds"] > 0.0
    assert observed["upper_hours"] == pytest.approx(
        observed["upper_seconds"] / 3600.0
    )


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="exact CUDA validation parity"
            ),
        ),
    ],
)
def test_scalar_and_detailed_exact_validation_are_bit_identical(
    monkeypatch, device: str
) -> None:
    payload = _tiny_payload()
    model = _TinyPilotModel().to(device).eval()
    chunks = (0, 1)
    weights = {0: 0.5, 1: 0.5}
    monkeypatch.setattr(
        training,
        "_runtime_graph_for_chunk",
        lambda _bundle, _chunk, *, device: torch.tensor(1.0, device=device),
    )
    kwargs = dict(
        chunks=chunks,
        weights=weights,
        torch=torch,
        device=device,
        direction_loss_weight=0.25,
        shrinkage=1e-4,
        precision={"active": "fp32", "autocast_enabled": False},
    )
    state_before = _runtime_value_sha256(model.state_dict())
    scalar = training._evaluate_full_runtime_coverage(
        model, payload["bundle"], payload["validation_batches"], **kwargs
    )
    detailed = training.evaluate_full_runtime_coverage_detailed(
        model, payload["bundle"], payload["validation_batches"], **kwargs
    )
    assert scalar.hex() == detailed.objective_value.hex()
    assert scalar == detailed.objective_value
    assert detailed.runtime_chunks_evaluated == 2
    assert _runtime_value_sha256(model.state_dict()) == state_before


@pytest.mark.skipif(not torch.cuda.is_available(), reason="small real-CUDA E2E")
def test_local_real_cuda_small_equal_step_e2e(monkeypatch) -> None:
    reference, proposed = _run_tiny_pair(monkeypatch, "cuda")
    assert reference["peak_reserved_bytes"] > 0
    assert proposed["peak_reserved_bytes"] > 0
    assert reference["training_call_telemetry"]["optimizer_steps"] == 3
    assert proposed["training_call_telemetry"]["optimizer_steps"] == 3
    assert pilot.compare_equal_step_arms(reference, proposed)["pass"] is True


def test_training_and_validation_oom_are_typed_fail_closed(monkeypatch) -> None:
    payload = _tiny_payload()
    model = _TinyPilotModel()
    training._seed_everything(torch, 20260726)
    initial_state = pilot._clone_state(model)
    initial_rng = training._capture_torch_rng_state(torch)
    identity = pilot._validation_identity(
        payload, runtime_value_sha256=_runtime_value_sha256
    )
    chunks = (0, 1)
    permutation = training._frozen_chunk_permutation(
        chunks, seed=20260726, fold=0
    )
    offsets = training._frozen_pass_offset_permutation(
        chunks, seed=20260726, fold=0
    )

    def invoke():
        return pilot._execute_arm(
            pilot_id="tiny-pilot",
            authorization_run_id="tiny-run",
            authorization_task_id="tiny-task",
            variant="G2",
            estimator=pilot.PROPOSED_ESTIMATOR,
            model=model,
            initial_model_state=initial_state,
            initial_model_sha256=_runtime_value_sha256(initial_state),
            initial_rng_state=initial_rng,
            initial_rng_sha256=_runtime_value_sha256(initial_rng),
            train_batches=payload["train_batches"],
            validation_batches=payload["validation_batches"],
            validation_identity=identity,
            bundle=payload["bundle"],
            runtime_chunks=chunks,
            runtime_chunk_weights={0: 0.5, 1: 0.5},
            runtime_permutation=permutation,
            pass_offset_permutation=offsets,
            runtime_plan=SimpleNamespace(
                candidate_microbatch_size=2, gradient_accumulation=2
            ),
            precision={"active": "fp32", "autocast_enabled": False},
            primary={
                "learning_rate": 1e-3,
                "weight_decay": 1e-5,
                "direction_loss_weight": 0.25,
                "residual_shrinkage": 1e-4,
            },
            graph_for_chunk=lambda _: torch.tensor(1.0),
            torch=torch,
            training=training,
            runtime_value_sha256=_runtime_value_sha256,
            device="cpu",
        )

    monkeypatch.setattr(
        training,
        "shared_encoder_conventional_group_backward",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            torch.OutOfMemoryError("CUDA out of memory: injected training")
        ),
    )
    with pytest.raises(RuntimeError, match="TRAINING_FAIL_CLOSED"):
        invoke()

    # Restore the real helper, then inject OOM only at exact validation.
    monkeypatch.undo()
    monkeypatch.setattr(
        training,
        "evaluate_full_runtime_coverage_detailed",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            torch.OutOfMemoryError("CUDA out of memory: injected validation")
        ),
    )
    with pytest.raises(RuntimeError, match="VALIDATION_FAIL_CLOSED"):
        invoke()


def test_guarded_callable_reverification_failure_emits_only_typed_stdout(
    monkeypatch, capsys
) -> None:
    context = pilot.EqualStepPilotAuthorizationContext(object(), object(), object())
    monkeypatch.setattr(
        pilot,
        "_reverify_context",
        lambda _context: (_ for _ in ()).throw(RuntimeError("DENIED_BY_TEST_GUARD")),
    )
    assert pilot.run_authorized_equal_step_candidate_pilot(context) == 42
    receipt = json.loads(capsys.readouterr().out.strip())
    assert receipt["status"] == pilot.RUNTIME_FAIL_STATUS
    assert receipt["formal_artifacts_written"] == 0
    assert receipt["formal_checkpoint_written"] is False
    assert receipt["formal_prediction_written"] is False
    assert "DENIED_BY_TEST_GUARD" in receipt["reason"]


def test_reverify_binds_exact_pilot_trainer_and_rejects_swap(monkeypatch) -> None:
    context = TrainingAuthorizationContext(
        run_id="v32-g012-g2-equal-step-test",
        task_id="G2|PATIENT_FOLD_0|SEED_20260726",
        endpoint_id="endpoint",
        hardware_class="RTX_4090_24GB",
        repo_root="repo",
        config_path="config",
        input_manifest_path="input",
        task_manifest_path="tasks",
        approval_path="approval",
        artifact_hashes={"code_sha256": "1" * 64},
        policy={"training_authorized": True},
        authorized_trainer=pilot.AUTHORIZED_TRAINER_SPECIFICATION,
    )
    observed = {}

    def guarded(**kwargs):
        observed.update(kwargs)
        return context

    import cc_hhgt.v32.training_guard as training_guard

    monkeypatch.setattr(training_guard, "guard_training_entry", guarded)
    assert pilot._reverify_context(context) == context
    assert observed["trainer_specification"] == pilot.AUTHORIZED_TRAINER_SPECIFICATION
    swapped = replace(
        context,
        authorized_trainer="cc_hhgt.v32.training:run_authorized_task",
    )
    with pytest.raises(RuntimeError, match="AUTHORIZED_TRAINER_DRIFT"):
        pilot._reverify_context(swapped)


def test_average_rank_ties_are_canonical() -> None:
    observed = pilot._average_ranks(np.asarray([5.0, 1.0, 1.0, 9.0]))
    assert observed.tolist() == pytest.approx([3.0, 1.5, 1.5, 4.0])
