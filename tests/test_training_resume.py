from __future__ import annotations

import random

import numpy as np
import pytest

from cc_hhgt.training_resume import (
    CHECKPOINT_FORMAT,
    capture_rng_state,
    restore_rng_state,
    training_resume_contract_sha256,
    training_runtime_fingerprint,
    validate_training_resume_checkpoint,
)


def _config() -> dict:
    return {
        "analysis_version": "v31-test",
        "_formal_training_gate_sha256": "gate-sha",
        "_run_id": "run-1",
        "training": {"epochs": 1000, "patience": 25},
        "state_training": {"auxiliary_loss_weight": 0.5},
        "runtime_graph_sampling": {"edge_chunk_size": 250000},
        "graph_contract": {"primary": "CONTRACT-T"},
        "pathway_prediction": {"target_level": "exact_pathway"},
    }


def test_resume_contract_locks_effective_epoch_cap_but_not_runtime_paths() -> None:
    cfg = _config()
    first = training_resume_contract_sha256(
        cfg, model_name="cc_hhgt", fold_id="LOCO_ACC", seed=20260726
    )
    cfg["_results"] = "/relocated/assets"
    assert first == training_resume_contract_sha256(
        cfg, model_name="cc_hhgt", fold_id="LOCO_ACC", seed=20260726
    )
    cfg["training"]["epochs"] = 500
    assert first != training_resume_contract_sha256(
        cfg, model_name="cc_hhgt", fold_id="LOCO_ACC", seed=20260726
    )


def test_rng_state_round_trip_is_exact() -> None:
    torch = pytest.importorskip("torch")
    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    state = capture_rng_state(torch)
    expected = (random.random(), float(np.random.random()), float(torch.rand(1)))
    restore_rng_state(torch, state)
    observed = (random.random(), float(np.random.random()), float(torch.rand(1)))
    assert observed == expected


def test_resume_checkpoint_validation_rejects_contract_drift() -> None:
    checkpoint = {
        "checkpoint_format": CHECKPOINT_FORMAT,
        "resume_contract_sha256": "contract-a",
        "model_name": "cc_hhgt",
        "fold_id": "LOCO_ACC",
        "seed": 20260726,
        "last_epoch": 5,
        "model_state": {},
        "state_decoder_state": {},
        "optimizer_state": {},
        "scaler_state": {},
        "rng_state": {},
        "history": [{"epoch": epoch} for epoch in range(1, 6)],
        "early_stopping_state": {},
        "runtime_fingerprint": {"device_type": "cpu"},
    }
    validate_training_resume_checkpoint(
        checkpoint,
        expected_contract_sha256="contract-a",
        expected_model_name="cc_hhgt",
        expected_fold_id="LOCO_ACC",
        expected_seed=20260726,
        expected_runtime_fingerprint={"device_type": "cpu"},
    )
    with pytest.raises(RuntimeError, match="resume contract mismatch"):
        validate_training_resume_checkpoint(
            checkpoint,
            expected_contract_sha256="contract-b",
            expected_model_name="cc_hhgt",
            expected_fold_id="LOCO_ACC",
            expected_seed=20260726,
            expected_runtime_fingerprint={"device_type": "cpu"},
        )


def test_runtime_fingerprint_records_deterministic_execution_contract() -> None:
    torch = pytest.importorskip("torch")
    fingerprint = training_runtime_fingerprint(torch, "cpu")
    assert fingerprint["device_type"] == "cpu"
    assert fingerprint["torch_version"] == str(torch.__version__)
    assert "deterministic_algorithms" in fingerprint


def test_model_optimizer_and_rng_resume_matches_uninterrupted_steps() -> None:
    torch = pytest.importorskip("torch")

    def build():
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 8), torch.nn.Dropout(0.25), torch.nn.Linear(8, 1)
        )
        return model, torch.optim.AdamW(model.parameters(), lr=0.01)

    def step(model, optimizer, x, y):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()

    x = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 10
    y = torch.arange(6, dtype=torch.float32).reshape(6, 1) / 10

    torch.manual_seed(101)
    uninterrupted, uninterrupted_optimizer = build()
    for _ in range(6):
        step(uninterrupted, uninterrupted_optimizer, x, y)

    torch.manual_seed(101)
    interrupted, interrupted_optimizer = build()
    for _ in range(3):
        step(interrupted, interrupted_optimizer, x, y)
    saved_model = {
        key: value.detach().clone() for key, value in interrupted.state_dict().items()
    }
    saved_optimizer = interrupted_optimizer.state_dict()
    saved_rng = capture_rng_state(torch)

    resumed, resumed_optimizer = build()
    resumed.load_state_dict(saved_model)
    resumed_optimizer.load_state_dict(saved_optimizer)
    restore_rng_state(torch, saved_rng)
    for _ in range(3):
        step(resumed, resumed_optimizer, x, y)

    for key, expected in uninterrupted.state_dict().items():
        assert torch.equal(expected, resumed.state_dict()[key]), key
