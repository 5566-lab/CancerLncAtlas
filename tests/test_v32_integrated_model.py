from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from cc_hhgt.v32.integrated_model import (
    CoreIsolationError,
    assert_v32_core_unchanged,
    build_private_auxiliary_head,
    freeze_v32_core,
    module_state_sha256,
    validate_private_head_checkpoint,
)


def test_private_head_detaches_core_embedding() -> None:
    core = torch.nn.Linear(3, 4)
    before = freeze_v32_core(core)
    head, initialization = build_private_auxiliary_head(
        "state", core_features=4, domain_features=2, hidden_features=8, seed=17
    )
    core_embedding = core(torch.ones(5, 3))
    logits = head(core_embedding, torch.ones(5, 2))
    logits.sum().backward()
    assert all(parameter.grad is None for parameter in core.parameters())
    assert any(parameter.grad is not None for parameter in head.parameters())
    assert_v32_core_unchanged(core, before)
    validate_private_head_checkpoint(initialization.as_dict(), head)


def test_private_head_initialization_is_reproducible_and_checkpoint_free() -> None:
    first, first_record = build_private_auxiliary_head(
        "drug", core_features=3, domain_features=4, seed=20260726
    )
    second, second_record = build_private_auxiliary_head(
        "drug", core_features=3, domain_features=4, seed=20260726
    )
    assert module_state_sha256(first) == module_state_sha256(second)
    assert first_record == second_record
    assert first_record.source_checkpoint_sha256 is None


def test_private_head_rejects_gradient_shape_and_invalid_availability() -> None:
    head, _ = build_private_auxiliary_head(
        "evidence", core_features=3, domain_features=2, output_features=2, seed=3
    )
    with pytest.raises(ValueError, match="batch shapes"):
        head(torch.ones(4, 3), torch.ones(5, 2))
    with pytest.raises(ValueError, match="availability"):
        head(torch.ones(4, 3), torch.ones(4, 2), torch.ones(3, dtype=torch.bool))


def test_unavailable_predictions_are_nan_not_false_zero_or_half() -> None:
    head, _ = build_private_auxiliary_head(
        "mutation_cnv", core_features=3, domain_features=2, seed=3
    )
    output = head(
        torch.ones(2, 3),
        torch.ones(2, 2),
        torch.tensor([True, False]),
    )
    assert torch.isfinite(output[0]).all()
    assert torch.isnan(output[1]).all()


def test_core_mutation_is_detected() -> None:
    core = torch.nn.Linear(2, 2)
    before = freeze_v32_core(core)
    with torch.no_grad():
        core.weight.add_(1.0)
    with pytest.raises(CoreIsolationError, match="parameter hash changed"):
        assert_v32_core_unchanged(core, before)


def test_core_gradient_or_trainability_is_detected() -> None:
    core = torch.nn.Linear(2, 2)
    before = freeze_v32_core(core)
    core.weight.requires_grad_(True)
    with pytest.raises(CoreIsolationError, match="trainable"):
        assert_v32_core_unchanged(core, before)


def test_checkpoint_metadata_rejects_any_source_checkpoint() -> None:
    head, record = build_private_auxiliary_head(
        "clinical", core_features=2, domain_features=2, seed=9
    )
    metadata = record.as_dict()
    metadata["source_checkpoint_sha256"] = "old-checkpoint"
    with pytest.raises(CoreIsolationError, match="source checkpoint"):
        validate_private_head_checkpoint(metadata, head)
