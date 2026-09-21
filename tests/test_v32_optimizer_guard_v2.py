"""Regression tests for the versioned global-delta optimizer guard."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from cc_hhgt.v32.optimizer_guard_v2 import (  # noqa: E402
    OPTIMIZER_GUARD_FORMAT,
    optimizer_step_with_global_delta_guard,
)


class _ScalarRoundoffModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # At one, eps/4 is below float32's representable half-ulp and rounds
        # back to one.  The vector receives a clearly representable update.
        self.scalar = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.vector = torch.nn.Parameter(
            torch.tensor([1.0, -1.0], dtype=torch.float32)
        )


class _ScalarNoOpVectorUpdate:
    def __init__(self, model: _ScalarRoundoffModel, *, update_vector: bool) -> None:
        self.model = model
        self.update_vector = update_vector

    def step(self) -> None:
        with torch.no_grad():
            self.model.scalar.add_(torch.finfo(torch.float32).eps / 4)
            if self.update_vector:
                self.model.vector.add_(0.25)


def _set_nonzero_gradients(model: torch.nn.Module) -> None:
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)


def test_scalar_roundoff_does_not_hide_other_representable_update() -> None:
    model = _ScalarRoundoffModel()
    scalar_before = model.scalar.detach().clone()
    vector_before = model.vector.detach().clone()
    _set_nonzero_gradients(model)

    result = optimizer_step_with_global_delta_guard(
        model,
        _ScalarNoOpVectorUpdate(model, update_vector=True),
        torch=torch,
        objective_value=1.0,
    )

    assert torch.equal(model.scalar, scalar_before)
    assert torch.allclose(model.vector, vector_before + 0.25)
    assert result["guard_format"] == OPTIMIZER_GUARD_FORMAT
    assert result["parameter_delta_positive"] is True
    assert result["update_probe_parameter"] == "vector"
    assert result["update_probe_mode"] == "GLOBAL_MAX_REPRESENTABLE_DELTA_V2"
    assert result["update_probe_max_abs_delta"] == pytest.approx(0.25)


def test_global_zero_representable_delta_still_fails_closed() -> None:
    model = _ScalarRoundoffModel()
    _set_nonzero_gradients(model)

    with pytest.raises(
        RuntimeError,
        match="ZERO_PARAMETER_DELTA_AFTER_STEP=GLOBAL_MAX_REPRESENTABLE_DELTA",
    ):
        optimizer_step_with_global_delta_guard(
            model,
            _ScalarNoOpVectorUpdate(model, update_vector=False),
            torch=torch,
            objective_value=1.0,
        )

