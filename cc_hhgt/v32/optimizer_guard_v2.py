"""Versioned optimizer-step guard for the V3.2 G0/G1/G2 trainer.

The original :func:`cc_hhgt.v32.training.optimizer_step_with_guards` selected
the smallest parameter tensor with a non-zero gradient and used only that
tensor to decide whether an optimizer step made progress.  A scalar parameter
can legitimately have a non-zero gradient while its update rounds to the
parameter dtype's representable value of zero (the G1F2 failure observed on
RTX 4090).  The old probe therefore reported a false training failure even
when another parameter changed.

This module is intentionally additive and versioned.  It does not modify the
formal/old ``training.py``.  A new code namespace can import or inline
``optimizer_step_with_global_delta_guard`` after the static audit has recorded
the new code hash.
"""
from __future__ import annotations

import math
from typing import Any


OPTIMIZER_GUARD_FORMAT = (
    "CC_HHGT_V3_2_OPTIMIZER_GUARD_V2_GLOBAL_REPRESENTABLE_DELTA"
)


def optimizer_step_with_global_delta_guard(
    model,
    optimizer,
    *,
    torch,
    objective_value: float,
) -> dict[str, Any]:
    """Validate one optimizer step using all trainable parameters.

    ``optimizer.step()`` is allowed to leave an individual parameter unchanged
    when its update is below that parameter's representable precision.  The
    step is considered productive when *any* trainable parameter has a finite,
    strictly positive representable absolute delta.  If every trainable
    parameter has a zero delta, the guard fails closed because the optimizer
    made no observable progress.

    The returned field names intentionally retain the V1 guard contract so the
    existing checkpoint/SUCCESS validators can consume the result.  The
    additional ``update_probe_mode`` and ``update_probe_parameter`` identify
    the parameter with the largest observed delta, rather than the smallest
    gradient tensor.
    """

    if not math.isfinite(float(objective_value)):
        raise RuntimeError(f"NONFINITE_OPTIMIZER_OBJECTIVE={objective_value}")

    gradient_rows: list[tuple[str, Any, Any]] = []
    trainable_rows: list[tuple[str, Any]] = []
    for name, parameter in model.named_parameters():
        if bool(getattr(parameter, "requires_grad", False)):
            trainable_rows.append((name, parameter))
        gradient = parameter.grad
        if gradient is None:
            continue
        gradient_rows.append((name, parameter, gradient))

    if not trainable_rows:
        raise RuntimeError("OPTIMIZER_STEP_HAS_NO_TRAINABLE_PARAMETERS")
    gradient_tensors = len(gradient_rows)
    if gradient_tensors == 0:
        raise RuntimeError("OPTIMIZER_STEP_HAS_NO_GRADIENTS")

    finite_flags = torch.stack(
        [torch.isfinite(gradient).all() for _, _, gradient in gradient_rows]
    )
    if not bool(finite_flags.all()):
        bad = [
            name
            for (name, _, _), finite in zip(
                gradient_rows,
                finite_flags.detach().cpu().tolist(),
                strict=True,
            )
            if not finite
        ]
        raise RuntimeError(f"NONFINITE_GRADIENT={bad}")

    nonzero_flags = torch.stack(
        [gradient.detach().ne(0).any() for _, _, gradient in gradient_rows]
    ).detach().cpu().tolist()
    if not any(nonzero_flags):
        raise RuntimeError("OPTIMIZER_STEP_HAS_ONLY_ZERO_GRADIENTS")

    # Snapshot every trainable parameter before the step.  Cloning all tensors
    # is deliberate: probing only one scalar is the source of the old false
    # positive.  ``detach`` keeps the guard out of autograd.
    before = {
        name: parameter.detach().clone()
        for name, parameter in trainable_rows
    }
    optimizer.step()

    deltas: list[tuple[str, Any, float]] = []
    nonfinite_delta_names: list[str] = []
    for name, parameter in trainable_rows:
        delta = parameter.detach() - before[name]
        if not bool(torch.isfinite(delta).all()):
            # Defer the error until the post-step parameter scan below.  This
            # preserves the established ``NONFINITE_PARAMETER_AFTER_STEP``
            # diagnostic for a parameter that became inf/nan during the step,
            # including when that parameter is not the selected telemetry
            # probe.
            nonfinite_delta_names.append(name)
            maximum_delta = 0.0
        else:
            maximum_delta = float(delta.abs().max().detach().cpu())
        deltas.append((name, delta, maximum_delta))

    parameter_finite_flags = torch.stack(
        [torch.isfinite(parameter.detach()).all() for _, parameter in trainable_rows]
    )
    if not bool(parameter_finite_flags.all()):
        bad = [
            name
            for (name, _), finite in zip(
                trainable_rows,
                parameter_finite_flags.detach().cpu().tolist(),
                strict=True,
            )
            if not finite
        ]
        raise RuntimeError(f"NONFINITE_PARAMETER_AFTER_STEP={bad}")
    if nonfinite_delta_names:
        raise RuntimeError(f"NONFINITE_PARAMETER_DELTA={nonfinite_delta_names}")

    # Select the largest representable update for telemetry.  Ties are made
    # deterministic by parameter name; no parameter is privileged because it
    # happens to be a scalar or have the smallest gradient tensor.
    probe_name, _, maximum_delta = max(
        deltas,
        key=lambda row: (row[2], row[0]),
    )
    if not maximum_delta > 0.0:
        raise RuntimeError(
            "ZERO_PARAMETER_DELTA_AFTER_STEP=GLOBAL_MAX_REPRESENTABLE_DELTA"
        )

    return {
        "guard_format": OPTIMIZER_GUARD_FORMAT,
        "grad_finite": True,
        "gradient_tensors": gradient_tensors,
        "parameters_finite": True,
        "trainable_parameter_tensors": len(trainable_rows),
        "parameter_delta_positive": True,
        "update_probe_mode": "GLOBAL_MAX_REPRESENTABLE_DELTA_V2",
        "update_probe_parameter": probe_name,
        "update_probe_max_abs_delta": maximum_delta,
    }
