"""Isolated real-data oracle for the V3.2 group-shared estimator.

This module is comparison-only.  Its public callable is reachable through the
guarded ``run-shard`` entry point, reads only G2/PATIENT_FOLD_0 training batch
0..3, and emits typed JSON receipts to stdout.  It never resolves a formal
output root and never writes a checkpoint, SUCCESS/FAILURE marker, prediction,
or winner-selection artifact.

The scientific protocol is frozen in
``docs/v32_group_shared_encoder_estimator_preregistration_20260901.md``.
Importantly, the real-data comparison is fail-closed if the nnPU clamp branch
is not constant across all 58 legacy/proposed assignments at either theta.
When the branch is constant, both K-cycle averages contain the same
candidate-batch/chunk pairs and therefore must agree to the preregistered
near-numerical identity thresholds.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


RECEIPT_FORMAT = "CC_HHGT_V3_2_GROUP_SHARED_ORACLE_COMPARISON_V1"
ARTIFACT_CLASS = "TIMING_AND_ORACLE_COMPARISON_ONLY"
PASS_STATUS = (
    "PASS_REAL_DATA_ORACLE_COMPARISON_ONLY_NOT_AUTHORIZED_FOR_FORMAL_TRAINING"
)
SCIENTIFIC_FAIL_STATUS = "SCIENTIFIC_FAIL_REAL_DATA_ORACLE_COMPARISON_ONLY"
RUNTIME_FAIL_STATUS = "RUNTIME_FAIL_REAL_DATA_ORACLE_COMPARISON_ONLY"
NONZERO_RETURN_CODE = 42

EXPECTED_VARIANT = "G2"
EXPECTED_FOLD = 0
EXPECTED_SEED = 20260726
EXPECTED_BATCH_COUNT = 403
EXPECTED_GROUP_BATCHES = 4
EXPECTED_BATCH_ROWS = 8192
EXPECTED_GROUP_ROWS = EXPECTED_GROUP_BATCHES * EXPECTED_BATCH_ROWS
EXPECTED_RUNTIME_CHUNKS = 29

COMPONENT_NAMES = (
    "membership_risk",
    "direction_loss",
    "shrinkage_penalty",
    "total_objective",
)
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 20260901

SCALAR_ABSOLUTE_FLOOR = 1e-7
SCALAR_RELATIVE_TOLERANCE = 1e-6
BOOTSTRAP_EQUIVALENCE_FRACTION = 0.01
GRADIENT_RELATIVE_L2_MAX = 1e-3
GRADIENT_COSINE_MIN = 0.9999
GRADIENT_NORM_RATIO_MIN = 0.999
GRADIENT_NORM_RATIO_MAX = 1.001
MODULE_GRADIENT_COSINE_MIN = 0.999
IMPLEMENTATION_LOSS_ATOL = 1e-6
IMPLEMENTATION_LOSS_RTOL = 1e-6
IMPLEMENTATION_GRADIENT_MAX_ABS = 1e-5
IMPLEMENTATION_GRADIENT_RELATIVE_L2 = 1e-4
IMPLEMENTATION_GRADIENT_COSINE_MIN = 0.99999


@dataclass(frozen=True)
class ScheduleEvaluation:
    """The complete K-assignment result for one estimator and theta."""

    component_values: Mapping[str, tuple[float, ...]]
    branches: tuple[bool, ...]
    mean_gradients: Mapping[str, np.ndarray]
    mean_gradient_sha256: str
    assignment_count: int
    encoder_forward_calls: int
    decoder_forward_calls: int
    global_loss_calls: int
    backward_calls: int


@dataclass(frozen=True)
class AssignmentObservation:
    """One fixed-theta group objective and its complete parameter gradient."""

    components: Mapping[str, float]
    branch_active: bool
    gradients: Mapping[str, np.ndarray]
    encoder_forward_calls: int
    decoder_forward_calls: int
    global_loss_calls: int
    backward_calls: int


class _GradientKahanAccumulator:
    """Float64/Kahan accumulation without retaining per-assignment gradients."""

    def __init__(self) -> None:
        self._sums: dict[str, np.ndarray] = {}
        self._corrections: dict[str, np.ndarray] = {}
        self.count = 0

    def add(self, gradients: Mapping[str, np.ndarray]) -> None:
        names = set(gradients)
        if self._sums and names != set(self._sums):
            raise RuntimeError("ORACLE_GRADIENT_PARAMETER_TOPOLOGY_DRIFT")
        for name in sorted(gradients):
            value = np.asarray(gradients[name], dtype=np.float64)
            if not np.isfinite(value).all():
                raise RuntimeError(f"NONFINITE_ORACLE_GRADIENT={name}")
            if name not in self._sums:
                self._sums[name] = np.zeros_like(value, dtype=np.float64)
                self._corrections[name] = np.zeros_like(value, dtype=np.float64)
            elif value.shape != self._sums[name].shape:
                raise RuntimeError(f"ORACLE_GRADIENT_SHAPE_DRIFT={name}")
            adjusted = value - self._corrections[name]
            updated = self._sums[name] + adjusted
            self._corrections[name] = (updated - self._sums[name]) - adjusted
            self._sums[name] = updated
        self.count += 1

    def mean(self) -> dict[str, np.ndarray]:
        if self.count < 1:
            raise RuntimeError("ORACLE_GRADIENT_ACCUMULATOR_EMPTY")
        return {
            name: np.ascontiguousarray(value / float(self.count), dtype=np.float64)
            for name, value in self._sums.items()
        }


def _kahan_mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Kahan mean requires at least one value")
    total = 0.0
    correction = 0.0
    for raw in values:
        value = float(raw)
        if not math.isfinite(value):
            raise RuntimeError(f"NONFINITE_ORACLE_SCALAR={value}")
        adjusted = value - correction
        updated = total + adjusted
        correction = (updated - total) - adjusted
        total = updated
    return total / len(values)


def _gradient_sha256(gradients: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(gradients):
        value = np.ascontiguousarray(gradients[name], dtype="<f8")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
        digest.update(b"\0")
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _paired_bootstrap_ci(
    reference: Sequence[float],
    proposed: Sequence[float],
    *,
    seed: int,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> tuple[float, float]:
    """Return a deterministic paired percentile CI for proposed-reference."""

    if len(reference) != len(proposed) or not reference:
        raise ValueError("Paired bootstrap requires aligned non-empty values")
    if replicates < 100:
        raise ValueError("Paired bootstrap replicate count is too small")
    differences = np.asarray(proposed, dtype=np.float64) - np.asarray(
        reference, dtype=np.float64
    )
    if not np.isfinite(differences).all():
        raise RuntimeError("NONFINITE_BOOTSTRAP_DIFFERENCE")
    generator = np.random.default_rng(int(seed))
    indices = generator.integers(
        0, len(differences), size=(int(replicates), len(differences))
    )
    means = differences[indices].mean(axis=1, dtype=np.float64)
    lower, upper = np.quantile(means, (0.025, 0.975), method="linear")
    return float(lower), float(upper)


def _module_parameter_groups(names: Sequence[str]) -> dict[str, tuple[str, ...]]:
    """Map complete parameter names onto the four preregistered module heads."""

    groups: dict[str, list[str]] = {
        "encoder": [],
        "residual_map_output": [],
        "gate": [],
        "direction_head": [],
    }
    for name in sorted(names):
        if name.startswith("encoder."):
            groups["encoder"].append(name)
        elif name.startswith(("residual_map.", "residual_output.")):
            groups["residual_map_output"].append(name)
        elif name.startswith("direction_output."):
            groups["direction_head"].append(name)
        elif name.startswith(
            (
                "gate.",
                "graph_gate.",
                "global_modality_logit",
                "cancer_modality_gate.",
                "signal_modality_gate",
                "evidence_projection.",
                "modality_message.",
                "lncrna_message_projection.",
                "pathway_message_projection.",
            )
        ):
            groups["gate"].append(name)
    return {key: tuple(value) for key, value in groups.items()}


def _gradient_metrics(
    reference: Mapping[str, np.ndarray],
    proposed: Mapping[str, np.ndarray],
    *,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    if set(reference) != set(proposed):
        raise RuntimeError("ORACLE_MEAN_GRADIENT_PARAMETER_TOPOLOGY_DRIFT")
    selected = tuple(sorted(reference) if names is None else sorted(names))
    if any(name not in reference for name in selected):
        raise RuntimeError("ORACLE_MODULE_GRADIENT_PARAMETER_MISSING")
    reference_sq = 0.0
    proposed_sq = 0.0
    difference_sq = 0.0
    dot = 0.0
    for name in selected:
        left = np.asarray(reference[name], dtype=np.float64)
        right = np.asarray(proposed[name], dtype=np.float64)
        if left.shape != right.shape:
            raise RuntimeError(f"ORACLE_MEAN_GRADIENT_SHAPE_DRIFT={name}")
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise RuntimeError(f"NONFINITE_ORACLE_MEAN_GRADIENT={name}")
        reference_sq += float(np.vdot(left.reshape(-1), left.reshape(-1)))
        proposed_sq += float(np.vdot(right.reshape(-1), right.reshape(-1)))
        delta = right - left
        difference_sq += float(np.vdot(delta.reshape(-1), delta.reshape(-1)))
        dot += float(np.vdot(left.reshape(-1), right.reshape(-1)))
    reference_norm = math.sqrt(max(0.0, reference_sq))
    proposed_norm = math.sqrt(max(0.0, proposed_sq))
    difference_norm = math.sqrt(max(0.0, difference_sq))
    if reference_norm == 0.0 and proposed_norm == 0.0:
        return {
            "status": "TYPED_NA_BOTH_ZERO",
            "reference_norm": 0.0,
            "proposed_norm": 0.0,
            "cosine": None,
            "norm_ratio": None,
            "relative_l2": 0.0,
        }
    if reference_norm == 0.0 or proposed_norm == 0.0:
        return {
            "status": "ZERO_NONZERO_MISMATCH",
            "reference_norm": reference_norm,
            "proposed_norm": proposed_norm,
            "cosine": None,
            "norm_ratio": None,
            "relative_l2": None,
        }
    return {
        "status": "COMPARABLE_NONZERO",
        "reference_norm": reference_norm,
        "proposed_norm": proposed_norm,
        "cosine": dot / (reference_norm * proposed_norm),
        "norm_ratio": proposed_norm / reference_norm,
        "relative_l2": difference_norm / reference_norm,
    }


def compare_schedule_evaluations(
    reference: ScheduleEvaluation,
    proposed: ScheduleEvaluation,
    *,
    theta_name: str,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """Apply every frozen scalar, branch, and gradient gate for one theta."""

    if reference.assignment_count != proposed.assignment_count:
        raise RuntimeError("ORACLE_ASSIGNMENT_COUNT_DRIFT")
    if reference.assignment_count != EXPECTED_RUNTIME_CHUNKS:
        raise RuntimeError(
            "ORACLE_REQUIRES_EXACT_K29_ASSIGNMENTS="
            f"{reference.assignment_count}"
        )
    if set(reference.component_values) != set(COMPONENT_NAMES) or set(
        proposed.component_values
    ) != set(COMPONENT_NAMES):
        raise RuntimeError("ORACLE_COMPONENT_SCHEMA_DRIFT")
    for values in (*reference.component_values.values(), *proposed.component_values.values()):
        if len(values) != reference.assignment_count:
            raise RuntimeError("ORACLE_COMPONENT_ASSIGNMENT_COVERAGE_DRIFT")

    all_branches = tuple(reference.branches) + tuple(proposed.branches)
    if len(all_branches) != 2 * EXPECTED_RUNTIME_CHUNKS:
        raise RuntimeError("ORACLE_BRANCH_COVERAGE_DRIFT")
    true_count = sum(bool(value) for value in all_branches)
    branch_constant = true_count in {0, len(all_branches)}
    paired_branch_disagreements = sum(
        bool(left) != bool(right)
        for left, right in zip(reference.branches, proposed.branches, strict=True)
    )

    component_results: dict[str, Any] = {}
    component_pass = True
    for component_index, name in enumerate(COMPONENT_NAMES):
        left = tuple(float(value) for value in reference.component_values[name])
        right = tuple(float(value) for value in proposed.component_values[name])
        reference_mean = _kahan_mean(left)
        proposed_mean = _kahan_mean(right)
        absolute_difference = abs(proposed_mean - reference_mean)
        identity_tolerance = max(
            SCALAR_ABSOLUTE_FLOOR,
            SCALAR_RELATIVE_TOLERANCE * abs(reference_mean),
        )
        lower, upper = _paired_bootstrap_ci(
            left,
            right,
            seed=BOOTSTRAP_SEED
            + 1000 * (0 if theta_name == "theta_0" else 1)
            + component_index,
            replicates=bootstrap_replicates,
        )
        bootstrap_margin = BOOTSTRAP_EQUIVALENCE_FRACTION * abs(reference_mean)
        identity_pass = absolute_difference <= identity_tolerance
        bootstrap_pass = lower >= -bootstrap_margin and upper <= bootstrap_margin
        passed = bool(identity_pass and bootstrap_pass)
        component_pass = component_pass and passed
        component_results[name] = {
            "reference_mean": reference_mean,
            "proposed_mean": proposed_mean,
            "absolute_mean_difference": absolute_difference,
            "identity_tolerance": identity_tolerance,
            "identity_pass": identity_pass,
            "paired_bootstrap_95_ci": [lower, upper],
            "bootstrap_equivalence_margin": bootstrap_margin,
            "bootstrap_pass": bootstrap_pass,
            "pass": passed,
        }

    complete_gradient = _gradient_metrics(
        reference.mean_gradients, proposed.mean_gradients
    )
    complete_gradient_pass = bool(
        complete_gradient["status"] == "COMPARABLE_NONZERO"
        and complete_gradient["relative_l2"] <= GRADIENT_RELATIVE_L2_MAX
        and complete_gradient["cosine"] >= GRADIENT_COSINE_MIN
        and GRADIENT_NORM_RATIO_MIN
        <= complete_gradient["norm_ratio"]
        <= GRADIENT_NORM_RATIO_MAX
    )
    complete_gradient["pass"] = complete_gradient_pass

    module_results: dict[str, Any] = {}
    modules_pass = True
    groups = _module_parameter_groups(tuple(reference.mean_gradients))
    for module, names in groups.items():
        if not names:
            metrics = {"status": "MODULE_PARAMETER_SET_EMPTY", "pass": False}
        else:
            metrics = _gradient_metrics(
                reference.mean_gradients, proposed.mean_gradients, names=names
            )
            if metrics["status"] == "TYPED_NA_BOTH_ZERO":
                metrics["pass"] = True
            elif metrics["status"] == "COMPARABLE_NONZERO":
                metrics["pass"] = bool(
                    metrics["cosine"] >= MODULE_GRADIENT_COSINE_MIN
                )
            else:
                metrics["pass"] = False
        modules_pass = modules_pass and bool(metrics["pass"])
        module_results[module] = {**metrics, "parameter_tensors": len(names)}

    reasons: list[str] = []
    if not branch_constant:
        reasons.append("NNPU_BRANCH_NONCONSTANT_ACROSS_58_EVALUATIONS")
    if paired_branch_disagreements:
        reasons.append("NNPU_PAIRED_BRANCH_DISAGREEMENT")
    if not component_pass:
        reasons.append("SCALAR_OR_BOOTSTRAP_IDENTITY_GATE_FAILED")
    if not complete_gradient_pass:
        reasons.append("COMPLETE_MEAN_GRADIENT_GATE_FAILED")
    if not modules_pass:
        reasons.append("MODULE_MEAN_GRADIENT_GATE_FAILED")
    passed = not reasons
    return {
        "theta": theta_name,
        "pass": passed,
        "scientific_fail_reasons": reasons,
        "branch_gate": {
            "evaluations": len(all_branches),
            "active_count": true_count,
            "inactive_count": len(all_branches) - true_count,
            "constant_across_all_58": branch_constant,
            "paired_disagreement_count": paired_branch_disagreements,
            "pass": branch_constant and paired_branch_disagreements == 0,
        },
        "components": component_results,
        "complete_mean_gradient": complete_gradient,
        "module_mean_gradients": module_results,
        "reference_mean_gradient_sha256": reference.mean_gradient_sha256,
        "proposed_mean_gradient_sha256": proposed.mean_gradient_sha256,
        "reference_call_counts": {
            "encoder_forward": reference.encoder_forward_calls,
            "decoder_forward": reference.decoder_forward_calls,
            "global_loss": reference.global_loss_calls,
            "backward": reference.backward_calls,
        },
        "proposed_call_counts": {
            "encoder_forward": proposed.encoder_forward_calls,
            "decoder_forward": proposed.decoder_forward_calls,
            "global_loss": proposed.global_loss_calls,
            "backward": proposed.backward_calls,
        },
    }


def _group_zero_assignments(
    permutation: Sequence[int], *, group_batches: int = EXPECTED_GROUP_BATCHES
) -> tuple[
    tuple[tuple[tuple[int, int], ...], ...],
    tuple[tuple[tuple[int, int], ...], ...],
]:
    """Return canonical K legacy and K same-chunk group-0 assignments."""

    chunks = tuple(int(value) for value in permutation)
    if len(chunks) != EXPECTED_RUNTIME_CHUNKS or len(set(chunks)) != len(chunks):
        raise RuntimeError("ORACLE_REQUIRES_ONE_PERMUTATION_OF_K29_CHUNKS")
    legacy = tuple(
        tuple(
            (batch, chunks[(batch + offset) % len(chunks)])
            for batch in range(group_batches)
        )
        for offset in range(len(chunks))
    )
    proposed = tuple(
        tuple((batch, chunks[offset]) for batch in range(group_batches))
        for offset in range(len(chunks))
    )
    expected_pairs = {
        (batch, chunk)
        for batch in range(group_batches)
        for chunk in chunks
    }
    for name, assignments in (("legacy", legacy), ("proposed", proposed)):
        flattened = [pair for assignment in assignments for pair in assignment]
        if len(flattened) != len(expected_pairs) or set(flattened) != expected_pairs:
            raise AssertionError(f"{name} oracle assignments do not cover B4 x K29")
    if any(len({chunk for _, chunk in assignment}) != 1 for assignment in proposed):
        raise AssertionError("Proposed oracle assignment is not group-shared")
    return legacy, proposed


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _base_receipt(status: str) -> dict[str, Any]:
    return {
        "format": RECEIPT_FORMAT,
        "status": status,
        "artifact_class": ARTIFACT_CLASS,
        "comparison_only": True,
        "formal_training_authorized": False,
        "formal_artifacts_written": 0,
        "checkpoint_written": False,
        "formal_log_written": False,
        "success_json_written": False,
        "failure_json_written": False,
        "prediction_written": False,
        "winner_selection_input": False,
        "test_labels_read": False,
        "test_metrics_computed": False,
        "validation_payload_schema_validated": False,
        "validation_rows_used": 0,
        "validation_labels_read": False,
    }


def _emit_receipt(receipt: Mapping[str, Any]) -> None:
    print(
        json.dumps(
            dict(receipt), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ),
        flush=True,
    )


@dataclass(frozen=True)
class _GroupLossResult:
    components: Mapping[str, Any]
    nnpu_correction_active: bool
    positive_weight_denominator: float
    unlabeled_count: int
    direction_count: int
    residual_count: int


def _global_group_loss_components(
    outputs,
    batches,
    *,
    torch,
    direction_loss_weight: float,
    shrinkage: float,
) -> _GroupLossResult:
    """One differentiable global reduction, decomposed for the audit receipt.

    This intentionally mirrors ``training._global_loss_from_outputs`` while
    exposing the preregistered components and clamp branch.  A unit test binds
    its total to that production helper.  All reductions are explicitly FP32;
    K-assignment means are accumulated later on CPU in float64/Kahan order.
    """

    if len(outputs) != len(batches) or not outputs:
        raise RuntimeError("Oracle global loss requires aligned non-empty outputs")
    functional = torch.nn.functional
    logits = torch.cat([item["final_logit"].reshape(-1) for item in outputs]).to(
        dtype=torch.float32
    )
    proxy = torch.cat([item["proxy_label"].reshape(-1) for item in batches]).to(
        device=logits.device
    )
    weak = torch.cat([item["weak_positive"].reshape(-1) for item in batches]).to(
        device=logits.device, dtype=torch.bool
    )
    if not (len(logits) == len(proxy) == len(weak)):
        raise RuntimeError("Oracle global nnPU rows are misaligned")

    association_available = torch.cat(
        [
            item.get(
                "association_available",
                torch.ones_like(item["proxy_label"], dtype=torch.bool),
            ).reshape(-1)
            for item in batches
        ]
    ).to(device=logits.device, dtype=torch.bool)
    if len(association_available) != len(logits):
        raise RuntimeError("Oracle association-availability rows are misaligned")
    positive = (proxy > 0.5) & association_available
    unlabeled = (proxy <= 0.5) & association_available
    zero = logits.sum() * 0.0
    positive_weight_denominator = 0.0
    if bool(positive.any()):
        positive_logits = logits[positive]
        weights = torch.where(
            weak[positive],
            torch.full_like(positive_logits, 0.35),
            torch.ones_like(positive_logits),
        )
        denominator = weights.sum().clamp_min(1e-8)
        positive_weight_denominator = float(denominator.detach().cpu())
        positive_risk = 0.10 * (
            functional.softplus(-positive_logits) * weights
        ).sum() / denominator
        positive_negative_risk = 0.10 * (
            functional.softplus(positive_logits) * weights
        ).sum() / denominator
    else:
        positive_risk = zero
        positive_negative_risk = zero
    unlabeled_count = int(unlabeled.sum().item())
    unlabeled_risk = (
        functional.softplus(logits[unlabeled]).mean()
        if unlabeled_count
        else zero
    )
    correction = unlabeled_risk - positive_negative_risk
    branch_active = bool(float(correction.detach().cpu()) >= 0.0)
    membership = positive_risk + 0.12 * torch.clamp(correction, min=0.0)

    direction_sum = zero
    direction_count = 0
    for output, batch in zip(outputs, batches, strict=True):
        if "direction_label" not in batch:
            continue
        direction_logit = output["direction_logit"].reshape(-1).to(
            dtype=torch.float32
        )
        label = batch["direction_label"].reshape(-1).to(
            device=direction_logit.device, dtype=direction_logit.dtype
        )
        available = batch.get(
            "direction_available", torch.ones_like(label, dtype=torch.bool)
        ).reshape(-1).to(device=label.device, dtype=torch.bool)
        association_available = batch.get(
            "association_available", torch.ones_like(label, dtype=torch.bool)
        ).reshape(-1).to(device=label.device, dtype=torch.bool)
        available = available & association_available
        if len(label) != len(direction_logit):
            raise RuntimeError("Oracle direction rows are misaligned")
        if bool(available.any()):
            direction_sum = direction_sum + functional.binary_cross_entropy_with_logits(
                direction_logit[available],
                label[available],
                reduction="sum",
            )
            direction_count += int(available.sum().item())
    direction = direction_sum / direction_count if direction_count else zero

    residuals = torch.cat(
        [item["raw_graph_residual"].reshape(-1) for item in outputs]
    ).to(dtype=torch.float32)
    residual_count = int(residuals.numel())
    shrinkage_penalty = (
        float(shrinkage) * residuals.square().mean()
        if residual_count
        else zero
    )
    total = membership + float(direction_loss_weight) * direction + shrinkage_penalty
    components = {
        "membership_risk": membership,
        "direction_loss": direction,
        "shrinkage_penalty": shrinkage_penalty,
        "total_objective": total,
    }
    if not all(
        bool(torch.isfinite(value.detach()).all()) for value in components.values()
    ):
        raise RuntimeError("NONFINITE_ORACLE_LOSS_COMPONENT")
    return _GroupLossResult(
        components=components,
        nnpu_correction_active=branch_active,
        positive_weight_denominator=positive_weight_denominator,
        unlabeled_count=unlabeled_count,
        direction_count=direction_count,
        residual_count=residual_count,
    )


def _component_floats(result: _GroupLossResult) -> dict[str, float]:
    values = {
        name: float(result.components[name].detach().cpu())
        for name in COMPONENT_NAMES
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise RuntimeError("NONFINITE_ORACLE_COMPONENT_RECEIPT")
    return values


def _capture_complete_gradients(model) -> dict[str, np.ndarray]:
    gradients: dict[str, np.ndarray] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            value = np.zeros(tuple(parameter.shape), dtype=np.float64)
        else:
            value = (
                parameter.grad.detach()
                .to(device="cpu", dtype=__import__("torch").float32)
                .numpy()
                .astype(np.float64, copy=True)
            )
        if not np.isfinite(value).all():
            raise RuntimeError(f"NONFINITE_ORACLE_GRADIENT={name}")
        gradients[str(name)] = np.ascontiguousarray(value)
    if not gradients:
        raise RuntimeError("ORACLE_MODEL_HAS_NO_TRAINABLE_PARAMETERS")
    return gradients


def _runtime_value_sha256(value: Any) -> str:
    """Hash tensors/mappings in memory without serializing to a file."""

    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if hasattr(item, "detach") and hasattr(item, "dtype"):
            tensor = item.detach().to(device="cpu").contiguous()
            digest.update(b"TENSOR\0")
            digest.update(str(tensor.dtype).encode("utf-8"))
            digest.update(b"\0")
            digest.update(
                json.dumps(list(tensor.shape), separators=(",", ":")).encode()
            )
            digest.update(b"\0")
            raw = (
                tensor.reshape(-1)
                .view(__import__("torch").uint8)
                .numpy()
                .tobytes(order="C")
            )
            digest.update(raw)
            return
        if isinstance(item, Mapping):
            digest.update(b"MAPPING\0")
            for key in sorted(item, key=lambda value: str(value)):
                digest.update(str(key).encode("utf-8"))
                digest.update(b"\0")
                update(item[key])
            return
        if isinstance(item, (list, tuple)):
            digest.update(b"SEQUENCE\0")
            digest.update(str(len(item)).encode("ascii"))
            digest.update(b"\0")
            for child in item:
                update(child)
            return
        digest.update(b"SCALAR\0")
        digest.update(
            json.dumps(item, sort_keys=True, default=str, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        digest.update(b"\0")

    update(value)
    return digest.hexdigest()


def _legacy_assignment_observation(
    model,
    raw_batches,
    assignment: Sequence[tuple[int, int]],
    graph_for_chunk: Callable[[int], Any],
    *,
    training,
    torch,
    device: str,
    precision: Mapping[str, Any],
    direction_loss_weight: float,
    shrinkage: float,
) -> AssignmentObservation:
    """Memory-safe exact mixed-chunk group gradient at one frozen theta."""

    if len(assignment) != EXPECTED_GROUP_BATCHES:
        raise RuntimeError("LEGACY_ORACLE_ASSIGNMENT_MUST_CONTAIN_FOUR_BATCHES")
    model.eval()
    model.zero_grad(set_to_none=True)
    start_rng = training._capture_torch_rng_state(torch)
    start_buffers = training._capture_model_buffers(model)
    probe_outputs = []
    moved_labels = []
    for batch_index, chunk in assignment:
        batch = training._move(raw_batches[int(batch_index)], device)
        graph = graph_for_chunk(int(chunk))
        with torch.no_grad(), training._autocast_context(torch, device, precision):
            output = model(
                graph,
                batch["candidate_batch"],
                batch["base_logit"],
                batch["conservation_context"],
                batch.get("graph_available"),
                admitted=True,
            )
        probe_outputs.append(
            {
                key: output[key].detach().to(dtype=torch.float32)
                for key in ("final_logit", "direction_logit", "raw_graph_residual")
            }
        )
        moved_labels.append(
            {
                key: value.detach()
                for key, value in batch.items()
                if key
                in {
                    "proxy_label",
                    "weak_positive",
                    "association_available",
                    "direction_label",
                    "direction_available",
                }
            }
        )
        del output, graph, batch
    result = _global_group_loss_components(
        probe_outputs,
        moved_labels,
        torch=torch,
        direction_loss_weight=direction_loss_weight,
        shrinkage=shrinkage,
    )
    plan = training.GlobalLossPlan(
        positive_weight_denominator=result.positive_weight_denominator,
        unlabeled_count=result.unlabeled_count,
        direction_count=result.direction_count,
        residual_count=result.residual_count,
        nnpu_correction_active=result.nnpu_correction_active,
        objective_value=float(result.components["total_objective"].detach().cpu()),
    )

    replay_outputs = []
    for batch_index, chunk in assignment:
        batch = training._move(raw_batches[int(batch_index)], device)
        graph = graph_for_chunk(int(chunk))
        with training._autocast_context(torch, device, precision):
            output = model(
                graph,
                batch["candidate_batch"],
                batch["base_logit"],
                batch["conservation_context"],
                batch.get("graph_available"),
                admitted=True,
            )
            fragment_loss = training._streaming_fragment_loss(
                output,
                batch,
                plan,
                torch=torch,
                direction_loss_weight=direction_loss_weight,
                shrinkage=shrinkage,
            )
        if not bool(torch.isfinite(fragment_loss.detach()).all()):
            raise RuntimeError("NONFINITE_LEGACY_ORACLE_FRAGMENT_LOSS")
        replay_outputs.append(
            {
                key: output[key].detach().to(dtype=torch.float32)
                for key in ("final_logit", "direction_logit", "raw_graph_residual")
            }
        )
        fragment_loss.backward()
        del fragment_loss, output, graph, batch
    replay = _global_group_loss_components(
        replay_outputs,
        moved_labels,
        torch=torch,
        direction_loss_weight=direction_loss_weight,
        shrinkage=shrinkage,
    )
    if replay.nnpu_correction_active != result.nnpu_correction_active:
        raise RuntimeError("LEGACY_ORACLE_REPLAY_NNPU_BRANCH_DRIFT")
    observed = float(replay.components["total_objective"].detach().cpu())
    tolerance = 1e-6 * max(1.0, abs(plan.objective_value))
    if abs(observed - plan.objective_value) > tolerance:
        raise RuntimeError(
            "LEGACY_ORACLE_REPLAY_OBJECTIVE_DRIFT="
            f"{observed}!={plan.objective_value}"
        )
    if not training._torch_rng_states_equal(
        torch, start_rng, training._capture_torch_rng_state(torch)
    ):
        raise RuntimeError("DROPOUT_DISABLED_LEGACY_ORACLE_RNG_DRIFT")
    if not training._model_buffers_equal(torch, model, start_buffers):
        raise RuntimeError("DROPOUT_DISABLED_LEGACY_ORACLE_BUFFER_DRIFT")
    gradients = _capture_complete_gradients(model)
    model.zero_grad(set_to_none=True)
    return AssignmentObservation(
        components=_component_floats(result),
        branch_active=result.nnpu_correction_active,
        gradients=gradients,
        encoder_forward_calls=2 * len(assignment),
        decoder_forward_calls=2 * len(assignment),
        global_loss_calls=2,
        backward_calls=len(assignment),
    )


def _shared_assignment_observation(
    model,
    raw_batches,
    assignment: Sequence[tuple[int, int]],
    graph_for_chunk: Callable[[int], Any],
    *,
    training,
    torch,
    device: str,
    precision: Mapping[str, Any],
    direction_loss_weight: float,
    shrinkage: float,
) -> AssignmentObservation:
    """One-encode, one-global-loss, one-backward proposed assignment."""

    if len(assignment) != EXPECTED_GROUP_BATCHES:
        raise RuntimeError("SHARED_ORACLE_ASSIGNMENT_MUST_CONTAIN_FOUR_BATCHES")
    chunks = {int(chunk) for _, chunk in assignment}
    if len(chunks) != 1:
        raise RuntimeError(f"SHARED_ORACLE_REQUIRES_UNIQUE_CHUNK={sorted(chunks)}")
    if [int(batch) for batch, _ in assignment] != list(
        range(EXPECTED_GROUP_BATCHES)
    ):
        raise RuntimeError("SHARED_ORACLE_CHANGED_CANONICAL_BATCH_ORDER")
    model.eval()
    model.zero_grad(set_to_none=True)
    start_rng = training._capture_torch_rng_state(torch)
    start_buffers = training._capture_model_buffers(model)
    batches = [
        training._move(raw_batches[int(batch_index)], device)
        for batch_index, _ in assignment
    ]
    graph = graph_for_chunk(next(iter(chunks)))
    with training._autocast_context(torch, device, precision):
        encoded = model.encoder.encode(graph)
        outputs = [
            model(
                graph,
                batch["candidate_batch"],
                batch["base_logit"],
                batch["conservation_context"],
                batch.get("graph_available"),
                admitted=True,
                encoded=encoded,
            )
            for batch in batches
        ]
        result = _global_group_loss_components(
            outputs,
            batches,
            torch=torch,
            direction_loss_weight=direction_loss_weight,
            shrinkage=shrinkage,
        )
    result.components["total_objective"].backward()
    if not training._torch_rng_states_equal(
        torch, start_rng, training._capture_torch_rng_state(torch)
    ):
        raise RuntimeError("DROPOUT_DISABLED_SHARED_ORACLE_RNG_DRIFT")
    if not training._model_buffers_equal(torch, model, start_buffers):
        raise RuntimeError("DROPOUT_DISABLED_SHARED_ORACLE_BUFFER_DRIFT")
    gradients = _capture_complete_gradients(model)
    model.zero_grad(set_to_none=True)
    observation = AssignmentObservation(
        components=_component_floats(result),
        branch_active=result.nnpu_correction_active,
        gradients=gradients,
        encoder_forward_calls=1,
        decoder_forward_calls=len(batches),
        global_loss_calls=1,
        backward_calls=1,
    )
    del outputs, encoded, graph, batches
    return observation


def _evaluate_assignments(
    assignments: Sequence[Sequence[tuple[int, int]]],
    evaluator: Callable[[Sequence[tuple[int, int]]], AssignmentObservation],
    *,
    theta_name: str,
    estimator_name: str,
    heartbeat: Callable[[Mapping[str, Any]], None] | None = None,
) -> ScheduleEvaluation:
    component_values = {name: [] for name in COMPONENT_NAMES}
    branches: list[bool] = []
    gradients = _GradientKahanAccumulator()
    encoder_calls = 0
    decoder_calls = 0
    global_loss_calls = 0
    backward_calls = 0
    for index, assignment in enumerate(assignments):
        observation = evaluator(assignment)
        for name in COMPONENT_NAMES:
            component_values[name].append(float(observation.components[name]))
        branches.append(bool(observation.branch_active))
        gradients.add(observation.gradients)
        encoder_calls += int(observation.encoder_forward_calls)
        decoder_calls += int(observation.decoder_forward_calls)
        global_loss_calls += int(observation.global_loss_calls)
        backward_calls += int(observation.backward_calls)
        if heartbeat is not None:
            heartbeat(
                {
                    "format": RECEIPT_FORMAT,
                    "status": "ORACLE_COMPARISON_HEARTBEAT",
                    "artifact_class": ARTIFACT_CLASS,
                    "comparison_only": True,
                    "formal_training_authorized": False,
                    "theta": theta_name,
                    "estimator": estimator_name,
                    "completed_assignments": index + 1,
                    "total_assignments": len(assignments),
                    "formal_artifacts_written": 0,
                    "validation_rows_used": 0,
                    "validation_labels_read": False,
                    "validation_payload_schema_validated": True,
                    "test_labels_read": False,
                }
            )
    mean_gradients = gradients.mean()
    return ScheduleEvaluation(
        component_values={
            name: tuple(values) for name, values in component_values.items()
        },
        branches=tuple(branches),
        mean_gradients=mean_gradients,
        mean_gradient_sha256=_gradient_sha256(mean_gradients),
        assignment_count=len(assignments),
        encoder_forward_calls=encoder_calls,
        decoder_forward_calls=decoder_calls,
        global_loss_calls=global_loss_calls,
        backward_calls=backward_calls,
    )


def _shared_api_conformance(
    model,
    raw_batches,
    assignment: Sequence[tuple[int, int]],
    graph_for_chunk: Callable[[int], Any],
    *,
    training,
    torch,
    device: str,
    precision: Mapping[str, Any],
    direction_loss_weight: float,
    shrinkage: float,
) -> dict[str, Any]:
    """Bind the independent audit implementation to the production helper."""

    local = _shared_assignment_observation(
        model,
        raw_batches,
        assignment,
        graph_for_chunk,
        training=training,
        torch=torch,
        device=device,
        precision=precision,
        direction_loss_weight=direction_loss_weight,
        shrinkage=shrinkage,
    )
    model.eval()
    model.zero_grad(set_to_none=True)
    work_items = [
        (raw_batches[int(batch_index)], int(chunk))
        for batch_index, chunk in assignment
    ]
    plan, telemetry = training.shared_encoder_conventional_group_backward(
        model,
        work_items,
        graph_for_chunk,
        torch=torch,
        device=device,
        precision=precision,
        direction_loss_weight=direction_loss_weight,
        shrinkage=shrinkage,
    )
    production_gradients = _capture_complete_gradients(model)
    model.zero_grad(set_to_none=True)

    loss_reference = float(local.components["total_objective"])
    loss_observed = float(plan.objective_value)
    loss_tolerance = IMPLEMENTATION_LOSS_ATOL + IMPLEMENTATION_LOSS_RTOL * abs(
        loss_reference
    )
    loss_difference = abs(loss_observed - loss_reference)
    failing_parameters: list[dict[str, Any]] = []
    parameter_tensors_checked = 0
    all_parameter_pass = True
    maximum_absolute_difference = 0.0
    if set(local.gradients) != set(production_gradients):
        raise RuntimeError("SHARED_API_CONFORMANCE_PARAMETER_TOPOLOGY_DRIFT")
    for name in sorted(local.gradients):
        parameter_tensors_checked += 1
        reference = np.asarray(local.gradients[name], dtype=np.float64)
        observed = np.asarray(production_gradients[name], dtype=np.float64)
        delta = observed - reference
        max_abs = float(np.max(np.abs(delta))) if delta.size else 0.0
        reference_norm = float(np.linalg.norm(reference.reshape(-1)))
        relative_l2 = float(np.linalg.norm(delta.reshape(-1))) / max(
            reference_norm, 1e-30
        )
        passed = bool(
            max_abs <= IMPLEMENTATION_GRADIENT_MAX_ABS
            and relative_l2 <= IMPLEMENTATION_GRADIENT_RELATIVE_L2
        )
        all_parameter_pass = all_parameter_pass and passed
        maximum_absolute_difference = max(maximum_absolute_difference, max_abs)
        if not passed:
            failing_parameters.append(
                {
                    "name": name,
                    "max_abs_difference": max_abs,
                    "relative_l2": relative_l2,
                }
            )
    flattened = _gradient_metrics(local.gradients, production_gradients)
    flattened_pass = bool(
        flattened["status"] == "COMPARABLE_NONZERO"
        and flattened["cosine"] >= IMPLEMENTATION_GRADIENT_COSINE_MIN
    )
    telemetry_pass = bool(
        int(telemetry.unique_chunks) == 1
        and int(telemetry.group_rows) == EXPECTED_GROUP_ROWS
        and int(telemetry.encoder_forward_calls) == 1
        and int(telemetry.decoder_forward_calls) == EXPECTED_GROUP_BATCHES
        and int(telemetry.global_loss_calls) == 1
        and int(telemetry.backward_calls) == 1
    )
    branch_pass = bool(
        bool(plan.nnpu_correction_active) == bool(local.branch_active)
    )
    passed = bool(
        loss_difference <= loss_tolerance
        and all_parameter_pass
        and flattened_pass
        and telemetry_pass
        and branch_pass
    )
    return {
        "pass": passed,
        "loss_reference": loss_reference,
        "loss_production": loss_observed,
        "loss_absolute_difference": loss_difference,
        "loss_tolerance": loss_tolerance,
        "nnpu_branch_identical": branch_pass,
        "maximum_parameter_gradient_abs_difference": maximum_absolute_difference,
        "all_parameter_gradient_tolerances_pass": all_parameter_pass,
        "flattened_gradient": flattened,
        "flattened_gradient_cosine_pass": flattened_pass,
        "production_call_telemetry": {
            "unique_chunks": int(telemetry.unique_chunks),
            "group_rows": int(telemetry.group_rows),
            "encoder_forward_calls": int(telemetry.encoder_forward_calls),
            "decoder_forward_calls": int(telemetry.decoder_forward_calls),
            "global_loss_calls": int(telemetry.global_loss_calls),
            "backward_calls": int(telemetry.backward_calls),
        },
        "production_call_telemetry_pass": telemetry_pass,
        "parameter_tensors_checked": parameter_tensors_checked,
        "failing_parameters": failing_parameters,
    }


def _require_oracle_task(task: Mapping[str, Any]) -> None:
    required = {
        "task_type": "CC_HHGT_PATIENT_FOLD",
        "model": "CC-HHGT",
        "status": "PENDING",
        "blocked_reason": "",
        "paid_task": "true",
        "patient_fold": str(EXPECTED_FOLD),
        "seed": str(EXPECTED_SEED),
    }
    drift = {
        key: {"observed": str(task.get(key, "")), "required": value}
        for key, value in required.items()
        if str(task.get(key, "")).strip().lower() != value.lower()
    }
    if drift:
        raise RuntimeError(f"ORACLE_AUTHORIZED_TASK_CONTRACT_DRIFT={drift}")


def _clone_model_state(model) -> dict[str, Any]:
    return {
        name: value.detach().to(device="cpu").clone()
        for name, value in model.state_dict().items()
    }


def _threshold_receipt() -> dict[str, Any]:
    return {
        "branch_constant_across_all_58_required": True,
        "paired_branch_disagreement_max": 0,
        "scalar_absolute_floor": SCALAR_ABSOLUTE_FLOOR,
        "scalar_relative_tolerance": SCALAR_RELATIVE_TOLERANCE,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_equivalence_fraction": BOOTSTRAP_EQUIVALENCE_FRACTION,
        "complete_gradient_relative_l2_max": GRADIENT_RELATIVE_L2_MAX,
        "complete_gradient_cosine_min": GRADIENT_COSINE_MIN,
        "complete_gradient_norm_ratio": [
            GRADIENT_NORM_RATIO_MIN,
            GRADIENT_NORM_RATIO_MAX,
        ],
        "module_gradient_cosine_min": MODULE_GRADIENT_COSINE_MIN,
        "implementation_loss_atol": IMPLEMENTATION_LOSS_ATOL,
        "implementation_loss_rtol": IMPLEMENTATION_LOSS_RTOL,
        "implementation_gradient_max_abs": IMPLEMENTATION_GRADIENT_MAX_ABS,
        "implementation_gradient_relative_l2": (
            IMPLEMENTATION_GRADIENT_RELATIVE_L2
        ),
        "implementation_gradient_cosine_min": (
            IMPLEMENTATION_GRADIENT_COSINE_MIN
        ),
    }


def _run_verified_oracle_comparison(context) -> dict[str, Any]:
    """Execute the frozen G2/Fold0 comparison after guard revalidation."""

    # Torch and production-training imports remain below the authorization
    # boundary enforced by ``run_authorized_oracle_comparison``.
    import torch

    from ..gnn import move_graph
    from . import training
    from .gpu_backward_probe import (
        _build_formal_model,
        _resolve_gpu_identity,
        _validate_graph_variant_binding,
    )

    started = time.perf_counter()
    ctx = training._context_mapping(context)
    root = Path(ctx["repo_root"])
    config = training._load_mapping(Path(ctx["config_path"]))
    task = training._task_row(Path(ctx["task_manifest_path"]), ctx["task_id"])
    _require_oracle_task(task)
    protocol_path = (
        root / "docs" / "v32_group_shared_encoder_estimator_preregistration_20260901.md"
    )
    runner_path = Path(__file__).resolve()
    if not protocol_path.is_file():
        raise RuntimeError(f"ORACLE_PREREGISTRATION_DOCUMENT_MISSING={protocol_path}")
    protocol_sha256 = training._file_sha256(protocol_path)
    runner_sha256 = training._file_sha256(runner_path)
    production_training_path = Path(training.__file__).resolve()
    production_training_sha256 = training._file_sha256(production_training_path)
    fold = int(task["patient_fold"])
    seed = int(task["seed"])

    prepared_path = training._resolve_prepared_path(config, fold, root)
    payload, prepared_sha256 = training.load_prepared_artifact_from_authorized_handle(
        torch,
        ctx["input_manifest_path"],
        fold=fold,
        prepared_path=prepared_path,
    )
    if not isinstance(payload, Mapping):
        raise RuntimeError("Oracle prepared fold artifact must contain a mapping")
    training._validate_prepared(
        payload, fold=fold, artifact_hashes=ctx["artifact_hashes"]
    )
    graph_variant = _validate_graph_variant_binding(
        run_id=str(ctx["run_id"]),
        config=config,
        payload=payload,
        prepared_path=prepared_path,
    )
    if graph_variant != EXPECTED_VARIANT:
        raise RuntimeError(
            f"ORACLE_REQUIRES_GRAPH_VARIANT_{EXPECTED_VARIANT}={graph_variant}"
        )
    input_authority_hashes = training._prepared_input_authority_hashes(payload)

    if not torch.cuda.is_available():
        raise RuntimeError("REAL_DATA_ORACLE_COMPARISON_REQUIRES_CUDA")
    device = "cuda"
    gpu_identity = _resolve_gpu_identity(torch)
    runtime = config.get("runtime_profile", {})
    precision = training.resolve_mixed_precision_contract(
        torch, runtime, device=device
    )
    if not (
        precision.get("requested") == "bf16"
        and precision.get("active") == "bf16"
        and precision.get("autocast_enabled") is True
    ):
        raise RuntimeError(
            f"REAL_DATA_ORACLE_COMPARISON_REQUIRES_ACTIVE_BF16={precision}"
        )
    runtime_plan = training.resolve_runtime_training_plan(runtime)
    if (
        int(runtime_plan.candidate_microbatch_size) != EXPECTED_BATCH_ROWS
        or int(runtime_plan.gradient_accumulation) != EXPECTED_GROUP_BATCHES
    ):
        raise RuntimeError(
            "ORACLE_PRIMARY_GROUP_CONTRACT_DRIFT="
            f"{runtime_plan.candidate_microbatch_size}x"
            f"{runtime_plan.gradient_accumulation}"
        )

    train_batches = payload["train_batches"]
    if len(train_batches) != EXPECTED_BATCH_COUNT:
        raise RuntimeError(
            f"ORACLE_REQUIRES_B403_TRAIN_BATCHES={len(train_batches)}"
        )
    raw_group = train_batches[:EXPECTED_GROUP_BATCHES]
    row_counts = [training._batch_row_count(batch) for batch in raw_group]
    if row_counts != [EXPECTED_BATCH_ROWS] * EXPECTED_GROUP_BATCHES:
        raise RuntimeError(f"ORACLE_GROUP_ZERO_ROW_CONTRACT_DRIFT={row_counts}")
    group_input_sha256 = _runtime_value_sha256(
        {
            "batch_indices": list(range(EXPECTED_GROUP_BATCHES)),
            "batches": raw_group,
        }
    )

    training._seed_everything(torch, seed)
    model, architecture_id = _build_formal_model(
        training, payload, config, device=device
    )
    bundle = payload["bundle"]
    runtime_chunks, runtime_chunk_weights = training._runtime_chunk_contract(bundle)
    if len(runtime_chunks) != EXPECTED_RUNTIME_CHUNKS:
        raise RuntimeError(
            f"ORACLE_REQUIRES_K29_RUNTIME_CHUNKS={len(runtime_chunks)}"
        )
    runtime_permutation = training._frozen_chunk_permutation(
        runtime_chunks, seed=seed, fold=fold
    )
    legacy_assignments, proposed_assignments = _group_zero_assignments(
        runtime_permutation
    )
    assignment_sha256 = _json_sha256(
        {
            "legacy": legacy_assignments,
            "proposed": proposed_assignments,
            "canonical_offsets": list(range(EXPECTED_RUNTIME_CHUNKS)),
        }
    )

    has_runtime_schedule = getattr(bundle, "runtime_schedule", None) is not None
    fixed_graph = payload.get("graph")
    if not has_runtime_schedule:
        raise RuntimeError("K29_ORACLE_REQUIRES_REGISTERED_RUNTIME_GRAPH_SCHEDULE")
    if fixed_graph is not None:
        raise RuntimeError(
            "A fixed prepared graph cannot override the registered runtime schedule"
        )

    def graph_for_chunk(chunk: int):
        return training._runtime_graph_for_chunk(bundle, int(chunk), device=device)

    primary = config.get("primary_model", {})
    direction_loss_weight = float(primary.get("direction_loss_weight", 0.25))
    shrinkage = float(primary.get("residual_shrinkage", 1e-4))
    learning_rate = float(primary.get("learning_rate", 1e-3))
    weight_decay = float(primary.get("weight_decay", 1e-5))

    theta_0_state = _clone_model_state(model)
    theta_0_sha256 = _runtime_value_sha256(theta_0_state)
    theta_0_rng = training._capture_torch_rng_state(torch)
    theta_0_rng_sha256 = _runtime_value_sha256(theta_0_rng)
    torch.cuda.reset_peak_memory_stats()

    def heartbeat(value: Mapping[str, Any]) -> None:
        _emit_receipt(
            {
                **dict(value),
                "run_id": str(ctx["run_id"]),
                "task_id": str(ctx["task_id"]),
                "patient_fold": fold,
                "seed": seed,
                "graph_variant": graph_variant,
                "prepared_artifact_sha256": prepared_sha256,
                "comparison_group_input_sha256": group_input_sha256,
                "assignment_sha256": assignment_sha256,
            }
        )

    def evaluate_theta(
        theta_name: str,
        state: Mapping[str, Any],
        state_sha256: str,
        rng_state: Mapping[str, Any],
    ) -> dict[str, Any]:
        model.load_state_dict(state)
        training._restore_torch_rng_state(torch, rng_state)
        model.eval()
        state_before = _runtime_value_sha256(model.state_dict())
        if state_before != state_sha256:
            raise RuntimeError(f"{theta_name.upper()}_STATE_RESTORE_SHA_DRIFT")
        conformance = _shared_api_conformance(
            model,
            raw_group,
            proposed_assignments[0],
            graph_for_chunk,
            training=training,
            torch=torch,
            device=device,
            precision=precision,
            direction_loss_weight=direction_loss_weight,
            shrinkage=shrinkage,
        )
        if not conformance["pass"]:
            raise RuntimeError(
                f"{theta_name.upper()}_PRODUCTION_SHARED_API_CONFORMANCE_FAILED"
            )
        if _runtime_value_sha256(model.state_dict()) != state_sha256:
            raise RuntimeError(
                f"{theta_name.upper()}_PRODUCTION_SHARED_API_MUTATED_MODEL_STATE"
            )
        if not training._torch_rng_states_equal(
            torch, rng_state, training._capture_torch_rng_state(torch)
        ):
            raise RuntimeError(
                f"{theta_name.upper()}_DROPOUT_DISABLED_SHARED_API_RNG_DRIFT"
            )
        # Make the exhaustive schedules independent of the conformance audit's
        # execution order even though the checks above require zero drift.
        model.load_state_dict(state)
        training._restore_torch_rng_state(torch, rng_state)
        reference = _evaluate_assignments(
            legacy_assignments,
            lambda assignment: _legacy_assignment_observation(
                model,
                raw_group,
                assignment,
                graph_for_chunk,
                training=training,
                torch=torch,
                device=device,
                precision=precision,
                direction_loss_weight=direction_loss_weight,
                shrinkage=shrinkage,
            ),
            theta_name=theta_name,
            estimator_name="exact_cartesian_v1_group0_mixed_chunk_oracle",
            heartbeat=heartbeat,
        )
        proposed = _evaluate_assignments(
            proposed_assignments,
            lambda assignment: _shared_assignment_observation(
                model,
                raw_group,
                assignment,
                graph_for_chunk,
                training=training,
                torch=torch,
                device=device,
                precision=precision,
                direction_loss_weight=direction_loss_weight,
                shrinkage=shrinkage,
            ),
            theta_name=theta_name,
            estimator_name="balanced_group_latin_shared_encoder_v1",
            heartbeat=heartbeat,
        )
        comparison = compare_schedule_evaluations(
            reference, proposed, theta_name=theta_name
        )
        state_after = _runtime_value_sha256(model.state_dict())
        if state_after != state_sha256:
            raise RuntimeError(f"{theta_name.upper()}_MODEL_STATE_MUTATED_BY_ORACLE")
        comparison["model_state_sha256"] = state_sha256
        comparison["production_shared_api_conformance"] = conformance
        comparison["dropout_disabled"] = True
        comparison["optimizer_steps"] = 0
        return comparison

    theta_0_result = evaluate_theta(
        "theta_0", theta_0_state, theta_0_sha256, theta_0_rng
    )

    # Construct theta_1 exactly once in memory from the formal initialization.
    model.load_state_dict(theta_0_state)
    training._restore_torch_rng_state(torch, theta_0_rng)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    optimizer.zero_grad(set_to_none=True)
    legacy_offset_zero = [
        (raw_group[int(batch_index)], int(chunk))
        for batch_index, chunk in legacy_assignments[0]
    ]
    theta_1_plan = training.streaming_group_backward(
        model,
        legacy_offset_zero,
        graph_for_chunk,
        torch=torch,
        device=device,
        precision=precision,
        direction_loss_weight=direction_loss_weight,
        shrinkage=shrinkage,
    )
    theta_1_update = training.optimizer_step_with_guards(
        model,
        optimizer,
        torch=torch,
        objective_value=theta_1_plan.objective_value,
    )
    optimizer.zero_grad(set_to_none=True)
    theta_1_state = _clone_model_state(model)
    theta_1_sha256 = _runtime_value_sha256(theta_1_state)
    if theta_1_sha256 == theta_0_sha256:
        raise RuntimeError("THETA_1_LEGACY_STEP_DID_NOT_CHANGE_MODEL_STATE")
    theta_1_rng = training._capture_torch_rng_state(torch)
    theta_1_rng_sha256 = _runtime_value_sha256(theta_1_rng)
    theta_1_result = evaluate_theta(
        "theta_1", theta_1_state, theta_1_sha256, theta_1_rng
    )

    passed = bool(theta_0_result["pass"] and theta_1_result["pass"])
    status = PASS_STATUS if passed else SCIENTIFIC_FAIL_STATUS
    if training._file_sha256(protocol_path) != protocol_sha256:
        raise RuntimeError("ORACLE_PREREGISTRATION_CHANGED_DURING_COMPARISON")
    if training._file_sha256(runner_path) != runner_sha256:
        raise RuntimeError("ORACLE_RUNNER_CHANGED_DURING_COMPARISON")
    if training._file_sha256(production_training_path) != production_training_sha256:
        raise RuntimeError("PRODUCTION_TRAINING_CODE_CHANGED_DURING_COMPARISON")
    torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - started
    return {
        **_base_receipt(status),
        "scientific_pass": passed,
        "validation_payload_schema_validated": True,
        "run_id": str(ctx["run_id"]),
        "task_id": str(ctx["task_id"]),
        "patient_fold": fold,
        "seed": seed,
        "graph_variant": graph_variant,
        "architecture_id": architecture_id,
        "prepared_artifact_sha256": prepared_sha256,
        "authorization_artifact_hashes": dict(ctx["artifact_hashes"]),
        "input_authority_hashes": input_authority_hashes,
        "preregistration_sha256": protocol_sha256,
        "oracle_runner_sha256": runner_sha256,
        "production_training_sha256": production_training_sha256,
        "candidate_batch_count": len(train_batches),
        "comparison_batch_indices": list(range(EXPECTED_GROUP_BATCHES)),
        "comparison_batch_row_counts": row_counts,
        "comparison_group_rows": sum(row_counts),
        "comparison_group_input_sha256": group_input_sha256,
        "runtime_chunks": list(runtime_chunks),
        "runtime_chunk_weights": {
            str(key): float(value)
            for key, value in runtime_chunk_weights.items()
        },
        "runtime_chunk_permutation": list(runtime_permutation),
        "canonical_offset_order": list(range(EXPECTED_RUNTIME_CHUNKS)),
        "assignment_sha256": assignment_sha256,
        "reference_estimator": "exact_cartesian_v1_group0_mixed_chunk_oracle",
        "proposed_estimator": "balanced_group_latin_shared_encoder_v1",
        "precision_contract": precision,
        "thresholds": _threshold_receipt(),
        "theta_0_model_state_sha256": theta_0_sha256,
        "theta_0_rng_sha256": theta_0_rng_sha256,
        "theta_1_model_state_sha256": theta_1_sha256,
        "theta_1_rng_sha256": theta_1_rng_sha256,
        "theta_1_construction": {
            "legacy_cartesian_pass_offset": 0,
            "optimizer_steps": 1,
            "objective": float(theta_1_plan.objective_value),
            "grad_finite": bool(theta_1_update["grad_finite"]),
            "parameters_finite": bool(theta_1_update["parameters_finite"]),
            "parameter_delta_positive": bool(
                theta_1_update["parameter_delta_positive"]
            ),
        },
        "theta_results": {
            "theta_0": theta_0_result,
            "theta_1": theta_1_result,
        },
        "gpu_identity": gpu_identity,
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "elapsed_seconds": elapsed_seconds,
        "training_batches_used": EXPECTED_GROUP_BATCHES,
        "training_batch_indices_used": list(range(EXPECTED_GROUP_BATCHES)),
        "oof_modality_lineage_preserved": True,
        "patient_fold_authority_preserved": True,
        "optimizer_boundary_preserved": True,
        "immutable_batch_composition_preserved": True,
    }


def run_authorized_oracle_comparison(context: Any) -> int:
    """Guarded public entry; emit one final typed receipt and no artifact file."""

    from .training_guard import TrainingAuthorizationContext, TrainingAuthorizationError

    if type(context) is not TrainingAuthorizationContext:
        raise TrainingAuthorizationError(
            "GROUP_SHARED_ORACLE_REQUIRES_GUARDED_RUN_SHARD_CONTEXT"
        )
    required_trainer = (
        "cc_hhgt.v32.group_shared_encoder_oracle:"
        "run_authorized_oracle_comparison"
    )
    if context.authorized_trainer != required_trainer:
        raise TrainingAuthorizationError(
            "GROUP_SHARED_ORACLE_AUTHORIZED_TRAINER_DRIFT="
            f"expected={required_trainer!r},observed={context.authorized_trainer!r}"
        )
    # Revalidate before importing Torch or any model/training module.
    from .gpu_backward_probe import _reverify_authorization_context

    verified = _reverify_authorization_context(context)
    try:
        receipt = _run_verified_oracle_comparison(verified)
    except Exception as exc:
        receipt = {
            **_base_receipt(RUNTIME_FAIL_STATUS),
            "scientific_pass": False,
            "run_id": str(verified.run_id),
            "task_id": str(verified.task_id),
            "patient_fold": EXPECTED_FOLD,
            "seed": EXPECTED_SEED,
            "graph_variant": EXPECTED_VARIANT,
            "endpoint_id": str(verified.endpoint_id),
            "hardware_class": str(verified.hardware_class),
            "error_type": type(exc).__name__,
            "reason": str(exc),
        }
        _emit_receipt(receipt)
        return NONZERO_RETURN_CODE
    _emit_receipt(receipt)
    return 0 if receipt.get("scientific_pass") is True else NONZERO_RETURN_CODE


__all__ = [
    "AssignmentObservation",
    "ScheduleEvaluation",
    "compare_schedule_evaluations",
    "run_authorized_oracle_comparison",
]
