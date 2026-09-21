"""Validation-only winner lock for V3.2 routing architectures."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .multimodal_fusion import FUSION_FOLD_COLUMN, TARGET_KEYS, artifact_sha256
from .routed_fair_comparison import exact_candidate_and_fold_hashes


CONTRACT_FORMAT = "CANCERLNCATLAS_V32_ROUTING_VALIDATION_COMPARISON_CONTRACT_V1"
WINNER_FORMAT = "CANCERLNCATLAS_V32_ROUTING_VALIDATION_WINNER_LOCK_V1"
ARCHITECTURES = ("current_primary", "external_router", "hierarchical_end_to_end")
MODALITIES = ("mutation", "cnv", "atac")


class RoutingValidationWinnerError(RuntimeError):
    """Raised when architecture selection could observe non-validation data."""


def _json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RoutingValidationWinnerError(f"Unreadable {label}: {path}") from exc
    if not isinstance(value, dict):
        raise RoutingValidationWinnerError(f"{label} is not a JSON object")
    return value


def _canonical(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    if missing := sorted(set(TARGET_KEYS) - set(frame.columns)):
        raise RoutingValidationWinnerError(f"{label} lacks exact keys: {missing}")
    value = frame.copy()
    value["cancer_id"] = value.cancer_id.astype(str).str.upper()
    for column in ("lncrna_id", "pathway_id"):
        value[column] = value[column].astype(str)
    if value[list(TARGET_KEYS)].duplicated().any():
        raise RoutingValidationWinnerError(f"{label} has duplicate exact keys")
    return value.sort_values(list(TARGET_KEYS), kind="stable").reset_index(drop=True)


def _probability(frame: pd.DataFrame, column: str, label: str) -> np.ndarray:
    if column not in frame:
        raise RoutingValidationWinnerError(f"{label} lacks {column}")
    value = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
    if not np.isfinite(value).all() or ((value < 0) | (value > 1)).any():
        raise RoutingValidationWinnerError(f"{label} {column} is not finite in [0, 1]")
    return value


def _metric(target: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    return {
        "logloss": float(
            -np.mean(
                target * np.log(clipped) + (1.0 - target) * np.log1p(-clipped)
            )
        ),
        "brier": float(np.mean(np.square(target - probability))),
    }


def _validate_contract(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    required = {
        "format": CONTRACT_FORMAT,
        "selection_split": "INNER_VALIDATION_ONLY",
        "outer_test_queries": 0,
    }
    drift = {
        key: {"expected": expected, "observed": value.get(key)}
        for key, expected in required.items()
        if value.get(key) != expected
    }
    if drift:
        raise RoutingValidationWinnerError(f"{label} validation contract drift: {drift}")
    for key in ("candidate_universe_sha256", "validation_pair_fold_sha256"):
        observed = str(value.get(key, ""))
        if len(observed) != 64:
            raise RoutingValidationWinnerError(f"{label} lacks SHA256 {key}")
    if not isinstance(value.get("seed"), int) or not str(
        value.get("training_budget_id", "")
    ).strip():
        raise RoutingValidationWinnerError(f"{label} lacks seed/budget")
    return dict(value)


def _validate_success(
    value: Mapping[str, Any], *, status: str, prediction_sha: str, contract_sha: str
) -> None:
    required = {
        "status": status,
        "outer_test_predictions_written": False,
        "outer_test_metrics_computed": False,
        "architecture_winner_selected": False,
    }
    if any(value.get(key) != expected for key, expected in required.items()):
        raise RoutingValidationWinnerError(
            f"Validation success contract drift for {status}"
        )
    declared_prediction = value.get("validation_predictions_sha256")
    if declared_prediction is None:
        declared_prediction = value.get("validation_predictions", {}).get("sha256")
    declared_contract = value.get("comparison_contract_sha256")
    if declared_contract is None:
        declared_contract = value.get("validation_comparison_contract", {}).get(
            "sha256"
        )
    if declared_prediction != prediction_sha or declared_contract != contract_sha:
        raise RoutingValidationWinnerError(
            f"Validation success artifact binding drift for {status}"
        )


def lock_routing_validation_winner(
    *,
    external_predictions_path: str | Path,
    hierarchical_predictions_path: str | Path,
    external_contract_path: str | Path,
    hierarchical_contract_path: str | Path,
    external_success_path: str | Path,
    hierarchical_success_path: str | Path,
    output_root: str | Path,
    run_id: str,
    minimum_logloss_improvement: float = 1e-4,
    brier_tolerance: float = 0.0,
) -> dict[str, Any]:
    if minimum_logloss_improvement < 0 or brier_tolerance < 0:
        raise RoutingValidationWinnerError("Winner thresholds must be non-negative")
    paths = {
        "external_predictions": Path(external_predictions_path).resolve(),
        "hierarchical_predictions": Path(hierarchical_predictions_path).resolve(),
        "external_contract": Path(external_contract_path).resolve(),
        "hierarchical_contract": Path(hierarchical_contract_path).resolve(),
        "external_success": Path(external_success_path).resolve(),
        "hierarchical_success": Path(hierarchical_success_path).resolve(),
    }
    for label, path in paths.items():
        if not path.is_file() or path.stat().st_size <= 0:
            raise RoutingValidationWinnerError(f"Missing {label}: {path}")
    external_prediction_sha = artifact_sha256(paths["external_predictions"])
    hierarchical_prediction_sha = artifact_sha256(paths["hierarchical_predictions"])
    external_contract_sha = artifact_sha256(paths["external_contract"])
    hierarchical_contract_sha = artifact_sha256(paths["hierarchical_contract"])
    external_contract = _validate_contract(
        _json(paths["external_contract"], "external contract"), "external"
    )
    hierarchical_contract = _validate_contract(
        _json(paths["hierarchical_contract"], "hierarchical contract"),
        "hierarchical",
    )
    if external_contract != hierarchical_contract:
        raise RoutingValidationWinnerError(
            "Routing arms do not share one validation comparison contract"
        )
    _validate_success(
        _json(paths["external_success"], "external success"),
        status="PASS_VALIDATION_ONLY_EXTERNAL_ROUTER",
        prediction_sha=external_prediction_sha,
        contract_sha=external_contract_sha,
    )
    _validate_success(
        _json(paths["hierarchical_success"], "hierarchical success"),
        status="PASS_VALIDATION_ONLY_HIERARCHICAL",
        prediction_sha=hierarchical_prediction_sha,
        contract_sha=hierarchical_contract_sha,
    )
    external = _canonical(
        pd.read_parquet(paths["external_predictions"]), "external validation"
    )
    hierarchical = _canonical(
        pd.read_parquet(paths["hierarchical_predictions"]),
        "hierarchical validation",
    )
    if len(external) != len(hierarchical) or not external[
        list(TARGET_KEYS)
    ].equals(hierarchical[list(TARGET_KEYS)]):
        raise RoutingValidationWinnerError("Validation candidate universes differ")
    candidate_sha, validation_fold_sha = exact_candidate_and_fold_hashes(external)
    if (
        external_contract["candidate_universe_sha256"] != candidate_sha
        or external_contract["validation_pair_fold_sha256"]
        != validation_fold_sha
    ):
        raise RoutingValidationWinnerError(
            "Validation comparison contract is not bound to the supplied frame"
        )
    required_common = {
        FUSION_FOLD_COLUMN,
        "selection_outer_pair_fold",
        "selection_validation_pair_fold",
        "outer_test_queried",
        "fusion_target",
        "primary_probability",
    }
    required_common |= {
        f"{modality}_{suffix}"
        for modality in MODALITIES
        for suffix in ("probability", "available")
    }
    for label, frame in (("external", external), ("hierarchical", hierarchical)):
        if missing := sorted(required_common - set(frame.columns)):
            raise RoutingValidationWinnerError(
                f"{label} validation lacks parity columns: {missing}"
            )
        if not frame.outer_test_queried.eq(False).all():
            raise RoutingValidationWinnerError(
                f"{label} validation queried an outer test row"
            )
        if not np.array_equal(
            pd.to_numeric(frame.selection_validation_pair_fold).to_numpy(int),
            pd.to_numeric(frame[FUSION_FOLD_COLUMN]).to_numpy(int),
        ):
            raise RoutingValidationWinnerError(
                f"{label} validation fold role is inconsistent"
            )
    parity_numeric = (
        "fusion_target",
        "primary_probability",
        FUSION_FOLD_COLUMN,
        "selection_outer_pair_fold",
        "selection_validation_pair_fold",
    )
    for column in parity_numeric:
        left = pd.to_numeric(external[column], errors="coerce").to_numpy(float)
        right = pd.to_numeric(hierarchical[column], errors="coerce").to_numpy(float)
        if not np.allclose(left, right, rtol=0.0, atol=1e-7, equal_nan=False):
            raise RoutingValidationWinnerError(f"Validation parity drift: {column}")
    for modality in MODALITIES:
        available_column = f"{modality}_available"
        probability_column = f"{modality}_probability"
        left_available = external[available_column].astype(bool).to_numpy()
        right_available = hierarchical[available_column].astype(bool).to_numpy()
        if not np.array_equal(left_available, right_available):
            raise RoutingValidationWinnerError(
                f"Validation availability drift: {modality}"
            )
        left_probability = pd.to_numeric(
            external[probability_column], errors="coerce"
        ).to_numpy(float)
        right_probability = pd.to_numeric(
            hierarchical[probability_column], errors="coerce"
        ).to_numpy(float)
        for label, probability, available in (
            ("external", left_probability, left_available),
            ("hierarchical", right_probability, right_available),
        ):
            if np.isnan(probability[available]).any() or np.isfinite(
                probability[~available]
            ).any():
                raise RoutingValidationWinnerError(
                    f"{label} {modality} violates typed missingness"
                )
        if not np.allclose(
            left_probability[left_available],
            right_probability[right_available],
            rtol=0.0,
            atol=1e-7,
        ):
            raise RoutingValidationWinnerError(
                f"Validation modality probabilities differ: {modality}"
            )
    target = _probability(external, "fusion_target", "external")
    scores = {
        "current_primary": _probability(
            external, "primary_probability", "external"
        ),
        "external_router": _probability(
            external, "discovery_adjusted_probability", "external"
        ),
        "hierarchical_end_to_end": _probability(
            hierarchical, "hierarchical_probability", "hierarchical"
        ),
    }
    rows: list[dict[str, Any]] = []
    groups = [("ALL_CANCERS", np.ones(len(external), dtype=bool))] + [
        (str(cancer), external.cancer_id.eq(cancer).to_numpy())
        for cancer in sorted(external.cancer_id.unique())
    ]
    for cancer, mask in groups:
        primary_metric = _metric(target[mask], scores["current_primary"][mask])
        for architecture, probability in scores.items():
            metric = _metric(target[mask], probability[mask])
            rows.append(
                {
                    "cancer_id": cancer,
                    "architecture_id": architecture,
                    "rows": int(mask.sum()),
                    **metric,
                    "delta_logloss_vs_primary": primary_metric["logloss"]
                    - metric["logloss"],
                    "delta_brier_vs_primary": primary_metric["brier"]
                    - metric["brier"],
                }
            )
    metrics = pd.DataFrame(rows)
    global_metrics = metrics.loc[metrics.cancer_id.eq("ALL_CANCERS")].set_index(
        "architecture_id"
    )
    primary_metric = global_metrics.loc["current_primary"]
    eligible = ["current_primary"]
    for architecture in ARCHITECTURES[1:]:
        metric = global_metrics.loc[architecture]
        if (
            float(primary_metric.logloss) - float(metric.logloss)
            >= minimum_logloss_improvement
            and float(metric.brier)
            <= float(primary_metric.brier) + brier_tolerance
        ):
            eligible.append(architecture)
    priority = {name: index for index, name in enumerate(ARCHITECTURES)}
    winner = min(
        eligible,
        key=lambda name: (
            float(global_metrics.loc[name].logloss),
            float(global_metrics.loc[name].brier),
            priority[name],
        ),
    )
    output = Path(output_root).resolve()
    staging = output.with_name(f"{output.name}.partial")
    if output.exists() or staging.exists():
        raise RoutingValidationWinnerError(
            f"Routing winner lock refuses output/staging reuse: {output}"
        )
    staging.mkdir(parents=True)
    try:
        metrics_path = staging / "validation_architecture_metrics.parquet"
        metrics.to_parquet(metrics_path, index=False)
        source_records = {
            label: {"path": str(path), "sha256": artifact_sha256(path)}
            for label, path in paths.items()
        }
        winner_payload = {
            "format": WINNER_FORMAT,
            "status": "PASS_ROUTING_VALIDATION_WINNER_LOCK",
            "run_id": run_id,
            "winner_id": winner,
            "compared_models": list(ARCHITECTURES),
            "eligible_models": eligible,
            "selection_split": "validation_only",
            "heldout_test_used_for_selection": False,
            "outer_test_predictions_available_during_selection": False,
            "fair_comparison": True,
            "winner_score_name": "validation_logloss",
            "winner_score": float(global_metrics.loc[winner].logloss),
            "minimum_logloss_improvement": float(minimum_logloss_improvement),
            "brier_tolerance": float(brier_tolerance),
            "comparison_contract": external_contract,
            "metrics": {
                "path": str(output / metrics_path.name),
                "sha256": artifact_sha256(metrics_path),
            },
            "sources": source_records,
            "test_metrics_used_for_selection": False,
            "winner_locked_before_outer_test_inference": True,
            "requires_untouched_external_validation_before_primary_promotion": True,
        }
        winner_path = staging / "ROUTING_VALIDATION_WINNER_LOCK.json"
        winner_path.write_text(
            json.dumps(winner_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        success = {
            "status": "PASS_ROUTING_VALIDATION_WINNER_LOCK",
            "run_id": run_id,
            "winner_id": winner,
            "winner_declaration": {
                "path": str(output / winner_path.name),
                "sha256": artifact_sha256(winner_path),
            },
            "metrics_sha256": artifact_sha256(metrics_path),
            "selection_split": "validation_only",
            "heldout_test_used_for_selection": False,
            "outer_test_predictions_available_during_selection": False,
        }
        success_path = staging / "SUCCESS.json"
        success_path.write_text(
            json.dumps(success, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(staging, output)
        return success
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def materialize_primary_outer_after_winner_lock(
    *,
    primary_frame_path: str | Path,
    external_validation_predictions_path: str | Path,
    routing_winner_declaration_path: str | Path,
    routing_winner_declaration_sha256: str,
    output_root: str | Path,
    run_id: str,
) -> dict[str, Any]:
    """Materialize the unchanged primary fallback after it wins validation."""

    primary_path = Path(primary_frame_path).resolve()
    validation_path = Path(external_validation_predictions_path).resolve()
    winner_path = Path(routing_winner_declaration_path).resolve()
    for label, path in (
        ("primary frame", primary_path),
        ("external validation predictions", validation_path),
        ("routing winner", winner_path),
    ):
        if not path.is_file() or path.stat().st_size <= 0:
            raise RoutingValidationWinnerError(f"Missing {label}: {path}")
    expected_winner_sha = str(routing_winner_declaration_sha256).lower()
    if len(expected_winner_sha) != 64 or artifact_sha256(winner_path) != expected_winner_sha:
        raise RoutingValidationWinnerError("Routing winner declaration SHA256 drift")
    winner = _json(winner_path, "routing winner")
    required = {
        "format": WINNER_FORMAT,
        "status": "PASS_ROUTING_VALIDATION_WINNER_LOCK",
        "winner_id": "current_primary",
        "selection_split": "validation_only",
        "heldout_test_used_for_selection": False,
        "outer_test_predictions_available_during_selection": False,
        "test_metrics_used_for_selection": False,
        "winner_locked_before_outer_test_inference": True,
    }
    if any(winner.get(key) != value for key, value in required.items()):
        raise RoutingValidationWinnerError(
            "Routing winner does not authorize primary outer inference"
        )
    source = winner.get("sources", {}).get("external_predictions", {})
    if source.get("sha256") != artifact_sha256(validation_path):
        raise RoutingValidationWinnerError(
            "Primary outer input is not bound through validation predictions"
        )
    primary = _canonical(pd.read_parquet(primary_path), "primary outer frame")
    validation = _canonical(
        pd.read_parquet(validation_path), "winner-bound external validation"
    )
    if (
        len(primary) != len(validation)
        or not primary[list(TARGET_KEYS)].equals(validation[list(TARGET_KEYS)])
    ):
        raise RoutingValidationWinnerError("Primary outer candidate universe drift")
    for column in ("fusion_target", "primary_probability", FUSION_FOLD_COLUMN):
        left = pd.to_numeric(primary[column], errors="coerce").to_numpy(float)
        right = pd.to_numeric(validation[column], errors="coerce").to_numpy(float)
        if not np.allclose(left, right, rtol=0.0, atol=1e-7, equal_nan=False):
            raise RoutingValidationWinnerError(f"Primary outer parity drift: {column}")
    target = _probability(primary, "fusion_target", "primary outer")
    probability = _probability(
        primary, "primary_probability", "primary outer"
    )
    prediction = primary[
        list(TARGET_KEYS)
        + [FUSION_FOLD_COLUMN, "fusion_target", "primary_probability"]
    ].copy()
    prediction["winner_probability"] = probability
    prediction["winner_id"] = "current_primary"
    prediction["winner_declaration_sha256"] = expected_winner_sha
    prediction["outer_inference_after_winner_lock"] = True
    rows: list[dict[str, Any]] = []
    groups = [("ALL_CANCERS", np.ones(len(primary), dtype=bool))] + [
        (str(cancer), primary.cancer_id.eq(cancer).to_numpy())
        for cancer in sorted(primary.cancer_id.unique())
    ]
    for cancer, mask in groups:
        metric = _metric(target[mask], probability[mask])
        rows.append(
            {
                "cancer_id": cancer,
                "rows": int(mask.sum()),
                **metric,
                "winner_id": "current_primary",
                "outer_metric_computed_after_winner_lock": True,
            }
        )
    metrics = pd.DataFrame(rows)
    output = Path(output_root).resolve()
    staging = output.with_name(f"{output.name}.partial")
    if output.exists() or staging.exists():
        raise RoutingValidationWinnerError(
            f"Primary outer inference refuses output/staging reuse: {output}"
        )
    staging.mkdir(parents=True)
    try:
        prediction_path = staging / "primary_winner_locked_oof.PRIVATE.parquet"
        metrics_path = staging / "primary_winner_locked_metrics.parquet"
        prediction.to_parquet(prediction_path, index=False)
        metrics.to_parquet(metrics_path, index=False)
        success = {
            "status": "PASS_PRIMARY_OUTER_AFTER_WINNER_LOCK",
            "run_id": run_id,
            "winner_id": "current_primary",
            "routing_winner_declaration": {
                "path": str(winner_path),
                "sha256": expected_winner_sha,
            },
            "outer_predictions": {
                "path": str(output / prediction_path.name),
                "sha256": artifact_sha256(prediction_path),
            },
            "outer_metrics": {
                "path": str(output / metrics_path.name),
                "sha256": artifact_sha256(metrics_path),
            },
            "outer_inference_started_after_winner_lock": True,
            "test_metrics_used_for_selection": False,
            "formal_v32_primary_unchanged": True,
        }
        success_path = staging / "SUCCESS.json"
        success_path.write_text(
            json.dumps(success, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(staging, output)
        return success
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = [
    "ARCHITECTURES",
    "CONTRACT_FORMAT",
    "RoutingValidationWinnerError",
    "WINNER_FORMAT",
    "lock_routing_validation_winner",
    "materialize_primary_outer_after_winner_lock",
]
