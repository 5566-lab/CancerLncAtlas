from __future__ import annotations

import pandas as pd
import pytest

from cc_hhgt.prediction_contract import (
    PredictionScale,
    candidate_universe_sha256,
    metric_scope_for_split,
    validate_metric_provenance,
    validate_probability_aliases,
    validate_prediction_scale,
)


def test_prediction_and_metric_provenance_contract() -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": ["b", "a"],
            "split": ["test", "test"],
            "proxy_positive_probability": [0.2, 0.8],
            "prediction_scale": ["raw_probability", "raw_probability"],
        }
    )
    validate_prediction_scale(frame, PredictionScale.RAW_PROBABILITY)
    assert candidate_universe_sha256(frame) == candidate_universe_sha256(frame.iloc[::-1])
    assert metric_scope_for_split("val") == "validation"
    assert metric_scope_for_split("test", evaluation="LOCO") == "LOCO_test"
    metrics = pd.DataFrame(
        {
            "prediction_scale": ["raw_probability"],
            "metric_scope": ["LOCO_test"],
            "prediction_file_sha256": ["a" * 64],
            "candidate_universe_sha256": ["b" * 64],
            "calibration_model_sha256": ["NOT_APPLICABLE_RAW"],
        }
    )
    validate_metric_provenance(metrics)


def test_scale_mismatch_fails_closed() -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": ["a"],
            "proxy_positive_probability": [0.5],
            "prediction_scale": ["calibrated_probability"],
        }
    )
    with pytest.raises(RuntimeError):
        validate_prediction_scale(frame, PredictionScale.RAW_PROBABILITY)


def test_calibrated_contract_permits_raw_lineage_alias() -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": ["a", "b"],
            "proxy_positive_probability": [0.25, 0.75],
            "prediction_scale": ["calibrated_probability"] * 2,
            "raw_probability": [0.10, 0.90],
            "calibrated_probability": [0.25, 0.75],
        }
    )
    validate_probability_aliases(frame, PredictionScale.CALIBRATED_PROBABILITY)


def test_probability_alias_must_match_declared_scale() -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": ["a"],
            "proxy_positive_probability": [0.25],
            "prediction_scale": ["calibrated_probability"],
            "raw_probability": [0.10],
            "calibrated_probability": [0.30],
        }
    )
    with pytest.raises(RuntimeError, match="disagrees with canonical"):
        validate_probability_aliases(frame, PredictionScale.CALIBRATED_PROBABILITY)
