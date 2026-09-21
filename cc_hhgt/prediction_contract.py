"""Machine-readable probability-scale and metric provenance contracts."""
from __future__ import annotations

import hashlib
from enum import Enum
from typing import Iterable

import pandas as pd


class PredictionScale(str, Enum):
    RAW_LOGIT = "raw_logit"
    RAW_PROBABILITY = "raw_probability"
    CALIBRATED_PROBABILITY = "calibrated_probability"


class MetricScope(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    OOF = "OOF"
    LOCO_TEST = "LOCO_test"
    PF_TEST = "PF_test"


def metric_scope_for_split(split: str, *, evaluation: str = "LOCO") -> str:
    split = str(split)
    if split == "train":
        return MetricScope.TRAIN.value
    if split in {"val", "validation"}:
        return MetricScope.VALIDATION.value
    if split != "test":
        raise ValueError(f"Unknown metric split: {split}")
    if evaluation == "LOCO":
        return MetricScope.LOCO_TEST.value
    if evaluation == "PF":
        return MetricScope.PF_TEST.value
    if evaluation == "OOF":
        return MetricScope.OOF.value
    raise ValueError(f"Unknown evaluation scope: {evaluation}")


def candidate_universe_sha256(frame: pd.DataFrame, key_columns: Iterable[str] | None = None) -> str:
    keys = list(key_columns or [
        column
        for column in (
            "candidate_id",
            "cancer_id",
            "lncrna_id",
            "pathway_id",
            "pathway_family_id",
            "state_id",
            "split",
        )
        if column in frame
    ])
    if not keys:
        raise ValueError("Candidate universe has no registered key columns")
    canonical = frame[keys].astype("string").fillna("").drop_duplicates().sort_values(keys, kind="stable")
    digest = hashlib.sha256()
    for row in canonical.itertuples(index=False, name=None):
        digest.update("\x1f".join(map(str, row)).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def validate_prediction_scale(frame: pd.DataFrame, expected: PredictionScale | str) -> None:
    expected = PredictionScale(expected)
    required = {"prediction_scale", "proxy_positive_probability"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Prediction table lacks scale contract fields: {missing}")
    observed = set(frame.prediction_scale.astype(str).unique())
    if observed != {expected.value}:
        raise RuntimeError(f"Prediction scale mismatch: expected={expected.value}, observed={sorted(observed)}")
    probability = pd.to_numeric(frame.proxy_positive_probability, errors="coerce")
    if probability.isna().any() or not probability.between(0, 1).all():
        raise RuntimeError("Prediction probabilities are missing or outside [0, 1]")


def validate_probability_aliases(
    frame: pd.DataFrame,
    expected: PredictionScale | str,
    *,
    atol: float = 1e-12,
) -> None:
    """Validate canonical probability semantics while permitting lineage aliases.

    Formal prediction tables use ``proxy_positive_probability`` plus an explicit
    ``prediction_scale`` as the canonical contract.  ``raw_probability`` and
    ``calibrated_probability`` may coexist so downstream consumers can retain
    both pre- and post-calibration lineage.  The alias matching the declared
    scale must equal the canonical column; any other alias must still be a
    finite probability in [0, 1].
    """
    expected = PredictionScale(expected)
    validate_prediction_scale(frame, expected)
    canonical = pd.to_numeric(frame.proxy_positive_probability, errors="coerce")
    for alias in (
        PredictionScale.RAW_PROBABILITY.value,
        PredictionScale.CALIBRATED_PROBABILITY.value,
    ):
        if alias not in frame:
            continue
        values = pd.to_numeric(frame[alias], errors="coerce")
        if values.isna().any() or not values.between(0, 1).all():
            raise RuntimeError(f"{alias} contains missing or out-of-range probabilities")
        if alias == expected.value and (values - canonical).abs().gt(atol).any():
            raise RuntimeError(
                f"{alias} disagrees with canonical proxy_positive_probability"
            )


def validate_metric_provenance(metrics: pd.DataFrame) -> None:
    required = {
        "prediction_scale",
        "metric_scope",
        "prediction_file_sha256",
        "candidate_universe_sha256",
        "calibration_model_sha256",
    }
    missing = sorted(required - set(metrics.columns))
    if missing:
        raise RuntimeError(f"Metric table lacks provenance fields: {missing}")
    if metrics[list(required)].astype("string").apply(lambda column: column.str.len().eq(0).any()).any():
        raise RuntimeError("Metric provenance contains blank values")
