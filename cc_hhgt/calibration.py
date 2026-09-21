from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit

from .common import read_table, write_json, write_table
from .metrics import proxy_binary_metrics


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels, dtype=float)
    def objective(log_temperature: float) -> float:
        temperature = np.exp(log_temperature)
        probability = np.clip(expit(logits / temperature), 1e-7, 1 - 1e-7)
        return float(-np.mean(labels * np.log(probability) + (1 - labels) * np.log(1 - probability)))
    result = minimize_scalar(objective, bounds=(-4, 4), method="bounded")
    return float(np.exp(result.x))


def calibrate_fold_model(
    cfg: dict[str, Any],
    model_name: str,
    fold_id: str,
    model_root: Path | None = None,
) -> dict[str, Any]:
    model_root = model_root or cfg["_results"] / "models"
    model_dir = model_root / model_name / fold_id
    pred_path = model_dir / "prediction_raw.parquet"
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    pred = read_table(pred_path)
    val = pred.loc[pred.split == "val"]
    if val.empty:
        raise RuntimeError(f"No validation predictions: {model_name}/{fold_id}")
    temperature = fit_temperature(val.raw_logit.to_numpy(float), val.proxy_label.to_numpy(float))
    pred["calibrated_probability"] = expit(pred.raw_logit.to_numpy(float) / temperature)
    pred["uncertainty"] = 1.0 - np.abs(pred.calibrated_probability - 0.5) * 2.0
    if "direction_probability" in pred:
        pred["predicted_direction"] = np.where(pred.direction_probability >= 0.5, "positive", "negative")
    else:
        pred["predicted_direction"] = pred.get("direction", "unknown")
    metrics = []
    for split, group in pred.groupby("split", observed=True):
        metrics.append({"fold_id": fold_id, "model_name": model_name, "split": split, "calibrated": True, **proxy_binary_metrics(group, group.calibrated_probability, cfg["calibration"]["bins"])})
    write_table(pred, model_dir / "prediction_calibrated.parquet")
    write_table(pd.DataFrame(metrics), model_dir / "metrics_calibrated.tsv")
    payload = {"method": "temperature", "temperature": temperature, "model_name": model_name, "fold_id": fold_id}
    write_json(payload, model_dir / "calibration.json")
    return payload


def apply_calibration(prob_or_logit: np.ndarray, calibration: dict[str, Any], input_is_logit: bool = True) -> np.ndarray:
    logits = np.asarray(prob_or_logit, dtype=float)
    if not input_is_logit:
        p = np.clip(logits, 1e-7, 1 - 1e-7)
        logits = np.log(p / (1 - p))
    return expit(logits / float(calibration.get("temperature", 1.0)))
