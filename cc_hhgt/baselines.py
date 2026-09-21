from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .common import LOGGER, write_json, write_table
from .metrics import proxy_binary_metrics
from .training_data import split_for_fold


def feature_columns(cfg: dict[str, Any], frame: pd.DataFrame) -> list[str]:
    return [x for x in cfg["training"]["feature_columns"] if x in frame.columns]


def _make_model(name: str, seed: int):
    if name == "logistic":
        return Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed, n_jobs=1)),
        ])
    if name == "hist_gradient_boosting":
        return Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("model", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0, random_state=seed)),
        ])
    raise ValueError(name)


def train_baseline_fold(cfg: dict[str, Any], fold_row: pd.Series, model_name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    train, val, test = split_for_fold(cfg, fold_row)
    if train.empty or val.empty or test.empty:
        raise RuntimeError(f"Fold {fold_row.fold_id} has an empty split")
    features = feature_columns(cfg, train)
    model = _make_model(model_name, int(fold_row.split_seed))
    weights = train.sample_weight.to_numpy(float)
    model.fit(train[features], train.proxy_label, model__sample_weight=weights)
    rows = []
    metrics = []
    for split_name, frame in [("train", train), ("val", val), ("test", test)]:
        probability = model.predict_proba(frame[features])[:, 1]
        part = frame[[c for c in ["candidate_id", "cancer_id", "lncrna_id", "pathway_id", "pathway_family_id", "label_class", "proxy_label", "direction", "observed_evidence_score"] if c in frame]].copy()
        part["split"] = split_name
        part["model_name"] = model_name
        part["model_analysis_version"] = cfg["analysis_version"]
        part["raw_probability"] = probability
        part["raw_logit"] = np.log(np.clip(probability, 1e-7, 1 - 1e-7) / np.clip(1 - probability, 1e-7, 1))
        part["fold_id"] = fold_row.fold_id
        rows.append(part)
        metrics.append({"fold_id": fold_row.fold_id, "model_name": model_name, "split": split_name, **proxy_binary_metrics(frame, probability, bins=cfg["calibration"]["bins"])})
    pred = pd.concat(rows, ignore_index=True)
    metric = pd.DataFrame(metrics)
    model_dir = cfg["_results"] / "models" / model_name / fold_row.fold_id
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": features, "fold": fold_row.to_dict()}, model_dir / "model.joblib")
    write_table(pred, model_dir / "prediction_raw.parquet")
    write_table(metric, model_dir / "metrics.tsv")
    write_json({
        "features": features,
        "fold": fold_row.to_dict(),
        "proxy_positive_policy": "strong_positive+weak_positive",
        "weak_positive_weight": cfg["training"]["weak_positive_weight"],
    }, model_dir / "metadata.json")
    return pred, {"metrics": metric.to_dict(orient="records"), "features": features}
