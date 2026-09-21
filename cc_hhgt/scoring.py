from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .calibration import apply_calibration
from .common import LOGGER, read_table, write_table
from .gnn import score_gnn_candidates
from .training_data import candidate_path


def available_model_for_fold(cfg: dict[str, Any], fold_id: str) -> tuple[str, Path]:
    for model_name in cfg["scoring"]["model_preference"]:
        model_dir = cfg["_results"] / "models" / model_name / fold_id
        marker = model_dir / ("best.pt" if model_name in {"rgcn", "hgt", "cc_hhgt"} else "model.joblib")
        if marker.exists():
            return model_name, model_dir
    raise FileNotFoundError(f"No trained model for {fold_id}")


def score_fold_candidates(cfg: dict[str, Any], fold_row: pd.Series) -> pd.DataFrame:
    model_name, model_dir = available_model_for_fold(cfg, fold_row.fold_id)
    candidates = pd.read_parquet(candidate_path(cfg, fold_row.test_cancer))
    if model_name in {"rgcn", "hgt", "cc_hhgt"}:
        scored = score_gnn_candidates(cfg, model_dir, candidates)
    else:
        bundle = joblib.load(model_dir / "model.joblib")
        features = bundle["features"]
        probability = bundle["model"].predict_proba(candidates[features])[:, 1]
        scored = candidates.copy()
        scored["raw_probability"] = probability
        scored["raw_logit"] = np.log(np.clip(probability, 1e-7, 1 - 1e-7) / np.clip(1 - probability, 1e-7, 1))
        scored["direction_probability"] = np.where(scored.get("direction", "unknown") == "positive", 1.0, np.where(scored.get("direction", "unknown") == "negative", 0.0, 0.5))
    calibration_path = model_dir / "calibration.json"
    calibration = json.loads(calibration_path.read_text()) if calibration_path.exists() else {"temperature": 1.0}
    scored["calibrated_probability"] = apply_calibration(scored.raw_logit, calibration, input_is_logit=True)
    scored["uncertainty"] = 1.0 - np.abs(scored.calibrated_probability - 0.5) * 2.0
    scored["predicted_direction"] = np.where(scored.direction_probability >= 0.5, "positive", "negative")
    scored["model_name"] = model_name
    scored["model_analysis_version"] = cfg["analysis_version"]
    scored["fold_id"] = fold_row.fold_id
    scored["test_cancer"] = fold_row.test_cancer
    out = cfg["_results"] / "tables" / "all_candidate_prediction" / f"cancer_id={fold_row.test_cancer}" / "part-0.parquet"
    write_table(scored, out)
    return scored
