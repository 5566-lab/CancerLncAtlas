from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "78_v31_selected_model_seed_ensemble_inference.py"
SPEC = importlib.util.spec_from_file_location("v31_selected_seed_ensemble", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _prediction(offset: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_id": ["B", "A"],
            "cancer_id": ["C", "C"],
            "lncrna_id": ["L2", "L1"],
            "pathway_id": ["P2", "P1"],
            "pathway_family_id": ["F2", "F1"],
            "proxy_positive_probability": [0.8 + offset, 0.2 + offset],
            "raw_logit": [1.0 + offset, -1.0 + offset],
            "direction_positive_probability": [0.7, 0.3],
            "proxy_label": [1, 0],
            "label_class": ["strong_positive", "unlabeled"],
        }
    )


def test_accumulator_aligns_identity_and_computes_seed_moments() -> None:
    state = (None, None, None, None, None)
    for offset in (0.0, 0.05, 0.10):
        reference, total, square_total, raw_total, direction_total = state
        state = MODULE._align_and_accumulate(
            reference,
            _prediction(offset),
            total,
            square_total,
            raw_total,
            direction_total,
        )
    reference, total, square_total, raw_total, direction_total = state
    assert reference.candidate_id.tolist() == ["A", "B"]
    mean = total / 3
    assert np.allclose(mean, [0.25, 0.85])
    assert np.all(square_total / 3 - mean**2 >= -1e-12)
    assert np.all(np.isfinite(raw_total))
    assert np.allclose(direction_total / 3, [0.3, 0.7])


def test_accumulator_rejects_identity_drift() -> None:
    state = MODULE._align_and_accumulate(None, _prediction(), None, None, None, None)
    drift = _prediction()
    drift.loc[0, "pathway_id"] = "DRIFT"
    reference, total, square_total, raw_total, direction_total = state
    with pytest.raises(RuntimeError, match="drift across seeds"):
        MODULE._align_and_accumulate(
            reference,
            drift,
            total,
            square_total,
            raw_total,
            direction_total,
        )


def test_selection_forbids_test_driven_or_incomplete_choice(tmp_path: Path) -> None:
    path = tmp_path / "selection.json"
    valid = {
        "status": "PASS",
        "run_id": "RUN",
        "selected_model": "hgt",
        "selection_split": "val_only",
        "test_metrics_used": False,
        "seed_ensemble_size": 3,
        "folds": 33,
        "matrix_audit_release_eligible": True,
    }
    path.write_text(json.dumps(valid), encoding="utf-8")
    assert MODULE._selection(path, "RUN")["selected_model"] == "hgt"
    valid["test_metrics_used"] = True
    path.write_text(json.dumps(valid), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Invalid validation-only"):
        MODULE._selection(path, "RUN")
