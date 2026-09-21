from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT = Path(__file__).with_name("73_v31_full_universe_inference.py")
if not SCRIPT.exists():
    SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / SCRIPT.name
SPEC = importlib.util.spec_from_file_location("v31_full_universe_inference", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _formal_prediction() -> pd.DataFrame:
    logits = np.asarray([-1.0, 0.0, 1.0], dtype=float)
    calibrated = 1.0 / (1.0 + np.exp(-logits / 2.0))
    return pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c"],
            "split": ["test", "test", "test"],
            "raw_logit": logits,
            "proxy_positive_probability": calibrated,
        }
    )


def test_registered_calibration_and_overlap_pass(tmp_path: Path) -> None:
    calibration = {
        "method": "temperature",
        "task": "pathway",
        "independent_by_state": False,
        "groups": {"pathway": {"temperature": 2.0}},
    }
    (tmp_path / "calibration_pathway.json").write_text(
        json.dumps(calibration), encoding="utf-8"
    )
    formal = _formal_prediction()
    formal.to_parquet(tmp_path / "prediction_pathway_calibrated.parquet", index=False)
    full = pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c", "d"],
            "raw_logit": [-1.0, 0.0, 1.0, 2.0],
            "raw_probability": [0.0, 0.0, 0.0, 0.0],
        }
    )
    calibrated, temperature = MODULE._apply_registered_calibration(tmp_path, full)
    assert temperature == 2.0
    assert set(calibrated.prediction_scale) == {"calibrated_probability"}
    assert calibrated.proxy_positive_probability.between(0, 1).all()
    audit = MODULE._audit_overlap(tmp_path, calibrated)
    assert audit["status"] == "PASS"
    assert audit["formal_test_rows"] == 3
    assert audit["overlap_rows"] == 3
    assert audit["raw_logit_max_abs_delta"] == 0.0
    assert audit["calibrated_probability_max_abs_delta"] == 0.0


def test_overlap_rejects_probability_drift(tmp_path: Path) -> None:
    formal = _formal_prediction()
    formal.to_parquet(tmp_path / "prediction_pathway_calibrated.parquet", index=False)
    full = formal.drop(columns="split").copy()
    full.loc[0, "proxy_positive_probability"] += MODULE.OVERLAP_ATOL * 2
    with pytest.raises(RuntimeError, match="overlap mismatch"):
        MODULE._audit_overlap(tmp_path, full)


def test_task_success_rejects_hard_cap(tmp_path: Path) -> None:
    (tmp_path / "SUCCESS.json").write_text(
        json.dumps(
            {
                "status": "COMPLETED",
                "hit_hard_epoch_cap": True,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="Hard-cap"):
        MODULE._task_success(tmp_path)
