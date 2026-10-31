from __future__ import annotations

import json
from pathlib import Path

import pytest


pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from cla_engine.backend_check import run


def test_current_backend_check_is_self_contained_and_joint(tmp_path: Path):
    result = run(tmp_path, "cpu")
    assert result["state"] == "REAL_PYG_BACKEND_CHECK_PASS_NOT_TRAINED"
    assert result["backend_contract"] == "CURRENT_CC_HHGT_PYG_JOINT_CONTEXT_V1"
    assert result["historical_reference_loaded"] is False
    assert result["full_r1_kernel_equivalence_claimed"] is False
    assert "reference_gnn_sha256" not in result
    assert result["task_weights"] == {"pathway": 1.0, "state": 0.5}
    assert min(result["shared_membership_gradients"].values()) > 0.0
    assert result["global_representable_delta_guard"]["parameter_delta_positive"] is True
    assert all(
        row["forward_exact"] and row["gradient_exact"]
        for row in result["current_encoder_zero_context_equivalence"]
    )
    saved = json.loads((tmp_path / "BACKEND_TEST.json").read_text())
    assert saved == result


def test_backend_source_has_no_historical_reference_loader():
    source = Path(__file__).resolve().parents[1] / "cla_engine" / "backend_check.py"
    text = source.read_text(encoding="utf-8")
    assert "gnn_full_r1.py" not in text
    assert "importlib" not in text
    assert "spec_from_file_location" not in text
