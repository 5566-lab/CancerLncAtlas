from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from cc_hhgt.common import file_sha256


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "79_materialize_v31_exact_pathway_website.py"
)
SPEC = importlib.util.spec_from_file_location("v31_exact_website_materializer", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _success(root: Path, *, gate_sha256: str = "a" * 64) -> Path:
    prediction = root / "prediction_exact_pathway_three_seed_ensemble.parquet"
    pd.DataFrame(
        {
            "candidate_id": ["C"],
            "cancer_id": ["A"],
            "lncrna_id": ["L"],
            "pathway_id": ["P"],
            "pathway_family_id": ["F"],
            "calibrated_probability": [0.8],
        }
    ).to_parquet(prediction, index=False)
    path = root / "SUCCESS.json"
    path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "run_id": "RUN",
                "selected_model": "hgt",
                "pathway_target_level": "exact_pathway",
                "seed_count": 3,
                "heldout_label_columns_in_output": False,
                "formal_gate_sha256": gate_sha256,
                "prediction_file": "/original/training/host/prediction.parquet",
                "prediction_file_sha256": file_sha256(prediction),
            }
        ),
        encoding="utf-8",
    )
    return path


def test_read_success_rewrites_training_host_path_to_verified_local_copy(
    tmp_path: Path,
) -> None:
    path = _success(tmp_path)
    payload = MODULE._read_success(
        path,
        run_id="RUN",
        selected_model="hgt",
        training_lineage_gate_sha256="a" * 64,
    )
    assert Path(payload["prediction_file"]) == (
        tmp_path / "prediction_exact_pathway_three_seed_ensemble.parquet"
    )


def test_read_success_rejects_inference_from_another_training_gate(
    tmp_path: Path,
) -> None:
    path = _success(tmp_path, gate_sha256="b" * 64)
    with pytest.raises(RuntimeError, match="Invalid exact-pathway ensemble inference"):
        MODULE._read_success(
            path,
            run_id="RUN",
            selected_model="hgt",
            training_lineage_gate_sha256="a" * 64,
        )
