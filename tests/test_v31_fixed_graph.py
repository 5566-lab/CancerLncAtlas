from __future__ import annotations

import json

import numpy as np
import pandas as pd

from cc_hhgt.v31_fixed_graph import _read_task_predictions, expected_b1_matrix


def _write_task(root) -> None:
    root.mkdir(parents=True)
    (root / "SUCCESS.json").write_text(
        json.dumps(
            {
                "status": "COMPLETED",
                "contract": "CONTRACT-T",
                "test_cancer": "BRCA",
                "seed": 20260726,
            }
        ),
        encoding="utf-8",
    )
    (root / "TRAINING_SUCCESS.json").write_text(
        json.dumps({"status": "COMPLETED", "graph_contract": "CONTRACT-T", "residual_learning": False}),
        encoding="utf-8",
    )
    pathway = pd.DataFrame(
        {
            # The first two probabilities are below the numerical clipping
            # threshold used by calibration metrics.  They must remain
            # distinct for AUROC/AUPRC so a monotone temperature transform
            # cannot spuriously appear to change ranking.
            "candidate_id": ["p1", "p2", "p3", "p4"],
            "cancer_id": ["BRCA"] * 4,
            "lncrna_id": ["L1", "L2", "L3", "L4"],
            "pathway_family_id": ["P"] * 4,
            "proxy_label": [0, 1, 0, 1],
            "split": ["test"] * 4,
            "raw_logit": [-20.0, -18.0, -16.0, -14.0],
            "proxy_positive_probability": 1.0
            / (1.0 + np.exp(-np.asarray([-20.0, -18.0, -16.0, -14.0]))),
            "prediction_scale": ["raw_probability"] * 4,
        }
    )
    state = pd.DataFrame(
        {
            "candidate_id": ["s1", "s2", "s3", "s4"],
            "cancer_id": ["BRCA"] * 4,
            "lncrna_id": ["L1", "L2", "L1", "L2"],
            "state_id": [
                "stemness_rna::RNAss",
                "stemness_rna::RNAss",
                "stemness_dna::DNAss",
                "stemness_dna::DNAss",
            ],
            "proxy_label": [0, 1, 1, 0],
            "split": ["test"] * 4,
            "raw_logit": [-2.0, 2.0, 1.0, -1.0],
            "proxy_positive_probability": [0.1192029220, 0.8807970780, 0.7310585786, 0.2689414214],
            "prediction_scale": ["raw_probability"] * 4,
        }
    )
    for task, frame in (("pathway", pathway), ("state", state)):
        frame.to_parquet(root / f"prediction_{task}_raw.parquet", index=False)
        calibrated = frame.copy()
        calibrated["proxy_positive_probability"] = 1.0 / (
            1.0 + np.exp(-calibrated.raw_logit.to_numpy(float) / 2.0)
        )
        calibrated["prediction_scale"] = "calibrated_probability"
        calibrated.to_parquet(root / f"prediction_{task}_calibrated.parquet", index=False)
        (root / f"calibration_{task}.json").write_text(
            json.dumps({"method": "temperature", "temperature": 2.0}), encoding="utf-8"
        )


def test_fixed_graph_task_collection_preserves_contract_and_scale(tmp_path) -> None:
    task_root = tmp_path / "seed"
    _write_task(task_root)
    predictions, audit = _read_task_predictions(
        task_root, "CONTRACT-T", "BRCA", 20260726
    )
    assert len(predictions) == 16
    assert set(predictions.prediction_scale) == {
        "raw_probability",
        "calibrated_probability",
    }
    assert set(predictions.target_subtype) == {"Pathway", "RNAss", "DNAss"}
    assert all(row["status"] == "PASS" for row in audit)


def test_b1_matrix_is_exact_and_keeps_contracts_separate() -> None:
    matrix = expected_b1_matrix()
    assert len(matrix) == 12
    assert sum(contract == "CONTRACT-T" for contract, _, _ in matrix) == 9
    assert sum(contract == "CONTRACT-S" for contract, _, _ in matrix) == 3
