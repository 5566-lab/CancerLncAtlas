from __future__ import annotations

import numpy as np
import pandas as pd

from cc_hhgt.v31_graph_residual_stage import (
    expected_graph_residual_matrix,
    lambda_token,
    load_fold_residual_base,
)


def test_graph_residual_matrix_is_exact_three_cancer_three_seed_grid() -> None:
    matrix = expected_graph_residual_matrix([0.001, 0.01, 0.1])
    assert len(matrix) == 27
    assert {cancer for cancer, _, _ in matrix} == {"BRCA", "COAD", "KIRP"}
    assert {seed for _, seed, _ in matrix} == {20260726, 20261726, 20262726}
    assert {value for _, _, value in matrix} == {0.001, 0.01, 0.1}


def test_lambda_token_is_path_stable() -> None:
    assert lambda_token(0.001) == "0p001"
    assert lambda_token(0.1) == "0p1"


def test_load_fold_residual_base_preserves_exact_splits(tmp_path) -> None:
    rows = []
    for task in ("pathway", "state"):
        for split, scope in (
            ("train", "cancer_crossfit_OOF"),
            ("val", "val"),
            ("test", "test"),
        ):
            rows.append(
                {
                    "candidate_id": f"{task}-{split}",
                    "cancer_id": "A" if split == "train" else "B",
                    "task": task,
                    "split": split,
                    "loco_cancer": "BRCA",
                    "best_simple_score": 0.5,
                    "z_base": 0.0,
                    "metric_scope": scope,
                    "simple_base_fit_cancers": "A|C",
                }
            )
    path = tmp_path / "base.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    loaded = load_fold_residual_base(path, heldout_cancer="BRCA")
    assert set(loaded) == {"pathway", "state"}
    assert set(loaded["pathway"]) == {"train", "val", "test"}
    assert np.isclose(loaded["state"]["test"].z_base.iloc[0], 0.0)
