from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cc_hhgt.v32.exact_release import ensemble_fold_predictions
from cc_hhgt.v32.full_model_contract import validate_public_module_frame


def test_exact_release_averages_five_new_folds_and_drops_labels(tmp_path: Path) -> None:
    paths = []
    for fold in range(5):
        frame = pd.DataFrame(
            {
                "cancer_id": ["A"],
                "lncrna_id": ["LNC:X"],
                "pathway_id": ["P1"],
                "pathway_family_id": ["F1"],
                "shared_or_local_scope": ["shared"],
                "patient_fold_id": [fold],
                "association_membership_probability": [0.1 * (fold + 1)],
                "association_direction_probability": [0.7],
                "l1_probability": [0.2],
                "ridge_probability": [0.3],
                "graph_residual": [0.0],
                "graph_gate": [0.5],
                "held_out_proxy_label": [1],
            }
        )
        path = tmp_path / f"fold_{fold}.parquet"
        frame.to_parquet(path, index=False)
        paths.append(path)
    result = ensemble_fold_predictions(paths, run_id="v32-test")
    assert np.isclose(result.association_membership_probability.iloc[0], 0.3)
    assert "held_out_proxy_label" not in result
    validate_public_module_frame("exact_pathway", result)
