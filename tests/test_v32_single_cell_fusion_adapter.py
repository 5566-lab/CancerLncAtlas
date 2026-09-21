from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cc_hhgt.v32.single_cell_fusion_adapter import (
    ANALYSIS_VERSION,
    SingleCellFusionAdapterError,
    build_single_cell_exact_fusion_expert,
)


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cancer_id": ["BRCA", "BRCA", "ACC"],
            "lncrna_id": ["LNC:1", "LNC:2", "LNC:3"],
            "pathway_id": ["P1", "P2", "P3"],
        }
    )


def _sparse() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cancer_id": ["BRCA", "ACC"],
            "lncrna_id": ["LNC:1", "LNC:3"],
            "pathway_id": ["P1", "P3"],
            "single_cell_replication_probability": [0.8, np.nan],
            "single_cell_available": [True, False],
            "single_cell_unavailable_reason": [pd.NA, "INSUFFICIENT_DONORS"],
            "analysis_version": [ANALYSIS_VERSION, ANALYSIS_VERSION],
        }
    )


def test_sparse_single_cell_is_typed_to_full_exact_universe() -> None:
    result = build_single_cell_exact_fusion_expert(_candidates(), _sparse())
    assert len(result) == 3
    assert not result[["cancer_id", "lncrna_id", "pathway_id"]].duplicated().any()
    available = result.loc[result.single_cell_available]
    assert available.single_cell_replication_probability.tolist() == [0.8]
    absent = result.loc[result.lncrna_id.eq("LNC:2")].iloc[0]
    assert not absent.single_cell_available
    assert pd.isna(absent.single_cell_replication_probability)
    assert absent.single_cell_unavailable_reason == "NO_SINGLE_CELL_EXACT_PAIR_MEASUREMENT"
    assert result.family_to_exact_broadcast.eq(False).all()
    assert result.changes_primary_ranking.eq(False).all()


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda frame: frame.assign(single_cell_replication_probability=0.7), "unavailable.*score"),
        (lambda frame: frame.assign(analysis_version="CancerLncAtlas_V3.1"), "not current"),
        (
            lambda frame: pd.concat(
                [
                    frame,
                    frame.iloc[[0]].assign(
                        cancer_id="UVM", lncrna_id="LNC:OUT", pathway_id="P:OUT"
                    ),
                ],
                ignore_index=True,
            ),
            "escape primary",
        ),
    ],
)
def test_adapter_fails_closed_on_invalid_sparse_rows(edit, message: str) -> None:
    with pytest.raises(SingleCellFusionAdapterError, match=message):
        build_single_cell_exact_fusion_expert(_candidates(), edit(_sparse()))
