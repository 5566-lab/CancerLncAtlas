from __future__ import annotations

import numpy as np
import pandas as pd

from cc_hhgt.v32.directional_cnv import (
    aggregate_pathway_directional,
    map_segments_directional,
)


def test_direction_is_preserved_and_unavailable_is_nan() -> None:
    segments = pd.DataFrame({
        "chromosome": ["1", "1", "2"],
        "start": [0, 100, 0],
        "end": [99, 199, 100],
        "value": [0.5, -0.6, 0.1],
    })
    intervals = pd.DataFrame({
        "entity_id": ["amp", "del", "neutral", "missing"],
        "chromosome": ["1", "1", "2", "3"],
        "start": [10, 110, 10, 1],
        "end": [20, 120, 20, 2],
    })
    result = map_segments_directional(
        segments, intervals, ["amp", "del", "neutral", "missing"]
    )
    assert np.allclose(result.signed_mean[:3], [0.5, -0.6, 0.1])
    assert np.isnan(result.signed_mean[3])
    assert result.callable.tolist() == [True, True, True, False]
    assert result.amplification.tolist() == [True, False, False, False]
    assert result.deletion.tolist() == [False, True, False, False]
    assert result.neutral.tolist() == [False, False, True, False]


def test_pathway_aggregation_keeps_amp_and_deletion_separate() -> None:
    segments = pd.DataFrame({
        "chromosome": ["1", "1", "1"],
        "start": [0, 100, 200],
        "end": [99, 199, 299],
        "value": [0.5, -0.4, 0.1],
    })
    intervals = pd.DataFrame({
        "entity_id": ["g1", "g2", "g3"],
        "chromosome": ["1", "1", "1"],
        "start": [10, 110, 210],
        "end": [20, 120, 220],
    })
    genes = map_segments_directional(segments, intervals, ["g1", "g2", "g3"])
    pathways = aggregate_pathway_directional(
        genes, [np.array([0, 1]), np.array([2])],
        canonical_callable=np.array([True, False]),
    )
    assert np.isclose(pathways["signed_mean"][0], 0.05)
    assert pathways["amplification"][0]
    assert pathways["deletion"][0]
    assert np.isnan(pathways["signed_mean"][1])
    assert not pathways["callable"][1]
