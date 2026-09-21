from __future__ import annotations

import numpy as np
import pandas as pd

from cc_hhgt.v32.hierarchical_candidate_preparation import _canonical_predictions


def test_modality_preparation_joins_shuffled_sources_by_exact_key() -> None:
    genomic = pd.DataFrame(
        {
            "cancer_id": ["coad", "brca"],
            "lncrna_id": ["L2", "L1"],
            "pathway_id": ["P2", "P1"],
            "mutation_context_probability": [0.8, 0.2],
            "mutation_available": [True, True],
            "cnv_context_probability": [np.nan, np.nan],
            "cnv_available": [False, False],
        }
    )
    atac = pd.DataFrame(
        {
            "cancer_id": ["BRCA", "COAD"],
            "lncrna_id": ["L1", "L2"],
            "pathway_id": ["P1", "P2"],
            "atac_context_probability": [0.1, 0.9],
            "atac_available": [True, True],
        }
    )
    result = _canonical_predictions(genomic, atac).reset_index()
    assert result.cancer_id.tolist() == ["BRCA", "COAD"]
    assert result.mutation_probability.tolist() == [0.2, 0.8]
    assert result.atac_probability.tolist() == [0.1, 0.9]
