from __future__ import annotations

import numpy as np
import pandas as pd

from cc_hhgt.v30_state_assets import _association


def test_state_association_uses_only_outcome_complete_cases(tmp_path) -> None:
    cfg = {
        "_root": tmp_path,
        "input_root": tmp_path,
        "inputs": {"bulk_covariates": None},
        "sample_contract": {"minimum_observed_per_cancer_state": 3},
        "state_graph": {"covariates": []},
    }
    features = pd.DataFrame(
        {"L1": [1.0, 2.0, 3.0, 4.0], "L2": [4.0, 1.0, 3.0, 2.0]},
        index=["S1", "S2", "S3", "S4"],
    )
    # S2 has no outcome row. It must not be median-imputed into the statistic.
    complete_cases = pd.DataFrame(
        {
            "cancer_id": ["BRCA", "BRCA", "BRCA"],
            "sample_id": ["S1", "S3", "S4"],
            "state_id": ["stemness_rna::RNAss"] * 3,
            "state_value": [0.1, 0.5, 0.9],
        }
    )
    table, audit = _association(
        cfg,
        "BRCA",
        "stemness_rna::RNAss",
        features,
        complete_cases,
        canonical_count=4,
    )
    assert audit["n_observed"] == 3
    assert audit["n_missing"] == 1
    assert audit["residual_df"] == 1
    assert audit["fdr_family_size"] == len(table) == 2
    assert np.isfinite(table.effect).all()
    assert np.isfinite(table.p_value).all()


def test_state_association_emits_unavailable_instead_of_silent_rows(tmp_path) -> None:
    cfg = {
        "_root": tmp_path,
        "input_root": tmp_path,
        "inputs": {"bulk_covariates": None},
        "sample_contract": {"minimum_observed_per_cancer_state": 4},
        "state_graph": {"covariates": []},
    }
    features = pd.DataFrame({"L1": [1.0, 2.0, 3.0]}, index=["S1", "S2", "S3"])
    cases = pd.DataFrame(
        {
            "cancer_id": ["OV"] * 3,
            "sample_id": ["S1", "S2", "S3"],
            "state_id": ["stemness_dna::DNAss"] * 3,
            "state_value": [0.1, 0.2, 0.3],
        }
    )
    table, audit = _association(cfg, "OV", "stemness_dna::DNAss", features, cases, 5)
    assert table.empty
    assert audit["eligibility"] == "UNAVAILABLE"
    assert audit["unavailable_reason"] == "N_OBSERVED_LT_4"
    assert audit["n_observed"] == 3
    assert audit["n_missing"] == 2
