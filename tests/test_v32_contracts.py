from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from cc_hhgt.v32.contracts import (
    PREDICTION_COLUMNS,
    assert_matched_candidate_universe,
    default_contract,
    validate_contract,
    validate_prediction_frame,
)


def test_default_contract_is_code_only_exact_pathway() -> None:
    contract = default_contract()
    validate_contract(contract)
    assert contract["execution_control"]["training_authorized"] is False
    assert contract["execution_control"]["max_cost_cny"] == 0


def test_family_target_fails_closed() -> None:
    contract = copy.deepcopy(default_contract())
    contract["task_contract"]["target_level"] = "pathway_family"
    contract["task_contract"]["target_column"] = "pathway_family_id"
    with pytest.raises(RuntimeError, match="exact_pathway"):
        validate_contract(contract)


def test_prediction_schema_and_matched_candidates() -> None:
    row = {
        "cancer_id": "BRCA",
        "lncrna_id": "L1",
        "pathway_id": "P1",
        "pathway_family_id": "F1",
        "association_membership_probability": 0.8,
        "association_direction": "positive",
        "l1_probability": 0.7,
        "ridge_probability": 0.75,
        "graph_residual": 0.1,
        "graph_gate": 0.5,
        "fold_rank_percentile": 0.9,
        "fold_selection_frequency": 0.8,
        "shared_or_local_scope": "shared",
        "regulatory_evidence_confidence": 0.2,
    }
    frame = pd.DataFrame([row], columns=PREDICTION_COLUMNS)
    validate_prediction_frame(frame)
    assert len(assert_matched_candidate_universe(lasso=frame, ridge=frame.copy(), cc=frame.copy())) == 64
    bad = frame.copy()
    bad["association_membership_probability"] = np.nan
    with pytest.raises(RuntimeError):
        validate_prediction_frame(bad)
