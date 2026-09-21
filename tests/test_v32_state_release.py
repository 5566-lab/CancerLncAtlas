from __future__ import annotations

import numpy as np
import pandas as pd

from cc_hhgt.v32.state_release import complete_state_universe
from cc_hhgt.v32.state_training import HISTORICAL_STATE_IDS


def test_complete_state_universe_uses_null_not_zero_for_unmeasured_pairs() -> None:
    release = pd.DataFrame(
        [
            {
                "cancer_id": "A",
                "lncrna_id": "L1",
                "state_id": HISTORICAL_STATE_IDS[0],
                "state_membership_probability": 0.8,
                "state_effect": 0.4,
                "association_direction": "positive",
                "availability": True,
                "availability_reason": pd.NA,
                "folds_available": 5,
                "folds_expected": 5,
                "model_version": "V3.2",
                "probability_source": "fresh",
                "effect_source": "fresh",
            }
        ]
    )
    result = complete_state_universe(
        release, cancers=["A", "B"], lncrnas=["L1", "L2"], training_run_id="v32-test"
    )
    assert len(result) == 2 * 2 * len(HISTORICAL_STATE_IDS)
    missing = result.loc[
        result.cancer_id.eq("B") & result.lncrna_id.eq("L2")
    ]
    assert not missing.availability.any()
    assert missing.state_membership_probability.isna().all()
    assert missing.availability_reason.eq(
        "LNCRNA_EXPRESSION_UNAVAILABLE_FOR_CANCER"
    ).all()
    observed = result.loc[
        result.cancer_id.eq("A")
        & result.lncrna_id.eq("L1")
        & result.state_id.eq(HISTORICAL_STATE_IDS[0])
    ].iloc[0]
    assert np.isclose(observed.state_membership_probability, 0.8)
