from __future__ import annotations

import pandas as pd

from cc_hhgt.v32.full_model_contract import validate_public_module_frame


def test_interaction_audit_is_null_with_reason() -> None:
    frame = pd.DataFrame(
        {
            "lncrna_id": ["LNC:X"],
            "protein_id": ["P1"],
            "physical_interaction_probability": [pd.NA],
            "availability": [False],
            "failure_reason": ["STRICT_CONNECTED_COMPONENT_COUNT_1_LT_5"],
            "analysis_version": ["CancerLncAtlas_V3.2_FULL_MULTITASK"],
            "training_run_id": ["v32-test"],
        }
    )
    validate_public_module_frame("interaction", frame)
