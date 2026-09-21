from __future__ import annotations

import pandas as pd
import pytest

from cc_hhgt.v32.multimodal_fusion import FUSION_FOLD_COLUMN
from cc_hhgt.v32.routed_fair_comparison import (
    RoutedComparisonError,
    compare_routed_candidates,
    validate_equal_contracts,
)


def _contract(seed=7):
    return {
        "candidate_universe_sha256": "a" * 64,
        "patient_modality_oof_sha256": "b" * 64,
        "outer_pair_fold_sha256": "c" * 64,
        "seed": seed,
        "training_budget_id": "equal-budget-v1",
    }


def test_fair_comparison_rejects_contract_drift_and_reports_regressions() -> None:
    assert validate_equal_contracts(_contract(), _contract()).seed == 7
    with pytest.raises(RoutedComparisonError, match="not fair"):
        validate_equal_contracts(_contract(), _contract(seed=8))
    primary_rows = []
    for fold in range(5):
        for index in range(4):
            primary_rows.append(
                {
                    "cancer_id": "BRCA",
                    "lncrna_id": f"L{fold}_{index}",
                    "pathway_id": f"P{index}",
                    "fusion_target": float(index % 2 == 0),
                    "primary_probability": 0.5,
                    FUSION_FOLD_COLUMN: fold,
                }
            )
    primary = pd.DataFrame(primary_rows)
    external = primary[["cancer_id", "lncrna_id", "pathway_id", FUSION_FOLD_COLUMN]].copy()
    external["fusion_target"] = primary.fusion_target
    external["primary_probability"] = primary.primary_probability
    external["discovery_adjusted_probability"] = primary.fusion_target * 0.8 + 0.1
    hierarchical = external[["cancer_id", "lncrna_id", "pathway_id", FUSION_FOLD_COLUMN]].copy()
    hierarchical["fusion_target"] = primary.fusion_target
    hierarchical["primary_probability"] = primary.primary_probability
    hierarchical["hierarchical_probability"] = 0.5
    for frame in (external, hierarchical):
        frame["mutation_probability"] = primary.fusion_target * 0.8 + 0.1
        frame["mutation_available"] = True
        for modality in ("cnv", "atac"):
            frame[f"{modality}_probability"] = float("nan")
            frame[f"{modality}_available"] = False
    metrics = compare_routed_candidates(primary, external, hierarchical)
    overall = metrics.loc[metrics.cancer_id.eq("ALL_CANCERS")].set_index("method")
    assert overall.loc["external_router", "delta_logloss_vs_primary"] > 0
    assert overall.loc["hierarchical_end_to_end", "delta_logloss_vs_primary"] == pytest.approx(0)

    drifted = hierarchical.copy()
    drifted.loc[0, "fusion_target"] = 1.0 - drifted.loc[0, "fusion_target"]
    with pytest.raises(RoutedComparisonError, match="labels differ"):
        compare_routed_candidates(primary, external, drifted)

    drifted = hierarchical.copy()
    drifted.loc[0, "mutation_available"] = False
    drifted.loc[0, "mutation_probability"] = float("nan")
    with pytest.raises(RoutedComparisonError, match="availability/callability differs"):
        compare_routed_candidates(primary, external, drifted)
