from __future__ import annotations

import numpy as np
import pandas as pd

from cc_hhgt.v32.distal_regulatory_mutation_head import (
    DistalRegulatoryMutationHeadConfig,
    DistalRegulatoryMutationHeadError,
    crossfit_distal_regulatory_pathway_head,
)
from cc_hhgt.v32.epigenetic_expression_head import (
    EpigeneticExpressionConfig,
    nested_outer_crossfit_epigenetic_expression,
)


def _frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    component_rows = []
    activity_rows = []
    for patient in range(40):
        patient_id = f"TCGA-AA-{patient:04d}"
        fold = patient % 5
        x = (patient - 19.5) / 10.0
        component_rows.append(
            {
                "cancer_id": "BRCA",
                "patient_id": patient_id,
                "lncrna_id": "L1",
                "patient_fold_id": fold,
                "mutation_regulatory_expression_component_z": x,
                "mutation_regulatory_expression_component_available": True,
            }
        )
        activity_rows.extend(
            [
                {
                    "cancer_id": "BRCA",
                    "patient_id": patient_id,
                    "pathway_id": "P1",
                    "activity_score": 1.5 * x + 0.05 * np.sin(patient),
                },
                {
                    "cancer_id": "BRCA",
                    "patient_id": patient_id,
                    "pathway_id": "P2",
                    "activity_score": np.cos(patient * 1.7),
                },
            ]
        )
    candidates = pd.DataFrame(
        [
            {"cancer_id": "BRCA", "lncrna_id": "L1", "pathway_id": "P1"},
            {"cancer_id": "BRCA", "lncrna_id": "L1", "pathway_id": "P2"},
            {"cancer_id": "BRCA", "lncrna_id": "MISSING", "pathway_id": "P1"},
        ]
    )
    return pd.DataFrame(component_rows), pd.DataFrame(activity_rows), candidates


def test_independent_pathway_head_is_patient_oof_and_typed() -> None:
    components, activity, candidates = _frames()
    config = DistalRegulatoryMutationHeadConfig(
        min_train_patients=15,
        min_heldout_patients=3,
        min_oof_folds=5,
        candidate_batch_size=2,
    )
    output, audit = crossfit_distal_regulatory_pathway_head(
        components, activity, candidates, config=config
    )
    supported = output.loc[output.pathway_id.eq("P1") & output.lncrna_id.eq("L1")].iloc[0]
    missing = output.loc[output.lncrna_id.eq("MISSING")].iloc[0]
    assert bool(supported.distal_regulatory_mutation_pathway_available)
    assert supported.oof_prediction_correlation > 0.99
    assert supported.oof_delta_mse > 0
    assert not bool(missing.distal_regulatory_mutation_pathway_available)
    assert missing.distal_regulatory_mutation_pathway_unavailable_reason
    assert audit["heldout_pathway_activity_used_for_own_prediction_fit"] is False
    assert audit["causal_claimed"] is False


def test_heldout_pathway_change_affects_metric_not_its_fold_fit() -> None:
    components, activity, candidates = _frames()
    config = DistalRegulatoryMutationHeadConfig(
        min_train_patients=15,
        min_heldout_patients=3,
        min_oof_folds=5,
    )
    original, _ = crossfit_distal_regulatory_pathway_head(
        components, activity, candidates.iloc[[0]], config=config
    )
    changed = activity.copy()
    mask = changed.patient_id.str[-4:].astype(int).mod(5).eq(0) & changed.pathway_id.eq("P1")
    changed.loc[mask, "activity_score"] += 100.0
    repeated, audit = crossfit_distal_regulatory_pathway_head(
        components, changed, candidates.iloc[[0]], config=config
    )
    # The held-out observations properly change OOF evaluation.  The audit
    # contract records that those values never entered their own fold fit.
    assert repeated.oof_rmse.iloc[0] > original.oof_rmse.iloc[0]
    assert audit["heldout_pathway_activity_used_for_own_prediction_fit"] is False


def test_component_typed_null_contract_fails_closed() -> None:
    components, activity, candidates = _frames()
    components.loc[0, "mutation_regulatory_expression_component_z"] = np.nan
    try:
        crossfit_distal_regulatory_pathway_head(components, activity, candidates)
    except DistalRegulatoryMutationHeadError as exc:
        assert "typed-null" in str(exc)
    else:
        raise AssertionError("Typed-null drift was not rejected")


def test_nested_head_consumes_the_matching_outer_fold_component() -> None:
    components, activity, candidates = _frames()
    nested = pd.concat(
        [components.assign(outer_fold_id=fold) for fold in range(5)],
        ignore_index=True,
    )
    output, audit = crossfit_distal_regulatory_pathway_head(
        nested,
        activity,
        candidates.iloc[[0]],
        config=DistalRegulatoryMutationHeadConfig(
            min_train_patients=15,
            min_heldout_patients=3,
            min_oof_folds=5,
        ),
    )
    assert bool(output.distal_regulatory_mutation_pathway_available.iloc[0])
    assert audit["nested_outer_crossfit"] is True
    assert audit["outer_training_components_are_inner_oof"] is True


def test_nested_expression_excludes_outer_heldout_expression_from_every_fit() -> None:
    rows = []
    for patient in range(50):
        x = (patient - 24.5) / 10.0
        rows.append(
            {
                "cancer_id": "BRCA",
                "patient_id": f"TCGA-AA-{patient:04d}",
                "lncrna_id": "L1",
                "patient_fold_id": patient % 5,
                "lncrna_expression": 1.7 * x + 0.1 * np.sin(patient),
                "distal_mutation_burden": x,
                "distal_mutation_burden__available": True,
            }
        )
    frame = pd.DataFrame(rows)
    config = EpigeneticExpressionConfig(
        signal_features=("distal_mutation_burden",),
        nuisance_features=(),
        min_train_patients=10,
    )
    _, original, audit = nested_outer_crossfit_epigenetic_expression(
        frame, config=config
    )
    changed = frame.copy()
    changed.loc[changed.patient_fold_id.eq(0), "lncrna_expression"] += 1000.0
    _, repeated, _ = nested_outer_crossfit_epigenetic_expression(
        changed, config=config
    )
    columns = [
        "patient_id",
        "lncrna_id",
        "regulatory_expression_component_z",
        "epigenetic_expression_available",
    ]
    left = original.loc[original.outer_fold_id.eq(0), columns].sort_values(
        ["patient_id", "lncrna_id"]
    )
    right = repeated.loc[repeated.outer_fold_id.eq(0), columns].sort_values(
        ["patient_id", "lncrna_id"]
    )
    assert np.allclose(
        left.regulatory_expression_component_z.to_numpy(float),
        right.regulatory_expression_component_z.to_numpy(float),
        equal_nan=True,
    )
    assert np.array_equal(
        left.epigenetic_expression_available.to_numpy(bool),
        right.epigenetic_expression_available.to_numpy(bool),
    )
    assert audit["outer_heldout_expression_used_for_any_component_fit"] is False
