from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cc_hhgt.v32.epigenetic_expression_head import (
    EpigeneticExpressionConfig,
    EpigeneticExpressionError,
    crossfit_epigenetic_expression,
)


def _fixture(seed: int = 20260830) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for lnc_index, lnc in enumerate(("LNC:ENSG1", "LNC:ENSG2")):
        for patient_index in range(75):
            atac = rng.normal()
            mutation = float(rng.binomial(2, 0.16))
            cnv = rng.normal(scale=0.7)
            purity = rng.uniform(0.45, 0.95)
            methylation_available = patient_index % 4 != 0
            promoter_methylation = rng.beta(2, 5) if methylation_available else np.nan
            distal_methylation = rng.beta(2, 4) if methylation_available else np.nan
            expression = (
                1.55 * atac
                + 0.55 * mutation
                - 1.10 * (promoter_methylation if methylation_available else 2 / 7)
                + 0.80 * cnv
                + 0.35 * purity
                + rng.normal(scale=0.28)
                + lnc_index * 0.2
            )
            rows.append(
                {
                    "cancer_id": "BRCA",
                    "patient_id": f"TCGA-AA-{patient_index:04d}",
                    "lncrna_id": lnc,
                    "patient_fold_id": patient_index % 5,
                    "lncrna_expression": expression,
                    "atac_distal_accessibility": atac,
                    "atac_distal_accessibility__available": True,
                    "distal_mutation_burden": mutation,
                    "distal_mutation_burden__available": True,
                    "promoter_methylation_beta": promoter_methylation,
                    "promoter_methylation_beta__available": methylation_available,
                    "distal_methylation_beta": distal_methylation,
                    "distal_methylation_beta__available": methylation_available,
                    "local_cnv_log2": cnv,
                    "local_cnv_log2__available": True,
                    "tumor_purity": purity,
                    "tumor_purity__available": True,
                }
            )
    return pd.DataFrame(rows)


def test_crossfit_predicts_expression_and_preserves_typed_missingness() -> None:
    frame = _fixture()
    output, audit = crossfit_epigenetic_expression(
        frame,
        config=EpigeneticExpressionConfig(min_train_patients=30, ridge_alpha=0.5),
    )
    assert audit["status"] == "PASS"
    assert output.epigenetic_expression_available.all()
    observed_z = output.groupby("lncrna_id", observed=True).lncrna_expression.transform(
        lambda value: (value - value.mean()) / value.std(ddof=0)
    )
    correlation = np.corrcoef(observed_z, output.predicted_expression_z)[0, 1]
    assert correlation > 0.85
    assert np.isfinite(output.regulatory_expression_component_z).all()
    assert not audit["heldout_expression_used_for_own_prediction"]
    assert not audit["missing_assay_assumed_zero"]


def test_heldout_target_cannot_change_its_own_fold_predictions() -> None:
    frame = _fixture()
    config = EpigeneticExpressionConfig(min_train_patients=30)
    original, _ = crossfit_epigenetic_expression(frame, config=config)
    changed = frame.copy()
    heldout = changed.patient_fold_id.eq(2)
    changed.loc[heldout, "lncrna_expression"] += 10_000.0
    rerun, _ = crossfit_epigenetic_expression(changed, config=config)
    columns = ["predicted_expression_z", "regulatory_expression_component_z"]
    output_heldout = original.patient_fold_id.eq(2)
    np.testing.assert_allclose(
        original.loc[output_heldout, columns],
        rerun.loc[output_heldout, columns],
        rtol=0,
        atol=0,
    )


def test_no_signal_is_explicitly_unavailable_not_zero() -> None:
    frame = _fixture()
    for feature in (
        "atac_distal_accessibility",
        "distal_mutation_burden",
        "promoter_methylation_beta",
        "distal_methylation_beta",
    ):
        frame[feature] = np.nan
        frame[f"{feature}__available"] = False
    output, audit = crossfit_epigenetic_expression(frame)
    assert audit["status"] == "TYPED_UNAVAILABLE_NO_OOF_PREDICTIONS"
    assert not output.epigenetic_expression_available.any()
    assert output.predicted_expression_z.isna().all()
    assert output.regulatory_expression_component_z.isna().all()
    assert output.epigenetic_expression_unavailable_reason.notna().all()


def test_finite_value_with_false_availability_fails_closed() -> None:
    frame = _fixture()
    frame.loc[0, "atac_distal_accessibility__available"] = False
    with pytest.raises(EpigeneticExpressionError, match="typed-null availability"):
        crossfit_epigenetic_expression(frame)
