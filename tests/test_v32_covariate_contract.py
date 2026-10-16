import numpy as np
import pandas as pd
import pytest

from cc_hhgt.stats import prepare_design
from cc_hhgt.v32 import bulk_coexpression_release as bulk
from cc_hhgt.v32.bulk_coexpression_release import DEFAULT_COVARIATES
from scripts.run_v32_local_cnv_confounding_audit_r2 import a0_full


def _covariates(n=40):
    patients = [f"P{i:03d}" for i in range(n)]
    frame = pd.DataFrame({"cancer_id": "BRCA", "patient_id": patients})
    frame["purity"] = np.linspace(0.15, 0.85, n)
    frame["age_years"] = np.linspace(30, 80, n)
    frame["sex"] = np.where(np.arange(n) % 2, "M", "F")
    frame["stage"] = np.where(np.arange(n) % 3, "II", "III")
    frame["molecular_subtype"] = np.where(np.arange(n) % 2, "B", "A")
    frame["clinical_subtype"] = np.where(np.arange(n) % 2, "Y", "X")
    frame["technical_batch"] = [f"batch_{i % 25:02d}" for i in range(n)]
    frame["leukocyte_fraction"] = np.linspace(0.02, 0.30, n)
    for i, column in enumerate(
        (
            "cell_fraction_Macrophages.M0",
            "cell_fraction_Macrophages.M1",
            "cell_fraction_Macrophages.M2",
        )
    ):
        frame[column] = np.linspace(0.01 + i * 0.01, 0.20 + i * 0.01, n)
    return frame


def test_declared_covariates_are_strict_and_high_cardinality_is_retained():
    frame = _covariates()
    patients = frame.patient_id.tolist()
    design, used, missing = bulk._covariate_design(
        frame, cancer="BRCA", patients=patients, columns=DEFAULT_COVARIATES
    )
    assert used == list(DEFAULT_COVARIATES)
    assert missing == 0
    assert np.isfinite(design).all()
    # 25 technical batches produce 24 one-hot columns when no implicit cap is
    # applied; the full declared contract must therefore materially enter.
    assert design.shape[1] >= 25
    assert np.linalg.matrix_rank(design) > 1

    with pytest.raises(RuntimeError, match="covariate contract"):
        bulk._covariate_design(
            frame.drop(columns="purity"),
            cancer="BRCA",
            patients=patients,
            columns=DEFAULT_COVARIATES,
        )


def test_prepare_design_none_disables_legacy_category_drop():
    n = 40
    frame = pd.DataFrame({"batch": [f"b{i % 25}" for i in range(n)]})
    capped = prepare_design(frame)
    uncapped = prepare_design(frame, max_categories=None)
    assert capped.shape[1] == 1  # intercept only under the legacy guard
    assert uncapped.shape[1] == 25  # intercept + 24 batch indicators


def test_shared_covariates_change_formal_association_and_rank():
    covariates = _covariates()
    # Keep this association example away from the high-cardinality stress case
    # above so residual degrees of freedom are comfortably positive.
    covariates["technical_batch"] = "batch_00"
    patients = covariates.patient_id.tolist()
    purity = covariates.purity.to_numpy(float)
    t = np.arange(len(patients), dtype=float)
    rng = np.random.default_rng(42)
    lnc = pd.DataFrame({"L1": purity + 0.25 * rng.normal(size=len(t))}, index=patients)
    pathway = pd.DataFrame({"P1": purity + 0.25 * rng.normal(size=len(t))}, index=patients)
    pairs = pd.DataFrame(
        {"cancer_id": ["BRCA"], "lncrna_id": ["L1"], "pathway_id": ["P1"]}
    )
    rho_shared, _, _, rank_shared = a0_full(
        lnc, pathway, pairs, patients, covariates=covariates, cancer="BRCA"
    )
    shuffled = covariates.copy()
    shuffled["purity"] = shuffled.purity.sample(frac=1.0, random_state=17).to_numpy()
    rho_shuffled, _, _, rank_shuffled = a0_full(
        lnc, pathway, pairs, patients, covariates=shuffled, cancer="BRCA"
    )
    assert rank_shared > 1
    assert rank_shuffled > 1
    assert not np.isclose(rho_shared[0], rho_shuffled[0])
