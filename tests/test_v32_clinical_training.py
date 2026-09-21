from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cc_hhgt.v32.clinical_training import (
    CLINICAL_ENDPOINTS,
    ClinicalTrainingError,
    discrete_time_nll,
    patient_fold_manifest,
    project_measurements_to_core,
    standardize_tcga_cdr_workbook,
    survival_risk_probability,
    train_private_clinical_head,
)


def _workbook(path) -> None:
    main = pd.DataFrame(
        {
            "bcr_patient_barcode": ["TCGA-AA-0001", "TCGA-AA-0002"],
            "type": ["BRCA", "BRCA"],
            "OS": [1, 0],
            "OS.time": [100, 200],
            "DSS": [1, 0],
            "DSS.time": [90, 200],
            "PFI": [1, 0],
            "PFI.time": [80, 180],
            "DFI": [1, np.nan],
            "DFI.time": [70, np.nan],
            "age_at_initial_pathologic_diagnosis": [50, 60],
            "gender": ["FEMALE", "MALE"],
            "ajcc_pathologic_tumor_stage": ["Stage II", "Stage III"],
            "clinical_stage": [np.nan, np.nan],
        }
    )
    extra = pd.DataFrame(
        {
            "bcr_patient_barcode": ["TCGA-AA-0001", "TCGA-AA-0002"],
            "type": ["BRCA", "BRCA"],
            "PFS": [1, 0],
            "PFS.time": [80, 180],
        }
    )
    with pd.ExcelWriter(path) as writer:
        main.to_excel(writer, sheet_name="TCGA-CDR", index=False)
        extra.to_excel(writer, sheet_name="ExtraEndpoints", index=False)


def test_tcga_cdr_keeps_dfi_and_does_not_invent_dfs(tmp_path) -> None:
    workbook = tmp_path / "TCGA-CDR.xlsx"
    _workbook(workbook)
    endpoints, covariates = standardize_tcga_cdr_workbook(workbook)
    assert set(endpoints.clinical_endpoint) == set(CLINICAL_ENDPOINTS)
    dfi = endpoints.loc[endpoints.clinical_endpoint.eq("DFI")]
    dfs = endpoints.loc[endpoints.clinical_endpoint.eq("DFS")]
    assert dfi.endpoint_available.any()
    assert not dfs.endpoint_available.any()
    assert dfs.time_days.isna().all() and dfs.event.isna().all()
    assert set(dfs.failure_reason) == {"NO_DISTINCT_DFS_SOURCE"}
    assert len(covariates) == 2


def test_patient_fold_manifest_rejects_patient_leakage() -> None:
    frame = pd.DataFrame(
        {
            "cancer_id": ["BRCA", "BRCA"],
            "sample_id": ["TCGA-AA-0001-01A", "TCGA-AA-0001-02A"],
            "patient_id": ["TCGA-AA-0001", "TCGA-AA-0001"],
            "patient_fold_id": [0, 1],
        }
    )
    with pytest.raises(ClinicalTrainingError, match="multiple"):
        patient_fold_manifest(frame)


def test_projection_scaler_is_fit_on_training_patients_only() -> None:
    measurements = pd.DataFrame(
        {
            "patient_id": ["A", "A", "B", "B", "C", "C"],
            "lncrna_id": ["L1", "L2"] * 3,
            "logcpm": [1.0, 2.0, 2.0, 4.0, 100.0, 200.0],
        }
    )
    embeddings = pd.DataFrame(
        {
            "node_id": ["L1", "L2"],
            "core_feature_000": [1.0, 0.0],
            "core_feature_001": [0.0, 1.0],
        }
    )
    first = project_measurements_to_core(
        measurements,
        embeddings,
        entity_column="lncrna_id",
        value_column="logcpm",
        train_patient_ids=["A", "B"],
        output_prefix="core_lnc",
    )
    changed = measurements.copy()
    changed.loc[changed.patient_id.eq("C"), "logcpm"] *= 1000
    second = project_measurements_to_core(
        changed,
        embeddings,
        entity_column="lncrna_id",
        value_column="logcpm",
        train_patient_ids=["A", "B"],
        output_prefix="core_lnc",
    )
    columns = ["core_lnc_000", "core_lnc_001"]
    assert np.allclose(
        first.loc[first.patient_id.isin(["A", "B"]), columns],
        second.loc[second.patient_id.isin(["A", "B"]), columns],
    )


def test_discrete_survival_loss_and_risk_are_valid() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.zeros((3, 4), requires_grad=True)
    loss = discrete_time_nll(
        logits,
        torch.tensor([0, 2, 3]),
        torch.tensor([1.0, 0.0, 1.0]),
        torch.tensor([1.0, 1.0, 0.0]),
    )
    assert torch.isfinite(loss)
    loss.backward()
    risk = survival_risk_probability(np.zeros((3, 4)))
    assert np.all((risk >= 0) & (risk <= 1))


def test_clinical_private_head_is_freshly_trained() -> None:
    pytest.importorskip("torch")
    rng = np.random.default_rng(4)
    rows = 48
    core = rng.normal(size=(rows, 4)).astype("float32")
    domain = rng.normal(size=(rows, 2)).astype("float32")
    arrays = {
        "OS": {
            "bin": rng.integers(0, 3, size=rows),
            "event": rng.integers(0, 2, size=rows).astype("float32"),
            "available": np.ones(rows, dtype="float32"),
        }
    }
    head, metadata, history = train_private_clinical_head(
        core,
        domain,
        arrays,
        train_indices=np.arange(0, 32),
        validation_indices=np.arange(32, 48),
        n_bins=3,
        seed=11,
        epochs=2,
        patience=2,
        batch_size=16,
    )
    assert head is not None and history
    assert metadata["module_id"] == "clinical"
    assert metadata["source_checkpoint_sha256"] is None
    assert metadata["initialization_policy"].endswith("FROM_SCRATCH")
