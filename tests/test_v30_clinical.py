from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from cc_hhgt.clinical_v30 import (
    clinical_replication_label,
    discrete_time_nll,
    geometric_priority,
    harrell_c_index,
    standardise_endpoint_table,
)


def test_standardise_os_from_tcga_fields():
    source = pd.DataFrame({
        "cancer_id": ["LUAD", "LUAD", "BRCA"],
        "sample_id": ["TCGA-AA-0001-01A", "TCGA-AA-0002-01A", "TCGA-BB-0003-01A"],
        "days_to_death": [100.0, np.nan, 400.0],
        "days_to_last_follow_up": [np.nan, 250.0, np.nan],
        "vital_status": ["Dead", "Alive", "Dead"],
    })
    endpoint, summary = standardise_endpoint_table(
        source,
        {"OS": {"time": [], "event": []}},
        required_endpoints=["OS"],
    )
    os = endpoint.loc[endpoint.endpoint.eq("OS")].sort_values("patient_id")
    assert os.endpoint_available.sum() == 3
    assert set(os.event.astype(int)) == {0, 1}
    assert set(os.patient_id) == {"TCGA-AA-0001", "TCGA-AA-0002", "TCGA-BB-0003"}


def test_pfi_and_dfi_remain_distinct_from_pfs_and_dfs():
    source = pd.DataFrame(
        {
            "cancer_id": ["LUAD"],
            "sample_id": ["TCGA-AA-0001-01A"],
            "PFI.time": [100.0],
            "PFI": [1],
            "PFS.time": [200.0],
            "PFS": [0],
            "DFI.time": [300.0],
            "DFI": [1],
            "DFS.time": [400.0],
            "DFS": [0],
        }
    )
    aliases = {
        endpoint: {"time": [f"{endpoint}.time"], "event": [endpoint]}
        for endpoint in ["PFI", "PFS", "DFI", "DFS"]
    }
    endpoint, _ = standardise_endpoint_table(source, aliases, required_endpoints=[])
    observed = endpoint.set_index("endpoint")
    assert observed.loc["PFI", "time_days"] == 100.0
    assert observed.loc["PFS", "time_days"] == 200.0
    assert observed.loc["DFI", "time_days"] == 300.0
    assert observed.loc["DFS", "time_days"] == 400.0


def test_discrete_survival_loss_is_finite():
    logits = torch.zeros((4, 5), dtype=torch.float32)
    bins = torch.tensor([0, 1, 3, 4])
    event = torch.tensor([1, 0, 1, 0], dtype=torch.float32)
    available = torch.ones(4)
    loss = discrete_time_nll(logits, bins, event, available)
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_clinical_replication_label():
    settings = {
        "replication_min_abs_beta": 0.1,
        "replication_max_test_p": 0.2,
        "replication_min_test_cindex": 0.52,
    }
    assert clinical_replication_label(0.2, 0.05, 0.6, 0.3, settings) == 1.0
    assert clinical_replication_label(-0.2, 0.05, 0.6, 0.3, settings) == 0.0


def test_geometric_priority_and_cindex():
    score = geometric_priority(pd.Series([0.81, 0.4, np.nan]), pd.Series([0.64, np.nan, 0.5]))
    assert abs(score.iloc[0] - 0.72) < 1e-8
    assert score.iloc[1] == 0.4
    assert score.iloc[2] == 0.5
    c = harrell_c_index(np.array([1, 2, 3, 4]), np.array([1, 1, 0, 0]), np.array([4, 3, 2, 1]))
    assert c == 1.0


def test_phreg_helper_returns_target_statistics():
    from cc_hhgt.clinical_v30 import fit_phreg
    rng = np.random.default_rng(7)
    n = 80
    x = rng.normal(size=(n, 2))
    baseline = rng.exponential(200, size=n)
    time = baseline * np.exp(-0.35 * x[:, 0])
    event = (rng.random(n) < 0.7).astype(float)
    result = fit_phreg(time, event, x, ["subject", "covariate"])
    assert result["status"] == "PASS"
    assert "subject" in result["params"]


def test_small_multi_endpoint_training(tmp_path):
    import importlib.util
    from pathlib import Path
    script = Path(__file__).resolve().parents[1] / "scripts" / "63_train_v30_survival_expert.py"
    spec = importlib.util.spec_from_file_location("v30_survival_script", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    rng = np.random.default_rng(9)
    n_train, n_val, n_test = 48, 16, 18
    n = n_train + n_val + n_test
    split = ["train"] * n_train + ["validation"] * n_val + ["test"] * n_test
    lnc = rng.normal(size=n)
    time = rng.exponential(500 * np.exp(-0.35 * lnc)) + 5
    event = (rng.random(n) < 0.65).astype(float)
    frame = pd.DataFrame({
        "patient_id": [f"P{i}" for i in range(n)],
        "patient_fold_id": "F1",
        "split": split,
        "age_years": rng.normal(60, 8, n),
        "sex": np.where(rng.random(n) > 0.5, "male", "female"),
        "lnc::LNC1": lnc,
        "pf::PF1": rng.normal(size=n),
        "stemness_rna::RNAss": rng.normal(size=n),
        "time_days::OS": time,
        "event::OS": event,
        "endpoint_available::OS": 1,
    })
    cfg = {
        "_results": tmp_path,
        "state_graph": {"required_states": ["stemness_rna::RNAss"]},
        "clinical": {
            "clinical_covariates": {"numeric": ["age_years"], "categorical": ["sex"]},
            "min_patients": 30,
            "min_events": 8,
            "n_time_bins": 5,
            "max_patient_model_molecular_features": 8,
            "min_patient_model_feature_coverage": 0.5,
            "hidden_dim": 16,
            "dropout": 0.1,
            "epochs": 8,
            "patience": 3,
            "batch_size": 32,
            "learning_rate": 0.003,
            "weight_decay": 0.0001,
        },
    }
    pred, importance, meta = module.train_one("TEST", "F1", 7, frame, cfg)
    assert meta["status"] == "COMPLETED"
    assert not pred.empty
    assert pred.risk_score.between(0, 1).all()
    assert (tmp_path / "models" / "clinical_survival" / "TEST" / "F1" / "seed_7" / "best.pt").exists()


def test_clinical_covariate_aliases():
    from cc_hhgt.clinical_v30 import build_clinical_covariates
    source = pd.DataFrame({
        "type": ["LUAD", "LUAD"],
        "bcr_patient_barcode": ["TCGA-AA-0001", "TCGA-AA-0002"],
        "age_at_initial_pathologic_diagnosis": [60, 71],
        "gender": ["MALE", "FEMALE"],
        "ajcc_pathologic_tumor_stage": ["Stage II", "Stage III"],
    })
    endpoints = pd.DataFrame({"cancer_id": ["LUAD", "LUAD"], "patient_id": ["TCGA-AA-0001", "TCGA-AA-0002"]})
    out = build_clinical_covariates(source, endpoints, {
        "age_years": ["age_years", "age_at_initial_pathologic_diagnosis"],
        "sex": ["sex", "gender"],
        "stage": ["stage", "ajcc_pathologic_tumor_stage"],
    })
    assert list(out.age_years) == [60, 71]
    assert set(out.sex) == {"MALE", "FEMALE"}
    assert set(out.stage) == {"Stage II", "Stage III"}
