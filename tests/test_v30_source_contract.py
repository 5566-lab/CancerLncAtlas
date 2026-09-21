from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_clinical_integration_preserves_discovery():
    text = (ROOT / "scripts" / "65_train_and_integrate_v30_clinical_expert.py").read_text()
    assert 'final["discovery_ranking_probability"] =' not in text
    assert 'final["survival_used_in_discovery"] = False' in text
    audit = (ROOT / "scripts" / "66_audit_v30_clinical.py").read_text()
    assert "survival_does_not_modify_discovery" in audit


def test_v30_does_not_retrain_v29_graph():
    pipeline = (ROOT / "scripts" / "69_run_v30_full_pipeline.py").read_text()
    assert "43_run_v29_strict_matrix" not in pipeline
    assert "42_train_v29_strict_multitask" not in pipeline
    assert '"60_preflight"' in pipeline
    assert '"68_finalize"' in pipeline


def test_endpoint_specific_heads_and_masks_exist():
    model = (ROOT / "cc_hhgt" / "clinical_v30.py").read_text()
    assert "nn.ModuleDict" in model
    assert "endpoint_available" in (ROOT / "scripts" / "61_standardize_v30_clinical_endpoints.py").read_text()
    assert "right-censored" in model


def test_interval_and_survival_endpoints_are_not_silently_renamed():
    config = (ROOT / "config" / "model_v3_0_clinical.yaml").read_text(encoding="utf-8")
    model = (ROOT / "cc_hhgt" / "clinical_v30.py").read_text(encoding="utf-8")
    assert 'ENDPOINTS = ("OS", "DSS", "PFI", "PFS", "DFI", "DFS")' in model
    assert 'time: [PFI.time, PFI_time, pfi_time, progression_free_interval_days]' in config
    assert 'time: [PFS.time, PFS_time, pfs_time, progression_free_survival_days]' in config
    assert 'time: [DFI.time, DFI_time, dfi_time, disease_free_interval_days]' in config
    assert 'time: [DFS.time, DFS_time, dfs_time, disease_free_survival_days]' in config


def test_clinical_meta_expert_uses_target_cancer_exclusion():
    trainer = (ROOT / "scripts" / "65_train_and_integrate_v30_clinical_expert.py").read_text(encoding="utf-8")
    assert 'leave_one_cancer_out_meta_expert_on_patient_fold_oof_statistics' in trainer
    assert 'target_cancer_rows_used_to_fit' in trainer
