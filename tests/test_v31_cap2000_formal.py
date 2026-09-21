from __future__ import annotations

from cc_hhgt.common import load_config


CONFIG = "config/model_v3_1_exact_pathway_33c_cap2000.yaml"
SUPERSEDED = (
    "V3STATE-exact_pathway_33c_site_cap1000_resume-"
    "20260823T075437Z-c37890fd3321"
)


def test_cap2000_is_frozen_fail_closed_full_retrain() -> None:
    cfg = load_config(CONFIG, create_dirs=False)
    contract = cfg["formal_contract"]
    selection = cfg["convergence_cap_selection"]

    assert cfg["analysis_version"].endswith("cap2000_resume5_2026-08")
    assert cfg["training"]["epochs"] == 2000
    assert contract["configured_epoch_cap"] == 2000
    assert contract["convergence_policy"] == "fail_if_any_task_reaches_hard_epoch_cap"
    assert contract["expected_tasks"] == 297
    assert contract["supersedes_run_id"] == SUPERSEDED
    assert contract["require_full_297_retrain"] is True
    assert contract["permit_prior_task_import"] is False
    assert contract["permit_prior_checkpoint_reuse"] is False
    assert selection["selection_status"] == "frozen_before_cap2000_formal_run"
    assert selection["validation_interval_epochs"] == 35
    assert selection["earliest_third_regular_stale_validation_epoch"] == 1015
    assert selection["selected_safety_cap"] == 2000
    assert selection["selection_uses_heldout_metric"] is False


def test_cap2000_keeps_exact_pathway_33c_and_pancancer_filter() -> None:
    cfg = load_config(CONFIG, create_dirs=False)
    filter_cfg = cfg["candidate_universe"]["pancancer_lnc_filter"]

    assert cfg["formal_contract"]["required_formal_cancers"] == ["HNSC", "LGG"]
    assert cfg["analysis_cancers"]["reference_only"] == []
    assert cfg["cancer_scope"]["reference_only"] == []
    assert cfg["task_contract"]["pathway"]["target_level"] == "exact_pathway"
    assert cfg["task_contract"]["pathway"]["target_column"] == "pathway_id"
    assert cfg["task_contract"]["pathway_family"]["may_replace_primary_target"] is False
    assert filter_cfg["enabled"] is True
    assert filter_cfg["within_cancer_min_sample_detection_rate"] == 0.10
    assert filter_cfg["minimum_detected_cancers"] == 3
    assert filter_cfg["evidence_cannot_bypass_filter"] is True
    assert cfg["runtime_graph_sampling"]["checkpoint_patience_cycles"] == 3
    assert cfg["training"]["resume_checkpoint_every_epochs"] == 5
    assert cfg["state_training"]["strict_pair_evidence_mask_probability"] == 1.0
