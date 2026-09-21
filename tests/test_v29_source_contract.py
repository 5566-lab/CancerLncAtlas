from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_v29_required_states_and_training_matrix():
    config = yaml.safe_load((ROOT / "config/model_v2_9_state_graph.yaml").read_text())
    required = set(config["state_graph"]["required_states"])
    assert {
        "EXTEND::published_score",
        "stemness_rna::RNAss",
        "stemness_dna::DNAss",
        "stemness_rna::EREG.EXPss",
    }.issubset(required)
    v30 = yaml.safe_load((ROOT / "config/model_v3_0_state_formal.yaml").read_text())
    assert v30["formal_contract"]["expected_folds"] == 33
    assert v30["formal_contract"]["expected_tasks"] == 297
    assert v30["analysis_cancers"]["reference_only"] == []
    assert set(v30["state_graph"]["required_states"]) == {
        "stemness_rna::RNAss",
        "stemness_dna::DNAss",
    }
    source = (ROOT / "scripts/43c_run_v29_strict_matrix_parallel.py").read_text()
    assert 'MODELS = ("rgcn", "hgt", "cc_hhgt")' in source
    assert "expected = 9 if args.pilot else configured_tasks" in source


def test_state_edges_are_signed_and_direct_target_edges_are_absent():
    source = (ROOT / "cc_hhgt/graph_build.py").read_text()
    assert "positively_associated_with_state" in source
    assert "negatively_associated_with_state" in source
    assert '"lncRNA", "state"' not in source


def test_v29_release_requires_state_embeddings_and_reports():
    audit = (ROOT / "scripts/49_audit_v29_release.py").read_text()
    assert '"embeddings/state.parquet"' in audit
    assert '"pancancer_lncrna_rnass_association.parquet"' in audit
    assert '"state_mean__EXTEND::published_score"' in audit
    assert '"state_mean__stemness_dna::DNAss"' in audit


def test_v30_pipeline_is_gate_then_isolated_runner_then_audit():
    pipeline = (ROOT / "scripts/73_run_v30_state_formal_pipeline.py").read_text()
    order = [
        "71_v30_formal_training_gate.py",
        "43c_run_v29_strict_matrix_parallel.py",
        "72_audit_v30_state_matrix.py",
    ]
    positions = [pipeline.index(name) for name in order]
    assert positions == sorted(positions)


def test_runner_and_final_audit_share_calibrated_probability_contract():
    runner = (ROOT / "scripts/43c_run_v29_strict_matrix_parallel.py").read_text()
    audit = (ROOT / "scripts/72_audit_v30_state_matrix.py").read_text()
    for source in (runner, audit):
        assert "validate_probability_aliases" in source
        assert "PredictionScale.CALIBRATED_PROBABILITY" in source
        assert 'if {"raw_probability", "calibrated_probability"} & set(frame)' not in source


def test_cc_hhgt_strict_pair_evidence_is_masked_for_validation_and_test():
    multitask = (ROOT / "cc_hhgt/v29_multitask.py").read_text()
    audit = (ROOT / "scripts/49_audit_v29_release.py").read_text()
    assert '"strict_pair_evidence_policy": "validation_and_test_fully_masked"' in multitask
    assert 'strict_pair_evidence_train_row_mask_probability' in multitask
    assert 'policy != "validation_and_test_fully_masked"' in audit


def test_v30_parallel_runner_requires_isolated_output_and_fixed_matrix_contract():
    runner = (ROOT / "scripts/43c_run_v29_strict_matrix_parallel.py").read_text()
    assert 'parser.add_argument("--output-root", required=True)' in runner
    assert 'state_training_sample_audit.json' in runner
    assert 'positive <= 0 or unlabeled <= 0' in runner
    assert 'V3 runner refuses every resume/reuse path' in runner
    assert 'expected = 9 if args.pilot else configured_tasks' in runner
    assert 'validate_formal_training_gate(' in runner
    assert 'verify_assets=True' in runner


def test_v29_release_builder_accepts_isolated_retrain_roots():
    builder = (ROOT / "scripts/44_build_v29_strict_release.py").read_text()
    assert 'parser.add_argument("--strict-root"' in builder
    assert 'parser.add_argument("--release-root"' in builder
    assert 'collect(strict_root, model, "state", expected_runs)' in builder
    assert 'folds.fold_id.astype(str).nunique()' in builder
    assert 'found {len(paths)}, expected {expected_runs}' in builder


def test_retrained_state_oof_analysis_uses_prespecified_complete_case_ensembles():
    analysis = (ROOT / "scripts/56_analyze_retrained_state_oof.py").read_text()
    assert '"rgcn_hgt_equal"' in analysis
    assert '"three_model_equal"' in analysis
    assert "mean(axis=1, skipna=False)" in analysis
    assert 'parser.add_argument("--state-oof", required=True)' in analysis
    assert 'duplicate_keys == 0' in analysis


def test_lasso_stemness_baseline_is_nested_inside_patient_folds():
    baseline = (ROOT / "scripts/57_run_nested_lasso_stemness_baseline.py").read_text()
    assert '"RNAss": "stemness_rna::RNAss"' in baseline
    assert '"DNAss": "stemness_dna::DNAss"' in baseline
    assert "screen_features(x_train[:, variable], y_train" in baseline
    assert "LassoCV(" in baseline
    assert '"evaluation_unit": "patient"' in baseline
    assert "existing patient_fold_id" in baseline


def test_lasso_graph_comparison_uses_matched_heldout_patient_association_labels():
    comparison = (ROOT / "scripts/58_compare_lasso_with_patient_graph.py").read_text()
    assert '"stemness_rna::RNAss"' in comparison
    assert '"stemness_dna::DNAss"' in comparison
    assert '"test_membership_label"' in comparison
    assert 'matched.absolute_lasso_coefficient.fillna(0.0)' in comparison
    assert '"evaluation_unit": "cancer x patient_fold x lncRNA x state association replication"' in comparison
    assert 'root.rglob("prediction.parquet")' in comparison


def test_retrained_patient_state_heads_are_isolated_and_use_fixed_embeddings():
    runner = (ROOT / "scripts/59_run_retrained_state_patient_heads.py").read_text()
    assert 'parser.add_argument("--strict-root", required=True)' in runner
    assert 'parser.add_argument("--output-root", required=True)' in runner
    assert '"CC_HHGT_V29_STRICT_ROOT": str(strict_root)' in runner
    assert '"CC_HHGT_STATE_DATA_ROOT": str(state_data)' in runner
    assert 'PATIENT_STATE_DATA_GATE.json' in runner
    assert 'nonpositive_direction_label_violations' in runner
    assert '"graph_encoder_retrained": False' in runner
    assert '"decoder_heads_retrained": True' in runner


def test_patient_state_direction_supervision_is_positive_only():
    builder = (ROOT / "scripts/27_build_lncrna_state_data.py").read_text(encoding="utf-8")
    model = (ROOT / "src/cc_hhgt_v26/state_model.py").read_text(encoding="utf-8")
    audit = (ROOT / "scripts/30_audit_lncrna_state_model.py").read_text(encoding="utf-8")
    assert "observed & positive_membership" in builder
    assert "prepared.train_label.gt(0.5)" in model
    assert '"direction_supervision_scope": "positive_membership_pairs_only"' in model
    assert '"patient_state_unlabeled_direction_count_zero"' in audit


def test_retrained_matrix_audit_enforces_full_pu_sample_quality():
    audit = (ROOT / "scripts/55_audit_retrained_state_matrix.py").read_text()
    assert "training_rows != 150_000" in audit
    assert "training_positive <= 0 or training_unlabeled <= 0" in audit
    assert "0.15 <= training_positive_rate <= 0.25" in audit
    assert '"missing_tasks_preview"' in audit
    assert 'formal training gate provenance mismatch' in audit
    assert 'best.pt SHA256 mismatch' in audit


def test_v30_formal_training_is_fail_closed_on_complete_asset_manifests():
    single = (ROOT / "scripts/42_train_v29_strict_multitask.py").read_text(encoding="utf-8")
    matrix = (ROOT / "scripts/43c_run_v29_strict_matrix_parallel.py").read_text(encoding="utf-8")
    gate = (ROOT / "scripts/71_v30_formal_training_gate.py").read_text(encoding="utf-8")
    asset_builder = (ROOT / "scripts/70_build_v30_formal_assets.py").read_text(encoding="utf-8")
    assert 'parser.add_argument("--formal-gate", required=True)' in single
    assert 'validate_formal_training_gate(' in matrix
    assert 'verify_assets=True' in matrix
    for requirement in [
        'canonical_zero_normal',
        'state_nan_never_enters_statistics',
        'exact_{expected_tasks}_tasks',
        'heldout_only_expression_edges',
        'reference_context_edges_removed',
        'candidate_fdr_family_predefined',
        'unlabeled_direction_supervision_zero',
        'heldout_cancer_features_deterministic_distinguishable',
        'pancancer_lncrna_filter_frozen',
        'pathway_candidates_cover_every_formal_cancer',
        'v31_fixed_graph_state_auxiliary_loss_weight',
    ]:
        assert requirement in gate
    assert 'state_training.auxiliary_loss_weight=0.50' in asset_builder


def test_state_moe_formal_predictions_use_target_cancer_excluded_oof_gate():
    moe = (ROOT / "scripts/29_integrate_lncrna_state_moe.py").read_text(encoding="utf-8")
    assert 'leave_one_cancer_out_gate_applied_to_patient_fold_oof_features' in moe
    assert 'target_cancer_rows_used_to_fit_oof_gate' in moe
    assert 'fold = frame.merge(oof[oof_columns]' in moe


def test_final_state_lasso_report_is_gated_by_all_three_audits():
    report = (ROOT / "scripts/59b_build_state_lasso_comparison_report.py").read_text(encoding="utf-8")
    assert 'require_pass(matrix_dir / "retrained_matrix_audit.json"' in report
    assert 'require_pass(oof_dir / "state_oof_analysis_audit.json"' in report
    assert 'require_pass(matched_dir / "MATCHED_LASSO_GRAPH_AUDIT.json"' in report
    assert 'CC_HHGT_v3_RNAss_DNAss_vs_LASSO_report.pptx' in report
    assert '当前工作站未挂载 `${DATA_ROOT}/stem`' in report


def test_v30_pipeline_requires_explicit_frozen_paths_and_historical_evidence_is_optional():
    aggregate = (ROOT / "scripts/09_aggregate_adapter_and_probabilities.py").read_text()
    pipeline = (ROOT / "scripts/73_run_v30_state_formal_pipeline.py").read_text()
    assert 'CC_HHGT_ADAPTER_RUN_ROOT' in aggregate
    assert 'CC_HHGT_ADAPTER_RESULT_ROOT' in aggregate
    assert 'CC_HHGT_STRICT_OOF_PATH' in aggregate
    assert 'CC_HHGT_GRAPH_NODE' in aggregate
    assert 'HISTORICAL_EVIDENCE_UNAVAILABLE.json' in aggregate
    for required in ["--run-id", "--run-root", "--input-root", "--input-manifest", "--training-root"]:
        assert required in pipeline
    assert "--pilot" in pipeline


def test_v29_release_contains_full_expert_and_state_outputs():
    finalizer = (ROOT / "scripts/51_finalize_v29_release.py").read_text()
    assert 'final_expert_fusion_table.parquet' in finalizer
    assert 'evidence_transformer_prediction.parquet' in finalizer
    assert 'graph_expert_prediction.parquet' in finalizer
    assert 'pancancer_lncrna_rnass_association.parquet' in finalizer


def test_v29_web_materialization_covers_main_and_state_modules():
    materialize = (ROOT / "scripts/52_materialize_v29_web_tables.py").read_text()
    for name in [
        'web_cancer_overview.parquet',
        'web_cancer_lncRNA_ranking.parquet',
        'web_cancer_pathway_ranking.parquet',
        'web_three_probability.parquet',
        'web_lncRNA_state_ranking.parquet',
        'web_pancancer_rnass.parquet',
    ]:
        assert name in materialize
