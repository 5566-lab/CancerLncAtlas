"""Observed rule aggregation and real producer/release consumers; no fitting."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cc_hhgt.v32.current_core_export import sha256
from cc_hhgt.v32.evidence_rule_sources import native_events
from cc_hhgt.v32.evidence_rule_summary import (KEYS, review_value, assign_study_components, attach_candidates, summarize_events)
from cc_hhgt.v32.evidence_rule_audit import sql_summary, assert_summary_equal
from cc_hhgt.v32.evidence_rule_release import audit_rule_release, attach_rule_annotations, RuleEvidenceQuery


def candidates():
    return pd.DataFrame([(c, l, "P1") for c in ("BRCA", "LAML") for l in ("LNC:L1", "LNC:ENSG0001", "LNC:MISSING")], columns=KEYS)


def native_fixture():
    rows = []
    for index, (database, pmid, direction, review, cancer) in enumerate([
        ("DB1", "12345678", "positive", "validated", "BRCA"),
        ("DB2", "12345678;23456789", "positive", "validated", "BRCA"),
        ("DB3", "23456789", "negative", "rejected", "BRCA"),
        ("DB1", "34567890", "positive", "not validated", "GLOBAL"),
        ("DB1", "", "positive", "validated", "BRCA"),
    ]):
        rows.append({"event_id": f"EV{index}", "cancer_id": cancer, "lncrna_id": "L1", "pathway_id": "P1",
            "partner_id": "", "route_type": "DIRECT_EXACT_ASSERTION", "static_member_id": "",
            "is_experimental": True, "is_computational": False, "is_model_prediction": False,
            "experiment_type": "knockdown", "graph_assay_class": "unknown", "source_database": database,
            "source_record_id": f"REC{index}", "pmid": pmid, "direction_raw": direction,
            "manual_review_status": review, "confidence_target": 1., "cell_line": "MCF7", "species": "human"})
    events = pd.DataFrame(rows)
    lineage = pd.DataFrame([{"event_id": row["event_id"], "lineage_id": "LIN"+str(i), "source_row_sha256": "a"*64,
        "source_sha256": "b"*64, "family_broadcast_used": False, "mapping_route": row["route_type"], "static_member_id": ""}
        for i, row in enumerate(rows)])
    members = pd.DataFrame({"pathway_id": ["P1", "P1"], "member_id": ["G0", "RBP1"]})
    return events, lineage, members


def ledger():
    raw, lineage, members = native_fixture()
    source = native_events(raw, lineage, members, source_id="native", artifacts={"events": "fixture"})
    return attach_candidates(assign_study_components(source), candidates())


@pytest.mark.parametrize("value, expected", [("not validated", None), ("invalid", 0), ("validated", 1),
    ("unvalidated", None), ("pending", None), ("not approved", 0)])
def test_review_status_is_explicit_and_negation_is_not_accepted(value, expected):
    observed = review_value(value)
    assert np.isnan(observed) if expected is None else observed == expected


def test_connected_studies_database_duplicates_and_conflicts_match_independent_sql():
    events = ledger(); summary, contexts = summarize_events(events, candidates())
    assert_summary_equal(summary, sql_summary(events, candidates()))
    brca = summary.loc[summary.cancer_id.eq("BRCA") & summary.lncrna_id.eq("LNC:L1")].iloc[0]
    assert brca.raw_records == 5 and brca.deduplicated_context_events == 3
    assert brca.independent_studies == 2 and brca.direct_studies == 2
    assert brca.observed_direct_direction == "conflicting" and brca.direction_conflict and brca.review_conflict
    assert brca.reviewed_direct_studies == 0 and np.isnan(brca.reviewed_study_fraction_accepted)
    assert brca.unverified_provenance_records == 1 and brca.global_studies == 1 and brca.cancer_specific_studies == 1
    laml = summary.loc[summary.cancer_id.eq("LAML") & summary.lncrna_id.eq("LNC:L1")].iloc[0]
    assert laml.independent_studies == 1 and laml.cancer_specific_studies == 0
    assert laml.observed_direct_direction == "positive" and laml.reviewed_direct_studies == 0
    unknown = summary.loc[summary.lncrna_id.eq("LNC:MISSING")]
    assert not unknown.evidence_available.any() and unknown.evidence_unavailable_reason.str.contains("NOT_A_NEGATIVE").all()
    assert not summary.independent_external_validation.any() and not summary.neural_score_available.any()
    assert set(contexts.source_cancer_id) == {"BRCA", "GLOBAL"}


def test_row_order_and_database_copy_invariance_and_preserved_contexts():
    events = ledger()
    first, _ = summarize_events(events, candidates())
    second, _ = summarize_events(events.sample(frac=1., random_state=7), candidates())
    assert_summary_equal(first, second)
    duplicates = events.loc[events.source_event_id.eq("EV0")].copy()
    duplicates["source_id"] = "copied_database"; duplicates["evidence_record_id"] += "_copy"
    copied, contexts = summarize_events(pd.concat([events, duplicates], ignore_index=True), candidates())
    for field in ("independent_studies", "direct_studies", "observed_direct_direction", "direction_conflict"):
        assert first[field].equals(copied[field])
    # A separate cell context remains inspectable even within the same study.
    duplicates["cell_line"] = "OTHER_CELL"
    _, contexts = summarize_events(pd.concat([events, duplicates], ignore_index=True), candidates())
    assert "OTHER_CELL" in set(contexts.cell_line)


def test_member_prediction_does_not_set_direct_direction_or_review():
    raw, lineage, members = native_fixture()
    raw = raw.iloc[:1].copy(); lineage = lineage.iloc[:1].copy()
    raw["is_computational"] = True
    source = native_events(raw, lineage, members, source_id="predicted", artifacts={})
    events = attach_candidates(assign_study_components(source), candidates())
    result, _ = summarize_events(events, candidates())
    assert result.independent_studies.eq(0).all() and result.direct_studies.eq(0).all()
    assert result.observed_direct_direction.eq("unknown").all() and result.prediction_sources.sum() == 1
    assert_summary_equal(result, sql_summary(events, candidates()))


@pytest.mark.parametrize("fault", ["lineage", "family", "member", "typed_bool"])
def test_native_events_reject_invalid_lineage_or_exact_mapping(fault):
    events, lineage, members = native_fixture()
    if fault == "lineage": lineage.loc[0, "mapping_route"] = "PARTNER_EXACT_MEMBER"
    elif fault == "family": lineage.loc[0, "family_broadcast_used"] = True
    elif fault == "typed_bool": events["is_experimental"] = "true"
    else:
        events.loc[0, "route_type"] = lineage.loc[0, "mapping_route"] = "PARTNER_EXACT_MEMBER"
        events.loc[0, "partner_id"] = "NOT_A_MEMBER"
    with pytest.raises(ValueError): native_events(events, lineage, members, source_id="bad", artifacts={})


def create_release(tmp_path, universe=None):
    from test_v32_encode_functional import test_complete_command_consumes_cached_native_data_without_training as run_kd
    from test_v32_rbns_sequence import test_full_command_streams_native_scores_and_writes_predicted_binding as run_rbns
    from scripts.materialize_v32_current_rule_evidence import main
    kd = tmp_path/"kd"; kd.mkdir(); run_kd(kd)
    rbns = tmp_path/"rbns"; rbns.mkdir(); run_rbns(rbns)
    def record(path): return {"path": str(path), "sha256": sha256(path)}
    raw, lineage, members = native_fixture()
    tables = {"events": raw, "lineage": lineage, "candidates": candidates() if universe is None else universe, "members": members}
    paths = {}
    for name, frame in tables.items():
        path = tmp_path/(name+".parquet"); frame.to_parquet(path, index=False); paths[name] = record(path)
    manifest = {"format": "CURRENT_RULE_EVIDENCE_INPUT_V1", "candidates": paths["candidates"],
        "exact_membership": paths["members"], "sources": [
            {"source_id": "native", "kind": "native_exact_events", "events": paths["events"], "lineage": paths["lineage"]},
            {"source_id": "kd", "kind": "encode_functional", "audit": record(kd/"output/AUDIT.json"), "table": record(kd/"output/FUNCTIONAL_EVIDENCE.parquet")},
            {"source_id": "rbns", "kind": "rbns_sequence", "audit": record(rbns/"output/AUDIT.json"), "table": record(rbns/"output/GENE_SEQUENCE_PREFERENCE.parquet")}]}
    manifest_path = tmp_path/"INPUT.json"; manifest_path.write_text(json.dumps(manifest))
    output = tmp_path/"release"
    assert main(["--manifest", str(manifest_path), "--output", str(output)]) == 0
    return output/"RULE_EVIDENCE_BINDING.json"


def test_actual_kd_and_rbns_commands_feed_rule_release_query_and_exact_ensemble(tmp_path, monkeypatch):
    import torch
    from cc_hhgt.v32.exact_release import ensemble_fold_predictions
    def forbidden(*args, **kwargs): raise AssertionError("Rule evidence must not create an optimizer")
    monkeypatch.setattr(torch.optim, "Adam", forbidden); monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    binding = create_release(tmp_path); digest = sha256(binding)
    result = audit_rule_release(binding, expected_sha=digest)
    assert result["summary_recomputed_by_independent_sql"] and not result["training_started"]
    from scripts.audit_v32_current_rule_evidence import main as audit_main
    assert audit_main(["--binding", str(binding), "--expected-sha256", digest, "--output", str(tmp_path/"REAUDIT.json")]) == 0
    assert json.loads((tmp_path/"REAUDIT.json.preflight.json").read_text())["target_host"] == "149"
    query = RuleEvidenceQuery(binding, expected_sha=digest)
    kd = query.query(cancer_id="LAML", lncrna_id="LNC:L1")["rows"][0]
    assert kd["functional_bridge_studies"] == 1 and kd["cancer_specific_studies"] == 0
    predicted = query.query(cancer_id="BRCA", lncrna_id="LNC:ENSG0001")["rows"][0]
    assert predicted["prediction_sources"] == 1 and predicted["independent_studies"] == 0
    assert predicted["observed_direct_direction"] == "unknown"
    details = query.query(cancer_id="LAML", lncrna_id="LNC:L1", table="events")["rows"]
    bridge = [r for r in details if r["route_type"] == "RBP_PERTURBATION_PATHWAY_BRIDGE"]
    assert bridge and all(r["direction"] == "unknown" and r["biosample"] == "K562" for r in bridge)
    assert all("qvalue" in json.loads(r["detail_json"]) for r in bridge)
    assert query.query(cancer_id="LAML' OR 1=1 --", lncrna_id="LNC:L1")["rows"] == []
    fold_paths = []
    for fold in range(5):
        frame = candidates().assign(pathway_family_id="F1", shared_or_local_scope="shared", patient_fold_id=fold,
            association_membership_probability=.1*(fold+1), association_direction_probability=.7,
            l1_probability=.2, ridge_probability=.3, graph_residual=0., graph_gate=.5)
        path = tmp_path/f"fold{fold}.parquet"; frame.to_parquet(path, index=False); fold_paths.append(path)
    before = ensemble_fold_predictions(fold_paths, run_id="synthetic-current")
    after = ensemble_fold_predictions(fold_paths, run_id="synthetic-current", rule_evidence_binding={"path": str(binding), "sha256": digest})
    pd.testing.assert_frame_equal(before, after[list(before)])
    assert after.rule_functional_bridge_studies.sum() == 2 and after.rule_prediction_sources.sum() == 2
    assert not after.rule_changes_primary_ranking.any()
    coverage = pd.read_parquet(binding.parent/"SOURCE_COVERAGE.parquet").set_index("source_id")
    assert coverage.loc["kd", "matched_source_events"] == 1 and coverage.loc["kd", "candidate_evidence_rows"] == 2
    assert coverage.loc["rbns", "matched_source_events"] == 1 and coverage.loc["rbns", "candidate_evidence_rows"] == 2


def test_independent_sql_detects_incorrect_production_counts():
    events = ledger(); observed, _ = summarize_events(events, candidates())
    observed.loc[0, "independent_studies"] += 1
    with pytest.raises(ValueError, match="independent_studies"):
        assert_summary_equal(observed, sql_summary(events, candidates()))


def test_empty_sources_keep_typed_missing_candidate_universe():
    events = ledger().iloc[:0]
    observed, contexts = summarize_events(events, candidates())
    assert_summary_equal(observed, sql_summary(events, candidates()))
    assert len(observed) == 6 and not observed.evidence_available.any() and contexts.empty


def test_actual_native_builder_preserves_review_conflicts_context_and_provenance():
    from cc_hhgt.v32.evidence_training import build_exact_event_bags
    _, _, members = native_fixture()
    common = dict(cancer_id="BRCA", lncrna_id="L1", pathway_id="P1", source_record_id="SAME_RECORD",
        source_database="DB", experimental=True, pmid="12345678", doi="10.1000/example",
        source_sha256="b"*64, experiment_type="knockdown", experiment_accession="ENCSR000BBB",
        biosample="MCF7", cell_line="MCF7", intervention="LNC_DEPLETION", readout="RNA_EXPRESSION",
        direction="positive", manual_review_status="validated")
    raw = pd.DataFrame([common, {**common, "manual_review_status": "rejected"}, {**common, "cell_line": "OTHER"}, common])
    built = build_exact_event_bags(raw, pd.DataFrame(), members)
    assert len(built.events) == 3 and built.events.source_occurrence_count.sum() == 4
    source = native_events(built.events, built.lineage, members, source_id="rebuilt", artifacts={})
    events = attach_candidates(assign_study_components(source), candidates())
    result, contexts = summarize_events(events, candidates())
    brca = result.loc[result.cancer_id.eq("BRCA") & result.lncrna_id.eq("LNC:L1")].iloc[0]
    assert brca.independent_studies == 1 and brca.review_conflict and brca.reviewed_direct_studies == 0
    assert set(contexts.cell_line) == {"MCF7", "OTHER"}
    assert set(contexts.experiment_id) == {"ENCSR000BBB"}
    assert events.study_tokens_json.str.contains("DOI:10.1000/example").all()
    assert_summary_equal(result, sql_summary(events, candidates()))


@pytest.mark.parametrize("fault", ["alias", "version", "family", "empty", "nonstring"])
def test_candidate_mapping_never_guesses_versions_or_families(fault):
    from cc_hhgt.v32.evidence_rule_summary import candidate_aliases
    universe = candidates()
    if fault == "alias": universe.loc[0, "lncrna_id"] = "L1"; universe.loc[1, "lncrna_id"] = "LNC:L1"
    elif fault == "empty": universe = universe.iloc[:0]
    elif fault == "nonstring": universe["pathway_id"] = 1
    if fault in {"alias", "empty", "nonstring"}:
        with pytest.raises(ValueError): candidate_aliases(universe)
        return
    source = ledger().drop(columns="evidence_record_id")
    if fault == "version": source["lncrna_id"] = "LNC:L1.2"
    else: source["pathway_id"] = "FAMILY:P1"
    assert attach_candidates(source, universe).empty


@pytest.mark.parametrize("fault", ["summary", "ledger", "extra_cancer", "coverage", "source"])
def test_independent_audit_rejects_rehashed_output_or_changed_raw_source(tmp_path, fault):
    binding_path = create_release(tmp_path)
    binding = json.loads(binding_path.read_text())
    if fault == "source":
        path = tmp_path/"events.parquet"
        frame = pd.read_parquet(path); frame.loc[0, "pmid"] = "99999999"; frame.to_parquet(path, index=False)
    else:
        record = (binding["outputs"]["summary"] if fault == "summary" else binding["outputs"]["source_coverage"]
                  if fault == "coverage" else binding["event_parts"][0])
        path = Path(record["path"]); frame = pd.read_parquet(path)
        if fault == "summary": frame.loc[0, "independent_studies"] += 1
        elif fault == "coverage": frame.loc[0, "normalized_events"] += 1
        elif fault == "ledger": frame.loc[0, "direction"] = "neutral"
        else:
            path = tmp_path/"EXTRA_EVENTS.parquet"; frame["cancer_id"] = "UNKNOWN_CANCER"
            record = {"path": str(path)}; binding["event_parts"].append(record)
        frame.to_parquet(path, index=False); record["sha256"] = sha256(path)
        binding_path.write_text(json.dumps(binding))
    with pytest.raises(ValueError): audit_rule_release(binding_path, expected_sha=sha256(binding_path))


def test_current_fusion_entry_and_public_materializer_consume_rules_without_fitting(tmp_path, monkeypatch):
    from dataclasses import asdict
    import cc_hhgt.v32.multimodal_fusion as fusion
    import scripts.run_v32_current_multimodal_fusion as runner
    import scripts.run_v32_multimodal_fusion_formal as formal
    universe = pd.concat([candidates(), pd.DataFrame({"cancer_id": "BRCA",
        "lncrna_id": [f"LNC:EXTRA{i}" for i in range(80)], "pathway_id": "P1"})], ignore_index=True)
    binding = create_release(tmp_path, universe=universe)
    def record(path): return {"path": str(path), "sha256": sha256(path)}
    primary_path = tmp_path/"primary.parquet"
    universe.assign(**{fusion.PRIMARY_COLUMN: .4, "analysis_version": fusion.ANALYSIS_VERSION}).to_parquet(primary_path, index=False)
    folds = []
    for fold in range(5):
        path = tmp_path/f"primary_fold{fold}.parquet"
        universe.assign(patient_fold_id=fold, held_out_proxy_label=fold%2).to_parquet(path, index=False)
        folds.append({"fold_id": fold, "artifact": record(path)})
    core = tmp_path/"CORE_BINDING.json"
    core.write_text(json.dumps({"status": "PASS", "run_id": "SYNTHETIC_CORE", "prediction": record(primary_path),
        "fold_prediction_sha256": [r["artifact"]["sha256"] for r in folds]}))
    # Synthetic native expert values, not trained predictions or scientific results.
    spec = fusion.ExpertSpec("fixture_sc", "rho", "available", ("discovery", "confidence"), "donor", value_transform="signed_fisher")
    expert_path = tmp_path/"expert.parquet"
    universe.assign(rho=np.linspace(-.5, .5, len(universe)), available=True,
        analysis_version=fusion.ANALYSIS_VERSION).to_parquet(expert_path, index=False)
    expert_audit = tmp_path/"EXPERT_AUDIT.json"
    expert_audit.write_text(json.dumps({"status": "PASS", "run_id": "SYNTHETIC_EXPERT", "estimand": "FIXTURE_ONLY",
        "core_dependency": "INDEPENDENT_MODEL", "expert_specs": [asdict(spec)], "outputs": {"prediction": record(expert_path)}}))
    manifest = {"format": "CURRENT_MULTIMODAL_FUSION_MANIFEST_V1", "core_run_id": "SYNTHETIC_CORE",
        "primary_prediction": record(primary_path), "primary_folds": folds, "primary_binding_audit": record(core),
        "rule_evidence_binding": record(binding), "experts": [{"spec": asdict(spec), "prediction": record(expert_path),
            "binding_audit": record(expert_audit), "run_id": "SYNTHETIC_EXPERT", "estimand": "FIXTURE_ONLY"}]}
    path = tmp_path/"FUSION_INPUT.json"; path.write_text(json.dumps(manifest))
    def forbidden(*args, **kwargs): raise AssertionError("No optimizer or fitted fusion is permitted")
    monkeypatch.setattr(runner, "train_pair_blocked_crossfit_fusion", forbidden)
    monkeypatch.setattr(fusion, "fit_residual_fusion_model", forbidden)
    assert formal.main(["--manifest", str(path), "--expected-core-run-id", "SYNTHETIC_CORE", "--output", str(tmp_path/"prepared")]) == 0
    frame = pd.read_parquet(tmp_path/"prepared/PRIVATE_FUSION_INPUT.parquet")
    assert frame.rule_functional_bridge_studies.sum() == 2
    input_audit = json.loads((tmp_path/"prepared/INPUT_AUDIT.json").read_text())
    assert input_audit["rule_evidence_used_as_model_input"] is False and input_audit["training_started"] is False
    # Fixed synthetic coefficients let us exercise real public serialization,
    # without invoking fit/crossfit or any optimizer.
    results = []
    for endpoint in ("discovery", "confidence"):
        model = fusion.ResidualFusionModel(endpoint, (spec.expert_id,), (0.,), (.3,), 1, "0"*64, "1"*64, 0, 0.,
                                          feature_transforms=("signed_fisher",))
        score = fusion.apply_residual_fusion_model(model, frame)
        clean = frame.drop(columns=[c for c in frame if c.startswith("rule_")])
        pd.testing.assert_frame_equal(score, fusion.apply_residual_fusion_model(model, clean))
        results.append(fusion.CrossfitFusionResult(endpoint, (spec,), score, pd.DataFrame({"fixture_only": [True]}),
            (model,)*5, model, "SYNTHETIC_NOT_ESTIMATED", "SYNTHETIC_NOT_ESTIMATED"))
    declarations = [{**record(path), "source_role": "raw_source_data", "generation": "raw_source"}]
    before = fusion.materialize_multimodal_fusion_release(tmp_path/"public_before", frame, *results,
        source_declarations=declarations, training_run_id="V32-FUSION-SYNTHETIC_ONLY", input_audit=input_audit)
    after = fusion.materialize_multimodal_fusion_release(tmp_path/"public_after", frame, *results,
        source_declarations=declarations, training_run_id="V32-FUSION-SYNTHETIC_ONLY", input_audit=input_audit,
        rule_evidence_binding=record(binding))
    before_frame = pd.read_parquet(before["prediction_path"]); after_frame = pd.read_parquet(after["prediction_path"])
    pd.testing.assert_frame_equal(before_frame, after_frame[list(before_frame)])
    assert after_frame.rule_functional_bridge_studies.sum() == 2 and after_frame.rule_prediction_sources.sum() == 2
    assert not after_frame.rule_changes_primary_ranking.any()


def test_empty_native_source_can_materialize_missing_evidence_release(tmp_path):
    from cc_hhgt.v32.evidence_training import build_exact_event_bags
    from scripts.materialize_v32_current_rule_evidence import main
    _, _, members = native_fixture()
    built = build_exact_event_bags(pd.DataFrame(), pd.DataFrame(), members)
    records = {}
    for name, frame in {"events": built.events, "lineage": built.lineage, "members": members, "candidates": candidates()}.items():
        path = tmp_path/(name+".parquet"); frame.to_parquet(path, index=False)
        records[name] = {"path": str(path), "sha256": sha256(path)}
    manifest = {"format": "CURRENT_RULE_EVIDENCE_INPUT_V1", "candidates": records["candidates"],
        "exact_membership": records["members"], "sources": [{"source_id": "empty", "kind": "native_exact_events",
            "events": records["events"], "lineage": records["lineage"]}]}
    path = tmp_path/"INPUT.json"; path.write_text(json.dumps(manifest))
    assert main(["--manifest", str(path), "--output", str(tmp_path/"output")]) == 0
    binding = tmp_path/"output/RULE_EVIDENCE_BINDING.json"
    query = RuleEvidenceQuery(binding, expected_sha=sha256(binding))
    result = query.query(cancer_id="BRCA", lncrna_id="LNC:L1")["rows"][0]
    assert result["evidence_available"] is False and result["reviewed_study_fraction_accepted"] is None
    assert query.query(cancer_id="BRCA", lncrna_id="LNC:L1", table="events")["rows"] == []
