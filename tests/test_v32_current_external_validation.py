"""Synthetic source isolation and numerical external evaluation; no optimizer."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cc_hhgt.v32.current_core_export import sha256
from cc_hhgt.v32.evidence_rule_summary import KEYS
from cc_hhgt.v32.external_validation_sources import identity_maps, normalize_source, isolate_sources
from cc_hhgt.v32.external_validation_metrics import (classification_metrics, rank_details, evaluate_external,
    validate_scores, cluster_labels, deduplicate_targets)


def crosswalk():
    rows = [("lncrna", f"L{i}", f"LNC:L{i}") for i in range(12)]
    rows += [("cancer", "breast", "BRCA"), ("cancer", "lung", "LUAD"), ("pathway", "p1", "P1"), ("pathway", "p2", "P2")]
    return pd.DataFrame(rows, columns=["namespace", "raw_id", "canonical_id"])


def raw_assays():
    rows = []
    for i, label in enumerate([1, 0, 1, 0, 1, 0, 1, 0]):
        rows.append({"record_id": f"OBS{i}", "lncrna": f"L{i}", "cancer": "breast", "pathway": "p1", "label": label,
            "pmid": str(12345000+i), "is_experimental": True, "is_predicted": False, "assay": "measured exact assay"})
    return pd.DataFrame(rows)


def model_use():
    return pd.DataFrame({"lncrna_id": ["LNC:L8"], "pathway_id": ["P1"], "cancer_id": ["LUAD"], "pmid": ["99999999"]})


def observations(raw=None, used=None):
    maps = identity_maps(crosswalk())
    ext = normalize_source(raw_assays() if raw is None else raw,
        {"kind": "exact_assay", "source_id": "external", "table": {"sha256": "a"*64},
         "measured_negative_definition": "Measured null effect in the registered assay"}, maps)
    use = normalize_source(model_use() if used is None else used,
        {"kind": "canonical_usage", "phase": "TRAINING", "source_id": "used", "table": {"sha256": "b"*64}}, maps, usage=True)
    return isolate_sources(ext, use)


def model_scores():
    frame = pd.DataFrame({"cancer_id": "BRCA", "lncrna_id": [f"LNC:L{i}" for i in range(10)], "pathway_id": "P1",
        "score": [.9, .1, .8, .2, .7, .3, .6, .4, .5, .0]})
    return {"primary": frame, "baseline": frame.assign(score=.5)}


def test_metrics_match_analytic_auc_average_precision_and_tied_rank_recovery():
    metrics = classification_metrics([1, 0, 1, 0], [.9, .8, .7, .1])
    assert metrics["auroc"] == .75 and metrics["average_precision"] == pytest.approx(5/6)
    assert metrics["brier"] == pytest.approx((.01+.64+.09+.01)/4)
    tied = classification_metrics([1, 0, 1, 0], [.5]*4)
    assert tied["auroc"] == .5 and tied["average_precision"] == .5
    assert all(np.isnan(v) for v in classification_metrics([1, 1], [.9, .8]).values())
    frame = model_scores()["baseline"]
    targets = frame.iloc[[0, 2]][KEYS].assign(label=1)
    details = rank_details(targets, frame, [1, 3, 20])
    assert details.hit_1.eq(.1).all() and details.hit_3.eq(.3).all() and details.hit_20.eq(1.).all()
    assert details.rank_midpoint.eq(5.5).all()
    labels = np.array([1, 0, 1, 0]); scores = np.array([.7, .7, .3, .1]); weights = np.array([3, 2, 1, 0])
    weighted = classification_metrics(labels, scores, weights)
    expanded = classification_metrics(np.repeat(labels, weights), np.repeat(scores, weights))
    for key in weighted: assert weighted[key] == pytest.approx(expanded[key])


def test_connected_multi_pmid_dataset_and_entity_exclusions_are_cancer_agnostic():
    raw = raw_assays()
    raw.loc[0, "pmid"] = "12345678; 23456789"
    raw.loc[1, "pmid"] = "23456789;34567890"
    raw.loc[2, "pmid"] = "34567890"
    raw.loc[3, "dataset_id"] = "GSE12345"
    used = pd.concat([model_use(), pd.DataFrame([
        {"lncrna_id": "LNC:L11", "pathway_id": "P2", "cancer_id": "LUAD", "pmid": "PMID:12345678"},
        {"lncrna_id": "LNC:L10", "pathway_id": "P2", "cancer_id": "LUAD", "dataset_id": "GSE12345"},
        {"lncrna_id": "LNC:L4", "pathway_id": "P1", "cancer_id": "LUAD", "pmid": "88888888"},
        {"lncrna_id": "LNC:L5", "pathway_id": "P2", "cancer_id": "LUAD", "pmid": "77777777"}])], ignore_index=True)
    isolated, _ = observations(raw, used)
    assert isolated.iloc[:4].study_overlap.all() and not isolated.iloc[:4].provenance_held_out.any()
    assert isolated.iloc[4].provenance_held_out and not isolated.iloc[4].pair_held_out
    assert isolated.iloc[5].pair_held_out and not isolated.iloc[5].lncrna_held_out
    assert isolated.iloc[6:].lncrna_held_out.all()


def test_missing_use_provenance_or_unknown_entity_cannot_claim_independence():
    used = model_use(); used["pmid"] = ""
    isolated, _ = observations(used=used)
    assert isolated.descriptive_eligible.all() and not isolated.provenance_held_out.any()
    used.loc[0, "lncrna_id"] = "UNKNOWN"
    with pytest.raises(ValueError, match="Unmapped model-use"): observations(used=used)
    used = model_use(); used["lncrna_id"] = ""
    isolated, _ = observations(used=used)
    assert isolated.provenance_held_out.all() and not isolated.pair_held_out.any() and not isolated.lncrna_held_out.any()


def test_exact_source_uses_explicit_negatives_and_typed_flags():
    maps = identity_maps(crosswalk()); frame = raw_assays()
    spec = {"kind": "exact_assay", "source_id": "external", "table": {"sha256": "a"*64}}
    with pytest.raises(ValueError, match="negative"): normalize_source(frame, spec, maps)
    frame["is_experimental"] = "False"
    with pytest.raises(ValueError, match="typed"): normalize_source(frame, {**spec, "measured_negative_definition": "measured"}, maps)


def test_duplicate_database_rows_do_not_change_estimates_and_conflicts_are_visible():
    isolated, used = observations()
    kwargs = dict(primary_model="primary", cutoffs=(1, 3), bootstrap_replicates=20, seed=13)
    original, _, _ = evaluate_external(isolated, used, model_scores(), **kwargs)
    copied = pd.concat([isolated, isolated], ignore_index=True)
    repeated, _, coverage = evaluate_external(copied, used, model_scores(), **kwargs)
    pd.testing.assert_frame_equal(original, repeated)
    conflict = isolated.iloc[[0]].copy(); conflict["label"] = 0
    _, details, coverage = evaluate_external(pd.concat([isolated, conflict]), used, model_scores(), **kwargs)
    assert coverage.conflicting_targets_excluded.eq(1).all()
    assert not details.lncrna_id.eq("LNC:L0").any()


def test_paired_cluster_intervals_and_no_false_classification_without_negatives():
    isolated, used = observations()
    metrics, details, coverage = evaluate_external(isolated, used, model_scores(), primary_model="primary", cutoffs=(1, 3),
        bootstrap_replicates=40, seed=91)
    select = metrics.loc[metrics.source_group.eq("ALL_SOURCES") & metrics.isolation_scope.eq("provenance_held_out")
        & metrics.metric.eq("auroc")]
    assert select.loc[select.baseline_model_id.eq("baseline"), "value"].iloc[0] == .5
    delta = select.loc[select.baseline_model_id.eq("baseline")].iloc[0]
    assert delta.ci_available and delta.ci_low == .5 and delta.ci_high == .5
    only_positive = isolated.loc[isolated.label.eq(1)]
    point, _, _ = evaluate_external(only_positive, used, model_scores(), primary_model="primary", cutoffs=(3,), bootstrap_replicates=0)
    assert not point.loc[point.metric.isin(["auroc", "average_precision", "brier", "logloss"]), "available"].any()
    assert point.loc[point.metric.eq("recall_all_positive_at_3"), "available"].all()
    assert not point.negative_labels_synthesized.any()


def test_study_or_lncrna_dependence_forms_one_resampling_block():
    events, _ = observations(); events["study_id"] = "SAME_STUDY"
    targets = deduplicate_targets(events)
    assert len(set(cluster_labels(targets))) == 1


def test_versions_and_families_are_not_guessed_and_paired_universes_must_match():
    raw = raw_assays(); raw.loc[0, "lncrna"] = "L0.2"; raw.loc[1, "pathway"] = "FAMILY:p1"
    isolated, _ = observations(raw)
    assert not isolated.iloc[:2].mapped.any()
    models = model_scores(); models["baseline"] = models["baseline"].iloc[:-1]
    with pytest.raises(ValueError, match="same complete"): validate_scores(models)


def test_native_mixed_methods_numeric_pmids_and_followup_cohorts_are_not_synthetic_truth():
    maps = identity_maps(pd.concat([crosswalk(), pd.DataFrame([["cancer", "MCF7", "BRCA"]], columns=crosswalk().columns)]))
    raw = pd.DataFrame({"ncRNA Symbol": ["L0", "L1", "L2"], "Disease Name": "breast",
        "Validated Method//Prediction Method": ["qPCR", "not experimentally validated", "new_unknown_method"],
        "PubMed ID": [12345678., np.nan, 23456789.]})
    spec = {"kind": "lncrnadisease", "source_id": "mixed", "table": {"sha256": "a"*64},
        "method_classes": {"qPCR": "experimental", "not experimentally validated": "predicted"}}
    source = normalize_source(raw, spec, maps)
    assert source.is_experimental.tolist() == [True, False, False]
    assert source.is_predicted.tolist() == [False, True, False]
    assert json.loads(source.iloc[0].study_tokens_json) == ["PMID:12345678"]
    followup = pd.DataFrame({"gsm": ["GSM1", "GSM2", "GSM3"], "target_raw": ["L0", "L1", "L2"], "cell_line": "MCF7",
        "dataset_accession": "GSE85011", "processed_data_file": ["rnaseq_tpm.gz", "rnaseq_tpm.gz", "other.gz"],
        "is_control": [False, True, False]})
    result = normalize_source(followup, {"kind": "gse85011_followup", "source_id": "gse", "table": {"sha256": "b"*64},
        "followup_selection_semantics": "PUBLISHED_GROWTH_MODIFIER_HITS"}, maps)
    assert result.label.notna().tolist() == [True, False, False] and result.label.dropna().eq(1).all()
    assert result.pathway_id.eq("").all() and result.biosample.eq("MCF7").all()
    assert result.study_tokens_json.str.contains("DATASET:GSE85011").all()


def test_actual_fusion_public_scores_are_reconstructed_and_changed_scores_rejected(tmp_path):
    import cc_hhgt.v32.multimodal_fusion as fusion
    from cc_hhgt.v32.external_validation_current import producer_scores
    frame = model_scores()["primary"].rename(columns={"score": "primary_probability"})
    frame["sc_probability"] = np.linspace(-.4, .4, len(frame)); frame["sc_available"] = True
    spec = fusion.ExpertSpec("sc", "rho", "available", ("discovery", "confidence"), "donor", value_transform="signed_fisher")
    fitted = []
    for endpoint in ("discovery", "confidence"):
        model = fusion.ResidualFusionModel(endpoint, ("sc",), (0.,), (.3,), 1, "0"*64, "1"*64, 0, 0., feature_transforms=("signed_fisher",))
        fitted.append(fusion.CrossfitFusionResult(endpoint, (spec,), fusion.apply_residual_fusion_model(model, frame),
            pd.DataFrame({"synthetic_only": [True]}), (model,)*5, model, "SYNTHETIC_NOT_ESTIMATED", "SYNTHETIC_NOT_ESTIMATED"))
    source = tmp_path/"SOURCE.txt"; source.write_text("SYNTHETIC ONLY; NO OPTIMIZATION")
    result = fusion.materialize_multimodal_fusion_release(tmp_path/"fusion", frame, *fitted,
        source_declarations=[{"path": str(source), "sha256": sha256(source), "source_role": "raw_source_data", "generation": "raw_source"}],
        training_run_id="V32-FUSION-SYNTHETIC_EXTERNAL")
    binding = Path(result["binding_path"])
    specification = {"model_id": "fusion", "producer_kind": "fusion_release", "producer_binding": {"path": str(binding), "sha256": sha256(binding)},
        "run_id": "V32-FUSION-SYNTHETIC_EXTERNAL", "score_column": "discovery_adjusted_probability"}
    scores, provenance = producer_scores(specification, tmp_path)
    assert len(scores) == len(frame) and provenance["producer_recomputed"]
    payload = json.loads(binding.read_text()); record = payload["artifacts"]["secondary_scores"]
    path = Path(record["path"]); public = pd.read_parquet(path); public.loc[0, "discovery_adjusted_probability"] = .99
    public.to_parquet(path, index=False); record["sha256"] = sha256(path); binding.write_text(json.dumps(payload))
    specification["producer_binding"]["sha256"] = sha256(binding)
    with pytest.raises(ValueError, match="producer recomputation"): producer_scores(specification, tmp_path)


def build_exact_producer(tmp_path):
    from cc_hhgt.v32.exact_release import materialize_exact_release
    frame = model_scores()["primary"]
    paths = {key: [] for key in ("fold_prediction_paths", "fold_metrics_paths", "training_success_paths", "checkpoint_paths")}
    hashes = {key: "a"*64 for key in ("code_sha256", "config_sha256", "input_manifest_sha256", "task_manifest_sha256")}
    for fold in range(5):
        pred = frame.rename(columns={"score": "association_membership_probability"}).assign(pathway_family_id="F1",
            shared_or_local_scope="shared", patient_fold_id=fold, association_direction_probability=.5,
            l1_probability=.5, ridge_probability=.5, graph_residual=0., graph_gate=.5)
        path = tmp_path/f"fold{fold}.parquet"; pred.to_parquet(path, index=False); paths["fold_prediction_paths"].append(path)
        metric = {"fold": fold, "candidate_alignment": "FULL_ROW_EXACT", "checkpoint_cycle": 0, "rows": len(frame),
            "artifact_hashes": hashes, **{k: {"rows": len(frame)} for k in ("cc_hhgt", "l1", "ridge")}}
        path = tmp_path/f"metrics{fold}.json"; path.write_text(json.dumps(metric)); paths["fold_metrics_paths"].append(path)
        success = {"status": "SUCCESS", "patient_fold": fold, "did_not_hit_hard_cap": True, "completed_cycles": 1,
            "seed": 13, "artifact_hashes": hashes, "fixture_only": True}
        path = tmp_path/f"success{fold}.json"; path.write_text(json.dumps(success)); paths["training_success_paths"].append(path)
        path = tmp_path/f"checkpoint{fold}.txt"; path.write_text("SYNTHETIC SERIALIZATION FIXTURE; NO MODEL WAS FITTED")
        paths["checkpoint_paths"].append(path)
    candidate = tmp_path/"candidates.parquet"; frame[KEYS].to_parquet(candidate, index=False)
    folds = tmp_path/"folds.json"; folds.write_text("{}")
    prep = tmp_path/"prep.json"; prep.write_text("{}")
    result = materialize_exact_release(**paths, candidate_path=candidate, fold_manifest_path=folds, prep_summary_path=prep,
        output_root=tmp_path/"exact_release", run_id="CURRENT_SYNTHETIC_FIXTURE")
    return Path(result["lineage_path"])


def test_actual_exact_producer_external_command_and_queries_without_training(tmp_path, monkeypatch):
    import torch
    from scripts.analyze_v32_current_external_validation import main
    from cc_hhgt.v32.external_validation_current import CurrentExternalValidationQuery, audit_current_external
    def forbidden(*args, **kwargs): raise AssertionError("External evaluation must not fit any model")
    monkeypatch.setattr(torch.optim, "Adam", forbidden); monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    producer = build_exact_producer(tmp_path)
    def record(path): return {"path": str(path), "sha256": sha256(path)}
    tables = {}
    for name, frame in {"crosswalk": crosswalk(), "usage": model_use(), "assay": raw_assays()}.items():
        path = tmp_path/(name+".parquet"); frame.to_parquet(path, index=False); tables[name] = record(path)
    native = tmp_path/"lnc2cancer.tsv"
    pd.DataFrame({"name": ["L0", "L2"], "cancer type": ["breast", "breast"], "pubmed id": ["12345000", "12345002"],
        "methods": ["measured", "measured"]}).to_csv(native, sep="\t", index=False)
    manifest = {"format": "CURRENT_EXTERNAL_VALIDATION_INPUT_V1", "identity_crosswalk": tables["crosswalk"],
        "models": [{"model_id": name, "producer_kind": "exact_release", "producer_binding": record(producer),
            "run_id": "CURRENT_SYNTHETIC_FIXTURE", "score_column": column} for name, column in
            (("primary", "association_membership_probability"), ("baseline", "l1_probability"))],
        "model_use_sources": [{"kind": "canonical_usage", "source_id": "used", "phase": "TRAINING",
            "model_ids": ["primary", "baseline"], "table": tables["usage"]}],
        "external_sources": [{"kind": "exact_assay", "source_id": "assay", "table": tables["assay"],
            "measured_negative_definition": "Registered measured null effect"},
            {"kind": "lnc2cancer", "source_id": "lnc2cancer", "table": record(native)}],
        "evaluation": {"primary_model": "primary", "cutoffs": [1, 3], "lncrna_aggregation": "MAX", "bootstrap_replicates": 20, "seed": 0}}
    path = tmp_path/"INPUT.json"; path.write_text(json.dumps(manifest))
    output = tmp_path/"external"
    assert main(["--manifest", str(path), "--output", str(output)]) == 0
    binding = output/"EXTERNAL_BINDING.json"
    query = CurrentExternalValidationQuery(binding, expected_sha=sha256(binding))
    results = query.query(filters={"model_id": "primary", "baseline_model_id": "", "endpoint": "EXACT_PATHWAY",
        "metric": "auroc", "isolation_scope": "provenance_held_out", "source_group": "ALL_SOURCES"})["rows"]
    assert len(results) == 1 and results[0]["value"] == 1.
    assert query.query(filters={"cancer_id": "BRCA' OR true --"})["rows"] == []
    assert query.binding["independent_external_validation_attested"] is False
    # A forged result with an updated hash still fails raw-source recomputation.
    bound = json.loads(binding.read_text()); metric_record = bound["outputs"]["metrics"]
    metric_path = Path(metric_record["path"]); frame = pd.read_parquet(metric_path)
    frame.loc[frame.value.notna().idxmax(), "value"] += .25
    frame.to_parquet(metric_path, index=False); metric_record["sha256"] = sha256(metric_path); binding.write_text(json.dumps(bound))
    with pytest.raises(ValueError, match="recomputation disagrees"): audit_current_external(binding, expected_sha=sha256(binding))
