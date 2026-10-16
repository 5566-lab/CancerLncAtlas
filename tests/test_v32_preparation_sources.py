"""Actual preparation reads and current consumers, using only synthetic inputs."""
import copy
import json
from pathlib import Path
import socket

import numpy as np
import pandas as pd
import pytest
import torch

from cc_hhgt.v32.preparation_sources import (SourceCapture, AUTHORITY_KEY, attach_sources,
    canonical_sha, validate_payload_sources, source_members)
from cc_hhgt.v32.current_model_use import Inventory, read, write, record
from scripts.prepare_v32_formal import load_adjusted_associations, make_batches, attach_replication_labels


def forbidden(*args, **kwargs): raise AssertionError("Model fitting is forbidden in preparation source verification")


@pytest.fixture(autouse=True)
def no_fitting(monkeypatch):
    assert socket.gethostname() == "149"
    for name in ("Adam", "AdamW", "SGD"): monkeypatch.setattr(torch.optim, name, forbidden)
    from sklearn.linear_model import LogisticRegression
    monkeypatch.setattr(LogisticRegression, "fit", forbidden)


def adjusted_capture(root, fold=0, *, train_pmid="12345678", validation_pmid="87654321"):
    root.mkdir(parents=True)
    capture = SourceCapture()
    for role in ("authorization", "patient_map", "patient_receipt", "train_audit_success",
                 "train_audit_independent", "train_audit_decision", "validation_audit_success", "validation_audit_independent"):
        path = root / (role + ".json")
        write(path, {"synthetic": True, "role": role})
        capture.read(path, read, role=role, kind="control")
    for role in ("detection", "pathway_hierarchy", "candidate_authority"):
        path = root / (role + ".parquet")
        pd.DataFrame({"cancer_id": ["BRCA"], "lncrna_id": ["LNC:Z"], "pathway_id": ["P1"]}).to_parquet(path, index=False)
        capture.read(path, pd.read_parquet, role=role)
    for split, directory, prefix, pmid in (("train", "pair_level", "A1_local", train_pmid),
                                           ("validation", "validation_associations", "A1_validation", validation_pmid)):
        path = root / split / directory / "cancer=BRCA" / f"fold={fold}" / "part.parquet"
        path.parent.mkdir(parents=True)
        pd.DataFrame({"cancer_id": ["BRCA"], "lncrna_id": ["LNC:Z"], "pathway_id": ["P1"],
            "fold_id": [fold], "availability": ["available"], "unavailable_reason": [""],
            "rho_"+prefix: [0.4], "p_"+prefix: [0.001], "fdr_official_"+prefix: [0.01],
            "label_"+prefix: ["strong_positive"], "direction_"+prefix: ["positive"],
            "n_"+prefix: [15], "residual_design_rank_"+prefix:[2], "correlation_df_"+prefix:[12],
            "pmid": [pmid]}).to_parquet(path, index=False)
    frames = load_adjusted_associations(capture, root / "train", root / "validation", fold=fold, cancers=["BRCA"])
    receipt = capture.receipt(fold=fold, mode="local-cnv-adjusted", cancers=["BRCA"])
    return capture, receipt, frames


def test_actual_adjusted_reader_preserves_used_files_and_separate_study_sources(tmp_path):
    _, receipt, frames = adjusted_capture(tmp_path / "inputs")
    tables = {r["role"]: r for r in receipt["source_reads"] if r["role"].endswith("_associations")}
    assert tables["train_associations"]["members"][0]["study_tokens"] == ["PMID:12345678"]
    assert tables["validation_associations"]["members"][0]["study_tokens"] == ["PMID:87654321"]
    assert frames["train"].association_available.tolist() == [True]
    assert frames["validation"].discovery_effect.tolist() == [.4]
    assert not receipt["raw_study_ancestry_complete"]


def test_changed_partition_with_unchanged_success_receipt_is_detected(tmp_path):
    capture, receipt, _ = adjusted_capture(tmp_path / "inputs")
    success = next(r for r in receipt["source_reads"] if r["role"] == "train_audit_success")
    before = record(success["path"])
    source = next(r for r in receipt["source_reads"] if r["role"] == "train_associations")
    frame = pd.read_parquet(source["path"]); frame.loc[0, "rho_A1_local"] = .9
    frame.to_parquet(source["path"], index=False)
    assert record(success["path"]) == before
    with pytest.raises(ValueError, match="changed before output"):
        capture.receipt(fold=0, mode="local-cnv-adjusted", cancers=["BRCA"])


def test_changed_during_read_and_extra_directory_partition_fail(tmp_path):
    path = tmp_path / "source.parquet"
    pd.DataFrame({"value": [1]}).to_parquet(path, index=False)
    def changing_reader(file):
        result = pd.read_parquet(file)
        result.assign(value=2).to_parquet(file, index=False)
        return result
    with pytest.raises(ValueError, match="changed during read"):
        SourceCapture().read(path, changing_reader, role="expression")
    tree = tmp_path / "activity"; tree.mkdir()
    pd.DataFrame({"value": [1], "dataset_id": ["GSE85011"]}).to_parquet(tree / "a.parquet", index=False)
    capture = SourceCapture(); capture.read(tree, pd.read_parquet, role="activity")
    pd.DataFrame({"value": [2], "dataset_id": ["GSE99999"]}).to_parquet(tree / "b.parquet", index=False)
    with pytest.raises(ValueError, match="changed between reads"):
        capture.read(tree, pd.read_parquet, role="activity")


def test_relocated_native_source_keeps_hash_and_explicit_missing_metadata(tmp_path):
    import shutil
    _, receipt, _ = adjusted_capture(tmp_path / "inputs")
    payload = {"patient_fold": 0, "input_scope": {"association_mode": "local-cnv-adjusted"}}
    attach_sources(payload, receipt)
    source = next(r for r in receipt["source_reads"] if r["role"] == "train_associations")
    relocated = tmp_path / "relocated.parquet"
    shutil.move(source["path"], relocated)
    inventory = Inventory([record(relocated)], tmp_path)
    inventory.preparation(payload, tmp_path, "m", 0, False)
    assert source["sha256"] in inventory.used_locations
    assert any(r["path"] == str(relocated) and r["sha256"] == source["sha256"] for r in inventory.dependencies)
    assert any(r["reason"] == "SOURCE_STUDY_METADATA_ABSENT" for r in inventory.gaps)
    assert any("PMID:12345678" in r["study_tokens_json"] for r in inventory.usage)


def test_metadata_is_explicit_and_retains_multiple_accession_columns(tmp_path):
    path = tmp_path / "GSE00000.tsv"
    pd.DataFrame({"pmid": [12345678], "dataset_id": ["GSE85011"], "dataset_accession": ["SRP12345"],
                  "experiment_id": ["ENCSR123ABC"], "patient_id": ["PRIVATE_PATIENT"]}).to_csv(path, sep="\t", index=False)
    observed = source_members(path, "table")[0]
    assert observed["study_tokens"] == ["DATASET:GSE85011", "DATASET:SRP12345", "EXPERIMENT:ENCSR123ABC", "PMID:12345678"]
    assert "PRIVATE_PATIENT" not in json.dumps(observed) and "GSE00000" not in observed["study_tokens"]


def test_formal_entrypoint_serializes_consumed_legacy_sources_in_all_arms_without_fitting(tmp_path, monkeypatch):
    """Exercise main's writers with fixed baseline/graph and small synthetic scope.

    The strict real-cohort-size/frozen-map checks and already tested graph builder
    are replaced only here. Expression/activity reads, association calculation,
    batch construction, source capture and all three payload writers are real.
    """
    from types import SimpleNamespace
    import sys
    import scripts.prepare_v32_formal as prepare
    import cc_hhgt.v32.patient_fold_authority as patient
    import cc_hhgt.v32.formal_graph_authority as graph
    import cc_hhgt.v32.formal_graph as variants
    from test_v32_current_binding_expansion import current_bundle
    from cc_hhgt.v32.input_lineage import artifact_sha256
    sample_ids = [f"SYNTHETIC_SAMPLE_{i}" for i in range(25)]
    manifest = pd.DataFrame({"sample_id": sample_ids, "patient_id": [f"SYNTHETIC_PATIENT_{i}" for i in range(25)],
                             "cancer_id": "BRCA", "patient_fold_id": np.repeat(np.arange(5), 5)})
    patient_map = tmp_path / "patients.tsv"; manifest.to_csv(patient_map, sep="\t", index=False)
    receipt = tmp_path / "patients.json"; write(receipt, {"synthetic_only": True})
    hashes = tmp_path / "hashes.json"; write(hashes, {"synthetic_only": True})
    scope = pd.DataFrame({"cancer_id": ["BRCA"], "lncrna_id": ["LNC:Z"], "detection_rate": [.8],
        "within_cancer_eligible": [True], "detected_cancers": [5], "shared_or_local_scope": ["shared"]})
    detection = tmp_path / "detection.parquet"; scope.to_parquet(detection, index=False)
    hierarchy = tmp_path / "hierarchy.tsv"
    pd.DataFrame({"pathway_id": ["P1"], "pathway_family_id": ["F1"], "weight": [1.]}).to_csv(hierarchy, sep="\t", index=False)
    activity = tmp_path / "activity"; activity.mkdir()
    values = np.sin(np.arange(25))
    manifest[["cancer_id", "sample_id", "patient_id"]].assign(pathway_id="P1", activity_score=values,
        dataset_accession="GSE12345").to_parquet(activity / "part.parquet", index=False)
    expression = tmp_path / "expression" / "cancer_id=BRCA"; expression.mkdir(parents=True)
    manifest[["cancer_id", "sample_id", "patient_id"]].assign(lncrna_id="LNC:Z", logcpm=values).to_parquet(expression / "part-0.parquet", index=False)
    monkeypatch.setattr(patient, "validate_frozen_v32_patient_fold_binding", lambda *a: {
        "manifest_sha256": artifact_sha256(patient_map), "receipt_sha256": artifact_sha256(receipt)})
    inputs = SimpleNamespace(receipt_sha256="a"*64, static_paths={"detection": detection, "pathway_hierarchy": hierarchy},
                             static_sha256={"detection": artifact_sha256(detection), "pathway_hierarchy": artifact_sha256(hierarchy)})
    monkeypatch.setattr(graph, "load_formal_graph_input_authority", lambda **kw: inputs)
    monkeypatch.setattr(prepare, "load_scope", lambda frame: frame)
    bundle = current_bundle()
    authority = SimpleNamespace(manifest={"variant_edge_counts": {arm: len(bundle.edges) for arm in ("G0", "G1", "G2")}})
    monkeypatch.setattr(prepare, "build_graph", lambda *a, **kw: SimpleNamespace(authority=authority, binding={"synthetic_only": True}))
    monkeypatch.setattr(variants, "build_variant_runtime_bundle", lambda *a: bundle)
    monkeypatch.setattr(graph, "bind_formal_graph_variant", lambda binding, authority, arm: {"synthetic_only": True, "arm": arm})
    fixed = SimpleNamespace(decision_function=lambda x: np.zeros(len(x)))
    monkeypatch.setattr(prepare, "_fast_l1", lambda frames, **kw: (fixed, {"synthetic_fixed_no_fit": True}, frames["train"], frames["validation"]))
    output = tmp_path / "prepared"
    argv = ["prepare", "--hashes", str(hashes), "--patient-folds", str(patient_map),
            "--patient-fold-authority-receipt", str(receipt), "--graph-authority-receipt", str(receipt),
            "--graph-authority-receipt-sha256", "a"*64, "--graph-fold-expression-pattern", "unused{fold}",
            "--graph-fold-coexpression-pattern", "unused{fold}", "--formal-expression-root", str(expression.parent),
            "--formal-activity-root", str(activity), "--budget-per-cancer", "1", "--fold-count", "1", "--output", str(output)]
    for role in ("detection", "signed-membership", "pathway-hierarchy", "lnc-protein-binding", "protein-gene-encoding", "ppi"):
        argv += ["--graph-"+role, str(detection), "--graph-"+role+"-sha256", "a"*64]
    monkeypatch.setattr(sys, "argv", argv)
    assert prepare.main() == 0
    assert read(output / "LAUNCH_PREFLIGHT.json")["target_host"] == "149"
    captured = read(output / "SOURCE_READS/PATIENT_FOLD_0.json")
    assert {r["role"] for r in captured["source_reads"]} == {"authorization", "patient_map", "patient_receipt", "detection", "pathway_hierarchy", "activity", "expression"}
    for arm in ("G0", "G1", "G2"):
        payload = torch.load(output / arm / "PATIENT_FOLD_0.pt", weights_only=False)
        assert validate_payload_sources(payload) == captured
        assert len(payload["train_batches"]) == len(payload["validation_batches"]) == 1
    assert "DATASET:GSE12345" in json.dumps(captured)


@pytest.mark.parametrize("fault", ["receipt_hash", "missing_receipt", "missing_role", "fold", "cancer", "invented_metadata"])
def test_native_training_and_inventory_reject_altered_preparation_sources(tmp_path, fault):
    from test_v32_current_binding_expansion import verified_payload
    from cc_hhgt.v32.training import _validate_prepared
    _, receipt, _ = adjusted_capture(tmp_path / "inputs")
    payload = verified_payload(); payload["input_scope"]["association_mode"] = "local-cnv-adjusted"
    attach_sources(payload, receipt)
    if fault == "receipt_hash": payload["preparation_sources"]["cancer_ids"] = ["LUAD"]
    elif fault == "missing_receipt": del payload["preparation_sources"]
    elif fault == "missing_role":
        receipt["source_reads"] = [r for r in receipt["source_reads"] if r["role"] != "train_associations"]
    elif fault == "fold": receipt["source_reads"][-1]["fold"] = 4
    elif fault == "cancer": receipt["source_reads"][-1]["cancer_id"] = "LUAD"
    else: receipt["source_reads"][-1]["members"][0]["study_tokens"] = ["PMID:99999999"]
    if fault in {"missing_role", "fold", "cancer", "invented_metadata"}:
        for key in ("artifact_hashes", "input_authority_hashes"): payload[key][AUTHORITY_KEY] = canonical_sha(receipt)
    if fault == "invented_metadata":
        with pytest.raises(ValueError, match="metadata differs"):
            Inventory([], tmp_path).preparation(payload, tmp_path, "m", 0, False)
    else:
        with pytest.raises(ValueError): _validate_prepared(payload, fold=0, artifact_hashes={}, authorized_graph_overlay=True)


@pytest.fixture(scope="module")
def native_sources(tmp_path_factory):
    import test_v32_current_binding_expansion as fixtures
    from test_v32_current_core_prediction import current_fixture
    from cc_hhgt.v32.current_core_prediction import materialize_current_predictions
    from cc_hhgt.v32.current_model_use import materialize_current_model_use, INPUT_FORMAT
    root = tmp_path_factory.mktemp("preparation_source_chain")
    original = fixtures.verified_payload
    def payload(fold=0):
        result = original(fold)
        _, receipt, frames = adjusted_capture(root / f"source{fold}", fold)
        for frame in frames.values():
            frame["shared_or_local_scope"] = "shared"; frame["detection_rate"] = .8; frame["detected_cancers"] = 5
        frames["validation"] = attach_replication_labels(frames["train"], frames["validation"])
        for split in ("train", "validation"):
            result[split+"_batches"] = make_batches(frames[split], np.zeros(1), result["bundle"], split=split)
        result["input_scope"]["association_mode"] = "local-cnv-adjusted"
        result["label_contract"].update(train_direction_label_source="train_local_cnv_adjusted_effect",
            validation_direction_label_source="validation_local_cnv_adjusted_replication_effect",
            association_target="A1_LOCAL", association_available_mask="association_available")
        attach_sources(result, receipt)
        return result
    with pytest.MonkeyPatch.context() as patch:
        for name in ("Adam", "AdamW", "SGD"): patch.setattr(torch.optim, name, forbidden)
        from sklearn.linear_model import LogisticRegression
        patch.setattr(LogisticRegression, "fit", forbidden)
        patch.setattr(fixtures, "verified_payload", payload)
        manifest = current_fixture(root / "native")
        binding = materialize_current_predictions(manifest, root / "predictions", expected_run_id="CURRENT-V4-SYNTHETIC")
        models = [{"model_id": model, "producer_kind": "current_core_release", "producer_binding": record(binding),
                   "run_id": "CURRENT-V4-SYNTHETIC", "score_column": score}
                  for model, score in (("primary", "association_membership_probability"), ("baseline", "l1_probability"))]
        source = root / "INPUT.json"; write(source, {"format": INPUT_FORMAT, "models": models})
        import scripts.export_v32_current_core_embeddings as exporter
        patch.setattr(exporter, "_build_model", forbidden)
        result = materialize_current_model_use(source, root / "inventory")
    return root, models, Path(result["binding_path"])


def test_real_current_producer_propagates_train_sources_and_keeps_l1_validation_diagnostic(native_sources, monkeypatch):
    from cc_hhgt.v32.current_model_use import load_current_model_use
    import scripts.export_v32_current_core_embeddings as exporter
    monkeypatch.setattr(exporter, "_build_model", forbidden)
    _, models, path = native_sources
    binding, tables = load_current_model_use(path, expected_sha=record(path)["sha256"], models=models)
    usage = tables["usage"]; metadata = usage.loc[usage.use_view.eq("PREPARATION_SOURCE_METADATA")]
    baseline = metadata.loc[metadata.model_id.eq("baseline")]
    assert not baseline.study_tokens_json.str.contains("87654321").any()
    assert not baseline.phase.eq("MODEL_SELECTION").any()
    assert baseline.study_tokens_json.str.contains("12345678").any()
    primary = metadata.loc[metadata.model_id.eq("primary") & metadata.study_tokens_json.str.contains("87654321")]
    assert primary.phase.eq("MODEL_SELECTION").all() and set(primary.patient_fold_id) == set(range(5))
    assert metadata.lncrna_id.eq("").all() and metadata.candidate_rows.eq(0).all()
    assert not binding["model_use_inventory_completeness_verified"]
    assert not tables["dependencies"].loc[tables["dependencies"].model_id.eq("baseline"), "role"].str.startswith("graph.").any()


def test_actual_external_v2_excludes_shared_study_even_for_an_unseen_entity(native_sources, tmp_path):
    from cc_hhgt.v32.external_validation_current import materialize_current_external, CurrentExternalValidationQuery
    _, models, inventory = native_sources
    files = {}
    for name, frame in {
        "crosswalk": pd.DataFrame([("lncrna", "new", "LNC:NEW"), ("cancer", "breast", "BRCA"), ("pathway", "p", "P1")],
                                   columns=["namespace", "raw_id", "canonical_id"]),
        "external": pd.DataFrame({"record_id": ["synthetic"], "lncrna": ["new"], "cancer": ["breast"], "pathway": ["p"],
            "label": [1], "is_experimental": [True], "is_predicted": [False], "assay": ["Synthetic assay"], "pmid": ["12345678"]})
    }.items():
        path = tmp_path / (name + ".parquet"); frame.to_parquet(path, index=False); files[name] = record(path)
    source = tmp_path / "INPUT.json"
    write(source, {"format": "CURRENT_EXTERNAL_VALIDATION_INPUT_V2", "models": models,
        "identity_crosswalk": files["crosswalk"], "model_use_inventory": record(inventory),
        "external_sources": [{"kind": "exact_assay", "source_id": "independent_entity_same_study", "table": files["external"]}],
        "evaluation": {"primary_model": "primary", "cutoffs": [1], "lncrna_aggregation": "MAX", "bootstrap_replicates": 0, "seed": 0}})
    result = materialize_current_external(source, tmp_path / "external")
    query = CurrentExternalValidationQuery(result["binding_path"], expected_sha=result["binding_sha256"])
    rows = query.query(table="isolation")["rows"]
    assert rows and all(r["study_overlap"] and not r["lncrna_overlap"] and not r["provenance_held_out"] for r in rows)
