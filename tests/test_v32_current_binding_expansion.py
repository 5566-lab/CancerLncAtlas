"""Synthetic current-C expansion; actual tensors/forwards, never an optimizer."""
import copy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import torch

from cc_hhgt.v32.current_binding_graph import (
    ARMS, LINEAGE_HASH_COLUMNS, adapt_october_edges, normalize_binding_source,
    build_current_binding_arms, binding_arm_runtime_bundle,
)
from cc_hhgt.v32.formal_graph import (
    build_formal_graph_authority, build_variant_runtime_bundle,
    materialize_expressed_in, materialize_signed_coexpression,
    materialize_signed_membership, materialize_pathway_hierarchy,
    materialize_typed_lnc_protein_binding, materialize_protein_gene_encoding,
)
from cc_hhgt.v32.safe_graph import build_safe_graph


def source(rows):
    frame = pd.DataFrame(rows, columns=["lncrna_id", "protein_id", "graph_assay_class", "weight"])
    frame["source_database"] = "synthetic_source"; frame["cancer_id"] = None
    frame["is_context_specific"] = False
    return frame


def current_bundle(fold=0, pathway_id="P1"):
    # Added IDs sort BEFORE current IDs: a full resort would break candidates.
    binding = source([("LNC:Z", "UNIPROT:Z", "eclip", 1.)])
    coexp = pd.DataFrame({"cancer_id": ["BRCA"]*3, "lncrna_id": ["LNC:Z"]*3,
        "gene_id": ["G1", "G2", "G3"], "rho": [.8, -.7, .6], "source_split": ["train"]*3,
        "edge_outer_fold": [fold]*3})
    expression = pd.DataFrame({"cancer_id": ["BRCA"], "lncrna_id": ["LNC:Z"], "detection_rate": [.8]})
    authority = build_formal_graph_authority([
        materialize_expressed_in(expression, candidate_pairs=expression),
        materialize_signed_coexpression(coexp, outer_fold=fold),
        materialize_signed_membership(pd.DataFrame({"gene_id": ["G1", "G2"], "pathway_id": [pathway_id, pathway_id], "weight": [1., -1.]})),
        materialize_pathway_hierarchy(pd.DataFrame({"pathway_id": [pathway_id], "pathway_family_id": ["F1"], "weight": [1.]})),
        materialize_protein_gene_encoding(pd.DataFrame({"protein_id": ["UNIPROT:Z"], "gene_id": ["G1"], "weight": [1.]})),
        materialize_typed_lnc_protein_binding(binding, require_global_binding=True),
    ], outer_fold=fold)
    return build_variant_runtime_bundle(authority, "G2", edge_chunk_size=2)


def expansion():
    return normalize_binding_source(source([
        ("LNC:Z", "UNIPROT:Z", "rip", 1.),  # already in current C
        ("LNC:A", "UNIPROT:A", "rip", 1.),
        ("LNC:A", "UNIPROT:A", "predicted", .4),  # physical takes precedence
        ("LNC:Z", "UNIPROT:A", "predicted", .4),
    ]), source_id="native_plus_rbns")


def test_three_arms_preserve_old_indices_features_edges_and_rotating_chunks():
    base = current_bundle(); original = copy.deepcopy(base)
    arms = build_current_binding_arms(base, [expansion()], fold=0, edge_chunk_size=2)
    bundles = [binding_arm_runtime_bundle(arms, arm) for arm in ARMS]
    assert [len(b.edges) for b in bundles] == [len(base.edges)+i for i in range(3)]
    assert arms.audit["appended_nodes"] == 2 and arms.audit["excluded_rows"] == 2
    for bundle in bundles:
        for kind, mapping in base.node_maps.items():
            assert all(bundle.node_maps[kind][key] == value for key, value in mapping.items())
            assert np.array_equal(base.node_degrees[kind], bundle.node_degrees[kind][:len(mapping)])
        assert bundle.runtime_num_chunks == 2
        retained = bundle.edges.loc[bundle.edges.edge_id.isin(base.edges.edge_id), base.edges.columns]
        pd.testing.assert_frame_equal(retained.reset_index(drop=True), base.edges.reset_index(drop=True))
    assert all(b.node_maps == bundles[0].node_maps for b in bundles)
    for kind in bundles[0].node_maps:
        assert all(torch.equal(b.hetero_data[kind].x, bundles[0].hetero_data[kind].x) for b in bundles)
    assert bundles[2].edges.loc[bundles[2].edges.relation_type.eq("binds_protein_predicted"), "weight"].tolist() == [.25]
    pd.testing.assert_frame_equal(base.edges, original.edges)
    assert base.node_maps == original.node_maps


def test_each_arm_has_identical_model_state_schema_and_real_candidate_forward():
    from cc_hhgt.gnn import build_model
    from cc_hhgt.v32.training import _runtime_graph_for_chunk
    base = current_bundle(); arms = build_current_binding_arms(base, [expansion()], fold=0, edge_chunk_size=2)
    config = {"training": {"hidden_channels": 8, "num_layers": 1, "num_heads": 2, "dropout": 0., "pair_hidden_channels": 4}}
    state = None
    for arm in ARMS:
        bundle = binding_arm_runtime_bundle(arms, arm)
        torch.manual_seed(7); model = build_model("cc_hhgt", bundle, 2, config).eval()
        if state is None: state = copy.deepcopy(model.state_dict())
        else: model.load_state_dict(state, strict=True)
        with torch.inference_mode():
            for chunk in range(2):
                output = model.encode(_runtime_graph_for_chunk(bundle, chunk, device="cpu"))
                assert torch.isfinite(output["lncRNA"][base.node_maps["lncRNA"]["LNC:Z"]]).all()
                assert torch.isfinite(output["pathway"][base.node_maps["pathway"]["P1"]]).all()


def native_source():
    cls = "experimental_physical"; lnc = "LNC:A"; protein = "UNIPROT:A"
    evidence = "EVIDEDGE:"+hashlib.sha256("\x1f".join([cls, lnc, protein]).encode()).hexdigest()
    line = dict(edge_evidence_id=evidence, lncrna_id=lnc, protein_id=protein, evidence_class=cls,
        experiment_type="RIP-seq", source_database="NPInter", is_experimental=True, is_computational=False,
        is_physical=True, family_broadcast_used=False, lineage_id="line1", event_id="ev1", raw_event_id="raw1",
        partner_type="rbp", source_row_sha256="a"*64, source_sha256="b"*64)
    lineage_sha = hashlib.sha256("\x1f".join(str(line[k]) for k in LINEAGE_HASH_COLUMNS).encode()).hexdigest()
    edges = pd.DataFrame([dict(edge_evidence_id=evidence, lncrna_id=lnc, protein_id=protein, evidence_class=cls,
        mapping_multiplicity=1, source_lineage_sha256=lineage_sha, source_event_count=1,
        cancer_id=None, is_context_specific=False, weight=1.)])
    return edges, pd.DataFrame([line])


def test_native_lineage_adapter_preserves_experimental_assay_and_provenance():
    edges, lineage = native_source(); output = adapt_october_edges(edges, lineage)
    assert output.graph_assay_class.tolist() == ["rip"]
    assert output.edge_evidence_id.tolist() == edges.edge_evidence_id.tolist()
    assert output.source_lineage_sha256.tolist() == edges.source_lineage_sha256.tolist()


@pytest.mark.parametrize("drift", ["event_hash", "endpoint", "flag", "functional", "family", "mapping"])
def test_native_lineage_rejects_corrupted_or_nonphysical_evidence(drift):
    edges, lineage = native_source()
    if drift == "event_hash": lineage.loc[0, "source_row_sha256"] = "c"*64
    elif drift == "endpoint": lineage.loc[0, "protein_id"] = "UNIPROT:B"
    elif drift == "flag": lineage.loc[0, "is_computational"] = True
    elif drift == "functional": lineage.loc[0, "experiment_type"] = "knockdown"
    elif drift == "family": lineage.loc[0, "family_broadcast_used"] = True
    else: edges.loc[0, "mapping_multiplicity"] = 2
    with pytest.raises(ValueError): adapt_october_edges(edges, lineage)


def test_predictions_are_unobserved_and_cannot_impersonate_experimental_edges():
    frame = materialize_typed_lnc_protein_binding(source([("LNC:Z", "UNIPROT:A", "predicted", .25)]), include_predicted=True)
    assert frame.observed.tolist() == [False]
    build_safe_graph(frame, outer_fold=0)
    frame["observed"] = True
    with pytest.raises(RuntimeError, match="Unsafe edges"): build_safe_graph(frame, outer_fold=0)
    measured = materialize_typed_lnc_protein_binding(source([("LNC:Z", "UNIPROT:A", "rip", 1.)]))
    measured["observed"] = False
    with pytest.raises(RuntimeError, match="Unsafe edges"): build_safe_graph(measured, outer_fold=0)


def test_three_arm_builder_rejects_vacuous_comparison_and_changed_current_features():
    base = current_bundle(); rows = expansion()
    with pytest.raises(ValueError, match="both new"):
        build_current_binding_arms(base, [rows.loc[rows.graph_assay_class.ne("predicted")]], fold=0, edge_chunk_size=2)
    base.node_degrees["protein"][0] = 999
    with pytest.raises(ValueError, match="node features"):
        build_current_binding_arms(base, [rows], fold=0, edge_chunk_size=2)


def verified_payload(fold=0):
    from test_v32_training_objective import _graph_binding
    from cc_hhgt.v32 import patient_fold_authority as authority
    from cc_hhgt.v32.training import PREPARED_FORMAT
    # Synthetic serialized metadata, never a replacement patient-map artifact.
    patient = {"format": authority.FORMAL_PREPARED_BINDING_FORMAT,
        "status": "PASS_FROZEN_V32_PATIENT_FIRST_AUTHORITY", "sample_id_patient_fallback_used": False,
        "legacy_patient_fold_manifest_used": False,
        "sample_patient_fold_map": {"sha256": authority.FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256},
        "authority_receipt": {"sha256": authority.FROZEN_V32_RECEIPT_SHA256},
        "authority": {"binding_format": authority.FROZEN_V32_BINDING_FORMAT,
            "status": "PASS_FROZEN_V32_PATIENT_FIRST_AUTHORITY",
            "manifest_sha256": authority.FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256,
            "receipt_sha256": authority.FROZEN_V32_RECEIPT_SHA256,
            "patient_authority_sha256": authority.FROZEN_V32_PATIENT_AUTHORITY_LOGICAL_SHA256,
            "sample_patient_map_sha256": authority.FROZEN_V32_SAMPLE_PATIENT_MAP_LOGICAL_SHA256,
            "seed": authority.DEFAULT_SEED, "folds": authority.N_FOLDS,
            "patients": authority.FROZEN_V32_PATIENTS, "samples": authority.FROZEN_V32_PATIENTS,
            "cancers": sorted(authority.TCGA_CANCERS), "explicit_patient_id": True,
            "sample_fallback_used": False, "legacy_patient_fold_manifest_accepted": False, "receipt_required": True}}
    batch = {"candidate_batch": {"l": torch.tensor([0]), "p": torch.tensor([0]), "c": torch.tensor([0])},
        "base_logit": torch.zeros(1), "conservation_context": torch.zeros((1, 4)),
        "proxy_label": torch.ones(1), "weak_positive": torch.tensor([False]),
        "association_available": torch.tensor([True]),
        "direction_label": torch.ones(1), "direction_available": torch.tensor([True])}
    payload = {"prepared_format": PREPARED_FORMAT, "patient_fold": fold, "feature_dim": 2,
        "patient_fold_authority": patient, "formal_graph_variant": "G2", "train_batches": [batch],
        "validation_batches": [copy.deepcopy(batch)],
        "input_scope": {"candidate_rows_per_split": 1, "discovery_features_split": "train_patients_only",
            "replication_labels_splits": ["validation"], "held_out_effects_as_features": False,
            "sealed_test_accessed": False},
        "label_contract": {
            "train_direction_label_source": "train_discovery_effect", "validation_direction_label_source": "validation_replication_effect",
            "heldout_direction_is_model_input": False, "test_labels_in_training_payload": False,
            "test_logits_in_training_payload": False, "test_metrics_computed_before_winner_lock": False}}
    bundle = current_bundle(fold)
    payload.update(bundle=bundle, patient_fold=fold, formal_graph_authority=_graph_binding(bundle),
        artifact_hashes={"code_sha256": "f"*64}, input_authority_hashes={"source": "a"*64},
        graph_overlay_authority={"verified": True})
    payload["formal_graph_authority"]["outer_fold"] = fold
    payload["formal_graph_authority"]["binding_generation"] = "TYPED_ASSAY_CLASS_V1"
    payload["legacy_model_config"] = {"training": {"hidden_channels": 8, "num_layers": 1,
        "num_heads": 2, "dropout": 0., "pair_hidden_channels": 4}}
    return payload


def comparison_config():
    return {"task_contract": {"rbp_evidence_mode": "C_TYPED_ENCODE_V2", "graph_variant": "G2"},
        "runtime_profile": {"max_coverage_cycles": 2, "device": "cpu", "target_host": "149", "mixed_precision": False},
        "training_io": {}, "primary_model": {"hidden_channels": 8, "dropout": 0.}}


def overlay_inputs():
    from cc_hhgt.v32.current_binding_overlay import prepare_binding_overlay, comparison_config_sha
    payload = verified_payload(); config = comparison_config()
    comparison = {"format": "CURRENT_C_THREE_ARM_COMPARISON_V1", "config_sha256": comparison_config_sha(config), "predicted_dropout": .5}
    arms = build_current_binding_arms(payload["bundle"], [expansion()], fold=0, edge_chunk_size=2)
    return payload, config, prepare_binding_overlay(payload, arms, "experimental_predicted", comparison)


def test_expansion_sidecar_uses_actual_training_validator_and_reuses_all_candidate_tensors():
    from cc_hhgt.v32.current_binding_overlay import apply_binding_overlay, non_graph_payload_sha
    from cc_hhgt.v32.training import _validate_prepared
    payload, config, overlay = overlay_inputs()
    result = apply_binding_overlay(payload, overlay, fold=0, arm="experimental_predicted", overlay_sha="b"*64,
        comparison_sha=overlay["formal_graph_authority"]["comparison_sha256"], config=config)
    _validate_prepared(result, fold=0, artifact_hashes={}, authorized_graph_overlay=True)
    assert result["train_batches"] is payload["train_batches"]
    assert result["validation_batches"] is payload["validation_batches"]
    assert non_graph_payload_sha(result) == non_graph_payload_sha(payload)
    assert result["binding_expansion_authority"]["verified"]


@pytest.mark.parametrize("drift", ["label", "candidate", "new_input", "budget", "arm", "parent", "edge", "dropout"])
def test_expansion_loader_rejects_changed_comparison_semantics(drift):
    from cc_hhgt.v32.current_binding_overlay import apply_binding_overlay
    payload, config, overlay = overlay_inputs(); arm = "experimental_predicted"
    if drift == "label": payload["train_batches"][0]["proxy_label"][0] = 0
    elif drift == "candidate": payload["train_batches"][0]["candidate_batch"]["l"][0] = 1
    elif drift == "new_input": payload["new_modality"] = torch.ones(2)
    elif drift == "budget": config["runtime_profile"]["max_coverage_cycles"] = 3
    elif drift == "arm": arm = "experimental"
    elif drift == "parent": payload["input_authority_hashes"]["source"] = "c"*64
    elif drift == "edge": overlay["bundle"].edges.loc[0, "weight"] = .9
    else: overlay["bundle"].relation_edge_dropout = {}
    with pytest.raises(ValueError):
        apply_binding_overlay(payload, overlay, fold=0, arm=arm, overlay_sha="b"*64,
            comparison_sha=overlay["formal_graph_authority"]["comparison_sha256"], config=config)


def test_server149_cpu_device_is_explicit_and_has_no_cuda_calls(monkeypatch):
    from types import SimpleNamespace
    from cc_hhgt.v32 import training_device
    from cc_hhgt.v32.orchestration import build_task_manifest
    monkeypatch.setattr(training_device.socket, "gethostname", lambda: "149")
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    runtime = {"device": "cpu", "target_host": "149", "mixed_precision": False}
    context = {"endpoint_id": "server149", "hardware_class": "SERVER149_CPU"}
    assert training_device.resolve_training_device(fake_torch, runtime, context) == "cpu"
    for event in ("empty_cache", "reset_peak_memory_stats", "max_memory_reserved"):
        assert training_device.cuda_memory_event(fake_torch, "cpu", event) == 0
    assert all(t["status"] == "BLOCKED" and not t["paid_task"] for t in build_task_manifest(owner="server149", hardware_class="SERVER149_CPU"))
    with pytest.raises(RuntimeError, match="fallback"):
        training_device.resolve_training_device(fake_torch, {}, context)
    monkeypatch.setattr(training_device.socket, "gethostname", lambda: "windows_local")
    with pytest.raises(RuntimeError, match="server149"):
        training_device.resolve_training_device(fake_torch, runtime, context)


def write_preparation_inputs(tmp_path):
    import yaml
    from cc_hhgt.v32.c_graph_overlay import OVERLAY_FORMAT
    from cc_hhgt.v32.current_core_export import sha256
    from cc_hhgt.v32.current_binding_overlay import GRAPH_FIELDS
    def record(path): return {"path": str(path), "sha256": sha256(path)}
    config_path = tmp_path/"config.yaml"; config_path.write_text(yaml.safe_dump(comparison_config()))
    graph_root = tmp_path/"G2"; graph_root.mkdir()
    folds = []
    for fold in range(5):
        payload = verified_payload(fold)
        old = {k: v for k, v in payload.items() if k not in GRAPH_FIELDS-{ "bundle", "formal_graph_authority"}}
        prepared = graph_root/f"A_FOLD_{fold}.pt"; torch.save(old, prepared)
        graph_receipt_sha = payload["formal_graph_authority"]["receipt"]["sha256"]
        sidecar = {"format": OVERLAY_FORMAT, "patient_fold": fold, "formal_graph_variant": "G2",
            "source_prepared_sha256": sha256(prepared), "graph_authority_receipt_sha256": graph_receipt_sha,
            "bundle": payload["bundle"], "formal_graph_authority": payload["formal_graph_authority"]}
        path = graph_root/f"C_FOLD_{fold}.pt"; torch.save(sidecar, path)
        current_manifest = tmp_path/f"current_{fold}.json"
        current_manifest.write_text(json.dumps({"fold_inputs": [{"fold": fold, **record(prepared),
            "graph_overlay": {**record(path), "format": OVERLAY_FORMAT, "bytes": path.stat().st_size,
                "source_prepared_sha256": sha256(prepared), "graph_authority_receipt_sha256": graph_receipt_sha}}]}))
        folds.append({"fold": fold, "prepared": record(prepared), "graph_overlay": record(path), "input_manifest": record(current_manifest)})
    native_edges, native_lineage = native_source()
    edges_path = tmp_path/"edges.parquet"; native_edges.to_parquet(edges_path)
    lineage_path = tmp_path/"lineage.parquet"; native_lineage.to_parquet(lineage_path)
    rbns_path = tmp_path/"RBNS_TYPED_BINDING.parquet"
    source([("LNC:Z", "UNIPROT:A", "predicted", .4)]).to_parquet(rbns_path)
    rbns_audit = tmp_path/"rbns_audit.json"
    rbns_audit.write_text(json.dumps({"format": "RBNS_SEQUENCE_ANALYSIS_AUDIT_V1", "status": "PASS",
        "annotation_id": "synthetic_annotation", "lncrna_binding_measured": False,
        "outputs": {"RBNS_TYPED_BINDING.parquet": record(rbns_path)}}))
    manifest = tmp_path/"comparison_input.json"
    manifest.write_text(json.dumps({"format": "CURRENT_C_BINDING_COMPARISON_INPUT_V1", "config": record(config_path),
        "fold_inputs": folds, "sources": [
            {"source_id": "october", "kind": "native_october", "edges": record(edges_path), "lineage": record(lineage_path)},
            {"source_id": "rbns", "kind": "rbns_sequence", "audit": record(rbns_audit), "binding": record(rbns_path)}]}))
    return manifest


def test_fivefold_prepare_to_training_loader_and_frozen_export_entry_without_optimizer(tmp_path, monkeypatch):
    import yaml
    from scripts import prepare_v32_current_binding_comparison as prepare
    from scripts.export_v32_current_core_embeddings import _validated_payload, _build_model
    from cc_hhgt.v32.current_binding_overlay import load_declared_binding_expansion
    from cc_hhgt.v32.current_core_export import sha256
    from cc_hhgt.v32.training import _runtime_graph_for_chunk, _prepared_input_authority_hashes
    from cc_hhgt.v32.cli import main as pipeline
    from cc_hhgt.gnn import build_model
    from cc_hhgt.v32.model import build_v32_cc_hhgt_residual
    def forbidden(*args, **kwargs): raise AssertionError("No optimizer may be created during preparation/export")
    monkeypatch.setattr(torch.optim, "Adam", forbidden); monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    manifest = write_preparation_inputs(tmp_path); output = tmp_path/"comparison"
    assert prepare.main(["--manifest", str(manifest), "--output", str(output), "--release", "synthetic", "--edge-chunk-size", "2"]) == 0
    result = json.loads((output/"COMPARISON.json").read_text())
    assert len(result["commands"]) == 15 and not result["training_started"]
    # The generated public trainer command is executable and remains blocked.
    assert pipeline(result["commands"][0]["argv"][2:]) != 0
    source_records = json.loads(manifest.read_text())["fold_inputs"]
    graph_hashes = {}
    for arm in ARMS:
        folder = output/arm; config_path = folder/"config.yaml"; config = yaml.safe_load(config_path.read_text())
        input_path = folder/"INPUT_MANIFEST.json"; input_doc = json.loads(input_path.read_text())
        for fold, authorized in enumerate(input_doc["fold_inputs"]):
            current, _ = prepare.load_current_fold(source_records[fold], tmp_path)
            payload = load_declared_binding_expansion(torch, payload=current, config=config,
                input_manifest_path=input_path, fold=fold, root=tmp_path)
            graph_hashes[(arm, fold)] = payload["formal_graph_authority"]["graph"]["edge_sha256"]
            assert payload["train_batches"] is current["train_batches"]
            if fold: continue
            receipt = tmp_path/f"synthetic_{arm}_receipt.json"
            receipt.write_text(json.dumps({"artifact_hashes": {}, "input_authority_hashes": _prepared_input_authority_hashes(payload)}))
            paths = {k: __import__("pathlib").Path(source_records[0][k]["path"]) for k in ("prepared", "graph_overlay")}
            paths.update(config=config_path, input_manifest=input_path, training_receipt=receipt,
                binding_expansion=__import__("pathlib").Path(authorized["binding_expansion"]["path"]))
            record = {key: {"sha256": sha256(path)} for key, path in paths.items()}
            exported_payload = _validated_payload(paths, record, 0)
            encoder = build_model("cc_hhgt", payload["bundle"], 2, payload["legacy_model_config"])
            model = build_v32_cc_hhgt_residual(encoder, hidden_channels=8, context_features=4, dropout=0.)
            restored = _build_model(exported_payload, config, {"model_state": model.state_dict()}, "cpu").eval()
            with torch.inference_mode():
                graph = _runtime_graph_for_chunk(exported_payload["bundle"], 0, device="cpu")
                encoded = restored.encoder.encode(graph)
                batch = exported_payload["train_batches"][0]
                prediction = restored(graph, batch["candidate_batch"], batch["base_logit"],
                    batch["conservation_context"], encoded=encoded)
            assert all(torch.isfinite(value).all() for value in encoded.values())
            assert prediction["final_logit"].shape == (1,) and torch.isfinite(prediction["direction_logit"]).all()
    assert all(len({graph_hashes[(arm, fold)] for arm in ARMS}) == 3 for fold in range(5))
    current, _ = prepare.load_current_fold(source_records[0], tmp_path)
    with pytest.raises(ValueError, match="seed"):
        load_declared_binding_expansion(torch, payload=current, config=config,
            input_manifest_path=input_path, fold=0, root=tmp_path, task_seed=1)
    sidecar_path = paths["binding_expansion"]
    with sidecar_path.open("ab") as handle: handle.write(b"changed")
    with pytest.raises(ValueError, match="hash/size"):
        load_declared_binding_expansion(torch, payload=current, config=config,
            input_manifest_path=input_path, fold=0, root=tmp_path)
