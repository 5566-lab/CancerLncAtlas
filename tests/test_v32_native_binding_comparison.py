"""Actual native prepared files through comparison, trainer loading and export."""
import copy
from pathlib import Path
import socket

import pandas as pd
import pytest
import torch
import yaml

from cc_hhgt.v32.formal_expression_sources import read, write, record
from cc_hhgt.v32.native_graph_input import NATIVE_MODE, validate_native_base, load_training_graph_inputs
from cc_hhgt.v32.current_binding_graph import ARMS
from test_v32_typed_graph_preparation import normalized, scoped, native_authority, prepared, patch_controls
from test_v32_fold_local_expression import forbidden


@pytest.fixture(autouse=True)
def no_fit(monkeypatch):
    assert socket.gethostname() == "149"
    for name in ("Adam", "AdamW", "SGD"): monkeypatch.setattr(torch.optim, name, forbidden)
    from sklearn.linear_model import LogisticRegression
    monkeypatch.setattr(LogisticRegression, "fit", forbidden)


@pytest.fixture(scope="module")
def comparison(prepared, native_authority, scoped):
    import scripts.prepare_v32_current_binding_comparison as command
    from test_v32_current_binding_expansion import native_source, source
    root = native_authority[0] / "native_comparison"; root.mkdir()
    base_manifest = prepared / "G2/INPUT_MANIFEST.json"
    rows = read(base_manifest)["fold_inputs"]
    assert len(rows) == 5 and all(r["graph_input_mode"] == NATIVE_MODE for r in rows)
    config = root / "config.yaml"
    config.write_text(yaml.safe_dump({"task_contract": {"rbp_evidence_mode": NATIVE_MODE, "graph_variant": "G2"},
        "runtime_profile": {"max_coverage_cycles": 2, "device": "cpu", "target_host": "149", "mixed_precision": False},
        "primary_model": {"hidden_channels": 96, "dropout": .2}, "training_io": {}}))
    edges, lineage = native_source()
    edges.to_parquet(root / "edges.parquet", index=False)
    lineage.to_parquet(root / "lineage.parquet", index=False)
    binding = root / "rbns.parquet"
    source([("LNC:Z", "UNIPROT:A", "predicted", .4)]).to_parquet(binding, index=False)
    audit = root / "rbns_audit.json"
    write(audit, {"format": "RBNS_SEQUENCE_ANALYSIS_AUDIT_V1", "status": "PASS",
        "annotation_id": "synthetic", "lncrna_binding_measured": False,
        "outputs": {"RBNS_TYPED_BINDING.parquet": record(binding)}})
    manifest = root / "input.json"
    write(manifest, {"format": "CURRENT_C_BINDING_COMPARISON_INPUT_V1", "config": record(config),
        "fold_inputs": [{"fold": r["fold"], "graph_input_mode": NATIVE_MODE,
            "prepared": {"path": r["path"], "sha256": r["sha256"]}, "input_manifest": record(base_manifest)} for r in rows],
        "sources": [{"source_id": "native", "kind": "native_october", "edges": record(root / "edges.parquet"),
                     "lineage": record(root / "lineage.parquet")},
                    {"source_id": "rbns", "kind": "rbns_sequence", "audit": record(audit), "binding": record(binding)}]})
    with pytest.MonkeyPatch.context() as patch:
        patch_controls(patch, scoped.parent / "patient_map.tsv", native_authority[0] / "patients.json")
        for name in ("Adam", "AdamW", "SGD"): patch.setattr(torch.optim, name, forbidden)
        assert command.main(["--manifest", str(manifest), "--output", str(root / "output"), "--release", "native-synthetic"]) == 0
    return root


@pytest.mark.parametrize("fold", range(5))
@pytest.mark.parametrize("arm", ARMS)
def test_actual_native_three_arm_training_loader_and_frozen_export(comparison, native_authority, scoped, monkeypatch, fold, arm):
    from cc_hhgt.v32.training import (load_prepared_artifact_from_authorized_handle, _validate_prepared,
        _prepared_input_authority_hashes, _runtime_graph_for_chunk, validate_formal_graph_variant_binding)
    from cc_hhgt.v32.current_binding_overlay import non_graph_payload_sha
    from cc_hhgt.v32.integrated_model import module_state_sha256
    from cc_hhgt.gnn import build_model
    from cc_hhgt.v32.model import build_v32_cc_hhgt_residual
    from scripts.export_v32_current_core_embeddings import _validated_payload, _build_model
    patch_controls(monkeypatch, scoped.parent / "patient_map.tsv", native_authority[0] / "patients.json")
    folder = comparison / "output" / arm
    config_path = folder / "config.yaml"; config = yaml.safe_load(config_path.read_text())
    manifest = folder / "INPUT_MANIFEST.json"; authorized = read(manifest)["fold_inputs"][fold]
    path = Path(authorized["path"])
    assert "graph_overlay" not in authorized and "c_graph_overlay_pattern" not in config["training_io"]
    payload, digest = load_prepared_artifact_from_authorized_handle(torch, manifest, fold=fold, prepared_path=path)
    base = payload
    payload, overlay_active = load_training_graph_inputs(torch, payload=payload, config=config,
        input_manifest_path=manifest, fold=fold, root=comparison, prepared_sha256=digest, task_seed=20260726)
    assert not overlay_active and "graph_overlay_authority" not in payload
    assert payload["binding_expansion_authority"]["verified"] is True
    assert all(payload["input_authority_hashes"][k] == v for k, v in _prepared_input_authority_hashes(base).items())
    _validate_prepared(payload, fold=fold, artifact_hashes=payload["artifact_hashes"])
    validate_formal_graph_variant_binding(run_id="v32-g012-g2-native-comparison", config=config,
        payload=payload, prepared_path=path)
    assert payload["train_batches"] is base["train_batches"] and payload["validation_batches"] is base["validation_batches"]
    assert non_graph_payload_sha(payload) == non_graph_payload_sha(base)
    for kind, nodes in base["bundle"].node_maps.items():
        assert all(payload["bundle"].node_maps[kind][k] == v for k, v in nodes.items())
    predicted = payload["bundle"].edges.relation_type.eq("binds_protein_predicted")
    assert predicted.any() == (arm == "experimental_predicted")
    schema = payload["bundle"].relation_schema_edges
    assert set(schema.loc[schema.relation_type.eq("expressed_in"), "edge_role"]) == {"fold_train_expression_eligibility"}
    # This is a source-binding receipt for untrained synthetic serialization,
    # not a claim of a completed selected training run.
    receipt = folder / f"synthetic_receipt_{fold}.json"
    write(receipt, {"synthetic_no_optimizer": True, "artifact_hashes": payload["artifact_hashes"],
                   "input_authority_hashes": _prepared_input_authority_hashes(payload)})
    paths = {"prepared": path, "config": config_path, "input_manifest": manifest,
             "binding_expansion": Path(authorized["binding_expansion"]["path"]), "training_receipt": receipt}
    records = {k: record(v) for k, v in paths.items()}
    exported = _validated_payload(paths, records, fold)
    assert non_graph_payload_sha(exported) == non_graph_payload_sha(base)
    encoder = build_model("cc_hhgt", payload["bundle"], payload["feature_dim"], payload["legacy_model_config"])
    original = build_v32_cc_hhgt_residual(encoder, hidden_channels=96, context_features=4, dropout=.2)
    model = _build_model(exported, config, {"model_state": original.state_dict()}, "cpu").eval()
    state = module_state_sha256(model)
    with torch.inference_mode():
        graph = _runtime_graph_for_chunk(exported["bundle"], 0, device="cpu")
        batch = exported["train_batches"][0]
        result = model(graph, batch["candidate_batch"], batch["base_logit"], batch["conservation_context"])
    assert result["final_logit"].shape == (3,) and all(torch.isfinite(v).all() for v in result.values())
    assert module_state_sha256(model) == state


def test_generated_native_commands_stay_code_only(comparison):
    from cc_hhgt.v32.cli import main as pipeline
    output = comparison / "output"
    document = read(output / "COMPARISON.json")
    assert document["base_graph_input_mode"] == NATIVE_MODE and len(document["commands"]) == 15
    assert read(output / "SUCCESS.json")["sidecars"] == 15
    assert pipeline(document["commands"][0]["argv"][2:]) != 0
    for arm in ARMS:
        config = yaml.safe_load((output / arm / "config.yaml").read_text())
        assert config["execution_control"]["training_authorized"] is False
        assert not (output / arm / "training_outputs").exists()


@pytest.mark.parametrize("fault", ["legacy_cohort", "generation", "binding_source", "trained", "overlay_metadata"])
def test_native_mode_rejects_incompatible_bases(prepared, native_authority, scoped, monkeypatch, fault):
    patch_controls(monkeypatch, scoped.parent / "patient_map.tsv", native_authority[0] / "patients.json")
    payload = torch.load(prepared / "G2/PATIENT_FOLD_0.pt", weights_only=False)
    binding = payload["formal_graph_authority"]
    if fault == "legacy_cohort": binding["gates"]["cohort_detection_used"] = True
    elif fault == "generation": binding["binding_generation"] = "GENERIC_GLOBAL"
    elif fault == "binding_source": binding["typed_binding_preparation"]["binding_source"]["sha256"] = "a" * 64
    elif fault == "trained": payload["contains_trained_parameters"] = True
    else: payload["graph_overlay_authority"] = {"verified": True}
    with pytest.raises(ValueError, match="Native direct G2"): validate_native_base(payload, 0)


@pytest.mark.parametrize("fault", ["mode", "mixed_fold", "historical_overlay"])
def test_comparison_cli_rejects_mixed_mode_before_materialization(comparison, tmp_path, fault):
    import scripts.prepare_v32_current_binding_comparison as command
    spec = copy.deepcopy(read(comparison / "input.json"))
    if fault == "mode":
        config = yaml.safe_load(Path(spec["config"]["path"]).read_text())
        config["task_contract"]["rbp_evidence_mode"] = "C_TYPED_ENCODE_V2"
        path = tmp_path / "config.yaml"; path.write_text(yaml.safe_dump(config)); spec["config"] = record(path)
    elif fault == "mixed_fold": spec["fold_inputs"][2].pop("graph_input_mode")
    else: spec["fold_inputs"][0]["graph_overlay"] = spec["fold_inputs"][0]["prepared"]
    manifest = tmp_path / "input.json"; write(manifest, spec)
    with pytest.raises(ValueError, match="graph input mode|mix historical"):
        command.main(["--manifest", str(manifest), "--output", str(tmp_path / "output"), "--release", "invalid"])
    assert not (tmp_path / "output").exists()


def test_native_runtime_rejects_undeclared_mode_and_overlay_pattern(comparison, native_authority, scoped, tmp_path, monkeypatch):
    patch_controls(monkeypatch, scoped.parent / "patient_map.tsv", native_authority[0] / "patients.json")
    folder = comparison / "output/current_c"; manifest = folder / "INPUT_MANIFEST.json"
    config = yaml.safe_load((folder / "config.yaml").read_text())
    declaration = read(manifest); item = declaration["fold_inputs"][0]
    payload = torch.load(item["path"], weights_only=False)
    declaration["fold_inputs"][0].pop("graph_input_mode")
    altered = tmp_path / "input.json"; write(altered, declaration)
    with pytest.raises(ValueError, match="mode differs"):
        load_training_graph_inputs(torch, payload=payload, config=config, input_manifest_path=altered,
            fold=0, root=comparison, prepared_sha256=item["sha256"])
    missing_mode = copy.deepcopy(config); missing_mode["task_contract"].pop("rbp_evidence_mode")
    with pytest.raises(ValueError, match="mode differs"):
        load_training_graph_inputs(torch, payload=payload, config=missing_mode, input_manifest_path=manifest,
            fold=0, root=comparison, prepared_sha256=item["sha256"])
    config["training_io"]["c_graph_overlay_pattern"] = "not_used_{fold}.pt"
    with pytest.raises(ValueError, match="cannot also declare"):
        load_training_graph_inputs(torch, payload=payload, config=config, input_manifest_path=manifest,
            fold=0, root=comparison, prepared_sha256=item["sha256"])
