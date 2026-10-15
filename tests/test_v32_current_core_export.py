"""Synthetic inference only: no optimizer or real checkpoint is executed."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from cc_hhgt.v32.current_core_export import (EXPORT_FORMAT, CHECKPOINT_FORMAT, ESTIMAND,
    stream_embeddings, validate_checkpoint, validate_current_export, verified_path, sha256)


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor([2., 3.]))
        self.calls = []

    def encode(self, graph):
        assert not self.training and not torch.is_grad_enabled()
        self.calls.append(graph)
        result = torch.tensor([[1., 2.], [3., 4.]]) * self.scale + graph
        return {"lncRNA": result, "pathway": result + 1}


def tiny():
    model = torch.nn.Module(); model.encoder = TinyEncoder()
    bundle = SimpleNamespace(node_maps={"lncRNA": {"LNC:L1": 0, "LNC:L2": 1},
                                        "pathway": {"PATH:P1": 0, "PATH:P2": 1}})
    return model, bundle


def test_registered_weighted_sum_and_parameter_freeze():
    model, bundle = tiny()
    before = model.encoder.scale.detach().clone()
    values, fingerprint = stream_embeddings(model, bundle, (0, 8), {0: .25, 8: .75},
                                             lambda bundle, chunk, device: chunk)
    expected = np.array([[8., 12.], [12., 18.]])
    np.testing.assert_allclose(values["lncRNA"].numpy(), expected)
    assert model.encoder.calls == [0, 8]
    assert torch.equal(model.encoder.scale, before)
    assert not model.encoder.scale.requires_grad and model.encoder.scale.grad is None
    assert len(fingerprint) == 64


@pytest.mark.parametrize("weights", [{0: .4}, {0: 1., 8: -1.}, {0: float("nan"), 8: 1.}])
def test_partial_or_bad_weights_rejected_before_encoding(weights):
    model, bundle = tiny()
    with pytest.raises(ValueError):
        stream_embeddings(model, bundle, (0, 8), weights, lambda b, c, device: c)
    assert not model.encoder.calls


@pytest.mark.parametrize("fault", ["missing_type", "shape", "nan", "alignment"])
def test_no_silent_chunk_or_node_drift(fault):
    model, bundle = tiny()
    original = model.encoder.encode
    def encode(graph):
        result = original(graph)
        if fault == "missing_type" and graph == 1: result.pop("pathway")
        if fault == "shape" and graph == 1: result["pathway"] = result["pathway"][:, :1]
        if fault == "nan": result["lncRNA"][0, 0] = float("nan")
        return result
    model.encoder.encode = encode
    if fault == "alignment": bundle.node_maps["lncRNA"]["LNC:L2"] = 3
    with pytest.raises(ValueError):
        stream_embeddings(model, bundle, (0, 1), {0: .5, 1: .5}, lambda b, c, device: c)


def test_artifact_hash_is_enforced(tmp_path):
    path = tmp_path / "source.json"; path.write_text("{}")
    record = {"path": path.name, "sha256": sha256(path)}
    assert verified_path(record, tmp_path) == path
    path.write_text("[]")
    with pytest.raises(ValueError, match="SHA256 mismatch"): verified_path(record, tmp_path)


def export_fixture(tmp_path):
    manifest = {"export_format": EXPORT_FORMAT, "status": "PASS", "core_run_id": "current-G2",
        "analysis_version": "CancerLncAtlas_V3.2_CURRENT_CORE", "training_generation": "V3.2",
        "graph_variant": "G2", "embedding_estimand": ESTIMAND,
        "all_registered_chunks_exported": True, "labels_used_for_export": False,
        "optimizer_started_during_export": False, "folds": {}}
    for fold in range(5):
        receipt = {"status": "SUCCESS", "run_id": "current-G2", "patient_fold": fold,
            "graph_variant": "G2", "best_model_state_sha256": str(fold)*64,
            "task_id": f"current-G2__PATIENT_FOLD_{fold}", "optimizer_steps": 10,
            "parameter_delta_positive": True, "validation_full_runtime_coverage": True,
            "runtime_chunk_weights": {"0": .25, "8": .75}, "runtime_num_chunks": 2}
        receipt_path = tmp_path / f"receipt_{fold}.json"; receipt_path.write_text(json.dumps(receipt))
        exports = {}
        for kind, ids in [("lncRNA", ["LNC:L1", "LNC:L2"]), ("pathway", ["PATHWAY:P1", "PATHWAY:P2"]),
                          ("gene", ["GENE:G1", "GENE:G2"]), ("cancer", ["CANCER:BRCA", "CANCER:LUAD"])]:
            frame = pd.DataFrame({"node_index": [0, 1], "node_id": ids,
                                  "core_feature_000": [1., 2.], "core_feature_001": [3., 4.]})
            path = tmp_path / f"{fold}_{kind}.parquet"; frame.to_parquet(path, index=False)
            exports[kind] = {"path": path.name, "sha256": sha256(path), "rows": 2, "features": 2}
        manifest["folds"][str(fold)] = {"patient_fold": fold, "checkpoint_format": CHECKPOINT_FORMAT,
            "checkpoint_sha256": str(fold)*64, "core_parameter_sha256": "a"*64,
            "core_unchanged_after_export": True, "runtime_chunks_total": 2,
            "chunk_weights": receipt["runtime_chunk_weights"], "exports": exports,
            "training_receipt": {"path": receipt_path.name, "sha256": sha256(receipt_path)}}
    path = tmp_path / "CORE_EMBEDDING_MANIFEST.json"; path.write_text(json.dumps(manifest))
    return path, manifest


def test_current_export_consumed_by_native_heads(tmp_path):
    from cc_hhgt.v32.genomic_training import _validate_core_manifest, load_fold_core_embeddings
    from cc_hhgt.v32.drug_training import _validate_core_manifest as drug_validate
    from cc_hhgt.v32.drug_training import load_fold_core_embeddings as drug_load
    from cc_hhgt.v32.state_training import load_v32_core_embeddings
    path, manifest = export_fixture(tmp_path)
    assert _validate_core_manifest(path)[0]["core_run_id"] == "current-G2"
    assert drug_validate(path)[0]["core_run_id"] == "current-G2"
    core = load_fold_core_embeddings(path, manifest, 0)
    assert core.lncrna.values.shape == (2, 2)
    assert drug_load(path, manifest, 0).lncrna.values.shape == (2, 2)
    embeddings, lineage, source = load_v32_core_embeddings(path, repo_root=tmp_path / "unrelated_checkout")
    assert set(embeddings) == set(range(5))
    assert lineage[0]["core_run_id"] == "current-G2"
    assert lineage[0]["core_checkpoint_format"] == CHECKPOINT_FORMAT


@pytest.mark.parametrize("fault", ["fold", "run", "weights", "receipt", "frozen"])
def test_incomplete_or_wrong_current_export_rejected(tmp_path, fault):
    path, manifest = export_fixture(tmp_path)
    if fault == "fold": manifest["folds"].pop("4")
    if fault == "run": manifest["core_run_id"] = "old-core"
    if fault == "weights": manifest["folds"]["0"]["chunk_weights"] = {"0": .5, "8": .5}
    if fault == "receipt": (tmp_path / "receipt_0.json").write_text("{}")
    if fault == "frozen": manifest["folds"]["0"]["core_unchanged_after_export"] = False
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError): validate_current_export(path, expected_run_id="current-G2")


def test_checkpoint_and_manifest_binding():
    receipt = {"patient_fold": 0, "task_id": "current0", "graph_variant": "G2",
        "artifact_hashes": {"config_sha256": "c"*64, "input_manifest_sha256": "d"*64},
        "input_authority_hashes": {"data": "f"*64}}
    checkpoint = {**copy.deepcopy(receipt), "checkpoint_format": CHECKPOINT_FORMAT,
                  "prepared_artifact_sha256": "e"*64}
    kwargs = dict(fold=0, prepared_sha="e"*64, config_sha="c"*64, input_manifest_sha="d"*64)
    validate_checkpoint(checkpoint, receipt, **kwargs)
    checkpoint["input_authority_hashes"]["data"] = "b"*64
    with pytest.raises(ValueError, match="input_authority_hashes"):
        validate_checkpoint(checkpoint, receipt, **kwargs)


def test_command_exports_five_folds_and_native_reader_consumes_them(tmp_path, monkeypatch):
    import scripts.export_v32_current_core_embeddings as runner
    import cc_hhgt.v32.training as training
    from cc_hhgt.v32.genomic_training import _validate_core_manifest, load_fold_core_embeddings

    def record(path): return {"path": path.name, "sha256": sha256(path)}
    inputs = {"format": "CURRENT_CORE_EXPORT_INPUT_V1", "core_run_id": "current-G2",
              "graph_variant": "G2", "fold_inputs": []}
    for fold in range(5):
        paths = {key: tmp_path / f"{fold}_{key}.json" for key in
                 ["config", "input_manifest", "prepared", "checkpoint", "training_receipt"]}
        for key in ["config", "input_manifest", "prepared"]: paths[key].write_text("{}")
        authority = {"config_sha256": sha256(paths["config"]),
                     "input_manifest_sha256": sha256(paths["input_manifest"])}
        checkpoint = {"checkpoint_format": CHECKPOINT_FORMAT, "patient_fold": fold,
            "task_id": f"current-G2__PATIENT_FOLD_{fold}", "graph_variant": "G2",
            "artifact_hashes": authority, "input_authority_hashes": {"fixture": "a"*64},
            "prepared_artifact_sha256": sha256(paths["prepared"])}
        torch.save(checkpoint, paths["checkpoint"])
        receipt = {**checkpoint, "status": "SUCCESS", "run_id": "current-G2",
            "best_model_state_sha256": sha256(paths["checkpoint"]), "optimizer_steps": 10,
            "parameter_delta_positive": True, "validation_full_runtime_coverage": True,
            "runtime_chunk_weights": {"0": .25, "8": .75}, "runtime_num_chunks": 2}
        paths["training_receipt"].write_text(json.dumps(receipt))
        inputs["fold_inputs"].append({"fold": fold, **{key: record(path) for key, path in paths.items()}})
    manifest_path = tmp_path / "inputs.json"; manifest_path.write_text(json.dumps(inputs))
    # Keep orchestration, serialization, receipt checks, aggregation and consumers
    # real. Substitute only the large scientific graph/model with deterministic
    # hand-built tensors. Receipt counts above describe the synthetic source.
    monkeypatch.setattr(runner, "_validated_payload", lambda *args: {"bundle": tiny()[1]})
    monkeypatch.setattr(runner, "_build_model", lambda *args: tiny()[0])
    monkeypatch.setattr(training, "_runtime_chunk_contract", lambda b: ((0, 8), {0: .25, 8: .75}))
    monkeypatch.setattr(training, "_runtime_graph_for_chunk", lambda b, c, device: c)
    result = tmp_path / "export"
    assert runner.main(["--manifest", str(manifest_path), "--expected-core-run-id", "current-G2",
                        "--output", str(result)]) == 0
    exported, _, _ = _validate_core_manifest(result / "CORE_EMBEDDING_MANIFEST.json")
    core = load_fold_core_embeddings(result / "CORE_EMBEDDING_MANIFEST.json", exported, 4)
    np.testing.assert_allclose(core.lncrna.values, [[8., 12.], [12., 18.]])
    assert not json.loads((result / "SUCCESS.json").read_text())["training_started"]
    assert len(exported["folds"]) == 5


def test_training_receipt_requires_optimizer_evidence(tmp_path):
    path, manifest = export_fixture(tmp_path)
    receipt_path = tmp_path / "receipt_0.json"
    receipt = json.loads(receipt_path.read_text()); receipt["optimizer_steps"] = 0
    receipt_path.write_text(json.dumps(receipt))
    manifest["folds"]["0"]["training_receipt"]["sha256"] = sha256(receipt_path)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="optimizer work"):
        validate_current_export(path)


def test_clinical_feature_entry_uses_current_core_hashes_and_fold_local_scaler(tmp_path):
    from scripts.run_v32_clinical_training import _build_fold_features, _load_core_tables
    _, manifest = export_fixture(tmp_path)
    ids = [f"person{i}" for i in range(10)]
    folds = pd.DataFrame({"cancer_id": "BRCA", "patient_id": ids, "patient_fold_id": np.arange(10) % 5})
    covariates = pd.DataFrame({"cancer_id": "BRCA", "patient_id": ids, "age_years": np.arange(10)+40})
    for name, entity, column in [("expression", "lncrna_id", "logcpm"), ("activity", "pathway_id", "activity_score")]:
        folder = tmp_path / name / "cancer_id=BRCA"; folder.mkdir(parents=True)
        frame = pd.DataFrame({"patient_id": ids, entity: "L1" if name == "expression" else "P1",
                              column: np.arange(10, dtype=float)})
        frame.to_parquet(folder / "part-0.parquet", index=False)
    args = (0, folds, covariates, tmp_path/"expression", tmp_path/"activity", tmp_path, manifest)
    before = _build_fold_features(*args).set_index("patient_id")
    expression_path = tmp_path / "expression/cancer_id=BRCA/part-0.parquet"
    expression = pd.read_parquet(expression_path); expression.loc[0, "logcpm"] = 999.
    expression.to_parquet(expression_path, index=False)
    after = _build_fold_features(*args).set_index("patient_id")
    assert len(after) == 10
    pd.testing.assert_frame_equal(before.drop(index="person0"), after.drop(index="person0"))
    assert before.loc["person0", "core_lnc_000"] != after.loc["person0", "core_lnc_000"]
    assert before["core_cancer_000"].eq(1.).all()
    path = tmp_path / manifest["folds"]["0"]["exports"]["lncRNA"]["path"]
    path.write_bytes(b"not the declared embedding")
    with pytest.raises(ValueError, match="SHA256 mismatch"): _load_core_tables(tmp_path, manifest, 0)


def test_clinical_node_alias_collision_is_not_averaged():
    from cc_hhgt.v32.clinical_training import project_measurements_to_core, ClinicalTrainingError
    measurement = pd.DataFrame({"patient_id": ["a", "b"], "lncrna_id": "L1", "logcpm": [1., 2.]})
    embedding = pd.DataFrame({"node_id": ["LNC:L1", "L1"], "core_feature_000": [1., 2.]})
    with pytest.raises(ClinicalTrainingError, match="collide"):
        project_measurements_to_core(measurement, embedding, entity_column="lncrna_id", value_column="logcpm",
                                     train_patient_ids=["a", "b"], output_prefix="core")


@pytest.mark.parametrize("fault", [None, "core", "input", "config", "file"])
def test_clinical_resume_cannot_mix_runs_or_outputs(tmp_path, fault):
    from scripts.run_v32_clinical_training import _validate_resume_receipt
    paths = {key: tmp_path / (key+".bin") for key in ["public", "private", "checkpoint"]}
    for path in paths.values(): path.write_bytes(b"fixture")
    meta = {"status": "SUCCESS", "training_run_id": "clinical-current", "core_manifest_sha256": "c"*64,
            "input_lineage_sha256": "d"*64, "training_config_sha256": "e"*64,
            "public_prediction_sha256": sha256(paths["public"]), "private_prediction_sha256": sha256(paths["private"]),
            "checkpoint_sha256": sha256(paths["checkpoint"])}
    args = dict(training_run_id="clinical-current", core_manifest_sha="c"*64, input_lineage_sha="d"*64,
                config_sha="e"*64, public_path=paths["public"], private_path=paths["private"], checkpoint_path=paths["checkpoint"])
    if fault == "core": meta["core_manifest_sha256"] = "b"*64
    if fault == "input": meta["input_lineage_sha256"] = "b"*64
    if fault == "config": meta["training_config_sha256"] = "b"*64
    if fault == "file": paths["private"].write_bytes(b"changed")
    if fault:
        with pytest.raises(RuntimeError): _validate_resume_receipt(meta, **args)
    else:
        _validate_resume_receipt(meta, **args)
