"""Synthetic assay -> node -> real loader/encoder checks; optimizers forbidden."""
import copy
import json

import numpy as np
import pandas as pd
import pytest
import torch

from cc_hhgt.gnn import build_model, move_graph, runtime_bundle_for_step
from cc_hhgt.v32 import omics_node_encoding as encoding
from cc_hhgt.v32 import omics_node_inputs as inputs
from cc_hhgt.v32 import patient_conditioning as pc
from cc_hhgt.v32.native_graph_input import load_training_graph_inputs
from test_v32_omics_node_encoding import _bundle, _model_cfg


@pytest.fixture(autouse=True)
def no_fit(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("Parameter fitting is forbidden")
    for name in ("Adam", "AdamW", "SGD"):
        monkeypatch.setattr(torch.optim, name, denied)


def _write(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    return path


def _config(tmp_path):
    authority = pd.DataFrame([
        ("a1", "A", "C1", True, "train"),
        ("a2", "A", "C1", True, "train"),
        ("b", "B", "C1", True, "train"),
        ("v", "V", "C1", True, "validation"),
        ("t", "T", "C1", True, "test"),
        ("n", "N", "C1", False, "train"),
    ], columns=["sample_id", "patient_id", "cancer_id", "is_tumor", "partition"])
    ap = tmp_path / "samples.parquet"
    authority.to_parquet(ap, index=False)
    mapping = pd.DataFrame([("p1", "gene", "G1"), ("p2", "gene", "G1"),
                            ("p3", "lncRNA", "L1")],
                           columns=["feature_id", "node_type", "canonical_id"])
    mp = tmp_path / "mapping.parquet"
    mapping.to_parquet(mp, index=False)
    assays = []
    for channel, unit, vals in [
        ("methylation_promoter_beta", "beta", [0.0, .4, .8, .2, .1, .9, .9, 0.0, np.nan]),
        ("cnv_signed", "log2_copy_ratio", [-2.0, 0.0, 2.0, -1.0, 5., 5., 5., 0.0, np.nan])]:
        frame = pd.DataFrame(zip(["a1", "a1", "a2", "b", "v", "t", "n", "a1", "b"],
                                 ["p1", "p2", "p1", "p1", "p1", "p1", "p1", "p3", "p3"], vals),
                             columns=["sample_id", "feature_id", "raw_value"])
        p = tmp_path / (channel + ".parquet")
        frame.to_parquet(p, index=False)
        assays.append({"channel": channel, "unit": unit, "measurements": inputs._artifact(p),
                       "mapping": inputs._artifact(mp)})
    return {"format": inputs.FORMAT, "target_host": "149", "outer_fold": 0,
            "scope": "cohort", "cancer_id": "C1", "sample_authority": inputs._artifact(ap),
            "assays": assays}


def _produced(tmp_path, config=None):
    config = _config(tmp_path) if config is None else config
    path = inputs.produce(_write(tmp_path / "config.json", config), tmp_path / "output")
    auth = _write(tmp_path / "authority.json", {"fold_inputs": [{"fold": 0, "node_omics": inputs._artifact(path)}]})
    cfg = {"task_contract": {"node_omics_channels": [a["channel"] for a in config["assays"]]}}
    payload = {"bundle": _bundle(), "input_authority_hashes": {"base": "a" * 64},
               "train_batches": [object()], "validation_batches": [object()]}
    return path, auth, cfg, payload


def test_equal_patient_weight_missing_zero_and_heldout_exclusion(tmp_path):
    config = _config(tmp_path)
    frame, lineage, manifest = inputs._prepare(config, tmp_path)
    series = frame.set_index(["canonical_id", "channel"]).raw_value
    # A methylation: mean(mean(0,.4), .8) = .5; B=.2 -> .35.
    assert series["G1", "methylation_promoter_beta"] == pytest.approx(.35)
    # A CNV: mean(mean(-2,0),2)=.5; B=-1 -> -.25, preserves sign.
    assert series["G1", "cnv_signed"] == pytest.approx(-.25)
    assert series["L1", "methylation_promoter_beta"] == 0.0
    assert set(lineage.patient_id) == {"A", "B"}
    assert manifest["standardisation_policy"] == "IDENTITY_NO_PARAMETER_FIT"
    for assay in config["assays"]:
        path = assay["measurements"]["path"]
        raw = pd.read_parquet(path)
        raw.loc[raw.sample_id.isin(["v", "t", "n"]), "raw_value"] = .7
        raw.to_parquet(path, index=False)
        assay["measurements"] = inputs._artifact(path)
    altered, _, _ = inputs._prepare(config, tmp_path)
    pd.testing.assert_frame_equal(frame, altered)


def test_actual_training_loader_encoder_and_export_loader_preserve_legacy_inputs(tmp_path, monkeypatch):
    path, auth, cfg, payload = _produced(tmp_path)
    original = payload["bundle"].hetero_data["gene"].x.clone()
    loaded, overlay = load_training_graph_inputs(torch, payload=payload, config=cfg,
        input_manifest_path=auth, fold=0, root=tmp_path, prepared_sha256="b" * 64)
    assert not overlay and loaded["bundle"].node_input_dim == 6
    assert loaded["train_batches"] is payload["train_batches"]
    assert loaded["validation_batches"] is payload["validation_batches"]
    torch.testing.assert_close(payload["bundle"].hetero_data["gene"].x, original)
    block = loaded["bundle"].node_omics
    np.testing.assert_allclose(block["gene"][0], [.35, 1., -.25, 1.])
    np.testing.assert_array_equal(block["gene"][1], [0., 0., 0., 0.])
    np.testing.assert_array_equal(block["lncRNA"][0], [0., 1., 0., 1.])
    assert loaded["input_authority_hashes"]["node_omics_manifest_sha256"] == inputs.sha256(path)
    model = build_model("cc_hhgt", loaded["bundle"], 2, _model_cfg())
    assert encoding.assert_encoder_consumes_omics(model, loaded["bundle"])["verified"]
    with torch.no_grad():
        result = model.eval().encode(move_graph(loaded["bundle"], "cpu", "cc_hhgt"))
    assert torch.isfinite(result["lncRNA"]).all()
    # Export's real entry point calls this same loader; validate its wiring using
    # fixed receipts and bypass only unrelated historical graph authority checks.
    import scripts.export_v32_current_core_embeddings as export
    import cc_hhgt.v32.training as training
    monkeypatch.setattr(training, "_validate_prepared", lambda *args, **kwargs: None)
    prepared = tmp_path / "prepared.pt"
    torch.save(payload, prepared)
    prepared_record = inputs._artifact(prepared)
    _write(auth, {"fold_inputs": [{"fold": 0, "sha256": prepared_record["sha256"],
                                  "node_omics": inputs._artifact(path)}]})
    receipt = _write(tmp_path / "receipt.json", {"artifact_hashes": {},
                        "input_authority_hashes": loaded["input_authority_hashes"]})
    model_config = _write(tmp_path / "model.json", cfg)
    exported = export._validated_payload({"prepared": prepared, "input_manifest": auth,
        "config": model_config, "training_receipt": receipt}, {"prepared": prepared_record}, 0)
    np.testing.assert_array_equal(exported["bundle"].node_omics["gene"], block["gene"])
    assert exported["input_authority_hashes"] == loaded["input_authority_hashes"]


@pytest.mark.parametrize("fault", ["beta", "cnv_units", "duplicate", "sample", "feature", "split", "normal_only"])
def test_invalid_sources_fail_closed(tmp_path, fault):
    config = _config(tmp_path)
    if fault == "cnv_units":
        config["assays"][1]["unit"] = "absolute_copy_count"
    elif fault in {"split", "normal_only"}:
        p = config["sample_authority"]["path"]
        table = pd.read_parquet(p)
        if fault == "split": table.loc[1, "partition"] = "test"
        else: table["is_tumor"] = False
        table.to_parquet(p, index=False)
        config["sample_authority"] = inputs._artifact(p)
    else:
        assay = config["assays"][0]
        p = assay["measurements"]["path"]
        table = pd.read_parquet(p)
        if fault == "beta": table.loc[0, "raw_value"] = 1.1
        if fault == "duplicate": table = pd.concat([table, table.iloc[:1]])
        if fault == "sample": table.loc[0, "sample_id"] = "UNKNOWN"
        if fault == "feature": table.loc[0, "feature_id"] = "UNKNOWN"
        table.to_parquet(p, index=False)
        assay["measurements"] = inputs._artifact(p)
    with pytest.raises(ValueError, match="NODE_INPUT"):
        inputs._prepare(config, tmp_path)


@pytest.mark.parametrize("fault", ["channel_order", "fold", "source_drift", "repinned_value", "declaration", "cancer"])
def test_loader_rejects_source_and_contract_substitution(tmp_path, fault):
    path, auth, cfg, payload = _produced(tmp_path)
    manifest = json.loads(path.read_text())
    if fault == "channel_order": cfg["task_contract"]["node_omics_channels"].reverse()
    if fault == "fold":
        manifest["outer_fold"] = 1
        _write(path, manifest)
    if fault == "source_drift":
        source = manifest["source_artifacts"]["cnv_signed:measurements"]["path"]
        with open(source, "ab") as out: out.write(b"changed")
    if fault == "repinned_value":
        table = path.parent / manifest["table_path"]
        frame = pd.read_parquet(table)
        frame.loc[0, "raw_value"] += .01
        frame.to_parquet(table, index=False)
        manifest["table_sha256"] = inputs.sha256(table)
        _write(path, manifest)
    if fault == "declaration": cfg["task_contract"] = {}
    if fault == "cancer": payload["bundle"].node_maps["cancer"]["C2"] = 1
    _write(auth, {"fold_inputs": [{"fold": 0, "node_omics": inputs._artifact(path)}]})
    with pytest.raises((ValueError, AssertionError)):
        inputs.load_declared_node_inputs(payload, cfg, auth, 0)


def test_subject_output_is_not_cohort_training_input(tmp_path):
    config = _config(tmp_path)
    config.update(scope="subject", subject_id="V")
    path, auth, cfg, payload = _produced(tmp_path, config)
    evidence = encoding.read_evidence(path.parent / "node_evidence.parquet", path)
    assert evidence.scope == "subject" and evidence.subject_id == "V"
    with pytest.raises(ValueError, match="SCOPE"):
        inputs.load_declared_node_inputs(payload, cfg, auth, 0)
    bundle = payload["bundle"]
    bundle.node_omics_channels = tuple(reversed(evidence.channels))
    intervention = pc.Intervention(cancer_id="C1", subject_id="V", node_type="gene",
                                    node_id="G1", channel="cnv_signed", value=0.)
    with pytest.raises(RuntimeError, match="CHANNEL_LAYOUT"):
        pc.validate_intervention(intervention, evidence, bundle)


def test_producer_refuses_other_host_and_existing_output(tmp_path, monkeypatch):
    config = _write(tmp_path / "config.json", _config(tmp_path))
    monkeypatch.setattr(inputs.socket, "gethostname", lambda: "windows")
    with pytest.raises(RuntimeError, match="149"):
        inputs.produce(config, tmp_path / "output")
    monkeypatch.setattr(inputs.socket, "gethostname", lambda: "149")
    inputs.produce(config, tmp_path / "output")
    with pytest.raises(FileExistsError):
        inputs.produce(config, tmp_path / "output")


def test_conditioned_runtime_rebuild_and_exception_restore(tmp_path):
    path, auth, cfg, payload = _produced(tmp_path)
    bundle = inputs.load_declared_node_inputs(payload, cfg, auth, 0)["bundle"]
    # Force the real runtime rematerializer rather than its no-schedule shortcut.
    bundle.runtime_schedule = bundle.edges.copy()
    bundle.runtime_schedule["cancer_id"] = "C1"
    bundle.runtime_chunk_positions = {0: np.arange(len(bundle.edges))}
    bundle.runtime_num_chunks = 1
    original = bundle.node_omics
    block = copy.deepcopy(original)
    block["gene"][0, 0] = .91
    with pc.conditioned(bundle, block):
        rebuilt = runtime_bundle_for_step(bundle, 0)
        assert rebuilt.hetero_data["gene"].x[0, 2].item() == pytest.approx(.91)
    assert bundle.node_omics is original
    snapshot = {kind: bundle.hetero_data[kind].x.clone() for kind in bundle.node_maps}
    block["gene"] = np.zeros((99, 4))
    with pytest.raises((ValueError, RuntimeError)):
        with pc.conditioned(bundle, block):
            pytest.fail("Invalid rows must not reach the body")
    for kind, value in snapshot.items():
        torch.testing.assert_close(bundle.hetero_data[kind].x, value)
    assert bundle.node_omics is original


def test_pooled_features_are_explicit_and_subject_cancers_are_not_mixed(tmp_path):
    config = _config(tmp_path)
    config["cancer_id"] = inputs.POOL
    path = config["sample_authority"]["path"]
    authority = pd.read_parquet(path)
    authority.loc[authority.patient_id.eq("B"), "cancer_id"] = "C2"
    authority.to_parquet(path, index=False)
    config["sample_authority"] = inputs._artifact(path)
    _, _, manifest = inputs._prepare(config, tmp_path)
    assert manifest["cancer_ids"] == ["C1", "C2"]
    assert manifest["cancer_id"] == inputs.POOL
    config.update(scope="subject", subject_id="B")
    with pytest.raises(ValueError, match="SUBJECT_SCOPE"):
        inputs._prepare(config, tmp_path)
    config["cancer_id"] = "C1"
    with pytest.raises(ValueError, match="ELIGIBLE_PATIENTS_EMPTY"):
        inputs._prepare(config, tmp_path)
