"""Actual methylation production and frozen native CNV arrays into prepared G2."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from cc_hhgt.v32 import omics_node_inputs as inputs
from cc_hhgt.v32.formal_expression_sources import read, write, record
from test_v32_native_methylation_workflow import completed as methylation_completed
from test_v32_typed_graph_preparation import normalized, scoped, native_authority, prepared, patch_controls, no_fit


def _native_config(completed, root):
    _, _, state, _ = completed
    patients = ["TCGA-AA-0001", "TCGA-AA-0002"]
    authority = root / "patients.parquet"
    # A has two declared aliquots; its already aggregated patient value must
    # count once, not twice, when these native patient tables are joined.
    pd.DataFrame({"sample_id": ["a1", "a2", "b"], "patient_id": [patients[0], *patients],
        "cancer_id": "BRCA", "is_tumor": True, "partition": "train"}).to_parquet(authority, index=False)
    mapping = root / "nodes.parquet"
    pd.DataFrame({"feature_id": ["LNC:Z", "LNC:Y"], "node_type": "lncRNA",
                  "canonical_id": ["LNC:Z", "LNC:Y"]}).to_parquet(mapping, index=False)
    cnv = root / "cnv"; cnv.mkdir()
    write(cnv / "patient_ids.json", patients); write(cnv / "lncrna_ids.json", ["LNC:Z"])
    np.save(cnv / "lncrna_signed_mean.npy", np.array([[-.4], [.8]], dtype=np.float32))
    np.save(cnv / "lncrna_callable.npy", np.ones((2, 1), dtype=np.uint8))
    write(cnv / "SUCCESS.json", {"format": "CC_HHGT_V3_2_DIRECTIONAL_SEGMENT_CNV_V1",
        "status": "SUCCESS", "cancer_id": "BRCA", "signed_values_preserved": True,
        "amplification_deletion_separate": True, "absolute_burden_used_as_continuous": False,
        "files": {p.name: record(p)["sha256"] for p in cnv.iterdir()},
        "synthetic_frozen_matrix_fixture": True})
    return {"format": inputs.FORMAT, "target_host": "149", "outer_fold": 4, "scope": "cohort",
        "cancer_id": "BRCA", "sample_authority": record(authority), "assays": [
            {"channel": "methylation_promoter_beta", "unit": "beta", "layout": "native_methylation_patient",
             "measurements": state["results"]["recovered"], "mapping": record(mapping)},
            {"channel": "cnv_signed", "unit": "log2_copy_ratio", "layout": "native_directional_cnv",
             "measurements": record(cnv / "SUCCESS.json"), "mapping": record(mapping)}]}


def test_native_products_pass_formal_prepared_and_export_embedding(
    methylation_completed, prepared, native_authority, scoped, monkeypatch, tmp_path
):
    from cc_hhgt.v32.native_graph_input import NATIVE_MODE, load_training_graph_inputs
    from cc_hhgt.v32.training import _validate_prepared, _runtime_chunk_contract, _runtime_graph_for_chunk
    from cc_hhgt.v32.current_core_export import stream_embeddings, node_frame
    from cc_hhgt.v32.model import build_v32_cc_hhgt_residual
    from cc_hhgt.gnn import build_model
    from scripts.export_v32_current_core_embeddings import _validated_payload, _build_model
    patch_controls(monkeypatch, scoped.parent / "patient_map.tsv", native_authority[0] / "patients.json")
    config = _native_config(methylation_completed, tmp_path)
    config_path = tmp_path / "producer.json"; write(config_path, config)
    evidence = inputs.produce(config_path, tmp_path / "node_inputs")
    manifest = read(evidence)
    assert all(v["input_granularity"] == "patient" for v in manifest["coverage"].values())
    model_config = {"task_contract": {"rbp_evidence_mode": NATIVE_MODE, "graph_variant": "G2",
        "node_omics_channels": manifest["channels"]}, "primary_model": {"hidden_channels": 96, "dropout": .2}}
    authority = read(prepared / "G2/INPUT_MANIFEST.json")
    row = next(r for r in authority["fold_inputs"] if r["fold"] == 4)
    row["node_omics"] = record(evidence)
    auth = tmp_path / "INPUT_MANIFEST.json"; write(auth, authority)
    original = torch.load(row["path"], map_location="cpu", weights_only=False)
    loaded, _ = load_training_graph_inputs(torch, payload=original, config=model_config,
        input_manifest_path=auth, fold=4, root=prepared, prepared_sha256=row["sha256"])
    _validate_prepared(loaded, fold=4, artifact_hashes=original["artifact_hashes"])
    index = loaded["bundle"].node_maps["lncRNA"]["LNC:Z"]
    # Original methylation producer: mean(mean(.2,.6),.9) = .65.
    # Native CNV: equal patient mean(-.4,.8) = .2, not aliquot-weighted zero.
    np.testing.assert_allclose(loaded["bundle"].node_omics["lncRNA"][index], [.65, 1., .2, 1.], atol=1e-7)
    cfg_path = tmp_path / "model.yaml"; cfg_path.write_text(yaml.safe_dump(model_config))
    receipt = tmp_path / "receipt.json"
    write(receipt, {"synthetic_no_optimizer": True, "artifact_hashes": original["artifact_hashes"],
                    "input_authority_hashes": loaded["input_authority_hashes"]})
    paths = {"prepared": Path(row["path"]), "config": cfg_path, "input_manifest": auth, "training_receipt": receipt}
    exported = _validated_payload(paths, {k: record(v) for k, v in paths.items()}, 4)
    encoder = build_model("cc_hhgt", loaded["bundle"], loaded["feature_dim"], loaded["legacy_model_config"])
    original_model = build_v32_cc_hhgt_residual(encoder, hidden_channels=96, context_features=4, dropout=.2)
    model = _build_model(exported, model_config, {"model_state": original_model.state_dict()}, "cpu")
    chunks, weights = _runtime_chunk_contract(exported["bundle"])
    embeddings, _ = stream_embeddings(model, exported["bundle"], chunks, weights, _runtime_graph_for_chunk)
    frame, _ = node_frame(embeddings["lncRNA"], exported["bundle"].node_maps["lncRNA"])
    assert "LNC:Z" in set(frame.node_id)
    assert np.isfinite(frame.filter(like="core_feature_").to_numpy()).all()
    frame.to_parquet(tmp_path / "native_lncrna_embeddings.parquet", index=False)


@pytest.mark.parametrize("fault", ["incomplete", "matrix_drift", "availability", "shape", "wrong_cancer", "heldout"])
def test_native_products_refuse_incomplete_or_misaligned_sources(methylation_completed, tmp_path, fault):
    config = _native_config(methylation_completed, tmp_path)
    if fault == "heldout":
        config["outer_fold"] = 0
    else:
        success = Path(config["assays"][1]["measurements"]["path"])
        receipt = read(success)
        if fault == "incomplete": receipt["status"] = "INCOMPLETE"
        if fault == "wrong_cancer": receipt["cancer_id"] = "LUAD"
        if fault in {"matrix_drift", "availability", "shape"}:
            name = "lncrna_callable.npy" if fault == "availability" else "lncrna_signed_mean.npy"
            value = np.zeros((2, 1)) if fault == "availability" else np.zeros((1, 1))
            np.save(success.parent / name, value)
            if fault != "matrix_drift": receipt["files"][name] = record(success.parent / name)["sha256"]
        write(success, receipt)
        config["assays"][1]["measurements"] = record(success)
    expected = {"incomplete": "NODE_INPUT_NATIVE_CNV_INCOMPLETE", "matrix_drift": "Artifact SHA256 mismatch",
                "availability": "NODE_INPUT_NATIVE_CNV_SHAPE_OR_AVAILABILITY", "shape": "NODE_INPUT_NATIVE_CNV_SHAPE_OR_AVAILABILITY",
                "wrong_cancer": "NODE_INPUT_NATIVE_PATIENT_CANCER_MISMATCH", "heldout": "NODE_INPUT_NATIVE_HELDOUT_PATIENT_IN_TRAIN"}
    with pytest.raises(ValueError, match=expected[fault]):
        inputs._prepare(config, tmp_path)
