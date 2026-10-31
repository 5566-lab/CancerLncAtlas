from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cla_engine.common import Stop, sha
from cla_engine.node_omics import CHANNELS, VerifiedGeneNodeOmics


def _record(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


def _product(root, heldout_cnv=9.0):
    root.mkdir()
    genes = pd.DataFrame({"gene_id": ["ENSG000001", "ENSG000002"]})
    patients = pd.DataFrame({"cancer_id": ["BRCA"] * 3, "patient_id": ["p0", "p1", "p2"],
                             "patient_fold_id": [2, 3, 0]})
    genes.to_parquet(root / "genes.parquet", index=False)
    patients.to_parquet(root / "patients.parquet", index=False)
    part = root / "BRCA"; part.mkdir()
    _write_json(part / "patient_ids.json", ["p0", "p1", "p2"])
    cnv = np.asarray([[0.0, np.nan], [2.0, np.nan], [heldout_cnv, 5.0]], dtype=np.float32)
    methyl = np.asarray([[0.2, np.nan], [0.4, np.nan], [0.8, 0.6]], dtype=np.float32)
    for channel, values in (("cnv_signed", cnv), ("methylation_promoter_beta", methyl)):
        np.save(part / f"{channel}.npy", values, allow_pickle=False)
        np.save(part / f"{channel}__available.npy", np.isfinite(values), allow_pickle=False)
    partition = {"cancer_id": "BRCA", "patients": _record(part / "patient_ids.json"), "channels": {}}
    for channel in CHANNELS:
        partition["channels"][channel] = {"values": _record(part / f"{channel}.npy"),
                                             "available": _record(part / f"{channel}__available.npy")}
    complete = {
        "format": "GENE_PATIENT_ASSAYS_VERIFIED_V1", "status": "PASS", "target_host": "149",
        "training_started": False, "node_type": "gene", "gene_type": "protein_coding",
        "genome_build": "GRCh38", "matrix_orientation": "patient_by_gene",
        "units": {"cnv_signed": "log2_copy_ratio", "methylation_promoter_beta": "beta"},
        "genes": _record(root / "genes.parquet"), "patients": _record(root / "patients.parquet"),
        "partitions": [partition],
    }
    _write_json(root / "COMPLETE.json", complete)
    return {"enabled": True, "channels": list(CHANNELS), "complete": _record(root / "COMPLETE.json")}


def test_view_reads_only_source_patients_and_preserves_zero_mask(tmp_path):
    first = _product(tmp_path / "first", heldout_cnv=9.0)
    second = _product(tmp_path / "second", heldout_cnv=-7.0)
    a = VerifiedGeneNodeOmics(first, ["ENSG000001", "ENSG000002"], ["BRCA"])
    b = VerifiedGeneNodeOmics(second, ["ENSG000001", "ENSG000002"], ["BRCA"])
    one, audit = a.build_view("BRCA", ["p0", "p1"])
    two, _ = b.build_view("BRCA", ["p0", "p1"])
    np.testing.assert_array_equal(one, two)
    np.testing.assert_allclose(one[0], [1.0, 1.0, 0.3, 1.0])
    np.testing.assert_array_equal(one[1], [0.0, 0.0, 0.0, 0.0])
    zero, _ = a.build_view("BRCA", ["p0"])
    np.testing.assert_allclose(zero[0], [0.0, 1.0, 0.2, 1.0])
    assert audit["outer_or_label_patients_used"] is False
    assert audit["source_patient_sha256"]


def test_product_hash_and_axis_drift_are_rejected(tmp_path):
    cfg = _product(tmp_path / "hash")
    path = tmp_path / "hash" / "BRCA" / "cnv_signed.npy"
    changed = np.load(path, allow_pickle=False).copy(); changed[0, 0] = 7
    np.save(path, changed, allow_pickle=False)
    with pytest.raises(Stop) as error:
        VerifiedGeneNodeOmics(cfg, ["ENSG000001"], ["BRCA"]).verify_cancers(["BRCA"])
    assert error.value.code == "GENE_NODE_OMICS_INPUT_DRIFT"

    cfg = _product(tmp_path / "axis")
    ids = tmp_path / "axis" / "BRCA" / "patient_ids.json"
    _write_json(ids, ["p1", "p0", "p2"])
    complete_path = tmp_path / "axis" / "COMPLETE.json"
    complete = json.loads(complete_path.read_text())
    complete["partitions"][0]["patients"] = _record(ids)
    _write_json(complete_path, complete)
    cfg["complete"] = _record(complete_path)
    with pytest.raises(Stop) as error:
        VerifiedGeneNodeOmics(cfg, ["ENSG000001"], ["BRCA"]).verify_cancers(["BRCA"])
    assert error.value.code == "GENE_NODE_OMICS_PARTITION_AXIS_DRIFT"


torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
from cla_engine.model import ARCHITECTURE, JointHGT, to_graph


def _model_fixture(enabled):
    nodes = {"lncRNA": ["L0", "L1"], "pathway": ["P0", "P1"], "cancer": ["C0"],
             "pathway_family": ["F0"], "gene": ["G0", "G1"], "protein": ["R0", "R1"]}
    relations = [("lncRNA", "targets", "pathway"), ("pathway", "rev_targets", "lncRNA"),
                 ("cancer", "conditions", "lncRNA"), ("lncRNA", "rev_conditions", "cancer"),
                 ("gene", "gene_context", "lncRNA"), ("protein", "protein_context", "pathway")]
    spec = {"architecture": ARCHITECTURE, "nodes": nodes, "relations": relations, "cancers": ["C0"],
            "states": ["S0", "S1"], "state_families": {"S0": "a", "S1": "b"},
            "feature_dims": {kind: 6 for kind in nodes}, "sc_dim": 1,
            "singlecell_role": "interpretation_only", "genomics_enabled": False, "numeric_features": 8,
            "gene_node_omics_enabled": enabled, "node_input_dim": 6 if enabled else 2}
    if enabled:
        spec["node_input_feature_order"] = ["log1p_degree", "constant", "cnv_signed_value",
            "cnv_signed_available", "methylation_promoter_beta_value", "methylation_promoter_beta_available"]
    arrays = {f"x::{kind}": np.ones((len(ids), 6), np.float32) for kind, ids in nodes.items()}
    for relation in relations:
        key = "|".join(relation); count = min(len(nodes[relation[0]]), len(nodes[relation[2]]))
        arrays["edge::" + key] = np.stack([np.arange(count), np.arange(count)]).astype(np.int64)
        arrays["weight::" + key] = np.ones(count, np.float32)
    if enabled:
        for kind, ids in nodes.items(): arrays["node_omics::" + kind] = np.zeros((len(ids), 4), np.float32)
        arrays["node_omics::gene"] = np.asarray([[1, 1, .2, 1], [0, 1, 0, 0]], np.float32)
    config = {"task_weights": {"pathway": 1.0, "state": 0.5}, "hidden": 8, "heads": 2,
              "layers": 2, "dropout": 0.0, "residual_bound": 1.0, "gate_penalty": 0.0}
    return spec, config, arrays


@pytest.mark.parametrize("enabled,width", [(False, 2), (True, 6)])
def test_two_and_six_column_inputs_reach_both_joint_heads(enabled, width):
    spec, config, arrays = _model_fixture(enabled)
    model = JointHGT(spec, config).eval(); graph = to_graph(arrays, "cpu")
    assert all(graph["data"][kind].x.shape[1] == width for kind in spec["nodes"])
    encoded = model.encode(graph, 0)
    common = {"lnc": torch.tensor([0, 1]), "target": torch.tensor([0, 1]),
              "numeric": torch.zeros((2, 8)), "base": torch.zeros(2)}
    for task in ("pathway", "state"):
        output = model.forward_task(task, encoded, common)
        assert output["logit"].shape == (2,) and torch.isfinite(output["logit"]).all()


def test_model_rejects_graph_from_other_node_input_width():
    spec, config, _ = _model_fixture(True)
    _, _, old_arrays = _model_fixture(False)
    model = JointHGT(spec, config); graph = to_graph(old_arrays, "cpu")
    with pytest.raises(Stop) as error: model.encode(graph, 0)
    assert error.value.code == "NODE_INPUT_WIDTH_MISMATCH"
