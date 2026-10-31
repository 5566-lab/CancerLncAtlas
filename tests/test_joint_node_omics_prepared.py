from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from cla_engine.common import Stop, atomic_json, atomic_npz, read_json, sha
from cla_engine.fixture import make_fixture
from cla_engine.lineage import validate_prepared
from cla_engine.model import JointHGT, loss_one, to_graph
from cla_engine.prepare import prepare
from cla_engine.train import load_npz, tensor_batch


CHANNELS = ("cnv_signed", "methylation_promoter_beta")


def _record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path)}


def _stable_gene_fixture(root):
    """Create the standard fixture but give every graph gene a real namespace."""
    cfg = make_fixture(root / "sources")
    mapping = {f"G{i}": f"ENSG{i + 1:011d}" for i in range(6)}
    for role in ("exact_membership", "weighted_membership", "protein_gene"):
        path = Path(cfg["sources"][role]["path"])
        frame = pd.read_csv(path, sep="\t")
        frame["gene_id"] = frame.gene_id.map(mapping)
        frame.to_csv(path, sep="\t", index=False)
        cfg["sources"][role]["sha256"] = sha(path)
    for path in sorted(Path(cfg["sources"]["gene_expression_root"]["path"]).glob("*.tsv")):
        frame = pd.read_csv(path, sep="\t")
        frame["gene_id"] = frame.gene_id.map(mapping)
        frame.to_csv(path, sep="\t", index=False)
    # The test isolates graph node inputs; the independent mutation projection
    # has its own coverage and need not be renamed for this integration check.
    cfg["genomics"]["enabled"] = False
    return cfg, list(mapping.values())


def _write_product(root, cfg, genes, *, changed_label_patient=False):
    root.mkdir(parents=True)
    patient_map = pd.read_csv(cfg["sources"]["patient_map"]["path"], sep="\t")
    patients = (patient_map[["cancer_id", "patient_id", "patient_fold_id"]]
                .drop_duplicates().sort_values(["cancer_id", "patient_id"]).reset_index(drop=True))
    pd.DataFrame({"gene_id": genes}).to_parquet(root / "genes.parquet", index=False)
    patients.to_parquet(root / "patients.parquet", index=False)
    partitions = []
    for cancer, axis in patients.groupby("cancer_id", sort=True):
        part = root / str(cancer); part.mkdir()
        identifiers = axis.patient_id.astype(str).tolist()
        atomic_json(part / "patient_ids.json", identifiers)
        row = np.arange(len(axis), dtype=np.float32)[:, None]
        column = np.arange(len(genes), dtype=np.float32)[None, :]
        cnv = ((row % 7) - 3 + column / 10).astype(np.float32)
        methyl = (((row + column) % 11) / 10).astype(np.float32)
        # Preserve an observed zero and structural missingness explicitly.
        cnv[0, 0] = 0.0; methyl[0, 0] = 0.0
        cnv[1, 1] = np.nan; methyl[1, 1] = np.nan
        if changed_label_patient and str(cancer) == "C1":
            label = np.flatnonzero(axis.patient_fold_id.to_numpy(int) == 2)[0]
            cnv[label, 0] = 123.0
            methyl[label, 0] = 0.99
        channel_records = {}
        for channel, values in (("cnv_signed", cnv), ("methylation_promoter_beta", methyl)):
            np.save(part / f"{channel}.npy", values, allow_pickle=False)
            np.save(part / f"{channel}__available.npy", np.isfinite(values), allow_pickle=False)
            channel_records[channel] = {"values": _record(part / f"{channel}.npy"),
                                        "available": _record(part / f"{channel}__available.npy")}
        partitions.append({"cancer_id": str(cancer), "patients": _record(part / "patient_ids.json"),
                           "channels": channel_records})
    complete = {
        "format": "GENE_PATIENT_ASSAYS_VERIFIED_V1", "status": "PASS", "target_host": "149",
        "training_started": False, "node_type": "gene", "gene_type": "protein_coding",
        "genome_build": "GRCh38", "matrix_orientation": "patient_by_gene",
        "units": {"cnv_signed": "log2_copy_ratio", "methylation_promoter_beta": "beta"},
        "genes": _record(root / "genes.parquet"), "patients": _record(root / "patients.parquet"),
        "partitions": partitions,
    }
    atomic_json(root / "COMPLETE.json", complete)
    return {"enabled": True, "channels": list(CHANNELS), "complete": _record(root / "COMPLETE.json")}


def _prepare_with_product(cfg, product, output):
    variant = copy.deepcopy(cfg)
    variant["gene_node_omics"] = product
    prepare(variant, output, fold=0, profile="diagnostic")
    manifest, spec, stored, lineage = validate_prepared(output)
    return variant, manifest, spec, stored, lineage


def test_prepared_six_column_joint_path_and_tamper_rejection(tmp_path):
    cfg, genes = _stable_gene_fixture(tmp_path)
    product_a = _write_product(tmp_path / "product_a", cfg, genes)
    product_b = _write_product(tmp_path / "product_b", cfg, genes, changed_label_patient=True)
    cfg_a, manifest_a, spec, stored_cfg, _ = _prepare_with_product(cfg, product_a, tmp_path / "prepared_a")
    _, _, spec_b, _, _ = _prepare_with_product(cfg, product_b, tmp_path / "prepared_b")
    prepared = tmp_path / "prepared_a"

    assert stored_cfg == cfg_a
    assert spec["gene_node_omics_enabled"] is True
    assert spec["node_input_dim"] == 6 == spec_b["node_input_dim"]
    assert spec["node_input_feature_order"] == ["log1p_degree", "constant", "cnv_signed_value",
        "cnv_signed_available", "methylation_promoter_beta_value", "methylation_promoter_beta_available"]
    view = "C1__train2"  # source folds 3/4; fold 2 is label-only for this view.
    expected_files = {f"views/{view}/graph.npz", f"views/{view}/NODE_OMICS_AUDIT.json"}
    assert expected_files <= set(manifest_a["files"])
    audit = read_json(prepared / "views" / view / "NODE_OMICS_AUDIT.json")
    assert audit["node_input_dim"] == 6
    assert audit["contract"] == "SOURCE_PATIENT_MEAN_EXPLICIT_MASK_NO_IMPUTATION"
    assert audit["missing_value_imputation"] is False and audit["parameter_fit"] is False

    graph_a = load_npz(prepared / "views" / view / "graph.npz")
    graph_b = load_npz(tmp_path / "prepared_b" / "views" / view / "graph.npz")
    for key in sorted(k for k in graph_a if k.startswith("node_omics::")):
        np.testing.assert_array_equal(graph_a[key], graph_b[key])
    audit_b = read_json(tmp_path / "prepared_b" / "views" / view / "NODE_OMICS_AUDIT.json")
    assert audit["block_sha256"] == audit_b["block_sha256"]

    model = JointHGT(spec, cfg_a["model"])
    loaded = JointHGT(spec, cfg_a["model"])
    loaded.load_state_dict(model.state_dict())
    graph = to_graph(graph_a, "cpu")
    assert all(graph["data"][kind].x.shape[1] == 6 for kind in spec["nodes"])
    encoded = loaded.encode(graph, spec["cancers"].index("C1"))
    batches = {}
    for task in ("pathway", "state"):
        arrays = load_npz(prepared / "views" / view / f"{task}.npz")
        positions = np.flatnonzero(arrays["available"])[:8]
        assert len(positions)
        batches[task] = tensor_batch(arrays, positions, "cpu")
    outputs = {task: loaded.forward_task(task, encoded, batch) for task, batch in batches.items()}
    losses = {task: loss_one(outputs[task], batches[task], cfg_a["model"]["direction_weight"])
              for task in ("pathway", "state")}
    joint = losses["pathway"] + 0.5 * losses["state"] + loaded.regularizer()
    assert torch.isfinite(joint)
    before = {name: value.detach().clone() for name, value in loaded.named_parameters()}
    optimizer = torch.optim.AdamW(loaded.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True); joint.backward(); optimizer.step()
    assert any(not torch.equal(before[name], value.detach()) for name, value in loaded.named_parameters())

    graph_path = prepared / "views" / view / "graph.npz"
    original_graph = graph_path.read_bytes()
    changed = load_npz(graph_path)
    changed["node_omics::gene"] = changed["node_omics::gene"].copy()
    changed["node_omics::gene"][0, 0] += 1
    atomic_npz(graph_path, **changed)
    with pytest.raises(Stop) as error: validate_prepared(prepared)
    assert error.value.code == "INPUT_DRIFT"
    graph_path.write_bytes(original_graph)
    validate_prepared(prepared)

    audit_path = prepared / "views" / view / "NODE_OMICS_AUDIT.json"
    original_audit = audit_path.read_bytes()
    changed_audit = read_json(audit_path); changed_audit["block_sha256"] = "0" * 64
    atomic_json(audit_path, changed_audit)
    with pytest.raises(Stop) as error: validate_prepared(prepared)
    assert error.value.code == "INPUT_DRIFT"
    audit_path.write_bytes(original_audit)
    validate_prepared(prepared)
