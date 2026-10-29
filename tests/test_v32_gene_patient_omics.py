"""Synthetic raw assays through genuine gene preparation; no fitting permitted."""
import copy
import gzip
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from cc_hhgt.v32 import gene_patient_omics as gene, omics_node_inputs as inputs
from cc_hhgt.v32.formal_expression_sources import read, write, record
from test_v32_omics_node_inputs import no_fit


def config(root):
    root.mkdir(parents=True, exist_ok=True)
    lines = ["##description: human GRCh38 version 36"]
    for gid, kind, start, end, strand in [("ENSG1", "protein_coding", 2001, 4000, "+"),
                                          ("ENSG2", "protein_coding", 9001, 10000, "-"),
                                          ("ENSG3", "lncRNA", 2001, 4000, "+")]:
        for feature, tid in [("gene", ""), ("transcript", "A"), ("transcript", "B")]:
            attrs = f'gene_id "{gid}.7"; gene_type "{kind}"; transcript_id "T{gid}{tid}";'
            lines.append(f"chr1\tGENCODE\t{feature}\t{start}\t{end}\t.\t{strand}\t.\t{attrs}")
    with gzip.open(root / "reference.gtf.gz", "wt") as f: f.write("\n".join(lines)+"\n")
    pd.DataFrame({"probeID": ["cg1", "cg2", "cg3", "cg4"], "CpG_chrm": "chr1",
                  "CpG_beg": [1000, 2010, 9900, 2101], "CpG_end": [1002, 2012, 9902, 2103]}).to_csv(root / "hm.tsv", sep="\t", index=False)
    write(root / "genes.json", ["ENSG1", "ENSG2", "ENSG3"])
    pd.DataFrame({"patient_id": ["A", "B", "H"], "cancer_id": "C1", "patient_fold_id": [2, 3, 0]}).to_csv(root / "folds.tsv", sep="\t", index=False)
    rows = []
    def raw(channel, patient, sample, fid, content):
        p = root / (fid + ".tsv"); p.write_text(content)
        rows.append({"channel": channel, "patient_id": patient, "sample_id": sample, "file_id": fid,
                     "cancer_id": "C1", "path": str(p), "bytes": p.stat().st_size,
                     "md5": hashlib.md5(p.read_bytes()).hexdigest(), "sample_type": "Primary Tumor"})
    for patient, value in [("A", -.4), ("B", .8), ("H", 5.)]:
        raw("cnv_signed", patient, patient+"c", patient+"c", f"Chromosome\tStart\tEnd\tSegment_Mean\n1\t1\t6000\t{value}\n")
    for patient, sample, fid, a, b in [("A", "a1", "f1", .2, .6), ("A", "a1", "f2", .8, .8),
                                      ("A", "a2", "f3", .2, .2), ("B", "b", "f4", 0., 0.), ("H", "h", "f5", .9, .9)]:
        raw("methylation_promoter_beta", patient, sample, fid, f"cg1\t{a}\ncg2\t{b}\ncg3\tNA\ncg4\t1\n")
    pd.DataFrame(rows).to_parquet(root / "raw.parquet", index=False)
    cfg = {"format": gene.INPUT, "target_host": "149", "genome_build": "GRCh38",
           **{k: record(root / p) for k, p in {"gtf": "reference.gtf.gz", "hm450": "hm.tsv", "genes": "genes.json",
                                               "patient_folds": "folds.tsv", "raw_files": "raw.parquet"}.items()}}
    path = root / "config.json"; write(path, cfg)
    return path


@pytest.fixture(scope="module")
def product(tmp_path_factory):
    root = tmp_path_factory.mktemp("gene_raw")
    path = config(root)
    return gene.materialize(path, root / "product", workers=2)


def node_config(product, root):
    authority = root / "authority.parquet"
    pd.DataFrame({"sample_id": ["a1", "a2", "b", "h"], "patient_id": ["A", "A", "B", "H"],
                  "cancer_id": "C1", "is_tumor": True, "partition": ["train", "train", "train", "test"]}).to_parquet(authority, index=False)
    return {"format": inputs.FORMAT, "target_host": "149", "outer_fold": 0, "scope": "cohort", "cancer_id": "C1",
            "sample_authority": record(authority), "assays": [{"channel": c, "unit": u, "layout": "verified_gene_patient",
            "measurements": record(product)} for c, u in gene.CHANNELS.items()]}


def test_actual_raw_gene_values_and_train_only_aggregation(product, tmp_path):
    data = read(product)
    assert read(data["excluded_requested_genes"]["path"]) == ["ENSG3"]
    mapping = pd.read_parquet(data["promoter_probe_map"]["path"])
    assert not mapping.duplicated().any()
    assert set(mapping.probe_id) == {"cg1", "cg2", "cg3"}  # Right endpoint is exclusive.
    methyl = np.load(product.parent / "C1/methylation_promoter_beta.npy")
    np.testing.assert_allclose(methyl[:, 0], [.4, 0., .9], atol=1e-7)
    assert np.isnan(methyl[:, 1]).all()
    cfg = node_config(product, tmp_path)
    values, lineage, manifest = inputs._prepare(cfg, tmp_path)
    assert set(values.canonical_id) == {"GENE:ENSG1"}
    np.testing.assert_allclose(values.raw_value, [.2, .2], atol=1e-7)
    assert set(lineage.patient_id) == {"A", "B"}
    assert all(v["contributing_patients"] == 2 for v in manifest["coverage"].values())
    cfg["scope"] = "subject"; cfg["subject_id"] = "B"
    values, _, _ = inputs._prepare(cfg, tmp_path)
    assert values.set_index("channel").loc["methylation_promoter_beta", "raw_value"] == 0.


def test_gene_product_reaches_encoder_and_changes_final_vector(product, tmp_path):
    from test_v32_omics_node_encoding import _nodes, _edges, _bundle, _model_cfg
    from cc_hhgt.gnn import build_model, move_graph
    cfg = node_config(product, tmp_path); write(tmp_path / "cfg.json", cfg)
    produced = inputs.produce(tmp_path / "cfg.json", tmp_path / "prepared")
    nodes, edges = _nodes(), _edges()
    replacements = {"G1": "GENE:ENSG1", "G2": "GENE:ENSG2"}
    nodes["canonical_id"] = nodes.canonical_id.replace(replacements)
    nodes["node_id"] = nodes.node_type + ":" + nodes.canonical_id
    for kind in ["source", "target"]:
        edges[kind+"_canonical_id"] = edges[kind+"_canonical_id"].replace(replacements)
        edges[kind+"_node_id"] = edges[kind+"_type"] + ":" + edges[kind+"_canonical_id"]
    payload = {"bundle": _bundle(nodes, edges), "input_authority_hashes": {"base": "a"*64}}
    authority = tmp_path / "input_manifest.json"
    write(authority, {"fold_inputs": [{"fold": 0, "node_omics": record(produced)}]})
    loaded = inputs.load_declared_node_inputs(payload, {"task_contract": {"node_omics_channels": list(gene.CHANNELS)}}, authority, 0)
    bundle = loaded["bundle"]
    np.testing.assert_allclose(bundle.node_omics["gene"][0], [.2, 1, .2, 1], atol=1e-7)
    np.testing.assert_array_equal(bundle.node_omics["gene"][1], [0, 0, 0, 0])
    torch.manual_seed(1729)
    model = build_model("cc_hhgt", bundle, 2, _model_cfg()).eval()
    weights = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        baseline = model.encode(move_graph(bundle, "cpu", "cc_hhgt"))["gene"].clone()
        for column in [2, 4]:
            original = bundle.hetero_data["gene"].x.clone()
            bundle.hetero_data["gene"].x[0, column] += .2
            changed = model.encode(move_graph(bundle, "cpu", "cc_hhgt"))["gene"]
            assert (changed[0] - baseline[0]).abs().max().item() > 1e-7
            bundle.hetero_data["gene"].x = original
    for k, v in weights.items(): torch.testing.assert_close(model.state_dict()[k], v, rtol=0, atol=0)


@pytest.mark.parametrize("fault", ["build", "normal", "identity", "duplicate", "md5", "beta", "overlap"])
def test_raw_sources_fail_closed(tmp_path, fault):
    path = config(tmp_path); cfg = read(path)
    if fault == "build":
        cfg["genome_build"] = "GRCh37"; write(path, cfg)
        with pytest.raises(ValueError, match="GENE_INPUT_FORMAT_HOST_BUILD"): gene.input_context(path)
        return
    table = pd.read_parquet(cfg["raw_files"]["path"])
    if fault in {"normal", "identity", "duplicate"}:
        if fault == "normal": table.loc[0, "sample_type"] = "Solid Tissue Normal"
        if fault == "identity": table.loc[0, "cancer_id"] = "WRONG"
        if fault == "duplicate": table = pd.concat([table, table.iloc[[0]]], ignore_index=True)
        table.to_parquet(cfg["raw_files"]["path"], index=False); cfg["raw_files"] = record(cfg["raw_files"]["path"]); write(path, cfg)
        with pytest.raises(ValueError, match={"normal": "GENE_RAW_NOT_TUMOR", "identity": "GENE_RAW_PATIENT_CANCER_MISMATCH", "duplicate": "GENE_RAW_FILES_DUPLICATED_OR_CHANNEL"}[fault]):
            gene.input_context(path)
        return
    _, files, requested, _, _ = gene.input_context(path)
    bodies, _, mapping, _ = gene.reference(files["gtf"], requested, files["hm450"]); gene.init_worker(bodies, mapping)
    row = table.iloc[3 if fault == "beta" else 0].to_dict()
    if fault == "md5": row["md5"] = "0"*32
    else:
        p = Path(row["path"])
        p.write_text("cg1\t1.5\n" if fault == "beta" else "Chromosome\tStart\tEnd\tSegment_Mean\n1\t1\t6000\t0\n1\t5000\t7000\t1\n")
        row.update(bytes=p.stat().st_size, md5=hashlib.md5(p.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match={"md5": "GENE_RAW_SIZE_OR_MD5_MISMATCH", "beta": "GENE_METHYLATION_BETA_INVALID", "overlap": "GENE_CNV_SEGMENTS_OVERLAP"}[fault]): gene.assay_file(row)


@pytest.mark.parametrize("outer_fold", [0, 4])
def test_heldout_patient_cannot_be_relabeled_training(product, tmp_path, outer_fold):
    cfg = node_config(product, tmp_path)
    p = Path(cfg["sample_authority"]["path"]); a = pd.read_parquet(p); a["partition"] = "train"; a.to_parquet(p, index=False)
    cfg["sample_authority"] = record(p)
    cfg["outer_fold"] = outer_fold
    with pytest.raises(ValueError, match="GENE_PRODUCT_PATIENT_FOLD_MISMATCH"):
        inputs._prepare(cfg, tmp_path)


def test_shared_segment_boundary_is_missing_not_arbitrary_direction(tmp_path):
    path = config(tmp_path)
    _, files, requested, _, raw = gene.input_context(path)
    bodies, _, mapping, _ = gene.reference(files["gtf"], requested, files["hm450"])
    gene.init_worker(bodies, mapping)
    row = raw.iloc[0].to_dict(); p = Path(row["path"])
    p.write_text("Chromosome\tStart\tEnd\tSegment_Mean\n1\t1\t3000\t-1\n1\t3000\t6000\t1\n")
    row.update(bytes=p.stat().st_size, md5=hashlib.md5(p.read_bytes()).hexdigest())
    values, receipt = gene.assay_file(row)
    assert np.isnan(values[0])
    assert receipt["shared_boundary_count"] == receipt["ambiguous_gene_count"] == 1
