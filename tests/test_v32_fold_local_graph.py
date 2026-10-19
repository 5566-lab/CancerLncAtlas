"""Actual native E1 producer and fold normalization joins; no model fitting."""
from dataclasses import replace
from pathlib import Path
import copy
import socket

import pandas as pd
import pytest
import torch

from cc_hhgt.v32.fold_local_graph import verify, replay_binding
from cc_hhgt.v32.formal_expression_sources import read, write, record
from cc_hhgt.v32.input_lineage import artifact_sha256
from test_v32_fold_local_expression import normalized, forbidden, RUNNER as CONVERT_RUNNER
from test_v32_fold_local_scope import scoped, graph_fixture


@pytest.fixture(autouse=True)
def no_fit(monkeypatch):
    assert socket.gethostname() == "149"
    for name in ("Adam", "AdamW", "SGD"): monkeypatch.setattr(torch.optim, name, forbidden)
    from sklearn.linear_model import LogisticRegression
    monkeypatch.setattr(LogisticRegression, "fit", forbidden)


def artifact(path): return {"path": str(path), "sha256": artifact_sha256(path)}


@pytest.fixture(scope="module")
def graph_inputs(scoped, tmp_path_factory):
    root = tmp_path_factory.mktemp("fold_normalization_graph")
    return {fold: graph_fixture(root/str(fold), scoped, fold) for fold in range(5)}


@pytest.mark.parametrize("fold", range(5))
def test_actual_five_fold_native_graph_sources_match_both_expression_kinds(graph_inputs, scoped, fold):
    inputs, _ = graph_inputs[fold]
    edge = inputs.receipt["fold_artifacts"][str(fold)]["train_coexpression"]
    proof = verify(record(inputs.receipt_path), edge, record(scoped/f"normal/fold={fold}/SCOPE_SOURCES.json"), scoped, fold)
    assert set(proof["cancers"]) == {"BRCA"}
    cancer = proof["cancers"]["BRCA"]
    assert set(cancer["expression_inputs"]) == {"lncrna_expression", "gene_expression"}
    assert cancer["normalization"] == read(scoped/f"normal/fold={fold}/SCOPE_SOURCES.json")["normalizations"]["BRCA"]
    assert cancer["edge_rows"] == len(pd.read_parquet(edge["path"]))
    # The actual synthetic native calculations produce no E1 edges in these
    # folds. Their cancer and normalization evidence must still be retained.
    if fold in (0, 4): assert cancer["edge_rows"] == 0
    else: assert cancer["edge_rows"] > 0


@pytest.fixture(scope="module")
def changed_conversion(scoped, tmp_path_factory):
    from cc_hhgt.v32.fold_local_expression import convert
    root = tmp_path_factory.mktemp("different_graph_normalization")/"converted"
    convert([scoped.parent/"changed"], root, runner=CONVERT_RUNNER)
    return root


def altered_receipt(root, inputs, mutation, changed_conversion):
    """Repin every changed metadata layer, so failures test semantics, not SHA typos."""
    document = copy.deepcopy(inputs.receipt)
    producer = read(document["local_cnv_adjustment"]["source_receipts"][0]["path"])
    output = next(r for r in producer["outputs"] if r["fold"] == 0)
    edge = copy.deepcopy(document["fold_artifacts"]["0"]["train_coexpression"])
    if mutation == "no_ancestry": output["upstream_transformation"] = None
    elif mutation == "missing_partition":
        empty = root/"empty"; empty.mkdir(); write(empty/"metadata.json", {"no_parquet_partitions": True}); edge = artifact(empty)
    elif mutation == "duplicate_cancer":
        document["local_cnv_adjustment"]["source_receipts"] *= 2
    elif mutation == "changed_edge":
        frame = pd.read_parquet(output["path"])
        if len(frame): frame["rho"] = -frame.rho
        else:
            frame = pd.DataFrame({"cancer_id": ["BRCA"], "lncrna_id": ["LNC:Z"], "gene_id": ["GENE:G1"],
                "rho": [.8], "source_split": ["train"], "edge_outer_fold": [0]})
        path = root/"changed_edges.parquet"; frame.to_parquet(path, index=False)
        output.update(record(path), rows=len(frame)); edge = record(path)
    else:
        binding = copy.deepcopy(output["upstream_transformation"])
        source = read(binding["sources"]["path"])
        if mutation == "patient_map":
            item = next(r for r in source["records"] if r["role"] == "patient_map")
            file = root/"patient_map.tsv"; file.write_bytes(Path(item["path"]).read_bytes()); item.update(record(file))
        else:
            role = "gene_expression" if mutation == "gene_normalization" else "lncrna_expression"
            item = next(r for r in source["records"] if r["role"] == role and r.get("fold") == 0)
            if mutation == "legacy": item.pop("upstream_expression")
            elif mutation == "conversion_values":
                path = root/"expression.parquet"
                pd.read_parquet(item["path"]).assign(logcpm=lambda f: f.logcpm+.5).to_parquet(path, index=False)
                conversion = read(item["upstream_expression"]["path"]); conversion["output"] = record(path)
                conv_path = root/"CONVERSION.json"; write(conv_path, conversion)
                item.update(record(path), upstream_expression=record(conv_path))
            else:
                kind = "gene" if role == "gene_expression" else "lncrna"
                path = changed_conversion/kind/"fold=0/cancer_id=BRCA/part-0.parquet"
                item.update(record(path), upstream_expression=record(path.parent/"EXPRESSION_SOURCES.json"))
        path = root/"TRANSFORMATION.json"; write(path, source)
        native_producer = read(binding["receipt"]["path"]); native_producer["consumed_sources"] = record(path)
        native_path = root/"NATIVE_PRODUCER.json"; write(native_path, native_producer)
        binding.update(sources=record(path), receipt=record(native_path)); output["upstream_transformation"] = binding
    path = root/"E1_PRODUCER.json"; write(path, producer)
    document["local_cnv_adjustment"]["source_receipts"][0] = record(path)
    document["fold_artifacts"]["0"]["train_coexpression"] = edge
    path = root/"GRAPH_RECEIPT.json"; write(path, document)
    return record(path), edge


@pytest.mark.parametrize("mutation,reason", [
    ("lncrna_normalization", "normalization sources differ"), ("gene_normalization", "normalization sources differ"),
    ("patient_map", "patient maps differ"), ("legacy", "requires native fold-local"),
    ("conversion_values", "native conversion"), ("no_ancestry", "lacks native transformation"),
    ("duplicate_cancer", "cancer declaration"), ("missing_partition", "explicit partitions"),
    ("changed_edge", "actual native E1 selection")])
def test_repinned_inconsistent_graph_inputs_fail(graph_inputs, scoped, changed_conversion, tmp_path, mutation, reason):
    inputs, _ = graph_inputs[0]
    receipt, edge = altered_receipt(tmp_path, inputs, mutation, changed_conversion)
    with pytest.raises(ValueError, match=reason):
        verify(receipt, edge, record(scoped/"normal/fold=0/SCOPE_SOURCES.json"), scoped, 0)


def test_actual_graph_builder_rejects_other_normalization_before_materializing(graph_inputs, scoped, changed_conversion, tmp_path):
    from cc_hhgt.v32.formal_graph_authority import build_bound_formal_graph
    from cc_hhgt.v32.fold_local_scope import scope_input
    inputs, patients = graph_inputs[0]
    item, _ = altered_receipt(tmp_path, inputs, "gene_normalization", changed_conversion)
    changed = replace(inputs, receipt_path=Path(item["path"]), receipt_sha256=item["sha256"], receipt=read(item["path"]))
    _, detection = scope_input(read(scoped/"normal/SCOPE_RELEASE.json"), scoped, 0)
    with pytest.raises(ValueError, match="normalization sources differ"):
        build_bound_formal_graph(changed, outer_fold=0, split_manifest=patients,
            candidate_pairs=pd.read_parquet(scoped/"normal/candidates.parquet"), detection_override=detection)


def test_bound_graph_replays_sources_and_cannot_drop_or_forge_match(graph_inputs, scoped):
    from cc_hhgt.v32.formal_graph_authority import build_bound_formal_graph, bind_formal_graph_variant, validate_formal_graph_payload_binding
    from cc_hhgt.v32.fold_local_scope import scope_input
    inputs, patients = graph_inputs[0]
    _, detection = scope_input(read(scoped/"normal/SCOPE_RELEASE.json"), scoped, 0)
    graph = build_bound_formal_graph(inputs, outer_fold=0, split_manifest=patients,
        candidate_pairs=pd.read_parquet(scoped/"normal/candidates.parquet"), detection_override=detection)
    binding = bind_formal_graph_variant(graph.binding, graph.authority, "G2")
    assert replay_binding(binding, scoped, 0) == binding["fold_normalization_match"]
    assert replay_binding({"base_authority": binding}, scoped, 0) == binding["fold_normalization_match"]
    for mutation in ("missing", "partial", "changed"):
        altered = copy.deepcopy(binding)
        if mutation == "missing": del altered["fold_normalization_match"]
        elif mutation == "partial": altered["fold_normalization_match"]["cancers"]["BRCA"]["expression_inputs"].pop("gene_expression")
        else: altered["fold_normalization_match"]["cancers"]["BRCA"]["edge_rows"] += 1
        with pytest.raises(ValueError, match="normalization"):
            if mutation == "changed": replay_binding(altered, scoped, 0)
            else: validate_formal_graph_payload_binding(altered, outer_fold=0, variant="G2")
