"""Fixture checks for multi-omics node encoding into the typed graph encoder.

Everything here is synthetic.  A passing suite means the code path exists and
fails closed on the recorded fault classes; it says nothing about whether
multi-omics conditioning improves a real prediction.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from cc_hhgt.gnn import (
    GraphBundle,
    _materialize_graph_bundle,
    build_model,
    fold_node_features,
    move_graph,
    validate_node_omics,
)
from cc_hhgt.v32 import omics_node_encoding as omics


def _nodes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "node_id": [
                "lncRNA:L1",
                "lncRNA:L2",
                "lncRNA:L3",
                "gene:G1",
                "gene:G2",
                "pathway:PW1",
                "cancer:C1",
            ],
            "node_type": [
                "lncRNA",
                "lncRNA",
                "lncRNA",
                "gene",
                "gene",
                "pathway",
                "cancer",
            ],
            "canonical_id": ["L1", "L2", "L3", "G1", "G2", "PW1", "C1"],
            "node_index_within_type": [0, 1, 2, 0, 1, 0, 0],
        }
    )


def _edges() -> pd.DataFrame:
    rows = [
        ("e1", "lncRNA", "L1", "coexpressed_with", "gene", "G1"),
        ("e2", "lncRNA", "L3", "coexpressed_with", "gene", "G2"),
        ("e3", "gene", "G1", "member_of", "pathway", "PW1"),
        ("e4", "lncRNA", "L1", "expressed_in", "cancer", "C1"),
    ]
    frame = pd.DataFrame(
        rows,
        columns=[
            "edge_id",
            "source_type",
            "source_canonical_id",
            "relation_type",
            "target_type",
            "target_canonical_id",
        ],
    )
    frame["source_node_id"] = frame.source_type + ":" + frame.source_canonical_id
    frame["target_node_id"] = frame.target_type + ":" + frame.target_canonical_id
    frame["weight"] = 0.8
    return frame


def _model_cfg() -> dict:
    return {
        "training": {
            "hidden_channels": 8,
            "num_layers": 1,
            "num_heads": 2,
            "dropout": 0.0,
            "pair_hidden_channels": 4,
        },
        "task_contract": {"pathway": {"target_level": "exact"}},
    }


def _bundle(nodes=None, edges=None, node_omics=None):
    nodes = _nodes() if nodes is None else nodes
    edges = _edges() if edges is None else edges
    return _materialize_graph_bundle(
        nodes,
        edges,
        feature_edges=edges,
        canonical_edges=edges,
        graph_contract="CONTRACT-T",
        node_omics=node_omics,
    )


def _rows():
    return [
        ("lncRNA", "L1", "mutation_damage", 2.0),
        ("lncRNA", "L1", "cnv_signed", -0.5),
        ("gene", "G1", "mutation_damage", 1.0),
        ("gene", "G1", "atac_accessibility", 1.0),
    ]


def _evidence(tmp_path, rows=None, *, channels=omics.CHANNELS, suffix=".tsv", **overrides):
    rows = _rows() if rows is None else rows
    table = tmp_path / ("evidence" + suffix)
    frame = pd.DataFrame(rows, columns=["node_type", "canonical_id", "channel", "raw_value"])
    if suffix.endswith(".parquet"):
        frame.to_parquet(table, index=False)
    elif suffix.endswith(".csv"):
        frame.to_csv(table, index=False)
    else:
        frame.to_csv(table, sep="\t", index=False)
    standardisation = {
        channel: {"center": 0.0, "scale": 1.0} for channel in channels
    }
    standardisation["atac_accessibility"] = {"center": 0.5, "scale": 0.25}
    manifest = {
        "format": omics.EVIDENCE_FORMAT,
        "scope": "cohort",
        "cancer_id": "C1",
        "subject_id": None,
        "channels": list(channels),
        "standardisation": {k: standardisation[k] for k in channels},
        "table_sha256": hashlib.sha256(table.read_bytes()).hexdigest(),
        "row_count": len(frame),
        "clipped": True,
    }
    manifest.update(overrides)
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return table, path


def _evidence_object(tmp_path, rows=None, **overrides):
    table, manifest = _evidence(tmp_path, rows, **overrides)
    return omics.read_evidence(table, manifest)


def test_channel_layout_is_value_then_observed_per_channel() -> None:
    assert omics.channel_layout(("mutation_damage", "cnv_signed")) == [
        "mutation_damage::value",
        "mutation_damage::observed",
        "cnv_signed::value",
        "cnv_signed::observed",
    ]


@pytest.mark.parametrize("suffix", [".tsv", ".csv", ".parquet"])
def test_read_evidence_accepts_every_declared_table_format(tmp_path, suffix) -> None:
    evidence = _evidence_object(tmp_path, suffix=suffix)
    assert evidence.width == len(omics.CHANNELS) * 2
    assert evidence.scope == "cohort"
    assert evidence.subject_id == ""
    assert len(evidence.frame) == len(_rows())


def test_read_evidence_applies_frozen_standardisation(tmp_path) -> None:
    evidence = _evidence_object(tmp_path)
    values = evidence.standardised.set_index(["node_type", "canonical_id", "channel"]).value
    # atac_accessibility uses center 0.5 / scale 0.25 and must be recomputed
    # from the manifest, never from the observed data.
    assert values[("gene", "G1", "atac_accessibility")] == pytest.approx(2.0)
    assert values[("lncRNA", "L1", "mutation_damage")] == pytest.approx(2.0)
    assert values[("lncRNA", "L1", "cnv_signed")] == pytest.approx(-0.5)


def test_read_evidence_clips_outliers(tmp_path) -> None:
    rows = [("gene", "G1", "mutation_damage", 1000.0)]
    evidence = _evidence_object(tmp_path, rows)
    assert evidence.standardised.value.iloc[0] == pytest.approx(omics.CLIP)


def test_read_evidence_accepts_subject_scope(tmp_path) -> None:
    evidence = _evidence_object(tmp_path, scope="subject", subject_id="PATIENT-7")
    assert evidence.scope == "subject"
    assert evidence.subject_id == "PATIENT-7"


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"format": "SOMETHING_ELSE"}, "OMICS_EVIDENCE_FORMAT_INVALID"),
        ({"scope": "fold"}, "OMICS_EVIDENCE_SCOPE_INVALID"),
        ({"scope": "subject", "subject_id": None}, "OMICS_EVIDENCE_SUBJECT_ID_MISSING"),
        (
            {"scope": "cohort", "subject_id": "PATIENT-7"},
            "OMICS_EVIDENCE_SUBJECT_ID_NOT_ALLOWED_FOR_COHORT",
        ),
        ({"cancer_id": ""}, "OMICS_EVIDENCE_CANCER_ID_MISSING"),
        ({"channels": ["mutation_damage", "rna_velocity"]}, "OMICS_EVIDENCE_CHANNEL_UNKNOWN"),
        (
            {"channels": ["mutation_damage", "mutation_damage"]},
            "OMICS_EVIDENCE_CHANNELS_INVALID",
        ),
        ({"channels": []}, "OMICS_EVIDENCE_CHANNELS_INVALID"),
        ({"table_sha256": ""}, "OMICS_EVIDENCE_TABLE_SHA256_MISSING"),
        ({"table_sha256": "0" * 64}, "OMICS_EVIDENCE_TABLE_SHA256_MISMATCH"),
        ({"row_count": 99}, "OMICS_EVIDENCE_ROW_COUNT_MISMATCH"),
        ({"clipped": "yes"}, "OMICS_EVIDENCE_CLIP_DECLARATION_INVALID"),
    ],
)
def test_read_evidence_rejects_manifest_faults(tmp_path, overrides, message) -> None:
    table, manifest = _evidence(tmp_path, **overrides)
    with pytest.raises(RuntimeError, match=message):
        omics.read_evidence(table, manifest)


@pytest.mark.parametrize(
    "standardisation, message",
    [
        ({"mutation_damage": {"center": 0.0, "scale": 0.0}}, "OMICS_EVIDENCE_SCALE_INVALID"),
        ({"mutation_damage": {"center": 0.0, "scale": -1.0}}, "OMICS_EVIDENCE_SCALE_INVALID"),
        (
            {"mutation_damage": {"center": float("nan"), "scale": 1.0}},
            "OMICS_EVIDENCE_CENTER_INVALID",
        ),
        ({"mutation_damage": "scaled"}, "OMICS_EVIDENCE_CENTER_INVALID"),
    ],
)
def test_read_evidence_rejects_invalid_standardisation(tmp_path, standardisation, message) -> None:
    table, manifest = _evidence(tmp_path, standardisation=standardisation)
    with pytest.raises(RuntimeError, match=message):
        omics.read_evidence(table, manifest)


def test_read_evidence_rejects_duplicate_node_channel(tmp_path) -> None:
    rows = _rows() + [("lncRNA", "L1", "mutation_damage", 3.0)]
    table, manifest = _evidence(tmp_path, rows)
    with pytest.raises(RuntimeError, match="OMICS_EVIDENCE_DUPLICATE_NODE_CHANNEL"):
        omics.read_evidence(table, manifest)


def test_read_evidence_rejects_undeclared_channel_rows(tmp_path) -> None:
    table, manifest = _evidence(tmp_path, channels=("mutation_damage",))
    with pytest.raises(RuntimeError, match="OMICS_EVIDENCE_CHANNEL_NOT_DECLARED"):
        omics.read_evidence(table, manifest)


def test_read_evidence_rejects_non_finite_raw_values(tmp_path) -> None:
    rows = _rows()[:-1] + [("gene", "G1", "atac_accessibility", float("nan"))]
    table, manifest = _evidence(tmp_path, rows)
    with pytest.raises(RuntimeError, match="OMICS_EVIDENCE_RAW_VALUE_INVALID"):
        omics.read_evidence(table, manifest)


def test_read_evidence_rejects_empty_identifiers(tmp_path) -> None:
    rows = _rows()[:-1] + [("gene", "", "atac_accessibility", 1.0)]
    table, manifest = _evidence(tmp_path, rows)
    with pytest.raises(RuntimeError, match="OMICS_EVIDENCE_EMPTY_IDENTIFIER"):
        omics.read_evidence(table, manifest)


def test_read_evidence_rejects_missing_columns(tmp_path) -> None:
    table = tmp_path / "bad.tsv"
    pd.DataFrame({"node_type": ["gene"], "canonical_id": ["G1"]}).to_csv(
        table, sep="\t", index=False
    )
    manifest = tmp_path / "bad.json"
    manifest.write_text(
        json.dumps(
            {
                "format": omics.EVIDENCE_FORMAT,
                "scope": "cohort",
                "cancer_id": "C1",
                "channels": ["mutation_damage"],
                "standardisation": {
                    "mutation_damage": {"center": 0.0, "scale": 1.0}
                },
                "table_sha256": hashlib.sha256(table.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="OMICS_EVIDENCE_COLUMNS_MISSING"):
        omics.read_evidence(table, manifest)


def test_read_evidence_rejects_unsupported_table_format(tmp_path) -> None:
    table = tmp_path / "evidence.xlsx"
    table.write_bytes(b"not a table")
    manifest = tmp_path / "evidence.json"
    manifest.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="OMICS_EVIDENCE_FORMAT_INVALID"):
        omics.read_evidence(table, manifest)


def test_node_feature_block_aligns_by_node_index_not_row_order(tmp_path) -> None:
    evidence = _evidence_object(tmp_path)
    block = omics.node_feature_block(_nodes(), evidence)
    shuffled = _nodes().sample(frac=1.0, random_state=7).reset_index(drop=True)
    other = omics.node_feature_block(shuffled, evidence)
    for node_type, value in block["blocks"].items():
        assert np.array_equal(value, other["blocks"][node_type])


def test_node_feature_block_marks_absent_channels_instead_of_imputing(tmp_path) -> None:
    evidence = _evidence_object(tmp_path)
    result = omics.node_feature_block(_nodes(), evidence)
    genes = result["blocks"]["gene"]
    lncrna = result["blocks"]["lncRNA"]
    column = omics.channel_layout(omics.CHANNELS)
    mutation = column.index("mutation_damage::value")
    mutation_observed = column.index("mutation_damage::observed")
    # G2 has no measurement at all: both the value and the flag stay zero.
    assert genes[1, mutation] == 0.0
    assert genes[1, mutation_observed] == 0.0
    assert genes[0, mutation] == pytest.approx(1.0)
    assert genes[0, mutation_observed] == 1.0
    # A pathway node type carries no modality: explicit zeros, never a mean.
    pathway = result["blocks"]["pathway"]
    assert pathway.shape == (1, len(omics.CHANNELS) * 2)
    assert np.all(pathway == 0.0)
    assert result["report"]["covered_pairs"] == {"lncRNA": 2, "gene": 2, "pathway": 0, "cancer": 0}
    # L1 has two channels, L3 has none.
    assert lncrna[0, mutation_observed] == 1.0
    assert lncrna[2, mutation_observed] == 0.0
    assert result["report"]["node_feature_width"] == 2 + len(omics.CHANNELS) * 2


def test_node_feature_block_rejects_evidence_node_outside_graph(tmp_path) -> None:
    rows = _rows() + [("gene", "G9", "mutation_damage", 1.0)]
    evidence = _evidence_object(tmp_path, rows)
    with pytest.raises(RuntimeError, match="OMICS_EVIDENCE_NODE_NOT_IN_GRAPH"):
        omics.node_feature_block(_nodes(), evidence)


def test_node_feature_block_rejects_missing_index_column(tmp_path) -> None:
    evidence = _evidence_object(tmp_path)
    nodes = _nodes().drop(columns="node_index_within_type")
    with pytest.raises(RuntimeError, match="OMICS_NODES_LACK_INDEX_COLUMN"):
        omics.node_feature_block(nodes, evidence)


def test_node_feature_block_rejects_duplicate_canonical_ids(tmp_path) -> None:
    evidence = _evidence_object(tmp_path)
    nodes = _nodes()
    nodes.loc[nodes.canonical_id.eq("L2"), "canonical_id"] = "L1"
    with pytest.raises(RuntimeError, match="OMICS_NODE_CANONICAL_ID_DUPLICATE"):
        omics.node_feature_block(nodes, evidence)


def test_materializer_widens_node_inputs(tmp_path) -> None:
    evidence = _evidence_object(tmp_path)
    result = omics.node_feature_block(_nodes(), evidence)
    bundle = _bundle(node_omics=result["blocks"])
    width = 2 + len(omics.CHANNELS) * 2
    assert bundle.node_input_dim == width
    assert bundle.homogeneous["node_x"].shape == (len(_nodes()), width)
    for node_type in ("lncRNA", "gene", "pathway", "cancer"):
        assert bundle.hetero_data[node_type].x.shape[1] == width
    # The two degree columns stay first and unchanged.
    degrees = fold_node_features(
        _nodes().loc[_nodes().node_type.eq("gene")].sort_values("node_index_within_type"),
        _edges(),
        "gene",
    )
    assert np.allclose(
        bundle.hetero_data["gene"].x.numpy()[:, :2], degrees, atol=1e-6
    )


def test_materializer_keeps_two_channel_contract_by_default() -> None:
    bundle = _bundle()
    assert bundle.node_input_dim == 2
    assert bundle.node_omics is None
    assert bundle.homogeneous["node_x"].shape == (len(_nodes()), 2)
    assert bundle.hetero_data["gene"].x.shape[1] == 2


def test_fold_node_features_appends_omics_columns() -> None:
    nodes = _nodes().loc[_nodes().node_type.eq("gene")].sort_values("node_index_within_type")
    block = np.asarray([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    features = fold_node_features(nodes, _edges(), "gene", omics=block)
    assert features.shape == (2, 6)
    assert np.allclose(features[:, 2:], block)
    base = fold_node_features(nodes, _edges(), "gene")
    assert np.allclose(features[:, :2], base)


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda blocks: blocks.pop("pathway"), id="missing_node_type"),
        pytest.param(
            lambda blocks: blocks.update({"compound": np.zeros((1, 6))}),
            id="unknown_node_type",
        ),
        pytest.param(
            lambda blocks: blocks.update({"gene": np.zeros((1, 6))}), id="row_mismatch"
        ),
        pytest.param(
            lambda blocks: blocks.update({"gene": np.zeros((2, 4))}), id="width_mismatch"
        ),
        pytest.param(
            lambda blocks: blocks["gene"].__setitem__((0, 0), np.nan), id="non_finite"
        ),
        pytest.param(
            lambda blocks: blocks.update({"gene": np.zeros((2, 6))[:, :0]}),
            id="empty_block",
        ),
    ],
)
def test_materializer_rejects_invalid_omics_blocks(tmp_path, mutate) -> None:
    evidence = _evidence_object(tmp_path)
    blocks = {key: value.copy() for key, value in omics.node_feature_block(_nodes(), evidence)["blocks"].items()}
    mutate(blocks)
    with pytest.raises((RuntimeError, ValueError, TypeError)):
        _bundle(node_omics=blocks)


def test_validate_node_omics_returns_defensive_copies(tmp_path) -> None:
    evidence = _evidence_object(tmp_path)
    blocks = omics.node_feature_block(_nodes(), evidence)["blocks"]
    cleaned, width = validate_node_omics(blocks, _nodes())
    assert width == len(omics.CHANNELS) * 2
    cleaned["gene"][:] = 99.0
    assert not np.allclose(blocks["gene"], 99.0)


def test_validate_node_omics_rejects_empty_mapping() -> None:
    with pytest.raises(RuntimeError, match="OMICS_NODE_OMICS_NOT_A_MAPPING"):
        validate_node_omics({}, _nodes())


def test_build_model_reads_the_widened_input_width(tmp_path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    evidence = _evidence_object(tmp_path)
    blocks = omics.node_feature_block(_nodes(), evidence)["blocks"]
    bundle = _bundle(node_omics=blocks)
    model = build_model("cc_hhgt", bundle, 3, _model_cfg())
    observed = omics.assert_encoder_consumes_omics(model, bundle)
    assert observed["verified"] is True
    assert observed["layers"] == {"lncRNA": 8, "gene": 8, "pathway": 8, "cancer": 8}
    rgcn = build_model("rgcn", bundle, 3, _model_cfg())
    assert omics.assert_encoder_consumes_omics(rgcn, bundle)["layers"] == {"homogeneous": 8}


def test_assert_encoder_refuses_a_degree_only_bundle() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    bundle = _bundle()
    model = build_model("cc_hhgt", bundle, 3, _model_cfg())
    with pytest.raises(RuntimeError, match="OMICS_ENCODER_NOT_WIDENED"):
        omics.assert_encoder_consumes_omics(model, bundle)


def test_attach_node_omics_writes_and_verifies(tmp_path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    evidence = _evidence_object(tmp_path)
    blocks = omics.node_feature_block(_nodes(), evidence)["blocks"]
    bundle = _bundle()
    checks = omics.attach_node_omics(bundle, blocks)
    assert checks["node_input_dim"] == 2 + len(omics.CHANNELS) * 2
    assert checks["width"] == len(omics.CHANNELS) * 2
    assert bundle.node_omics is not None
    for node_type, expected in blocks.items():
        stored = bundle.hetero_data[node_type].x.numpy()[:, 2:]
        assert np.allclose(stored, expected, atol=1e-6)
    model = build_model("cc_hhgt", bundle, 3, _model_cfg())
    assert omics.assert_encoder_consumes_omics(model, bundle)["verified"] is True


def test_attach_node_omics_rejects_a_second_attach(tmp_path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    blocks = omics.node_feature_block(_nodes(), _evidence_object(tmp_path))["blocks"]
    bundle = _bundle()
    omics.attach_node_omics(bundle, blocks)
    with pytest.raises(RuntimeError, match="OMICS_BUNDLE_ALREADY_AUGMENTED"):
        omics.attach_node_omics(bundle, blocks)


def test_attach_node_omics_rejects_row_mismatch(tmp_path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    blocks = dict(omics.node_feature_block(_nodes(), _evidence_object(tmp_path))["blocks"])
    blocks["gene"] = np.zeros((1, blocks["gene"].shape[1]))
    bundle = _bundle()
    with pytest.raises(RuntimeError, match="OMICS_BLOCK_ROW_MISMATCH"):
        omics.attach_node_omics(bundle, blocks)


def test_attach_node_omics_rejects_a_model_built_before_augmentation(tmp_path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    blocks = omics.node_feature_block(_nodes(), _evidence_object(tmp_path))["blocks"]
    bundle = _bundle()
    model = build_model("cc_hhgt", bundle, 3, _model_cfg())
    omics.attach_node_omics(bundle, blocks)
    with pytest.raises(RuntimeError, match="OMICS_ENCODER_INPUT_WIDTH_MISMATCH"):
        omics.assert_encoder_consumes_omics(model, bundle)


def _two_graphs(tmp_path):
    evidence = _evidence_object(tmp_path)
    base = omics.node_feature_block(_nodes(), evidence)["blocks"]
    perturbed = {key: value.copy() for key, value in base.items()}
    columns = omics.channel_layout(omics.CHANNELS)
    mutation = columns.index("mutation_damage::value")
    observed = columns.index("mutation_damage::observed")
    gene_rows = _nodes().loc[_nodes().node_type.eq("gene")].sort_values(
        "node_index_within_type"
    )
    position = int(np.flatnonzero(gene_rows.canonical_id.eq("G1").to_numpy())[0])
    perturbed["gene"][position, mutation] = 3.0
    perturbed["gene"][position, observed] = 1.0
    return base, perturbed


def test_omics_value_propagates_through_message_passing(tmp_path) -> None:
    """A gene-level omics change must reach a connected lncRNA embedding.

    This is the actual claim behind "multi-omics participates in graph
    encoding": the value is written on the gene node, and the lncRNA embedding
    that HGT produces for its partner changes without any decoder term.
    """

    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    base, perturbed = _two_graphs(tmp_path)
    bundle_base = _bundle(node_omics=base)
    bundle_perturbed = _bundle(node_omics=perturbed)
    model = build_model("cc_hhgt", bundle_base, 3, _model_cfg())
    graph_base = move_graph(bundle_base, "cpu", "cc_hhgt")
    graph_perturbed = move_graph(bundle_perturbed, "cpu", "cc_hhgt")
    connected_base = omics.encode_forward(
        model, graph_base, node_type="lncRNA", node_id="L1", bundle=bundle_base
    )
    connected_perturbed = omics.encode_forward(
        model, graph_perturbed, node_type="lncRNA", node_id="L1", bundle=bundle_perturbed
    )
    gene_base = omics.encode_forward(
        model, graph_base, node_type="gene", node_id="G1", bundle=bundle_base
    )
    gene_perturbed = omics.encode_forward(
        model, graph_perturbed, node_type="gene", node_id="G1", bundle=bundle_perturbed
    )
    assert float(np.max(np.abs(gene_perturbed - gene_base))) > 1e-6
    assert float(np.max(np.abs(connected_perturbed - connected_base))) > 1e-6


def test_unrelated_nodes_are_unaffected_by_a_local_omics_change(tmp_path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    base, perturbed = _two_graphs(tmp_path)
    bundle_base = _bundle(node_omics=base)
    bundle_perturbed = _bundle(node_omics=perturbed)
    model = build_model("cc_hhgt", bundle_base, 3, _model_cfg())
    graph_base = move_graph(bundle_base, "cpu", "cc_hhgt")
    graph_perturbed = move_graph(bundle_perturbed, "cpu", "cc_hhgt")
    for node_type, node_id in (("lncRNA", "L2"), ("lncRNA", "L3"), ("gene", "G2")):
        left = omics.encode_forward(
            model, graph_base, node_type=node_type, node_id=node_id, bundle=bundle_base
        )
        right = omics.encode_forward(
            model,
            graph_perturbed,
            node_type=node_type,
            node_id=node_id,
            bundle=bundle_perturbed,
        )
        assert float(np.max(np.abs(right - left))) < 1e-6


def test_repeated_forward_is_deterministic_in_eval_mode(tmp_path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    base, _ = _two_graphs(tmp_path)
    bundle = _bundle(node_omics=base)
    model = build_model("cc_hhgt", bundle, 3, _model_cfg())
    graph = move_graph(bundle, "cpu", "cc_hhgt")
    first = omics.encode_forward(model, graph, node_type="lncRNA", node_id="L1", bundle=bundle)
    second = omics.encode_forward(model, graph, node_type="lncRNA", node_id="L1", bundle=bundle)
    assert np.array_equal(first, second)


def test_module_exposes_no_fitting_entry_point() -> None:
    forbidden = {"fit", "train", "fit_transform", "backward", "optimizer", "step"}
    assert forbidden.isdisjoint(dir(omics))
    source = (omics.__file__ or "")
    assert source.endswith("omics_node_encoding.py")


def test_graph_bundle_defaults_are_backward_compatible() -> None:
    fields = GraphBundle.__dataclass_fields__
    assert fields["node_input_dim"].default == 2
    assert fields["node_omics"].default is None
