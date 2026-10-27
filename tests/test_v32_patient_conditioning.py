"""Fixture checks for patient-level intervention and interaction attribution.

Synthetic graphs only.  A pass here means the counterfactual contrast is defined,
executed and falsifiable; it is not a patient result and not a mediation proof.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from cc_hhgt.gnn import _materialize_graph_bundle, build_model, move_graph
from cc_hhgt.v32 import omics_node_encoding as omics
from cc_hhgt.v32 import patient_conditioning as pc

FEATURES = ["pair_expression", "pair_membership"]
CHANNELS = ("mutation_damage", "cnv_signed")


def _nodes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "node_id": [
                "lncRNA:L1",
                "lncRNA:L2",
                "gene:G1",
                "gene:G2",
                "pathway:PW1",
                "pathway:PW2",
                "cancer:C1",
            ],
            "node_type": [
                "lncRNA",
                "lncRNA",
                "gene",
                "gene",
                "pathway",
                "pathway",
                "cancer",
            ],
            "canonical_id": ["L1", "L2", "G1", "G2", "PW1", "PW2", "C1"],
            "node_index_within_type": [0, 1, 0, 1, 0, 1, 0],
        }
    )


def _edges() -> pd.DataFrame:
    rows = [
        ("e1", "lncRNA", "L1", "coexpressed_with", "gene", "G1", 0.8),
        ("e2", "gene", "G1", "member_of", "pathway", "PW1", 0.8),
        ("e3", "gene", "G2", "member_of", "pathway", "PW2", 0.8),
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
            "weight",
        ],
    )
    frame["source_node_id"] = frame.source_type + ":" + frame.source_canonical_id
    frame["target_node_id"] = frame.target_type + ":" + frame.target_canonical_id
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


def _rows():
    return [
        ("gene", "G1", "mutation_damage", 0.5),
        ("gene", "G1", "cnv_signed", -0.2),
        ("gene", "G2", "mutation_damage", -0.3),
    ]


def _evidence(tmp_path, rows=None, *, scope="subject", subject_id="PATIENT-7", cancer_id="C1"):
    rows = _rows() if rows is None else rows
    table = tmp_path / "patient_evidence.tsv"
    pd.DataFrame(rows, columns=["node_type", "canonical_id", "channel", "raw_value"]).to_csv(
        table, sep="\t", index=False
    )
    manifest = {
        "format": omics.EVIDENCE_FORMAT,
        "scope": scope,
        "cancer_id": cancer_id,
        "subject_id": subject_id,
        "channels": list(CHANNELS),
        "standardisation": {channel: {"center": 0.0, "scale": 1.0} for channel in CHANNELS},
        "table_sha256": hashlib.sha256(table.read_bytes()).hexdigest(),
        "row_count": len(rows),
        "clipped": True,
    }
    path = tmp_path / "patient_evidence.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return omics.read_evidence(table, path)


def _bundle(tmp_path, *, nodes=None, edges=None, rows=None):
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    nodes = _nodes() if nodes is None else nodes
    edges = _edges() if edges is None else edges
    bundle = _materialize_graph_bundle(
        nodes,
        edges,
        feature_edges=edges,
        canonical_edges=edges,
        graph_contract="CONTRACT-T",
    )
    blocks = pc.observed_blocks(bundle, _evidence(tmp_path, rows))
    omics.attach_node_omics(bundle, blocks)
    return bundle


def _model(bundle, seed: int = 0):
    torch = pytest.importorskip("torch")
    torch.manual_seed(seed)
    return build_model("cc_hhgt", bundle, len(FEATURES), _model_cfg())


def _pairs(pathways=("PW1", "PW2")) -> pd.DataFrame:
    rows = []
    for lncrna in ("L1", "L2"):
        for index, pathway in enumerate(pathways):
            rows.append(
                {
                    "lncrna_id": lncrna,
                    "pathway_id": pathway,
                    "cancer_id": "C1",
                    "pair_expression": 0.5 + 0.1 * index,
                    "pair_membership": 1.0 if pathway == "PW1" else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _intervention(**overrides) -> pc.Intervention:
    payload = {
        "cancer_id": "C1",
        "subject_id": "PATIENT-7",
        "node_type": "gene",
        "node_id": "G1",
        "channel": "mutation_damage",
        "mode": "inactivated",
        "value": 3.0,
    }
    payload.update(overrides)
    return pc.Intervention(**payload)


def _row(table: pd.DataFrame, lncrna: str, pathway: str) -> pd.Series:
    selected = table[
        table.lncrna_id.astype(str).eq(lncrna) & table.pathway_id.astype(str).eq(pathway)
    ]
    assert len(selected) == 1
    return selected.iloc[0]


def test_validate_intervention_records_a_non_causal_estimand(tmp_path) -> None:
    bundle = _bundle(tmp_path)
    evidence = _evidence(tmp_path)
    record = pc.validate_intervention(_intervention(), evidence, bundle)
    assert record["scope"] == pc.SCOPE
    assert record["causal_claim"] is False
    assert record["requires_orthogonal_validation"] is True
    assert "do(gene:G1.mutation_damage" in record["estimand"]
    assert record["intervention"]["observed_value"] == pytest.approx(0.5)
    assert record["intervention"]["intervention_value"] == pytest.approx(3.0)
    assert record["intervention"]["counterfactual_value_is_declared"] is True


@pytest.mark.parametrize(
    "intervention_kwargs, message",
    [
        ({"mode": "resistant"}, "PATIENT_CONDITIONING_UNKNOWN_MODE"),
        ({"subject_id": "PATIENT-8"}, "PATIENT_CONDITIONING_SUBJECT_MISMATCH"),
        ({"cancer_id": "BRCA"}, "PATIENT_CONDITIONING_CANCER_MISMATCH"),
        ({"node_type": "protein", "node_id": "P1"}, "PATIENT_CONDITIONING_NODE_TYPE_UNKNOWN"),
        ({"node_id": "G9"}, "PATIENT_CONDITIONING_NODE_NOT_IN_GRAPH"),
        ({"channel": "atac_accessibility"}, "PATIENT_CONDITIONING_CHANNEL_MISSING"),
        ({"node_id": "G2", "channel": "cnv_signed"}, "PATIENT_CONDITIONING_PARTICIPANT_UNMEASURED"),
        ({"value": None}, "PATIENT_CONDITIONING_INTERVENTION_VALUE_MISSING"),
        ({"value": 0.5}, "PATIENT_CONDITIONING_NO_INTERVENTION_CONTRAST"),
        ({"value": float("nan")}, "PATIENT_CONDITIONING_VALUE_NOT_FINITE"),
    ],
)
def test_validate_intervention_fails_closed_without_a_usable_contrast(
    tmp_path, intervention_kwargs, message
) -> None:
    bundle = _bundle(tmp_path)
    evidence = _evidence(tmp_path)
    with pytest.raises(RuntimeError, match=message):
        pc.validate_intervention(_intervention(**intervention_kwargs), evidence, bundle)


def test_cohort_evidence_cannot_be_presented_as_a_patient(tmp_path) -> None:
    bundle = _bundle(tmp_path)
    evidence = _evidence(tmp_path, scope="cohort", subject_id=None)
    with pytest.raises(RuntimeError, match="PATIENT_CONDITIONING_SUBJECT_SCOPE_REQUIRED"):
        pc.validate_intervention(_intervention(), evidence, bundle)


def test_degree_only_bundle_is_refused(tmp_path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    nodes, edges = _nodes(), _edges()
    bundle = _materialize_graph_bundle(
        nodes, edges, feature_edges=edges, canonical_edges=edges, graph_contract="CONTRACT-T"
    )
    with pytest.raises(RuntimeError, match="PATIENT_CONDITIONING_BUNDLE_NOT_AUGMENTED"):
        pc.validate_intervention(_intervention(), _evidence(tmp_path), bundle)


def test_intervened_blocks_change_exactly_one_cell(tmp_path) -> None:
    bundle = _bundle(tmp_path)
    evidence = _evidence(tmp_path)
    blocks = pc.observed_blocks(bundle, evidence)
    updated, writeback = pc.intervened_blocks(bundle, blocks, evidence, _intervention())
    for node_type, block in blocks.items():
        assert updated[node_type].shape == block.shape
    difference = updated["gene"] - blocks["gene"]
    assert int(np.count_nonzero(difference)) == 1
    assert writeback["node_type"] == "gene"
    assert writeback["observed_value"] == pytest.approx(0.5)
    assert writeback["intervention_value"] == pytest.approx(3.0)
    assert updated["gene"][writeback["row"], writeback["column"]] == pytest.approx(3.0)
    # The observed flag must survive: this is a declared state, not a missing assay.
    assert updated["gene"][writeback["row"], writeback["column"] + 1] == 1.0


def test_intervened_blocks_mode_observed_is_a_no_op(tmp_path) -> None:
    bundle = _bundle(tmp_path)
    evidence = _evidence(tmp_path)
    blocks = pc.observed_blocks(bundle, evidence)
    updated, _ = pc.intervened_blocks(
        bundle, blocks, evidence, _intervention(mode="observed")
    )
    for node_type, block in blocks.items():
        assert np.array_equal(updated[node_type], block)


def test_intervened_blocks_reject_unmeasured_and_unknown_channels(tmp_path) -> None:
    bundle = _bundle(tmp_path)
    evidence = _evidence(tmp_path)
    blocks = pc.observed_blocks(bundle, evidence)
    with pytest.raises(RuntimeError, match="PATIENT_CONDITIONING_CHANNEL_MISSING"):
        pc.intervened_blocks(bundle, blocks, evidence, _intervention(channel="atac_accessibility"))
    with pytest.raises(RuntimeError, match="PATIENT_CONDITIONING_PARTICIPANT_UNMEASURED"):
        pc.intervened_blocks(
            bundle, blocks, evidence, _intervention(node_id="G2", channel="cnv_signed")
        )


def test_conditioned_writes_and_restores_the_bundle_exactly(tmp_path) -> None:
    bundle = _bundle(tmp_path)
    evidence = _evidence(tmp_path)
    blocks = pc.observed_blocks(bundle, evidence)
    updated, _ = pc.intervened_blocks(bundle, blocks, evidence, _intervention())
    before = {node: bundle.hetero_data[node].x.detach().clone() for node in bundle.hetero_data.node_types}
    before_homogeneous = bundle.homogeneous["node_x"].detach().clone()
    with pc.conditioned(bundle, updated):
        inside = bundle.hetero_data["gene"].x.detach().numpy()
        changed = np.abs(inside - before["gene"].numpy()).max()
        assert changed > 1e-6
    for node, tensor in before.items():
        assert np.array_equal(bundle.hetero_data[node].x.detach().numpy(), tensor.numpy())
    assert np.array_equal(
        bundle.homogeneous["node_x"].detach().numpy(), before_homogeneous.numpy()
    )


def test_pair_batch_requires_columns_and_finite_features(tmp_path) -> None:
    bundle = _bundle(tmp_path)
    with pytest.raises(RuntimeError, match="PATIENT_CONDITIONING_PAIRS_EMPTY"):
        pc.pair_batch(bundle, _pairs().iloc[0:0], FEATURES, "cc_hhgt")
    with pytest.raises(RuntimeError, match="PATIENT_CONDITIONING_PAIR_COLUMNS_MISSING"):
        pc.pair_batch(bundle, _pairs().drop(columns=["pair_membership"]), FEATURES, "cc_hhgt")
    broken = _pairs()
    broken.loc[0, "pair_expression"] = np.nan
    with pytest.raises(RuntimeError, match="PATIENT_CONDITIONING_PAIR_FEATURE_INVALID"):
        pc.pair_batch(bundle, broken, FEATURES, "cc_hhgt")
    unknown = _pairs()
    unknown.loc[0, "lncrna_id"] = "L9"
    with pytest.raises(RuntimeError, match="PATIENT_CONDITIONING_PAIR_NODE_MISSING"):
        pc.pair_batch(bundle, unknown, FEATURES, "cc_hhgt")


def test_pair_batch_supports_both_model_kinds(tmp_path) -> None:
    bundle = _bundle(tmp_path)
    frame, batch = pc.pair_batch(bundle, _pairs(), FEATURES, "cc_hhgt")
    assert len(frame) == 4
    assert set(batch) == {"l", "p", "c", "x"}
    assert batch["x"].shape == (4, len(FEATURES))
    _, rgcn_batch = pc.pair_batch(bundle, _pairs(), FEATURES, "rgcn")
    assert rgcn_batch["x"].shape == (4, len(FEATURES))


def test_score_membership_is_frozen_and_finite(tmp_path) -> None:
    bundle = _bundle(tmp_path)
    model = _model(bundle)
    _, batch = pc.pair_batch(bundle, _pairs(), FEATURES, "cc_hhgt")
    graph = move_graph(bundle, "cpu", "cc_hhgt")
    scores = pc.score_membership(model, graph, batch)
    assert scores.shape == (4,)
    assert np.isfinite(scores).all()
    assert not model.training


def test_patient_relation_shift_reports_deltas_only_where_the_graph_connects(tmp_path) -> None:
    bundle = _bundle(tmp_path)
    model = _model(bundle)
    table, record = pc.patient_relation_shift(
        model,
        bundle,
        evidence=_evidence(tmp_path),
        intervention=_intervention(),
        pairs=_pairs(),
        features=FEATURES,
    )
    interaction = _row(table, "L1", "PW1")
    unaffected = _row(table, "L2", "PW2")
    assert abs(interaction.delta) > 1e-6
    assert abs(interaction.membership_intervened - interaction.membership_observed) > 1e-6
    assert abs(unaffected.delta) < 1e-9
    assert record["changed_pair_count"] == int((table.absolute_delta > pc.EFFECT_TOLERANCE).sum())
    assert record["causal_claim"] is False
    assert record["scope"] == pc.SCOPE
    assert record["pair_count"] == 4
    assert record["kind"] == "cc_hhgt"
    assert record["node_input_dim"] == bundle.node_input_dim
    assert "do(gene:G1.mutation_damage" in record["estimand"]
    assert record["intervention_identifier"] == _intervention().identifier()
    assert "per_cancer" in record["rank"]
    assert record["rank"]["per_cancer"]["C1"]["pairs"] == 4


def test_patient_relation_shift_with_observed_mode_has_no_effect(tmp_path) -> None:
    bundle = _bundle(tmp_path)
    model = _model(bundle)
    table, record = pc.patient_relation_shift(
        model,
        bundle,
        evidence=_evidence(tmp_path),
        intervention=_intervention(mode="observed"),
        pairs=_pairs(),
        features=FEATURES,
    )
    assert float(table.absolute_delta.max()) < 1e-9
    assert record["changed_pair_count"] == 0
    assert record["sign_flip_count"] == 0


def test_patient_relation_shift_is_deterministic(tmp_path) -> None:
    bundle = _bundle(tmp_path)
    model = _model(bundle)
    first, _ = pc.patient_relation_shift(
        model,
        bundle,
        evidence=_evidence(tmp_path),
        intervention=_intervention(),
        pairs=_pairs(),
        features=FEATURES,
    )
    second, _ = pc.patient_relation_shift(
        model,
        bundle,
        evidence=_evidence(tmp_path),
        intervention=_intervention(),
        pairs=_pairs(),
        features=FEATURES,
    )
    assert np.array_equal(first.membership_intervened, second.membership_intervened)
    assert np.array_equal(first.membership_observed, second.membership_observed)


def test_rank_summary_ranks_within_each_cancer() -> None:
    table = pd.DataFrame(
        {
            "cancer_id": ["C1"] * 4,
            "lncrna_id": ["L1", "L1", "L2", "L2"],
            "membership_observed": [3.0, 2.0, 1.0, 0.0],
            "membership_intervened": [0.0, 2.0, 1.0, 3.0],
        }
    )
    table["delta"] = table.membership_intervened - table.membership_observed
    summary = pc.rank_summary(table, top_k=2)
    rows = summary["rank_table"]
    assert [row["rank_observed"] for row in rows] == [1.0, 2.0, 3.0, 4.0]
    assert [row["rank_intervened"] for row in rows] == [4.0, 2.0, 3.0, 1.0]
    assert summary["per_cancer"]["C1"]["max_absolute_rank_shift"] == 3.0
    assert summary["per_cancer"]["C1"]["top_k_overlap"] == 1
    assert summary["per_cancer"]["C1"]["top_k"] == 2
    assert summary["per_cancer"]["C1"]["top_k_jaccard"] == pytest.approx(1 / 3)


def test_delta_table_flags_only_real_sign_changes() -> None:
    frame = pd.DataFrame(
        {
            "lncrna_id": ["L1", "L2", "L3"],
            "pathway_id": ["PW1", "PW1", "PW1"],
            "cancer_id": ["C1", "C1", "C1"],
        }
    )
    observed = np.asarray([1.0, 1.0, -1e-9])
    intervened = np.asarray([-1.0, 2.0, 1e-9])
    table = pc._delta_table(frame, observed, intervened)
    assert table.sign_flip.tolist() == [True, False, False]
    assert table.delta.tolist() == pytest.approx([-2.0, 1.0, 2e-9])
    assert table.absolute_delta.tolist() == pytest.approx([2.0, 1.0, 2e-9])


def test_interaction_attribution_passes_its_controls_on_a_connected_graph(tmp_path) -> None:
    bundle = _bundle(tmp_path)
    model = _model(bundle)
    record = pc.interaction_attribution(
        model,
        bundle,
        evidence=_evidence(tmp_path),
        intervention=_intervention(),
        pairs=_pairs(),
        features=FEATURES,
        path_target="L1",
        path_target_pathway="PW1",
        placebo_node="G2",
    )
    assert record["verdict"] == "INTERACTION_ATTRIBUTED_UNDER_FROZEN_MODEL"
    assert record["total_effect"] > 1e-6
    assert record["path_removed_effect"] < 1e-7
    assert record["attribution_share"] == pytest.approx(1.0)
    assert record["removed_edge_count"] == 2
    assert record["rewire_target_max_absolute_delta"] < 1e-7
    assert record["rewire_specific"] is True
    assert record["placebo_clean"] is True
    assert record["placebo"]["target_max_absolute_delta"] < 1e-7
    assert record["target_pair_count"] == 1
    assert record["biological_mediation_proven"] is False
    assert record["causal_claim"] is False
    assert record["verdict_scope"] == "WITHIN_FROZEN_MODEL_ONLY"
    assert record["total_effect_is_not_a_proof"] is True


def test_interaction_attribution_reports_no_effect_for_a_null_contrast(tmp_path) -> None:
    bundle = _bundle(tmp_path)
    model = _model(bundle)
    record = pc.interaction_attribution(
        model,
        bundle,
        evidence=_evidence(tmp_path),
        intervention=_intervention(mode="observed"),
        pairs=_pairs(),
        features=FEATURES,
        path_target="L1",
        path_target_pathway="PW1",
    )
    assert record["verdict"] == "NO_EFFECT"
    assert record["total_effect"] < 1e-9
    assert record["placebo"] is None


def test_interaction_attribution_requires_a_target_pair(tmp_path) -> None:
    bundle = _bundle(tmp_path)
    model = _model(bundle)
    with pytest.raises(RuntimeError, match="PATIENT_CONDITIONING_PATH_TARGET_NOT_IN_PAIRS"):
        pc.interaction_attribution(
            model,
            bundle,
            evidence=_evidence(tmp_path),
            intervention=_intervention(),
            pairs=_pairs(),
            features=FEATURES,
            path_target="G1",
        )


def test_interaction_attribution_rejects_an_unknown_placebo(tmp_path) -> None:
    bundle = _bundle(tmp_path)
    model = _model(bundle)
    with pytest.raises(RuntimeError, match="PATIENT_CONDITIONING_NODE_NOT_IN_GRAPH"):
        pc.interaction_attribution(
            model,
            bundle,
            evidence=_evidence(tmp_path),
            intervention=_intervention(),
            pairs=_pairs(),
            features=FEATURES,
            path_target="L1",
            path_target_pathway="PW1",
            placebo_node="G9",
        )


def test_rewiring_needs_an_alternative_target(tmp_path) -> None:
    nodes = _nodes()
    nodes = nodes[nodes.canonical_id.ne("PW2")].reset_index(drop=True)
    edges = _edges()
    edges = edges[edges.target_canonical_id.ne("PW2")].reset_index(drop=True)
    edges = edges[edges.source_canonical_id.ne("G2")].reset_index(drop=True)
    rows = [("gene", "G1", "mutation_damage", 0.5), ("gene", "G1", "cnv_signed", -0.2)]
    bundle = _bundle(tmp_path, nodes=nodes, edges=edges, rows=rows)
    model = _model(bundle)
    with pytest.raises(RuntimeError, match="PATIENT_CONDITIONING_REWIRE_UNAVAILABLE"):
        pc.interaction_attribution(
            model,
            bundle,
            evidence=_evidence(tmp_path, rows),
            intervention=_intervention(),
            pairs=_pairs(pathways=("PW1",)),
            features=FEATURES,
            path_target="L1",
            path_target_pathway="PW1",
        )


def test_interaction_attribution_is_unsupported_for_the_homogeneous_model(tmp_path) -> None:
    bundle = _bundle(tmp_path)
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    model = build_model("rgcn", bundle, len(FEATURES), _model_cfg())
    with pytest.raises(RuntimeError, match="PATIENT_CONDITIONING_GRAPH_KIND_UNSUPPORTED"):
        pc.interaction_attribution(
            model,
            bundle,
            evidence=_evidence(tmp_path),
            intervention=_intervention(),
            pairs=_pairs(),
            features=FEATURES,
            path_target="L1",
            path_target_pathway="PW1",
            kind="rgcn",
        )


def test_intervention_identifier_is_stable_and_sensitive() -> None:
    first = _intervention().identifier()
    assert first == _intervention().identifier()
    assert first != _intervention(value=2.0).identifier()
    assert first != _intervention(subject_id="PATIENT-8").identifier()
    assert len(first) == 32


def test_module_exposes_no_fitting_entry_point() -> None:
    forbidden = {"fit", "train", "fit_transform", "optimizer", "backward"}
    assert forbidden.isdisjoint(dir(pc))
    assert pc.SCOPE.endswith("NON_CAUSAL")
