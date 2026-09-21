from __future__ import annotations

import pandas as pd
import pytest

from cc_hhgt.gnn import (
    _materialize_graph_bundle,
    fold_node_features,
    runtime_bundle_for_step,
)
from cc_hhgt.graph_contract import build_fold_graph
from cc_hhgt.relation_sampling import RuntimeSamplingPolicy, schedule_runtime_edges


def test_runtime_chunk_is_connected_to_graph_tensors_and_pseudoheldout_filter() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    nodes = pd.DataFrame(
        {
            "node_id": ["lncRNA:L1", "lncRNA:L2", "gene:G1", "gene:G2", "state:S1", "cancer:A", "cancer:B"],
            "node_type": ["lncRNA", "lncRNA", "gene", "gene", "state", "cancer", "cancer"],
            "canonical_id": ["L1", "L2", "G1", "G2", "S1", "A", "B"],
            "node_index_within_type": [0, 1, 0, 1, 0, 0, 1],
        }
    )
    rows = []
    for index in range(8):
        source = "L1" if index % 2 == 0 else "L2"
        target = "G1" if index % 3 else "G2"
        cancer = "A" if index < 4 else "B"
        rows.append(
            {
                "edge_id": f"c{index}", "source_type": "lncRNA", "relation_type": "coexpressed_with",
                "target_type": "gene", "source_canonical_id": source, "target_canonical_id": target,
                "source_node_id": f"lncRNA:{source}", "target_node_id": f"gene:{target}",
                "cancer_id": cancer, "is_context_specific": True, "source_database": "TCGA",
                "weight": 0.5 + index / 20, "raw_effect": (-1 if index % 2 else 1) * (0.5 + index / 20),
                "relation_provenance": "cancer_unlabeled_context",
            }
        )
    for index, cancer in enumerate(("A", "B", "A", "B")):
        gene = "G1" if index % 2 == 0 else "G2"
        rows.append(
            {
                "edge_id": f"s{index}", "source_type": "gene", "relation_type": "positively_associated_with_state",
                "target_type": "state", "source_canonical_id": gene, "target_canonical_id": "S1",
                "source_node_id": f"gene:{gene}", "target_node_id": "state:S1", "cancer_id": cancer,
                "is_context_specific": True, "source_database": "TCGA_sample_scores", "weight": 0.7,
                "raw_effect": 0.7, "relation_provenance": "cancer_outcome_derived",
            }
        )
    for index, (lnc, cancer) in enumerate((("L1", "A"), ("L2", "B"), ("L2", "A"), ("L1", "B"))):
        rows.append(
            {
                "edge_id": f"e{index}", "source_type": "lncRNA", "relation_type": "expressed_in",
                "target_type": "cancer", "source_canonical_id": lnc, "target_canonical_id": cancer,
                "source_node_id": f"lncRNA:{lnc}", "target_node_id": f"cancer:{cancer}", "cancer_id": cancer,
                "is_context_specific": True, "source_database": "TCGA", "weight": 1.0,
                "raw_effect": None, "relation_provenance": "cancer_unlabeled_context",
            }
        )
    rows.append(
        {
            "edge_id": "global_static_na_cancer",
            "source_type": "lncRNA",
            "relation_type": "curated_relation",
            "target_type": "gene",
            "source_canonical_id": "L1",
            "target_canonical_id": "G1",
            "source_node_id": "lncRNA:L1",
            "target_node_id": "gene:G1",
            "cancer_id": None,
            "is_context_specific": False,
            "source_database": "curated",
            "weight": 1.0,
            "raw_effect": None,
            "relation_provenance": "global_static",
        }
    )
    edges = pd.DataFrame(rows)
    scheduled = schedule_runtime_edges(edges, RuntimeSamplingPolicy(edge_chunk_size=9))
    active = scheduled.loc[scheduled.runtime_chunk.eq(0)]
    base = _materialize_graph_bundle(
        nodes,
        active,
        feature_edges=scheduled,
        canonical_edges=scheduled,
        runtime_schedule=scheduled,
        relation_schema_edges=scheduled,
        active_runtime_chunk=0,
        graph_contract="CONTRACT-T",
    )
    episode = runtime_bundle_for_step(
        base,
        0,
        pseudoheldout_cancer="A",
        formal_heldout_cancers=set(),
    )
    assert episode.active_runtime_chunk == 0
    assert len(episode.active_edges) <= 9
    assert not (
        episode.active_edges.cancer_id.astype(str).eq("A")
        & episode.active_edges.relation_provenance.astype(str).eq("cancer_outcome_derived")
    ).any()
    assert set(base.hetero_data.edge_types) == set(episode.hetero_data.edge_types)

    # Degree features must be computed from the fully pseudoheldout-filtered
    # schedule.  Using the unfiltered base schedule here would silently encode
    # cancer A's outcome edges even though they are absent from active_edges.
    expected_edges = build_fold_graph(
        scheduled,
        heldout_cancers={"A"},
        mode="pseudoheldout",
        contract="CONTRACT-T",
        compute_fingerprint=False,
    ).edges
    expected_active = build_fold_graph(
        scheduled.loc[scheduled.runtime_chunk.eq(0)],
        heldout_cancers={"A"},
        mode="pseudoheldout",
        contract="CONTRACT-T",
        compute_fingerprint=False,
    ).edges
    assert episode.active_edges.edge_id.astype(str).tolist() == (
        expected_active.edge_id.astype(str).tolist()
    )
    global_chunk = int(
        scheduled.loc[
            scheduled.edge_id.astype(str).eq("global_static_na_cancer"),
            "runtime_chunk",
        ].iloc[0]
    )
    global_episode = runtime_bundle_for_step(
        base,
        global_chunk,
        pseudoheldout_cancer="A",
        formal_heldout_cancers=set(),
    )
    assert "global_static_na_cancer" in set(
        global_episode.active_edges.edge_id.astype(str)
    )
    gene_nodes = nodes.loc[nodes.node_type.eq("gene")].sort_values(
        "node_index_within_type"
    )
    expected_gene_x = fold_node_features(gene_nodes, expected_edges, "gene")
    assert episode.hetero_data["gene"].x.cpu().numpy() == pytest.approx(
        expected_gene_x
    )
    assert episode.hetero_data["gene"].x.cpu().numpy() != pytest.approx(
        base.hetero_data["gene"].x.cpu().numpy()
    )
