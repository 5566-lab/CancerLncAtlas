from __future__ import annotations

import pandas as pd

from cc_hhgt.relation_sampling import (
    RuntimeSamplingPolicy,
    schedule_runtime_edges,
    signal_retention_audit,
)


def test_runtime_schedule_is_non_destructive_and_retains_both_coexpression_signs() -> None:
    edges = pd.DataFrame(
        {
            "edge_id": [f"e{i}" for i in range(8)],
            "source_type": ["lncRNA"] * 6 + ["gene"] * 2,
            "relation_type": ["coexpressed_with"] * 6 + ["member_of"] * 2,
            "target_type": ["gene"] * 6 + ["pathway"] * 2,
            "source_canonical_id": ["L1", "L1", "L1", "L2", "L2", "L2", "G1", "G2"],
            "target_canonical_id": ["G1", "G2", "G3", "G1", "G2", "G3", "P1", "P1"],
            "cancer_id": ["BRCA"] * 6 + [None, None],
            "weight": [0.9, 0.8, 0.3, 0.7, 0.6, 0.2, 1.0, 1.0],
            "raw_effect": [0.9, -0.8, 0.3, -0.7, 0.6, -0.2, None, None],
        }
    )
    policy = RuntimeSamplingPolicy(edge_chunk_size=3)
    scheduled = schedule_runtime_edges(edges, policy)
    assert len(scheduled) == len(edges)
    assert set(scheduled.edge_id) == set(edges.edge_id)
    assert scheduled.groupby("runtime_chunk").size().max() <= 3
    n_chunks = scheduled.runtime_chunk.nunique()
    frequent = set(
        scheduled.relation_type.value_counts().loc[lambda x: x >= n_chunks].index
    )
    for _, chunk in scheduled.groupby("runtime_chunk"):
        assert frequent.issubset(set(chunk.relation_type))
    audit = signal_retention_audit(edges, edges, scheduled, policy)
    assert audit.audit_status.eq("PASS").all()
    assert audit.weighted_signal_retention.eq(1.0).all()
    coexpression = audit.loc[audit.relation_family.str.contains("coexpressed_with")].iloc[0]
    assert coexpression.positive_mass_retention == 1.0
    assert coexpression.negative_mass_retention == 1.0
