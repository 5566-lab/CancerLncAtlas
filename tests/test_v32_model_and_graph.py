from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cc_hhgt.v32.baselines import (
    assert_no_forbidden_requested_features,
    build_safe_hashed_design,
)
from cc_hhgt.v32.model import (
    assert_exact_l1_fallback,
    build_v32_cc_hhgt_residual,
    build_v32_hierarchical_evidence_hhgt,
    compose_bounded_residual_numpy,
)
from cc_hhgt.v32.safe_graph import build_safe_graph, graph_is_invariant_to_external_evidence


def _edges():
    rows = [
        ("protein", "PR1", "encoded_by", "gene", "G1", "static_protein_gene_encoding"),
        ("gene", "G1", "member_of_positive", "pathway", "P1", "static_signed_pathway_membership_positive"),
        ("pathway", "P1", "member_of_family", "pathway_family", "F1", "static_pathway_hierarchy"),
    ]
    frame = pd.DataFrame(
        rows,
        columns=["source_type", "source_id", "relation_type", "target_type", "target_id", "edge_role"],
    )
    frame["source_split"] = "static"
    frame["outcome_derived"] = False
    frame["observed"] = True
    frame["weight"] = 1.0
    frame["relation_polarity"] = 1.0
    frame["raw_effect"] = pd.NA
    frame["requires_fold_localization"] = False
    frame["edge_outer_fold"] = pd.NA
    frame["cancer_id"] = pd.NA
    frame["is_context_specific"] = False
    return frame


def _fresh_relation(
    source_type,
    source_id,
    relation_type,
    target_type,
    target_id,
    edge_role,
    *,
    source_split="static",
    polarity=1.0,
    edge_outer_fold=None,
    cancer_id=None,
    is_context_specific=False,
):
    return {
        "source_type": source_type,
        "source_id": source_id,
        "relation_type": relation_type,
        "target_type": target_type,
        "target_id": target_id,
        "edge_role": edge_role,
        "source_split": source_split,
        "outcome_derived": False,
        "observed": True,
        "weight": 0.8,
        "relation_polarity": polarity,
        "raw_effect": None,
        "requires_fold_localization": False,
        "edge_outer_fold": edge_outer_fold,
        "cancer_id": cancer_id,
        "is_context_specific": is_context_specific,
    }


def test_external_evidence_cannot_change_primary_graph() -> None:
    base = _edges()
    evidence = pd.DataFrame({"edge_role": ["literature"]})
    assert graph_is_invariant_to_external_evidence(base, evidence, outer_fold=0)
    unsafe = base.iloc[[0]].copy()
    unsafe[["source_type", "target_type", "edge_role"]] = ["lncRNA", "pathway", "pair_evidence"]
    with pytest.raises(RuntimeError, match="Unsafe edges"):
        build_safe_graph(unsafe, outer_fold=0)


def test_fresh_safe_graph_contract_accepts_typed_roles_and_reciprocal_ppi() -> None:
    rows = [
        _fresh_relation(
            "lncRNA", "L1", "expressed_in", "cancer", "BRCA",
            "transductive_expression_eligibility",
            source_split="all_samples_pre_outcome", cancer_id="BRCA",
            is_context_specific=True,
        ),
        _fresh_relation(
            "lncRNA", "L1", "coexpressed_negative", "gene", "G1",
            "fold_train_coexpression_negative",
            source_split="train", polarity=-1.0, edge_outer_fold=2, cancer_id="BRCA",
            is_context_specific=True,
        ),
        _fresh_relation(
            "gene", "G1", "member_of_negative", "pathway", "P1",
            "static_signed_pathway_membership_negative", polarity=-1.0,
        ),
        _fresh_relation(
            "pathway", "P1", "member_of_family", "pathway_family", "F1",
            "static_pathway_hierarchy",
        ),
        _fresh_relation(
            "lncRNA", "L1", "binds_protein", "protein", "PR1",
            "static_global_lnc_protein_binding",
        ),
        _fresh_relation(
            "protein", "PR1", "encoded_by", "gene", "G1",
            "static_protein_gene_encoding",
        ),
        _fresh_relation(
            "protein", "PR1", "physical_interaction", "protein", "PR2",
            "static_symmetric_ppi",
        ),
        _fresh_relation(
            "protein", "PR2", "physical_interaction", "protein", "PR1",
            "static_symmetric_ppi",
        ),
    ]
    safe = build_safe_graph(pd.DataFrame(rows), outer_fold=2)
    assert safe.manifest["ppi_reciprocal_message_count"] == 2
    assert safe.manifest["transductive_expression_eligibility_declared"] is True
    assert safe.manifest["relation_counts"]["coexpressed_negative"] == 1


def test_fresh_safe_graph_rejects_wrong_split_polarity_context_and_half_ppi() -> None:
    coexpression = pd.DataFrame(
        [
            _fresh_relation(
                "lncRNA", "L1", "coexpressed_positive", "gene", "G1",
                "fold_train_coexpression_positive",
                source_split="validation", edge_outer_fold=0, cancer_id="BRCA",
            )
        ]
    )
    with pytest.raises(RuntimeError, match="Unsafe edges"):
        build_safe_graph(coexpression, outer_fold=0)

    signed = _edges()
    signed.loc[signed.relation_type.eq("member_of_positive"), "relation_polarity"] = -1.0
    with pytest.raises(RuntimeError, match="Unsafe edges"):
        build_safe_graph(signed, outer_fold=0)

    context_binding = pd.DataFrame(
        [
            _fresh_relation(
                "lncRNA", "L1", "binds_protein", "protein", "PR1",
                "static_global_lnc_protein_binding",
                cancer_id="BRCA", is_context_specific=True,
            )
        ]
    )
    with pytest.raises(RuntimeError, match="Unsafe edges"):
        build_safe_graph(context_binding, outer_fold=0)

    half_ppi = pd.DataFrame(
        [
            _fresh_relation(
                "protein", "PR1", "physical_interaction", "protein", "PR2",
                "static_symmetric_ppi",
            )
        ]
    )
    with pytest.raises(RuntimeError, match="exactly two reciprocal"):
        build_safe_graph(half_ppi, outer_fold=0)


def test_baseline_design_ignores_evidence_and_rejects_requested_evidence() -> None:
    frame = pd.DataFrame(
        {
            "cancer_id": ["A", "B"],
            "lncrna_id": ["L1", "L2"],
            "pathway_id": ["P1", "P2"],
            "pathway_family_id": ["F1", "F2"],
            "discovery_effect": [0.3, -0.2],
            "discovery_neglog10_fdr": [3.0, 2.0],
            "detection_rate": [0.5, 0.4],
        }
    )
    first = build_safe_hashed_design(frame, n_features=2048)
    with_evidence = frame.assign(drug_support=[999, -999], regulatory_evidence_confidence=[1, 0])
    second = build_safe_hashed_design(with_evidence, n_features=2048)
    assert (first != second).nnz == 0
    with pytest.raises(RuntimeError):
        assert_no_forbidden_requested_features(["discovery_effect", "drug_support"])


def test_bounded_residual_and_exact_fallback() -> None:
    base = np.array([-1.0, 0.0, 1.0])
    final, contribution = compose_bounded_residual_numpy(
        base, np.array([100.0, -100.0, 0.5]), np.array([1.0, 0.5, 0.0])
    )
    assert np.max(np.abs(contribution)) <= 1.0
    fallback, contribution = compose_bounded_residual_numpy(
        base, np.ones(3), np.zeros(3)
    )
    assert np.array_equal(contribution, np.zeros(3))
    assert_exact_l1_fallback(base, fallback)


def test_torch_forward_is_zero_residual_without_optimizer() -> None:
    torch = pytest.importorskip("torch")

    class Encoder(torch.nn.Module):
        def encode(self, graph):
            return graph

    model = build_v32_cc_hhgt_residual(Encoder(), hidden_channels=8, context_features=4, dropout=0.0)
    encoded = {
        "lncRNA": torch.randn(3, 8),
        "pathway": torch.randn(2, 8),
        "cancer": torch.randn(2, 8),
    }
    base = torch.tensor([0.2, -0.4])
    with torch.no_grad():
        output = model(
            encoded,
            {"l": torch.tensor([0, 1]), "p": torch.tensor([0, 1]), "c": torch.tensor([1, 0])},
            base,
            torch.zeros(2, 4),
            encoded=encoded,
        )
    assert torch.equal(output["final_logit"], base)
    assert torch.count_nonzero(output["graph_residual"]) == 0


def test_hierarchical_evidence_gate_is_exactly_zero_when_unavailable() -> None:
    torch = pytest.importorskip("torch")

    class Encoder(torch.nn.Module):
        def encode(self, graph):
            return graph

    model = build_v32_hierarchical_evidence_hhgt(
        Encoder(), hidden_channels=8, context_features=4, dropout=0.0
    )
    encoded = {
        "lncRNA": torch.randn(2, 8),
        "pathway": torch.randn(2, 8),
        "cancer": torch.randn(1, 8),
    }
    batch = {
        "l": torch.tensor([0, 1]),
        "p": torch.tensor([0, 1]),
        "c": torch.tensor([0, 0]),
        "modality_probability": torch.full((2, 3), float("nan")),
        "modality_available": torch.zeros(2, 3, dtype=torch.bool),
    }
    base = torch.tensor([0.2, -0.4])
    output = model(encoded, batch, base, torch.zeros(2, 4), encoded=encoded)
    assert torch.equal(output["final_logit"], base)
    assert torch.count_nonzero(output["modality_gates"]) == 0
    assert torch.count_nonzero(output["evidence_message"]) == 0
    batch["modality_probability"] = torch.tensor([[0.9, float("nan"), float("nan")]] * 2)
    batch["modality_available"][:, 0] = True
    output = model(encoded, batch, base, torch.zeros(2, 4), encoded=encoded)
    assert torch.all(output["modality_gates"][:, 0] > 0)
    assert torch.count_nonzero(output["evidence_message"]) > 0
