"""Phase 3/7 tests: typed binding relations inside the formal G0/G1/G2 graph.

Contracts pinned here:

* typed relation types are emitted per graph assay class;
* context-specific eCLIP is admitted under its own edge role, keeps its cancer
  context, and is never globalised;
* G0 contains no binding at all, G1 gains binding but still no PPI, G2 adds PPI --
  the scientific meaning of the three arms is unchanged;
* predictions do not enter message passing unless explicitly requested.
"""

from __future__ import annotations

import pandas as pd
import pytest

from cc_hhgt.v32.formal_graph import (
    CONTEXT_ECLIP_ROLE,
    G1_ROLES,
    G2_ONLY_ROLES,
    GLOBAL_BINDING_ROLE,
    _variant_mask,
    materialize_typed_lnc_protein_binding,
    typed_global_role,
)


def _binding(rows: list[dict]) -> pd.DataFrame:
    base = {
        "lncrna_id": "LNC:1",
        "protein_id": "UNIPROT:P1",
        "weight": 1.0,
        "cancer_id": pd.NA,
        "is_context_specific": False,
        "graph_assay_class": "eclip",
        "relation_type": "binds_protein_eclip",
        "source_database": "NPInter",
    }
    return pd.DataFrame([{**base, **row} for row in rows])


# ---------------------------------------------------------------------------
# Typed emission
# ---------------------------------------------------------------------------


def test_typed_relations_are_emitted_per_assay_class() -> None:
    frame = _binding(
        [
            {"graph_assay_class": "eclip", "relation_type": "binds_protein_eclip"},
            {"graph_assay_class": "rip", "relation_type": "binds_protein_rip",
             "protein_id": "UNIPROT:P2"},
            {"graph_assay_class": "other_clip", "relation_type": "binds_protein_other_clip",
             "protein_id": "UNIPROT:P3"},
        ]
    )
    edges = materialize_typed_lnc_protein_binding(frame)
    assert set(edges.relation_type) == {
        "binds_protein_eclip", "binds_protein_rip", "binds_protein_other_clip"
    }
    # one dedicated role per relation: safe_graph.ROLE_CONTRACTS is 1:1
    assert set(edges.edge_role) == {
        typed_global_role("eclip"),
        typed_global_role("rip"),
        typed_global_role("other_clip"),
    }
    assert edges.is_context_specific.eq(False).all()


def test_declared_relation_type_must_match_the_assay_class() -> None:
    frame = _binding([{"graph_assay_class": "rip", "relation_type": "binds_protein_eclip"}])
    with pytest.raises(RuntimeError):
        materialize_typed_lnc_protein_binding(frame)


def test_unknown_assay_class_is_rejected() -> None:
    frame = _binding([{"graph_assay_class": "not_a_class"}])
    with pytest.raises(RuntimeError):
        materialize_typed_lnc_protein_binding(frame)


# ---------------------------------------------------------------------------
# Predictions are opt-in
# ---------------------------------------------------------------------------


def test_predictions_are_excluded_by_default() -> None:
    frame = _binding(
        [
            {"graph_assay_class": "eclip", "relation_type": "binds_protein_eclip"},
            {"graph_assay_class": "predicted", "relation_type": "binds_protein_predicted",
             "protein_id": "UNIPROT:P9"},
        ]
    )
    edges = materialize_typed_lnc_protein_binding(frame)
    assert "binds_protein_predicted" not in set(edges.relation_type)


def test_predictions_can_be_admitted_explicitly() -> None:
    frame = _binding(
        [
            {"graph_assay_class": "eclip", "relation_type": "binds_protein_eclip"},
            {"graph_assay_class": "predicted", "relation_type": "binds_protein_predicted",
             "protein_id": "UNIPROT:P9"},
        ]
    )
    edges = materialize_typed_lnc_protein_binding(frame, include_predicted=True)
    assert "binds_protein_predicted" in set(edges.relation_type)


# ---------------------------------------------------------------------------
# Context handling
# ---------------------------------------------------------------------------


def test_context_eclip_keeps_its_cancer_and_its_own_role() -> None:
    # distinct pairs, so the two edges do not collide on EDGE_KEYS
    frame = _binding(
        [
            {"cancer_id": "LIHC", "is_context_specific": True,
             "protein_id": "UNIPROT:P1"},
            {"cancer_id": pd.NA, "is_context_specific": False,
             "protein_id": "UNIPROT:P2"},
        ]
    )
    edges = materialize_typed_lnc_protein_binding(frame)
    contextual = edges.loc[edges.is_context_specific]
    assert len(contextual) == 1
    assert contextual.iloc[0].cancer_id == "LIHC"
    assert contextual.iloc[0].edge_role == CONTEXT_ECLIP_ROLE
    global_rows = edges.loc[~edges.is_context_specific]
    assert len(global_rows) == 1
    assert global_rows.iloc[0].edge_role == typed_global_role("eclip")
    assert pd.isna(global_rows.iloc[0].cancer_id)


def test_context_rows_are_never_globalised() -> None:
    frame = _binding([{"cancer_id": "LIHC", "is_context_specific": True}])
    edges = materialize_typed_lnc_protein_binding(frame)
    assert edges.is_context_specific.all()
    assert set(edges.cancer_id) == {"LIHC"}


def test_non_eclip_context_rows_are_not_admitted() -> None:
    """Only eCLIP has an approved context role; nothing else is broadened."""

    frame = _binding(
        [
            {"graph_assay_class": "rip", "relation_type": "binds_protein_rip",
             "cancer_id": "LIHC", "is_context_specific": True},
        ]
    )
    edges = materialize_typed_lnc_protein_binding(frame)
    assert edges.empty


def test_context_admission_can_be_switched_off() -> None:
    frame = _binding(
        [
            {"cancer_id": "LIHC", "is_context_specific": True},
            {"cancer_id": pd.NA, "is_context_specific": False},
        ]
    )
    edges = materialize_typed_lnc_protein_binding(frame, include_context_eclip=False)
    assert len(edges) == 1
    assert edges.is_context_specific.eq(False).all()


def test_a_context_specific_edge_never_becomes_pan_cancer() -> None:
    """A single mapped cancer must not fan out to other cancers."""

    frame = _binding([{"cancer_id": "LIHC", "is_context_specific": True}])
    edges = materialize_typed_lnc_protein_binding(frame)
    assert edges.cancer_id.nunique() == 1
    assert edges.cancer_id.iloc[0] == "LIHC"


def test_hepg2_like_context_and_k562_like_absence_behave_differently() -> None:
    """Mapped context enters; unmapped context was already excluded upstream.

    The context authority maps HepG2->LIHC and has no K562 entry, so a K562 row
    never reaches this builder as a context-specific row.  If it arrives with a
    null cancer it is treated as global, not promoted to a cancer.
    """

    mapped = _binding([{"cancer_id": "LIHC", "is_context_specific": True}])
    assert len(materialize_typed_lnc_protein_binding(mapped)) == 1

    unmapped = _binding([{"cancer_id": pd.NA, "is_context_specific": False}])
    edges = materialize_typed_lnc_protein_binding(unmapped)
    assert pd.isna(edges.iloc[0].cancer_id)
    assert edges.iloc[0].edge_role == typed_global_role("eclip")


def test_candidate_restriction_applies_to_both_paths() -> None:
    frame = _binding(
        [
            {"lncrna_id": "LNC:1", "cancer_id": "LIHC", "is_context_specific": True},
            {"lncrna_id": "LNC:2", "cancer_id": "LIHC", "is_context_specific": True,
             "protein_id": "UNIPROT:P2"},
            {"lncrna_id": "LNC:3"},
        ]
    )
    edges = materialize_typed_lnc_protein_binding(frame, candidate_lncrnas=["LNC:1"])
    assert set(edges.source_id) == {"LNC:1"}


# ---------------------------------------------------------------------------
# G0 / G1 / G2 invariants
# ---------------------------------------------------------------------------


def _mask(roles: list[str]) -> dict[str, bool]:
    edges = pd.DataFrame({"edge_role": roles})
    return {arm: bool(_variant_mask(edges, arm).all()) for arm in ("G0", "G1", "G2")}


def test_binding_roles_are_absent_from_g0_and_present_from_g1() -> None:
    edges = pd.DataFrame(
        {
            "edge_role": [
                "transductive_expression_eligibility",
                GLOBAL_BINDING_ROLE,
                CONTEXT_ECLIP_ROLE,
                "static_protein_gene_encoding",
            ]
        }
    )
    g0 = _variant_mask(edges, "G0")
    g1 = _variant_mask(edges, "G1")
    g2 = _variant_mask(edges, "G2")
    # G0 keeps only the non-binding eligibility edge
    assert g0.tolist() == [True, False, False, False]
    # G1 gains binding and protein encoding
    assert g1.tolist() == [True, True, True, True]
    assert g2.all()


def test_g1_still_has_no_ppi_and_g2_does() -> None:
    edges = pd.DataFrame({"edge_role": ["static_symmetric_ppi"]})
    assert not _variant_mask(edges, "G0").any()
    assert not _variant_mask(edges, "G1").any()
    assert _variant_mask(edges, "G2").all()


def test_context_eclip_role_is_a_binding_role() -> None:
    assert CONTEXT_ECLIP_ROLE in G1_ROLES
    assert CONTEXT_ECLIP_ROLE not in G2_ONLY_ROLES
    assert GLOBAL_BINDING_ROLE in G1_ROLES


def test_variant_mask_rejects_an_unknown_arm() -> None:
    with pytest.raises(ValueError):
        _variant_mask(pd.DataFrame({"edge_role": []}), "G9")


def test_emitted_edges_select_correctly_in_every_arm() -> None:
    frame = _binding(
        [
            {"cancer_id": pd.NA, "is_context_specific": False},
            {"cancer_id": "LIHC", "is_context_specific": True, "protein_id": "UNIPROT:P2"},
        ]
    )
    edges = materialize_typed_lnc_protein_binding(frame)
    assert not _variant_mask(edges, "G0").any(), "G0 must contain no eCLIP binding"
    assert _variant_mask(edges, "G1").all(), "G1 must contain the allowed eCLIP edges"
    assert _variant_mask(edges, "G2").all()


def test_context_edge_takes_precedence_over_a_global_edge_for_the_same_pair() -> None:
    """EDGE_KEYS carries neither edge_role nor cancer_id, so one pair can hold
    only one edge per relation type.  When a pair is supported both ways the
    context-specific edge must win: keeping the global edge would broadcast
    context-specific evidence to every cancer, which the task forbids.
    """

    frame = _binding(
        [
            {"cancer_id": "LIHC", "is_context_specific": True},
            {"cancer_id": pd.NA, "is_context_specific": False},
        ]
    )
    edges = materialize_typed_lnc_protein_binding(frame)
    assert len(edges) == 1
    row = edges.iloc[0]
    assert row.edge_role == CONTEXT_ECLIP_ROLE
    assert bool(row.is_context_specific) is True
    assert row.cancer_id == "LIHC"


def test_global_edge_survives_when_no_context_edge_competes() -> None:
    frame = _binding([{"cancer_id": pd.NA, "is_context_specific": False}])
    edges = materialize_typed_lnc_protein_binding(frame)
    assert len(edges) == 1
    assert edges.iloc[0].edge_role == typed_global_role("eclip")