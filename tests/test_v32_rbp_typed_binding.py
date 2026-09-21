"""Phase 3 tests: typed lncRNA-protein binding materialisation.

Contracts pinned here:

* legacy mode reproduces the frozen pipeline exactly (ablation mode A anchor);
* ``partner_type == "rbp"`` rows are dropped by legacy and included by typed;
* RBPs never create a second entity -- partner identifiers are reused verbatim,
  so there can be no ``NONO:protein`` / ``NONO:RBP`` duplication;
* one pair may produce several typed relations, but never duplicate rows inside
  a (pair, context, graph_assay_class) cell;
* predictions are typed, never merged into physical binding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cc_hhgt.v32.rbp_assay import GRAPH_ASSAY_CLASSES, classify_assay
from cc_hhgt.v32.rbp_typed_binding import (
    LEGACY_RELATION_TYPE,
    BindingTyping,
    build_typed_lnc_protein_binding,
    verify_legacy_equivalence,
)


def _protein_gene() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gene_id": ["GENE:A", "GENE:B", "GENE:C"],
            "protein_id": ["UNIPROT:P1", "UNIPROT:P2", "UNIPROT:P3"],
            "uniprot_accession": ["P1", "P2", "P3"],
            "mapping_multiplicity": [1, 1, 1],
            "mapping_weight": [1.0, 1.0, 1.0],
        }
    )


def _dim_lnc() -> pd.DataFrame:
    return pd.DataFrame({"lncrna_id": ["LNC:1", "LNC:2"]})


def _dim_cancer() -> pd.DataFrame:
    return pd.DataFrame({"cancer_id": ["BRCA"], "tcga_code": ["BRCA"],
                         "english_name": ["Breast"], "chinese_name": ["乳腺癌"],
                         "synonyms": ["breast invasive carcinoma"]})


def _interaction(rows: list[dict]) -> pd.DataFrame:
    base = {
        "interaction_id": "", "lncrna_id": "LNC:1", "partner_id": "GENE:A",
        "partner_type": "protein", "relation_type": "binding_or_interaction",
        "direction": "", "disease_raw": "", "cell_line": "", "tissue": "",
        "experiment_raw": "eCLIP", "experiment_family": "physical_binding",
        "throughput": "", "is_experimental": True, "is_predicted": False,
        "pmid": "111", "source_database": "NPInter", "source_record_id": "R1",
        "mapping_status": "", "interaction_row_id": "",
    }
    out = []
    for index, row in enumerate(rows):
        merged = {**base, **row}
        merged["interaction_id"] = merged.get("interaction_id") or f"I{index}"
        out.append(merged)
    return pd.DataFrame(out)


def _build(frame: pd.DataFrame, typing: BindingTyping) -> pd.DataFrame:
    table, _ = build_typed_lnc_protein_binding(
        frame, _protein_gene(), _dim_lnc(), _dim_cancer(),
        typing=typing, classify=classify_assay,
    )
    return table


LEGACY = BindingTyping(include_rbp_partner_type=False, group_by_assay_class=False)
TYPED = BindingTyping(include_rbp_partner_type=True, group_by_assay_class=True)


# ---------------------------------------------------------------------------
# Legacy equivalence
# ---------------------------------------------------------------------------


def test_legacy_mode_emits_the_historical_relation_type() -> None:
    table = _build(_interaction([{}]), LEGACY)
    assert set(table.relation_type) == {LEGACY_RELATION_TYPE}
    assert "graph_assay_class" not in table.columns


def test_legacy_mode_drops_rbp_rows_exactly_as_before() -> None:
    frame = _interaction(
        [
            {"partner_type": "protein", "partner_id": "GENE:A"},
            {"partner_type": "rbp", "partner_id": "GENE:B"},
        ]
    )
    table = _build(frame, LEGACY)
    assert table.protein_id.tolist() == ["UNIPROT:P1"]


def test_typed_mode_includes_rbp_without_creating_a_new_entity() -> None:
    frame = _interaction(
        [
            {"partner_type": "protein", "partner_id": "GENE:A"},
            {"partner_type": "rbp", "partner_id": "GENE:B"},
        ]
    )
    table = _build(frame, TYPED)
    # the RBP partner maps into the SAME protein identifier space
    assert set(table.protein_id) == {"UNIPROT:P1", "UNIPROT:P2"}
    # no "RBP:" style identifier may appear anywhere
    assert not table.protein_id.astype(str).str.contains("RBP").any()
    assert not table.lncrna_id.astype(str).str.contains("RBP").any()


def test_verify_legacy_equivalence_detects_a_row_count_difference() -> None:
    produced = _build(_interaction([{}]), LEGACY)
    frozen = produced.iloc[0:0]
    result = verify_legacy_equivalence(produced, frozen)
    assert result["status"] == "FAIL_ROW_COUNT"


def test_verify_legacy_equivalence_detects_a_value_difference() -> None:
    produced = _build(_interaction([{}]), LEGACY)
    frozen = produced.copy()
    frozen.loc[:, "weight"] = frozen.weight + 1.0
    result = verify_legacy_equivalence(produced, frozen)
    assert result["status"] == "FAIL_VALUE_MISMATCH"
    assert result["mismatches"].get("weight") == 1


def test_verify_legacy_equivalence_passes_on_identical_tables() -> None:
    produced = _build(_interaction([{}]), LEGACY)
    assert verify_legacy_equivalence(produced, produced)["status"] == "PASS"


# ---------------------------------------------------------------------------
# Typed grouping
# ---------------------------------------------------------------------------


def test_one_pair_can_produce_several_typed_relations() -> None:
    frame = _interaction(
        [
            {"experiment_raw": "eCLIP", "source_record_id": "R1"},
            {"experiment_raw": "RIP-seq", "source_record_id": "R2"},
        ]
    )
    table = _build(frame, TYPED)
    assert sorted(table.relation_type) == [
        "binds_protein_eclip",
        "binds_protein_rip",
    ]
    assert len(table) == 2
    assert table.lncrna_id.nunique() == 1 and table.protein_id.nunique() == 1


def test_duplicate_database_records_collapse_within_an_assay_class() -> None:
    """Ten identical eCLIP records must not become ten identical edges."""

    frame = _interaction(
        [{"experiment_raw": "eCLIP", "source_database": f"DB{i}",
          "source_record_id": f"R{i}", "pmid": f"{100+i}"} for i in range(10)]
    )
    table = _build(frame, TYPED)
    assert len(table) == 1
    row = table.iloc[0]
    assert row.n_source_records == 10
    assert row.n_pmids == 10
    assert len(str(row.source_database).split("|")) == 10


def test_aggregation_keeps_max_weight_and_joins_sources() -> None:
    frame = _interaction(
        [
            {"experiment_raw": "eCLIP", "is_experimental": True,
             "source_database": "NPInter", "pmid": "111", "source_record_id": "R1"},
            {"experiment_raw": "eCLIP", "is_experimental": True,
             "source_database": "RNAInter", "pmid": "222", "source_record_id": "R2"},
        ]
    )
    table = _build(frame, TYPED)
    assert len(table) == 1
    row = table.iloc[0]
    assert float(row.weight) == pytest.approx(1.0)
    assert row.source_database == "NPInter|RNAInter"
    assert int(row.n_source_records) == 2
    assert int(row.n_pmids) == 2


def test_non_experimental_flag_forces_the_predicted_class() -> None:
    """An explicitly non-experimental record must not be grouped with a measured one.

    ``is_experimental=False`` is forced to the ``predicted`` class by design, so
    it can never be aggregated into the same relation as a real assay.
    """

    frame = _interaction(
        [
            {"experiment_raw": "eCLIP", "is_experimental": True, "source_record_id": "R1"},
            {"experiment_raw": "eCLIP", "is_experimental": False, "source_record_id": "R2"},
        ]
    )
    table = _build(frame, TYPED)
    assert len(table) == 2
    assert set(table.relation_type) == {"binds_protein_eclip", "binds_protein_predicted"}


def test_relation_types_stay_in_the_closed_vocabulary() -> None:
    frame = _interaction(
        [
            {"experiment_raw": "eCLIP"},
            {"experiment_raw": "RIP"},
            {"experiment_raw": "catRAPID", "source_record_id": "R2"},
            {"experiment_raw": "", "source_record_id": "R3"},
        ]
    )
    table = _build(frame, TYPED)
    allowed = {f"binds_protein_{cls}" for cls in GRAPH_ASSAY_CLASSES}
    assert set(table.relation_type) <= allowed


# ---------------------------------------------------------------------------
# Predictions must never masquerade as measurements
# ---------------------------------------------------------------------------


def test_predicted_rows_get_their_own_relation_type() -> None:
    frame = _interaction(
        [
            {"experiment_raw": "Scan pipeline widely used MATCH algorithm",
             "source_record_id": "R1", "is_experimental": False, "is_predicted": True},
            {"experiment_raw": "eCLIP", "source_record_id": "R2"},
        ]
    )
    table = _build(frame, TYPED)
    assert set(table.relation_type) == {"binds_protein_predicted", "binds_protein_eclip"}
    predicted = table.loc[table.relation_type.eq("binds_protein_predicted")].iloc[0]
    assert predicted.graph_assay_class == "predicted"


def test_predicted_never_shares_a_relation_with_eclip() -> None:
    frame = _interaction(
        [
            {"experiment_raw": "catRAPID", "source_record_id": "R1"},
            {"experiment_raw": "eCLIP", "source_record_id": "R2"},
        ]
    )
    table = _build(frame, TYPED)
    assert "binds_protein_predicted" in set(table.relation_type)
    assert "binds_protein_eclip" in set(table.relation_type)
    # the two must be separate rows even for the same pair and context
    assert len(table) == 2


def test_unknown_assay_fails_closed_into_its_own_class() -> None:
    frame = _interaction([{"experiment_raw": "", "is_predicted": False,
                           "is_experimental": None, "experiment_family": ""}])
    table = _build(frame, TYPED)
    assert table.iloc[0].graph_assay_class == "unknown"
    assert table.iloc[0].relation_type == "binds_protein_unknown"


# ---------------------------------------------------------------------------
# Context handling must not change
# ---------------------------------------------------------------------------


def test_context_mapping_is_untouched_by_typing() -> None:
    """Typing must not globalise or re-map any cancer context."""

    frame = _interaction(
        [
            {"experiment_raw": "eCLIP", "cell_line": "HepG2"},
            {"experiment_raw": "eCLIP", "source_record_id": "R2", "cell_line": "K562"},
        ]
    )
    table = _build(frame, TYPED)
    # HepG2 maps to LIHC via the existing authority; K562 has no mapping and is
    # excluded rather than guessed.
    assert table.cancer_id.dropna().astype(str).unique().tolist() == ["LIHC"]
    assert bool(table.is_context_specific.all())


def test_context_free_rows_stay_global() -> None:
    table = _build(_interaction([{"experiment_raw": "eCLIP"}]), TYPED)
    assert pd.isna(table.iloc[0].cancer_id)
    assert bool(table.iloc[0].is_context_specific) is False


def test_missing_required_column_fails_loudly() -> None:
    frame = _interaction([{}]).drop(columns=["partner_type"])
    with pytest.raises(ValueError):
        _build(frame, TYPED)
