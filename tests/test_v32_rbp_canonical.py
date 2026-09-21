"""Phase 4 tests: canonical experiment identity and duplicate collapse."""

from __future__ import annotations

import pandas as pd
import pytest

from cc_hhgt.v32.rbp_canonical import (
    CANONICAL_KEY_FIELDS,
    CANONICAL_KEY_VERSION,
    canonical_experiment_key,
    collapse_canonical_experiments,
    duplicate_report,
)


def _frame(rows: list[dict]) -> pd.DataFrame:
    base = {
        "lncrna_id": "LNC:1", "partner_id": "GENE:A", "pmid": "111",
        "assay_subtype": "eclip", "cell_line": "HepG2", "tissue": "liver",
        "source_database": "NPInter", "source_record_id": "R1",
    }
    return pd.DataFrame([{**base, **row} for row in rows])


# ---------------------------------------------------------------------------
# Key determinism and shape
# ---------------------------------------------------------------------------


def test_key_is_deterministic() -> None:
    args = ("LNC:1", "GENE:A", "111", "eclip", "HepG2", "liver")
    first = canonical_experiment_key(*args)
    for _ in range(20):
        assert canonical_experiment_key(*args) == first


def test_key_is_versioned_and_namespaced() -> None:
    key = canonical_experiment_key("L", "P", "1", "eclip")
    assert key.startswith("CEXPV1:")
    assert CANONICAL_KEY_VERSION == "RBP_CANONICAL_EXPERIMENT_KEY_V1"


@pytest.mark.parametrize(
    "field, alt",
    [
        ("lncrna_id", "LNC:2"),
        ("partner_id", "GENE:B"),
        ("pmid", "222"),
        ("assay_subtype", "rip"),
        ("cell_line", "K562"),
        ("tissue", "brain"),
    ],
)
def test_every_key_component_changes_the_key(field: str, alt: str) -> None:
    base = {
        "lncrna_id": "LNC:1", "partner_id": "GENE:A", "pmid": "111",
        "assay_subtype": "eclip", "cell_line": "HepG2", "tissue": "liver",
    }
    other = {**base, field: alt}
    assert canonical_experiment_key(**base) != canonical_experiment_key(**other)


def test_key_field_order_is_part_of_the_contract() -> None:
    assert CANONICAL_KEY_FIELDS == (
        "lncrna_id", "partner_id", "pmid", "assay_subtype", "cell_line", "tissue",
    )


def test_case_and_padding_do_not_change_identity() -> None:
    a = canonical_experiment_key("lnc:1", "gene:a", "111", "eCLIP", " hepg2 ", "Liver")
    b = canonical_experiment_key("LNC:1", "GENE:A", "111", "eclip", "HepG2", "liver")
    assert a == b


def test_placeholder_values_are_treated_as_absent() -> None:
    a = canonical_experiment_key("L", "P", "1", "eclip", "NA", "unknown")
    b = canonical_experiment_key("L", "P", "1", "eclip", "", None)
    assert a == b


# ---------------------------------------------------------------------------
# Collapse behaviour
# ---------------------------------------------------------------------------


def test_same_experiment_in_three_databases_collapses_to_one() -> None:
    frame = _frame(
        [
            {"source_database": "NPInter", "source_record_id": "A"},
            {"source_database": "RNAInter", "source_record_id": "B"},
            {"source_database": "LncTarD", "source_record_id": "C"},
        ]
    )
    result = collapse_canonical_experiments(frame)
    assert len(result.collapsed) == 1
    row = result.collapsed.iloc[0]
    assert row.source_record_count == 3
    assert row.source_database_count == 3
    assert row.source_database_list == "LncTarD|NPInter|RNAInter"
    assert result.audit["canonical_experiments_seen_in_multiple_databases"] == 1


def test_different_assays_on_the_same_pmid_stay_separate() -> None:
    frame = _frame(
        [
            {"assay_subtype": "eclip", "source_database": "NPInter"},
            {"assay_subtype": "rip", "source_database": "NPInter", "source_record_id": "R2"},
        ]
    )
    result = collapse_canonical_experiments(frame)
    assert len(result.collapsed) == 2
    assert result.audit["canonical_experiments_seen_in_multiple_databases"] == 0


def test_different_cell_lines_stay_separate() -> None:
    frame = _frame(
        [
            {"cell_line": "HepG2", "source_database": "NPInter"},
            {"cell_line": "K562", "source_database": "NPInter", "source_record_id": "R2"},
        ]
    )
    assert len(collapse_canonical_experiments(frame).collapsed) == 2


def test_rows_without_pmid_are_never_merged() -> None:
    """Fail-closed: no publication anchor means no merge, however similar."""

    frame = _frame(
        [
            {"pmid": "", "source_database": "RNAInter", "source_record_id": "A"},
            {"pmid": "", "source_database": "RNAInter", "source_record_id": "B"},
            {"pmid": None, "source_database": "NPInter", "source_record_id": "C"},
        ]
    )
    result = collapse_canonical_experiments(frame)
    assert len(result.collapsed) == 3
    assert result.audit["rows_without_pmid"] == 3
    assert result.audit["canonical_experiments"] == 0
    assert result.audit["unanchored_rows_preserved_separately"] == 3
    assert set(result.collapsed.source_record_count) == {1}


def test_anchored_and_unanchored_rows_are_handled_separately() -> None:
    frame = _frame(
        [
            {"pmid": "111", "source_database": "NPInter", "source_record_id": "A"},
            {"pmid": "111", "source_database": "RNAInter", "source_record_id": "B"},
            {"pmid": "", "source_database": "RNAInter", "source_record_id": "C"},
        ]
    )
    result = collapse_canonical_experiments(frame)
    assert result.audit["rows_with_pmid"] == 2
    assert result.audit["rows_without_pmid"] == 1
    assert len(result.collapsed) == 2  # one collapsed experiment + one unanchored row


def test_missing_required_column_fails_loudly() -> None:
    frame = _frame([{}]).drop(columns=["assay_subtype"])
    with pytest.raises(ValueError):
        collapse_canonical_experiments(frame)


def test_empty_frame_is_safe() -> None:
    frame = _frame([]).iloc[0:0]
    result = collapse_canonical_experiments(frame)
    assert len(result.collapsed) == 0
    assert result.audit["input_rows"] == 0


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def test_duplicate_report_surfaces_cross_database_groups() -> None:
    frame = _frame(
        [
            {"source_database": "NPInter", "pmid": "111"},
            {"source_database": "RNAInter", "pmid": "111"},
            {"source_database": "NPInter", "pmid": "999", "lncrna_id": "LNC:Z"},
        ]
    )
    report = duplicate_report(frame)
    assert len(report) >= 1
    top = report.iloc[0]
    assert top.database_count == 2
    assert "NPInter" in top.databases and "RNAInter" in top.databases


def test_duplicate_report_is_empty_for_a_single_database() -> None:
    frame = _frame(
        [
            {"source_database": "NPInter", "pmid": "111"},
            {"source_database": "NPInter", "pmid": "222", "lncrna_id": "LNC:Z"},
        ]
    )
    assert duplicate_report(frame).empty
