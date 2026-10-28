"""Fixture checks for the public field-definition registry and completeness gate."""
from __future__ import annotations

import copy

import pytest

from cc_hhgt.v32 import public_field_definitions as definitions


def _catalog(columns, *, component="selected", table_id="selected.geneset_members"):
    return {
        "format": "CURRENT_PUBLIC_TABLE_CATALOG_V1",
        "web_binding_sha256": "a" * 64,
        "selected_analysis_sha256": "b" * 64,
        "scope": definitions.INCOMPLETE_SCOPE,
        "tables": [
            {
                "table_id": table_id,
                "component": component,
                "kind": "parquet",
                "rows": 3,
                "table_semantics": "synthetic table",
                "fields": [
                    {"name": name, "arrow_type": "string", "nullable": False}
                    for name in columns
                ],
                "parts": [{"part": 0, "sha256": "c" * 64}],
            }
        ],
    }


DEFINED = ("cancer_id", "lncrna_id", "pathway_id", "score")
MIXED = ("cancer_id", "lncrna_id", "totally_unknown_column")


def test_every_registry_entry_is_structurally_valid() -> None:
    assert definitions.COLUMN_DEFINITIONS
    for name, definition in definitions.COLUMN_DEFINITIONS.items():
        record = definition.as_record()
        assert record["kind"] in definitions.KINDS, name
        assert definition.status == "authored", name
        assert len(record["definition"]) >= 20, name
        assert record["source"].strip(), name
        if record["kind"] in {"measure", "count", "probability", "score"}:
            assert record["unit"].strip(), name


def test_registry_digest_is_stable_and_sensitive(monkeypatch) -> None:
    first = definitions.registry_digest()
    assert first == definitions.registry_digest()
    monkeypatch.setitem(
        definitions.COLUMN_DEFINITIONS,
        "cancer_id",
        definitions._d("identifier", "none", "test", "A different sentence for the same column."),
    )
    assert definitions.registry_digest() != first


@pytest.mark.parametrize(
    "definition, message",
    [
        (definitions.FieldDefinition("nonsense", "x", "s", "A long enough sentence here."),
         "PUBLIC_FIELD_KIND_INVALID"),
        (definitions.FieldDefinition("measure", "x", "s", "A long enough sentence here.",
                                     status="guessed"), "PUBLIC_FIELD_STATUS_INVALID"),
        (definitions.FieldDefinition("measure", "x", "s", "too short"), "PUBLIC_FIELD_DEFINITION_TOO_SHORT"),
        (definitions.FieldDefinition("measure", "  ", "s", "A long enough sentence here."),
         "PUBLIC_FIELD_UNIT_REQUIRED"),
        (definitions.FieldDefinition("count", "", "s", "A long enough sentence here."),
         "PUBLIC_FIELD_UNIT_REQUIRED"),
    ],
)
def test_invalid_definition_entries_are_rejected(definition, message) -> None:
    with pytest.raises(ValueError, match=message):
        definition.as_record()


def test_coverage_counts_defined_and_undefined() -> None:
    report = definitions.coverage(_catalog(MIXED))
    assert report["tables"] == 1
    assert report["columns"] == 3
    assert report["defined"] == 2
    assert report["undefined"] == 1
    assert report["complete"] is False
    assert report["definition_count"] == len(definitions.COLUMN_DEFINITIONS)


def test_coverage_lists_the_missing_column_with_its_context() -> None:
    report = definitions.coverage(_catalog(MIXED))
    assert report["undefined_columns"] == [
        {
            "table_id": "selected.geneset_members",
            "component": "selected",
            "column": "totally_unknown_column",
            "arrow_type": "string",
        }
    ]


def test_coverage_reports_per_component_counts() -> None:
    catalog = _catalog(MIXED)
    catalog["tables"].append(
        copy.deepcopy(catalog["tables"][0]) | {"table_id": "scores.pairs", "component": "scores"}
    )
    report = definitions.coverage(catalog)
    assert report["per_component"]["selected"] == {"defined": 2, "total": 3}
    assert report["per_component"]["scores"] == {"defined": 2, "total": 3}


def test_coverage_rejects_an_empty_catalog() -> None:
    with pytest.raises(RuntimeError, match="PUBLIC_FIELD_CATALOG_EMPTY"):
        definitions.coverage({"tables": []})
    with pytest.raises(RuntimeError, match="PUBLIC_FIELD_CATALOG_EMPTY"):
        definitions.coverage({})


def test_require_complete_passes_on_full_coverage() -> None:
    report = definitions.require_complete(_catalog(DEFINED))
    assert report["complete"] is True
    assert report["schema_unverified"] == report["defined"]


def test_require_complete_refuses_a_partial_catalog() -> None:
    with pytest.raises(RuntimeError, match="PUBLIC_FIELD_DEFINITIONS_INCOMPLETE") as error:
        definitions.require_complete(_catalog(MIXED))
    assert "2/3 defined" in str(error.value)
    assert "selected:totally_unknown_column" in str(error.value)


def test_pending_definitions_never_count(monkeypatch) -> None:
    monkeypatch.setitem(
        definitions.COLUMN_DEFINITIONS,
        "lncrna_id",
        definitions.FieldDefinition(
            "identifier", "none", "registry", "Draft text that is long enough to pass.",
            status="pending",
        ),
    )
    report = definitions.coverage(_catalog(DEFINED))
    assert report["defined"] == 3
    assert report["undefined"] == 1
    assert [row["column"] for row in report["undefined_columns"]] == ["lncrna_id"]


def test_augment_attaches_definitions_and_keeps_schema_keys() -> None:
    catalog = _catalog(MIXED)
    result = definitions.augment_catalog(catalog)
    fields = result["tables"][0]["fields"]
    assert [row["name"] for row in fields] == list(MIXED)
    for row in fields:
        assert set(row) >= {"name", "arrow_type", "nullable", "field_definition"}
    defined = [row for row in fields if row["field_definition"] is not None]
    assert len(defined) == 2
    assert defined[0]["field_definition"]["kind"] == "identifier"
    assert defined[0]["field_definition"]["status"] == "authored"


def test_augment_leaves_undefined_columns_explicitly_null() -> None:
    result = definitions.augment_catalog(_catalog(MIXED))
    missing = [
        row for row in result["tables"][0]["fields"]
        if row["name"] == "totally_unknown_column"
    ]
    assert missing[0]["field_definition"] is None


def test_augment_scope_stays_incomplete_until_every_column_is_defined() -> None:
    result = definitions.augment_catalog(_catalog(MIXED))
    assert result["scope"] == definitions.INCOMPLETE_SCOPE
    assert result["field_definition_scope"] == definitions.INCOMPLETE_SCOPE
    coverage = result["field_definition_coverage"]
    assert coverage["complete"] is False
    assert coverage["defined"] == 2 and coverage["columns"] == 3
    assert result["definitions_schema_verified"] is False


def test_augment_scope_becomes_complete_only_on_full_coverage() -> None:
    result = definitions.augment_catalog(_catalog(DEFINED))
    assert result["scope"] == definitions.COMPLETE_SCOPE
    assert result["field_definition_coverage"]["complete"] is True
    assert result["field_definition_coverage"]["undefined"] == 0


def test_augment_does_not_mutate_its_input() -> None:
    catalog = _catalog(MIXED)
    original = copy.deepcopy(catalog)
    definitions.augment_catalog(catalog)
    assert catalog == original


def test_component_note_overrides_the_shared_semantics() -> None:
    base = definitions._definition_for("score", None)
    selected = definitions._definition_for("score", "selected")
    other = definitions._definition_for("score", "unregistered_component")
    assert base is not None and selected is not None
    assert selected.definition != base.definition
    assert "winner-locked" in selected.definition
    assert other.definition == base.definition
    assert selected.schema_verified == base.schema_verified


def test_definitions_do_not_claim_validation() -> None:
    for name, definition in definitions.COLUMN_DEFINITIONS.items():
        lowered = definition.definition.lower()
        assert " proves " not in lowered, name
        assert " validated " not in lowered, name
        assert "causally" not in lowered, name
