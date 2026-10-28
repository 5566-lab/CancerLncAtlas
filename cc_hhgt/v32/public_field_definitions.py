"""Scientific field definitions for the public publication tables.

The public catalog publishes Arrow schemas: for every table it lists each column
name, Arrow type and nullability.  That is a schema, not a definition.  A reader
cannot tell from ``weight`` whether it is an edge weight or a sample weight, from
``raw_effect`` whether it is raw or standardised, or from a zero whether the value
was measured as zero or simply unavailable.

This module is the definition registry and the completeness gate:

* ``COLUMN_DEFINITIONS`` gives one checkable sentence per column - what the value
  is, its kind, its unit or scale, and which stage produces it.
* ``COMPONENT_NOTES`` adds a component-specific clarification where the same
  column name means something different in another module.
* ``coverage`` reports ``defined / total`` over an actual catalog and lists every
  undefined column.
* ``require_complete`` raises instead of claiming completeness, and
  ``augment_catalog`` attaches the definitions and upgrades the declared scope
  only when coverage is exactly 100%.

Honesty rules baked into the data model:

* every definition is either ``authored`` (a human wrote the sentence and the
  grading below records whether the actual published schema was inspected) or
  ``pending``.  A pending entry never counts toward coverage.
* a definition that was authored from code without inspecting the published
  Arrow schema is marked ``schema_verified: false``, so a reader can see which
  sentences are still unreviewed against a real publication.
* no definition asserts a scientific finding.  It states what a column contains.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Mapping

FORMAT = "CANCERLNCATLAS_V32_PUBLIC_FIELD_DEFINITIONS_V1"
COMPLETE_SCOPE = "PUBLIC_SCHEMAS_AND_COMPLETE_FIELD_DEFINITIONS"
INCOMPLETE_SCOPE = "PUBLIC_SCHEMAS_AND_TABLE_SEMANTICS_NOT_COMPLETE_FIELD_DEFINITIONS"
KINDS = (
    "identifier",      # opaque or namespaced key, never a measurement
    "category",        # closed or documented vocabulary
    "count",           # non-negative integer quantity
    "measure",         # real-valued quantity with a unit or scale
    "probability",     # bounded to [0, 1]
    "score",           # unbounded real score / logit
    "flag",            # boolean or 0/1
    "text",            # free text
    "timestamp",       # UTC instant
    "digest",          # content hash
)
REQUIRED_KEYS = ("kind", "unit", "source", "definition", "status", "schema_verified")


@dataclass(frozen=True)
class FieldDefinition:
    """One public column definition."""

    kind: str
    unit: str
    source: str
    definition: str
    status: str = "authored"
    schema_verified: bool = False

    def as_record(self) -> dict:
        if self.kind not in KINDS:
            raise ValueError(f"PUBLIC_FIELD_KIND_INVALID: {self.kind}")
        if self.status not in {"authored", "pending"}:
            raise ValueError(f"PUBLIC_FIELD_STATUS_INVALID: {self.status}")
        if self.status == "authored" and len(self.definition.strip()) < 20:
            raise ValueError("PUBLIC_FIELD_DEFINITION_TOO_SHORT")
        if self.kind in {"measure", "count", "probability", "score"} and not self.unit.strip():
            raise ValueError(f"PUBLIC_FIELD_UNIT_REQUIRED: {self.kind}")
        return {
            "kind": self.kind,
            "unit": self.unit,
            "source": self.source,
            "definition": self.definition,
            "status": self.status,
            "schema_verified": bool(self.schema_verified),
        }


def _d(kind: str, unit: str, source: str, definition: str, *, verified: bool = False):
    return FieldDefinition(
        kind=kind,
        unit=unit,
        source=source,
        definition=definition,
        status="authored",
        schema_verified=verified,
    )


IDENTIFIER_UNIT = "none (namespaced identifier)"
DIMENSIONLESS = "dimensionless"
COUNT_UNIT = "rows"
PROBABILITY_UNIT = "probability [0,1]"
LOGIT_UNIT = "log-odds (unbounded)"

# Shared column semantics.  A column name that means something else in one
# component must be overridden in COMPONENT_NOTES instead of redefined here.
COLUMN_DEFINITIONS: dict[str, FieldDefinition] = {
    "cancer_id": _d("identifier", IDENTIFIER_UNIT, "cancer scope declaration",
                    "TCGA cancer-type code (for example BRCA) that scopes the row; it is the exact-pathway analysis unit, not a patient."),
    "lncrna_id": _d("identifier", IDENTIFIER_UNIT, "canonical lncRNA node table",
                    "Canonical identifier of one lncRNA node from the frozen graph node universe; the same string denotes the same node in every table."),
    "pathway_id": _d("identifier", IDENTIFIER_UNIT, "canonical exact-pathway node table",
                     "Canonical identifier of one exact pathway from the registered pathway set; pathway families are a different unit and are never substituted."),
    "pathway_family_id": _d("identifier", IDENTIFIER_UNIT, "canonical pathway-family node table",
                            "Canonical identifier of a pathway family; present only in tables whose estimand is the family unit, not the exact pathway."),
    "geneset_id": _d("identifier", IDENTIFIER_UNIT, "GeneSet registry",
                     "Registered GeneSet identifier for a cancer-specific lncRNA set; not a pathway and not interchangeable with one."),
    "subject_id": _d("identifier", IDENTIFIER_UNIT, "entity registry",
                     "Identifier of the row subject whose type is declared in subject_type; the same string may be a lncRNA, a pathway or a GeneSet."),
    "subject_type": _d("category", DIMENSIONLESS, "entity registry",
                       "Declared type of subject_id, one of lncRNA, exact_pathway, pathway_family or geneset; it selects how subject_id must be read."),
    "node_type": _d("category", DIMENSIONLESS, "frozen graph node table",
                    "Typed node class in the heterogeneous graph (lncRNA, gene, protein, pathway, pathway_family, cancer, state); node identifiers are only unique inside their type."),
    "canonical_id": _d("identifier", IDENTIFIER_UNIT, "frozen graph node table",
                       "Identifier of a node inside its own node_type; it is only comparable with canonical_id values of the same type."),
    "weight": _d("measure", DIMENSIONLESS, "edge construction stage",
                 "Non-negative message weight of the edge after fold-local normalisation; it scales the message, it is not a sample weight or a probability that the edge is true.", verified=False),
    "raw_effect": _d("measure", DIMENSIONLESS, "edge construction stage",
                     "Effect size measured on the source statistic before fold-local standardisation; unlike weight it keeps its original sign and scale."),
    "relation_type": _d("category", DIMENSIONLESS, "typed relation schema",
                        "Declared typed relation of an edge (for example coexpressed_with or member_of); the relation carries the scientific meaning, not the endpoints alone."),
    "source_type": _d("category", DIMENSIONLESS, "typed relation schema",
                      "Node type of the edge source; a node index is only meaningful together with its type."),
    "target_type": _d("category", DIMENSIONLESS, "typed relation schema",
                      "Node type of the edge target; a node index is only meaningful together with its type."),
    "edge_role": _d("category", DIMENSIONLESS, "typed relation schema",
                    "Registered scientific role of an edge class, used to decide which evidence may enter the primary graph at all."),
    "edge_weight": _d("measure", DIMENSIONLESS, "graph tensor materialisation",
                      "Signed weight actually used by the message-passing tensor for this edge, after the declared polarity is applied."),
    "score": _d("score", LOGIT_UNIT, "model scoring stage",
                "Model membership logit for the row's candidate pair before any calibration; higher means the model finds the pair more consistent with the pathway."),
    "membership_logit": _d("score", LOGIT_UNIT, "model scoring stage",
                           "Logit of the lncRNA-pathway membership head; the bounded residual is already included."),
    "direction_logit": _d("score", LOGIT_UNIT, "model scoring stage",
                          "Logit of the association-direction head; it orders the pair along the reported direction and does not measure magnitude."),
    "probability": _d("probability", PROBABILITY_UNIT, "calibration stage",
                      "Calibrated probability derived from the score by the declared link function; it is a model output, not an observed frequency."),
    "rank": _d("count", "rank position (1 = highest score)", "ranking stage",
               "Within-cancer rank of the row by score, ties resolved by the declared stable rule; ranks are only comparable inside one cancer and one release."),
    "fold": _d("count", "fold index", "outer cross-validation split",
               "Zero-based outer fold that produced the row; a row's fold must be read before comparing scores across rows."),
    "patient_fold_id": _d("identifier", IDENTIFIER_UNIT, "outer cross-validation split",
                          "Opaque fold-assignment identifier for a patient; it isolates the split without publishing the patient identifier."),
    "proxy_label": _d("flag", DIMENSIONLESS, "training label construction",
                      "Proxy binary label used to train the head; it is derived from the declared evidence arm and is not a measured binding outcome."),
    "label_class": _d("category", DIMENSIONLESS, "training label construction",
                      "Declared label class of the row (for example weak_positive or unlabeled); it records how the training label was formed."),
    "direction_label": _d("category", DIMENSIONLESS, "training label construction",
                          "Declared association direction used for supervision, one of positive, negative or unavailable."),
    "coverage_count": _d("count", COUNT_UNIT, "coverage accounting",
                         "Number of distinct patients contributing to the row after the declared eligibility rules; zero means no patient coverage and never an imputed value."),
    "patient_count": _d("count", COUNT_UNIT, "coverage accounting",
                        "Number of patients declared by the upstream stage for this scope before eligibility filtering."),
    "available": _d("flag", DIMENSIONLESS, "availability accounting",
                    "True when the upstream measurement exists for this row; false means the value is absent, and a zero value must never be read as unavailable."),
    "unavailable_reason": _d("category", DIMENSIONLESS, "availability accounting",
                             "Declared reason the measurement is unavailable (for example an assay gap or a failed completeness gate); set only when available is false."),
    "failure_reason": _d("text", DIMENSIONLESS, "stage diagnostics",
                         "Human-readable reason a stage produced no usable value; presence of text here never stands in for a numerical result."),
    "sha256": _d("digest", DIMENSIONLESS, "artifact publication",
                 "SHA256 of the exact file bytes named by the row; changing the file without changing this digest is detected by every consumer."),
    "format": _d("category", DIMENSIONLESS, "artifact publication",
                 "Declared payload format string; consumers refuse a payload whose format is not the one they expect."),
    "analysis_version": _d("category", DIMENSIONLESS, "release declaration",
                           "Declared analysis version of the producing release; tables from different versions are never mixed silently."),
    "artifact_kind": _d("category", DIMENSIONLESS, "artifact publication",
                        "Declared kind of artifact the row describes, which selects the validator applied to it."),
}

# Component-specific clarifications.  Keyed by component then column.
COMPONENT_NOTES: dict[str, dict[str, str]] = {
    "selected": {
        "score": "Selected current score for the lncRNA-pathway pair, produced by the registered winner-locked pipeline; it is not the fused or historical score.",
    },
    "cs_graph": {
        "weight": "Context-specific graph edge weight after the typed relation's own normalisation; it is never comparable across relation types.",
    },
    "distal": {
        "weight": "Distal-link weight derived from the declared peak-gene rule; it encodes a statistical link, not a physical contact.",
    },
    "drug": {
        "probability": "Predicted drug association probability from the declared link function; it is not a patient response probability.",
    },
    "external": {
        "score": "External-evaluation score for the same pair under the external input slice; it is not an independent cohort measurement.",
    },
}


def _definition_for(column: str, component: str | None) -> FieldDefinition | None:
    definition = COLUMN_DEFINITIONS.get(str(column))
    if definition is None:
        return None
    if component is None:
        return definition
    note = COMPONENT_NOTES.get(str(component), {}).get(str(column))
    if note is None:
        return definition
    return FieldDefinition(
        kind=definition.kind,
        unit=definition.unit,
        source=definition.source,
        definition=note,
        status=definition.status,
        schema_verified=definition.schema_verified,
    )


def coverage(catalog: Mapping[str, Any]) -> dict:
    """Report defined/total columns over an actual catalog payload."""

    tables = catalog.get("tables") if isinstance(catalog, Mapping) else None
    if not isinstance(tables, list) or not tables:
        raise RuntimeError("PUBLIC_FIELD_CATALOG_EMPTY")
    total = 0
    defined = 0
    undefined: list[dict] = []
    unverified = 0
    per_component: dict[str, dict[str, int]] = {}
    for table in tables:
        component = str(table.get("component"))
        bucket = per_component.setdefault(component, {"defined": 0, "total": 0})
        for field_row in table.get("fields") or []:
            column = str(field_row.get("name"))
            total += 1
            bucket["total"] += 1
            definition = _definition_for(column, component)
            if definition is None or definition.status != "authored":
                undefined.append(
                    {
                        "table_id": str(table.get("table_id")),
                        "component": component,
                        "column": column,
                        "arrow_type": str(field_row.get("arrow_type")),
                    }
                )
                continue
            defined += 1
            bucket["defined"] += 1
            if not definition.schema_verified:
                unverified += 1
    return {
        "format": FORMAT,
        "tables": len(tables),
        "columns": int(total),
        "defined": int(defined),
        "undefined": int(total - defined),
        "schema_unverified": int(unverified),
        "complete": bool(total and defined == total),
        "per_component": per_component,
        "undefined_columns": undefined,
        "definition_count": len(COLUMN_DEFINITIONS),
        "component_note_count": sum(len(value) for value in COMPONENT_NOTES.values()),
    }


def require_complete(catalog: Mapping[str, Any]) -> dict:
    """Raise unless every published column carries an authored definition."""

    report = coverage(catalog)
    if not report["complete"]:
        sample = ", ".join(
            f"{row['component']}:{row['column']}"
            for row in report["undefined_columns"][:5]
        )
        raise RuntimeError(
            "PUBLIC_FIELD_DEFINITIONS_INCOMPLETE: "
            f"{report['defined']}/{report['columns']} defined; missing {sample}"
        )
    return report


def augment_catalog(catalog: Mapping[str, Any]) -> dict:
    """Attach definitions to every field and set the honest scope string."""

    report = coverage(catalog)
    output = dict(catalog)
    tables = []
    for table in catalog.get("tables") or []:
        component = str(table.get("component"))
        fields = []
        for field_row in table.get("fields") or []:
            row = dict(field_row)
            definition = _definition_for(str(field_row.get("name")), component)
            row["field_definition"] = (
                None if definition is None else definition.as_record()
            )
            fields.append(row)
        updated = dict(table)
        updated["fields"] = fields
        tables.append(updated)
    output["tables"] = tables
    output["field_definition_coverage"] = {
        key: report[key]
        for key in (
            "format",
            "tables",
            "columns",
            "defined",
            "undefined",
            "schema_unverified",
            "complete",
            "per_component",
            "definition_count",
            "component_note_count",
        )
    }
    output["field_definition_scope"] = (
        COMPLETE_SCOPE if report["complete"] else INCOMPLETE_SCOPE
    )
    output["scope"] = output["field_definition_scope"]
    output["definitions_schema_verified"] = report["schema_unverified"] == 0
    return output


def registry_digest() -> str:
    payload = {
        "columns": {
            name: definition.as_record() for name, definition in COLUMN_DEFINITIONS.items()
        },
        "component_notes": COMPONENT_NOTES,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


__all__ = [
    "COLUMN_DEFINITIONS",
    "COMPLETE_SCOPE",
    "COMPONENT_NOTES",
    "FieldDefinition",
    "FORMAT",
    "INCOMPLETE_SCOPE",
    "KINDS",
    "REQUIRED_KEYS",
    "augment_catalog",
    "coverage",
    "registry_digest",
    "require_complete",
]
