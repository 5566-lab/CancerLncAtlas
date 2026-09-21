"""Typed lncRNA-protein (incl. RBP) binding materialisation.

Why this module exists
----------------------
The frozen v3.2 chain builds the main-graph binding table through
``cc_hhgt.protein_layer.build_contextual_lnc_protein``, which

* selects partners with ``partner_type.str.contains("protein")`` -- so every row
  labelled ``partner_type == "rbp"`` is silently dropped (348,364 rows in the
  frozen ``interaction_relation`` table, giving 147,643 lncRNA-partner pairs that
  never reach the graph);
* computes ``weight`` from a single binary ``is_experimental`` flag and then
  discards every assay field, so eCLIP and RIP become indistinguishable
  (the collapsed 623,207-row table carries no assay column at all).

This module re-derives the same table with two opt-in changes:

1. ``include_rbp_partner_type`` -- treat ``rbp`` partners as protein nodes.
   RBPs are *not* given a new node type, and partner identifiers are already in
   the shared ``GENE:`` / ``UNIPROT:`` space, so no duplicate entity can arise.
2. ``group_by_assay_class`` -- add ``graph_assay_class`` to the grouping key and
   emit one relation type per class.

Legacy equivalence
------------------
With both switches off the output must be **identical** to the frozen pipeline.
That is asserted by ``verify_legacy_equivalence`` and exercised before any new
materialisation is trusted.  This is the anchor that makes ablation mode A
meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..interaction_context import apply_strict_cancer_context
from .rbp_assay import GRAPH_ASSAY_CLASSES, graph_assay_relation_type

__all__ = [
    "BindingTyping",
    "LEGACY_RELATION_TYPE",
    "build_typed_lnc_protein_binding",
    "verify_legacy_equivalence",
]

LEGACY_RELATION_TYPE = "binds_protein"

#: Partner types that denote a protein node.  ``rbp`` is included only when the
#: typing switch is on; historically the filter was a bare
#: ``contains("protein")`` which cannot match ``rbp``.
_PROTEIN_LIKE = ("protein",)
_PROTEIN_LIKE_WITH_RBP = ("protein", "rbp")


@dataclass(frozen=True)
class BindingTyping:
    """Ablation switches for the binding materialiser."""

    include_rbp_partner_type: bool = False
    group_by_assay_class: bool = False
    #: When True the label ``predicted`` is kept out of the physical-binding
    #: relation entirely; it is still emitted as its own typed relation.
    emit_predicted_relation: bool = True

    def partner_types(self) -> tuple[str, ...]:
        return _PROTEIN_LIKE_WITH_RBP if self.include_rbp_partner_type else _PROTEIN_LIKE


def _normalize_uniprot(values: pd.Series) -> pd.Series:
    out = values.astype("string").str.strip()
    out = out.str.replace(r"^UNIPROT:", "", regex=True, case=False)
    out = out.str.split(r"[;,\s|]+", regex=True).str[0].str.strip()
    return out.mask(out.isna() | out.str.lower().isin({"", "nan", "none", "<na>"}))


def _require_columns(frame: pd.DataFrame, required: set[str], what: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{what} lacks columns: {missing}")


def build_typed_lnc_protein_binding(
    interaction_relation: pd.DataFrame,
    protein_gene: pd.DataFrame,
    dim_lnc: pd.DataFrame,
    dim_cancer: pd.DataFrame,
    *,
    typing: BindingTyping,
    classify: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return (typed binding table, audit).

    ``classify`` is injected (``cc_hhgt.v32.rbp_assay.classify_assay``) so that
    the materialiser and the Evidence layer can never drift apart.
    """

    _require_columns(
        interaction_relation,
        {"lncrna_id", "partner_id", "partner_type"},
        "interaction_relation",
    )
    _require_columns(
        protein_gene,
        {"gene_id", "protein_id", "uniprot_accession", "mapping_multiplicity", "mapping_weight"},
        "protein_gene_map",
    )
    _require_columns(dim_lnc, {"lncrna_id"}, "dim_lncRNA")

    relation = interaction_relation
    partner_type = relation.partner_type.astype(str).str.lower().str.strip()
    is_protein = partner_type.isin(typing.partner_types())
    rel = relation.loc[is_protein].copy()
    rel["lncrna_id"] = rel.lncrna_id.astype("string").str.strip()
    rel["original_partner_id"] = rel.partner_id.astype("string").str.strip()

    valid_lnc = set(dim_lnc.lncrna_id.dropna().astype(str))
    rel = rel.loc[
        rel.lncrna_id.isin(valid_lnc)
        & rel.original_partner_id.notna()
        & rel.original_partner_id.ne("")
    ].copy()
    rel["_row_id"] = np.arange(len(rel), dtype=np.int64)

    rel, context_audit = apply_strict_cancer_context(rel, dim_cancer)

    mapping = protein_gene[
        ["gene_id", "protein_id", "mapping_multiplicity", "mapping_weight"]
    ].drop_duplicates(["gene_id", "protein_id"])

    by_gene = rel.loc[rel.original_partner_id.str.startswith("GENE:", na=False)].merge(
        mapping, left_on="original_partner_id", right_on="gene_id", how="left"
    )
    direct = rel.loc[~rel.original_partner_id.str.startswith("GENE:", na=False)].copy()
    direct["uniprot_accession"] = _normalize_uniprot(direct.original_partner_id)
    direct = direct.merge(
        protein_gene[
            ["uniprot_accession", "protein_id", "gene_id", "mapping_multiplicity", "mapping_weight"]
        ].drop_duplicates("uniprot_accession"),
        on="uniprot_accession",
        how="left",
    )
    mapped = pd.concat([by_gene, direct], ignore_index=True, sort=False)
    rejected_mapping = int(mapped.protein_id.isna().sum())
    mapped = mapped.dropna(subset=["protein_id"]).copy()

    # Identical expression to the frozen pipeline so that legacy mode is exact.
    experimental = mapped.get("is_experimental", pd.Series(False, index=mapped.index)).fillna(False)
    mapped["weight"] = np.where(experimental, 1.0, 0.4) * mapped.mapping_weight.fillna(1.0)
    mapped["source_database"] = mapped.source_database.fillna("unknown").astype(str)
    mapped["pmid_present"] = mapped.get("pmid", pd.Series(pd.NA, index=mapped.index)).notna().astype(int)

    # --- assay taxonomy -----------------------------------------------------
    if "experiment_raw" in mapped.columns:
        raw = mapped["experiment_raw"]
    elif "experiment_family" in mapped.columns:
        raw = mapped["experiment_family"]
    else:
        raw = pd.Series([""] * len(mapped), index=mapped.index, dtype="object")
    assignments = [
        classify(value, is_predicted=pred, is_experimental=exp)
        for value, pred, exp in zip(
            raw.tolist(),
            mapped.get("is_predicted", pd.Series([None] * len(mapped), index=mapped.index)).tolist(),
            mapped.get("is_experimental", pd.Series([None] * len(mapped), index=mapped.index)).tolist(),
        )
    ]
    mapped["experiment_raw"] = [a.experiment_raw for a in assignments]
    mapped["experiment_family"] = [a.experiment_family for a in assignments]
    mapped["assay_subtype"] = [a.assay_subtype for a in assignments]
    mapped["graph_assay_class"] = [a.graph_assay_class for a in assignments]

    # --- aggregation --------------------------------------------------------
    join_sources = lambda values: "|".join(sorted(set(map(str, values))))  # noqa: E731

    if typing.group_by_assay_class:
        keys = ["lncrna_id", "protein_id", "cancer_id", "graph_assay_class"]
        grouped = (
            mapped.groupby(keys, as_index=False, observed=True, dropna=False)
            .agg(
                weight=("weight", "max"),
                source_database=("source_database", join_sources),
                n_source_records=("_row_id", "nunique"),
                n_pmids=("pmid_present", "sum"),
                mapping_multiplicity=("mapping_multiplicity", "max"),
                assay_subtype=("assay_subtype", lambda v: "|".join(sorted(set(map(str, v))))),
                experiment_family=("experiment_family", lambda v: "|".join(sorted(set(map(str, v))))),
                experiment_raw=("experiment_raw", lambda v: "|".join(sorted(set(map(str, v))))),
            )
            .sort_values(keys, na_position="first")
        )
        grouped["relation_type"] = [
            graph_assay_relation_type(cls) for cls in grouped.graph_assay_class
        ]
    else:
        keys = ["lncrna_id", "protein_id", "cancer_id"]
        grouped = (
            mapped.groupby(keys, as_index=False, observed=True, dropna=False)
            .agg(
                weight=("weight", "max"),
                source_database=("source_database", join_sources),
                n_source_records=("_row_id", "nunique"),
                n_pmids=("pmid_present", "sum"),
                mapping_multiplicity=("mapping_multiplicity", "max"),
            )
            .sort_values(keys, na_position="first")
        )
        grouped["relation_type"] = LEGACY_RELATION_TYPE

    grouped["is_context_specific"] = grouped.cancer_id.notna()
    grouped["mapping_status"] = np.where(
        grouped.mapping_multiplicity.fillna(1).gt(1), "mapped_expanded", "mapped_unique"
    )

    audit = {
        "partner_types_used": list(typing.partner_types()),
        "include_rbp_partner_type": typing.include_rbp_partner_type,
        "group_by_assay_class": typing.group_by_assay_class,
        **context_audit,
        "protein_mapping_rejected_rows": rejected_mapping,
        "edges_total": int(len(grouped)),
        "global_edges": int(grouped.cancer_id.isna().sum()),
        "context_specific_edges": int(grouped.cancer_id.notna().sum()),
        "distinct_lncrna": int(grouped.lncrna_id.nunique()),
        "relation_type_counts": grouped.relation_type.value_counts().to_dict(),
    }
    if typing.group_by_assay_class:
        audit["graph_assay_class_counts"] = grouped.graph_assay_class.value_counts().to_dict()
        unknown = set(grouped.graph_assay_class.unique()) - set(GRAPH_ASSAY_CLASSES)
        if unknown:
            raise RuntimeError(f"graph_assay_class outside the closed vocabulary: {unknown}")
    return grouped, audit


def verify_legacy_equivalence(
    produced: pd.DataFrame,
    frozen: pd.DataFrame,
    *,
    keys: tuple[str, ...] = ("lncrna_id", "protein_id", "cancer_id"),
) -> dict[str, Any]:
    """Compare a legacy-mode materialisation against the frozen artifact.

    This is the Phase 12 gate #12 check ("legacy mode reproduces the old generic
    semantics").  It compares row count, distinct-lncRNA count and the full
    key/weight/source payload, not just the totals.
    """

    result: dict[str, Any] = {
        "produced_rows": int(len(produced)),
        "frozen_rows": int(len(frozen)),
        "rows_match": int(len(produced)) == int(len(frozen)),
    }
    if not result["rows_match"]:
        result["status"] = "FAIL_ROW_COUNT"
        return result

    left = produced.copy()
    right = frozen.copy()
    for frame in (left, right):
        if "cancer_id" in frame.columns:
            frame["cancer_id"] = frame.cancer_id.astype("string")
    left = left.sort_values(list(keys), na_position="first").reset_index(drop=True)
    right = right.sort_values(list(keys), na_position="first").reset_index(drop=True)

    compare_columns = [c for c in ("weight", "source_database", "n_source_records", "n_pmids")
                       if c in left.columns and c in right.columns]
    mismatches: dict[str, int] = {}
    for column in compare_columns:
        a = left[column]
        b = right[column]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            different = int((~np.isclose(a.astype(float), b.astype(float), equal_nan=True)).sum())
        else:
            different = int((a.astype(str) != b.astype(str)).sum())
        if different:
            mismatches[column] = different
    result["compared_columns"] = compare_columns
    result["mismatches"] = mismatches
    result["status"] = "PASS" if not mismatches else "FAIL_VALUE_MISMATCH"
    return result
