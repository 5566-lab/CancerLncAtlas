"""Canonical experiment identity for lncRNA-protein / RBP evidence.

Problem
-------
The same underlying wet-lab experiment is routinely indexed by several curated
databases.  RNAInter and NPInter both absorb primary CLIP/RIP studies, so one
eCLIP experiment on one lncRNA-RBP pair can appear three or four times with a
different ``source_database``.  Treating those as independent observations
inflates apparent evidence multiplicity, and in the Evidence layer it lets one
experiment contribute several "independent" events.

What this module does
---------------------
Builds a ``canonical_experiment_key`` from the *experiment's own* identity --

    lncrna_id | partner_id | pmid | assay_subtype | cell_line | tissue

-- and collapses rows that share it, while recording how many databases and
records contributed:

    source_database_list, source_database_count, source_record_count

Fail-closed rules
-----------------
* A row **without a PMID is never collapsed**.  Without a publication anchor we
  cannot distinguish "one experiment indexed twice" from "two experiments", so
  the row keeps its own identity and is counted separately in the audit as a
  *potential* duplicate only.  Guessing here would silently destroy evidence.
* The key is deterministic: it is a pure function of the declared fields.
* ``assay_subtype`` participates in the key, so an eCLIP and a RIP experiment on
  the same pair and PMID stay distinct.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

__all__ = [
    "CANONICAL_KEY_VERSION",
    "CANONICAL_KEY_FIELDS",
    "CollapseResult",
    "canonical_experiment_key",
    "collapse_canonical_experiments",
    "duplicate_report",
]

CANONICAL_KEY_VERSION = "RBP_CANONICAL_EXPERIMENT_KEY_V1"

#: Fields that define one experiment.  Order is part of the contract.
CANONICAL_KEY_FIELDS: tuple[str, ...] = (
    "lncrna_id",
    "partner_id",
    "pmid",
    "assay_subtype",
    "cell_line",
    "tissue",
)

_MISSING = {"", "-", "na", "n/a", "nan", "none", "null", "<na>", "unknown"}


def _norm(value: object) -> str:
    """Normalise one key component without inventing information."""

    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip().lower()
    return "" if text in _MISSING else text


def canonical_experiment_key(
    lncrna_id: object,
    partner_id: object,
    pmid: object,
    assay_subtype: object,
    cell_line: object = "",
    tissue: object = "",
) -> str:
    """Deterministic identity of a single experiment."""

    parts = (
        _norm(lncrna_id),
        _norm(partner_id),
        _norm(pmid),
        _norm(assay_subtype),
        _norm(cell_line),
        _norm(tissue),
    )
    payload = "\x1f".join(parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return "CEXPV1:" + digest[:32]


@dataclass(frozen=True)
class CollapseResult:
    """Output of a canonical collapse."""

    collapsed: pd.DataFrame
    audit: dict[str, Any]


def _join_unique(values: Iterable[object]) -> str:
    return "|".join(sorted({str(v) for v in values if _norm(v)}))


def collapse_canonical_experiments(
    frame: pd.DataFrame,
    *,
    key_function=canonical_experiment_key,
    pmid_column: str = "pmid",
    database_column: str = "source_database",
    record_column: str | None = "source_record_id",
) -> CollapseResult:
    """Collapse rows that describe the same experiment.

    Rows without a PMID are passed through untouched (never merged).
    """

    required = {"lncrna_id", "partner_id", "assay_subtype", pmid_column, database_column}
    if len(frame) == 0:
        # A genuinely empty input ("no interactions") is not a contract
        # violation; only a non-empty frame missing columns is.
        empty = frame.copy()
        for column in sorted(required | {"cell_line", "tissue", "source_record_count",
                                         "source_database_list", "source_database_count",
                                         "_canonical_experiment_key"}):
            if column not in empty.columns:
                empty[column] = pd.Series(dtype="object")
        return CollapseResult(
            collapsed=empty,
            audit={
                "input_rows": 0, "rows_with_pmid": 0, "rows_without_pmid": 0,
                "canonical_experiments": 0, "canonical_collapsed_rows": 0,
                "canonical_experiments_seen_in_multiple_databases": 0,
                "unanchored_rows_preserved_separately": 0, "output_rows": 0,
                "key_version": CANONICAL_KEY_VERSION,
            },
        )

    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"canonical collapse lacks columns: {missing}")

    work = frame.copy()
    for column in ("cell_line", "tissue"):
        if column not in work.columns:
            work[column] = ""

    pmids = work[pmid_column].map(_norm)
    has_pmid = pmids.ne("")

    work["_canonical_experiment_key"] = [
        key_function(lnc, partner, pmid, subtype, cell, tissue) if anchor else ""
        for lnc, partner, pmid, subtype, cell, tissue, anchor in zip(
            work.lncrna_id, work.partner_id, pmids, work.assay_subtype,
            work.cell_line, work.tissue, has_pmid,
        )
    ]

    anchored = work.loc[has_pmid].copy()
    unanchored = work.loc[~has_pmid].copy()

    grouped_records: dict[str, int] = {}
    grouped_databases: dict[str, str] = {}
    grouped_database_count: dict[str, int] = {}
    if not anchored.empty:
        for key, block in anchored.groupby("_canonical_experiment_key", observed=True):
            grouped_records[key] = int(len(block))
            databases = sorted({str(v) for v in block[database_column] if _norm(v)})
            grouped_databases[key] = "|".join(databases)
            grouped_database_count[key] = len(databases)

    if not anchored.empty:
        anchored = anchored.drop_duplicates("_canonical_experiment_key", keep="first").copy()
        anchored["source_record_count"] = anchored["_canonical_experiment_key"].map(grouped_records)
        anchored["source_database_list"] = anchored["_canonical_experiment_key"].map(grouped_databases)
        anchored["source_database_count"] = anchored["_canonical_experiment_key"].map(grouped_database_count)

    # Rows without a PMID keep their own identity and are never merged.
    if not unanchored.empty:
        unanchored["source_record_count"] = 1
        unanchored["source_database_list"] = unanchored[database_column].astype(str)
        unanchored["source_database_count"] = 1

    combined = pd.concat([anchored, unanchored], ignore_index=True, sort=False)

    cross_database = 0
    if not anchored.empty:
        cross_database = int((anchored.source_database_count > 1).sum())

    audit = {
        "input_rows": int(len(frame)),
        "rows_with_pmid": int(has_pmid.sum()),
        "rows_without_pmid": int((~has_pmid).sum()),
        "canonical_experiments": int(len(anchored)),
        "canonical_collapsed_rows": int(len(anchored) and (int(has_pmid.sum()) - len(anchored))),
        "canonical_experiments_seen_in_multiple_databases": cross_database,
        "unanchored_rows_preserved_separately": int(len(unanchored)),
        "output_rows": int(len(combined)),
        "key_version": CANONICAL_KEY_VERSION,
    }
    return CollapseResult(collapsed=combined, audit=audit)


def duplicate_report(
    frame: pd.DataFrame,
    *,
    key_function=canonical_experiment_key,
    pmid_column: str = "pmid",
    database_column: str = "source_database",
    top: int = 25,
) -> pd.DataFrame:
    """Report potential cross-database duplication, including PMID-less rows.

    PMID-less rows are *reported* using a relaxed key (no PMID) but are never
    collapsed by :func:`collapse_canonical_experiments`.
    """

    work = frame.copy()
    for column in ("cell_line", "tissue"):
        if column not in work.columns:
            work[column] = ""

    def relaxed(row: Mapping[str, Any]) -> str:
        return canonical_experiment_key(
            row.get("lncrna_id"), row.get("partner_id"), "",
            row.get("assay_subtype"), row.get("cell_line"), row.get("tissue"),
        )

    work["_relaxed_key"] = [relaxed(row) for row in work.to_dict("records")]
    grouped = (
        work.groupby("_relaxed_key", observed=True)
        .agg(
            rows=(database_column, "size"),
            databases=(database_column, lambda v: "|".join(sorted({str(x) for x in v}))),
            database_count=(database_column, lambda v: len({str(x) for x in v})),
            pmids=(pmid_column, lambda v: "|".join(sorted({str(x) for x in v if _norm(x)}))),
            assay_subtype=("assay_subtype", "first"),
        )
        .reset_index()
    )
    potential = grouped.loc[grouped.database_count.gt(1)].sort_values(
        ["database_count", "rows"], ascending=False
    )
    return potential.head(top).reset_index(drop=True)
