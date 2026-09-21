from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from .external_validation import build_cancer_aliases, map_cancer_value


# Only high-confidence, widely used cancer cell-line aliases are assigned to a
# TCGA context.  Kidney-derived HEK293/293T and lineage-ambiguous/mixed entries
# are deliberately absent and therefore excluded from strict LOCO graphs.
CELL_LINE_CANCER_HINTS = {
    "HELA": "CESC",
    "HELAS3": "CESC",
    "HEPG2": "LIHC",
    "HUH7": "LIHC",
    "MCF7": "BRCA",
    "MDAMB231": "BRCA",
    "MDAMB468": "BRCA",
    "T47D": "BRCA",
    "BT474": "BRCA",
    "HCT116": "COAD",
    "HT29": "COAD",
    "SW480": "COAD",
    "DLD1": "COAD",
    "RKO": "COAD",
    "A549": "LUAD",
    "H1299": "LUAD",
    "H1975": "LUAD",
    "PC3": "PRAD",
    "DU145": "PRAD",
    "LNCAP": "PRAD",
    "U87": "GBM",
    "U87MG": "GBM",
    "U251": "GBM",
    "T24": "BLCA",
    "PANC1": "PAAD",
    "AGS": "STAD",
    "786O": "KIRC",
    "CAKI1": "KIRC",
    "C666": "HNSC",
}

_EMPTY_CONTEXT = {"", "-", "NA", "N/A", "NAN", "NONE", "NULL", "UNKNOWN", "<NA>"}


def _clean(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.upper() in _EMPTY_CONTEXT else text


def _cell_line_hints(value: object) -> set[str]:
    text = _clean(value)
    if not text:
        return set()
    hints: set[str] = set()
    for token in re.split(r"[;,|+/]", text):
        normalized = re.sub(r"[^A-Z0-9]", "", token.upper())
        if normalized in CELL_LINE_CANCER_HINTS:
            hints.add(CELL_LINE_CANCER_HINTS[normalized])
    return hints


def _map_context_tuple(
    disease_raw: object,
    tissue: object,
    cell_line: object,
    aliases: dict[str, str],
) -> tuple[object, str]:
    values = [_clean(disease_raw), _clean(tissue), _clean(cell_line)]
    if not any(values):
        return pd.NA, "global_context_free"
    text = " | ".join(value for value in values if value)
    phrase_id, _, phrase_status = map_cancer_value(text, aliases)
    candidates = _cell_line_hints(cell_line)
    if not pd.isna(phrase_id):
        candidates.add(str(phrase_id))
    if len(candidates) == 1:
        return next(iter(candidates)), "mapped_context"
    if len(candidates) > 1:
        return pd.NA, "excluded_ambiguous_context"
    return pd.NA, f"excluded_unmapped_context:{phrase_status}"


def apply_strict_cancer_context(
    frame: pd.DataFrame,
    dim_cancer: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Map contextual interaction rows or exclude them; never globalize them.

    Context-free evidence remains global.  Any row carrying disease, tissue or
    cell-line context enters the strict graph only when that context can be
    mapped unambiguously to one TCGA cancer.
    """
    out = frame.copy()
    for column in ("disease_raw", "tissue", "cell_line"):
        if column not in out:
            out[column] = ""
    key_columns = ["disease_raw", "tissue", "cell_line"]
    keys = out[key_columns].fillna("").astype(str)
    unique = keys.drop_duplicates()
    aliases = build_cancer_aliases(dim_cancer)
    mapped: dict[tuple[str, str, str], tuple[object, str]] = {}
    for row in unique.itertuples(index=False, name=None):
        mapped[tuple(row)] = _map_context_tuple(*row, aliases)
    results = [mapped[tuple(row)] for row in keys.itertuples(index=False, name=None)]
    out["cancer_id"] = pd.Series([item[0] for item in results], index=out.index, dtype="string")
    out["context_mapping_status"] = pd.Series([item[1] for item in results], index=out.index, dtype="string")
    contextual = ~out.context_mapping_status.eq("global_context_free")
    mapped_context = out.context_mapping_status.eq("mapped_context")
    excluded = contextual & ~mapped_context
    audit = {
        "input_rows": int(len(out)),
        "context_free_rows": int((~contextual).sum()),
        "context_specific_input_rows": int(contextual.sum()),
        "mapped_context_rows": int(mapped_context.sum()),
        "excluded_unmapped_or_ambiguous_context_rows": int(excluded.sum()),
        "context_specific_lost_to_global": 0,
        "mapped_cancer_counts": out.loc[mapped_context, "cancer_id"].astype(str).value_counts().to_dict(),
    }
    return out.loc[~excluded].copy(), audit
