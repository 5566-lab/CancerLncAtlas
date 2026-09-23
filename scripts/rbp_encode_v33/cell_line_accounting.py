"""Which cell lines, from which databases, never reach the primary graph?

interaction_context.py treats a named cell line as a disease-context claim: a row
carrying disease, tissue or cell line is either mapped to one of the 33 TCGA
cancers or removed.  The ruling is that this is wrong for an interaction - an
RBP-lncRNA binding is not cancer-type specific, it only needs to exist in some
cell line.

This produces the full accounting of what the rule removes, by database and by
cell line, and quantifies what each part of the ruling recovers.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

WORK = Path("${PRIVATE_WORK_ROOT}/CancerLncAtlas/rbp_encode_v33_20260921_r1")
sys.path.insert(0, str(WORK / "code"))

from cc_hhgt.interaction_context import (  # noqa: E402
    CELL_LINE_CANCER_HINTS,
    _clean,
    apply_strict_cancer_context,
)

STANDARDISED = Path(
    "${PRIVATE_ARCHIVE_ROOT}/CancerLncAtlas/results/v3_state_formal_runs/"
    "V3STATE-formal500-20260810T101223Z-a8803782dcf0_server_handoff/run/assets/standardized"
)
OUT = WORK / "outputs" / "phase3_typed_binding"
OUT.mkdir(parents=True, exist_ok=True)

interaction = pd.read_parquet(STANDARDISED / "interaction_relation.parquet")
dim_cancer = pd.read_parquet(STANDARDISED / "dim_cancer.parquet")
scoped, receipt = apply_strict_cancer_context(interaction, dim_cancer)

print("=== inputs ===")
print(f"  interaction rows : {len(interaction):,}")

frame = interaction.copy()
status = pd.Series("excluded_dropped", index=frame.index, dtype=object)
status.loc[scoped.index] = scoped.context_mapping_status.astype(str).to_numpy()
cancer = pd.Series(pd.NA, index=frame.index, dtype=object)
cancer.loc[scoped.index] = scoped.cancer_id.astype(str).to_numpy()
frame["graph_status"] = status.to_numpy()
frame["graph_cancer"] = cancer.to_numpy()
print()
print("=== graph status ===")
print(f"  {frame.graph_status.value_counts().to_dict()}")

frame["ctx_disease"] = frame.disease_raw.map(_clean) if "disease_raw" in frame else ""
frame["ctx_tissue"] = frame.tissue.map(_clean) if "tissue" in frame else ""
frame["ctx_cell"] = frame.cell_line.map(_clean) if "cell_line" in frame else ""
frame["has_disease"] = frame.ctx_disease.ne("")
frame["has_tissue"] = frame.ctx_tissue.ne("")
frame["has_cell"] = frame.ctx_cell.ne("")

out_of_graph = frame.loc[frame.graph_status.eq("excluded_dropped")].copy()
print()
print(f"=== never reach the primary graph: {len(out_of_graph):,} ===")

buckets = {
    "cell_line only": int((out_of_graph.has_cell & ~out_of_graph.has_tissue
                           & ~out_of_graph.has_disease).sum()),
    "tissue only": int((~out_of_graph.has_cell & out_of_graph.has_tissue
                        & ~out_of_graph.has_disease).sum()),
    "cell_line AND tissue": int((out_of_graph.has_cell & out_of_graph.has_tissue
                                 & ~out_of_graph.has_disease).sum()),
    "any row with disease_raw": int(out_of_graph.has_disease.sum()),
    "any row naming a cell line": int(out_of_graph.has_cell.sum()),
    "any row naming a tissue": int(out_of_graph.has_tissue.sum()),
}
print()
print("=== which context field removed them (overlapping counts) ===")
for key, value in buckets.items():
    print(f"  {key:28} {value:>9,}")

by_db_cell = (out_of_graph.loc[out_of_graph.has_cell]
              .groupby(["source_database", "ctx_cell"], dropna=False)
              .size().reset_index(name="excluded_rows")
              .sort_values("excluded_rows", ascending=False))
print()
print("=== excluded rows by source_database x cell_line ===")
print(f"  distinct (database, cell line) pairs: {len(by_db_cell):,}")
for row in by_db_cell.head(30).itertuples(index=False):
    key = str(row.ctx_cell).upper().replace("-", "").replace(" ", "")
    hint = CELL_LINE_CANCER_HINTS.get(key, "no hint")
    print(f"    {str(row.source_database)[:20]:22} {str(row.ctx_cell)[:30]:32} "
          f"{row.excluded_rows:>8,}   {hint}")

print()
print("=== what the tissue field actually contains (excluded rows) ===")
tissue_counts = out_of_graph.loc[out_of_graph.has_tissue, "ctx_tissue"].value_counts()
print(f"  distinct tissue strings: {len(tissue_counts):,}")
for name, count in tissue_counts.head(20).items():
    print(f"    {str(name)[:44]:46} {count:>8,}")

print()
print("=== excluded rows by source_database ===")
by_db = out_of_graph.groupby("source_database").size().sort_values(ascending=False)
for name, count in by_db.items():
    total = int((frame.source_database.astype(str) == str(name)).sum())
    print(f"    {str(name)[:24]:26} excluded {count:>9,} / {total:>10,}  ({count / max(total, 1):.1%})")

in_graph = frame.loc[frame.graph_status.ne("excluded_dropped")]
present = {c.upper() for c in in_graph.loc[in_graph.has_cell, "ctx_cell"].unique()}
all_cells = set(frame.loc[frame.has_cell, "ctx_cell"].unique())
never = sorted(c for c in all_cells if c.upper() not in present)
never_counts = (frame.loc[frame.ctx_cell.isin(never)]
                .groupby("ctx_cell").size().sort_values(ascending=False))
print()
print("=== cell lines with no row in the primary graph ===")
print(f"  distinct cell-line strings : {len(all_cells):,}")
print(f"  absent entirely            : {len(never):,}")
for name, count in never_counts.head(30).items():
    dbs = sorted(frame.loc[frame.ctx_cell.eq(name), "source_database"].astype(str).unique())
    print(f"    {str(name)[:34]:36} {count:>8,}   {','.join(dbs)[:44]}")

print()
print("=== what kind of evidence is lost ===")
for column in ("is_experimental", "is_predicted", "partner_type", "experiment_family"):
    if column in out_of_graph.columns:
        print(f"  {column:20} {out_of_graph[column].astype(str).value_counts().head(5).to_dict()}")

print()
print("=== counterfactual: what each part of the ruling recovers ===")
cell_only = out_of_graph.loc[out_of_graph.has_cell & ~out_of_graph.has_tissue
                             & ~out_of_graph.has_disease]
both = out_of_graph.loc[out_of_graph.has_cell & out_of_graph.has_tissue
                        & ~out_of_graph.has_disease]
tissue_any = out_of_graph.loc[out_of_graph.has_tissue & ~out_of_graph.has_disease]
print(f"  drop cell_line as an axis         -> {len(cell_only):>9,} rows return")
print(f"  drop cell_line AND tissue as axes -> {len(both) + len(cell_only):>9,} rows return")
print(f"  (tissue is the binding constraint: {len(tissue_any):,} excluded rows name a tissue)")

mapped = frame.loc[frame.graph_status.eq("mapped_context")]
print()
print("=== contrast: rows that DID map into the graph as context edges ===")
print(f"  rows: {len(mapped):,}")
print(f"  by cancer: {mapped.graph_cancer.astype(str).value_counts().head(12).to_dict()}")

receipt_out = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "question": (
        "Which cell lines from which databases never reach the primary graph, and "
        "how much of that loss is the cell-line axis itself?"
    ),
    "ruling": (
        "A protein-lncRNA interaction is not cancer-type specific. A global edge "
        "being reachable from all 33 cancers is not a problem; the requirement is "
        "only that the interaction exist in some cell line."
    ),
    "graph_status_counts": {str(k): int(v) for k, v in frame.graph_status.value_counts().items()},
    "excluded_total": int(len(out_of_graph)),
    "excluded_by_context_field": buckets,
    "excluded_by_database": {str(k): int(v) for k, v in by_db.items()},
    "top_tissue_values": {str(k): int(v) for k, v in tissue_counts.head(40).items()},
    "tissue_distinct": int(len(tissue_counts)),
    "excluded_by_database_and_cell_line": [
        {"source_database": str(r.source_database), "cell_line": str(r.ctx_cell),
         "excluded_rows": int(r.excluded_rows)}
        for r in by_db_cell.head(300).itertuples(index=False)],
    "cell_lines_absent_from_graph": [
        {"cell_line": str(k), "rows": int(v),
         "databases": sorted(frame.loc[frame.ctx_cell.eq(k), "source_database"].astype(str).unique())}
        for k, v in never_counts.head(100).items()],
    "cell_lines_absent_count": len(never),
    "cell_lines_total": len(all_cells),
    "counterfactual": {
        "drop_cell_line_axis_recovers": int(len(cell_only)),
        "drop_cell_line_and_tissue_axes_recovers": int(len(both) + len(cell_only)),
        "rows_naming_a_tissue": int(len(tissue_any)),
        "conclusion": (
            "cell_line alone is not the binding constraint: almost no excluded row "
            "names a cell line without also naming a tissue, so tissue has to be "
            "reconsidered too before the ruling changes anything."
        ),
    },
    "library_receipt": receipt,
}
path = OUT / "CELL_LINE_GRAPH_ACCOUNTING.json"
path.write_text(json.dumps(receipt_out, indent=2, default=str), encoding="utf-8")
by_db_cell.to_csv(OUT / "CELL_LINE_GRAPH_ACCOUNTING.tsv", sep="\t", index=False)
print()
print(f"written: {path}")
print(f"written: {OUT / 'CELL_LINE_GRAPH_ACCOUNTING.tsv'}")