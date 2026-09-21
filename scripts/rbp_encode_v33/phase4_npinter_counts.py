"""Phase 4 / final-report questions A and B.

Quantify, on the REAL source tables:
  A. how many lncRNA-protein (RBP) relations carry a concrete assay type;
  B. how many were collapsed into the single token ``physical_binding``.

Run with the work-root environment sourced.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb

sys.path.insert(0, "${PRIVATE_WORK_ROOT}/CancerLncAtlas/rbp_encode_v33_20260921_r1/code")
from cc_hhgt.v32.rbp_assay import classify_assay, legacy_experiment_family  # noqa: E402

INPUT = Path("${PRIVATE_WORK_ROOT}/CancerLncAtlas/input")
NPINTER = INPUT / "npinter5_human_lncRNA_interactions.comp.tsv"
OUT = Path("${PRIVATE_WORK_ROOT}/CancerLncAtlas/rbp_encode_v33_20260921_r1/manifests")

con = duckdb.connect()
q = lambda s: con.execute(s).fetchall()

print("=" * 78)
print("NPInter5 comprehensive table")
print("=" * 78)
con.execute(
    f"CREATE VIEW npi AS SELECT * FROM read_csv('{NPINTER}', delim='\t', header=true, "
    "all_varchar=true, ignore_errors=true)"
)
total = q("SELECT count(*) FROM npi")[0][0]
print(f"total rows            : {total:,}")

print("\n-- distinct partner_type --")
for value, n in q(
    "SELECT partner_type, count(*) FROM npi GROUP BY 1 ORDER BY 2 DESC LIMIT 20"
):
    print(f"   {str(value)[:34]:36s} {n:>10,}")

print("\n-- rows with any partner_type matching RBP-like protein classes --")
rbp_like = q(
    "SELECT count(*) FROM npi WHERE lower(coalesce(partner_type,'')) "
    "SIMILAR TO '%(rbp|protein|rna binding)%'"
)[0][0]
print(f"   RBP/protein-like rows : {rbp_like:,}")


def classify_rows(rows):
    from collections import Counter

    legacy = Counter()
    subtype = Counter()
    graph = Counter()
    pairs = set()
    blank = 0
    for lnc, partner, method in rows:
        raw = "" if method is None else str(method).strip()
        if not raw:
            blank += 1
        legacy[legacy_experiment_family(raw)] += 1
        a = classify_assay(raw)
        subtype[a.assay_subtype] += 1
        graph[a.graph_assay_class] += 1
        if lnc and partner:
            pairs.add((str(lnc), str(partner)))
    return legacy, subtype, graph, pairs, blank


print("\n-- assay distribution over ALL NPInter rows --")
rows = q("SELECT RNA_name, partner_name, methods FROM npi")
legacy, subtype, graph, pairs, blank = classify_rows(rows)
print(f"   distinct lncRNA-partner pairs : {len(pairs):,}")
print(f"   rows with empty methods       : {blank:,}")
print("\n   LEGACY experiment_family:")
for k, v in legacy.most_common():
    print(f"      {k:26s} {v:>10,}")
print("\n   NEW assay_subtype:")
for k, v in subtype.most_common():
    print(f"      {k:26s} {v:>10,}")
print("\n   NEW graph_assay_class:")
for k, v in graph.most_common():
    print(f"      {k:26s} {v:>10,}")

collapsed = legacy.get("physical_binding", 0)
recovered = sum(
    v for k, v in subtype.items()
    if k in {"eclip", "par_clip", "iclip", "hits_clip", "clip_unspecified",
             "rip", "chirp", "rap", "chart", "rna_pulldown", "emsa", "co_ip",
             "other_experimental"}
) - subtype.get("other_experimental", 0)
print("\n" + "=" * 78)
print("QUESTION B: rows historically collapsed into 'physical_binding'")
print(f"   {collapsed:,}")
print("   of which the new taxonomy recovers a finer subtype:")
print(f"   {recovered:,}")
print("=" * 78)

report = {
    "npinter_rows": total,
    "npinter_distinct_pairs": len(pairs),
    "npinter_blank_methods": blank,
    "legacy_family": dict(legacy),
    "assay_subtype": dict(subtype),
    "graph_assay_class": dict(graph),
    "collapsed_to_physical_binding": collapsed,
    "recovered_finer_subtype": recovered,
}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "PHASE4_NPINTER_ASSAY_COUNTS.json").write_text(
    json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(f"\nwritten: {OUT / 'PHASE4_NPINTER_ASSAY_COUNTS.json'}")
