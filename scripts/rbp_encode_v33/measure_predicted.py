"""Measure what admitting `binds_protein_predicted` would actually do.

The question is not "are there predictions" - there are 773,283 rows of them - but
what changes in the graph under the same candidate-universe filter the authority
build uses.  Two numbers matter and neither had been measured:

* how many predicted edges survive the filter (the denominator of any claim about
  "87% of the binding layer is prediction");
* how much of the lncRNA and protein coverage exists *only* because of predicted
  edges, which is the real form of the "too few edges" worry.

No graph is written.  Nothing is toggled.  This only counts.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

WORK = Path("${PRIVATE_WORK_ROOT}/CancerLncAtlas/rbp_encode_v33_20260921_r1")
sys.path.insert(0, str(WORK / "code"))

from cc_hhgt.v32.formal_graph import materialize_typed_lnc_protein_binding  # noqa: E402

PREPARED = Path("${PRIVATE_WORK_ROOT}/CancerLncAtlas/inputs/v32_g012_local_cnv_formal_prepared_20260903_r1")
TYPED = WORK / "outputs" / "phase3_typed_binding" / "lnc_protein_binding_typed.parquet"
OUT = WORK / "outputs" / "phase3_typed_binding"
PREDICTED_CLASS = "predicted"

typed = pd.read_parquet(TYPED)
candidates = pd.read_parquet(PREPARED / "FORMAL_CANDIDATE_UNIVERSE.parquet",
                             columns=["lncrna_id"])
candidate_lncrnas = candidates.lncrna_id.astype(str).unique()
print(f"=== inputs ===")
print(f"  typed binding rows    : {len(typed):,}")
print(f"  candidate-universe lncRNAs : {len(candidate_lncrnas):,}")
print(f"  graph_assay_class present  : "
      f"{sorted(typed.graph_assay_class.astype(str).unique())}")

# --- the filter the authority build actually applies ---------------------------
in_scope = typed.lncrna_id.astype(str).isin(set(candidate_lncrnas))
print(f"\n=== candidate-universe filter is not a no-op ===")
print(f"  rows in scope    : {int(in_scope.sum()):,} ({in_scope.mean():.2%})")
print(f"  rows filtered out: {int((~in_scope).sum()):,}")

for cls in sorted(typed.graph_assay_class.astype(str).unique()):
    sub = typed.loc[typed.graph_assay_class.astype(str).eq(cls)]
    kept = int(sub.lncrna_id.astype(str).isin(set(candidate_lncrnas)).sum())
    print(f"    {cls:28} {len(sub):>9,} -> {kept:>9,} in scope "
          f"({kept / max(len(sub), 1):.1%})")

# --- materialise both ways, with the same filter -------------------------------
print(f"\n=== materialised edges under the candidate-universe filter ===")
results = {}
for label, predicted in (("physical_only", False), ("with_predicted", True)):
    edges = materialize_typed_lnc_protein_binding(
        typed, include_predicted=predicted, candidate_lncrnas=candidate_lncrnas)
    counts = {str(k): int(v) for k, v in edges.relation_type.value_counts().items()}
    results[label] = {
        "edges": int(len(edges)),
        "relation_counts": counts,
        "lncrnas": int(edges.source_id.nunique()),
        "proteins": int(edges.target_id.nunique()),
    }
    print(f"  [{label:15}] edges={len(edges):>8,}  "
          f"lncRNAs={edges.source_id.nunique():>6,}  proteins={edges.target_id.nunique():>6,}")
    for key, value in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"        {key:44s} {value:>9,}")

physical = results["physical_only"]
withpred = results["with_predicted"]
predicted_edges = withpred["relation_counts"].get("binds_protein_predicted", 0)
print(f"\n=== the two numbers that were missing ===")
print(f"  predicted edges after the filter : {predicted_edges:,}")
print(f"  predicted share of the binding layer : "
      f"{predicted_edges / max(withpred['edges'], 1):.2%}")
print(f"  binding layer growth : {physical['edges']:,} -> {withpred['edges']:,} "
      f"({withpred['edges'] / max(physical['edges'], 1):.2f}x)")

# --- coverage: what only predictions can reach ---------------------------------
print(f"\n=== coverage contribution (the real form of 'too few edges') ===")
phys_edges = materialize_typed_lnc_protein_binding(
    typed, include_predicted=False, candidate_lncrnas=candidate_lncrnas)
pred_edges = materialize_typed_lnc_protein_binding(
    typed.loc[typed.graph_assay_class.astype(str).eq(PREDICTED_CLASS)],
    include_predicted=True, candidate_lncrnas=candidate_lncrnas)
phys_pairs = set(zip(phys_edges.source_id.astype(str), phys_edges.target_id.astype(str)))
pred_pairs = set(zip(pred_edges.source_id.astype(str), pred_edges.target_id.astype(str)))
phys_lnc = set(phys_edges.source_id.astype(str))
pred_lnc = set(pred_edges.source_id.astype(str))
phys_pro = set(phys_edges.target_id.astype(str))
pred_pro = set(pred_edges.target_id.astype(str))

coverage = {
    "pairs_physical_only": len(phys_pairs),
    "pairs_predicted": len(pred_pairs),
    "pairs_shared": len(phys_pairs & pred_pairs),
    "pairs_predicted_only": len(pred_pairs - phys_pairs),
    "lncrnas_physical_only": len(phys_lnc),
    "lncrnas_predicted_only": len(pred_lnc - phys_lnc),
    "lncrnas_predicted_only_fraction_of_universe":
        len(pred_lnc - phys_lnc) / max(len(candidate_lncrnas), 1),
    "proteins_physical_only": len(phys_pro),
    "proteins_predicted_only": len(pred_pro - phys_pro),
}
for key, value in coverage.items():
    print(f"  {key:46} {value:,}")

# --- what the predicted rows actually are --------------------------------------
print(f"\n=== provenance of the predicted rows ===")
pred_rows = typed.loc[typed.graph_assay_class.astype(str).eq(PREDICTED_CLASS)]
prov = pred_rows.experiment_raw.astype(str).value_counts()
for key, value in prov.head(10).items():
    print(f"    {key[:60]:62} {value:>9,}")
print(f"    distinct experiment_raw values: {pred_rows.experiment_raw.nunique():,}")
print(f"    distinct source_database      : "
      f"{sorted(pred_rows.source_database.astype(str).unique())}")

receipt = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "question": (
        "What changes if binds_protein_predicted is admitted, measured under the "
        "same candidate-universe filter the authority build uses?"
    ),
    "inputs": {
        "typed_binding_rows": int(len(typed)),
        "candidate_universe_lncrnas": int(len(candidate_lncrnas)),
        "rows_in_scope": int(in_scope.sum()),
        "rows_filtered_out": int((~in_scope).sum()),
    },
    "materialised": results,
    "predicted_edges_after_filter": int(predicted_edges),
    "predicted_share_of_binding_layer": predicted_edges / max(withpred["edges"], 1),
    "binding_layer_growth": withpred["edges"] / max(physical["edges"], 1),
    "coverage": coverage,
    "provenance": {str(k): int(v) for k, v in prov.head(20).items()},
    "distinct_prediction_methods": int(pred_rows.experiment_raw.nunique()),
    "note": (
        "Counts only. No graph was written and no switch was toggled. This is the "
        "measurement that has to exist before 'let the model decide' can mean "
        "anything, because it fixes the denominator."
    ),
}
path = OUT / "PREDICTED_ADMISSION_MEASUREMENT.json"
path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
print(f"\nwritten: {path}")
