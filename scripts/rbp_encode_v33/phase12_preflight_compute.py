"""Phase 12 CPU hard gate: compute every preflight item and emit the report.

Gates:
  1  relevant pytest                                              -> from logs
  2  full suite, no new failures                                  -> from logs
  3  ENCODE manifest SHA                                          -> blocked, reported
  4  genome assembly gate                                         -> Phase 5/6
  5  RBP -> UniProt mapping audit                                 -> computed here
  6  assay_subtype missingness report                             -> computed here
  7  cross-database duplicate audit                               -> Phase 4
  8  graph count report                                           -> computed here
  9  context leakage test                                         -> computed here
 10  G0/G1/G2 invariant                                           -> Phase 3 supplement
 11  synthetic forward/backward                                   -> separate run
 12  legacy mode reproduces old generic semantics                 -> Phase 3, PASS
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

WORK = Path("${PRIVATE_WORK_ROOT}/CancerLncAtlas/rbp_encode_v33_20260921_r1")
sys.path.insert(0, str(WORK / "code"))

from cc_hhgt.v32.formal_graph import (  # noqa: E402
    CONTEXT_ECLIP_ROLE,
    GLOBAL_BINDING_ROLE,
    materialize_typed_lnc_protein_binding,
)
from cc_hhgt.v32.rbp_assay import ASSAY_SUBTYPES, GRAPH_ASSAY_CLASSES  # noqa: E402

TYPED = WORK / "outputs" / "phase3_typed_binding" / "lnc_protein_binding_typed.parquet"
AUDIT = WORK / "outputs" / "phase3_typed_binding" / "PHASE3_TYPED_BINDING_AUDIT.json"
OUT = WORK / "manifests" / "PHASE12_CPU_PREFLIGHT.json"
REPORT = WORK / "reports" / "RBP_ENCODE_CPU_PREFLIGHT.md"

typed = pd.read_parquet(TYPED)
audit = json.loads(AUDIT.read_text())
legacy_audit = audit["legacy_audit"]
typed_audit = audit["typed_audit"]

results: dict[str, object] = {}

# ---------------------------------------------------------------- gate 5
print("=== gate 5: RBP -> UniProt mapping audit ===")
input_rows = {"legacy": legacy_audit["input_rows"], "typed": typed_audit["input_rows"]}
rejected = {
    "legacy": legacy_audit["protein_mapping_rejected_rows"],
    "typed": typed_audit["protein_mapping_rejected_rows"],
}
mapping = {
    "input_relation_rows": input_rows,
    "unmappable_partner_rows": rejected,
    "mapped_fraction": {
        key: round(1.0 - rejected[key] / input_rows[key], 6) for key in input_rows
    },
    "all_emitted_proteins_are_uniprot": bool(
        typed.protein_id.astype(str).str.startswith("UNIPROT:").all()
    ),
    "distinct_proteins": int(typed.protein_id.nunique()),
    "distinct_lncrnas": int(typed.lncrna_id.nunique()),
}
for key, value in mapping.items():
    print(f"    {key:34s} {value}")
results["rbp_uniprot_mapping"] = mapping

# ---------------------------------------------------------------- gate 6
print("\n=== gate 6: assay_subtype missingness ===")
classes = typed.graph_assay_class.astype(str)
subtype_counts = (
    typed.assign(_s=typed.assay_subtype.astype(str))
    .groupby("_s", observed=True)
    .size()
    .sort_values(ascending=False)
)
total_rows = int(len(typed))
uninformative = int(classes.isin({"unknown", "experimental_unspecified"}).sum())
missingness = {
    "rows": total_rows,
    "assay_subtype_counts": {k: int(v) for k, v in subtype_counts.items()},
    "graph_assay_class_counts": {
        k: int(v) for k, v in classes.value_counts().items()
    },
    "uninformative_rows": uninformative,
    "uninformative_fraction": round(uninformative / total_rows, 6),
    "vocabulary_coverage": len(set(subtype_counts.index) & set(ASSAY_SUBTYPES)),
}
print(f"    rows                       : {total_rows:,}")
print(f"    uninformative (unknown/unspecified) : {uninformative:,} "
      f"({100.0 * uninformative / total_rows:.2f}%)")
for key, value in missingness["graph_assay_class_counts"].items():
    print(f"       {key:28s} {value:>10,}")
results["assay_subtype_missingness"] = missingness

# ---------------------------------------------------------------- gate 8 + 9
print("\n=== gate 8/9: graph counts and context leakage ===")
graph_report: dict[str, object] = {}
for predicted in (False, True):
    edges = materialize_typed_lnc_protein_binding(typed, include_predicted=predicted)
    label = "with_predicted" if predicted else "physical_only"
    contextual = edges.loc[edges.edge_role.eq(CONTEXT_ECLIP_ROLE)]
    globals_ = edges.loc[edges.edge_role.eq(GLOBAL_BINDING_ROLE)]
    leakage = int(
        (
            contextual.cancer_id.isna()
            | ~contextual.is_context_specific.astype(bool)
        ).sum()
    )
    graph_report[label] = {
        "edges": int(len(edges)),
        "relation_type_counts": {
            k: int(v) for k, v in edges.relation_type.value_counts().items()
        },
        "edge_role_counts": {
            k: int(v) for k, v in edges.edge_role.value_counts().items()
        },
        "context_specific_edges": int(len(contextual)),
        "global_edges": int(len(globals_)),
        "context_edges_without_a_cancer": leakage,
        "distinct_context_cancers": sorted(
            contextual.cancer_id.dropna().astype(str).unique()
        ),
    }
    print(f"    [{label}] edges={len(edges):,} "
          f"context={len(contextual):,} global={len(globals_):,} "
          f"leakage={leakage}")
    for key, value in sorted(
        graph_report[label]["relation_type_counts"].items(), key=lambda x: -x[1]
    ):
        print(f"         {key:44s} {value:>10,}")
results["graph_counts"] = graph_report

leakage_total = sum(v["context_edges_without_a_cancer"] for v in graph_report.values())
results["context_leakage_gate"] = {
    "context_edges_without_a_cancer": leakage_total,
    "pass": leakage_total == 0,
}
print(f"    context leakage gate: "
      f"{'PASS' if leakage_total == 0 else 'FAIL'} ({leakage_total} leaking edges)")

# ---------------------------------------------------------------- context exclusions
results["context_policy"] = {
    "context_specific_input_rows": typed_audit["context_specific_input_rows"],
    "mapped_context_rows": typed_audit["mapped_context_rows"],
    "excluded_unmapped_or_ambiguous_rows":
        typed_audit["excluded_unmapped_or_ambiguous_context_rows"],
    "excluded_fraction": round(
        typed_audit["excluded_unmapped_or_ambiguous_context_rows"]
        / max(typed_audit["context_specific_input_rows"], 1),
        6,
    ),
    "context_specific_lost_to_global": typed_audit["context_specific_lost_to_global"],
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
print(f"\nwritten: {OUT}")
