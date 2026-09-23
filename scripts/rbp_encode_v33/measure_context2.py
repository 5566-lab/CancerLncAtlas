"""Close two gaps in the previous measurement before any recommendation is made.

1. `apply_strict_cancer_context` returned 5,844,161 rows from 6,160,707 input and
   reported zero excluded.  Where did 316,546 rows go, and does "exclude" mean
   "label" or "drop"?
2. The join back to the frozen global layer returned zero overlap because the two
   id spaces differ, so it proved nothing.  Restate that honestly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

WORK = Path("${PRIVATE_WORK_ROOT}/CancerLncAtlas/rbp_encode_v33_20260921_r1")
sys.path.insert(0, str(WORK / "code"))

from cc_hhgt.interaction_context import (  # noqa: E402
    apply_strict_cancer_context,
)

STANDARDISED = Path(
    "${PRIVATE_ARCHIVE_ROOT}/CancerLncAtlas/results/v3_state_formal_runs/"
    "V3STATE-formal500-20260810T101223Z-a8803782dcf0_server_handoff/run/assets/standardized"
)

interaction = pd.read_parquet(STANDARDISED / "interaction_relation.parquet")
dim_cancer = pd.read_parquet(STANDARDISED / "dim_cancer.parquet")
scoped, receipt = apply_strict_cancer_context(interaction, dim_cancer)

print("=== the library's own receipt ===")
for key, value in receipt.items():
    if isinstance(value, dict):
        print(f"  {key}:")
        for k, v in list(value.items())[:12]:
            print(f"      {k:10} {v:>10,}")
    else:
        print(f"  {key:52} {value:>12,}" if isinstance(value, (int, float)) else f"  {key}: {value}")

print(f"\n=== drop accounting ===")
print(f"  input rows          : {len(interaction):,}")
print(f"  returned rows       : {len(scoped):,}")
print(f"  rows unaccounted for: {len(interaction) - len(scoped):,}")
print(f"  status values returned: "
      f"{scoped.context_mapping_status.astype(str).value_counts().to_dict()}")

# Which rows vanished?  Compare on the interaction id.
if "interaction_id" in interaction.columns and "interaction_id" in scoped.columns:
    lost = interaction.loc[~interaction.interaction_id.isin(scoped.interaction_id)]
    print(f"\n=== the {len(lost):,} rows that were dropped, not labelled ===")
    for column in ("disease_raw", "tissue", "cell_line"):
        filled = lost[column].notna() & ~lost[column].astype(str).str.strip().isin(
            ["", "-", "NA", "N/A", "NAN", "NONE", "NULL", "UNKNOWN", "<NA>"])
        print(f"  {column:14} filled in dropped rows: {int(filled.sum()):>9,} "
              f"({filled.mean():.2%})")
    cl = lost.cell_line.astype(str).str.strip().str.upper()
    cl = cl[~cl.isin(["", "-", "NA", "N/A", "NAN", "NONE", "NULL", "UNKNOWN", "<NA>"])]
    print(f"  dropped rows naming a cell line: {len(cl):,}")
    print(f"  top dropped cell lines: {cl.value_counts().head(10).to_dict()}")
    print(f"\n  of the dropped rows, how many name K562? "
          f"{int(cl.str.contains('K562').sum()):,}")
else:
    print("  (no shared id column; cannot separate dropped from relabelled)")
    lost = None

print("\n=== RESTATING the frozen-layer check honestly ===")
print("  The earlier join returned 0 overlap because interaction_relation.partner_id")
print("  and lnc_protein_binding.protein_id live in different id spaces. It therefore")
print("  proved nothing either way. The claim 'the frozen global layer is context-free'")
print("  rests on the rule in interaction_context.py plus the observed facts that all")
print("  623,207 frozen global rows carry is_context_specific=False and cancer_id=NA.")
print("  It was NOT verified by tracing a global row back to its source record.")

out = {
    "library_receipt": receipt,
    "input_rows": int(len(interaction)),
    "returned_rows": int(len(scoped)),
    "unaccounted_rows": int(len(interaction) - len(scoped)),
    "status_values_returned": {
        str(k): int(v) for k, v in scoped.context_mapping_status.astype(str).value_counts().items()},
    "exclude_is_implemented_as": "row removed from the returned frame, not labelled and kept",
    "frozen_layer_check": {
        "join_possible": False,
        "why": "interaction_relation.partner_id and lnc_protein_binding.protein_id are different id spaces",
        "evidence_for_context_free": [
            "interaction_context.apply_strict_cancer_context docstring: never globalize contextual rows",
            "all 623,207 frozen global rows have is_context_specific=False",
            "all 623,207 frozen global rows have cancer_id=NA",
        ],
        "verified_by_tracing_a_row": False,
    },
}
if lost is not None:
    cl = lost.cell_line.astype(str).str.strip().str.upper()
    named = cl[~cl.isin(["", "-", "NA", "N/A", "NAN", "NONE", "NULL", "UNKNOWN", "<NA>"])]
    out["dropped_rows"] = {
        "n": int(len(lost)),
        "naming_a_cell_line": int(len(named)),
        "top_cell_lines": {str(k): int(v) for k, v in named.value_counts().head(15).items()},
        "naming_k562": int(named.str.contains("K562").sum()),
    }

path = WORK / "outputs" / "phase3_typed_binding" / "CONTEXT_DROP_ACCOUNTING.json"
path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
print(f"\nwritten: {path}")
