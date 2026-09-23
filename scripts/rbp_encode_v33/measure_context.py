"""What does "global" actually mean in this project, and what does the K562 ruling change?

`cc_hhgt/interaction_context.py` states the rule in code:

    "Map contextual interaction rows or exclude them; never globalize them.
     Context-free evidence remains global."

and `CELL_LINE_CANCER_HINTS` deliberately omits K562, because K562 is CML and CML
is not one of the 33 TCGA cancers.  So under the standing rule an eCLIP file whose
biosample is K562 is *contextual* and *unmappable*, which means excluded.

This script measures that claim instead of trusting the docstring, and then
quantifies what changes if ENCODE eCLIP edges are declared context-free (the
ruling), while the frozen layer keeps the original rule.
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
    apply_strict_cancer_context,
)

STANDARDISED = Path(
    "${PRIVATE_ARCHIVE_ROOT}/CancerLncAtlas/results/v3_state_formal_runs/"
    "V3STATE-formal500-20260810T101223Z-a8803782dcf0_server_handoff/run/assets/standardized"
)
ENC_DIR = WORK / "outputs" / "phase3_encode_eclip"
OUT = WORK / "outputs" / "phase3_typed_binding"

print("=== the mapping table the project actually uses ===")
print(f"  cell lines with a cancer hint : {len(CELL_LINE_CANCER_HINTS)}")
print(f"  K562 present                  : {'K562' in CELL_LINE_CANCER_HINTS}")
print(f"  cancers covered               : {sorted(set(CELL_LINE_CANCER_HINTS.values()))}")

interaction = pd.read_parquet(STANDARDISED / "interaction_relation.parquet")
dim_cancer = pd.read_parquet(STANDARDISED / "dim_cancer.parquet")
print(f"\n=== interaction_relation ===")
print(f"  rows: {len(interaction):,}")

# ---------------------------------------------------------------- the three-way split
print("\n=== applying the project's own strict context rule ===")
scoped, ctx_receipt = apply_strict_cancer_context(interaction, dim_cancer)
summary = {
    "rows": int(len(scoped)),
    "global_context_free": int(scoped.context_mapping_status.eq("global_context_free").sum()),
    "mapped_context": int(scoped.context_mapping_status.eq("mapped_context").sum()),
    "excluded_contextual": int(
        (~scoped.context_mapping_status.eq("global_context_free")
         & ~scoped.context_mapping_status.eq("mapped_context")).sum()),
}
for key, value in summary.items():
    if key != "rows":
        print(f"  {key:24} {value:>10,}  ({value / len(scoped):.2%})")
print(f"  {'rows':24} {summary['rows']:>10,}")

excluded_status = scoped.loc[
    ~scoped.context_mapping_status.isin(["global_context_free", "mapped_context"]),
    "context_mapping_status"].value_counts()
print("\n  exclusion reasons:")
for key, value in excluded_status.head(8).items():
    print(f"    {str(key)[:64]:66} {value:>9,}")

# ---------------------------------------------------------------- is "global" context-free?
print("\n=== is the frozen global layer really context-free? ===")
frozen_global = pd.read_parquet(
    Path("${PRIVATE_WORK_ROOT}/CancerLncAtlas/inputs/v32_g012_patient_first_20260830_r1"
         "/static/lnc_protein_binding.parquet"))
print(f"  frozen global binding rows : {len(frozen_global):,}")
print(f"  frozen is_context_specific : "
      f"{frozen_global.is_context_specific.value_counts().to_dict()}")
print(f"  frozen cancer_id non-null  : {int(frozen_global.cancer_id.notna().sum()):,}")
print(f"  frozen source_database     : "
      f"{frozen_global.source_database.astype(str).value_counts().to_dict()}")

# The context-free rows should be the pool the global layer was built from.
free = scoped.loc[scoped.context_mapping_status.eq("global_context_free")]
free_pairs = set(zip(free.lncrna_id.astype(str), free.partner_id.astype(str)))
glob_pairs = set(zip(frozen_global.lncrna_id.astype(str), frozen_global.protein_id.astype(str)))
print(f"  context-free (lncRNA, partner) pairs : {len(free_pairs):,}")
print(f"  frozen global (lncRNA, protein) pairs: {len(glob_pairs):,}")
print(f"  overlap                              : {len(free_pairs & glob_pairs):,}")
print("  (the two id spaces need not be identical, so a partial overlap is expected;")
print("   what matters is whether ANY frozen global row traces to a named cell line)")

# How often does any context field appear at all, by assay family?
print("\n=== how often does a context field appear? ===")
for column in ("disease_raw", "tissue", "cell_line"):
    filled = interaction[column].notna() & ~interaction[column].astype(str).str.strip().isin(
        ["", "-", "NA", "N/A", "NAN", "NONE", "NULL", "UNKNOWN", "<NA>"])
    print(f"  {column:14} filled {int(filled.sum()):>10,} ({filled.mean():.2%})")

print("\n=== the most common named cell lines ===")
cl = interaction.loc[:, "cell_line"].astype(str).str.strip().str.upper()
cl = cl[~cl.isin(["", "-", "NA", "N/A", "NAN", "NONE", "NULL", "UNKNOWN", "<NA>"])]
top = cl.value_counts().head(15)
for key, value in top.items():
    hint = CELL_LINE_CANCER_HINTS.get(key, "NO HINT -> excluded")
    print(f"    {key[:22]:24} {value:>9,}   {hint}")

# ---------------------------------------------------------------- eCLIP specifically
print("\n=== eCLIP rows specifically ===")
is_eclip = interaction.experiment_raw.astype(str).str.contains("eclip", case=False, na=False)
eclip_rows = scoped.loc[is_eclip]
print(f"  eCLIP rows            : {len(eclip_rows):,}")
print(f"  status breakdown      : "
      f"{eclip_rows.context_mapping_status.astype(str).value_counts().to_dict()}")
eclip_cl = interaction.loc[is_eclip, "cell_line"].astype(str).str.strip().str.upper()
eclip_cl = eclip_cl[~eclip_cl.isin(["", "-", "NA", "N/A", "NAN", "NONE", "NULL", "UNKNOWN", "<NA>"])]
print(f"  eCLIP named cell lines: {eclip_cl.value_counts().head(10).to_dict()}")

# ---------------------------------------------------------------- what the ruling changes
print("\n=== what the ruling changes for the ENCODE layer ===")
encode_global = pd.read_parquet(ENC_DIR / "ENCODE_ECLIP_LNCRNA_RBP_GLOBAL.parquet")
has_k562 = encode_global.biosamples.astype(str).str.contains("K562", regex=False)
has_hepg2 = encode_global.biosamples.astype(str).str.contains("HepG2", regex=False)
has_adrenal = encode_global.biosamples.astype(str).str.contains("adrenal", regex=False)
groups = {
    "K562 only": int((has_k562 & ~has_hepg2 & ~has_adrenal).sum()),
    "HepG2 only": int((~has_k562 & has_hepg2 & ~has_adrenal).sum()),
    "adrenal only": int((~has_k562 & ~has_hepg2 & has_adrenal).sum()),
    "K562 + HepG2": int((has_k562 & has_hepg2).sum()),
    "other combinations": int(len(encode_global) - (has_k562 | has_hepg2 | has_adrenal).sum()),
}
for key, value in groups.items():
    print(f"  {key:22} {value:>9,}")
print(f"  total ENCODE global edges: {len(encode_global):,}")
print(f"\n  Under the standing rule, every one of these is contextual and unmappable")
print(f"  except the HepG2 subset, which maps to LIHC.")
print(f"  Under the ruling, all of them are context-free and may sit in the global layer.")

receipt = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "question": (
        "Does 'global' mean 'aggregated across cell lines' or 'no context attached', "
        "and what does declaring ENCODE eCLIP edges context-free change?"
    ),
    "project_rule": {
        "source": "cc_hhgt/interaction_context.py",
        "verbatim": "Map contextual interaction rows or exclude them; never globalize them.",
        "global_means": "context-free: no disease, tissue or cell line attached",
        "cell_lines_with_hints": len(CELL_LINE_CANCER_HINTS),
        "k562_has_a_hint": "K562" in CELL_LINE_CANCER_HINTS,
        "why_absent": "K562 is CML, which is not one of the 33 TCGA cancers",
    },
    "split": summary,
    "library_receipt": ctx_receipt,
    "exclusion_reasons": {str(k): int(v) for k, v in excluded_status.items()},
    "frozen_global_layer": {
        "rows": int(len(frozen_global)),
        "is_context_specific_values": {
            str(k): int(v) for k, v in frozen_global.is_context_specific.value_counts().items()},
        "cancer_id_non_null": int(frozen_global.cancer_id.notna().sum()),
    },
    "eclip_rows": int(len(eclip_rows)),
    "eclip_status": {str(k): int(v) for k, v in
                     eclip_rows.context_mapping_status.astype(str).value_counts().items()},
    "encode_global_edges_by_biosample": groups,
    "encode_global_edges_total": int(len(encode_global)),
    "ruling": {
        "text": "ENCODE eCLIP K562 edges are not context-specific",
        "effect": (
            "Overrides the standing rule for ENCODE eCLIP edges only. The frozen "
            "layer keeps the original rule; nothing is retroactively reinterpreted."
        ),
        "rationale": (
            "An eCLIP biosample is the material the assay was run in, not a claim "
            "about which cancer the association holds in. The standing rule treats "
            "a named cell line as a disease-context claim, which is right for a "
            "curated association and wrong for a physical-binding measurement."
        ),
    },
}
path = OUT / "CONTEXT_SEMANTICS_MEASUREMENT.json"
path.write_text(json.dumps(receipt, indent=2, default=str), encoding="utf-8")
print(f"\nwritten: {path}")
