"""What does ruling B actually change in the global binding layer?

Ruling B: the 316,546 rows that `apply_strict_cancer_context` removes are not
removed.  They become global.  The 213,449 already-mapped rows keep going to
context edges, unchanged.

The chain being reproduced, verbatim from the frozen code:

  interaction_relation
    -> protein_layer._rebuild  (filters protein partners, then
       apply_strict_cancer_context at line 73  <-- the 316,546 are removed here)
    -> lncRNA_protein_relation_context.parquet
    -> materialize_v32_g012_graph_authorities: keep cancer_id IS NA and
       ~is_context_specific, restrict to the candidate universe and UNIPROT
       proteins, dedup on (lncrna_id, protein_id)
    -> lnc_protein_binding.parquet   (frozen: 623,207 rows)

The reproduction is validated against that frozen 623,207 before any change is
reported, so the numbers below rest on a chain that is known to land on the
frozen value.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

WORK = Path("${PRIVATE_WORK_ROOT}/CancerLncAtlas/rbp_encode_v33_20260921_r1")
sys.path.insert(0, str(WORK / "code"))

from cc_hhgt.interaction_context import apply_strict_cancer_context  # noqa: E402
from cc_hhgt.protein_layer import _normalize_uniprot  # noqa: E402

STANDARDISED = Path(
    "${PRIVATE_ARCHIVE_ROOT}/CancerLncAtlas/results/v3_state_formal_runs/"
    "V3STATE-formal500-20260810T101223Z-a8803782dcf0_server_handoff/run/assets/standardized"
)
PREPARED = Path("${PRIVATE_WORK_ROOT}/CancerLncAtlas/inputs/v32_g012_local_cnv_formal_prepared_20260903_r1")
OUT = WORK / "outputs" / "phase3_typed_binding"
OUT.mkdir(parents=True, exist_ok=True)

relation = pd.read_parquet(STANDARDISED / "interaction_relation.parquet")
dim_lnc = pd.read_parquet(STANDARDISED / "dim_lncRNA.parquet")
dim_cancer = pd.read_parquet(STANDARDISED / "dim_cancer.parquet")
protein_gene = pd.read_parquet(STANDARDISED / "protein_gene_map.parquet")
validated = pd.read_parquet(STANDARDISED / "lncRNA_protein_relation.parquet")
frozen_global = pd.read_parquet(
    Path("${PRIVATE_WORK_ROOT}/CancerLncAtlas/inputs/v32_g012_patient_first_20260830_r1"
         "/static/lnc_protein_binding.parquet"))
candidates = pd.read_parquet(PREPARED / "FORMAL_CANDIDATE_UNIVERSE.parquet",
                             columns=["lncrna_id"])
candidate_lncs = set(candidates.lncrna_id.astype(str))
validated_pairs = set(zip(validated.lncrna_id.astype(str), validated.protein_id.astype(str)))

print("=== frozen references ===")
print(f"  validated pairs (unfiltered rebuild) : {len(validated_pairs):,}")
print(f"  frozen global binding rows           : {len(frozen_global):,}")

# --- reproduce protein_layer up to and including the context call --------------
is_protein = relation.partner_type.astype(str).str.lower().str.contains("protein")
rel = relation.loc[is_protein].copy()
rel["lncrna_id"] = rel.lncrna_id.astype("string").str.strip()
rel["original_partner_id"] = rel.partner_id.astype("string").str.strip()
valid_lnc = set(dim_lnc.lncrna_id.dropna().astype(str))
rel = rel.loc[rel.lncrna_id.isin(valid_lnc)
              & rel.original_partner_id.notna()
              & rel.original_partner_id.ne("")].copy()
rel["_row_id"] = np.arange(len(rel), dtype=np.int64)
print(f"\n=== protein-partner rows entering the context gate: {len(rel):,} ===")

before, audit = apply_strict_cancer_context(rel, dim_cancer)
removed_index = rel.index.difference(before.index)
print(f"  kept by the gate   : {len(before):,}")
print(f"  removed by the gate: {len(removed_index):,}  "
      f"(library audit says {audit['excluded_unmapped_or_ambiguous_context_rows']:,})")
if len(removed_index) != audit["excluded_unmapped_or_ambiguous_context_rows"]:
    raise SystemExit("FAIL-CLOSED: the reproduced gate does not match the library audit")

# --- ruling B: the removed rows become global instead --------------------------
admitted = rel.loc[removed_index].copy()
admitted["cancer_id"] = pd.Series(pd.NA, index=admitted.index, dtype="string")
admitted["context_mapping_status"] = "global_by_ruling"
after = pd.concat([before, admitted], axis=0).sort_index()
print(f"  after ruling B     : {len(after):,}")


def finish(rows: pd.DataFrame) -> pd.DataFrame:
    """The mapping and grouping half of protein_layer._rebuild."""
    mapping = protein_gene[["gene_id", "protein_id", "mapping_multiplicity",
                            "mapping_weight"]].drop_duplicates(["gene_id", "protein_id"])
    by_gene = rows.loc[rows.original_partner_id.str.startswith("GENE:", na=False)].merge(
        mapping, left_on="original_partner_id", right_on="gene_id", how="left")
    direct = rows.loc[~rows.original_partner_id.str.startswith("GENE:", na=False)].copy()
    direct["uniprot_accession"] = _normalize_uniprot(direct.original_partner_id)
    direct = direct.merge(
        protein_gene[["uniprot_accession", "protein_id", "gene_id",
                      "mapping_multiplicity", "mapping_weight"]]
        .drop_duplicates("uniprot_accession"),
        on="uniprot_accession", how="left")
    mapped = pd.concat([by_gene, direct], ignore_index=True, sort=False)
    mapped = mapped.dropna(subset=["protein_id"]).copy()
    experimental = mapped.get("is_experimental", pd.Series(False, index=mapped.index)).fillna(False)
    mapped["weight"] = np.where(experimental, 1.0, 0.4) * mapped.mapping_weight.fillna(1.0)
    grouped = (mapped.groupby(["lncrna_id", "protein_id", "cancer_id"],
                              as_index=False, observed=True, dropna=False)
               .agg(weight=("weight", "max"),
                    n_source_records=("_row_id", "nunique")))
    return grouped


def to_global_artifact(grouped: pd.DataFrame) -> pd.DataFrame:
    """The materialize_v32_g012 selection, verbatim."""
    out = grouped.loc[
        grouped.cancer_id.isna()
        & grouped.lncrna_id.astype(str).isin(candidate_lncs)
        & grouped.protein_id.astype(str).str.startswith("UNIPROT:")
    ].copy()
    out = out.drop_duplicates(["lncrna_id", "protein_id"], keep="first")
    return out


print("\n=== running both branches through the frozen chain ===")
grouped_before = finish(before)
grouped_after = finish(after)
artifact_before = to_global_artifact(grouped_before)
artifact_after = to_global_artifact(grouped_after)

print(f"  current chain  -> global binding rows : {len(artifact_before):,}")
print(f"  frozen reference                      : {len(frozen_global):,}")
reproduces = len(artifact_before) == len(frozen_global)
print(f"  chain reproduces the frozen value     : {reproduces}")
if not reproduces:
    print("  WARNING: the reproduction misses the frozen value, so the delta below is")
    print("  indicative only and must not be quoted as the effect of the ruling.")

print(f"  ruling B chain -> global binding rows : {len(artifact_after):,}")
delta = len(artifact_after) - len(artifact_before)
print(f"  delta                                 : {delta:+,} "
      f"({len(artifact_after) / max(len(artifact_before), 1):.2f}x)")

before_pairs = set(zip(artifact_before.lncrna_id.astype(str), artifact_before.protein_id.astype(str)))
after_pairs = set(zip(artifact_after.lncrna_id.astype(str), artifact_after.protein_id.astype(str)))
new_pairs = after_pairs - before_pairs
print(f"\n=== what the ruling adds ===")
print(f"  new (lncRNA, protein) pairs          : {len(new_pairs):,}")
print(f"  lncRNAs: {artifact_before.lncrna_id.nunique():,} -> {artifact_after.lncrna_id.nunique():,} "
      f"({artifact_after.lncrna_id.nunique() - artifact_before.lncrna_id.nunique():+,})")
print(f"  proteins: {artifact_before.protein_id.nunique():,} -> {artifact_after.protein_id.nunique():,} "
      f"({artifact_after.protein_id.nunique() - artifact_before.protein_id.nunique():+,})")
inside_validated = len(new_pairs & validated_pairs)
print(f"  new pairs inside the validated superset: {inside_validated:,} / {len(new_pairs):,}")
print(f"  new pairs OUTSIDE the validated set    : {len(new_pairs - validated_pairs):,}")

# --- the context side must be untouched ---------------------------------------
ctx_before = grouped_before.loc[grouped_before.cancer_id.notna()]
ctx_after = grouped_after.loc[grouped_after.cancer_id.notna()]
print(f"\n=== context side must be untouched ===")
print(f"  context rows before: {len(ctx_before):,}")
print(f"  context rows after : {len(ctx_after):,}")
print(f"  identical          : {len(ctx_before) == len(ctx_after)}")

receipt = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "ruling": (
        "B: keep the context edges as they are; the rows that the strict context "
        "gate removes become global instead of being dropped."
    ),
    "chain_reproduced": [
        "interaction_relation -> protein partners -> apply_strict_cancer_context",
        "-> partner_id mapped to protein_id -> grouped by (lncrna_id, protein_id, cancer_id)",
        "-> keep cancer_id IS NA and candidate-universe lncRNAs and UNIPROT proteins",
        "-> dedup on (lncrna_id, protein_id)",
    ],
    "reproduction_validated": bool(reproduces),
    "gate": {
        "protein_partner_rows": int(len(rel)),
        "kept": int(len(before)),
        "removed": int(len(removed_index)),
        "library_audit_removed": int(audit["excluded_unmapped_or_ambiguous_context_rows"]),
    },
    "global_binding_rows": {
        "current": int(len(artifact_before)),
        "frozen_reference": int(len(frozen_global)),
        "after_ruling_b": int(len(artifact_after)),
        "delta": int(delta),
        "growth_factor": len(artifact_after) / max(len(artifact_before), 1),
    },
    "coverage": {
        "lncrnas_current": int(artifact_before.lncrna_id.nunique()),
        "lncrnas_after": int(artifact_after.lncrna_id.nunique()),
        "proteins_current": int(artifact_before.protein_id.nunique()),
        "proteins_after": int(artifact_after.protein_id.nunique()),
        "new_pairs": int(len(new_pairs)),
        "new_pairs_inside_validated_superset": int(inside_validated),
        "new_pairs_outside_validated_superset": int(len(new_pairs - validated_pairs)),
    },
    "context_side_untouched": bool(len(ctx_before) == len(ctx_after)),
    "note": (
        "The frozen artifact is not modified. This measures what a new generation "
        "would contain; producing it requires a new receipt and output directory."
    ),
}
path = OUT / "RULING_B_IMPACT.json"
path.write_text(json.dumps(receipt, indent=2, default=str), encoding="utf-8")
print(f"\nwritten: {path}")
