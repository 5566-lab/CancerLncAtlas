"""Step D2: ruling B on the protein layer, plus the ENCODE v2 edges, in one table.

Two changes land together because both widen the same binding layer and both must
reach the same new generation:

Ruling B
    `apply_strict_cancer_context` removes rows whose recorded context cannot be
    mapped to exactly one of the 33 cancers.  Under the ruling those rows are not
    removed: an interaction is not cancer-type specific, so they become global.
    Measured effect: 623,207 -> 650,571 global binding rows, +27,364 edges, with
    every new pair inside the validated superset.

ENCODE v2
    The reproducible-peak, exon-level layer: 13,053 global edges that are 3.09x
    enriched over chance, replacing the 428,949 that were 0.71x depleted.

Neither the frozen artifact nor the frozen code is modified.  The ruling-B behaviour
is implemented here, against the same declarations the library uses, so the original
function keeps its published contract.
"""

from __future__ import annotations

import hashlib
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
    "V3STATE-formal500-20260810T101223Z-a8803782dcf0_server_handoff/run/assets/standardized")
PREPARED = Path("${PRIVATE_WORK_ROOT}/CancerLncAtlas/inputs/v32_g012_local_cnv_formal_prepared_20260903_r1")
ENC2 = WORK / "outputs" / "phase3_encode_eclip_v2"
TYPED_DIR = WORK / "outputs" / "phase3_typed_binding"
OUT = WORK / "outputs" / "phase3_typed_binding_v2"
OUT.mkdir(parents=True, exist_ok=True)

SCHEMA = ["lncrna_id", "protein_id", "cancer_id", "graph_assay_class", "weight",
          "source_database", "n_source_records", "n_pmids", "mapping_multiplicity",
          "assay_subtype", "experiment_family", "experiment_raw", "relation_type",
          "is_context_specific", "mapping_status"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------- ruling B
print("=== ruling B: the context gate's removals become global ===")
relation = pd.read_parquet(STANDARDISED / "interaction_relation.parquet")
dim_lnc = pd.read_parquet(STANDARDISED / "dim_lncRNA.parquet")
dim_cancer = pd.read_parquet(STANDARDISED / "dim_cancer.parquet")
protein_gene = pd.read_parquet(STANDARDISED / "protein_gene_map.parquet")
validated = pd.read_parquet(STANDARDISED / "lncRNA_protein_relation.parquet")
typed = pd.read_parquet(TYPED_DIR / "lnc_protein_binding_typed.parquet")
candidates = pd.read_parquet(PREPARED / "FORMAL_CANDIDATE_UNIVERSE.parquet",
                             columns=["lncrna_id"])
candidate_lncs = set(candidates.lncrna_id.astype(str))

valid_lnc = set(dim_lnc.lncrna_id.dropna().astype(str))
rel = relation.loc[relation.partner_type.astype(str).str.lower().str.contains("protein")].copy()
rel["lncrna_id"] = rel.lncrna_id.astype("string").str.strip()
rel["original_partner_id"] = rel.partner_id.astype("string").str.strip()
rel = rel.loc[rel.lncrna_id.isin(valid_lnc) & rel.original_partner_id.notna()
              & rel.original_partner_id.ne("")].copy()
rel["_row_id"] = np.arange(len(rel), dtype=np.int64)
before, audit = apply_strict_cancer_context(rel, dim_cancer)
removed = rel.index.difference(before.index)
print(f"  protein-partner rows entering the gate : {len(rel):,}")
print(f"  removed by the gate                    : {len(removed):,}")
if len(removed) != audit["excluded_unmapped_or_ambiguous_context_rows"]:
    raise SystemExit("FAIL-CLOSED: reproduction does not match the library audit")
admitted = rel.loc[removed].copy()
admitted["cancer_id"] = pd.Series(pd.NA, index=admitted.index, dtype="string")
admitted["context_mapping_status"] = "global_by_ruling_B"
after = pd.concat([before, admitted], axis=0).sort_index()
print(f"  after ruling B                          : {len(after):,}")


def map_to_proteins(rows: pd.DataFrame) -> pd.DataFrame:
    mapping = protein_gene[["gene_id", "protein_id", "mapping_multiplicity", "mapping_weight"]
                           ].drop_duplicates(["gene_id", "protein_id"])
    by_gene = rows.loc[rows.original_partner_id.str.startswith("GENE:", na=False)].merge(
        mapping, left_on="original_partner_id", right_on="gene_id", how="left")
    direct = rows.loc[~rows.original_partner_id.str.startswith("GENE:", na=False)].copy()
    direct["uniprot_accession"] = _normalize_uniprot(direct.original_partner_id)
    direct = direct.merge(
        protein_gene[["uniprot_accession", "protein_id", "gene_id",
                      "mapping_multiplicity", "mapping_weight"]].drop_duplicates("uniprot_accession"),
        on="uniprot_accession", how="left")
    mapped = pd.concat([by_gene, direct], ignore_index=True, sort=False)
    mapped = mapped.dropna(subset=["protein_id"]).copy()
    experimental = mapped.get("is_experimental", pd.Series(False, index=mapped.index)).fillna(False)
    mapped["weight"] = np.where(experimental, 1.0, 0.4) * mapped.mapping_weight.fillna(1.0)
    return mapped


def to_global_rows(rows: pd.DataFrame) -> pd.DataFrame:
    mapped = map_to_proteins(rows)
    def _col(name: str, default: str) -> pd.Series:
        """Read an optional column, falling back rather than assuming it exists.

        `interaction_relation` carries `experiment_family` but no `assay_subtype`,
        and a row admitted by ruling B was never classified by the assay taxonomy
        because the gate removed it before classification ever ran.
        """
        if name in mapped.columns:
            return mapped[name].fillna(default).astype(str)
        return pd.Series(default, index=mapped.index, dtype=str)

    frame = pd.DataFrame({
        "lncrna_id": mapped.lncrna_id.astype(str),
        "protein_id": mapped.protein_id.astype(str),
        "cancer_id": pd.Series(pd.NA, index=mapped.index, dtype="string"),
        "graph_assay_class": "experimental_unspecified",
        "weight": mapped.weight.astype(float),
        "source_database": mapped.source_database.fillna("unknown").astype(str),
        "n_source_records": 1,
        "n_pmids": 0,
        "mapping_multiplicity": mapped.mapping_multiplicity.fillna(1.0).astype(float),
        "assay_subtype": _col("assay_subtype", "unspecified"),
        "experiment_family": _col("experiment_family", "physical_binding"),
        "experiment_raw": _col("experiment_raw", ""),
        "relation_type": "binds_protein_experimental_unspecified",
        "is_context_specific": False,
        "mapping_status": "mapped_unique",
    })
    frame = frame.loc[frame.lncrna_id.isin(candidate_lncs)
                      & frame.protein_id.str.startswith("UNIPROT:")]
    return frame.drop_duplicates(["lncrna_id", "protein_id"])


ruling_b_rows = to_global_rows(admitted)
frozen_global = pd.read_parquet(
    Path("${PRIVATE_WORK_ROOT}/CancerLncAtlas/inputs/v32_g012_patient_first_20260830_r1"
         "/static/lnc_protein_binding.parquet"))
frozen_pairs = set(zip(frozen_global.lncrna_id.astype(str), frozen_global.protein_id.astype(str)))
new_pairs = set(zip(ruling_b_rows.lncrna_id, ruling_b_rows.protein_id)) - frozen_pairs
validated_pairs = set(zip(validated.lncrna_id.astype(str), validated.protein_id.astype(str)))
outside = new_pairs - validated_pairs
print(f"  ruling-B global rows after the frozen chain filters: {len(ruling_b_rows):,}")
print(f"  new (lncRNA, protein) pairs vs the frozen layer    : {len(new_pairs):,}")
print(f"  of which OUTSIDE the validated superset            : {len(outside):,}")
if outside:
    raise SystemExit(f"FAIL-CLOSED: {len(outside)} pairs outside the validated superset")

# ---------------------------------------------------------------- ENCODE v2
print("\n=== ENCODE v2: reproducible peaks on exons ===")
enc_global = pd.read_parquet(ENC2 / "ENCODE_V2_GLOBAL.parquet")
enc_context = pd.read_parquet(ENC2 / "ENCODE_V2_CONTEXT.parquet")
dim_gene = pd.read_parquet(STANDARDISED / "dim_gene.parquet")
symbol_to_gene: dict[str, list[str]] = {}
for symbol, gene in zip(dim_gene.gene_symbol.astype(str), dim_gene.gene_id.astype(str)):
    symbol_to_gene.setdefault(symbol.upper(), []).append(gene)
gene_to_protein: dict[str, list[str]] = {}
for gene, protein in zip(protein_gene.gene_id.astype(str), protein_gene.protein_id.astype(str)):
    gene_to_protein.setdefault(gene, []).append(protein)
symbols = sorted(enc_global.rbp.astype(str).unique())
resolved = {s: sorted({p for g in symbol_to_gene.get(s.upper(), [])
                       for p in gene_to_protein.get(g, [])}) for s in symbols}
unresolved = [s for s in symbols if not resolved[s]]
print(f"  RBP symbols {len(symbols)}  resolved {len(symbols) - len(unresolved)}  "
      f"unresolved {unresolved}")
unmapped_edges = int(enc_global.rbp.isin(unresolved).sum())
enc_global = enc_global.loc[~enc_global.rbp.isin(unresolved)].copy()
enc_global["protein_id"] = enc_global.rbp.map(lambda s: resolved[s][0])
enc_context = enc_context.loc[~enc_context.rbp.isin(unresolved)].copy()
enc_context["protein_id"] = enc_context.rbp.map(lambda s: resolved[s][0])
print(f"  edges dropped for unresolved symbols: {unmapped_edges:,}")
print(f"  global {len(enc_global):,}   context {len(enc_context):,}")


def encode_block(frame: pd.DataFrame, contextual: bool) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    out["lncrna_id"] = frame.lncrna_id.astype("string")
    out["protein_id"] = frame.protein_id.astype(str)
    out["cancer_id"] = (frame.cancer_id.astype("string") if contextual
                        else pd.Series(pd.NA, index=frame.index, dtype="string"))
    out["graph_assay_class"] = "eclip"
    out["weight"] = 1.0
    out["source_database"] = "ENCODE_eCLIP_reproducible"
    out["n_source_records"] = pd.to_numeric(frame.n_experiments, errors="coerce").fillna(0).astype("int64")
    out["n_pmids"] = 0
    out["mapping_multiplicity"] = 1.0
    out["assay_subtype"] = "eclip"
    out["experiment_family"] = "physical_binding"
    out["experiment_raw"] = "eCLIP"
    out["relation_type"] = "binds_protein_eclip"
    out["is_context_specific"] = bool(contextual)
    out["mapping_status"] = "mapped_unique"
    return out[SCHEMA]


encode_rows = pd.concat([encode_block(enc_global, False), encode_block(enc_context, True)],
                        ignore_index=True)
print(f"  ENCODE rows projected onto the legacy schema: {len(encode_rows):,}")

# ---------------------------------------------------------------- merge
print("\n=== the new generation ===")
merged = pd.concat([typed[SCHEMA], ruling_b_rows[SCHEMA], encode_rows], ignore_index=True)
merged = merged.drop_duplicates().reset_index(drop=True)
merged_path = OUT / "lnc_protein_binding_typed_v2.parquet"
merged.to_parquet(merged_path, index=False)
print(f"  frozen typed rows  : {len(typed):,}")
print(f"  + ruling B rows    : {len(ruling_b_rows):,}")
print(f"  + ENCODE v2 rows   : {len(encode_rows):,}")
print(f"  = v2 rows          : {len(merged):,}")
print(f"  path               : {merged_path}")
print(f"\n  relation counts:")
for key, value in merged.relation_type.astype(str).value_counts().items():
    print(f"    {key:44s} {value:>10,}")
print(f"\n  source_database counts:")
for key, value in merged.source_database.astype(str).value_counts().head(8).items():
    print(f"    {key:44s} {value:>10,}")

receipt = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "step": "Step D2 - ruling B plus the ENCODE v2 layer, one merged binding table",
    "ruling_b": {
        "gate_rows": int(len(rel)), "removed_by_the_gate": int(len(removed)),
        "library_audit_removed": int(audit["excluded_unmapped_or_ambiguous_context_rows"]),
        "global_rows_after_filters": int(len(ruling_b_rows)),
        "new_pairs_vs_frozen": len(new_pairs),
        "new_pairs_outside_validated_superset": len(outside),
    },
    "encode_v2": {
        "global_edges": int(len(enc_global)), "context_edges": int(len(enc_context)),
        "rbp_symbols": len(symbols), "unresolved": unresolved,
        "edges_dropped_for_unresolved_symbols": unmapped_edges,
        "relation": "binds_protein_eclip",
        "source_database": "ENCODE_eCLIP_reproducible",
    },
    "merged": {
        "frozen_typed_rows": int(len(typed)),
        "ruling_b_rows": int(len(ruling_b_rows)),
        "encode_rows": int(len(encode_rows)),
        "v2_rows": int(len(merged)),
        "path": str(merged_path), "sha256": sha256_file(merged_path),
        "relation_counts": {str(k): int(v) for k, v in merged.relation_type.astype(str).value_counts().items()},
    },
    "frozen_artifacts_touched": False,
    "frozen_code_modified": False,
}
path = OUT / "PHASE3_V2_MERGE_AUDIT.json"
path.write_text(json.dumps(receipt, indent=2, default=str), encoding="utf-8")
print(f"\nwritten: {path}")