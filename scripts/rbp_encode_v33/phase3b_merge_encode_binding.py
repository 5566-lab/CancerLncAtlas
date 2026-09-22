"""Phase 3B: merge the ENCODE eCLIP edges into the typed binding layer.

The legacy typed layer and the ENCODE layer are both eCLIP evidence, so they
belong to the same graph relation (``binds_protein_eclip``).  They are kept
distinguishable by ``source_database``: RNAInter / NPInter for the frozen
cross-database records, ENCODE_eCLIP for the peak-overlap evidence.

The existing artifact is never overwritten.  The merged table is written to a new
path so the pre-ENCODE generation stays on disk and the two can be compared.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

WORK = Path("${PRIVATE_WORK_ROOT}/CancerLncAtlas/rbp_encode_v33_20260921_r1")
TYPED_DIR = WORK / "outputs" / "phase3_typed_binding"
ENC_DIR = WORK / "outputs" / "phase3_encode_eclip"
MAN = WORK / "manifests"

LEGACY = TYPED_DIR / "lnc_protein_binding_typed.parquet"
ENCODE_GLOBAL = ENC_DIR / "ENCODE_ECLIP_LNCRNA_RBP_GLOBAL.parquet"
ENCODE_CONTEXT = ENC_DIR / "ENCODE_ECLIP_LNCRNA_RBP_CONTEXT.parquet"
MERGED = TYPED_DIR / "lnc_protein_binding_typed_with_encode.parquet"

#: The exact column set of the legacy typed artifact.  The merged table keeps it
#: so every downstream consumer - including the graph materialiser, which only
#: requires a subset - sees one schema.
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


legacy = pd.read_parquet(LEGACY)
missing = sorted(set(SCHEMA) - set(legacy.columns))
if missing:
    raise SystemExit(f"FAIL-CLOSED: the legacy typed artifact lacks columns: {missing}")
print(f"=== legacy typed binding: {len(legacy):,} rows ===")
print(f"    scenario sha256 = {sha256_file(LEGACY)[:24]}…")

global_edges = pd.read_parquet(ENCODE_GLOBAL)
context_edges = pd.read_parquet(ENCODE_CONTEXT)
print(f"=== ENCODE edges: global {len(global_edges):,}  context {len(context_edges):,} ===")


def to_legacy_schema(frame: pd.DataFrame, *, contextual: bool) -> pd.DataFrame:
    """Project an ENCODE edge table onto the legacy typed schema."""
    if frame.empty:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in SCHEMA})
    out = pd.DataFrame(index=frame.index)
    out["lncrna_id"] = frame.lncrna_id.astype("string")
    out["protein_id"] = frame.protein_id.astype(str)
    out["cancer_id"] = (frame.cancer_id.astype("string") if contextual
                        else pd.Series(pd.NA, index=frame.index, dtype="string"))
    out["graph_assay_class"] = "eclip"
    out["weight"] = 1.0
    out["source_database"] = "ENCODE_eCLIP"
    # One peak-overlap call is one independent piece of evidence per source file;
    # n_pmids stays 0 because the peak files themselves carry no PMID attribution.
    out["n_source_records"] = pd.to_numeric(frame.n_source_files, errors="coerce").fillna(0).astype("int64")
    out["n_pmids"] = 0
    out["mapping_multiplicity"] = 1.0
    out["assay_subtype"] = "eclip"
    out["experiment_family"] = "physical_binding"
    out["experiment_raw"] = "eCLIP"
    out["relation_type"] = "binds_protein_eclip"
    out["is_context_specific"] = bool(contextual)
    out["mapping_status"] = "mapped_unique"
    return out[SCHEMA]


encode_rows = pd.concat(
    [to_legacy_schema(global_edges, contextual=False),
     to_legacy_schema(context_edges, contextual=True)],
    ignore_index=True,
)
print(f"    ENCODE rows projected onto the legacy schema: {len(encode_rows):,}")

# --- what the ENCODE layer adds that the frozen databases did not already have --
legacy_eclip = legacy.loc[legacy.graph_assay_class.astype(str).eq("eclip")]
legacy_pairs = set(zip(legacy_eclip.lncrna_id.astype(str), legacy_eclip.protein_id.astype(str)))
encode_pairs = set(zip(encode_rows.lncrna_id.astype(str), encode_rows.protein_id.astype(str)))
overlap = encode_pairs & legacy_pairs
new_pairs = encode_pairs - legacy_pairs
print(f"\n=== novelty of the ENCODE layer (against the frozen databases) ===")
print(f"    distinct ENCODE pairs              : {len(encode_pairs):,}")
print(f"      already in the frozen eCLIP layer: {len(overlap):,}")
print(f"      new to the graph                 : {len(new_pairs):,} "
      f"({len(new_pairs) / max(len(encode_pairs), 1) * 100:.1f}%)")

merged = pd.concat([legacy[SCHEMA], encode_rows], ignore_index=True)
merged = merged.drop_duplicates().reset_index(drop=True)
merged.to_parquet(MERGED, index=False)
print(f"\n=== merged ===")
print(f"    rows      : {len(legacy):,} -> {len(merged):,} (+{len(merged) - len(legacy):,})")
print(f"    path      : {MERGED}")
print(f"    sha256    : {sha256_file(MERGED)[:24]}…")
print(f"\n    relation counts after the merge:")
for key, value in merged.relation_type.astype(str).value_counts().items():
    print(f"      {key:40s} {value:>10,}")
print(f"\n    source_database after the merge:")
for key, value in merged.source_database.astype(str).value_counts().items():
    print(f"      {key:40s} {value:>10,}")

receipt = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "phase": "Phase 3B - merge ENCODE eCLIP evidence into the typed binding layer",
    "legacy_binding": {"path": str(LEGACY), "sha256": sha256_file(LEGACY), "rows": int(len(legacy))},
    "encode_global": {"path": str(ENCODE_GLOBAL), "sha256": sha256_file(ENCODE_GLOBAL),
                      "rows": int(len(global_edges))},
    "encode_context": {"path": str(ENCODE_CONTEXT), "sha256": sha256_file(ENCODE_CONTEXT),
                       "rows": int(len(context_edges))},
    "encode_rows_projected": int(len(encode_rows)),
    "distinct_encode_pairs": len(encode_pairs),
    "encode_pairs_already_in_frozen_eclip": len(overlap),
    "encode_pairs_new": len(new_pairs),
    "novelty_fraction": len(new_pairs) / max(len(encode_pairs), 1),
    "merged": {"path": str(MERGED), "sha256": sha256_file(MERGED), "rows": int(len(merged))},
    "merged_relation_counts": merged.relation_type.astype(str).value_counts().to_dict(),
    "merged_source_database_counts": merged.source_database.astype(str).value_counts().to_dict(),
    "legacy_artifact_overwritten": False,
    "schema": SCHEMA,
    "notes": [
        "ENCODE evidence enters the same binds_protein_eclip relation as the frozen "
        "cross-database records; source_database keeps the two separable.",
        "n_pmids is 0 for ENCODE rows because peak files carry no PMID attribution.",
    ],
}
receipt_path = ENC_DIR / "PHASE3B_MERGE_AUDIT.json"
receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
print(f"\nwritten: {MERGED}")
print(f"written: {receipt_path}")
