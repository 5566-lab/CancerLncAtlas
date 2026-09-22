"""Path B step 0: pin the GRCh38 coordinate authority for the lncRNA nodes.

The graph can only accept an eCLIP peak if a peak interval can be compared with
a lncRNA interval.  The V3.2 node universe stores lncRNAs as ``LNC:ENSG…``, and
the project's own standardised asset ``dim_lncRNA.parquet`` carries chromosome /
start / end / strand / annotation_release for exactly that identifier space.  No
external annotation is introduced: the coordinate authority is the same asset
the rest of V3.2 already uses, so the ID space cannot drift.

The assembly is *verified*, not assumed.  Three loci whose GRCh38 coordinates are
common knowledge (MALAT1, NEAT1, XIST) are compared against the table; a table
that disagrees would be an hg19 table wearing a GENCODE label, and the script
fails closed rather than lifting every peak to the wrong place.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

WORK = Path("${PRIVATE_WORK_ROOT}/CancerLncAtlas/rbp_encode_v33_20260921_r1")
PREPARED = Path("${PRIVATE_WORK_ROOT}/CancerLncAtlas/inputs/v32_g012_local_cnv_formal_prepared_20260903_r1")
STANDARDISED = Path(
    "${PRIVATE_ARCHIVE_ROOT}/CancerLncAtlas/results/v3_state_formal_runs/"
    "V3STATE-formal500-20260810T101223Z-a8803782dcf0_server_handoff/run/assets/standardized"
)
OUT = WORK / "inputs" / "encode"
OUT.mkdir(parents=True, exist_ok=True)
MAN = WORK / "manifests"
MAN.mkdir(parents=True, exist_ok=True)

#: symbol -> (ensembl id, chromosome, start, end) on GRCh38.
GRCH38_TRUTH = {
    "NEAT1": ("ENSG00000245532", "chr11", 65_422_774, 65_445_540),
    "XIST": ("ENSG00000229807", "chrX", 73_820_649, 73_852_753),
    "MALAT1": ("ENSG00000251562", "chr11", 65_497_688, 65_506_516),
}
#: GENCODE occasionally extends a 3' end, so the end coordinate is allowed to
#: differ by more than the start.  hg19 is ~230 kb away on these loci, so a
#: generous tolerance still separates the two assemblies decisively.
START_TOLERANCE = 5_000
END_TOLERANCE = 20_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


print("=== the lncRNA node universe ===", flush=True)
universe = pd.read_parquet(PREPARED / "FORMAL_CANDIDATE_UNIVERSE.parquet", columns=["lncrna_id"])
nodes = sorted(universe.lncrna_id.astype(str).unique())
print(f"  distinct lncRNA nodes: {len(nodes):,}")

print("\n=== the project's own coordinate asset ===", flush=True)
dim = pd.read_parquet(STANDARDISED / "dim_lncRNA.parquet")
dim["_ens"] = dim.ensembl_gene_id.astype(str).str.split(".").str[0]
print(f"  dim_lncRNA rows: {len(dim):,}")
print(f"  annotation_release: {dim.annotation_release.astype(str).value_counts().to_dict()}")

print("\n=== verifying the assembly against known GRCh38 loci ===", flush=True)
checks = []
for symbol, (ens, chrom, start, end) in GRCH38_TRUTH.items():
    hit = dim.loc[dim._ens.eq(ens)]
    if hit.empty:
        raise SystemExit(f"FAIL-CLOSED: {symbol} ({ens}) is absent from dim_lncRNA")
    row = hit.iloc[0]
    row_chrom = str(row.chromosome)
    row_chrom = row_chrom if row_chrom.startswith("chr") else f"chr{row_chrom}"
    observed = (row_chrom, int(row.start), int(row.end))
    d_start = abs(observed[1] - start)
    d_end = abs(observed[2] - end)
    ok = row_chrom == chrom and d_start <= START_TOLERANCE and d_end <= END_TOLERANCE
    checks.append({"symbol": symbol, "ensembl_gene_id": ens, "observed": list(observed),
                   "expected_GRCh38": [chrom, start, end],
                   "delta_start": d_start, "delta_end": d_end, "verdict": "GRCh38" if ok else "NOT_GRCh38"})
    print(f"  {symbol:8} {observed[0]} {observed[1]:,}-{observed[2]:,}  "
          f"delta=({d_start:,},{d_end:,})  -> {checks[-1]['verdict']}")

if any(c["verdict"] != "GRCh38" for c in checks):
    raise SystemExit("FAIL-CLOSED: dim_lncRNA does not carry GRCh38 coordinates on the probe loci")

print("\n=== projecting the node universe onto coordinates ===", flush=True)
by_ens = dim.drop_duplicates("_ens").set_index("_ens")
frame = pd.DataFrame({"lncrna_id": nodes})
frame["ensembl_gene_id"] = frame.lncrna_id.str.split(":", n=1).str[1].str.split(".").str[0]
joined = frame.join(by_ens[["chromosome", "start", "end", "strand", "gene_type",
                            "annotation_release", "gene_symbol"]], on="ensembl_gene_id")

unmapped = joined.loc[joined.chromosome.isna(), "lncrna_id"].tolist()
print(f"  nodes with coordinates : {int(joined.chromosome.notna().sum()):,} / {len(joined):,}")
print(f"  nodes without          : {len(unmapped):,}")
if unmapped:
    raise SystemExit(
        "FAIL-CLOSED: {} lncRNA nodes have no coordinate authority, e.g. {}".format(
            len(unmapped), unmapped[:5])
    )

coords = joined.copy()
coords["chromosome"] = coords.chromosome.astype(str).map(
    lambda c: c if c.startswith("chr") else f"chr{c}")
coords["start"] = coords.start.astype("int64")
coords["end"] = coords.end.astype("int64")
coords["assembly"] = "GRCh38"
coords["coordinate_source"] = "dim_lncRNA.parquet"
coords = coords.sort_values("lncrna_id").reset_index(drop=True)

# A locus must be a sane half-open interval before anything is overlapped with it.
bad = coords.loc[~(coords.end > coords.start) | coords.chromosome.eq("chrnan")]
if len(bad):
    raise SystemExit(f"FAIL-CLOSED: {len(bad)} loci are not valid intervals: {bad.lncrna_id.head(3).tolist()}")

path = OUT / "LNCRNA_COORDINATE_AUTHORITY.parquet"
coords.to_parquet(path, index=False)
print(f"\nwritten: {path}  ({len(coords):,} loci)")

receipt = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "purpose": "GRCh38 interval authority for overlapping ENCODE eCLIP peaks with lncRNA nodes",
    "node_universe_source": str(PREPARED / "FORMAL_CANDIDATE_UNIVERSE.parquet"),
    "node_universe_sha256": sha256_file(PREPARED / "FORMAL_CANDIDATE_UNIVERSE.parquet"),
    "coordinate_source": str(STANDARDISED / "dim_lncRNA.parquet"),
    "coordinate_source_sha256": sha256_file(STANDARDISED / "dim_lncRNA.parquet"),
    "coordinate_source_annotation_release": dim.annotation_release.astype(str).value_counts().to_dict(),
    "assembly": "GRCh38",
    "assembly_verification": checks,
    "assembly_verified": True,
    "nodes": len(coords),
    "nodes_without_coordinates": 0,
    "chromosomes": sorted(coords.chromosome.unique().tolist()),
    "output": str(path),
    "output_sha256": sha256_file(path),
    "external_annotation_introduced": False,
    "note": (
        "Coordinates come from the project's own standardised dim_lncRNA asset so "
        "the LNC:ENSG identifier space cannot drift. The assembly is verified on "
        "three probe loci and the script fails closed if the table is not GRCh38."
    ),
}
receipt_path = MAN / "LNCRNA_COORDINATE_AUTHORITY.json"
receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
print(f"written: {receipt_path}")
print(f"coordinate authority sha256 = {receipt['output_sha256']}")
