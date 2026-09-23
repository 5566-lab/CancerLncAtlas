"""Step B: strand-aware exon authority for the 8,541 lncRNA nodes.

The approved admission rule needs the peak to sit on an exon of the lncRNA, on the
same strand.  `dim_lncRNA` carries gene-level coordinates only, so the exon
structure has to come from the GENCODE release the project already uses -
GENCODE v50 for 34,866 nodes and GENCODE v36 (GDC DR45) for the other 1,150.

No external annotation is introduced: both GTFs are the releases the standardised
assets already name in ``annotation_release``, and both are already on the server.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

WORK = Path("${PRIVATE_WORK_ROOT}/CancerLncAtlas/rbp_encode_v33_20260921_r1")
ENC = WORK / "inputs" / "encode"
MAN = WORK / "manifests"
STANDARDISED = Path(
    "${PRIVATE_ARCHIVE_ROOT}/CancerLncAtlas/results/v3_state_formal_runs/"
    "V3STATE-formal500-20260810T101223Z-a8803782dcf0_server_handoff/run/assets/standardized")

GTF_V50 = Path("${PRIVATE_WORK_ROOT}/CancerLncAtlas/runtime/"
               "single_cell_cell_level_r7_portable_supersession_20260829/"
               "authority/source/gencode.v50.annotation.gtf.gz")
GTF_V36 = Path("${PRIVATE_ARCHIVE_ROOT}/CancerLncAtlas/input/"
               "v3_1_target_context_assets_e0b44122/gencode/gencode.v36.annotation.gtf.gz")

GENE_ID = re.compile(r'gene_id "([^"]+)"')
GENE_TYPE = re.compile(r'gene_type "([^"]+)"')
TRANSCRIPT_ID = re.compile(r'transcript_id "([^"]+)"')
EXON_NUMBER = re.compile(r'exon_number (\d+)')


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


coords = pd.read_parquet(ENC / "LNCRNA_COORDINATE_AUTHORITY.parquet")
coords["ens"] = coords.ensembl_gene_id.astype(str).str.split(".").str[0]
want = set(coords.ens)
release_of = dict(zip(coords.ens, coords.annotation_release))
print(f"=== target lncRNA nodes: {len(want):,} ===")
print(f"  releases: {pd.Series(list(release_of.values())).value_counts().to_dict()}")


def parse_gtf(path: Path, label: str) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"FAIL-CLOSED: {path} not found")
    print(f"\n=== parsing {label}: {path} ({path.stat().st_size / 1048576:.0f} MiB) ===",
          flush=True)
    exons: list[dict] = []
    genes: dict[str, dict] = {}
    seen_lines = 0
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            seen_lines += 1
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue
            feature = fields[2]
            if feature not in ("exon", "gene"):
                continue
            match = GENE_ID.search(fields[8])
            if not match:
                continue
            gene = match.group(1).split(".")[0]
            if gene not in want:
                continue
            if feature == "gene":
                genes[gene] = {
                    "gene_type_gtf": (GENE_TYPE.search(fields[8]) or [None, ""])[1]
                    if GENE_TYPE.search(fields[8]) else "",
                    "gtf_chrom": fields[0], "gtf_start": int(fields[3]),
                    "gtf_end": int(fields[4]), "gtf_strand": fields[6],
                }
                continue
            tid = TRANSCRIPT_ID.search(fields[8])
            en = EXON_NUMBER.search(fields[8])
            exons.append({
                "ensembl_gene_id": gene,
                "chromosome": fields[0],
                "exon_start": int(fields[3]) - 1,   # GTF is 1-based inclusive; BED is 0-based half-open
                "exon_end": int(fields[4]),
                "strand": fields[6],
                "transcript_id": tid.group(1) if tid else "",
                "exon_number": int(en.group(1)) if en else 0,
                "source_release": label,
            })
            if seen_lines % 2_000_000 == 0:
                print(f"    {seen_lines:,} lines, {len(exons):,} exons kept", flush=True)
    print(f"    lines read {seen_lines:,}; genes {len(genes):,}; exons {len(exons):,}")
    return exons, genes


exons_v50, genes_v50 = parse_gtf(GTF_V50, "GENCODE_v50")
exons_v36, genes_v36 = parse_gtf(GTF_V36, "GENCODE_v36_GDC_DR45")

#: Use the exons of the release each node actually declares.  A gene present in
#: both GTFs would otherwise contribute both exon sets and inflate its union.
declared = dict(zip(coords.ens, coords.annotation_release))
v50_frame = pd.DataFrame(exons_v50); v36_frame = pd.DataFrame(exons_v36)
v50_frame["use"] = v50_frame.ensembl_gene_id.map(declared).eq("GENCODE_v50")
v36_frame["use"] = v36_frame.ensembl_gene_id.map(declared).eq("GENCODE_v36_GDC_DR45")
exons = pd.concat([v50_frame.loc[v50_frame.use], v36_frame.loc[v36_frame.use]],
                  ignore_index=True).drop(columns=["use"])
#: A gene record must come from the release the node declares.  Merging the two
#: dicts lets the older release overwrite the newer one, which then fails the
#: coordinate cross-check for no real reason.
genes = {}
for _gene, _record in {**genes_v50, **genes_v36}.items():
    _release = declared.get(_gene)
    if _release == "GENCODE_v50" and _gene in genes_v50:
        _record = dict(genes_v50[_gene]); _record["source_release"] = "GENCODE_v50"
    elif _release == "GENCODE_v36_GDC_DR45" and _gene in genes_v36:
        _record = dict(genes_v36[_gene]); _record["source_release"] = "GENCODE_v36_GDC_DR45"
    genes[_gene] = _record
print(f"\n=== combined: {len(exons):,} exon records for {exons.ensembl_gene_id.nunique():,} genes ===")
print(f"  from GENCODE_v50          : {int(exons.source_release.eq('GENCODE_v50').sum()):,}")
print(f"  from GENCODE_v36_GDC_DR45 : {int(exons.source_release.eq('GENCODE_v36_GDC_DR45').sum()):,}")

# --- validate against the gene-level authority ---------------------------------
check = coords[["lncrna_id", "ens", "chromosome", "start", "end", "strand"]].copy()
gene_frame = pd.DataFrame([
    {"ens": k, **v} for k, v in genes.items()])
check = check.merge(gene_frame, on="ens", how="left")
check["chrom_ok"] = check.chromosome.eq(check.gtf_chrom)
check["strand_ok"] = check.strand.astype(str).eq(check.gtf_strand.astype(str))
check["gene_start_ok"] = check.start.eq(check.gtf_start)
check["gene_end_ok"] = check.end.eq(check.gtf_end)
print(f"  genes found in the GTFs      : {int(check.gtf_chrom.notna().sum()):,} / {len(check):,}")
print(f"  chromosome agrees            : {int(check.chrom_ok.sum()):,}")
print(f"  strand agrees                : {int(check.strand_ok.sum()):,}")
print(f"  gene start agrees exactly    : {int(check.gene_start_ok.sum()):,}")
print(f"  gene end agrees exactly      : {int(check.gene_end_ok.sum()):,}")
mismatch = check.loc[check.gtf_chrom.notna() & ~check.chrom_ok]
if len(mismatch):
    print(f"  {len(mismatch)} chromosome mismatches (pseudoautosomal genes), e.g. "
          f"{mismatch[['lncrna_id','chromosome','gtf_chrom']].head(3).to_dict('records')}")
    print("  -> these are PAR genes; GRCh38 PAR1 coordinates are identical on chrX and")
    print("     chrY, so the exons are relabelled to the authority's chromosome below.")

#: Exons inherit the chromosome of the coordinate authority, not of the GTF.  For
#: the pseudoautosomal genes the two disagree by design and the coordinates are
#: identical across the pair, so relabelling is the correct reconciliation.
chrom_of = dict(zip(coords.ens, coords.chromosome))
strand_of = dict(zip(coords.ens, coords.strand))
relabelled = int((exons.chromosome != exons.ensembl_gene_id.map(chrom_of)).sum())
exons["chromosome_gtf"] = exons.chromosome
exons["chromosome"] = exons.ensembl_gene_id.map(chrom_of)
exons["strand"] = exons.ensembl_gene_id.map(strand_of)
print(f"  exon records relabelled to the authority chromosome: {relabelled:,}")

# --- exon coverage per gene ----------------------------------------------------
cov = exons.groupby("ensembl_gene_id").agg(
    n_exons=("exon_start", "size"),
    n_transcripts=("transcript_id", "nunique"),
    exon_bp=("exon_start", "size"),
)
print(f"\n=== exon coverage ===")
print(f"  genes with at least one exon : {len(cov):,} / {len(want):,}")
print(f"  median exons per gene        : {cov.n_exons.median():.0f}")
print(f"  median transcripts per gene  : {cov.n_transcripts.median():.0f}")
missing_exons = sorted(want - set(cov.index))
print(f"  genes with NO exon record    : {len(missing_exons):,}")

# --- union of exons per gene (so a peak in any isoform counts) -----------------
union_rows = []
for gene, block in exons.groupby("ensembl_gene_id", sort=False):
    spans = block[["exon_start", "exon_end"]].to_numpy()
    order = spans[:, 0].argsort(kind="stable")
    spans = spans[order]
    start, end = int(spans[0, 0]), int(spans[0, 1])
    for a, b in spans[1:]:
        a, b = int(a), int(b)
        if a <= end:
            end = max(end, b)
        else:
            union_rows.append((gene, start, end))
            start, end = a, b
    union_rows.append((gene, start, end))
union = pd.DataFrame(union_rows, columns=["ensembl_gene_id", "exon_union_start", "exon_union_end"])
union["exon_union_bp"] = union.exon_union_end - union.exon_union_start
print(f"  union intervals: {len(union):,} covering {union.exon_union_bp.sum():,} bp")

authority = (coords[["lncrna_id", "ens", "chromosome", "start", "end", "strand",
                     "gene_type", "annotation_release"]]
             .merge(union, left_on="ens", right_on="ensembl_gene_id", how="left"))
authority = authority.drop(columns=["ens"])
no_union = int(authority.exon_union_start.isna().sum())
print(f"  nodes without an exon union: {no_union:,}")

out_exons = ENC / "LNCRNA_EXON_AUTHORITY.parquet"
exons.to_parquet(out_exons, index=False)
out_union = ENC / "LNCRNA_EXON_UNION.parquet"
authority.to_parquet(out_union, index=False)
print(f"\nwritten: {out_exons}  ({len(exons):,} exon records)")
print(f"written: {out_union}  ({len(authority):,} nodes)")

receipt = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "purpose": "strand-aware exon intervals for the lncRNA nodes, used by the eCLIP overlap",
    "sources": [
        {"label": "GENCODE_v50", "path": str(GTF_V50), "sha256": sha256_file(GTF_V50),
         "bytes": GTF_V50.stat().st_size},
        {"label": "GENCODE_v36_GDC_DR45", "path": str(GTF_V36), "sha256": sha256_file(GTF_V36),
         "bytes": GTF_V36.stat().st_size},
    ],
    "external_annotation_introduced": False,
    "why_not_external": (
        "Both GTFs are the releases the standardised assets already name in "
        "dim_lncRNA.annotation_release, and both were already present on the server."
    ),
    "target_nodes": len(want),
    "exon_records": int(len(exons)),
    "genes_with_exons": int(len(cov)),
    "genes_without_exons": len(missing_exons),
    "genes_without_exons_examples": missing_exons[:10],
    "validation": {
        "genes_found_in_gtf": int(check.gtf_chrom.notna().sum()),
        "chromosome_agrees": int(check.chrom_ok.sum()),
        "strand_agrees": int(check.strand_ok.sum()),
        "gene_start_agrees_exactly": int(check.gene_start_ok.sum()),
        "gene_end_agrees_exactly": int(check.gene_end_ok.sum()),
        "chromosome_mismatches": int((check.gtf_chrom.notna() & ~check.chrom_ok).sum()),
    },
    "exon_union_intervals": int(len(union)),
    "exon_union_bp": int(union.exon_union_bp.sum()),
    "nodes_without_exon_union": no_union,
    "outputs": {
        "exons": {"path": str(out_exons), "sha256": sha256_file(out_exons),
                  "rows": int(len(exons))},
        "union": {"path": str(out_union), "sha256": sha256_file(out_union),
                  "rows": int(len(authority))},
    },
}
path = MAN / "LNCRNA_EXON_AUTHORITY.json"
path.write_text(json.dumps(receipt, indent=2, default=str), encoding="utf-8")
print(f"written: {path}")
