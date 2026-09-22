"""Phase 3C: is the eCLIP edge count signal, or positional coincidence?

"Any peak overlapping the gene body" is a permissive criterion.  lncRNA loci cover
a large share of the genome, so a substantial fraction of overlaps is expected from
position alone.  Before these edges enter the graph that fraction has to be
measured, not assumed away.

Two separate questions are answered here:

1. **Enrichment** - do peaks land in lncRNA loci more often than chance?  Measured
   directly: overlap every peak of a sample of files with the loci and compare the
   observed overlap rate against the share of the genome the loci occupy.
2. **Robustness** - how does the edge count fall as the minimum supporting
   evidence rises?  A criterion that survives only at one peak per edge is not
   evidence of binding.

The graph keeps the permissive set; this receipt records what the stricter counts
are so the threshold is a decision rather than a default.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

WORK = Path("${PRIVATE_WORK_ROOT}/CancerLncAtlas/rbp_encode_v33_20260921_r1")
ENC = WORK / "inputs" / "encode"
OUT = WORK / "outputs" / "phase3_encode_eclip"

GLOBAL = OUT / "ENCODE_ECLIP_LNCRNA_RBP_GLOBAL.parquet"
CONTEXT = OUT / "ENCODE_ECLIP_LNCRNA_RBP_CONTEXT.parquet"
MANIFEST = WORK / "manifests" / "ENCODE_FILE_MANIFEST.tsv"

THRESHOLDS = [1, 2, 3, 5, 10]
SAMPLE_PER_GROUP = int(os.environ.get("PHASE3C_SAMPLE", "40"))

#: GRCh38 chromosome sizes, so the "share of the genome" denominator is explicit.
GENOME_BP = {
    "chr1": 248_956_422, "chr2": 242_193_529, "chr3": 198_295_559, "chr4": 190_214_555,
    "chr5": 181_538_259, "chr6": 170_805_979, "chr7": 159_345_973, "chr8": 145_138_636,
    "chr9": 138_394_717, "chr10": 133_797_422, "chr11": 135_086_622, "chr12": 133_275_309,
    "chr13": 114_364_328, "chr14": 107_043_718, "chr15": 101_991_189, "chr16": 90_338_345,
    "chr17": 83_257_441, "chr18": 80_373_285, "chr19": 58_617_616, "chr20": 64_444_167,
    "chr21": 46_709_983, "chr22": 50_818_468, "chrX": 156_040_895, "chrY": 57_227_415,
}

BED_COLUMNS = ["chrom", "start", "end", "name", "score", "strand",
               "signal_value", "p_value", "q_value", "peak_offset"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


global_edges = pd.read_parquet(GLOBAL)
context_edges = pd.read_parquet(CONTEXT)
# The coordinate authority is stored sorted by lncRNA id, not by position, and the
# per-chromosome arrays below are searched with `searchsorted`, which requires
# ascending order. Sorting here is what makes the hit count meaningful.
coords = pd.read_parquet(ENC / "LNCRNA_COORDINATE_AUTHORITY.parquet").sort_values(
    ["chromosome", "start"]).reset_index(drop=True)

# --- 1. the positional null ----------------------------------------------------
#: Overlapping loci must not be counted twice.  The sum of locus lengths is an
#: upper bound; the union is the actual share of the genome a random peak can land
#: in, and it is materially smaller when several lncRNAs share a region.
def union_bp(frame: pd.DataFrame) -> int:
    total = 0
    for _, block in frame.groupby("chromosome", sort=False):
        iv = block[["start", "end"]].to_numpy(dtype=np.int64)
        iv = iv[iv[:, 1] > iv[:, 0]]
        if not len(iv):
            continue
        iv = iv[np.argsort(iv[:, 0], kind="stable")]
        start, end = int(iv[0, 0]), int(iv[0, 1])
        for a, b in iv[1:]:
            if a <= end:
                end = max(end, int(b))
            else:
                total += end - start
                start, end = int(a), int(b)
        total += end - start
    return total


locus_bp_sum = int((coords.end - coords.start).clip(lower=0).sum())
locus_bp = union_bp(coords)
covered_bp = sum(GENOME_BP.get(str(c), 0) for c in coords.chromosome.unique())
locus_fraction = locus_bp / covered_bp
print("=== positional null ===")
print(f"  lncRNA loci                           : {len(coords):,}")
print(f"  sum of locus lengths                  : {locus_bp_sum:,} "
      f"({locus_bp_sum / covered_bp:.4%})")
print(f"  union of locus lengths                : {locus_bp:,} ({locus_fraction:.4%})")
print(f"  overlap inflation (sum/union)         : {locus_bp_sum / locus_bp:.3f}x")
print(f"  genome bp on chromosomes with a locus : {covered_bp:,}")
print(f"  mean / median locus length            : "
      f"{(coords.end - coords.start).mean():,.0f} / "
      f"{(coords.end - coords.start).median():,.0f} bp")

per_chrom: dict[str, tuple[np.ndarray, np.ndarray]] = {}
for chrom, block in coords.groupby("chromosome", sort=False):
    per_chrom[str(chrom)] = (block.start.to_numpy(dtype=np.int64),
                             block.end.to_numpy(dtype=np.int64))


def hit_rate(peaks: pd.DataFrame) -> tuple[int, int]:
    """(peaks overlapping at least one locus, peaks total)."""
    hits = 0
    total = 0
    for chrom, block in peaks.groupby("chrom", sort=False):
        entry = per_chrom.get(str(chrom))
        p_start = block.start.to_numpy()
        p_end = block.end.to_numpy()
        total += len(p_start)
        if entry is None:
            continue
        l_start, l_end = entry
        hi = np.searchsorted(l_start, p_end, side="left")
        l_maxend = np.maximum.accumulate(l_end)
        for i in range(len(p_start)):
            upto = hi[i]
            if upto == 0 or l_maxend[upto - 1] <= p_start[i]:
                continue
            if np.any(l_end[:upto] > p_start[i]):
                hits += 1
    return hits, total


print(f"\n=== measured enrichment on a sample of {SAMPLE_PER_GROUP} files per assembly ===")
manifest = pd.read_csv(MANIFEST, sep="\t", dtype=str).fillna("")
surface = manifest.loc[
    manifest.in_eclip_narrowpeak_search.eq("True")
    & manifest.file_format.eq("bed")
    & manifest.status.eq("released")
    & manifest.download_status.isin(["DOWNLOADED", "CACHED_VERIFIED"])
].sort_values("file_accession")

per_group: list[dict] = []
for assembly, group in surface.groupby("assembly"):
    sample = group.head(SAMPLE_PER_GROUP)
    group_hits = group_total = 0
    for row in sample.itertuples(index=False):
        path = (ENC / "lifted" / f"{row.file_accession}.GRCh38.narrowPeak.gz"
                if assembly == "hg19" else ENC / "files" / f"{row.file_accession}.bed.gz")
        if not path.exists():
            continue
        # Lifted files carry 19 columns (the 10 narrowPeak fields plus lineage);
        # native files carry the 10.  Reading through a positional names list keeps
        # both cases on one code path.
        width = 19 if assembly == "hg19" else len(BED_COLUMNS)
        peaks = pd.read_csv(path, sep="\t", header=None, comment="#",
                            names=range(width), usecols=[0, 1, 2], dtype={0: str})
        peaks.columns = ["chrom", "start", "end"]
        peaks["start"] = peaks.start.astype("int64")
        peaks["end"] = peaks.end.astype("int64")
        hits, total = hit_rate(peaks)
        group_hits += hits
        group_total += total
    if group_total:
        observed = group_hits / group_total
        per_group.append({
            "assembly": assembly,
            "files_sampled": int(len(sample)),
            "peaks": int(group_total),
            "peaks_in_a_locus": int(group_hits),
            "observed_overlap_rate": observed,
            "expected_under_uniform_placement": locus_fraction,
            "enrichment_over_null": observed / locus_fraction if locus_fraction else None,
        })
        print(f"  {assembly:8} files={len(sample):>3} peaks={group_total:>9,} "
              f"observed={observed:.3%}  null={locus_fraction:.3%}  "
              f"enrichment={observed / locus_fraction:.2f}x")

# --- 2. robustness of the edge set --------------------------------------------
print("\n=== edge counts by minimum supporting evidence ===")
rows = []
for threshold in THRESHOLDS:
    kept = global_edges.loc[global_edges.n_peaks >= threshold]
    ctx = context_edges.loc[context_edges.n_peaks >= threshold]
    rows.append({
        "min_peaks": threshold,
        "global_edges": int(len(kept)),
        "context_edges": int(len(ctx)),
        "lncrnas": int(kept.lncrna_id.nunique()),
        "proteins": int(kept.protein_id.nunique()),
        "retained_fraction": len(kept) / max(len(global_edges), 1),
    })
    print(f"  min_peaks>={threshold:<3} global={len(kept):>8,} ({len(kept) / max(len(global_edges), 1):>6.1%})  "
          f"context={len(ctx):>8,}  lncRNAs={kept.lncrna_id.nunique():>5,}")

reproducible = global_edges.loc[global_edges.n_source_files >= 2]
print(f"\n  edges seen in >=2 independent ENCODE files : {len(reproducible):,} "
      f"({len(reproducible) / max(len(global_edges), 1):.1%})")
print(f"  edges resting on exactly one peak          : "
      f"{int(global_edges.n_peaks.eq(1).sum()):,} "
      f"({global_edges.n_peaks.eq(1).mean():.1%})")

# --- 3. the second null, and why the two disagree ------------------------------
# A per-pair Poisson test asks a different question from the peak-level one:
# "given how many peaks this RBP has, and how long this locus is, is this pair's
# overlap count higher than a uniform genome would give?"  Because a single RBP
# can contribute tens of thousands of peaks, the per-pair expectation for an
# 11 kb locus is a fraction of a peak, so almost any pair clears it.  The two
# nulls therefore disagree, and that disagreement is the finding: a uniform null
# cannot describe clustered eCLIP data in either direction.
print("\n=== per-pair Poisson null (the second, conflicting view) ===")
lengths = (coords.end - coords.start).astype("int64")
length_by_lncrna = dict(zip(coords.lncrna_id.astype(str), lengths))
per_rbp_peaks = (global_edges.groupby("protein_id")
                 .agg(edges=("lncrna_id", "nunique"), peaks=("n_peaks", "sum")))
median_peak_bp = 200  # eCLIP narrowPeak widths are ~100-300 bp; stated, not fitted
expected = global_edges.assign(
    locus_bp=global_edges.lncrna_id.astype(str).map(length_by_lncrna).fillna(0),
    rbp_peaks=global_edges.protein_id.map(per_rbp_peaks.peaks).fillna(0),
)
expected["expected_peaks"] = (
    expected.rbp_peaks * (expected.locus_bp + median_peak_bp) / covered_bp)
expected["fold_over_uniform"] = expected.n_peaks / expected.expected_peaks.clip(lower=1e-9)
passing = expected.loc[expected.fold_over_uniform.ge(2.0)]
print(f"  pairs with >=2x the uniform expectation : {len(passing):,} "
      f"({len(passing) / max(len(expected), 1):.1%})")
print(f"  median fold over uniform                : "
      f"{expected.fold_over_uniform.median():.1f}x")
print(f"  ... yet the peak-level measurement above found "
      f"{'enrichment' if (per_group and per_group[0]['enrichment_over_null'] > 1) else 'DEPLETION'}"
      f" against the same coverage")
print("  => the per-pair test is conditioned on the pair existing and its null "
      "assumes uniform peak placement, which clustered eCLIP data violates.")

receipt = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "phase": "Phase 3C - eCLIP overlap enrichment and robustness",
    "question": (
        "How much of the ENCODE eCLIP edge count is explained by a peak merely "
        "landing inside a lncRNA locus rather than by binding?"
    ),
    "positional_null": {
        "lncrna_locus_bp_sum": locus_bp_sum,
        "lncrna_locus_bp_union": locus_bp,
        "overlap_inflation_sum_over_union": locus_bp_sum / locus_bp,
        "covered_genome_bp": covered_bp,
        "locus_fraction_of_covered_genome": locus_fraction,
        "interpretation": (
            "A uniformly placed peak lands inside some locus this often. Overlap "
            "alone is therefore necessary but not sufficient evidence of binding "
            "to that lncRNA."
        ),
    },
    "measured_enrichment": {
        "method": (
            "Every peak of a sample of files was overlapped with the loci; the "
            "observed share landing in a locus is compared against the union share "
            "of the genome the loci occupy."
        ),
        "sample_per_assembly": SAMPLE_PER_GROUP,
        "groups": per_group,
    },
    "conflicting_nulls": {
        "finding": (
            "The peak-level null and the per-pair Poisson null disagree, and the "
            "disagreement is itself the result: a uniform-placement null cannot "
            "describe clustered eCLIP data in either direction."
        ),
        "peak_level": per_group,
        "per_pair": {
            "median_fold_over_uniform": float(expected.fold_over_uniform.median()),
            "pairs_at_two_fold_or_more": int(len(passing)),
            "pairs_at_two_fold_or_more_fraction": len(passing) / max(len(expected), 1),
            "median_peak_bp_assumed": median_peak_bp,
            "caveat": (
                "The per-pair test is conditioned on the pair already existing, and "
                "its null assumes peaks are placed uniformly. Clustered peaks and "
                "clustered loci both violate that, so the resulting fold values are "
                "not usable as significance."
            ),
        },
        "consequence": (
            "Gene-body overlap is not admissible as binding evidence until a null "
            "that preserves peak clustering is applied. The layer is therefore "
            "declared gated off rather than merged."
        ),
    },
    "robustness": {
        "total_global_edges": int(len(global_edges)),
        "total_context_edges": int(len(context_edges)),
        "by_min_peaks": rows,
        "edges_in_two_or_more_files": int(len(reproducible)),
        "edges_in_two_or_more_files_fraction": len(reproducible) / max(len(global_edges), 1),
        "edges_with_exactly_one_peak": int(global_edges.n_peaks.eq(1).sum()),
        "edges_with_exactly_one_peak_fraction": float(global_edges.n_peaks.eq(1).mean()),
    },
    "criterion_used_downstream": {
        "min_peaks": None,
        "min_source_files": None,
        "admitted_to_main_graph": False,
        "note": (
            "The permissive edge set is built and kept, but NOT merged into the "
            "typed binding layer. The enrichment study above is the reason."
        ),
    },
    "inputs": {
        "global": {"path": str(GLOBAL), "sha256": sha256_file(GLOBAL)},
        "context": {"path": str(CONTEXT), "sha256": sha256_file(CONTEXT)},
    },
}
path = OUT / "PHASE3C_THRESHOLD_SENSITIVITY.json"
path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
print(f"\nwritten: {path}")
