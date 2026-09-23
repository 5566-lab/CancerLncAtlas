"""Step F: the two shuffle controls the ablation needs.

C vs B answers "does ENCODE help".  It cannot answer "is the RBP assignment
carrying the signal", because a shuffled layer has the same number of edges, the
same degree distribution, the same relation type and the same biosample mix.  Only
the pairing changes.  Two pairings are worth breaking:

shuffle_A - permute the RBP label across files
    Every file keeps its peaks exactly where they are; only the identity of the
    protein it is attributed to changes.  If C is no better than C-shuffle_A, the
    layer is saying "this locus has peaks", not "this RBP binds this lncRNA".
    This needs no re-overlap: the evidence table already records which file
    contributed which peaks, so relabelling the file is enough.

shuffle_B - permute peak positions within a chromosome
    Every file keeps its peak count and its length distribution; the intervals move.
    If C is no better than C-shuffle_B, position carries no information beyond
    "this lncRNA's locus is covered".  This does need a re-overlap.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

WORK = Path("${PRIVATE_WORK_ROOT}/CancerLncAtlas/rbp_encode_v33_20260921_r1")
ENC = WORK / "inputs" / "encode"
FILES = ENC / "files"
MAN = WORK / "manifests"
SRC = WORK / "outputs" / "phase3_encode_eclip_v2"
OUT = WORK / "outputs" / "phase3_encode_eclip_v2_shuffle"
OUT.mkdir(parents=True, exist_ok=True)
SHUF_DIR = ENC / "shuffled"
SHUF_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20260726
BED_COLUMNS = ["chrom", "start", "end", "name", "score", "strand",
               "signal_value", "p_value", "q_value", "peak_offset"]
GENOME_BP = {
    "chr1": 248_956_422, "chr2": 242_193_529, "chr3": 198_295_559, "chr4": 190_214_555,
    "chr5": 181_538_259, "chr6": 170_805_979, "chr7": 159_345_973, "chr8": 145_138_636,
    "chr9": 138_394_717, "chr10": 133_797_422, "chr11": 135_086_622, "chr12": 133_275_309,
    "chr13": 114_364_328, "chr14": 107_043_718, "chr15": 101_991_189, "chr16": 90_338_345,
    "chr17": 83_257_441, "chr18": 80_373_285, "chr19": 58_617_616, "chr20": 64_444_167,
    "chr21": 46_709_983, "chr22": 50_818_468, "chrX": 156_040_895, "chrY": 57_227_415,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


evidence = pd.read_parquet(SRC / "ENCODE_V2_EVIDENCE.parquet")
peaks_meta = pd.read_csv(MAN / "ENCODE_PEAK_CLASSES.tsv", sep="\t")
reproducible = peaks_meta.loc[peaks_meta.peak_class.eq("combined")].copy()
exon_union = pd.read_parquet(ENC / "LNCRNA_EXON_UNION.parquet")
covered_bp = sum(GENOME_BP.get(str(c), 0) for c in exon_union.chromosome.unique())
exon_fraction = int(exon_union.exon_union_bp.sum()) / covered_bp
print(f"=== source layer ===")
print(f"  evidence rows {len(evidence):,}   files {evidence.file_accession.nunique():,}")

# ------------------------------------------------------------------ shuffle A
print("\n=== shuffle A: permute the RBP label across files ===")
rng = np.random.default_rng(SEED)
file_table = evidence[["file_accession", "assembly", "biosample", "rbp"]].drop_duplicates()
print(f"  files: {len(file_table):,}")
shuffled = file_table.copy()
#: Permute inside each (assembly, biosample) stratum so the marginal RBP counts and
#: the biosample composition are preserved exactly.  A global permutation would move
#: RBPs between biosamples, which is a different experiment.
strata = 0
for (assembly, biosample), block in file_table.groupby(["assembly", "biosample"]):
    idx = block.index.to_numpy()
    if len(idx) < 2:
        continue
    shuffled.loc[idx, "rbp"] = rng.permutation(block.rbp.to_numpy())
    strata += 1
changed = int((shuffled.rbp.to_numpy() != file_table.rbp.to_numpy()).sum())
print(f"  strata permuted            : {strata}")
print(f"  files whose RBP changed    : {changed:,} / {len(file_table):,} ({changed / len(file_table):.1%})")
print(f"  RBP marginal preserved     : "
      f"{sorted(shuffled.rbp.value_counts().items()) == sorted(file_table.rbp.value_counts().items())}")

evidence_a = evidence.drop(columns=["rbp"]).merge(
    shuffled[["file_accession", "rbp"]], on="file_accession", how="left")
evidence_a.to_parquet(OUT / "SHUFFLE_A_EVIDENCE.parquet", index=False)

# ------------------------------------------------------------------ shuffle B
print("\n=== shuffle B: permute peak positions within a chromosome ===")
#: Exons are where the signal lives, so shuffling must preserve the per-chromosome
#: peak count and the peak-length distribution while moving the positions.  Within a
#: chromosome the starts are permuted among the peaks, keeping lengths attached to
#: their own peak.
exon_by_chrom = {}
for chrom, block in exon_union.groupby("chromosome", sort=False):
    block = block.sort_values("exon_union_start")
    exon_by_chrom[str(chrom)] = (
        block.exon_union_start.to_numpy(dtype=np.int64),
        block.exon_union_end.to_numpy(dtype=np.int64),
        block.lncrna_id.to_numpy(dtype=object),
        np.maximum.accumulate(block.exon_union_end.to_numpy(dtype=np.int64)),
    )

rng_b = np.random.default_rng(SEED + 1)
rows = []
diag = []
for counter, row in enumerate(reproducible.sort_values("file_accession").itertuples(index=False), 1):
    if row.assembly == "hg19":
        peaks = pd.read_parquet(ENC / "lifted_reproducible" / f"{row.file_accession}.parquet")
    else:
        peaks = pd.read_csv(FILES / f"{row.file_accession}.bed.gz", sep="\t", header=None,
                            comment="#", names=BED_COLUMNS, dtype={0: str})
    peaks = peaks.loc[peaks.chrom.isin(exon_by_chrom.keys())].reset_index(drop=True)
    lengths = (peaks.end - peaks.start).to_numpy()
    new_start = peaks.start.to_numpy().copy()
    for chrom, block in peaks.groupby("chrom", sort=False):
        idx = block.index.to_numpy()
        if len(idx) < 2:
            continue
        #: Sample positions uniformly from the covered part of the chromosome, then
        #: keep the file's own length distribution by pairing sorted starts with the
        #: original lengths.
        limit = GENOME_BP.get(str(chrom), 0)
        draws = rng_b.integers(0, max(limit - int(lengths[idx].max()) - 1, 1), size=len(idx))
        new_start[idx] = draws
    total = len(peaks)
    hit_idx = set()
    counts: dict[tuple, int] = {}
    for chrom, block in peaks.groupby("chrom", sort=False):
        table = exon_by_chrom.get(str(chrom))
        if table is None:
            continue
        l_start, l_end, l_id, l_maxend = table
        idx = block.index.to_numpy()
        starts = new_start[idx]
        ends = starts + lengths[idx]
        order = np.argsort(starts, kind="stable")
        starts = starts[order]; ends = ends[order]; idx = idx[order]
        hi = np.searchsorted(l_start, ends, side="left")
        for i in range(len(starts)):
            upto = hi[i]
            if upto == 0 or l_maxend[upto - 1] <= starts[i]:
                continue
            for c in np.flatnonzero(l_end[:upto] > starts[i]):
                key = str(l_id[c])
                counts[key] = counts.get(key, 0) + 1
                hit_idx.add(int(idx[i]))
    for lncrna, n in counts.items():
        rows.append({"lncrna_id": lncrna, "rbp": row.rbp, "file_accession": row.file_accession,
                     "n_peaks": int(n)})
    diag.append({"file_accession": row.file_accession, "assembly": row.assembly,
                 "peaks": total, "peaks_in_exon": len(hit_idx)})
    if counter % 100 == 0 or counter == len(reproducible):
        print(f"    {counter:>4}/{len(reproducible)}  rows={len(rows):,}", flush=True)

evidence_b = pd.DataFrame(rows)
diag_b = pd.DataFrame(diag)
evidence_b.to_parquet(OUT / "SHUFFLE_B_EVIDENCE.parquet", index=False)
p_b = int(diag_b.peaks.sum()); h_b = int(diag_b.peaks_in_exon.sum())
obs_b = h_b / max(p_b, 1)
print(f"  shuffled peaks {p_b:,}  in_exon {h_b:,}  observed={obs_b:.3%}  "
      f"null={exon_fraction:.3%}  enrichment={obs_b / exon_fraction:.2f}x")

receipt = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "step": "Step F - shuffle controls for the C vs C-shuffle comparison",
    "seed": SEED,
    "shuffle_A": {
        "what": "RBP label permuted across files within each (assembly, biosample) stratum",
        "why": "tests whether the protein identity carries the signal",
        "strata_permuted": strata,
        "files_relabelled": changed,
        "files_total": int(len(file_table)),
        "rbp_marginal_preserved": bool(
            sorted(shuffled.rbp.value_counts().items()) == sorted(file_table.rbp.value_counts().items())),
        "peaks_moved": 0,
    },
    "shuffle_B": {
        "what": "peak starts drawn uniformly within the chromosome, lengths preserved",
        "why": "tests whether peak position carries information beyond locus coverage",
        "peaks": p_b,
        "peaks_in_exon": h_b,
        "observed_overlap": obs_b,
        "null": exon_fraction,
        "enrichment": obs_b / exon_fraction,
        "note": "this enrichment should be ~1.0; it is the null the real layer must beat",
    },
    "outputs": str(OUT),
}
path = OUT / "PHASE3_SHUFFLE_AUDIT.json"
path.write_text(json.dumps(receipt, indent=2, default=str), encoding="utf-8")
print(f"\nwritten: {path}")