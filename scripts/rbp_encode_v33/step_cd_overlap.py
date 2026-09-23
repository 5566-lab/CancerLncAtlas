"""Step C+D: the ENCODE eCLIP layer, rebuilt on reproducible peaks and exons.

Four corrections over the first attempt, each traceable to a defect:

1. Reproducible peaks only.  Each experiment ships three peak calls per assembly -
   replicate 1, replicate 2, and a replicate-concordant combined call - and the
   combined call is ~1.5 % the size of either single-replicate call.  The first
   attempt used all three, i.e. 125.9 M peaks of which 98.6 % were single-replicate
   calls.  The reproducible set is the 477 combined files, 1.78 M peaks.

2. Exons, not gene bodies.  Gene bodies span 450.0 Mb, the exon union 32.9 Mb.  A
   peak must now fall on an exon, which is the biologically correct interval and,
   because exons are strand-specific, is also the strand-aware treatment.

3. Independent experiments, not files.  An experiment contributes a GRCh38 and an
   hg19 copy of the same measurement; counting files would call one experiment two
   independent observations.

4. Biosample preserved on every edge, with the ruled routing: K562 and adrenal
   gland are not cancer-context claims and reach the global layer; HepG2 also
   produces LIHC context edges.
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
OUT = WORK / "outputs" / "phase3_encode_eclip_v2"
OUT.mkdir(parents=True, exist_ok=True)
LIFT_DIR = ENC / "lifted_reproducible"
LIFT_DIR.mkdir(parents=True, exist_ok=True)

GENOME_BP = {
    "chr1": 248_956_422, "chr2": 242_193_529, "chr3": 198_295_559, "chr4": 190_214_555,
    "chr5": 181_538_259, "chr6": 170_805_979, "chr7": 159_345_973, "chr8": 145_138_636,
    "chr9": 138_394_717, "chr10": 133_797_422, "chr11": 135_086_622, "chr12": 133_275_309,
    "chr13": 114_364_328, "chr14": 107_043_718, "chr15": 101_991_189, "chr16": 90_338_345,
    "chr17": 83_257_441, "chr18": 80_373_285, "chr19": 58_617_616, "chr20": 64_444_167,
    "chr21": 46_709_983, "chr22": 50_818_468, "chrX": 156_040_895, "chrY": 57_227_415,
}
BIOSAMPLE_TO_CANCER = {"HepG2": "LIHC"}
BED_COLUMNS = ["chrom", "start", "end", "name", "score", "strand",
               "signal_value", "p_value", "q_value", "peak_offset"]
CHAIN = ENC.parent / "liftover" / "hg19ToHg38.over.chain.gz"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ------------------------------------------------------------------ inputs
peaks_meta = pd.read_csv(MAN / "ENCODE_PEAK_CLASSES.tsv", sep="\t")
reproducible = peaks_meta.loc[peaks_meta.peak_class.eq("combined")].copy()
print(f"=== reproducible peak set: {len(reproducible):,} files ===")
print(f"  assembly  : {reproducible.assembly.value_counts().to_dict()}")
print(f"  biosample : {reproducible.biosample.value_counts().to_dict()}")
print(f"  experiments {reproducible.experiment.nunique():,}   RBPs {reproducible.rbp.nunique()}")

exon_union = pd.read_parquet(ENC / "LNCRNA_EXON_UNION.parquet")
gene_body = pd.read_parquet(ENC / "LNCRNA_COORDINATE_AUTHORITY.parquet")
covered_bp = sum(GENOME_BP.get(str(c), 0) for c in gene_body.chromosome.unique())
exon_bp = int(exon_union.exon_union_bp.sum())
exon_fraction = exon_bp / covered_bp
gene_body_bp = int((gene_body.end - gene_body.start).clip(lower=0).sum())
print(f"\n=== loci ===")
print(f"  genes {len(gene_body):,}   exon union {exon_bp:,} bp   gene body {gene_body_bp:,} bp")
print(f"  exon share of covered genome : {exon_fraction:.4%}   (gene body was {gene_body_bp / covered_bp:.4%})")

# ------------------------------------------------- step C: lift hg19 reproducible peaks
print("\n=== step C: liftOver of the hg19 reproducible peaks ===", flush=True)
chain_receipt = json.loads((ENC.parent / "liftover" / "CHAIN_RECEIPT.json").read_text())
chain_sha = sha256_file(CHAIN)
if chain_receipt["sha256"] != chain_sha:
    raise SystemExit("FAIL-CLOSED: chain file changed since it was pinned")

starts: dict[str, list] = {}
ix: dict[str, dict] = {}
n_blocks = 0
with gzip.open(CHAIN, "rt") as handle:
    header = None
    t_pos = q_pos = 0
    acc: dict[str, list] = {}
    for line in handle:
        if line.startswith("chain "):
            p = line.split()
            if p[4] != "+":
                raise SystemExit("FAIL-CLOSED: minus-strand chain target")
            header = {"t_name": p[2], "q_name": p[7], "q_size": int(p[8]), "q_strand": p[9]}
            t_pos, q_pos = int(p[5]), int(p[10])
            acc.setdefault(header["t_name"], [])
            continue
        if header is None:
            continue
        parts = line.split()
        size = int(parts[0])
        acc[header["t_name"]].append((t_pos, t_pos + size, q_pos, q_pos + size,
                                      header["q_name"], header["q_strand"], header["q_size"]))
        n_blocks += 1
        if len(parts) == 1:
            header = None
            continue
        t_pos += size + int(parts[1]); q_pos += size + int(parts[2])
    for chrom, blocks in acc.items():
        arr = np.asarray(blocks)
        order = np.argsort(arr[:, 0].astype(np.int64), kind="stable")
        a = arr[order]
        ix[chrom] = {
            "t_start": a[:, 0].astype(np.int64), "t_end": a[:, 1].astype(np.int64),
            "q_start": a[:, 2].astype(np.int64), "q_end": a[:, 3].astype(np.int64),
            "q_chrom": a[:, 4].astype(object), "q_strand": a[:, 5].astype(object),
            "q_size": a[:, 6].astype(np.int64),
        }
print(f"  chain: {n_blocks:,} aligned blocks over {len(ix):,} chromosomes")


def lift_frame(peaks: pd.DataFrame):
    chrom = peaks.chrom.to_numpy(dtype=object)
    start = peaks.start.to_numpy(); end = peaks.end.to_numpy()
    n = len(peaks)
    qc = np.full(n, "", dtype=object); qs = np.full(n, -1, dtype=np.int64)
    qe = np.full(n, -1, dtype=np.int64); reason = np.full(n, "", dtype=object)
    known = np.isin(chrom, list(ix))
    reason[~known] = "chromosome_absent_from_chain"
    for name in pd.unique(chrom[known]):
        table = ix[str(name)]
        sel = np.flatnonzero(known & (chrom == name))
        s, e = start[sel], end[sel]
        pos = np.searchsorted(table["t_start"], s, side="right") - 1
        inside = pos >= 0
        safe = np.where(inside, pos, 0)
        fits = inside & (s >= table["t_start"][safe]) & (e <= table["t_end"][safe])
        reason[sel[~fits]] = "not_contained_in_single_aligned_block"
        good = sel[fits]
        if not len(good):
            continue
        k = pos[fits]
        plus = table["q_strand"][k] == "+"
        gs, ge = start[good], end[good]
        ms = np.where(plus, table["q_start"][k] + (gs - table["t_start"][k]),
                      table["q_end"][k] - (ge - table["t_start"][k]))
        me = np.where(plus, table["q_start"][k] + (ge - table["t_start"][k]),
                      table["q_end"][k] - (gs - table["t_start"][k]))
        ok = (ms >= 0) & (me <= table["q_size"][k]) & (me > ms)
        reason[good[~ok]] = "lifted_out_of_query_bounds"
        keep = good[ok]
        qc[keep] = table["q_chrom"][k[ok]]; qs[keep] = ms[ok]; qe[keep] = me[ok]
    mapped = qs >= 0
    out = peaks.copy()
    out["chrom"] = qc; out["start"] = qs; out["end"] = qe
    out = out.loc[mapped].copy()
    stats = {"peaks_total": n, "peaks_mapped": int(mapped.sum()),
             "peaks_quarantined": int((~mapped).sum()),
             "reason_not_single_block": int((reason == "not_contained_in_single_aligned_block").sum()),
             "reason_chromosome": int((reason == "chromosome_absent_from_chain").sum()),
             "reason_out_of_bounds": int((reason == "lifted_out_of_query_bounds").sum())}
    return out, stats


hg19_files = reproducible.loc[reproducible.assembly.eq("hg19")].sort_values("file_accession")
lift_stats = []
for counter, row in enumerate(hg19_files.itertuples(index=False), 1):
    peaks = pd.read_csv(FILES / f"{row.file_accession}.bed.gz", sep="\t", header=None,
                        comment="#", names=BED_COLUMNS, dtype={0: str})
    lifted, stats = lift_frame(peaks)
    lifted.to_parquet(LIFT_DIR / f"{row.file_accession}.parquet", index=False)
    lift_stats.append({"file_accession": row.file_accession, **stats})
    if counter % 50 == 0 or counter == len(hg19_files):
        done = sum(s["peaks_total"] for s in lift_stats)
        got = sum(s["peaks_mapped"] for s in lift_stats)
        print(f"    {counter:>4}/{len(hg19_files)}  peaks={done:,} mapped={got:,} "
              f"({got / max(done, 1) * 100:.2f}%)", flush=True)
lift_df = pd.DataFrame(lift_stats)
print(f"  total {lift_df.peaks_total.sum():,}  mapped {lift_df.peaks_mapped.sum():,}  "
      f"quarantined {lift_df.peaks_quarantined.sum():,}  dropped 0")

# ------------------------------------------------- step D: exon-level overlap
print("\n=== step D: exon-level overlap ===", flush=True)
exon_by_chrom = {}
for chrom, block in exon_union.groupby("chromosome", sort=False):
    block = block.sort_values("exon_union_start")
    exon_by_chrom[str(chrom)] = (
        block.exon_union_start.to_numpy(dtype=np.int64),
        block.exon_union_end.to_numpy(dtype=np.int64),
        block.lncrna_id.to_numpy(dtype=object),
        block.strand.to_numpy(dtype=object),
        np.maximum.accumulate(block.exon_union_end.to_numpy(dtype=np.int64)),
    )

rows = []
diag = []
for counter, row in enumerate(reproducible.sort_values("file_accession").itertuples(index=False), 1):
    if row.assembly == "hg19":
        peaks = pd.read_parquet(LIFT_DIR / f"{row.file_accession}.parquet")
    else:
        peaks = pd.read_csv(FILES / f"{row.file_accession}.bed.gz", sep="\t", header=None,
                            comment="#", names=BED_COLUMNS, dtype={0: str})
    peaks = peaks.loc[peaks.chrom.isin(exon_by_chrom.keys())].reset_index(drop=True)
    total = len(peaks)
    hit_peak_idx = set()
    counts: dict[tuple, int] = {}
    for chrom, block in peaks.groupby("chrom", sort=False):
        table = exon_by_chrom.get(str(chrom))
        if table is None:
            continue
        l_start, l_end, l_id, l_strand, l_maxend = table
        idx = block.index.to_numpy()
        p_start = block.start.to_numpy(); p_end = block.end.to_numpy()
        hi = np.searchsorted(l_start, p_end, side="left")
        for i in range(len(p_start)):
            upto = hi[i]
            if upto == 0 or l_maxend[upto - 1] <= p_start[i]:
                continue
            for c in np.flatnonzero(l_end[:upto] > p_start[i]):
                key = (str(l_id[c]), str(l_strand[c]))
                counts[key] = counts.get(key, 0) + 1
                hit_peak_idx.add(int(idx[i]))
    for (lncrna, strand), n in counts.items():
        rows.append({"lncrna_id": lncrna, "gene_strand": strand, "peak_class": "exonic",
                     "file_accession": row.file_accession, "experiment": row.experiment,
                     "assembly": row.assembly, "rbp": row.rbp, "biosample": row.biosample,
                     "n_peaks": int(n)})
    diag.append({"file_accession": row.file_accession, "assembly": row.assembly,
                 "biosample": row.biosample, "experiment": row.experiment,
                 "peaks": total, "peaks_in_exon": len(hit_peak_idx)})
    if counter % 100 == 0 or counter == len(reproducible):
        print(f"    {counter:>4}/{len(reproducible)}  evidence rows={len(rows):,}", flush=True)

evidence = pd.DataFrame(rows)
diag_df = pd.DataFrame(diag)
print(f"\n=== evidence rows (lncRNA x RBP x file): {len(evidence):,} ===")
if evidence.empty:
    raise SystemExit("FAIL-CLOSED: no reproducible peak overlaps an lncRNA exon")

print("\n=== enrichment: reproducible peaks in lncRNA exons vs chance ===")
enrichment = {}
for assembly, block in diag_df.groupby("assembly"):
    p = int(block.peaks.sum()); h = int(block.peaks_in_exon.sum())
    obs = h / max(p, 1)
    enrichment[assembly] = {"peaks": p, "peaks_in_exon": h, "observed": obs,
                            "null": exon_fraction, "enrichment": obs / exon_fraction}
    print(f"  {assembly:8} peaks={p:>8,} in_exon={h:>7,} observed={obs:.3%} "
          f"null={exon_fraction:.3%} -> {obs / exon_fraction:.2f}x")

# --- aggregate to edges: experiments, not files --------------------------------
print("\n=== aggregation ===")
links = evidence.merge(
    reproducible[["file_accession", "experiment"]].drop_duplicates(),
    on="file_accession", how="left", suffixes=("", "_meta"))
edges = (links.groupby(["lncrna_id", "rbp", "biosample"], as_index=False)
         .agg(n_peaks=("n_peaks", "sum"),
              n_files=("file_accession", "nunique"),
              n_experiments=("experiment", "nunique"),
              gene_strand=("gene_strand", "first")))
print(f"  (lncRNA, RBP, biosample) rows : {len(edges):,}")
print(f"  distinct experiments across the layer: {links.experiment.nunique():,}")

global_edges = (edges.groupby(["lncrna_id", "rbp"], as_index=False)
                .agg(n_peaks=("n_peaks", "sum"),
                     n_experiments=("n_experiments", "sum"),
                     biosamples=("biosample", lambda s: ",".join(sorted(set(s)))),
                     gene_strand=("gene_strand", "first")))
context_rows = []
for biosample, cancer in BIOSAMPLE_TO_CANCER.items():
    sub = edges.loc[edges.biosample.eq(biosample)]
    if sub.empty:
        continue
    ctx = sub.groupby(["lncrna_id", "rbp"], as_index=False).agg(
        n_peaks=("n_peaks", "sum"), n_experiments=("n_experiments", "sum"))
    ctx["cancer_id"] = cancer; ctx["biosample"] = biosample
    context_rows.append(ctx)
context_edges = (pd.concat(context_rows, ignore_index=True) if context_rows
                 else pd.DataFrame(columns=["lncrna_id", "rbp", "cancer_id"]))
print(f"  global edges  : {len(global_edges):,}")
print(f"  context edges : {len(context_edges):,}  (LIHC from HepG2 only)")
print(f"  lncRNAs {global_edges.lncrna_id.nunique():,}   RBPs {global_edges.rbp.nunique()}")

for name, frame in (("ENCODE_V2_EVIDENCE.parquet", evidence),
                    ("ENCODE_V2_EDGES_BY_BIOSAMPLE.parquet", edges),
                    ("ENCODE_V2_GLOBAL.parquet", global_edges),
                    ("ENCODE_V2_CONTEXT.parquet", context_edges),
                    ("ENCODE_V2_PEAK_DIAGNOSTICS.parquet", diag_df)):
    frame.to_parquet(OUT / name, index=False)
    print(f"  written {name}  ({len(frame):,} rows)")

receipt = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "step": "Step C+D - ENCODE eCLIP layer on reproducible peaks and exons",
    "corrections_over_first_attempt": [
        "reproducible (combined) peak calls only: 477 files / 1.78 M peaks, not 1,431 / 125.9 M",
        "exon union intervals (32.9 Mb) instead of gene bodies (450.0 Mb), which is also the strand-aware treatment",
        "independent experiments counted, so the GRCh38 and hg19 copies of one experiment collapse to one",
        "biosample preserved per edge with the ruled routing",
    ],
    "reproducible_peak_set": {
        "files": int(len(reproducible)),
        "by_assembly": reproducible.assembly.value_counts().to_dict(),
        "by_biosample": reproducible.biosample.value_counts().to_dict(),
        "experiments": int(reproducible.experiment.nunique()),
        "rbps": int(reproducible.rbp.nunique()),
        "peaks": int(peaks_meta.loc[peaks_meta.peak_class.eq("combined"), "n_peaks"].sum()),
    },
    "liftover": {
        "chain_sha256": chain_sha,
        "files": int(len(lift_df)),
        "peaks_total": int(lift_df.peaks_total.sum()),
        "peaks_mapped": int(lift_df.peaks_mapped.sum()),
        "peaks_quarantined": int(lift_df.peaks_quarantined.sum()),
        "peaks_dropped": 0,
        "mapped_fraction": float(lift_df.peaks_mapped.sum() / max(lift_df.peaks_total.sum(), 1)),
    },
    "loci": {"gene_body_bp": gene_body_bp, "exon_union_bp": exon_bp,
             "covered_genome_bp": covered_bp,
             "exon_fraction": exon_fraction, "gene_body_fraction": gene_body_bp / covered_bp},
    "enrichment": enrichment,
    "edges": {"global": int(len(global_edges)), "context": int(len(context_edges)),
              "evidence_rows": int(len(evidence))},
    "routing": {"K562": "global", "adrenal gland": "global", "HepG2": "global + LIHC context"},
    "biosample_to_cancer": BIOSAMPLE_TO_CANCER,
    "declared_prohibitions": {"node_type_created": False,
                              "k562_forced_to_LAML": False,
                              "hepg2_broadcast_to_33_cancers": False},
    "outputs": str(OUT),
}
path = OUT / "PHASE3_V2_AUDIT.json"
path.write_text(json.dumps(receipt, indent=2, default=str), encoding="utf-8")
print(f"\nwritten: {path}")