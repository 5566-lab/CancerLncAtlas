"""Phase 3 (ENCODE): turn eCLIP peaks into typed lncRNA-protein binding evidence.

Every peak - GRCh38-native or explicitly lifted from hg19 - is overlapped with
the 8,541 lncRNA loci of the node universe.  A peak that touches a locus is
evidence that the file's RBP bound that lncRNA in that biosample.

Layering rules, all of them deliberate:

* the global relation ``binds_protein_eclip`` aggregates every biosample.  This
  matches how the frozen binding layer already treats RNAInter/NPInter records,
  which are likewise cell-line specific measurements aggregated into a global
  physical-interaction layer.  It is a *binding* statement, not a cancer claim.
* a context-specific edge is emitted only where a biosample maps onto one of the
  33 cancers.  HepG2 maps to LIHC.  K562 is CML, which is not one of the 33, so
  no context edge is invented for it and it is never forced onto LAML.  adrenal
  gland is normal tissue and produces no context edge either.
* provenance is kept per edge: assembly, chain hash where lifted, source file
  accessions and peak counts, so the ENCODE contribution can be separated from
  the legacy contribution at any time.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

WORK = Path("${PRIVATE_WORK_ROOT}/CancerLncAtlas/rbp_encode_v33_20260921_r1")
ENC = WORK / "inputs" / "encode"
FILES = ENC / "files"
LIFTED = ENC / "lifted"
MAN = WORK / "manifests"
OUT = WORK / "outputs" / "phase3_encode_eclip"
OUT.mkdir(parents=True, exist_ok=True)

STANDARDISED = Path(
    "${PRIVATE_ARCHIVE_ROOT}/CancerLncAtlas/results/v3_state_formal_runs/"
    "V3STATE-formal500-20260810T101223Z-a8803782dcf0_server_handoff/run/assets/standardized"
)

#: The only biosample -> cancer mapping the project's own dim_cancer supports.
#: K562 (CML) and "adrenal gland" (normal tissue) are deliberately absent.
BIOSAMPLE_TO_CANCER = {"HepG2": "LIHC"}

BED_COLUMNS = ["chrom", "start", "end", "name", "score", "strand",
               "signal_value", "p_value", "q_value", "peak_offset"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


print("=== authorities ===", flush=True)
coords = pd.read_parquet(ENC / "LNCRNA_COORDINATE_AUTHORITY.parquet")
coords_receipt = json.loads((MAN / "LNCRNA_COORDINATE_AUTHORITY.json").read_text(encoding="utf-8"))
if coords_receipt["assembly"] != "GRCh38":
    raise SystemExit("FAIL-CLOSED: the coordinate authority is not GRCh38")
print(f"  lncRNA loci : {len(coords):,} on {coords.chromosome.nunique()} chromosomes")

manifest = pd.read_csv(MAN / "ENCODE_FILE_MANIFEST.tsv", sep="\t", dtype=str).fillna("")
eclip = manifest.loc[
    manifest.in_eclip_narrowpeak_search.eq("True")
    & manifest.file_format.eq("bed")
    & manifest.status.eq("released")
].copy()
native = eclip.loc[eclip.assembly.eq("GRCh38")]
lifted = eclip.loc[eclip.assembly.eq("hg19")]
print(f"  eCLIP files : {len(eclip):,}  (GRCh38-native {len(native):,}, lifted from hg19 {len(lifted):,})")

# --- RBP symbol -> protein node -------------------------------------------------
print("\n=== RBP symbol -> protein node ===", flush=True)
dim_gene = pd.read_parquet(STANDARDISED / "dim_gene.parquet")
pgm = pd.read_parquet(STANDARDISED / "protein_gene_map.parquet")
symbol_to_gene: dict[str, list[str]] = defaultdict(list)
for symbol, gene in zip(dim_gene.gene_symbol.astype(str), dim_gene.gene_id.astype(str)):
    if symbol:
        symbol_to_gene[symbol.upper()].append(gene)
gene_to_protein: dict[str, list[str]] = defaultdict(list)
for gene, protein in zip(pgm.gene_id.astype(str), pgm.protein_id.astype(str)):
    gene_to_protein[gene].append(protein)

symbols = sorted({s for s in eclip.rbp.astype(str) if s})
resolved: dict[str, list[str]] = {}
unresolved: list[str] = []
for symbol in symbols:
    genes = symbol_to_gene.get(symbol.upper(), [])
    proteins = sorted({p for g in genes for p in gene_to_protein.get(g, [])})
    if proteins:
        resolved[symbol] = proteins
    else:
        unresolved.append(symbol)
n_pairs = sum(len(v) for v in resolved.values())
print(f"  RBP symbols in the eCLIP resource : {len(symbols):,}")
print(f"  resolved to at least one protein  : {len(resolved):,}")
print(f"  unresolved                        : {len(unresolved):,} {unresolved[:8]}")
print(f"  symbol -> protein pairs           : {n_pairs:,}")
if not resolved:
    raise SystemExit("FAIL-CLOSED: no eCLIP RBP symbol maps onto a protein node")

# --- overlap --------------------------------------------------------------------
print("\n=== overlapping peaks with lncRNA loci ===", flush=True)
coords = coords.sort_values(["chromosome", "start"]).reset_index(drop=True)
per_chrom: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
for chrom, block in coords.groupby("chromosome", sort=False):
    per_chrom[str(chrom)] = (
        block.start.to_numpy(dtype=np.int64),
        block.end.to_numpy(dtype=np.int64),
        block.index.to_numpy(dtype=np.int64),
    )


def read_peaks(path: Path, lifted_file: bool) -> pd.DataFrame:
    if lifted_file:
        frame = pd.read_csv(path, sep="\t", header=None, comment="#",
                            names=BED_COLUMNS + ["source_file", "source_assembly",
                                                 "source_chrom", "source_start", "source_end",
                                                 "chain_id", "chain_strand", "chain_sha256",
                                                 "map_status"],
                            usecols=[0, 1, 2], dtype={0: str})
    else:
        frame = pd.read_csv(path, sep="\t", header=None, comment="#",
                            names=BED_COLUMNS, usecols=[0, 1, 2], dtype={0: str})
    frame.columns = ["chrom", "start", "end"]
    frame["start"] = frame.start.astype("int64")
    frame["end"] = frame.end.astype("int64")
    return frame


def overlap_counts(peaks: pd.DataFrame) -> dict[int, int]:
    """lncRNA row index -> number of peaks touching it."""
    counts: dict[int, int] = {}
    for chrom, block in peaks.groupby("chrom", sort=False):
        entry = per_chrom.get(str(chrom))
        if entry is None:
            continue
        l_start, l_end, l_index = entry
        p_start = block.start.to_numpy()
        p_end = block.end.to_numpy()
        order = np.argsort(p_start, kind="stable")
        p_start = p_start[order]
        p_end = p_end[order]
        # peaks are sorted by start: any locus starting before a peak's end is a
        # candidate, and the locus must also end after the peak's start.
        l_maxend = np.maximum.accumulate(l_end)
        hi = np.searchsorted(l_start, p_end, side="left")
        for i in range(len(p_start)):
            upto = hi[i]
            # If no locus in the prefix reaches past this peak's start, the peak
            # cannot overlap anything and the slice test can be skipped entirely.
            if upto == 0 or l_maxend[upto - 1] <= p_start[i]:
                continue
            candidates = np.flatnonzero(l_end[:upto] > p_start[i])
            for c in candidates:
                key = int(l_index[c])
                counts[key] = counts.get(key, 0) + 1
    return counts


records: list[dict] = []
for label, frame, lifted_file in (("GRCh38-native", native, False), ("lifted-from-hg19", lifted, True)):
    print(f"\n--- {label}: {len(frame):,} files", flush=True)
    for counter, row in enumerate(frame.sort_values("file_accession").itertuples(index=False), 1):
        if lifted_file:
            path = LIFTED / f"{row.file_accession}.GRCh38.narrowPeak.gz"
        else:
            path = FILES / f"{row.file_accession}.bed.gz"
        if not path.exists():
            raise SystemExit(f"FAIL-CLOSED: {path} is missing")
        peaks = read_peaks(path, lifted_file)
        counts = overlap_counts(peaks)
        for row_index, n_peaks in counts.items():
            records.append({
                "lncrna_id": coords.lncrna_id.iat[row_index],
                "rbp": row.rbp,
                "biosample": row.biosample,
                "file_accession": row.file_accession,
                "assembly_provenance": "native_GRCh38" if not lifted_file else "lifted_from_hg19",
                "n_peaks": int(n_peaks),
                "peaks_in_file": int(len(peaks)),
            })
        if counter % 100 == 0 or counter == len(frame):
            print(f"    {counter:>5}/{len(frame)}  lncRNA-RBP-file hits so far: {len(records):,}", flush=True)

evidence = pd.DataFrame(records)
print(f"\n=== evidence rows (lncRNA x RBP x file) : {len(evidence):,} ===")
if evidence.empty:
    raise SystemExit("FAIL-CLOSED: no eCLIP peak overlaps any lncRNA locus")

# --- expand symbols to protein nodes and aggregate -------------------------------
exploded = evidence.assign(protein_id=evidence.rbp.map(
    lambda s: resolved.get(s) or [None])).explode("protein_id")
unmapped_evidence = exploded.loc[exploded.protein_id.isna()]
exploded = exploded.dropna(subset=["protein_id"])
print(f"  evidence rows dropped for unmapped RBP symbols: {len(unmapped_evidence):,}")

group_cols = ["lncrna_id", "protein_id", "rbp", "assembly_provenance"]
aggregated = exploded.groupby(group_cols, as_index=False).agg(
    n_peaks=("n_peaks", "sum"),
    n_source_files=("file_accession", "nunique"),
    source_files=("file_accession", lambda s: ",".join(sorted(set(s)))),
    biosamples=("biosample", lambda s: ",".join(sorted(set(s)))),
    peaks_in_source_files=("peaks_in_file", "sum"),
)
aggregated["relation_type"] = "binds_protein_eclip"
aggregated["source_database"] = "ENCODE_eCLIP"
aggregated["evidence_role"] = "physical_binding"
aggregated["graph_assay_class"] = "eclip"
aggregated["assembly_provenance_detail"] = np.where(
    aggregated.assembly_provenance.eq("lifted_from_hg19"),
    "hg19 peaks lifted to GRCh38 with the pinned UCSC chain",
    "GRCh38-native peaks, no coordinate change",
)
aggregated["weight"] = 1.0
aggregated = aggregated.sort_values(["lncrna_id", "protein_id", "assembly_provenance"]).reset_index(drop=True)

# --- global vs context layering --------------------------------------------------
global_edges = aggregated.groupby(["lncrna_id", "protein_id"], as_index=False).agg(
    n_peaks=("n_peaks", "sum"),
    n_source_files=("n_source_files", "sum"),
    source_files=("source_files", lambda s: ",".join(sorted({f for x in s for f in x.split(",") if f}))),
    biosamples=("biosamples", lambda s: ",".join(sorted({b for x in s for b in x.split(",") if b}))),
)
global_edges["relation_type"] = "binds_protein_eclip"
global_edges["edge_role"] = "global_physical_binding"
global_edges["cancer_id"] = pd.NA
global_edges["is_context_specific"] = False
global_edges["source_database"] = "ENCODE_eCLIP"
global_edges["evidence_scope"] = "cell_line_agnostic_physical_binding"

context_frames = []
for biosample, cancer in BIOSAMPLE_TO_CANCER.items():
    sub = aggregated.loc[aggregated.biosamples.str.contains(biosample, regex=False)]
    if sub.empty:
        continue
    ctx = sub.groupby(["lncrna_id", "protein_id"], as_index=False).agg(
        n_peaks=("n_peaks", "sum"),
        n_source_files=("n_source_files", "sum"),
        source_files=("source_files", lambda s: ",".join(sorted({f for x in s for f in x.split(",") if f}))),
    )
    ctx["relation_type"] = "binds_protein_eclip"
    ctx["edge_role"] = "context_physical_binding"
    ctx["cancer_id"] = cancer
    ctx["is_context_specific"] = True
    ctx["source_database"] = "ENCODE_eCLIP"
    ctx["biosample"] = biosample
    ctx["evidence_scope"] = f"{biosample} eCLIP mapped to {cancer}"
    context_frames.append(ctx)

context_edges = (pd.concat(context_frames, ignore_index=True) if context_frames
                 else pd.DataFrame(columns=list(global_edges.columns)))
unmapped_biosamples = sorted({b for s in aggregated.biosamples for b in s.split(",")
                              if b and b not in BIOSAMPLE_TO_CANCER})

print(f"\n=== edges ===")
print(f"  global   (binds_protein_eclip)      : {len(global_edges):,}")
print(f"  context  (binds_protein_eclip,LIHC) : {len(context_edges):,}")
print(f"  lncRNAs touched                     : {global_edges.lncrna_id.nunique():,}")
print(f"  protein nodes touched               : {global_edges.protein_id.nunique():,}")
print(f"  biosamples with no cancer mapping   : {unmapped_biosamples}")

global_path = OUT / "ENCODE_ECLIP_LNCRNA_RBP_GLOBAL.parquet"
context_path = OUT / "ENCODE_ECLIP_LNCRNA_RBP_CONTEXT.parquet"
evidence_path = OUT / "ENCODE_ECLIP_EVIDENCE_LONG.parquet"
aggregated.to_parquet(evidence_path, index=False)
global_edges.to_parquet(global_path, index=False)
context_edges.to_parquet(context_path, index=False)

receipt = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "phase": "Phase 3 (ENCODE eCLIP) - typed lncRNA-protein binding evidence",
    "coordinate_authority": str(ENC / "LNCRNA_COORDINATE_AUTHORITY.parquet"),
    "coordinate_authority_sha256": coords_receipt["output_sha256"],
    "manifest_sha256": sha256_file(MAN / "ENCODE_FILE_MANIFEST.tsv"),
    "eclip_files": {"total": len(eclip), "native_GRCh38": len(native), "lifted_from_hg19": len(lifted)},
    "rbp_symbols": len(symbols),
    "rbp_symbols_resolved": len(resolved),
    "rbp_symbols_unresolved": unresolved,
    "rbp_symbol_to_protein_pairs": n_pairs,
    "evidence_rows_lncrna_rbp_file": len(evidence),
    "evidence_rows_without_protein": len(unmapped_evidence),
    "global_edges": len(global_edges),
    "context_edges": len(context_edges),
    "lncrnas_touched": int(global_edges.lncrna_id.nunique()),
    "proteins_touched": int(global_edges.protein_id.nunique()),
    "biosample_to_cancer": BIOSAMPLE_TO_CANCER,
    "biosamples_without_cancer_mapping": unmapped_biosamples,
    "relation_type": "binds_protein_eclip",
    "graph_assay_class": "eclip",
    "node_type_created": False,
    "knockdown_used_as_ground_truth": False,
    "hepg2_broadcast_to_33_cancers": False,
    "k562_forced_to_LAML": False,
    "outputs": {
        "evidence_long": {"path": str(evidence_path), "sha256": sha256_file(evidence_path)},
        "global": {"path": str(global_path), "sha256": sha256_file(global_path)},
        "context": {"path": str(context_path), "sha256": sha256_file(context_path)},
    },
}
receipt_path = OUT / "PHASE3_ENCODE_ECLIP_AUDIT.json"
receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
print(f"\nwritten: {evidence_path}")
print(f"written: {global_path}")
print(f"written: {context_path}")
print(f"written: {receipt_path}")
