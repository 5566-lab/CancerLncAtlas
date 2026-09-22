"""Path B step 2: lift the hg19 eCLIP peaks to GRCh38 with the pinned chain.

Silent liftOver is forbidden, so this is the explicit kind:

* the chain file is identified by SHA256 and its direction was already verified
  by chromosome *size* in Path B step 1;
* every mapped peak records the chain id, the chain strand and its original
  hg19 coordinates, so the mapping can be replayed;
* a peak is mapped only if it lies entirely inside one aligned block of one
  chain.  A peak that straddles an alignment gap is *not* silently truncated to
  the part that happens to map - it is quarantined with the reason.

Nothing is dropped.  Unmapped peaks land in a quarantine file, and the receipt
records how many were quarantined per file and why.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

WORK = Path("${PRIVATE_WORK_ROOT}/CancerLncAtlas/rbp_encode_v33_20260921_r1")
ENC = WORK / "inputs" / "encode"
FILES = ENC / "files"
LIFTED = ENC / "lifted"
QUAR = ENC / "quarantine"
LIFTED.mkdir(parents=True, exist_ok=True)
QUAR.mkdir(parents=True, exist_ok=True)
MAN = WORK / "manifests"

CHAIN = ENC.parent / "liftover" / "hg19ToHg38.over.chain.gz"
CHAIN_RECEIPT = ENC.parent / "liftover" / "CHAIN_RECEIPT.json"
MANIFEST = MAN / "ENCODE_FILE_MANIFEST.tsv"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


print("=== chain authority ===", flush=True)
receipt = json.loads(CHAIN_RECEIPT.read_text(encoding="utf-8"))
chain_sha = sha256_file(CHAIN)
print(f"  file   : {CHAIN}")
print(f"  sha256 : {chain_sha}")
if receipt.get("sha256") != chain_sha:
    raise SystemExit("FAIL-CLOSED: the chain file no longer matches CHAIN_RECEIPT.json")
if not receipt.get("direction_verified"):
    raise SystemExit("FAIL-CLOSED: the chain direction was never verified")
print(f"  direction: {receipt['direction_verified_by']}")

#: Per target chromosome: parallel arrays of aligned blocks, sorted by target
#: start.  Chain blocks never overlap in target coordinates, so one sorted array
#: per chromosome is a complete index.
print("\n=== parsing the chain file ===", flush=True)
t_start: dict[str, list[int]] = defaultdict(list)
t_end: dict[str, list[int]] = defaultdict(list)
q_chrom: dict[str, list[str]] = defaultdict(list)
q_start: dict[str, list[int]] = defaultdict(list)
q_end: dict[str, list[int]] = defaultdict(list)
q_size: dict[str, list[int]] = defaultdict(list)
t_strand: dict[str, list[str]] = defaultdict(list)
q_strand: dict[str, list[str]] = defaultdict(list)
chain_id: dict[str, list[int]] = defaultdict(list)

n_chains = n_blocks = 0
with gzip.open(CHAIN, "rt") as handle:
    header = None
    t_pos = q_pos = 0
    for line in handle:
        if line.startswith("chain "):
            parts = line.split()
            # chain score tName tSize tStrand tStart tEnd qName qSize qStrand qStart qEnd id
            header = {
                "score": int(parts[1]), "t_name": parts[2], "t_size": int(parts[3]),
                "t_strand": parts[4], "t_start": int(parts[5]), "t_end": int(parts[6]),
                "q_name": parts[7], "q_size": int(parts[8]), "q_strand": parts[9],
                "q_start": int(parts[10]), "q_end": int(parts[11]), "id": int(parts[12]),
            }
            if header["t_strand"] != "+":
                raise SystemExit("FAIL-CLOSED: a chain targets the minus strand; unsupported")
            t_pos, q_pos = header["t_start"], header["q_start"]
            n_chains += 1
            continue
        if header is None:
            continue
        parts = line.split()
        if len(parts) == 1:  # last block of the chain
            size = int(parts[0])
            t_start[header["t_name"]].append(t_pos)
            t_end[header["t_name"]].append(t_pos + size)
            q_chrom[header["t_name"]].append(header["q_name"])
            q_start[header["t_name"]].append(q_pos)
            q_end[header["t_name"]].append(q_pos + size)
            q_size[header["t_name"]].append(header["q_size"])
            t_strand[header["t_name"]].append(header["t_strand"])
            q_strand[header["t_name"]].append(header["q_strand"])
            chain_id[header["t_name"]].append(header["id"])
            n_blocks += 1
            header = None
            continue
        size, dt, dq = int(parts[0]), int(parts[1]), int(parts[2])
        t_start[header["t_name"]].append(t_pos)
        t_end[header["t_name"]].append(t_pos + size)
        q_chrom[header["t_name"]].append(header["q_name"])
        q_start[header["t_name"]].append(q_pos)
        q_end[header["t_name"]].append(q_pos + size)
        q_size[header["t_name"]].append(header["q_size"])
        t_strand[header["t_name"]].append(header["t_strand"])
        q_strand[header["t_name"]].append(header["q_strand"])
        chain_id[header["t_name"]].append(header["id"])
        n_blocks += 1
        t_pos += size + dt
        q_pos += size + dq

print(f"  chains: {n_chains:,}   aligned blocks: {n_blocks:,}")
print(f"  target chromosomes: {len(t_start):,}")

index: dict[str, dict] = {}
for chrom, starts in t_start.items():
    order = np.argsort(np.asarray(starts, dtype=np.int64), kind="stable")
    index[chrom] = {
        "t_start": np.asarray(starts, dtype=np.int64)[order],
        "t_end": np.asarray(t_end[chrom], dtype=np.int64)[order],
        "q_chrom": np.asarray(q_chrom[chrom], dtype=object)[order],
        "q_start": np.asarray(q_start[chrom], dtype=np.int64)[order],
        "q_end": np.asarray(q_end[chrom], dtype=np.int64)[order],
        "q_size": np.asarray(q_size[chrom], dtype=np.int64)[order],
        "q_strand": np.asarray(q_strand[chrom], dtype=object)[order],
        "chain_id": np.asarray(chain_id[chrom], dtype=np.int64)[order],
    }
overlaps = sum(
    1 for chrom, ix in index.items()
    if np.any(ix["t_start"][1:] < ix["t_end"][:-1])
)
if overlaps:
    raise SystemExit(f"FAIL-CLOSED: chain blocks overlap in target space on {overlaps} chromosomes")
print(f"  block overlap check: clean on all {len(index):,} target chromosomes")


def lift(chrom: np.ndarray, start: np.ndarray, end: np.ndarray) -> tuple[dict, np.ndarray]:
    """Map half-open intervals.  Returns (columns, mapped_mask).

    A peak maps only when one single aligned block contains it end to end.
    """
    n = len(start)
    out = {
        "q_chrom": np.full(n, "", dtype=object),
        "q_start": np.full(n, -1, dtype=np.int64),
        "q_end": np.full(n, -1, dtype=np.int64),
        "chain_id": np.full(n, -1, dtype=np.int64),
        "chain_strand": np.full(n, "", dtype=object),
        "reason": np.full(n, "", dtype=object),
    }
    known = np.isin(chrom, list(index))
    out["reason"][~known] = "chromosome_absent_from_chain"

    for name in pd.unique(chrom[known]):
        ix = index[str(name)]
        sel = np.flatnonzero(known & (chrom == name))
        s = start[sel]
        e = end[sel]
        pos = np.searchsorted(ix["t_start"], s, side="right") - 1
        inside = (pos >= 0)
        safe = np.where(inside, pos, 0)
        fits = inside & (s >= ix["t_start"][safe]) & (e <= ix["t_end"][safe])
        out["reason"][sel[~fits]] = "not_contained_in_single_aligned_block"
        good = sel[fits]
        if not len(good):
            continue
        p = pos[fits]
        qs = ix["q_start"][p]
        qe = ix["q_end"][p]
        strand = ix["q_strand"][p]
        plus = strand == "+"
        gs = start[good]
        ge = end[good]
        mapped_start = np.where(plus, qs + (gs - ix["t_start"][p]), qe - (ge - ix["t_start"][p]))
        mapped_end = np.where(plus, qs + (ge - ix["t_start"][p]), qe - (gs - ix["t_start"][p]))
        in_bounds = (mapped_start >= 0) & (mapped_end <= ix["q_size"][p]) & (mapped_end > mapped_start)
        out["reason"][good[~in_bounds]] = "lifted_interval_out_of_query_bounds"
        keep = good[in_bounds]
        k = in_bounds
        out["q_chrom"][keep] = ix["q_chrom"][p[k]]
        out["q_start"][keep] = mapped_start[k]
        out["q_end"][keep] = mapped_end[k]
        out["chain_id"][keep] = ix["chain_id"][p[k]]
        out["chain_strand"][keep] = strand[k]
    mapped = out["q_start"] >= 0
    return out, mapped


manifest = pd.read_csv(MANIFEST, sep="\t", dtype=str).fillna("")
hg19 = manifest.loc[
    manifest.assembly.eq("hg19")
    & manifest.assembly_policy.eq("explicit_liftover_hg19_to_GRCh38")
    & manifest.file_format.eq("bed")
    & manifest.status.eq("released")
    & manifest.in_eclip_narrowpeak_search.eq("True")
].sort_values("file_accession").reset_index(drop=True)

#: A limit exists only for smoke-testing the mapper before every file has landed.
#: It must be 0 for the real run, where a missing file is a hard failure.
LIMIT = int(os.environ.get("PATHB_LIFTOVER_LIMIT", "0"))
if LIMIT:
    present = [row.file_accession for row in hg19.itertuples(index=False)
               if (FILES / f"{row.file_accession}.bed.gz").exists()]
    hg19 = hg19.loc[hg19.file_accession.isin(present[:LIMIT])].reset_index(drop=True)
    print(f"    LIMIT={LIMIT}: smoke-testing {len(hg19)} already-downloaded files")
print(f"\n=== hg19 eCLIP files to lift: {len(hg19):,} ===", flush=True)

BED_COLUMNS = ["chrom", "start", "end", "name", "score", "strand",
               "signal_value", "p_value", "q_value", "peak_offset"]
stats = []
for counter, row in enumerate(hg19.itertuples(index=False), 1):
    source = FILES / f"{row.file_accession}.bed.gz"
    if not source.exists():
        raise SystemExit(f"FAIL-CLOSED: {source} is missing; run Phase 6B first")
    peaks = pd.read_csv(source, sep="\t", header=None, comment="#",
                        names=BED_COLUMNS, dtype={0: str})
    peaks["start"] = peaks.start.astype("int64")
    peaks["end"] = peaks.end.astype("int64")
    peaks["source_chrom"] = peaks.chrom
    peaks["source_start"] = peaks.start
    peaks["source_end"] = peaks.end

    lifted, mapped = lift(peaks.chrom.to_numpy(dtype=object),
                          peaks.start.to_numpy(), peaks.end.to_numpy())
    peaks["map_status"] = np.where(mapped, "mapped", "unmapped")
    peaks["unmapped_reason"] = lifted["reason"]

    mapped_rows = peaks.loc[mapped].copy()
    mapped_rows["chrom"] = lifted["q_chrom"][mapped]
    mapped_rows["start"] = lifted["q_start"][mapped]
    mapped_rows["end"] = lifted["q_end"][mapped]
    mapped_rows["chain_id"] = lifted["chain_id"][mapped]
    mapped_rows["chain_strand"] = lifted["chain_strand"][mapped]
    mapped_rows["source_file"] = row.file_accession
    mapped_rows["source_assembly"] = "hg19"
    mapped_rows["chain_sha256"] = chain_sha
    mapped_rows = mapped_rows.sort_values(["chrom", "start", "end", "name"])

    out_cols = BED_COLUMNS + ["source_file", "source_assembly", "source_chrom",
                              "source_start", "source_end", "chain_id", "chain_strand",
                              "chain_sha256", "map_status"]
    mapped_path = LIFTED / f"{row.file_accession}.GRCh38.narrowPeak.gz"
    mapped_rows[out_cols].to_csv(mapped_path, sep="\t", header=False, index=False,
                                 compression="gzip")

    unmapped_rows = peaks.loc[~mapped, BED_COLUMNS + ["source_chrom", "source_start",
                                                      "source_end", "unmapped_reason"]]
    quar_path = QUAR / f"{row.file_accession}.unmapped.tsv.gz"
    unmapped_rows.to_csv(quar_path, sep="\t", header=True, index=False, compression="gzip")

    stats.append({
        "file_accession": row.file_accession,
        "rbp": row.rbp, "biosample": row.biosample, "assembly": "hg19",
        "peaks_total": int(len(peaks)),
        "peaks_mapped": int(mapped.sum()),
        "peaks_unmapped": int((~mapped).sum()),
        "mapped_fraction": float(mapped.sum() / max(len(peaks), 1)),
        "unmapped_chromosome_absent": int((lifted["reason"] == "chromosome_absent_from_chain").sum()),
        "unmapped_not_in_single_block": int((lifted["reason"] == "not_contained_in_single_aligned_block").sum()),
        "unmapped_out_of_bounds": int((lifted["reason"] == "lifted_interval_out_of_query_bounds").sum()),
        "lifted_file": str(mapped_path),
        "quarantine_file": str(quar_path),
    })
    if counter % 50 == 0 or counter == len(hg19):
        total_p = sum(s["peaks_total"] for s in stats)
        total_m = sum(s["peaks_mapped"] for s in stats)
        print(f"    {counter:>4}/{len(hg19)}  peaks={total_p:,}  mapped={total_m:,} "
              f"({total_m / max(total_p, 1) * 100:.2f}%)", flush=True)

stats_frame = pd.DataFrame(stats)
stats_path = MAN / "PATHB_LIFTOVER_STATS.tsv"
stats_frame.to_csv(stats_path, sep="\t", index=False)

total_peaks = int(stats_frame.peaks_total.sum())
total_mapped = int(stats_frame.peaks_mapped.sum())
print(f"\n=== liftOver summary ===")
print(f"  files            : {len(stats_frame):,}")
print(f"  peaks            : {total_peaks:,}")
print(f"  mapped           : {total_mapped:,} ({total_mapped / max(total_peaks, 1) * 100:.2f}%)")
print(f"  quarantined      : {total_peaks - total_mapped:,}")
print(f"    chromosome absent from chain : {int(stats_frame.unmapped_chromosome_absent.sum()):,}")
print(f"    not in a single aligned block: {int(stats_frame.unmapped_not_in_single_block.sum()):,}")
print(f"    lifted outside query bounds  : {int(stats_frame.unmapped_out_of_bounds.sum()):,}")

out_receipt = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "step": "Path B step 2 - explicit hg19 -> GRCh38 liftOver of ENCODE eCLIP peaks",
    "chain_file": str(CHAIN),
    "chain_sha256": chain_sha,
    "chain_receipt": str(CHAIN_RECEIPT),
    "chain_direction_verified_by": receipt["direction_verified_by"],
    "chain_records": receipt["chain_records"],
    "aligned_blocks": n_blocks,
    "target_chromosomes": len(index),
    "criterion": (
        "A peak maps only if a single aligned block of a single chain contains it "
        "end to end and the lifted interval stays inside the query chromosome. "
        "Peaks straddling an alignment gap are quarantined, never truncated."
    ),
    "files": len(stats_frame),
    "peaks_total": total_peaks,
    "peaks_mapped": total_mapped,
    "peaks_quarantined": total_peaks - total_mapped,
    "mapped_fraction": total_mapped / max(total_peaks, 1),
    "quarantine_reasons": {
        "chromosome_absent_from_chain": int(stats_frame.unmapped_chromosome_absent.sum()),
        "not_contained_in_single_aligned_block": int(stats_frame.unmapped_not_in_single_block.sum()),
        "lifted_interval_out_of_query_bounds": int(stats_frame.unmapped_out_of_bounds.sum()),
    },
    "stats_tsv": str(stats_path),
    "stats_sha256": sha256_file(stats_path),
    "lifted_dir": str(LIFTED),
    "quarantine_dir": str(QUAR),
    "silent_liftover": False,
    "peaks_dropped": 0,
}
receipt_path = MAN / "PATHB_LIFTOVER_RECEIPT.json"
receipt_path.write_text(json.dumps(out_receipt, indent=2), encoding="utf-8")
print(f"\nwritten: {stats_path}")
print(f"written: {receipt_path}")
