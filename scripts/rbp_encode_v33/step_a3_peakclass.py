"""Confirm which file class is the reproducible peak set, by counting real peaks.

The structure says each experiment ships three peak files per assembly: replicate 1,
replicate 2, and a combined call.  If the combined call is the replicate-concordant
reproducible set it must be markedly smaller than either single-replicate call.  The
files are already on disk, so this is measured rather than assumed.
"""

from __future__ import annotations

import gzip
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

WORK = Path("${PRIVATE_WORK_ROOT}/CancerLncAtlas/rbp_encode_v33_20260921_r1")
ENC = WORK / "inputs" / "encode"
RAW = ENC / "raw"
MAN = WORK / "manifests"


def count_peaks(path: Path) -> int:
    with gzip.open(path, "rt") as handle:
        return sum(1 for line in handle if line.strip() and not line.startswith("#"))


payload = json.loads((RAW / "ENCODE_SEARCH_ECLIP_PREFERRED_DEFAULT.json").read_text(encoding="utf-8"))
rows = []
for record in payload["@graph"]:
    dataset = str(record.get("dataset") or "")
    bio = ",".join(str(x) for x in (record.get("biological_replicates") or []))
    rows.append({
        "file_accession": record.get("accession"),
        "experiment": dataset.rstrip("/").rsplit("/", 1)[-1],
        "assembly": record.get("assembly"),
        "preferred_default": bool(record.get("preferred_default")),
        "bio": bio,
        "peak_class": "combined" if bio == "1,2" else f"replicate_{bio}",
        "rbp": (record.get("target") or {}).get("label"),
        "biosample": (record.get("biosample_ontology") or {}).get("term_name"),
        "file_size": record.get("file_size"),
    })
frame = pd.DataFrame(rows)


def peaks_for(row) -> int:
    path = (ENC / "lifted" / f"{row.file_accession}.GRCh38.narrowPeak.gz"
            if row.assembly == "hg19" else ENC / "files" / f"{row.file_accession}.bed.gz")
    if not path.exists():
        return -1
    return count_peaks(path)


print(f"=== counting peaks in {len(frame):,} files ===", flush=True)
with ThreadPoolExecutor(max_workers=16) as pool:
    frame["n_peaks"] = list(pool.map(peaks_for, frame.itertuples(index=False)))
missing = int(frame.n_peaks.lt(0).sum())
if missing:
    raise SystemExit(f"FAIL-CLOSED: {missing} files are not on disk")

print("\n=== peaks by file class ===")
summary = frame.groupby("peak_class").agg(
    files=("file_accession", "size"),
    total_peaks=("n_peaks", "sum"),
    median_peaks=("n_peaks", "median"),
    mean_peaks=("n_peaks", "mean"),
    median_bytes=("file_size", "median"),
)
print(summary.to_string())

print("\n=== within-experiment comparison (GRCh38 only) ===")
g = frame.loc[frame.assembly.eq("GRCh38")]
wide = g.pivot_table(index="experiment", columns="peak_class", values="n_peaks", aggfunc="first")
wide = wide.dropna()
print(f"  experiments compared: {len(wide):,}")
for column in ("replicate_1", "replicate_2", "combined"):
    if column in wide:
        print(f"    median peaks {column:14}: {wide[column].median():>8,.0f}")
if {"replicate_1", "combined"} <= set(wide.columns):
    ratio = (wide.combined / wide[["replicate_1", "replicate_2"]].max(axis=1))
    print(f"    combined / max(single replicate)  : median {ratio.median():.3f}, "
          f"mean {ratio.mean():.3f}")
    print(f"    experiments where combined is smaller: "
          f"{int((ratio < 1).sum()):,} / {len(ratio):,}")

print("\n=== is preferred_default the combined file? ===")
print(pd.crosstab(frame.peak_class, frame.preferred_default).to_string())

print("\n=== which class to use as the reproducible set ===")
combined = frame.loc[frame.peak_class.eq("combined")]
print(f"  combined files        : {len(combined):,}")
print(f"  total peaks           : {combined.n_peaks.sum():,}")
print(f"  RBPs covered          : {combined.rbp.nunique()} / {frame.rbp.nunique()}")
print(f"  biosamples            : {sorted(combined.biosample.unique())}")
print(f"  assemblies            : {combined.assembly.value_counts().to_dict()}")
allp = int(frame.n_peaks.sum())
print(f"\n  all files total peaks : {allp:,}")
print(f"  redundancy removed    : {allp / max(int(combined.n_peaks.sum()), 1):.2f}x")

out = {
    "files": int(len(frame)),
    "by_class": summary.to_dict(),
    "combined_is_smaller_than_single_replicate": {
        "experiments_compared": int(len(wide)),
        "median_ratio": float((wide.combined / wide[["replicate_1", "replicate_2"]].max(axis=1)).median()),
        "experiments_where_smaller": int(((wide.combined / wide[["replicate_1", "replicate_2"]].max(axis=1)) < 1).sum()),
    },
    "preferred_default_is_not_the_combined_file": bool(
        pd.crosstab(frame.peak_class, frame.preferred_default).loc["combined", True] == 0),
    "decision": (
        "The reproducible peak set is the combined (bio=1,2) file per (experiment, "
        "assembly), not preferred_default. preferred_default picks one of the two "
        "single-replicate files, so using it would select a single replicate and "
        "discard the replicate-concordant call."
    ),
    "combined": {
        "files": int(len(combined)),
        "total_peaks": int(combined.n_peaks.sum()),
        "rbps": int(combined.rbp.nunique()),
        "all_rbps": int(frame.rbp.nunique()),
        "biosamples": sorted(combined.biosample.unique().tolist()),
        "assemblies": combined.assembly.value_counts().to_dict(),
    },
    "all_files_total_peaks": allp,
    "redundancy_removed": allp / max(int(combined.n_peaks.sum()), 1),
}
path = MAN / "ENCODE_PEAK_CLASS_AUDIT.json"
path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")

frame.to_csv(MAN / "ENCODE_PEAK_CLASSES.tsv", sep="\t", index=False)
print(f"\nwritten: {path}")
print(f"written: {MAN / 'ENCODE_PEAK_CLASSES.tsv'}")
