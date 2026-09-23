"""What does preferred_default actually select inside an experiment?

Step A showed both `preferred_default=True` and `n_biological_replicates==2` pick
exactly 477 files while sharing only 27, which is too neat to be a coincidence and
too disjoint to be the same rule.  Before either is used as the reproducible-peak
filter, the within-experiment structure has to be understood.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

WORK = Path("${PRIVATE_WORK_ROOT}/CancerLncAtlas/rbp_encode_v33_20260921_r1")
RAW = WORK / "inputs" / "encode" / "raw"

payload = json.loads((RAW / "ENCODE_SEARCH_ECLIP_PREFERRED_DEFAULT.json").read_text(encoding="utf-8"))
rows = []
for record in payload["@graph"]:
    dataset = str(record.get("dataset") or "")
    rows.append({
        "file_accession": record.get("accession"),
        "experiment": dataset.rstrip("/").rsplit("/", 1)[-1],
        "assembly": record.get("assembly"),
        "preferred_default": bool(record.get("preferred_default")),
        "n_bio": len(record.get("biological_replicates") or []),
        "bio": ",".join(str(x) for x in (record.get("biological_replicates") or [])),
        "tech": ",".join(str(x) for x in (record.get("technical_replicates") or [])),
        "rbp": (record.get("target") or {}).get("label"),
        "biosample": (record.get("biosample_ontology") or {}).get("term_name"),
        "size": record.get("file_size"),
    })
frame = pd.DataFrame(rows)
print(f"=== {len(frame):,} files across {frame.experiment.nunique():,} experiments ===")

# --- how many files does an experiment contribute? ----------------------------
per_exp = frame.groupby("experiment").agg(
    files=("file_accession", "size"),
    preferred=("preferred_default", "sum"),
    assemblies=("assembly", "nunique"),
    biosamples=("biosample", "nunique"),
)
print("\n=== files per experiment ===")
print(per_exp.files.value_counts().sort_index().to_string())
print("\n=== preferred per experiment ===")
print(per_exp.preferred.value_counts().sort_index().to_string())
print(f"\n  experiments with exactly one preferred file: "
      f"{int(per_exp.preferred.eq(1).sum()):,} / {len(per_exp):,}")

# --- is preferred a per (experiment, assembly) choice? ------------------------
per_exp_asm = frame.groupby(["experiment", "assembly"]).agg(
    files=("file_accession", "size"), preferred=("preferred_default", "sum"))
print("\n=== preferred per (experiment, assembly) ===")
print(per_exp_asm.preferred.value_counts().sort_index().to_string())
print(f"  groups: {len(per_exp_asm):,}")

# --- look at one experiment in full -------------------------------------------
print("\n=== a single experiment in full ===")
busiest = per_exp.sort_values("files", ascending=False).head(3)
for experiment in busiest.index:
    block = frame.loc[frame.experiment.eq(experiment)].sort_values(
        ["assembly", "preferred_default", "n_bio", "file_accession"])
    print(f"\n--- {experiment}  ({block.rbp.iloc[0]}, {block.biosample.iloc[0]})")
    print(block[["file_accession", "assembly", "preferred_default", "n_bio", "bio",
                 "tech", "size"]].to_string(index=False))

# --- and the replicate structure overall --------------------------------------
print("\n=== replicate structure ===")
print(pd.crosstab(frame.n_bio, frame.preferred_default, margins=True).to_string())
print("\n=== biological replicate labels seen ===")
print(frame.bio.value_counts().head(10).to_string())
print("\n=== technical replicate labels seen ===")
print(frame.tech.value_counts().head(10).to_string())

out = {
    "files": int(len(frame)),
    "experiments": int(frame.experiment.nunique()),
    "files_per_experiment": per_exp.files.value_counts().sort_index().to_dict(),
    "preferred_per_experiment": per_exp.preferred.value_counts().sort_index().to_dict(),
    "experiments_with_exactly_one_preferred": int(per_exp.preferred.eq(1).sum()),
    "preferred_per_experiment_assembly": per_exp_asm.preferred.value_counts().sort_index().to_dict(),
    "n_bio_x_preferred": pd.crosstab(frame.n_bio, frame.preferred_default).to_dict(),
    "bio_label_counts": frame.bio.value_counts().head(10).to_dict(),
}
path = WORK / "manifests" / "ENCODE_PREFERRED_DEFAULT_STRUCTURE.json"
path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
print(f"\nwritten: {path}")
