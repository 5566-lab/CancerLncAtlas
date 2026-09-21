#!/usr/bin/env python3
"""Convert the 149-generated formal TMM logCPM matrices to V3.2 parquet."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyreadr


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "artifacts" / "input" / "formal_lnc_logcpm"
DST = ROOT / "artifacts" / "input" / "formal_lncRNA_expression"


def main() -> int:
    DST.mkdir(parents=True, exist_ok=True)
    records = []
    for source in sorted(SRC.glob("TCGA-*_formal_logcpm.RDS")):
        cancer = source.name.split("-")[1].split("_")[0]
        matrix = pyreadr.read_r(str(source))[None]
        matrix.index = matrix.index.astype(str)
        matrix.columns = matrix.columns.astype(str)
        long = matrix.rename_axis("lncrna_id").stack(future_stack=True).rename("logcpm").reset_index()
        long = long.rename(columns={"level_1": "sample_id"})
        long["cancer_id"] = cancer
        long["patient_id"] = long.sample_id.astype(str).str.slice(0, 12)
        long = long[["cancer_id", "sample_id", "patient_id", "lncrna_id", "logcpm"]]
        target = DST / f"cancer_id={cancer}" / "part-0.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        long.to_parquet(target, index=False, compression="zstd")
        qc = {"cancer_id": cancer, "rows": int(len(long)), "lncrna_n": int(long.lncrna_id.nunique()), "sample_n": int(long.sample_id.nunique())}
        (target.parent / "QC.json").write_text(json.dumps(qc, indent=2) + "\n", encoding="utf-8")
        records.append(qc)
        print(json.dumps(qc), flush=True)
    if len(records) != 33:
        raise RuntimeError(f"Expected 33 formal cancer matrices, observed {len(records)}")
    (DST / "SUCCESS.json").write_text(json.dumps({"status": "PASS", "cancers": records}, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
