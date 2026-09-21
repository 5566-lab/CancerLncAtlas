"""Build strict-SPF patient-fold adapter data from sample-level matrices."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cc_hhgt_v26.adapter_data import (  # noqa: E402
    ADAPTER_DATA,
    OOF,
    PATIENT_FOLD,
    build_cancer,
    write_contract,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cancer")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    fold_manifest = pd.read_csv(PATIENT_FOLD, sep="\t", usecols=["cancer_id"])
    cancers = sorted(fold_manifest["cancer_id"].dropna().unique())
    if args.cancer:
        cancers = [args.cancer]
    write_contract()
    audit_rows: list[dict[str, object]] = []
    for cancer in cancers:
        output = ADAPTER_DATA / f"cancer_id={cancer}" / "part-0.parquet"
        provenance_path = (
            ADAPTER_DATA / f"cancer_id={cancer}" / "fold_provenance.tsv"
        )
        if args.resume and output.exists() and provenance_path.exists():
            existing_columns = set(pq.read_schema(output).names)
            required_columns = {
                "candidate_from_strict",
                "candidate_from_patient",
                "candidate_from_evidence",
                "strict_available",
                "candidate_source",
            }
            if required_columns.issubset(existing_columns):
                print(f"[skip completed] {cancer}", flush=True)
                continue
            print(
                f"[rebuild incompatible legacy adapter data] {cancer}",
                flush=True,
            )
        print(f"[build] {cancer}", flush=True)
        frame, provenance = build_cancer(cancer)
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(output, index=False, compression="zstd")
        pd.DataFrame(provenance).to_csv(provenance_path, sep="\t", index=False)
        audit_rows.extend(provenance)
        print(f"[completed] {cancer}: {len(frame):,} rows", flush=True)
    if not args.cancer:
        all_provenance = []
        for path in ADAPTER_DATA.glob("cancer_id=*/fold_provenance.tsv"):
            all_provenance.append(pd.read_csv(path, sep="\t"))
        provenance = pd.concat(all_provenance, ignore_index=True)
        provenance.to_parquet(
            ADAPTER_DATA / "fold_provenance.parquet",
            index=False,
            compression="zstd",
        )
        status = {
            "completed_at": datetime.now().isoformat(),
            "status": "COMPLETED",
            "n_cancers": int(provenance["cancer_id"].nunique()),
            "n_patient_folds": int(len(provenance)),
            "old_pf_used": False,
            "static_spf_used": True,
        }
        (ADAPTER_DATA / "SUCCESS.json").write_text(
            json.dumps(status, indent=2), encoding="utf-8"
        )
        print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
