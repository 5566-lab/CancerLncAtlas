#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description="Derive gdc-client and reuse-gap manifests")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=250)
    args = parser.parse_args()
    frame = pd.read_csv(args.input, sep="\t", dtype={"file_id": str, "md5sum": str})
    selected = frame.loc[frame.selected_for_patient.astype(str).str.lower().isin(["true", "1"])].copy()
    reusable = selected.existing_reusable.astype(str).str.lower().isin(["true", "1"])
    missing = selected.loc[~reusable].copy()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    official = missing[["file_id", "file_name", "md5sum", "file_size"]].rename(
        columns={"file_id": "id", "md5sum": "md5", "file_size": "size"}
    )
    official["state"] = "released"
    official.to_csv(args.output_dir / "gdc-client.missing.manifest.txt", sep="\t", index=False)
    chunk_dir = args.output_dir / "gdc_client_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_rows = []
    for start in range(0, len(official), args.chunk_size):
        index = start // args.chunk_size + 1
        path = chunk_dir / f"manifest_{index:03d}.txt"
        part = official.iloc[start : start + args.chunk_size]
        part.to_csv(path, sep="\t", index=False)
        chunk_rows.append({"chunk": index, "path": path.name, "files": len(part), "bytes": int(part["size"].sum())})
    pd.DataFrame(chunk_rows).to_csv(chunk_dir / "CHUNK_INDEX.tsv", sep="\t", index=False)
    gap = selected.assign(existing_reusable=reusable).groupby("cancer_id", observed=True).agg(
        selected_files=("file_id", "size"),
        reusable_files=("existing_reusable", "sum"),
        total_bytes=("file_size", "sum"),
    ).reset_index()
    gap["missing_files"] = gap.selected_files - gap.reusable_files
    missing_bytes = missing.groupby("cancer_id", observed=True).file_size.sum()
    gap["missing_bytes"] = gap.cancer_id.map(missing_bytes).fillna(0).astype("int64")
    gap.to_csv(args.output_dir / "REUSE_GAP_BY_CANCER.tsv", sep="\t", index=False)
    summary = {
        "format": "CC_HHGT_V3_2_GDC_CLIENT_SEGMENT_MANIFEST_V1",
        "candidate_only": True,
        "selected_files": int(len(selected)),
        "reusable_files": int(reusable.sum()),
        "missing_files": int(len(missing)),
        "missing_bytes": int(missing.file_size.sum()),
        "chunk_size": args.chunk_size,
        "chunks": len(chunk_rows),
        "gdc_client_uuid_directory_layout_preserved": True,
        "file_case_sample_mapping": args.input.name,
    }
    (args.output_dir / "GDC_CLIENT_MANIFEST_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
