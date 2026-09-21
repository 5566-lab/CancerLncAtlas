#!/usr/bin/env python3
"""Rebuild direction-preserving CNV arrays from the verified staged segments."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cc_hhgt.v32.cnv_only_training import TCGA_CANCERS
from cc_hhgt.v32.directional_cnv import (
    aggregate_pathway_directional,
    map_segments_directional,
)


FORMAT = "CC_HHGT_V3_2_DIRECTIONAL_SEGMENT_CNV_V1"
FILES = (
    "patient_ids.json", "lncrna_ids.json", "pathway_ids.json",
    "lncrna_signed_mean.npy", "lncrna_callable.npy", "lncrna_amplification.npy",
    "lncrna_deletion.npy", "lncrna_neutral.npy", "pathway_signed_mean.npy",
    "pathway_callable.npy", "pathway_amplification.npy", "pathway_deletion.npy",
    "pathway_neutral.npy", "pathway_amplification_fraction.npy",
    "pathway_deletion_fraction.npy",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalise_intervals(path: Path, needed: set[str]) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    lower = {str(column).lower(): column for column in frame.columns}
    entity_column = lower.get("entity_id") or lower.get("gene_id")
    type_column = lower.get("entity_type") or lower.get("gene_type")
    chromosome_column = lower.get("chromosome") or lower.get("chrom")
    if entity_column is None or chromosome_column is None or not {"start", "end"}.issubset(lower):
        raise RuntimeError("interval authority lacks entity/chromosome/start/end")
    selected = [entity_column, chromosome_column, lower["start"], lower["end"]]
    if type_column is not None:
        selected.append(type_column)
    result = frame[selected].copy()
    names = ["entity_id", "chromosome", "start", "end"]
    if type_column is not None:
        names.append("entity_type")
    result.columns = names
    result["entity_id"] = result.entity_id.astype(str)
    result = result.loc[result.entity_id.isin(needed)].drop_duplicates()
    return result


def segment_file_map(root: Path, cancer: str) -> dict[str, Path]:
    cancer_root = root / cancer
    if not cancer_root.is_dir():
        return {}
    mapping: dict[str, Path] = {}
    for path in sorted(item for item in cancer_root.iterdir() if item.is_file()):
        patient = path.name[:12].upper()
        if not patient.startswith("TCGA-"):
            continue
        if patient in mapping:
            raise RuntimeError(f"duplicate staged CNV file for {cancer}/{patient}")
        mapping[patient] = path
    return mapping


def open_array(root: Path, name: str, shape: tuple[int, int], dtype: Any) -> np.memmap:
    return np.lib.format.open_memmap(root / name, mode="w+", dtype=dtype, shape=shape)


def materialize_cancer(args: argparse.Namespace) -> dict[str, Any]:
    cancer = args.cancer.upper()
    if cancer not in TCGA_CANCERS:
        raise RuntimeError(f"unknown cancer: {cancer}")
    output = Path(args.output_root).resolve()
    final = output / f"cancer={cancer}"
    if final.exists():
        raise RuntimeError(f"immutable directional CNV output exists: {final}")
    old_success_path = Path(args.old_root).resolve() / "SUCCESS.json"
    old_success = load_json(old_success_path)
    old = Path(old_success["cancers"][cancer]["path"]).resolve()
    patient_ids = tuple(map(str, load_json(old / "patient_ids.json")))
    lncrna_ids = tuple(map(str, load_json(old / "lncrna_ids.json")))
    pathway_ids = tuple(map(str, load_json(old / "pathway_ids.json")))
    membership = pd.read_parquet(args.membership, columns=["pathway_id", "gene_id"])
    membership = membership.dropna().astype(str).drop_duplicates()
    membership = membership.loc[membership.pathway_id.isin(pathway_ids)]
    genes = tuple(sorted(membership.gene_id.unique()))
    gene_index = {value: position for position, value in enumerate(genes)}
    pathway_members = [
        np.asarray([
            gene_index[value] for value in membership.loc[
                membership.pathway_id.eq(pathway), "gene_id"
            ]
        ], dtype=int)
        for pathway in pathway_ids
    ]
    intervals = normalise_intervals(Path(args.intervals), set(lncrna_ids) | set(genes))
    lnc_intervals = intervals.loc[intervals.entity_id.isin(lncrna_ids)]
    gene_intervals = intervals.loc[intervals.entity_id.isin(genes)]
    staged = segment_file_map(Path(args.staging_root).resolve(), cancer)
    token = hashlib.sha256(f"{cancer}|{old}|{args.intervals}|{args.membership}".encode()).hexdigest()[:12]
    building = output / f".cancer={cancer}.building.{token}.{os.getpid()}"
    building.mkdir(parents=True)
    atomic_json(building / "INCOMPLETE.json", {"status": "BUILDING", "cancer_id": cancer})
    old_lc = np.load(old / "lncrna_callable.npy", mmap_mode="r")
    old_pc = np.load(old / "pathway_callable.npy", mmap_mode="r")
    shape_l = (len(patient_ids), len(lncrna_ids))
    shape_p = (len(patient_ids), len(pathway_ids))
    arrays = {
        "lncrna_signed_mean.npy": open_array(building, "lncrna_signed_mean.npy", shape_l, np.float32),
        "lncrna_callable.npy": open_array(building, "lncrna_callable.npy", shape_l, np.uint8),
        "lncrna_amplification.npy": open_array(building, "lncrna_amplification.npy", shape_l, np.uint8),
        "lncrna_deletion.npy": open_array(building, "lncrna_deletion.npy", shape_l, np.uint8),
        "lncrna_neutral.npy": open_array(building, "lncrna_neutral.npy", shape_l, np.uint8),
        "pathway_signed_mean.npy": open_array(building, "pathway_signed_mean.npy", shape_p, np.float32),
        "pathway_callable.npy": open_array(building, "pathway_callable.npy", shape_p, np.uint8),
        "pathway_amplification.npy": open_array(building, "pathway_amplification.npy", shape_p, np.uint8),
        "pathway_deletion.npy": open_array(building, "pathway_deletion.npy", shape_p, np.uint8),
        "pathway_neutral.npy": open_array(building, "pathway_neutral.npy", shape_p, np.uint8),
        "pathway_amplification_fraction.npy": open_array(building, "pathway_amplification_fraction.npy", shape_p, np.float32),
        "pathway_deletion_fraction.npy": open_array(building, "pathway_deletion_fraction.npy", shape_p, np.float32),
    }
    for name, value in arrays.items():
        value[:] = np.nan if np.issubdtype(value.dtype, np.floating) else 0
    processed = 0
    unavailable = 0
    local_callable_mismatch = 0
    source_records = []
    try:
        for row, patient in enumerate(patient_ids):
            source = staged.get(patient)
            if source is None:
                unavailable += 1
                if bool(np.asarray(old_lc[row], dtype=bool).any()) or bool(np.asarray(old_pc[row], dtype=bool).any()):
                    raise RuntimeError(f"old store callable but staged source missing: {cancer}/{patient}")
                continue
            raw = pd.read_csv(source, sep="\t")
            columns = {str(column).lower(): column for column in raw.columns}
            required = ("chromosome", "start", "end", "segment_mean")
            if any(column not in columns for column in required):
                raise RuntimeError(f"malformed segment source: {source}")
            segment = raw[[columns[column] for column in required]].copy()
            segment.columns = ["chromosome", "start", "end", "value"]
            lnc = map_segments_directional(segment, lnc_intervals, lncrna_ids)
            genes_result = map_segments_directional(segment, gene_intervals, genes)
            canonical_lc = np.asarray(old_lc[row], dtype=bool)
            mismatch = int(np.sum(lnc.callable != canonical_lc))
            local_callable_mismatch += mismatch
            if mismatch:
                raise RuntimeError(
                    f"directional remap changes canonical local callability: {cancer}/{patient}/{mismatch}"
                )
            pathway = aggregate_pathway_directional(
                genes_result, pathway_members,
                canonical_callable=np.asarray(old_pc[row], dtype=bool),
            )
            arrays["lncrna_signed_mean.npy"][row] = lnc.signed_mean
            arrays["lncrna_callable.npy"][row] = lnc.callable
            arrays["lncrna_amplification.npy"][row] = lnc.amplification
            arrays["lncrna_deletion.npy"][row] = lnc.deletion
            arrays["lncrna_neutral.npy"][row] = lnc.neutral
            for key in (
                "signed_mean", "callable", "amplification", "deletion", "neutral",
                "amplification_fraction", "deletion_fraction",
            ):
                arrays[f"pathway_{key}.npy"][row] = pathway[key]
            processed += 1
            source_records.append({
                "patient_id": patient, "path": str(source), "bytes": source.stat().st_size,
                "mtime_ns": source.stat().st_mtime_ns,
            })
        for value in arrays.values():
            value.flush()
        del value
        (building / "patient_ids.json").write_text(json.dumps(patient_ids) + "\n", encoding="utf-8")
        (building / "lncrna_ids.json").write_text(json.dumps(lncrna_ids) + "\n", encoding="utf-8")
        (building / "pathway_ids.json").write_text(json.dumps(pathway_ids) + "\n", encoding="utf-8")
        atomic_json(building / "SOURCE_FILES.json", {"records": source_records})
        files = {name: sha256(building / name) for name in FILES}
        receipt = {
            "format": FORMAT, "status": "SUCCESS", "cancer_id": cancer,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "patients": len(patient_ids), "processed_segment_patients": processed,
            "typed_unavailable_patients": unavailable, "lncrnas": len(lncrna_ids),
            "pathways": len(pathway_ids), "genes": len(genes),
            "event_threshold": float(args.event_threshold),
            "local_callable_mismatch_vs_canonical": local_callable_mismatch,
            "signed_values_preserved": True, "absolute_burden_used_as_continuous": False,
            "amplification_deletion_separate": True,
            "old_partition": str(old), "old_success_sha256": sha256(old / "SUCCESS.json"),
            "intervals_path": str(Path(args.intervals).resolve()),
            "intervals_sha256": sha256(Path(args.intervals)),
            "membership_path": str(Path(args.membership).resolve()),
            "membership_sha256": sha256(Path(args.membership)), "files": files,
        }
        atomic_json(building / "SUCCESS.json", receipt)
        (building / "INCOMPLETE.json").unlink()
        os.replace(building, final)
        return receipt
    except BaseException:
        atomic_json(building / "RUN_FAILED.json", {"status": "RUN_FAILED", "cancer_id": cancer})
        raise


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_root).resolve()
    success_path = root / "SUCCESS.json"
    if success_path.exists():
        raise RuntimeError(f"immutable directional CNV SUCCESS exists: {success_path}")
    records = {}
    for cancer in TCGA_CANCERS:
        partition = root / f"cancer={cancer}"
        receipt_path = partition / "SUCCESS.json"
        if not receipt_path.is_file():
            raise RuntimeError(f"missing directional partition: {cancer}")
        receipt = load_json(receipt_path)
        if receipt.get("status") != "SUCCESS" or receipt.get("cancer_id") != cancer:
            raise RuntimeError(f"invalid directional partition: {cancer}")
        records[cancer] = {
            "path": str(partition), "success_sha256": sha256(receipt_path),
            "processed_segment_patients": receipt["processed_segment_patients"],
            "typed_unavailable_patients": receipt["typed_unavailable_patients"],
        }
    payload = {
        "format": FORMAT, "status": "SUCCESS", "cancers": records,
        "cancer_count": len(records), "signed_values_preserved": True,
        "absolute_burden_used_as_continuous": False,
        "amplification_deletion_separate": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(success_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("cancer", "finalize"), required=True)
    parser.add_argument("--old-root", required=True)
    parser.add_argument("--staging-root", required=True)
    parser.add_argument("--intervals", required=True)
    parser.add_argument("--membership", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--cancer")
    parser.add_argument("--event-threshold", type=float, default=0.30)
    args = parser.parse_args()
    if args.phase == "cancer":
        if not args.cancer:
            parser.error("--cancer is required for phase=cancer")
        result = materialize_cancer(args)
    else:
        result = finalize(args)
    print(json.dumps({"status": result["status"], "phase": args.phase, "cancer": args.cancer}, sort_keys=True))


if __name__ == "__main__":
    main()
