#!/usr/bin/env python3
"""Validate a fully written UVM staging payload and emit the missing receipt."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import h5py
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    h5_path = root / "raw_feature_bc_matrix.h5"
    metadata_path = root / "cell_metadata.tsv.gz"
    audit_path = root / "AUDIT.json"
    success_path = root / "SUCCESS.json"
    if success_path.exists():
        raise RuntimeError("immutable SUCCESS already exists")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS" or audit.get("independent_h5_reopen") != "PASS":
        raise RuntimeError("partial UVM audit did not reach the completed validation state")
    with h5py.File(h5_path, "r") as handle:
        shape = tuple(int(value) for value in handle["matrix/shape"][:])
        cells = int(handle["matrix/barcodes"].shape[0])
        features = int(handle["matrix/features/id"].shape[0])
        nonzero = int(handle["matrix/data"].shape[0])
        if shape != (features, cells) or int(handle["matrix/indptr"].shape[0]) != cells + 1:
            raise RuntimeError("UVM H5 sparse-matrix closure failed")
    metadata = pd.read_csv(metadata_path, sep="\t", low_memory=False)
    required = {"cell_id", "patient_id", "cancer_id", "cell_type_major"}
    if required - set(metadata.columns):
        raise RuntimeError("UVM metadata schema is incomplete")
    if len(metadata) != cells or metadata.cell_id.astype(str).duplicated().any():
        raise RuntimeError("UVM metadata/H5 cell closure failed")
    if set(metadata.cancer_id.astype(str)) != {"UVM"}:
        raise RuntimeError("UVM cancer identity drift")
    observed = {
        "cells": cells,
        "features": features,
        "nonzero": nonzero,
        "patients": int(metadata.patient_id.astype(str).nunique()),
    }
    for field, value in observed.items():
        if int(audit.get(field, -1)) != value:
            raise RuntimeError(f"UVM audit {field} drift")
    payload = {
        "format": str(audit["format"]),
        "status": "SUCCESS_RESCUE_SOURCE_MATERIALIZED",
        "audit_sha256": sha256(audit_path),
        "h5_sha256": sha256(h5_path),
        "metadata_sha256": sha256(metadata_path),
        "cells": cells,
        "features": features,
        "lncrna_features": int(audit["lncrna_features"]),
        "patients": observed["patients"],
        "recovery": "COMPLETED_PAYLOAD_COPIED_FROM_SSHFS_STAGING_AND_REHASHED_ON_LOCAL_DISK",
    }
    temporary = success_path.with_name(f".{success_path.name}.partial.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, success_path)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
