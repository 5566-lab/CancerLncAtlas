#!/usr/bin/env python3
"""Re-audit donor/cell-type replication for the five typed rescue cancers."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import h5py


EXPECTED_ROOT = Path("./data/CancerLncAtlas")
SOURCE_ROOT = Path("./data/CancerLncAtlas/processed/sc_tool_input")
RUN_ID = "single_cell_rescue_20260828"
MIN_DONORS = 5
LNCRNA_COUNTS_FROM_V32_EXPLICIT_ID_REAUDIT = {
    "KICH": 12366,
    "KIRP": 12267,
    "TGCT": 15502,
    "UCS": 53,
    "UVM": 446,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(cancer: str) -> dict[str, object]:
    metadata = SOURCE_ROOT / cancer / "cell_metadata.tsv.gz"
    h5 = SOURCE_ROOT / cancer / "raw_feature_bc_matrix.h5"
    row: dict[str, object] = {
        "cancer_id": cancer,
        "metadata_path": str(metadata),
        "h5_path": str(h5),
        "formal_min_association_observations": MIN_DONORS,
        "lncrna_features_v32_explicit_id_reaudit": LNCRNA_COUNTS_FROM_V32_EXPLICIT_ID_REAUDIT[cancer],
        "lncrna_count_authority": "V3.2_EXPLICIT_ID_REAUDIT_20260826",
    }
    if not metadata.is_file() or not h5.is_file():
        return {**row, "status": "MISSING_CURRENT_INPUT"}

    donors: set[str] = set()
    contexts: dict[str, set[str]] = defaultdict(set)
    cells = 0
    with gzip.open(metadata, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        columns = reader.fieldnames or []
        patient_col = next((x for x in ("patient_id", "donor_id", "patient_id_raw") if x in columns), None)
        celltype_col = next((x for x in ("cell_type_major", "cell_type_raw", "cell_type") if x in columns), None)
        state_col = next((x for x in ("cell_state",) if x in columns), None)
        if patient_col is None or celltype_col is None:
            return {**row, "metadata_columns": columns, "status": "DONOR_OR_CELLTYPE_COLUMN_MISSING"}
        for record in reader:
            cells += 1
            donor = (record.get(patient_col) or "").strip()
            celltype = (record.get(celltype_col) or "").strip()
            state = (record.get(state_col) or "").strip() if state_col else ""
            if not donor or not celltype:
                continue
            donors.add(donor)
            contexts[f"{celltype}||{state}"].add(donor)

    donor_counts = sorted((len(value) for value in contexts.values()), reverse=True)
    with h5py.File(h5, "r") as handle:
        shape = tuple(int(x) for x in handle["matrix/shape"][:])
    passing = sum(value >= MIN_DONORS for value in donor_counts)
    status = "PASS_HAS_CONTEXT_WITH_MINIMUM_DONOR_REPLICATION" if passing else "BLOCKED_NO_CONTEXT_WITH_MINIMUM_DONOR_REPLICATION"
    if cancer in {"UCS", "UVM"}:
        status = "BLOCKED_LOW_LNCRNA_FEATURE_UNIVERSE_REGARDLESS_OF_DONOR_GATE"
    return {
        **row,
        "metadata_sha256": sha256(metadata),
        "h5_sha256": sha256(h5),
        "metadata_cells": cells,
        "h5_features": shape[0],
        "h5_cells": shape[1],
        "donors": len(donors),
        "celltype_state_contexts": len(contexts),
        "contexts_with_at_least_5_donors": passing,
        "max_donors_in_any_context": donor_counts[0] if donor_counts else 0,
        "status": status,
    }


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} {EXPECTED_ROOT}")
    root = Path(sys.argv[1]).resolve()
    if root != EXPECTED_ROOT:
        raise SystemExit(f"Refusing output root outside {EXPECTED_ROOT}: {root}")
    rows = [audit(cancer) for cancer in ("KICH", "KIRP", "TGCT", "UCS", "UVM")]
    out = root / "manifests" / RUN_ID
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "typed_gap_context_reaudit.json"
    json_path.write_text(json.dumps({
        "format": "CANCERLNCATLAS_SINGLE_CELL_TYPED_GAP_CONTEXT_REAUDIT_V1",
        "contract_source": "cc_hhgt.v32.single_cell_partition_builder:min_association_observations=5",
        "rows": rows,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tsv_path = out / "typed_gap_context_reaudit.tsv"
    columns = sorted({key for row in rows for key in row})
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"RESULT_PATH\t{json_path}")
    print(f"RESULT_SHA256\t{sha256(json_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
