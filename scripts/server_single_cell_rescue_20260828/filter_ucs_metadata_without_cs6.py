#!/usr/bin/env python3
"""Create a typed UCS metadata staging view excluding GSM9042132/CS6."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path


EXPECTED_ROOT = Path("./data/CancerLncAtlas")
SOURCE = Path("./data/CancerLncAtlas/processed/sc_tool_input/UCS/cell_metadata.tsv.gz")
RUN_ID = "single_cell_rescue_20260828"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} {EXPECTED_ROOT}")
    root = Path(sys.argv[1]).resolve()
    if root != EXPECTED_ROOT:
        raise SystemExit(f"Refusing output root outside {EXPECTED_ROOT}: {root}")
    out_dir = root / "metadata" / RUN_ID
    manifest_dir = root / "manifests" / RUN_ID
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "UCS_GSE299623_without_GSM9042132_CS6_cell_metadata.tsv.gz"

    source_rows = excluded_rows = output_rows = 0
    source_donors: set[str] = set()
    output_donors: set[str] = set()
    with gzip.open(SOURCE, "rt", encoding="utf-8", newline="") as source, gzip.open(
        output, "wt", encoding="utf-8", newline="", compresslevel=6
    ) as target:
        reader = csv.DictReader(source, delimiter="\t")
        required = {"patient_id_raw", "sample_id_raw", "raw_cell_id", "patient_id", "cell_id"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise SystemExit(f"UCS metadata lacks fail-closed identity fields: {missing}")
        fields = list(reader.fieldnames or []) + ["rescue_exclusion_status", "rescue_analysis_role"]
        writer = csv.DictWriter(target, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in reader:
            source_rows += 1
            source_donors.add(row["patient_id"])
            signals = {
                "patient_id_raw": row["patient_id_raw"] == "CS6",
                "sample_id_raw": row["sample_id_raw"].startswith("GSM9042132"),
                "raw_cell_id": row["raw_cell_id"].startswith("GSM9042132"),
                "patient_id": row["patient_id"].endswith(":CS6"),
                "cell_id": ":GSM9042132_" in row["cell_id"],
            }
            if any(signals.values()):
                # Any partial identifier disagreement is a lineage error, not
                # a row that may be silently retained or dropped.
                if not all(signals.values()):
                    raise SystemExit(f"Inconsistent CS6 identity signals at source row {source_rows}: {signals}")
                excluded_rows += 1
                continue
            row["rescue_exclusion_status"] = "PASS_NOT_GSM9042132_CS6"
            row["rescue_analysis_role"] = "ANNOTATION_CANDIDATE_PENDING_REBUILT_MATRIX_BARCODE_REMAP"
            writer.writerow(row)
            output_rows += 1
            output_donors.add(row["patient_id"])

    if excluded_rows <= 0:
        raise SystemExit("No GSM9042132/CS6 metadata rows were excluded")
    if source_rows != excluded_rows + output_rows:
        raise SystemExit("Row accounting mismatch")
    if any("CS6" in donor or "GSM9042132" in donor for donor in output_donors):
        raise SystemExit("Excluded donor remains in staged metadata")

    gate = {
        "format": "CANCERLNCATLAS_UCS_GSE299623_CS6_METADATA_EXCLUSION_GATE_V1",
        "status": "PASS",
        "source_path": str(SOURCE),
        "source_sha256": sha256(SOURCE),
        "output_path": str(output),
        "output_sha256": sha256(output),
        "source_rows": source_rows,
        "excluded_rows": excluded_rows,
        "output_rows": output_rows,
        "source_donors": len(source_donors),
        "output_donors": len(output_donors),
        "excluded_sample_accession": "GSM9042132",
        "excluded_source_label": "CS6",
        "raw_source_deleted": False,
        "output_analysis_role": "ANNOTATION_CANDIDATE_PENDING_REBUILT_MATRIX_BARCODE_REMAP",
    }
    gate_path = manifest_dir / "UCS_GSE299623_METADATA_EXCLUSION_GATE.json"
    gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"RESULT_PATH\t{gate_path}")
    print(f"RESULT_SHA256\t{sha256(gate_path)}")
    print(f"EXCLUDED_ROWS\t{excluded_rows}")
    print(f"OUTPUT_DONORS\t{len(output_donors)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
