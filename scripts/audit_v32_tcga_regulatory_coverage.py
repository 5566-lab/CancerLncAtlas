#!/usr/bin/env python3
"""Audit patient-level TCGA ATAC coverage against the frozen V3.2 cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ATAC_CANCER_ALIASES = {"ACCx": "ACC", "GBMx": "GBM", "LGGx": "LGG"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _patient_id(barcode: str) -> str:
    fields = str(barcode).strip().split("-")
    if len(fields) < 3 or fields[0] != "TCGA":
        raise RuntimeError(f"Invalid TCGA barcode: {barcode!r}")
    return "-".join(fields[:3])


def audit(mapping_path: Path, fold_path: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    formal_by_cancer: dict[str, set[str]] = defaultdict(set)
    with fold_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"cancer_id", "patient_id"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise RuntimeError("Frozen patient-fold map lacks cancer_id/patient_id")
        for row in reader:
            formal_by_cancer[str(row["cancer_id"])].add(str(row["patient_id"]))

    libraries: dict[str, int] = defaultdict(int)
    specimens: dict[str, set[str]] = defaultdict(set)
    aliquots: dict[str, set[str]] = defaultdict(set)
    patients: dict[str, set[str]] = defaultdict(set)
    with mapping_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"bam_prefix", "stanfordUUID", "aliquot_id"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise RuntimeError(f"ATAC mapping lacks columns: {sorted(required)}")
        for row in reader:
            raw_cancer_id = str(row["bam_prefix"]).split("-", 1)[0]
            cancer_id = ATAC_CANCER_ALIASES.get(raw_cancer_id, raw_cancer_id)
            libraries[cancer_id] += 1
            specimens[cancer_id].add(str(row["stanfordUUID"]))
            aliquot = str(row["aliquot_id"])
            aliquots[cancer_id].add(aliquot)
            patients[cancer_id].add(_patient_id(aliquot))

    formal_cancer_count = len(formal_by_cancer)
    atac_cancer_count = len(libraries)
    all_cancers = sorted(set(formal_by_cancer) | set(libraries))
    rows: list[dict[str, object]] = []
    for cancer_id in all_cancers:
        cancer_patients = patients.get(cancer_id, set())
        formal_patients = formal_by_cancer.get(cancer_id, set())
        overlap = cancer_patients & formal_patients
        formal_n = len(formal_patients)
        rows.append(
            {
                "cancer_id": cancer_id,
                "atac_library_rows": libraries.get(cancer_id, 0),
                "atac_specimens": len(specimens.get(cancer_id, set())),
                "atac_aliquots": len(aliquots.get(cancer_id, set())),
                "atac_patients": len(cancer_patients),
                "formal_patients": formal_n,
                "formal_atac_overlap_patients": len(overlap),
                "formal_atac_overlap_fraction": len(overlap) / formal_n if formal_n else None,
                "availability": "AVAILABLE" if overlap else "TYPED_UNAVAILABLE",
            }
        )

    available = [row for row in rows if row["availability"] == "AVAILABLE"]
    payload: dict[str, object] = {
        "format": "CC_HHGT_V3_2_TCGA_REGULATORY_COVERAGE_AUDIT_V1",
        "status": "PASS_ATAC_COVERAGE_AUDITED",
        "mapping": {"path": str(mapping_path.resolve()), "sha256": _sha256(mapping_path)},
        "patient_fold_authority": {"path": str(fold_path.resolve()), "sha256": _sha256(fold_path)},
        "totals": {
            "mapping_library_rows": sum(libraries.values()),
            "atac_cancers": atac_cancer_count,
            "atac_specimens": len(set().union(*specimens.values())),
            "atac_aliquots": len(set().union(*aliquots.values())),
            "atac_patients": len(set().union(*patients.values())),
            "formal_cancers": formal_cancer_count,
            "formal_patients": len(set().union(*formal_by_cancer.values())),
            "formal_atac_overlap_cancers": len(available),
            "formal_atac_overlap_patients": sum(int(row["formal_atac_overlap_patients"]) for row in available),
        },
        "typed_unavailable_cancers": [
            row["cancer_id"] for row in rows if row["availability"] != "AVAILABLE"
        ],
        "cancer_id_aliases": ATAC_CANCER_ALIASES,
    }
    return rows, payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--patient-folds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Coverage audit output reuse is forbidden: {output}")
    rows, payload = audit(args.mapping.resolve(strict=True), args.patient_folds.resolve(strict=True))
    output.mkdir(parents=True)
    table_path = output / "TCGA_ATAC_PATIENT_COVERAGE.tsv"
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    payload["coverage_table"] = {
        "path": str(table_path),
        "sha256": _sha256(table_path),
        "rows": len(rows),
    }
    (output / "AUDIT.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
