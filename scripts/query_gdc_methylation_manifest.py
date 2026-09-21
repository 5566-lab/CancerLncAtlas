"""Query an auditable TCGA 450K methylation download manifest from GDC.

The script writes metadata and manifests only.  It does not download assay
files and never treats an absent patient/file as a biological zero.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


API = "https://api.gdc.cancer.gov/files"
CANCERS = (
    "ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA",
    "GBM", "HNSC", "KICH", "KIRC", "KIRP", "LAML", "LGG", "LIHC",
    "LUAD", "LUSC", "MESO", "OV", "PAAD", "PCPG", "PRAD", "READ",
    "SARC", "SKCM", "STAD", "TGCT", "THCA", "THYM", "UCEC", "UCS",
    "UVM",
)
FIELDS = (
    "file_id", "file_name", "md5sum", "file_size", "state", "access",
    "data_category", "data_type", "data_format", "experimental_strategy",
    "platform", "analysis.workflow_type", "cases.case_id",
    "cases.submitter_id", "cases.project.project_id", "cases.samples.sample_id",
    "cases.samples.submitter_id", "cases.samples.sample_type",
    "cases.samples.tissue_type",
)
PRIMARY_TUMOR = "Primary Tumor"
RESULT_DATA_TYPE = "Methylation Beta Value"
FORMAT = "CANCERLNCATLAS_V32_GDC_METHYLATION_MANIFEST_V1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="")
    os.replace(temporary, path)


def _post(payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        API,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        value = json.load(response)
    if not isinstance(value, dict) or not isinstance(value.get("data"), dict):
        raise RuntimeError("GDC files API returned a malformed response")
    return value


def _filters() -> dict[str, Any]:
    return {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content": {
                    "field": "cases.project.project_id",
                    "value": [f"TCGA-{cancer}" for cancer in CANCERS],
                },
            },
            {
                "op": "in",
                "content": {"field": "data_category", "value": ["DNA Methylation"]},
            },
            {
                "op": "in",
                "content": {
                    "field": "experimental_strategy",
                    "value": ["Methylation Array"],
                },
            },
            {
                "op": "in",
                "content": {
                    "field": "platform",
                    "value": ["Illumina Human Methylation 450"],
                },
            },
            {
                "op": "in",
                "content": {"field": "data_type", "value": [RESULT_DATA_TYPE]},
            },
            {"op": "in", "content": {"field": "access", "value": ["open"]}},
        ],
    }


def query_all(page_size: int = 5000) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    offset = 0
    pagination: dict[str, Any] = {}
    while True:
        response = _post(
            {
                "filters": _filters(),
                "format": "JSON",
                "fields": ",".join(FIELDS),
                "expand": "cases,cases.project,cases.samples,analysis",
                "from": offset,
                "size": page_size,
            }
        )
        data = response["data"]
        page = data.get("hits", [])
        if not isinstance(page, list):
            raise RuntimeError("GDC files API hits are malformed")
        hits.extend(page)
        pagination = dict(data.get("pagination") or {})
        total = int(pagination.get("total", len(hits)))
        if not page or len(hits) >= total:
            break
        offset += len(page)
    if len(hits) != int(pagination.get("total", len(hits))):
        raise RuntimeError("GDC files pagination did not recover the declared total")
    return hits, pagination


def _workflow(hit: dict[str, Any]) -> str:
    analysis = hit.get("analysis") or {}
    return str(analysis.get("workflow_type") or "")


def flatten(hits: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for hit in hits:
        cases = hit.get("cases") or [None]
        for case in cases:
            case = case or {}
            project = str((case.get("project") or {}).get("project_id") or "")
            cancer = project.removeprefix("TCGA-").upper()
            samples = case.get("samples") or [None]
            for sample in samples:
                sample = sample or {}
                rows.append(
                    {
                        "file_id": str(hit.get("file_id") or hit.get("id") or ""),
                        "file_name": str(hit.get("file_name") or ""),
                        "md5sum": str(hit.get("md5sum") or ""),
                        "file_size": int(hit.get("file_size") or 0),
                        "state": str(hit.get("state") or ""),
                        "access": str(hit.get("access") or ""),
                        "data_category": str(hit.get("data_category") or ""),
                        "data_type": str(hit.get("data_type") or ""),
                        "data_format": str(hit.get("data_format") or ""),
                        "experimental_strategy": str(
                            hit.get("experimental_strategy") or ""
                        ),
                        "platform": str(hit.get("platform") or ""),
                        "workflow_type": _workflow(hit),
                        "case_id": str(case.get("case_id") or case.get("id") or ""),
                        "patient_id": str(case.get("submitter_id") or "")[:12],
                        "cancer_id": cancer,
                        "sample_id": str(sample.get("submitter_id") or "")[:16],
                        "sample_uuid": str(sample.get("sample_id") or sample.get("id") or ""),
                        "sample_type": str(sample.get("sample_type") or ""),
                        "tissue_type": str(sample.get("tissue_type") or ""),
                    }
                )
    return rows


def _write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient-folds", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    folds_path = args.patient_folds.resolve(strict=True)
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"GDC methylation manifest output reuse refused: {output}")

    import pandas as pd

    folds = pd.read_csv(folds_path, sep="\t", dtype=str)
    required = {"cancer_id", "patient_id"}
    if missing := sorted(required - set(folds.columns)):
        raise RuntimeError(f"Patient folds lack {missing}")
    formal_pairs = set(
        zip(folds.cancer_id.astype(str).str.upper(), folds.patient_id.astype(str).str[:12])
    )
    if folds.cancer_id.nunique() != 33:
        raise RuntimeError("Patient folds are not the formal 33-cancer authority")

    hits, pagination = query_all()
    rows = flatten(hits)
    fields = list(rows[0]) if rows else ["file_id"]
    selected = [
        row
        for row in rows
        if row["sample_type"] == PRIMARY_TUMOR
        and row["data_type"] == RESULT_DATA_TYPE
        and (row["cancer_id"], row["patient_id"]) in formal_pairs
        and row["file_id"]
        and row["md5sum"]
        and row["file_size"] > 0
    ]
    # A file can inherit multiple sample rows through an expanded case.  Keep
    # one exact file/patient/sample key, but retain technical/aliquot replicates.
    selected = list(
        {
            (row["file_id"], row["patient_id"], row["sample_id"]): row
            for row in selected
        }.values()
    )
    selected.sort(key=lambda row: (row["cancer_id"], row["patient_id"], row["file_id"]))
    unique_files = {row["file_id"]: row for row in selected}
    patients_by_cancer: dict[str, set[str]] = defaultdict(set)
    for row in selected:
        patients_by_cancer[row["cancer_id"]].add(row["patient_id"])

    output.mkdir(parents=True)
    _write_tsv(output / "GDC_METHYLATION_FILES_ALL.tsv", rows, fields)
    _write_tsv(output / "SELECTED_PRIMARY_TUMOR_FILES.tsv", selected, fields)
    manifest_rows = [
        {
            "id": row["file_id"],
            "filename": row["file_name"],
            "md5": row["md5sum"],
            "size": row["file_size"],
            "state": row["state"],
        }
        for row in sorted(unique_files.values(), key=lambda value: value["file_id"])
    ]
    _write_tsv(output / "gdc_manifest.tsv", manifest_rows, ["id", "filename", "md5", "size", "state"])

    data_type_counts = Counter(row["data_type"] for row in rows)
    workflow_counts = Counter(row["workflow_type"] for row in rows)
    summary = {
        "format": FORMAT,
        "status": "PASS_GDC_METHYLATION_MANIFEST_READY",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gdc_api": API,
        "gdc_total_files": int(pagination.get("total", len(hits))),
        "flattened_file_sample_rows": len(rows),
        "selected_primary_tumor_file_sample_rows": len(selected),
        "selected_unique_files": len(unique_files),
        "selected_download_bytes": sum(row["file_size"] for row in unique_files.values()),
        "selected_formal_patients": len({(row["cancer_id"], row["patient_id"]) for row in selected}),
        "formal_patients": len(formal_pairs),
        "patient_coverage_by_cancer": {
            cancer: len(patients_by_cancer.get(cancer, set())) for cancer in CANCERS
        },
        "data_type_counts": dict(sorted(data_type_counts.items())),
        "workflow_type_counts": dict(sorted(workflow_counts.items())),
        "filters": _filters(),
        "patient_fold_sha256": _sha256(folds_path),
        "missing_patient_assumed_zero": False,
        "selected_result_data_type": RESULT_DATA_TYPE,
        "download_started": False,
    }
    _atomic_text(
        output / "MANIFEST_SUMMARY.json",
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    hashes = {
        path.name: _sha256(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "FILE_SHA256SUMS.json"
    }
    _atomic_text(
        output / "FILE_SHA256SUMS.json",
        json.dumps(hashes, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
