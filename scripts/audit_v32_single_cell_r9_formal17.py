#!/usr/bin/env python3
"""Independently audit the 17-cancer V3.2 single-cell r9 release binding."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


FORMAL_CANCERS = frozenset(
    {
        "ACC", "CHOL", "DLBC", "ESCA", "GBM", "HNSC", "KIRC", "LAML",
        "LGG", "LUSC", "MESO", "PCPG", "READ", "SARC", "SKCM", "THYM",
        "UCEC",
    }
)
FORMAL_COMPARTMENTS = ("malignant", "immune", "stromal")
COMPARTMENT_ORDER = {value: index for index, value in enumerate(FORMAL_COMPARTMENTS)}
RELEASE_FILES = (
    "lncrna_donor_celltype_summary.parquet",
    "pathway_availability.parquet",
    "pathway_donor_celltype_summary.parquet",
    "association_evidence.parquet",
    "association_lncrna_testability.parquet",
    "association_context_availability.parquet",
    "RESOURCE_ESTIMATE.json",
)
FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_R9_FORMAL17_INDEPENDENT_AUDIT_V1"


class Formal17AuditError(RuntimeError):
    """Raised when a formal single-cell binding or output invariant fails."""


def sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise Formal17AuditError(f"absent or unsafe file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise Formal17AuditError(f"absent or unsafe JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Formal17AuditError(f"JSON root is not an object: {path}")
    return value


def exclusive_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def python_scalar(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def verify_manifest(output: Path, success: dict[str, Any]) -> dict[str, Any]:
    manifest_path = output / "FILE_MANIFEST.parquet"
    observed_manifest_sha = sha256_file(manifest_path)
    if observed_manifest_sha != success.get("file_manifest_sha256"):
        raise Formal17AuditError(f"FILE_MANIFEST SHA drift: {output}")
    manifest = pd.read_parquet(manifest_path)
    required_columns = {"relative_path", "bytes", "sha256"}
    if set(manifest.columns) != required_columns:
        raise Formal17AuditError(f"FILE_MANIFEST schema drift: {output}")
    if manifest.relative_path.astype(str).tolist() != sorted(RELEASE_FILES):
        raise Formal17AuditError(f"FILE_MANIFEST release set drift: {output}")
    rows = []
    for row in manifest.itertuples(index=False):
        relative = str(row.relative_path)
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise Formal17AuditError(f"unsafe FILE_MANIFEST relative path: {relative}")
        target = output / relative
        observed_bytes = int(target.stat().st_size) if target.is_file() else -1
        observed_sha = sha256_file(target)
        if observed_bytes != int(row.bytes) or observed_sha != str(row.sha256):
            raise Formal17AuditError(f"manifest leaf drift: {target}")
        rows.append(
            {"relative_path": relative, "bytes": observed_bytes, "sha256": observed_sha}
        )
    return {
        "path": str(manifest_path),
        "sha256": observed_manifest_sha,
        "entry_count": len(rows),
        "entries": rows,
    }


def audit_evidence(path: Path, expected_rows: int) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    try:
        observed_rows = int(parquet.metadata.num_rows)
        schema = parquet.schema_arrow.remove_metadata()
        schema_sha = hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()
        columns = [
            "compartment", "bh_q_global_tests", "nominal_p", "lncrna_id",
            "pathway_id", "fdr_0_10_pass", "cell_as_independent_replicate",
        ]
        if not set(columns).issubset(schema.names):
            raise Formal17AuditError(f"association evidence schema drift: {path}")
        previous: tuple[Any, ...] | None = None
        compartment_rows = {value: 0 for value in FORMAL_COMPARTMENTS}
        fdr_pass_rows = {value: 0 for value in FORMAL_COMPARTMENTS}
        for batch in parquet.iter_batches(batch_size=16_384, columns=columns):
            values = [column.to_pylist() for column in batch.columns]
            for row in zip(*values, strict=True):
                compartment = str(row[0])
                if compartment not in COMPARTMENT_ORDER:
                    raise Formal17AuditError(f"unknown evidence compartment: {compartment}")
                q_value = float(row[1])
                nominal_p = float(row[2])
                if not (0.0 <= q_value <= 1.0 and 0.0 <= nominal_p <= 1.0):
                    raise Formal17AuditError(f"invalid p/q value in {path}")
                key = (
                    COMPARTMENT_ORDER[compartment], q_value, nominal_p,
                    str(row[3]), str(row[4]),
                )
                if previous is not None and key < previous:
                    raise Formal17AuditError(f"association evidence sort drift: {path}")
                previous = key
                if bool(row[5]) != (q_value <= 0.10):
                    raise Formal17AuditError(f"fdr_0_10_pass drift: {path}")
                if bool(row[6]):
                    raise Formal17AuditError(f"cell treated as independent replicate: {path}")
                compartment_rows[compartment] += 1
                fdr_pass_rows[compartment] += int(bool(row[5]))
    finally:
        parquet.close()
    if observed_rows != int(expected_rows):
        raise Formal17AuditError(
            f"association evidence row drift: {observed_rows} != {expected_rows}"
        )
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
        "rows": observed_rows,
        "schema_sha256": schema_sha,
        "sorted_by_compartment_q_p_lncrna_pathway": True,
        "compartment_rows": compartment_rows,
        "fdr_0_10_pass_rows": fdr_pass_rows,
    }


def audit_testability(path: Path, mapped_lncrnas: int) -> dict[str, Any]:
    frame = pd.read_parquet(path)
    required = {
        "compartment", "lncrna_id", "eligible_donor_count",
        "detecting_donor_count", "context_detect_rate", "test_status",
        "unavailable_reason",
    }
    if not required.issubset(frame.columns):
        raise Formal17AuditError(f"testability schema drift: {path}")
    expected_rows = int(mapped_lncrnas) * len(FORMAL_COMPARTMENTS)
    if len(frame) != expected_rows:
        raise Formal17AuditError(f"testability row drift: {path}")
    if set(frame.compartment.astype(str)) != set(FORMAL_COMPARTMENTS):
        raise Formal17AuditError(f"testability compartment set drift: {path}")
    by_compartment: dict[str, Any] = {}
    eligible_sets: dict[str, set[str]] = {}
    for compartment in FORMAL_COMPARTMENTS:
        selected = frame.loc[frame.compartment.astype(str).eq(compartment)].copy()
        if len(selected) != int(mapped_lncrnas) or selected.lncrna_id.duplicated().any():
            raise Formal17AuditError(f"testability lncRNA universe drift: {path}")
        eligible = selected.test_status.astype(str).eq("ELIGIBLE_FOR_DONOR_ASSOCIATION")
        eligible_sets[compartment] = set(selected.loc[eligible, "lncrna_id"].astype(str))
        reasons = (
            selected.loc[~eligible, "unavailable_reason"]
            .fillna("<NULL>").astype(str).value_counts().sort_index().to_dict()
        )
        by_compartment[compartment] = {
            "eligible_lncrnas": int(eligible.sum()),
            "typed_unavailable_lncrnas": int((~eligible).sum()),
            "eligible_donor_count": int(selected.eligible_donor_count.iloc[0]),
            "unavailable_reason_counts": {str(k): int(v) for k, v in reasons.items()},
        }
    union = set().union(*eligible_sets.values())
    intersection = set.intersection(*eligible_sets.values())
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
        "rows": len(frame),
        "mapped_source_universe_lncrnas": int(mapped_lncrnas),
        "eligible_union_lncrnas": len(union),
        "eligible_all_three_compartments_lncrnas": len(intersection),
        "by_compartment": by_compartment,
        "success_lncrnas_semantics": "MAPPED_SOURCE_FEATURE_UNIVERSE_NOT_TESTABLE_COUNT",
    }


def audit_groups(path: Path, expected_cells: int) -> dict[str, Any]:
    columns = ["patient_id", "cell_type_major", "compartment", "cell_count"]
    parquet = pq.ParquetFile(path)
    groups: dict[tuple[str, str, str], int] = {}
    try:
        if not set(columns).issubset(parquet.schema_arrow.names):
            raise Formal17AuditError(f"lncRNA summary schema drift: {path}")
        for batch in parquet.iter_batches(batch_size=32_768, columns=columns):
            values = [column.to_pylist() for column in batch.columns]
            for patient, celltype, compartment, cell_count in zip(*values, strict=True):
                key = (str(patient), str(celltype), str(compartment))
                count = int(cell_count)
                previous = groups.setdefault(key, count)
                if previous != count:
                    raise Formal17AuditError(f"group cell count drift: {path}")
    finally:
        parquet.close()
    total_cells = sum(groups.values())
    if total_cells != int(expected_cells):
        raise Formal17AuditError(f"group cells drift: {total_cells} != {expected_cells}")
    compartment_cells: dict[str, int] = {}
    compartment_donors: dict[str, set[str]] = {}
    label_cells: dict[tuple[str, str], int] = {}
    for (patient, celltype, compartment), count in groups.items():
        compartment_cells[compartment] = compartment_cells.get(compartment, 0) + count
        compartment_donors.setdefault(compartment, set()).add(patient)
        label_cells[(celltype, compartment)] = label_cells.get((celltype, compartment), 0) + count
    unresolved_cells = compartment_cells.get("other_unresolved", 0)
    return {
        "donor_celltype_groups": len(groups),
        "cells": total_cells,
        "compartment_cell_counts": dict(sorted(compartment_cells.items())),
        "compartment_unique_donors_before_min_cell_filter": {
            key: len(value) for key, value in sorted(compartment_donors.items())
        },
        "other_unresolved_cells": unresolved_cells,
        "other_unresolved_fraction": unresolved_cells / total_cells if total_cells else 0.0,
        "cell_type_label_counts": [
            {"cell_type_major": key[0], "compartment": key[1], "cells": value}
            for key, value in sorted(label_cells.items())
        ],
    }


def audit_context(path: Path) -> dict[str, Any]:
    frame = pd.read_parquet(path)
    if len(frame) != 3 or set(frame.compartment.astype(str)) != set(FORMAL_COMPARTMENTS):
        raise Formal17AuditError(f"association context row/compartment drift: {path}")
    records = []
    for row in frame.sort_values("compartment", key=lambda x: x.map(COMPARTMENT_ORDER)).to_dict("records"):
        records.append({str(key): python_scalar(value) for key, value in row.items()})
    return {"path": str(path), "sha256": sha256_file(path), "rows": records}


def audit_pathways(path: Path, success: dict[str, Any]) -> dict[str, Any]:
    frame = pd.read_parquet(path, columns=["pathway_id", "ucell_available", "unavailable_reason"])
    if len(frame) != int(success["pathways_total"]) or frame.pathway_id.duplicated().any():
        raise Formal17AuditError(f"pathway availability universe drift: {path}")
    available = frame.ucell_available.astype(bool)
    if int(available.sum()) != int(success["pathways_available"]):
        raise Formal17AuditError(f"available pathway count drift: {path}")
    reasons = (
        frame.loc[~available, "unavailable_reason"].fillna("<NULL>")
        .astype(str).value_counts().sort_index().to_dict()
    )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "total": len(frame),
        "available": int(available.sum()),
        "typed_unavailable": int((~available).sum()),
        "unavailable_reason_counts": {str(k): int(v) for k, v in reasons.items()},
    }


def audit_one(binding: dict[str, Any]) -> dict[str, Any]:
    cancer = str(binding["cancer_id"])
    declared_output = Path(str(binding["output_root"]))
    if declared_output.is_symlink() or not declared_output.is_dir():
        raise Formal17AuditError(f"unsafe cancer output root: {declared_output}")
    output = declared_output.resolve()
    if not output.is_dir() or output.name != f"cancer_id={cancer}":
        raise Formal17AuditError(f"unsafe cancer output root: {output}")
    success_path = output / "SUCCESS.json"
    success_sha = sha256_file(success_path)
    if success_sha != binding.get("success_sha256"):
        raise Formal17AuditError(f"bound SUCCESS SHA drift: {cancer}")
    success = load_json(success_path)
    if success.get("status") != "SUCCESS" or success.get("cancer_id") != cancer:
        raise Formal17AuditError(f"SUCCESS semantics drift: {cancer}")
    lineage_path = output / "LINEAGE.json"
    lineage_sha = sha256_file(lineage_path)
    if lineage_sha != success.get("lineage_sha256"):
        raise Formal17AuditError(f"LINEAGE SHA drift: {cancer}")
    lineage = load_json(lineage_path)
    if lineage.get("contract_sha256") != success.get("contract_sha256"):
        raise Formal17AuditError(f"contract binding drift: {cancer}")
    manifest = verify_manifest(output, success)
    evidence = audit_evidence(
        output / "association_evidence.parquet",
        int(success["association_evidence_rows"]),
    )
    testability = audit_testability(
        output / "association_lncrna_testability.parquet", int(success["lncrnas"])
    )
    groups = audit_groups(
        output / "lncrna_donor_celltype_summary.parquet", int(success["cells"])
    )
    context = audit_context(output / "association_context_availability.parquet")
    pathways = audit_pathways(output / "pathway_availability.parquet", success)
    resource = load_json(output / "RESOURCE_ESTIMATE.json")
    return {
        "cancer_id": cancer,
        "generated_in_current_r11_run": bool(
            binding.get("generated_in_current_r11_run")
        ),
        "output_root": str(output),
        "success_path": str(success_path),
        "success_sha256": success_sha,
        "contract_sha256": str(success["contract_sha256"]),
        "lineage_sha256": lineage_sha,
        "file_manifest": manifest,
        "dataset_id": str(success["dataset_id"]),
        "cells": int(success["cells"]),
        "donors": int(success["donors"]),
        "mapped_source_universe_lncrnas": int(success["lncrnas"]),
        "pathways": pathways,
        "association_evidence": evidence,
        "association_testability": testability,
        "association_context": context,
        "cell_groups": groups,
        "measurement_scale": str(lineage.get("measurement_scale")),
        "feature_mapping_policy": str(lineage.get("feature_mapping_policy")),
        "duplicate_policy": str(lineage.get("duplicate_policy")),
        "source_h5_sha256": str(lineage.get("input_shas", {}).get("h5_sha256")),
        "source_metadata_sha256": str(
            lineage.get("input_shas", {}).get("metadata_sha256")
        ),
        "resource_estimated_peak_ram_bytes": int(resource["estimated_peak_ram_bytes"]),
    }


def write_summary_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "cancer_id", "generated_in_current_r11_run", "cells", "donors",
        "mapped_source_universe_lncrnas", "eligible_union_lncrnas",
        "eligible_malignant_lncrnas", "eligible_immune_lncrnas",
        "eligible_stromal_lncrnas", "association_evidence_rows",
        "fdr_0_10_pass_rows", "pathways_available", "pathways_typed_unavailable",
        "other_unresolved_cells", "other_unresolved_fraction", "success_sha256",
    ]
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            testability = row["association_testability"]
            writer.writerow(
                {
                    "cancer_id": row["cancer_id"],
                    "generated_in_current_r11_run": str(
                        row["generated_in_current_r11_run"]
                    ).lower(),
                    "cells": row["cells"],
                    "donors": row["donors"],
                    "mapped_source_universe_lncrnas": row["mapped_source_universe_lncrnas"],
                    "eligible_union_lncrnas": testability["eligible_union_lncrnas"],
                    "eligible_malignant_lncrnas": testability["by_compartment"]["malignant"]["eligible_lncrnas"],
                    "eligible_immune_lncrnas": testability["by_compartment"]["immune"]["eligible_lncrnas"],
                    "eligible_stromal_lncrnas": testability["by_compartment"]["stromal"]["eligible_lncrnas"],
                    "association_evidence_rows": row["association_evidence"]["rows"],
                    "fdr_0_10_pass_rows": sum(row["association_evidence"]["fdr_0_10_pass_rows"].values()),
                    "pathways_available": row["pathways"]["available"],
                    "pathways_typed_unavailable": row["pathways"]["typed_unavailable"],
                    "other_unresolved_cells": row["cell_groups"]["other_unresolved_cells"],
                    "other_unresolved_fraction": f"{row['cell_groups']['other_unresolved_fraction']:.12g}",
                    "success_sha256": row["success_sha256"],
                }
            )
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort-success", required=True, type=Path)
    parser.add_argument("--audit-root", required=True, type=Path)
    parser.add_argument(
        "--expected-cancers", nargs="+",
        help="Optional explicit cancer universe for an independently generated extension cohort.",
    )
    args = parser.parse_args()
    cohort_path = args.cohort_success.resolve()
    audit_root = args.audit_root.resolve()
    if audit_root.exists() or audit_root.is_symlink():
        raise Formal17AuditError(f"audit root reuse forbidden: {audit_root}")
    cohort = load_json(cohort_path)
    expected = (
        frozenset(str(value).upper() for value in args.expected_cancers)
        if args.expected_cancers else FORMAL_CANCERS
    )
    if not expected or len(expected) != len(args.expected_cancers or expected):
        raise Formal17AuditError("expected cancer universe is empty or duplicated")
    expected_count = len(expected)
    expected_status = f"BOUND_{expected_count}_OF_{expected_count}_PENDING_INDEPENDENT_AUDIT"
    if (
        cohort.get("status") != expected_status
        or int(cohort.get("formal_bound_cancer_count", 0)) != expected_count
    ):
        raise Formal17AuditError("cohort SUCCESS is not a complete expected binding")
    bindings = cohort.get("formal_bindings")
    if not isinstance(bindings, list) or len(bindings) != expected_count:
        raise Formal17AuditError("formal binding list count drift")
    if {str(row.get("cancer_id")) for row in bindings} != expected:
        raise Formal17AuditError("formal cancer universe drift")
    audit_root.mkdir(parents=True)
    try:
        rows = [audit_one(binding) for binding in sorted(bindings, key=lambda x: x["cancer_id"])]
        summary_path = audit_root / f"FORMAL{expected_count}_SUMMARY.tsv"
        write_summary_tsv(summary_path, rows)
        value = {
            "format": (
                FORMAT if not args.expected_cancers
                else "CC_HHGT_V3_2_SINGLE_CELL_R11_FORMAL23_INDEPENDENT_AUDIT_V1"
            ),
            "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
            "status": f"PASS_{expected_count}_OF_{expected_count}_INDEPENDENTLY_VERIFIED",
            "cohort_success_path": str(cohort_path),
            "cohort_success_sha256": sha256_file(cohort_path),
            "formal_cancer_universe": sorted(expected),
            "formal_cancer_count": len(rows),
            "generated_in_current_r11_run_count": sum(
                row["generated_in_current_r11_run"] for row in rows
            ),
            "immutable_upstream_binding_count": sum(
                not row["generated_in_current_r11_run"] for row in rows
            ),
            "total_cells": sum(row["cells"] for row in rows),
            "total_association_evidence_rows": sum(row["association_evidence"]["rows"] for row in rows),
            "mapped_source_universe_lncrna_union_not_computed_from_success_counts": True,
            "success_lncrnas_semantics": "MAPPED_SOURCE_FEATURE_UNIVERSE_NOT_TESTABLE_COUNT",
            "per_cancer_testable_counts_derived_from": "association_lncrna_testability.parquet",
            "summary_tsv_path": str(summary_path),
            "summary_tsv_sha256": sha256_file(summary_path),
            "per_cancer": rows,
            "production_deployed": False,
        }
        value["audit_contract_sha256"] = canonical_sha256(value)
        report_path = audit_root / "AUDIT.json"
        exclusive_json(report_path, value)
        print(json.dumps({**value, "audit_sha256": sha256_file(report_path)}, ensure_ascii=False, indent=2, sort_keys=True))
    except Exception:
        if not any(audit_root.iterdir()):
            audit_root.rmdir()
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
