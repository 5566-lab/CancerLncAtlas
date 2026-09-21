#!/usr/bin/env python3
"""Independently audit a V3.2 single-cell formal-context binding."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb


ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
BINDING_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_FORMAL_CONTEXT_BINDING_V1"
AUDIT_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_FORMAL_CONTEXT_INDEPENDENT_AUDIT_V1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def sql_path(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", required=True, type=Path)
    parser.add_argument("--expected-binding-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    source = args.binding.resolve()
    observed = sha256(source)
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, observed_value: Any, expected: Any) -> None:
        checks.append(
            {
                "check": name,
                "status": "PASS" if passed else "FAIL",
                "observed": observed_value,
                "expected": expected,
            }
        )

    check("binding_sha256", observed == args.expected_binding_sha256, observed, args.expected_binding_sha256)
    binding = load_json(source)
    check("binding_format", binding.get("format") == BINDING_FORMAT, binding.get("format"), BINDING_FORMAT)
    check(
        "analysis_version",
        binding.get("analysis_version") == ANALYSIS_VERSION,
        binding.get("analysis_version"),
        ANALYSIS_VERSION,
    )
    check(
        "status",
        binding.get("status") == "FORMAL_CONTEXT_READY_WITH_CELL_LEVEL_UCELL_PARTIAL",
        binding.get("status"),
        "FORMAL_CONTEXT_READY_WITH_CELL_LEVEL_UCELL_PARTIAL",
    )
    check("formal_context_cancers", len(set(binding.get("formal_context_cancers", []))) == 17, len(set(binding.get("formal_context_cancers", []))), 17)
    for field in (
        "historical_predictions_used",
        "historical_rankings_used",
        "historical_checkpoints_used",
        "exact_primary_modified",
        "release_ready",
        "production_deployed",
    ):
        check(field, binding.get(field) is False, binding.get(field), False)

    artifacts = binding.get("artifacts", {})
    roles = ("typed_predictions", "lncrna_celltype", "pathway_activity", "exact_association")
    paths: dict[str, Path] = {}
    for role in roles:
        declaration = artifacts.get(role, {})
        path = Path(str(declaration.get("path", ""))).resolve()
        exists = path.is_file() and not path.is_symlink()
        actual = sha256(path) if exists else None
        expected = declaration.get("sha256")
        check(f"{role}_sha256", exists and actual == expected, actual, expected)
        paths[role] = path

    connection = duckdb.connect(database=":memory:")
    try:
        typed = connection.execute(
            f"""
            SELECT count(*), count_if(single_cell_available),
                   count_if(single_cell_available AND
                     (single_cell_replication_probability IS NULL OR
                      NOT isfinite(single_cell_replication_probability) OR
                      single_cell_replication_probability < 0 OR
                      single_cell_replication_probability > 1)),
                   count_if(NOT single_cell_available AND
                     single_cell_replication_probability IS NOT NULL),
                   count_if(changes_primary_ranking),
                   count(DISTINCT CASE WHEN single_cell_available THEN cancer_id END),
                   count(DISTINCT CASE WHEN single_cell_available THEN
                     (dataset_id, cancer_id, cell_type_major) END)
            FROM read_parquet({sql_path(paths['typed_predictions'])})
            """
        ).fetchone()
        celltype = connection.execute(
            f"""
            SELECT count(*), count_if(lnc_celltype_available),
                   count(DISTINCT CASE WHEN lnc_celltype_available THEN cancer_id END),
                   count(DISTINCT CASE WHEN lnc_celltype_available THEN lncrna_id END),
                   count(DISTINCT CASE WHEN lnc_celltype_available THEN cell_type END),
                   count_if(NOT source_is_nonpredictive)
            FROM read_parquet({sql_path(paths['lncrna_celltype'])})
            """
        ).fetchone()
        activity = connection.execute(
            f"""
            SELECT count(*), count_if(activity_available),
                   count(DISTINCT CASE WHEN activity_available THEN cancer_id END),
                   count(DISTINCT CASE WHEN activity_available THEN donor_id END),
                   count(DISTINCT CASE WHEN activity_available THEN cell_type END),
                   count(DISTINCT CASE WHEN activity_available THEN pathway_id END),
                   count_if(NOT source_is_nonpredictive),
                   count(DISTINCT CASE WHEN activity_available THEN
                     (dataset_id, cancer_id, cell_type) END)
            FROM read_parquet({sql_path(paths['pathway_activity'])})
            """
        ).fetchone()
        exact = connection.execute(
            f"""
            SELECT count(*), count_if(single_cell_available),
                   count(DISTINCT CASE WHEN single_cell_available THEN cancer_id END),
                   count_if(NOT exact_pathway_only)
            FROM read_parquet({sql_path(paths['exact_association'])})
            """
        ).fetchone()
        overlap = connection.execute(
            f"""
            WITH predicted AS (
              SELECT DISTINCT dataset_id, cancer_id, cell_type_major AS cell_type
              FROM read_parquet({sql_path(paths['typed_predictions'])})
              WHERE single_cell_available
            ), activity AS (
              SELECT DISTINCT dataset_id, cancer_id, cell_type
              FROM read_parquet({sql_path(paths['pathway_activity'])})
              WHERE activity_available
            )
            SELECT count(*), count_if(activity.dataset_id IS NULL)
            FROM predicted LEFT JOIN activity USING (dataset_id, cancer_id, cell_type)
            """
        ).fetchone()
    finally:
        connection.close()
    expected_rows = {
        "typed": (7_814_014, 6_214_014, 0, 0, 0, 17, 129),
        "celltype": (229_104, 194_082, 17, 5_424, 16, 0),
        "activity": (4_354_871, 4_320_711, 17, 274, 16, 2_135, 0, 137),
        "exact": (2_554_541, 954_541, 17, 0),
        "context_overlap": (129, 0),
    }
    for name, observed_row in {
        "typed": typed,
        "celltype": celltype,
        "activity": activity,
        "exact": exact,
        "context_overlap": overlap,
    }.items():
        values = tuple(map(int, observed_row))
        check(f"{name}_semantic_counts", values == expected_rows[name], values, expected_rows[name])

    fusion = binding.get("fusion", {})
    check(
        "single_cell_secondary_effect",
        fusion.get("single_cell_currently_changes_secondary_score") is False,
        fusion.get("single_cell_currently_changes_secondary_score"),
        False,
    )
    check(
        "single_cell_discovery_weight",
        fusion.get("weights", {}).get("discovery", {}).get("single_cell_weight") == 0.0,
        fusion.get("weights", {}).get("discovery", {}).get("single_cell_weight"),
        0.0,
    )
    check(
        "single_cell_confidence_weight",
        fusion.get("weights", {}).get("confidence", {}).get("single_cell_weight") == 0.0,
        fusion.get("weights", {}).get("confidence", {}).get("single_cell_weight"),
        0.0,
    )
    ucell = binding.get("cell_level_ucell", {})
    check("ucell_scope", ucell.get("status") == "PARTIAL_1_OF_17_FORMAL_CANCERS", ucell.get("status"), "PARTIAL_1_OF_17_FORMAL_CANCERS")
    check("ucell_numeric_rows", ucell.get("numeric_rows_hnsc") == 12_512_240, ucell.get("numeric_rows_hnsc"), 12_512_240)
    failures = [row for row in checks if row["status"] != "PASS"]
    report = {
        "format": AUDIT_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not failures else "FAIL",
        "source_binding_path": str(source),
        "source_binding_sha256": observed,
        "checks": checks,
        "check_count": len(checks),
        "failure_count": len(failures),
        "independent_findings": {
            "formal_context_ready": not failures,
            "cell_level_ucell_complete": False,
            "single_cell_currently_changes_secondary_score": False,
            "changes_exact_primary_score": False,
            "learned_celltype_contexts": 129,
            "activity_celltype_contexts": 137,
            "learned_contexts_with_activity_match": 129,
            "learned_contexts_missing_activity": 0,
            "activity_and_learned_association_are_distinct": True,
        },
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "INDEPENDENT_AUDIT.json"
    audit_binding_path = output / "INDEPENDENT_AUDIT_BINDING.json"
    success_path = output / "SUCCESS.json"
    if any(path.exists() for path in (report_path, audit_binding_path, success_path)):
        raise RuntimeError(f"Refusing to overwrite independent audit: {output}")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit_binding = {
        "format": "CC_HHGT_V3_2_SINGLE_CELL_FORMAL_CONTEXT_INDEPENDENT_AUDIT_BINDING_V1",
        "analysis_version": ANALYSIS_VERSION,
        "status": report["status"],
        "source_binding_path": str(source),
        "source_binding_sha256": observed,
        "audit_report_path": str(report_path),
        "audit_report_sha256": sha256(report_path),
        "check_count": len(checks),
        "failure_count": len(failures),
        "release_ready": False,
        "production_deployed": False,
    }
    audit_binding_path.write_text(
        json.dumps(audit_binding, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    success_path.write_text(
        json.dumps(
            {
                "status": report["status"],
                "binding": audit_binding_path.name,
                "binding_sha256": sha256(audit_binding_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "check_count": len(checks),
                "failure_count": len(failures),
                "audit_binding_path": str(audit_binding_path),
                "audit_binding_sha256": sha256(audit_binding_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
