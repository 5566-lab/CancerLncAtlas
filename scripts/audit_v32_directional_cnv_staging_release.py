#!/usr/bin/env python3
"""Independent, fail-closed audit of the directional CNV website release."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import duckdb


EXPECTED_ROWS = 3_300_000
EXPECTED_CANCERS = 33
EXPECTED_COLUMNS = {
    "cancer_id",
    "lncrna_id",
    "pathway_id",
    "cnv_context_probability",
    "cnv_context_probability_sd",
    # The materializer publishes fold-range bounds as part of the public
    # contract.  Audit them explicitly instead of treating them as schema
    # drift.
    "cnv_context_probability_min",
    "cnv_context_probability_max",
    "cnv_patient_folds_with_prediction",
    "cnv_patient_fold_count",
    "cnv_available",
    "cnv_unavailable_reason",
    "local_cnv_available_fold_count",
    "pathway_cnv_available_fold_count",
    "cnv_pair_callable_patients_across_oof",
    "training_run_id",
    "analysis_version",
    "module_id",
    "target_level",
    "prediction_format",
    "changes_primary_ranking",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} is missing or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def declared_artifact(
    release: Path,
    artifacts: dict[str, Any],
    name: str,
) -> tuple[Path, str, int | None]:
    record = artifacts.get(name)
    if not isinstance(record, dict):
        raise RuntimeError(f"release binding lacks {name}")
    path = Path(str(record.get("path", "")))
    expected = str(record.get("sha256", "")).lower()
    if (
        path.parent.resolve() != release
        or path.is_symlink()
        or not path.is_file()
        or len(expected) != 64
        or sha256(path) != expected
    ):
        raise RuntimeError(f"release {name} declaration or hash drift")
    rows = int(record["rows"]) if "rows" in record else None
    return path.resolve(), expected, rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--memory-limit", default="32GB")
    args = parser.parse_args()

    release = args.release_root.resolve(strict=True)
    output = args.output_root.resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"immutable independent audit output exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.partial.{os.getpid()}")
    if staging.exists() or staging.is_symlink():
        raise RuntimeError(f"independent audit staging path exists: {staging}")
    staging.mkdir()

    binding_path = release / "DIRECTIONAL_CNV_WEBSITE_BINDING.json"
    success_path = release / "SUCCESS.json"
    binding = load_json(binding_path, "directional CNV release binding")
    success = load_json(success_path, "directional CNV release SUCCESS")
    if (
        binding.get("format")
        != "CC_HHGT_V3_2_DIRECTIONAL_CNV_WEBSITE_BINDING_V1"
        or binding.get("status") != "PASS"
        or binding.get("analysis_version")
        != "CancerLncAtlas_V3.2_FULL_MULTITASK"
        or binding.get("staging_queryable") is not True
        or binding.get("release_ready") is not False
        or binding.get("release_blocker") != "CNV_ROUTER_INCREMENT_NOT_TESTED"
        or binding.get("changes_primary_ranking") is not False
        or success.get("format")
        != "CC_HHGT_V3_2_DIRECTIONAL_CNV_WEBSITE_SUCCESS_V1"
        or success.get("status") != "SUCCESS"
        or success.get("binding_sha256") != sha256(binding_path)
        or success.get("production_deployed") is not False
        or success.get("release_ready") is not False
    ):
        raise RuntimeError("directional CNV release header or SUCCESS contract drift")
    artifacts = binding.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "predictions",
        "coverage",
        "lineage",
        "transformation_audit",
    }:
        raise RuntimeError("directional CNV release artifact set drift")
    predictions, prediction_sha, prediction_rows = declared_artifact(
        release, artifacts, "predictions"
    )
    coverage, coverage_sha, coverage_rows = declared_artifact(
        release, artifacts, "coverage"
    )
    lineage_path, _, _ = declared_artifact(release, artifacts, "lineage")
    transformation_path, transformation_sha, _ = declared_artifact(
        release, artifacts, "transformation_audit"
    )
    if prediction_rows != EXPECTED_ROWS or coverage_rows != EXPECTED_CANCERS:
        raise RuntimeError("directional CNV declared row counts drift")
    lineage = load_json(lineage_path, "directional CNV release lineage")
    transformation = load_json(
        transformation_path, "directional CNV transformation audit"
    )
    checks = transformation.get("checks")
    if (
        lineage.get("format")
        != "CC_HHGT_V3_2_DIRECTIONAL_CNV_WEBSITE_LINEAGE_V1"
        or lineage.get("status") != "PASS"
        or lineage.get("mutation_features_used") is not False
        or lineage.get("primary_ranking_changed") is not False
        or transformation.get("format")
        != "CC_HHGT_V3_2_DIRECTIONAL_CNV_WEBSITE_TRANSFORMATION_AUDIT_V1"
        or transformation.get("status") != "PASS"
        or transformation.get("prediction_sha256") != prediction_sha
        or transformation.get("coverage_sha256") != coverage_sha
        or not isinstance(checks, dict)
        or checks.get("mutation_features_used") is not False
        or checks.get("superseded_combined_cnv_read") is not False
        or checks.get("primary_ranking_changed") is not False
    ):
        raise RuntimeError("directional CNV transformation provenance drift")

    source_oof = Path(str(lineage.get("source_oof_root", ""))).resolve(strict=True)
    source_local = Path(
        str(lineage.get("source_local_cnv_audit_root", ""))
    ).resolve(strict=True)
    source_manifest = source_oof / "OOF_MANIFEST.json"
    source_audit = source_oof / "CNV_OOF_INDEPENDENT_AUDIT.json"
    source_success = source_oof / "SUCCESS.json"
    local_success = source_local / "SUCCESS.json"
    for path, label in (
        (source_manifest, "source OOF manifest"),
        (source_audit, "source OOF independent audit"),
        (source_success, "source OOF SUCCESS"),
        (local_success, "local-CNV audit SUCCESS"),
    ):
        load_json(path, label)
    if (
        sha256(source_manifest) != lineage.get("source_manifest_sha256")
        or sha256(source_audit) != lineage.get("source_independent_audit_sha256")
        or sha256(source_success) != transformation.get("source_success_sha256")
        or sha256(local_success) != transformation.get("local_cnv_success_sha256")
    ):
        raise RuntimeError("directional CNV source lineage hash drift")

    connection = duckdb.connect()
    connection.execute(f"SET memory_limit='{args.memory_limit}'")
    connection.execute("SET threads=8")
    columns = {
        row[0]
        for row in connection.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)", [str(predictions)]
        ).fetchall()
    }
    if columns != EXPECTED_COLUMNS or any("mutation" in name.lower() for name in columns):
        raise RuntimeError(
            f"directional CNV public schema drift: missing={sorted(EXPECTED_COLUMNS-columns)}, "
            f"extra={sorted(columns-EXPECTED_COLUMNS)}"
        )
    summary = connection.execute(
        """
        SELECT count(*), count(DISTINCT (cancer_id, lncrna_id, pathway_id)),
               count(DISTINCT cancer_id), min(rows_per_cancer), max(rows_per_cancer),
               count_if(cnv_available AND
                        (cnv_context_probability IS NULL OR
                         NOT isfinite(cnv_context_probability) OR
                         cnv_context_probability < 0 OR
                         cnv_context_probability > 1 OR
                         cnv_context_probability_min IS NULL OR
                         cnv_context_probability_max IS NULL OR
                         NOT isfinite(cnv_context_probability_min::DOUBLE) OR
                         NOT isfinite(cnv_context_probability_max::DOUBLE) OR
                         cnv_context_probability_min < 0 OR
                         cnv_context_probability_max > 1 OR
                         cnv_context_probability_min > cnv_context_probability_max OR
                         cnv_context_probability < cnv_context_probability_min OR
                         cnv_context_probability > cnv_context_probability_max)),
               count_if(NOT cnv_available AND
                        (cnv_context_probability IS NOT NULL OR
                         cnv_context_probability_sd IS NOT NULL OR
                         cnv_context_probability_min IS NOT NULL OR
                         cnv_context_probability_max IS NOT NULL)),
               count_if(cnv_available AND
                        nullif(trim(cnv_unavailable_reason), '') IS NOT NULL),
               count_if(NOT cnv_available AND
                        nullif(trim(cnv_unavailable_reason), '') IS NULL),
               count_if(cnv_patient_fold_count != 5 OR
                        cnv_patient_folds_with_prediction < 0 OR
                        cnv_patient_folds_with_prediction > 5),
               count_if(analysis_version != 'CancerLncAtlas_V3.2_FULL_MULTITASK'
                        OR module_id != 'cnv' OR changes_primary_ranking)
        FROM (
          SELECT *, count(*) OVER (PARTITION BY cancer_id) AS rows_per_cancer
          FROM read_parquet(?)
        )
        """,
        [str(predictions)],
    ).fetchone()
    if tuple(map(int, summary)) != (
        EXPECTED_ROWS,
        EXPECTED_ROWS,
        EXPECTED_CANCERS,
        100_000,
        100_000,
        0,
        0,
        0,
        0,
        0,
        0,
    ):
        raise RuntimeError(f"directional CNV prediction closure failed: {summary}")
    coverage_check = connection.execute(
        """
        WITH expected AS (
          SELECT cancer_id, count(*) AS candidate_rows,
                 count_if(cnv_available) AS available_rows,
                 count_if(NOT cnv_available) AS typed_unavailable_rows,
                 avg(cnv_patient_folds_with_prediction) AS mean_available_folds,
                 avg(cnv_pair_callable_patients_across_oof)
                   AS mean_pair_callable_patients
          FROM read_parquet(?) GROUP BY cancer_id
        ), observed AS (SELECT * FROM read_parquet(?))
        SELECT
          (SELECT count(*) FROM observed),
          (SELECT count(DISTINCT cancer_id) FROM observed),
          (SELECT count(*) FROM expected e FULL OUTER JOIN observed o USING(cancer_id)
           WHERE e.cancer_id IS NULL OR o.cancer_id IS NULL
              OR e.candidate_rows != o.candidate_rows
              OR e.available_rows != o.available_rows
              OR e.typed_unavailable_rows != o.typed_unavailable_rows
              OR abs(e.mean_available_folds-o.mean_available_folds) > 1e-12
              OR abs(e.mean_pair_callable_patients-o.mean_pair_callable_patients) > 1e-12)
        """,
        [str(predictions), str(coverage)],
    ).fetchone()
    connection.close()
    if tuple(map(int, coverage_check)) != (33, 33, 0):
        raise RuntimeError(f"directional CNV coverage rederivation failed: {coverage_check}")

    audited_at = datetime.now(timezone.utc).isoformat()
    report = {
        "format": "CC_HHGT_V3_2_DIRECTIONAL_CNV_WEBSITE_INDEPENDENT_AUDIT_V1",
        "status": "PASS",
        "audited_at_utc": audited_at,
        "independent_of_materializer_implementation": True,
        "materializer_imported": False,
        "accepted_for_staging_api_integration": True,
        "checks": {
            "release_rows": EXPECTED_ROWS,
            "unique_keys": EXPECTED_ROWS,
            "cancers": EXPECTED_CANCERS,
            "rows_per_cancer": 100_000,
            "artifact_hashes_recomputed": True,
            "source_hashes_recomputed": True,
            "coverage_rederived": True,
            "signed_directional_source_only": True,
            "mutation_columns_absent": True,
            "typed_null_contract": True,
            "probability_contract": True,
            "five_fold_count_contract": True,
            "superseded_combined_cnv_exposed": False,
            "changes_primary_ranking": False,
        },
        "release_binding_sha256": sha256(binding_path),
        "release_success_sha256": sha256(success_path),
        "prediction_sha256": prediction_sha,
        "coverage_sha256": coverage_sha,
        "transformation_audit_sha256": transformation_sha,
    }
    report_path = staging / "AUDIT.json"
    write_json(report_path, report)
    audit_binding = {
        "format": (
            "CC_HHGT_V3_2_DIRECTIONAL_CNV_WEBSITE_"
            "INDEPENDENT_AUDIT_BINDING_V1"
        ),
        "status": "PASS",
        "accepted_for_staging_api_integration": True,
        "release_binding": {
            "path": str(binding_path),
            "sha256": sha256(binding_path),
        },
        "report": {
            "path": str(output / report_path.name),
            "sha256": sha256(report_path),
        },
        "production_deployed": False,
        "release_ready": False,
    }
    audit_binding_path = staging / "INDEPENDENT_AUDIT_BINDING.json"
    write_json(audit_binding_path, audit_binding)
    write_json(
        staging / "SUCCESS.json",
        {
            "format": (
                "CC_HHGT_V3_2_DIRECTIONAL_CNV_WEBSITE_"
                "INDEPENDENT_AUDIT_SUCCESS_V1"
            ),
            "status": "SUCCESS",
            "binding_sha256": sha256(audit_binding_path),
            "report_sha256": sha256(report_path),
            "accepted_for_staging_api_integration": True,
            "production_deployed": False,
            "release_ready": False,
        },
    )
    os.replace(staging, output)
    print(json.dumps({"status": "PASS", "rows": EXPECTED_ROWS, "cancers": 33}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
