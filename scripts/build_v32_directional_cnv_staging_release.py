#!/usr/bin/env python3
"""Build an audited website asset from the directional CNV-only OOF.

The source contains five candidate-level OOF predictions per cancer/lncRNA/
exact-pathway key.  This release averages only available fold predictions and
keeps zero available folds as a typed null.  It never reads the superseded
combined genomic CNV score or any Mutation feature.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

import duckdb
import pandas as pd


CANCERS = (
    "ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA",
    "GBM", "HNSC", "KICH", "KIRC", "KIRP", "LAML", "LGG", "LIHC",
    "LUAD", "LUSC", "MESO", "OV", "PAAD", "PCPG", "PRAD", "READ",
    "SARC", "SKCM", "STAD", "TGCT", "THCA", "THYM", "UCEC", "UCS",
    "UVM",
)
SOURCE_MANIFEST_FORMAT = "CC_HHGT_V3_2_DIRECTIONAL_CNV_HEAD_OOF_V2"
SOURCE_SUCCESS_FORMAT = "CC_HHGT_V3_2_DIRECTIONAL_CNV_HEAD_OOF_FINAL_V1"
SOURCE_AUDIT_FORMAT = "CC_HHGT_V3_2_CNV_OOF_INDEPENDENT_AUDIT_V1"
LOCAL_SUCCESS_FORMAT = "CC_HHGT_V3_2_LOCAL_CNV_FINAL_SUCCESS_V1"
EXPECTED_SOURCE_ROWS = 16_500_000
EXPECTED_RELEASE_ROWS = 3_300_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} is missing or unsafe: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
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


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof-root", required=True, type=Path)
    parser.add_argument("--local-audit-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--memory-limit", default="64GB")
    parser.add_argument(
        "--work-root",
        type=Path,
        help="Optional local filesystem used for construction before verified copy",
    )
    args = parser.parse_args()

    oof = args.oof_root.resolve(strict=True)
    local = args.local_audit_root.resolve(strict=True)
    output = args.output_root.resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"immutable directional CNV website release exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    work_parent = (
        args.work_root.resolve()
        if args.work_root is not None
        else output.parent
    )
    work_parent.mkdir(parents=True, exist_ok=True)
    staging = work_parent / f".{output.name}.partial.{os.getpid()}"
    if staging.exists() or staging.is_symlink():
        raise RuntimeError(f"staging path already exists: {staging}")
    staging.mkdir()

    success_path = oof / "SUCCESS.json"
    audit_path = oof / "CNV_OOF_INDEPENDENT_AUDIT.json"
    manifest_path = oof / "OOF_MANIFEST.json"
    local_success_path = local / "SUCCESS.json"
    source_success = load_json(success_path, "directional CNV SUCCESS")
    source_audit = load_json(audit_path, "directional CNV independent audit")
    source_manifest = load_json(manifest_path, "directional CNV OOF manifest")
    local_success = load_json(local_success_path, "local-CNV audit SUCCESS")
    if (
        source_success.get("format") != SOURCE_SUCCESS_FORMAT
        or source_success.get("status") != "SUCCESS"
        or source_success.get("CNV_HEAD_OOF_COMPLETE") is not True
    ):
        raise RuntimeError("directional CNV source SUCCESS contract drift")
    if (
        source_audit.get("format") != SOURCE_AUDIT_FORMAT
        or source_audit.get("status") != "PASS"
    ):
        raise RuntimeError("directional CNV independent audit did not pass")
    if (
        source_manifest.get("format") != SOURCE_MANIFEST_FORMAT
        or source_manifest.get("status") != "READY_FOR_INDEPENDENT_AUDIT"
        or source_manifest.get("cnv_only") is not True
        or source_manifest.get("mutation_features_used") is not False
        or source_manifest.get("target")
        != "current_fold_local_unadjusted_lncrna_exact_pathway_association"
    ):
        raise RuntimeError("directional CNV OOF manifest contract drift")
    if (
        local_success.get("format") != LOCAL_SUCCESS_FORMAT
        or local_success.get("status") != "SUCCESS"
    ):
        raise RuntimeError("local-CNV confounding audit is not closed")
    if source_success.get("independent_audit_sha256") != sha256(audit_path):
        raise RuntimeError("directional CNV SUCCESS does not bind the independent audit")
    if source_audit.get("oof_manifest_sha256") != sha256(manifest_path):
        raise RuntimeError("directional CNV audit does not bind the OOF manifest")

    prediction_records = source_manifest.get("predictions")
    if not isinstance(prediction_records, list) or len(prediction_records) != 165:
        raise RuntimeError("directional CNV manifest must contain 165 prediction records")
    expected_partitions = {(fold, cancer) for fold in range(5) for cancer in CANCERS}
    observed_partitions: set[tuple[int, str]] = set()
    prediction_paths: list[str] = []
    source_rows_declared = 0
    for record in prediction_records:
        fold, cancer = int(record.get("fold", -1)), str(record.get("cancer", ""))
        if (fold, cancer) in observed_partitions:
            raise RuntimeError(f"duplicate directional CNV partition: {fold}/{cancer}")
        observed_partitions.add((fold, cancer))
        path = Path(str(record.get("path", ""))).resolve(strict=True)
        if path.is_symlink() or sha256(path) != str(record.get("sha256", "")):
            raise RuntimeError(f"directional CNV prediction hash drift: {fold}/{cancer}")
        prediction_paths.append(str(path))
        source_rows_declared += int(record.get("rows", -1))
    if observed_partitions != expected_partitions:
        raise RuntimeError("directional CNV prediction partition closure drift")
    if source_rows_declared != EXPECTED_SOURCE_ROWS:
        raise RuntimeError(f"directional CNV declared row count drift: {source_rows_declared}")

    connection = duckdb.connect()
    connection.execute(f"SET memory_limit={sql_literal(args.memory_limit)}")
    connection.execute("SET threads=8")
    connection.read_parquet(prediction_paths).create_view("source")
    source_check = connection.execute(
        """
        SELECT count(*) AS rows,
               count(DISTINCT (cancer_id, lncrna_id, pathway_id)) AS keys,
               min(folds_per_key) AS min_folds,
               max(folds_per_key) AS max_folds
        FROM (
          SELECT cancer_id, lncrna_id, pathway_id, count(*) AS folds_per_key
          FROM source GROUP BY ALL
        )
        """
    ).fetchone()
    # The outer count is a key count, so independently inspect raw source rows.
    raw_rows = int(connection.execute("SELECT count(*) FROM source").fetchone()[0])
    if raw_rows != EXPECTED_SOURCE_ROWS or tuple(map(int, source_check[1:])) != (
        EXPECTED_RELEASE_ROWS,
        5,
        5,
    ):
        raise RuntimeError(
            "directional CNV source key/fold closure drift: "
            f"raw={raw_rows}, summary={source_check}"
        )
    cancers = tuple(
        row[0]
        for row in connection.execute(
            "SELECT DISTINCT cancer_id FROM source ORDER BY cancer_id"
        ).fetchall()
    )
    if cancers != CANCERS:
        raise RuntimeError(f"directional CNV cancer universe drift: {cancers}")
    invalid = connection.execute(
        """
        SELECT
          count_if(cnv_available AND
                   (cnv_probability IS NULL OR cnv_probability < 0 OR cnv_probability > 1)),
          count_if(NOT cnv_available AND cnv_probability IS NOT NULL),
          count_if(cnv_available AND nullif(trim(unavailable_reason), '') IS NOT NULL),
          count_if(NOT cnv_available AND nullif(trim(unavailable_reason), '') IS NULL),
          count_if(run_id IS NULL OR model_version != 'V3.2')
        FROM source
        """
    ).fetchone()
    if any(int(value) != 0 for value in invalid):
        raise RuntimeError(f"directional CNV typed-availability contract failed: {invalid}")

    prediction_path = staging / "directional_cnv_typed_predictions.parquet"
    connection.execute(
        f"""
        COPY (
          SELECT cancer_id, lncrna_id, pathway_id,
                 avg(cnv_probability) FILTER (WHERE cnv_available)
                   AS cnv_context_probability,
                 stddev_samp(cnv_probability) FILTER (WHERE cnv_available)
                   AS cnv_context_probability_sd,
                 min(cnv_probability) FILTER (WHERE cnv_available)
                   AS cnv_context_probability_min,
                 max(cnv_probability) FILTER (WHERE cnv_available)
                   AS cnv_context_probability_max,
                 count_if(cnv_available)::INTEGER AS cnv_patient_folds_with_prediction,
                 count(*)::INTEGER AS cnv_patient_fold_count,
                 count_if(cnv_available) > 0 AS cnv_available,
                 CASE WHEN count_if(cnv_available) > 0 THEN NULL
                      ELSE string_agg(DISTINCT nullif(trim(unavailable_reason), ''), ';'
                                      ORDER BY nullif(trim(unavailable_reason), ''))
                 END AS cnv_unavailable_reason,
                 count_if(lncrna_local_cnv_available)::INTEGER
                   AS local_cnv_available_fold_count,
                 count_if(pathway_cnv_available)::INTEGER
                   AS pathway_cnv_available_fold_count,
                 sum(callable_patient_n)::BIGINT AS cnv_pair_callable_patients_across_oof,
                 first(run_id) AS training_run_id,
                 'CancerLncAtlas_V3.2_FULL_MULTITASK' AS analysis_version,
                 'cnv' AS module_id,
                 'cancer_x_lncrna_x_exact_pathway_cnv_context' AS target_level,
                 'CC_HHGT_V3_2_DIRECTIONAL_CNV_TYPED_PREDICTIONS_V1' AS prediction_format,
                 false AS changes_primary_ranking
          FROM source
          GROUP BY cancer_id, lncrna_id, pathway_id
          ORDER BY cancer_id, lncrna_id, pathway_id
        ) TO {sql_literal(str(prediction_path))}
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
    )
    release = connection.read_parquet(str(prediction_path))
    release.create_view("release")
    release_check = connection.execute(
        """
        SELECT count(*), count(DISTINCT cancer_id),
               count_if(cnv_available AND cnv_context_probability IS NULL),
               count_if(NOT cnv_available AND cnv_context_probability IS NOT NULL),
               count_if(cnv_patient_fold_count != 5),
               count_if(cnv_patient_folds_with_prediction < 0 OR
                        cnv_patient_folds_with_prediction > 5)
        FROM release
        """
    ).fetchone()
    if tuple(map(int, release_check)) != (EXPECTED_RELEASE_ROWS, 33, 0, 0, 0, 0):
        raise RuntimeError(f"directional CNV release validation failed: {release_check}")
    coverage = connection.execute(
        """
        SELECT cancer_id, count(*) AS candidate_rows,
               count_if(cnv_available) AS available_rows,
               count_if(NOT cnv_available) AS typed_unavailable_rows,
               avg(cnv_patient_folds_with_prediction) AS mean_available_folds,
               avg(cnv_pair_callable_patients_across_oof) AS mean_pair_callable_patients
        FROM release GROUP BY cancer_id ORDER BY cancer_id
        """
    ).fetchdf()
    if tuple(coverage.cancer_id.astype(str)) != CANCERS or not (
        coverage.candidate_rows.astype(int) == 100_000
    ).all():
        raise RuntimeError("directional CNV website coverage summary drift")
    coverage_path = staging / "directional_cnv_coverage_33c.parquet"
    coverage.to_parquet(coverage_path, index=False, compression="zstd")
    coverage.to_csv(staging / "directional_cnv_coverage_33c.tsv", sep="\t", index=False)
    connection.close()

    created = datetime.now(timezone.utc).isoformat()
    transformation_audit = {
        "format": "CC_HHGT_V3_2_DIRECTIONAL_CNV_WEBSITE_TRANSFORMATION_AUDIT_V1",
        "status": "PASS",
        "audited_at_utc": created,
        "checks": {
            "source_rows": raw_rows,
            "release_rows": EXPECTED_RELEASE_ROWS,
            "cancers": 33,
            "folds_per_key": 5,
            "prediction_partitions": 165,
            "source_prediction_hashes_recomputed": True,
            "source_independent_audit_bound": True,
            "signed_directional_source_only": True,
            "mutation_features_used": False,
            "superseded_combined_cnv_read": False,
            "available_probability_finite": True,
            "typed_unavailable_probability_null": True,
            "primary_ranking_changed": False,
        },
        "source_success_sha256": sha256(success_path),
        "source_audit_sha256": sha256(audit_path),
        "source_manifest_sha256": sha256(manifest_path),
        "local_cnv_success_sha256": sha256(local_success_path),
        "prediction_sha256": sha256(prediction_path),
        "coverage_sha256": sha256(coverage_path),
    }
    transformation_audit_output = staging / "TRANSFORMATION_AUDIT.json"
    write_json(transformation_audit_output, transformation_audit)
    lineage = {
        "format": "CC_HHGT_V3_2_DIRECTIONAL_CNV_WEBSITE_LINEAGE_V1",
        "status": "PASS",
        "generated_at_utc": created,
        "source_oof_root": str(oof),
        "source_local_cnv_audit_root": str(local),
        "source_manifest_sha256": transformation_audit["source_manifest_sha256"],
        "source_independent_audit_sha256": transformation_audit["source_audit_sha256"],
        "transformation_code_path": str(Path(__file__).resolve()),
        "transformation_code_sha256": sha256(Path(__file__).resolve()),
        "aggregation": "mean_of_available_patient_fold_oof_probabilities",
        "target": source_manifest["target"],
        "mutation_features_used": False,
        "primary_ranking_changed": False,
    }
    lineage_path = staging / "LINEAGE.json"
    write_json(lineage_path, lineage)
    binding = {
        "format": "CC_HHGT_V3_2_DIRECTIONAL_CNV_WEBSITE_BINDING_V1",
        "status": "PASS",
        "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
        "staging_queryable": True,
        "release_ready": False,
        "release_blocker": "CNV_ROUTER_INCREMENT_NOT_TESTED",
        "changes_primary_ranking": False,
        "artifacts": {
            "predictions": {
                "path": str(output / prediction_path.name),
                "sha256": sha256(prediction_path),
                "rows": EXPECTED_RELEASE_ROWS,
            },
            "coverage": {
                "path": str(output / coverage_path.name),
                "sha256": sha256(coverage_path),
                "rows": 33,
            },
            "lineage": {
                "path": str(output / lineage_path.name),
                "sha256": sha256(lineage_path),
            },
            "transformation_audit": {
                "path": str(output / transformation_audit_output.name),
                "sha256": sha256(transformation_audit_output),
            },
        },
    }
    binding_path = staging / "DIRECTIONAL_CNV_WEBSITE_BINDING.json"
    write_json(binding_path, binding)
    (staging / "README.md").write_text(
        "# V3.2 directional CNV website asset\n\n"
        "This staging asset contains 3.3 million cancer × lncRNA × exact-pathway "
        "rows aggregated from five independently audited directional CNV-only OOF "
        "predictions. Typed-unavailable probabilities remain null. The asset is an "
        "independent evidence head and does not yet change the primary ranking.\n",
        encoding="utf-8",
    )
    final_success = {
        "format": "CC_HHGT_V3_2_DIRECTIONAL_CNV_WEBSITE_SUCCESS_V1",
        "status": "SUCCESS",
        "binding_sha256": sha256(binding_path),
        "transformation_audit_sha256": sha256(transformation_audit_output),
        "prediction_sha256": sha256(prediction_path),
        "success_written_last": True,
        "production_deployed": False,
        "release_ready": False,
    }
    write_json(staging / "SUCCESS.json", final_success)
    if staging.parent == output.parent:
        os.replace(staging, output)
    else:
        transfer = output.with_name(f".{output.name}.transfer.{os.getpid()}")
        if transfer.exists() or transfer.is_symlink():
            raise RuntimeError(f"directional CNV transfer path exists: {transfer}")
        shutil.copytree(staging, transfer)
        copied_binding = load_json(
            transfer / binding_path.name,
            "copied directional CNV website binding",
        )
        copied_success = load_json(
            transfer / "SUCCESS.json",
            "copied directional CNV website SUCCESS",
        )
        copied_artifacts = copied_binding.get("artifacts")
        if not isinstance(copied_artifacts, dict):
            raise RuntimeError("copied directional CNV binding lacks artifacts")
        for name, record in copied_artifacts.items():
            if not isinstance(record, dict):
                raise RuntimeError(f"copied directional CNV {name} declaration drift")
            declared = Path(str(record.get("path", "")))
            if declared.parent != output:
                raise RuntimeError(
                    f"copied directional CNV {name} final path drift: {declared}"
                )
            copied_path = transfer / declared.name
            if not copied_path.is_file() or sha256(copied_path) != record.get("sha256"):
                raise RuntimeError(f"copied directional CNV {name} hash drift")
        if copied_success.get("binding_sha256") != sha256(transfer / binding_path.name):
            raise RuntimeError("copied directional CNV binding receipt drift")
        os.replace(transfer, output)
        shutil.rmtree(staging)
    print(json.dumps(final_success, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
