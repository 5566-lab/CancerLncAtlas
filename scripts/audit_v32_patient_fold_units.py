#!/usr/bin/env python3
"""Audit whether a nominal V3.2 patient-fold manifest is actually patient-safe."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb


FORMAT = "CANCERLNCATLAS_V32_PATIENT_FOLD_UNIT_AUDIT_V1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _activity_inventory(pattern: str) -> tuple[list[dict[str, object]], str]:
    paths = sorted(Path(item).resolve() for item in glob.glob(pattern, recursive=True))
    if not paths:
        raise RuntimeError("The activity glob matched no files")
    common = Path(paths[0]).parent
    while not all(path.is_relative_to(common) for path in paths):
        common = common.parent
    rows: list[dict[str, object]] = []
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"Unsafe activity input: {path}")
        rows.append(
            {
                "path": path.relative_to(common).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return rows, hashlib.sha256(canonical).hexdigest()


def audit(activity_glob: str, fold_manifest: Path, *, memory_limit: str) -> dict:
    manifest = fold_manifest.resolve()
    if manifest.is_symlink() or not manifest.is_file():
        raise RuntimeError(f"Unsafe fold manifest: {manifest}")
    inventory, tree_sha256 = _activity_inventory(activity_glob)
    connection = duckdb.connect()
    connection.execute("SET threads=1")
    connection.execute(f"SET memory_limit='{memory_limit}'")
    base = """
        WITH sp AS (
          SELECT cancer_id::VARCHAR AS cancer_id,
                 sample_id::VARCHAR AS sample_id,
                 patient_id::VARCHAR AS patient_id
          FROM read_parquet(?)
          GROUP BY ALL
        ), pf AS (
          SELECT cancer_id::VARCHAR AS cancer_id,
                 sample_id::VARCHAR AS sample_id,
                 patient_fold_id::INTEGER AS patient_fold_id
          FROM read_csv_auto(?, delim='\\t', header=true)
        ), joined AS (
          SELECT sp.*, pf.patient_fold_id
          FROM sp LEFT JOIN pf USING(cancer_id, sample_id)
        ), patient_groups AS (
          SELECT cancer_id, patient_id,
                 count(DISTINCT sample_id) AS n_samples,
                 count(DISTINCT patient_fold_id) AS n_folds
          FROM joined GROUP BY ALL
        )
    """
    summary_query = base + """
        SELECT
          (SELECT count(*) FROM sp) AS distinct_cancer_sample_patient_rows,
          (SELECT count(DISTINCT cancer_id || '|' || sample_id) FROM sp) AS cancer_sample_keys,
          (SELECT count(DISTINCT cancer_id || '|' || patient_id) FROM sp) AS cancer_patient_keys,
          (SELECT count(*) FROM (
             SELECT cancer_id, sample_id FROM sp GROUP BY ALL
             HAVING count(DISTINCT patient_id) != 1
           )) AS ambiguous_sample_patient_keys,
          (SELECT count(*) FROM patient_groups WHERE n_samples > 1) AS patients_with_multiple_samples,
          (SELECT coalesce(sum(n_samples - 1), 0) FROM patient_groups WHERE n_samples > 1) AS excess_samples,
          (SELECT count(*) FROM patient_groups WHERE n_folds > 1) AS patients_crossing_folds,
          (SELECT coalesce(max(n_samples), 0) FROM patient_groups) AS max_samples_per_patient,
          (SELECT count(*) FROM joined WHERE patient_fold_id IS NULL) AS samples_missing_fold,
          (SELECT count(*) FROM pf) AS fold_manifest_rows,
          (SELECT count(*) FROM (
             SELECT cancer_id, sample_id FROM pf GROUP BY ALL HAVING count(*) != 1
           )) AS duplicate_fold_keys,
          (SELECT count(*) FROM pf LEFT JOIN sp USING(cancer_id, sample_id)
             WHERE sp.sample_id IS NULL) AS fold_rows_without_activity
    """
    row = connection.execute(summary_query, [activity_glob, manifest.as_posix()]).fetchone()
    summary = dict(zip([item[0] for item in connection.description], row, strict=True))
    examples_query = base + """
        SELECT cancer_id, patient_id,
               list(sample_id ORDER BY sample_id) AS sample_ids,
               list(DISTINCT patient_fold_id ORDER BY patient_fold_id) AS patient_fold_ids
        FROM joined
        GROUP BY cancer_id, patient_id
        HAVING count(DISTINCT sample_id) > 1 OR count(DISTINCT patient_fold_id) > 1
        ORDER BY cancer_id, patient_id
        LIMIT 20
    """
    examples = connection.execute(
        examples_query, [activity_glob, manifest.as_posix()]
    ).fetchall()
    connection.close()
    hard_failures = {
        key: int(summary[key])
        for key in (
            "ambiguous_sample_patient_keys",
            "patients_crossing_folds",
            "samples_missing_fold",
            "duplicate_fold_keys",
            "fold_rows_without_activity",
        )
        if int(summary[key]) != 0
    }
    one_to_one = (
        int(summary["cancer_sample_keys"]) == int(summary["cancer_patient_keys"])
        and int(summary["patients_with_multiple_samples"]) == 0
        and int(summary["max_samples_per_patient"]) == 1
    )
    return {
        "format": FORMAT,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_CURRENT_INPUT_ONE_SAMPLE_PER_PATIENT" if not hard_failures and one_to_one else "FAIL_PATIENT_FOLD_UNIT",
        "classification": "LATENT_SAMPLE_KEYED_PROCESSING_CONTRACT_RISK_NOT_CURRENT_DATA_GAP" if not hard_failures and one_to_one else "ACTIVE_PATIENT_SPLIT_PROCESSING_BUG",
        "inputs": {
            "activity_glob": activity_glob,
            "activity_files": len(inventory),
            "activity_bytes": sum(int(item["bytes"]) for item in inventory),
            "activity_tree_sha256": tree_sha256,
            "activity_inventory": inventory,
            "fold_manifest": manifest.as_posix(),
            "fold_manifest_bytes": manifest.stat().st_size,
            "fold_manifest_sha256": _sha256_file(manifest),
        },
        "observed": summary,
        "multi_sample_or_cross_fold_examples": [
            {
                "cancer_id": cancer,
                "patient_id": patient,
                "sample_ids": samples,
                "patient_fold_ids": folds,
            }
            for cancer, patient, samples, folds in examples
        ],
        "gates": {
            "hard_failures": hard_failures,
            "current_activity_is_one_sample_per_patient": one_to_one,
            "current_patients_crossing_folds": int(summary["patients_crossing_folds"]) == 0,
            "current_result_may_be_called_patient_safe": not hard_failures and one_to_one,
            "sample_only_fold_builder_is_future_safe": False,
        },
        "required_remediation": [
            "Pass an explicit patient_id into the fresh fold builder and hash patient identity, not only sample_id.",
            "Assign every sample from one patient to exactly one fold and reject patient or cancer crossover.",
            "Retain the current one-sample-per-patient observation as an input-specific PASS, not a general contract waiver.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activity-glob", required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--memory-limit", default="1GB")
    args = parser.parse_args()
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    result = audit(args.activity_glob, args.fold_manifest, memory_limit=args.memory_limit)
    output = destination / "AUDIT.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "audit": output.as_posix()}))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
