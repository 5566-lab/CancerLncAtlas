#!/usr/bin/env python3
"""Independently audit a completed fresh V3.2 ATAC OOF expert."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


FORMAL_CANCERS = (
    "ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA",
    "GBM", "HNSC", "KICH", "KIRC", "KIRP", "LAML", "LGG", "LIHC",
    "LUAD", "LUSC", "MESO", "OV", "PAAD", "PCPG", "PRAD", "READ",
    "SARC", "SKCM", "STAD", "TGCT", "THCA", "THYM", "UCEC", "UCS",
    "UVM",
)
RAW_GAPS = (
    "DLBC", "KICH", "LAML", "OV", "PAAD", "READ", "SARC", "THYM",
    "UCS", "UVM",
)
RAW_COVERED = tuple(value for value in FORMAL_CANCERS if value not in RAW_GAPS)
KEYS = ["cancer_id", "lncrna_id", "pathway_id"]


class AuditError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-success", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    success_path = Path(args.training_success).resolve()
    candidates_path = Path(args.candidates).resolve()
    output = Path(args.output_root).resolve()
    if output.exists():
        raise AuditError(f"ATAC result audit refuses output reuse: {output}")
    success = read_json(success_path)
    if success.get("status") != "SUCCESS" or success.get("success_written_last") is not True:
        raise AuditError("ATAC training success is not a final SUCCESS")
    if success.get("old_predictions_or_checkpoints_used") is not False:
        raise AuditError("ATAC training success permits old results")
    prediction_path = Path(str(success["prediction_path"]))
    lineage_path = Path(str(success["lineage_path"]))
    checkpoint_manifest_path = Path(str(success["checkpoint_manifest_path"]))
    for path in (prediction_path, lineage_path, checkpoint_manifest_path, candidates_path):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)
    prediction_sha = sha256(prediction_path)
    lineage_sha = sha256(lineage_path)
    checkpoint_manifest_sha = sha256(checkpoint_manifest_path)
    if prediction_sha != success.get("prediction_sha256"):
        raise AuditError("ATAC typed prediction SHA differs from SUCCESS")
    if lineage_sha != success.get("lineage_sha256"):
        raise AuditError("ATAC lineage SHA differs from SUCCESS")
    if checkpoint_manifest_sha != success.get("checkpoint_manifest_sha256"):
        raise AuditError("ATAC checkpoint manifest SHA differs from SUCCESS")
    if pq.ParquetFile(prediction_path).metadata.num_rows != 3_300_000:
        raise AuditError("ATAC typed prediction table is not 3,300,000 rows")

    lineage = read_json(lineage_path)
    lineage_checks = {
        "analysis_version_is_v32": str(lineage.get("analysis_version", "")).startswith(
            "CancerLncAtlas_V3.2"
        ),
        "modality_atac": lineage.get("modality") == "atac",
        "folds_five": lineage.get("folds") == 5,
        "patient_level_oof": lineage.get("patient_level_modality_oof") is True,
        "unaveraged_fold_outputs": (
            lineage.get("patient_fold_oof_predictions_not_fold_averaged") is True
        ),
        "outer_test_never_fit": lineage.get(
            "outer_test_patients_used_in_any_upstream_fit"
        ) == 0,
        "no_old_checkpoint": lineage.get("old_checkpoint_loaded") is False,
        "no_old_predictions": lineage.get("old_predictions_used_as_features") is False,
        "no_old_rankings": lineage.get("old_rankings_used_as_outputs") is False,
        "prediction_hash_bound": lineage.get("predictions_sha256") == prediction_sha,
        "no_family_broadcast": lineage.get("exact_pathway_family_broadcast_used") is False,
        "missing_not_zero": lineage.get("missing_atac_assumed_zero") is False,
        "no_positive_claim_before_compare": (
            lineage.get("formal_positive_contribution_claimed") is False
        ),
    }
    if not all(lineage_checks.values()):
        raise AuditError(
            f"ATAC lineage contract failed: {[key for key, value in lineage_checks.items() if not value]}"
        )

    candidate_sha = sha256(candidates_path)
    cancer_rows: list[dict[str, Any]] = []
    for cancer in FORMAL_CANCERS:
        expected = pq.read_table(
            candidates_path, columns=KEYS, filters=[("cancer_id", "=", cancer)]
        ).to_pandas()
        observed = pq.read_table(
            prediction_path,
            columns=KEYS
            + [
                "atac_context_probability",
                "atac_available",
                "atac_unavailable_reason",
                "atac_patient_folds_with_prediction",
                "analysis_version",
                "training_run_id",
                "module_id",
                "changes_primary_ranking",
            ],
            filters=[("cancer_id", "=", cancer)],
        ).to_pandas()
        if len(expected) != 100_000 or len(observed) != 100_000:
            raise AuditError(f"{cancer} does not preserve exact 100,000-row scope")
        if expected.duplicated(KEYS).any() or observed.duplicated(KEYS).any():
            raise AuditError(f"{cancer} contains duplicate exact keys")
        if not expected[KEYS].reset_index(drop=True).equals(observed[KEYS].reset_index(drop=True)):
            raise AuditError(f"{cancer} candidate keys differ from the authority")
        available = observed.atac_available
        if available.isna().any() or not pd.api.types.is_bool_dtype(available):
            raise AuditError(f"{cancer} ATAC availability is not a non-null boolean")
        probability = pd.to_numeric(observed.atac_context_probability, errors="coerce")
        reasons = observed.atac_unavailable_reason.astype("string")
        fold_count = pd.to_numeric(
            observed.atac_patient_folds_with_prediction, errors="raise"
        ).astype(int)
        if probability[available].isna().any() or not probability[available].between(0, 1).all():
            raise AuditError(f"{cancer} available ATAC probability is invalid")
        if probability[~available].notna().any():
            raise AuditError(f"{cancer} unavailable ATAC row has a probability")
        if reasons[available].notna().any() or reasons[~available].isna().any():
            raise AuditError(f"{cancer} typed unavailable reason is invalid")
        if not (fold_count[available].between(1, 5).all() and fold_count[~available].eq(0).all()):
            raise AuditError(f"{cancer} fold prediction count is inconsistent")
        if not observed.analysis_version.eq("CancerLncAtlas_V3.2_FULL_MULTITASK").all():
            raise AuditError(f"{cancer} analysis version drift")
        if not observed.module_id.eq("atac_coaccessibility").all():
            raise AuditError(f"{cancer} module ID drift")
        if observed.changes_primary_ranking.any():
            raise AuditError(f"{cancer} incorrectly changes the primary ranking")
        if cancer in RAW_GAPS:
            if available.any() or set(reasons.dropna()) != {
                "ATAC_RAW_NOT_AVAILABLE_FOR_CANCER"
            }:
                raise AuditError(f"{cancer} raw gap is not an exact typed null")
        elif not available.any():
            raise AuditError(f"{cancer} raw-covered cancer has zero ATAC prediction")
        cancer_rows.append(
            {
                "cancer_id": cancer,
                "raw_covered": cancer in RAW_COVERED,
                "rows": len(observed),
                "available_rows": int(available.sum()),
                "unavailable_rows": int((~available).sum()),
                "probability_min": float(probability[available].min()) if available.any() else None,
                "probability_max": float(probability[available].max()) if available.any() else None,
                "probability_mean": float(probability[available].mean()) if available.any() else None,
                "fold_count_min": int(fold_count[available].min()) if available.any() else 0,
                "fold_count_max": int(fold_count[available].max()) if available.any() else 0,
            }
        )

    checkpoint_manifest = read_json(checkpoint_manifest_path)
    records = checkpoint_manifest.get("records")
    if not isinstance(records, list) or len(records) != 5:
        raise AuditError("ATAC checkpoint manifest does not contain five records")
    fold_records: list[dict[str, Any]] = []
    for record in records:
        fold = int(record["patient_fold"])
        if fold not in range(5) or record.get("status") != "SUCCESS":
            raise AuditError("ATAC checkpoint record is malformed")
        checkpoint_path = Path(str(record["path"]))
        fold_path = Path(str(record["prediction_path"]))
        fold_success_path = Path(str(record["success_path"]))
        fold_success = read_json(fold_success_path)
        checkpoint_sha = sha256(checkpoint_path)
        fold_sha = sha256(fold_path)
        if checkpoint_sha != record.get("sha256") or checkpoint_sha != fold_success.get(
            "checkpoint_sha256"
        ):
            raise AuditError(f"ATAC fold {fold} checkpoint SHA mismatch")
        if fold_sha != record.get("prediction_sha256") or fold_sha != fold_success.get(
            "prediction_sha256"
        ):
            raise AuditError(f"ATAC fold {fold} prediction SHA mismatch")
        if pq.ParquetFile(fold_path).metadata.num_rows != 3_300_000:
            raise AuditError(f"ATAC fold {fold} prediction row count mismatch")
        if fold_success.get("heldout_patients_used_for_fit") is not False:
            raise AuditError(f"ATAC fold {fold} does not prove held-out isolation")
        fold_records.append(
            {
                "patient_fold": fold,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_sha,
                "prediction_path": str(fold_path),
                "prediction_sha256": fold_sha,
                "prediction_rows": 3_300_000,
                "success_path": str(fold_success_path),
                "success_sha256": sha256(fold_success_path),
            }
        )
    if {record["patient_fold"] for record in fold_records} != set(range(5)):
        raise AuditError("ATAC fold set is not exactly 0..4")

    output.mkdir(parents=True)
    audit_path = output / "ATAC_FRESH_OOF_AUDIT.json"
    audit = {
        "format": "CC_HHGT_V3_2_ATAC_FRESH_OOF_INDEPENDENT_AUDIT_V1",
        "status": "PASS",
        "training_success_path": str(success_path),
        "training_success_sha256": sha256(success_path),
        "candidate_authority_path": str(candidates_path),
        "candidate_authority_sha256": candidate_sha,
        "prediction_path": str(prediction_path),
        "prediction_sha256": prediction_sha,
        "prediction_rows": 3_300_000,
        "lineage_path": str(lineage_path),
        "lineage_sha256": lineage_sha,
        "lineage_checks": lineage_checks,
        "formal_cancers": list(FORMAL_CANCERS),
        "raw_covered_cancers": list(RAW_COVERED),
        "typed_raw_gap_cancers": list(RAW_GAPS),
        "cancer_records": cancer_rows,
        "fold_records": sorted(fold_records, key=lambda item: item["patient_fold"]),
        "exact_candidate_keys_match_authority": True,
        "typed_null_never_zero": True,
        "old_predictions_checkpoints_or_rankings_used": False,
        "formal_positive_contribution_claimed": False,
        "release_ready": False,
        "release_reason": "REQUIRES_FAIR_ROUTER_VS_HIERARCHICAL_VALIDATION",
    }
    atomic_json(audit_path, audit)
    atomic_json(
        output / "SUCCESS.json",
        {
            "status": "PASS",
            "audit_path": str(audit_path),
            "audit_sha256": sha256(audit_path),
            "prediction_sha256": prediction_sha,
            "prediction_rows": 3_300_000,
            "covered_cancers_with_predictions": list(RAW_COVERED),
            "typed_raw_gap_cancers": list(RAW_GAPS),
            "success_written_last": True,
        },
    )
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
