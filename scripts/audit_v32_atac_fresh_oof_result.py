#!/usr/bin/env python3
"""Independently audit a completed fresh V3.2 ATAC OOF expert."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
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


def main(argv=None) -> int:
    from cc_hhgt.v32.cnv_input_sources import require_host
    require_host()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-success", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--expected-training-run-id")
    args = parser.parse_args(argv)

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

    lineage = read_json(lineage_path)
    from cc_hhgt.v32.current_core_export import verified_path
    from cc_hhgt.v32.atac_cohort import training_cohort
    input_path = verified_path({"path": lineage["input_lineage_audit_path"],
        "sha256": lineage["input_lineage_audit_sha256"]}, lineage_path.parent)
    input_audit = read_json(input_path)
    cohort = training_cohort(success, lineage, input_audit, input_path.parent, candidates=candidates_path)
    if cohort and not args.expected_training_run_id:
        raise AuditError("Native ATAC audit requires --expected-training-run-id")
    run_id = args.expected_training_run_id or success.get("training_run_id")
    if not run_id or success.get("training_run_id") != run_id or lineage.get("training_run_id") != run_id:
        raise AuditError("ATAC training run identity differs")
    cancers = tuple(cohort["cancers"]) if cohort else FORMAL_CANCERS
    covered = tuple(c for c in cancers if c not in RAW_GAPS)
    gaps = tuple(c for c in cancers if c in RAW_GAPS)
    counts = cohort["candidate_rows_by_cancer"] if cohort else {c: 100_000 for c in cancers}
    total_rows = sum(counts.values())
    for p in (prediction_path, candidates_path):
        if pq.ParquetFile(p).metadata.num_rows != total_rows:
            raise AuditError("ATAC total rows differ from declared candidate scope")
    if cohort and (lineage.get("raw_covered_cancers") != list(covered)
                   or lineage.get("typed_raw_gap_cancers") != list(gaps)):
        raise AuditError("ATAC lineage raw coverage differs from declared scope")
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
    for cancer in cancers:
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
        if len(expected) != counts[cancer] or len(observed) != counts[cancer]:
            raise AuditError(f"{cancer} does not preserve declared candidate scope")
        if expected.duplicated(KEYS).any() or observed.duplicated(KEYS).any():
            raise AuditError(f"{cancer} contains duplicate exact keys")
        if not expected[KEYS].reset_index(drop=True).equals(observed[KEYS].reset_index(drop=True)):
            raise AuditError(f"{cancer} candidate keys differ from the authority")
        if not observed.training_run_id.eq(run_id).all(): raise AuditError("ATAC typed rows have a different run")
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
        checkpoint = read_json(checkpoint_path)
        if (checkpoint.get("training_run_id") != run_id or checkpoint.get("patient_fold") != fold
                or checkpoint.get("outer_test_patients_used_for_fit") is not False
                or checkpoint.get("cohort") != cohort or fold_success.get("cohort") != cohort):
            raise AuditError("ATAC checkpoint fold/run/cohort differs")
        if cohort and (checkpoint.get("input_hashes", {}).get("candidates") != candidate_sha
                       or checkpoint.get("input_hashes", {}).get("folds") != cohort["patient_map"]["sha256"]
                       or fold_success.get("training_run_id") != run_id):
            raise AuditError("ATAC checkpoint input/fold-success run differs")
        if sha256(fold_success_path) != record.get("success_sha256"):
            raise AuditError("ATAC fold success SHA mismatch")
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
        if pq.ParquetFile(fold_path).metadata.num_rows != total_rows:
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
                "prediction_rows": total_rows,
                "success_path": str(fold_success_path),
                "success_sha256": sha256(fold_success_path),
            }
        )
    if {record["patient_fold"] for record in fold_records} != set(range(5)):
        raise AuditError("ATAC fold set is not exactly 0..4")

    # Independently reconcile all five key/fold tables and the published mean.
    # Process one cancer at a time, bounded by its declared candidate budget.
    for cancer in cancers:
        expected = pq.read_table(candidates_path, columns=KEYS, filters=[("cancer_id", "=", cancer)]).to_pandas()
        values = []
        for record in fold_records:
            frame = pq.read_table(record["prediction_path"], filters=[("cancer_id", "=", cancer)]).to_pandas()
            if (not frame[KEYS].reset_index(drop=True).equals(expected.reset_index(drop=True))
                    or not frame.patient_fold_id.eq(record["patient_fold"]).all()):
                raise AuditError("ATAC fold keys/identity differ from authority")
            available = frame.atac_available
            probability = pd.to_numeric(frame.atac_context_probability, errors="raise")
            reason = frame.atac_unavailable_reason.astype("string")
            if (not pd.api.types.is_bool_dtype(available) or available.isna().any()
                    or not probability[available].between(0, 1).all() or probability[~available].notna().any()
                    or reason[available].notna().any() or reason[~available].isna().any()):
                raise AuditError("ATAC fold typed availability/probability differs")
            if cancer in gaps and (available.any() or not reason.eq("ATAC_RAW_NOT_AVAILABLE_FOR_CANCER").all()):
                raise AuditError("ATAC fold raw gap is not typed null")
            values.append(probability.to_numpy(float))
        matrix = np.stack(values); n = np.isfinite(matrix).sum(axis=0)
        mean = np.divide(np.nansum(matrix, axis=0), n, out=np.full(len(expected), np.nan), where=n > 0)
        typed = pq.read_table(prediction_path, filters=[("cancer_id", "=", cancer)]).to_pandas()
        if (not np.array_equal(n, typed.atac_patient_folds_with_prediction)
                or not np.allclose(mean, typed.atac_context_probability, atol=1e-7, rtol=1e-6, equal_nan=True)):
            raise AuditError("ATAC typed mean/count differs from actual five fold tables")

    output.mkdir(parents=True)
    atomic_json(output / "AUDIT_PREFLIGHT.json", {"target_host": "149", "training_started": False,
        "training_run_id": run_id, "cohort": cohort})
    audit_path = output / "ATAC_FRESH_OOF_AUDIT.json"
    audit = {
        "format": "CC_HHGT_V3_2_ATAC_FRESH_OOF_INDEPENDENT_AUDIT_V1",
        "status": "PASS",
        "training_success_path": str(success_path),
        "cohort": cohort, "training_run_id": run_id,
        "training_success_sha256": sha256(success_path),
        "candidate_authority_path": str(candidates_path),
        "candidate_authority_sha256": candidate_sha,
        "prediction_path": str(prediction_path),
        "prediction_sha256": prediction_sha,
        "prediction_rows": total_rows,
        "lineage_path": str(lineage_path),
        "lineage_sha256": lineage_sha,
        "lineage_checks": lineage_checks,
        "formal_cancers": list(cancers),
        "raw_covered_cancers": list(covered),
        "typed_raw_gap_cancers": list(gaps),
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
            "prediction_rows": total_rows,
            "covered_cancers_with_predictions": list(covered),
            "typed_raw_gap_cancers": list(gaps),
            "success_written_last": True,
        },
    )
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
