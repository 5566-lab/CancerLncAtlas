#!/usr/bin/env python3
"""Run fresh V3.2 entity-level clinical association calculations."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.common import file_sha256, write_json
from cc_hhgt.v32.clinical_entity_training import (
    ANALYSIS_VERSION,
    CALCULATION_FORMAT,
    HISTORICAL_STATE_IDS,
    SUBJECT_SPECS,
    analyse_entity_matrix,
    fold_metrics,
    input_path_sha256,
    json_sha256,
    load_long_entity_matrix,
    patient_fold_manifest,
    standardize_tcga_cdr_workbook,
    summarise_oof_associations,
    validate_entity_clinical_contract,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Five-fold V3.2 lncRNA/exact-pathway/State clinical associations"
    )
    parser.add_argument("--tcga-cdr", required=True)
    parser.add_argument("--patient-folds", required=True)
    parser.add_argument("--patient-fold-authority-receipt", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--lncrna-expression-root", required=True)
    parser.add_argument("--pathway-activity-root", required=True)
    parser.add_argument("--state-measurements", required=True)
    parser.add_argument("--v32-state-lineage", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--training-run-id", required=True)
    parser.add_argument(
        "--subject-types", default="lncRNA,exact_pathway,state",
        help="Comma-separated subset of lncRNA,exact_pathway,state",
    )
    parser.add_argument("--cancers", default=None, help="Comma-separated subset; default all")
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--min-train-patients", type=int, default=30)
    parser.add_argument("--min-train-events", type=int, default=5)
    parser.add_argument("--min-validation-patients", type=int, default=10)
    parser.add_argument("--min-validation-events", type=int, default=2)
    parser.add_argument("--min-test-patients", type=int, default=10)
    parser.add_argument("--min-test-events", type=int, default=2)
    parser.add_argument("--min-train-measurements", type=int, default=20)
    parser.add_argument("--min-folds-available", type=int, default=3)
    parser.add_argument(
        "--max-entities", type=int, default=None,
        help="Explicit smoke-test cap; lineage will set full_entity_universe=false",
    )
    return parser


def _state_frame(path: Path) -> pd.DataFrame:
    columns = ["cancer_id", "patient_id", "state_id", "state_value"]
    try:
        frame = pd.read_parquet(
            path, columns=columns,
            filters=[("state_id", "in", list(HISTORICAL_STATE_IDS))],
        )
    except (TypeError, ValueError):
        frame = pd.read_parquet(path, columns=columns)
        frame = frame.loc[frame.state_id.astype(str).isin(HISTORICAL_STATE_IDS)]
    if frame.patient_id.isna().any():
        raise ValueError("State measurements contain null explicit patient_id")
    frame["patient_id"] = frame.patient_id.astype(str).str.strip()
    if frame.patient_id.eq("").any():
        raise ValueError("State measurements contain empty explicit patient_id")
    frame["cancer_id"] = frame.cancer_id.astype("string")
    frame["state_id"] = frame.state_id.astype(str)
    frame["state_value"] = pd.to_numeric(frame.state_value, errors="coerce")
    return frame.loc[frame.state_id.isin(HISTORICAL_STATE_IDS)].dropna(
        subset=["cancer_id", "patient_id", "state_id", "state_value"]
    )


def _measurement_matrix(
    subject_type: str,
    cancer_id: str,
    *,
    lncrna_root: Path,
    pathway_root: Path,
    state: pd.DataFrame,
    entity_universe: list[str],
) -> pd.DataFrame:
    if subject_type == "lncRNA":
        matrix = load_long_entity_matrix(
            lncrna_root / f"cancer_id={cancer_id}" / "part-0.parquet",
            cancer_id=cancer_id, entity_column="lncrna_id", value_column="logcpm",
        )
        return matrix.reindex(columns=entity_universe)
    if subject_type == "exact_pathway":
        matrix = load_long_entity_matrix(
            pathway_root / f"cancer_id={cancer_id}" / "part-0.parquet",
            cancer_id=cancer_id, entity_column="pathway_id", value_column="activity_score",
        )
        return matrix.reindex(columns=entity_universe)
    subset = state.loc[state.cancer_id.astype(str).eq(str(cancer_id))]
    wide = subset.pivot_table(
        index="patient_id", columns="state_id", values="state_value", aggfunc="mean"
    ).reindex(columns=list(HISTORICAL_STATE_IDS))
    # Preserve all seven historical subjects even when one measurement is
    # completely absent in a cancer; it will become an explicit null row.
    return wide.sort_index(axis=0).reindex(columns=entity_universe)


def _write_part(frame: pd.DataFrame, root: Path, cancer: str, subject_type: str) -> Path:
    target = root / f"cancer_id={cancer}" / f"subject_type={subject_type}" / "part-0.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target, index=False, compression="zstd")
    return target


def _input_artifact(path: Path, kind: str, usage: str) -> dict[str, str]:
    return {
        "path": str(path.resolve()),
        "sha256": input_path_sha256(path),
        "artifact_kind": kind,
        "usage": usage,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cdr_path = Path(args.tcga_cdr)
    fold_path = Path(args.patient_folds)
    fold_receipt_path = Path(args.patient_fold_authority_receipt)
    candidate_path = Path(args.candidates)
    lncrna_root = Path(args.lncrna_expression_root)
    pathway_root = Path(args.pathway_activity_root)
    state_path = Path(args.state_measurements)
    state_lineage_path = Path(args.v32_state_lineage)
    output = Path(args.output_root)
    from cc_hhgt.v32.patient_fold_authority import (
        validate_frozen_v32_patient_fold_binding,
    )

    patient_authority_audit = validate_frozen_v32_patient_fold_binding(
        fold_path, fold_receipt_path
    )
    output.mkdir(parents=True, exist_ok=True)

    subject_types = tuple(value.strip() for value in args.subject_types.split(",") if value.strip())
    unknown = sorted(set(subject_types) - set(SUBJECT_SPECS))
    if unknown:
        raise ValueError(f"Unknown subject types: {unknown}")
    endpoints, covariates = standardize_tcga_cdr_workbook(cdr_path)
    sample_folds = pd.read_csv(fold_path, sep="\t")
    folds = patient_fold_manifest(sample_folds)
    candidates = pd.read_parquet(
        candidate_path, columns=["cancer_id", "lncrna_id", "pathway_id"]
    ).drop_duplicates()
    candidates["cancer_id"] = candidates.cancer_id.astype(str)
    candidates["lncrna_id"] = candidates.lncrna_id.astype(str)
    candidates["pathway_id"] = candidates.pathway_id.astype(str)
    available_cancers = sorted(set(folds.cancer_id.astype(str)))
    if args.cancers:
        requested = {value.strip() for value in args.cancers.split(",") if value.strip()}
        cancers = sorted(set(available_cancers).intersection(requested))
    else:
        cancers = available_cancers
    if not cancers:
        raise ValueError("No requested cancer is present in the V3.2 fold manifest")

    state_lineage = json.loads(state_lineage_path.read_text(encoding="utf-8"))
    if not str(state_lineage.get("analysis_version", "")).startswith("CancerLncAtlas_V3.2"):
        raise ValueError("State generation anchor is not V3.2")
    if state_lineage.get("old_checkpoint_loaded") is not False:
        raise ValueError("State generation anchor does not exclude old checkpoints")
    state = _state_frame(state_path) if "state" in subject_types else pd.DataFrame()

    all_summary: list[pd.DataFrame] = []
    all_metrics: list[pd.DataFrame] = []
    parts: list[dict[str, object]] = []
    for cancer in cancers:
        cancer_folds = folds.loc[folds.cancer_id.astype(str).eq(cancer)].copy()
        cancer_folds = cancer_folds.drop_duplicates("patient_id").set_index("patient_id")
        for subject_type in subject_types:
            candidate_subset = candidates.loc[candidates.cancer_id.eq(cancer)]
            if subject_type == "lncRNA":
                entity_universe = sorted(candidate_subset.lncrna_id.unique().tolist())
            elif subject_type == "exact_pathway":
                entity_universe = sorted(candidate_subset.pathway_id.unique().tolist())
            else:
                entity_universe = list(HISTORICAL_STATE_IDS)
            try:
                wide = _measurement_matrix(
                    subject_type, cancer,
                    lncrna_root=lncrna_root, pathway_root=pathway_root, state=state,
                    entity_universe=entity_universe,
                )
            except Exception as exc:
                parts.append({
                    "cancer_id": cancer, "subject_type": subject_type,
                    "status": "INPUT_UNAVAILABLE", "failure_reason": repr(exc),
                    "entities": 0, "patients": 0,
                })
                continue
            if args.max_entities is not None:
                wide = wide.iloc[:, : int(args.max_entities)]
            patients = cancer_folds.index.intersection(wide.index.astype(str), sort=False)
            if len(patients) == 0 or wide.shape[1] == 0:
                parts.append({
                    "cancer_id": cancer, "subject_type": subject_type,
                    "status": "INPUT_UNAVAILABLE", "failure_reason": "NO_ALIGNED_PATIENT_ENTITY_MATRIX",
                    "entities": int(wide.shape[1]), "patients": int(len(patients)),
                })
                continue
            wide = wide.reindex(index=patients)
            fold_records = analyse_entity_matrix(
                cancer_id=cancer,
                subject_type=subject_type,
                patient_ids=patients.tolist(),
                entity_ids=wide.columns.astype(str).tolist(),
                values=wide.to_numpy(np.float64),
                fold_ids=cancer_folds.reindex(patients).patient_fold_id.to_numpy(int),
                endpoint_frame=endpoints,
                covariate_frame=covariates,
                block_size=args.block_size,
                min_train_patients=args.min_train_patients,
                min_train_events=args.min_train_events,
                min_validation_patients=args.min_validation_patients,
                min_validation_events=args.min_validation_events,
                min_test_patients=args.min_test_patients,
                min_test_events=args.min_test_events,
                min_train_measurements=args.min_train_measurements,
            )
            summary = summarise_oof_associations(
                fold_records,
                wide.columns.astype(str).tolist(),
                cancer_id=cancer,
                subject_type=subject_type,
                min_folds_available=args.min_folds_available,
            )
            fold_path_part = _write_part(
                fold_records, output / "entity_clinical_association_fold_records", cancer, subject_type
            )
            summary_path_part = _write_part(
                summary, output / "entity_clinical_association_summary_parts", cancer, subject_type
            )
            metrics = fold_metrics(fold_records)
            all_summary.append(summary)
            all_metrics.append(metrics)
            parts.append({
                "cancer_id": cancer,
                "subject_type": subject_type,
                "status": "SUCCESS",
                "failure_reason": "",
                "entities": int(wide.shape[1]),
                "patients": int(len(wide)),
                "fold_rows": int(len(fold_records)),
                "summary_rows": int(len(summary)),
                "available_summary_rows": int(summary.availability.sum()),
                "fold_records_path": str(fold_path_part.resolve()),
                "fold_records_sha256": file_sha256(fold_path_part),
                "summary_part_path": str(summary_path_part.resolve()),
                "summary_part_sha256": file_sha256(summary_path_part),
            })
            print(json.dumps(parts[-1], ensure_ascii=False), flush=True)

    if not all_summary:
        raise RuntimeError("No entity clinical subject matrix was calculated")
    summary = pd.concat(all_summary, ignore_index=True)
    expected_summary_rows = int(
        sum(
            (
                candidates.loc[candidates.cancer_id.eq(cancer), "lncrna_id"].nunique()
                if subject_type == "lncRNA"
                else candidates.loc[candidates.cancer_id.eq(cancer), "pathway_id"].nunique()
                if subject_type == "exact_pathway"
                else len(HISTORICAL_STATE_IDS)
            )
            * 6
            for cancer in cancers
            for subject_type in subject_types
        )
    )
    if args.max_entities is None and len(summary) != expected_summary_rows:
        raise RuntimeError(
            f"Typed entity universe is incomplete: {len(summary)} != {expected_summary_rows}"
        )
    summary_path = output / "entity_clinical_associations.parquet"
    summary.to_parquet(summary_path, index=False, compression="zstd")
    metrics = pd.concat(all_metrics, ignore_index=True)
    metrics_path = output / "entity_clinical_fold_metrics.tsv"
    metrics.to_csv(metrics_path, sep="\t", index=False)
    part_manifest = pd.DataFrame(parts)
    part_manifest_path = output / "ENTITY_CLINICAL_PART_MANIFEST.tsv"
    part_manifest.to_csv(part_manifest_path, sep="\t", index=False)

    config = {
        "format": CALCULATION_FORMAT,
        "subject_types": list(subject_types),
        "cancers": cancers,
        "block_size": args.block_size,
        "minimums": {
            "train_patients": args.min_train_patients,
            "train_events": args.min_train_events,
            "validation_patients": args.min_validation_patients,
            "validation_events": args.min_validation_events,
            "test_patients": args.min_test_patients,
            "test_events": args.min_test_events,
            "train_measurements": args.min_train_measurements,
            "folds_available": args.min_folds_available,
        },
        "max_entities": args.max_entities,
    }
    calculation_manifest = {
        "format": CALCULATION_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "training_run_id": args.training_run_id,
        "folds": 5,
        "split_policy": {
            str(fold): {
                "test": [fold],
                "validation": [(fold + 1) % 5],
                "train": [value for value in range(5) if value not in {fold, (fold + 1) % 5}],
            }
            for fold in range(5)
        },
        "method": "TRAIN_LOCAL_COVARIATE_RESIDUALIZED_BRESLOW_COX_NULL_SCORE_ONE_STEP",
        "partitions": parts,
    }
    calculation_manifest_path = output / "CALCULATION_MANIFEST.json"
    write_json(calculation_manifest, calculation_manifest_path)

    input_artifacts = [
        _input_artifact(cdr_path, "raw_data", "fresh endpoint standardisation"),
        _input_artifact(fold_path, "split_manifest", "fixed V3.2 patient folds"),
        _input_artifact(
            fold_receipt_path,
            "split_authority_receipt",
            "frozen V3.2 patient-first fold binding",
        ),
        _input_artifact(candidate_path, "annotation", "complete typed lncRNA/exact-pathway universe"),
    ]
    if "lncRNA" in subject_types:
        input_artifacts.append(
            _input_artifact(lncrna_root, "standardized_input", "patient lncRNA expression only")
        )
    if "exact_pathway" in subject_types:
        input_artifacts.append(
            _input_artifact(pathway_root, "standardized_input", "patient exact-pathway activity only")
        )
    if "state" in subject_types:
        input_artifacts.extend(
            [
                _input_artifact(state_path, "standardized_input", "seven patient State measurements only"),
                _input_artifact(
                    state_lineage_path,
                    "v32_generation_anchor",
                    "V3.2 State lineage verification only; never a feature or label",
                ),
            ]
        )
    seed_values = sorted(set(pd.to_numeric(sample_folds.fold_seed, errors="coerce").dropna().astype(int)))
    lineage = {
        "format": CALCULATION_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "module_id": "clinical_entity_association_extension",
        "training_run_id": args.training_run_id,
        "training_status": "SUCCESS",
        "statistical_training_status": "FRESH_V32_COX_SCORE_CALCULATION",
        "folds": 5,
        "seeds": seed_values,
        "split_unit": "patient",
        "split_policy": "3_TRAIN_1_VALIDATION_1_OOF_TEST",
        "patient_id_contract": {
            "source": "EXPLICIT_PATIENT_ID_COLUMN",
            "whitespace_trim_only": True,
            "unconditional_tcga_12_character_truncation_forbidden": True,
            "sample_id_patient_fallback_used": False,
            "null_or_empty_patient_id_fails_closed": True,
        },
        "patient_fold_authority": patient_authority_audit,
        "preprocessing_scope": "TRAIN_PATIENTS_ONLY_PER_OUTER_FOLD",
        "association_method": "BRESLOW_COX_NULL_SCORE_ONE_STEP_AND_OOF_RANDOM_EFFECTS",
        "code_sha256": file_sha256(ROOT / "cc_hhgt" / "v32" / "clinical_entity_training.py"),
        "runner_sha256": file_sha256(Path(__file__)),
        "config_sha256": json_sha256(config),
        "input_manifest_sha256": json_sha256(input_artifacts),
        "calculation_manifest_sha256": file_sha256(calculation_manifest_path),
        "input_artifacts": input_artifacts,
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": file_sha256(summary_path),
        "summary_rows": int(len(summary)),
        "expected_summary_rows_from_candidate_universe": expected_summary_rows,
        "all_candidate_entities_enumerated": bool(len(summary) == expected_summary_rows),
        "available_rows": int(summary.availability.sum()),
        "unavailable_rows": int((~summary.availability).sum()),
        "cancers": int(summary.cancer_id.nunique()),
        "subject_types": sorted(summary.subject_type.unique().tolist()),
        "clinical_endpoints": sorted(summary.clinical_endpoint.unique().tolist()),
        "dfs_policy": "NULL_WITH_NO_DISTINCT_DFS_SOURCE",
        "full_entity_universe": args.max_entities is None,
        "private_head_trained_from_scratch": False,
        "fresh_statistical_calculation": True,
        "old_checkpoint_loaded": False,
        "old_predictions_used_as_features": False,
        "old_rankings_used_as_outputs": False,
        "v30_clinical_result_loaded": False,
        "changes_primary_ranking": False,
        "patient_risk_artifacts_overwritten": False,
    }
    lineage_path = output / "ENTITY_CLINICAL_LINEAGE.json"
    write_json(lineage, lineage_path)
    contract_validation = validate_entity_clinical_contract(summary, lineage)
    contract_validation["lineage_sha256"] = file_sha256(lineage_path)
    contract_validation["summary_sha256"] = file_sha256(summary_path)
    contract_validation_path = output / "ENTITY_CLINICAL_CONTRACT_VALIDATION.json"
    write_json(contract_validation, contract_validation_path)
    success = {
        "status": "SUCCESS",
        "analysis_version": ANALYSIS_VERSION,
        "training_run_id": args.training_run_id,
        "summary_rows": int(len(summary)),
        "available_rows": int(summary.availability.sum()),
        "unavailable_rows": int((~summary.availability).sum()),
        "fold_record_rows": int(part_manifest.fold_rows.fillna(0).sum()),
        "cancers": int(summary.cancer_id.nunique()),
        "subject_types": sorted(summary.subject_type.unique().tolist()),
        "endpoints": sorted(summary.clinical_endpoint.unique().tolist()),
        "dfs_non_null_rows": int(
            summary.loc[summary.clinical_endpoint.eq("DFS"), "meta_beta"].notna().sum()
        ),
        "all_non_null_results_newly_computed_v32": True,
        "old_results_used": False,
        "changes_primary_ranking": False,
        "lineage_sha256": file_sha256(lineage_path),
        "contract_validation": "PASS",
        "contract_validation_sha256": file_sha256(contract_validation_path),
    }
    write_json(success, output / "SUCCESS.json")
    print(json.dumps(success, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
