#!/usr/bin/env python3
"""Train the pooled five-fold V3.2 clinical private head from raw TCGA-CDR."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
RUN_ID = "V3_2_FULL_MULTITASK_CLINICAL_20260825"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _partition(root: Path, cancer: str) -> Path:
    path = root / f"cancer_id={cancer}" / "part-0.parquet"
    if not path.is_file():
        raise RuntimeError(f"Missing V3.2 molecular partition: {path}")
    return path


def _build_fold_features(
    fold: int,
    folds: pd.DataFrame,
    covariates: pd.DataFrame,
    expression_root: Path,
    activity_root: Path,
    core_root: Path,
) -> pd.DataFrame:
    from cc_hhgt.v32.clinical_training import (
        core_feature_columns,
        encode_fold_covariates,
        project_measurements_to_core,
    )

    validation_fold = (int(fold) + 1) % 5
    embedding_root = core_root / f"patient_fold={fold}"
    lnc_embedding = pd.read_parquet(embedding_root / "lncRNA.parquet")
    pathway_embedding = pd.read_parquet(embedding_root / "pathway.parquet")
    cancer_embedding = pd.read_parquet(embedding_root / "cancer.parquet")
    cancer_features = core_feature_columns(cancer_embedding)
    cancer_embedding = cancer_embedding.set_index("node_id")
    output: list[pd.DataFrame] = []
    for cancer in sorted(folds.cancer_id.astype(str).unique()):
        cohort = folds.loc[folds.cancer_id.astype(str).eq(cancer)].copy()
        train_ids = cohort.loc[
            ~cohort.patient_fold_id.isin([fold, validation_fold]), "patient_id"
        ].astype(str).tolist()
        if len(train_ids) < 2:
            continue
        expression = pd.read_parquet(_partition(expression_root, cancer))
        activity = pd.read_parquet(_partition(activity_root, cancer))
        wanted = set(cohort.patient_id.astype(str))
        expression = expression.loc[expression.patient_id.astype(str).isin(wanted)]
        activity = activity.loc[activity.patient_id.astype(str).isin(wanted)]
        lnc = project_measurements_to_core(
            expression,
            lnc_embedding,
            entity_column="lncrna_id",
            value_column="logcpm",
            train_patient_ids=train_ids,
            output_prefix="core_lnc",
        )
        pathway = project_measurements_to_core(
            activity,
            pathway_embedding,
            entity_column="pathway_id",
            value_column="activity_score",
            train_patient_ids=train_ids,
            output_prefix="core_pathway",
        )
        local_covariates = covariates.loc[
            covariates.cancer_id.astype(str).eq(cancer)
            & covariates.patient_id.astype(str).isin(wanted)
        ]
        domain = encode_fold_covariates(local_covariates, train_ids)
        frame = cohort.merge(lnc, on="patient_id", how="inner", validate="one_to_one")
        frame = frame.merge(pathway, on="patient_id", how="inner", validate="one_to_one")
        frame = frame.merge(domain, on="patient_id", how="left", validate="one_to_one")
        if cancer not in cancer_embedding.index:
            raise RuntimeError(f"Cancer {cancer} lacks a fold-{fold} V3.2 core embedding")
        cancer_vector = cancer_embedding.loc[cancer, cancer_features].to_numpy(np.float32)
        for index, value in enumerate(cancer_vector):
            frame[f"core_cancer_{index:03d}"] = value
        output.append(frame)
    if not output:
        raise RuntimeError(f"Fold {fold} produced no pooled clinical patient features")
    features = pd.concat(output, ignore_index=True)
    expected = folds[["cancer_id", "patient_id"]].drop_duplicates()
    observed = features[["cancer_id", "patient_id"]].drop_duplicates()
    missing = expected.merge(observed, how="left", indicator=True).query("_merge == 'left_only'")
    if not missing.empty:
        # A missing molecular intersection is explicit and never filled with
        # an old patient vector.  The training cohort remains auditable.
        print(f"[clinical] fold={fold} molecular-unavailable patients={len(missing)}", flush=True)
    return features.sort_values(["cancer_id", "patient_id"], kind="stable").reset_index(drop=True)


def _endpoint_arrays(
    features: pd.DataFrame,
    endpoints: pd.DataFrame,
    train_indices: np.ndarray,
    *,
    n_bins: int,
    min_patients: int,
    min_events: int,
):
    from cc_hhgt.v32.clinical_training import CLINICAL_ENDPOINTS, endpoint_time_bins, time_to_bin

    aligned = features[["cancer_id", "patient_id"]].copy()
    arrays = {}
    edges = {}
    private = []
    for endpoint in CLINICAL_ENDPOINTS:
        source = endpoints.loc[
            endpoints.clinical_endpoint.eq(endpoint),
            ["cancer_id", "patient_id", "time_days", "event", "endpoint_available", "failure_reason"],
        ].copy()
        source = source.rename(columns={"failure_reason": "label_failure_reason"})
        part = aligned.merge(
            source, on=["cancer_id", "patient_id"], how="left", validate="one_to_one"
        )
        time = pd.to_numeric(part.time_days, errors="coerce").to_numpy(float)
        event = pd.to_numeric(part.event, errors="coerce").to_numpy(float)
        available = np.isfinite(time) & np.isfinite(event)
        train_available = available[train_indices]
        if train_available.sum() >= min_patients and event[train_indices][train_available].sum() >= min_events:
            endpoint_edges = endpoint_time_bins(
                time[train_indices][train_available],
                event[train_indices][train_available],
                n_bins,
            )
            bins = np.zeros(len(features), dtype=np.int64)
            bins[available] = time_to_bin(time[available], endpoint_edges)
            arrays[endpoint] = {
                "time": time,
                "event": np.where(available, event, 0.0).astype("float32"),
                "available": available.astype("float32"),
                "bin": bins,
            }
            edges[endpoint] = endpoint_edges.tolist()
        private_part = part[["cancer_id", "patient_id", "time_days", "event", "endpoint_available"]].copy()
        private_part["clinical_endpoint"] = endpoint
        private.append(private_part)
    return arrays, edges, pd.concat(private, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--workbook",
        default="inputs/v32_full_multitask/clinical/TCGA-CDR-SupplementalTableS1.xlsx",
    )
    parser.add_argument(
        "--fold-manifest",
        default="artifacts/v32_patient_fold_authority_20260829_r1/SAMPLE_PATIENT_FOLD_MAP.tsv",
    )
    parser.add_argument(
        "--patient-fold-authority-receipt",
        default="artifacts/v32_patient_fold_authority_20260829_r1/PATIENT_FOLD_AUTHORITY_RECEIPT.json",
    )
    parser.add_argument(
        "--expression-root", default="artifacts/input/formal_lncRNA_expression"
    )
    parser.add_argument(
        "--activity-root", default="artifacts/input/formal_pathway_activity"
    )
    parser.add_argument(
        "--core-root", default="artifacts/v32_full_multitask/core_embeddings"
    )
    parser.add_argument(
        "--output-root", default="artifacts/v32_full_multitask/clinical"
    )
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--n-bins", type=int, default=12)
    parser.add_argument("--min-patients", type=int, default=40)
    parser.add_argument("--min-events", type=int, default=12)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    sys.path.insert(0, str(root))
    import torch

    from cc_hhgt.v32.clinical_training import (
        CLINICAL_CHECKPOINT_FORMAT,
        CLINICAL_ENDPOINTS,
        harrell_c_index,
        patient_fold_manifest,
        reshape_endpoint_logits,
        standardize_tcga_cdr_workbook,
        survival_risk_probability,
        train_private_clinical_head,
    )
    from cc_hhgt.v32.input_lineage import InputLineageArtifact, artifact_sha256, validate_input_lineage
    from cc_hhgt.v32.patient_fold_authority import (
        validate_frozen_v32_patient_fold_binding,
    )

    folds_requested = sorted(set(args.folds))
    if not folds_requested or any(fold not in range(5) for fold in folds_requested):
        raise RuntimeError("--folds must be a non-empty subset of 0..4")
    workbook = root / args.workbook
    fold_manifest_path = root / args.fold_manifest
    fold_authority_receipt_path = root / args.patient_fold_authority_receipt
    expression_root = root / args.expression_root
    activity_root = root / args.activity_root
    core_root = root / args.core_root
    output_root = root / args.output_root
    patient_authority_audit = validate_frozen_v32_patient_fold_binding(
        fold_manifest_path, fold_authority_receipt_path
    )
    output_root.mkdir(parents=True, exist_ok=True)

    input_audit = validate_input_lineage(
        [
            InputLineageArtifact(
                workbook, "external_raw_TCGA_CDR", "raw_data", True, False,
                use_role="training_target", artifact_id="tcga_cdr_raw_outcomes"
            ),
            InputLineageArtifact(
                fold_manifest_path, "V3.2", "split_manifest", False, False,
                use_role="split_control", artifact_id="v32_patient_folds"
            ),
            InputLineageArtifact(
                expression_root, "V3.2", "standardized_input", False, False,
                use_role="aux_input", artifact_id="v32_expression"
            ),
            InputLineageArtifact(
                activity_root, "V3.2", "standardized_input", False, False,
                use_role="aux_input", artifact_id="v32_exact_pathway_activity"
            ),
        ]
    )
    input_audit["patient_fold_authority"] = patient_authority_audit
    (output_root / "INPUT_LINEAGE_AUDIT.json").write_text(
        json.dumps(input_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    endpoints, covariates = standardize_tcga_cdr_workbook(workbook)
    endpoints.to_parquet(output_root / "v32_clinical_endpoints_private.parquet", index=False)
    covariates.to_parquet(output_root / "v32_clinical_covariates.parquet", index=False)
    raw_folds = pd.read_csv(fold_manifest_path, sep="\t")
    folds = patient_fold_manifest(raw_folds)
    endpoints = endpoints.merge(
        folds, on=["cancer_id", "patient_id"], how="inner", validate="many_to_one"
    )
    covariates = covariates.merge(
        folds[["cancer_id", "patient_id"]],
        on=["cancer_id", "patient_id"], how="inner", validate="one_to_one"
    )
    core_manifest_path = core_root / "CORE_EMBEDDING_MANIFEST.json"
    core_manifest = json.loads(core_manifest_path.read_text(encoding="utf-8"))
    if set(core_manifest.get("folds", {})) != set(map(str, range(5))):
        raise RuntimeError("V3.2 core embedding manifest does not contain all five folds")
    if core_manifest.get("historical_checkpoint_loaded") is not False:
        raise RuntimeError("V3.2 core lineage did not exclude historical checkpoints")

    public_outputs = []
    private_outputs = []
    task_lineage = []
    for fold in folds_requested:
        fold_root = output_root / f"patient_fold={fold}"
        fold_root.mkdir(parents=True, exist_ok=True)
        public_path = fold_root / "clinical_patient_risk.parquet"
        private_path = fold_root / "clinical_patient_risk_evaluation_labels.parquet"
        success_path = fold_root / "SUCCESS.json"
        if args.resume and public_path.is_file() and private_path.is_file() and success_path.is_file():
            public_outputs.append(pd.read_parquet(public_path))
            private_outputs.append(pd.read_parquet(private_path))
            task_lineage.append(json.loads(success_path.read_text(encoding="utf-8")))
            continue
        features = _build_fold_features(
            fold, folds, covariates, expression_root, activity_root, core_root
        )
        feature_path = fold_root / "patient_features.fold_local.parquet"
        features.to_parquet(feature_path, index=False, compression="zstd")
        validation_fold = (fold + 1) % 5
        train_indices = np.flatnonzero(
            ~features.patient_fold_id.isin([fold, validation_fold]).to_numpy()
        )
        validation_indices = np.flatnonzero(
            features.patient_fold_id.eq(validation_fold).to_numpy()
        )
        test_indices = np.flatnonzero(features.patient_fold_id.eq(fold).to_numpy())
        arrays, edges, private_labels = _endpoint_arrays(
            features, endpoints, train_indices,
            n_bins=args.n_bins, min_patients=args.min_patients, min_events=args.min_events,
        )
        core_columns = sorted(column for column in features if column.startswith("core_"))
        domain_columns = sorted(column for column in features if column.startswith("domain_"))
        core_values = features[core_columns].to_numpy(np.float32)
        domain_values = features[domain_columns].fillna(0).to_numpy(np.float32)
        head, initialization, history = train_private_clinical_head(
            core_values, domain_values, arrays,
            train_indices=train_indices, validation_indices=validation_indices,
            n_bins=args.n_bins, seed=args.seed,
            epochs=args.epochs, patience=args.patience,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        head.to(device)
        with torch.no_grad():
            logits = reshape_endpoint_logits(
                head(
                    torch.as_tensor(core_values[test_indices], device=device),
                    torch.as_tensor(domain_values[test_indices], device=device),
                ),
                args.n_bins,
            ).cpu().numpy()
        rows = []
        private_rows = []
        test = features.iloc[test_indices].reset_index(drop=True)
        labels_aligned = private_labels.merge(
            test[["cancer_id", "patient_id"]],
            on=["cancer_id", "patient_id"], how="inner", validate="many_to_one"
        )
        for endpoint_index, endpoint in enumerate(CLINICAL_ENDPOINTS):
            trained = endpoint in arrays
            probability = (
                survival_risk_probability(logits[:, endpoint_index])
                if trained else np.full(len(test), np.nan)
            )
            for row_index, patient in test.iterrows():
                rows.append(
                    {
                        "cancer_id": patient.cancer_id,
                        "subject_type": "patient",
                        "subject_id": patient.patient_id,
                        "clinical_endpoint": endpoint,
                        "clinical_relevance_probability": float(probability[row_index]) if trained else np.nan,
                        "availability": bool(trained),
                        "failure_reason": "" if trained else "NO_TRAINABLE_DISTINCT_ENDPOINT_SOURCE",
                        "patient_fold_id": fold,
                        "analysis_version": ANALYSIS_VERSION,
                        "training_run_id": RUN_ID,
                        "model_artifact_id": f"clinical-fold-{fold}",
                        "changes_primary_ranking": False,
                    }
                )
            observed = labels_aligned.loc[labels_aligned.clinical_endpoint.eq(endpoint)].copy()
            observed = observed.merge(
                pd.DataFrame(
                    {
                        "cancer_id": test.cancer_id,
                        "patient_id": test.patient_id,
                        "clinical_relevance_probability": probability,
                    }
                ),
                on=["cancer_id", "patient_id"], how="left", validate="one_to_one",
            )
            observed["patient_fold_id"] = fold
            private_rows.append(observed)
        public = pd.DataFrame(rows)
        private = pd.concat(private_rows, ignore_index=True)
        public.to_parquet(public_path, index=False, compression="zstd")
        private.to_parquet(private_path, index=False, compression="zstd")
        pd.DataFrame(history).to_csv(fold_root / "training_history.tsv", sep="\t", index=False)

        core_fold = core_manifest["folds"][str(fold)]
        checkpoint = {
            "checkpoint_format": CLINICAL_CHECKPOINT_FORMAT,
            "analysis_version": ANALYSIS_VERSION,
            "training_run_id": RUN_ID,
            "module_id": "clinical",
            "patient_fold": fold,
            "seed": args.seed,
            "initialization": initialization,
            "source_checkpoint_sha256": None,
            "parent_v32_core_checkpoint_sha256": core_fold["checkpoint_sha256"],
            "parent_v32_core_parameter_sha256": core_fold["core_parameter_sha256"],
            "core_parameters_before_sha256": core_fold["core_parameter_sha256"],
            "core_parameters_after_sha256": core_fold["core_parameter_sha256"],
            "core_parameters_frozen": True,
            "old_checkpoint_loaded": False,
            "old_predictions_used_as_features": False,
            "endpoint_time_bins": edges,
            "core_columns": core_columns,
            "domain_columns": domain_columns,
            "model_state": head.cpu().state_dict(),
        }
        checkpoint_path = fold_root / "best_model_state.pt"
        torch.save(checkpoint, checkpoint_path)
        meta = {
            "status": "SUCCESS",
            "module_id": "clinical",
            "patient_fold": fold,
            "seed": args.seed,
            "n_train": int(len(train_indices)),
            "n_validation": int(len(validation_indices)),
            "n_test": int(len(test_indices)),
            "trained_endpoints": sorted(arrays),
            "unavailable_endpoints": sorted(set(CLINICAL_ENDPOINTS) - set(arrays)),
            "checkpoint_path": checkpoint_path.relative_to(root).as_posix(),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "parent_v32_core_checkpoint_sha256": core_fold["checkpoint_sha256"],
            "parent_v32_core_parameter_sha256": core_fold["core_parameter_sha256"],
            "private_head_trained_from_scratch": True,
            "source_checkpoint_sha256": None,
            "core_parameters_frozen": True,
            "core_parameters_unchanged": True,
        }
        success_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        public_outputs.append(public)
        private_outputs.append(private)
        task_lineage.append(meta)

    public_all = pd.concat(public_outputs, ignore_index=True)
    private_all = pd.concat(private_outputs, ignore_index=True)
    public_all.to_parquet(output_root / "clinical_patient_risk.parquet", index=False, compression="zstd")
    private_all.to_parquet(
        output_root / "clinical_patient_risk_evaluation_labels.PRIVATE.parquet",
        index=False, compression="zstd",
    )
    metric_rows = []
    for endpoint, group in private_all.groupby("clinical_endpoint", observed=True):
        available = (
            group.endpoint_available.fillna(False).astype(bool)
            & group.clinical_relevance_probability.notna()
        )
        metric_rows.append(
            {
                "clinical_endpoint": endpoint,
                "n_evaluation": int(available.sum()),
                "n_events": int(pd.to_numeric(group.loc[available, "event"], errors="coerce").sum()),
                "c_index": harrell_c_index(
                    group.loc[available, "time_days"],
                    group.loc[available, "event"],
                    group.loc[available, "clinical_relevance_probability"],
                ),
            }
        )
    pd.DataFrame(metric_rows).to_csv(output_root / "clinical_oof_metrics.tsv", sep="\t", index=False)

    aggregate_core_hash = _canonical_hash(
        {str(fold): core_manifest["folds"][str(fold)]["checkpoint_sha256"] for fold in range(5)}
    )
    lineage = {
        "module_id": "clinical",
        "analysis_version": ANALYSIS_VERSION,
        "training_run_id": RUN_ID,
        "training_status": "SUCCESS" if set(folds_requested) == set(range(5)) else "PARTIAL",
        "initialization_policy": "FROZEN_V32_CORE_PRIVATE_HEAD_FROM_SCRATCH",
        "folds": len(folds_requested),
        "seeds": sorted({row["seed"] for row in task_lineage}),
        "code_sha256": artifact_sha256(root / "cc_hhgt" / "v32"),
        "config_sha256": _canonical_hash(vars(args)),
        "input_manifest_sha256": input_audit["lineage_sha256"],
        "checkpoint_manifest_sha256": _canonical_hash(task_lineage),
        "old_checkpoint_loaded": False,
        "old_predictions_used_as_features": False,
        "old_rankings_used_as_outputs": False,
        "private_head_trained_from_scratch": True,
        "core_parameters_frozen": True,
        "v32_core_checkpoint_sha256": aggregate_core_hash,
        "core_parameters_before_sha256": aggregate_core_hash,
        "core_parameters_after_sha256": aggregate_core_hash,
        "input_artifacts": [
            {"path": str(workbook), "sha256": _sha256(workbook), "artifact_kind": "training_label"},
            {"path": str(fold_manifest_path), "sha256": _sha256(fold_manifest_path), "artifact_kind": "split_manifest"},
            {"path": str(core_manifest_path), "sha256": aggregate_core_hash, "artifact_kind": "v32_core_checkpoint"},
        ],
        "task_lineage": task_lineage,
        "public_prediction_rows": int(len(public_all)),
        "private_label_rows": int(len(private_all)),
        "labels_in_public_output": False,
    }
    (output_root / "MODULE_LINEAGE.json").write_text(
        json.dumps(lineage, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": lineage["training_status"],
                "folds": folds_requested,
                "public_rows": len(public_all),
                "trained_endpoints": sorted(
                    set().union(*(set(row["trained_endpoints"]) for row in task_lineage))
                ),
                "unavailable_endpoint_policy": "NULL_WITH_REASON_NO_DFI_TO_DFS_RENAME",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
