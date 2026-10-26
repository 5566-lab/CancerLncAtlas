#!/usr/bin/env python3
"""Train the pooled five-fold V3.2 clinical private head from raw TCGA-CDR."""
from __future__ import annotations

import argparse
import hashlib
import json
import socket
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


def _load_core_tables(core_root: Path, manifest: dict, fold: int):
    """Read the exact hash-bound exported files, including relocated V4 output."""
    from cc_hhgt.v32.clinical_training import core_feature_columns, canonical_core_entity_ids
    from cc_hhgt.v32.current_core_export import verified_path
    tables = {}
    for kind in ("lncRNA", "pathway", "cancer"):
        declaration = manifest["folds"][str(fold)]["exports"][kind]
        path = verified_path(declaration, core_root)
        frame = pd.read_parquet(path)
        columns = core_feature_columns(frame)
        if (len(frame) != declaration.get("rows") or len(columns) != declaration.get("features")
                or not np.isfinite(frame[columns].to_numpy(float)).all()):
            raise RuntimeError(f"Fold-{fold} {kind} embedding shape or values differ from its manifest")
        frame["node_id"] = canonical_core_entity_ids(frame.node_id, kind.lower())
        if frame.node_id.duplicated().any():
            raise RuntimeError(f"Fold-{fold} {kind} embedding IDs collide")
        tables[kind] = frame
    return tables


def _validate_resume_receipt(meta, *, training_run_id, core_manifest_sha, input_lineage_sha,
                             config_sha, public_path, private_path, checkpoint_path):
    if (meta.get("status") != "SUCCESS" or meta.get("training_run_id") != training_run_id
            or meta.get("core_manifest_sha256") != core_manifest_sha
            or meta.get("input_lineage_sha256") != input_lineage_sha
            or meta.get("training_config_sha256") != config_sha):
        raise RuntimeError("Clinical resume belongs to a different run, core or input lineage")
    for key, path in [("public_prediction_sha256", public_path), ("private_prediction_sha256", private_path),
                      ("checkpoint_sha256", checkpoint_path)]:
        if meta.get(key) != _sha256(path):
            raise RuntimeError(f"Clinical resume artifact changed: {key}")


def _build_fold_features(
    fold: int,
    folds: pd.DataFrame,
    covariates: pd.DataFrame,
    expression_root: Path,
    activity_root: Path,
    core_root: Path,
    core_manifest: dict | None = None,
) -> pd.DataFrame:
    from cc_hhgt.v32.clinical_training import (
        core_feature_columns,
        encode_fold_covariates,
        project_measurements_to_core,
    )

    validation_fold = (int(fold) + 1) % 5
    if core_manifest is None:
        from cc_hhgt.v32.genomic_training import _validate_core_manifest
        core_manifest = _validate_core_manifest(core_root / "CORE_EMBEDDING_MANIFEST.json")[0]
    tables = _load_core_tables(core_root, core_manifest, fold)
    lnc_embedding, pathway_embedding, cancer_embedding = (tables[k] for k in ("lncRNA", "pathway", "cancer"))
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


def align_clinical_cohort(raw_folds, endpoints, covariates):
    from cc_hhgt.v32.clinical_training import patient_fold_manifest
    folds = patient_fold_manifest(raw_folds)
    endpoints = endpoints.merge(folds, on=["cancer_id", "patient_id"], how="inner", validate="many_to_one")
    covariates = covariates.merge(folds[["cancer_id", "patient_id"]], on=["cancer_id", "patient_id"], how="inner", validate="one_to_one")
    return folds, endpoints, covariates


def collect_fold_inputs(fold, folds, covariates, endpoints, expression_root, activity_root,
                        core_root, core_manifest, *, n_bins, min_patients, min_events):
    """Original patient projection, eligibility and train-defined time bins; no fit."""
    from cc_hhgt.v32.contracts import dataframe_sha256
    features = _build_fold_features(fold, folds, covariates, expression_root, activity_root, core_root, core_manifest)
    train = np.flatnonzero(~features.patient_fold_id.isin([fold, (fold+1)%5]).to_numpy())
    validation = np.flatnonzero(features.patient_fold_id.eq((fold+1)%5).to_numpy())
    test = np.flatnonzero(features.patient_fold_id.eq(fold).to_numpy())
    arrays, edges, private = _endpoint_arrays(features, endpoints, train, n_bins=n_bins, min_patients=min_patients, min_events=min_events)
    if not arrays or any(not len(ids) for ids in (train, validation, test)):
        raise RuntimeError("Clinical fold has no trainable endpoint or an empty patient split")
    core_columns = sorted(c for c in features if c.startswith("core_"))
    domain_columns = sorted(c for c in features if c.startswith("domain_"))
    core = features[core_columns].to_numpy(np.float32)
    domain = features[domain_columns].fillna(0).to_numpy(np.float32)
    if not np.isfinite(core).all() or not np.isfinite(domain).all(): raise RuntimeError("Clinical collected features are nonfinite")
    values = {"core":core, "domain":domain, "train":train, "validation":validation, "test":test,
        **{endpoint+"."+key:value for endpoint, target in arrays.items() for key,value in target.items()}}
    hashes = {}
    for name,value in values.items():
        array=np.ascontiguousarray(value); digest=hashlib.sha256(); view=memoryview(array).cast("B")
        for start in range(0,len(view),8*1024*1024): digest.update(view[start:start+8*1024*1024])
        hashes[name]={"shape":list(array.shape),"dtype":str(array.dtype),"sha256":digest.hexdigest()}
    description={"n_train":len(train),"n_validation":len(validation),"n_test":len(test),
        "core_columns":core_columns,"domain_columns":domain_columns,"endpoint_time_bins":edges,
        "patient_order_sha256":dataframe_sha256(features[["cancer_id","patient_id","patient_fold_id"]], ["cancer_id","patient_id"]),
        "training_input_sha256":_canonical_hash(hashes)}
    return dict(features=features, arrays=arrays, edges=edges, private_labels=private,
        train_indices=train, validation_indices=validation, test_indices=test, core_columns=core_columns,
        domain_columns=domain_columns, core_values=core, domain_values=domain, description=description)


def _parser():
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
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--patient-authority-mode", choices=["frozen", "current"], default="frozen")
    parser.add_argument("--training-run-id", help="Required for a current-core V2 run")
    parser.add_argument("--expected-core-run-id", help="Required for a current-core V2 run")
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if socket.gethostname() != "149":
        raise RuntimeError("Clinical preparation and training require target host 149")
    run_id = args.training_run_id or RUN_ID
    if any(getattr(args,k)<1 for k in ("epochs","patience","n_bins","min_patients","min_events")):
        raise ValueError("Clinical counts and training limits must be positive")
    if args.prepare_only and args.resume: raise ValueError("Clinical input preparation cannot resume fitting outputs")

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
        validate_frozen_v32_patient_fold_binding, validate_patient_fold_authority,
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
    core_manifest_path = core_root / "CORE_EMBEDDING_MANIFEST.json"
    from cc_hhgt.v32.genomic_training import _validate_core_manifest
    from cc_hhgt.v32.current_core_export import EXPORT_FORMAT, validate_current_export
    core_manifest, core_manifest_sha, _ = _validate_core_manifest(core_manifest_path)
    if core_manifest.get("export_format") == EXPORT_FORMAT:
        if not args.training_run_id or not args.expected_core_run_id:
            raise RuntimeError("Current clinical run requires --training-run-id and --expected-core-run-id")
        validate_current_export(core_manifest_path, expected_run_id=args.expected_core_run_id)
    if args.patient_authority_mode == "current" and (not args.expected_core_run_id or core_manifest.get("export_format") != EXPORT_FORMAT):
        raise RuntimeError("Current Clinical patient authority requires the expected current core")
    authority_validator = validate_frozen_v32_patient_fold_binding if args.patient_authority_mode == "frozen" else validate_patient_fold_authority
    patient_authority_audit = authority_validator(
        fold_manifest_path, fold_authority_receipt_path
    )
    output_root.mkdir(parents=True, exist_ok=args.resume)
    training_config = {
        "parameters": {key: getattr(args, key) for key in
                       ["seed", "epochs", "patience", "n_bins", "min_patients", "min_events"]},
        "runner_sha256": _sha256(Path(__file__)),
        "implementation_sha256": _sha256(Path(__file__).resolve().parents[1] / "cc_hhgt/v32/clinical_training.py")}
    training_config_sha = _canonical_hash(training_config)
    source_contract = {"format": "NATIVE_CLINICAL_RISK_INPUTS_V1", "patient_authority_mode": args.patient_authority_mode,
        "files": {key:{"path":str(path.resolve()),"sha256":artifact_sha256(path)} for key,path in {
            "workbook":workbook,"patient_folds":fold_manifest_path,"patient_authority_receipt":fold_authority_receipt_path,
            "expression_root":expression_root,"activity_root":activity_root,"core_manifest":core_manifest_path}.items()}}
    preflight = {"target_host": "149",
        "training_started": False, "training_run_id": run_id, "core_run_id": core_manifest.get("core_run_id"),
        "core_manifest_sha256": core_manifest_sha, "training_config_sha256": training_config_sha}
    preflight.update(prepare_only=args.prepare_only, source_contract=source_contract)
    preflight_path = output_root / "LAUNCH_PREFLIGHT.json"
    if args.resume:
        if not preflight_path.is_file() or json.loads(preflight_path.read_text()) != preflight:
            raise RuntimeError("Clinical resume preflight differs from its original run/core/configuration")
    else:
        preflight_path.write_text(json.dumps(preflight, indent=2))

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
    if args.resume:
        previous_audit_path = output_root / "INPUT_LINEAGE_AUDIT.json"
        if (not previous_audit_path.is_file() or
                json.loads(previous_audit_path.read_text()).get("lineage_sha256") != input_audit["lineage_sha256"]):
            raise RuntimeError("Clinical resume raw input lineage changed")
    (output_root / "INPUT_LINEAGE_AUDIT.json").write_text(
        json.dumps(input_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    endpoints, covariates = standardize_tcga_cdr_workbook(workbook)
    endpoints.to_parquet(output_root / "v32_clinical_endpoints_private.parquet", index=False)
    covariates.to_parquet(output_root / "v32_clinical_covariates.parquet", index=False)
    folds, endpoints, covariates = align_clinical_cohort(pd.read_csv(fold_manifest_path, sep="\t"), endpoints, covariates)
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
            prior = json.loads(success_path.read_text(encoding="utf-8"))
            _validate_resume_receipt(prior, training_run_id=run_id, core_manifest_sha=core_manifest_sha,
                input_lineage_sha=input_audit["lineage_sha256"], public_path=public_path,
                config_sha=training_config_sha, private_path=private_path, checkpoint_path=fold_root / "best_model_state.pt")
            public_outputs.append(pd.read_parquet(public_path))
            private_outputs.append(pd.read_parquet(private_path))
            task_lineage.append(json.loads(success_path.read_text(encoding="utf-8")))
            continue
        collected = collect_fold_inputs(fold, folds, covariates, endpoints, expression_root, activity_root,
            core_root, core_manifest, n_bins=args.n_bins, min_patients=args.min_patients, min_events=args.min_events)
        features, arrays, edges, private_labels = (collected[k] for k in ("features", "arrays", "edges", "private_labels"))
        train_indices, validation_indices, test_indices = (collected[k] for k in ("train_indices", "validation_indices", "test_indices"))
        core_columns, domain_columns = collected["core_columns"], collected["domain_columns"]
        core_values, domain_values = collected["core_values"], collected["domain_values"]
        feature_path = fold_root / "patient_features.fold_local.parquet"
        features.to_parquet(feature_path, index=False, compression="zstd")
        if args.prepare_only:
            task_lineage.append({"patient_fold": fold, **collected["description"],
                "feature_artifact": {"path": str(feature_path.resolve()), "sha256": _sha256(feature_path)}})
            continue
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
                        "training_run_id": run_id,
                        "core_run_id": core_manifest.get("core_run_id"),
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
            "training_run_id": run_id,
            "core_run_id": core_manifest.get("core_run_id"),
            "core_manifest_sha256": core_manifest_sha,
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
            "training_run_id": run_id,
            "core_run_id": core_manifest.get("core_run_id"),
            "core_manifest_sha256": core_manifest_sha,
            "input_lineage_sha256": input_audit["lineage_sha256"],
            "training_config_sha256": training_config_sha,
            "input_description": collected["description"],
            "public_prediction_sha256": _sha256(public_path),
            "feature_artifact": {"path": str(feature_path.resolve()), "sha256": _sha256(feature_path)},
            "private_prediction_sha256": _sha256(private_path),
            "module_id": "clinical",
            "patient_fold": fold,
            "seed": args.seed,
            "n_train": int(len(train_indices)),
            "n_validation": int(len(validation_indices)),
            "n_test": int(len(test_indices)),
            "trained_endpoints": sorted(arrays),
            "unavailable_endpoints": sorted(set(CLINICAL_ENDPOINTS) - set(arrays)),
            "checkpoint_path": str(checkpoint_path.resolve()),
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

    for item in source_contract["files"].values():
        if artifact_sha256(item["path"]) != item["sha256"]: raise RuntimeError("Clinical source changed during execution")
    if args.prepare_only:
        ready={"format":"NATIVE_CLINICAL_RISK_PREFLIGHT_V1","status":"PASS_READY","target_host":"149",
            "training_started":False,"model_forward_started":False,"training_run_id":run_id,
            "core_run_id":core_manifest.get("core_run_id"),"source_contract":source_contract,
            "training_config":training_config,"patient_fold_authority":patient_authority_audit,"folds":task_lineage}
        (output_root/"HEAD_PREFLIGHT.json").write_text(json.dumps(ready,indent=2)+"\n",encoding="utf-8")
        print(json.dumps({"status":"PASS_READY","training_started":False,"folds":len(task_lineage)}));return 0
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

    if _sha256(core_manifest_path) != core_manifest_sha:
        raise RuntimeError("Selected core manifest changed during clinical training")
    aggregate_core_hash = _canonical_hash(
        {str(fold): core_manifest["folds"][str(fold)]["checkpoint_sha256"] for fold in range(5)}
    )
    lineage = {
        "module_id": "clinical",
        "analysis_version": ANALYSIS_VERSION,
        "training_run_id": run_id,
        "core_run_id": core_manifest.get("core_run_id"),
        "core_manifest_sha256": core_manifest_sha,
        "training_status": "SUCCESS" if set(folds_requested) == set(range(5)) else "PARTIAL",
        "initialization_policy": "FROZEN_V32_CORE_PRIVATE_HEAD_FROM_SCRATCH",
        "folds": len(folds_requested),
        "seeds": sorted({row["seed"] for row in task_lineage}),
        "code_sha256": artifact_sha256(root / "cc_hhgt" / "v32"),
        "config_sha256": _canonical_hash(vars(args)),
        "training_config": training_config,
        "training_config_sha256": training_config_sha,
        "source_contract": source_contract,
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
            {"path": str(core_manifest_path), "sha256": core_manifest_sha, "artifact_kind": "v32_core_embedding_manifest"},
        ],
        "task_lineage": task_lineage,
        "public_prediction_rows": int(len(public_all)),
        "public_prediction_artifact": {
            "path": str((output_root / "clinical_patient_risk.parquet").resolve()),
            "sha256": _sha256(output_root / "clinical_patient_risk.parquet"),
        },
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
