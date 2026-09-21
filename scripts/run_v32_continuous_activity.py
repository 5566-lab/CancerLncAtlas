#!/usr/bin/env python3
"""Train and materialize the fresh V3.2 continuous pathway-activity head."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.continuous_activity import (  # noqa: E402
    ANALYSIS_VERSION,
    CHECKPOINT_FORMAT,
    MODULE_ID,
    ContinuousActivityConfig,
    ContinuousActivityError,
    audit_frozen_lnc_embeddings,
    fit_fresh_pca_ridge_head,
)
from cc_hhgt.v32.patient_folds import assign_outer_split  # noqa: E402


RUN_FORMAT = "CC_HHGT_V3_2_CONTINUOUS_ACTIVITY_RELEASE_V1"
BINDING_FORMAT = "CC_HHGT_V3_2_CONTINUOUS_ACTIVITY_BINDING_V1"
N_FOLDS = 5


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _tree_manifest(root: Path, cancers: list[str]) -> dict[str, Any]:
    files = [root / f"cancer_id={cancer}" / "part-0.parquet" for cancer in cancers]
    success = root / "SUCCESS.json"
    if success.is_file():
        files.append(success)
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise ContinuousActivityError(f"Source tree is incomplete: {missing[:5]}")
    rows = []
    digest = hashlib.sha256()
    for path in sorted(files):
        relative = path.relative_to(root).as_posix()
        sha = file_sha256(path)
        size = path.stat().st_size
        rows.append({"path": relative, "bytes": int(size), "sha256": sha})
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(sha.encode("ascii"))
        digest.update(b"\n")
    return {"root": str(root.resolve()), "tree_sha256": digest.hexdigest(), "files": rows}


def _stable_cancer_seed(cancer: str) -> int:
    return int(hashlib.sha256(cancer.encode("utf-8")).hexdigest()[:8], 16)


def _load_fold_manifest(path: Path, cancers: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        sep="\t",
        dtype={"cancer_id": str, "sample_id": str, "patient_id": str},
    )
    required = {
        "cancer_id", "sample_id", "patient_id", "patient_fold_id", "fold_seed"
    }
    if missing := sorted(required - set(frame.columns)):
        raise ContinuousActivityError(f"Patient fold manifest lacks columns: {missing}")
    frame = frame.loc[frame.cancer_id.isin(cancers)].copy()
    frame["patient_fold_id"] = pd.to_numeric(frame.patient_fold_id, errors="raise").astype(int)
    frame["fold_seed"] = pd.to_numeric(frame.fold_seed, errors="raise").astype(int)
    if frame.empty or frame.duplicated(["cancer_id", "sample_id"]).any():
        raise ContinuousActivityError("Patient fold manifest is empty or duplicates samples")
    for column in ("sample_id", "patient_id"):
        frame[column] = frame[column].astype(str).str.strip()
        if frame[column].eq("").any():
            raise ContinuousActivityError(
                f"Patient fold manifest contains empty explicit {column}"
            )
    if (
        frame.groupby(["cancer_id", "patient_id"], observed=True)
        .patient_fold_id.nunique()
        .gt(1)
        .any()
    ):
        raise ContinuousActivityError("One explicit patient crosses folds")
    if set(frame.patient_fold_id.unique()) != set(range(N_FOLDS)):
        raise ContinuousActivityError("Formal continuous head requires patient folds 0..4")
    if frame.fold_seed.nunique() != 1 or int(frame.fold_seed.iloc[0]) != 20260726:
        raise ContinuousActivityError("Patient fold manifest seed is not the frozen V3.2 seed")
    observed = set(frame.cancer_id.unique())
    if observed != set(cancers):
        raise ContinuousActivityError(
            f"Patient fold manifest cancer mismatch: {sorted(set(cancers) - observed)}"
        )
    return frame.sort_values(["cancer_id", "patient_fold_id", "sample_id"], kind="stable")


def _read_cancer_matrices(
    expression_root: Path,
    activity_root: Path,
    fold_manifest: pd.DataFrame,
    cancer: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    expression = pd.read_parquet(expression_root / f"cancer_id={cancer}" / "part-0.parquet")
    activity = pd.read_parquet(activity_root / f"cancer_id={cancer}" / "part-0.parquet")
    required_expression = {"cancer_id", "sample_id", "patient_id", "lncrna_id", "logcpm"}
    required_activity = {
        "cancer_id",
        "sample_id",
        "patient_id",
        "pathway_id",
        "activity_score",
    }
    if missing := sorted(required_expression - set(expression.columns)):
        raise ContinuousActivityError(f"{cancer} expression lacks columns: {missing}")
    if missing := sorted(required_activity - set(activity.columns)):
        raise ContinuousActivityError(f"{cancer} activity lacks columns: {missing}")
    if set(expression.cancer_id.astype(str)) != {cancer} or set(activity.cancer_id.astype(str)) != {cancer}:
        raise ContinuousActivityError(f"{cancer} source partition contains another cancer")
    if "quality_flag" in activity:
        activity = activity.loc[activity.quality_flag.astype(str).str.lower().eq("pass")]
    if expression.duplicated(["sample_id", "lncrna_id"]).any():
        raise ContinuousActivityError(f"{cancer} expression duplicates sample/lncRNA")
    if activity.duplicated(["sample_id", "pathway_id"]).any():
        raise ContinuousActivityError(f"{cancer} activity duplicates sample/pathway")
    expression["logcpm"] = pd.to_numeric(expression.logcpm, errors="coerce")
    activity["activity_score"] = pd.to_numeric(activity.activity_score, errors="coerce")

    expression_patient = expression[["sample_id", "patient_id"]].astype(str).drop_duplicates()
    activity_patient = activity[["sample_id", "patient_id"]].astype(str).drop_duplicates()
    if expression_patient.sample_id.duplicated().any() or activity_patient.sample_id.duplicated().any():
        raise ContinuousActivityError(f"{cancer} has conflicting sample/patient mappings")
    patients = expression_patient.merge(
        activity_patient,
        on="sample_id",
        how="inner",
        suffixes=("_expression", "_activity"),
        validate="one_to_one",
    )
    if not patients.patient_id_expression.eq(patients.patient_id_activity).all():
        raise ContinuousActivityError(f"{cancer} expression/activity patient identifiers disagree")
    local_folds = fold_manifest.loc[
        fold_manifest.cancer_id.eq(cancer), ["cancer_id", "sample_id", "patient_fold_id", "fold_seed"]
    ]
    common = sorted(set(patients.sample_id.astype(str)) & set(local_folds.sample_id.astype(str)))
    if len(common) < 15:
        raise ContinuousActivityError(f"{cancer} has only {len(common)} aligned patients")
    expression_wide = expression.pivot(
        index="sample_id", columns="lncrna_id", values="logcpm"
    ).reindex(index=common).sort_index(axis=1)
    activity_wide = activity.pivot(
        index="sample_id", columns="pathway_id", values="activity_score"
    ).reindex(index=common).sort_index(axis=1)
    finite_pathway = np.isfinite(activity_wide.to_numpy(float)).all(axis=0)
    activity_wide = activity_wide.loc[:, finite_pathway]
    if activity_wide.shape[1] < 1:
        raise ContinuousActivityError(f"{cancer} has no complete pathway outcomes")
    patient_map = patients.set_index("sample_id").patient_id_expression.reindex(common)
    local_folds = local_folds.loc[local_folds.sample_id.isin(common)].copy()
    return expression_wide, activity_wide, patient_map, local_folds


def _fold_paths(work: Path, cancer: str, fold: int) -> dict[str, Path]:
    root = work / "folds" / f"cancer_{cancer}" / f"fold_{fold}"
    return {
        "root": root,
        "checkpoint": root / "CONTINUOUS_HEAD.npz",
        "oof": root / "OOF.parquet",
        "metrics": root / "METRICS.parquet",
        "attributions": root / "ATTRIBUTIONS.parquet",
        "success": root / "FOLD_SUCCESS.json",
    }


def _valid_completed_fold(paths: dict[str, Path], run_signature: str) -> bool:
    if not paths["success"].is_file():
        return False
    try:
        payload = json.loads(paths["success"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if payload.get("status") != "PASS" or payload.get("run_signature") != run_signature:
        return False
    for key in ("checkpoint", "oof", "metrics", "attributions"):
        path = paths[key]
        declaration = payload.get("artifacts", {}).get(key, {})
        if not path.is_file() or declaration.get("sha256") != file_sha256(path):
            return False
    return True


def _train_fold(
    *,
    work: Path,
    cancer: str,
    fold: int,
    expression: pd.DataFrame,
    activity: pd.DataFrame,
    patient_map: pd.Series,
    fold_manifest: pd.DataFrame,
    config: ContinuousActivityConfig,
    min_train_detection: float,
    run_signature: str,
    base_seed: int,
) -> dict[str, Any]:
    paths = _fold_paths(work, cancer, fold)
    paths["root"].mkdir(parents=True, exist_ok=True)
    split = assign_outer_split(fold_manifest, fold, n_folds=N_FOLDS, validation_offset=1)
    sample_ids = {
        name: sorted(split.loc[split.split.eq(name), "sample_id"].astype(str))
        for name in ("train", "validation", "test")
    }
    if min(map(len, sample_ids.values())) < 3:
        raise ContinuousActivityError(f"{cancer}/fold{fold} has a split below three patients")

    train_expression = expression.reindex(sample_ids["train"])
    train_values = train_expression.to_numpy(float)
    detection = np.mean(np.isfinite(train_values) & (train_values > 0), axis=0)
    variance = np.nanvar(train_values, axis=0, ddof=0)
    keep = (detection >= float(min_train_detection)) & np.isfinite(variance) & (variance > 1.0e-8)
    feature_ids = expression.columns.to_numpy(str)[keep]
    if len(feature_ids) < 2:
        raise ContinuousActivityError(f"{cancer}/fold{fold} has fewer than two train-only features")
    pathway_ids = activity.columns.to_numpy(str)
    x = {
        name: expression.reindex(sample_ids[name], columns=feature_ids).to_numpy(float)
        for name in sample_ids
    }
    y = {
        name: activity.reindex(sample_ids[name], columns=pathway_ids).to_numpy(float)
        for name in sample_ids
    }
    seed = int(base_seed + fold * 1009 + _stable_cancer_seed(cancer) % 1_000_003)
    fitted = fit_fresh_pca_ridge_head(
        x["train"],
        y["train"],
        x["validation"],
        y["validation"],
        x["test"],
        y["test"],
        feature_ids=feature_ids,
        pathway_ids=pathway_ids,
        seed=seed,
        config=config,
    )
    temporary_checkpoint = paths["checkpoint"].with_name(".CONTINUOUS_HEAD.partial.npz")
    fitted.model.write(temporary_checkpoint)
    os.replace(temporary_checkpoint, paths["checkpoint"])

    n_test, n_pathways = fitted.test_prediction.shape
    oof = pd.DataFrame(
        {
            "cancer_id": cancer,
            "patient_fold_id": int(fold),
            "sample_id": np.repeat(np.asarray(sample_ids["test"], dtype=str), n_pathways),
            "patient_id": np.repeat(
                patient_map.reindex(sample_ids["test"]).astype(str).to_numpy(), n_pathways
            ),
            "pathway_id": np.tile(pathway_ids, n_test),
            "observed_activity": y["test"].reshape(-1),
            "predicted_activity": fitted.test_prediction.reshape(-1),
            "null_predicted_activity": np.tile(fitted.model.outcome_mean, n_test),
            "split": "outer_test",
            "available": True,
            "unavailable_reason": None,
            "analysis_version": ANALYSIS_VERSION,
            "module_id": MODULE_ID,
        }
    )
    metrics = fitted.metrics.copy()
    metrics.insert(0, "patient_fold_id", int(fold))
    metrics.insert(0, "cancer_id", cancer)
    null_residual = y["test"] - fitted.model.outcome_mean
    metrics["test_null_rmse"] = np.sqrt(np.mean(np.square(null_residual), axis=0))
    metrics["test_mse_improvement"] = (
        np.square(metrics.test_null_rmse) - np.square(metrics.test_rmse)
    )
    metrics["metric_available"] = np.isfinite(metrics.test_r2) & np.isfinite(metrics.test_spearman)
    metrics["unavailable_reason"] = np.where(
        metrics.metric_available, None, "CONSTANT_OUTER_TEST_OUTCOME_OR_PREDICTION"
    )
    metrics["analysis_version"] = ANALYSIS_VERSION
    metrics["module_id"] = MODULE_ID
    attributions = fitted.attributions.copy()
    attributions.insert(0, "patient_fold_id", int(fold))
    attributions.insert(0, "cancer_id", cancer)
    attributions["direction"] = np.where(
        attributions.standardized_coefficient.ge(0), "positive", "negative"
    )
    attributions["analysis_version"] = ANALYSIS_VERSION
    attributions["module_id"] = MODULE_ID
    _atomic_parquet(oof, paths["oof"])
    _atomic_parquet(metrics, paths["metrics"])
    _atomic_parquet(attributions, paths["attributions"])
    artifacts = {
        key: {
            "path": str(paths[key].relative_to(work).as_posix()),
            "bytes": int(paths[key].stat().st_size),
            "sha256": file_sha256(paths[key]),
            "rows": int(
                {"oof": len(oof), "metrics": len(metrics), "attributions": len(attributions)}.get(
                    key, 1
                )
            ),
        }
        for key in ("checkpoint", "oof", "metrics", "attributions")
    }
    payload = {
        "status": "PASS",
        "run_format": RUN_FORMAT,
        "run_signature": run_signature,
        "analysis_version": ANALYSIS_VERSION,
        "module_id": MODULE_ID,
        "cancer_id": cancer,
        "patient_fold_id": int(fold),
        "split_counts": {key: int(len(value)) for key, value in sample_ids.items()},
        "train_only_expression_features": int(len(feature_ids)),
        "pathways": int(len(pathway_ids)),
        "checkpoint": fitted.model.metadata(),
        "artifacts": artifacts,
    }
    _atomic_json(paths["success"], payload)
    return payload


def _copy_query(connection, sql: str, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    quoted = temporary.as_posix().replace("'", "''")
    connection.execute(
        f"COPY ({sql}) TO '{quoted}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)"
    )
    os.replace(temporary, destination)


def _materialize_release(work: Path, top_attributions: int) -> dict[str, Any]:
    import duckdb

    connection = duckdb.connect()
    fold_glob = (work / "folds" / "**" / "OOF.parquet").as_posix().replace("'", "''")
    metric_glob = (work / "folds" / "**" / "METRICS.parquet").as_posix().replace("'", "''")
    attribution_glob = (work / "folds" / "**" / "ATTRIBUTIONS.parquet").as_posix().replace("'", "''")
    connection.execute(
        f"CREATE VIEW oof AS SELECT * FROM read_parquet('{fold_glob}', hive_partitioning=false)"
    )
    connection.execute(
        f"CREATE VIEW metrics AS SELECT * FROM read_parquet('{metric_glob}', hive_partitioning=false)"
    )
    connection.execute(
        f"CREATE VIEW raw_attribution AS SELECT * FROM read_parquet('{attribution_glob}', hive_partitioning=false)"
    )
    oof_rows, oof_keys = connection.execute(
        "SELECT count(*), count(DISTINCT (cancer_id, sample_id, pathway_id)) FROM oof"
    ).fetchone()
    if int(oof_rows) != int(oof_keys):
        raise ContinuousActivityError("OOF predictions duplicate cancer/sample/pathway keys")
    if connection.execute(
        "SELECT count(*) FROM oof WHERE NOT isfinite(observed_activity) OR NOT isfinite(predicted_activity)"
    ).fetchone()[0]:
        raise ContinuousActivityError("OOF predictions contain non-finite values")
    fold_count = connection.execute(
        "SELECT count(DISTINCT patient_fold_id) FROM metrics"
    ).fetchone()[0]
    if int(fold_count) != N_FOLDS:
        raise ContinuousActivityError("Continuous release does not contain five folds")

    oof_path = work / "continuous_pathway_activity_oof.parquet"
    metric_path = work / "continuous_pathway_activity_metrics.parquet"
    attribution_path = work / "continuous_pathway_activity_attributions.parquet"
    _copy_query(
        connection,
        "SELECT * FROM oof ORDER BY cancer_id, sample_id, pathway_id",
        oof_path,
    )
    _copy_query(
        connection,
        "SELECT * FROM metrics ORDER BY cancer_id, patient_fold_id, pathway_id",
        metric_path,
    )
    attribution_sql = f"""
        WITH aggregate AS (
          SELECT cancer_id, pathway_id, lncrna_id,
                 avg(standardized_coefficient) AS mean_standardized_coefficient,
                 avg(abs(standardized_coefficient)) AS mean_absolute_standardized_coefficient,
                 count(DISTINCT patient_fold_id) AS selected_folds,
                 sum(CASE WHEN standardized_coefficient > 0 THEN 1 ELSE 0 END) AS positive_folds,
                 sum(CASE WHEN standardized_coefficient < 0 THEN 1 ELSE 0 END) AS negative_folds
          FROM raw_attribution
          GROUP BY cancer_id, pathway_id, lncrna_id
        ), ranked AS (
          SELECT *, selected_folds / {float(N_FOLDS)} AS selection_stability,
                 greatest(positive_folds, negative_folds) / selected_folds AS sign_consistency,
                 CASE WHEN mean_standardized_coefficient >= 0 THEN 'positive' ELSE 'negative' END AS direction,
                 row_number() OVER (
                   PARTITION BY cancer_id, pathway_id
                   ORDER BY mean_absolute_standardized_coefficient DESC, lncrna_id
                 ) AS attribution_rank
          FROM aggregate
        )
        SELECT *, '{ANALYSIS_VERSION}' AS analysis_version, '{MODULE_ID}' AS module_id
        FROM ranked WHERE attribution_rank <= {int(top_attributions)}
        ORDER BY cancer_id, pathway_id, attribution_rank
    """
    _copy_query(connection, attribution_sql, attribution_path)
    model_mse, null_mse = connection.execute(
        "SELECT avg(pow(observed_activity-predicted_activity, 2)), "
        "avg(pow(observed_activity-null_predicted_activity, 2)) FROM oof"
    ).fetchone()
    improvement = float(null_mse - model_mse)
    outcome = "INCREMENT" if improvement > 1.0e-4 else "NO_INCREMENT"
    artifacts = {}
    for key, path in {
        "v32_continuous_pathway_activity_oof": oof_path,
        "v32_continuous_pathway_activity_metrics": metric_path,
        "v32_continuous_pathway_activity_attributions": attribution_path,
    }.items():
        rows = int(
            connection.execute(
                f"SELECT count(*) FROM read_parquet('{path.as_posix().replace(chr(39), chr(39) * 2)}')"
            ).fetchone()[0]
        )
        artifacts[key] = {
            "path": path.name,
            "rows": rows,
            "bytes": int(path.stat().st_size),
            "sha256": file_sha256(path),
        }
    return {
        "artifacts": artifacts,
        "model_mse": float(model_mse),
        "null_mse": float(null_mse),
        "mse_improvement": improvement,
        "performance_outcome": outcome,
        "scientific_status": "READY" if outcome == "INCREMENT" else "DIAGNOSTIC_ONLY",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expression-root", default="artifacts/input/formal_lncRNA_expression")
    parser.add_argument("--activity-root", default="artifacts/input/formal_pathway_activity")
    parser.add_argument(
        "--fold-manifest",
        default="artifacts/v32_patient_fold_authority_20260829_r1/SAMPLE_PATIENT_FOLD_MAP.tsv",
    )
    parser.add_argument(
        "--patient-fold-authority-receipt",
        default="artifacts/v32_patient_fold_authority_20260829_r1/PATIENT_FOLD_AUTHORITY_RECEIPT.json",
    )
    parser.add_argument("--core-embedding-root", default="artifacts/v32_full_multitask/core_embeddings")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--cancers", help="Optional comma-separated pilot subset")
    parser.add_argument("--components", type=int, default=64)
    parser.add_argument("--top-attributions", type=int, default=25)
    parser.add_argument("--min-train-detection", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output = Path(args.output_root).resolve()
    if output.exists():
        raise ContinuousActivityError(f"Continuous release refuses overwrite: {output}")
    expression_root = Path(args.expression_root).resolve()
    activity_root = Path(args.activity_root).resolve()
    fold_path = Path(args.fold_manifest).resolve()
    fold_receipt_path = Path(args.patient_fold_authority_receipt).resolve()
    core_root = Path(args.core_embedding_root).resolve()
    from cc_hhgt.v32.patient_fold_authority import (
        validate_frozen_v32_patient_fold_binding,
    )

    patient_authority_audit = validate_frozen_v32_patient_fold_binding(
        fold_path, fold_receipt_path
    )
    all_cancers = sorted(
        path.name.split("=", 1)[1]
        for path in expression_root.glob("cancer_id=*")
        if path.is_dir()
    )
    cancers = (
        sorted({value.strip() for value in args.cancers.split(",") if value.strip()})
        if args.cancers
        else all_cancers
    )
    if not cancers or not set(cancers).issubset(all_cancers):
        raise ContinuousActivityError("Requested cancers are empty or outside expression source")
    if not 0 < float(args.min_train_detection) <= 1:
        raise ContinuousActivityError("min-train-detection must be in (0,1]")
    config = ContinuousActivityConfig(
        n_components=int(args.components), top_attributions=int(args.top_attributions)
    )
    config.validate()

    expression_manifest = _tree_manifest(expression_root, cancers)
    activity_manifest = _tree_manifest(activity_root, cancers)
    fold_sha = file_sha256(fold_path)
    core_manifest_path = core_root / "CORE_EMBEDDING_MANIFEST.json"
    if not core_manifest_path.is_file():
        raise ContinuousActivityError("Current V3.2 core embedding manifest is missing")
    source_payload = {
        "analysis_version": ANALYSIS_VERSION,
        "module_id": MODULE_ID,
        "cancers": cancers,
        "components": int(args.components),
        "top_attributions": int(args.top_attributions),
        "min_train_detection": float(args.min_train_detection),
        "seed": int(args.seed),
        "expression_tree_sha256": expression_manifest["tree_sha256"],
        "activity_tree_sha256": activity_manifest["tree_sha256"],
        "fold_manifest_sha256": fold_sha,
        "patient_fold_authority_receipt_sha256": file_sha256(fold_receipt_path),
        "patient_fold_authority_binding": patient_authority_audit,
        "core_embedding_manifest_sha256": file_sha256(core_manifest_path),
        "script_sha256": file_sha256(Path(__file__).resolve()),
        "module_sha256": file_sha256(ROOT / "cc_hhgt" / "v32" / "continuous_activity.py"),
    }
    run_signature = hashlib.sha256(
        json.dumps(source_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    work = output.parent / f".{output.name}.work"
    if work.exists():
        if not args.resume:
            raise ContinuousActivityError(f"Partial run exists; pass --resume: {work}")
        state = json.loads((work / "RUN_STATE.json").read_text(encoding="utf-8"))
        if state.get("run_signature") != run_signature:
            raise ContinuousActivityError("Resume signature does not match current code/inputs")
    else:
        work.mkdir(parents=True)
        _atomic_json(
            work / "RUN_STATE.json",
            {
                "status": "RUN_IN_PROGRESS",
                "run_format": RUN_FORMAT,
                "run_signature": run_signature,
                **source_payload,
            },
        )
        _atomic_json(
            work / "SOURCE_MANIFEST.json",
            {
                "expression": expression_manifest,
                "activity": activity_manifest,
                "fold_manifest": {"path": str(fold_path), "sha256": fold_sha},
                "patient_fold_authority_receipt": {
                    "path": str(fold_receipt_path),
                    "sha256": file_sha256(fold_receipt_path),
                },
                "activity_source_role": "provenance_audited_standardized_source",
                "activity_is_training_target_not_historical_model_output": True,
                "old_checkpoints_used": False,
                "old_predictions_used": False,
            },
        )

    fold_manifest = _load_fold_manifest(fold_path, cancers)
    embedding_audits = {}
    for fold in range(N_FOLDS):
        embedding_path = core_root / f"patient_fold={fold}" / "lncRNA.parquet"
        audit = audit_frozen_lnc_embeddings(pd.read_parquet(embedding_path))
        audit.update({"patient_fold_id": fold, "path": str(embedding_path), "sha256": file_sha256(embedding_path)})
        embedding_audits[str(fold)] = audit
    _atomic_json(
        work / "FROZEN_CORE_LNCRNA_EMBEDDING_USABILITY_AUDIT.json",
        {
            "status": "PASS",
            "decision": "FRESH_FOLD_LOCAL_EXPRESSION_PCA",
            "core_embeddings_used_as_continuous_features": False,
            "reason": "CONSTANT_LNCRNA_EMBEDDINGS_CANNOT_DISTINGUISH_LNCRNAS",
            "folds": embedding_audits,
        },
    )

    checkpoint_rows = []
    for cancer in cancers:
        expression, activity, patient_map, local_folds = _read_cancer_matrices(
            expression_root, activity_root, fold_manifest, cancer
        )
        for fold in range(N_FOLDS):
            paths = _fold_paths(work, cancer, fold)
            if args.resume and _valid_completed_fold(paths, run_signature):
                payload = json.loads(paths["success"].read_text(encoding="utf-8"))
            else:
                payload = _train_fold(
                    work=work,
                    cancer=cancer,
                    fold=fold,
                    expression=expression,
                    activity=activity,
                    patient_map=patient_map,
                    fold_manifest=local_folds,
                    config=config,
                    min_train_detection=float(args.min_train_detection),
                    run_signature=run_signature,
                    base_seed=int(args.seed),
                )
            checkpoint_rows.append(
                {
                    "cancer_id": cancer,
                    "patient_fold_id": fold,
                    "checkpoint_sha256": payload["artifacts"]["checkpoint"]["sha256"],
                    "final_parameter_sha256": payload["checkpoint"]["final_parameter_sha256"],
                    "initial_parameter_sha256": payload["checkpoint"]["initial_parameter_sha256"],
                    "selected_alpha": payload["checkpoint"]["selected_alpha"],
                    "n_features": payload["checkpoint"]["n_features"],
                    "n_components": payload["checkpoint"]["n_components"],
                    "n_pathways": payload["checkpoint"]["n_pathways"],
                }
            )
            print(
                json.dumps(
                    {
                        "status": "FOLD_COMPLETE",
                        "cancer_id": cancer,
                        "patient_fold_id": fold,
                        "features": payload["checkpoint"]["n_features"],
                        "pathways": payload["checkpoint"]["n_pathways"],
                    }
                ),
                flush=True,
            )

    checkpoint_manifest_path = work / "CHECKPOINT_MANIFEST.parquet"
    _atomic_parquet(pd.DataFrame(checkpoint_rows), checkpoint_manifest_path)
    release = _materialize_release(work, int(args.top_attributions))
    checkpoint_manifest_sha = file_sha256(checkpoint_manifest_path)
    binding = {
        "binding_format": BINDING_FORMAT,
        "status": "PASS",
        "analysis_version": ANALYSIS_VERSION,
        "model_version": "V3.2",
        "module_id": MODULE_ID,
        "target_level": "pathway_activity",
        "run_signature": run_signature,
        "new_training": {
            "completed": True,
            "model_version": "V3.2",
            "run_id": output.name,
            "initialization": "random",
            "new_parameters_from_scratch": True,
            "checkpoint_format": CHECKPOINT_FORMAT,
            "checkpoint_sha256": checkpoint_manifest_sha,
            "input_roles": [
                "provenance_audited_standardized_source",
                "split_definition",
                "task_definition",
            ],
            "legacy_derived_inputs": [],
            "old_checkpoint_loaded": False,
            "old_predictions_used_as_features": False,
        },
        "folds": N_FOLDS,
        "cancers": len(cancers),
        "primary_exact_pathway_ranking_changed": False,
        "independent_from_exact_primary": True,
        "unavailable_encoding": "null_with_reason",
        "unavailable_fill_value": None,
        "source": source_payload,
        "checkpoint_manifest": {
            "path": checkpoint_manifest_path.name,
            "rows": len(checkpoint_rows),
            "bytes": checkpoint_manifest_path.stat().st_size,
            "sha256": checkpoint_manifest_sha,
        },
        **release,
    }
    _atomic_json(work / "OUTPUT_BINDING.json", binding)
    binding_sha = file_sha256(work / "OUTPUT_BINDING.json")
    success = {
        "status": "PASS",
        "run_format": RUN_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "model_version": "V3.2",
        "module_id": MODULE_ID,
        "run_signature": run_signature,
        "output_binding_sha256": binding_sha,
        "cancers": len(cancers),
        "folds": N_FOLDS,
        "fresh_checkpoints": len(checkpoint_rows),
        "performance_outcome": release["performance_outcome"],
        "scientific_status": release["scientific_status"],
        "primary_exact_pathway_ranking_changed": False,
    }
    _atomic_json(work / "SUCCESS.json", success)
    os.replace(work, output)
    print(json.dumps(success, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
