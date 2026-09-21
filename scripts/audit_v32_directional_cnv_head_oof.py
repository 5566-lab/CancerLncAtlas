#!/usr/bin/env python3
"""Independent auditor/evaluator for the V3.2 directional CNV-only OOF head.

This program does not import the generator.  It opens every checkpoint and
prediction partition, derives each checkpoint's genuinely unseen OOF label
from the adjacent-fold validation authority only after training has ended,
recomputes metrics, and is the sole writer of the final SUCCESS receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CANCERS = (
    "ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA", "GBM",
    "HNSC", "KICH", "KIRC", "KIRP", "LAML", "LGG", "LIHC", "LUAD", "LUSC",
    "MESO", "OV", "PAAD", "PCPG", "PRAD", "READ", "SARC", "SKCM", "STAD",
    "TGCT", "THCA", "THYM", "UCEC", "UCS", "UVM",
)
POSITIVE = {"strong_positive", "weak_positive"}
EXPECTED_RUN = "V32_DIRECTIONAL_CNV_HEAD_OOF_20260903_R1"
REQUIRED_COLUMNS = {
    "cancer_id", "lncrna_id", "pathway_id", "fold_id", "cnv_probability",
    "cnv_logit", "lncrna_local_cnv_available", "pathway_cnv_available",
    "cnv_available", "local_cnv_feature_count", "pathway_cnv_feature_count",
    "callable_patient_n", "train_patient_n", "validation_patient_n",
    "oof_patient_n", "unavailable_reason", "run_id", "model_version",
    "checkpoint_id", "checkpoint_sha256", "candidate_authority_sha256",
    "fold_manifest_sha256", "directional_cnv_success_sha256", "cnv_mapping_sha256",
    "association_audit_success_sha256", "core_manifest_sha256",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def calibration_coefficients(y: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    p = np.clip(probability.astype(float), 1e-7, 1 - 1e-7)
    x = np.column_stack([np.ones(len(p)), np.log(p / (1 - p))])
    beta = np.array([0.0, 1.0])
    for _ in range(30):
        eta = np.clip(x @ beta, -30, 30)
        mu = 1 / (1 + np.exp(-eta))
        weight = np.maximum(mu * (1 - mu), 1e-8)
        hessian = x.T @ (x * weight[:, None])
        score = x.T @ (y - mu)
        try:
            step = np.linalg.solve(hessian, score)
        except np.linalg.LinAlgError:
            return np.nan, np.nan
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return float(beta[0]), float(beta[1])


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, roc_auc_score
    valid = np.isfinite(y) & np.isfinite(p)
    y = y[valid].astype(float); p = p[valid].astype(float)
    if not len(y):
        return {"evaluation_rows": 0, "auprc": np.nan, "auroc": np.nan,
                "logloss": np.nan, "brier": np.nan, "ece_10bin": np.nan,
                "calibration_intercept": np.nan, "calibration_slope": np.nan,
                "positive_prevalence": np.nan}
    clipped = np.clip(p, 1e-7, 1 - 1e-7)
    ece = 0.0
    for index in range(10):
        low, high = index / 10, (index + 1) / 10
        chosen = (p >= low) & ((p < high) if index < 9 else (p <= high))
        if chosen.any():
            ece += float(chosen.mean()) * abs(float(p[chosen].mean()) - float(y[chosen].mean()))
    intercept, slope = calibration_coefficients(y, p) if len(np.unique(y)) == 2 else (np.nan, np.nan)
    return {
        "evaluation_rows": len(y), "positive_prevalence": float(y.mean()),
        "auprc": float(average_precision_score(y, p)) if y.sum() else np.nan,
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
        "logloss": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))),
        "brier": float(np.mean((p - y) ** 2)), "ece_10bin": ece,
        "calibration_intercept": intercept, "calibration_slope": slope,
    }


def calibration_rows(scope: str, scope_id: str, y: np.ndarray, p: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for index in range(10):
        low, high = index / 10, (index + 1) / 10
        chosen = np.isfinite(y) & np.isfinite(p) & (p >= low) & ((p < high) if index < 9 else (p <= high))
        rows.append({
            "scope": scope, "scope_id": scope_id, "bin_id": index,
            "lower": low, "upper": high, "n": int(chosen.sum()),
            "mean_probability": float(np.mean(p[chosen])) if chosen.any() else np.nan,
            "observed_prevalence": float(np.mean(y[chosen])) if chosen.any() else np.nan,
        })
    return rows


def load_oof_labels(association_root: Path, prediction_fold: int, cancer: str,
                    keys: pd.DataFrame) -> np.ndarray:
    # Audit fold j validation patients are (j+1)%5. Therefore validation from
    # j=(prediction_fold-1)%5 is precisely this checkpoint's unseen OOF fold.
    source_fold = (prediction_fold - 1) % 5
    source = association_root / "pair_level" / f"cancer={cancer}" / f"fold={source_fold}" / "part.parquet"
    frame = pd.read_parquet(source, columns=["lncrna_id", "pathway_id", "label_A0_validation"])
    frame[["lncrna_id", "pathway_id"]] = frame[["lncrna_id", "pathway_id"]].astype(str)
    if not keys.equals(frame[["lncrna_id", "pathway_id"]]):
        frame = keys.merge(frame, on=["lncrna_id", "pathway_id"], how="left", validate="one_to_one")
    value = frame.label_A0_validation.astype(str)
    out = np.where(value.isin(POSITIVE), 1.0, 0.0).astype(float)
    out[value.eq("unavailable").to_numpy()] = np.nan
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--patient-folds", required=True, type=Path)
    parser.add_argument("--association-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if (root / "SUCCESS.json").exists():
        raise RuntimeError("immutable CNV head SUCCESS already exists")
    ready = json.loads((root / "PACKAGE_READY.json").read_text(encoding="utf-8"))
    manifest_path = root / "OOF_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if ready.get("status") != "READY_FOR_INDEPENDENT_AUDIT" or manifest.get("status") != "READY_FOR_INDEPENDENT_AUDIT":
        raise RuntimeError("generator package is not ready for independent audit")
    if ready.get("training_run_id") != EXPECTED_RUN or manifest.get("training_run_id") != EXPECTED_RUN:
        raise RuntimeError("training run id drift")
    if manifest.get("mutation_features_used") or manifest.get("sealed_test_read") or manifest.get("old_checkpoint_loaded"):
        raise RuntimeError("forbidden training provenance flag")
    if len(manifest.get("checkpoints", [])) != 5 or len(manifest.get("predictions", [])) != 165:
        raise RuntimeError("5-fold/165-partition closure drift")
    if manifest.get("core_parameters_before_sha256") != manifest.get("core_parameters_after_sha256"):
        raise RuntimeError("frozen core changed")

    folds = pd.read_csv(args.patient_folds, sep="\t")
    folds.patient_fold_id = pd.to_numeric(folds.patient_fold_id).astype(int)
    if set(folds.patient_fold_id) != set(range(5)) or folds[["cancer_id", "patient_id"]].duplicated().any():
        raise RuntimeError("patient fold authority drift")
    for fold in range(5):
        validation = (fold + 1) % 5
        for cancer in CANCERS:
            local = folds.loc[folds.cancer_id.astype(str).eq(cancer)]
            train_ids = set(local.loc[~local.patient_fold_id.isin([fold, validation]), "patient_id"].astype(str))
            validation_ids = set(local.loc[local.patient_fold_id.eq(validation), "patient_id"].astype(str))
            oof_ids = set(local.loc[local.patient_fold_id.eq(fold), "patient_id"].astype(str))
            if train_ids & validation_ids or train_ids & oof_ids or validation_ids & oof_ids:
                raise RuntimeError(f"patient overlap: {cancer}/{fold}")

    candidates = pd.read_parquet(args.candidates, columns=["cancer_id", "lncrna_id", "pathway_id"]).astype(str)
    if len(candidates) != 3_300_000 or candidates.duplicated().any() or set(candidates.cancer_id) != set(CANCERS):
        raise RuntimeError("formal candidate closure drift")
    candidate_sha = sha256(args.candidates.resolve())

    import torch
    checkpoint_by_fold: dict[int, dict[str, Any]] = {}
    for record in manifest["checkpoints"]:
        fold = int(record["fold"]); path = Path(record["path"]).resolve()
        if fold in checkpoint_by_fold or sha256(path) != record["sha256"]:
            raise RuntimeError(f"checkpoint closure/hash drift: {fold}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if int(payload.get("patient_fold", -1)) != fold or payload.get("modality") != "cnv":
            raise RuntimeError(f"checkpoint identity drift: {fold}")
        if payload.get("old_checkpoint_loaded") or payload.get("mutation_features_used"):
            raise RuntimeError(f"checkpoint forbidden provenance: {fold}")
        if int(payload.get("optimizer_step_count", 0)) <= 0:
            raise RuntimeError(f"checkpoint has no optimizer updates: {fold}")
        initial = payload.get("initialization", {}).get("initial_parameter_sha256")
        if not initial or initial == payload.get("final_parameter_sha256"):
            raise RuntimeError(f"checkpoint is not freshly trained: {fold}")
        checkpoint_by_fold[fold] = record
    if set(checkpoint_by_fold) != set(range(5)):
        raise RuntimeError("checkpoint fold ids drift")

    partition_metrics: list[dict[str, Any]] = []
    calibration: list[dict[str, Any]] = []
    values_by_fold: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {fold: [] for fold in range(5)}
    values_by_cancer: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {cancer: [] for cancer in CANCERS}
    availability: list[dict[str, Any]] = []
    interpretation: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for record in manifest["predictions"]:
        fold, cancer = int(record["fold"]), str(record["cancer"])
        if (fold, cancer) in seen or cancer not in CANCERS:
            raise RuntimeError(f"prediction identity duplication/drift: {fold}/{cancer}")
        seen.add((fold, cancer))
        path = Path(record["path"]).resolve()
        if sha256(path) != record["sha256"]:
            raise RuntimeError(f"prediction SHA drift: {fold}/{cancer}")
        frame = pd.read_parquet(path)
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing or len(frame) != 100_000:
            raise RuntimeError(f"prediction schema/row drift {fold}/{cancer}: {sorted(missing)}")
        if set(frame.cancer_id.astype(str)) != {cancer} or set(frame.fold_id.astype(int)) != {fold}:
            raise RuntimeError(f"prediction partition identity drift: {fold}/{cancer}")
        keys = frame[["lncrna_id", "pathway_id"]].astype(str).reset_index(drop=True)
        authority = candidates.loc[candidates.cancer_id.eq(cancer), ["lncrna_id", "pathway_id"]].reset_index(drop=True)
        if not keys.equals(authority):
            raise RuntimeError(f"candidate key/order drift: {fold}/{cancer}")
        available = frame.cnv_available.astype(bool).to_numpy()
        probability = pd.to_numeric(frame.cnv_probability, errors="coerce").to_numpy(float)
        logit = pd.to_numeric(frame.cnv_logit, errors="coerce").to_numpy(float)
        if np.any(~np.isfinite(probability[available])) or np.any((probability[available] < 0) | (probability[available] > 1)):
            raise RuntimeError(f"available probability contract drift: {fold}/{cancer}")
        if np.any(np.isfinite(probability[~available])) or np.any(np.isfinite(logit[~available])):
            raise RuntimeError(f"typed-null contract drift: {fold}/{cancer}")
        expected_logit = np.log(np.clip(probability[available], 1e-7, 1 - 1e-7) / (1 - np.clip(probability[available], 1e-7, 1 - 1e-7)))
        if not np.allclose(logit[available], expected_logit, atol=2e-5, rtol=2e-5):
            raise RuntimeError(f"probability/logit mismatch: {fold}/{cancer}")
        if set(frame.run_id.astype(str)) != {EXPECTED_RUN} or set(frame.checkpoint_id.astype(str)) != {f"cnv_patient_fold_{fold}"}:
            raise RuntimeError(f"prediction model lineage drift: {fold}/{cancer}")
        if set(frame.checkpoint_sha256.astype(str)) != {checkpoint_by_fold[fold]["sha256"]}:
            raise RuntimeError(f"prediction checkpoint SHA drift: {fold}/{cancer}")
        if set(frame.candidate_authority_sha256.astype(str)) != {candidate_sha}:
            raise RuntimeError(f"candidate SHA lineage drift: {fold}/{cancer}")
        declared = folds.loc[folds.cancer_id.astype(str).eq(cancer)]
        validation_fold = (fold + 1) % 5
        expected_counts = {
            "train_patient_n": int((~declared.patient_fold_id.isin([fold, validation_fold])).sum()),
            "validation_patient_n": int(declared.patient_fold_id.eq(validation_fold).sum()),
            "oof_patient_n": int(declared.patient_fold_id.eq(fold).sum()),
        }
        for column, expected in expected_counts.items():
            if set(pd.to_numeric(frame[column]).astype(int)) != {expected}:
                raise RuntimeError(f"declared patient split count drift: {fold}/{cancer}/{column}")
        forbidden = [column for column in frame if "mutation" in column.lower() or "sealed" in column.lower() or "router" in column.lower()]
        if forbidden:
            raise RuntimeError(f"forbidden prediction fields: {forbidden}")

        labels = load_oof_labels(args.association_root.resolve(), fold, cancer, keys)
        observed = available & np.isfinite(labels)
        score = metrics(labels[observed], probability[observed])
        partition_metrics.append({
            "fold_id": fold, "cancer_id": cancer, "evaluation_split": "OOF",
            "prediction_rows": len(frame), "available_rows": int(available.sum()),
            "prediction_coverage": float(available.mean()), **score,
        })
        calibration.extend(calibration_rows("cancer_fold", f"{cancer}:{fold}", labels, probability))
        values_by_fold[fold].append((labels[observed], probability[observed]))
        values_by_cancer[cancer].append((labels[observed], probability[observed]))
        local_available = frame.lncrna_local_cnv_available.astype(bool).to_numpy()
        pathway_available = frame.pathway_cnv_available.astype(bool).to_numpy()
        for label, selected in {
            "local_available": local_available,
            "local_unavailable": ~local_available,
            "pathway_available": pathway_available,
            "pathway_unavailable": ~pathway_available,
            "both_available": local_available & pathway_available,
            "local_only_available": local_available & ~pathway_available,
            "pathway_only_available": ~local_available & pathway_available,
            "both_unavailable": ~local_available & ~pathway_available,
        }.items():
            finite = selected & np.isfinite(probability)
            interpretation.append({"fold_id": fold, "cancer_id": cancer, "group": label,
                                   "rows": int(selected.sum()), "finite_predictions": int(finite.sum()),
                                   "mean_probability": float(np.mean(probability[finite])) if finite.any() else np.nan})
        sensitivity_path = args.association_root.resolve() / "pair_level" / f"cancer={cancer}" / f"fold={fold}" / "part.parquet"
        sensitivity = pd.read_parquet(
            sensitivity_path,
            columns=["lncrna_id", "pathway_id", "cnv_sensitive", "cnv_independent_candidate"],
        )
        sensitivity[["lncrna_id", "pathway_id"]] = sensitivity[["lncrna_id", "pathway_id"]].astype(str)
        if not keys.equals(sensitivity[["lncrna_id", "pathway_id"]]):
            sensitivity = keys.merge(sensitivity, on=["lncrna_id", "pathway_id"], how="left", validate="one_to_one")
        for label, selected in {
            "train_derived_cnv_sensitive_pair": sensitivity.cnv_sensitive.fillna(False).astype(bool).to_numpy(),
            "train_derived_cnv_independent_pair": sensitivity.cnv_independent_candidate.fillna(False).astype(bool).to_numpy(),
        }.items():
            finite = selected & np.isfinite(probability)
            interpretation.append({"fold_id": fold, "cancer_id": cancer, "group": label,
                                   "rows": int(selected.sum()), "finite_predictions": int(finite.sum()),
                                   "mean_probability": float(np.mean(probability[finite])) if finite.any() else np.nan})

    if seen != {(fold, cancer) for fold in range(5) for cancer in CANCERS}:
        raise RuntimeError("prediction cancer/fold closure drift")

    part = pd.DataFrame(partition_metrics)
    fold_rows = []
    for fold, chunks in values_by_fold.items():
        y = np.concatenate([chunk[0] for chunk in chunks]); p = np.concatenate([chunk[1] for chunk in chunks])
        fold_rows.append({"fold_id": fold, "evaluation_split": "OOF", **metrics(y, p),
                          "prediction_coverage": float(part.loc[part.fold_id.eq(fold), "available_rows"].sum() / part.loc[part.fold_id.eq(fold), "prediction_rows"].sum())})
    cancer_rows = []
    for cancer, chunks in values_by_cancer.items():
        y = np.concatenate([chunk[0] for chunk in chunks]); p = np.concatenate([chunk[1] for chunk in chunks])
        local = part.loc[part.cancer_id.eq(cancer)]
        cancer_rows.append({"cancer_id": cancer, "evaluation_split": "OOF", **metrics(y, p),
                            "prediction_coverage": float(local.available_rows.sum() / local.prediction_rows.sum()),
                            "available_candidate_rows": int(local.available_rows.sum())})
    all_y = np.concatenate([chunk[0] for chunks in values_by_fold.values() for chunk in chunks])
    all_p = np.concatenate([chunk[1] for chunks in values_by_fold.values() for chunk in chunks])
    global_metrics = {
        "format": "CC_HHGT_V3_2_CNV_HEAD_OOF_METRICS_V1", "evaluation_split": "OOF",
        **metrics(all_y, all_p), "prediction_coverage": float(part.available_rows.sum() / part.prediction_rows.sum()),
        "fold_mean": pd.DataFrame(fold_rows).select_dtypes(include=[np.number]).mean().to_dict(),
        "fold_std": pd.DataFrame(fold_rows).select_dtypes(include=[np.number]).std(ddof=1).to_dict(),
    }
    part.to_csv(root / "PRIVATE_EVALUATION" / "cnv_oof_metrics_cancer_fold.tsv", sep="\t", index=False)
    pd.DataFrame(fold_rows).to_csv(root / "cnv_oof_metrics_fold.tsv", sep="\t", index=False)
    pd.DataFrame(cancer_rows).to_csv(root / "cnv_oof_metrics_cancer.tsv", sep="\t", index=False)
    pd.DataFrame(calibration).to_csv(root / "cnv_calibration.tsv", sep="\t", index=False)
    pd.DataFrame(interpretation).to_csv(root / "PRIVATE_EVALUATION" / "availability_interpretation.tsv", sep="\t", index=False)
    write_json(root / "cnv_oof_metrics_global.json", global_metrics)

    audit = {
        "format": "CC_HHGT_V3_2_CNV_OOF_INDEPENDENT_AUDIT_V1", "status": "PASS",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": {"folds": 5, "fresh_checkpoints": 5, "prediction_partitions": 165,
                   "candidate_rows": 16_500_000, "candidate_closure": True,
                   "patient_split_overlap": False, "checkpoint_parameters_changed": True,
                   "old_checkpoint_loaded": False, "core_hash_unchanged": True,
                   "cnv_only_features": True, "typed_null": True, "probability_range": True,
                   "metrics_recomputed_from_unseen_oof_labels": True,
                   "mutation_used": False, "router_used": False, "sealed_test_read": False},
        "oof_manifest_sha256": sha256(manifest_path),
        "global_metrics_sha256": sha256(root / "cnv_oof_metrics_global.json"),
    }
    write_json(root / "CNV_OOF_INDEPENDENT_AUDIT.json", audit)
    report = (
        "# CancerLncAtlas V3.2 directional CNV head\n\n"
        "Status: **PASS** (independently audited)\n\n"
        f"Five fresh checkpoints produced 165 typed OOF partitions. Global OOF AUCPR: {global_metrics['auprc']:.6g}; "
        f"AUROC: {global_metrics['auroc']:.6g}; log loss: {global_metrics['logloss']:.6g}; "
        f"coverage: {global_metrics['prediction_coverage']:.3%}.\n\n"
        "- CNV_HEAD_OOF_COMPLETE\n- CNV_ROUTER_INCREMENT_NOT_TESTED\n"
        "- CNV_NOT_IN_PRIMARY_HHGT\n- CNV_NOT_USED_TO_REDEFINE_TARGET\n"
        "- No Mutation, router, old checkpoint, or sealed test was used.\n"
    )
    (root / "CNV_HEAD_REPORT.md").write_text(report, encoding="utf-8")
    write_json(root / "SUCCESS.json", {
        "format": "CC_HHGT_V3_2_DIRECTIONAL_CNV_HEAD_OOF_FINAL_V1", "status": "SUCCESS",
        "training_run_id": EXPECTED_RUN, "checkpoint_count": 5, "prediction_file_count": 165,
        "CNV_HEAD_OOF_COMPLETE": True, "CNV_ROUTER_INCREMENT_NOT_TESTED": True,
        "CNV_NOT_IN_PRIMARY_HHGT": True, "CNV_NOT_USED_TO_REDEFINE_TARGET": True,
        "independent_audit_sha256": sha256(root / "CNV_OOF_INDEPENDENT_AUDIT.json"),
        "metrics_sha256": sha256(root / "cnv_oof_metrics_global.json"),
    })
    print(json.dumps({"status": "SUCCESS", "folds": 5, "partitions": 165}, sort_keys=True))


if __name__ == "__main__":
    main()
