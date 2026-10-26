"""Direction-preserving, formal-target V3.2 CNV-only five-fold head.

Unlike the superseded implementation, CNV events never define the target.
Targets are the current fold-local expression--pathway association labels from
the completed local-CNV audit.  Local and pathway CNV remain distinct inputs
and distinct availability fields.  No Mutation input exists in this API.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .cnv_only_training import TCGA_CANCERS, _atomic_json, _canonical_patient
from .genomic_training import (
    ANALYSIS_VERSION,
    N_FOLDS,
    PRIVATE_CHECKPOINT_FORMAT,
    GenomicTrainingConfig,
    _balanced_indices,
    _candidate_core,
    _file_sha256,
    _fit_head,
    _predict_head,
    _read_table,
    _validate_core_manifest,
    load_fold_core_embeddings,
)
from .integrated_model import module_state_sha256


CNV_DOMAIN_FEATURES = (
    "log_pair_callable",
    "pair_callable_fraction",
    "lncrna_signed_mean",
    "lncrna_mean_absolute_cnv",
    "lncrna_amplification_fraction",
    "lncrna_deletion_fraction",
    "pathway_signed_mean",
    "pathway_mean_absolute_cnv",
    "pathway_amplification_fraction",
    "pathway_deletion_fraction",
    "signed_cnv_correlation",
    "direction_concordance_fraction",
)
POSITIVE_LABELS = {"strong_positive", "weak_positive"}
EXPECTED_ASSOCIATION_RUN_ID = "V32_LOCAL_CNV_CONFOUNDING_20260902_R3"


@dataclass(frozen=True)
class CNVCandidateStatistics:
    domain: np.ndarray
    cnv_available: np.ndarray
    lncrna_local_available: np.ndarray
    pathway_available: np.ndarray
    local_callable_n: np.ndarray
    pathway_callable_n: np.ndarray
    pair_callable_n: np.ndarray
    reasons: np.ndarray


class DirectionalCompactCNV:
    """Read-only compact directional CNV matrices for one cancer."""

    def __init__(self, root: str | Path, *, capture=None, cancer="") -> None:
        from .cnv_input_sources import SourceCapture
        self.capture = capture if capture is not None else SourceCapture()
        self.root = Path(root).resolve()
        context = dict(role="compact", cancer=cancer)
        receipt = self.capture.json(self.root / "SUCCESS.json", **context)
        if (
            receipt.get("status") != "SUCCESS"
            or receipt.get("signed_values_preserved") is not True
            or receipt.get("amplification_deletion_separate") is not True
            or receipt.get("absolute_burden_used_as_continuous") is not False
        ):
            raise RuntimeError(f"directional CNV receipt contract drift: {self.root}")
        self.patient_ids = tuple(self.capture.json(self.root / "patient_ids.json", **context))
        self.lncrna_ids = tuple(self.capture.json(self.root / "lncrna_ids.json", **context))
        self.pathway_ids = tuple(self.capture.json(self.root / "pathway_ids.json", **context))
        self.patient_index = {value: index for index, value in enumerate(self.patient_ids)}
        self.lncrna_index = {value: index for index, value in enumerate(self.lncrna_ids)}
        self.pathway_index = {value: index for index, value in enumerate(self.pathway_ids)}
        self.ls = self.capture.array(self.root / "lncrna_signed_mean.npy", **context)
        self.lc = self.capture.array(self.root / "lncrna_callable.npy", **context)
        self.la = self.capture.array(self.root / "lncrna_amplification.npy", **context)
        self.ld = self.capture.array(self.root / "lncrna_deletion.npy", **context)
        self.ps = self.capture.array(self.root / "pathway_signed_mean.npy", **context)
        self.pc = self.capture.array(self.root / "pathway_callable.npy", **context)
        self.pa = self.capture.array(self.root / "pathway_amplification.npy", **context)
        self.pd = self.capture.array(self.root / "pathway_deletion.npy", **context)
        self.capture.check_partition(self.root, receipt)

    def candidate_statistics(
        self,
        candidates: pd.DataFrame,
        patients: Sequence[str],
        min_pair_callable: int,
    ) -> CNVCandidateStatistics:
        size = len(candidates)
        domain = np.zeros((size, len(CNV_DOMAIN_FEATURES)), dtype=np.float32)
        cnv_available = np.zeros(size, dtype=bool)
        local_available = np.zeros(size, dtype=bool)
        pathway_available = np.zeros(size, dtype=bool)
        local_n = np.zeros(size, dtype=np.int32)
        pathway_n = np.zeros(size, dtype=np.int32)
        pair_n = np.zeros(size, dtype=np.int32)
        reasons = np.full(size, "CNV_NO_EXPLICIT_PAIR_CALLABILITY", dtype=object)
        rows = np.asarray(
            [self.patient_index[value] for value in patients if value in self.patient_index],
            dtype=int,
        )
        if not len(rows):
            reasons[:] = "CNV_NOT_AVAILABLE_FOR_CANCER_SPLIT"
            return CNVCandidateStatistics(
                domain, cnv_available, local_available, pathway_available,
                local_n, pathway_n, pair_n, reasons,
            )
        li_all = np.asarray(
            [self.lncrna_index.get(str(value), -1) for value in candidates.lncrna_id],
            dtype=int,
        )
        pi_all = np.asarray(
            [self.pathway_index.get(str(value), -1) for value in candidates.pathway_id],
            dtype=int,
        )
        for start in range(0, size, 512):
            stop = min(size, start + 512)
            li, pi = li_all[start:stop], pi_all[start:stop]
            valid = (li >= 0) & (pi >= 0)
            local_positions = np.flatnonzero(valid)
            destination = start + local_positions
            reasons[start:stop][li < 0] = "CNV_LNCRNA_UNAVAILABLE"
            reasons[start:stop][(li >= 0) & (pi < 0)] = "CNV_PATHWAY_UNAVAILABLE"
            if not len(local_positions):
                continue
            lix, pix = li[valid], pi[valid]
            lc = np.asarray(self.lc[np.ix_(rows, lix)], dtype=bool)
            pc = np.asarray(self.pc[np.ix_(rows, pix)], dtype=bool)
            pair = lc & pc
            ln = lc.sum(axis=0).astype(np.int32)
            pn = pc.sum(axis=0).astype(np.int32)
            count = pair.sum(axis=0).astype(np.int32)
            local_n[destination], pathway_n[destination], pair_n[destination] = ln, pn, count
            local_available[destination] = ln >= int(min_pair_callable)
            pathway_available[destination] = pn >= int(min_pair_callable)
            eligible = count >= int(min_pair_callable)
            if not eligible.any():
                reasons[destination] = "CNV_INSUFFICIENT_PAIR_CALLABILITY"
                continue
            ls = np.asarray(self.ls[np.ix_(rows, lix)], dtype=float)
            ps = np.asarray(self.ps[np.ix_(rows, pix)], dtype=float)
            la = np.asarray(self.la[np.ix_(rows, lix)], dtype=bool)
            ld = np.asarray(self.ld[np.ix_(rows, lix)], dtype=bool)
            pa = np.asarray(self.pa[np.ix_(rows, pix)], dtype=bool)
            pdv = np.asarray(self.pd[np.ix_(rows, pix)], dtype=bool)
            safe = np.maximum(count.astype(float), 1.0)

            def mean_where(values: np.ndarray) -> np.ndarray:
                return np.where(pair, values, 0.0).sum(axis=0) / safe

            lmean, pmean = mean_where(ls), mean_where(ps)
            lcenter = np.where(pair, ls - lmean, 0.0)
            pcenter = np.where(pair, ps - pmean, 0.0)
            denominator = np.sqrt(
                np.sum(lcenter * lcenter, axis=0) * np.sum(pcenter * pcenter, axis=0)
            )
            correlation = np.divide(
                np.sum(lcenter * pcenter, axis=0), denominator,
                out=np.zeros(len(count), dtype=float), where=denominator > 1e-12,
            )
            concordance = mean_where((np.sign(ls) == np.sign(ps)).astype(float))
            values = np.column_stack([
                np.log1p(count), count / len(rows), lmean, mean_where(np.abs(ls)),
                mean_where(la), mean_where(ld), pmean, mean_where(np.abs(ps)),
                mean_where(pa), mean_where(pdv), correlation, concordance,
            ]).astype(np.float32)
            if not np.isfinite(values[eligible]).all():
                raise RuntimeError("finite directional CNV feature contract failed")
            domain[destination] = values
            cnv_available[destination[eligible]] = True
            reasons[destination] = np.where(
                eligible, "", "CNV_INSUFFICIENT_PAIR_CALLABILITY"
            )
        return CNVCandidateStatistics(
            domain, cnv_available, local_available, pathway_available,
            local_n, pathway_n, pair_n, reasons,
        )


def _patient_splits(folds: pd.DataFrame, outer_fold: int) -> dict[str, dict[str, list[str]]]:
    validation_fold = (int(outer_fold) + 1) % N_FOLDS
    result = {name: {} for name in ("train", "validation", "oof")}
    for cancer, local in folds.groupby("cancer_id", observed=True, sort=False):
        result["oof"][cancer] = local.loc[local.patient_fold_id.eq(outer_fold), "patient_id"].tolist()
        result["validation"][cancer] = local.loc[local.patient_fold_id.eq(validation_fold), "patient_id"].tolist()
        result["train"][cancer] = local.loc[
            ~local.patient_fold_id.isin([outer_fold, validation_fold]), "patient_id"
        ].tolist()
    return result


def _formal_labels(
    association_root: Path,
    cancer: str,
    fold: int,
    candidates: pd.DataFrame,
    split: str,
    capture=None,
    authority=None,
) -> np.ndarray:
    if split not in {"train", "validation"}:
        raise ValueError("CNV formal labels permit only train/validation reads")
    column = "label_A0_full" if split == "train" else "label_A0_validation"
    source = association_root / "pair_level" / f"cancer={cancer}" / f"fold={fold}" / "part.parquet"
    if authority is None:
        from .cnv_head_association import association_contract
        authority = (association_contract(association_root) if (association_root/"SUCCESS.json").exists()
                     else {"cohort": None, "run_id": EXPECTED_ASSOCIATION_RUN_ID})
    if authority["cohort"] is not None:
        item = authority["label_sources"].get((cancer, fold))
        if item is None or Path(item["path"]) != source or _file_sha256(source) != item["sha256"]:
            raise ValueError("Native CNV label file differs from completed association authority")
    columns = ["lncrna_id", "pathway_id", column, "run_id"]
    if capture is None:
        frame = pd.read_parquet(source, columns=columns)
    else:
        frame = capture.read(source, lambda p: pd.read_parquet(p, columns=columns),
                             role="formal_labels", cancer=cancer, fold=fold, split=split, columns=columns)
    if len(frame) != len(candidates) or set(frame.run_id.astype(str)) != {authority["run_id"]}:
        raise RuntimeError(f"formal association label authority drift: {cancer}/{fold}/{split}")
    keys = ["lncrna_id", "pathway_id"]
    left = candidates[keys].astype(str).reset_index(drop=True)
    right = frame[keys].astype(str).reset_index(drop=True)
    if not left.equals(right):
        merged = left.merge(frame[keys + [column]], on=keys, how="left", validate="one_to_one")
        if len(merged) != len(left) or merged[column].isna().any():
            raise RuntimeError(f"formal label key closure drift: {cancer}/{fold}/{split}")
        values = merged[column].astype(str)
    else:
        values = frame[column].astype(str)
    if not set(values).issubset(POSITIVE_LABELS | {"unlabeled", "unavailable"}):
        raise RuntimeError(f"unknown formal labels: {cancer}/{fold}/{split}")
    labels = np.where(values.isin(POSITIVE_LABELS), 1.0, 0.0).astype(np.float32)
    labels[values.eq("unavailable").to_numpy()] = np.nan
    return labels


def _metric(labels: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    valid = np.isfinite(labels) & np.isfinite(probability)
    y, p = labels[valid], probability[valid]
    if not len(y):
        return {"n": 0, "available": False}
    clipped = np.clip(p, 1e-7, 1 - 1e-7)
    record: dict[str, Any] = {
        "n": int(len(y)), "available": True, "positive_prevalence": float(y.mean()),
        "brier": float(np.mean((p - y) ** 2)),
        "logloss": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))),
        "auprc": float(average_precision_score(y, p)) if y.sum() else None,
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
    }
    ece = 0.0
    for low in np.linspace(0.0, 1.0, 11)[:-1]:
        high = low + 0.1
        selected = (p >= low) & ((p < high) if high < 1 else (p <= high))
        if selected.any():
            ece += float(selected.mean()) * abs(float(p[selected].mean()) - float(y[selected].mean()))
    record["ece_10bin"] = ece
    return record


def prepare_directional_inputs(*, candidates_path, folds_path, directional_root,
                               association_root, core_manifest_path):
    """Load the trainer's actual inputs without creating outputs or fitting."""
    from .cnv_input_sources import SourceCapture, require_host
    from .cnv_head_association import association_contract
    require_host(); capture = SourceCapture()
    directional_root, association_root = Path(directional_root).resolve(), Path(association_root).resolve()
    core_manifest_path = Path(core_manifest_path).resolve()
    association_success = capture.json(association_root/"SUCCESS.json", role="association_success")
    directional_success = capture.json(directional_root/"SUCCESS.json", role="materialization_success")
    if association_success.get("status") not in {"SUCCESS", "PASS"} or directional_success.get("status") != "SUCCESS":
        raise RuntimeError("CNV head requires completed directional and association inputs")
    authority = association_contract(association_root, candidates=candidates_path, patient_map=folds_path, capture=capture)
    cohort = authority["cohort"]
    cancers = tuple(cohort["cancers"]) if cohort is not None else TCGA_CANCERS
    if cohort is None:
        if int(directional_success.get("cancer_count", -1)) != 33:
            raise RuntimeError("directional CNV aggregate contract drift")
    elif (set(directional_success.get("cancers", {})) != set(cancers)
            or directional_success.get("cancer_count", len(cancers)) != len(cancers)):
        raise ValueError("Directional CNV coverage differs from the declared association cohort")
    raw = capture.read(folds_path, _read_table, role="folds")
    patient_column = "patient_id" if "patient_id" in raw else "sample_id"
    folds = raw[["cancer_id", patient_column, "patient_fold_id"]].copy()
    folds.columns = ["cancer_id", "patient_id", "patient_fold_id"]
    if folds.isna().any().any(): raise ValueError("CNV patient identities contain nulls")
    folds.cancer_id = folds.cancer_id.astype(str).str.upper()
    folds.patient_id = folds.patient_id.map(_canonical_patient)
    values = pd.to_numeric(folds.patient_fold_id, errors="raise")
    if pd.api.types.is_bool_dtype(values) or not values.isin(range(5)).all():
        raise ValueError("CNV patient fold IDs must be integers 0..4")
    folds.patient_fold_id = values.astype(int)
    folds = folds.drop_duplicates()
    selected = folds.loc[folds.cancer_id.isin(cancers)]
    if (selected.patient_id.eq("").any() or selected.patient_id.duplicated().any()
            or any(set(selected.loc[selected.cancer_id.eq(c), "patient_fold_id"]) != set(range(5)) for c in cancers)):
        raise ValueError("CNV cohort requires distinct patients and all five folds per cancer")
    candidates = capture.read(candidates_path,
        lambda p: pd.read_parquet(p, columns=["cancer_id", "lncrna_id", "pathway_id"]), role="candidates",
        columns=["cancer_id", "lncrna_id", "pathway_id"])
    if candidates.isna().any().any() or candidates.duplicated().any():
        raise ValueError("CNV candidates contain null or duplicated keys")
    candidates = candidates.astype(str)
    counts = candidates.groupby("cancer_id").size().to_dict()
    expected_counts = cohort["candidate_rows_by_cancer"] if cohort is not None else {c: 100_000 for c in cancers}
    if counts != expected_counts: raise RuntimeError("formal exact candidate closure drift")
    manifest, core_sha, composite = capture.read(core_manifest_path, _validate_core_manifest, role="core")
    compacts = {c: DirectionalCompactCNV(directional_root/f"cancer={c}", capture=capture, cancer=c) for c in cancers}
    hashes = {"candidate_authority_sha256": _file_sha256(candidates_path), "fold_manifest_sha256": _file_sha256(folds_path),
        "directional_cnv_success_sha256": _file_sha256(directional_root/"SUCCESS.json"),
        "cnv_mapping_sha256": _file_sha256(directional_root/"SUCCESS.json"),
        "association_audit_success_sha256": _file_sha256(association_root/"SUCCESS.json"), "core_manifest_sha256": core_sha}
    return dict(capture=capture, authority=authority, cohort=cohort, cancers=cancers, candidates=candidates,
        folds=folds, manifest=manifest, compacts=compacts, input_hashes=hashes,
        core_manifest_path=core_manifest_path, core_manifest_sha=core_sha, core_composite=composite,
        association_root=association_root)


def collect_directional_examples(prepared, core, fold, settings):
    """Original per-cancer sampling and balancing, over the declared scope."""
    split_patients = _patient_splits(prepared["folds"], fold)
    train_parts, validation_parts = [], []
    cancers = prepared["cancers"]; candidates = prepared["candidates"]
    train_limit = max(2, int(math.ceil(settings.max_train_rows/len(cancers))))
    validation_limit = max(2, int(math.ceil(settings.max_validation_rows/len(cancers))))
    for index, cancer in enumerate(cancers):
        local = candidates.loc[candidates.cancer_id.eq(cancer)].reset_index(drop=True)
        for split, destination, maximum, seed in (
            ("train", train_parts, train_limit, settings.seed+fold),
            ("validation", validation_parts, validation_limit, settings.seed+50_000+fold)):
            stats = prepared["compacts"][cancer].candidate_statistics(local, split_patients[split].get(cancer, ()), settings.min_pair_callable)
            target = _formal_labels(prepared["association_root"], cancer, fold, local, split,
                capture=prepared["capture"], authority=prepared["authority"])
            core_values, core_available = _candidate_core(local, core)
            eligible = np.flatnonzero(stats.cnv_available & core_available & np.isfinite(target))
            if not len(eligible): continue
            rng = np.random.default_rng(seed+index)
            chosen = rng.choice(eligible, size=min(len(eligible), maximum), replace=False)
            destination.append((core_values[chosen], stats.domain[chosen], target[chosen]))
    if not train_parts or not validation_parts:
        raise RuntimeError(f"CNV fold {fold} has no formal train/validation examples")
    outputs = []
    for parts, seed in ((train_parts, settings.seed+fold+100_000), (validation_parts, settings.seed+fold+150_000)):
        values, domain, labels = [np.concatenate([p[i] for p in parts]) for i in range(3)]
        choice = _balanced_indices(labels, len(labels), seed)
        if not len(choice): raise RuntimeError(f"CNV fold {fold} lacks two formal target classes")
        outputs.extend((values[choice], domain[choice], labels[choice]))
    return tuple(outputs)


def run_directional_cnv_only_oof(
    *, candidates_path: str | Path, folds_path: str | Path,
    directional_root: str | Path, association_root: str | Path,
    core_manifest_path: str | Path, output_root: str | Path,
    training_run_id: str, config: GenomicTrainingConfig | None = None,
) -> dict[str, Any]:
    from .cnv_input_sources import require_host
    require_host()
    settings = config or GenomicTrainingConfig()
    settings.validate()
    output = Path(output_root).resolve()
    runtime_only = {"supervisor.log", "audit_supervisor.log", "training.log", "independent_audit.log", "HEAD_PREFLIGHT.json", "LAUNCH_PREFLIGHT.json"}
    unexpected = [path.name for path in output.iterdir() if path.name not in runtime_only] if output.exists() else []
    if unexpected:
        raise RuntimeError(f"CNV-only scientific output reuse refused: {output}: {sorted(unexpected)}")
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output/"LAUNCH_PREFLIGHT.json", {"target_host": "149", "status": "CHECKING_INPUTS",
        "training_started": False, "training_run_id": str(training_run_id)})
    prepared = prepare_directional_inputs(candidates_path=candidates_path, folds_path=folds_path,
        directional_root=directional_root, association_root=association_root, core_manifest_path=core_manifest_path)
    capture = prepared["capture"]; candidates = prepared["candidates"]; folds = prepared["folds"]
    manifest = prepared["manifest"]; compacts = prepared["compacts"]; cancers = prepared["cancers"]
    input_hashes = prepared["input_hashes"]; core_manifest_sha = prepared["core_manifest_sha"]
    core_composite = prepared["core_composite"]; core_manifest_path = prepared["core_manifest_path"]
    association_root = prepared["association_root"]
    scope = {"association_run_id": prepared["authority"]["run_id"], "cancers": list(cancers),
             **({"cohort": prepared["cohort"]} if prepared["cohort"] is not None else {})}
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "INPUT_PRECHECK.json", {
        "format": "CC_HHGT_V3_2_DIRECTIONAL_CNV_ONLY_INPUT_V2", "status": "PASS",
        **input_hashes, **scope, "cnv_domain_features": list(CNV_DOMAIN_FEATURES),
        "target": "current_fold_local_unadjusted_lncrna_exact_pathway_association",
        "mutation_features_used": False, "sealed_test_read": False,
        "old_checkpoint_loaded": False, "old_predictions_used": False,
    })
    checkpoint_root = output / "checkpoints"; checkpoint_root.mkdir()
    prediction_root = output / "cnv_oof_predictions"; prediction_root.mkdir()
    private_root = output / "PRIVATE_EVALUATION"; private_root.mkdir()
    checkpoints: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    validation_metrics: list[dict[str, Any]] = []
    core_hashes_before: dict[str, str] = {}
    for fold in range(N_FOLDS):
        core = load_fold_core_embeddings(core_manifest_path, manifest, fold)
        core_hashes_before.update(core.input_hashes)
        split_patients = _patient_splits(folds, fold)
        train_core, train_domain, train_target, validation_core, validation_domain, validation_target = collect_directional_examples(prepared, core, fold, settings)
        head, initialization, mean, scale, history = _fit_head(
            train_core, train_domain, train_target,
            validation_core, validation_domain, validation_target,
            modality="cnv", fold=fold, config=settings,
        )
        final_parameter_sha = module_state_sha256(head)
        if final_parameter_sha == initialization["initial_parameter_sha256"]:
            raise RuntimeError(f"CNV head parameters did not change: fold {fold}")
        optimizer_steps = int(math.ceil(len(train_target) / settings.batch_size) * len(history))
        import torch
        checkpoint = checkpoint_root / f"cnv_patient_fold_{fold}.pt"
        checkpoint_payload = {
            "checkpoint_format": PRIVATE_CHECKPOINT_FORMAT,
            "analysis_version": ANALYSIS_VERSION, "modality": "cnv",
            "patient_fold": fold, "initialization": initialization,
            "final_parameter_sha256": final_parameter_sha,
            "optimizer_step_count": optimizer_steps,
            "model_state": head.state_dict(), "domain_features": list(CNV_DOMAIN_FEATURES),
            "domain_mean": mean, "domain_scale": scale, "history": history,
            "core_checkpoint_sha256": core.checkpoint_sha256,
            "core_parameter_sha256": core.core_parameter_sha256,
            "target_semantics": "current_formal_unadjusted_association_proxy_label",
            "old_checkpoint_loaded": False, "old_predictions_used": False,
            "mutation_features_used": False,
        }
        torch.save(checkpoint_payload, checkpoint)
        checkpoint_sha = _file_sha256(checkpoint)
        checkpoints.append({
            "fold": fold, "path": str(checkpoint), "sha256": checkpoint_sha,
            "train_rows": len(train_target), "validation_rows": len(validation_target),
            "optimizer_step_count": optimizer_steps,
            "initial_parameter_sha256": initialization["initial_parameter_sha256"],
            "final_parameter_sha256": final_parameter_sha,
            "core_parameter_sha256": core.core_parameter_sha256,
        })
        for cancer in cancers:
            local = candidates.loc[candidates.cancer_id.eq(cancer)].reset_index(drop=True)
            oof_patients = split_patients["oof"].get(cancer, ())
            stats = compacts[cancer].candidate_statistics(local, oof_patients, settings.min_pair_callable)
            core_values, core_available = _candidate_core(local, core)
            mask = stats.cnv_available & core_available
            probability = np.full(len(local), np.nan, dtype=np.float32)
            if mask.any():
                probability[mask] = _predict_head(
                    head, core_values[mask], stats.domain[mask], mean, scale,
                    settings.prediction_batch_size,
                )
            logit = np.full(len(local), np.nan, dtype=np.float32)
            if mask.any():
                clipped = np.clip(probability[mask], 1e-7, 1 - 1e-7)
                logit[mask] = np.log(clipped / (1 - clipped))
            reason = stats.reasons.astype(object).copy()
            reason[stats.cnv_available & ~core_available] = "FROZEN_CORE_ENTITY_UNAVAILABLE"
            reason[mask] = ""
            output_frame = pd.DataFrame({
                "cancer_id": cancer, "lncrna_id": local.lncrna_id,
                "pathway_id": local.pathway_id, "fold_id": fold,
                "cnv_probability": probability, "cnv_logit": logit,
                "lncrna_local_cnv_available": stats.lncrna_local_available,
                "pathway_cnv_available": stats.pathway_available,
                "cnv_available": mask,
                "local_cnv_feature_count": stats.local_callable_n,
                "pathway_cnv_feature_count": stats.pathway_callable_n,
                "callable_patient_n": stats.pair_callable_n,
                "train_patient_n": len(split_patients["train"].get(cancer, ())),
                "validation_patient_n": len(split_patients["validation"].get(cancer, ())),
                "oof_patient_n": len(oof_patients), "unavailable_reason": reason,
                "run_id": str(training_run_id), "model_version": "V3.2",
                "checkpoint_id": f"cnv_patient_fold_{fold}",
                "checkpoint_sha256": checkpoint_sha, **input_hashes,
            })
            path = prediction_root / f"fold_id={fold}" / f"cancer_id={cancer}" / "part.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            output_frame.to_parquet(path, index=False, compression="zstd")
            predictions.append({
                "fold": fold, "cancer": cancer, "path": str(path),
                "sha256": _file_sha256(path), "rows": len(output_frame),
                "available": int(mask.sum()), "unavailable": int((~mask).sum()),
            })

            # Validation is the only target-bearing evaluation permitted before
            # winner lock.  Keep row labels private and publish aggregates only.
            validation_stats = compacts[cancer].candidate_statistics(
                local, split_patients["validation"].get(cancer, ()), settings.min_pair_callable
            )
            validation_labels = _formal_labels(association_root, cancer, fold, local, "validation", capture=capture, authority=prepared["authority"])
            validation_mask = validation_stats.cnv_available & core_available & np.isfinite(validation_labels)
            validation_probability = np.full(len(local), np.nan, dtype=np.float32)
            if validation_mask.any():
                validation_probability[validation_mask] = _predict_head(
                    head, core_values[validation_mask], validation_stats.domain[validation_mask],
                    mean, scale, settings.prediction_batch_size,
                )
            validation_metrics.append({
                "fold_id": fold, "cancer_id": cancer,
                "prediction_coverage": float(validation_mask.mean()),
                **_metric(validation_labels, validation_probability),
            })
        del head, train_core, train_domain, train_target, validation_core, validation_domain, validation_target

    core_hashes_after = {path: _file_sha256(path) for path in core_hashes_before}
    if core_hashes_after != core_hashes_before or _file_sha256(core_manifest_path) != core_manifest_sha:
        raise RuntimeError("frozen core embedding or manifest changed during CNV-only OOF")
    validation_table = pd.DataFrame(validation_metrics)
    validation_table.to_csv(private_root / "validation_metrics.tsv", sep="\t", index=False)
    availability_rows = []
    for cancer in cancers:
        records = [row for row in predictions if row["cancer"] == cancer]
        availability_rows.append({
            "cancer_id": cancer, "folds": len(records),
            "candidate_rows": sum(row["rows"] for row in records),
            "available_rows": sum(row["available"] for row in records),
            "unavailable_rows": sum(row["unavailable"] for row in records),
        })
    availability_name = "cnv_availability.tsv" if prepared["cohort"] is not None else "cnv_availability_33c.tsv"
    pd.DataFrame(availability_rows).to_csv(output / availability_name, sep="\t", index=False)
    consumed_sources = capture.write(output / "CONSUMED_SOURCES.json", kind="cnv", run_id=training_run_id)
    manifest_payload = {
        "format": "CC_HHGT_V3_2_DIRECTIONAL_CNV_HEAD_OOF_V2", "status": "READY_FOR_INDEPENDENT_AUDIT",
        "training_run_id": str(training_run_id), "cnv_only": True,
        "target": "current_fold_local_unadjusted_lncrna_exact_pathway_association",
        "five_fold_oof": True, "checkpoint_count": len(checkpoints),
        "prediction_file_count": len(predictions), "checkpoints": checkpoints,
        "predictions": predictions, "domain_features": list(CNV_DOMAIN_FEATURES),
        "core_manifest_sha256": core_manifest_sha,
        "core_parameter_composite_sha256": core_composite,
        "core_parameters_before_sha256": core_hashes_before,
        "core_parameters_after_sha256": core_hashes_after,
        "mutation_features_used": False, "sealed_test_read": False,
        "old_checkpoint_loaded": False, "old_predictions_used": False,
        "validation_metrics_path": str(private_root / "validation_metrics.tsv"),
        "consumed_sources": consumed_sources,
        "training_config": asdict(settings),
        **input_hashes, **scope,
    }
    _atomic_json(output / "OOF_MANIFEST.json", manifest_payload)
    _atomic_json(output / "CHECKPOINT_MANIFEST.json", {
        "format": "CC_HHGT_V3_2_CNV_CHECKPOINT_MANIFEST_V1",
        "status": "READY_FOR_INDEPENDENT_AUDIT", "checkpoints": checkpoints,
        "fresh_initialization": True, "old_checkpoint_loaded": False,
        "core_parameters_before_sha256": core_hashes_before,
        "core_parameters_after_sha256": core_hashes_after,
    })
    _atomic_json(output / "LINEAGE.json", {
        "format": "CC_HHGT_V3_2_CNV_HEAD_LINEAGE_V1",
        "training_run_id": str(training_run_id), **input_hashes, **scope,
        "oof_manifest_sha256": _file_sha256(output / "OOF_MANIFEST.json"),
        "mutation_features_used": False, "router_used": False,
        "sealed_test_read": False, "primary_or_g012_modified": False,
    })
    _atomic_json(output / "PACKAGE_READY.json", {
        "format": manifest_payload["format"], "status": "READY_FOR_INDEPENDENT_AUDIT",
        "training_run_id": str(training_run_id), "checkpoint_count": 5,
        "prediction_file_count": len(predictions), **scope,
        "oof_manifest_sha256": _file_sha256(output / "OOF_MANIFEST.json"),
    })
    return {"status": "READY_FOR_INDEPENDENT_AUDIT", "checkpoints": 5, "prediction_files": len(predictions)}


__all__ = [
    "CNV_DOMAIN_FEATURES", "CNVCandidateStatistics", "DirectionalCompactCNV",
    "run_directional_cnv_only_oof",
]
