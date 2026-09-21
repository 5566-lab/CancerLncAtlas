"""OOF stacking for the three already-trained strict graph experts.

This module never trains or modifies R-GCN, HGT, or CC-HHGT checkpoints.  It
only reads their out-of-fold predictions and learns an availability-masked
stacking gate.  The gate is grouped by cancer so a cancer's stacker prediction
is produced by a gate trained on other cancers.
"""
from __future__ import annotations

import glob
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .moe_model import AvailabilityMaskedMoE, fit_moe, predict_moe, probability_to_logit, sigmoid_numpy

KEYS = ["cancer_id", "lncrna_id", "pathway_family_id"]
MODEL_ALIASES = {
    "rgcn": "rgcn",
    "hgt": "hgt",
    "cc_hhgt_strict": "cc_hhgt",
}


def _metrics(labels: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(labels) & np.isfinite(probability)
    labels = labels[valid]
    probability = probability[valid]
    if not len(labels):
        return {"auprc": math.nan, "auroc": math.nan, "brier": math.nan}
    return {
        "auprc": float(average_precision_score(labels, probability)) if len(np.unique(labels)) > 1 else math.nan,
        "auroc": float(roc_auc_score(labels, probability)) if len(np.unique(labels)) > 1 else math.nan,
        "brier": float(brier_score_loss(labels, probability)),
    }


def _read_prediction_file(path: Path, model_name: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if "split" in frame:
        frame = frame.loc[frame["split"].astype(str).eq("test")].copy()
    required = {*KEYS, "calibrated_probability"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"{path} lacks graph prediction columns: {missing}")
    keep = [*KEYS, "calibrated_probability"]
    for column in ["proxy_label", "label_class", "seed", "fold_id"]:
        if column in frame:
            keep.append(column)
    frame = frame[keep].copy()
    frame["model_name"] = model_name
    return frame


def discover_model_prediction_files(strict_root: Path, model_name: str) -> list[Path]:
    explicit = os.getenv(f"CC_HHGT_{model_name.upper()}_OOF_GLOB")
    if explicit:
        return sorted(Path(path) for path in glob.glob(explicit, recursive=True))
    files = sorted((strict_root / model_name).glob("LOCO_*/seed_*/prediction_pathway_calibrated.parquet"))
    if files:
        return files
    return sorted((strict_root / model_name).glob("LOCO_*/seed_*/prediction.parquet"))


def load_model_oof(strict_root: Path, model_name: str, expected_seeds: int = 3) -> pd.DataFrame:
    files = discover_model_prediction_files(strict_root, model_name)
    if not files:
        raise FileNotFoundError(f"No OOF prediction files found for {model_name} under {strict_root}")
    parts = [_read_prediction_file(path, model_name) for path in files]
    raw = pd.concat(parts, ignore_index=True, sort=False)
    alias = MODEL_ALIASES[model_name]
    agg_spec: dict[str, tuple[str, object]] = {
        f"{alias}_probability": ("calibrated_probability", "mean"),
        f"{alias}_probability_sd": ("calibrated_probability", "std"),
        f"{alias}_n_seeds": ("calibrated_probability", "count"),
    }
    if "proxy_label" in raw:
        agg_spec["proxy_label"] = ("proxy_label", "mean")
    result = raw.groupby(KEYS, observed=True, as_index=False).agg(**agg_spec)
    result[f"{alias}_available"] = result[f"{alias}_n_seeds"].gt(0).astype("int8")
    result[f"{alias}_seed_complete"] = result[f"{alias}_n_seeds"].ge(expected_seeds).astype("int8")
    return result


def load_primary_full_oof(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=KEYS)
    frame = pd.read_parquet(path)
    required = {*KEYS, "cross_cancer_probability"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"{path} lacks primary strict columns: {missing}")
    keep = [*KEYS, "cross_cancer_probability"]
    for column in ["cross_cancer_probability_sd", "direction_probability", "proxy_label"]:
        if column in frame:
            keep.append(column)
    frame = frame[keep].copy()
    frame = frame.rename(columns={
        "cross_cancer_probability": "cc_hhgt_probability",
        "cross_cancer_probability_sd": "cc_hhgt_probability_sd",
    })
    frame["cc_hhgt_available"] = frame["cc_hhgt_probability"].notna().astype("int8")
    return frame.drop_duplicates(KEYS)


def build_graph_expert_frame(
    strict_root: Path,
    primary_oof: Path,
    expected_seeds: int = 3,
) -> pd.DataFrame:
    frames: dict[str, pd.DataFrame] = {}
    for model_name in MODEL_ALIASES:
        try:
            frames[model_name] = load_model_oof(strict_root, model_name, expected_seeds)
        except FileNotFoundError:
            frames[model_name] = pd.DataFrame(columns=KEYS)
    full_primary = load_primary_full_oof(primary_oof)
    if not full_primary.empty:
        cc = frames.get("cc_hhgt_strict", pd.DataFrame(columns=KEYS))
        if cc.empty:
            cc = full_primary
        else:
            cc = full_primary.merge(cc, on=KEYS, how="outer", suffixes=("_full", "_sample"))
            for column in ["cc_hhgt_probability", "cc_hhgt_probability_sd", "proxy_label"]:
                full_col = f"{column}_full"
                sample_col = f"{column}_sample"
                full_series = cc[full_col] if full_col in cc else pd.Series(np.nan, index=cc.index)
                sample_series = cc[sample_col] if sample_col in cc else pd.Series(np.nan, index=cc.index)
                if full_col in cc or sample_col in cc:
                    cc[column] = full_series.combine_first(sample_series)
            for column in ["cc_hhgt_available", "cc_hhgt_n_seeds", "cc_hhgt_seed_complete"]:
                full_col = f"{column}_full"
                sample_col = f"{column}_sample"
                if sample_col in cc:
                    cc[column] = cc[sample_col]
                elif full_col in cc:
                    cc[column] = cc[full_col]
            cc = cc[[c for c in cc.columns if not c.endswith("_full") and not c.endswith("_sample")]]
        frames["cc_hhgt_strict"] = cc

    nonempty = [frame for frame in frames.values() if not frame.empty]
    if not nonempty:
        raise RuntimeError("No graph expert OOF predictions were found")
    merged = nonempty[0]
    for frame in nonempty[1:]:
        merged = merged.merge(frame, on=KEYS, how="outer", suffixes=("", "_dup"))
        duplicate_cols = [c for c in merged.columns if c.endswith("_dup")]
        for dup in duplicate_cols:
            base = dup[:-4]
            if base in merged:
                merged[base] = merged[base].combine_first(merged[dup])
            else:
                merged[base] = merged[dup]
        merged = merged.drop(columns=duplicate_cols)
    proxy_cols = [c for c in merged.columns if c == "proxy_label" or c.startswith("proxy_label_")]
    if proxy_cols:
        merged["proxy_label"] = merged[proxy_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    return merged.sort_values(KEYS).reset_index(drop=True)


def _expert_arrays(frame: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    probabilities = frame.reindex(columns=columns).apply(pd.to_numeric, errors="coerce")
    availability = probabilities.notna().to_numpy(dtype=np.float32)
    logits = np.full(probabilities.shape, np.nan, dtype=np.float32)
    for index, column in enumerate(columns):
        valid = probabilities[column].notna().to_numpy()
        if valid.any():
            logits[valid, index] = probability_to_logit(probabilities.loc[valid, column].to_numpy(np.float32))
    return logits, availability


def _quality(frame: pd.DataFrame, quality_columns: list[str], mean=None, scale=None):
    q = frame.reindex(columns=quality_columns).apply(pd.to_numeric, errors="coerce")
    q = q.replace([np.inf, -np.inf], np.nan)
    if mean is None:
        mean = q.mean().fillna(0).to_numpy(np.float32)
    if scale is None:
        scale = q.std().replace(0, 1).fillna(1).to_numpy(np.float32)
    values = q.fillna(pd.Series(mean, index=q.columns)).to_numpy(np.float32)
    return ((values - mean) / scale).astype(np.float32), np.asarray(mean, np.float32), np.asarray(scale, np.float32)


def _cancer_meta_fold(cancer: str, cancers: list[str], n_folds: int = 5) -> int:
    return cancers.index(cancer) % n_folds


@dataclass
class GraphStackingResult:
    oof: pd.DataFrame
    full: pd.DataFrame
    checkpoint: dict
    metrics: pd.DataFrame


def train_graph_stacker(
    frame: pd.DataFrame,
    *,
    seed: int = 20260731,
    n_folds: int = 5,
    epochs: int = 80,
    patience: int = 10,
) -> GraphStackingResult:
    expert_columns = ["rgcn_probability", "hgt_probability", "cc_hhgt_probability"]
    for column in expert_columns:
        if column not in frame:
            frame[column] = np.nan
    frame = frame.copy()
    frame["proxy_label"] = pd.to_numeric(frame.get("proxy_label"), errors="coerce")
    labeled = frame[frame["proxy_label"].notna()].copy()
    if labeled.empty:
        raise RuntimeError("Graph stacker has no OOF proxy labels")
    cancers = sorted(labeled["cancer_id"].astype(str).unique().tolist())
    labeled["stack_fold"] = labeled["cancer_id"].astype(str).map(lambda x: _cancer_meta_fold(x, cancers, n_folds))
    quality_columns = []
    for prefix in ["rgcn", "hgt", "cc_hhgt"]:
        for suffix in ["probability_sd", "n_seeds", "seed_complete", "available"]:
            column = f"{prefix}_{suffix}"
            if column not in labeled:
                labeled[column] = np.nan if suffix == "probability_sd" else 0
            quality_columns.append(column)

    oof_parts = []
    metric_rows = []
    for heldout in range(n_folds):
        validation_fold = (heldout + 1) % n_folds
        train = labeled.loc[~labeled["stack_fold"].isin([heldout, validation_fold])].copy()
        validation = labeled.loc[labeled["stack_fold"].eq(validation_fold)].copy()
        test = labeled.loc[labeled["stack_fold"].eq(heldout)].copy()
        train_q, mean, scale = _quality(train, quality_columns)
        val_q, _, _ = _quality(validation, quality_columns, mean, scale)
        test_q, _, _ = _quality(test, quality_columns, mean, scale)
        train_logits, train_available = _expert_arrays(train, expert_columns)
        val_logits, val_available = _expert_arrays(validation, expert_columns)
        test_logits, test_available = _expert_arrays(test, expert_columns)
        fit = fit_moe(
            train_logits, train_available, train_q, train["proxy_label"].to_numpy(np.float32),
            val_logits, val_available, val_q, validation["proxy_label"].to_numpy(np.float32),
            seed=seed + heldout, epochs=epochs, patience=patience,
        )
        test_logit, test_weight = predict_moe(fit.model, test_logits, test_available, test_q)
        part = test[[*KEYS, "proxy_label", "stack_fold"]].copy()
        part["graph_ensemble_probability"] = sigmoid_numpy(test_logit)
        for index, expert in enumerate(["rgcn", "hgt", "cc_hhgt"]):
            part[f"graph_weight_{expert}"] = test_weight[:, index].astype("float32")
        oof_parts.append(part)
        metric_rows.append({"heldout_fold": heldout, "validation_fold": validation_fold, **_metrics(test["proxy_label"].to_numpy(float), part["graph_ensemble_probability"].to_numpy(float))})

    validation_fold = n_folds - 1
    final_train = labeled.loc[~labeled["stack_fold"].eq(validation_fold)].copy()
    final_validation = labeled.loc[labeled["stack_fold"].eq(validation_fold)].copy()
    train_q, mean, scale = _quality(final_train, quality_columns)
    val_q, _, _ = _quality(final_validation, quality_columns, mean, scale)
    train_logits, train_available = _expert_arrays(final_train, expert_columns)
    val_logits, val_available = _expert_arrays(final_validation, expert_columns)
    fit = fit_moe(
        train_logits, train_available, train_q, final_train["proxy_label"].to_numpy(np.float32),
        val_logits, val_available, val_q, final_validation["proxy_label"].to_numpy(np.float32),
        seed=seed + 100, epochs=epochs, patience=patience,
    )
    full_q, _, _ = _quality(frame.assign(**{c: frame.get(c, 0) for c in quality_columns}), quality_columns, mean, scale)
    full_logits, full_available = _expert_arrays(frame, expert_columns)
    valid = full_available.sum(axis=1) > 0
    full = frame.copy()
    full["graph_ensemble_probability"] = np.nan
    for expert in ["rgcn", "hgt", "cc_hhgt"]:
        full[f"graph_weight_{expert}"] = 0.0
    if valid.any():
        logit, weight = predict_moe(fit.model, full_logits[valid], full_available[valid], full_q[valid])
        full.loc[valid, "graph_ensemble_probability"] = sigmoid_numpy(logit)
        for index, expert in enumerate(["rgcn", "hgt", "cc_hhgt"]):
            full.loc[valid, f"graph_weight_{expert}"] = weight[:, index]
    probabilities = frame[expert_columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    full["graph_expert_disagreement_sd"] = np.nanstd(probabilities, axis=1).astype("float32")
    checkpoint = {
        "model_state": fit.state_dict,
        "n_experts": 3,
        "quality_dim": len(quality_columns),
        "expert_columns": expert_columns,
        "quality_columns": quality_columns,
        "quality_mean": mean,
        "quality_scale": scale,
        "training_source": "already-trained graph expert OOF predictions only",
        "group_split": "cancer-grouped five-fold stacking",
        "seed": seed + 100,
    }
    return GraphStackingResult(
        oof=pd.concat(oof_parts, ignore_index=True),
        full=full,
        checkpoint=checkpoint,
        metrics=pd.DataFrame(metric_rows),
    )


def save_graph_stacking(result: GraphStackingResult, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    result.oof.to_parquet(output_root / "graph_expert_oof_prediction.parquet", index=False, compression="zstd")
    result.full.to_parquet(output_root / "graph_expert_prediction.parquet", index=False, compression="zstd")
    result.metrics.to_csv(output_root / "graph_expert_crossfit_metrics.tsv", sep="\t", index=False)
    checkpoint = dict(result.checkpoint)
    for key in ["quality_mean", "quality_scale"]:
        checkpoint[key] = np.asarray(checkpoint[key], dtype=np.float32)
    torch.save(checkpoint, output_root / "graph_expert_stacker.pt")
    summary = {
        "status": "COMPLETED",
        "oof_rows": int(len(result.oof)),
        "scored_rows": int(result.full["graph_ensemble_probability"].notna().sum()),
        "models": ["R-GCN", "HGT", "CC-HHGT-Strict"],
        "graph_models_retrained_by_this_stage": False,
        "underlying_graph_models_retrained": True,
        "underlying_graph_release": "V2.9 state-node strict retraining",
        "availability_masked": True,
    }
    (output_root / "SUCCESS.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
