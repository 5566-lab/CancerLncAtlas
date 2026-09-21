#!/usr/bin/env python3
"""Train availability-masked state-specific MoE and create final rankings."""
from __future__ import annotations

import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from cc_hhgt_v26.moe_model import (  # noqa: E402
    AvailabilityMaskedMoE,
    fit_moe,
    predict_moe,
    probability_to_logit,
    sigmoid_numpy,
)
from cc_hhgt.v30_integrity import stable_partition  # noqa: E402

RESULT = Path(os.getenv("CC_HHGT_RESULT_ROOT", str(ROOT / "results")))
INPUT_ROOT = Path(os.getenv("CC_HHGT_STATE_RESULT_ROOT", str(RESULT / "lncrna_state_experts")))
RUN = Path(os.getenv("CC_HHGT_STATE_RUN_ROOT", str(RESULT / "lncrna_state_run")))
OUT = Path(os.getenv("CC_HHGT_STATE_RELEASE_ROOT", str(RESULT / "lncrna_state_release")))
KEYS = ["cancer_id", "patient_fold_id", "lncrna_id", "state_id"]
EXPERTS = ["state_patient_probability", "state_rgcn_probability", "state_hgt_probability", "state_cc_hhgt_strict_probability"]
QUALITY = [
    "state_patient_probability_sd",
    "state_rgcn_probability_sd",
    "state_hgt_probability_sd",
    "state_cc_hhgt_strict_probability_sd",
    "train_detection_rate",
    "train_n_patients",
    "train_state_sd",
]
SEED = int(os.getenv("CC_HHGT_STATE_MOE_SEED", "20260731"))


def fold_index(value: object) -> int:
    match = re.search(r"PF[_-]?(\d+)", str(value), re.I)
    if match:
        return int(match.group(1))
    digits = re.findall(r"\d+", str(value))
    return int(digits[-1]) if digits else stable_partition(str(value), 5, 20260810) + 1


def quality_matrix(frame: pd.DataFrame, mean=None, scale=None):
    value = frame.reindex(columns=QUALITY).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if mean is None:
        mean = value.mean().fillna(0).to_numpy(np.float32)
    if scale is None:
        scale = value.std().replace(0, 1).fillna(1).to_numpy(np.float32)
    array = value.fillna(pd.Series(mean, index=QUALITY)).to_numpy(np.float32)
    return ((array - mean) / scale).astype(np.float32), mean, scale


def expert_arrays(frame: pd.DataFrame):
    probability = frame.reindex(columns=EXPERTS).apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    availability = np.isfinite(probability)
    logits = np.zeros_like(probability, dtype=np.float32)
    logits[availability] = probability_to_logit(probability[availability])
    return logits, availability.astype(np.float32)


def collect() -> pd.DataFrame:
    if not (RUN / "SUCCESS.json").exists():
        raise RuntimeError("lncRNA-state experts are incomplete")
    frames = []
    for path in sorted(INPUT_ROOT.glob("*/*/seed_*/prediction.parquet")):
        frames.append(pd.read_parquet(path))
    if not frames:
        raise RuntimeError("no lncRNA-state prediction files")
    raw = pd.concat(frames, ignore_index=True, sort=False)
    aggregations = {}
    for column in raw.columns:
        if column in {*KEYS, "seed"}:
            continue
        if pd.api.types.is_numeric_dtype(raw[column]):
            aggregations[column] = "mean"
    summary = raw.groupby(KEYS, observed=True).agg(aggregations).reset_index()
    for expert in EXPERTS:
        if expert in raw:
            sd = raw.groupby(KEYS, observed=True)[expert].std().rename(f"{expert}_sd").reset_index()
            summary = summary.merge(sd, on=KEYS, how="left")
        elif expert not in summary:
            summary[expert] = np.nan
            summary[f"{expert}_sd"] = np.nan
    summary["fusion_fold"] = summary["patient_fold_id"].map(fold_index)
    return summary


def train_gate(frame: pd.DataFrame):
    oof_parts = []
    metrics = []
    cancers = sorted(frame["cancer_id"].astype(str).unique())
    for cancer_index, heldout_cancer in enumerate(cancers):
        validation_cancer = cancers[(cancer_index + 1) % len(cancers)]
        # Excluding the entire held-out cancer guarantees that no upstream
        # statistic used to fit the gate contains any patient from the target
        # cancer.  The applied PF record itself was computed from that PF's
        # train patients only, yielding strict nested patient OOF predictions.
        train = frame[~frame["cancer_id"].astype(str).isin([heldout_cancer, validation_cancer])].copy()
        validation = frame[frame["cancer_id"].astype(str).eq(validation_cancer)].copy()
        test = frame[frame["cancer_id"].astype(str).eq(heldout_cancer)].copy()
        tq, mean, scale = quality_matrix(train)
        vq, _, _ = quality_matrix(validation, mean, scale)
        eq, _, _ = quality_matrix(test, mean, scale)
        tl, ta = expert_arrays(train); vl, va = expert_arrays(validation); el, ea = expert_arrays(test)
        fit = fit_moe(tl, ta, tq, train["test_membership_label"].to_numpy(np.float32), vl, va, vq, validation["test_membership_label"].to_numpy(np.float32), seed=SEED + cancer_index)
        logit, weight = predict_moe(fit.model, el, ea, eq)
        part = test[KEYS + ["test_membership_label", "test_effect", "state_direction_probability"]].copy()
        part["lncrna_state_probability"] = sigmoid_numpy(logit)
        for index, expert in enumerate(EXPERTS):
            part[f"state_weight_{expert}"] = weight[:, index]
        oof_parts.append(part)
        metrics.append({
            "heldout_cancer": heldout_cancer,
            "validation_cancer": validation_cancer,
            "heldout_patient_folds": int(test.patient_fold_id.nunique()),
            "target_cancer_rows_used_to_fit_gate": 0,
            **fit.validation_metrics,
        })

    validation_fold = int(frame["fusion_fold"].max())
    train = frame[frame["fusion_fold"].ne(validation_fold)].copy()
    validation = frame[frame["fusion_fold"].eq(validation_fold)].copy()
    tq, mean, scale = quality_matrix(train); vq, _, _ = quality_matrix(validation, mean, scale)
    tl, ta = expert_arrays(train); vl, va = expert_arrays(validation)
    fit = fit_moe(tl, ta, tq, train["test_membership_label"].to_numpy(np.float32), vl, va, vq, validation["test_membership_label"].to_numpy(np.float32), seed=SEED + 100)
    checkpoint = {
        "model_state": fit.state_dict,
        "n_experts": len(EXPERTS),
        "quality_dim": len(QUALITY),
        "expert_columns": EXPERTS,
        "quality_columns": QUALITY,
        "quality_mean": mean,
        "quality_scale": scale,
        "task": "cancer_x_lncrna_x_tumor_state",
        "availability_mask": "masked_softmax",
        "graph_encoder_retrained": False,
        "crossfit_scope": "leave_one_cancer_out_gate_applied_to_patient_fold_oof_features",
        "target_cancer_rows_used_to_fit_oof_gate": 0,
    }
    return pd.concat(oof_parts, ignore_index=True), fit.model, checkpoint, pd.DataFrame(metrics)


def apply_gate(model, checkpoint, frame):
    quality, _, _ = quality_matrix(frame, checkpoint["quality_mean"], checkpoint["quality_scale"])
    logits, availability = expert_arrays(frame)
    valid = availability.sum(axis=1) > 0
    output = np.full(len(frame), np.nan, np.float32)
    weights = np.zeros((len(frame), len(EXPERTS)), np.float32)
    if valid.any():
        logit, weight = predict_moe(model, logits[valid], availability[valid], quality[valid])
        output[valid] = sigmoid_numpy(logit)
        weights[valid] = weight
    return output, weights


def main() -> int:
    frame = collect()
    OUT.mkdir(parents=True, exist_ok=True)
    oof, model, checkpoint, metrics = train_gate(frame)
    oof.to_parquet(OUT / "lncrna_state_moe_oof_prediction.parquet", index=False, compression="zstd")
    gate_root = OUT / "state_moe_gate"; gate_root.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, gate_root / "state_moe.pt")
    metrics.to_csv(gate_root / "crossfit_metrics.tsv", sep="\t", index=False)

    oof_columns = KEYS + ["lncrna_state_probability", *[f"state_weight_{expert}" for expert in EXPERTS]]
    fold = frame.merge(oof[oof_columns], on=KEYS, how="left", validate="one_to_one")
    if fold["lncrna_state_probability"].isna().any():
        raise RuntimeError("Nested state MoE OOF predictions do not cover every patient-fold record")
    fold.to_parquet(OUT / "lncrna_state_fold_prediction.parquet", index=False, compression="zstd")

    agg = {
        "lncrna_state_probability": ["mean", "std", "count"],
        "test_effect": "mean",
        "state_direction_probability": "mean",
    }
    for column in EXPERTS + [f"state_weight_{expert}" for expert in EXPERTS]:
        if column in fold:
            agg[column] = "mean"
    final = fold.groupby(["cancer_id", "lncrna_id", "state_id"], observed=True).agg(agg)
    final.columns = ["_".join([str(x) for x in column if x]) if isinstance(column, tuple) else str(column) for column in final.columns]
    final = final.reset_index().rename(columns={
        "lncrna_state_probability_mean": "lncrna_state_probability",
        "lncrna_state_probability_std": "lncrna_state_probability_sd",
        "lncrna_state_probability_count": "n_patient_folds_available",
        "test_effect_mean": "state_effect",
        "state_direction_probability_mean": "state_direction_probability",
    })
    final["fold_selection_frequency"] = final["n_patient_folds_available"] / 5.0
    final["replication_class"] = np.select(
        [final["n_patient_folds_available"].ge(4), final["n_patient_folds_available"].eq(3)],
        ["robust_core", "replicated"],
        default="exploratory",
    )
    final["direction"] = np.where(final["state_direction_probability"].ge(0.5), "positive", "negative")
    final["prediction_scope"] = "direct_lncrna_tumor_state"
    final.to_parquet(OUT / "lncrna_state_final.parquet", index=False, compression="zstd")
    summary = {
        "status": "COMPLETED",
        "completed_at": datetime.now().isoformat(),
        "fold_rows": len(fold),
        "final_rows": len(final),
        "states": sorted(final["state_id"].astype(str).unique()),
        "experts": EXPERTS,
        "graph_encoder_retrained": False,
        "formal_probability_source": "leave_one_cancer_out_nested_oof_gate",
        "target_cancer_rows_used_to_fit_oof_gate": 0,
    }
    (OUT / "LNCRNA_STATE_MODEL_SUCCESS.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
