#!/usr/bin/env python3
"""Final OOF fusion of graph ensemble, patient-native, and event experts.

This stage is intentionally light-weight.  It reuses all trained graph and
patient models and trains only two availability-masked gates:
- discovery: graph ensemble + patient-native + indirect event evidence;
- confidence: discovery experts + direct evidence + frozen historical evidence.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from cc_hhgt.v30_integrity import stable_partition  # noqa: E402
from cc_hhgt_v26.moe_model import AvailabilityMaskedMoE, fit_moe, predict_moe, probability_to_logit, sigmoid_numpy  # noqa: E402

RELEASE = Path(os.getenv("CC_HHGT_RELEASE_DIR", str(ROOT / "results" / "v2_8_cancer_native_moe_release")))
GRAPH_ROOT = Path(os.getenv("CC_HHGT_GRAPH_STACK_ROOT", str(ROOT / "results" / "graph_expert_stacking")))
EVENT_ROOT = Path(os.getenv("CC_HHGT_EVIDENCE_EVENT_ROOT", str(ROOT / "results" / "evidence_event_model")))
KEYS = ["cancer_id", "lncrna_id", "pathway_family_id"]
EXPECTED_FOLDS = 5
SEED = int(os.getenv("CC_HHGT_FULL_FUSION_SEED", "20260731"))
MAX_TRAIN = int(os.getenv("CC_HHGT_FULL_FUSION_MAX_TRAIN_ROWS", "2000000"))

DISCOVERY_EXPERTS = [
    "graph_ensemble_oof_probability",
    "cancer_native_probability",
    "indirect_mechanism_probability",
]
CONFIDENCE_EXPERTS = [
    "graph_ensemble_oof_probability",
    "cancer_native_probability",
    "indirect_mechanism_probability",
    "direct_evidence_probability",
    "evidence_integrated_probability",
]
DISCOVERY_QUALITY = [
    "graph_expert_disagreement_sd", "cancer_native_probability_sd",
    "train_detection_rate", "train_n_patients", "n_evidence_events",
    "n_evidence_pmids", "strict_available", "patient_available",
    "indirect_evidence_available",
]
CONFIDENCE_QUALITY = [
    *DISCOVERY_QUALITY, "direct_evidence_available", "n_direct_events",
    "evidence_crosswalk_max_coverage",
]


def fold_index(value: object) -> int:
    match = re.search(r"PF[_-]?(\d+)", str(value), flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return stable_partition(str(value), EXPECTED_FOLDS, 20260810) + 1


def quality_matrix(frame: pd.DataFrame, columns: list[str], mean=None, scale=None):
    q = frame.reindex(columns=columns).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if mean is None:
        mean = q.mean().fillna(0).to_numpy(np.float32)
    if scale is None:
        scale = q.std().replace(0, 1).fillna(1).to_numpy(np.float32)
    values = q.fillna(pd.Series(mean, index=q.columns)).to_numpy(np.float32)
    return ((values - mean) / scale).astype(np.float32), np.asarray(mean, np.float32), np.asarray(scale, np.float32)


def expert_arrays(frame: pd.DataFrame, columns: list[str]):
    p = frame.reindex(columns=columns).apply(pd.to_numeric, errors="coerce")
    availability = p.notna().to_numpy(np.float32)
    logits = np.full(p.shape, np.nan, np.float32)
    for index, column in enumerate(columns):
        valid = p[column].notna().to_numpy()
        if valid.any():
            logits[valid, index] = probability_to_logit(p.loc[valid, column].to_numpy(np.float32))
    return logits, availability


def subsample(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    if len(frame) <= MAX_TRAIN:
        return frame
    positive = frame.loc[frame["test_membership_label"].ge(0.5)]
    negative = frame.loc[frame["test_membership_label"].lt(0.5)]
    remaining = max(MAX_TRAIN - len(positive), 0)
    if remaining == 0:
        return positive.sample(MAX_TRAIN, random_state=seed)
    negative = negative.sample(min(remaining, len(negative)), random_state=seed)
    return pd.concat([positive, negative], ignore_index=True).sample(frac=1, random_state=seed)


def train_gate(frame: pd.DataFrame, experts: list[str], quality_columns: list[str], name: str):
    frame = frame.copy()
    frame["fusion_fold"] = frame["patient_fold_id"].map(fold_index)
    frame["test_membership_label"] = pd.to_numeric(frame["test_membership_label"], errors="coerce")
    oof_parts = []
    metrics = []
    for heldout in sorted(frame["fusion_fold"].dropna().unique()):
        validation_fold = heldout % EXPECTED_FOLDS + 1
        train = subsample(frame.loc[~frame["fusion_fold"].isin([heldout, validation_fold])].copy(), SEED + int(heldout))
        validation = frame.loc[frame["fusion_fold"].eq(validation_fold)].copy()
        test = frame.loc[frame["fusion_fold"].eq(heldout)].copy()
        tq, mean, scale = quality_matrix(train, quality_columns)
        vq, _, _ = quality_matrix(validation, quality_columns, mean, scale)
        eq, _, _ = quality_matrix(test, quality_columns, mean, scale)
        tl, ta = expert_arrays(train, experts); vl, va = expert_arrays(validation, experts); el, ea = expert_arrays(test, experts)
        fit = fit_moe(
            tl, ta, tq, train["test_membership_label"].to_numpy(np.float32),
            vl, va, vq, validation["test_membership_label"].to_numpy(np.float32),
            seed=SEED + int(heldout),
        )
        logit, weight = predict_moe(fit.model, el, ea, eq)
        part = test[[*KEYS, "patient_fold_id", "test_membership_label"]].copy()
        part[f"{name}_oof_probability"] = sigmoid_numpy(logit)
        for index, expert in enumerate(experts):
            part[f"{name}_weight_{expert}"] = weight[:, index]
        oof_parts.append(part)
        metrics.append({"gate": name, "heldout_fold": int(heldout), "validation_fold": int(validation_fold), **fit.validation_metrics})

    validation_fold = EXPECTED_FOLDS
    train = subsample(frame.loc[frame["fusion_fold"].ne(validation_fold)].copy(), SEED + 100)
    validation = frame.loc[frame["fusion_fold"].eq(validation_fold)].copy()
    tq, mean, scale = quality_matrix(train, quality_columns); vq, _, _ = quality_matrix(validation, quality_columns, mean, scale)
    tl, ta = expert_arrays(train, experts); vl, va = expert_arrays(validation, experts)
    fit = fit_moe(
        tl, ta, tq, train["test_membership_label"].to_numpy(np.float32),
        vl, va, vq, validation["test_membership_label"].to_numpy(np.float32),
        seed=SEED + 100,
    )
    checkpoint = {
        "model_state": fit.state_dict, "n_experts": len(experts), "quality_dim": len(quality_columns),
        "expert_columns": experts, "quality_columns": quality_columns,
        "quality_mean": mean, "quality_scale": scale,
        "training_source": "OOF graph stacker + OOF patient expert + OOF event expert",
        "availability_mask": "masked_softmax",
        "direct_evidence_policy": "excluded from discovery; confidence only",
    }
    return pd.concat(oof_parts, ignore_index=True), fit.model, checkpoint, pd.DataFrame(metrics)


def apply_gate(model: AvailabilityMaskedMoE, checkpoint: dict, frame: pd.DataFrame):
    quality, _, _ = quality_matrix(frame, checkpoint["quality_columns"], checkpoint["quality_mean"], checkpoint["quality_scale"])
    logits, availability = expert_arrays(frame, checkpoint["expert_columns"])
    valid = availability.sum(axis=1) > 0
    final_logit = np.full(len(frame), np.nan, np.float32)
    weights = np.zeros((len(frame), len(checkpoint["expert_columns"])), np.float32)
    if valid.any():
        logit, weight = predict_moe(model, logits[valid], availability[valid], quality[valid])
        final_logit[valid] = logit; weights[valid] = weight
    return final_logit, weights


def main() -> int:
    required = [
        RELEASE / "moe_oof_training_frame.parquet",
        RELEASE / "candidate_union.parquet",
        RELEASE / "final_three_probability_table.parquet",
        GRAPH_ROOT / "graph_expert_oof_prediction.parquet",
        GRAPH_ROOT / "graph_expert_prediction.parquet",
        EVENT_ROOT / "evidence_transformer_oof_prediction.parquet",
        EVENT_ROOT / "evidence_transformer_prediction.parquet",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Full expert fusion inputs missing: {missing}")

    training = pd.read_parquet(RELEASE / "moe_oof_training_frame.parquet")
    graph_oof = pd.read_parquet(GRAPH_ROOT / "graph_expert_oof_prediction.parquet")
    event_oof = pd.read_parquet(EVENT_ROOT / "evidence_transformer_oof_prediction.parquet")
    graph_keep = [*KEYS, "graph_ensemble_probability"]
    graph_oof = graph_oof[graph_keep].drop_duplicates(KEYS).rename(columns={"graph_ensemble_probability": "graph_ensemble_oof_probability"})
    event_keep = [
        *KEYS, "indirect_mechanism_probability", "direct_evidence_probability",
        "neural_evidence_probability", "indirect_evidence_available",
        "direct_evidence_available", "n_evidence_events", "n_evidence_pmids",
        "n_direct_events",
    ]
    event_oof = event_oof[event_keep].drop_duplicates(KEYS)
    training = training.merge(graph_oof, on=KEYS, how="left").merge(event_oof, on=KEYS, how="left")
    # For strict candidates not covered by the sampled multi-model OOF tables,
    # retain the already-OOF CC-HHGT probability as the single available graph expert.
    training["graph_ensemble_oof_probability"] = training["graph_ensemble_oof_probability"].combine_first(training.get("cross_cancer_probability"))
    for column in ["graph_expert_disagreement_sd", "n_evidence_events", "n_evidence_pmids", "n_direct_events"]:
        if column not in training:
            training[column] = 0.0
    training["patient_available"] = training["cancer_native_probability"].notna().astype("int8")
    training["indirect_evidence_available"] = training["indirect_mechanism_probability"].notna().astype("int8")
    training["direct_evidence_available"] = training["direct_evidence_probability"].notna().astype("int8")

    discovery_oof, discovery_model, discovery_ckpt, discovery_metrics = train_gate(training, DISCOVERY_EXPERTS, DISCOVERY_QUALITY, "discovery_full")
    confidence_oof, confidence_model, confidence_ckpt, confidence_metrics = train_gate(training, CONFIDENCE_EXPERTS, CONFIDENCE_QUALITY, "confidence_full")
    full_oof = discovery_oof.merge(confidence_oof, on=[*KEYS, "patient_fold_id", "test_membership_label"], how="outer", validate="one_to_one")
    full_oof.to_parquet(RELEASE / "full_expert_moe_oof_prediction.parquet", index=False, compression="zstd")
    gate_root = RELEASE / "full_expert_moe_gate"; gate_root.mkdir(parents=True, exist_ok=True)
    torch.save(discovery_ckpt, gate_root / "discovery.pt"); torch.save(confidence_ckpt, gate_root / "confidence.pt")
    pd.concat([discovery_metrics, confidence_metrics], ignore_index=True).to_csv(gate_root / "crossfit_metrics.tsv", sep="\t", index=False)

    baseline_path = RELEASE / "final_three_probability_table.parquet"
    baseline_backup = RELEASE / "final_three_probability_table_pre_event_graphstack.parquet"
    if not baseline_backup.exists():
        shutil.copy2(baseline_path, baseline_backup)
    final = pd.read_parquet(baseline_backup)
    graph = pd.read_parquet(GRAPH_ROOT / "graph_expert_prediction.parquet")
    event = pd.read_parquet(EVENT_ROOT / "evidence_transformer_prediction.parquet")
    graph_columns = [c for c in graph.columns if c in KEYS or c.startswith("graph_") or c in {"rgcn_probability", "hgt_probability", "cc_hhgt_probability"}]
    event_columns = [c for c in event.columns if c in KEYS or c.endswith("_probability") or c.endswith("_available") or c.startswith("n_evidence") or c.startswith("n_direct") or c.startswith("n_global")]
    union = pd.concat([
        pd.read_parquet(RELEASE / "candidate_union.parquet")[KEYS],
        graph[KEYS], event[KEYS], final[KEYS],
    ], ignore_index=True).drop_duplicates(KEYS)
    final = union.merge(final, on=KEYS, how="left").merge(graph[graph_columns], on=KEYS, how="left").merge(event[event_columns], on=KEYS, how="left")
    final["graph_ensemble_probability"] = final["graph_ensemble_probability"].combine_first(final.get("cross_cancer_probability"))
    final["strict_available"] = final["graph_ensemble_probability"].notna().astype("int8")
    final["patient_available"] = final["cancer_native_probability"].notna().astype("int8")
    final["indirect_evidence_available"] = final["indirect_mechanism_probability"].notna().astype("int8")
    final["direct_evidence_available"] = final["direct_evidence_probability"].notna().astype("int8")
    for column in set(DISCOVERY_QUALITY + CONFIDENCE_QUALITY):
        if column not in final:
            final[column] = 0.0
    discovery_logit, discovery_weight = apply_gate(discovery_model, discovery_ckpt, final)
    confidence_logit, confidence_weight = apply_gate(confidence_model, confidence_ckpt, final)
    final["discovery_ranking_probability"] = sigmoid_numpy(discovery_logit)
    final["target_evidence_masked_fused_probability"] = final["discovery_ranking_probability"]
    final["fused_confidence_probability"] = sigmoid_numpy(confidence_logit)
    for index, expert in enumerate(DISCOVERY_EXPERTS):
        final[f"discovery_weight_{expert}"] = discovery_weight[:, index]
    for index, expert in enumerate(CONFIDENCE_EXPERTS):
        final[f"confidence_weight_{expert}"] = confidence_weight[:, index]
    final["prediction_scope"] = "graph_stacking_plus_patient_native_plus_direct_indirect_event_transformer"
    final["direct_evidence_used_in_discovery"] = False
    scored = final[["graph_ensemble_probability", "cancer_native_probability", "indirect_mechanism_probability", "direct_evidence_probability", "evidence_integrated_probability"]].notna().any(axis=1)
    unscored = final.loc[~scored].copy(); unscored["failure_reason"] = "no_graph_patient_or_event_expert"
    unscored.to_parquet(RELEASE / "candidate_union_unscored.parquet", index=False, compression="zstd")
    final = final.loc[scored].sort_values(KEYS).reset_index(drop=True)
    final.to_parquet(RELEASE / "final_expert_fusion_table.parquet", index=False, compression="zstd")
    final.to_parquet(baseline_path, index=False, compression="zstd")

    summary = {
        "status": "COMPLETED", "completed_at": datetime.now().isoformat(),
        "final_rows": int(len(final)), "unscored_rows": int(len(unscored)),
        "graph_models_retrained": True, "graph_release": "V2.9 state-node strict retraining", "patient_model_retrained_by_this_stage": False,
        "graph_experts": ["R-GCN", "HGT", "CC-HHGT-Strict"],
        "discovery_experts": DISCOVERY_EXPERTS,
        "confidence_experts": CONFIDENCE_EXPERTS,
        "direct_evidence_used_in_discovery": False,
        "event_transformer_enabled": True,
        "simple_mean_removed": True,
    }
    (RELEASE / "FULL_EXPERT_FUSION_SUCCESS.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
