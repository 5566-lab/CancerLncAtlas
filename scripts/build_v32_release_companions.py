#!/usr/bin/env python3
"""Build exact matched L2 baselines and row-aligned release companions.

This intentionally reuses the already materialized protein-coding-gene
exact-pathway activity.  Association features are recomputed fold-locally from
training patients so Ridge sees exactly the same safe feature contract as L1.
No CC-HHGT optimizer step is executed here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def _probability_from_batches(payload: dict, split: str) -> np.ndarray:
    import torch

    logits = torch.cat([batch["base_logit"].detach().cpu() for batch in payload[f"{split}_batches"]])
    return torch.sigmoid(logits).numpy().astype("float32", copy=False)


def _metrics(labels: np.ndarray, probability: np.ndarray) -> dict[str, float | None]:
    from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

    labels = np.asarray(labels, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-7, 1 - 1e-7)
    return {
        "auprc": float(average_precision_score(labels, probability)),
        "auroc": float(roc_auc_score(labels, probability)) if np.unique(labels).size == 2 else None,
        "log_loss": float(log_loss(labels, probability, labels=[0, 1])),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-root", default="artifacts/formal_release_prepared_1seed")
    parser.add_argument("--static-root", default="artifacts/formal_prepared")
    parser.add_argument("--output", default="artifacts/formal_release_companions_1seed")
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--fold", type=int, action="append")
    args = parser.parse_args()

    import torch
    from sklearn.linear_model import LogisticRegression
    from cc_hhgt.v32.baselines import build_safe_hashed_design
    from cc_hhgt.v32.patient_folds import assign_outer_split
    from cc_hhgt.v32.patient_fold_authority import (
        validate_frozen_v32_patient_fold_binding,
        validate_frozen_v32_prepared_fold_binding,
    )
    from prepare_v32_formal import (
        attach_replication_labels,
        load_activity,
        load_expression_cancer,
        sampled_associations,
    )

    prepared_root = ROOT / args.prepared_root
    static_root = ROOT / args.static_root
    output = ROOT / args.output
    validate_frozen_v32_patient_fold_binding(
        static_root / "SAMPLE_PATIENT_FOLD_MAP.tsv",
        static_root / "PATIENT_FOLD_AUTHORITY_RECEIPT.json",
    )
    validate_frozen_v32_prepared_fold_binding(prepared_root)
    output.mkdir(parents=True, exist_ok=True)
    folds = sorted(set(args.fold if args.fold is not None else range(5)))

    candidates = pd.read_parquet(static_root / "FORMAL_CANDIDATE_UNIVERSE.parquet")
    candidates = candidates.sort_values(["cancer_id", "lncrna_id", "pathway_id"], kind="stable").reset_index(drop=True)
    hierarchy = pd.read_parquet(static_root / "PATHWAY_HIERARCHY.parquet")
    scope = pd.read_parquet(static_root / "FORMAL_LNCRNA_SCOPE.parquet")
    fold_manifest = pd.read_csv(
        static_root / "SAMPLE_PATIENT_FOLD_MAP.tsv", sep="\t", dtype=str
    )
    fold_manifest["patient_fold_id"] = pd.to_numeric(fold_manifest["patient_fold_id"], errors="raise").astype(int)
    activity = load_activity()
    cancers = sorted(activity.cancer_id.astype(str).unique())
    expr_by = {cancer: load_expression_cancer(cancer) for cancer in cancers}
    activity_by = {cancer: activity.loc[activity.cancer_id.eq(cancer)].copy() for cancer in cancers}
    support = scope.groupby("lncrna_id", observed=True).detected_cancers.max()

    summaries: list[dict] = []
    keys = ["cancer_id", "lncrna_id", "pathway_id"]
    for fold in folds:
        print(json.dumps({"status": "RIDGE_PREP_START", "fold": fold}), flush=True)
        split_manifest = assign_outer_split(fold_manifest, fold, n_folds=5, validation_offset=1)
        train = sampled_associations(expr_by, activity_by, candidates, split_manifest, "train", hierarchy, scope)
        validation_replication = sampled_associations(expr_by, activity_by, candidates, split_manifest, "validation", hierarchy, scope)
        test_replication = sampled_associations(expr_by, activity_by, candidates, split_manifest, "test", hierarchy, scope)
        frames = {
            "train": train,
            "validation": attach_replication_labels(train, validation_replication),
            "test": attach_replication_labels(train, test_replication),
        }
        for frame in frames.values():
            frame["cross_cancer_support_frequency"] = frame.lncrna_id.map(support).fillna(0).to_numpy(float) / 33.0
            frame["cross_cancer_direction_consistency"] = 0.0
            frame["cross_cancer_i2"] = 0.0
            if not frame[keys].reset_index(drop=True).equals(candidates[keys]):
                raise RuntimeError(f"Fold {fold} candidate order drift")

        designs = {name: build_safe_hashed_design(frame, n_features=4096) for name, frame in frames.items()}
        labels = frames["train"].proxy_label.to_numpy(int)
        weights = np.where(
            frames["train"].label_class.astype(str).eq("weak_positive"),
            0.35,
            np.where(labels == 0, 0.12, 1.0),
        )
        ridge = LogisticRegression(
            penalty="l2",
            solver="liblinear",
            C=0.1,
            max_iter=250,
            random_state=args.seed,
        )
        ridge.fit(designs["train"], labels, sample_weight=weights)
        ridge_probability = {
            name: ridge.predict_proba(design)[:, 1].astype("float32")
            for name, design in designs.items()
        }

        payload = torch.load(prepared_root / f"PATIENT_FOLD_{fold}.pt", map_location="cpu", weights_only=False)
        l1_probability = _probability_from_batches(payload, "test")
        if len(l1_probability) != len(candidates):
            raise RuntimeError(f"Fold {fold} L1 row count mismatch")
        companion = pd.DataFrame(
            {
                "candidate_row_index": np.arange(len(candidates), dtype=np.int32),
                "l1_probability": l1_probability,
                "ridge_probability": ridge_probability["test"],
                "regulatory_evidence_confidence": np.full(len(candidates), 0.5, dtype=np.float32),
                "regulatory_evidence_available": np.zeros(len(candidates), dtype=bool),
            }
        )
        companion.to_parquet(output / f"FOLD_{fold}_COMPANION.parquet", index=False, compression="zstd")
        summary = {
            "fold": fold,
            "rows": int(len(companion)),
            "ridge_contract": {
                "penalty": "l2",
                "solver": "liblinear",
                "C": 0.1,
                "n_hashed_features": 4096,
                "feature_scope": "same_fold_safe_discovery_features_as_l1",
                "test_used_for_tuning": False,
            },
            "l1_test": _metrics(frames["test"].proxy_label.to_numpy(int), l1_probability),
            "ridge_validation": _metrics(frames["validation"].proxy_label.to_numpy(int), ridge_probability["validation"]),
            "ridge_test": _metrics(frames["test"].proxy_label.to_numpy(int), ridge_probability["test"]),
            "regulatory_evidence": "UNAVAILABLE_NEUTRAL_0P5_NOT_USED_FOR_RANKING",
        }
        (output / f"FOLD_{fold}_BASELINE_METRICS.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        summaries.append(summary)
        print(json.dumps({"status": "RIDGE_PREP_DONE", **summary}, sort_keys=True), flush=True)

    (output / "BASELINE_METRICS_SUMMARY.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
