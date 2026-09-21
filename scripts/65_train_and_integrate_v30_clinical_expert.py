#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math

import joblib
import numpy as np
import pandas as pd
import torch

from cc_hhgt.clinical_v30 import ClinicalReplicationMLP, clean_na_columns, clinical_replication_label, geometric_priority, make_preprocessor
from cc_hhgt.common import load_config, seed_everything, write_json, write_table

NUMERIC = ["train_beta", "train_se", "train_p_value", "train_hazard_ratio", "n_train", "events_train"]
CATEGORICAL = ["endpoint", "subject_type"]


def fit_crossfit(records: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list[dict]]:
    settings = cfg["clinical"]
    frame = records.loc[records.status.eq("PASS")].copy()
    frame["replication_label"] = [
        clinical_replication_label(tb, tp, tc, rb, settings)
        for tb, tp, tc, rb in zip(frame.test_beta, frame.test_p_value, frame.test_c_index, frame.train_beta, strict=True)
    ]
    frame = frame.loc[frame.replication_label.notna()].reset_index(drop=True)
    if frame.replication_label.nunique() < 2:
        raise RuntimeError("Clinical replication labels contain fewer than two classes")
    cancers = sorted(frame.cancer_id.astype(str).unique())
    if len(cancers) < 3:
        raise RuntimeError("Nested clinical replication cross-fit requires at least three cancers")
    predictions = []
    metrics = []
    model_root = cfg["_results"] / "models" / "clinical_replication_expert"
    for cancer_index, held_cancer in enumerate(cancers):
        validation_cancer = cancers[(cancer_index + 1) % len(cancers)]
        # Upstream Cox statistics from any other PF of the target cancer can
        # contain the held-out PF's patients. Excluding the entire target
        # cancer from meta-expert fitting removes that indirect overlap.
        tr = frame.loc[~frame.cancer_id.astype(str).isin([held_cancer, validation_cancer])].copy()
        va = frame.loc[frame.cancer_id.astype(str).eq(validation_cancer)].copy()
        test = frame.loc[frame.cancer_id.astype(str).eq(held_cancer)].copy()
        tr, va, test = clean_na_columns(tr), clean_na_columns(va), clean_na_columns(test)
        pre = make_preprocessor(NUMERIC, CATEGORICAL)
        x_train = np.asarray(pre.fit_transform(tr), np.float32)
        x_val = np.asarray(pre.transform(va), np.float32)
        x_test = np.asarray(pre.transform(test), np.float32)
        y_train = tr.replication_label.to_numpy(np.float32)
        y_val = va.replication_label.to_numpy(np.float32)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seed_everything(20260726 + cancer_index)
        model = ClinicalReplicationMLP.build(x_train.shape[1], hidden_dim=96, dropout=0.15).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / max(y_train.sum(), 1.0)], dtype=torch.float32, device=device)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        xt = torch.tensor(x_train, dtype=torch.float32, device=device)
        yt = torch.tensor(y_train, dtype=torch.float32, device=device).unsqueeze(1)
        xv = torch.tensor(x_val, dtype=torch.float32, device=device)
        yv = torch.tensor(y_val, dtype=torch.float32, device=device).unsqueeze(1)
        best = None
        best_val = math.inf
        patience = 0
        for epoch in range(150):
            model.train(); optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xt), yt); loss.backward(); optimizer.step()
            model.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(model(xv), yv).cpu())
            if val_loss < best_val - 1e-5:
                best_val = val_loss; best = copy.deepcopy(model.state_dict()); patience = 0
            else:
                patience += 1
                if patience >= 15:
                    break
        model.load_state_dict(best); model.eval()
        with torch.no_grad():
            prob = torch.sigmoid(model(torch.tensor(x_test, dtype=torch.float32, device=device))).squeeze(1).cpu().numpy()
        out = test[["cancer_id", "patient_fold_id", "endpoint", "subject_type", "subject_id", "modifier_id", "replication_label"]].copy()
        out["clinical_survival_probability"] = prob
        predictions.append(out)
        fold_dir = model_root / f"held_cancer_{held_cancer}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": model.cpu().state_dict(),
            "input_dim": x_train.shape[1],
            "numeric_features": NUMERIC,
            "categorical_features": CATEGORICAL,
            "held_cancer": held_cancer,
            "validation_cancer": validation_cancer,
            "target": "held-out clinical association replication",
            "crossfit_scope": "leave_one_cancer_out_meta_expert_on_patient_fold_oof_statistics",
            "target_cancer_rows_used_to_fit": 0,
        }, fold_dir / "best.pt")
        joblib.dump(pre, fold_dir / "preprocessor.joblib")
        metrics.append({
            "held_cancer": held_cancer,
            "validation_cancer": validation_cancer,
            "n_train": len(tr),
            "n_validation": len(va),
            "n_test": len(test),
            "target_cancer_rows_used_to_fit": 0,
            "positive_rate_test": float(test.replication_label.mean()),
            "mean_probability": float(np.mean(prob)),
            "best_validation_loss": best_val,
        })
    return pd.concat(predictions, ignore_index=True), metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the V3.0 candidate clinical replication expert and integrate translational priority")
    parser.add_argument("--config", default="config/model_v3_0_clinical.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    assoc_root = cfg["_results"] / "clinical_association_release"
    records = pd.read_parquet(assoc_root / "clinical_association_fold_records.parquet")
    oof, metrics = fit_crossfit(records, cfg)
    release = cfg["_results"] / "clinical_expert_release"
    write_table(oof, release / "clinical_replication_oof_prediction.parquet")
    write_table(pd.DataFrame(metrics), release / "clinical_replication_metrics.tsv")

    endpoint_summary = oof.groupby(["cancer_id", "endpoint", "subject_type", "subject_id", "modifier_id"], observed=True, dropna=False).agg(
        clinical_survival_probability=("clinical_survival_probability", "mean"),
        clinical_survival_probability_sd=("clinical_survival_probability", "std"),
        n_clinical_folds=("patient_fold_id", "nunique"),
        n_replication_positive=("replication_label", "sum"),
    ).reset_index()
    write_table(endpoint_summary, release / "clinical_endpoint_relevance.parquet")
    clinical = endpoint_summary.groupby(["cancer_id", "subject_type", "subject_id", "modifier_id"], observed=True, dropna=False).agg(
        clinical_relevance_score=("clinical_survival_probability", "max"),
        mean_clinical_probability=("clinical_survival_probability", "mean"),
        clinical_endpoint_count=("endpoint", "nunique"),
        clinical_best_endpoint=("endpoint", lambda x: x.iloc[0]),
        clinical_fold_coverage=("n_clinical_folds", "max"),
    ).reset_index()
    # Recover the actual best endpoint after aggregation.
    best = endpoint_summary.sort_values("clinical_survival_probability", ascending=False).drop_duplicates(["cancer_id", "subject_type", "subject_id", "modifier_id"])
    clinical = clinical.drop(columns="clinical_best_endpoint").merge(best[["cancer_id", "subject_type", "subject_id", "modifier_id", "endpoint"]].rename(columns={"endpoint": "clinical_best_endpoint"}), on=["cancer_id", "subject_type", "subject_id", "modifier_id"], how="left")
    write_table(clinical, release / "clinical_relevance_summary.parquet")

    v29 = cfg["_root"] / cfg["clinical"]["v29_result_root"]
    functional = pd.read_parquet(v29 / "v2_9_downstream" / "v2_9_release" / "final_expert_fusion_table.parquet")
    pair_clin = clinical.loc[clinical.subject_type.eq("lncRNA_pathway")].rename(columns={"subject_id": "lncrna_id", "modifier_id": "pathway_family_id"})
    final = functional.merge(pair_clin[["cancer_id", "lncrna_id", "pathway_family_id", "clinical_relevance_score", "mean_clinical_probability", "clinical_endpoint_count", "clinical_best_endpoint", "clinical_fold_coverage"]], on=["cancer_id", "lncrna_id", "pathway_family_id"], how="left")
    functional_col = cfg.get("clinical_integration", {}).get("functional_probability_column", "fused_confidence_probability")
    final["translational_priority_score"] = geometric_priority(final[functional_col], final["clinical_relevance_score"])
    final["discovery_probability_preserved"] = True
    final["survival_used_in_discovery"] = False
    final["clinical_model_version"] = "CC-HHGT_v3.0-clinical"
    final["prediction_scope"] = final.get("prediction_scope", "v2.9_functional")
    write_table(final, release / "v3_0_functional_clinical_table.parquet")

    payload = {
        "status": "COMPLETED",
        "oof_rows": int(len(oof)),
        "endpoint_rows": int(len(endpoint_summary)),
        "clinical_summary_rows": int(len(clinical)),
        "integrated_rows": int(len(final)),
        "discovery_probability_modified": False,
        "crossfit_scope": "leave_one_cancer_out_meta_expert_on_patient_fold_oof_statistics",
        "target_cancer_rows_used_to_fit": 0,
    }
    write_json(payload, release / "SUCCESS.json")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
