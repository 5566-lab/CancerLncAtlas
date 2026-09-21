#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from cc_hhgt.clinical_v30 import (
    ENDPOINTS,
    MultiEndpointSurvivalMLP,
    discrete_time_nll,
    endpoint_time_bins,
    gradient_input_importance,
    harrell_c_index,
    make_preprocessor,
    survival_risk_from_logits,
    time_to_bin,
)
from cc_hhgt.common import load_config, seed_everything, write_json, write_table


def endpoint_arrays(frame: pd.DataFrame, endpoint: str, edges: np.ndarray):
    time = pd.to_numeric(frame.get(f"time_days::{endpoint}"), errors="coerce").to_numpy(float)
    event = pd.to_numeric(frame.get(f"event::{endpoint}"), errors="coerce").to_numpy(float)
    available = np.isfinite(time) & np.isfinite(event)
    bins = np.zeros(len(frame), dtype=np.int64)
    bins[available] = time_to_bin(time[available], edges)
    return time, event, available.astype(np.float32), bins


def train_one(cancer: str, fold: str, seed: int, frame: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    settings = cfg["clinical"]
    seed_everything(seed)
    train = frame.loc[(frame.patient_fold_id.astype(str).eq(fold)) & frame.split.eq("train")].copy()
    val = frame.loc[(frame.patient_fold_id.astype(str).eq(fold)) & frame.split.eq("validation")].copy()
    test = frame.loc[(frame.patient_fold_id.astype(str).eq(fold)) & frame.split.eq("test")].copy()
    if min(len(train), len(val), len(test)) == 0:
        raise RuntimeError(f"{cancer} fold={fold}: empty train/validation/test")

    clinical_num = [c for c in settings.get("clinical_covariates", {}).get("numeric", []) if c in frame]
    clinical_cat = [c for c in settings.get("clinical_covariates", {}).get("categorical", []) if c in frame]
    molecular_all = [c for c in frame.columns if c.startswith("lnc::") or c.startswith("pf::")]
    states = [c for c in cfg.get("state_graph", {}).get("required_states", []) if c in frame]
    # Outcome-independent, train-only feature filtering prevents a 50-patient
    # cancer from receiving hundreds of sparse molecular inputs.
    coverage_min = float(settings.get("min_patient_model_feature_coverage", 0.50))
    max_molecular = int(settings.get("max_patient_model_molecular_features", 256))
    feature_quality = []
    for column in molecular_all:
        values = pd.to_numeric(train[column], errors="coerce")
        coverage = float(values.notna().mean())
        variance = float(values.var()) if values.notna().sum() > 1 else 0.0
        if coverage >= coverage_min and np.isfinite(variance) and variance > 1e-8:
            feature_quality.append((variance, coverage, column))
    feature_quality.sort(reverse=True)
    molecular = [column for _, _, column in feature_quality[:max_molecular]]
    numeric = list(dict.fromkeys([*clinical_num, *molecular, *states]))
    if not numeric:
        raise RuntimeError(f"{cancer}: no numeric clinical/molecular features")
    pre = make_preprocessor(numeric, clinical_cat)
    def _clean(frame):
        frame = frame.copy()
        for column in frame.columns:
            series = frame[column]
            if isinstance(series.dtype, pd.StringDtype) or series.dtype == object:
                frame[column] = series.astype(object).where(series.notna(), np.nan)
        return frame
    x_train = np.asarray(pre.fit_transform(_clean(train)), dtype=np.float32)
    x_val = np.asarray(pre.transform(_clean(val)), dtype=np.float32)
    x_test = np.asarray(pre.transform(_clean(test)), dtype=np.float32)
    feature_names = list(pre.get_feature_names_out())

    endpoints = []
    edges_by_endpoint = {}
    arrays = {"train": {}, "validation": {}, "test": {}}
    for endpoint in ENDPOINTS:
        time_col = f"time_days::{endpoint}"
        event_col = f"event::{endpoint}"
        if time_col not in frame or event_col not in frame:
            continue
        time_train = pd.to_numeric(train[time_col], errors="coerce").to_numpy(float)
        event_train = pd.to_numeric(train[event_col], errors="coerce").to_numpy(float)
        available = np.isfinite(time_train) & np.isfinite(event_train)
        if available.sum() < int(settings.get("min_patients", 50)) or event_train[available].sum() < int(settings.get("min_events", 12)):
            continue
        endpoint_edges = endpoint_time_bins(time_train[available], event_train[available], int(settings.get("n_time_bins", 12)))
        edges_by_endpoint[endpoint] = endpoint_edges
        endpoints.append(endpoint)
        for split_name, part in [("train", train), ("validation", val), ("test", test)]:
            arrays[split_name][endpoint] = endpoint_arrays(part, endpoint, endpoint_edges)
    if not endpoints:
        raise RuntimeError(f"{cancer} fold={fold}: no endpoint meets minimum patient/event requirements")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiEndpointSurvivalMLP.build(
        x_train.shape[1], endpoints, int(settings.get("n_time_bins", 12)),
        int(settings.get("hidden_dim", 192)), float(settings.get("dropout", 0.2)),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings.get("learning_rate", 1e-3)), weight_decay=float(settings.get("weight_decay", 1e-4)))

    train_tensor = torch.tensor(x_train, dtype=torch.float32)
    loader = DataLoader(TensorDataset(train_tensor, torch.arange(len(train_tensor))), batch_size=int(settings.get("batch_size", 128)), shuffle=True)

    def loss_for(split_name: str, x: np.ndarray, index: np.ndarray | None = None):
        tensor = torch.tensor(x if index is None else x[index], dtype=torch.float32, device=device)
        outputs, _ = model(tensor)
        loss = torch.tensor(0.0, device=device)
        used = 0
        for endpoint in endpoints:
            _, event, available, bins = arrays[split_name][endpoint]
            if index is not None:
                event, available, bins = event[index], available[index], bins[index]
            if available.sum() == 0:
                continue
            loss = loss + discrete_time_nll(
                outputs[endpoint],
                torch.tensor(bins, dtype=torch.long, device=device),
                torch.tensor(event, dtype=torch.float32, device=device),
                torch.tensor(available, dtype=torch.float32, device=device),
            )
            used += 1
        return loss / max(used, 1)

    best_state = None
    best_val = math.inf
    patience = 0
    history = []
    for epoch in range(int(settings.get("epochs", 120))):
        model.train()
        train_losses = []
        for (batch_x, batch_idx) in loader:
            batch_x = batch_x.to(device)
            batch_idx_np = batch_idx.numpy()
            optimizer.zero_grad(set_to_none=True)
            outputs, _ = model(batch_x)
            batch_loss = torch.tensor(0.0, device=device)
            used = 0
            for endpoint in endpoints:
                _, event, available, bins = arrays["train"][endpoint]
                if available[batch_idx_np].sum() == 0:
                    continue
                batch_loss = batch_loss + discrete_time_nll(
                    outputs[endpoint],
                    torch.tensor(bins[batch_idx_np], dtype=torch.long, device=device),
                    torch.tensor(event[batch_idx_np], dtype=torch.float32, device=device),
                    torch.tensor(available[batch_idx_np], dtype=torch.float32, device=device),
                )
                used += 1
            if used == 0:
                continue
            batch_loss = batch_loss / used
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_losses.append(float(batch_loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_for("validation", x_val).detach().cpu())
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(train_losses)) if train_losses else math.nan, "validation_loss": val_loss})
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"[{cancer} {fold} {seed}] epoch={epoch+1} train_loss={history[-1]['train_loss']:.4f} val_loss={val_loss:.4f}", flush=True)
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= int(settings.get("patience", 15)):
                break
    if best_state is None:
        raise RuntimeError(f"{cancer} fold={fold} seed={seed}: no valid model state")
    model.load_state_dict(best_state)
    model.eval()

    rows = []
    importance_rows = []
    with torch.no_grad():
        test_outputs, _ = model(torch.tensor(x_test, dtype=torch.float32, device=device))
    for endpoint in endpoints:
        logits = test_outputs[endpoint].detach().cpu().numpy()
        risk = survival_risk_from_logits(logits)
        time, event, available, _ = arrays["test"][endpoint]
        cindex = harrell_c_index(time[available.astype(bool)], event[available.astype(bool)], risk[available.astype(bool)])
        for idx, row in test.reset_index(drop=True).iterrows():
            rows.append({
                "cancer_id": cancer,
                "patient_fold_id": fold,
                "seed": seed,
                "patient_id": row.patient_id,
                "endpoint": endpoint,
                "time_days": time[idx],
                "event": event[idx],
                "endpoint_available": int(available[idx]),
                "risk_score": float(risk[idx]),
                "fold_c_index": cindex,
            })
        if available.sum() >= 4:
            importance = gradient_input_importance(model.cpu(), x_test[available.astype(bool)], endpoint)
            model.to(device)
            for name, value in zip(feature_names, importance, strict=True):
                importance_rows.append({
                    "cancer_id": cancer,
                    "patient_fold_id": fold,
                    "seed": seed,
                    "endpoint": endpoint,
                    "feature_name": name,
                    "importance": float(value),
                })

    model_root = cfg["_results"] / "models" / "clinical_survival" / cancer / fold / f"seed_{seed}"
    model_root.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.cpu().state_dict(),
        "input_dim": x_train.shape[1],
        "endpoints": endpoints,
        "n_bins": int(settings.get("n_time_bins", 12)),
        "hidden_dim": int(settings.get("hidden_dim", 192)),
        "dropout": float(settings.get("dropout", 0.2)),
        "time_bin_edges": {k: v.tolist() for k, v in edges_by_endpoint.items()},
        "feature_names": feature_names,
        "clinical_covariates": {"numeric": clinical_num, "categorical": clinical_cat},
        "molecular_features": molecular,
        "state_features": states,
        "preprocessing_fit_scope": "train_patients_only",
        "survival_endpoints_used_in_discovery": False,
    }, model_root / "best.pt")
    joblib.dump(pre, model_root / "preprocessor.joblib")
    pd.DataFrame(history).to_csv(model_root / "training_history.tsv", sep="\t", index=False)
    meta = {
        "status": "COMPLETED",
        "cancer_id": cancer,
        "patient_fold_id": fold,
        "seed": seed,
        "endpoints": endpoints,
        "n_train": len(train),
        "n_validation": len(val),
        "n_test": len(test),
        "best_validation_loss": best_val,
        "device": str(device),
    }
    write_json(meta, model_root / "SUCCESS.json")
    return pd.DataFrame(rows), pd.DataFrame(importance_rows), meta


def main() -> int:
    parser = argparse.ArgumentParser(description="Train V3.0 multi-endpoint patient survival expert")
    parser.add_argument("--config", default="config/model_v3_0_clinical.yaml")
    parser.add_argument("--cancer", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    patient_root = cfg["_results"] / "clinical_data" / "patients"
    outputs = []
    importance = []
    task_rows = []
    for path in sorted(patient_root.rglob("part-0.parquet")):
        cancer = path.parent.name.split("=", 1)[-1]
        if args.cancer and cancer != args.cancer:
            continue
        frame = pd.read_parquet(path)
        folds = sorted(frame.patient_fold_id.astype(str).unique())
        for fold in folds:
            for seed in cfg["clinical"].get("seeds", [20260726]):
                success = cfg["_results"] / "models" / "clinical_survival" / cancer / fold / f"seed_{seed}" / "SUCCESS.json"
                pred_file = cfg["_results"] / "clinical_survival_run" / cancer / fold / f"seed_{seed}" / "prediction.parquet"
                imp_file = cfg["_results"] / "clinical_survival_run" / cancer / fold / f"seed_{seed}" / "feature_importance.parquet"
                if args.resume and success.exists() and pred_file.exists():
                    outputs.append(pd.read_parquet(pred_file))
                    if imp_file.exists():
                        importance.append(pd.read_parquet(imp_file))
                    task_rows.append(json.loads(success.read_text(encoding="utf-8")))
                    continue
                try:
                    pred, imp, meta = train_one(cancer, fold, int(seed), frame, cfg)
                except RuntimeError as exc:
                    print(f"[skip] {cancer} {fold} seed={seed}: {exc}", flush=True)
                    continue
                pred_file.parent.mkdir(parents=True, exist_ok=True)
                write_table(pred, pred_file)
                write_table(imp, imp_file)
                outputs.append(pred)
                importance.append(imp)
                task_rows.append(meta)
    if not outputs:
        raise RuntimeError("No V3.0 survival tasks were completed")
    release = cfg["_results"] / "clinical_survival_release"
    all_pred = pd.concat(outputs, ignore_index=True)
    all_imp = pd.concat(importance, ignore_index=True) if importance else pd.DataFrame()
    write_table(all_pred, release / "patient_survival_oof_prediction.parquet")
    write_table(all_imp, release / "survival_feature_importance.parquet")
    task = pd.DataFrame(task_rows)
    write_table(task, release / "survival_task_manifest.tsv")
    summary = all_pred.loc[all_pred.endpoint_available.eq(1)].groupby(["cancer_id", "endpoint"], observed=True).agg(
        n_patients=("patient_id", "nunique"), n_events=("event", "sum"), c_index=("fold_c_index", "mean"), c_index_sd=("fold_c_index", "std")
    ).reset_index()
    write_table(summary, release / "survival_oof_metrics.tsv")
    payload = {"status": "COMPLETED", "tasks": len(task), "prediction_rows": len(all_pred), "cancers": int(all_pred.cancer_id.nunique()), "endpoints": sorted(all_pred.endpoint.unique())}
    write_json(payload, release / "SUCCESS.json")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
