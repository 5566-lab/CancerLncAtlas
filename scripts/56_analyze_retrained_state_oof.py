#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from cc_hhgt.common import read_table, write_json, write_table


BASE_PROBABILITIES = {
    "rgcn": "rgcn_probability",
    "hgt": "hgt_probability",
    "cc_hhgt_strict": "cc_hhgt_strict_probability",
}
ENSEMBLE_COMPONENTS = {
    "rgcn_hgt_equal": ["rgcn_probability", "hgt_probability"],
    "three_model_equal": list(BASE_PROBABILITIES.values()),
}


def calibration_error(labels: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_id = np.clip(np.digitize(probability, edges[1:-1], right=False), 0, bins - 1)
    error = 0.0
    for index in range(bins):
        mask = bin_id == index
        if mask.any():
            error += float(mask.mean()) * abs(float(labels[mask].mean()) - float(probability[mask].mean()))
    return float(error)


def binary_metrics(labels: np.ndarray, probability: np.ndarray) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=float)
    probability = np.asarray(probability, dtype=float)
    valid = np.isfinite(labels) & np.isfinite(probability) & np.isin(labels, [0.0, 1.0])
    labels = labels[valid]
    probability = probability[valid]
    result: dict[str, float | int] = {
        "n": int(len(labels)),
        "n_positive": int(labels.sum()),
        "positive_rate": float(labels.mean()) if len(labels) else float("nan"),
        "auroc": float("nan"),
        "auprc": float("nan"),
        "brier": float("nan"),
        "ece_10bin": float("nan"),
    }
    if len(labels) and np.unique(labels).size == 2:
        result["auroc"] = float(roc_auc_score(labels, probability))
        result["auprc"] = float(average_precision_score(labels, probability))
    if len(labels):
        clipped = np.clip(probability, 0.0, 1.0)
        result["brier"] = float(brier_score_loss(labels, clipped))
        result["ece_10bin"] = calibration_error(labels, clipped)
    return result


def add_prespecified_ensembles(frame: pd.DataFrame) -> dict[str, str]:
    model_columns = dict(BASE_PROBABILITIES)
    for model_name, components in ENSEMBLE_COMPONENTS.items():
        missing = sorted(set(components) - set(frame.columns))
        if missing:
            raise RuntimeError(f"Cannot build {model_name}; missing columns: {missing}")
        output_column = f"{model_name}_probability"
        # Complete-case averaging avoids silently changing the ensemble model per row.
        frame[output_column] = frame[components].mean(axis=1, skipna=False)
        model_columns[model_name] = output_column
    return model_columns


def evaluate(frame: pd.DataFrame, model_columns: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    pooled_rows: list[dict] = []
    cancer_rows: list[dict] = []
    scopes = [("all_states", "__ALL__", frame)]
    scopes.extend(("state", str(state_id), group) for state_id, group in frame.groupby("state_id", observed=True))
    for scope, state_id, group in scopes:
        labels = group.proxy_label.to_numpy(float)
        for model_name, column in model_columns.items():
            pooled_rows.append(
                {
                    "scope": scope,
                    "state_id": state_id,
                    "model_name": model_name,
                    **binary_metrics(labels, group[column].to_numpy(float)),
                }
            )
    for (cancer_id, state_id), group in frame.groupby(["cancer_id", "state_id"], observed=True):
        labels = group.proxy_label.to_numpy(float)
        for model_name, column in model_columns.items():
            cancer_rows.append(
                {
                    "cancer_id": str(cancer_id),
                    "state_id": str(state_id),
                    "model_name": model_name,
                    **binary_metrics(labels, group[column].to_numpy(float)),
                }
            )
    return pd.DataFrame(pooled_rows), pd.DataFrame(cancer_rows)


def summarize(pooled: pd.DataFrame, by_cancer: pd.DataFrame) -> pd.DataFrame:
    pooled_state = pooled.loc[pooled.scope.eq("state")].rename(
        columns={
            "n": "pooled_n",
            "n_positive": "pooled_n_positive",
            "positive_rate": "pooled_positive_rate",
            "auroc": "pooled_auroc",
            "auprc": "pooled_auprc",
            "brier": "pooled_brier",
            "ece_10bin": "pooled_ece_10bin",
        }
    )
    pooled_state = pooled_state.drop(columns=["scope"])
    cancer_summary = (
        by_cancer.groupby(["model_name", "state_id"], observed=True)
        .agg(
            cancer_runs=("auroc", "size"),
            valid_cancer_runs=("auroc", "count"),
            macro_cancer_auroc=("auroc", "mean"),
            median_cancer_auroc=("auroc", "median"),
            std_cancer_auroc=("auroc", "std"),
            macro_cancer_auprc=("auprc", "mean"),
            median_cancer_auprc=("auprc", "median"),
        )
        .reset_index()
    )
    threshold = (
        by_cancer.assign(ge_060=by_cancer.auroc.ge(0.60).where(by_cancer.auroc.notna()))
        .groupby(["model_name", "state_id"], observed=True)
        .ge_060.mean()
        .rename("fraction_cancers_auroc_ge_060")
        .reset_index()
    )
    return pooled_state.merge(cancer_summary, on=["model_name", "state_id"], how="left").merge(
        threshold, on=["model_name", "state_id"], how="left"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Quantify isolated fixed-sampler strict state OOF predictions")
    parser.add_argument("--state-oof", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-cancers", type=int, default=33)
    parser.add_argument("--expected-states", type=int, default=7)
    args = parser.parse_args()

    state_oof = Path(args.state_oof).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = read_table(state_oof)
    required = {"candidate_id", "cancer_id", "lncrna_id", "state_id", "proxy_label", *BASE_PROBABILITIES.values()}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"State OOF file is missing required columns: {missing}")
    frame = frame.copy()
    model_columns = add_prespecified_ensembles(frame)
    pooled, by_cancer = evaluate(frame, model_columns)
    summary = summarize(pooled, by_cancer)

    write_table(frame, output_dir / "state_oof_prediction_with_prespecified_ensembles.parquet")
    pooled.to_csv(output_dir / "state_oof_pooled_metrics.tsv", sep="\t", index=False)
    by_cancer.to_csv(output_dir / "state_oof_cancer_state_metrics.tsv", sep="\t", index=False)
    summary.to_csv(output_dir / "state_oof_state_summary.tsv", sep="\t", index=False)

    cancers = int(frame.cancer_id.nunique())
    states = int(frame.state_id.nunique())
    duplicate_keys = int(frame.duplicated(["candidate_id", "cancer_id", "lncrna_id", "state_id"]).sum())
    missing_probability = {model: int(frame[column].isna().sum()) for model, column in model_columns.items()}
    seed_columns = [column for column in frame.columns if column.endswith("_n_seeds")]
    incomplete_seed_rows = {
        column: int(frame[column].fillna(0).lt(3).sum()) for column in seed_columns
    }
    labels = sorted(frame.proxy_label.dropna().astype(float).unique().tolist())
    passed = (
        cancers == args.expected_cancers
        and states == args.expected_states
        and duplicate_keys == 0
        and labels == [0.0, 1.0]
        and not any(missing_probability.values())
        and not any(incomplete_seed_rows.values())
    )
    audit = {
        "status": "PASS" if passed else "FAIL",
        "state_oof": str(state_oof),
        "rows": int(len(frame)),
        "cancers": cancers,
        "states": states,
        "state_ids": sorted(frame.state_id.astype(str).unique().tolist()),
        "labels": labels,
        "duplicate_keys": duplicate_keys,
        "missing_probability": missing_probability,
        "incomplete_seed_rows": incomplete_seed_rows,
        "prespecified_ensembles": ENSEMBLE_COMPONENTS,
    }
    write_json(audit, output_dir / "state_oof_analysis_audit.json")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
