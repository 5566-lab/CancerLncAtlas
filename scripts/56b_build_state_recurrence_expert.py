#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from cc_hhgt.common import load_config, read_table, write_json, write_table
from cc_hhgt.state_recurrence import (
    add_fold_local_recurrence,
    add_split_ranks,
    binary_ranking_metrics,
    select_recurrence_weight,
)


MODEL_NAMES = ("rgcn", "hgt", "cc_hhgt_strict")
STEMNESS_STATES = ("stemness_rna::RNAss", "stemness_dna::DNAss")
KEYS = ["candidate_id", "cancer_id", "lncrna_id", "state_id", "proxy_label", "split"]


def load_candidates(root: Path) -> pd.DataFrame:
    files = sorted(root.glob("cancer_id=*/*.parquet"))
    if not files:
        return read_table(root)
    columns = ["candidate_id", "cancer_id", "lncrna_id", "state_id", "proxy_label"]
    return pd.concat([pd.read_parquet(path, columns=columns) for path in files], ignore_index=True)


def collect_complete_fold(strict_root: Path, fold_id: str, expected_seeds: int) -> pd.DataFrame | None:
    paths = sorted(strict_root.glob(f"*/{fold_id}/seed_*/prediction_state_raw.parquet"))
    if len(paths) != len(MODEL_NAMES) * expected_seeds:
        return None
    raw = pd.concat(
        [
            pd.read_parquet(
                path,
                columns=[*KEYS, "model_name", "seed", "raw_probability"],
            )
            for path in paths
        ],
        ignore_index=True,
    )
    raw = raw.loc[raw.state_id.astype(str).isin(STEMNESS_STATES) & raw.split.astype(str).isin(["val", "test"])]
    duplicate = int(raw.duplicated([*KEYS, "model_name", "seed"]).sum())
    if duplicate:
        raise RuntimeError(f"{fold_id}: duplicate seed predictions={duplicate}")
    seed_counts = raw.groupby([*KEYS, "model_name"], observed=True).seed.nunique()
    if not seed_counts.eq(expected_seeds).all():
        return None
    mean = (
        raw.groupby([*KEYS, "model_name"], observed=True)
        .agg(model_probability=("raw_probability", "mean"), n_seeds=("seed", "nunique"))
        .reset_index()
    )
    probability = mean.pivot(index=KEYS, columns="model_name", values="model_probability").reset_index()
    seed_count = mean.pivot(index=KEYS, columns="model_name", values="n_seeds").reset_index()
    missing_models = sorted(set(MODEL_NAMES) - set(probability.columns))
    if missing_models:
        raise RuntimeError(f"{fold_id}: missing models={missing_models}")
    probability["graph_equal_probability"] = probability[list(MODEL_NAMES)].mean(axis=1, skipna=False)
    for model in MODEL_NAMES:
        probability[f"{model}_probability"] = probability.pop(model)
        probability[f"{model}_n_seeds"] = seed_count[model].astype(int)
    return probability


def evaluate_methods(frame: pd.DataFrame, fold_id: str, selected: dict[str, float]) -> pd.DataFrame:
    rows: list[dict] = []
    methods = {
        "graph_equal": "graph_equal_probability",
        "recurrence": "recurrence_probability",
        "validation_selected_fusion": "recurrence_graph_fusion_score",
    }
    for (split, state_id), group in frame.groupby(["split", "state_id"], observed=True):
        for method, column in methods.items():
            rows.append(
                {
                    "fold_id": fold_id,
                    "cancer_id": str(group.cancer_id.iloc[0]),
                    "split": str(split),
                    "state_id": str(state_id),
                    "method": method,
                    "selected_recurrence_weight": selected[str(state_id)],
                    **binary_ranking_metrics(group.proxy_label, group[column]),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a leakage-controlled cross-cancer state recurrence expert")
    parser.add_argument("--config", default="config/model_v2_9_state_graph_local_run.yaml")
    parser.add_argument("--strict-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-folds", type=int, default=33)
    parser.add_argument("--expected-seeds", type=int, default=3)
    parser.add_argument("--prior-strength", type=float, default=2.0)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    strict_root = Path(args.strict_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    table_root = Path(cfg["_results"]) / "tables"
    candidates = load_candidates(table_root / "strict_state_candidate")
    candidates = candidates.loc[candidates.state_id.astype(str).isin(STEMNESS_STATES)].copy()
    folds = read_table(table_root / "fold_manifest.tsv")

    predictions: list[pd.DataFrame] = []
    metrics: list[pd.DataFrame] = []
    selection_rows: list[pd.DataFrame] = []
    incomplete: list[str] = []
    for fold in folds.itertuples(index=False):
        fold_id = str(fold.fold_id)
        graph = collect_complete_fold(strict_root, fold_id, args.expected_seeds)
        if graph is None:
            incomplete.append(fold_id)
            continue
        train_cancers = set(str(fold.train_cancers).split(";"))
        train = candidates.loc[candidates.cancer_id.astype(str).isin(train_cancers)]
        enriched = add_fold_local_recurrence(train, graph, prior_strength=args.prior_strength)
        enriched = add_split_ranks(enriched)
        selected: dict[str, float] = {}
        enriched["selected_recurrence_weight"] = float("nan")
        enriched["recurrence_graph_fusion_score"] = float("nan")
        for state_id, state_rows in enriched.groupby("state_id", observed=True):
            validation = state_rows.loc[state_rows.split.astype(str).eq("val")]
            weight, weight_metrics = select_recurrence_weight(validation)
            selected[str(state_id)] = weight
            weight_metrics.insert(0, "state_id", str(state_id))
            weight_metrics.insert(0, "fold_id", fold_id)
            selection_rows.append(weight_metrics)
            mask = enriched.state_id.astype(str).eq(str(state_id))
            enriched.loc[mask, "selected_recurrence_weight"] = weight
            enriched.loc[mask, "recurrence_graph_fusion_score"] = (
                weight * enriched.loc[mask, "recurrence_rank"]
                + (1.0 - weight) * enriched.loc[mask, "graph_equal_rank"]
            )
        enriched["fold_id"] = fold_id
        enriched["validation_cancer"] = str(fold.validation_cancer)
        enriched["test_cancer"] = str(fold.test_cancer)
        enriched["target_cancer_labels_used_for_history_or_weight_selection"] = 0
        predictions.append(enriched)
        metrics.append(evaluate_methods(enriched, fold_id, selected))

    if not predictions:
        raise RuntimeError("No complete strict folds are available")
    prediction = pd.concat(predictions, ignore_index=True)
    metric = pd.concat(metrics, ignore_index=True)
    selection = pd.concat(selection_rows, ignore_index=True)
    write_table(prediction, output_dir / "state_recurrence_expert_prediction.parquet")
    metric.to_csv(output_dir / "state_recurrence_expert_metrics.tsv", sep="\t", index=False)
    selection.to_csv(output_dir / "state_recurrence_validation_weight_grid.tsv", sep="\t", index=False)

    complete_folds = int(prediction.fold_id.nunique())
    duplicate_keys = int(prediction.duplicated(["fold_id", *KEYS]).sum())
    expected_seed_failures = {
        column: int(prediction[column].ne(args.expected_seeds).sum())
        for column in prediction.columns
        if column.endswith("_n_seeds")
    }
    full_expected = complete_folds == args.expected_folds and not incomplete
    passed = duplicate_keys == 0 and not any(expected_seed_failures.values()) and (full_expected or args.allow_partial)
    audit = {
        "status": "PASS" if passed else "FAIL",
        "scope": "PARTIAL" if incomplete else "COMPLETE",
        "strict_root": str(strict_root),
        "complete_folds": complete_folds,
        "expected_folds": args.expected_folds,
        "incomplete_folds": incomplete,
        "prediction_rows": int(len(prediction)),
        "duplicate_keys": duplicate_keys,
        "expected_seed_failures": expected_seed_failures,
        "prior_strength": args.prior_strength,
        "weight_grid": [0.0, 0.25, 0.5, 0.75, 1.0],
        "history_scope": "fold train cancers only",
        "weight_selection_scope": "fold validation cancer only",
        "target_cancer_labels_used_for_history_or_weight_selection": 0,
        "states": list(STEMNESS_STATES),
    }
    write_json(audit, output_dir / "state_recurrence_expert_audit.json")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

