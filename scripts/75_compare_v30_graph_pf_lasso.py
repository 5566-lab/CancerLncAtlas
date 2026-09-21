#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from cc_hhgt.v30_integrity import atomic_write_json


MODELS = ("rgcn", "hgt", "cc_hhgt")
SEEDS = (20260726, 20261726, 20262726)


def metric(label: np.ndarray, score: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(label) & np.isfinite(score)
    label = label[valid].astype(int)
    score = score[valid]
    if len(label) == 0 or np.unique(label).size < 2:
        return {"auroc": np.nan, "auprc": np.nan, "positive_rate": float(label.mean()) if len(label) else np.nan, "auprc_lift": np.nan}
    auprc = float(average_precision_score(label, score))
    positive_rate = float(label.mean())
    return {"auroc": float(roc_auc_score(label, score)), "auprc": auprc, "positive_rate": positive_rate, "auprc_lift": auprc - positive_rate}


def bootstrap(values: np.ndarray, statistic, seed: int, iterations: int = 4000) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    estimates = np.asarray([statistic(values[rng.integers(0, len(values), len(values))]) for _ in range(iterations)])
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def main() -> int:
    parser = argparse.ArgumentParser(description="Matched PF association-replication comparison: V3 graph/effect vs nested LASSO")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--lasso-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    training_root = Path(args.training_root).resolve()
    lasso_root = Path(args.lasso_root).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise RuntimeError(f"Refusing to reuse matched comparison output: {output_root}")
    lasso_success = json.loads((lasso_root / "SUCCESS.json").read_text(encoding="utf-8"))
    matrix_success = json.loads((training_root / "run_control" / "PARALLEL_SUMMARY.json").read_text(encoding="utf-8"))
    expected_mode = f"FULL_{int(matrix_success.get('n_tasks', 0))}"
    if (
        lasso_success.get("run_id") != args.run_id
        or matrix_success.get("run_id") != args.run_id
        or matrix_success.get("status") != "PASS"
        or matrix_success.get("mode") != expected_mode
        or int(matrix_success.get("n_tasks", 0)) != 297
    ):
        raise RuntimeError("Matched comparison requires the same completed formal run_id and full 297-task matrix")

    associations = pd.read_parquet(lasso_root / "association_replication.parquet")
    graph_parts = []
    for model in MODELS:
        for seed in SEEDS:
            for path in sorted((training_root / model).glob(f"LOCO_*/seed_{seed}/prediction_state_calibrated.parquet")):
                frame = pd.read_parquet(path)
                test = frame.loc[frame.split.astype(str).eq("test"), ["cancer_id", "state_id", "lncrna_id", "proxy_positive_probability"]].copy()
                test["model"] = model
                test["seed"] = seed
                graph_parts.append(test)
    graph = pd.concat(graph_parts, ignore_index=True)
    duplicate = graph.duplicated(["cancer_id", "state_id", "lncrna_id", "model", "seed"]).sum()
    if duplicate:
        raise RuntimeError(f"Duplicate graph test predictions: {duplicate}")
    ensemble = graph.groupby(["cancer_id", "state_id", "lncrna_id"], observed=True).agg(graph_probability=("proxy_positive_probability", "mean"), graph_models=("model", "nunique"), graph_seeds=("seed", "nunique"), graph_predictions=("proxy_positive_probability", "size")).reset_index()
    if not ensemble.graph_predictions.eq(9).all() or not ensemble.graph_models.eq(3).all() or not ensemble.graph_seeds.eq(3).all():
        raise RuntimeError("Graph ensemble is not complete 3 models x 3 seeds per candidate")
    matched = associations.merge(ensemble, on=["cancer_id", "state_id", "lncrna_id"], how="inner", validate="many_to_one")
    if matched.empty:
        raise RuntimeError("No matched graph/LASSO association rows")
    matched["lasso_score"] = matched.absolute_lasso_coefficient.fillna(0.0)
    matched["train_effect_evidence"] = matched.train_effect.abs() * np.clip(-np.log10(matched.train_fdr.clip(lower=1e-300)) / 10.0, 0, 1)
    group_keys = ["cancer_id", "state_id", "patient_fold_id"]
    matched["graph_rank"] = matched.groupby(group_keys, observed=True).graph_probability.rank(pct=True, method="average")
    matched["effect_rank"] = matched.groupby(group_keys, observed=True).train_effect_evidence.rank(pct=True, method="average")
    # Fixed, preregistered blend: no outer-test label is used to fit a meta learner.
    matched["graph_effect_fixed_score"] = 0.5 * matched.graph_rank + 0.5 * matched.effect_rank

    rows = []
    for key, group in matched.groupby(group_keys, observed=True):
        label = group.test_membership_label.to_numpy(float)
        for score_name in ("graph_probability", "train_effect_evidence", "graph_effect_fixed_score", "lasso_score"):
            values = group[score_name].to_numpy(float)
            rows.append({"cancer_id": key[0], "state_id": key[1], "patient_fold_id": key[2], "score_name": score_name, "n_candidates": len(group), "n_positive": int(np.nansum(label)), **metric(label, values)})
    fold_metrics = pd.DataFrame(rows)
    valid_metrics = fold_metrics.dropna(subset=["auroc"]).copy()
    summary_rows = []
    for (state_id, score_name), group in valid_metrics.groupby(["state_id", "score_name"], observed=True):
        values = group.auroc.to_numpy(float)
        low, high = bootstrap(values, np.median, 20260810)
        summary_rows.append({"state_id": state_id, "score_name": score_name, "evaluable_folds": len(group), "cancers": group.cancer_id.nunique(), "macro_median_auroc": float(np.median(values)), "macro_mean_auroc": float(np.mean(values)), "macro_std_auroc": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan, "cluster_bootstrap_median_ci95_lower": low, "cluster_bootstrap_median_ci95_upper": high, "macro_mean_auprc_lift": float(group.auprc_lift.mean())})
    summary = pd.DataFrame(summary_rows)

    eligibility = pd.read_csv(run_root / "assets" / "results" / "tables" / "state_model_eligibility.tsv", sep="\t")
    eligible_counts = eligibility.loc[eligibility.cancer_role.eq("FORMAL") & eligibility.evaluation_eligibility.eq("ELIGIBLE")].groupby("state_id", observed=True).size().mul(5).to_dict()
    primary_rows = summary.loc[summary.score_name.eq("graph_effect_fixed_score")].copy()
    primary_rows["eligible_folds"] = primary_rows.state_id.map(eligible_counts)
    primary_rows["eligible_fold_coverage"] = primary_rows.evaluable_folds / primary_rows.eligible_folds
    primary_rows["state_problem_solved"] = (primary_rows.macro_median_auroc >= 0.60) & (primary_rows.cluster_bootstrap_median_ci95_lower > 0.50) & (primary_rows.eligible_fold_coverage >= 0.80)

    paired = valid_metrics.pivot_table(index=group_keys, columns="score_name", values="auroc").reset_index()
    paired = paired.dropna(subset=["graph_effect_fixed_score", "lasso_score"])
    paired["paired_delta_graph_minus_lasso"] = paired.graph_effect_fixed_score - paired.lasso_score
    advantage_rows = []
    for state_id, group in paired.groupby("state_id", observed=True):
        delta = group.paired_delta_graph_minus_lasso.to_numpy(float)
        low, high = bootstrap(delta, np.mean, 20260811)
        mean_delta = float(np.mean(delta))
        advantage_rows.append({"state_id": state_id, "paired_folds": len(group), "paired_mean_delta_auroc": mean_delta, "paired_median_delta_auroc": float(np.median(delta)), "cluster_bootstrap_mean_delta_ci95_lower": low, "cluster_bootstrap_mean_delta_ci95_upper": high, "performance_advantage_supported": bool(mean_delta >= 0.02 and low > 0)})
    advantage = pd.DataFrame(advantage_rows)

    pooled_rows = []
    for state_id, group in matched.groupby("state_id", observed=True):
        for score_name in ("graph_probability", "train_effect_evidence", "graph_effect_fixed_score", "lasso_score"):
            pooled_rows.append({"state_id": state_id, "score_name": score_name, **metric(group.test_membership_label.to_numpy(float), group[score_name].to_numpy(float))})
    pooled = pd.DataFrame(pooled_rows)
    patient_score = pd.read_csv(lasso_root / "patient_score_fold_metrics.tsv", sep="\t")
    patient_summary = patient_score.groupby("state_id", observed=True).agg(folds=("patient_fold_id", "size"), median_r2=("test_r2", "median"), median_spearman=("test_spearman", "median"), median_rmse=("test_rmse", "median"), median_high_low_auroc=("test_high_low_auroc", "median")).reset_index()

    temporary = output_root.parent / f".{output_root.name}.tmp"; temporary.mkdir(parents=True)
    matched.to_parquet(temporary / "matched_candidate_predictions.parquet", index=False, compression="zstd")
    fold_metrics.to_csv(temporary / "patient_fold_association_metrics.tsv", sep="\t", index=False)
    summary.to_csv(temporary / "macro_state_metrics.tsv", sep="\t", index=False)
    primary_rows.to_csv(temporary / "state_acceptance.tsv", sep="\t", index=False)
    paired.to_csv(temporary / "paired_fold_delta.tsv", sep="\t", index=False)
    advantage.to_csv(temporary / "graph_vs_lasso_advantage.tsv", sep="\t", index=False)
    pooled.to_csv(temporary / "appendix_pooled_metrics.tsv", sep="\t", index=False)
    patient_summary.to_csv(temporary / "lasso_patient_score_summary.tsv", sep="\t", index=False)
    success = {"status": "PASS", "run_id": args.run_id, "matched_rows": len(matched), "evaluation_unit": "cancer x patient_fold x lncRNA x state association replication", "primary_score": "graph_effect_fixed_score", "primary_score_fit": "fixed 0.5 rank blend; no outer-test fitting", "graph_component_policy": "secondary target-unlabeled transductive LOCO prior", "pooled_auroc_role": "APPENDIX_ONLY", "state_acceptance": primary_rows.to_dict("records"), "graph_vs_lasso_advantage": advantage.to_dict("records"), "advantage_claim_rule": "paired mean delta >= 0.02 and clustered-bootstrap 95% CI lower > 0", "patient_state_score_metrics_reported_separately": True}
    atomic_write_json(temporary / "SUCCESS.json", success)
    os.replace(temporary, output_root)
    print(json.dumps(success, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
