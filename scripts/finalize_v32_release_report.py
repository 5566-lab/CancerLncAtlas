#!/usr/bin/env python3
"""Create the evidence-backed one-seed release-candidate report and audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def _mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--prediction-pattern", required=True)
    parser.add_argument("--metrics-pattern", required=True)
    parser.add_argument("--materialization-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cost-json")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    metrics = [json.loads(Path(args.metrics_pattern.format(fold=fold)).read_text(encoding="utf-8")) for fold in range(5)]
    success_files = sorted(Path(args.training_root).glob("*/SUCCESS.json"))
    successes = [json.loads(path.read_text(encoding="utf-8")) for path in success_files]
    materialization = json.loads((Path(args.materialization_root) / "MATERIALIZATION_SUCCESS.json").read_text(encoding="utf-8"))

    model_summary: dict[str, dict[str, dict[str, float]]] = {}
    for model in ("l1", "ridge", "cc_hhgt"):
        model_summary[model] = {
            metric: _mean_std([float(item[model][metric]) for item in metrics])
            for metric in ("auprc", "auroc", "log_loss")
        }
    fold_rows = []
    for item in metrics:
        fold_rows.append(
            {
                "fold": item["fold"],
                **{f"{model}_{metric}": item[model][metric] for model in ("l1", "ridge", "cc_hhgt") for metric in ("auprc", "auroc", "log_loss")},
            }
        )
    pd.DataFrame(fold_rows).to_csv(output / "FOLD_METRICS.tsv", sep="\t", index=False)

    family_rows = []
    residual_rows = []
    for fold in range(5):
        frame = pd.read_parquet(
            args.prediction_pattern.format(fold=fold),
            columns=[
                "pathway_family_id", "held_out_proxy_label", "l1_probability",
                "ridge_probability", "association_membership_probability",
                "graph_residual", "graph_gate",
            ],
        )
        residual_rows.append(
            {
                "fold": fold,
                "mean_abs_graph_residual": float(frame.graph_residual.abs().mean()),
                "p95_abs_graph_residual": float(frame.graph_residual.abs().quantile(0.95)),
                "near_l1_fallback_fraction": float(frame.graph_residual.abs().le(1e-6).mean()),
                "mean_graph_gate": float(frame.graph_gate.mean()),
            }
        )
        for family, group in frame.groupby("pathway_family_id", observed=True, sort=True):
            y = group.held_out_proxy_label.to_numpy(int)
            if len(group) < 100 or np.unique(y).size != 2:
                continue
            record = {"fold": fold, "pathway_family_id": str(family), "rows": len(group), "prevalence": float(y.mean())}
            for model, column in (
                ("l1", "l1_probability"),
                ("ridge", "ridge_probability"),
                ("cc_hhgt", "association_membership_probability"),
            ):
                probability = group[column].to_numpy(float)
                record[f"{model}_auprc"] = float(average_precision_score(y, probability))
                record[f"{model}_auroc"] = float(roc_auc_score(y, probability))
            record["delta_auprc_vs_l1"] = record["cc_hhgt_auprc"] - record["l1_auprc"]
            record["delta_auprc_vs_ridge"] = record["cc_hhgt_auprc"] - record["ridge_auprc"]
            family_rows.append(record)
    family_fold = pd.DataFrame(family_rows)
    family_fold.to_csv(output / "PATHWAY_FAMILY_FOLD_METRICS.tsv", sep="\t", index=False)
    family = (
        family_fold.groupby("pathway_family_id", observed=True, sort=True)
        .agg(
            folds=("fold", "nunique"),
            mean_rows=("rows", "mean"),
            mean_prevalence=("prevalence", "mean"),
            mean_l1_auprc=("l1_auprc", "mean"),
            mean_ridge_auprc=("ridge_auprc", "mean"),
            mean_cc_hhgt_auprc=("cc_hhgt_auprc", "mean"),
            mean_delta_auprc_vs_l1=("delta_auprc_vs_l1", "mean"),
            mean_delta_auprc_vs_ridge=("delta_auprc_vs_ridge", "mean"),
            positive_folds_vs_l1=("delta_auprc_vs_l1", lambda x: int((x > 0).sum())),
            positive_folds_vs_ridge=("delta_auprc_vs_ridge", lambda x: int((x > 0).sum())),
        )
        .reset_index()
        .sort_values(["mean_delta_auprc_vs_l1", "pathway_family_id"], ascending=[False, True], kind="stable")
    )
    family.to_csv(output / "PATHWAY_FAMILY_METRICS.tsv", sep="\t", index=False)
    residual = pd.DataFrame(residual_rows)
    residual.to_csv(output / "RESIDUAL_USAGE.tsv", sep="\t", index=False)

    cost = None
    if args.cost_json and Path(args.cost_json).is_file():
        cost = json.loads(Path(args.cost_json).read_text(encoding="utf-8"))
    audit = {
        "status": "PROMISING_1SEED_RELEASE_CANDIDATE",
        "website_replacement_authorized": False,
        "run_id": "V3_2_ONESEED_CC_HHGT_FORMAL_RELEASE",
        "successful_training_folds": len(successes),
        "all_folds_early_stopped_before_cap": len(successes) == 5 and all(item.get("did_not_hit_hard_cap") is True for item in successes),
        "seed": 20260726,
        "model_summary": model_summary,
        "cc_hhgt_auprc_wins_vs_l1": int(sum(item["cc_hhgt"]["auprc"] > item["l1"]["auprc"] for item in metrics)),
        "cc_hhgt_auprc_wins_vs_ridge": int(sum(item["cc_hhgt"]["auprc"] > item["ridge"]["auprc"] for item in metrics)),
        "pathway_family_target_used": False,
        "pair_evidence_in_primary": False,
        "regulatory_evidence_status": "UNAVAILABLE_NEUTRAL_0P5_NOT_USED_FOR_RANKING",
        "subtype_feedback_to_model": False,
        "materialization": materialization,
        "paid_compute": cost,
        "limitations": [
            "Only one random seed; this does not establish seed stability.",
            "Regulatory evidence confidence is not available for the exact-pathway universe in this run and is never used for ranking.",
            "Website replacement requires separate review and authorization.",
        ],
    }
    (output / "FORMAL_RELEASE_AUDIT.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    top = family.head(10)
    lines = [
        "# CancerLncAtlas V3.2 one-seed CC-HHGT formal report",
        "",
        f"Status: `{audit['status']}`. This run is not yet authorized to replace the website model.",
        "",
        "## Held-out five-fold results",
        "",
        "| Model | AUCPR mean ± SD | AUROC mean ± SD | Log loss mean ± SD |",
        "|---|---:|---:|---:|",
    ]
    for model, label in (("l1", "L1 logistic LASSO"), ("ridge", "L2 logistic Ridge"), ("cc_hhgt", "CC-HHGT bounded residual")):
        item = model_summary[model]
        lines.append(f"| {label} | {item['auprc']['mean']:.6f} ± {item['auprc']['std']:.6f} | {item['auroc']['mean']:.6f} ± {item['auroc']['std']:.6f} | {item['log_loss']['mean']:.6f} ± {item['log_loss']['std']:.6f} |")
    lines += [
        "",
        f"CC-HHGT AUCPR wins: {audit['cc_hhgt_auprc_wins_vs_l1']}/5 vs L1 and {audit['cc_hhgt_auprc_wins_vs_ridge']}/5 vs Ridge.",
        "",
        "## Largest pathway-family AUCPR gains versus L1",
        "",
        "| Pathway family | Mean ΔAUCPR | Positive folds |",
        "|---|---:|---:|",
    ]
    for row in top.itertuples(index=False):
        lines.append(f"| {row.pathway_family_id} | {row.mean_delta_auprc_vs_l1:.6f} | {row.positive_folds_vs_l1}/{row.folds} |")
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "The final target is cancer × lncRNA × exact pathway. Pathway family is auxiliary hierarchy and stratification only. Pair-level evidence is excluded from the primary score. This is a one-seed release candidate, not proof of seed stability and not an automatic website replacement.",
    ]
    report_text = "\n".join(lines) + "\n"
    # The historical Windows copy contains three double-decoded CP936 glyph
    # pairs in otherwise UTF-8 report literals.  Normalize them at the output
    # boundary so the published report always uses the intended math symbols.
    for corrupted, intended in {
        "\u00a1\u00c0": "±",
        "\u00a6\u00a4": "Δ",
        "\u00a1\u00c1": "×",
    }.items():
        report_text = report_text.replace(corrupted, intended)
    (output / "MODEL_REPORT.md").write_text(report_text, encoding="utf-8")
    print(json.dumps(audit, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
