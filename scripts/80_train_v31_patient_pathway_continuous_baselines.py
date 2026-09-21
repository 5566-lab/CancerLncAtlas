#!/usr/bin/env python3
"""Nested patient-level LASSO/Ridge regression for exact pathway activity."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from cc_hhgt.common import file_sha256, load_config, read_table, stable_id, write_table
from cc_hhgt.formal_gate import validate_formal_training_gate
from cc_hhgt.pathway_target import require_exact_pathway_contract
from cc_hhgt.v30_integrity import atomic_write_json
from cc_hhgt.v31_continuous_baselines import fit_nested_continuous_baselines


MODELS = ("lasso", "ridge")
PURPOSE = "SEPARATE_PATIENT_EXACT_PATHWAY_CONTINUOUS_REGRESSION_BASELINE"


def _cancer_matrix(
    input_root: Path,
    canonical: pd.DataFrame,
    cancer: str,
    eligible_lnc: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sample = canonical.loc[
        canonical.cancer_id.astype(str).eq(cancer),
        ["sample_id", "patient_id"],
    ].drop_duplicates()
    if sample.sample_id.astype(str).duplicated().any() or sample.patient_id.astype(
        str
    ).duplicated().any():
        raise RuntimeError(f"Canonical sample map is not one-to-one for {cancer}")
    expression = read_table(
        input_root
        / "parquet"
        / "bulk_lncRNA_expression"
        / f"cancer_id={cancer}"
        / "part-0.parquet"
    )
    expression = expression.loc[
        expression.lncrna_id.astype(str).isin(eligible_lnc)
    ].merge(
        sample,
        on=["sample_id", "patient_id"],
        how="inner",
        validate="many_to_one",
    )
    if expression.duplicated(["patient_id", "lncrna_id"]).any():
        raise RuntimeError(f"Duplicate patient-lncRNA expression for {cancer}")
    logcpm = expression.pivot(
        index="patient_id", columns="lncrna_id", values="logcpm"
    ).sort_index(axis=1)
    tpm = expression.pivot(
        index="patient_id", columns="lncrna_id", values="tpm"
    ).reindex(columns=logcpm.columns)

    activity = read_table(
        input_root
        / "parquet"
        / "bulk_pathway_activity"
        / f"cancer_id={cancer}"
        / "part-0.parquet"
    ).merge(
        sample,
        on=["sample_id", "patient_id"],
        how="inner",
        validate="many_to_one",
    )
    if "quality_flag" in activity:
        activity = activity.loc[activity.quality_flag.astype(str).str.lower().eq("pass")]
    if activity.duplicated(["patient_id", "pathway_id"]).any():
        raise RuntimeError(f"Duplicate patient-pathway activity for {cancer}")
    pathway = activity.pivot(
        index="patient_id", columns="pathway_id", values="activity_score"
    ).sort_index(axis=1)
    return logcpm, tpm, pathway


def _coefficient_genesets(
    coefficients: pd.DataFrame,
    metrics: pd.DataFrame,
    dim_lnc: pd.DataFrame,
    dim_pathway: pd.DataFrame,
    analysis_version: str,
    min_members: int,
    max_members: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lasso_metrics = metrics.loc[metrics.model.eq("lasso")]
    denominator = (
        lasso_metrics.groupby(["cancer_id", "pathway_id"], observed=True)
        .patient_fold_id.nunique()
        .rename("available_folds")
        .reset_index()
    )
    if coefficients.empty:
        return pd.DataFrame(), pd.DataFrame()
    grouped = (
        coefficients.groupby(
            ["cancer_id", "pathway_id", "lncrna_id"], observed=True
        )
        .agg(
            selected_folds=("patient_fold_id", "nunique"),
            mean_standardized_coefficient=("standardized_coefficient", "mean"),
            positive_folds=("standardized_coefficient", lambda x: int((x > 0).sum())),
            negative_folds=("standardized_coefficient", lambda x: int((x < 0).sum())),
        )
        .reset_index()
        .merge(denominator, on=["cancer_id", "pathway_id"], validate="many_to_one")
    )
    grouped["selection_stability"] = grouped.selected_folds / grouped.available_folds
    grouped["sign_consistency"] = grouped[["positive_folds", "negative_folds"]].max(
        axis=1
    ) / grouped.selected_folds
    grouped["direction"] = np.where(
        grouped.mean_standardized_coefficient.ge(0), "positive", "negative"
    )
    grouped["absolute_mean_coefficient"] = grouped.mean_standardized_coefficient.abs()
    selected = grouped.loc[
        grouped.selection_stability.ge(0.60) & grouped.sign_consistency.ge(0.80)
    ].copy()
    selected = selected.sort_values(
        [
            "cancer_id",
            "pathway_id",
            "direction",
            "selection_stability",
            "absolute_mean_coefficient",
            "lncrna_id",
        ],
        ascending=[True, True, True, False, False, True],
        kind="stable",
    )
    selected = selected.groupby(
        ["cancer_id", "pathway_id", "direction"], observed=True, sort=False
    ).head(max_members)
    selected = selected.merge(
        dim_lnc, on="lncrna_id", how="left", validate="many_to_one"
    ).merge(dim_pathway, on="pathway_id", how="left", validate="many_to_one")
    master_rows: list[dict] = []
    member_rows: list[pd.DataFrame] = []
    for (cancer, pathway_id, direction), group in selected.groupby(
        ["cancer_id", "pathway_id", "direction"], observed=True, sort=True
    ):
        if len(group) < min_members:
            continue
        geneset_id = stable_id(
            "LGSREG", cancer, pathway_id, direction, analysis_version
        )
        master_rows.append(
            {
                "geneset_id": geneset_id,
                "geneset_name": f"{cancer}__{pathway_id}__{direction.upper()}__LASSO_REGRESSION",
                "geneset_type": "patient_exact_pathway_lasso_coefficient",
                "cancer_id": cancer,
                "pathway_id": pathway_id,
                "pathway_name": group.pathway_name.iloc[0]
                if "pathway_name" in group
                else pathway_id,
                "direction": direction,
                "member_count": int(len(group)),
                "analysis_version": analysis_version,
                "estimand": "patient_exact_pathway_continuous_activity",
            }
        )
        ranked = group.copy()
        ranked["geneset_id"] = geneset_id
        ranked["rank"] = np.arange(1, len(ranked) + 1)
        member_rows.append(ranked)
    return (
        pd.DataFrame(master_rows),
        pd.concat(member_rows, ignore_index=True)
        if member_rows
        else pd.DataFrame(),
    )


def _write_gmt(path: Path, master: pd.DataFrame, member: pd.DataFrame) -> None:
    lines: list[str] = []
    if not master.empty and not member.empty:
        names = master.set_index("geneset_id").geneset_name.to_dict()
        for geneset_id, group in member.groupby("geneset_id", observed=True, sort=True):
            genes = group.sort_values("rank", kind="stable").gene_symbol.fillna("").astype(str)
            fallback = group.sort_values("rank", kind="stable").lncrna_id.astype(str)
            values = [symbol if symbol else lnc for symbol, lnc in zip(genes, fallback)]
            lines.append(
                "\t".join(
                    [names[geneset_id], "CancerLncAtlas nested LASSO regression", *values]
                )
            )
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--formal-gate", required=True)
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--cancers", help="Optional comma-separated pilot subset")
    parser.add_argument("--lasso-alpha-count", type=int, default=30)
    parser.add_argument("--ridge-alpha-count", type=int, default=25)
    args = parser.parse_args()
    if args.lasso_alpha_count < 3 or args.ridge_alpha_count < 3:
        raise ValueError("Continuous baseline alpha grids require at least three values")

    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise RuntimeError(f"Continuous baseline refuses reuse: {output_root}")
    gate_path = Path(args.formal_gate).resolve()
    gate_document = json.loads(gate_path.read_text(encoding="utf-8"))
    training_root = Path(args.training_root).resolve()
    cfg = load_config(
        args.config,
        project_root_override=Path(gate_document["paths"]["run_root"]).resolve(),
        create_dirs=False,
    )
    require_exact_pathway_contract(cfg)
    gate, gate_sha256 = validate_formal_training_gate(
        gate_path,
        cfg,
        verify_assets=True,
        run_id=args.run_id,
        input_root=gate_document["paths"]["input_root"],
        input_manifest=gate_document["paths"]["input_manifest"],
        output_root=training_root,
    )
    run_root = Path(gate["paths"]["run_root"]).resolve()
    input_root = Path(gate["paths"]["input_root"]).resolve()
    tables = Path(gate["paths"]["asset_results"]).resolve() / "tables"
    canonical = read_table(run_root / "canonical" / "canonical_samples.parquet")
    patient_folds = read_table(
        run_root / "canonical" / "canonical_patient_fold_manifest.parquet"
    )
    eligible = read_table(tables / "pancancer_lncrna_eligibility.parquet")
    eligible_lnc = set(eligible.loc[eligible.eligible.astype(bool), "lncrna_id"].astype(str))
    all_cancers = sorted(canonical.cancer_id.astype(str).unique())
    cancers = (
        [item.strip() for item in args.cancers.split(",") if item.strip()]
        if args.cancers
        else all_cancers
    )
    if not cancers or not set(cancers).issubset(all_cancers):
        raise RuntimeError("Continuous baseline requested unknown cancers")

    temporary = output_root.parent / f".{output_root.name}.{os.getpid()}.tmp"
    temporary.mkdir(parents=True)
    metric_rows: list[dict] = []
    prediction_rows: list[pd.DataFrame] = []
    coefficient_rows: list[pd.DataFrame] = []
    availability_rows: list[dict] = []
    detection_min = float(
        cfg["candidate_universe"]["pancancer_lnc_filter"]
        ["within_cancer_min_sample_detection_rate"]
    )
    lasso_ratios = np.logspace(0, -3, args.lasso_alpha_count)
    ridge_alphas = np.logspace(4, -4, args.ridge_alpha_count)

    for cancer in cancers:
        logcpm, tpm, pathway = _cancer_matrix(
            input_root, canonical, cancer, eligible_lnc
        )
        folds = patient_folds.loc[patient_folds.cancer_id.astype(str).eq(cancer)]
        for fold_id in sorted(folds.patient_fold_id.astype(str).unique()):
            assignment = folds.loc[
                folds.patient_fold_id.astype(str).eq(fold_id), ["patient_id", "split"]
            ].drop_duplicates()
            if assignment.patient_id.astype(str).duplicated().any():
                raise RuntimeError(f"Conflicting patient assignment: {fold_id}")
            assignment = assignment.set_index("patient_id").split.astype(str)
            patients = logcpm.index.intersection(pathway.index).intersection(
                assignment.index
            )
            split_ids = {
                split: patients[assignment.reindex(patients).eq(split)]
                for split in ("train", "validation", "test")
            }
            if min(map(len, split_ids.values())) < 3:
                availability_rows.append(
                    {
                        "cancer_id": cancer,
                        "patient_fold_id": fold_id,
                        "status": "UNAVAILABLE",
                        "reason": "MINIMUM_SPLIT_PATIENTS",
                    }
                )
                continue
            train_tpm = tpm.reindex(split_ids["train"])
            detection = np.nanmean(train_tpm.to_numpy(float) > 0, axis=0)
            variance = np.nanvar(
                logcpm.reindex(split_ids["train"]).to_numpy(float), axis=0, ddof=1
            )
            keep = (detection >= detection_min) & np.isfinite(variance) & (variance > 1e-8)
            feature_ids = logcpm.columns[keep].astype(str)
            if len(feature_ids) == 0:
                raise RuntimeError(f"No outcome-free lncRNA predictors: {fold_id}")
            x = {
                split: logcpm.reindex(ids)[feature_ids].to_numpy(float)
                for split, ids in split_ids.items()
            }
            for pathway_id in pathway.columns.astype(str):
                y = {
                    split: pathway.reindex(ids)[pathway_id].to_numpy(float)
                    for split, ids in split_ids.items()
                }
                if not all(np.isfinite(value).all() for value in y.values()):
                    availability_rows.append(
                        {
                            "cancer_id": cancer,
                            "patient_fold_id": fold_id,
                            "pathway_id": pathway_id,
                            "status": "UNAVAILABLE",
                            "reason": "NONFINITE_EXACT_PATHWAY_ACTIVITY",
                        }
                    )
                    continue
                fitted = fit_nested_continuous_baselines(
                    x["train"],
                    y["train"],
                    x["validation"],
                    y["validation"],
                    x["test"],
                    y["test"],
                    lasso_alpha_ratios=lasso_ratios,
                    ridge_alphas=ridge_alphas,
                )
                for model_name in MODELS:
                    result = fitted[model_name]
                    metric_rows.append(
                        {
                            "cancer_id": cancer,
                            "patient_fold_id": fold_id,
                            "pathway_id": pathway_id,
                            "model": model_name,
                            "alpha": result["alpha"],
                            "validation_mse": result["validation_mse"],
                            "test_r2": result["test_r2"],
                            "test_rmse": result["test_rmse"],
                            "test_spearman": result["test_spearman"],
                            "n_nonzero": result["n_nonzero"],
                            **fitted["audit"],
                        }
                    )
                    prediction_rows.append(
                        pd.DataFrame(
                            {
                                "cancer_id": cancer,
                                "patient_fold_id": fold_id,
                                "pathway_id": pathway_id,
                                "model": model_name,
                                "patient_id": split_ids["test"].astype(str),
                                "observed_activity": y["test"],
                                "predicted_activity": result["prediction"],
                                "split": "outer_test",
                            }
                        )
                    )
                lasso_coef = np.asarray(fitted["lasso"]["coefficients"])
                nonzero = np.flatnonzero(np.abs(lasso_coef) > 1e-12)
                if len(nonzero):
                    coefficient_rows.append(
                        pd.DataFrame(
                            {
                                "cancer_id": cancer,
                                "patient_fold_id": fold_id,
                                "pathway_id": pathway_id,
                                "lncrna_id": feature_ids[nonzero],
                                "standardized_coefficient": lasso_coef[nonzero],
                                "selected_by_lasso": True,
                            }
                        )
                    )
                availability_rows.append(
                    {
                        "cancer_id": cancer,
                        "patient_fold_id": fold_id,
                        "pathway_id": pathway_id,
                        "status": "AVAILABLE",
                        "reason": None,
                    }
                )

    metrics = pd.DataFrame(metric_rows)
    predictions = (
        pd.concat(prediction_rows, ignore_index=True)
        if prediction_rows
        else pd.DataFrame()
    )
    coefficients = (
        pd.concat(coefficient_rows, ignore_index=True)
        if coefficient_rows
        else pd.DataFrame()
    )
    availability = pd.DataFrame(availability_rows)
    dim_lnc = read_table(run_root / "assets" / "standardized" / "dim_lncRNA.parquet")[
        ["lncrna_id", "ensembl_gene_id", "gene_symbol", "gene_name"]
    ].drop_duplicates("lncrna_id")
    dim_pathway = read_table(
        run_root / "assets" / "standardized" / "dim_pathway.parquet"
    )[["pathway_id", "pathway_name", "pathway_source", "pathway_collection"]].drop_duplicates(
        "pathway_id"
    )
    master, member = _coefficient_genesets(
        coefficients,
        metrics,
        dim_lnc,
        dim_pathway,
        cfg["analysis_version"],
        int(cfg["materialization"]["min_members"]),
        int(cfg["materialization"]["max_members"]),
    )
    write_table(metrics, temporary / "patient_exact_pathway_regression_metrics.tsv")
    write_table(predictions, temporary / "patient_exact_pathway_outer_test.parquet")
    write_table(coefficients, temporary / "lasso_nonzero_coefficients.parquet")
    write_table(availability, temporary / "availability.tsv")
    write_table(master, temporary / "lasso_regression_geneset_master.parquet")
    write_table(member, temporary / "lasso_regression_geneset_member.parquet")
    _write_gmt(temporary / "lasso_regression_exact_pathway_lncRNA.gmt", master, member)
    if not metrics.empty and bool(metrics.outer_test_used_for_tuning.any()):
        raise RuntimeError("Outer-test patients entered continuous baseline tuning")
    payload = {
        "status": "PASS",
        "purpose": PURPOSE,
        "run_id": args.run_id,
        "formal_gate_sha256": gate_sha256,
        "cancers": len(cancers),
        "patient_pathway_model_folds": int(len(metrics)),
        "outer_test_prediction_rows": int(len(predictions)),
        "lasso_nonzero_coefficient_rows": int(len(coefficients)),
        "lasso_regression_genesets": int(len(master)),
        "estimand": "patient_exact_pathway_continuous_activity",
        "may_enter_primary_exact_pathway_classification_ranking": False,
        "hyperparameter_selection_split": "validation_only",
        "outer_test_used_for_tuning": False,
        "historical_elastic_net_alpha_0_5_is_not_this_baseline": True,
        "script_sha256": file_sha256(Path(__file__).resolve()),
    }
    atomic_write_json(temporary / "SUCCESS.json", payload)
    os.replace(temporary, output_root)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
