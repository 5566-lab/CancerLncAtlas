#!/usr/bin/env python3
"""Materialize V2.9 main lncRNA--pathway and lncRNA--state web tables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cc_hhgt.common import input_path, load_config, read_table, write_json, write_table

KEYS = ["cancer_id", "lncrna_id", "pathway_family_id"]


def _assert_unique_keys(
    frame: pd.DataFrame, keys: list[str], label: str
) -> None:
    """Fail closed on relationship-key collisions; never silently de-duplicate."""

    missing = sorted(set(keys) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing relationship keys: {missing}")
    duplicated = frame.duplicated(keys, keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, keys].head(10).to_dict("records")
        raise ValueError(
            f"{label} contains duplicate relationship keys {keys}: {examples}"
        )


def _classify_relationships(
    frame: pd.DataFrame, materialization: dict[str, object]
) -> pd.Series:
    """Assign mutually exclusive display classes in evidence-first order.

    A high discovery score with little direct evidence is a predicted candidate,
    not a model-supported observation. The historical condition order tested the
    broader confidence condition first and swallowed those candidate rows.
    """

    observed_min = float(materialization.get("observed_core_evidence_min", 0.70))
    supported_min = float(
        materialization.get("model_supported_probability_min", 0.80)
    )
    predicted_min = float(
        materialization.get("predicted_candidate_probability_min", 0.90)
    )
    predicted_evidence_max = float(
        materialization.get("predicted_candidate_max_observed_evidence", 0.35)
    )
    direct = frame["direct_evidence_probability"].combine_first(
        frame["evidence_integrated_probability"]
    )
    discovery = frame["discovery_ranking_probability"]
    confidence = frame["fused_confidence_probability"]
    return pd.Series(
        np.select(
            [
                direct.ge(observed_min),
                discovery.ge(predicted_min)
                & direct.fillna(0).le(predicted_evidence_max),
                confidence.ge(supported_min),
            ],
            ["observed_core", "predicted_candidate", "model_supported"],
            default="exploratory",
        ),
        index=frame.index,
        dtype="string",
    )


def _lncrna_dimension(cfg: dict) -> pd.DataFrame:
    dim_path = cfg["_standardized"] / "dim_lncRNA.parquet"
    if not dim_path.exists():
        configured = input_path(cfg, "dim_lncRNA")
        dim_path = configured if configured is not None else dim_path
    if not dim_path.exists():
        return pd.DataFrame(columns=["lncrna_id"])
    dim = read_table(dim_path)
    keep = [
        c for c in [
            "lncrna_id", "gene_symbol", "gene_name", "ensembl_gene_id",
            "gene_type", "chromosome",
        ] if c in dim
    ]
    return dim[keep].drop_duplicates("lncrna_id")


def _pathway_dimension(cfg: dict) -> pd.DataFrame:
    candidates = [
        cfg["_results"] / "tables" / "pathway_family.parquet",
        cfg["_standardized"] / "pathway_family.parquet",
    ]
    for path in candidates:
        if path.exists():
            frame = read_table(path)
            key = "pathway_family_id"
            if key in frame:
                keep = [
                    c for c in [
                        key, "pathway_family_name", "family_name", "display_name",
                        "representative_pathway_id", "n_pathways",
                    ] if c in frame
                ]
                return frame[keep].drop_duplicates(key)
    return pd.DataFrame(columns=["pathway_family_id"])


def _materialize_main(cfg: dict, web: Path, dim_lnc: pd.DataFrame) -> dict:
    release = cfg["_results"] / "v2_9_downstream" / "v2_9_release"
    final_path = release / "final_expert_fusion_table.parquet"
    if not final_path.exists():
        fallback = release / "final_three_probability_table.parquet"
        if not fallback.exists():
            raise FileNotFoundError(
                f"Neither full expert nor baseline final table exists: {final_path}, {fallback}"
            )
        final_path = fallback
        model_mode = "BASELINE_WITHOUT_FULL_EVENT_FUSION"
    else:
        model_mode = "FULL_EXPERT_V2_9"

    final = read_table(final_path)
    missing = sorted(set(KEYS) - set(final.columns))
    if missing:
        raise ValueError(f"Final prediction table missing keys: {missing}")
    _assert_unique_keys(final, KEYS, "Final prediction table")
    final = final.merge(dim_lnc, on="lncrna_id", how="left")
    family = _pathway_dimension(cfg)
    if not family.empty:
        final = final.merge(family, on="pathway_family_id", how="left")
    _assert_unique_keys(final, KEYS, "Enriched final prediction table")

    for column in [
        "discovery_ranking_probability", "fused_confidence_probability",
        "graph_ensemble_probability", "cross_cancer_probability",
        "cancer_native_probability", "cancer_specific_probability",
        "indirect_mechanism_probability", "direct_evidence_probability",
        "evidence_integrated_probability", "specificity_score",
    ]:
        if column not in final:
            final[column] = np.nan
        final[column] = pd.to_numeric(final[column], errors="coerce")

    # Keep the legacy strict score visible when graph stacking is unavailable for a pair.
    final["graph_probability"] = final["graph_ensemble_probability"].combine_first(
        final["cross_cancer_probability"]
    )
    final["display_probability"] = final["fused_confidence_probability"].combine_first(
        final["discovery_ranking_probability"]
    ).combine_first(final["cancer_native_probability"]).combine_first(final["graph_probability"])

    materialization = cfg.get("materialization", {})
    final["relationship_class"] = _classify_relationships(final, materialization)
    final["rank_within_cancer"] = final.groupby("cancer_id", observed=True)[
        "display_probability"
    ].rank(method="first", ascending=False)
    final["rank_within_cancer_pathway"] = final.groupby(
        ["cancer_id", "pathway_family_id"], observed=True
    )["display_probability"].rank(method="first", ascending=False)
    final["rank_within_cancer_lncrna"] = final.groupby(
        ["cancer_id", "lncrna_id"], observed=True
    )["display_probability"].rank(method="first", ascending=False)
    final["analysis_version"] = cfg["analysis_version"]
    final["model_mode"] = model_mode

    ranking_columns = [
        c for c in [
            *KEYS, "gene_symbol", "gene_name", "pathway_family_name", "family_name",
            "direction", "graph_probability", "cancer_native_probability",
            "cancer_specific_probability", "indirect_mechanism_probability",
            "direct_evidence_probability", "evidence_integrated_probability",
            "discovery_ranking_probability", "fused_confidence_probability",
            "display_probability", "specificity_score", "cold_start_status",
            "candidate_source", "relationship_class", "rank_within_cancer",
            "rank_within_cancer_pathway", "rank_within_cancer_lncrna",
            "analysis_version", "model_mode",
        ] if c in final
    ]
    ranking = final[ranking_columns].sort_values(
        ["cancer_id", "rank_within_cancer"], na_position="last"
    )
    write_table(ranking, web / "web_cancer_lncRNA_pathway_ranking.parquet")
    # Compatibility alias used by the production cancer page.
    write_table(ranking, web / "web_cancer_lncRNA_ranking.parquet")
    write_table(
        ranking.sort_values(
            ["cancer_id", "pathway_family_id", "rank_within_cancer_pathway"],
            na_position="last",
        ),
        web / "web_cancer_pathway_lncRNA_ranking.parquet",
    )

    pathway_summary = (
        final.groupby(["cancer_id", "pathway_family_id"], observed=True)
        .agg(
            n_lncRNAs=("lncrna_id", "nunique"),
            n_observed_core=("relationship_class", lambda x: int((x == "observed_core").sum())),
            n_model_supported=("relationship_class", lambda x: int((x == "model_supported").sum())),
            n_predicted_candidates=("relationship_class", lambda x: int((x == "predicted_candidate").sum())),
            mean_discovery_probability=("discovery_ranking_probability", "mean"),
            mean_confidence_probability=("fused_confidence_probability", "mean"),
            max_display_probability=("display_probability", "max"),
        )
        .reset_index()
    )
    pathway_summary["pathway_rank_within_cancer"] = pathway_summary.groupby(
        "cancer_id", observed=True
    )["max_display_probability"].rank(method="first", ascending=False)
    pathway_summary["rank_within_cancer"] = pathway_summary["pathway_rank_within_cancer"]
    pathway_summary["analysis_version"] = cfg["analysis_version"]
    write_table(pathway_summary, web / "web_cancer_pathway_ranking.parquet")

    # Cancer overview required by the production portal.
    cancer_summary = (
        final.groupby("cancer_id", observed=True)
        .agg(
            detectable_lncRNAs=("lncrna_id", "nunique"),
            significant_lncRNA_pathway_relations=(
                "relationship_class",
                lambda x: int((x != "exploratory").sum()),
            ),
            scored_relationship_count=("display_probability", "count"),
            high_confidence_relations=("relationship_class", lambda x: int(x.isin(["observed_core", "model_supported"]).sum())),
            predicted_candidate_relations=("relationship_class", lambda x: int((x == "predicted_candidate").sum())),
            mean_discovery_probability=("discovery_ranking_probability", "mean"),
            mean_confidence_probability=("fused_confidence_probability", "mean"),
        )
        .reset_index()
    )
    cancer_summary["non_exploratory_relations"] = cancer_summary[
        "significant_lncRNA_pathway_relations"
    ]
    cancer_summary["significant_relation_semantics"] = (
        "compatibility name for thresholded non-exploratory display classes; "
        "not a hypothesis-test significance count"
    )
    dim_cancer_path = cfg["_standardized"] / "dim_cancer.parquet"
    if dim_cancer_path.exists():
        cancer_dim = read_table(dim_cancer_path)
    else:
        configured = input_path(cfg, "dim_cancer")
        cancer_dim = read_table(configured) if configured is not None and configured.exists() else pd.DataFrame(columns=["cancer_id"])
    if not cancer_dim.empty:
        keep = [c for c in ["cancer_id", "english_name", "cancer_name", "full_name", "reference_only", "analysis_tier"] if c in cancer_dim]
        cancer_summary = cancer_summary.merge(cancer_dim[keep].drop_duplicates("cancer_id"), on="cancer_id", how="left")
    if "english_name" not in cancer_summary:
        cancer_summary["english_name"] = cancer_summary.get("full_name", cancer_summary.get("cancer_name", cancer_summary["cancer_id"]))
    cancer_summary["english_name"] = cancer_summary["english_name"].fillna(cancer_summary["cancer_id"])
    if "reference_only" not in cancer_summary:
        reference_only = set(map(str, cfg.get("analysis_cancers", {}).get("reference_only", [])))
        cancer_summary["reference_only"] = cancer_summary["cancer_id"].astype(str).isin(reference_only)
    cancer_summary["model_version"] = "CC-HHGT_v2.9-state-graph"
    cancer_summary["analysis_version"] = cfg["analysis_version"]
    write_table(cancer_summary, web / "web_cancer_overview.parquet")

    pan = (
        final.groupby("lncrna_id", observed=True)
        .agg(
            n_cancers=("cancer_id", "nunique"),
            n_pathway_families=("pathway_family_id", "nunique"),
            mean_discovery_probability=("discovery_ranking_probability", "mean"),
            max_discovery_probability=("discovery_ranking_probability", "max"),
            mean_confidence_probability=("fused_confidence_probability", "mean"),
            max_display_probability=("display_probability", "max"),
        )
        .reset_index()
        .merge(dim_lnc, on="lncrna_id", how="left")
    )
    pan["pancancer_rank"] = pan["max_display_probability"].rank(
        method="first", ascending=False
    )
    pan["analysis_version"] = cfg["analysis_version"]
    write_table(pan.sort_values("pancancer_rank"), web / "web_lncRNA_pan_cancer_ranking.parquet")

    probability_columns = [
        c for c in [
            *KEYS, "graph_probability", "cancer_native_probability",
            "cancer_specific_probability", "indirect_mechanism_probability",
            "direct_evidence_probability", "evidence_integrated_probability",
            "discovery_ranking_probability", "fused_confidence_probability",
            "relationship_class", "analysis_version", "model_mode",
        ] if c in final
    ]
    write_table(final[probability_columns], web / "web_three_probability.parquet")
    write_table(
        pd.DataFrame([
            {
                "analysis_version": cfg["analysis_version"],
                "model_version": "CC-HHGT_v2.9-state-graph",
                "model_mode": model_mode,
                "state_nodes_in_strict_graph": True,
                "strict_models_retrained": True,
            }
        ]),
        web / "web_model_version.parquet",
    )
    return {
        "main_prediction_rows": int(len(final)),
        "main_ranking_rows": int(len(ranking)),
        "pathway_summary_rows": int(len(pathway_summary)),
        "pancancer_lncRNA_rows": int(len(pan)),
        "cancer_overview_rows": int(len(cancer_summary)),
        "model_mode": model_mode,
    }


def _materialize_state(cfg: dict, web: Path, dim_lnc: pd.DataFrame) -> dict:
    release = cfg["_results"] / "v2_9_downstream" / "lncrna_state_release"
    report = read_table(release / "lncrna_state_significance_report.parquet")
    pan = read_table(release / "pancancer_lncrna_rnass_association.parquet")
    report = report.merge(dim_lnc, on="lncrna_id", how="left", suffixes=("", "_dim"))
    pan = pan.merge(dim_lnc, on="lncrna_id", how="left", suffixes=("", "_dim"))

    report["state_rank_within_cancer"] = (
        report.groupby(["cancer_id", "state_id"], observed=True)["state_patient_probability"]
        .rank(method="first", ascending=False)
    )
    report["effect_rank_within_cancer"] = (
        report.groupby(["cancer_id", "state_id"], observed=True)["partial_rho"]
        .rank(method="first", ascending=False)
    )
    report["is_high_confidence"] = (
        report["fdr"].le(0.05)
        & report["partial_rho"].abs().ge(0.15)
        & report["n_folds_same_direction"].ge(4)
    )
    report["evidence_scope"] = "patient_crossfit_plus_v2_9_state_graph"
    report["analysis_version"] = cfg["analysis_version"]

    ranking_columns = [
        c for c in [
            "cancer_id", "lncrna_id", "gene_symbol", "gene_name", "state_id",
            "partial_rho", "ci_lower", "ci_upper", "p_value", "fdr",
            "n_patients", "detection_rate", "n_folds_available",
            "n_folds_same_direction", "fold_selection_frequency",
            "state_patient_probability", "direction_probability", "direction",
            "replication_tier", "conclusion", "is_high_confidence",
            "state_rank_within_cancer", "effect_rank_within_cancer",
            "evidence_scope", "analysis_version",
        ] if c in report
    ]
    ranking = report[ranking_columns].sort_values(
        ["cancer_id", "state_id", "state_rank_within_cancer"]
    )
    write_table(ranking, web / "web_lncRNA_state_ranking.parquet")
    write_table(
        ranking.sort_values(["state_id", "cancer_id", "state_rank_within_cancer"]),
        web / "web_state_lncRNA_ranking.parquet",
    )

    matrix = report.pivot_table(
        index=["cancer_id", "lncrna_id"],
        columns="state_id",
        values="state_patient_probability",
        aggfunc="mean",
    ).reset_index()
    matrix.columns = [str(c) for c in matrix.columns]
    write_table(matrix, web / "web_lncRNA_state_matrix.parquet")

    summary = (
        report.groupby(["cancer_id", "state_id"], observed=True)
        .agg(
            n_tested_lncRNAs=("lncrna_id", "nunique"),
            n_significant=("is_high_confidence", "sum"),
            n_robust=("replication_tier", lambda x: int((x == "robust_core").sum())),
            median_probability=("state_patient_probability", "median"),
            median_abs_effect=("partial_rho", lambda x: float(np.nanmedian(np.abs(x)))),
        )
        .reset_index()
    )
    summary["analysis_version"] = cfg["analysis_version"]
    write_table(summary, web / "web_cancer_state_lncRNA_summary.parquet")
    write_table(summary, web / "web_cancer_state_summary.parquet")

    # Explicitly materialize unavailable optional modules instead of returning 503.
    cancers = sorted(report["cancer_id"].astype(str).unique())
    for table_name, module_name in [
        ("web_cancer_celltype_summary.parquet", "celltype"),
        ("web_cancer_drug_summary.parquet", "drug"),
        ("web_cancer_geneset_summary.parquet", "geneset"),
    ]:
        target = web / table_name
        if not target.exists():
            write_table(
                pd.DataFrame({
                    "cancer_id": cancers,
                    "module": module_name,
                    "data_available": False,
                    "failure_reason": "not_materialized_by_v2_9_state_pipeline",
                    "analysis_version": cfg["analysis_version"],
                }),
                target,
            )

    pan["pancancer_rank"] = pan["mean_cancer_state_probability"].rank(
        method="first", ascending=False
    )
    pan["analysis_version"] = cfg["analysis_version"]
    write_table(pan.sort_values("pancancer_rank"), web / "web_pancancer_rnass.parquet")
    return {
        "state_ranking_rows": int(len(ranking)),
        "state_matrix_rows": int(len(matrix)),
        "state_summary_rows": int(len(summary)),
        "pancancer_rnass_rows": int(len(pan)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize V2.9 main and state web tables")
    parser.add_argument("--config", default="config/model_v2_9_state_graph.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    web = cfg["_results"] / "web_tables"
    web.mkdir(parents=True, exist_ok=True)
    dim_lnc = _lncrna_dimension(cfg)
    main_summary = _materialize_main(cfg, web, dim_lnc)
    state_summary = _materialize_state(cfg, web, dim_lnc)

    required = [
        "web_cancer_overview.parquet",
        "web_cancer_lncRNA_pathway_ranking.parquet",
        "web_cancer_lncRNA_ranking.parquet",
        "web_cancer_pathway_lncRNA_ranking.parquet",
        "web_cancer_pathway_ranking.parquet",
        "web_lncRNA_pan_cancer_ranking.parquet",
        "web_three_probability.parquet",
        "web_model_version.parquet",
        "web_lncRNA_state_ranking.parquet",
        "web_state_lncRNA_ranking.parquet",
        "web_lncRNA_state_matrix.parquet",
        "web_cancer_state_lncRNA_summary.parquet",
        "web_cancer_state_summary.parquet",
        "web_cancer_celltype_summary.parquet",
        "web_cancer_drug_summary.parquet",
        "web_cancer_geneset_summary.parquet",
        "web_pancancer_rnass.parquet",
    ]
    missing = [name for name in required if not (web / name).exists()]
    payload = {
        "status": "COMPLETED" if not missing else "FAILED",
        "analysis_version": cfg["analysis_version"],
        **main_summary,
        **state_summary,
        "missing_tables": missing,
        "tables": required,
    }
    write_json(payload, web / "V2_9_WEB_TABLES_SUCCESS.json")
    # Backward-compatible state-only success marker.
    write_json(payload, web / "V2_9_STATE_WEB_TABLES_SUCCESS.json")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if missing:
        raise RuntimeError(f"V2.9 web tables missing: {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
