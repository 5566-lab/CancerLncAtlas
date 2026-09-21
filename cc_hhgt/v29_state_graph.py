from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats

from .common import (
    LOGGER,
    first_existing_column,
    input_path,
    read_table,
    stable_id,
    write_json,
    write_table,
)
from .io import read_cancer_partition
from .stats import (
    bh_fdr,
    correlation_p_values,
    design_rank,
    prepare_design,
    rank_transform,
    residualize,
    standardize,
)

DEFAULT_REQUIRED_STATES = [
    "EXTEND::published_score",
    "stemness_rna::RNAss",
    "stemness_dna::DNAss",
    "stemness_rna::EREG.EXPss",
]


@dataclass(frozen=True)
class StateGraphPaths:
    catalog: Path
    pathway_state: Path
    family_state: Path
    gene_state: Path
    cancer_state: Path
    strict_candidates: Path
    audit: Path


def paths(cfg: dict[str, Any]) -> StateGraphPaths:
    tables = cfg["_results"] / "tables"
    return StateGraphPaths(
        catalog=tables / "state_catalog.parquet",
        pathway_state=tables / "pathway_state_edge",
        family_state=tables / "pathway_family_state_edge",
        gene_state=tables / "gene_state_edge",
        cancer_state=tables / "cancer_state_edge.parquet",
        strict_candidates=tables / "strict_state_candidate",
        audit=tables / "V2_9_STATE_GRAPH_AUDIT.json",
    )


def _state_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    raw = dict(cfg.get("state_graph", {}))
    raw.setdefault("required_states", DEFAULT_REQUIRED_STATES)
    raw.setdefault("include_regex", r"(?i)(extend|telomerase|rnass|dnass|ereg[._:-]?expss|stemness)")
    raw.setdefault("min_samples", 30)
    raw.setdefault("min_abs_pathway_effect", 0.15)
    raw.setdefault("min_abs_gene_effect", 0.20)
    raw.setdefault("max_fdr", 0.05)
    raw.setdefault("max_gene_edges_per_state_direction", 250)
    raw.setdefault("min_gene_variance", 0.01)
    raw.setdefault("max_genes_per_cancer", 25000)
    raw.setdefault("candidate_strong_abs_effect", 0.20)
    raw.setdefault("candidate_strong_fdr", 0.05)
    raw.setdefault("candidate_weak_abs_effect", 0.12)
    raw.setdefault("candidate_weak_fdr", 0.20)
    raw.setdefault("candidate_top_unlabeled_per_state", 2500)
    raw.setdefault("candidate_random_unlabeled_per_state", 2500)
    raw.setdefault("candidate_min_detection", 0.05)
    raw.setdefault("candidate_max_lncRNAs", 16000)
    raw.setdefault("lnc_block_size", 256)
    raw.setdefault(
        "covariates",
        [
            "purity",
            "age_years",
            "sex",
            "stage",
            "molecular_subtype",
            "clinical_subtype",
            "technical_batch",
            "leukocyte_fraction",
        ],
    )
    return raw


def _sample_key(value: object) -> str:
    return str(value)[:15]


def _state_matrix(cfg: dict[str, Any], selected: set[str] | None = None) -> pd.DataFrame:
    state_path = cfg["_standardized"] / "tumor_state_long.parquet"
    if not state_path.exists():
        raise FileNotFoundError(
            f"{state_path} is missing. Run scripts/01_standardize_model_inputs.py first."
        )
    long = read_table(state_path)
    required = {"sample_id", "cancer_id", "state_id", "state_value"}
    missing = required - set(long.columns)
    if missing:
        raise ValueError(f"tumor_state_long missing columns: {sorted(missing)}")
    long = long.copy()
    long["sample_id"] = long["sample_id"].map(_sample_key)
    long["cancer_id"] = long["cancer_id"].astype(str)
    long["state_id"] = long["state_id"].astype(str)
    long["state_value"] = pd.to_numeric(long["state_value"], errors="coerce")
    long = long.dropna(subset=["state_value"])
    if selected is not None:
        long = long.loc[long.state_id.isin(selected)]
    return long


def select_states(cfg: dict[str, Any]) -> tuple[list[str], pd.DataFrame]:
    settings = _state_cfg(cfg)
    long = _state_matrix(cfg)
    observed = sorted(long.state_id.unique())
    required = [str(x) for x in settings["required_states"]]
    missing = sorted(set(required) - set(observed))
    if missing:
        raise RuntimeError(
            "Required tumor-state columns are missing from standardized data: "
            f"{missing}. Observed examples={observed[:30]}"
        )
    pattern = re.compile(str(settings["include_regex"]))
    selected = sorted(set(required) | {x for x in observed if pattern.search(x)})
    if not selected:
        raise RuntimeError("No tumor states selected for V2.9")
    return selected, long.loc[long.state_id.isin(selected)].copy()


def build_state_catalog(cfg: dict[str, Any]) -> pd.DataFrame:
    selected, long = select_states(cfg)
    summary = (
        long.groupby("state_id", observed=True)
        .agg(
            n_values=("state_value", "size"),
            n_samples=("sample_id", "nunique"),
            n_cancers=("cancer_id", "nunique"),
            mean_value=("state_value", "mean"),
            sd_value=("state_value", "std"),
            min_value=("state_value", "min"),
            max_value=("state_value", "max"),
        )
        .reset_index()
    )
    required = set(map(str, _state_cfg(cfg)["required_states"]))
    summary["required_for_v2_9"] = summary.state_id.isin(required)
    summary["state_group"] = np.select(
        [
            summary.state_id.str.contains("EXTEND|telomerase", case=False, regex=True),
            summary.state_id.str.contains("RNAss|EREG", case=False, regex=True),
            summary.state_id.str.contains("DNAss", case=False, regex=True),
        ],
        ["telomerase", "rna_stemness", "dna_stemness"],
        default="other_state",
    )
    summary["node_label"] = summary.state_id
    summary["analysis_version"] = cfg["analysis_version"]
    write_table(summary, paths(cfg).catalog)
    write_json(
        {
            "status": "PASS",
            "selected_states": selected,
            "required_states": sorted(required),
            "n_states": len(summary),
            "n_values": int(len(long)),
        },
        cfg["_results"] / "tables" / "state_catalog_summary.json",
    )
    return summary


def _aligned_state_for_cancer(long: pd.DataFrame, cancer: str, selected: Iterable[str]) -> pd.DataFrame:
    state = long.loc[
        long.cancer_id.astype(str).eq(str(cancer))
        & long.state_id.astype(str).isin(set(map(str, selected)))
    ].pivot_table(index="sample_id", columns="state_id", values="state_value", aggfunc="mean")
    state.index = state.index.map(_sample_key)
    return state.groupby(level=0).mean()


def _covariate_design(cfg: dict[str, Any], cancer: str, samples: list[str]) -> np.ndarray:
    cov_path = input_path(cfg, "bulk_covariates")
    if cov_path is None or not cov_path.exists():
        return np.ones((len(samples), 1), dtype=float)
    cov = read_table(cov_path)
    cov = cov.loc[cov.cancer_id.astype(str).eq(str(cancer))].copy()
    cov["sample_id"] = cov["sample_id"].map(_sample_key)
    cov = cov.drop_duplicates("sample_id").set_index("sample_id").reindex(samples)
    cov_cols = [x for x in _state_cfg(cfg)["covariates"] if x in cov.columns]
    if not cov_cols:
        return np.ones((len(samples), 1), dtype=float)
    return prepare_design(cov[cov_cols].reset_index(drop=True))


def _residual_matrix(frame: pd.DataFrame, design: np.ndarray) -> np.ndarray:
    filled = frame.copy()
    filled = filled.apply(pd.to_numeric, errors="coerce")
    all_missing = filled.isna().all(axis=0)
    if all_missing.any():
        filled.loc[:, all_missing] = 0.0
    filled = filled.fillna(filled.median()).fillna(0)
    return standardize(residualize(rank_transform(filled.to_numpy(float)), design))



def build_pathway_state_edges_v29(cfg: dict[str, Any], cancers: Iterable[str] | None = None) -> pd.DataFrame:
    settings = _state_cfg(cfg)
    selected, long = select_states(cfg)
    dim = read_table(cfg["_standardized"] / "dim_cancer.parquet")
    cancer_list = list(cancers) if cancers else dim.cancer_id.astype(str).tolist()
    pathway_root = input_path(cfg, "bulk_pathway_activity")
    if pathway_root is None:
        raise RuntimeError("bulk_pathway_activity is not configured")
    out_root = paths(cfg).pathway_state
    out_root.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for cancer in cancer_list:
        activity = read_cancer_partition(pathway_root, cancer)
        if activity.empty:
            summary_rows.append({"cancer_id": cancer, "status": "MISSING_ACTIVITY", "n_edges": 0})
            continue
        value_col = first_existing_column(activity, cfg["column_aliases"]["pathway_value"])
        activity["sample_id"] = activity["sample_id"].map(_sample_key)
        matrix = activity.pivot_table(index="sample_id", columns="pathway_id", values=value_col, aggfunc="mean").groupby(level=0).mean()
        state = _aligned_state_for_cancer(long, cancer, selected)
        samples = sorted(set(matrix.index) & set(state.index))
        if len(samples) < int(settings["min_samples"]):
            summary_rows.append({"cancer_id": cancer, "status": "INSUFFICIENT_SAMPLES", "n_samples": len(samples), "n_edges": 0})
            continue
        matrix = matrix.reindex(samples)
        state = state.reindex(samples)
        state = state.loc[:, state.notna().sum() >= int(settings["min_samples"])]
        if matrix.empty or state.empty:
            summary_rows.append({"cancer_id": cancer, "status": "NO_VARIABLE_FEATURES", "n_edges": 0})
            continue
        design = _covariate_design(cfg, cancer, samples)
        residual_rank = design_rank(design)
        correlation_df = len(samples) - residual_rank - 1
        if correlation_df <= 0:
            raise RuntimeError(f"Invalid pathway-state residual df for {cancer}: n={len(samples)}, rank={residual_rank}")
        ax = _residual_matrix(matrix, design)
        sx = _residual_matrix(state, design)
        corr = ax.T @ sx / max(len(samples) - 1, 1)
        pval = correlation_p_values(corr, len(samples), residual_design_rank=residual_rank)
        parts = []
        for j, state_id in enumerate(state.columns.astype(str)):
            effect = corr[:, j]
            pvalue = pval[:, j]
            fdr = bh_fdr(pvalue, total_tests=len(pvalue))
            part = pd.DataFrame({
                "pathway_id": matrix.columns.astype(str),
                "state_id": state_id,
                "effect": effect,
                "p_value": pvalue,
                "fdr": fdr,
            })
            part = part.loc[part.effect.abs().ge(float(settings["min_abs_pathway_effect"])) & part.fdr.le(float(settings["max_fdr"]))]
            if part.empty:
                continue
            part["direction"] = np.where(part.effect.ge(0), "positive", "negative")
            parts.append(part)
        if not parts:
            summary_rows.append({"cancer_id": cancer, "status": "NO_SIGNIFICANT_EDGES", "n_samples": len(samples), "n_edges": 0})
            continue
        out = pd.concat(parts, ignore_index=True)
        out["cancer_id"] = cancer
        out["n_samples"] = len(samples)
        out["residual_design_rank"] = residual_rank
        out["correlation_df"] = correlation_df
        out["edge_id"] = [stable_id("PSTATE29", cancer, p, st) for p, st in zip(out.pathway_id, out.state_id)]
        out["method"] = "covariate_residual_spearman_v2_9_effective_df"
        write_table(out, out_root / f"cancer_id={cancer}" / "part-0.parquet")
        summary_rows.append({"cancer_id": cancer, "status": "PASS", "n_samples": len(samples), "n_edges": len(out), "n_states": out.state_id.nunique()})
    summary = pd.DataFrame(summary_rows)
    write_table(summary, cfg["_results"] / "tables" / "pathway_state_summary.tsv")
    return summary

def build_gene_state_edges(cfg: dict[str, Any], cancers: Iterable[str] | None = None) -> pd.DataFrame:
    settings = _state_cfg(cfg)
    selected, long = select_states(cfg)
    dim = read_table(cfg["_standardized"] / "dim_cancer.parquet")
    cancer_list = list(cancers) if cancers else dim.cancer_id.astype(str).tolist()
    gene_root = input_path(cfg, "bulk_gene_expression")
    if gene_root is None:
        raise RuntimeError("bulk_gene_expression is not configured")
    out_root = paths(cfg).gene_state
    out_root.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    for cancer in cancer_list:
        out_file = out_root / f"cancer_id={cancer}" / "part-0.parquet"
        gene = read_cancer_partition(gene_root, cancer)
        if gene.empty:
            summary_rows.append({"cancer_id": cancer, "status": "MISSING_GENE_EXPRESSION", "n_edges": 0})
            continue
        value_col = first_existing_column(gene, cfg["column_aliases"]["expression_value"])
        gene["sample_id"] = gene["sample_id"].map(_sample_key)
        matrix = gene.pivot_table(index="sample_id", columns="gene_id", values=value_col, aggfunc="mean")
        state = _aligned_state_for_cancer(long, cancer, selected)
        samples = sorted(set(matrix.index) & set(state.index))
        if len(samples) < int(settings["min_samples"]):
            summary_rows.append({"cancer_id": cancer, "status": "INSUFFICIENT_SAMPLES", "n_samples": len(samples), "n_edges": 0})
            continue
        matrix = matrix.reindex(samples)
        state = state.reindex(samples)
        variance = matrix.var(axis=0, ddof=1)
        keep = variance.loc[variance >= float(settings["min_gene_variance"])].sort_values(ascending=False).head(int(settings["max_genes_per_cancer"])).index
        matrix = matrix.loc[:, keep]
        state = state.loc[:, state.notna().sum() >= int(settings["min_samples"])]
        if matrix.empty or state.empty:
            summary_rows.append({"cancer_id": cancer, "status": "NO_VARIABLE_FEATURES", "n_edges": 0})
            continue
        design = _covariate_design(cfg, cancer, samples)
        residual_rank = design_rank(design)
        correlation_df = len(samples) - residual_rank - 1
        if correlation_df <= 0:
            raise RuntimeError(f"Invalid gene-state residual df for {cancer}: n={len(samples)}, rank={residual_rank}")
        gx = _residual_matrix(matrix, design)
        sx = _residual_matrix(state, design)
        corr = gx.T @ sx / max(len(samples) - 1, 1)
        pval = correlation_p_values(corr, len(samples), residual_design_rank=residual_rank)
        rows: list[pd.DataFrame] = []
        for j, state_id in enumerate(state.columns.astype(str)):
            effect = corr[:, j]
            p = pval[:, j]
            fdr = bh_fdr(p, total_tests=len(p))
            candidate = pd.DataFrame(
                {
                    "gene_id": matrix.columns.astype(str),
                    "state_id": state_id,
                    "effect": effect,
                    "p_value": p,
                    "fdr": fdr,
                }
            )
            candidate = candidate.loc[
                candidate.effect.abs().ge(float(settings["min_abs_gene_effect"]))
                & candidate.fdr.le(float(settings["max_fdr"]))
            ]
            if candidate.empty:
                continue
            candidate["direction"] = np.where(candidate.effect.ge(0), "positive", "negative")
            candidate = (
                candidate.assign(abs_effect=candidate.effect.abs())
                .sort_values(["direction", "abs_effect"], ascending=[True, False])
                .groupby("direction", observed=True)
                .head(int(settings["max_gene_edges_per_state_direction"]))
                .drop(columns="abs_effect")
            )
            rows.append(candidate)
        if not rows:
            summary_rows.append({"cancer_id": cancer, "status": "NO_SIGNIFICANT_EDGES", "n_samples": len(samples), "n_edges": 0})
            continue
        out = pd.concat(rows, ignore_index=True)
        out["cancer_id"] = cancer
        out["n_samples"] = len(samples)
        out["residual_design_rank"] = residual_rank
        out["correlation_df"] = correlation_df
        out["edge_id"] = [stable_id("GSTATE", cancer, g, s) for g, s in zip(out.gene_id, out.state_id)]
        out["method"] = "train-cancer_covariate_residual_spearman_effective_df"
        write_table(out, out_file)
        summary_rows.append({"cancer_id": cancer, "status": "PASS", "n_samples": len(samples), "n_edges": len(out), "n_states": out.state_id.nunique()})
    summary = pd.DataFrame(summary_rows)
    write_table(summary, cfg["_results"] / "tables" / "gene_state_summary.tsv")
    return summary


def build_family_state_edges(cfg: dict[str, Any]) -> pd.DataFrame:
    pstate_root = paths(cfg).pathway_state
    if not pstate_root.exists():
        raise FileNotFoundError(pstate_root)
    ps = read_table(pstate_root)
    member = read_table(cfg["_results"] / "tables" / "pathway_family_member.parquet")
    needed = {"pathway_id", "pathway_family_id"}
    if not needed.issubset(member.columns):
        raise ValueError(f"pathway_family_member missing {sorted(needed - set(member.columns))}")
    weight_col = "membership_weight" if "membership_weight" in member else None
    use = member[["pathway_id", "pathway_family_id"] + ([weight_col] if weight_col else [])].copy()
    if weight_col is None:
        use["membership_weight"] = 1.0
    merged = ps.merge(use, on="pathway_id", how="inner")
    merged["weighted_effect"] = merged.effect * merged.membership_weight
    grouped = merged.groupby(["cancer_id", "pathway_family_id", "state_id"], observed=True)
    out = grouped.agg(
        effect_sum=("weighted_effect", "sum"),
        weight_sum=("membership_weight", "sum"),
        min_fdr=("fdr", "min"),
        n_pathways=("pathway_id", "nunique"),
        n_samples=("n_samples", "max"),
    ).reset_index()
    out["effect"] = out.effect_sum / out.weight_sum.clip(lower=1e-8)
    out["direction"] = np.where(out.effect.ge(0), "positive", "negative")
    out["edge_id"] = [stable_id("PFSTATE", c, p, s) for c, p, s in zip(out.cancer_id, out.pathway_family_id, out.state_id)]
    out["method"] = "membership_weighted_pathway_state"
    out = out.drop(columns=["effect_sum", "weight_sum"])
    write_table(out, paths(cfg).family_state, partition_cols=["cancer_id"])
    return out


def build_cancer_state_edges(cfg: dict[str, Any]) -> pd.DataFrame:
    selected, long = select_states(cfg)
    grouped = (
        long.groupby(["cancer_id", "state_id"], observed=True)
        .agg(effect=("state_value", "mean"), state_sd=("state_value", "std"), n_samples=("sample_id", "nunique"))
        .reset_index()
    )
    grouped = grouped.loc[grouped.state_id.isin(selected)]
    # Never normalize across all cancers here: the held-out LOCO cancer must
    # not influence a training edge's sign or magnitude.  The graph loader
    # standardizes these raw means after test/validation context removal.
    grouped["direction"] = "requires_fold_localization"
    grouped["edge_id"] = [stable_id("CSTATE", c, s) for c, s in zip(grouped.cancer_id, grouped.state_id)]
    grouped["method"] = "raw_cancer_mean_requires_fold_local_standardization"
    write_table(grouped, paths(cfg).cancer_state)
    return grouped


def _lnc_matrix(cfg: dict[str, Any], cancer: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = input_path(cfg, "bulk_lnc_expression")
    if root is None:
        raise RuntimeError("bulk_lnc_expression is not configured")
    frame = read_cancer_partition(root, cancer)
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()
    value_col = first_existing_column(frame, cfg["column_aliases"]["expression_value"])
    frame["sample_id"] = frame["sample_id"].map(_sample_key)
    expr = frame.pivot_table(index="sample_id", columns="lncrna_id", values=value_col, aggfunc="mean")
    if "tpm" in frame:
        detection = frame.assign(detected=pd.to_numeric(frame.tpm, errors="coerce").gt(0.1).astype(float)).pivot_table(index="sample_id", columns="lncrna_id", values="detected", aggfunc="max").fillna(0)
    else:
        detection = expr.notna().astype(float)
    return expr.groupby(level=0).mean(), detection.groupby(level=0).max()


def build_strict_state_candidates(cfg: dict[str, Any], cancers: Iterable[str] | None = None) -> pd.DataFrame:
    settings = _state_cfg(cfg)
    selected, long = select_states(cfg)
    dim = read_table(cfg["_standardized"] / "dim_cancer.parquet")
    cancer_list = list(cancers) if cancers else dim.cancer_id.astype(str).tolist()
    root = paths(cfg).strict_candidates
    root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    for cancer in cancer_list:
        out_file = root / f"cancer_id={cancer}" / "part-0.parquet"
        expression, detection = _lnc_matrix(cfg, cancer)
        state = _aligned_state_for_cancer(long, cancer, selected)
        samples = sorted(set(expression.index) & set(state.index))
        if len(samples) < int(settings["min_samples"]):
            summaries.append({"cancer_id": cancer, "status": "INSUFFICIENT_SAMPLES", "n_samples": len(samples), "n_candidates": 0})
            continue
        expression = expression.reindex(samples)
        detection = detection.reindex(samples).fillna(0)
        state = state.reindex(samples)
        detection_rate = detection.mean(axis=0)
        keep = detection_rate.loc[detection_rate >= float(settings["candidate_min_detection"])].index
        variance = expression[keep].var(axis=0, ddof=1).sort_values(ascending=False)
        keep = variance.head(int(settings["candidate_max_lncRNAs"])).index
        expression = expression[keep]
        state = state.loc[:, state.notna().sum() >= int(settings["min_samples"])]
        if expression.empty or state.empty:
            summaries.append({"cancer_id": cancer, "status": "NO_VARIABLE_FEATURES", "n_candidates": 0})
            continue
        design = _covariate_design(cfg, cancer, samples)
        residual_rank = design_rank(design)
        correlation_df = len(samples) - residual_rank - 1
        if correlation_df <= 0:
            raise RuntimeError(f"Invalid lncRNA-state residual df for {cancer}: n={len(samples)}, rank={residual_rank}")
        x = _residual_matrix(expression, design)
        y = _residual_matrix(state, design)
        corr = x.T @ y / max(len(samples) - 1, 1)
        pval = correlation_p_values(corr, len(samples), residual_design_rank=residual_rank)
        parts: list[pd.DataFrame] = []
        for j, state_id in enumerate(state.columns.astype(str)):
            effect = corr[:, j]
            p = pval[:, j]
            fdr = bh_fdr(p, total_tests=len(p))
            table = pd.DataFrame(
                {
                    "lncrna_id": expression.columns.astype(str),
                    "state_id": state_id,
                    "effect": effect,
                    "p_value": p,
                    "fdr": fdr,
                    "detection_rate": expression.columns.astype(str).map(detection_rate.to_dict()).astype(float),
                }
            )
            strong = table.effect.abs().ge(float(settings["candidate_strong_abs_effect"])) & table.fdr.le(float(settings["candidate_strong_fdr"]))
            weak = table.effect.abs().ge(float(settings["candidate_weak_abs_effect"])) & table.fdr.le(float(settings["candidate_weak_fdr"])) & ~strong
            # Retain the complete detectable lncRNA x state universe for
            # validation/test and calibration.  PU capping is applied only to
            # the training cancers later, never to held-out evaluation rows.
            selected_table = table.copy()
            selected_table["label_class"] = np.select(
                [
                    strong,
                    weak,
                ],
                ["strong_positive", "weak_positive"],
                default="unlabeled",
            )
            selected_table["proxy_label"] = selected_table.label_class.ne("unlabeled").astype("int8")
            direction_known = selected_table.label_class.ne("unlabeled")
            selected_table["direction"] = np.where(
                direction_known,
                np.where(selected_table.effect.ge(0), "positive", "negative"),
                "unknown",
            )
            selected_table["direction_label"] = np.where(
                direction_known,
                selected_table.effect.ge(0).astype(float),
                np.nan,
            )
            selected_table["cancer_id"] = cancer
            selected_table["n_samples"] = len(samples)
            selected_table["residual_design_rank"] = residual_rank
            selected_table["correlation_df"] = correlation_df
            selected_table["evaluation_sampling_policy"] = "full_detectable_lncrna_state_universe"
            selected_table["candidate_id"] = [stable_id("STATEPAIR", cancer, l, state_id) for l in selected_table.lncrna_id]
            selected_table["label_source"] = "covariate_residual_spearman_state_auxiliary_effective_df"
            selected_table["method"] = "covariate_residual_spearman_state_auxiliary_effective_df"
            parts.append(selected_table)
        if not parts:
            summaries.append({"cancer_id": cancer, "status": "NO_CANDIDATES", "n_candidates": 0})
            continue
        out = pd.concat(parts, ignore_index=True)
        write_table(out, out_file)
        summaries.append(
            {
                "cancer_id": cancer,
                "status": "PASS",
                "n_samples": len(samples),
                "n_states": out.state_id.nunique(),
                "n_candidates": len(out),
                "n_strong": int(out.label_class.eq("strong_positive").sum()),
                "n_weak": int(out.label_class.eq("weak_positive").sum()),
            }
        )
    summary = pd.DataFrame(summaries)
    write_table(summary, cfg["_results"] / "tables" / "strict_state_candidate_summary.tsv")
    return summary


def build_all_state_assets(cfg: dict[str, Any], cancers: Iterable[str] | None = None) -> dict[str, Any]:
    catalog = build_state_catalog(cfg)
    pathway_summary = build_pathway_state_edges_v29(cfg, cancers)
    gene_summary = build_gene_state_edges(cfg, cancers)
    family = build_family_state_edges(cfg)
    cancer_state = build_cancer_state_edges(cfg)
    candidate_summary = build_strict_state_candidates(cfg, cancers)
    payload = {
        "status": "PASS",
        "analysis_version": cfg["analysis_version"],
        "required_states": _state_cfg(cfg)["required_states"],
        "n_state_nodes": int(len(catalog)),
        "pathway_state_summary_rows": int(len(pathway_summary)),
        "gene_state_summary_rows": int(len(gene_summary)),
        "n_family_state_edges": int(len(family)),
        "n_cancer_state_edges": int(len(cancer_state)),
        "strict_candidate_summary_rows": int(len(candidate_summary)),
    }
    write_json(payload, paths(cfg).audit)
    return payload
