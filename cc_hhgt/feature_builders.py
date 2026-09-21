from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats

from .common import LOGGER, first_existing_column, input_path, read_table, stable_id, write_table
from .io import read_cancer_partition
from .stats import bh_fdr, correlation_p_values, design_rank, fisher_meta_correlations, prepare_design, rank_transform, residualize, robust_z, standardize


def _select_variable(matrix: pd.DataFrame, max_features: int, min_variance: float) -> pd.DataFrame:
    variance = matrix.var(axis=0, ddof=1)
    keep = variance.loc[variance >= min_variance].sort_values(ascending=False).head(max_features).index
    return matrix.loc[:, keep]


def _expression_matrix(
    path: Path,
    cancer: str,
    entity_col: str,
    value_aliases: list[str],
    allowed_samples: set[str] | None = None,
) -> pd.DataFrame:
    frame = read_cancer_partition(path, cancer)
    if frame.empty:
        return pd.DataFrame()
    frame["sample_id"] = frame.sample_id.astype(str)
    if allowed_samples is not None:
        frame = frame.loc[frame.sample_id.isin(allowed_samples)].copy()
    duplicate = frame.duplicated(["sample_id", entity_col], keep=False)
    if duplicate.any():
        raise RuntimeError(
            f"Duplicate exact sample/entity rows for {cancer}/{entity_col}: "
            f"{frame.loc[duplicate, ['sample_id', entity_col]].head(10).to_dict('records')}"
        )
    value_col = first_existing_column(frame, value_aliases)
    frame[value_col] = pd.to_numeric(frame[value_col], errors="coerce")
    return frame.pivot(index="sample_id", columns=entity_col, values=value_col)


def build_lnc_gene_coexpression(cfg: dict[str, Any], cancers: Iterable[str] | None = None) -> pd.DataFrame:
    c = cfg["coexpression"]
    lnc_path = input_path(cfg, "bulk_lnc_expression")
    gene_path = input_path(cfg, "bulk_gene_expression")
    cov_path = input_path(cfg, "bulk_covariates")
    dim_cancer = read_table(cfg["_standardized"] / "dim_cancer.parquet")
    cancer_list = list(cancers) if cancers else dim_cancer.cancer_id.astype(str).tolist()
    cov_all = read_table(cov_path)
    out_dir = cfg["_results"] / "tables" / "lnc_gene_coexpression"
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    canonical = None
    canonical_path = cfg.get("_canonical_samples_path")
    if canonical_path:
        canonical = read_table(Path(canonical_path))
        if canonical.duplicated(["cancer_id", "sample_id"]).any():
            raise RuntimeError("Canonical samples are not unique for coexpression")
        if not canonical.is_tumor.astype(bool).all():
            raise RuntimeError("Non-tumor samples entered canonical coexpression input")

    for cancer in cancer_list:
        out_file = out_dir / f"cancer_id={cancer}" / "part-0.parquet"
        if out_file.exists():
            summaries.append({"cancer_id": cancer, "status": "CACHED", "n_edges": len(pd.read_parquet(out_file, columns=["edge_id"]))})
            continue
        LOGGER.info("Computing lncRNA-gene coexpression for %s", cancer)
        allowed = (
            set(canonical.loc[canonical.cancer_id.astype(str).eq(str(cancer)), "sample_id"].astype(str))
            if canonical is not None
            else None
        )
        lnc = _expression_matrix(lnc_path, cancer, "lncrna_id", cfg["column_aliases"]["expression_value"], allowed)
        gene = _expression_matrix(gene_path, cancer, "gene_id", cfg["column_aliases"]["expression_value"], allowed)
        samples = sorted(set(lnc.index) & set(gene.index))
        if len(samples) < c["min_samples"]:
            summaries.append({"cancer_id": cancer, "status": "INSUFFICIENT_SAMPLES", "n_samples": len(samples), "n_edges": 0})
            continue
        lnc = lnc.reindex(samples)
        gene = gene.reindex(samples)
        detection = (lnc > 0).mean(axis=0)
        lnc = lnc.loc[:, detection >= c["min_detection_rate"]]
        lnc = _select_variable(lnc, c["max_lncRNAs_per_cancer"], c["min_variance"])
        gene = _select_variable(gene, c["max_genes_per_cancer"], c["min_variance"])
        if lnc.empty or gene.empty:
            summaries.append({"cancer_id": cancer, "status": "NO_VARIABLE_FEATURES", "n_edges": 0})
            continue

        cov = cov_all.loc[cov_all.cancer_id.astype(str) == cancer].copy()
        cov["sample_id"] = cov.sample_id.astype(str)
        cov = cov.loc[cov.sample_id.isin(samples)]
        if cov.duplicated("sample_id").any():
            raise RuntimeError(f"Duplicate exact covariate rows for coexpression/{cancer}")
        cov = cov.set_index("sample_id").reindex(samples)
        cov_cols = [x for x in c["covariates"] if x in cov.columns]
        design = prepare_design(cov[cov_cols].reset_index(drop=True)) if cov_cols else np.ones((len(samples), 1))
        residual_rank = design_rank(design)
        correlation_df = len(samples) - residual_rank - 1
        if correlation_df <= 0:
            raise RuntimeError(
                f"No residual correlation degrees of freedom for {cancer}: "
                f"n={len(samples)}, design_rank={residual_rank}"
            )
        lnc = lnc.fillna(lnc.median())
        gene = gene.fillna(gene.median())
        lnc_x = standardize(residualize(rank_transform(lnc.to_numpy(float)), design))
        gene_x = standardize(residualize(rank_transform(gene.to_numpy(float)), design))
        denom = max(len(samples) - 1, 1)
        total_tests = lnc_x.shape[1] * gene_x.shape[1]
        rows: list[pd.DataFrame] = []
        block_size = int(c["lnc_block_size"])
        for start in range(0, lnc_x.shape[1], block_size):
            stop = min(start + block_size, lnc_x.shape[1])
            corr = (lnc_x[:, start:stop].T @ gene_x) / denom
            p = correlation_p_values(corr, len(samples), residual_design_rank=residual_rank)
            mask = np.abs(corr) >= c["min_abs_rho"]
            if not mask.any():
                continue
            bi, gj = np.where(mask)
            part = pd.DataFrame({
                "lncrna_id": lnc.columns.to_numpy()[start:stop][bi],
                "gene_id": gene.columns.to_numpy()[gj],
                "rho": corr[bi, gj],
                "p_value": p[bi, gj],
            })
            part["fdr"] = bh_fdr(part.p_value.to_numpy(), total_tests=total_tests)
            part = part.loc[part.fdr <= c["max_fdr"]]
            rows.append(part)
        if not rows:
            summaries.append({"cancer_id": cancer, "status": "NO_SIGNIFICANT_EDGES", "n_samples": len(samples), "n_edges": 0})
            continue
        edges = pd.concat(rows, ignore_index=True)
        edges["direction"] = np.where(edges.rho >= 0, "positive", "negative")
        edges["abs_rho"] = edges.rho.abs()
        edges = edges.sort_values(["lncrna_id", "direction", "abs_rho"], ascending=[True, True, False])
        edges = edges.groupby(["lncrna_id", "direction"], observed=True).head(c["max_edges_per_lncRNA_direction"])
        edges["cancer_id"] = cancer
        edges["n_samples"] = len(samples)
        edges["residual_design_rank"] = residual_rank
        edges["correlation_df"] = correlation_df
        edges["n_observed"] = len(samples)
        edges["n_missing"] = (len(allowed) - len(samples)) if allowed is not None else 0
        if cfg.get("_sample_universe_sha256"):
            edges["sample_universe_sha256"] = cfg["_sample_universe_sha256"]
        for key, value in cfg.get("_formal_lineage", {}).items():
            edges[key] = value
        edges["edge_id"] = [stable_id("COEX", cancer, l, g) for l, g in zip(edges.lncrna_id, edges.gene_id)]
        edges["method"] = "covariate_residual_spearman_effective_df"
        write_table(edges.drop(columns="abs_rho"), out_file)
        summaries.append({"cancer_id": cancer, "status": "PASS", "n_samples": len(samples), "n_lncRNAs": lnc.shape[1], "n_genes": gene.shape[1], "n_edges": len(edges)})
    summary = pd.DataFrame(summaries)
    write_table(summary, cfg["_results"] / "tables" / "lnc_gene_coexpression_summary.tsv")
    return summary


def build_pathway_state_edges(cfg: dict[str, Any], cancers: Iterable[str] | None = None) -> pd.DataFrame:
    settings = cfg["pathway_state"]
    pathway_path = input_path(cfg, "bulk_pathway_activity")
    state_path = cfg["_standardized"] / "tumor_state_long.parquet"
    cov_path = input_path(cfg, "bulk_covariates")
    if not state_path.exists():
        raise FileNotFoundError(state_path)
    states_all = read_table(state_path)
    cov_all = read_table(cov_path)
    dim_cancer = read_table(cfg["_standardized"] / "dim_cancer.parquet")
    cancer_list = list(cancers) if cancers else dim_cancer.cancer_id.astype(str).tolist()
    out_dir = cfg["_results"] / "tables" / "pathway_state_edge"
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for cancer in cancer_list:
        out_file = out_dir / f"cancer_id={cancer}" / "part-0.parquet"
        if out_file.exists():
            summaries.append({"cancer_id": cancer, "status": "CACHED"})
            continue
        activity = read_cancer_partition(pathway_path, cancer)
        if activity.empty:
            summaries.append({"cancer_id": cancer, "status": "MISSING_ACTIVITY", "n_edges": 0})
            continue
        value_col = first_existing_column(activity, cfg["column_aliases"]["pathway_value"])
        activity = activity.pivot_table(index="sample_id", columns="pathway_id", values=value_col, aggfunc="mean")
        state = states_all.loc[states_all.cancer_id.astype(str) == cancer].pivot_table(index="sample_id", columns="state_id", values="state_value", aggfunc="mean")
        # Tumor-state table uses 15-char sample codes (TCGA-XX-XXXX-XX) while
        # bulk activity uses full aliquot barcodes; align on the 15-char code.
        activity.index = activity.index.astype(str).str[:15]
        activity = activity.groupby(level=0).mean()
        state.index = state.index.astype(str).str[:15]
        state = state.groupby(level=0).mean()
        samples = sorted(set(activity.index) & set(state.index))
        if len(samples) < settings["min_samples"]:
            summaries.append({"cancer_id": cancer, "status": "INSUFFICIENT_SAMPLES", "n_samples": len(samples), "n_edges": 0})
            continue
        activity = activity.reindex(samples)
        state = state.reindex(samples)
        state = state.loc[:, state.notna().sum() >= settings["min_samples"]]
        state = state.loc[:, state.var(ddof=1).sort_values(ascending=False).head(settings["max_states"]).index]
        a = standardize(rank_transform(activity.fillna(activity.median()).to_numpy(float)))
        s = standardize(rank_transform(state.fillna(state.median()).to_numpy(float)))
        corr = (a.T @ s) / max(len(samples) - 1, 1)
        p = correlation_p_values(corr, len(samples))
        total_tests = corr.size
        mask = np.abs(corr) >= settings["min_abs_effect"]
        i, j = np.where(mask)
        if len(i) == 0:
            summaries.append({"cancer_id": cancer, "status": "NO_SIGNIFICANT_EDGES", "n_edges": 0})
            continue
        out = pd.DataFrame({
            "pathway_id": activity.columns.to_numpy()[i],
            "state_id": state.columns.to_numpy()[j],
            "effect": corr[i, j],
            "p_value": p[i, j],
        })
        out["fdr"] = bh_fdr(out.p_value, total_tests=total_tests)
        out = out.loc[out.fdr <= settings["max_fdr"]]
        out["cancer_id"] = cancer
        out["n_samples"] = len(samples)
        out["direction"] = np.where(out.effect >= 0, "positive", "negative")
        out["edge_id"] = [stable_id("PSTATE", cancer, p, s) for p, s in zip(out.pathway_id, out.state_id)]
        write_table(out, out_file)
        summaries.append({"cancer_id": cancer, "status": "PASS", "n_samples": len(samples), "n_edges": len(out)})
    summary = pd.DataFrame(summaries)
    write_table(summary, cfg["_results"] / "tables" / "pathway_state_summary.tsv")
    return summary


def build_ucell_support(cfg: dict[str, Any]) -> pd.DataFrame:
    path = cfg["_standardized"] / "ucell_aggregate.parquet"
    if not path.exists():
        empty = pd.DataFrame(columns=["cancer_id", "pathway_id", "ucell_effect", "p_value", "fdr", "n_patients", "ucell_available"])
        write_table(empty, cfg["_results"] / "tables" / "ucell_pathway_support.parquet")
        return empty
    raw = read_table(path)
    settings = cfg["ucell"]
    rows = []
    grouped = raw.dropna(subset=["patient_id", "pathway_id", "mean_ucell", "mean_pseudotime"]).groupby(["cancer_id", "pathway_id"], observed=True)
    for (cancer, pathway), group in grouped:
        patient_r = []
        weights = []
        for patient, pg in group.groupby("patient_id", observed=True):
            if len(pg) < settings["min_bins_per_patient"] or pg.mean_pseudotime.nunique() < 3:
                continue
            r, _ = stats.spearmanr(pg.mean_pseudotime, pg.mean_ucell, nan_policy="omit")
            if np.isfinite(r):
                patient_r.append(r)
                weights.append(max(len(pg) - 3, 1))
        if len(patient_r) < settings["min_patients"]:
            continue
        effect, p_value = fisher_meta_correlations(np.asarray(patient_r), np.asarray(weights))
        rows.append({"cancer_id": cancer, "pathway_id": pathway, "ucell_effect": effect, "p_value": p_value, "n_patients": len(patient_r)})
    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(columns=["cancer_id", "pathway_id", "ucell_effect", "p_value", "fdr", "n_patients", "direction", "ucell_available"])
    else:
        out["fdr"] = out.groupby("cancer_id", observed=True).p_value.transform(lambda x: bh_fdr(x))
        out["direction"] = np.where(out.ucell_effect >= 0, "positive", "negative")
        out["ucell_available"] = True
        out["ucell_support"] = np.where((out.fdr <= settings["max_fdr"]) & (out.ucell_effect.abs() >= settings["min_abs_effect"]), np.minimum(out.ucell_effect.abs() / 0.5, 1.0) * np.minimum(-np.log10(out.fdr.clip(1e-12)) / 6, 1.0), 0.0)
    write_table(out, cfg["_results"] / "tables" / "ucell_pathway_support.parquet")
    return out
