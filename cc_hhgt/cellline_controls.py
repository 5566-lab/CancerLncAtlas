from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .common import LOGGER, input_path, read_table, write_table
from .stats import bh_fdr, correlation_p_values, rank_transform, standardize

MODEL_PATTERN = re.compile(r"^(ACH-|SIDM|SANGER|MODEL|CL-)", re.I)


def clean_gene_symbol(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"\s*\([^)]*\)\s*$", "", text)
    text = re.sub(r"\.\d+$", "", text)
    return text.upper()


def _looks_like_model(values: pd.Series) -> float:
    sample = values.dropna().astype(str).head(100)
    if sample.empty:
        return 0.0
    return float(sample.map(lambda x: bool(MODEL_PATTERN.search(x)) or x.startswith("ACH-")).mean())


def read_expression_matrix(path: Path, desired_symbols: set[str] | None = None) -> pd.DataFrame:
    """Return model x gene matrix from either model-row or gene-row CSV/TSV."""
    sep = "\t" if any(str(path).endswith(x) for x in [".tsv", ".tsv.gz", ".txt", ".txt.gz"]) else ","
    preview = pd.read_csv(path, sep=sep, nrows=8, low_memory=False)
    first = preview.columns[0]
    row_model_score = _looks_like_model(preview[first])
    col_model_score = float(np.mean([bool(MODEL_PATTERN.search(str(x))) or str(x).startswith("ACH-") for x in preview.columns[1:101]])) if len(preview.columns) > 1 else 0.0
    if row_model_score >= col_model_score:
        header = list(preview.columns)
        selected = header
        if desired_symbols:
            selected = [first] + [c for c in header[1:] if clean_gene_symbol(c) in desired_symbols]
        frame = pd.read_csv(path, sep=sep, usecols=selected, low_memory=False)
        frame = frame.set_index(first)
        frame.columns = [clean_gene_symbol(x) for x in frame.columns]
    else:
        chunks = []
        for chunk in pd.read_csv(path, sep=sep, chunksize=2000, low_memory=False):
            chunk[first] = chunk[first].map(clean_gene_symbol)
            if desired_symbols:
                chunk = chunk.loc[chunk[first].isin(desired_symbols)]
            chunks.append(chunk)
        genes = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        if genes.empty:
            return pd.DataFrame()
        frame = genes.set_index(first).T
        frame.columns = [clean_gene_symbol(x) for x in frame.columns]
    frame.index = frame.index.astype(str).str.strip()
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.loc[:, ~frame.columns.duplicated()].dropna(axis=0, how="all")
    return frame


def normalized_rank_pathway_family_activity(depmap: pd.DataFrame, pathway_member: pd.DataFrame, family_member: pd.DataFrame) -> pd.DataFrame:
    symbols = set(depmap.columns)
    member = pathway_member.copy()
    member["gene_symbol_norm"] = member.gene_symbol.map(clean_gene_symbol)
    member = member.loc[member.gene_symbol_norm.isin(symbols)]
    if member.empty:
        return pd.DataFrame(index=depmap.index)
    ranks = depmap.rank(axis=1, method="average", pct=True)
    pathway_scores = {}
    for pathway, group in member.groupby("pathway_id", observed=True):
        cols = [x for x in group.gene_symbol_norm.unique() if x in ranks.columns]
        if len(cols) >= 5:
            pathway_scores[str(pathway)] = ranks[cols].mean(axis=1) - 0.5
    if not pathway_scores:
        return pd.DataFrame(index=depmap.index)
    pmat = pd.DataFrame(pathway_scores, index=depmap.index)
    fm = family_member.merge(pd.DataFrame({"pathway_id": pmat.columns}), on="pathway_id")
    family_scores = {}
    for family, group in fm.groupby("pathway_family_id", observed=True):
        cols = [x for x in group.pathway_id.astype(str).unique() if x in pmat.columns]
        if cols:
            weights = group.drop_duplicates("pathway_id").set_index("pathway_id").reindex(cols).membership_weight.fillna(1.0).to_numpy(float)
            family_scores[str(family)] = np.average(pmat[cols].to_numpy(float), axis=1, weights=weights)
    return pd.DataFrame(family_scores, index=depmap.index)


def _model_crosswalk(cfg: dict[str, Any]) -> pd.DataFrame:
    path = cfg["_standardized"] / "cell_model_crosswalk.parquet"
    return read_table(path) if path.exists() else pd.DataFrame()


def align_models(lnc: pd.DataFrame, depmap: pd.DataFrame, crosswalk: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    if crosswalk.empty:
        common = sorted(set(lnc.index) & set(depmap.index))
        return lnc.reindex(common), depmap.reindex(common), pd.Series("PAN_CELL_LINE", index=common)
    dep_col = "depmap_model_id" if "depmap_model_id" in crosswalk.columns else None
    sanger_col = "sanger_model_id" if "sanger_model_id" in crosswalk.columns else None
    lineage_col = "lineage" if "lineage" in crosswalk.columns else None
    mapping = {}
    lineage = {}
    for row in crosswalk.itertuples(index=False):
        dep = str(getattr(row, dep_col)) if dep_col and pd.notna(getattr(row, dep_col)) else None
        san = str(getattr(row, sanger_col)) if sanger_col and pd.notna(getattr(row, sanger_col)) else None
        lin = str(getattr(row, lineage_col)) if lineage_col and pd.notna(getattr(row, lineage_col)) else "PAN_CELL_LINE"
        if dep:
            mapping[dep] = dep; lineage[dep] = lin
        if san and dep:
            mapping[san] = dep
    lnc = lnc.copy(); lnc["_depmap_id"] = [mapping.get(str(x), str(x)) for x in lnc.index]
    lnc = lnc.groupby("_depmap_id").mean(numeric_only=True)
    common = sorted(set(lnc.index) & set(depmap.index))
    return lnc.reindex(common), depmap.reindex(common), pd.Series({x: lineage.get(x, "PAN_CELL_LINE") for x in common})


def build_cellline_context_support(cfg: dict[str, Any]) -> pd.DataFrame:
    output = cfg["_results"] / "tables" / "cellline_lnc_family_support.parquet"
    if output.exists():
        return read_table(output)
    settings = cfg.get("cellline_context", {})
    if not settings.get("enabled", True):
        return pd.DataFrame()
    lnc_path = input_path(cfg, "cell_model_passports_lnc_expression")
    depmap_path = input_path(cfg, "depmap_protein_expression")
    if not lnc_path or not depmap_path or not lnc_path.exists() or not depmap_path.exists():
        LOGGER.warning("Cell-line raw expression inputs are incomplete; using precomputed drug associations only")
        return pd.DataFrame()
    dim_lnc = read_table(cfg["_standardized"] / "dim_lncRNA.parquet")
    pathway_member = read_table(cfg["_standardized"] / "pathway_gene_member.parquet")
    family_member = read_table(cfg["_results"] / "tables" / "pathway_family_member.parquet")
    lnc_symbols = set(dim_lnc.gene_symbol.dropna().map(clean_gene_symbol))
    gene_symbols = set(pathway_member.gene_symbol.dropna().map(clean_gene_symbol))
    LOGGER.info("Reading Cell Model Passports lncRNA expression")
    lnc = read_expression_matrix(lnc_path, desired_symbols=lnc_symbols)
    LOGGER.info("Reading DepMap protein-coding expression")
    depmap = read_expression_matrix(depmap_path, desired_symbols=gene_symbols)
    if lnc.empty or depmap.empty:
        return pd.DataFrame()
    lnc, depmap, lineage = align_models(lnc, depmap, _model_crosswalk(cfg))
    dim_cancer = read_table(cfg["_standardized"] / "dim_cancer.parquet")
    alias_map = {}
    for row in dim_cancer.itertuples(index=False):
        cancer_id = str(row.cancer_id)
        values = [cancer_id, getattr(row, "tcga_code", None), getattr(row, "english_name", None), getattr(row, "synonyms", None)]
        for value in values:
            if value is None or pd.isna(value):
                continue
            for token in re.split(r"[;|,]", str(value)):
                norm = re.sub(r"[^A-Z0-9]+", "", token.upper())
                if norm:
                    alias_map[norm] = cancer_id
    lineage = lineage.astype(str).map(lambda x: alias_map.get(re.sub(r"[^A-Z0-9]+", "", x.upper()), x))
    if len(lnc) < settings["min_models"]:
        return pd.DataFrame()
    symbol_to_id = dim_lnc.assign(symbol_norm=dim_lnc.gene_symbol.map(clean_gene_symbol)).drop_duplicates("symbol_norm").set_index("symbol_norm").lncrna_id
    lnc = lnc.rename(columns={c: symbol_to_id.get(clean_gene_symbol(c), c) for c in lnc.columns})
    lnc = lnc.loc[:, lnc.columns.isin(set(dim_lnc.lncrna_id.astype(str)))]
    if lnc.columns.duplicated().any():
        lnc = lnc.T.groupby(level=0).mean().T
    variance = lnc.var(axis=0, ddof=1).sort_values(ascending=False)
    lnc = lnc.loc[:, variance.head(settings["max_lncRNAs"]).index]
    family_activity = normalized_rank_pathway_family_activity(depmap, pathway_member, family_member)
    common = sorted(set(lnc.index) & set(family_activity.index))
    lnc = lnc.reindex(common).fillna(lnc.median())
    family_activity = family_activity.reindex(common).fillna(family_activity.median())
    lineage = lineage.reindex(common).fillna("PAN_CELL_LINE")
    contexts = {"PAN_CELL_LINE": np.ones(len(common), dtype=bool)}
    if settings.get("compute_per_lineage", True):
        for value, count in lineage.value_counts().items():
            if count >= settings["min_models"]:
                contexts[str(value)] = lineage.astype(str).eq(str(value)).to_numpy()
    rows = []
    block = int(settings["block_size"])
    for context, mask in contexts.items():
        n = int(mask.sum())
        if n < settings["min_models"]:
            continue
        lx = standardize(rank_transform(lnc.loc[mask].to_numpy(float)))
        px = standardize(rank_transform(family_activity.loc[mask].to_numpy(float)))
        total_tests = lx.shape[1] * px.shape[1]
        for start in range(0, lx.shape[1], block):
            stop = min(start + block, lx.shape[1])
            corr = lx[:, start:stop].T @ px / max(n - 1, 1)
            p = correlation_p_values(corr, n)
            i, j = np.where(np.abs(corr) >= settings["min_abs_rho"])
            if len(i) == 0:
                continue
            part = pd.DataFrame({
                "cancer_id": context,
                "lncrna_id": lnc.columns.to_numpy()[start:stop][i],
                "pathway_family_id": family_activity.columns.to_numpy()[j],
                "cellline_rho": corr[i, j],
                "p_value": p[i, j],
                "n_cell_lines": n,
            })
            part["fdr"] = bh_fdr(part.p_value, total_tests=total_tests)
            rows.append(part.loc[part.fdr <= settings["max_fdr"]])
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not out.empty:
        out["cellline_support"] = np.clip(np.abs(out.cellline_rho) / 0.5, 0, 1) * np.clip(-np.log10(out.fdr.clip(1e-12)) / 6, 0, 1)
        out["cellline_direction"] = np.where(out.cellline_rho >= 0, "positive", "negative")
        out["cellline_available"] = True
        write_table(out, output)
    return out
