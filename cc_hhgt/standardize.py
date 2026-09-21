from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import LOGGER, first_existing_column, input_path, read_collection, read_table, resolve_path, stable_id, write_table
from .io import discover_preferred, read_glob_collection, read_preferred_collection


def _copy_small(cfg: dict[str, Any], key: str, output_name: str, dedupe: list[str] | None = None) -> pd.DataFrame:
    path = input_path(cfg, key)
    if not path or not path.exists():
        return pd.DataFrame()
    df = read_table(path)
    if dedupe:
        df = df.drop_duplicates(dedupe)
    write_table(df, cfg["_standardized"] / output_name)
    return df


def standardize_dimensions(cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    outputs = {}
    for key, name, pk in [
        ("dim_lncRNA", "dim_lncRNA.parquet", ["lncrna_id"]),
        ("dim_gene", "dim_gene.parquet", ["gene_id"]),
        ("dim_pathway", "dim_pathway.parquet", ["pathway_id"]),
        ("dim_cancer", "dim_cancer.parquet", ["cancer_id"]),
        ("dim_drug", "dim_drug.parquet", ["drug_id"]),
    ]:
        outputs[key] = _copy_small(cfg, key, name, pk)
    return outputs


def standardize_pathway_members(cfg: dict[str, Any], dim_gene: pd.DataFrame) -> pd.DataFrame:
    raw = read_table(input_path(cfg, "pathway_gene_member"))
    symbol_map = dim_gene[["gene_id", "gene_symbol"]].dropna().copy()
    symbol_map["gene_symbol_norm"] = symbol_map.gene_symbol.astype(str).str.upper()
    symbol_map = symbol_map.sort_values("gene_id").drop_duplicates("gene_symbol_norm")
    raw["gene_symbol_norm"] = raw.gene_symbol.astype(str).str.upper()
    out = raw.merge(symbol_map[["gene_id", "gene_symbol_norm"]], on="gene_symbol_norm", how="left")
    out["mapping_status"] = np.where(out.gene_id.notna(), "mapped", "unmapped")
    out = out[["pathway_id", "gene_id", "gene_symbol", "weight", "mapping_status"]].drop_duplicates(["pathway_id", "gene_id", "gene_symbol"])
    write_table(out, cfg["_standardized"] / "pathway_gene_member.parquet")
    return out


def standardize_interactions(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    relation = read_table(input_path(cfg, "interaction_relation"))
    relation["interaction_row_id"] = [stable_id("IRROW", x, y, i) for i, (x, y) in enumerate(zip(relation.interaction_id, relation.source_record_id))]
    relation = relation.drop_duplicates("interaction_row_id")
    write_table(relation, cfg["_standardized"] / "interaction_relation.parquet")

    event_path = input_path(cfg, "evidence_event")
    if event_path and event_path.exists():
        events = read_table(event_path).drop_duplicates("evidence_event_id")
        write_table(events, cfg["_standardized"] / "evidence_event.parquet")
    else:
        events = pd.DataFrame()

    support_path = input_path(cfg, "interaction_pathway_support")
    if support_path and support_path.exists():
        support = read_table(support_path)
        support = support.drop_duplicates(["lncrna_id", "pathway_id", "cancer_id", "support_type", "direction"])
        write_table(support, cfg["_standardized"] / "interaction_pathway_support.parquet")
    else:
        support = pd.DataFrame()
    return relation, events, support


def standardize_tumor_state(cfg: dict[str, Any]) -> pd.DataFrame:
    path = input_path(cfg, "tumor_state")
    if not path or not path.exists():
        return pd.DataFrame()
    raw = read_table(path)
    cancer_col = first_existing_column(raw, ["cancer_id", "cancer_type", "tcga_code"], required=False)
    sample_col = first_existing_column(raw, ["sample_id", "sample", "barcode"], required=True)
    id_cols = [sample_col]
    if cancer_col:
        id_cols.append(cancer_col)
    patient_col = first_existing_column(raw, ["patient_id", "patient", "case_id"], required=False)
    if patient_col:
        id_cols.append(patient_col)
    numeric = [c for c in raw.columns if c not in id_cols and pd.api.types.is_numeric_dtype(raw[c])]
    long = raw[id_cols + numeric].melt(id_vars=id_cols, var_name="state_id", value_name="state_value")
    rename = {sample_col: "sample_id"}
    if cancer_col:
        rename[cancer_col] = "cancer_id"
    if patient_col:
        rename[patient_col] = "patient_id"
    long = long.rename(columns=rename)
    if "cancer_id" not in long.columns:
        long["cancer_id"] = long.sample_id.astype(str).str.split("-").str[0]
    long["state_value"] = pd.to_numeric(long.state_value, errors="coerce")
    long = long.dropna(subset=["state_value"])
    long["state_type"] = "continuous"
    write_table(long, cfg["_standardized"] / "tumor_state_long.parquet")
    return long


def standardize_single_cell(cfg: dict[str, Any]) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    sc, source = read_preferred_collection(cfg, "sc_lnc_pathway_formal", "sc_lnc_pathway_staging_glob")
    if not sc.empty:
        sc["source_tier"] = source
        write_table(sc, cfg["_standardized"] / "sc_lnc_pathway.parquet")
    outputs["sc_lnc_pathway"] = {"rows": len(sc), "source": source}

    pt, pt_source = read_preferred_collection(cfg, "sc_pseudotime_pathway_formal", "sc_pseudotime_pathway_staging_glob")
    if not pt.empty:
        pt["source_tier"] = pt_source
        write_table(pt, cfg["_standardized"] / "sc_pseudotime_pathway.parquet")
    outputs["sc_pseudotime_pathway"] = {"rows": len(pt), "source": pt_source}

    ucell = read_glob_collection(cfg, "ucell_aggregate_glob")
    if not ucell.empty:
        write_table(ucell, cfg["_standardized"] / "ucell_aggregate.parquet")
    outputs["ucell_aggregate"] = {"rows": len(ucell), "cancers": int(ucell.cancer_id.nunique()) if "cancer_id" in ucell else 0}

    sc_summary = read_glob_collection(cfg, "sc_lnc_celltype_glob")
    if not sc_summary.empty:
        write_table(sc_summary, cfg["_standardized"] / "sc_lnc_celltype_summary.parquet")
    outputs["sc_lnc_celltype_summary"] = {"rows": len(sc_summary)}
    return outputs


def standardize_drug_tables(cfg: dict[str, Any]) -> dict[str, int]:
    output: dict[str, int] = {}
    curated_path = input_path(cfg, "lnc_drug_curated")
    if curated_path and curated_path.exists():
        curated = read_table(curated_path).drop_duplicates()
        curated["curated_evidence_id"] = [stable_id("CDE", *r) for r in curated.astype(str).itertuples(index=False, name=None)]
        write_table(curated, cfg["_standardized"] / "lncRNA_drug_curated_evidence.parquet")
        output["curated"] = len(curated)
    for key, out_name in [
        ("prism_lnc_drug", "prism_lnc_drug.parquet"),
        ("gdsc_lnc_drug", "gdsc_lnc_drug.parquet"),
        ("drug_replication", "drug_replication.parquet"),
    ]:
        path = input_path(cfg, key)
        if not path or not path.exists():
            output[key] = 0
            continue
        df = read_table(path)
        source_cols = [c for c in ["resource", "source_dataset", "source_drug_id", "response_metric", "observed_or_predicted"] if c in df.columns]
        key_cols = [c for c in ["cancer_id", "lncrna_id", "drug_id"] + source_cols if c in df.columns]
        df["association_id"] = [stable_id("DA", *r) for r in df[key_cols].astype(str).itertuples(index=False, name=None)]
        sort_cols = [c for c in ["fdr", "p_value", "best_gdsc_fdr", "best_gdsc_p_value"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols)
        df = df.drop_duplicates("association_id")
        write_table(df, cfg["_standardized"] / out_name)
        output[key] = len(df)
    return output
