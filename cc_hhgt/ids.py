from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import LOGGER, first_existing_column, input_path, read_table, stable_id, write_table


def build_symbol_crosswalk(dim_gene: pd.DataFrame, dim_lnc: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for df, id_col, entity_type in [(dim_gene, "gene_id", "gene"), (dim_lnc, "lncrna_id", "lncRNA")]:
        base = df[[id_col, "gene_symbol", "ensembl_gene_id", "hgnc_id", "aliases"]].copy()
        base = base.rename(columns={id_col: "canonical_id"})
        for source_col, source_type in [("gene_symbol", "symbol"), ("ensembl_gene_id", "ensembl"), ("hgnc_id", "hgnc")]:
            part = base[["canonical_id", source_col]].dropna().rename(columns={source_col: "source_id"})
            part["source_id"] = part.source_id.astype(str).str.strip()
            part["source_id_norm"] = part.source_id.str.upper()
            part["source_type"] = source_type
            part["entity_type"] = entity_type
            rows.append(part)
        alias_rows = []
        for row in base[["canonical_id", "aliases"]].dropna().itertuples(index=False):
            for alias in str(row.aliases).replace("|", ";").replace(",", ";").split(";"):
                alias = alias.strip()
                if alias:
                    alias_rows.append({"canonical_id": row.canonical_id, "source_id": alias, "source_id_norm": alias.upper(), "source_type": "alias", "entity_type": entity_type})
        if alias_rows:
            rows.append(pd.DataFrame(alias_rows))
    out = pd.concat(rows, ignore_index=True).drop_duplicates()
    counts = out.groupby(["source_id_norm", "entity_type"]).canonical_id.transform("nunique")
    out["mapping_status"] = np.where(counts == 1, "unique", "ambiguous")
    return out


def build_uniprot_gene_map(cfg: dict[str, Any], dim_gene: pd.DataFrame) -> pd.DataFrame:
    path = input_path(cfg, "uniprot_mapping")
    if not path or not path.exists():
        return pd.DataFrame(columns=["protein_id", "gene_id", "uniprot_accession", "mapping_confidence"])
    raw = read_table(path)
    accession = first_existing_column(raw, ["uniprot_accession", "accession", "Entry", "UniProtKB-AC", "From"], required=False)
    gene_symbol = first_existing_column(raw, ["gene_symbol", "Gene Names (primary)", "gene", "GENE", "Gene_Name"], required=False)
    ensembl = first_existing_column(raw, ["ensembl_gene_id", "Ensembl", "To"], required=False)
    if not accession:
        LOGGER.warning("UniProt mapping lacks a recognizable accession column")
        return pd.DataFrame(columns=["protein_id", "gene_id", "uniprot_accession", "mapping_confidence"])
    symbol_map = dim_gene[["gene_id", "gene_symbol", "ensembl_gene_id"]].copy()
    symbol_map["symbol_norm"] = symbol_map.gene_symbol.astype(str).str.upper()
    frames = []
    if gene_symbol:
        x = raw[[accession, gene_symbol]].dropna().copy()
        x["symbol_norm"] = x[gene_symbol].astype(str).str.split().str[0].str.upper()
        x = x.merge(symbol_map[["gene_id", "symbol_norm"]].drop_duplicates("symbol_norm"), on="symbol_norm", how="left")
        x["mapping_confidence"] = "gene_symbol"
        frames.append(x.rename(columns={accession: "uniprot_accession"})[["uniprot_accession", "gene_id", "mapping_confidence"]])
    if ensembl:
        x = raw[[accession, ensembl]].dropna().copy()
        x["ensembl_norm"] = x[ensembl].astype(str).str.replace(r"\.\d+$", "", regex=True)
        gm = symbol_map.assign(ensembl_norm=symbol_map.ensembl_gene_id.astype(str).str.replace(r"\.\d+$", "", regex=True))
        x = x.merge(gm[["gene_id", "ensembl_norm"]].drop_duplicates("ensembl_norm"), on="ensembl_norm", how="left")
        x["mapping_confidence"] = "ensembl"
        frames.append(x.rename(columns={accession: "uniprot_accession"})[["uniprot_accession", "gene_id", "mapping_confidence"]])
    if not frames:
        return pd.DataFrame(columns=["protein_id", "gene_id", "uniprot_accession", "mapping_confidence"])
    out = pd.concat(frames, ignore_index=True).dropna(subset=["gene_id", "uniprot_accession"])
    out["uniprot_accession"] = out.uniprot_accession.astype(str).str.split(";").str[0].str.strip()
    out["protein_id"] = "UNIPROT:" + out.uniprot_accession
    priority = {"ensembl": 0, "gene_symbol": 1}
    out["_priority"] = out.mapping_confidence.map(priority).fillna(9)
    out = out.sort_values("_priority").drop_duplicates(["protein_id", "gene_id"]).drop(columns="_priority")
    return out


def build_drug_gene_target(cfg: dict[str, Any], dim_gene: pd.DataFrame, dim_drug: pd.DataFrame) -> pd.DataFrame:
    existing = input_path(cfg, "drug_target_relation")
    if existing and existing.exists():
        df = read_table(existing)
        if "gene_id" not in df.columns and "target_gene_id" in df.columns:
            df = df.rename(columns={"target_gene_id": "gene_id"})
        return df.drop_duplicates()
    path = input_path(cfg, "drugcentral_targets")
    if not path or not path.exists():
        return pd.DataFrame()
    raw = read_table(path)
    drug_name = first_existing_column(raw, ["drug_name", "DRUG_NAME", "STRUCT_ID", "drug"], required=False)
    gene_symbol = first_existing_column(raw, ["gene_symbol", "GENE", "TARGET_GENE", "gene"], required=False)
    accession = first_existing_column(raw, ["accession", "ACCESSION", "uniprot_accession"], required=False)
    action = first_existing_column(raw, ["action_type", "ACTION_TYPE", "MOA"], required=False)
    if not drug_name or not gene_symbol:
        return pd.DataFrame()
    dg = dim_gene[["gene_id", "gene_symbol"]].copy()
    dg["gene_symbol_norm"] = dg.gene_symbol.astype(str).str.upper()
    dd = dim_drug[["drug_id", "drug_name", "normalized_name"]].copy()
    dd["drug_norm"] = dd.normalized_name.fillna(dd.drug_name).astype(str).str.lower().str.replace(r"[^a-z0-9]+", "", regex=True)
    out = raw.copy()
    out["gene_symbol_norm"] = out[gene_symbol].astype(str).str.upper()
    out["drug_norm"] = out[drug_name].astype(str).str.lower().str.replace(r"[^a-z0-9]+", "", regex=True)
    out = out.merge(dg[["gene_id", "gene_symbol_norm"]].drop_duplicates("gene_symbol_norm"), on="gene_symbol_norm", how="left")
    out = out.merge(dd[["drug_id", "drug_norm"]].drop_duplicates("drug_norm"), on="drug_norm", how="left")
    out["source_drug_name"] = out[drug_name].astype(str)
    out["source_gene_symbol"] = out[gene_symbol].astype(str)
    out["uniprot_accession"] = out[accession].astype(str) if accession else pd.NA
    out["action_type"] = out[action].astype(str) if action else pd.NA
    out["mapping_status"] = np.select([out.drug_id.notna() & out.gene_id.notna(), out.drug_id.notna(), out.gene_id.notna()], ["mapped_both", "drug_only", "gene_only"], default="unmapped")
    out["drug_target_id"] = [stable_id("DT", a, b, c) for a, b, c in zip(out.drug_id, out.gene_id, out.action_type)]
    return out[["drug_target_id", "drug_id", "gene_id", "uniprot_accession", "action_type", "source_drug_name", "source_gene_symbol", "mapping_status"]].drop_duplicates("drug_target_id")


def build_string_edges(cfg: dict[str, Any], uniprot_gene: pd.DataFrame, dim_gene: pd.DataFrame) -> pd.DataFrame:
    path = input_path(cfg, "string_physical")
    if not path or not path.exists():
        return pd.DataFrame()
    raw = read_table(path)
    a = first_existing_column(raw, ["protein1", "protein_a", "string_protein_id_a", "item_id_a", "node1"], required=False)
    b = first_existing_column(raw, ["protein2", "protein_b", "string_protein_id_b", "item_id_b", "node2"], required=False)
    score = first_existing_column(raw, ["combined_score", "physical_score", "score"], required=False)
    if not a or not b:
        return pd.DataFrame()
    aliases_path = input_path(cfg, "string_aliases")
    alias_map = pd.DataFrame()
    if aliases_path and aliases_path.exists():
        aliases = read_table(aliases_path)
        sid = first_existing_column(aliases, ["string_protein_id", "#string_protein_id", "protein_id"], required=False)
        alias = first_existing_column(aliases, ["alias", "source_id"], required=False)
        source = first_existing_column(aliases, ["source", "alias_source"], required=False)
        if sid and alias:
            alias_map = aliases[[sid, alias] + ([source] if source else [])].copy()
            alias_map.columns = ["string_id", "alias"] + (["source"] if source else [])
            alias_map["alias_norm"] = alias_map.alias.astype(str).str.replace(r"\.\d+$", "", regex=True).str.upper()
            gm = dim_gene[["gene_id", "gene_symbol", "ensembl_gene_id"]].copy()
            gm_long = pd.concat([
                gm[["gene_id", "gene_symbol"]].rename(columns={"gene_symbol": "alias_norm"}),
                gm[["gene_id", "ensembl_gene_id"]].rename(columns={"ensembl_gene_id": "alias_norm"}),
            ])
            gm_long["alias_norm"] = gm_long.alias_norm.astype(str).str.replace(r"\.\d+$", "", regex=True).str.upper()
            alias_map = alias_map.merge(gm_long.drop_duplicates("alias_norm"), on="alias_norm", how="left").dropna(subset=["gene_id"])
            alias_map = alias_map.sort_values("gene_id").drop_duplicates("string_id")
    if alias_map.empty:
        return pd.DataFrame()
    mapping = alias_map.set_index("string_id").gene_id
    out = pd.DataFrame({
        "gene_id_a": raw[a].astype(str).map(mapping),
        "gene_id_b": raw[b].astype(str).map(mapping),
        "physical_score": pd.to_numeric(raw[score], errors="coerce") if score else 1.0,
    }).dropna(subset=["gene_id_a", "gene_id_b"])
    if out.physical_score.max() > 1:
        out["physical_score"] = out.physical_score / 1000.0
    out = out.loc[out.gene_id_a != out.gene_id_b]
    out["ppi_edge_id"] = [stable_id("PPI", *sorted([a, b])) for a, b in zip(out.gene_id_a, out.gene_id_b)]
    return out.sort_values("physical_score", ascending=False).drop_duplicates("ppi_edge_id")


def build_crosswalks(cfg: dict[str, Any]) -> dict[str, int]:
    std = cfg["_standardized"]
    dim_gene = read_table(std / "dim_gene.parquet")
    dim_lnc = read_table(std / "dim_lncRNA.parquet")
    dim_drug = read_table(std / "dim_drug.parquet")
    symbol = build_symbol_crosswalk(dim_gene, dim_lnc)
    write_table(symbol, std / "entity_id_crosswalk.parquet")
    gp = build_uniprot_gene_map(cfg, dim_gene)
    write_table(gp, std / "gene_protein_map.parquet")
    dt = build_drug_gene_target(cfg, dim_gene, dim_drug)
    write_table(dt, std / "drug_gene_target.parquet")
    ppi = build_string_edges(cfg, gp, dim_gene)
    write_table(ppi, std / "string_gene_edge.parquet")
    return {"entity_crosswalk": len(symbol), "gene_protein_map": len(gp), "drug_gene_target": len(dt), "string_gene_edge": len(ppi)}
