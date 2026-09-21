from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import read_table, stable_id, write_json, write_table
from .interaction_context import apply_strict_cancer_context


REQUIRED_PROTEIN_ASSETS = {
    "protein_gene": "protein_gene_map.parquet",
    "lnc_protein_validated": "lncRNA_protein_relation.parquet",
    "string_protein": "string_protein_edge.parquet",
    "drug_protein": "drug_protein_target.parquet",
}


def require_protein_assets(standardized: Path) -> dict[str, pd.DataFrame]:
    missing = [str(standardized / name) for name in REQUIRED_PROTEIN_ASSETS.values() if not (standardized / name).exists()]
    if missing:
        raise FileNotFoundError(
            "The validated V2.4 canonical protein layer is required before graph build; "
            f"missing={missing}"
        )
    assets = {key: read_table(standardized / name) for key, name in REQUIRED_PROTEIN_ASSETS.items()}
    endpoints = pd.concat(
        [
            assets["protein_gene"].protein_id,
            assets["lnc_protein_validated"].protein_id,
            assets["string_protein"].protein_id_a,
            assets["string_protein"].protein_id_b,
            assets["drug_protein"].protein_id,
        ],
        ignore_index=True,
    ).dropna().astype(str)
    invalid = endpoints.loc[~endpoints.str.startswith("UNIPROT:")]
    if len(invalid):
        raise ValueError(f"Non-canonical protein endpoints found: {invalid.head(10).tolist()}")
    return assets


def _normalize_uniprot(values: pd.Series) -> pd.Series:
    out = values.astype("string").str.strip()
    out = out.str.replace(r"^UNIPROT:", "", regex=True, case=False)
    out = out.str.split(r"[;,\s|]+", regex=True).str[0].str.strip()
    return out.mask(out.isna() | out.str.lower().isin({"", "nan", "none", "<na>"}))


def build_contextual_lnc_protein(
    standardized: Path,
    tables: Path,
    relation: pd.DataFrame,
    dim_lnc: pd.DataFrame,
    dim_cancer: pd.DataFrame,
    protein_gene: pd.DataFrame,
    validated: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rebuild canonical lncRNA-protein edges while retaining strict context."""
    is_protein = relation.partner_type.astype(str).str.lower().str.contains("protein")
    rel = relation.loc[is_protein].copy()
    rel["lncrna_id"] = rel.lncrna_id.astype("string").str.strip()
    rel["original_partner_id"] = rel.partner_id.astype("string").str.strip()
    valid_lnc = set(dim_lnc.lncrna_id.dropna().astype(str))
    rel = rel.loc[
        rel.lncrna_id.isin(valid_lnc)
        & rel.original_partner_id.notna()
        & rel.original_partner_id.ne("")
    ].copy()
    rel["_row_id"] = np.arange(len(rel), dtype=np.int64)
    rel, context_audit = apply_strict_cancer_context(rel, dim_cancer)

    mapping = protein_gene[
        ["gene_id", "protein_id", "mapping_multiplicity", "mapping_weight"]
    ].drop_duplicates(["gene_id", "protein_id"])
    by_gene = rel.loc[rel.original_partner_id.str.startswith("GENE:", na=False)].merge(
        mapping,
        left_on="original_partner_id",
        right_on="gene_id",
        how="left",
    )
    direct = rel.loc[~rel.original_partner_id.str.startswith("GENE:", na=False)].copy()
    direct["uniprot_accession"] = _normalize_uniprot(direct.original_partner_id)
    direct = direct.merge(
        protein_gene[
            ["uniprot_accession", "protein_id", "gene_id", "mapping_multiplicity", "mapping_weight"]
        ].drop_duplicates("uniprot_accession"),
        on="uniprot_accession",
        how="left",
    )
    mapped = pd.concat([by_gene, direct], ignore_index=True, sort=False)
    rejected_mapping = int(mapped.protein_id.isna().sum())
    mapped = mapped.dropna(subset=["protein_id"]).copy()
    experimental = mapped.get("is_experimental", pd.Series(False, index=mapped.index)).fillna(False)
    mapped["weight"] = np.where(experimental, 1.0, 0.4) * mapped.mapping_weight.fillna(1.0)
    mapped["source_database"] = mapped.source_database.fillna("unknown").astype(str)
    mapped["pmid_present"] = mapped.get("pmid", pd.Series(pd.NA, index=mapped.index)).notna().astype(int)

    grouped = (
        mapped.groupby(["lncrna_id", "protein_id", "cancer_id"], as_index=False, observed=True, dropna=False)
        .agg(
            weight=("weight", "max"),
            source_database=("source_database", lambda values: "|".join(sorted(set(map(str, values))))),
            n_source_records=("_row_id", "nunique"),
            n_pmids=("pmid_present", "sum"),
            mapping_multiplicity=("mapping_multiplicity", "max"),
        )
        .sort_values(["lncrna_id", "protein_id", "cancer_id"], na_position="first")
    )
    grouped["relation_type"] = "binds_protein"
    grouped["is_context_specific"] = grouped.cancer_id.notna()
    grouped["mapping_status"] = np.where(
        grouped.mapping_multiplicity.fillna(1).gt(1), "mapped_expanded", "mapped_unique"
    )
    grouped["edge_id"] = [
        stable_id("LNCPROTCTX", lnc, protein, cancer)
        for lnc, protein, cancer in zip(grouped.lncrna_id, grouped.protein_id, grouped.cancer_id)
    ]

    validated_pairs = set(zip(validated.lncrna_id.astype(str), validated.protein_id.astype(str)))
    observed_pairs = set(zip(grouped.lncrna_id.astype(str), grouped.protein_id.astype(str)))
    unexpected_pairs = observed_pairs - validated_pairs
    if unexpected_pairs:
        raise RuntimeError(
            "Context-preserving protein mapping diverged from validated V2.4 assets; "
            f"unexpected_examples={list(sorted(unexpected_pairs))[:10]}"
        )
    output = standardized / "lncRNA_protein_relation_context.parquet"
    write_table(grouped, output)
    audit = {
        "status": "PASS",
        **context_audit,
        "protein_mapping_rejected_rows": rejected_mapping,
        "validated_v2_4_pairs": len(validated_pairs),
        "retained_unique_pairs": len(observed_pairs),
        "retained_edges_with_context_expansion": int(len(grouped)),
        "global_edges": int(grouped.cancer_id.isna().sum()),
        "context_specific_edges": int(grouped.cancer_id.notna().sum()),
        "unexpected_pairs_vs_v2_4": 0,
        "output": str(output),
    }
    write_json(audit, tables / "protein_context_audit.json")
    return grouped, audit
