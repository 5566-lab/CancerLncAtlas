from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from .cellline_controls import build_cellline_context_support
from .common import LOGGER, first_existing_column, input_path, read_table, stable_id, write_table
from .stats import minmax01, signed_support


def parquet_glob(path: Path) -> str:
    if path.is_dir():
        return str(path / "**" / "*.parquet")
    return str(path)



def expand_global_cancer(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    if df.empty or "cancer_id" not in df.columns:
        return df
    text = df.cancer_id.astype("string")
    global_mask = (text.isna() | text.str.strip().isin(["", "NA", "None", "nan", "PAN_CANCER", "ALL"])).fillna(True)
    if not global_mask.any():
        return df
    cancers = read_table(cfg["_standardized"] / "dim_cancer.parquet").cancer_id.astype(str).tolist()
    local = df.loc[~global_mask].copy()
    global_rows = df.loc[global_mask].drop(columns="cancer_id")
    expanded = []
    for cancer in cancers:
        part = global_rows.copy(); part["cancer_id"] = cancer; expanded.append(part)
    return pd.concat([local] + expanded, ignore_index=True)

def _direction_sign(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.lower()
    return np.select([text.str.contains("pos|up|sens|increase|promot"), text.str.contains("neg|down|resist|decrease|inhibit")], [1.0, -1.0], default=0.0)


def _aggregate_pathway_to_family(df: pd.DataFrame, family_member: pd.DataFrame, support_col: str, effect_col: str | None = None, fdr_col: str | None = None) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    joined = df.merge(family_member[["pathway_id", "pathway_family_id", "membership_weight"]], on="pathway_id", how="inner")
    joined["weighted_support"] = joined[support_col].fillna(0.0) * joined.membership_weight.fillna(1.0)
    group_cols = [c for c in ["cancer_id", "lncrna_id", "pathway_family_id"] if c in joined.columns]
    agg_spec = {support_col: ("weighted_support", "max"), "n_pathways": ("pathway_id", "nunique")}
    if effect_col:
        joined["weighted_effect"] = joined[effect_col].fillna(0.0) * joined.membership_weight.fillna(1.0)
        agg_spec[effect_col] = ("weighted_effect", lambda x: x.iloc[np.argmax(np.abs(x.to_numpy()))] if len(x) else np.nan)
    if fdr_col:
        agg_spec[fdr_col] = (fdr_col, "min")
    return joined.groupby(group_cols, as_index=False, observed=True).agg(**agg_spec)


def build_bulk_support(cfg: dict[str, Any], family_member: pd.DataFrame) -> pd.DataFrame:
    path = input_path(cfg, "bulk_lnc_pathway")
    settings = cfg["pair_evidence"]
    con = duckdb.connect()
    con.register("family_member", family_member[["pathway_id", "pathway_family_id", "membership_weight"]])
    sql = f"""
    SELECT b.cancer_id, b.lncrna_id, f.pathway_family_id,
           max(abs(b.rho_adjusted) * least(-log10(greatest(b.fdr_global, 1e-300))/10.0, 1.0) * coalesce(f.membership_weight,1.0)) AS bulk_support,
           arg_max(b.rho_adjusted, abs(b.rho_adjusted)) AS bulk_effect,
           min(b.fdr_global) AS bulk_fdr,
           count(DISTINCT b.pathway_id) AS bulk_n_pathways,
           max(CASE WHEN b.purity_adjusted AND b.cell_fraction_adjusted THEN 1 ELSE 0 END) AS bulk_adjusted
    FROM read_parquet('{parquet_glob(path)}', hive_partitioning=1) b
    JOIN family_member f USING(pathway_id)
    WHERE b.fdr_global <= {max(settings['bulk_max_fdr'], 0.1)}
      AND abs(b.rho_adjusted) >= {min(settings['min_abs_bulk_effect'], 0.15)}
    GROUP BY 1,2,3
    """
    out = con.execute(sql).df()
    con.close()
    if not out.empty:
        out["bulk_support"] = out.bulk_support.clip(0, 1)
        out["bulk_direction"] = np.where(out.bulk_effect >= 0, "positive", "negative")
        out["bulk_available"] = True
    return out


def build_sc_support(cfg: dict[str, Any], family_member: pd.DataFrame) -> pd.DataFrame:
    path = cfg["_standardized"] / "sc_lnc_pathway.parquet"
    if not path.exists():
        return pd.DataFrame()
    sc = read_table(path)
    effect_col = first_existing_column(sc, cfg["column_aliases"]["sc_effect"])
    fdr_col = first_existing_column(sc, cfg["column_aliases"]["sc_fdr"])
    sc[effect_col] = pd.to_numeric(sc[effect_col], errors="coerce")
    sc[fdr_col] = pd.to_numeric(sc[fdr_col], errors="coerce")
    sc["sc_ssgsea_support"] = signed_support(sc[effect_col], sc[fdr_col], effect_scale=0.4)
    if "n_patients" in sc.columns:
        sc["sc_ssgsea_support"] *= np.clip(np.log1p(sc.n_patients.fillna(0)) / np.log(11), 0, 1)
    out = _aggregate_pathway_to_family(sc, family_member, "sc_ssgsea_support", effect_col, fdr_col)
    if out.empty:
        return out
    out = out.rename(columns={effect_col: "sc_effect", fdr_col: "sc_fdr"})
    out["sc_direction"] = np.where(out.sc_effect >= 0, "positive", "negative")
    out["sc_available"] = True
    return out


def build_ucell_pair_support(cfg: dict[str, Any], sc_support: pd.DataFrame, family_member: pd.DataFrame) -> pd.DataFrame:
    path = cfg["_results"] / "tables" / "ucell_pathway_support.parquet"
    if not path.exists():
        return pd.DataFrame()
    uc = read_table(path)
    if uc.empty:
        return pd.DataFrame()
    family_uc = _aggregate_pathway_to_family(uc, family_member, "ucell_support", "ucell_effect", "fdr")
    if family_uc.empty:
        return family_uc
    family_uc = family_uc.rename(columns={"ucell_effect": "ucell_effect", "fdr": "ucell_fdr"})
    family_uc["ucell_direction"] = np.where(family_uc.ucell_effect >= 0, "positive", "negative")
    # UCell is pathway-level; it supports a pair only when the pair's single-cell direction is consistent.
    if sc_support.empty:
        return pd.DataFrame()
    keys = ["cancer_id", "pathway_family_id"]
    out = sc_support[["cancer_id", "lncrna_id", "pathway_family_id", "sc_direction"]].merge(family_uc, on=keys, how="inner")
    out["ucell_direction_match"] = out.sc_direction == out.ucell_direction
    out["ucell_support"] = out.ucell_support * np.where(out.ucell_direction_match, 1.0, 0.25)
    out["ucell_available"] = True
    return out[["cancer_id", "lncrna_id", "pathway_family_id", "ucell_support", "ucell_effect", "ucell_fdr", "ucell_direction_match", "ucell_available"]]



def build_bulk_sc_replication_support(cfg: dict[str, Any], family_member: pd.DataFrame) -> pd.DataFrame:
    path = input_path(cfg, "bulk_sc_replication")
    if not path or not path.exists():
        return pd.DataFrame()
    x = read_table(path)
    if "pathway_id" not in x.columns or "lncrna_id" not in x.columns or "cancer_id" not in x.columns:
        return pd.DataFrame()
    if "replication_status" in x.columns:
        status = x.replication_status.astype(str).str.lower()
        base = np.select([status.str.contains("replicat|confirmed|fdr"), status.str.contains("nominal|same_direction"), status.str.contains("discord|opposite")], [1.0, 0.65, 0.0], default=0.25)
    else:
        same = x.get("same_direction", False)
        base = np.where(pd.Series(same).fillna(False).astype(bool), 0.7, 0.0)
    if "n_sc_datasets" in x.columns:
        base = base * np.clip(np.log1p(pd.to_numeric(x.n_sc_datasets, errors="coerce").fillna(0)) / np.log(4), 0.25, 1.0)
    x["replication_support"] = base
    joined = x.merge(family_member[["pathway_id", "pathway_family_id", "membership_weight"]], on="pathway_id", how="inner")
    joined["replication_support"] *= joined.membership_weight.fillna(1.0)
    out = joined.groupby(["cancer_id", "lncrna_id", "pathway_family_id"], as_index=False, observed=True).agg(
        replication_support=("replication_support", "max"),
        replication_n_pathways=("pathway_id", "nunique"),
    )
    out["replication_available"] = True
    return out

def build_interaction_support(cfg: dict[str, Any], family_member: pd.DataFrame) -> pd.DataFrame:
    path = cfg["_standardized"] / "interaction_pathway_support.parquet"
    if not path.exists():
        return pd.DataFrame()
    x = expand_global_cancer(read_table(path), cfg)
    x["fdr"] = pd.to_numeric(x.fdr, errors="coerce")
    x["interaction_support"] = np.clip(np.log1p(x.n_independent_events.fillna(0)) / np.log(11), 0, 1) * np.clip(-np.log10(x.fdr.clip(1e-12)) / 8, 0, 1)
    level = x.highest_experiment_level.astype(str).str.lower()
    x["interaction_support"] *= np.select([level.str.contains("low|reporter|rescue|western|rip|clip"), level.str.contains("high")], [1.0, 0.75], default=0.6)
    x["perturbation_support"] = np.where(x.support_type.astype(str).str.lower().str.contains("perturb|knock|overexpress|rescue|functional"), x.interaction_support, 0.0)
    joined = x.merge(family_member[["pathway_id", "pathway_family_id", "membership_weight"]], on="pathway_id", how="inner")
    joined["interaction_weighted"] = joined.interaction_support * joined.membership_weight.fillna(1.0)
    joined["perturbation_weighted"] = joined.perturbation_support * joined.membership_weight.fillna(1.0)
    out = joined.groupby(["cancer_id", "lncrna_id", "pathway_family_id"], as_index=False, observed=True).agg(
        interaction_support=("interaction_weighted", "max"),
        perturbation_support=("perturbation_weighted", "max"),
        independent_pmids=("n_independent_pmids", "sum"),
        independent_events=("n_independent_events", "sum"),
        interaction_fdr=("fdr", "min"),
        interaction_n_pathways=("pathway_id", "nunique"),
    )
    out["interaction_available"] = True
    if "target_ids" in x.columns:
        source_relation = read_table(cfg["_standardized"] / "interaction_relation.parquet", columns=["lncrna_id", "partner_id", "source_database"]).drop_duplicates()
        lineage = x[["cancer_id", "lncrna_id", "pathway_id", "target_ids", "interaction_support", "perturbation_support"]].copy()
        lineage["partner_id"] = lineage.target_ids.fillna("").astype(str).str.replace(r"[\[\]\"']", "", regex=True).str.split(r"[;,|]")
        lineage = lineage.explode("partner_id")
        lineage["partner_id"] = lineage.partner_id.astype(str).str.strip()
        lineage = lineage.loc[lineage.partner_id.ne("")].merge(source_relation, on=["lncrna_id", "partner_id"], how="inner")
        lineage = lineage.merge(family_member[["pathway_id", "pathway_family_id", "membership_weight"]], on="pathway_id", how="inner")
        lineage["source_support"] = lineage.interaction_support * lineage.membership_weight.fillna(1.0)
        lineage["source_perturbation_support"] = lineage.perturbation_support * lineage.membership_weight.fillna(1.0)
        source_out = lineage.groupby(["cancer_id", "lncrna_id", "pathway_family_id", "source_database"], as_index=False, observed=True).agg(
            source_support=("source_support", "max"),
            source_perturbation_support=("source_perturbation_support", "max"),
            n_source_targets=("partner_id", "nunique"),
        )
        write_table(source_out, cfg["_results"] / "tables" / "interaction_pair_source_support.parquet")
    return out


def build_drug_family_map(cfg: dict[str, Any], family_member: pd.DataFrame) -> pd.DataFrame:
    dt_path = cfg["_standardized"] / "drug_gene_target.parquet"
    pgm_path = cfg["_standardized"] / "pathway_gene_member.parquet"
    if not dt_path.exists() or not pgm_path.exists():
        return pd.DataFrame()
    dt = read_table(dt_path).dropna(subset=["drug_id", "gene_id"])
    pgm = read_table(pgm_path).dropna(subset=["gene_id"])
    joined = dt[["drug_id", "gene_id"]].drop_duplicates().merge(pgm[["pathway_id", "gene_id"]].drop_duplicates(), on="gene_id").merge(family_member[["pathway_id", "pathway_family_id"]], on="pathway_id")
    counts = joined.groupby(["drug_id", "pathway_family_id"], as_index=False).agg(n_target_genes=("gene_id", "nunique"), n_pathways=("pathway_id", "nunique"))
    counts["drug_family_support"] = np.clip(np.log1p(counts.n_target_genes) / np.log(6), 0, 1)
    write_table(counts, cfg["_results"] / "tables" / "drug_pathway_family_support.parquet")
    return counts


def build_drug_support(cfg: dict[str, Any], family_member: pd.DataFrame) -> pd.DataFrame:
    drug_family = build_drug_family_map(cfg, family_member)
    if drug_family.empty:
        return pd.DataFrame()
    frames = []
    # Significant single-resource associations are retained with lower weight;
    # PRISM-GDSC replicated associations receive the highest weight below.
    for assoc_name in ["prism_lnc_drug.parquet", "gdsc_lnc_drug.parquet"]:
        assoc_path = cfg["_standardized"] / assoc_name
        if not assoc_path.exists():
            continue
        a = read_table(assoc_path)
        if "fdr" not in a.columns:
            continue
        a = a.loc[pd.to_numeric(a.fdr, errors="coerce") <= cfg["pair_evidence"]["drug_max_fdr"]].copy()
        if a.empty:
            continue
        effect_col = "beta" if "beta" in a.columns else ("rho" if "rho" in a.columns else None)
        if effect_col is None:
            continue
        a["association_support"] = signed_support(pd.to_numeric(a[effect_col], errors="coerce"), pd.to_numeric(a.fdr, errors="coerce"), effect_scale=0.35) * 0.65
        cov = a.model_covariates.astype(str).str.lower() if "model_covariates" in a.columns else pd.Series("", index=a.index)
        a["association_support"] *= np.where(cov.str.contains("target|pathway|protein|depmap"), 1.10, 1.0)
        a["association_support"] = a.association_support.clip(0, 0.75)
        a["association_direction"] = a.direction.astype(str) if "direction" in a.columns else np.where(pd.to_numeric(a[effect_col], errors="coerce") >= 0, "positive", "negative")
        a["n_supporting_datasets"] = 1
        a["replication_status"] = a.get("resource", assoc_name).astype(str) if isinstance(a.get("resource", assoc_name), pd.Series) else assoc_name
        a["source_database"] = a["replication_status"]
        frames.append(a[["cancer_id", "lncrna_id", "drug_id", "association_support", "association_direction", "n_supporting_datasets", "replication_status", "source_database"]])
    rep_path = cfg["_standardized"] / "drug_replication.parquet"
    if rep_path.exists():
        rep = read_table(rep_path)
        rep["association_support"] = np.select(
            [rep.replication_status.astype(str).str.contains("fdr", case=False), rep.replication_status.astype(str).str.contains("nominal", case=False), rep.n_gdsc_same_direction.fillna(0) > 0],
            [1.0, 0.75, 0.5], default=0.15,
        )
        rep["association_direction"] = rep.prism_direction.astype(str)
        rep["source_database"] = "PRISM_GDSC_replication"
        frames.append(rep[["cancer_id", "lncrna_id", "drug_id", "association_support", "association_direction", "n_supporting_datasets", "replication_status", "source_database"]])
    curated_path = cfg["_standardized"] / "lncRNA_drug_curated_evidence.parquet"
    if curated_path.exists():
        cur = expand_global_cancer(read_table(curated_path), cfg)
        cur["association_support"] = 0.8
        cur["association_direction"] = cur.direction.astype(str)
        cur["n_supporting_datasets"] = 1
        cur["replication_status"] = "curated"
        frames.append(cur[["cancer_id", "lncrna_id", "drug_id", "association_support", "association_direction", "n_supporting_datasets", "replication_status", "source_database"]])
    if not frames:
        return pd.DataFrame()
    assoc = pd.concat(frames, ignore_index=True)
    assoc = assoc.merge(drug_family, on="drug_id", how="inner")
    assoc["weighted_support"] = assoc.association_support * assoc.drug_family_support
    source_out = assoc.groupby(["cancer_id", "lncrna_id", "pathway_family_id", "source_database"], as_index=False, observed=True).agg(
        source_support=("weighted_support", "max"),
        n_source_drugs=("drug_id", "nunique"),
    )
    write_table(source_out, cfg["_results"] / "tables" / "drug_pair_source_support.parquet")
    out = assoc.groupby(["cancer_id", "lncrna_id", "pathway_family_id"], as_index=False, observed=True).agg(
        drug_support=("weighted_support", "max"),
        drug_n_drugs=("drug_id", "nunique"),
        drug_n_datasets=("n_supporting_datasets", "max"),
        drug_replication_status=("replication_status", lambda x: ";".join(sorted(set(map(str, x))))),
    )
    out["drug_available"] = True
    return out



def build_cellline_pair_support(cfg: dict[str, Any]) -> pd.DataFrame:
    x = build_cellline_context_support(cfg)
    if x.empty:
        return x
    cancers = set(read_table(cfg["_standardized"] / "dim_cancer.parquet").cancer_id.astype(str))
    pan = x.loc[~x.cancer_id.astype(str).isin(cancers)].copy()
    local = x.loc[x.cancer_id.astype(str).isin(cancers)].copy()
    if not pan.empty:
        expanded = []
        for cancer in sorted(cancers):
            part = pan.copy(); part["cancer_id"] = cancer; part["cellline_context_source"] = "pan_cell_line"; expanded.append(part)
        pan = pd.concat(expanded, ignore_index=True)
    if not local.empty:
        local["cellline_context_source"] = "lineage"
    return pd.concat([local, pan], ignore_index=True).sort_values("cellline_context_source").drop_duplicates(["cancer_id", "lncrna_id", "pathway_family_id"], keep="first")

def build_state_support(cfg: dict[str, Any], family_member: pd.DataFrame) -> pd.DataFrame:
    root = cfg["_results"] / "tables" / "pathway_state_edge"
    if not root.exists():
        return pd.DataFrame()
    x = read_table(root)
    x["state_support"] = signed_support(x.effect, x.fdr, effect_scale=0.3)
    x = x.merge(family_member[["pathway_id", "pathway_family_id", "membership_weight"]], on="pathway_id")
    x["state_support"] *= x.membership_weight.fillna(1.0)
    return x.groupby(["cancer_id", "pathway_family_id"], as_index=False, observed=True).agg(state_support=("state_support", "max"), state_n_edges=("state_id", "nunique"), state_available=("state_id", lambda x: True))


def merge_evidence(cfg: dict[str, Any], components: list[pd.DataFrame]) -> pd.DataFrame:
    keys = ["cancer_id", "lncrna_id", "pathway_family_id"]
    nonempty = [x for x in components if x is not None and not x.empty]
    if not nonempty:
        raise RuntimeError("No evidence component available")
    out = nonempty[0]
    for frame in nonempty[1:]:
        merge_keys = [x for x in keys if x in frame.columns and x in out.columns]
        out = out.merge(frame, on=merge_keys, how="outer")
    for col in ["bulk_support", "sc_ssgsea_support", "ucell_support", "replication_support", "interaction_support", "perturbation_support", "drug_support", "cellline_support", "clinical_support", "state_support"]:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0).clip(0, 1)
    for col in ["bulk_available", "sc_available", "ucell_available", "interaction_available", "drug_available", "cellline_available"]:
        if col not in out.columns:
            out[col] = False
        out[col] = out[col].fillna(False).astype(bool)
    if "independent_pmids" not in out:
        out["independent_pmids"] = 0
    if "independent_events" not in out:
        out["independent_events"] = 0
    out["independent_pmid_score"] = np.clip(np.log1p(out.independent_pmids.fillna(0)) / np.log(11), 0, 1)
    out["independent_dataset_score"] = np.clip(np.log1p(out.get("drug_n_datasets", 0)) / np.log(4), 0, 1)

    direction_cols = [x for x in ["bulk_direction", "sc_direction"] if x in out.columns]
    if direction_cols:
        signs = np.column_stack([_direction_sign(out[x]) for x in direction_cols])
        nonzero = (signs != 0).sum(axis=1)
        out["direction_consistency"] = np.divide(np.abs(signs.sum(axis=1)), nonzero, out=np.zeros(len(out)), where=nonzero > 0)
        signed_total = signs.sum(axis=1)
        out["direction"] = np.select([signed_total > 0, signed_total < 0], ["positive", "negative"], default="unknown")
    else:
        out["direction_consistency"] = 0.0
        out["direction"] = "unknown"

    weights = cfg["pair_evidence"]["weights"]
    numerator = np.zeros(len(out))
    denominator = np.zeros(len(out))
    mapping = {"bulk": "bulk_support", "sc_ssgsea": "sc_ssgsea_support", "ucell": "ucell_support", "replication": "replication_support", "interaction": "interaction_support", "perturbation": "perturbation_support", "drug": "drug_support", "cellline": "cellline_support", "clinical": "clinical_support", "state": "state_support"}
    for name, col in mapping.items():
        weight = float(weights.get(name, 0.0))
        available_col = {"bulk": "bulk_available", "sc_ssgsea": "sc_available", "ucell": "ucell_available", "interaction": "interaction_available", "drug": "drug_available", "cellline": "cellline_available"}.get(name)
        available = out[available_col].to_numpy(bool) if available_col else (out[col].to_numpy() > 0)
        numerator += weight * out[col].to_numpy(float)
        denominator += weight * available.astype(float)
    out["observed_evidence_score"] = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    out["observed_evidence_score"] *= 0.75 + 0.25 * out.direction_consistency
    source_cols = ["bulk_support", "sc_ssgsea_support", "ucell_support", "interaction_support", "perturbation_support", "drug_support", "cellline_support"]
    out["n_positive_evidence_sources"] = (out[source_cols] > 0).sum(axis=1)
    pe = cfg["pair_evidence"]
    out["label_class"] = np.select(
        [(out.observed_evidence_score >= pe["strong_positive_min_score"]) & (out.n_positive_evidence_sources >= pe["strong_positive_min_sources"]), out.observed_evidence_score >= pe["weak_positive_min_score"]],
        ["strong_positive", "weak_positive"], default="unlabeled",
    )
    # Explicit estimands.  ``label`` remains a historical storage alias only;
    # fail-closed training entry points must select one of the named columns.
    out["association_proxy_label"] = out.label_class.isin(
        ["strong_positive", "weak_positive"]
    ).astype(np.int8)
    out["strong_evidence_label"] = out.label_class.eq("strong_positive").astype(np.int8)
    out["label"] = out.strong_evidence_label
    out["label_semantics"] = "PATHWAY_STRONG_EVIDENCE_V1:strong_evidence_label"
    out["sample_weight"] = np.select([out.label_class == "strong_positive", out.label_class == "weak_positive"], [1.0, cfg["training"]["weak_positive_weight"]], default=cfg["training"]["unlabeled_weight"])
    masks = []
    for row in out[["bulk_available", "sc_available", "ucell_available", "interaction_available", "drug_available", "cellline_available"]].itertuples(index=False):
        names = [name for name, available in zip(["bulk", "sc", "ucell", "interaction", "drug", "cellline"], row) if not available]
        masks.append(";".join(names))
    out["evidence_missing_mask"] = masks
    out["pair_id"] = [stable_id("PAIR", c, l, f) for c, l, f in zip(out.cancer_id, out.lncrna_id, out.pathway_family_id)]
    out["analysis_version"] = cfg["analysis_version"]
    return out


def build_pair_evidence(cfg: dict[str, Any]) -> pd.DataFrame:
    from .pathway_target import EXACT_PATHWAY_TARGET, pathway_target_level

    if pathway_target_level(cfg) == EXACT_PATHWAY_TARGET:
        from .exact_pathway_evidence import build_exact_pair_evidence

        return build_exact_pair_evidence(cfg)

    family_member = read_table(cfg["_results"] / "tables" / "pathway_family_member.parquet")
    bulk = build_bulk_support(cfg, family_member)
    sc = build_sc_support(cfg, family_member)
    ucell = build_ucell_pair_support(cfg, sc, family_member)
    replication = build_bulk_sc_replication_support(cfg, family_member)
    interaction = build_interaction_support(cfg, family_member)
    drug = build_drug_support(cfg, family_member)
    cellline = build_cellline_pair_support(cfg)
    state = build_state_support(cfg, family_member)
    pair = merge_evidence(cfg, [bulk, sc, ucell, replication, interaction, drug, cellline, state])
    source_frames = []
    for frame, source_name, support_col in [
        (bulk, "TCGA_bulk", "bulk_support"), (sc, "scRNA_true_ssGSEA", "sc_ssgsea_support"),
        (ucell, "UCell", "ucell_support"), (replication, "bulk_sc_replication", "replication_support"),
        (cellline, "Cell_Model_Passports+DepMap", "cellline_support")
    ]:
        if frame is not None and not frame.empty and support_col in frame.columns:
            part = frame[["cancer_id", "lncrna_id", "pathway_family_id", support_col]].copy()
            part["source_database"] = source_name; part = part.rename(columns={support_col: "source_support"}); source_frames.append(part)
    for source_file in [cfg["_results"] / "tables" / "interaction_pair_source_support.parquet", cfg["_results"] / "tables" / "drug_pair_source_support.parquet"]:
        if source_file.exists():
            part = read_table(source_file)
            source_frames.append(part[["cancer_id", "lncrna_id", "pathway_family_id", "source_database", "source_support"]])
    if source_frames:
        source_contribution = pd.concat(source_frames, ignore_index=True).groupby(["cancer_id", "lncrna_id", "pathway_family_id", "source_database"], as_index=False, observed=True).source_support.max()
        source_contribution["pair_id"] = [stable_id("PAIR", c, l, f) for c, l, f in zip(source_contribution.cancer_id, source_contribution.lncrna_id, source_contribution.pathway_family_id)]
        write_table(source_contribution, cfg["_results"] / "tables" / "pair_evidence_source_contribution.parquet")
    write_table(pair, cfg["_results"] / "tables" / "pair_evidence.parquet")
    component_summary = pd.DataFrame([
        {"component": "bulk", "rows": len(bulk)},
        {"component": "sc_ssgsea", "rows": len(sc)},
        {"component": "ucell", "rows": len(ucell)},
        {"component": "bulk_sc_replication", "rows": len(replication)},
        {"component": "interaction", "rows": len(interaction)},
        {"component": "drug", "rows": len(drug)},
        {"component": "cellline", "rows": len(cellline)},
        {"component": "state", "rows": len(state)},
        {"component": "pair_evidence", "rows": len(pair)},
    ])
    write_table(component_summary, cfg["_results"] / "tables" / "pair_evidence_component_summary.tsv")
    return pair
