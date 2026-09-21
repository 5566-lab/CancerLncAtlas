from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from .common import first_existing_column, input_path, read_table, write_table
from .stats import signed_support


KEYS = ["cancer_id", "lncrna_id", "pathway_id"]


def _parquet_glob(path: Path) -> str:
    value = path / "**" / "*.parquet" if path.is_dir() else path
    return value.as_posix()


def _direction_sign(values: pd.Series) -> np.ndarray:
    text = values.astype(str).str.lower()
    return np.select(
        [
            text.str.contains("pos|up|sens|increase|promot"),
            text.str.contains("neg|down|resist|decrease|inhibit"),
        ],
        [1.0, -1.0],
        default=0.0,
    )


def _expand_global_cancer(frame: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    if frame.empty or "cancer_id" not in frame:
        return frame
    text = frame.cancer_id.astype("string")
    mask = (
        text.isna()
        | text.str.strip().isin(["", "NA", "None", "nan", "PAN_CANCER", "ALL"])
    ).fillna(True)
    if not mask.any():
        return frame
    cancers = read_table(cfg["_standardized"] / "dim_cancer.parquet").cancer_id.astype(str)
    local = frame.loc[~mask].copy()
    global_rows = frame.loc[mask].drop(columns="cancer_id")
    expanded = []
    for cancer in cancers:
        part = global_rows.copy()
        part["cancer_id"] = cancer
        expanded.append(part)
    return pd.concat([local, *expanded], ignore_index=True)


def _aggregate_exact(
    frame: pd.DataFrame,
    support_col: str,
    effect_col: str | None = None,
    fdr_col: str | None = None,
    prefix: str = "",
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    missing = set(KEYS) - set(frame.columns)
    if missing:
        raise RuntimeError(f"Exact-pathway evidence lacks keys: {sorted(missing)}")
    value = frame.copy()
    value[support_col] = pd.to_numeric(value[support_col], errors="coerce").fillna(0.0)
    group = value.groupby(KEYS, as_index=False, observed=True)
    aggregate: dict[str, tuple[str, Any]] = {
        support_col: (support_col, "max"),
        f"{prefix}n_records": ("pathway_id", "size"),
    }
    if fdr_col:
        aggregate[fdr_col] = (fdr_col, "min")
    out = group.agg(**aggregate)
    if effect_col:
        ranked = value.assign(_abs_effect=pd.to_numeric(value[effect_col], errors="coerce").abs())
        best = (
            ranked.sort_values("_abs_effect", ascending=False, kind="stable")
            .drop_duplicates(KEYS)[KEYS + [effect_col]]
        )
        out = out.merge(best, on=KEYS, how="left", validate="one_to_one")
    return out


def _stable_ids(frame: pd.DataFrame, prefix: str) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype="string")
    con = duckdb.connect()
    con.register("keys", frame[KEYS].astype(str))
    result = con.execute(
        f"""
        SELECT '{prefix}:' || substr(
          sha256(cancer_id || '|' || lncrna_id || '|' || pathway_id), 1, 20
        ) AS stable_id
        FROM keys
        """
    ).fetchdf()["stable_id"]
    con.close()
    return result


def build_bulk_support(cfg: dict[str, Any]) -> pd.DataFrame:
    path = input_path(cfg, "bulk_lnc_pathway")
    policy = cfg["exact_pathway_evidence"]
    max_fdr = float(policy["bulk_discovery_max_fdr"])
    min_effect = float(policy["bulk_discovery_min_abs_effect"])
    con = duckdb.connect()
    out = con.execute(
        f"""
        SELECT
          cancer_id,
          lncrna_id,
          pathway_id,
          max(abs(rho_adjusted) *
              least(-log10(greatest(fdr_global, 1e-300)) / 10.0, 1.0)) AS bulk_support,
          arg_max(rho_adjusted, abs(rho_adjusted)) AS bulk_effect,
          min(fdr_global) AS bulk_fdr,
          count(*) AS bulk_n_records,
          max(CASE WHEN purity_adjusted AND cell_fraction_adjusted THEN 1 ELSE 0 END) AS bulk_adjusted
        FROM read_parquet('{_parquet_glob(path)}', hive_partitioning=1)
        WHERE fdr_global <= {max_fdr}
          AND abs(rho_adjusted) >= {min_effect}
        GROUP BY 1, 2, 3
        """
    ).fetchdf()
    con.close()
    if not out.empty:
        out["bulk_support"] = out.bulk_support.clip(0.0, 1.0)
        out["bulk_direction"] = np.where(out.bulk_effect >= 0, "positive", "negative")
        out["bulk_available"] = True
    return out


def build_sc_support(cfg: dict[str, Any]) -> pd.DataFrame:
    path = cfg["_standardized"] / "sc_lnc_pathway.parquet"
    if not path.exists():
        return pd.DataFrame()
    sc = read_table(path)
    effect_col = first_existing_column(sc, cfg["column_aliases"]["sc_effect"])
    fdr_col = first_existing_column(sc, cfg["column_aliases"]["sc_fdr"])
    sc[effect_col] = pd.to_numeric(sc[effect_col], errors="coerce")
    sc[fdr_col] = pd.to_numeric(sc[fdr_col], errors="coerce")
    policy = cfg["exact_pathway_evidence"]
    sc = sc.loc[
        sc[fdr_col].le(float(policy["sc_discovery_max_fdr"]))
        & sc[effect_col].abs().ge(float(policy["sc_discovery_min_abs_effect"]))
    ].copy()
    if sc.empty:
        return pd.DataFrame()
    sc["sc_ssgsea_support"] = signed_support(
        sc[effect_col], sc[fdr_col], effect_scale=0.4
    )
    if "n_patients" in sc:
        sc["sc_ssgsea_support"] *= np.clip(
            np.log1p(pd.to_numeric(sc.n_patients, errors="coerce").fillna(0)) / np.log(11),
            0,
            1,
        )
    out = _aggregate_exact(
        sc,
        "sc_ssgsea_support",
        effect_col=effect_col,
        fdr_col=fdr_col,
        prefix="sc_",
    ).rename(columns={effect_col: "sc_effect", fdr_col: "sc_fdr"})
    if not out.empty:
        out["sc_direction"] = np.where(out.sc_effect >= 0, "positive", "negative")
        out["sc_available"] = True
    return out


def build_ucell_support(cfg: dict[str, Any], sc_support: pd.DataFrame) -> pd.DataFrame:
    path = cfg["_results"] / "tables" / "ucell_pathway_support.parquet"
    if not path.exists() or sc_support.empty:
        return pd.DataFrame()
    ucell = read_table(path)
    if ucell.empty or "pathway_id" not in ucell:
        return pd.DataFrame()
    ucell["ucell_direction"] = np.where(
        pd.to_numeric(ucell.ucell_effect, errors="coerce") >= 0, "positive", "negative"
    )
    out = sc_support[
        ["cancer_id", "lncrna_id", "pathway_id", "sc_direction"]
    ].merge(ucell, on=["cancer_id", "pathway_id"], how="inner")
    out["ucell_direction_match"] = out.sc_direction.eq(out.ucell_direction)
    out["ucell_support"] = pd.to_numeric(out.ucell_support, errors="coerce").fillna(0.0)
    out["ucell_support"] *= np.where(out.ucell_direction_match, 1.0, 0.25)
    out["ucell_fdr"] = pd.to_numeric(out.get("fdr"), errors="coerce")
    out["ucell_available"] = True
    return out[
        KEYS
        + [
            "ucell_support",
            "ucell_effect",
            "ucell_fdr",
            "ucell_direction_match",
            "ucell_available",
        ]
    ]


def build_replication_support(cfg: dict[str, Any]) -> pd.DataFrame:
    path = input_path(cfg, "bulk_sc_replication")
    if not path.exists():
        return pd.DataFrame()
    value = read_table(path)
    if not set(KEYS).issubset(value.columns):
        return pd.DataFrame()
    if "replication_status" in value:
        status = value.replication_status.astype(str).str.lower()
        base = np.select(
            [
                status.str.contains("replicat|confirmed|fdr"),
                status.str.contains("nominal|same_direction"),
                status.str.contains("discord|opposite"),
            ],
            [1.0, 0.65, 0.0],
            default=0.25,
        )
    else:
        base = np.where(value.get("same_direction", False), 0.7, 0.0)
    if "n_sc_datasets" in value:
        base *= np.clip(
            np.log1p(pd.to_numeric(value.n_sc_datasets, errors="coerce").fillna(0))
            / np.log(4),
            0.25,
            1.0,
        )
    value["replication_support"] = base
    out = _aggregate_exact(value, "replication_support", prefix="replication_")
    out["replication_available"] = True
    return out


def build_interaction_support(
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = cfg["_standardized"] / "interaction_pathway_support.parquet"
    if not path.exists():
        return pd.DataFrame(), pd.DataFrame()
    value = _expand_global_cancer(read_table(path), cfg)
    if value.empty or not set(KEYS).issubset(value.columns):
        return pd.DataFrame(), pd.DataFrame()
    value["fdr"] = pd.to_numeric(value.fdr, errors="coerce")
    value["interaction_support"] = (
        np.clip(np.log1p(value.n_independent_events.fillna(0)) / np.log(11), 0, 1)
        * np.clip(-np.log10(value.fdr.clip(1e-12)) / 8, 0, 1)
    )
    level = value.highest_experiment_level.astype(str).str.lower()
    value["interaction_support"] *= np.select(
        [
            level.str.contains("low|reporter|rescue|western|rip|clip"),
            level.str.contains("high"),
        ],
        [1.0, 0.75],
        default=0.6,
    )
    value["perturbation_support"] = np.where(
        value.support_type.astype(str)
        .str.lower()
        .str.contains("perturb|knock|overexpress|rescue|functional"),
        value.interaction_support,
        0.0,
    )
    out = value.groupby(KEYS, as_index=False, observed=True).agg(
        interaction_support=("interaction_support", "max"),
        perturbation_support=("perturbation_support", "max"),
        independent_pmids=("n_independent_pmids", "sum"),
        independent_events=("n_independent_events", "sum"),
        interaction_fdr=("fdr", "min"),
        interaction_n_records=("pathway_id", "size"),
    )
    out["interaction_available"] = True

    source_out = pd.DataFrame()
    if "target_ids" in value:
        source_relation = read_table(
            cfg["_standardized"] / "interaction_relation.parquet",
            columns=["lncrna_id", "partner_id", "source_database"],
        ).drop_duplicates()
        lineage = value[
            KEYS
            + ["target_ids", "interaction_support", "perturbation_support"]
        ].copy()
        lineage["partner_id"] = (
            lineage.target_ids.fillna("")
            .astype(str)
            .str.replace(r"[\[\]\"']", "", regex=True)
            .str.split(r"[;,|]")
        )
        lineage = lineage.explode("partner_id")
        lineage["partner_id"] = lineage.partner_id.astype(str).str.strip()
        lineage = lineage.loc[lineage.partner_id.ne("")].merge(
            source_relation, on=["lncrna_id", "partner_id"], how="inner"
        )
        source_out = lineage.groupby(
            KEYS + ["source_database"], as_index=False, observed=True
        ).agg(
            source_support=("interaction_support", "max"),
            source_perturbation_support=("perturbation_support", "max"),
            n_source_targets=("partner_id", "nunique"),
        )
    return out, source_out


def _drug_pathway_map(cfg: dict[str, Any]) -> pd.DataFrame:
    target_path = cfg["_standardized"] / "drug_gene_target.parquet"
    member_path = cfg["_standardized"] / "pathway_gene_member.parquet"
    if not target_path.exists() or not member_path.exists():
        return pd.DataFrame()
    target = read_table(target_path).dropna(subset=["drug_id", "gene_id"])
    member = read_table(member_path).dropna(subset=["pathway_id", "gene_id"])
    joined = target[["drug_id", "gene_id"]].drop_duplicates().merge(
        member[["pathway_id", "gene_id"]].drop_duplicates(), on="gene_id"
    )
    counts = joined.groupby(["drug_id", "pathway_id"], as_index=False).agg(
        n_target_genes=("gene_id", "nunique")
    )
    counts["drug_pathway_support"] = np.clip(
        np.log1p(counts.n_target_genes) / np.log(6), 0, 1
    )
    write_table(counts, cfg["_results"] / "tables" / "drug_exact_pathway_support.parquet")
    return counts


def build_drug_support(
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    drug_pathway = _drug_pathway_map(cfg)
    if drug_pathway.empty:
        return pd.DataFrame(), pd.DataFrame()
    frames = []
    for name in ["prism_lnc_drug.parquet", "gdsc_lnc_drug.parquet"]:
        path = cfg["_standardized"] / name
        if not path.exists():
            continue
        value = read_table(path)
        if "fdr" not in value:
            continue
        value = value.loc[
            pd.to_numeric(value.fdr, errors="coerce")
            <= cfg["pair_evidence"]["drug_max_fdr"]
        ].copy()
        effect_col = "beta" if "beta" in value else ("rho" if "rho" in value else None)
        if value.empty or effect_col is None:
            continue
        value["association_support"] = (
            signed_support(
                pd.to_numeric(value[effect_col], errors="coerce"),
                pd.to_numeric(value.fdr, errors="coerce"),
                effect_scale=0.35,
            )
            * 0.65
        ).clip(0, 0.75)
        value["n_supporting_datasets"] = 1
        value["replication_status"] = name
        value["source_database"] = name
        frames.append(
            value[
                [
                    "cancer_id",
                    "lncrna_id",
                    "drug_id",
                    "association_support",
                    "n_supporting_datasets",
                    "replication_status",
                    "source_database",
                ]
            ]
        )
    replication = cfg["_standardized"] / "drug_replication.parquet"
    if replication.exists():
        value = read_table(replication)
        value["association_support"] = np.select(
            [
                value.replication_status.astype(str).str.contains("fdr", case=False),
                value.replication_status.astype(str).str.contains("nominal", case=False),
                value.n_gdsc_same_direction.fillna(0) > 0,
            ],
            [1.0, 0.75, 0.5],
            default=0.15,
        )
        value["source_database"] = "PRISM_GDSC_replication"
        frames.append(
            value[
                [
                    "cancer_id",
                    "lncrna_id",
                    "drug_id",
                    "association_support",
                    "n_supporting_datasets",
                    "replication_status",
                    "source_database",
                ]
            ]
        )
    curated = cfg["_standardized"] / "lncRNA_drug_curated_evidence.parquet"
    if curated.exists():
        value = _expand_global_cancer(read_table(curated), cfg)
        value["association_support"] = 0.8
        value["n_supporting_datasets"] = 1
        value["replication_status"] = "curated"
        frames.append(
            value[
                [
                    "cancer_id",
                    "lncrna_id",
                    "drug_id",
                    "association_support",
                    "n_supporting_datasets",
                    "replication_status",
                    "source_database",
                ]
            ]
        )
    if not frames:
        return pd.DataFrame(), pd.DataFrame()
    association = pd.concat(frames, ignore_index=True).merge(
        drug_pathway, on="drug_id", how="inner"
    )
    association["weighted_support"] = (
        association.association_support * association.drug_pathway_support
    )
    source_out = association.groupby(
        KEYS + ["source_database"], as_index=False, observed=True
    ).agg(
        source_support=("weighted_support", "max"),
        n_source_drugs=("drug_id", "nunique"),
    )
    out = association.groupby(KEYS, as_index=False, observed=True).agg(
        drug_support=("weighted_support", "max"),
        drug_n_drugs=("drug_id", "nunique"),
        drug_n_datasets=("n_supporting_datasets", "max"),
        drug_replication_status=(
            "replication_status",
            lambda x: ";".join(sorted(set(map(str, x)))),
        ),
    )
    out["drug_available"] = True
    return out, source_out


def build_exact_pathway_context(cfg: dict[str, Any]) -> pd.DataFrame:
    """Build pathway-only context without allowing it to define pair labels."""

    root = cfg["_results"] / "tables" / "pathway_state_edge"
    if not root.exists():
        return pd.DataFrame()
    value = read_table(root)
    if "pathway_id" not in value or "cancer_id" not in value:
        return pd.DataFrame()
    value["state_support"] = signed_support(value.effect, value.fdr, effect_scale=0.3)
    out = value.groupby(["cancer_id", "pathway_id"], as_index=False, observed=True).agg(
        state_support=("state_support", "max"),
        state_n_edges=("state_id", "nunique"),
    )
    out["state_available"] = True
    write_table(out, cfg["_results"] / "tables" / "exact_pathway_context.parquet")
    return out


def merge_exact_evidence(
    cfg: dict[str, Any], components: list[pd.DataFrame]
) -> pd.DataFrame:
    nonempty = [frame for frame in components if frame is not None and not frame.empty]
    if not nonempty:
        raise RuntimeError("No exact-pathway evidence component available")
    out = nonempty[0]
    for frame in nonempty[1:]:
        if not set(KEYS).issubset(frame.columns):
            raise RuntimeError("Family/pathway-only evidence cannot enter exact pair labels")
        out = out.merge(frame, on=KEYS, how="outer", validate="one_to_one")

    support_columns = [
        "bulk_support",
        "sc_ssgsea_support",
        "ucell_support",
        "replication_support",
        "interaction_support",
        "perturbation_support",
        "drug_support",
    ]
    for column in support_columns + ["clinical_support", "state_support", "cellline_support"]:
        if column not in out:
            out[column] = 0.0
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0).clip(0, 1)
    availability = {
        "bulk": "bulk_available",
        "sc_ssgsea": "sc_available",
        "ucell": "ucell_available",
        "replication": "replication_available",
        "interaction": "interaction_available",
        "perturbation": "interaction_available",
        "drug": "drug_available",
    }
    for column in set(availability.values()) | {"cellline_available"}:
        if column not in out:
            out[column] = False
        out[column] = out[column].fillna(False).astype(bool)
    if "independent_pmids" not in out:
        out["independent_pmids"] = 0
    if "independent_events" not in out:
        out["independent_events"] = 0
    out["independent_pmid_score"] = np.clip(
        np.log1p(out.independent_pmids.fillna(0)) / np.log(11), 0, 1
    )
    out["independent_dataset_score"] = np.clip(
        np.log1p(out.get("drug_n_datasets", 0)) / np.log(4), 0, 1
    )

    direction_columns = [
        column for column in ["bulk_direction", "sc_direction"] if column in out
    ]
    if direction_columns:
        signs = np.column_stack([_direction_sign(out[column]) for column in direction_columns])
        nonzero = (signs != 0).sum(axis=1)
        out["direction_consistency"] = np.divide(
            np.abs(signs.sum(axis=1)),
            nonzero,
            out=np.zeros(len(out)),
            where=nonzero > 0,
        )
        signed_total = signs.sum(axis=1)
        out["direction"] = np.select(
            [signed_total > 0, signed_total < 0],
            ["positive", "negative"],
            default="unknown",
        )
    else:
        out["direction_consistency"] = 0.0
        out["direction"] = "unknown"

    weights = cfg["pair_evidence"]["weights"]
    name_to_column = {
        "bulk": "bulk_support",
        "sc_ssgsea": "sc_ssgsea_support",
        "ucell": "ucell_support",
        "replication": "replication_support",
        "interaction": "interaction_support",
        "perturbation": "perturbation_support",
        "drug": "drug_support",
    }
    numerator = np.zeros(len(out), dtype=float)
    denominator = np.zeros(len(out), dtype=float)
    for name, column in name_to_column.items():
        weight = float(weights.get(name, 0.0))
        available = out[availability[name]].to_numpy(bool)
        numerator += weight * out[column].to_numpy(float)
        denominator += weight * available.astype(float)
    out["observed_evidence_score"] = np.divide(
        numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0
    )
    out["observed_evidence_score"] *= 0.75 + 0.25 * out.direction_consistency
    out["n_positive_evidence_sources"] = (out[support_columns] > 0).sum(axis=1)

    # The primary target is an association estimand. Literature, interaction,
    # perturbation and drug evidence are valuable annotations, but allowing any
    # of them to create the target would make the model predict the evidence
    # collection process rather than patient-derived pathway association.
    association_columns = ["bulk_support", "sc_ssgsea_support"]
    out["n_positive_association_sources"] = (
        out[association_columns].gt(0).sum(axis=1)
    )
    association_weights = {"bulk_support": 0.65, "sc_ssgsea_support": 0.35}
    out["association_support_score"] = sum(
        weight * out[column] for column, weight in association_weights.items()
    )

    policy = cfg.get("exact_pathway_evidence", {})
    if policy.get("family_level_evidence_may_define_label", False):
        raise RuntimeError("Family-level evidence cannot define exact-pathway labels")
    if policy.get("annotation_sources_may_define_label", False):
        raise RuntimeError("Annotation evidence cannot define association labels")
    bulk_qualified = out.bulk_support.gt(0)
    sc_qualified = out.sc_ssgsea_support.gt(0)
    bulk_fdr = pd.to_numeric(
        out.get("bulk_fdr", pd.Series(np.nan, index=out.index)), errors="coerce"
    )
    bulk_effect = pd.to_numeric(
        out.get("bulk_effect", pd.Series(np.nan, index=out.index)), errors="coerce"
    )
    stringent_bulk = (
        bulk_qualified
        & bulk_fdr.le(float(policy["bulk_strong_max_fdr"]))
        & bulk_effect.abs().ge(float(policy["bulk_strong_min_abs_effect"]))
    )
    replicated_bulk_sc = (
        bulk_qualified & sc_qualified & out.direction_consistency.ge(1.0)
    )
    strong = stringent_bulk | replicated_bulk_sc
    weak = bulk_qualified | sc_qualified
    out["label_class"] = np.select(
        [strong, weak], ["strong_positive", "weak_positive"], default="unlabeled"
    )
    out["association_proxy_label"] = out.label_class.isin(
        ["strong_positive", "weak_positive"]
    ).astype(np.int8)
    out["strong_association_label"] = out.label_class.eq("strong_positive").astype(np.int8)
    # Compatibility alias for the established training contract. Its semantics
    # are frozen by label_semantics below and audited as association, not
    # functional/mechanistic evidence.
    out["strong_evidence_label"] = out.strong_association_label
    out["label"] = out.strong_association_label
    out["label_semantics"] = (
        "EXACT_PATHWAY_PATIENT_ASSOCIATION_V2:cancer_x_lncrna_x_pathway_id"
    )
    out["sample_weight"] = np.select(
        [out.label_class.eq("strong_positive"), out.label_class.eq("weak_positive")],
        [1.0, cfg["training"]["weak_positive_weight"]],
        default=cfg["training"]["unlabeled_weight"],
    )
    masks = []
    ordered = [
        ("bulk", "bulk_available"),
        ("sc", "sc_available"),
        ("ucell", "ucell_available"),
        ("replication", "replication_available"),
        ("interaction", "interaction_available"),
        ("drug", "drug_available"),
    ]
    for row in out[[column for _, column in ordered]].itertuples(index=False):
        masks.append(
            ";".join(name for (name, _), present in zip(ordered, row) if not present)
        )
    out["evidence_missing_mask"] = masks
    out["pair_id"] = _stable_ids(out, "PAIR")
    out["analysis_version"] = cfg["analysis_version"]
    return out


def build_exact_pair_evidence(cfg: dict[str, Any]) -> pd.DataFrame:
    bulk = build_bulk_support(cfg)
    sc = build_sc_support(cfg)
    ucell = build_ucell_support(cfg, sc)
    replication = build_replication_support(cfg)
    interaction, interaction_sources = build_interaction_support(cfg)
    drug, drug_sources = build_drug_support(cfg)
    context = build_exact_pathway_context(cfg)
    pair = merge_exact_evidence(
        cfg, [bulk, sc, ucell, replication, interaction, drug]
    )
    family = read_table(
        cfg["_results"] / "tables" / "pathway_family_member.parquet",
        columns=["pathway_id", "pathway_family_id"],
    ).drop_duplicates("pathway_id")
    pair = pair.merge(family, on="pathway_id", how="left", validate="many_to_one")
    if pair.pathway_family_id.isna().any():
        missing = pair.loc[pair.pathway_family_id.isna(), "pathway_id"].astype(str).unique()
        raise RuntimeError(
            f"Exact-pathway evidence lacks hierarchy mapping for {len(missing)} pathways"
        )

    source_frames = []
    for frame, source_name, support_column in [
        (bulk, "TCGA_bulk", "bulk_support"),
        (sc, "scRNA_true_ssGSEA", "sc_ssgsea_support"),
        (ucell, "UCell", "ucell_support"),
        (replication, "bulk_sc_replication", "replication_support"),
    ]:
        if frame is not None and not frame.empty and support_column in frame:
            part = frame[KEYS + [support_column]].copy()
            part["source_database"] = source_name
            source_frames.append(part.rename(columns={support_column: "source_support"}))
    for frame in [interaction_sources, drug_sources]:
        if frame is not None and not frame.empty:
            source_frames.append(frame[KEYS + ["source_database", "source_support"]])
    if source_frames:
        source = pd.concat(source_frames, ignore_index=True).groupby(
            KEYS + ["source_database"], as_index=False, observed=True
        ).source_support.max()
        source["pair_id"] = _stable_ids(source, "PAIR")
        write_table(
            source,
            cfg["_results"] / "tables" / "pair_evidence_source_contribution.parquet",
        )

    write_table(pair, cfg["_results"] / "tables" / "pair_evidence.parquet")
    summary = pd.DataFrame(
        [
            {"component": "bulk_exact_pathway", "rows": len(bulk)},
            {"component": "sc_ssgsea_exact_pathway", "rows": len(sc)},
            {"component": "ucell_exact_pathway", "rows": len(ucell)},
            {"component": "bulk_sc_replication_exact_pathway", "rows": len(replication)},
            {"component": "interaction_exact_pathway", "rows": len(interaction)},
            {"component": "drug_exact_pathway", "rows": len(drug)},
            {"component": "pathway_context_not_label", "rows": len(context)},
            {"component": "pair_evidence_exact_pathway", "rows": len(pair)},
        ]
    )
    write_table(summary, cfg["_results"] / "tables" / "pair_evidence_component_summary.tsv")
    return pair
