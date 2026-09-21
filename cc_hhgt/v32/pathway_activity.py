from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


PROTEIN_CODING_TYPES = frozenset({"protein_coding", "protein-coding", "coding"})


def validate_protein_coding_pathway_membership(
    pathway_members: pd.DataFrame,
    gene_annotation: pd.DataFrame,
) -> pd.DataFrame:
    required_member = {"pathway_id", "gene_id"}
    required_annotation = {"gene_id", "gene_type"}
    if missing := sorted(required_member - set(pathway_members.columns)):
        raise ValueError(f"Pathway membership lacks columns: {missing}")
    if missing := sorted(required_annotation - set(gene_annotation.columns)):
        raise ValueError(f"Gene annotation lacks columns: {missing}")
    annotation = gene_annotation[["gene_id", "gene_type"]].copy()
    annotation["gene_id"] = annotation.gene_id.astype(str)
    annotation["gene_type"] = annotation.gene_type.astype(str).str.lower()
    if annotation.gene_id.duplicated().any():
        raise RuntimeError("Gene annotation contains duplicate gene IDs")
    members = pathway_members.copy()
    members[["pathway_id", "gene_id"]] = members[["pathway_id", "gene_id"]].astype(str)
    members = members.merge(annotation, on="gene_id", how="left", validate="many_to_one")
    if members.gene_type.isna().any():
        missing = members.loc[members.gene_type.isna(), "gene_id"].drop_duplicates().head(10).tolist()
        raise RuntimeError(f"Pathway members lack gene annotation: {missing}")
    noncoding = members.loc[~members.gene_type.isin(PROTEIN_CODING_TYPES)]
    if not noncoding.empty:
        examples = noncoding[["pathway_id", "gene_id", "gene_type"]].head(10).to_dict("records")
        raise RuntimeError(f"Non-protein-coding genes would leak into pathway activity: {examples}")
    if "membership_weight" not in members:
        members["membership_weight"] = 1.0
    members["membership_weight"] = pd.to_numeric(
        members.membership_weight, errors="coerce"
    )
    if members.membership_weight.isna().any() or (members.membership_weight <= 0).any():
        raise RuntimeError("Pathway membership weights must be finite and positive")
    return members[["pathway_id", "gene_id", "membership_weight"]].drop_duplicates(
        ["pathway_id", "gene_id"]
    )


def compute_rank_mean_activity(
    gene_expression: pd.DataFrame,
    pathway_members: pd.DataFrame,
    gene_annotation: pd.DataFrame,
    *,
    value_column: str = "expression",
) -> pd.DataFrame:
    """Deterministic rank-mean activity used for fixtures and fallback audits.

    Production can supply a precomputed ssGSEA table through
    :func:`validate_precomputed_activity`; both paths enforce protein-coding
    membership and never include lncRNA expression in the target.
    """

    required = {"sample_id", "gene_id", value_column}
    if missing := sorted(required - set(gene_expression.columns)):
        raise ValueError(f"Gene expression lacks columns: {missing}")
    members = validate_protein_coding_pathway_membership(pathway_members, gene_annotation)
    values = gene_expression[["sample_id", "gene_id", value_column]].copy()
    values[["sample_id", "gene_id"]] = values[["sample_id", "gene_id"]].astype(str)
    values[value_column] = pd.to_numeric(values[value_column], errors="coerce")
    if values[value_column].isna().any():
        raise RuntimeError("Gene expression contains non-numeric values")
    if values.duplicated(["sample_id", "gene_id"]).any():
        raise RuntimeError("Gene expression contains duplicate sample/gene rows")
    values["rank_percentile"] = values.groupby("sample_id", observed=True)[value_column].rank(
        method="average", pct=True
    )
    joined = values.merge(members, on="gene_id", how="inner", validate="many_to_many")
    joined["weighted_rank"] = joined.rank_percentile * joined.membership_weight
    activity = (
        joined.groupby(["sample_id", "pathway_id"], as_index=False, observed=True)
        .agg(weighted_sum=("weighted_rank", "sum"), weight_sum=("membership_weight", "sum"), n_genes=("gene_id", "nunique"))
    )
    activity["pathway_activity"] = activity.weighted_sum / activity.weight_sum
    return activity[["sample_id", "pathway_id", "pathway_activity", "n_genes"]].sort_values(
        ["sample_id", "pathway_id"], kind="stable"
    ).reset_index(drop=True)


def validate_precomputed_activity(activity: pd.DataFrame) -> pd.DataFrame:
    required = {"sample_id", "pathway_id", "pathway_activity"}
    if missing := sorted(required - set(activity.columns)):
        raise ValueError(f"Precomputed pathway activity lacks columns: {missing}")
    result = activity.copy()
    result[["sample_id", "pathway_id"]] = result[["sample_id", "pathway_id"]].astype(str)
    result["pathway_activity"] = pd.to_numeric(result.pathway_activity, errors="coerce")
    if result.pathway_activity.isna().any() or not np.isfinite(result.pathway_activity).all():
        raise RuntimeError("Precomputed pathway activity must be finite")
    if result.duplicated(["sample_id", "pathway_id"]).any():
        raise RuntimeError("Precomputed pathway activity contains duplicate keys")
    return result


@dataclass(frozen=True)
class ActivityScaler:
    mean: pd.Series
    scale: pd.Series

    @classmethod
    def fit(cls, activity: pd.DataFrame, train_sample_ids: set[str]) -> "ActivityScaler":
        frame = validate_precomputed_activity(activity)
        train = frame.loc[frame.sample_id.isin(set(map(str, train_sample_ids)))]
        if train.empty:
            raise RuntimeError("No training samples are available to fit pathway scaling")
        mean = train.groupby("pathway_id", observed=True).pathway_activity.mean()
        scale = train.groupby("pathway_id", observed=True).pathway_activity.std(ddof=0).replace(0, 1.0)
        return cls(mean=mean, scale=scale)

    def transform(self, activity: pd.DataFrame) -> pd.DataFrame:
        frame = validate_precomputed_activity(activity)
        frame["train_mean"] = frame.pathway_id.map(self.mean)
        frame["train_scale"] = frame.pathway_id.map(self.scale)
        if frame[["train_mean", "train_scale"]].isna().any().any():
            missing = frame.loc[frame.train_mean.isna(), "pathway_id"].unique()[:10]
            raise RuntimeError(f"Activity includes pathways absent from training: {missing.tolist()}")
        frame["pathway_activity_scaled"] = (
            frame.pathway_activity - frame.train_mean
        ) / frame.train_scale
        return frame.drop(columns=["train_mean", "train_scale"])
