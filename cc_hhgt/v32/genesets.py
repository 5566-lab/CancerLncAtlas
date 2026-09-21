"""Deterministic V3.2 exact-pathway ranked Gene Set materialization.

This module is deliberately post-training code.  It aggregates fold predictions
and never fits a model or imports a training framework.  Regulatory evidence is
carried to the website output, but is not used to select or rank members.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from cc_hhgt.common import stable_id


class GeneSetContractError(ValueError):
    """Raised when fold predictions violate the V3.2 materialization contract."""


@dataclass(frozen=True)
class GeneSetMaterializationConfig:
    """Frozen, inexpensive post-processing settings for ranked Gene Sets."""

    expected_folds: int = 5
    min_members: int = 10
    max_members: int = 200
    membership_probability_min: float = 0.5
    analysis_version: str = "V3.2"

    def validate(self) -> None:
        if self.expected_folds < 1:
            raise GeneSetContractError("expected_folds must be positive")
        if self.min_members < 1 or self.max_members < self.min_members:
            raise GeneSetContractError("invalid Gene Set member limits")
        if not 0.0 <= self.membership_probability_min <= 1.0:
            raise GeneSetContractError(
                "membership_probability_min must be within [0, 1]"
            )
        if not str(self.analysis_version).strip():
            raise GeneSetContractError("analysis_version must be non-empty")


FROZEN_MEMBER_COLUMNS = (
    "cancer_id",
    "lncrna_id",
    "pathway_id",
    "pathway_family_id",
    "association_membership_probability",
    "association_direction",
    "l1_probability",
    "ridge_probability",
    "graph_residual",
    "graph_gate",
    "fold_rank_percentile",
    "fold_selection_frequency",
    "shared_or_local_scope",
    "regulatory_evidence_confidence",
)

_REQUIRED_FOLD_COLUMNS = {
    "cancer_id",
    "lncrna_id",
    "pathway_id",
    "pathway_family_id",
    "association_membership_probability",
    "association_direction",
    "l1_probability",
    "ridge_probability",
    "graph_residual",
    "graph_gate",
    "shared_or_local_scope",
    "regulatory_evidence_confidence",
}

_PROBABILITY_COLUMNS = (
    "association_membership_probability",
    "l1_probability",
    "ridge_probability",
    "graph_gate",
    "regulatory_evidence_confidence",
)

_DIRECTION_ALIASES = {
    "positive": "positive",
    "pos": "positive",
    "up": "positive",
    "+": "positive",
    "+1": "positive",
    "1": "positive",
    "negative": "negative",
    "neg": "negative",
    "down": "negative",
    "-": "negative",
    "-1": "negative",
}

_SCOPE_ALIASES = {
    "shared": "shared",
    "pan_cancer_shared": "shared",
    "pancancer_shared": "shared",
    "shared_eligible": "shared",
    "local": "cancer_local",
    "local_only": "cancer_local",
    "cancer_local": "cancer_local",
    "cancer-local": "cancer_local",
}


def _fold_column(frame: pd.DataFrame) -> str:
    present = [name for name in ("patient_fold_id", "fold_id") if name in frame]
    if len(present) != 1:
        raise GeneSetContractError(
            "fold predictions require exactly one of patient_fold_id or fold_id"
        )
    return present[0]


def _normalise_direction(value: Any) -> str:
    key = str(value).strip().lower()
    if key not in _DIRECTION_ALIASES:
        raise GeneSetContractError(f"invalid association_direction: {value!r}")
    return _DIRECTION_ALIASES[key]


def _normalise_scope(value: Any) -> str:
    key = str(value).strip().lower()
    if key not in _SCOPE_ALIASES:
        raise GeneSetContractError(f"invalid shared_or_local_scope: {value!r}")
    return _SCOPE_ALIASES[key]


def validate_fold_predictions(frame: pd.DataFrame) -> str:
    """Validate fold predictions and return the fold identifier column name."""

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise GeneSetContractError("fold predictions must be a non-empty DataFrame")
    missing = sorted(_REQUIRED_FOLD_COLUMNS - set(frame.columns))
    if missing:
        raise GeneSetContractError(f"fold predictions lack columns: {missing}")
    fold_column = _fold_column(frame)

    for target_column in ("pathway_target_level", "target_level"):
        if target_column in frame:
            levels = set(frame[target_column].dropna().astype(str).str.lower())
            if levels != {"exact_pathway"}:
                raise GeneSetContractError(
                    "V3.2 Gene Sets must target exact_pathway, never pathway_family"
                )

    string_columns = (
        fold_column,
        "cancer_id",
        "lncrna_id",
        "pathway_id",
        "pathway_family_id",
    )
    for column in string_columns:
        values = frame[column].astype("string")
        if values.isna().any() or values.str.strip().eq("").any():
            raise GeneSetContractError(f"{column} contains missing or empty identifiers")

    if frame["pathway_id"].astype(str).eq(
        frame["pathway_family_id"].astype(str)
    ).any():
        raise GeneSetContractError(
            "pathway_id equals pathway_family_id; family-level targets are forbidden"
        )
    family_counts = frame.groupby("pathway_id", observed=True)[
        "pathway_family_id"
    ].nunique(dropna=False)
    if family_counts.gt(1).any():
        offenders = sorted(family_counts[family_counts.gt(1)].index.astype(str))
        raise GeneSetContractError(
            f"exact pathways map to multiple pathway families: {offenders[:5]}"
        )

    for column in _PROBABILITY_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(values).all() or values.lt(0).any() or values.gt(1).any():
            raise GeneSetContractError(f"{column} must contain finite [0, 1] values")
    residual = pd.to_numeric(frame["graph_residual"], errors="coerce")
    if not np.isfinite(residual).all():
        raise GeneSetContractError("graph_residual must be finite")

    duplicate_key = [fold_column, "cancer_id", "lncrna_id", "pathway_id"]
    if frame.duplicated(duplicate_key).any():
        raise GeneSetContractError(
            "fold predictions contain duplicate cancer-lncRNA-exact-pathway rows"
        )

    # A candidate has one measurement scope.  Mixing shared and local rows across
    # folds would contaminate the shared-universe subtype analysis.
    canonical_scope = frame["shared_or_local_scope"].map(_normalise_scope)
    scope_counts = (
        frame.assign(_scope=canonical_scope)
        .groupby(["cancer_id", "lncrna_id", "pathway_id"], observed=True)["_scope"]
        .nunique()
    )
    if scope_counts.gt(1).any():
        raise GeneSetContractError("candidate scope changes across patient folds")
    frame["association_direction"].map(_normalise_direction)
    return fold_column


def _add_fold_rank_percentile(work: pd.DataFrame, fold_column: str) -> pd.DataFrame:
    ordered = work.sort_values(
        [
            fold_column,
            "cancer_id",
            "pathway_id",
            "association_direction",
            "association_membership_probability",
            "lncrna_id",
        ],
        ascending=[True, True, True, True, False, True],
        kind="stable",
    ).copy()
    group_columns = [
        fold_column,
        "cancer_id",
        "pathway_id",
        "association_direction",
    ]
    ordered["_fold_rank"] = ordered.groupby(
        group_columns, observed=True, sort=False
    ).cumcount()
    ordered["_fold_size"] = ordered.groupby(
        group_columns, observed=True, sort=False
    )["lncrna_id"].transform("size")
    denominator = (ordered["_fold_size"] - 1).clip(lower=1)
    ordered["_fold_rank_percentile"] = 1.0 - ordered["_fold_rank"] / denominator
    ordered.loc[ordered["_fold_size"].eq(1), "_fold_rank_percentile"] = 1.0
    return ordered


def aggregate_fold_predictions(
    fold_predictions: pd.DataFrame,
    config: GeneSetMaterializationConfig | None = None,
) -> pd.DataFrame:
    """Aggregate fold predictions without allowing evidence to influence ranking."""

    config = config or GeneSetMaterializationConfig()
    config.validate()
    fold_column = validate_fold_predictions(fold_predictions)
    work = fold_predictions.copy()
    work[fold_column] = work[fold_column].astype(str)
    for column in ("cancer_id", "lncrna_id", "pathway_id", "pathway_family_id"):
        work[column] = work[column].astype(str)
    for column in _PROBABILITY_COLUMNS + ("graph_residual",):
        work[column] = pd.to_numeric(work[column], errors="raise").astype(float)
    work["association_direction"] = work["association_direction"].map(
        _normalise_direction
    )
    work["shared_or_local_scope"] = work["shared_or_local_scope"].map(
        _normalise_scope
    )
    work = _add_fold_rank_percentile(work, fold_column)

    identity = ["cancer_id", "lncrna_id", "pathway_id"]
    direction_votes = (
        work.groupby(identity + ["association_direction"], observed=True, sort=True)
        .agg(
            _direction_fold_count=(fold_column, "nunique"),
            _direction_probability_sum=("association_membership_probability", "sum"),
        )
        .reset_index()
        .sort_values(
            identity
            + [
                "_direction_fold_count",
                "_direction_probability_sum",
                "association_direction",
            ],
            ascending=[True, True, True, False, False, True],
            kind="stable",
        )
        .drop_duplicates(identity, keep="first")
    )
    consensus = direction_votes[identity + ["association_direction"]].rename(
        columns={"association_direction": "_consensus_direction"}
    )
    work = work.merge(consensus, on=identity, how="left", validate="many_to_one")
    work["_direction_matches"] = work["association_direction"].eq(
        work["_consensus_direction"]
    )
    work["_selected_in_fold"] = (
        work["_direction_matches"]
        & work["association_membership_probability"].ge(
            config.membership_probability_min
        )
    )
    work["_rank_when_consensus"] = work["_fold_rank_percentile"].where(
        work["_direction_matches"]
    )

    aggregate = (
        work.groupby(identity, observed=True, sort=True)
        .agg(
            pathway_family_id=("pathway_family_id", "first"),
            association_membership_probability=(
                "association_membership_probability",
                "mean",
            ),
            l1_probability=("l1_probability", "mean"),
            ridge_probability=("ridge_probability", "mean"),
            graph_residual=("graph_residual", "mean"),
            graph_gate=("graph_gate", "mean"),
            fold_rank_percentile=("_rank_when_consensus", "mean"),
            _selected_fold_count=("_selected_in_fold", "sum"),
            _direction_fold_count=("_direction_matches", "sum"),
            n_folds_available=(fold_column, "nunique"),
            shared_or_local_scope=("shared_or_local_scope", "first"),
            regulatory_evidence_confidence=(
                "regulatory_evidence_confidence",
                "mean",
            ),
        )
        .reset_index()
        .merge(consensus, on=identity, how="left", validate="one_to_one")
    )
    if aggregate["n_folds_available"].gt(config.expected_folds).any():
        raise GeneSetContractError("candidate appears in more than expected_folds")
    aggregate["association_direction"] = aggregate.pop("_consensus_direction")
    aggregate["fold_selection_frequency"] = (
        aggregate.pop("_selected_fold_count") / float(config.expected_folds)
    )
    aggregate["direction_stability"] = (
        aggregate.pop("_direction_fold_count") / aggregate["n_folds_available"]
    )
    aggregate["fold_rank_percentile"] = aggregate[
        "fold_rank_percentile"
    ].fillna(0.0)
    aggregate["pathway_target_level"] = "exact_pathway"
    return aggregate.sort_values(identity, kind="stable").reset_index(drop=True)


def materialize_ranked_genesets(
    fold_predictions: pd.DataFrame,
    config: GeneSetMaterializationConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create deterministic cancer-by-exact-pathway directional ranked Gene Sets."""

    config = config or GeneSetMaterializationConfig()
    config.validate()
    aggregate = aggregate_fold_predictions(fold_predictions, config)
    eligible = aggregate.loc[
        aggregate["association_membership_probability"].ge(
            config.membership_probability_min
        )
    ].copy()

    master_rows: list[dict[str, Any]] = []
    member_frames: list[pd.DataFrame] = []
    grouping = ["cancer_id", "pathway_id", "association_direction"]
    for (cancer, pathway, direction), group in eligible.groupby(
        grouping, observed=True, sort=True
    ):
        ordered = group.sort_values(
            [
                "association_membership_probability",
                "fold_selection_frequency",
                "fold_rank_percentile",
                "lncrna_id",
            ],
            ascending=[False, False, False, True],
            kind="stable",
        ).head(config.max_members)
        if len(ordered) < config.min_members:
            continue
        family_values = sorted(set(ordered["pathway_family_id"].astype(str)))
        if len(family_values) != 1:
            raise GeneSetContractError(
                f"exact pathway {pathway!r} has a non-unique family in materialization"
            )
        geneset_id = stable_id(
            "LGS32", cancer, pathway, direction, config.analysis_version
        )
        master_rows.append(
            {
                "geneset_id": geneset_id,
                "geneset_name": f"{cancer}__{pathway}__{str(direction).upper()}",
                "geneset_type": "cancer_exact_pathway_ranked",
                "cancer_id": str(cancer),
                "pathway_id": str(pathway),
                "pathway_family_id": family_values[0],
                "direction": str(direction),
                "member_count": int(len(ordered)),
                "shared_member_count": int(
                    ordered["shared_or_local_scope"].eq("shared").sum()
                ),
                "local_member_count": int(
                    ordered["shared_or_local_scope"].eq("cancer_local").sum()
                ),
                "analysis_version": config.analysis_version,
                "pathway_target_level": "exact_pathway",
                "ranking_uses_regulatory_evidence": False,
            }
        )
        ranked = ordered.copy()
        ranked.insert(0, "geneset_id", geneset_id)
        ranked.insert(1, "geneset_rank", np.arange(1, len(ranked) + 1))
        ranked["analysis_version"] = config.analysis_version
        member_frames.append(ranked)

    master_columns = [
        "geneset_id",
        "geneset_name",
        "geneset_type",
        "cancer_id",
        "pathway_id",
        "pathway_family_id",
        "direction",
        "member_count",
        "shared_member_count",
        "local_member_count",
        "analysis_version",
        "pathway_target_level",
        "ranking_uses_regulatory_evidence",
    ]
    master = pd.DataFrame(master_rows, columns=master_columns)
    member_columns = [
        "geneset_id",
        "geneset_rank",
        *FROZEN_MEMBER_COLUMNS,
        "direction_stability",
        "n_folds_available",
        "pathway_target_level",
        "analysis_version",
    ]
    if member_frames:
        members = pd.concat(member_frames, ignore_index=True)
        members = members.loc[:, member_columns]
    else:
        members = pd.DataFrame(columns=member_columns)
    return master, members


def genesets_to_gmt_lines(
    master: pd.DataFrame,
    members: pd.DataFrame,
    *,
    description: str = "CancerLncAtlas CC-HHGT V3.2 exact pathway",
) -> list[str]:
    """Render ranked Gene Sets as deterministic GMT lines."""

    required_master = {"geneset_id", "geneset_name", "pathway_target_level"}
    required_members = {"geneset_id", "lncrna_id", "geneset_rank"}
    if not required_master.issubset(master.columns):
        raise GeneSetContractError("Gene Set master schema is incomplete")
    if not required_members.issubset(members.columns):
        raise GeneSetContractError("Gene Set member schema is incomplete")
    if not master["pathway_target_level"].astype(str).eq("exact_pathway").all():
        raise GeneSetContractError("GMT export refuses family-level targets")
    names = master.set_index("geneset_id")["geneset_name"].astype(str).to_dict()
    unknown = sorted(set(members["geneset_id"].astype(str)) - set(names))
    if unknown:
        raise GeneSetContractError(f"members reference unknown Gene Sets: {unknown[:5]}")
    lines: list[str] = []
    for geneset_id, group in members.groupby("geneset_id", observed=True, sort=True):
        ordered = group.sort_values(
            ["geneset_rank", "lncrna_id"], kind="stable"
        )["lncrna_id"].astype(str)
        lines.append("\t".join([names[str(geneset_id)], description, *ordered]))
    return lines


def assert_required_member_columns(columns: Iterable[str]) -> None:
    """Small schema hook used by writers and dry-run audits."""

    missing = sorted(set(FROZEN_MEMBER_COLUMNS) - set(columns))
    if missing:
        raise GeneSetContractError(f"ranked Gene Set output lacks fields: {missing}")
