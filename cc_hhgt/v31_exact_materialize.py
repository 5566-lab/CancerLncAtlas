from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .common import stable_id


IDENTITY_COLUMNS = [
    "candidate_id",
    "cancer_id",
    "lncrna_id",
    "pathway_id",
    "pathway_family_id",
]
FORBIDDEN_LABEL_COLUMNS = {
    "label",
    "label_class",
    "proxy_label",
    "association_proxy_label",
    "strong_association_label",
    "strong_evidence_label",
    "sample_weight",
}


def classify_exact_pathway_scores(
    frame: pd.DataFrame, materialization: dict[str, Any]
) -> pd.DataFrame:
    """Add post-hoc evidence classes without allowing labels into model scores."""

    missing = sorted(
        set(
            IDENTITY_COLUMNS
            + [
                "calibrated_probability",
                "seed_probability_std",
                "direction_positive_probability",
                "observed_evidence_score",
            ]
        )
        - set(frame.columns)
    )
    if missing:
        raise ValueError(f"Exact-pathway website scores lack columns: {missing}")
    forbidden = sorted(FORBIDDEN_LABEL_COLUMNS & set(frame.columns))
    if forbidden:
        raise ValueError(f"Held-out label columns entered website materialization: {forbidden}")
    if frame.candidate_id.astype(str).duplicated().any():
        raise ValueError("Exact-pathway website scores contain duplicate candidates")

    result = frame.copy()
    probability = pd.to_numeric(result.calibrated_probability, errors="coerce")
    uncertainty = pd.to_numeric(result.seed_probability_std, errors="coerce")
    evidence = pd.to_numeric(result.observed_evidence_score, errors="coerce").fillna(0.0)
    direction_probability = pd.to_numeric(
        result.direction_positive_probability, errors="coerce"
    )
    if (
        not np.isfinite(probability).all()
        or not np.isfinite(uncertainty).all()
        or not np.isfinite(direction_probability).all()
        or probability.lt(0).any()
        or probability.gt(1).any()
        or uncertainty.lt(0).any()
        or direction_probability.lt(0).any()
        or direction_probability.gt(1).any()
    ):
        raise ValueError("Exact-pathway website scores contain invalid probabilities")

    observed_min = float(materialization["observed_core_evidence_min"])
    supported_min = float(materialization["model_supported_probability_min"])
    predicted_min = float(materialization["predicted_candidate_probability_min"])
    predicted_evidence_max = float(
        materialization["predicted_candidate_max_observed_evidence"]
    )
    uncertainty_max = float(materialization["max_uncertainty"])
    result["relationship_class"] = np.select(
        [
            evidence.ge(observed_min),
            evidence.le(predicted_evidence_max)
            & probability.ge(predicted_min)
            & uncertainty.le(uncertainty_max),
            evidence.gt(0)
            & probability.ge(supported_min)
            & uncertainty.le(uncertainty_max),
        ],
        ["observed_core", "predicted_candidate", "model_supported"],
        default="exploratory",
    )
    observed_direction = (
        result["observed_direction"].fillna("").astype(str).str.lower()
        if "observed_direction" in result
        else pd.Series("", index=result.index, dtype=str)
    )
    predicted_direction = np.where(
        direction_probability.ge(0.5), "positive", "negative"
    )
    result["final_direction"] = np.where(
        observed_direction.isin(["positive", "negative"]),
        observed_direction,
        predicted_direction,
    )
    result["member_weight"] = (
        0.55 * probability + 0.45 * evidence.clip(0.0, 1.0)
    ) * (1.0 - uncertainty.clip(0.0, 1.0))
    return result


def build_exact_pathway_genesets(
    classified: pd.DataFrame,
    *,
    analysis_version: str,
    min_members: int,
    max_members: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create deterministic cancer-by-exact-pathway directional gene sets."""

    if min_members < 1 or max_members < min_members:
        raise ValueError("Invalid exact-pathway gene-set member limits")
    missing = sorted(
        {
            "cancer_id",
            "lncrna_id",
            "pathway_id",
            "pathway_family_id",
            "relationship_class",
            "final_direction",
            "member_weight",
            "calibrated_probability",
            "observed_evidence_score",
        }
        - set(classified.columns)
    )
    if missing:
        raise ValueError(f"Classified exact-pathway scores lack columns: {missing}")

    selected = classified.loc[
        classified.relationship_class.astype(str).ne("exploratory")
    ].copy()
    group_columns = ["cancer_id", "pathway_id", "final_direction"]
    master_rows: list[dict[str, Any]] = []
    member_frames: list[pd.DataFrame] = []
    for (cancer, pathway, direction), group in selected.groupby(
        group_columns, observed=True, sort=True
    ):
        ordered = group.sort_values(
            [
                "member_weight",
                "calibrated_probability",
                "observed_evidence_score",
                "lncrna_id",
            ],
            ascending=[False, False, False, True],
            kind="stable",
        ).head(max_members)
        if len(ordered) < min_members:
            continue
        geneset_id = stable_id(
            "LGS", cancer, pathway, direction, analysis_version
        )
        family_ids = sorted(
            set(ordered.pathway_family_id.dropna().astype(str).tolist())
        )
        pathway_name = (
            str(ordered.pathway_name.dropna().iloc[0])
            if "pathway_name" in ordered and ordered.pathway_name.notna().any()
            else str(pathway)
        )
        master_rows.append(
            {
                "geneset_id": geneset_id,
                "geneset_name": f"{cancer}__{pathway}__{str(direction).upper()}",
                "geneset_type": "cancer_exact_pathway",
                "cancer_id": str(cancer),
                "pathway_id": str(pathway),
                "pathway_name": pathway_name,
                "pathway_family_id": ";".join(family_ids),
                "direction": str(direction),
                "member_count": int(len(ordered)),
                "n_observed_core": int(
                    ordered.relationship_class.eq("observed_core").sum()
                ),
                "n_model_supported": int(
                    ordered.relationship_class.eq("model_supported").sum()
                ),
                "n_predicted_candidate": int(
                    ordered.relationship_class.eq("predicted_candidate").sum()
                ),
                "analysis_version": analysis_version,
                "pathway_target_level": "exact_pathway",
            }
        )
        ordered = ordered.copy()
        ordered["geneset_id"] = geneset_id
        ordered["rank"] = np.arange(1, len(ordered) + 1)
        member_frames.append(ordered)

    master = pd.DataFrame(master_rows)
    member = (
        pd.concat(member_frames, ignore_index=True)
        if member_frames
        else pd.DataFrame(columns=["geneset_id", "rank"])
    )
    return master, member
