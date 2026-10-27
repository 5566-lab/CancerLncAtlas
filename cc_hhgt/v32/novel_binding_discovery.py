"""Candidate generation and held-out evaluation for novel binding evidence.

"Discover new real bindings" cannot be delivered by this repository: no score
computed here is a measurement, and nothing in the code can make a candidate
true.  What *can* be delivered, and is delivered here, is the part that decides
whether such a claim would ever be allowed:

1. a deterministic, hash-based split of the known binding edges into a scoring
   partition and a **held-out** partition that is never used for features;
2. a candidate universe of lncRNA-protein pairs that excludes every scored
   known edge, so a "discovery" can never be a memorised positive;
3. an explicitly declared, **unfitted** additive score with frozen
   standardisation parameters, so no coefficient is learned from the evaluation
   labels;
4. evaluation at a fixed candidate budget: precision@k, recall@k, average
   precision, and the lift over a degree-prior baseline computed from the
   scored partition only;
5. hard gates that refuse an evaluation that cannot mean anything: empty
   holdout, candidates fewer than the budget, the primary feature missing for
   every candidate, or a feature table that carries a label column.

Every emitted row says ``candidate_status = "unvalidated_candidate"``.  The
module never claims that a candidate binds, and ``discovery_claim`` is false in
every record.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .omics_ablation_comparison import average_precision as _average_precision

FORMAT = "CANCERLNCATLAS_V32_NOVEL_BINDING_DISCOVERY_V1"
SCOPE = "UNVALIDATED_CANDIDATE_GENERATION_WITH_HELD_OUT_EVALUATION"
CANDIDATE_STATUS = "unvalidated_candidate"
FORBIDDEN_FEATURE_COLUMNS = ("label", "is_known", "is_positive", "outcome", "known_edge")
KNOWN_EDGE_COLUMNS = ("lncrna_id", "protein_id")
UNIVERSE_KEY = ("lncrna_id", "protein_id")
DEFAULT_BUDGET = 50
DEFAULT_HOLDOUT_FRACTION = 0.3
DEFAULT_MINIMUM_HOLDOUT = 5
HOLDOUT_FRACTION_TOLERANCE = 0.25


@dataclass(frozen=True)
class DiscoveryConfig:
    """Everything the score needs, declared before any evaluation is read."""

    feature_weights: Mapping[str, float]
    standardisation: Mapping[str, Mapping[str, float]]
    primary_feature: str
    budget: int = DEFAULT_BUDGET
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION
    minimum_holdout: int = DEFAULT_MINIMUM_HOLDOUT
    seed: int = 0
    evidence_arm: str = "experimental"

    def digest(self) -> str:
        payload = {
            "feature_weights": {key: float(value) for key, value in self.feature_weights.items()},
            "standardisation": {
                key: {inner: float(value) for inner, value in parameters.items()}
                for key, parameters in self.standardisation.items()
            },
            "primary_feature": self.primary_feature,
            "budget": int(self.budget),
            "holdout_fraction": float(self.holdout_fraction),
            "minimum_holdout": int(self.minimum_holdout),
            "seed": int(self.seed),
            "evidence_arm": str(self.evidence_arm),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()


def _digest(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _edge_key(lncrna_id: str, protein_id: str) -> str:
    return f"{lncrna_id}\x1f{protein_id}"


def validate_config(config: DiscoveryConfig) -> dict:
    if not isinstance(config, DiscoveryConfig):
        raise RuntimeError("DISCOVERY_CONFIG_TYPE")
    if not config.feature_weights:
        raise RuntimeError("DISCOVERY_WEIGHTS_MISSING")
    weights = {str(key): float(value) for key, value in config.feature_weights.items()}
    if any(not np.isfinite(value) for value in weights.values()):
        raise RuntimeError("DISCOVERY_WEIGHTS_NOT_FINITE")
    if str(config.primary_feature) not in weights:
        raise RuntimeError("DISCOVERY_PRIMARY_FEATURE_UNWEIGHTED")
    if not 0.0 < float(config.holdout_fraction) < 1.0:
        raise RuntimeError("DISCOVERY_HOLDOUT_FRACTION_INVALID")
    if int(config.budget) < 1:
        raise RuntimeError("DISCOVERY_BUDGET_INVALID")
    if int(config.minimum_holdout) < 1:
        raise RuntimeError("DISCOVERY_MINIMUM_HOLDOUT_INVALID")
    for feature in weights:
        parameters = config.standardisation.get(feature)
        if not isinstance(parameters, Mapping):
            raise RuntimeError(f"DISCOVERY_STANDARDISATION_MISSING: {feature}")
        center = parameters.get("center")
        scale = parameters.get("scale")
        if center is None or not np.isfinite(float(center)):
            raise RuntimeError(f"DISCOVERY_STANDARDISATION_INVALID: {feature}")
        if scale is None or not np.isfinite(float(scale)) or float(scale) <= 0:
            raise RuntimeError(f"DISCOVERY_STANDARDISATION_INVALID: {feature}")
    return {
        "format": FORMAT,
        "scope": SCOPE,
        "weights": weights,
        "primary_feature": str(config.primary_feature),
        "budget": int(config.budget),
        "holdout_fraction": float(config.holdout_fraction),
        "minimum_holdout": int(config.minimum_holdout),
        "seed": int(config.seed),
        "evidence_arm": str(config.evidence_arm),
        "config_digest": config.digest(),
        "score_is_fitted": False,
        "candidates_are_unvalidated": True,
    }


def split_known_edges(known_edges: pd.DataFrame, config: DiscoveryConfig) -> pd.DataFrame:
    """Deterministic hash split of known edges into scored and held-out parts."""

    missing = [name for name in KNOWN_EDGE_COLUMNS if name not in known_edges.columns]
    if missing:
        raise RuntimeError(f"DISCOVERY_KNOWN_COLUMNS_MISSING: {missing}")
    frame = known_edges.loc[:, list(KNOWN_EDGE_COLUMNS)].copy()
    frame["lncrna_id"] = frame.lncrna_id.astype(str)
    frame["protein_id"] = frame.protein_id.astype(str)
    if frame.duplicated(UNIVERSE_KEY).any():
        raise RuntimeError("DISCOVERY_KNOWN_EDGE_DUPLICATE")
    if not len(frame):
        raise RuntimeError("DISCOVERY_KNOWN_EDGES_EMPTY")
    keys = [
        _edge_key(lncrna, protein)
        for lncrna, protein in zip(frame.lncrna_id, frame.protein_id, strict=True)
    ]
    frame["edge_digest"] = [_digest(f"{config.seed}\x1f{key}") for key in keys]
    # The split is a pure function of (seed, edge identity): no label, degree or
    # score information participates, so the held-out part cannot be selected
    # to look good.
    threshold = int(float(config.holdout_fraction) * (2**32))
    frame["partition"] = np.where(
        [int(digest[:8], 16) < threshold for digest in frame.edge_digest],
        "holdout",
        "scored",
    )
    holdout = int((frame.partition == "holdout").sum())
    scored = int((frame.partition == "scored").sum())
    if holdout < int(config.minimum_holdout):
        raise RuntimeError(
            f"DISCOVERY_HOLDOUT_TOO_SMALL: {holdout} < {int(config.minimum_holdout)}"
        )
    if scored < 1:
        raise RuntimeError("DISCOVERY_SCORED_PARTITION_EMPTY")
    observed = holdout / len(frame)
    if abs(observed - float(config.holdout_fraction)) > HOLDOUT_FRACTION_TOLERANCE:
        raise RuntimeError(
            f"DISCOVERY_HOLDOUT_FRACTION_DRIFT: {observed:.3f} vs {config.holdout_fraction:.3f}"
        )
    return frame


def candidate_universe(
    known_edges: pd.DataFrame, train_edges: pd.DataFrame
) -> pd.DataFrame:
    """Every lncRNA-protein pair except the scored known edges.

    Known edges in the held-out partition stay in the universe without being
    marked: they are exactly the signal the evaluation must find, and the score
    never sees which candidates they are.
    """

    missing = [name for name in KNOWN_EDGE_COLUMNS if name not in known_edges.columns]
    if missing:
        raise RuntimeError(f"DISCOVERY_KNOWN_COLUMNS_MISSING: {missing}")
    lncrnas = sorted(set(known_edges.lncrna_id.astype(str)))
    proteins = sorted(set(known_edges.protein_id.astype(str)))
    if not lncrnas or not proteins:
        raise RuntimeError("DISCOVERY_UNIVERSE_EMPTY")
    universe = pd.MultiIndex.from_product(
        [lncrnas, proteins], names=list(UNIVERSE_KEY)
    ).to_frame(index=False)
    blocked = set(
        zip(
            train_edges.lncrna_id.astype(str),
            train_edges.protein_id.astype(str),
            strict=True,
        )
    )
    mask = [
        (lncrna, protein) not in blocked
        for lncrna, protein in zip(
            universe.lncrna_id, universe.protein_id, strict=True
        )
    ]
    universe = universe.loc[mask].reset_index(drop=True)
    if not len(universe):
        raise RuntimeError("DISCOVERY_UNIVERSE_EMPTY")
    return universe


def _check_feature_columns(features: pd.DataFrame, config: DiscoveryConfig) -> None:
    leaked = [name for name in FORBIDDEN_FEATURE_COLUMNS if name in features.columns]
    if leaked:
        raise RuntimeError(f"DISCOVERY_LABEL_LEAKAGE: {leaked}")
    missing = [
        name
        for name in (list(config.feature_weights) + list(UNIVERSE_KEY))
        if name not in features.columns
    ]
    if missing:
        raise RuntimeError(f"DISCOVERY_FEATURE_COLUMNS_MISSING: {missing}")


def score_candidates(
    universe: pd.DataFrame, features: pd.DataFrame, config: DiscoveryConfig
) -> pd.DataFrame:
    """Score the universe with a declared, unfitted additive model.

    A missing feature value contributes exactly zero and is counted, so a
    candidate can never be promoted by an imputed value.
    """

    _check_feature_columns(features, config)
    frame = features.loc[
        :, list(UNIVERSE_KEY) + list(config.feature_weights)
    ].copy()
    frame["lncrna_id"] = frame.lncrna_id.astype(str)
    frame["protein_id"] = frame.protein_id.astype(str)
    if frame.duplicated(UNIVERSE_KEY).any():
        raise RuntimeError("DISCOVERY_FEATURE_ROW_DUPLICATE")
    merged = universe.merge(frame, on=list(UNIVERSE_KEY), how="left", validate="one_to_one")
    contributions: dict[str, np.ndarray] = {}
    missing_counts: dict[str, int] = {}
    for feature, weight in config.feature_weights.items():
        raw = pd.to_numeric(merged[feature], errors="coerce").to_numpy(float)
        present = np.isfinite(raw)
        missing_counts[feature] = int((~present).sum())
        parameters = config.standardisation[feature]
        standardised = np.zeros(len(raw), dtype=float)
        standardised[present] = (
            raw[present] - float(parameters["center"])
        ) / float(parameters["scale"])
        contributions[feature] = standardised * float(weight)
    total = np.sum(np.stack(list(contributions.values()), axis=0), axis=0)
    merged["score"] = total
    for feature in contributions:
        merged[f"contribution::{feature}"] = contributions[feature]
    presence = np.column_stack(
        [
            np.isfinite(pd.to_numeric(merged[feature], errors="coerce").to_numpy(float))
            for feature in config.feature_weights
        ]
    )
    labels = np.asarray(list(config.feature_weights), dtype=object)
    merged["missing_features"] = [
        ",".join(labels[~row].tolist()) for row in presence
    ]
    primary_missing = int(missing_counts[str(config.primary_feature)])
    if primary_missing == len(merged):
        raise RuntimeError("DISCOVERY_PRIMARY_FEATURE_ABSENT")
    merged.attrs["missing_counts"] = missing_counts
    return merged


def degree_prior(
    universe: pd.DataFrame, train_edges: pd.DataFrame
) -> tuple[np.ndarray, dict]:
    """Baseline score from the scored partition only, never from holdout edges."""

    lncrna_degree = train_edges.lncrna_id.astype(str).value_counts()
    protein_degree = train_edges.protein_id.astype(str).value_counts()
    left = universe.lncrna_id.astype(str).map(lncrna_degree).fillna(0).to_numpy(float)
    right = universe.protein_id.astype(str).map(protein_degree).fillna(0).to_numpy(float)
    return left + right, {
        "definition": "train_partition_lncrna_degree_plus_protein_degree",
        "uses_holdout_edges": False,
        "train_edges": int(len(train_edges)),
    }


def _precision_at(order: np.ndarray, labels: np.ndarray, budget: int) -> float:
    top = order[: min(budget, len(order))]
    return float(labels[top].sum() / max(1, len(top)))


def _recall_at(order: np.ndarray, labels: np.ndarray, budget: int) -> float:
    total = int(labels.sum())
    if total == 0:
        raise RuntimeError("DISCOVERY_HOLDOUT_EMPTY")
    top = order[: min(budget, len(order))]
    return float(labels[top].sum() / total)


def evaluate(
    scored: pd.DataFrame,
    holdout_edges: pd.DataFrame,
    train_edges: pd.DataFrame,
    config: DiscoveryConfig,
) -> dict:
    """Fixed-budget held-out evaluation with the degree-prior baseline."""

    if not len(holdout_edges):
        raise RuntimeError("DISCOVERY_HOLDOUT_EMPTY")
    key = scored.set_index(list(UNIVERSE_KEY)).index
    holdout_pairs = set(
        zip(
            holdout_edges.lncrna_id.astype(str),
            holdout_edges.protein_id.astype(str),
            strict=True,
        )
    )
    train_pairs = set(
        zip(
            train_edges.lncrna_id.astype(str),
            train_edges.protein_id.astype(str),
            strict=True,
        )
    )
    overlap = holdout_pairs & train_pairs
    if overlap:
        raise RuntimeError(f"DISCOVERY_PARTITION_OVERLAP: {len(overlap)}")
    labels = np.asarray([pair in holdout_pairs for pair in key], dtype=bool)
    if not labels.any():
        raise RuntimeError("DISCOVERY_HOLDOUT_EMPTY")
    scores = scored.score.to_numpy(float)
    if not np.isfinite(scores).all():
        raise RuntimeError("DISCOVERY_SCORES_NOT_FINITE")
    prior, prior_record = degree_prior(scored[list(UNIVERSE_KEY)], train_edges)
    budget = int(config.budget)
    if len(scored) < budget:
        raise RuntimeError(f"DISCOVERY_BUDGET_EXCEEDS_UNIVERSE: {len(scored)} < {budget}")
    order = np.argsort(-scores, kind="stable")
    prior_order = np.argsort(-prior, kind="stable")
    ap = _average_precision(scores, labels)
    prior_ap = _average_precision(prior, labels)
    record = {
        "format": FORMAT,
        "scope": SCOPE,
        "budget": budget,
        "universe": int(len(scored)),
        "holdout_positives": int(labels.sum()),
        "holdout_prevalence": float(labels.mean()),
        "precision_at_budget": _precision_at(order, labels, budget),
        "recall_at_budget": _recall_at(order, labels, budget),
        "average_precision": ap,
        "degree_prior": {
            **prior_record,
            "precision_at_budget": _precision_at(prior_order, labels, budget),
            "recall_at_budget": _recall_at(prior_order, labels, budget),
            "average_precision": prior_ap,
        },
        "lift_average_precision": ap - prior_ap,
        "lift_precision_at_budget": _precision_at(order, labels, budget)
        - _precision_at(prior_order, labels, budget),
        "score_is_fitted": False,
        "score_uses_holdout_edges": False,
        "known_train_edges_excluded_from_universe": True,
        "candidate_status": CANDIDATE_STATUS,
        "discovery_claim": False,
        "biological_validation": False,
        "config_digest": config.digest(),
        "seed": int(config.seed),
    }
    return record


def top_candidates(scored: pd.DataFrame, config: DiscoveryConfig) -> pd.DataFrame:
    ordered = scored.sort_values("score", ascending=False, kind="stable").head(
        int(config.budget)
    )
    output = ordered[
        list(UNIVERSE_KEY)
        + ["score", "missing_features"]
        + [f"contribution::{feature}" for feature in config.feature_weights]
    ].copy()
    output["rank"] = np.arange(1, len(output) + 1)
    output["candidate_status"] = CANDIDATE_STATUS
    output["evidence_required"] = "orthogonal_experimental_validation"
    return output.reset_index(drop=True)


def discover_candidates(
    known_edges: pd.DataFrame,
    features: pd.DataFrame,
    config: DiscoveryConfig,
) -> tuple[pd.DataFrame, dict]:
    """Full pipeline: split, universe, score, evaluate, emit unvalidated rows."""

    record = validate_config(config)
    if "evidence_arm" in known_edges.columns:
        arm = known_edges.evidence_arm.astype(str)
        known_edges = known_edges.loc[arm.eq(str(config.evidence_arm))]
    split = split_known_edges(known_edges, config)
    train = split.loc[split.partition.eq("scored"), list(KNOWN_EDGE_COLUMNS)]
    holdout = split.loc[split.partition.eq("holdout"), list(KNOWN_EDGE_COLUMNS)]
    universe = candidate_universe(split, train)
    if len(universe) < int(config.budget):
        raise RuntimeError(
            f"DISCOVERY_BUDGET_EXCEEDS_UNIVERSE: {len(universe)} < {int(config.budget)}"
        )
    scored = score_candidates(universe, features, config)
    evaluation = evaluate(scored, holdout, train, config)
    candidates = top_candidates(scored, config)
    record.update(
        evaluation,
        scored_known_edges=int(len(train)),
        held_out_known_edges=int(len(holdout)),
        candidates_emitted=int(len(candidates)),
        missing_features=scored.attrs.get("missing_counts", {}),
        candidate_columns=list(candidates.columns),
        held_out_edges_used_for_scoring=False,
        notes=(
            "Candidates are ranked by a declared additive score with frozen standardisation. "
            "The held-out edges are used only for evaluation, never for features or weights. "
            "A ranked candidate is not evidence of binding."
        ),
    )
    return candidates, record


__all__ = [
    "CANDIDATE_STATUS",
    "DiscoveryConfig",
    "FORMAT",
    "SCOPE",
    "candidate_universe",
    "degree_prior",
    "discover_candidates",
    "evaluate",
    "score_candidates",
    "split_known_edges",
    "top_candidates",
    "validate_config",
]
