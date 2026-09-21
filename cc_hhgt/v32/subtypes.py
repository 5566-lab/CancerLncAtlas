"""Rank-based, downstream-only cancer-context subtype discovery for V3.2.

The classifier consumes already materialized exact-pathway Gene Sets.  It does
not feed subtype labels back into CC-HHGT, so the reported clusters cannot be a
circular training feature.  The shared pan-cancer lncRNA universe is used for
the primary analysis; cancer-local members remain available in the Gene Set
table but are excluded here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import combinations
from typing import Any, Hashable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score


class SubtypeContractError(ValueError):
    """Raised when ranked Gene Sets violate the downstream subtype contract."""


@dataclass(frozen=True)
class RankedSubtypeConfig:
    """Frozen V3.2 rank/composition clustering settings."""

    top_n: int = 200
    rbo_p: float = 0.98
    rbo_weight: float = 0.5
    weighted_jaccard_weight: float = 0.5
    k_min: int = 1
    k_max: int = 4
    program_k_max: int = 6
    silhouette_threshold: float = 0.25
    bootstrap_ari_threshold: float = 0.75
    bootstrap_replicates: int = 200
    min_cancers: int = 6
    min_members: int = 10
    min_cluster_size: int = 3
    conservation_similarity_threshold: float = 0.60
    conservation_ci_lower_threshold: float = 0.50
    program_raw_weight: float = 0.70
    program_subtype_weight: float = 0.30
    min_program_pathways: int = 1
    random_seed: int = 20260726

    def validate(self) -> None:
        if self.top_n < 1:
            raise SubtypeContractError("top_n must be positive")
        if not 0.0 < self.rbo_p < 1.0:
            raise SubtypeContractError("rbo_p must be within (0, 1)")
        if not np.isclose(
            self.rbo_weight + self.weighted_jaccard_weight, 1.0
        ):
            raise SubtypeContractError("RBO and weighted-Jaccard weights must sum to 1")
        if min(self.rbo_weight, self.weighted_jaccard_weight) < 0:
            raise SubtypeContractError("similarity weights cannot be negative")
        if self.k_min != 1 or self.k_max < self.k_min:
            raise SubtypeContractError("V3.2 subtype K search must start at K=1")
        if self.k_max != 4:
            raise SubtypeContractError("V3.2 subtype K search is frozen at K=1..4")
        if self.program_k_max != 6:
            raise SubtypeContractError(
                "V3.2 global cancer-program K search is frozen at K=2..6"
            )
        if self.min_cancers < 2 or self.min_members < 1:
            raise SubtypeContractError("invalid minimum cancers or members")
        if self.min_cluster_size < 2:
            raise SubtypeContractError("min_cluster_size must be at least two")
        if self.bootstrap_replicates < 1:
            raise SubtypeContractError("bootstrap_replicates must be positive")
        for name, value in (
            ("silhouette_threshold", self.silhouette_threshold),
            ("bootstrap_ari_threshold", self.bootstrap_ari_threshold),
            ("conservation_similarity_threshold", self.conservation_similarity_threshold),
            (
                "conservation_ci_lower_threshold",
                self.conservation_ci_lower_threshold,
            ),
        ):
            if not 0.0 <= value <= 1.0:
                raise SubtypeContractError(f"{name} must be within [0, 1]")
        if not np.isclose(self.program_raw_weight + self.program_subtype_weight, 1.0):
            raise SubtypeContractError("program similarity weights must sum to 1")


@dataclass(frozen=True)
class _Profile:
    cancer_id: str
    pathway_id: str
    pathway_family_id: str
    direction_lists: Mapping[str, tuple[str, ...]]
    weights: Mapping[str, float]
    n_members: int


@dataclass(frozen=True)
class _ClusterDecision:
    labels: np.ndarray
    selected_k: int
    proposed_k: int
    silhouette: float
    bootstrap_ari: float
    accepted: bool


@dataclass(frozen=True)
class RankedSubtypeResult:
    """All downstream subtype outputs, kept separate from model predictions."""

    pathway_context_subtype: pd.DataFrame
    pathway_conservation: pd.DataFrame
    pathway_pairwise_similarity: pd.DataFrame
    cancer_program_subtype: pd.DataFrame
    cancer_program_similarity: pd.DataFrame


def rank_biased_overlap(
    left: Sequence[Hashable],
    right: Sequence[Hashable],
    *,
    p: float = 0.98,
    depth: int | None = None,
) -> float:
    """Finite extrapolated rank-biased overlap (RBO).

    Duplicate identifiers are forbidden because ranked Gene Sets contain one
    row per lncRNA and direction.  Identical lists return exactly one, while
    disjoint lists return zero.
    """

    if not 0.0 < p < 1.0:
        raise SubtypeContractError("RBO p must be within (0, 1)")
    left_values = list(left)
    right_values = list(right)
    if len(set(left_values)) != len(left_values) or len(set(right_values)) != len(
        right_values
    ):
        raise SubtypeContractError("RBO inputs cannot contain duplicate identifiers")
    maximum = max(len(left_values), len(right_values))
    if depth is None:
        depth = maximum
    depth = min(int(depth), maximum)
    if depth <= 0:
        return 1.0 if not left_values and not right_values else 0.0

    left_seen: set[Hashable] = set()
    right_seen: set[Hashable] = set()
    weighted = 0.0
    agreement = 0.0
    for position in range(1, depth + 1):
        if position <= len(left_values):
            left_seen.add(left_values[position - 1])
        if position <= len(right_values):
            right_seen.add(right_values[position - 1])
        agreement = len(left_seen & right_seen) / float(position)
        weighted += (1.0 - p) * agreement * (p ** (position - 1))
    score = weighted + agreement * (p**depth)
    return float(np.clip(score, 0.0, 1.0))


def weighted_jaccard(
    left: Mapping[Hashable, float], right: Mapping[Hashable, float]
) -> float:
    """Probability-weighted Jaccard similarity over directional lncRNA tokens."""

    keys = set(left) | set(right)
    if not keys:
        return 1.0
    left_values = {key: float(left.get(key, 0.0)) for key in keys}
    right_values = {key: float(right.get(key, 0.0)) for key in keys}
    if any(not np.isfinite(value) or value < 0 for value in left_values.values()):
        raise SubtypeContractError("weighted-Jaccard weights must be finite and nonnegative")
    if any(not np.isfinite(value) or value < 0 for value in right_values.values()):
        raise SubtypeContractError("weighted-Jaccard weights must be finite and nonnegative")
    denominator = sum(max(left_values[key], right_values[key]) for key in keys)
    if denominator == 0:
        return 1.0
    numerator = sum(min(left_values[key], right_values[key]) for key in keys)
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def _stable_seed(base: int, *parts: str) -> int:
    payload = "|".join([str(base), *map(str, parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def _canonical_scope(value: Any) -> str:
    key = str(value).strip().lower()
    if key in {"shared", "pan_cancer_shared", "pancancer_shared", "shared_eligible"}:
        return "shared"
    if key in {"local", "local_only", "cancer_local", "cancer-local"}:
        return "cancer_local"
    raise SubtypeContractError(f"invalid shared_or_local_scope: {value!r}")


def _validate_members(members: pd.DataFrame) -> str | None:
    required = {
        "cancer_id",
        "lncrna_id",
        "pathway_id",
        "pathway_family_id",
        "association_membership_probability",
        "association_direction",
    }
    if not isinstance(members, pd.DataFrame) or members.empty:
        raise SubtypeContractError("ranked Gene Set members must be non-empty")
    missing = sorted(required - set(members.columns))
    if missing:
        raise SubtypeContractError(f"ranked Gene Sets lack columns: {missing}")
    for target_column in ("pathway_target_level", "target_level"):
        if target_column in members:
            levels = set(members[target_column].dropna().astype(str).str.lower())
            if levels != {"exact_pathway"}:
                raise SubtypeContractError(
                    "pathway-context subtypes require exact_pathway targets"
                )
    for column in ("cancer_id", "lncrna_id", "pathway_id", "pathway_family_id"):
        values = members[column].astype("string")
        if values.isna().any() or values.str.strip().eq("").any():
            raise SubtypeContractError(f"{column} contains empty identifiers")
    if members["pathway_id"].astype(str).eq(
        members["pathway_family_id"].astype(str)
    ).any():
        raise SubtypeContractError("pathway family cannot replace exact pathway")
    family_count = members.groupby("pathway_id", observed=True)[
        "pathway_family_id"
    ].nunique(dropna=False)
    if family_count.gt(1).any():
        raise SubtypeContractError("an exact pathway maps to multiple families")
    probabilities = pd.to_numeric(
        members["association_membership_probability"], errors="coerce"
    )
    if (
        not np.isfinite(probabilities).all()
        or probabilities.lt(0).any()
        or probabilities.gt(1).any()
    ):
        raise SubtypeContractError("association probabilities must be finite [0, 1]")
    rank_columns = [column for column in ("geneset_rank", "rank") if column in members]
    if len(rank_columns) > 1:
        left = pd.to_numeric(members[rank_columns[0]], errors="coerce")
        right = pd.to_numeric(members[rank_columns[1]], errors="coerce")
        if not left.equals(right):
            raise SubtypeContractError("geneset_rank and rank disagree")
    return rank_columns[0] if rank_columns else None


def _build_profiles(
    members: pd.DataFrame, config: RankedSubtypeConfig
) -> dict[str, list[_Profile]]:
    rank_column = _validate_members(members)
    work = members.copy()
    for column in ("cancer_id", "lncrna_id", "pathway_id", "pathway_family_id"):
        work[column] = work[column].astype(str)
    work["association_membership_probability"] = pd.to_numeric(
        work["association_membership_probability"], errors="raise"
    ).astype(float)
    work["association_direction"] = (
        work["association_direction"].astype(str).str.strip().str.lower()
    )
    if not set(work["association_direction"]).issubset({"positive", "negative"}):
        raise SubtypeContractError("association_direction must be positive or negative")
    profile_universe = (
        work[["pathway_id", "cancer_id", "pathway_family_id"]]
        .drop_duplicates()
        .sort_values(["pathway_id", "cancer_id"], kind="stable")
    )
    if "shared_or_local_scope" in work:
        work["_scope"] = work["shared_or_local_scope"].map(_canonical_scope)
        work = work.loc[work["_scope"].eq("shared")].copy()
    duplicate_key = ["cancer_id", "pathway_id", "association_direction", "lncrna_id"]
    if work.duplicated(duplicate_key).any():
        raise SubtypeContractError("ranked Gene Sets contain duplicate directional members")
    if rank_column:
        work["_rank"] = pd.to_numeric(work[rank_column], errors="coerce")
        if not np.isfinite(work["_rank"]).all() or work["_rank"].lt(1).any():
            raise SubtypeContractError("Gene Set ranks must be finite positive values")
    else:
        work = work.sort_values(
            [
                "cancer_id",
                "pathway_id",
                "association_direction",
                "association_membership_probability",
                "lncrna_id",
            ],
            ascending=[True, True, True, False, True],
            kind="stable",
        )
        work["_rank"] = (
            work.groupby(
                ["cancer_id", "pathway_id", "association_direction"],
                observed=True,
                sort=False,
            ).cumcount()
            + 1
        )
    work = work.sort_values(
        [
            "cancer_id",
            "pathway_id",
            "association_direction",
            "_rank",
            "lncrna_id",
        ],
        kind="stable",
    )
    work = work.groupby(
        ["cancer_id", "pathway_id", "association_direction"],
        observed=True,
        sort=False,
    ).head(config.top_n)

    profiles: dict[str, list[_Profile]] = {}
    shared_groups = {
        (str(pathway), str(cancer)): group
        for (pathway, cancer), group in work.groupby(
            ["pathway_id", "cancer_id"], observed=True, sort=True
        )
    }
    for universe_row in profile_universe.itertuples(index=False):
        pathway = str(universe_row.pathway_id)
        cancer = str(universe_row.cancer_id)
        family = str(universe_row.pathway_family_id)
        group = shared_groups.get((pathway, cancer), work.iloc[0:0])
        directional: dict[str, tuple[str, ...]] = {}
        weights: dict[str, float] = {}
        for direction, direction_group in group.groupby(
            "association_direction", observed=True, sort=True
        ):
            ordered = direction_group.sort_values(["_rank", "lncrna_id"], kind="stable")
            tokens: list[str] = []
            for row in ordered.itertuples(index=False):
                token = f"{direction}\x1f{row.lncrna_id}"
                tokens.append(token)
                weights[token] = float(row.association_membership_probability)
            directional[str(direction)] = tuple(tokens)
        profile = _Profile(
            cancer_id=str(cancer),
            pathway_id=str(pathway),
            pathway_family_id=family,
            direction_lists=directional,
            weights=weights,
            n_members=len(weights),
        )
        profiles.setdefault(str(pathway), []).append(profile)
    for pathway in profiles:
        profiles[pathway] = sorted(profiles[pathway], key=lambda item: item.cancer_id)
    return profiles


def _profile_similarity(
    left: _Profile, right: _Profile, config: RankedSubtypeConfig
) -> tuple[float, float, float]:
    directions = sorted(set(left.direction_lists) | set(right.direction_lists))
    directional_rbo = [
        rank_biased_overlap(
            left.direction_lists.get(direction, ()),
            right.direction_lists.get(direction, ()),
            p=config.rbo_p,
            depth=config.top_n,
        )
        for direction in directions
    ]
    rbo = float(np.mean(directional_rbo)) if directional_rbo else 1.0
    jaccard = weighted_jaccard(left.weights, right.weights)
    combined = config.rbo_weight * rbo + config.weighted_jaccard_weight * jaccard
    return float(combined), rbo, jaccard


def _similarity_matrix(
    profiles: Sequence[_Profile], config: RankedSubtypeConfig
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    size = len(profiles)
    matrix = np.eye(size, dtype=float)
    pair_rows: list[dict[str, Any]] = []
    for left_index, right_index in combinations(range(size), 2):
        combined, rbo, jaccard = _profile_similarity(
            profiles[left_index], profiles[right_index], config
        )
        matrix[left_index, right_index] = matrix[right_index, left_index] = combined
        pair_rows.append(
            {
                "pathway_id": profiles[left_index].pathway_id,
                "pathway_family_id": profiles[left_index].pathway_family_id,
                "cancer_a": profiles[left_index].cancer_id,
                "cancer_b": profiles[right_index].cancer_id,
                "rbo_similarity": rbo,
                "weighted_jaccard_similarity": jaccard,
                "combined_similarity": combined,
            }
        )
    return matrix, pair_rows


def _bootstrap_profiles(
    profiles: Sequence[_Profile], rng: np.random.Generator
) -> list[_Profile]:
    lncrnas = sorted(
        {
            token.split("\x1f", 1)[1]
            for profile in profiles
            for token in profile.weights
        }
    )
    if not lncrnas:
        return list(profiles)
    sampled = rng.choice(np.asarray(lncrnas, dtype=object), size=len(lncrnas), replace=True)
    multiplicity: dict[str, int] = {}
    for value in sampled.tolist():
        multiplicity[str(value)] = multiplicity.get(str(value), 0) + 1
    bootstrapped: list[_Profile] = []
    for profile in profiles:
        directional: dict[str, tuple[str, ...]] = {}
        weights: dict[str, float] = {}
        for direction, tokens in sorted(profile.direction_lists.items()):
            expanded: list[str] = []
            for token in tokens:
                lncrna = token.split("\x1f", 1)[1]
                for copy_index in range(multiplicity.get(lncrna, 0)):
                    boot_token = f"{token}\x1e{copy_index}"
                    expanded.append(boot_token)
                    weights[boot_token] = profile.weights[token]
            directional[direction] = tuple(expanded)
        bootstrapped.append(
            _Profile(
                cancer_id=profile.cancer_id,
                pathway_id=profile.pathway_id,
                pathway_family_id=profile.pathway_family_id,
                direction_lists=directional,
                weights=weights,
                n_members=len(weights),
            )
        )
    return bootstrapped


def _distance(similarity: np.ndarray) -> np.ndarray:
    symmetric = (np.asarray(similarity, dtype=float) + np.asarray(similarity).T) / 2.0
    symmetric = np.clip(symmetric, 0.0, 1.0)
    distance = 1.0 - symmetric
    np.fill_diagonal(distance, 0.0)
    return distance


def _fit_labels(similarity: np.ndarray, k: int) -> np.ndarray:
    if k == 1:
        return np.zeros(similarity.shape[0], dtype=int)
    return AgglomerativeClustering(
        n_clusters=k,
        metric="precomputed",
        linkage="average",
    ).fit_predict(_distance(similarity))


def _canonicalise_labels(labels: np.ndarray, names: Sequence[str]) -> np.ndarray:
    clusters: list[tuple[str, int]] = []
    for label in sorted(set(map(int, labels))):
        members = sorted(name for name, value in zip(names, labels) if int(value) == label)
        clusters.append((members[0], label))
    mapping = {old: new for new, (_, old) in enumerate(sorted(clusters))}
    return np.asarray([mapping[int(value)] for value in labels], dtype=int)


def _mean_off_diagonal(matrix: np.ndarray) -> float:
    if matrix.shape[0] < 2:
        return 1.0
    return float(matrix[np.triu_indices(matrix.shape[0], k=1)].mean())


def _choose_clusters(
    similarity: np.ndarray,
    names: Sequence[str],
    bootstrap_matrices: Sequence[np.ndarray],
    config: RankedSubtypeConfig,
    *,
    k_max: int | None = None,
) -> _ClusterDecision:
    n_items = similarity.shape[0]
    candidates: list[tuple[float, int, np.ndarray]] = []
    maximum_k = min(
        config.k_max if k_max is None else int(k_max),
        n_items // config.min_cluster_size,
    )
    for k in range(max(2, config.k_min), maximum_k + 1):
        labels = _fit_labels(similarity, k)
        counts = np.bincount(labels, minlength=k)
        if counts.min() < config.min_cluster_size:
            continue
        score = float(silhouette_score(_distance(similarity), labels, metric="precomputed"))
        candidates.append((score, k, labels))
    if not candidates:
        return _ClusterDecision(
            labels=np.zeros(n_items, dtype=int),
            selected_k=1,
            proposed_k=1,
            silhouette=0.0,
            bootstrap_ari=1.0,
            accepted=False,
        )
    score, proposed_k, labels = sorted(candidates, key=lambda item: (-item[0], item[1]))[0]
    canonical = _canonicalise_labels(labels, names)
    if score < config.silhouette_threshold:
        return _ClusterDecision(
            labels=np.zeros(n_items, dtype=int),
            selected_k=1,
            proposed_k=proposed_k,
            silhouette=score,
            bootstrap_ari=0.0,
            accepted=False,
        )
    ari_values: list[float] = []
    for bootstrap in bootstrap_matrices:
        boot_labels = _fit_labels(bootstrap, proposed_k)
        ari_values.append(float(adjusted_rand_score(canonical, boot_labels)))
    stability = float(np.mean(ari_values)) if ari_values else 0.0
    accepted = stability >= config.bootstrap_ari_threshold
    return _ClusterDecision(
        labels=canonical if accepted else np.zeros(n_items, dtype=int),
        selected_k=proposed_k if accepted else 1,
        proposed_k=proposed_k,
        silhouette=score,
        bootstrap_ari=stability,
        accepted=accepted,
    )


def classify_pathway_context_subtypes(
    members: pd.DataFrame,
    config: RankedSubtypeConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Classify cancers separately for every exact pathway.

    Returns ``(context_rows, conservation_rows, pairwise_rows)``.  Every input
    cancer/pathway gets a context row; low-coverage rows are explicitly marked
    ``UNAVAILABLE`` rather than silently converted to a zero or a singleton.
    """

    config = config or RankedSubtypeConfig()
    config.validate()
    profiles_by_pathway = _build_profiles(members, config)
    context_rows: list[dict[str, Any]] = []
    conservation_rows: list[dict[str, Any]] = []
    all_pairs: list[dict[str, Any]] = []

    for pathway in sorted(profiles_by_pathway):
        all_profiles = profiles_by_pathway[pathway]
        evaluable = [item for item in all_profiles if item.n_members >= config.min_members]
        family = all_profiles[0].pathway_family_id
        if len(evaluable) < config.min_cancers:
            for profile in all_profiles:
                context_rows.append(
                    {
                        "cancer_id": profile.cancer_id,
                        "pathway_id": pathway,
                        "pathway_family_id": family,
                        "n_shared_members": profile.n_members,
                        "pathway_context_subtype_id": "UNAVAILABLE",
                        "pathway_context_subtype": "UNAVAILABLE",
                        "classification_status": "UNAVAILABLE",
                        "selected_k": 0,
                        "proposed_k": 0,
                        "silhouette": np.nan,
                        "bootstrap_ari": np.nan,
                        "mean_pairwise_similarity": np.nan,
                    }
                )
            conservation_rows.append(
                {
                    "pathway_id": pathway,
                    "pathway_family_id": family,
                    "n_evaluable_cancers": len(evaluable),
                    "mean_pairwise_similarity": np.nan,
                    "bootstrap_ci_lower": np.nan,
                    "bootstrap_ci_upper": np.nan,
                    "selected_k": 0,
                    "proposed_k": 0,
                    "silhouette": np.nan,
                    "bootstrap_ari": np.nan,
                    "classification_status": "UNAVAILABLE",
                }
            )
            continue

        similarity, pair_rows = _similarity_matrix(evaluable, config)
        all_pairs.extend(pair_rows)
        rng = np.random.default_rng(_stable_seed(config.random_seed, pathway))
        bootstrap_matrices: list[np.ndarray] = []
        bootstrap_means: list[float] = []
        for _ in range(config.bootstrap_replicates):
            boot_profiles = _bootstrap_profiles(evaluable, rng)
            boot_similarity, _ = _similarity_matrix(boot_profiles, config)
            bootstrap_matrices.append(boot_similarity)
            bootstrap_means.append(_mean_off_diagonal(boot_similarity))
        cancer_names = [item.cancer_id for item in evaluable]
        decision = _choose_clusters(
            similarity, cancer_names, bootstrap_matrices, config
        )
        mean_similarity = _mean_off_diagonal(similarity)
        ci_lower, ci_upper = np.quantile(bootstrap_means, [0.025, 0.975])
        if decision.selected_k > 1:
            status = "STABLE_SUBTYPES"
        elif (
            mean_similarity >= config.conservation_similarity_threshold
            and ci_lower >= config.conservation_ci_lower_threshold
        ):
            status = "CONSERVED_K1"
        else:
            status = "CONTINUOUS_HETEROGENEITY_K1"
        labels = dict(zip(cancer_names, decision.labels.tolist()))
        for profile in all_profiles:
            if profile.cancer_id not in labels:
                row_status = "UNAVAILABLE"
                subtype_id = "UNAVAILABLE"
                row_k = 0
            else:
                row_status = status
                subtype_id = f"PATHWAY_CONTEXT:{pathway}:S{labels[profile.cancer_id] + 1}"
                row_k = decision.selected_k
            context_rows.append(
                {
                    "cancer_id": profile.cancer_id,
                    "pathway_id": pathway,
                    "pathway_family_id": family,
                    "n_shared_members": profile.n_members,
                    "pathway_context_subtype_id": subtype_id,
                    "pathway_context_subtype": subtype_id,
                    "classification_status": row_status,
                    "selected_k": row_k,
                    "proposed_k": decision.proposed_k if row_k else 0,
                    "silhouette": decision.silhouette if row_k else np.nan,
                    "bootstrap_ari": decision.bootstrap_ari if row_k else np.nan,
                    "mean_pairwise_similarity": mean_similarity if row_k else np.nan,
                }
            )
        conservation_rows.append(
            {
                "pathway_id": pathway,
                "pathway_family_id": family,
                "n_evaluable_cancers": len(evaluable),
                "mean_pairwise_similarity": mean_similarity,
                "bootstrap_ci_lower": float(ci_lower),
                "bootstrap_ci_upper": float(ci_upper),
                "selected_k": decision.selected_k,
                "proposed_k": decision.proposed_k,
                "silhouette": decision.silhouette,
                "bootstrap_ari": decision.bootstrap_ari,
                "classification_status": status,
            }
        )

    context = pd.DataFrame(context_rows).sort_values(
        ["pathway_id", "cancer_id"], kind="stable"
    ).reset_index(drop=True)
    conservation = pd.DataFrame(conservation_rows).sort_values(
        "pathway_id", kind="stable"
    ).reset_index(drop=True)
    pairwise_columns = [
        "pathway_id",
        "pathway_family_id",
        "cancer_a",
        "cancer_b",
        "rbo_similarity",
        "weighted_jaccard_similarity",
        "combined_similarity",
    ]
    pairwise = pd.DataFrame(all_pairs, columns=pairwise_columns).sort_values(
        ["pathway_id", "cancer_a", "cancer_b"], kind="stable"
    ).reset_index(drop=True)
    return context, conservation, pairwise


def _program_similarity_matrix(
    cancers: Sequence[str],
    pairwise: pd.DataFrame,
    context: pd.DataFrame,
    conservation: pd.DataFrame,
    selected_pathways: Sequence[str],
    config: RankedSubtypeConfig,
) -> tuple[np.ndarray, dict[tuple[str, str], int]]:
    index = {cancer: position for position, cancer in enumerate(cancers)}
    matrix = np.eye(len(cancers), dtype=float)
    coverage: dict[tuple[str, str], int] = {}
    stable_paths = set(
        conservation.loc[conservation["selected_k"].gt(1), "pathway_id"].astype(str)
    )
    subtype_lookup = {
        (str(row.pathway_id), str(row.cancer_id)): str(row.pathway_context_subtype_id)
        for row in context.itertuples(index=False)
        if str(row.pathway_context_subtype_id) != "UNAVAILABLE"
    }
    pair_lookup = {
        (str(row.pathway_id), str(row.cancer_a), str(row.cancer_b)): (
            str(row.pathway_family_id),
            float(row.combined_similarity),
        )
        for row in pairwise.itertuples(index=False)
    }
    for left, right in combinations(cancers, 2):
        by_family: dict[str, list[float]] = {}
        used = 0
        for pathway in selected_pathways:
            key = (str(pathway), min(left, right), max(left, right))
            record = pair_lookup.get(key)
            if record is None:
                continue
            family, raw_similarity = record
            value = raw_similarity
            if pathway in stable_paths:
                left_label = subtype_lookup.get((pathway, left))
                right_label = subtype_lookup.get((pathway, right))
                if left_label is not None and right_label is not None:
                    match = float(left_label == right_label)
                    value = (
                        config.program_raw_weight * raw_similarity
                        + config.program_subtype_weight * match
                    )
            by_family.setdefault(family, []).append(value)
            used += 1
        coverage[(left, right)] = used
        if by_family:
            family_means = [float(np.mean(values)) for _, values in sorted(by_family.items())]
            score = float(np.mean(family_means))
        else:
            score = np.nan
        left_index, right_index = index[left], index[right]
        matrix[left_index, right_index] = matrix[right_index, left_index] = score
    missing = ~np.isfinite(matrix)
    if missing.any():
        observed = matrix[np.isfinite(matrix) & ~np.eye(len(cancers), dtype=bool)]
        fill = float(observed.mean()) if observed.size else 0.0
        matrix[missing] = fill
    return np.clip(matrix, 0.0, 1.0), coverage


def classify_cancer_program_subtypes(
    pathway_context: pd.DataFrame,
    pathway_conservation: pd.DataFrame,
    pathway_pairwise_similarity: pd.DataFrame,
    config: RankedSubtypeConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate exact-pathway results into optional global cancer programs."""

    config = config or RankedSubtypeConfig()
    config.validate()
    required_context = {
        "cancer_id",
        "pathway_id",
        "pathway_context_subtype_id",
        "classification_status",
    }
    required_conservation = {"pathway_id", "selected_k"}
    required_pairwise = {
        "pathway_id",
        "pathway_family_id",
        "cancer_a",
        "cancer_b",
        "combined_similarity",
    }
    if not required_context.issubset(pathway_context.columns):
        raise SubtypeContractError("pathway-context subtype schema is incomplete")
    if not required_conservation.issubset(pathway_conservation.columns):
        raise SubtypeContractError("pathway conservation schema is incomplete")
    if not required_pairwise.issubset(pathway_pairwise_similarity.columns):
        raise SubtypeContractError("pathway pairwise similarity schema is incomplete")

    available = pathway_context.loc[
        pathway_context["pathway_context_subtype_id"].astype(str).ne("UNAVAILABLE")
    ].copy()
    pathway_counts = available.groupby("cancer_id", observed=True)[
        "pathway_id"
    ].nunique()
    all_cancers = sorted(pathway_context["cancer_id"].astype(str).unique())
    evaluable_cancers = sorted(
        pathway_counts[pathway_counts.ge(config.min_program_pathways)].index.astype(str)
    )
    if len(evaluable_cancers) < config.min_cancers:
        rows = [
            {
                "cancer_id": cancer,
                "cancer_program_subtype_id": "UNAVAILABLE",
                "cancer_program_subtype": "UNAVAILABLE",
                "classification_status": "UNAVAILABLE",
                "n_evaluable_pathways": int(pathway_counts.get(cancer, 0)),
                "selected_k": 0,
                "proposed_k": 0,
                "silhouette": np.nan,
                "bootstrap_ari": np.nan,
            }
            for cancer in all_cancers
        ]
        return pd.DataFrame(rows), pd.DataFrame(
            columns=["cancer_a", "cancer_b", "program_similarity", "n_shared_pathways"]
        )

    pathways = sorted(pathway_pairwise_similarity["pathway_id"].astype(str).unique())
    matrix, coverage = _program_similarity_matrix(
        evaluable_cancers,
        pathway_pairwise_similarity,
        pathway_context,
        pathway_conservation,
        pathways,
        config,
    )
    rng = np.random.default_rng(_stable_seed(config.random_seed, "CANCER_PROGRAM"))
    boot_matrices: list[np.ndarray] = []
    for _ in range(config.bootstrap_replicates):
        selected = rng.choice(
            np.asarray(pathways, dtype=object), size=len(pathways), replace=True
        ).tolist()
        boot_matrix, _ = _program_similarity_matrix(
            evaluable_cancers,
            pathway_pairwise_similarity,
            pathway_context,
            pathway_conservation,
            [str(value) for value in selected],
            config,
        )
        boot_matrices.append(boot_matrix)
    decision = _choose_clusters(
        matrix,
        evaluable_cancers,
        boot_matrices,
        config,
        k_max=config.program_k_max,
    )
    labels = dict(zip(evaluable_cancers, decision.labels.tolist()))
    status = "STABLE_PROGRAM_SUBTYPES" if decision.selected_k > 1 else "ONE_PROGRAM_CLUSTER_K1"
    rows: list[dict[str, Any]] = []
    for cancer in all_cancers:
        if cancer not in labels:
            rows.append(
                {
                    "cancer_id": cancer,
                    "cancer_program_subtype_id": "UNAVAILABLE",
                    "cancer_program_subtype": "UNAVAILABLE",
                    "classification_status": "UNAVAILABLE",
                    "n_evaluable_pathways": int(pathway_counts.get(cancer, 0)),
                    "selected_k": 0,
                    "proposed_k": 0,
                    "silhouette": np.nan,
                    "bootstrap_ari": np.nan,
                }
            )
            continue
        subtype_id = f"CANCER_PROGRAM:S{labels[cancer] + 1}"
        rows.append(
            {
                "cancer_id": cancer,
                "cancer_program_subtype_id": subtype_id,
                "cancer_program_subtype": subtype_id,
                "classification_status": status,
                "n_evaluable_pathways": int(pathway_counts.get(cancer, 0)),
                "selected_k": decision.selected_k,
                "proposed_k": decision.proposed_k,
                "silhouette": decision.silhouette,
                "bootstrap_ari": decision.bootstrap_ari,
            }
        )
    pair_rows: list[dict[str, Any]] = []
    position = {cancer: index for index, cancer in enumerate(evaluable_cancers)}
    for left, right in combinations(evaluable_cancers, 2):
        pair_rows.append(
            {
                "cancer_a": left,
                "cancer_b": right,
                "program_similarity": float(matrix[position[left], position[right]]),
                "n_shared_pathways": int(coverage[(left, right)]),
            }
        )
    program = pd.DataFrame(rows).sort_values("cancer_id", kind="stable").reset_index(drop=True)
    similarity = pd.DataFrame(pair_rows).sort_values(
        ["cancer_a", "cancer_b"], kind="stable"
    ).reset_index(drop=True)
    return program, similarity


def infer_ranked_subtypes(
    members: pd.DataFrame,
    config: RankedSubtypeConfig | None = None,
) -> RankedSubtypeResult:
    """Run both downstream subtype layers without fitting or changing CC-HHGT."""

    config = config or RankedSubtypeConfig()
    pathway, conservation, pairwise = classify_pathway_context_subtypes(
        members, config
    )
    program, program_similarity = classify_cancer_program_subtypes(
        pathway, conservation, pairwise, config
    )
    return RankedSubtypeResult(
        pathway_context_subtype=pathway,
        pathway_conservation=conservation,
        pathway_pairwise_similarity=pairwise,
        cancer_program_subtype=program,
        cancer_program_similarity=program_similarity,
    )
