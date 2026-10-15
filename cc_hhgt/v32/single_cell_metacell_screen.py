"""Fold-local discovery screening and donor-resolved metacell targets.

The screen may use association effects from outer-training donors, but never
from an outer-query donor.  A CountSketch only proposes a bounded set of
lncRNA-pathway pairs; every retained source effect and every query target is
then recomputed exactly from within-donor metacell ranks.
"""
from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata, t as student_t


PAIR_KEYS = ("lncrna_id", "pathway_id")
TARGET_METHOD = "WITHIN_DONOR_METACELL_SPEARMAN"
SCREEN_POLICY = "OUTER_SOURCE_DONOR_COUNTSKETCH_TOPK_EXACT_REFINEMENT"


class MetacellScreenError(RuntimeError):
    """Raised when source/query separation or matrix identity is invalid."""


def donor_folds(cancer_id: str, donor_ids: Iterable[str], folds: int = 5,
                seed: int = 1729) -> pd.DataFrame:
    """Assign every donor once; cohorts with <5 donors use LODO-like folds."""
    donors = sorted({str(value).strip() for value in donor_ids})
    if len(donors) < 2 or any(not value for value in donors):
        raise MetacellScreenError("at least two non-empty donors are required")
    ordered = sorted(
        donors,
        key=lambda donor: hashlib.sha256(
            f"{seed}|{str(cancer_id).upper()}|{donor}".encode("utf-8")
        ).hexdigest(),
    )
    active = min(int(folds), len(ordered))
    return pd.DataFrame(
        {
            "cancer_id": str(cancer_id).upper(),
            "donor_id": ordered,
            "outer_fold_id": np.arange(len(ordered), dtype=np.int64) % active,
            "cohort_fold_count": active,
            "limited_replication": len(ordered) < int(folds),
        }
    )


def _dense(matrix) -> np.ndarray:
    values = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise MetacellScreenError("metacell matrix must be finite and two-dimensional")
    return values


def _standardized_ranks(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ranks = rankdata(values, axis=0).astype(np.float32, copy=False)
    ranks -= ranks.mean(axis=0, keepdims=True)
    norm = np.linalg.norm(ranks, axis=0)
    valid = norm > 1e-8
    ranks[:, valid] /= norm[valid]
    ranks[:, ~valid] = 0.0
    return ranks, valid


def _context_blocks(
    lnc_expression,
    pathway_activity,
    metadata: pd.DataFrame,
    source_donors: Iterable[str],
    *,
    min_metacells: int,
    min_detected: int,
    min_detection_fraction: float,
) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray]:
    x = _dense(lnc_expression)
    y = _dense(pathway_activity)
    if len(x) != len(y) or len(x) != len(metadata):
        raise MetacellScreenError("metacell matrices and metadata are misaligned")
    donors = metadata.donor_id.astype(str).to_numpy()
    allowed = {str(value) for value in source_donors}
    if not allowed:
        raise MetacellScreenError("at least one donor is required")
    allowed &= set(donors)
    blocks: list[dict[str, object]] = []
    lnc_support = np.zeros(x.shape[1], dtype=np.int32)
    pathway_support = np.zeros(y.shape[1], dtype=np.int32)
    for donor in sorted(allowed):
        rows = np.flatnonzero(donors == donor)
        if len(rows) < int(min_metacells):
            continue
        local_x = x[rows]
        local_y = y[rows]
        xr, x_variable = _standardized_ranks(local_x)
        yr, y_variable = _standardized_ranks(local_y)
        required = max(int(min_detected), int(np.ceil(len(rows) * min_detection_fraction)))
        x_detected = (local_x > 0).sum(axis=0) >= required
        x_valid = x_variable & x_detected
        lnc_support += x_valid
        pathway_support += y_variable
        blocks.append(
            {
                "donor_id": donor,
                "n_metacells": len(rows),
                "x_rank": xr,
                "y_rank": yr,
                "x_valid": x_valid,
                "y_valid": y_variable,
                "lnc_detection_rate": (local_x > 0).mean(axis=0).astype(np.float32),
            }
        )
    return blocks, lnc_support, pathway_support


def _countsketch(
    blocks: list[dict[str, object]],
    lnc_keep: np.ndarray,
    pathway_keep: np.ndarray,
    *,
    dimensions: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if dimensions < 16:
        raise MetacellScreenError("CountSketch dimensions must be at least 16")
    rng = np.random.default_rng(seed)
    sx = np.zeros((dimensions, len(lnc_keep)), dtype=np.float32)
    sy = np.zeros((dimensions, len(pathway_keep)), dtype=np.float32)
    donor_scale = float(np.sqrt(max(len(blocks), 1)))
    for block in blocks:
        x = block["x_rank"][:, lnc_keep] / donor_scale
        y = block["y_rank"][:, pathway_keep] / donor_scale
        bucket = rng.integers(0, dimensions, size=len(x))
        sign = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=len(x))
        np.add.at(sx, bucket, x * sign[:, None])
        np.add.at(sy, bucket, y * sign[:, None])
    return sx, sy


def _exact_pair_effects(
    blocks: list[dict[str, object]],
    lnc_index: np.ndarray,
    pathway_index: np.ndarray,
    *,
    batch_pairs: int = 8192,
) -> dict[str, np.ndarray]:
    count = len(lnc_index)
    rho_sum = np.zeros(count, dtype=np.float64)
    fisher_sum = np.zeros(count, dtype=np.float64)
    positive = np.zeros(count, dtype=np.int32)
    support = np.zeros(count, dtype=np.int32)
    donor_rhos: list[np.ndarray] = []
    donor_masks: list[np.ndarray] = []
    for block in blocks:
        valid = block["x_valid"][lnc_index] & block["y_valid"][pathway_index]
        rho = np.full(count, np.nan, dtype=np.float32)
        for start in range(0, count, batch_pairs):
            stop = min(start + batch_pairs, count)
            local_valid = valid[start:stop]
            if not local_valid.any():
                continue
            rows = np.flatnonzero(local_valid) + start
            rho[rows] = np.sum(
                block["x_rank"][:, lnc_index[rows]]
                * block["y_rank"][:, pathway_index[rows]],
                axis=0,
                dtype=np.float64,
            )
        clipped = np.clip(rho, -0.999999, 0.999999)
        rho_sum[valid] += rho[valid]
        fisher_sum[valid] += np.arctanh(clipped[valid])
        positive[valid] += rho[valid] > 0
        support[valid] += 1
        donor_rhos.append(rho)
        donor_masks.append(valid)
    mean_rho = np.divide(rho_sum, support, out=np.full(count, np.nan), where=support > 0)
    fisher_mean = np.divide(
        fisher_sum, support, out=np.full(count, np.nan), where=support > 0
    )
    meta_rho = np.tanh(fisher_mean)
    direction = np.divide(
        np.maximum(positive, support - positive),
        support,
        out=np.full(count, np.nan),
        where=support > 0,
    )
    p_value = np.full(count, np.nan)
    if donor_rhos:
        values = np.vstack(donor_rhos)
        for index in np.flatnonzero(support >= 2):
            local = np.arctanh(np.clip(values[:, index], -0.999999, 0.999999))
            local = local[np.isfinite(local)]
            sd = local.std(ddof=1)
            if sd <= 1e-12:
                p_value[index] = 0.0 if abs(local.mean()) > 1e-12 else 1.0
            else:
                statistic = local.mean() / (sd / np.sqrt(len(local)))
                p_value[index] = float(2.0 * student_t.sf(abs(statistic), len(local) - 1))
    return {
        "source_mean_rho": mean_rho,
        "source_meta_rho": meta_rho,
        "source_direction_consistency": direction,
        "source_donor_count": support,
        "source_meta_p_value": p_value,
    }


def screen_source_pairs(
    lnc_expression,
    pathway_activity,
    metadata: pd.DataFrame,
    lnc_ids: Iterable[str],
    pathway_ids: Iterable[str],
    source_donors: Iterable[str],
    *,
    top_k: int = 25,
    sketch_oversample: int = 4,
    sketch_dimensions: int = 64,
    min_source_donors: int = 3,
    min_metacells: int = 20,
    min_detected: int = 5,
    min_detection_fraction: float = 0.1,
    min_abs_meta_rho: float = 0.1,
    seed: int = 1729,
) -> pd.DataFrame:
    """Return exact-refined new-pair candidates selected without query donors."""
    lnc_ids = np.asarray(list(map(str, lnc_ids)), dtype=object)
    pathway_ids = np.asarray(list(map(str, pathway_ids)), dtype=object)
    if len(lnc_ids) != len(set(lnc_ids)) or len(pathway_ids) != len(set(pathway_ids)):
        raise MetacellScreenError("feature IDs must be unique")
    blocks, lnc_support, pathway_support = _context_blocks(
        lnc_expression, pathway_activity, metadata, source_donors,
        min_metacells=min_metacells,
        min_detected=min_detected,
        min_detection_fraction=min_detection_fraction,
    )
    lnc_keep = np.flatnonzero(lnc_support >= min_source_donors)
    pathway_keep = np.flatnonzero(pathway_support >= min_source_donors)
    columns = list(PAIR_KEYS) + [
        "source_mean_rho", "source_meta_rho", "source_direction_consistency",
        "source_donor_count", "source_meta_p_value", "screen_rank",
        "selection_policy", "candidate_selection_query_donor_used",
    ]
    if len(blocks) < min_source_donors or not len(lnc_keep) or not len(pathway_keep):
        return pd.DataFrame(columns=columns)
    sx, sy = _countsketch(
        blocks, lnc_keep, pathway_keep, dimensions=sketch_dimensions, seed=seed
    )
    proposed_lnc: list[int] = []
    proposed_path: list[int] = []
    approximate_rank: list[int] = []
    proposal_k = min(len(pathway_keep), max(top_k, top_k * sketch_oversample))
    for start in range(0, len(lnc_keep), 256):
        stop = min(start + 256, len(lnc_keep))
        approximate = sx[:, start:stop].T @ sy
        choose = np.argpartition(np.abs(approximate), -proposal_k, axis=1)[:, -proposal_k:]
        for local_row, candidates in enumerate(choose):
            ordered = candidates[np.argsort(
                -np.abs(approximate[local_row, candidates]), kind="stable"
            )]
            proposed_lnc.extend([lnc_keep[start + local_row]] * len(ordered))
            proposed_path.extend(pathway_keep[ordered].tolist())
            approximate_rank.extend(range(1, len(ordered) + 1))
    li = np.asarray(proposed_lnc, dtype=np.int64)
    pi = np.asarray(proposed_path, dtype=np.int64)
    effects = _exact_pair_effects(blocks, li, pi)
    result = pd.DataFrame(
        {
            "lncrna_id": lnc_ids[li],
            "pathway_id": pathway_ids[pi],
            "approximate_screen_rank": approximate_rank,
            **effects,
        }
    )
    result = result.loc[
        result.source_donor_count.ge(min_source_donors)
        & result.source_meta_rho.abs().ge(min_abs_meta_rho)
    ].copy()
    if result.empty:
        return pd.DataFrame(columns=columns)
    result = result.sort_values(
        ["lncrna_id", "source_meta_rho", "source_direction_consistency", "pathway_id"],
        ascending=[True, False, False, True],
        key=lambda column: column.abs() if column.name == "source_meta_rho" else column,
        kind="mergesort",
    )
    result = result.groupby("lncrna_id", observed=True, sort=False).head(top_k).copy()
    result["screen_rank"] = result.groupby("lncrna_id", observed=True).cumcount() + 1
    result["selection_policy"] = SCREEN_POLICY
    result["candidate_selection_query_donor_used"] = False
    return result.loc[:, columns].reset_index(drop=True)


def exact_donor_targets(
    lnc_expression,
    pathway_activity,
    metadata: pd.DataFrame,
    lnc_ids: Iterable[str],
    pathway_ids: Iterable[str],
    pairs: pd.DataFrame,
    donors: Iterable[str],
    *,
    min_metacells: int = 20,
    min_detected: int = 5,
    min_detection_fraction: float = 0.1,
) -> pd.DataFrame:
    """Compute exact signed targets for declared pairs, one row per donor."""
    lnc_ids = pd.Index(list(map(str, lnc_ids)))
    pathway_ids = pd.Index(list(map(str, pathway_ids)))
    if not set(PAIR_KEYS).issubset(pairs.columns):
        raise MetacellScreenError("pair table lacks exact IDs")
    clean = pairs.loc[:, list(PAIR_KEYS)].astype(str).drop_duplicates().reset_index(drop=True)
    li = lnc_ids.get_indexer(clean.lncrna_id)
    pi = pathway_ids.get_indexer(clean.pathway_id)
    if (li < 0).any() or (pi < 0).any():
        raise MetacellScreenError("pair table contains unavailable exact IDs")
    wanted = {str(value) for value in donors}
    blocks, _, _ = _context_blocks(
        lnc_expression, pathway_activity, metadata, wanted,
        min_metacells=min_metacells,
        min_detected=min_detected,
        min_detection_fraction=min_detection_fraction,
    )
    rows: list[pd.DataFrame] = []
    for block in blocks:
        valid = block["x_valid"][li] & block["y_valid"][pi]
        rho = np.full(len(clean), np.nan, dtype=np.float32)
        for start in range(0, len(clean), 8192):
            stop = min(start + 8192, len(clean))
            local = np.flatnonzero(valid[start:stop]) + start
            if len(local):
                rho[local] = np.sum(
                    block["x_rank"][:, li[local]] * block["y_rank"][:, pi[local]],
                    axis=0,
                    dtype=np.float64,
                )
        frame = clean.loc[valid].copy()
        frame["query_donor_id"] = str(block["donor_id"])
        frame["rho"] = rho[valid]
        frame["association_target"] = np.abs(rho[valid])
        frame["signed_target_unit_interval"] = (rho[valid] + 1.0) / 2.0
        frame["n_metacells"] = int(block["n_metacells"])
        frame["lnc_detection_rate"] = block["lnc_detection_rate"][li[valid]]
        frame["association_method"] = TARGET_METHOD
        frame["association_observation_unit"] = "within_donor_metacell"
        frame["association_target_level"] = (
            "donor_x_celltype_x_lncrna_x_exact_pathway"
        )
        frame["donor_resolved_target"] = True
        rows.append(frame)
    if not rows:
        return pd.DataFrame(
            columns=list(PAIR_KEYS) + [
                "query_donor_id", "rho", "association_target",
                "signed_target_unit_interval", "n_metacells",
                "lnc_detection_rate", "association_method",
                "association_observation_unit", "association_target_level",
                "donor_resolved_target",
            ]
        )
    return pd.concat(rows, ignore_index=True)


def exact_source_pair_effects(
    lnc_expression,
    pathway_activity,
    metadata: pd.DataFrame,
    lnc_ids: Iterable[str],
    pathway_ids: Iterable[str],
    pairs: pd.DataFrame,
    source_donors: Iterable[str],
    *,
    min_metacells: int = 20,
    min_detected: int = 5,
    min_detection_fraction: float = 0.1,
) -> pd.DataFrame:
    """Attach exact equal-donor source effects to an already frozen universe."""
    lnc_ids = pd.Index(list(map(str, lnc_ids)))
    pathway_ids = pd.Index(list(map(str, pathway_ids)))
    clean = pairs.loc[:, list(PAIR_KEYS)].astype(str).drop_duplicates().reset_index(drop=True)
    li = lnc_ids.get_indexer(clean.lncrna_id)
    pi = pathway_ids.get_indexer(clean.pathway_id)
    if (li < 0).any() or (pi < 0).any():
        raise MetacellScreenError("source universe contains unavailable exact IDs")
    blocks, _, _ = _context_blocks(
        lnc_expression, pathway_activity, metadata, source_donors,
        min_metacells=min_metacells,
        min_detected=min_detected,
        min_detection_fraction=min_detection_fraction,
    )
    effects = _exact_pair_effects(blocks, li, pi)
    return pd.concat([clean, pd.DataFrame(effects)], axis=1)


__all__ = [
    "MetacellScreenError", "SCREEN_POLICY", "TARGET_METHOD", "donor_folds",
    "exact_donor_targets", "exact_source_pair_effects", "screen_source_pairs",
]
