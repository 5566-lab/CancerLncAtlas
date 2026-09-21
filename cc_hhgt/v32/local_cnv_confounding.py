"""Fold-local local-CNV sensitivity primitives for V3.2.

The functions in this module make missingness explicit and deliberately avoid
prediction terminology.  They are used by the association audit and by the
real lncRNA--gene G0 edge stability audit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata, t as student_t

from ..stats import prepare_design


DEFAULT_COVARIATES = (
    "purity", "age_years", "sex", "stage", "molecular_subtype",
    "clinical_subtype", "technical_batch", "leukocyte_fraction",
    "cell_fraction_Macrophages.M0", "cell_fraction_Macrophages.M1",
    "cell_fraction_Macrophages.M2",
)


MIN_MATCHED_PATIENTS = 12
STRONG_FDR = 0.05
STRONG_EFFECT = 0.20
WEAK_FDR = 0.10
WEAK_EFFECT = 0.15


@dataclass(frozen=True)
class PartialSpearmanResult:
    rho: float
    p_value: float
    n_patients: int
    residual_design_rank: int
    correlation_df: int
    x_beta: tuple[float, ...]
    y_beta: tuple[float, ...]
    x_standard_error: float
    y_standard_error: float
    available: bool
    unavailable_reason: str


def canonical_patient(value: object) -> str:
    text = str(value).strip().replace(".", "-").upper()
    return text[:12] if text.startswith("TCGA-") else text


def patient_first_matrix(
    frame: pd.DataFrame,
    *,
    feature_column: str,
    value_column: str,
) -> tuple[pd.DataFrame, int]:
    """Average duplicate aliquots before creating a patient-by-feature matrix."""

    required = {"patient_id", feature_column, value_column}
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"patient-first matrix missing columns: {missing}")
    local = frame[["patient_id", feature_column, value_column]].copy()
    local["patient_id"] = local["patient_id"].map(canonical_patient)
    local[value_column] = pd.to_numeric(local[value_column], errors="coerce")
    local = local.dropna(subset=["patient_id", feature_column, value_column])
    duplicate_rows = int(local.duplicated(["patient_id", feature_column], keep=False).sum())
    matrix = local.pivot_table(
        index="patient_id",
        columns=feature_column,
        values=value_column,
        aggfunc="mean",
        observed=True,
    )
    matrix.index = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str)
    return matrix.sort_index().sort_index(axis=1), duplicate_rows


def covariate_design(
    covariates: pd.DataFrame,
    patients: Sequence[str],
    *,
    columns: Sequence[str] = DEFAULT_COVARIATES,
) -> tuple[np.ndarray, tuple[str, ...]]:
    local = covariates.copy()
    local["patient_id"] = local["patient_id"].map(canonical_patient)
    if local.duplicated("patient_id").any():
        agg = {
            column: ("mean" if pd.api.types.is_numeric_dtype(local[column]) else "first")
            for column in local.columns
            if column not in {"cancer_id", "sample_id", "patient_id"}
        }
        local = local.groupby("patient_id", as_index=False, observed=True).agg(agg)
    local = local.set_index("patient_id").reindex(list(patients))
    used = tuple(column for column in columns if column in local.columns)
    if not used:
        return np.ones((len(patients), 1), dtype=float), used
    design = prepare_design(local[list(used)].reset_index(drop=True))
    if len(design) != len(patients) or not np.isfinite(design).all():
        raise ValueError("covariate design is not finite and patient-aligned")
    return np.asarray(design, dtype=float), used


def _rank(values: np.ndarray) -> np.ndarray:
    return rankdata(np.asarray(values, dtype=float), method="average").astype(float)


def partial_spearman(
    x: Sequence[float],
    y: Sequence[float],
    base_design: np.ndarray,
    *,
    x_extras: Iterable[Sequence[float]] = (),
    y_extras: Iterable[Sequence[float]] = (),
    available_mask: Sequence[bool] | None = None,
    min_patients: int = MIN_MATCHED_PATIENTS,
) -> PartialSpearmanResult:
    """Compute an exact matched-patient partial Spearman association.

    Every outcome and continuous extra covariate is ranked *after* applying the
    pair-specific availability mask.  Unavailable CNV therefore cannot become
    a zero-valued biological observation.
    """

    xv = np.asarray(x, dtype=float).reshape(-1)
    yv = np.asarray(y, dtype=float).reshape(-1)
    design = np.asarray(base_design, dtype=float)
    if design.ndim != 2 or len(design) != len(xv) or len(yv) != len(xv):
        raise ValueError("partial Spearman inputs are not row-aligned")
    x_extra_values = [np.asarray(value, dtype=float).reshape(-1) for value in x_extras]
    y_extra_values = [np.asarray(value, dtype=float).reshape(-1) for value in y_extras]
    extra_values = x_extra_values + y_extra_values
    if any(len(value) != len(xv) for value in extra_values):
        raise ValueError("extra covariate is not row-aligned")
    mask = np.isfinite(xv) & np.isfinite(yv) & np.isfinite(design).all(axis=1)
    for value in extra_values:
        mask &= np.isfinite(value)
    if available_mask is not None:
        explicit = np.asarray(available_mask, dtype=bool).reshape(-1)
        if len(explicit) != len(xv):
            raise ValueError("availability mask is not row-aligned")
        mask &= explicit
    n = int(mask.sum())
    if n < int(min_patients):
        return PartialSpearmanResult(
            np.nan, np.nan, n, 0, 0, (), (), np.nan, np.nan, False,
            "INSUFFICIENT_MATCHED_PATIENTS",
        )
    xr = _rank(xv[mask])
    yr = _rank(yv[mask])
    x_design = np.column_stack(
        [design[mask]] + [_rank(value[mask])[:, None] for value in x_extra_values]
    )
    y_design = np.column_stack(
        [design[mask]] + [_rank(value[mask])[:, None] for value in y_extra_values]
    )
    union_design = np.column_stack(
        [design[mask]] + [_rank(value[mask])[:, None] for value in extra_values]
    )
    rank = int(np.linalg.matrix_rank(union_design))
    df = n - rank - 1
    if df < 1:
        return PartialSpearmanResult(
            np.nan, np.nan, n, rank, df, (), (), np.nan, np.nan, False,
            "NO_RESIDUAL_CORRELATION_DEGREES_OF_FREEDOM",
        )
    beta_x, *_ = np.linalg.lstsq(x_design, xr, rcond=None)
    beta_y, *_ = np.linalg.lstsq(y_design, yr, rcond=None)
    rx = xr - x_design @ beta_x
    ry = yr - y_design @ beta_y
    sx = float(np.linalg.norm(rx))
    sy = float(np.linalg.norm(ry))
    if sx <= 0.0 or sy <= 0.0:
        return PartialSpearmanResult(
            np.nan, np.nan, n, rank, df,
            tuple(float(v) for v in beta_x), tuple(float(v) for v in beta_y),
            np.nan, np.nan, False, "ZERO_RESIDUAL_VARIANCE",
        )
    rho = float(np.clip(np.dot(rx, ry) / (sx * sy), -1.0, 1.0))
    statistic = abs(rho) * np.sqrt(df / max(1.0 - rho * rho, 1e-15))
    p_value = float(2.0 * student_t.sf(statistic, df))
    x_variance = float(np.dot(rx, rx) / df)
    y_variance = float(np.dot(ry, ry) / df)
    x_inverse = np.linalg.pinv(x_design.T @ x_design, rcond=1e-10)
    y_inverse = np.linalg.pinv(y_design.T @ y_design, rcond=1e-10)
    x_standard_error = (
        float(np.sqrt(max(x_variance * x_inverse[-1, -1], 0.0)))
        if x_extra_values else np.nan
    )
    y_standard_error = (
        float(np.sqrt(max(y_variance * y_inverse[-1, -1], 0.0)))
        if y_extra_values else np.nan
    )
    return PartialSpearmanResult(
        rho, p_value, n, rank, df,
        tuple(float(v) for v in beta_x), tuple(float(v) for v in beta_y),
        x_standard_error, y_standard_error, True, "",
    )


def _rho_columns(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=0)
    denominator = np.sqrt(np.sum(left * left, axis=0) * np.sum(right * right, axis=0))
    return np.divide(
        numerator,
        denominator,
        out=np.full(left.shape[1], np.nan, dtype=float),
        where=denominator > 0,
    )


def _p_from_df(rho: np.ndarray, df: int) -> np.ndarray:
    if df < 1:
        return np.full(len(rho), np.nan, dtype=float)
    clipped = np.clip(np.abs(rho), 0.0, 1.0 - 1e-12)
    statistic = clipped * np.sqrt(df / np.maximum(1.0 - clipped * clipped, 1e-15))
    return 2.0 * student_t.sf(statistic, df)


def fold_local_pair_models(
    expression: np.ndarray,
    activity: np.ndarray,
    local_cnv: np.ndarray,
    pathway_cnv: np.ndarray,
    base_design: np.ndarray,
    *,
    local_callable: np.ndarray,
    pathway_callable: np.ndarray,
    min_patients: int = MIN_MATCHED_PATIENTS,
) -> dict[str, np.ndarray]:
    """Vectorized exact matched-patient A0_MATCHED/A1/A2 calculation.

    Columns are candidate pairs.  Columns sharing the same complete-case mask
    are handled together, so the C0 projection is fit once for that exact
    patient set.  Ranking happens after masking.  A1 adjusts only the lncRNA
    expression side; A2 additionally adjusts only the pathway-activity side.
    """

    x = np.asarray(expression, dtype=float)
    y = np.asarray(activity, dtype=float)
    zl = np.asarray(local_cnv, dtype=float)
    zp = np.asarray(pathway_cnv, dtype=float)
    design = np.asarray(base_design, dtype=float)
    lc = np.asarray(local_callable, dtype=bool)
    pc = np.asarray(pathway_callable, dtype=bool)
    if x.ndim != 2 or any(value.shape != x.shape for value in (y, zl, zp, lc, pc)):
        raise ValueError("pair model arrays must share an n-patient by n-pair shape")
    if design.ndim != 2 or design.shape[0] != x.shape[0]:
        raise ValueError("base design is not patient-aligned")
    pair_count = x.shape[1]
    result: dict[str, np.ndarray] = {
        "n": np.zeros(pair_count, dtype=np.int32),
        "n_a0_matched": np.zeros(pair_count, dtype=np.int32),
        "n_a1_local": np.zeros(pair_count, dtype=np.int32),
        "n_a2_local_pathway": np.zeros(pair_count, dtype=np.int32),
        "rho_a0_matched": np.full(pair_count, np.nan),
        "p_a0_matched": np.full(pair_count, np.nan),
        "rho_a1_local": np.full(pair_count, np.nan),
        "p_a1_local": np.full(pair_count, np.nan),
        "rho_a2_local_pathway": np.full(pair_count, np.nan),
        "p_a2_local_pathway": np.full(pair_count, np.nan),
        "local_cnv_beta": np.full(pair_count, np.nan),
        "local_cnv_beta_se": np.full(pair_count, np.nan),
        "pathway_cnv_beta": np.full(pair_count, np.nan),
        "pathway_cnv_beta_se": np.full(pair_count, np.nan),
        "residual_design_rank_a0": np.zeros(pair_count, dtype=np.int16),
        "residual_design_rank_a1": np.zeros(pair_count, dtype=np.int16),
        "residual_design_rank_a2": np.zeros(pair_count, dtype=np.int16),
        "available": np.zeros(pair_count, dtype=bool),
    }
    # A0_MATCHED and A1_LOCAL are matched on *local* CNV callability only.
    # Requiring pathway CNV here would silently mix the local-CNV effect with
    # extra patient loss.  A2 has its own, potentially smaller, complete-case
    # mask and its own A1 comparator on that exact A2 cohort.
    complete = (
        lc & np.isfinite(x) & np.isfinite(y) & np.isfinite(zl)
        & np.isfinite(design).all(axis=1)[:, None]
    )
    # A packed boolean mask is a deterministic complete-case identity.
    packed = np.packbits(complete, axis=0).T
    _, inverse = np.unique(packed, axis=0, return_inverse=True)
    for group_id in range(int(inverse.max()) + 1 if pair_count else 0):
        columns = np.flatnonzero(inverse == group_id)
        rows = complete[:, columns[0]]
        n = int(rows.sum())
        result["n"][columns] = n
        result["n_a0_matched"][columns] = n
        result["n_a1_local"][columns] = n
        if n < int(min_patients):
            continue
        local_design = design[rows]
        base_rank = int(np.linalg.matrix_rank(local_design))
        xr = rankdata(x[np.ix_(rows, columns)], axis=0, method="average")
        yr = rankdata(y[np.ix_(rows, columns)], axis=0, method="average")
        zlr = rankdata(zl[np.ix_(rows, columns)], axis=0, method="average")
        projection = np.linalg.pinv(local_design, rcond=1e-10)

        def residual(values: np.ndarray) -> np.ndarray:
            return values - local_design @ (projection @ values)

        x0, y0 = residual(xr), residual(yr)
        local0 = residual(zlr)
        rho0 = _rho_columns(x0, y0)
        local_den = np.sum(local0 * local0, axis=0)
        local_beta = np.divide(
            np.sum(local0 * x0, axis=0), local_den,
            out=np.full(len(columns), np.nan), where=local_den > 0,
        )
        x1 = x0 - local0 * local_beta
        rho1 = _rho_columns(x1, y0)
        rank_a1 = base_rank + (local_den > 1e-12).astype(np.int16)
        df0 = n - base_rank - 1
        p0 = _p_from_df(rho0, df0)
        p1 = np.asarray([
            _p_from_df(np.asarray([value]), n - int(rank_value) - 1)[0]
            for value, rank_value in zip(rho1, rank_a1, strict=True)
        ])
        local_df = np.maximum(n - rank_a1 - 1, 1)
        local_se = np.sqrt(
            np.divide(
                np.sum(x1 * x1, axis=0) / local_df,
                local_den,
                out=np.full(len(columns), np.nan), where=local_den > 0,
            )
        )
        result["rho_a0_matched"][columns] = rho0
        result["p_a0_matched"][columns] = p0
        result["rho_a1_local"][columns] = rho1
        result["p_a1_local"][columns] = p1
        result["local_cnv_beta"][columns] = local_beta
        result["local_cnv_beta_se"][columns] = local_se
        result["residual_design_rank_a0"][columns] = base_rank
        result["residual_design_rank_a1"][columns] = rank_a1
        result["available"][columns] = np.isfinite(rho0) & np.isfinite(rho1)
    # A2 is recomputed on its own exact complete-case cohort.  Store the A1
    # value on that same cohort separately so delta_rho_pathway_cnv is not a
    # missingness contrast.  The historical rho_a1_local remains the primary
    # A1-vs-A0 comparison on local-CNV-callable patients.
    result["rho_a1_on_a2_matched"] = np.full(pair_count, np.nan)
    result["p_a1_on_a2_matched"] = np.full(pair_count, np.nan)
    result["available_a2"] = np.zeros(pair_count, dtype=bool)
    complete_a2 = complete & pc & np.isfinite(zp)
    packed_a2 = np.packbits(complete_a2, axis=0).T
    _, inverse_a2 = np.unique(packed_a2, axis=0, return_inverse=True)
    for group_id in range(int(inverse_a2.max()) + 1 if pair_count else 0):
        columns = np.flatnonzero(inverse_a2 == group_id)
        rows = complete_a2[:, columns[0]]
        n = int(rows.sum())
        result["n_a2_local_pathway"][columns] = n
        if n < int(min_patients):
            continue
        local_design = design[rows]
        base_rank = int(np.linalg.matrix_rank(local_design))
        xr = rankdata(x[np.ix_(rows, columns)], axis=0, method="average")
        yr = rankdata(y[np.ix_(rows, columns)], axis=0, method="average")
        zlr = rankdata(zl[np.ix_(rows, columns)], axis=0, method="average")
        zpr = rankdata(zp[np.ix_(rows, columns)], axis=0, method="average")
        projection = np.linalg.pinv(local_design, rcond=1e-10)
        x0 = xr - local_design @ (projection @ xr)
        y0 = yr - local_design @ (projection @ yr)
        local0 = zlr - local_design @ (projection @ zlr)
        pathway0 = zpr - local_design @ (projection @ zpr)
        local_den = np.sum(local0 * local0, axis=0)
        local_beta = np.divide(
            np.sum(local0 * x0, axis=0), local_den,
            out=np.full(len(columns), np.nan), where=local_den > 0,
        )
        x1 = x0 - local0 * local_beta
        rho1 = _rho_columns(x1, y0)
        pathway_den = np.sum(pathway0 * pathway0, axis=0)
        pathway_beta = np.divide(
            np.sum(pathway0 * y0, axis=0), pathway_den,
            out=np.full(len(columns), np.nan), where=pathway_den > 0,
        )
        y2 = y0 - pathway0 * pathway_beta
        rho2 = _rho_columns(x1, y2)
        rank_a1 = base_rank + (local_den > 1e-12).astype(np.int16)
        rank_a2 = rank_a1 + (pathway_den > 1e-12).astype(np.int16)
        p1 = np.asarray([
            _p_from_df(np.asarray([value]), n - int(rank_value) - 1)[0]
            for value, rank_value in zip(rho1, rank_a1, strict=True)
        ])
        p2 = np.asarray([
            _p_from_df(np.asarray([value]), n - int(rank_value) - 1)[0]
            for value, rank_value in zip(rho2, rank_a2, strict=True)
        ])
        pathway_df = np.maximum(n - rank_a2 - 1, 1)
        pathway_se = np.sqrt(
            np.divide(
                np.sum(y2 * y2, axis=0) / pathway_df,
                pathway_den,
                out=np.full(len(columns), np.nan), where=pathway_den > 0,
            )
        )
        result["rho_a1_on_a2_matched"][columns] = rho1
        result["p_a1_on_a2_matched"][columns] = p1
        result["rho_a2_local_pathway"][columns] = rho2
        result["p_a2_local_pathway"][columns] = p2
        result["pathway_cnv_beta"][columns] = pathway_beta
        result["pathway_cnv_beta_se"][columns] = pathway_se
        result["residual_design_rank_a2"][columns] = rank_a2
        result["available_a2"][columns] = np.isfinite(rho1) & np.isfinite(rho2)
    result["available"] = (
        np.isfinite(result["rho_a0_matched"])
        & np.isfinite(result["rho_a1_local"])
    )
    return result


def bh_fdr(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values)
    if not valid.any():
        return out
    observed = values[valid]
    order = np.argsort(observed, kind="stable")
    ranked = observed[order] * len(observed) / np.arange(1, len(observed) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    restored = np.empty(len(observed), dtype=float)
    restored[order] = np.clip(adjusted, 0.0, 1.0)
    out[valid] = restored
    return out


def pathway_within_cancer_fdr(frame: pd.DataFrame, p_column: str) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, index in frame.groupby("pathway_id", observed=True, sort=False).groups.items():
        positions = list(index)
        result.loc[positions] = bh_fdr(frame.loc[positions, p_column].to_numpy(float))
    return result


def association_label(rho: float, fdr: float) -> str:
    if not np.isfinite(rho) or not np.isfinite(fdr):
        return "unavailable"
    if fdr <= STRONG_FDR and abs(rho) >= STRONG_EFFECT:
        return "strong_positive"
    if fdr <= WEAK_FDR and abs(rho) >= WEAK_EFFECT:
        return "weak_positive"
    return "unlabeled"


def direction(rho: float) -> str:
    if not np.isfinite(rho):
        return "unavailable"
    return "positive" if rho >= 0 else "negative"


def top_k_jaccard(left: Sequence[float], right: Sequence[float], k: int = 100) -> float:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    positions = np.flatnonzero(valid)
    if not len(positions):
        return np.nan
    size = min(int(k), len(positions))
    aa = set(positions[np.argsort(np.abs(a[positions]), kind="stable")[-size:]])
    bb = set(positions[np.argsort(np.abs(b[positions]), kind="stable")[-size:]])
    return len(aa & bb) / len(aa | bb)


def classify_impact(cancer_rows: Sequence[dict[str, float]]) -> str:
    """Apply the preregistered local-CNV impact thresholds verbatim."""

    rows = list(cancer_rows)
    if not rows:
        raise ValueError("impact classification requires cancer summaries")
    positive = np.asarray([row["positive_flip_rate"] for row in rows], float)
    direction_flip = np.asarray([row["direction_flip_rate"] for row in rows], float)
    rank_corr = np.asarray([row["median_rank_correlation"] for row in rows], float)
    top100 = np.asarray([row["top100_jaccard"] for row in rows], float)
    g0_sign = np.asarray([row["g0_sign_flip_rate"] for row in rows], float)
    g0_jaccard = np.asarray([row["g0_edge_jaccard"] for row in rows], float)
    # The preregistration defines global flip rates and medians across the
    # cancer-level summaries.  It does not promote a single noisy cancer's
    # rank/Jaccard minimum to a global retraining decision.  Per-cancer >15%
    # is the one explicitly count-based exception.
    weights = np.asarray([row.get("baseline_positive_count", 1.0) for row in rows], float)
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
    global_positive = (
        float(np.nansum(positive * weights) / np.sum(weights))
        if np.sum(weights) else float(np.nanmean(positive))
    )
    direction_weights = np.asarray([row.get("direction_evaluable_count", 1.0) for row in rows], float)
    direction_weights = np.where(
        np.isfinite(direction_weights) & (direction_weights > 0), direction_weights, 0.0
    )
    global_direction = (
        float(np.nansum(direction_flip * direction_weights) / np.sum(direction_weights))
        if np.sum(direction_weights) else float(np.nanmean(direction_flip))
    )
    median_rank = float(np.nanmedian(rank_corr))
    median_top100 = float(np.nanmedian(top100))
    global_g0_sign = float(np.nanmean(g0_sign))
    median_g0_jaccard = float(np.nanmedian(g0_jaccard))
    core = bool(
        global_positive > 0.10
        or np.sum(positive > 0.15) >= 5
        or global_direction > 0.05
        or median_rank < 0.90
        or median_top100 < 0.75
        or global_g0_sign > 0.05
        or median_g0_jaccard < 0.75
    )
    if core:
        return "CORE_RETRAIN_REQUIRED"
    small = bool(
        global_positive <= 0.05
        and global_direction <= 0.02
        and median_rank >= 0.95
        and median_top100 >= 0.85
        and global_g0_sign <= 0.02
        and median_g0_jaccard >= 0.85
    )
    return "IMPACT_SMALL" if small else "SENSITIVITY_ONLY_REVIEW"


__all__ = [
    "PartialSpearmanResult", "association_label", "bh_fdr", "canonical_patient",
    "classify_impact", "covariate_design", "direction", "partial_spearman",
    "fold_local_pair_models", "pathway_within_cancer_fdr", "patient_first_matrix",
    "top_k_jaccard",
]
