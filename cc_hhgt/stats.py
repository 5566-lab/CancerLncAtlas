from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


def bh_fdr(p_values: Iterable[float], total_tests: int | None = None) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    out = np.full(p.shape, np.nan, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return out
    observed = p[finite]
    n = int(total_tests or len(observed))
    order = np.argsort(observed)
    ranked = observed[order]
    adjusted = ranked * n / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    restore = np.empty_like(adjusted)
    restore[order] = adjusted
    out[finite] = restore
    return out


def robust_z(values: pd.Series | np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if not np.isfinite(x).any():
        return np.zeros_like(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    scale = mad / 0.6744897501960817 if np.isfinite(mad) and mad > 0 else np.nanstd(x, ddof=1)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return np.nan_to_num((x - med) / scale, nan=0.0, posinf=0.0, neginf=0.0)


def minmax01(values: pd.Series | np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    finite = np.isfinite(x)
    out = np.zeros_like(x, dtype=float)
    if not finite.any():
        return out
    lo, hi = np.nanmin(x[finite]), np.nanmax(x[finite])
    if hi <= lo:
        return out
    out[finite] = (x[finite] - lo) / (hi - lo)
    return out


def signed_support(effect: pd.Series | np.ndarray, fdr: pd.Series | np.ndarray, effect_scale: float = 0.3) -> np.ndarray:
    e = np.asarray(effect, dtype=float)
    q = np.asarray(fdr, dtype=float)
    magnitude = np.tanh(np.abs(e) / max(effect_scale, 1e-6))
    significance = np.clip(-np.log10(np.clip(q, 1e-300, 1.0)) / 10.0, 0.0, 1.0)
    return np.nan_to_num(magnitude * significance, nan=0.0)


def prepare_design(covariates: pd.DataFrame) -> np.ndarray:
    if covariates.empty:
        return np.ones((0, 1), dtype=float)
    blocks: list[pd.DataFrame] = []
    for col in covariates.columns:
        s = covariates[col]
        if pd.api.types.is_numeric_dtype(s):
            numeric = pd.to_numeric(s, errors="coerce")
            median = numeric.median()
            numeric = numeric.fillna(median if np.isfinite(median) else 0.0)
            blocks.append(pd.DataFrame({col: robust_z(numeric)}))
        else:
            text = s.astype("string").fillna("<NA>")
            if text.nunique(dropna=False) > max(20, len(text) // 4):
                continue
            blocks.append(pd.get_dummies(text, prefix=col, drop_first=True, dtype=float))
    design = pd.concat(blocks, axis=1) if blocks else pd.DataFrame(index=covariates.index)
    x = design.to_numpy(dtype=float)
    return np.column_stack([np.ones(len(covariates)), x])


def design_rank(design: np.ndarray) -> int:
    """Return the numerical rank used by a covariate residualization."""
    x = np.nan_to_num(np.asarray(design, dtype=float), nan=0.0)
    if x.ndim != 2 or x.shape[0] == 0:
        return 0
    return int(np.linalg.matrix_rank(x))


def residualize(matrix: np.ndarray, design: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if design.shape[0] == 0 or design.shape[1] <= 1:
        return matrix.astype(np.float32)
    x = np.nan_to_num(np.asarray(design, dtype=float), nan=0.0)
    # lstsq projects onto the actual column space.  A reduced QR projection
    # removes spurious dimensions when a one-hot design is rank deficient.
    fitted = x @ np.linalg.lstsq(x, matrix, rcond=None)[0]
    return (matrix - fitted).astype(np.float32)


def rank_transform(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    return np.apply_along_axis(stats.rankdata, 0, matrix).astype(np.float32)


def standardize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    mean = np.nanmean(matrix, axis=0, keepdims=True)
    sd = np.nanstd(matrix, axis=0, ddof=1, keepdims=True)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    return np.nan_to_num((matrix - mean) / sd, nan=0.0, posinf=0.0, neginf=0.0)


def correlation_p_values(r: np.ndarray, n: int, *, residual_design_rank: int = 1) -> np.ndarray:
    """Two-sided P values for (partial) correlations.

    ``residual_design_rank`` includes the intercept.  Thus an ordinary
    correlation uses the default rank 1 and df=n-2, while correlations of
    two variables residualized on a rank-q covariate design use df=n-q-1.
    """
    r = np.clip(np.asarray(r, dtype=float), -0.999999, 0.999999)
    df = int(n) - int(residual_design_rank) - 1
    if df <= 0:
        return np.ones_like(r)
    t = r * np.sqrt(df / np.maximum(1 - r * r, 1e-12))
    return 2 * stats.t.sf(np.abs(t), df=df)


def fisher_meta_correlations(correlations: np.ndarray, weights: np.ndarray | None = None) -> tuple[float, float]:
    r = np.clip(np.asarray(correlations, dtype=float), -0.999999, 0.999999)
    finite = np.isfinite(r)
    if not finite.any():
        return np.nan, np.nan
    r = r[finite]
    w = np.ones_like(r) if weights is None else np.asarray(weights, dtype=float)[finite]
    z = np.arctanh(r)
    mean_z = np.average(z, weights=np.maximum(w, 1e-6))
    se = np.sqrt(1.0 / np.maximum(w, 1e-6).sum())
    p = 2 * stats.norm.sf(abs(mean_z / max(se, 1e-12)))
    return float(np.tanh(mean_z)), float(p)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (y_prob >= lo) & ((y_prob < hi) if hi < 1.0 else (y_prob <= hi))
        if not mask.any():
            continue
        ece += mask.mean() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)
