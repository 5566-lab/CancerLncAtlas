"""Numerical estimability of rank-residual associations."""
from __future__ import annotations
import numpy as np


def rank_residual_estimable(raw_rank, residual, design_columns):
    """Return whether each rank vector retains variation after adjustment.

    The tolerance is relative to the rank vector and floating-point precision;
    expression units do not enter it. Exact constants and vectors in the nuisance
    span have no adjusted correlation even if roundoff leaves a small residual.
    """
    raw_rank = np.asarray(raw_rank, dtype=float)
    residual = np.asarray(residual, dtype=float)
    factor = 256 * np.finfo(float).eps * max(raw_rank.shape[0], design_columns + 1)
    tolerance = factor * np.maximum(1, np.linalg.norm(raw_rank, axis=0))
    return ((np.ptp(raw_rank, axis=0) > 0)
            & (np.linalg.norm(residual, axis=0) > tolerance))
