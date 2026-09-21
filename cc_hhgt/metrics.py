from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from .stats import expected_calibration_error


def binary_metrics(y_true, y_prob, bins: int = 10) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    # Clipping is required for logarithmic/calibration metrics, but it must not
    # be applied to ranking metrics.  In particular, clipping very small or
    # very large probabilities creates artificial ties.  A monotone
    # temperature transform can then appear to change AUROC/AUPRC merely by
    # moving scores out of the clipped region even though the model ranking is
    # unchanged.
    ranking_score = np.asarray(y_prob, dtype=float)
    p = np.clip(ranking_score, 1e-7, 1 - 1e-7)
    out = {
        "n": int(len(y)),
        "n_positive": int(y.sum()),
        "positive_rate": float(y.mean()) if len(y) else np.nan,
        "brier": float(brier_score_loss(y, p)) if len(y) else np.nan,
        "log_loss": float(log_loss(y, p, labels=[0, 1])) if len(y) else np.nan,
        "ece": expected_calibration_error(y, p, bins=bins) if len(y) else np.nan,
    }
    if len(np.unique(y)) == 2:
        out["auroc"] = float(roc_auc_score(y, ranking_score))
        out["auprc"] = float(average_precision_score(y, ranking_score))
    else:
        out["auroc"] = np.nan
        out["auprc"] = np.nan
    return out


def proxy_binary_metrics(frame: pd.DataFrame, y_prob, bins: int = 10) -> dict[str, float]:
    """Metrics for the documented strong/weak/unlabeled label policy.

    The primary proxy endpoint treats both evidence-supported classes as
    positive.  A strong-positive versus unlabeled sensitivity endpoint is
    reported separately so weak positives are never mislabeled as negatives.
    """
    probability = np.asarray(y_prob, dtype=float)
    out = binary_metrics(frame.proxy_label, probability, bins=bins)
    classes = frame.label_class.astype(str)
    out["n_strong_positive"] = int((classes == "strong_positive").sum())
    out["n_weak_positive"] = int((classes == "weak_positive").sum())
    out["n_unlabeled"] = int((classes == "unlabeled").sum())
    sensitivity_mask = classes != "weak_positive"
    sensitivity = binary_metrics(
        (classes.loc[sensitivity_mask] == "strong_positive").astype(np.int8),
        probability[sensitivity_mask.to_numpy()],
        bins=bins,
    )
    for key, value in sensitivity.items():
        out[f"strong_{key}"] = value
    return out
