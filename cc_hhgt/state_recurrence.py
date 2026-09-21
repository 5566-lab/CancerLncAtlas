from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


HISTORY_KEYS = ["lncrna_id", "state_id"]


def add_fold_local_recurrence(
    train_candidates: pd.DataFrame,
    target_candidates: pd.DataFrame,
    prior_strength: float = 2.0,
) -> pd.DataFrame:
    """Add train-cancer-only recurrence features to validation/test candidates."""
    if prior_strength < 0:
        raise ValueError("prior_strength must be non-negative")
    required = {*HISTORY_KEYS, "proxy_label"}
    for name, frame in (("train", train_candidates), ("target", target_candidates)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} candidates missing columns: {missing}")

    train = train_candidates.copy()
    train["proxy_label"] = pd.to_numeric(train.proxy_label, errors="raise").astype(float)
    history = (
        train.groupby(HISTORY_KEYS, observed=True)
        .proxy_label.agg(history_positive_count="sum", history_cancer_count="count")
        .reset_index()
    )
    state_prior = train.groupby("state_id", observed=True).proxy_label.mean()
    output = target_candidates.merge(history, on=HISTORY_KEYS, how="left", validate="many_to_one")
    output["history_positive_count"] = output.history_positive_count.fillna(0.0).astype(float)
    output["history_cancer_count"] = output.history_cancer_count.fillna(0).astype(int)
    output["history_state_prior"] = output.state_id.map(state_prior).astype(float)
    if output.history_state_prior.isna().any():
        missing_states = sorted(output.loc[output.history_state_prior.isna(), "state_id"].astype(str).unique())
        raise RuntimeError(f"No train-cancer prevalence is available for states: {missing_states}")
    denominator = output.history_cancer_count.astype(float) + float(prior_strength)
    numerator = output.history_positive_count + float(prior_strength) * output.history_state_prior
    output["recurrence_probability"] = np.where(
        denominator.gt(0), numerator / denominator, output.history_state_prior
    )
    return output


def add_split_ranks(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"split", "state_id", "graph_equal_probability", "recurrence_probability"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Cannot rank recurrence expert; missing columns: {missing}")
    output = frame.copy()
    groups = ["split", "state_id"]
    output["graph_equal_rank"] = output.groupby(groups, observed=True).graph_equal_probability.rank(
        method="average", pct=True
    )
    output["recurrence_rank"] = output.groupby(groups, observed=True).recurrence_probability.rank(
        method="average", pct=True
    )
    return output


def binary_ranking_metrics(labels: Iterable[float], score: Iterable[float]) -> dict[str, float | int]:
    labels_array = np.asarray(list(labels), dtype=float)
    score_array = np.asarray(list(score), dtype=float)
    valid = np.isfinite(labels_array) & np.isfinite(score_array) & np.isin(labels_array, [0.0, 1.0])
    labels_array = labels_array[valid]
    score_array = score_array[valid]
    result: dict[str, float | int] = {
        "n": int(len(labels_array)),
        "n_positive": int(labels_array.sum()),
        "auroc": float("nan"),
        "auprc": float("nan"),
    }
    if len(labels_array) and np.unique(labels_array).size == 2:
        result["auroc"] = float(roc_auc_score(labels_array, score_array))
        result["auprc"] = float(average_precision_score(labels_array, score_array))
    return result


def select_recurrence_weight(
    validation: pd.DataFrame,
    weights: Iterable[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> tuple[float, pd.DataFrame]:
    required = {"proxy_label", "graph_equal_rank", "recurrence_rank"}
    missing = sorted(required - set(validation.columns))
    if missing:
        raise ValueError(f"Cannot select recurrence weight; missing columns: {missing}")
    rows: list[dict[str, float | int]] = []
    for weight in sorted({float(value) for value in weights}):
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"recurrence weight must be in [0, 1], observed {weight}")
        score = weight * validation.recurrence_rank + (1.0 - weight) * validation.graph_equal_rank
        rows.append({"recurrence_weight": weight, **binary_ranking_metrics(validation.proxy_label, score)})
    metrics = pd.DataFrame(rows)
    valid = metrics.loc[metrics.auroc.notna()].copy()
    if valid.empty:
        # No validation discrimination can be estimated. Prefer the explicit,
        # train-only recurrence expert over a target-label-dependent fallback.
        return 1.0, metrics
    best_auroc = float(valid.auroc.max())
    # Deterministic tie break favours the transparent recurrence expert.
    best = float(valid.loc[np.isclose(valid.auroc, best_auroc), "recurrence_weight"].max())
    return best, metrics

