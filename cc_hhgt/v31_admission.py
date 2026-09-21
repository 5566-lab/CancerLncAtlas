"""Validation-only residual selection and admission policies."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from .common import require_columns
from .metrics import binary_metrics


def select_residual_lambda(
    validation: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    lambda_column: str = "residual_shrinkage_lambda",
    final_probability_column: str = "proxy_positive_probability",
    base_probability_column: str = "base_probability",
    label_column: str = "proxy_label",
    minimum_delta_auprc: float = 0.01,
    maximum_brier_worsening: float = 0.01,
    maximum_ece_worsening: float = 0.01,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select lambda and decide module admission without consulting test rows."""

    required = [
        "candidate_id", "metric_scope", *group_columns, lambda_column,
        final_probability_column, base_probability_column, label_column,
    ]
    require_columns(validation, required, "residual validation predictions")
    if not validation.metric_scope.astype(str).eq("validation").all():
        raise RuntimeError("residual lambda selection may only use validation rows")
    if validation.duplicated([*group_columns, lambda_column, "candidate_id"]).any():
        raise RuntimeError("duplicate residual validation candidates")
    for _, comparison in validation.groupby(list(group_columns), observed=True, sort=True):
        signatures = []
        for value, candidate_group in comparison.groupby(lambda_column, observed=True, sort=True):
            signature = candidate_group[
                ["candidate_id", label_column, base_probability_column]
            ].sort_values("candidate_id").reset_index(drop=True)
            signatures.append((value, signature))
        reference_value, reference = signatures[0]
        for value, signature in signatures[1:]:
            if not reference.equals(signature):
                raise RuntimeError(
                    "residual lambda candidates/labels/base drift: "
                    f"{reference_value} vs {value}"
                )
    rows: list[dict[str, Any]] = []
    for keys, group in validation.groupby(
        [*group_columns, lambda_column], observed=True, sort=True
    ):
        if not isinstance(keys, tuple):
            keys = (keys,)
        identifiers = dict(zip([*group_columns, lambda_column], keys))
        base = binary_metrics(group[label_column], group[base_probability_column])
        final = binary_metrics(group[label_column], group[final_probability_column])
        rows.append(
            {
                **identifiers,
                "n": final["n"],
                "positive_rate": final["positive_rate"],
                "base_auprc": base["auprc"],
                "final_auprc": final["auprc"],
                "delta_auprc": final["auprc"] - base["auprc"],
                "base_auroc": base["auroc"],
                "final_auroc": final["auroc"],
                "delta_auroc": final["auroc"] - base["auroc"],
                "base_brier": base["brier"],
                "final_brier": final["brier"],
                "delta_brier": final["brier"] - base["brier"],
                "base_ece": base["ece"],
                "final_ece": final["ece"],
                "delta_ece": final["ece"] - base["ece"],
            }
        )
    metrics = pd.DataFrame(rows)
    selections: list[dict[str, Any]] = []
    for keys, group in metrics.groupby(list(group_columns), observed=True, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        ordered = group.sort_values(
            ["delta_auprc", "delta_brier", "delta_ece", lambda_column],
            ascending=[False, True, True, True],
        )
        selected = ordered.iloc[0]
        admitted = bool(
            np.isfinite(selected.delta_auprc)
            and selected.delta_auprc >= minimum_delta_auprc
            and selected.delta_brier <= maximum_brier_worsening
            and selected.delta_ece <= maximum_ece_worsening
        )
        selections.append(
            {
                **dict(zip(group_columns, keys)),
                "selected_lambda": float(selected[lambda_column]),
                "validation_delta_auprc": float(selected.delta_auprc),
                "validation_delta_auroc": float(selected.delta_auroc),
                "validation_delta_brier": float(selected.delta_brier),
                "validation_delta_ece": float(selected.delta_ece),
                "minimum_delta_auprc": float(minimum_delta_auprc),
                "maximum_brier_worsening": float(maximum_brier_worsening),
                "maximum_ece_worsening": float(maximum_ece_worsening),
                "admitted": admitted,
                "decision": "PASS" if admitted else "OFF",
                "selection_scope": "validation",
                "test_metric_used_for_selection": False,
            }
        )
    return metrics, pd.DataFrame(selections)
