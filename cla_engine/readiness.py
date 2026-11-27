"""CPU checks performed before baseline fitting, capsule commit and GPU creation."""
from __future__ import annotations

import numpy as np

from .common import fail


def validate_task_arrays(data, view, task, numeric_features, allow_empty=False, objective='binary'):
    if objective not in {'binary', 'observed_fused_rms_effect'}:
        fail('RELATION_OBJECTIVE_UNKNOWN', view, objective=objective)
    required = {
        "lnc", "target", "numeric", "base", "label", "available", "weight",
        "direction_label", "direction_available",
    }
    if required - set(data):
        fail("TASK_ARRAY_FIELDS_MISSING", view, task=task, missing=sorted(required - set(data)))
    n = len(data["lnc"])
    available = np.asarray(data["available"])
    for key in required - {"numeric"}:
        if np.asarray(data[key]).shape != (n,):
            fail("TASK_ARRAY_SHAPE", view, task=task, field=key)
    if np.asarray(data["numeric"]).shape != (n, int(numeric_features)):
        fail("TASK_NUMERIC_SHAPE", view, task=task)
    if not np.isin(available, [0, 1]).all():
        fail("TASK_AVAILABILITY_INVALID", view, task=task)
    valid = available.astype(bool)
    if not valid.any() and not allow_empty:
        fail(
            "TASK_NOT_TRAINABLE_CPU",
            "No evaluable candidates; do not rent a GPU or silently remove a task",
            view=view,
            task=task,
            n_candidates=n,
            next_action="REVIEW_COHORT_COVERAGE_WITHOUT_CHANGING_FROZEN_FOLDS",
        )
    for key in ("numeric", "base"):
        if not np.isfinite(data[key]).all():
            fail("TASK_NONFINITE_FEATURE", view, task=task, field=key)
    y = np.asarray(data["label"])
    w = np.asarray(data["weight"])
    label_ok = (np.isin(y[valid], [0, 1]).all() if objective == 'binary' else
                np.isfinite(y[valid]).all() and np.all((y[valid] >= 0) & (y[valid] <= 1 + 1e-6)))
    if not label_ok or not np.isfinite(w[valid]).all() or np.any(w[valid] <= 0):
        fail("TASK_LABEL_OR_WEIGHT_INVALID", view, task=task)
    dm = np.asarray(data["direction_available"])
    if not np.isin(dm, [0, 1]).all() or np.any(dm.astype(bool) & ~valid):
        fail("TASK_DIRECTION_MASK_INVALID", view, task=task)
    if objective == 'observed_fused_rms_effect' and dm.any():
        fail('RELATION_STRENGTH_DIRECTION_HEAD_NOT_DECLARED', view, task=task)
    if objective == 'binary' and not np.isin(np.asarray(data["direction_label"])[dm.astype(bool)], [0, 1]).all():
        fail("TASK_DIRECTION_LABEL_INVALID", view, task=task)
    return {
        "view": view,
        "task": task,
        "rows": n,
        "usable": int(valid.sum()),
        "positive": int(y[valid].sum()) if objective == 'binary' else None,
        "label_min": float(np.min(y[valid])) if objective != 'binary' and valid.any() else None,
        "label_max": float(np.max(y[valid])) if objective != 'binary' and valid.any() else None,
    }


def write_prepared_readiness(root, spec, views, baselines, coverage):
    from pathlib import Path
    import html
    import json

    from .common import atomic_json

    root = Path(root)
    rows = []
    split_by_view = {view["id"]: view["split"] for view in views}
    for view in views:
        for task in ("pathway", "state"):
            with np.load(root / "views" / view["id"] / f"{task}.npz", allow_pickle=False) as values:
                # Empty cancer/fold views are typed unavailable. The frozen
                # contract checks estimability after pooling cancers by split.
                row = validate_task_arrays(
                    values,
                    view["id"],
                    task,
                    spec["numeric_features"],
                    allow_empty=True,
                    objective=spec.get('relation_objective', 'binary'),
                )
            row["split"] = split_by_view[view["id"]]
            rows.append(row)

    aggregate = {}
    for task in ("pathway", "state"):
        if task not in baselines:
            fail("TASK_NOT_ESTIMABLE", "Fitted pooled baseline receipt is missing", task=task)
        aggregate[task] = {}
        for split in ("train", "validation"):
            selected = [row for row in rows if row["task"] == task and row["split"] == split]
            usable = sum(row["usable"] for row in selected)
            objective = spec.get('relation_objective', 'binary')
            positive = sum(row["positive"] for row in selected) if objective == 'binary' else None
            if usable == 0 or (objective == 'binary' and (positive == 0 or positive == usable)):
                fail(
                    "TASK_NOT_ESTIMABLE",
                    "Pooled task split lacks estimable candidates",
                    task=task,
                    split=split,
                    usable=usable,
                    positive=positive,
                )
            aggregate[task][split] = {"usable": usable, "positive": positive}

    empty_views = [
        {"view": row["view"], "task": row["task"], "split": row["split"]}
        for row in rows
        if row["usable"] == 0
    ]
    receipt = {
        "state": "CPU_TRAINABILITY_VALIDATED_NOT_GPU_VALIDATED",
        "views": rows,
        "aggregate_trainability": aggregate,
        "typed_unavailable_empty_views": empty_views,
        "empty_view_policy": ("ALLOW_TYPED_UNAVAILABLE_SINGLE_VIEW_REQUIRE_POOLED_TWO_CLASS_BY_TASK_AND_SPLIT"
                              if spec.get('relation_objective', 'binary') == 'binary' else
                              "ALLOW_TYPED_UNAVAILABLE_SINGLE_VIEW_REQUIRE_POOLED_OBSERVED_BY_TASK_AND_SPLIT"),
        "task_weights": {"pathway": 1.0, "state": 0.5},
        "baselines": baselines,
        "singlecell_role": spec.get("singlecell_role"),
        "genomics_enabled": spec.get("genomics_enabled"),
        "graph_backend_checked": False,
        "paid_training_permitted_by_this_report_alone": False,
    }
    atomic_json(root / "CPU_READINESS.json", receipt)
    text = (
        '<!doctype html><html lang="zh"><meta charset="utf-8"><title>R6 CPU preflight</title>'
        '<h1>CPU 物化检查</h1><p>不是 CUDA 验收或正式发布许可。</p><pre>'
        + html.escape(json.dumps(receipt, ensure_ascii=False, indent=2))
        + "</pre></html>"
    )
    (root / "preflight_report.html").write_text(text, encoding="utf-8")
    return receipt
