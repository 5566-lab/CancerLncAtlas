from __future__ import annotations

import pandas as pd

from cc_hhgt.state_recurrence import (
    add_fold_local_recurrence,
    add_split_ranks,
    select_recurrence_weight,
)


def test_recurrence_uses_only_supplied_train_candidates() -> None:
    train = pd.DataFrame(
        {
            "lncrna_id": ["L1", "L1", "L2", "L2"],
            "state_id": ["S", "S", "S", "S"],
            "proxy_label": [1, 1, 0, 0],
        }
    )
    target = pd.DataFrame(
        {
            "lncrna_id": ["L1", "L2", "L3"],
            "state_id": ["S", "S", "S"],
            "proxy_label": [0, 1, 1],
        }
    )
    result = add_fold_local_recurrence(train, target, prior_strength=2.0)
    assert result.history_cancer_count.tolist() == [2, 2, 0]
    assert result.recurrence_probability.tolist() == [0.75, 0.25, 0.5]


def test_validation_weight_selection_can_choose_recurrence() -> None:
    validation = pd.DataFrame(
        {
            "proxy_label": [0, 0, 1, 1],
            "graph_equal_rank": [0.75, 1.0, 0.25, 0.5],
            "recurrence_rank": [0.25, 0.5, 0.75, 1.0],
        }
    )
    weight, metrics = select_recurrence_weight(validation)
    assert weight == 1.0
    assert len(metrics) == 5


def test_ranks_are_computed_separately_by_split_and_state() -> None:
    frame = pd.DataFrame(
        {
            "split": ["val", "val", "test", "test"],
            "state_id": ["S", "S", "S", "S"],
            "graph_equal_probability": [0.1, 0.2, 10.0, 20.0],
            "recurrence_probability": [0.2, 0.1, 20.0, 10.0],
        }
    )
    result = add_split_ranks(frame)
    assert result.graph_equal_rank.tolist() == [0.5, 1.0, 0.5, 1.0]
    assert result.recurrence_rank.tolist() == [1.0, 0.5, 1.0, 0.5]
