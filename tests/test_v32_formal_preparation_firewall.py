from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from scripts.prepare_v32_formal import (
    attach_replication_labels,
    build_graph,
    make_batches,
)


torch = pytest.importorskip("torch")


def _discovery(effect: float, label: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cancer_id": ["BRCA"],
            "lncrna_id": ["L1"],
            "pathway_id": ["P1"],
            "discovery_effect": [effect],
            "proxy_label": [label],
            "weak_positive": [False],
            "label_class": ["strong_positive" if label else "unlabeled"],
            "shared_or_local_scope": ["shared"],
            "detection_rate": [0.7],
            "detected_cancers": [5],
        }
    )


def test_validation_direction_comes_from_replication_not_train_effect() -> None:
    train = _discovery(0.8, 1)
    replication = _discovery(-0.7, 1)
    validation = attach_replication_labels(train, replication)
    assert validation.discovery_effect.iloc[0] == pytest.approx(0.8)
    assert validation.train_discovery_effect.iloc[0] == pytest.approx(0.8)
    assert validation.replication_effect.iloc[0] == pytest.approx(-0.7)
    assert validation.replication_direction_label.iloc[0] == 0

    bundle = SimpleNamespace(
        node_maps={"lncRNA": {"L1": 0}, "pathway": {"P1": 0}, "cancer": {"BRCA": 0}}
    )
    batch = make_batches(
        validation,
        logits=pd.Series([0.25]).to_numpy(),
        bundle=bundle,
        split="validation",
    )[0]
    assert batch["direction_label"].item() == 0.0
    assert batch["direction_available"].item() is True
    # Conservation context still contains only the allowed train effect.
    assert batch["conservation_context"][0, 2].item() == pytest.approx(0.8)


def test_training_preparer_refuses_to_materialize_a_test_batch() -> None:
    validation = attach_replication_labels(_discovery(0.8), _discovery(-0.7))
    bundle = SimpleNamespace(
        node_maps={"lncRNA": {"L1": 0}, "pathway": {"P1": 0}, "cancer": {"BRCA": 0}}
    )
    with pytest.raises(RuntimeError, match="sealed split"):
        make_batches(
            validation,
            logits=pd.Series([0.25]).to_numpy(),
            bundle=bundle,
            split="test",
        )


def test_legacy_implicit_graph_call_is_fail_closed() -> None:
    with pytest.raises(TypeError):
        build_graph(["L1"], ["P1"], ["BRCA"])
