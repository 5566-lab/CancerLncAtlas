from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cc_hhgt.v31_residual import (
    attach_residual_base,
    assert_exact_fallback,
    build_additive_residual_head,
    compose_residual_logits,
    logit_to_probability,
    probability_to_logit,
)


def test_zero_initialized_residual_is_exact_base() -> None:
    import torch

    torch.manual_seed(4)
    base = torch.tensor([-2.0, 0.0, 3.0], dtype=torch.float32)
    features = torch.randn(3, 5)
    head = build_additive_residual_head(5, hidden_dim=7)
    final, delta = head(base, features)
    assert torch.equal(delta, torch.zeros_like(delta))
    assert torch.equal(final, base)


def test_disabled_or_unavailable_module_is_exact_previous_score() -> None:
    base = np.array([-2.0, 0.5, 1.0])
    delta = np.array([3.0, -4.0, 2.0])
    disabled, parts = compose_residual_logits(base, {"Graph": delta}, admitted={"Graph": False})
    assert_exact_fallback(base, disabled)
    unavailable, parts = compose_residual_logits(
        base,
        {"Graph": delta},
        availability={"Graph": np.zeros(3, dtype=bool)},
    )
    assert_exact_fallback(base, unavailable)
    assert np.array_equal(parts["Graph"], np.zeros(3))


def test_sequential_module_off_returns_exact_previous_score() -> None:
    base = np.array([-1.0, 1.0])
    graph, _ = compose_residual_logits(base, {"Graph": np.array([0.2, -0.3])})
    graph_plus_rna, _ = compose_residual_logits(
        graph,
        {"RNA": np.array([10.0, 10.0])},
        admitted={"RNA": False},
    )
    assert_exact_fallback(graph, graph_plus_rna)


def test_probability_logit_roundtrip() -> None:
    probability = np.array([0.001, 0.2, 0.5, 0.9, 0.999])
    np.testing.assert_allclose(logit_to_probability(probability_to_logit(probability)), probability)


def test_attach_residual_base_preserves_candidates_and_rejects_leakage() -> None:
    candidates = pd.DataFrame(
        {"candidate_id": ["A:x", "B:x"], "cancer_id": ["A", "B"], "proxy_label": [1, 0]}
    )
    base = pd.DataFrame(
        {
            "candidate_id": ["A:x", "B:x"],
            "best_simple_score": [0.7, 0.2],
            "z_base": [0.8472978604, -1.3862943611],
            "metric_scope": ["cancer_crossfit_OOF", "cancer_crossfit_OOF"],
            "simple_base_fit_cancers": ["B|C", "A|C"],
        }
    )
    attached = attach_residual_base(
        candidates, base, split="train", train_cancers={"A", "B", "C"}
    )
    assert attached.candidate_id.tolist() == candidates.candidate_id.tolist()
    leaked = base.copy()
    leaked.loc[0, "simple_base_fit_cancers"] = "A|B|C"
    with pytest.raises(RuntimeError, match="own outcomes"):
        attach_residual_base(
            candidates, leaked, split="train", train_cancers={"A", "B", "C"}
        )
