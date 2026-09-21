from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

from cc_hhgt.v29_multitask import (
    _backward_encoder_once,
    _checkpoint_patience_improved,
    _detach_encoded_for_microbatch_accumulation,
    _sample_train,
    _state_candidates_in_graph,
    candidate_microbatch_indices,
)
from cc_hhgt.training_data import _label_independent_sample, _proportional_quotas


def _frame(positive: int, unlabeled: int) -> pd.DataFrame:
    rows = []
    for label, count in [(1.0, positive), (0.0, unlabeled)]:
        for index in range(count):
            rows.append(
                {
                    "candidate_id": f"{int(label)}-{index}",
                    "proxy_label": label,
                    "cancer_id": f"C{index % 3}",
                    "state_id": f"S{index % 4}",
                }
            )
    return pd.DataFrame(rows)


def test_state_sampler_retains_unlabeled_when_positives_exceed_cap():
    sampled = _sample_train(_frame(positive=1800, unlabeled=5000), cap=1000, seed=17, unlabeled_to_positive_ratio=4.0)
    counts = sampled.proxy_label.value_counts()
    assert len(sampled) == 1000
    assert counts[1.0] == 200
    assert counts[0.0] == 800
    assert sampled.groupby(["cancer_id", "state_id"]).size().gt(0).all()


def test_state_sampler_is_deterministic():
    frame = _frame(positive=1800, unlabeled=5000)
    first = _sample_train(frame, cap=777, seed=29, unlabeled_to_positive_ratio=4.0)
    second = _sample_train(frame, cap=777, seed=29, unlabeled_to_positive_ratio=4.0)
    assert first.candidate_id.tolist() == second.candidate_id.tolist()


def test_state_sampler_fails_closed_for_single_class_input():
    with pytest.raises(RuntimeError, match="both positive and unlabeled"):
        _sample_train(_frame(positive=100, unlabeled=0), cap=50, seed=1)


def test_state_candidates_are_filtered_to_graph_before_sampling():
    frame = pd.DataFrame(
        {
            "lncrna_id": ["L1", "L2", "L1", "L1"],
            "state_id": ["S1", "S1", "S2", "S1"],
            "cancer_id": ["C1", "C1", "C1", "C2"],
            "proxy_label": [1.0, 0.0, 0.0, 0.0],
        }
    )
    bundle = SimpleNamespace(
        node_maps={"lncRNA": {"L1": 0}, "state": {"S1": 0}, "cancer": {"C1": 0}}
    )
    mapped = _state_candidates_in_graph(frame, bundle)
    assert mapped[["lncrna_id", "state_id", "cancer_id"]].values.tolist() == [["L1", "S1", "C1"]]


def test_pathway_file_sampling_quota_is_exact_and_proportional():
    quotas = _proportional_quotas(pd.Series([10, 20, 70]).to_numpy(), 25)
    assert quotas.tolist() == [3, 5, 17]
    assert int(quotas.sum()) == 25


def test_evaluation_sampling_is_fixed_and_label_independent():
    frame = pd.DataFrame(
        {
            "candidate_id": [f"c{i}" for i in range(1000)],
            "label_class": ["strong_positive"] * 500 + ["unlabeled"] * 500,
        }
    )
    first = _label_independent_sample(frame, 100, seed=23)
    relabeled = frame.assign(label_class="unlabeled")
    second = _label_independent_sample(relabeled, 100, seed=23)
    assert first.candidate_id.tolist() == second.candidate_id.tolist()


def test_candidate_microbatches_are_deterministic_mixed_and_exact():
    labels = np.asarray([1] * 20 + [0] * 80, dtype=float)
    first = candidate_microbatch_indices(labels, microbatch_size=25, seed=19)
    second = candidate_microbatch_indices(labels, microbatch_size=25, seed=19)
    assert len(first) == 4
    assert all(np.array_equal(left, right) for left, right in zip(first, second, strict=True))
    observed = np.concatenate(first)
    assert sorted(observed.tolist()) == list(range(100))
    assert all(set(labels[index].tolist()) == {0.0, 1.0} for index in first)


def test_detached_microbatch_accumulation_matches_single_full_backward():
    torch = pytest.importorskip("torch")
    torch.manual_seed(31)

    class TinyEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.left = torch.nn.Linear(3, 4, dtype=torch.float64)
            self.right = torch.nn.Linear(3, 4, dtype=torch.float64)

        def forward(self, value):
            return {"left": self.left(value), "right": self.right(value)}

    direct_encoder = TinyEncoder()
    accumulated_encoder = TinyEncoder()
    accumulated_encoder.load_state_dict(direct_encoder.state_dict())
    direct_decoder = torch.nn.Linear(8, 1, dtype=torch.float64)
    accumulated_decoder = torch.nn.Linear(8, 1, dtype=torch.float64)
    accumulated_decoder.load_state_dict(direct_decoder.state_dict())
    value = torch.randn(9, 3, dtype=torch.float64)
    batches = [torch.tensor([0, 2, 4, 6]), torch.tensor([1, 3, 5, 7, 8])]
    scales = [0.4, 0.6]

    direct = direct_encoder(value)
    direct_loss = sum(
        scale
        * direct_decoder(
            torch.cat([direct["left"][index], direct["right"][index]], dim=1)
        ).square().mean()
        for index, scale in zip(batches, scales, strict=True)
    )
    direct_loss.backward()

    encoded = accumulated_encoder(value)
    leaf = _detach_encoded_for_microbatch_accumulation(encoded)
    for index, scale in zip(batches, scales, strict=True):
        loss = scale * accumulated_decoder(
            torch.cat([leaf["left"][index], leaf["right"][index]], dim=1)
        ).square().mean()
        loss.backward()
    assert _backward_encoder_once(torch, encoded, leaf) == 1

    for direct_parameter, accumulated_parameter in zip(
        direct_encoder.parameters(), accumulated_encoder.parameters(), strict=True
    ):
        assert torch.allclose(
            direct_parameter.grad, accumulated_parameter.grad, atol=1e-12, rtol=1e-10
        )
    for direct_parameter, accumulated_parameter in zip(
        direct_decoder.parameters(), accumulated_decoder.parameters(), strict=True
    ):
        assert torch.allclose(
            direct_parameter.grad, accumulated_parameter.grad, atol=1e-12, rtol=1e-10
        )


def test_checkpoint_patience_requires_registered_minimum_gain():
    assert _checkpoint_patience_improved(0.5, -np.inf, 1e-4)
    assert not _checkpoint_patience_improved(0.5, 0.5, 0.0)
    assert not _checkpoint_patience_improved(0.500099, 0.5, 1e-4)
    assert _checkpoint_patience_improved(0.5001, 0.5, 1e-4)
    assert not _checkpoint_patience_improved(np.nan, 0.5, 1e-4)
