from __future__ import annotations

import pandas as pd

from cc_hhgt.v29_multitask import (
    StateBatch,
    StateDecoder,
    _pathway_forward,
    _state_forward,
)


def test_pathway_forward_adds_frozen_base_without_overwrite() -> None:
    import torch

    class Decoder:
        def __call__(self, lnc, target, cancer, pair):
            return torch.full((len(lnc),), 0.25), torch.zeros(len(lnc))

    class Model:
        decoder = Decoder()

    encoded = {
        "lncRNA": torch.randn(2, 3),
        "pathway_family": torch.randn(2, 3),
        "cancer": torch.randn(2, 3),
    }
    batch = {
        "l": torch.tensor([0, 1]),
        "p": torch.tensor([1, 0]),
        "c": torch.tensor([0, 1]),
        "x": torch.randn(2, 4),
        "base_logit": torch.tensor([-1.0, 2.0]),
    }
    final, _, residual = _pathway_forward(Model(), encoded, batch, "cc_hhgt")
    assert torch.equal(residual, torch.tensor([0.25, 0.25]))
    assert torch.equal(final, torch.tensor([-0.75, 2.25]))


def test_state_residual_decoder_is_zero_at_epoch_zero() -> None:
    import torch

    hidden = 4
    decoder = StateDecoder.build(
        hidden, 0.0, ["RNAss", "DNAss"], zero_initialize_membership=True
    )
    encoded = {
        "lncRNA": torch.randn(2, hidden),
        "state": torch.randn(2, hidden),
        "cancer": torch.randn(2, hidden),
    }
    base = torch.tensor([-0.75, 1.25])
    batch = StateBatch(
        frame=pd.DataFrame({"candidate_id": ["a", "b"]}),
        l=torch.tensor([0, 1]),
        s=torch.tensor([0, 1]),
        c=torch.tensor([0, 1]),
        y=torch.tensor([0.0, 1.0]),
        weak=torch.tensor([False, False]),
        direction=torch.tensor([-1.0, 1.0]),
        head=torch.tensor([0, 1]),
        base_logit=base,
    )
    final, _, residual = _state_forward(decoder, encoded, batch, "cc_hhgt")
    assert torch.equal(residual, torch.zeros_like(residual))
    assert torch.equal(final, base)
