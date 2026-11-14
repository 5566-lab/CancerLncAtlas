import torch
import pytest

from cla_engine.model import loss_one


def batch():
    return {'available': torch.tensor([True, True, False]),
            'weight': torch.ones(3),
            'label': torch.tensor([.25, .5, 0.]),
            'direction_available': torch.tensor([False, False, False]),
            'direction_label': torch.zeros(3)}


def test_continuous_loss_ignores_unobserved_and_backpropagates():
    score = torch.tensor([.25, .3, 1000.], requires_grad=True)
    out = {'logit': score, 'delta': torch.zeros(3), 'direction': torch.zeros(3)}
    loss = loss_one(out, batch(), objective='observed_fused_rms_effect')
    assert float(loss.detach()) == pytest.approx(.02)
    loss.backward()
    assert score.grad.tolist() == pytest.approx([0., -.2, 0.])


def test_continuous_loss_rejects_invalid_observed_label():
    data = batch()
    data['label'][0] = float('nan')
    out = {'logit': torch.zeros(3), 'delta': torch.zeros(3), 'direction': torch.zeros(3)}
    with pytest.raises(Exception, match='Observed RMS effect must be finite'):
        loss_one(out, data, objective='observed_fused_rms_effect')
