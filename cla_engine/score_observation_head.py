"""Optional observation head on the original HHGT encoder; explicit targets only."""
import torch
from torch import nn


class ScoreObservationHead(nn.Module):
    def __init__(self, hidden, selected_score_ids):
        super().__init__()
        ids=tuple(selected_score_ids)
        if not ids or len(ids)!=len(set(ids)):raise ValueError('EXPLICIT_UNIQUE_SCORE_TARGET_SELECTION_REQUIRED')
        self.score_ids=ids
        # Separate original readouts can have different units/directions even
        # when they share scope. Do not force identical predictions for them.
        self.weight=nn.Parameter(torch.empty(len(ids),hidden))
        self.bias=nn.Parameter(torch.zeros(len(ids)))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, encoded, spec):
        if tuple(spec['nodes'].get('score_target',()))!=self.score_ids:
            raise ValueError('SCORE_HEAD_AXIS_MISMATCH')
        h=encoded[0] if isinstance(encoded,tuple) else encoded
        return (h['score_target']*self.weight).sum(-1)+self.bias
