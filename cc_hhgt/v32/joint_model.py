"""Pathway + State association heads on one trainable HHGT encoder.

The two tasks keep separate targets and scores.  Explicit task weights apply to the
training objective and checkpoint selection, not to unrelated prediction rows.
Direct literature/event evidence is deliberately not an input to either head.

Loss construction deliberately lives in :mod:`cla_engine.model` and
:mod:`cla_engine.train`.  The active R6.2 entrypoint uses availability-masked BCE
for each task and combines the normalized task losses at Pathway=1.0, State=0.5.
The retired prepared-payload runner's nnPU ``joint_loss`` is not exposed here.
"""
from __future__ import annotations
from collections.abc import Mapping
from typing import Any
import torch
from torch import nn

TASKS = ("pathway", "state")
TASK_WEIGHTS = {"pathway": 1.0, "state": 0.5}
JOINT_ARCHITECTURE = "V32_FULL_PYG_SHARED_HHGT_PATHWAY_STATE_1_05_R6"

class JointPathwayStateModel(nn.Module):
    def __init__(self, encoder: nn.Module, *, state_ids: tuple[str,...], hidden_channels: int=96,
                 context_features: int=4, dropout: float=.2):
        super().__init__()
        from .model import build_v32_cc_hhgt_residual
        if not state_ids or len(set(state_ids)) != len(state_ids):
            raise ValueError("State IDs must be a nonempty ordered unique tuple")
        # Own the shared encoder exactly once.  The pathway model already registers it.
        self.pathway_model=build_v32_cc_hhgt_residual(encoder, hidden_channels=hidden_channels,
                                                    context_features=context_features, dropout=dropout)
        self.state_ids=tuple(state_ids)
        self.state_embedding=nn.Embedding(len(state_ids),hidden_channels)
        width=9*hidden_channels
        self.state_residual_map=nn.Sequential(nn.Linear(width,hidden_channels),nn.LayerNorm(hidden_channels),
                                             nn.GELU(),nn.Dropout(dropout))
        self.state_residual_output=nn.Linear(hidden_channels,1)
        self.state_direction=nn.Sequential(nn.Linear(width,hidden_channels//2),nn.GELU(),nn.Linear(hidden_channels//2,1))
        self.state_gate=nn.Sequential(nn.Linear(context_features,hidden_channels//2),nn.GELU(),
                                      nn.Linear(hidden_channels//2,1),nn.Sigmoid())
        nn.init.zeros_(self.state_residual_output.weight);nn.init.zeros_(self.state_residual_output.bias)

    @property
    def encoder(self):
        return self.pathway_model.encoder

    def forward_task(self, task: str, graph: Any, batch: Mapping[str,Any], *, encoded=None):
        if task not in TASKS: raise ValueError(f"Unknown joint task {task!r}")
        if encoded is None: encoded=self.encoder.encode(graph)
        candidates=batch["candidate_batch"]
        base=batch["base_logit"]
        context=batch["conservation_context"]
        if task=="pathway":
            return self.pathway_model(graph,candidates,base,context,batch.get("graph_available"),encoded=encoded)
        l=encoded["lncRNA"][candidates["l"]]
        c=encoded["cancer"][candidates["c"]]
        s=self.state_embedding(candidates["s"])
        # No detach and no exported embedding matrix: this graph remains connected
        # to the exact same encoder optimized by the Pathway task.
        features=self.pathway_model._features(l,s,c)
        raw=self.state_residual_output(self.state_residual_map(features)).squeeze(-1)
        gate=self.state_gate(context).squeeze(-1)
        available=batch.get("graph_available",torch.ones_like(base,dtype=torch.bool))
        delta=torch.where(available.bool(),gate*torch.tanh(raw),torch.zeros_like(raw))
        return {"final_logit":base+delta,"graph_residual":delta,"raw_graph_residual":raw,
                "graph_gate":gate,"direction_logit":self.state_direction(features).squeeze(-1)}

def weighted_task_selection(pathway_macro_auprc: float, state_macro_auprc: float) -> float:
    import math
    values=(float(pathway_macro_auprc),float(state_macro_auprc))
    if not all(math.isfinite(x) and 0<=x<=1 for x in values):
        raise ValueError("Both task scores must be finite AUPRC values in [0,1]")
    return (values[0]+.5*values[1])/1.5

# Compatibility name only; the implementation now uses the frozen 1:0.5 ratio.
equal_task_selection = weighted_task_selection
