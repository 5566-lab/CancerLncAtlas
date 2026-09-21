"""Lightweight set encoders used by the legacy online query service."""

from __future__ import annotations

import torch
from torch import nn


class AttentionSetEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 96, hidden: int = 96) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, embedding_dim),
        )

    def forward(
        self, values: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        score = self.attention(values).squeeze(-1)
        score = score.masked_fill(~mask, -1e9)
        weight = torch.softmax(score, dim=-1)
        pooled = torch.sum(values * weight.unsqueeze(-1), dim=1)
        return self.projection(pooled)


class ProteinInteractionDecoder(nn.Module):
    def __init__(self, embedding_dim: int = 96, hidden: int = 96) -> None:
        super().__init__()
        self.protein_encoder = AttentionSetEncoder(embedding_dim, hidden)
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim * 4, hidden),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(
        self,
        lnc: torch.Tensor,
        protein_values: torch.Tensor,
        protein_mask: torch.Tensor,
    ) -> torch.Tensor:
        protein = self.protein_encoder(protein_values, protein_mask)
        features = torch.cat(
            [lnc, protein, lnc * protein, torch.abs(lnc - protein)], dim=-1
        )
        return self.decoder(features).squeeze(-1)
