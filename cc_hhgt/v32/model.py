from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


FALLBACK_ATOL = 1e-7


def compose_bounded_residual_numpy(
    base_logit,
    raw_residual,
    graph_gate,
    *,
    available=True,
    admitted: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    base = np.asarray(base_logit, dtype=float)
    residual = np.asarray(raw_residual, dtype=float)
    gate = np.asarray(graph_gate, dtype=float)
    availability = np.asarray(available, dtype=bool)
    if residual.shape != base.shape or gate.shape != base.shape:
        raise ValueError("base_logit, raw_residual and graph_gate must have identical shapes")
    if availability.ndim == 0:
        availability = np.full(base.shape, bool(availability), dtype=bool)
    if availability.shape != base.shape:
        raise ValueError("availability must align to base_logit")
    if ((gate < 0) | (gate > 1) | ~np.isfinite(gate)).any():
        raise ValueError("graph_gate must be finite and bounded within [0, 1]")
    contribution = gate * np.tanh(residual)
    contribution = np.where(availability & bool(admitted), contribution, 0.0)
    return base + contribution, contribution


def assert_exact_l1_fallback(base_logit, final_logit, *, atol: float = FALLBACK_ATOL) -> None:
    base = np.asarray(base_logit, dtype=float)
    final = np.asarray(final_logit, dtype=float)
    if base.shape != final.shape:
        raise RuntimeError("L1 fallback shapes differ")
    error = float(np.max(np.abs(base - final))) if base.size else 0.0
    if not np.isfinite(error) or error > float(atol):
        raise RuntimeError(f"Exact L1 fallback failed: max_abs_error={error}")


def probability_from_logit(logit) -> np.ndarray:
    value = np.asarray(logit, dtype=float)
    positive = value >= 0
    result = np.empty_like(value)
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def build_v32_cc_hhgt_residual(
    encoder: Any,
    *,
    hidden_channels: int = 96,
    context_features: int = 4,
    dropout: float = 0.20,
):
    """Build the cancer-conditioned bounded residual around an HGT encoder.

    Torch is imported lazily so importing or dry-running the V3.2 package can
    never initialize CUDA.  ``encoder`` must expose ``encode(graph)`` and
    return embeddings keyed by ``lncRNA``, ``pathway`` and ``cancer``.
    """

    import torch
    from torch import nn

    if hidden_channels < 4:
        raise ValueError("hidden_channels must be at least four")
    if context_features < 1:
        raise ValueError("context_features must be positive")

    class CancerConditionedResidual(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = encoder
            # l, p, c, three Hadamard interactions and three absolute
            # differences explicitly represent the cancer/pathway context.
            decoder_input = hidden_channels * 9
            self.residual_map = nn.Sequential(
                nn.Linear(decoder_input, hidden_channels),
                nn.LayerNorm(hidden_channels),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.residual_output = nn.Linear(hidden_channels, 1)
            self.direction_output = nn.Sequential(
                nn.Linear(decoder_input, hidden_channels // 2),
                nn.GELU(),
                nn.Linear(hidden_channels // 2, 1),
            )
            self.gate = nn.Sequential(
                nn.Linear(context_features, hidden_channels // 2),
                nn.GELU(),
                nn.Linear(hidden_channels // 2, 1),
                nn.Sigmoid(),
            )
            # Epoch zero is exactly the L1 base, independently of encoder
            # embeddings and gate values.
            nn.init.zeros_(self.residual_output.weight)
            nn.init.zeros_(self.residual_output.bias)

        @staticmethod
        def _features(lnc, pathway, cancer):
            if lnc.shape != pathway.shape or lnc.shape != cancer.shape:
                raise ValueError("Candidate embeddings must have identical shapes")
            return torch.cat(
                [
                    lnc,
                    pathway,
                    cancer,
                    lnc * pathway,
                    lnc * cancer,
                    pathway * cancer,
                    torch.abs(lnc - pathway),
                    torch.abs(lnc - cancer),
                    torch.abs(pathway - cancer),
                ],
                dim=-1,
            )

        def forward(
            self,
            graph,
            batch: Mapping[str, Any],
            base_logit,
            conservation_context,
            graph_available=None,
            *,
            admitted: bool = True,
            encoded: Mapping[str, Any] | None = None,
        ):
            if encoded is None:
                encoded = self.encoder.encode(graph)
            required = {"l", "p", "c"}
            if missing := sorted(required - set(batch)):
                raise ValueError(f"Candidate batch lacks indices: {missing}")
            lnc = encoded["lncRNA"][batch["l"]]
            pathway = encoded["pathway"][batch["p"]]
            cancer = encoded["cancer"][batch["c"]]
            features = self._features(lnc, pathway, cancer)
            raw_residual = self.residual_output(self.residual_map(features)).squeeze(-1)
            graph_gate = self.gate(conservation_context).squeeze(-1)
            if graph_available is None:
                graph_available = torch.ones_like(base_logit, dtype=torch.bool)
            contribution = graph_gate * torch.tanh(raw_residual)
            contribution = torch.where(
                graph_available.to(dtype=torch.bool) & bool(admitted),
                contribution,
                torch.zeros_like(contribution),
            )
            final_logit = base_logit + contribution
            return {
                "final_logit": final_logit,
                "graph_residual": contribution,
                "raw_graph_residual": raw_residual,
                "graph_gate": graph_gate,
                "direction_logit": self.direction_output(features).squeeze(-1),
            }

        @staticmethod
        def residual_shrinkage(raw_residual, coefficient: float = 1e-4):
            if coefficient < 0:
                raise ValueError("Residual shrinkage coefficient must be non-negative")
            return float(coefficient) * raw_residual.square().mean()

    return CancerConditionedResidual()


def build_v32_hierarchical_evidence_hhgt(
    encoder: Any,
    *,
    hidden_channels: int = 96,
    context_features: int = 4,
    modality_names: tuple[str, ...] = ("mutation", "cnv", "atac"),
    dropout: float = 0.20,
):
    """Build an end-to-end HHGT residual with cancer-conditioned evidence gates.

    Patient-level OOF modality probabilities are encoded as messages after the
    HGT encoder and before the cancer/lncRNA/pathway residual decoder.  A
    learned global gate and a cancer-embedding gate jointly control each
    modality.  Typed unavailable values are multiplied out exactly, so an
    absent modality cannot change the score or receive a gradient.

    This is a separate routed candidate architecture; it does not alter the
    formal V3.2 builder above.
    """

    import torch
    from torch import nn

    modalities = tuple(map(str, modality_names))
    if not modalities or len(modalities) != len(set(modalities)):
        raise ValueError("modality_names must be distinct and non-empty")
    if hidden_channels < 4 or context_features < 1:
        raise ValueError("Hierarchical HHGT dimensions are invalid")

    class HierarchicalEvidenceHHGT(nn.Module):
        integration_stage = "POST_HGT_PRE_RESIDUAL_GATED_MESSAGE"

        def __init__(self) -> None:
            super().__init__()
            self.encoder = encoder
            self.modality_names = modalities
            width = len(modalities)
            self.global_modality_logit = nn.Parameter(torch.zeros(width))
            self.cancer_modality_gate = nn.Linear(hidden_channels, width, bias=False)
            self.signal_modality_gate = nn.Parameter(torch.zeros(width))
            self.modality_message = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(1, hidden_channels),
                        nn.GELU(),
                        nn.Linear(hidden_channels, hidden_channels),
                    )
                    for _ in modalities
                ]
            )
            self.lncrna_message_projection = nn.Linear(hidden_channels, hidden_channels, bias=False)
            self.pathway_message_projection = nn.Linear(hidden_channels, hidden_channels, bias=False)
            decoder_input = hidden_channels * 9
            self.residual_map = nn.Sequential(
                nn.Linear(decoder_input, hidden_channels),
                nn.LayerNorm(hidden_channels),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.residual_output = nn.Linear(hidden_channels, 1)
            self.direction_output = nn.Sequential(
                nn.Linear(decoder_input, hidden_channels // 2),
                nn.GELU(),
                nn.Linear(hidden_channels // 2, 1),
            )
            self.graph_gate = nn.Sequential(
                nn.Linear(context_features, hidden_channels // 2),
                nn.GELU(),
                nn.Linear(hidden_channels // 2, 1),
                nn.Sigmoid(),
            )
            nn.init.zeros_(self.residual_output.weight)
            nn.init.zeros_(self.residual_output.bias)

        @staticmethod
        def _features(lnc, pathway, cancer):
            return torch.cat(
                [
                    lnc,
                    pathway,
                    cancer,
                    lnc * pathway,
                    lnc * cancer,
                    pathway * cancer,
                    torch.abs(lnc - pathway),
                    torch.abs(lnc - cancer),
                    torch.abs(pathway - cancer),
                ],
                dim=-1,
            )

        def _evidence_message(self, cancer, probability, available):
            if probability.ndim != 2 or probability.shape[1] != len(modalities):
                raise ValueError("modality_probability must have shape [batch, modalities]")
            if available.shape != probability.shape:
                raise ValueError("modality_available must align to modality_probability")
            availability = available.to(dtype=torch.bool)
            valid_values = probability[availability]
            if valid_values.numel() and (
                (~torch.isfinite(valid_values)).any()
                or (valid_values < 0).any()
                or (valid_values > 1).any()
            ):
                raise ValueError("Available modality probabilities must be finite in [0, 1]")
            neutral = torch.full_like(probability, 0.5)
            safe_probability = torch.where(availability, probability, neutral)
            clipped = safe_probability.clamp(1e-6, 1 - 1e-6)
            signal = torch.logit(clipped)
            gate_logit = (
                self.global_modality_logit[None, :]
                + self.cancer_modality_gate(cancer)
                + signal * self.signal_modality_gate[None, :]
            )
            gates = torch.sigmoid(gate_logit) * availability.to(dtype=probability.dtype)
            messages = torch.stack(
                [layer(signal[:, index : index + 1]) for index, layer in enumerate(self.modality_message)],
                dim=1,
            )
            gated = messages * gates.unsqueeze(-1)
            return gated.sum(dim=1), gates, gated

        def forward(
            self,
            graph,
            batch: Mapping[str, Any],
            base_logit,
            conservation_context,
            graph_available=None,
            *,
            admitted: bool = True,
            encoded: Mapping[str, Any] | None = None,
        ):
            if encoded is None:
                encoded = self.encoder.encode(graph)
            required = {"l", "p", "c", "modality_probability", "modality_available"}
            if missing := sorted(required - set(batch)):
                raise ValueError(f"Hierarchical candidate batch lacks fields: {missing}")
            lnc = encoded["lncRNA"][batch["l"]]
            pathway = encoded["pathway"][batch["p"]]
            cancer = encoded["cancer"][batch["c"]]
            message, modality_gates, gated_messages = self._evidence_message(
                cancer,
                batch["modality_probability"],
                batch["modality_available"],
            )
            lnc = lnc + self.lncrna_message_projection(message)
            pathway = pathway + self.pathway_message_projection(message)
            features = self._features(lnc, pathway, cancer)
            raw_residual = self.residual_output(self.residual_map(features)).squeeze(-1)
            graph_gate = self.graph_gate(conservation_context).squeeze(-1)
            if graph_available is None:
                graph_available = torch.ones_like(base_logit, dtype=torch.bool)
            contribution = graph_gate * torch.tanh(raw_residual)
            contribution = torch.where(
                graph_available.to(dtype=torch.bool) & bool(admitted),
                contribution,
                torch.zeros_like(contribution),
            )
            return {
                "final_logit": base_logit + contribution,
                "graph_residual": contribution,
                "raw_graph_residual": raw_residual,
                "graph_gate": graph_gate,
                "direction_logit": self.direction_output(features).squeeze(-1),
                "modality_gates": modality_gates,
                "gated_modality_messages": gated_messages,
                "evidence_message": message,
                "integration_stage": self.integration_stage,
            }

        @staticmethod
        def residual_shrinkage(raw_residual, coefficient: float = 1e-4):
            if coefficient < 0:
                raise ValueError("Residual shrinkage coefficient must be non-negative")
            return float(coefficient) * raw_residual.square().mean()

    return HierarchicalEvidenceHHGT()


def nnpu_membership_loss(
    logits,
    proxy_label,
    weak_positive,
    *,
    positive_prior: float = 0.10,
    unlabeled_weight: float = 0.12,
    weak_positive_weight: float = 0.35,
):
    """Non-negative PU risk imported lazily for the authorized trainer."""

    import torch
    import torch.nn.functional as functional

    positive = proxy_label > 0.5
    unlabeled = ~positive
    zero = logits.sum() * 0.0
    if positive.any():
        weights = torch.where(
            weak_positive[positive].to(dtype=torch.bool),
            torch.full_like(logits[positive], float(weak_positive_weight)),
            torch.ones_like(logits[positive]),
        )
        denominator = weights.sum().clamp_min(1e-8)
        positive_loss = functional.binary_cross_entropy_with_logits(
            logits[positive], torch.ones_like(logits[positive]), reduction="none"
        )
        positive_as_negative = functional.binary_cross_entropy_with_logits(
            logits[positive], torch.zeros_like(logits[positive]), reduction="none"
        )
        positive_risk = float(positive_prior) * (positive_loss * weights).sum() / denominator
        positive_negative_risk = (
            float(positive_prior) * (positive_as_negative * weights).sum() / denominator
        )
    else:
        positive_risk = zero
        positive_negative_risk = zero
    if unlabeled.any():
        unlabeled_risk = functional.binary_cross_entropy_with_logits(
            logits[unlabeled], torch.zeros_like(logits[unlabeled])
        )
    else:
        unlabeled_risk = zero
    return positive_risk + float(unlabeled_weight) * torch.clamp(
        unlabeled_risk - positive_negative_risk, min=0.0
    )
