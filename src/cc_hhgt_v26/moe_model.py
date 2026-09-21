"""Availability-masked mixture-of-experts utilities.

The gate is trained only on out-of-fold expert predictions.  Unavailable
experts are masked before softmax and therefore receive exactly zero weight.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from torch import nn


def probability_to_logit(probability: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    value = np.clip(np.asarray(probability, dtype=np.float32), eps, 1 - eps)
    return np.log(value / (1 - value)).astype(np.float32)


def sigmoid_numpy(logit: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(logit, dtype=np.float64), -40, 40)
    return (1.0 / (1.0 + np.exp(-value))).astype(np.float32)


class AvailabilityMaskedMoE(nn.Module):
    def __init__(self, n_experts: int, quality_dim: int, hidden: int = 32) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.gate = nn.Sequential(
            nn.Linear(n_experts * 2 + quality_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden, n_experts),
        )

    def forward(
        self,
        expert_logits: torch.Tensor,
        availability: torch.Tensor,
        quality: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        available = availability.bool()
        if torch.any(available.sum(dim=-1) == 0):
            raise RuntimeError("MoE received a row with no available expert")
        gate_input = torch.cat(
            [
                torch.nan_to_num(expert_logits, nan=0.0, posinf=10.0, neginf=-10.0),
                availability.float(),
                quality,
            ],
            dim=-1,
        )
        gate_logits = self.gate(gate_input).masked_fill(~available, -1e9)
        weights = torch.softmax(gate_logits, dim=-1)
        final_logit = torch.sum(
            weights * torch.nan_to_num(expert_logits, nan=0.0), dim=-1
        )
        return final_logit, weights


@dataclass
class MoEFitResult:
    model: AvailabilityMaskedMoE
    state_dict: dict[str, torch.Tensor]
    history: list[dict[str, float]]
    validation_metrics: dict[str, float]


def _metrics(labels: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(labels) & np.isfinite(probability)
    labels = labels[valid]
    probability = probability[valid]
    if not len(labels):
        return {"auprc": math.nan, "auroc": math.nan, "brier": math.nan}
    return {
        "auprc": float(average_precision_score(labels, probability))
        if len(np.unique(labels)) > 1
        else math.nan,
        "auroc": float(roc_auc_score(labels, probability))
        if len(np.unique(labels)) > 1
        else math.nan,
        "brier": float(brier_score_loss(labels, probability)),
    }


def fit_moe(
    train_expert_logits: np.ndarray,
    train_availability: np.ndarray,
    train_quality: np.ndarray,
    train_labels: np.ndarray,
    validation_expert_logits: np.ndarray,
    validation_availability: np.ndarray,
    validation_quality: np.ndarray,
    validation_labels: np.ndarray,
    *,
    seed: int = 20260731,
    epochs: int = 80,
    patience: int = 10,
    device: str | None = None,
) -> MoEFitResult:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    valid_train = (
        np.isfinite(train_labels)
        & (np.asarray(train_availability).sum(axis=1) > 0)
    )
    valid_validation = (
        np.isfinite(validation_labels)
        & (np.asarray(validation_availability).sum(axis=1) > 0)
    )
    if valid_train.sum() < 100 or len(np.unique(train_labels[valid_train])) < 2:
        raise RuntimeError("insufficient OOF rows/classes to train the MoE gate")

    train_logits_t = torch.as_tensor(train_expert_logits[valid_train], device=torch_device)
    train_availability_t = torch.as_tensor(train_availability[valid_train], device=torch_device)
    train_quality_t = torch.as_tensor(train_quality[valid_train], device=torch_device)
    train_labels_t = torch.as_tensor(train_labels[valid_train], dtype=torch.float32, device=torch_device)
    validation_logits_t = torch.as_tensor(validation_expert_logits[valid_validation], device=torch_device)
    validation_availability_t = torch.as_tensor(validation_availability[valid_validation], device=torch_device)
    validation_quality_t = torch.as_tensor(validation_quality[valid_validation], device=torch_device)

    model = AvailabilityMaskedMoE(
        train_expert_logits.shape[1], train_quality.shape[1]
    ).to(torch_device)
    positive = float(train_labels_t.sum().item())
    negative = float(len(train_labels_t) - positive)
    pos_weight = torch.tensor(
        min(max(negative / max(positive, 1.0), 1.0), 20.0),
        device=torch_device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=2e-3)
    best_state = None
    best_score = -math.inf
    stale = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        final_logit, weights = model(
            train_logits_t, train_availability_t, train_quality_t
        )
        loss = criterion(final_logit, train_labels_t)
        # Small entropy term discourages premature single-expert collapse while
        # preserving fully learned, row-specific routing.
        entropy = -torch.sum(weights * torch.log(weights.clamp_min(1e-8)), dim=-1).mean()
        objective = loss - 0.01 * entropy
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_logit, validation_weight = model(
                validation_logits_t,
                validation_availability_t,
                validation_quality_t,
            )
            validation_probability = torch.sigmoid(validation_logit).cpu().numpy()
        metrics = _metrics(validation_labels[valid_validation], validation_probability)
        score = metrics["auprc"]
        if not np.isfinite(score):
            score = -float(objective.item())
        history.append(
            {
                "epoch": epoch,
                "loss": float(loss.item()),
                "entropy": float(entropy.item()),
                "validation_auprc": float(metrics["auprc"]),
                "validation_auroc": float(metrics["auroc"]),
                "mean_max_weight": float(validation_weight.max(dim=1).values.mean().item()),
            }
        )
        if score > best_score + 1e-5:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("MoE training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        validation_logit, _ = model(
            validation_logits_t,
            validation_availability_t,
            validation_quality_t,
        )
    metrics = _metrics(
        validation_labels[valid_validation],
        torch.sigmoid(validation_logit).cpu().numpy(),
    )
    return MoEFitResult(model, best_state, history, metrics)


def predict_moe(
    model: AvailabilityMaskedMoE,
    expert_logits: np.ndarray,
    availability: np.ndarray,
    quality: np.ndarray,
    *,
    device: str | None = None,
    batch_size: int = 200000,
) -> tuple[np.ndarray, np.ndarray]:
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(torch_device)
    model.eval()
    logits_out: list[np.ndarray] = []
    weight_out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(expert_logits), batch_size):
            stop = min(start + batch_size, len(expert_logits))
            final_logit, weights = model(
                torch.as_tensor(expert_logits[start:stop], device=torch_device),
                torch.as_tensor(availability[start:stop], device=torch_device),
                torch.as_tensor(quality[start:stop], device=torch_device),
            )
            logits_out.append(final_logit.cpu().numpy())
            weight_out.append(weights.cpu().numpy())
    return np.concatenate(logits_out), np.concatenate(weight_out)
