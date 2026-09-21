"""Direct lncRNA--tumor-state modeling with frozen graph embeddings.

This module adds a separate biological task:

    cancer x lncRNA x tumor_state

Tumor states (for example EXTEND telomerase activity, RNAss, DNAss and
EREG.EXPss) are not treated as ordinary pathway families.  A patient-native
MLP learns train-patient lncRNA/state association reproducibility.  Three
lightweight graph decoders use frozen R-GCN, HGT and CC-HHGT embeddings.  No
strict graph encoder is retrained.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from torch import nn

from .adapter_model import fit_temperature


ROOT = Path(__file__).resolve().parents[2]
RESULT = Path(os.getenv("CC_HHGT_RESULT_ROOT", str(ROOT / "results")))
STATE_DATA = Path(os.getenv("CC_HHGT_STATE_DATA_ROOT", str(RESULT / "lncrna_state_data")))
STATE_MODEL_ROOT = Path(os.getenv("CC_HHGT_STATE_MODEL_ROOT", str(ROOT / "models" / "lncrna_state_experts")))
STATE_RESULT_ROOT = Path(os.getenv("CC_HHGT_STATE_RESULT_ROOT", str(RESULT / "lncrna_state_experts")))
SEEDS = [20260726, 20261726, 20262726]
GRAPH_MODELS = ["rgcn", "hgt", "cc_hhgt_strict"]
V29_STRICT_ROOT = Path(os.getenv("CC_HHGT_V29_STRICT_ROOT", str(RESULT / "model" / "cc_hhgt_v2_9_state_graph" / "v2_9_strict")))
if not V29_STRICT_ROOT.exists():
    V29_STRICT_ROOT = Path(os.getenv("CC_HHGT_V29_STRICT_ROOT", str(RESULT / "v2_9_strict")))
KEYS = ["cancer_id", "patient_fold_id", "lncrna_id", "state_id"]

PATIENT_FEATURES = [
    "train_effect",
    "train_fdr",
    "train_detection_rate",
    "train_n_patients",
    "train_state_mean",
    "train_state_sd",
    "train_state_missing_rate",
    "lncrna_available",
    "state_available",
    "pair_available",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def _numeric_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "canonical_id",
        "node_id",
        "node_type",
        "label",
        "node_index_within_type",
    }
    return [
        column
        for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    ]


def _embedding_frame(model: str, cancer: str, seed: int, node_type: str) -> pd.DataFrame | None:
    path = (
        V29_STRICT_ROOT
        / model
        / f"LOCO_{cancer}"
        / f"seed_{seed}"
        / "embeddings"
        / f"{node_type}.parquet"
    )
    if not path.exists():
        legacy = (
            ROOT
            / "models"
            / "strict_global"
            / model
            / f"LOCO_{cancer}"
            / f"seed_{seed}"
            / "embeddings"
            / f"{node_type}.parquet"
        )
        path = legacy if legacy.exists() else path
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    if "canonical_id" not in frame:
        return None
    return frame.drop_duplicates("canonical_id").set_index("canonical_id")


def _scaled_values(frame: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    value = frame.reindex(columns=columns).apply(pd.to_numeric, errors="coerce")
    value = value.replace([np.inf, -np.inf], np.nan)
    mean = value.mean(axis=0).fillna(0).to_numpy(np.float32)
    scale = value.std(axis=0).replace(0, 1).fillna(1).to_numpy(np.float32)
    array = value.fillna(pd.Series(mean, index=columns)).to_numpy(np.float32)
    return (array - mean) / scale, mean, scale


class StateExpertNetwork(nn.Module):
    """Patient-native head plus optional frozen-embedding graph decoders."""

    def __init__(
        self,
        patient_dim: int,
        graph_dims: dict[str, int],
        hidden: int = 96,
    ) -> None:
        super().__init__()
        self.patient_encoder = nn.Sequential(
            nn.Linear(patient_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.patient_membership_head = nn.Linear(hidden, 1)
        self.patient_direction_head = nn.Linear(hidden, 1)
        self.graph_heads = nn.ModuleDict()
        for model, dim in graph_dims.items():
            self.graph_heads[model] = nn.Sequential(
                nn.Linear(dim * 5, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(hidden, hidden // 2),
                nn.GELU(),
                nn.Linear(hidden // 2, 1),
            )

    def forward(
        self,
        patient_features: torch.Tensor,
        graph_features: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        hidden = self.patient_encoder(patient_features)
        result = {
            "patient_logit": self.patient_membership_head(hidden).squeeze(-1),
            "direction_logit": self.patient_direction_head(hidden).squeeze(-1),
        }
        for model, value in graph_features.items():
            if model in self.graph_heads:
                result[f"{model}_logit"] = self.graph_heads[model](value).squeeze(-1)
        return result


@dataclass
class PreparedStateFold:
    identity: pd.DataFrame
    patient_features: torch.Tensor
    graph_features: dict[str, torch.Tensor]
    graph_available: dict[str, torch.Tensor]
    train_label: torch.Tensor
    train_label_available: torch.Tensor
    validation_label: np.ndarray
    test_label: np.ndarray
    train_direction: torch.Tensor
    train_direction_available: torch.Tensor
    validation_direction: np.ndarray
    test_direction: np.ndarray
    patient_feature_columns: list[str]
    patient_mean: np.ndarray
    patient_scale: np.ndarray
    graph_dims: dict[str, int]


def _build_graph_feature(
    frame: pd.DataFrame,
    model: str,
    cancer: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int | None]:
    lnc = _embedding_frame(model, cancer, seed, "lncRNA")
    state = _embedding_frame(model, cancer, seed, "state")
    cancer_frame = _embedding_frame(model, cancer, seed, "cancer")
    if lnc is None or state is None or cancer_frame is None:
        return np.zeros((len(frame), 0), np.float32), np.zeros(len(frame), bool), None
    common = sorted(set(_numeric_columns(lnc)) & set(_numeric_columns(state)) & set(_numeric_columns(cancer_frame)))
    if not common:
        return np.zeros((len(frame), 0), np.float32), np.zeros(len(frame), bool), None
    dim = len(common)
    lnc_index = lnc.index.astype(str)
    state_index = state.index.astype(str)
    cancer_index = cancer_frame.index.astype(str)
    cancer_key = cancer if cancer in cancer_index else None
    if cancer_key is None:
        matches = [key for key in cancer_index if str(key).endswith(cancer)]
        cancer_key = matches[0] if matches else None
    if cancer_key is None:
        return np.zeros((len(frame), dim * 5), np.float32), np.zeros(len(frame), bool), dim

    available = frame["lncrna_id"].astype(str).isin(set(lnc_index)) & frame["state_id"].astype(str).isin(set(state_index))
    lnc_values = np.zeros((len(frame), dim), np.float32)
    state_values = np.zeros((len(frame), dim), np.float32)
    if available.any():
        idx = np.flatnonzero(available.to_numpy())
        lnc_values[idx] = lnc.loc[frame.loc[available, "lncrna_id"].astype(str), common].to_numpy(np.float32)
        state_values[idx] = state.loc[frame.loc[available, "state_id"].astype(str), common].to_numpy(np.float32)
    cancer_value = cancer_frame.loc[cancer_key, common].to_numpy(np.float32)
    cancer_values = np.repeat(cancer_value[None, :], len(frame), axis=0)
    feature = np.concatenate(
        [
            lnc_values,
            state_values,
            cancer_values,
            lnc_values * state_values,
            np.abs(lnc_values - state_values),
        ],
        axis=1,
    ).astype(np.float32)
    feature[~available.to_numpy()] = 0.0
    return feature, available.to_numpy(bool), dim


def prepare_state_fold(
    frame: pd.DataFrame,
    cancer: str,
    seed: int,
    device: torch.device,
) -> PreparedStateFold:
    patient_columns = [column for column in PATIENT_FEATURES if column in frame]
    optional_prefixes = ("mutation_", "tumor_state_", "sc_", "ucell_", "celltype_", "communication_")
    patient_columns += [
        column
        for column in frame.columns
        if column.startswith(optional_prefixes)
        and not column.startswith(("validation_", "test_"))
        and not column.endswith(("_label", "_pvalue"))
    ]
    patient_columns = sorted(set(patient_columns))
    if not patient_columns:
        raise RuntimeError("lncRNA-state patient head has no patient features")
    patient_values, patient_mean, patient_scale = _scaled_values(frame, patient_columns)

    graph_features: dict[str, torch.Tensor] = {}
    graph_available: dict[str, torch.Tensor] = {}
    graph_dims: dict[str, int] = {}
    for model in GRAPH_MODELS:
        values, available, dim = _build_graph_feature(frame, model, cancer, seed)
        if dim is None:
            continue
        graph_dims[model] = dim
        graph_features[model] = torch.as_tensor(values, device=device)
        graph_available[model] = torch.as_tensor(available, device=device)

    identity_columns = [
        "cancer_id",
        "patient_fold_id",
        "lncrna_id",
        "state_id",
        "train_effect",
        "test_effect",
        "train_detection_rate",
        "train_n_patients",
        "train_state_mean",
        "train_state_sd",
        "selected_for_model",
        "membership_threshold",
    ]
    identity = frame[[column for column in identity_columns if column in frame]].reset_index(drop=True)
    for model in GRAPH_MODELS:
        identity[f"{model}_available"] = (
            graph_available[model].detach().cpu().numpy().astype("int8")
            if model in graph_available
            else np.zeros(len(frame), dtype="int8")
        )

    return PreparedStateFold(
        identity=identity,
        patient_features=torch.as_tensor(patient_values, device=device),
        graph_features=graph_features,
        graph_available=graph_available,
        train_label=torch.as_tensor(frame["train_membership_label"].fillna(0).to_numpy(np.float32), device=device),
        train_label_available=torch.as_tensor(
            (
                frame["train_membership_label"].notna()
                & frame.get("selected_for_model", pd.Series(1, index=frame.index)).astype(bool)
            ).to_numpy(),
            device=device,
        ),
        validation_label=frame["validation_membership_label"].to_numpy(np.float32),
        test_label=frame["test_membership_label"].to_numpy(np.float32),
        train_direction=torch.as_tensor(frame["train_direction_label"].fillna(0).to_numpy(np.float32), device=device),
        train_direction_available=torch.as_tensor(
            frame["train_direction_label"].notna().to_numpy(), device=device
        ),
        validation_direction=frame["validation_direction_label"].to_numpy(np.float32),
        test_direction=frame["test_direction_label"].to_numpy(np.float32),
        patient_feature_columns=patient_columns,
        patient_mean=patient_mean,
        patient_scale=patient_scale,
        graph_dims=graph_dims,
    )


def _weighted_bce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    positive = labels.sum().clamp_min(1)
    negative = (labels.numel() - labels.sum()).clamp_min(1)
    pos_weight = (negative / positive).clamp(1, 20)
    return nn.functional.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)


def _mean_available_probability(
    output: dict[str, torch.Tensor],
    prepared: PreparedStateFold,
) -> np.ndarray:
    values = [torch.sigmoid(output["patient_logit"]).detach().cpu().numpy()]
    availability = [np.ones(len(prepared.identity), bool)]
    for model in GRAPH_MODELS:
        key = f"{model}_logit"
        if key in output:
            values.append(torch.sigmoid(output[key]).detach().cpu().numpy())
            availability.append(prepared.graph_available[model].detach().cpu().numpy().astype(bool))
    matrix = np.column_stack(values)
    mask = np.column_stack(availability)
    total = np.where(mask, matrix, 0).sum(axis=1)
    count = mask.sum(axis=1)
    return total / np.maximum(count, 1)


def train_state_task(
    cancer: str,
    patient_fold_id: str,
    seed: int,
    epochs: int = 60,
    patience: int = 8,
) -> tuple[pd.DataFrame, dict[str, float]]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = STATE_DATA / f"cancer_id={cancer}" / "part-0.parquet"
    frame = pd.read_parquet(data_path)
    frame = frame[frame["patient_fold_id"].astype(str).eq(str(patient_fold_id))].reset_index(drop=True)
    if frame.empty:
        raise RuntimeError(f"{cancer} {patient_fold_id}: empty lncRNA-state fold")
    prepared = prepare_state_fold(frame, cancer, seed, device)
    model = StateExpertNetwork(
        patient_dim=prepared.patient_features.shape[1],
        graph_dims=prepared.graph_dims,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=2e-3)
    train_mask = prepared.train_label_available
    if train_mask.sum() < 50 or prepared.train_label[train_mask].unique().numel() < 2:
        raise RuntimeError(f"{cancer} {patient_fold_id}: insufficient train labels/classes")
    best_state = None
    best_score = -math.inf
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        output = model(prepared.patient_features, prepared.graph_features)
        terms = [_weighted_bce(output["patient_logit"][train_mask], prepared.train_label[train_mask])]
        for graph_name in GRAPH_MODELS:
            key = f"{graph_name}_logit"
            if key not in output:
                continue
            mask = train_mask & prepared.graph_available[graph_name]
            if mask.sum() >= 20 and prepared.train_label[mask].unique().numel() > 1:
                terms.append(0.5 * _weighted_bce(output[key][mask], prepared.train_label[mask]))
        direction_mask = (
            train_mask
            & prepared.train_label.gt(0.5)
            & prepared.train_direction_available
        )
        if direction_mask.any():
            terms.append(0.2 * nn.functional.binary_cross_entropy_with_logits(
                output["direction_logit"][direction_mask], prepared.train_direction[direction_mask]
            ))
        loss = torch.stack(terms).sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_output = model(prepared.patient_features, prepared.graph_features)
            validation_probability = _mean_available_probability(validation_output, prepared)
        metric = _metrics(prepared.validation_label, validation_probability)
        score = metric["auprc"] if np.isfinite(metric["auprc"]) else -float(loss.item())
        history.append({"epoch": epoch, "loss": float(loss.item()), **{f"validation_{k}": v for k, v in metric.items()}})
        if score > best_score + 1e-5:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("lncRNA-state model produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        output = model(prepared.patient_features, prepared.graph_features)

    validation_mask = np.isfinite(prepared.validation_label)
    temperatures: dict[str, float] = {}
    calibration: dict[str, str] = {}
    head_logits: dict[str, np.ndarray] = {
        "patient": output["patient_logit"].detach().cpu().numpy(),
    }
    for graph_name in GRAPH_MODELS:
        key = f"{graph_name}_logit"
        if key in output:
            head_logits[graph_name] = output[key].detach().cpu().numpy()
    for name, logits in head_logits.items():
        valid = validation_mask.copy()
        if name in prepared.graph_available:
            valid &= prepared.graph_available[name].detach().cpu().numpy().astype(bool)
        temperature, method = fit_temperature(logits[valid], prepared.validation_label[valid])
        temperatures[name] = temperature
        calibration[name] = method

    prediction = prepared.identity.copy()
    for name, logits in head_logits.items():
        probability = 1.0 / (1.0 + np.exp(-np.clip(logits / max(temperatures[name], 1e-6), -40, 40)))
        if name in prepared.graph_available:
            available = prepared.graph_available[name].detach().cpu().numpy().astype(bool)
            probability = np.where(available, probability, np.nan)
        prediction[f"state_{name}_probability"] = probability.astype("float32")
    prediction["state_direction_probability"] = torch.sigmoid(output["direction_logit"]).detach().cpu().numpy().astype("float32")
    prediction["test_membership_label"] = prepared.test_label
    prediction["test_direction_label"] = prepared.test_direction
    prediction["seed"] = seed

    model_dir = STATE_MODEL_ROOT / cancer / str(patient_fold_id) / f"seed_{seed}"
    result_dir = STATE_RESULT_ROOT / cancer / str(patient_fold_id) / f"seed_{seed}"
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": best_state,
        "patient_feature_columns": prepared.patient_feature_columns,
        "patient_mean": prepared.patient_mean,
        "patient_scale": prepared.patient_scale,
        "graph_dims": prepared.graph_dims,
        "temperatures": temperatures,
        "calibration": calibration,
        "graph_encoder_retrained": False,
        "task": "cancer_x_lncrna_x_tumor_state",
        "state_semantics": "tumor phenotype/state; not ordinary pathway membership",
        "direction_supervision_scope": "positive_membership_pairs_only",
        "state_unlabeled_direction_used_in_loss": 0,
    }
    torch.save(checkpoint, model_dir / "best.pt")
    prediction.to_parquet(result_dir / "prediction.parquet", index=False, compression="zstd")
    pd.DataFrame(history).to_csv(result_dir / "history.tsv", sep="\t", index=False)
    test_mean = prediction[[column for column in prediction if column.startswith("state_") and column.endswith("_probability") and column != "state_direction_probability"]].mean(axis=1, skipna=True).to_numpy()
    metrics = _metrics(prepared.test_label, test_mean)
    pd.DataFrame([metrics]).to_csv(result_dir / "metrics.tsv", sep="\t", index=False)
    return prediction, metrics
