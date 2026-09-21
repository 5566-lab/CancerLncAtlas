"""Patient-cross-fitted cancer adapter with native and strict-joint branches.

The patient-native branch is deliberately independent of strict graph scores
and embeddings.  It can therefore score cancer-specific/cold-start candidates
that never entered the strict OOF universe.  The joint branch uses the frozen
strict embeddings and strict probabilities when available.  A masked internal
router combines both branches; unavailable strict inputs receive exactly zero
weight.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
RESULT_BASE = Path(os.getenv("CC_HHGT_RESULT_ROOT", str(ROOT / "results")))
ADAPTER_DATA = Path(os.getenv("CC_HHGT_ADAPTER_DATA_ROOT", str(RESULT_BASE / "adapter_data")))
MODEL_ROOT = Path(os.getenv("CC_HHGT_ADAPTER_MODEL_ROOT", str(ROOT / "models" / "cancer_adapter")))
RESULT_ROOT = Path(os.getenv("CC_HHGT_ADAPTER_RESULT_ROOT", str(RESULT_BASE / "cancer_adapter")))
STRICT_EMBEDDING_ROOT = Path(os.getenv("CC_HHGT_STRICT_EMBEDDING_ROOT", str(ROOT / "models" / "strict_global")))
SEEDS = [20260726, 20261726, 20262726]

STRICT_FEATURES = [
    "cross_cancer_probability",
    "cross_cancer_probability_sd",
    "direction_probability",
    "strict_available",
    "strict_candidate_available",
]
PATIENT_BASE_FEATURES = [
    "train_effect",
    "train_fdr",
    "train_detection_rate",
    "train_n_patients",
    "lncrna_available",
    "pathway_available",
    "pair_available",
    "patient_available",
    "patient_native_selection_score",
]
PATIENT_OPTIONAL_PREFIXES = (
    "state_mean__",
    "state_available__",
    "sc_",
    "ucell_",
    "celltype_",
    "drug_",
    "mutation_",
    "tumor_state_",
    "crosswalk_",
    "communication_",
)
EXCLUDED_FEATURE_PREFIXES = (
    "validation_",
    "test_",
)
EXCLUDED_FEATURE_SUFFIXES = (
    "_label",
    "_pvalue",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def expected_calibration_error(
    labels: np.ndarray, probability: np.ndarray, bins: int = 10
) -> float:
    edges = np.linspace(0, 1, bins + 1)
    value = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        mask = (probability >= lower) & (
            probability < upper if upper < 1 else probability <= upper
        )
        if mask.any():
            value += mask.mean() * abs(labels[mask].mean() - probability[mask].mean())
    return float(value)


def classification_metrics(labels: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(labels) & np.isfinite(probability)
    labels = labels[valid]
    probability = probability[valid]
    if not len(labels):
        return {"auprc": math.nan, "auroc": math.nan, "brier": math.nan, "ece": math.nan}
    return {
        "auprc": float(average_precision_score(labels, probability))
        if len(np.unique(labels)) > 1
        else math.nan,
        "auroc": float(roc_auc_score(labels, probability))
        if len(np.unique(labels)) > 1
        else math.nan,
        "brier": float(brier_score_loss(labels, probability)),
        "ece": expected_calibration_error(labels, probability),
    }


class CancerAdapter(nn.Module):
    """Two-branch patient expert with a fail-closed strict-availability router."""

    def __init__(
        self,
        embedding_dim: int,
        patient_dim: int,
        strict_dim: int,
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
        self.native_trunk = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(0.10),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.lnc_projection = nn.Linear(embedding_dim, hidden)
        self.pathway_projection = nn.Linear(embedding_dim, hidden)
        self.cancer_projection = nn.Linear(embedding_dim, hidden)
        self.strict_projection = nn.Linear(strict_dim, hidden)
        self.joint_trunk = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.native_membership_head = nn.Linear(hidden, 1)
        self.joint_membership_head = nn.Linear(hidden, 1)
        self.branch_gate = nn.Sequential(
            nn.Linear(hidden * 2 + 1, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 2),
        )
        self.direction_head = nn.Linear(hidden, 1)
        self.specificity_head = nn.Linear(hidden, 1)

    def forward(
        self,
        lnc: torch.Tensor,
        pathway: torch.Tensor,
        patient_features: torch.Tensor,
        strict_features: torch.Tensor,
        cancer_embedding: torch.Tensor,
        strict_available: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        patient_hidden = self.patient_encoder(patient_features)
        native_hidden = self.native_trunk(patient_hidden)
        joint_hidden = self.joint_trunk(
            patient_hidden
            + self.lnc_projection(lnc)
            + self.pathway_projection(pathway)
            + self.cancer_projection(cancer_embedding)
            + self.strict_projection(strict_features)
        )
        native_logit = self.native_membership_head(native_hidden).squeeze(-1)
        joint_logit = self.joint_membership_head(joint_hidden).squeeze(-1)
        availability_value = strict_available.float().unsqueeze(-1)
        gate_logit = self.branch_gate(
            torch.cat([native_hidden, joint_hidden, availability_value], dim=-1)
        )
        # Native branch is always available for rows with patient features.
        # Joint branch is exactly masked whenever strict score/embedding is absent.
        mask = torch.stack(
            [torch.ones_like(strict_available, dtype=torch.bool), strict_available.bool()],
            dim=-1,
        )
        gate_logit = gate_logit.masked_fill(~mask, -1e9)
        gate_weight = torch.softmax(gate_logit, dim=-1)
        fused_logit = (
            gate_weight[:, 0] * native_logit + gate_weight[:, 1] * joint_logit
        )
        fused_hidden = (
            gate_weight[:, 0:1] * native_hidden
            + gate_weight[:, 1:2] * joint_hidden
        )
        return {
            "membership_logit": fused_logit,
            "native_membership_logit": native_logit,
            "joint_membership_logit": joint_logit,
            "direction_logit": self.direction_head(fused_hidden).squeeze(-1),
            "specificity_logit": self.specificity_head(fused_hidden).squeeze(-1),
            "native_weight": gate_weight[:, 0],
            "joint_weight": gate_weight[:, 1],
        }


@dataclass
class PreparedFold:
    identity: pd.DataFrame
    lnc: torch.Tensor
    pathway: torch.Tensor
    patient_features: torch.Tensor
    strict_features: torch.Tensor
    cancer_embedding: torch.Tensor
    strict_available: torch.Tensor
    train_membership: torch.Tensor
    train_membership_available: torch.Tensor
    validation_membership: np.ndarray
    test_membership: np.ndarray
    train_direction: torch.Tensor
    validation_direction: np.ndarray
    test_direction: np.ndarray
    train_specificity: torch.Tensor
    patient_feature_columns: list[str]
    strict_feature_columns: list[str]
    patient_scaler_mean: np.ndarray
    patient_scaler_scale: np.ndarray
    strict_scaler_mean: np.ndarray
    strict_scaler_scale: np.ndarray


def _embedding_frame(cancer: str, seed: int, node_type: str) -> pd.DataFrame:
    path = (
        STRICT_EMBEDDING_ROOT
        / "cc_hhgt_strict"
        / f"LOCO_{cancer}"
        / f"seed_{seed}"
        / "embeddings"
        / f"{node_type}.parquet"
    )
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def _numeric_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.reindex(columns=columns).copy()
    for column in result:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.replace([np.inf, -np.inf], np.nan)


def _scale_frame(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = frame.mean(axis=0).fillna(0).to_numpy(dtype=np.float32)
    scale = frame.std(axis=0).replace(0, 1).fillna(1).to_numpy(dtype=np.float32)
    values = frame.fillna(pd.Series(mean, index=frame.columns)).to_numpy(dtype=np.float32)
    return (values - mean) / scale, mean, scale


def _patient_feature_columns(frame: pd.DataFrame) -> list[str]:
    selected: list[str] = []
    for column in frame.columns:
        if column in PATIENT_BASE_FEATURES or column.startswith(PATIENT_OPTIONAL_PREFIXES):
            if column.startswith(EXCLUDED_FEATURE_PREFIXES):
                continue
            if column.endswith(EXCLUDED_FEATURE_SUFFIXES):
                continue
            if pd.api.types.is_numeric_dtype(frame[column]) or frame[column].dtype == object:
                selected.append(column)
    # Prevent any strict score from silently entering the native branch.
    selected = [column for column in selected if column not in STRICT_FEATURES]
    selected = sorted(set(selected))
    if not selected:
        raise RuntimeError("patient-native branch has no numeric features")
    return selected


def prepare_fold(
    frame: pd.DataFrame, cancer: str, seed: int, device: torch.device
) -> PreparedFold:
    lnc_embedding = _embedding_frame(cancer, seed, "lncRNA").set_index("canonical_id")
    pathway_embedding = _embedding_frame(cancer, seed, "pathway_family").set_index(
        "canonical_id"
    )
    cancer_embedding = _embedding_frame(cancer, seed, "cancer").set_index(
        "canonical_id"
    )
    embedding_columns = [
        column for column in lnc_embedding if column.startswith("embedding_")
    ]

    lnc_values_frame = lnc_embedding.reindex(frame["lncrna_id"])[embedding_columns]
    pathway_values_frame = pathway_embedding.reindex(frame["pathway_family_id"])[
        embedding_columns
    ]
    lnc_embedding_available = lnc_values_frame.notna().all(axis=1).to_numpy()
    pathway_embedding_available = pathway_values_frame.notna().all(axis=1).to_numpy()
    lnc_values = lnc_values_frame.fillna(0).to_numpy(dtype=np.float32)
    pathway_values = pathway_values_frame.fillna(0).to_numpy(dtype=np.float32)

    cancer_key = f"CANCER:{cancer}"
    if cancer_key not in cancer_embedding.index:
        matches = [value for value in cancer_embedding.index if str(value).endswith(cancer)]
        if not matches:
            raise RuntimeError(f"{cancer}: cancer embedding unavailable")
        cancer_key = matches[0]
    cancer_values = np.repeat(
        cancer_embedding.loc[cancer_key, embedding_columns]
        .to_numpy(dtype=np.float32)[None, :],
        len(frame),
        axis=0,
    )

    patient_columns = _patient_feature_columns(frame)
    strict_columns = [column for column in STRICT_FEATURES if column in frame]
    if "strict_available" not in strict_columns:
        strict_columns.append("strict_available")
        frame = frame.copy()
        frame["strict_available"] = frame.get("cross_cancer_probability", np.nan).notna().astype("int8")

    patient_frame = _numeric_frame(frame, patient_columns)
    strict_frame = _numeric_frame(frame, strict_columns)
    patient_values, patient_mean, patient_scale = _scale_frame(patient_frame)
    strict_values, strict_mean, strict_scale = _scale_frame(strict_frame)

    score_available = frame.get("strict_available", pd.Series(0, index=frame.index)).fillna(0).astype(bool).to_numpy()
    strict_available = (
        score_available & lnc_embedding_available & pathway_embedding_available
    )

    identity_columns = [
        "cancer_id",
        "patient_fold_id",
        "lncrna_id",
        "pathway_family_id",
        "candidate_source",
        "candidate_from_strict",
        "candidate_from_patient",
        "candidate_from_evidence",
        "train_detection_rate",
        "train_n_patients",
        "pair_available",
        "cold_start",
    ]
    identity_columns = [column for column in identity_columns if column in frame]
    identity = frame[identity_columns].reset_index(drop=True)
    identity["strict_available"] = strict_available.astype("int8")

    return PreparedFold(
        identity=identity,
        lnc=torch.as_tensor(lnc_values, device=device),
        pathway=torch.as_tensor(pathway_values, device=device),
        patient_features=torch.as_tensor(patient_values, device=device),
        strict_features=torch.as_tensor(strict_values, device=device),
        cancer_embedding=torch.as_tensor(cancer_values, device=device),
        strict_available=torch.as_tensor(strict_available, device=device),
        train_membership=torch.as_tensor(
            frame["train_membership_label"].fillna(0).to_numpy(np.float32),
            device=device,
        ),
        train_membership_available=torch.as_tensor(
            frame["train_membership_label"].notna().to_numpy(),
            device=device,
        ),
        validation_membership=frame["validation_membership_label"].to_numpy(np.float32),
        test_membership=frame["test_membership_label"].to_numpy(np.float32),
        train_direction=torch.as_tensor(
            frame["train_direction_label"].fillna(0).to_numpy(np.float32),
            device=device,
        ),
        validation_direction=frame["validation_direction_label"].to_numpy(np.float32),
        test_direction=frame["test_direction_label"].to_numpy(np.float32),
        train_specificity=torch.as_tensor(
            frame["train_effect"].abs().fillna(0).to_numpy(np.float32),
            device=device,
        ),
        patient_feature_columns=patient_columns,
        strict_feature_columns=strict_columns,
        patient_scaler_mean=patient_mean,
        patient_scaler_scale=patient_scale,
        strict_scaler_mean=strict_mean,
        strict_scaler_scale=strict_scale,
    )


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> tuple[float, str]:
    valid = np.isfinite(logits) & np.isfinite(labels)
    logits = logits[valid]
    labels = labels[valid]
    if len(logits) < 10 or len(np.unique(labels)) < 2:
        return 1.0, "unavailable_single_class_or_small_validation"

    def objective(log_temperature: float) -> float:
        temperature = math.exp(log_temperature)
        scaled = logits / temperature
        return float(
            np.mean(
                np.maximum(scaled, 0)
                - scaled * labels
                + np.log1p(np.exp(-np.abs(scaled)))
            )
        )

    result = minimize_scalar(objective, bounds=(-3.0, 3.0), method="bounded")
    return float(math.exp(result.x)), "validation_patients_only"


def _sigmoid_numpy(logit: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.clip(logit / max(temperature, 1e-6), -40, 40)
    return 1.0 / (1.0 + np.exp(-scaled))


def train_task(
    cancer: str,
    patient_fold_id: str,
    seed: int,
    epochs: int = 80,
    patience: int = 10,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, float]]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source = ADAPTER_DATA / f"cancer_id={cancer}" / "part-0.parquet"
    frame = pd.read_parquet(source, filters=[("patient_fold_id", "=", patient_fold_id)])
    prepared = prepare_fold(frame, cancer, seed, device)
    model = CancerAdapter(
        prepared.lnc.shape[1],
        prepared.patient_features.shape[1],
        prepared.strict_features.shape[1],
    ).to(device)
    available_train = prepared.train_membership_available.bool()
    if not available_train.any():
        raise RuntimeError("patient adapter has no train-available candidate labels")
    positive = float(prepared.train_membership[available_train].sum().item())
    negative = float(available_train.sum().item() - positive)
    pos_weight = torch.tensor(
        min(max(negative / max(positive, 1.0), 1.0), 20.0), device=device
    )
    membership_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    direction_loss = nn.BCEWithLogitsLoss()
    specificity_loss = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=2e-3)
    best_state = None
    best_score = -math.inf
    stale = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        result = model(
            prepared.lnc,
            prepared.pathway,
            prepared.patient_features,
            prepared.strict_features,
            prepared.cancer_embedding,
            prepared.strict_available,
        )
        positive_mask = (prepared.train_membership > 0.5) & available_train
        dloss = (
            direction_loss(
                result["direction_logit"][positive_mask],
                prepared.train_direction[positive_mask],
            )
            if positive_mask.any()
            else result["direction_logit"].sum() * 0
        )
        joint_mask = prepared.strict_available.bool() & available_train
        joint_aux = (
            membership_loss(
                result["joint_membership_logit"][joint_mask],
                prepared.train_membership[joint_mask],
            )
            if joint_mask.any()
            else result["joint_membership_logit"].sum() * 0
        )
        loss = (
            membership_loss(
                result["membership_logit"][available_train],
                prepared.train_membership[available_train],
            )
            + 0.50
            * membership_loss(
                result["native_membership_logit"][available_train],
                prepared.train_membership[available_train],
            )
            + 0.25 * joint_aux
            + 0.25 * dloss
            + 0.20
            * specificity_loss(
                torch.sigmoid(result["specificity_logit"])[available_train],
                prepared.train_specificity[available_train],
            )
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_result = model(
                prepared.lnc,
                prepared.pathway,
                prepared.patient_features,
                prepared.strict_features,
                prepared.cancer_embedding,
                prepared.strict_available,
            )
            validation_probability = torch.sigmoid(
                validation_result["membership_logit"]
            ).cpu().numpy()
        metric = classification_metrics(
            prepared.validation_membership, validation_probability
        )
        score = metric["auprc"]
        if not np.isfinite(score):
            score = -float(loss.item())
        history.append(
            {
                "epoch": epoch,
                "loss": float(loss.item()),
                "validation_auprc": float(metric["auprc"]),
                "validation_auroc": float(metric["auroc"]),
                "mean_native_weight": float(result["native_weight"].mean().item()),
                "mean_joint_weight": float(result["joint_weight"].mean().item()),
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
        raise RuntimeError("adapter training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        result = model(
            prepared.lnc,
            prepared.pathway,
            prepared.patient_features,
            prepared.strict_features,
            prepared.cancer_embedding,
            prepared.strict_available,
        )

    fused_logit = result["membership_logit"].cpu().numpy()
    native_logit = result["native_membership_logit"].cpu().numpy()
    joint_logit = result["joint_membership_logit"].cpu().numpy()
    fused_temperature, calibration_status = fit_temperature(
        fused_logit, prepared.validation_membership
    )
    native_temperature, native_calibration_status = fit_temperature(
        native_logit, prepared.validation_membership
    )
    strict_mask_np = prepared.strict_available.cpu().numpy().astype(bool)
    joint_validation_labels = prepared.validation_membership.copy()
    joint_validation_labels[~strict_mask_np] = np.nan
    joint_temperature, joint_calibration_status = fit_temperature(
        joint_logit, joint_validation_labels
    )

    probability = _sigmoid_numpy(fused_logit, fused_temperature)
    native_probability = _sigmoid_numpy(native_logit, native_temperature)
    joint_probability = _sigmoid_numpy(joint_logit, joint_temperature)
    patient_available_np = prepared.identity.get(
        "pair_available", pd.Series(1, index=prepared.identity.index)
    ).fillna(0).to_numpy(dtype=bool)
    native_probability[~patient_available_np] = np.nan
    probability[~patient_available_np] = np.nan
    joint_probability[~(strict_mask_np & patient_available_np)] = np.nan
    direction_probability = torch.sigmoid(result["direction_logit"]).cpu().numpy()
    direction_probability[~patient_available_np] = np.nan
    specificity = torch.sigmoid(result["specificity_logit"]).cpu().numpy()
    validation_metrics = classification_metrics(
        prepared.validation_membership, probability
    )
    test_metrics = classification_metrics(prepared.test_membership, probability)
    native_test_metrics = classification_metrics(
        prepared.test_membership, native_probability
    )

    output = prepared.identity.copy()
    output["seed"] = seed
    output["raw_logit"] = fused_logit.astype("float32")
    output["native_raw_logit"] = native_logit.astype("float32")
    output["joint_raw_logit"] = joint_logit.astype("float32")
    output["cancer_specific_probability"] = probability.astype("float32")
    output["cancer_native_probability"] = native_probability.astype("float32")
    output["cancer_joint_probability"] = joint_probability.astype("float32")
    output["patient_gate_native_weight"] = result["native_weight"].cpu().numpy().astype("float32")
    output["patient_gate_joint_weight"] = result["joint_weight"].cpu().numpy().astype("float32")
    output["direction_probability"] = direction_probability.astype("float32")
    output["specificity_score"] = specificity.astype("float32")
    output["test_membership_label"] = prepared.test_membership
    output["test_direction_label"] = prepared.test_direction
    output["temperature"] = fused_temperature
    output["native_temperature"] = native_temperature
    output["joint_temperature"] = joint_temperature

    task_root = MODEL_ROOT / cancer / patient_fold_id / f"seed_{seed}"
    task_root.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": best_state,
            "embedding_dim": int(prepared.lnc.shape[1]),
            "patient_dim": int(prepared.patient_features.shape[1]),
            "strict_dim": int(prepared.strict_features.shape[1]),
            "patient_feature_columns": prepared.patient_feature_columns,
            "strict_feature_columns": prepared.strict_feature_columns,
            "patient_scaler_mean": prepared.patient_scaler_mean,
            "patient_scaler_scale": prepared.patient_scaler_scale,
            "strict_scaler_mean": prepared.strict_scaler_mean,
            "strict_scaler_scale": prepared.strict_scaler_scale,
            "seed": seed,
            "cancer_id": cancer,
            "patient_fold_id": patient_fold_id,
            "global_encoder_frozen": True,
            "patient_native_independent_of_strict": True,
            "strict_missing_router": "masked_softmax_zero_weight",
        },
        task_root / "best.pt",
    )
    pd.DataFrame(history).to_csv(
        task_root / "training_history.tsv", sep="\t", index=False
    )
    calibration = {
        "fused_temperature": fused_temperature,
        "fused_status": calibration_status,
        "native_temperature": native_temperature,
        "native_status": native_calibration_status,
        "joint_temperature": joint_temperature,
        "joint_status": joint_calibration_status,
        "source": "validation_patients_only",
    }
    (task_root / "calibration.json").write_text(
        json.dumps(calibration, indent=2), encoding="utf-8"
    )
    result_root = RESULT_ROOT / cancer / patient_fold_id / f"seed_{seed}"
    result_root.mkdir(parents=True, exist_ok=True)
    output.to_parquet(
        result_root / "prediction.parquet", index=False, compression="zstd"
    )
    metrics = {
        "cancer_id": cancer,
        "patient_fold_id": patient_fold_id,
        "seed": seed,
        "n_pairs": len(frame),
        "n_strict_available": int(strict_mask_np.sum()),
        "n_strict_unavailable": int((~strict_mask_np).sum()),
        "temperature": fused_temperature,
        "calibration_status": calibration_status,
        **{f"validation_{key}": value for key, value in validation_metrics.items()},
        **{f"test_{key}": value for key, value in test_metrics.items()},
        **{f"native_test_{key}": value for key, value in native_test_metrics.items()},
    }
    pd.DataFrame([metrics]).to_csv(
        result_root / "metrics.tsv", sep="\t", index=False
    )
    group_weights = {
        "patient_native_encoder": float(
            sum(parameter.norm().item() for parameter in model.patient_encoder.parameters())
        ),
        "patient_native_head": float(
            sum(parameter.norm().item() for parameter in model.native_membership_head.parameters())
        ),
        "strict_joint_embeddings": float(
            model.lnc_projection.weight.norm().item()
            + model.pathway_projection.weight.norm().item()
            + model.cancer_projection.weight.norm().item()
        ),
        "strict_joint_features": float(model.strict_projection.weight.norm().item()),
        "patient_branch_router": float(
            sum(parameter.norm().item() for parameter in model.branch_gate.parameters())
        ),
    }
    total = sum(group_weights.values())
    attribution = {key: value / total for key, value in group_weights.items()}
    return output, metrics, attribution
