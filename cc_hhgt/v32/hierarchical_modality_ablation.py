"""Post-lock modality ablation for the selected hierarchical V3.2 model."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .multimodal_fusion import FUSION_FOLD_COLUMN, TARGET_KEYS, artifact_sha256
from .routing_validation_winner import WINNER_FORMAT
from .validation_prediction_inference import (
    infer_final_logits_for_batches,
    inverse_node_maps,
)


STATUS = "PASS_HIERARCHICAL_MODALITY_ABLATION_AFTER_WINNER_LOCK"


class HierarchicalModalityAblationError(RuntimeError):
    pass


def _json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HierarchicalModalityAblationError(f"Unreadable {label}: {path}") from exc
    if not isinstance(value, dict):
        raise HierarchicalModalityAblationError(f"{label} must be a JSON object")
    return value


def _masked_batches(
    batches: Sequence[Mapping[str, Any]], modality_index: int
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for raw in batches:
        batch = dict(raw)
        candidate = dict(batch["candidate_batch"])
        available = candidate["modality_available"].clone()
        probability = candidate["modality_probability"].clone()
        if available.ndim != 2 or modality_index >= available.shape[1]:
            raise HierarchicalModalityAblationError(
                "Hierarchical modality tensor shape drift"
            )
        available[:, modality_index] = False
        probability[:, modality_index] = float("nan")
        candidate["modality_available"] = available
        candidate["modality_probability"] = probability
        batch["candidate_batch"] = candidate
        output.append(batch)
    return output


def _metric(target: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    clipped = np.clip(probability, 1e-7, 1.0 - 1e-7)
    logloss = float(
        -np.mean(
            target * np.log(clipped)
            + (1.0 - target) * np.log(1.0 - clipped)
        )
    )
    return logloss, float(np.mean((probability - target) ** 2))


def materialize_hierarchical_modality_ablation_after_winner_lock(
    *,
    repo_root: str | Path,
    config_path: str | Path,
    prepared_root: str | Path,
    checkpoint_pattern: str,
    outer_predictions_path: str | Path,
    outer_success_path: str | Path,
    routing_winner_declaration_path: str | Path,
    routing_winner_declaration_sha256: str,
    output_root: str | Path,
) -> dict[str, Any]:
    """Ablate each modality after selection and report per-cancer deltas."""

    import torch
    import yaml

    from ..gnn import build_model
    from .model import build_v32_hierarchical_evidence_hhgt

    del repo_root
    config_path = Path(config_path).resolve()
    prepared = Path(prepared_root).resolve()
    outer_path = Path(outer_predictions_path).resolve()
    outer_success_path = Path(outer_success_path).resolve()
    winner_path = Path(routing_winner_declaration_path).resolve()
    expected_winner_sha = str(routing_winner_declaration_sha256).lower()
    for label, path in (
        ("config", config_path),
        ("outer predictions", outer_path),
        ("outer success", outer_success_path),
        ("routing winner", winner_path),
    ):
        if not path.is_file() or path.stat().st_size <= 0:
            raise HierarchicalModalityAblationError(f"Missing {label}: {path}")
    if len(expected_winner_sha) != 64 or artifact_sha256(winner_path) != expected_winner_sha:
        raise HierarchicalModalityAblationError("Routing winner SHA256 drift")
    winner = _json(winner_path, "routing winner")
    required_winner = {
        "format": WINNER_FORMAT,
        "status": "PASS_ROUTING_VALIDATION_WINNER_LOCK",
        "winner_id": "hierarchical_end_to_end",
        "selection_split": "validation_only",
        "heldout_test_used_for_selection": False,
        "outer_test_predictions_available_during_selection": False,
        "test_metrics_used_for_selection": False,
        "winner_locked_before_outer_test_inference": True,
    }
    if any(winner.get(key) != value for key, value in required_winner.items()):
        raise HierarchicalModalityAblationError(
            "Routing winner does not authorize hierarchical ablation"
        )
    outer_success = _json(outer_success_path, "hierarchical outer success")
    if (
        outer_success.get("status") != "PASS_HIERARCHICAL_OUTER_AFTER_WINNER_LOCK"
        or outer_success.get("outer_inference_started_after_winner_lock") is not True
        or outer_success.get("test_metrics_used_for_selection") is not False
        or outer_success.get("routing_winner_declaration", {}).get("sha256")
        != expected_winner_sha
        or outer_success.get("oof_predictions", {}).get("sha256")
        != artifact_sha256(outer_path)
    ):
        raise HierarchicalModalityAblationError(
            "Hierarchical outer result is not bound to the routing winner"
        )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise HierarchicalModalityAblationError("Hierarchical config is invalid")
    primary = config.get("primary_model", {})
    modalities = tuple(
        primary.get("evidence_integration", {}).get("modalities", ())
    )
    if modalities != ("mutation", "cnv", "atac"):
        raise HierarchicalModalityAblationError(
            "Hierarchical modality order/contract drift"
        )
    full = (
        pd.read_parquet(outer_path)
        .sort_values(list(TARGET_KEYS), kind="stable")
        .reset_index(drop=True)
    )
    required_outer = set(TARGET_KEYS) | {
        FUSION_FOLD_COLUMN,
        "fusion_target",
        "hierarchical_probability",
    } | {f"{modality}_available" for modality in modalities}
    if missing := sorted(required_outer - set(full.columns)):
        raise HierarchicalModalityAblationError(
            f"Hierarchical outer predictions lack columns: {missing}"
        )
    if full[list(TARGET_KEYS)].duplicated().any():
        raise HierarchicalModalityAblationError(
            "Hierarchical outer predictions duplicate exact keys"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    parts: dict[str, list[pd.DataFrame]] = {modality: [] for modality in modalities}
    checkpoint_records: list[dict[str, Any]] = []
    for fold in range(5):
        payload_path = prepared / f"PATIENT_FOLD_{fold}.pt"
        checkpoint_path = Path(checkpoint_pattern.format(fold=fold)).resolve()
        payload = torch.load(payload_path, map_location="cpu", weights_only=False)
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        if (
            not isinstance(payload, Mapping)
            or payload.get("outer_inference_authorized_after_winner_lock") is not True
            or set(payload).intersection({"train_batches", "validation_batches"})
            or checkpoint.get("architecture_id")
            != "HHGT_HIERARCHICAL_EVIDENCE_GATE_CANDIDATE"
        ):
            raise HierarchicalModalityAblationError(
                f"Fold {fold} outer payload/checkpoint contract drift"
            )
        batches = payload.get("test_batches")
        if not isinstance(batches, Sequence) or not batches:
            raise HierarchicalModalityAblationError(
                f"Fold {fold} has no winner-locked test batches"
            )
        bundle = payload["bundle"]
        encoder = build_model(
            "cc_hhgt",
            bundle,
            int(payload["feature_dim"]),
            dict(payload["legacy_model_config"]),
        )
        model = build_v32_hierarchical_evidence_hhgt(
            encoder,
            hidden_channels=int(primary.get("hidden_channels", 96)),
            context_features=int(payload.get("conservation_context_features", 4)),
            modality_names=modalities,
            dropout=float(primary.get("dropout", 0.20)),
        )
        model.load_state_dict(checkpoint["model_state"])
        model.to(device).eval()
        inverse = inverse_node_maps(bundle)
        for modality_index, modality in enumerate(modalities):
            masked = _masked_batches(batches, modality_index)
            logits = infer_final_logits_for_batches(
                model=model,
                bundle=bundle,
                fixed_graph=payload.get("graph"),
                batches=masked,
                device=device,
                torch=torch,
            )
            for raw, final_logit in zip(masked, logits):
                candidate = raw["candidate_batch"]
                l_index = candidate["l"].detach().cpu().numpy().astype(int)
                p_index = candidate["p"].detach().cpu().numpy().astype(int)
                c_index = candidate["c"].detach().cpu().numpy().astype(int)
                pair_fold = (
                    candidate["outer_pair_fold"]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(int)
                )
                if not np.all(pair_fold == fold):
                    raise HierarchicalModalityAblationError(
                        f"Fold {fold} ablation batch contains another pair fold"
                    )
                parts[modality].append(
                    pd.DataFrame(
                        {
                            "cancer_id": [
                                inverse["cancer"][value].upper() for value in c_index
                            ],
                            "lncrna_id": [
                                inverse["lncRNA"][value] for value in l_index
                            ],
                            "pathway_id": [
                                inverse["pathway"][value] for value in p_index
                            ],
                            FUSION_FOLD_COLUMN: pair_fold,
                            "ablated_probability": (
                                1.0 / (1.0 + np.exp(-final_logit))
                            ).astype(np.float32),
                        }
                    )
                )
        checkpoint_records.append(
            {
                "fold": fold,
                "path": str(checkpoint_path),
                "sha256": artifact_sha256(checkpoint_path),
            }
        )

    target = pd.to_numeric(full.fusion_target, errors="raise").to_numpy(float)
    full_probability = pd.to_numeric(
        full.hierarchical_probability, errors="raise"
    ).to_numpy(float)
    groups = [("ALL_CANCERS", np.ones(len(full), dtype=bool))] + [
        (str(cancer), full.cancer_id.eq(cancer).to_numpy())
        for cancer in sorted(full.cancer_id.unique())
    ]
    metrics: list[dict[str, Any]] = []
    for modality in modalities:
        ablated = (
            pd.concat(parts[modality], ignore_index=True)
            .sort_values(list(TARGET_KEYS), kind="stable")
            .reset_index(drop=True)
        )
        if (
            len(ablated) != len(full)
            or not ablated[list(TARGET_KEYS)].equals(full[list(TARGET_KEYS)])
            or not np.array_equal(
                pd.to_numeric(ablated[FUSION_FOLD_COLUMN]).to_numpy(int),
                pd.to_numeric(full[FUSION_FOLD_COLUMN]).to_numpy(int),
            )
        ):
            raise HierarchicalModalityAblationError(
                f"{modality} ablation universe/fold drift"
            )
        ablated_probability = pd.to_numeric(
            ablated.ablated_probability, errors="raise"
        ).to_numpy(float)
        available = full[f"{modality}_available"].astype(bool).to_numpy()
        for cancer, mask in groups:
            full_loss, full_brier = _metric(target[mask], full_probability[mask])
            ablated_loss, ablated_brier = _metric(
                target[mask], ablated_probability[mask]
            )
            metrics.append(
                {
                    "cancer_id": cancer,
                    "modality": modality,
                    "rows": int(mask.sum()),
                    "available_rows": int((mask & available).sum()),
                    "full_logloss": full_loss,
                    "ablated_logloss": ablated_loss,
                    "conditional_ablation_logloss_delta": ablated_loss - full_loss,
                    "full_brier": full_brier,
                    "ablated_brier": ablated_brier,
                    "conditional_ablation_brier_delta": ablated_brier - full_brier,
                    "outer_metric_computed_after_winner_lock": True,
                    "used_for_model_or_architecture_selection": False,
                }
            )
    metric_frame = pd.DataFrame(metrics)
    output = Path(output_root).resolve()
    staging = output.with_name(f"{output.name}.partial")
    if output.exists() or staging.exists():
        raise HierarchicalModalityAblationError(
            f"Hierarchical ablation output/staging reuse is forbidden: {output}"
        )
    staging.mkdir(parents=True)
    try:
        metrics_path = staging / "hierarchical_modality_ablation_metrics.parquet"
        metric_frame.to_parquet(metrics_path, index=False)
        success = {
            "status": STATUS,
            "winner_id": "hierarchical_end_to_end",
            "routing_winner_declaration": {
                "path": str(winner_path),
                "sha256": expected_winner_sha,
            },
            "outer_predictions": {
                "path": str(outer_path),
                "sha256": artifact_sha256(outer_path),
            },
            "outer_success_sha256": artifact_sha256(outer_success_path),
            "modality_ablation_metrics": {
                "path": str(output / metrics_path.name),
                "sha256": artifact_sha256(metrics_path),
            },
            "checkpoints": checkpoint_records,
            "device": device,
            "outer_metric_computed_after_winner_lock": True,
            "used_for_model_or_architecture_selection": False,
        }
        (staging / "SUCCESS.json").write_text(
            json.dumps(success, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(staging, output)
        return success
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = [
    "HierarchicalModalityAblationError",
    "STATUS",
    "materialize_hierarchical_modality_ablation_after_winner_lock",
]
