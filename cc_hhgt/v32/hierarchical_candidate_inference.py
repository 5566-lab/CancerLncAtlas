"""Materialize pair-blocked OOF predictions from hierarchical HHGT folds."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .multimodal_fusion import FUSION_FOLD_COLUMN, TARGET_KEYS, artifact_sha256
from .routed_fair_comparison import (
    exact_candidate_and_fold_hashes,
    modality_source_composite_sha256,
)


class HierarchicalInferenceError(RuntimeError):
    pass


def _load_config(path: Path) -> Mapping[str, Any]:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise HierarchicalInferenceError("Hierarchical config is not a mapping")
    return value


def _move(value: Any, device: str):
    if hasattr(value, "to"):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _move(item, device) for key, item in value.items()}
    return value


def _inverse(bundle: Any) -> dict[str, dict[int, str]]:
    return {
        kind: {int(index): str(identifier) for identifier, index in bundle.node_maps[kind].items()}
        for kind in ("cancer", "lncRNA", "pathway")
    }


def materialize_hierarchical_oof(
    *,
    repo_root: str | Path,
    config_path: str | Path,
    prepared_root: str | Path,
    checkpoint_pattern: str,
    preparation_manifest_path: str | Path,
    output_root: str | Path,
    training_budget_id: str = "v32-routed-equal-budget-v1",
    primary_outer_predictions_path: str | Path | None = None,
) -> dict[str, Any]:
    import torch

    from ..gnn import build_model
    from .model import build_v32_hierarchical_evidence_hhgt
    from .validation_prediction_inference import (
        infer_final_logits_for_batches,
        inverse_node_maps,
    )

    root = Path(repo_root).resolve()
    config_file = Path(config_path).resolve()
    prepared = Path(prepared_root).resolve()
    prep_manifest_path = Path(preparation_manifest_path).resolve()
    output = Path(output_root).resolve()
    if output.exists():
        raise HierarchicalInferenceError(f"Hierarchical inference refuses output reuse: {output}")
    output.mkdir(parents=True)
    config = _load_config(config_file)
    primary = config["primary_model"]
    integration = primary["evidence_integration"]
    modalities = tuple(integration["modalities"])
    prep_manifest = json.loads(prep_manifest_path.read_text(encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parts: list[pd.DataFrame] = []
    checkpoint_records: list[dict[str, Any]] = []
    for fold in range(5):
        payload_path = prepared / f"PATIENT_FOLD_{fold}.pt"
        checkpoint_path = Path(checkpoint_pattern.format(fold=fold)).resolve()
        payload = torch.load(payload_path, map_location="cpu", weights_only=False)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("architecture_id") != "HHGT_HIERARCHICAL_EVIDENCE_GATE_CANDIDATE":
            raise HierarchicalInferenceError(f"Fold {fold} checkpoint is not hierarchical HHGT")
        bundle = payload["bundle"]
        encoder = build_model(
            "cc_hhgt", bundle, int(payload["feature_dim"]), dict(payload["legacy_model_config"])
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
        batches = payload["test_batches"]
        final_logits = infer_final_logits_for_batches(
            model=model,
            bundle=bundle,
            fixed_graph=payload.get("graph"),
            batches=batches,
            device=device,
            torch=torch,
        )
        for raw, final_logit in zip(batches, final_logits):
                batch = raw
                candidate = batch["candidate_batch"]
                l_index = candidate["l"].detach().cpu().numpy().astype(int)
                p_index = candidate["p"].detach().cpu().numpy().astype(int)
                c_index = candidate["c"].detach().cpu().numpy().astype(int)
                pair_fold = candidate["outer_pair_fold"].detach().cpu().numpy().astype(int)
                if not np.all(pair_fold == fold):
                    raise HierarchicalInferenceError(f"Fold {fold} test batch contains another pair fold")
                modality_probability = candidate["modality_probability"].detach().cpu().numpy().astype(float)
                modality_available = candidate["modality_available"].detach().cpu().numpy().astype(bool)
                if modality_probability.shape != modality_available.shape or modality_probability.shape[1] != len(modalities):
                    raise HierarchicalInferenceError("Hierarchical modality tensors violate the declared contract")
                modality_columns = {
                    f"{modality}_probability": modality_probability[:, index]
                    for index, modality in enumerate(modalities)
                }
                modality_columns.update(
                    {
                        f"{modality}_available": modality_available[:, index]
                        for index, modality in enumerate(modalities)
                    }
                )
                parts.append(
                    pd.DataFrame(
                        {
                            "cancer_id": [inverse["cancer"][value].upper() for value in c_index],
                            "lncrna_id": [inverse["lncRNA"][value] for value in l_index],
                            "pathway_id": [inverse["pathway"][value] for value in p_index],
                            FUSION_FOLD_COLUMN: pair_fold,
                            "fusion_target": batch["proxy_label"].detach().cpu().numpy().astype(float),
                            "primary_probability": torch.sigmoid(batch["base_logit"]).detach().cpu().numpy(),
                            "hierarchical_probability": (
                                1.0 / (1.0 + np.exp(-final_logit))
                            ).astype(np.float32),
                            **modality_columns,
                        }
                    )
                )
        checkpoint_records.append(
            {"fold": fold, "path": str(checkpoint_path), "sha256": artifact_sha256(checkpoint_path)}
        )
    frame = pd.concat(parts, ignore_index=True).sort_values(list(TARGET_KEYS), kind="stable").reset_index(drop=True)
    if frame[list(TARGET_KEYS)].duplicated().any() or set(frame[FUSION_FOLD_COLUMN]) != set(range(5)):
        raise HierarchicalInferenceError("Hierarchical OOF candidate/fold coverage is invalid")
    primary_outer_path = (
        Path(primary_outer_predictions_path).resolve()
        if primary_outer_predictions_path is not None
        else None
    )
    if primary_outer_path is not None:
        if not primary_outer_path.is_file() or primary_outer_path.stat().st_size <= 0:
            raise HierarchicalInferenceError(
                f"Missing G012 primary outer predictions: {primary_outer_path}"
            )
        primary_outer = (
            pd.read_parquet(primary_outer_path)
            .sort_values(list(TARGET_KEYS), kind="stable")
            .reset_index(drop=True)
        )
        required_primary = set(TARGET_KEYS) | {
            FUSION_FOLD_COLUMN,
            "fusion_target",
            "primary_probability",
        }
        if missing := sorted(required_primary - set(primary_outer.columns)):
            raise HierarchicalInferenceError(
                f"G012 primary outer frame lacks columns: {missing}"
            )
        if (
            len(primary_outer) != len(frame)
            or not primary_outer[list(TARGET_KEYS)].equals(frame[list(TARGET_KEYS)])
        ):
            raise HierarchicalInferenceError(
                "G012 and hierarchical outer candidate universes differ"
            )
        for column in (FUSION_FOLD_COLUMN, "fusion_target"):
            left = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
            right = pd.to_numeric(primary_outer[column], errors="coerce").to_numpy(float)
            if not np.allclose(left, right, rtol=0.0, atol=1e-7, equal_nan=False):
                raise HierarchicalInferenceError(
                    f"G012/hierarchical outer parity drift: {column}"
                )
        frame.rename(columns={"primary_probability": "base_l1_probability"}, inplace=True)
        frame["primary_probability"] = pd.to_numeric(
            primary_outer.primary_probability, errors="raise"
        ).astype(np.float32)
    oof_path = output / "hierarchical_oof_predictions.parquet"
    frame.to_parquet(oof_path, index=False)
    metric_rows: list[dict[str, Any]] = []
    target = pd.to_numeric(frame.fusion_target, errors="raise").to_numpy(float)
    primary_probability = pd.to_numeric(
        frame.primary_probability, errors="raise"
    ).to_numpy(float)
    hierarchical_probability = pd.to_numeric(
        frame.hierarchical_probability, errors="raise"
    ).to_numpy(float)
    groups = [("ALL_CANCERS", np.ones(len(frame), dtype=bool))] + [
        (str(cancer), frame.cancer_id.eq(cancer).to_numpy())
        for cancer in sorted(frame.cancer_id.unique())
    ]
    for cancer, mask in groups:
        clipped_primary = np.clip(primary_probability[mask], 1e-7, 1.0 - 1e-7)
        clipped_hierarchical = np.clip(
            hierarchical_probability[mask], 1e-7, 1.0 - 1e-7
        )
        local_target = target[mask]
        primary_logloss = float(
            -np.mean(
                local_target * np.log(clipped_primary)
                + (1.0 - local_target) * np.log(1.0 - clipped_primary)
            )
        )
        hierarchical_logloss = float(
            -np.mean(
                local_target * np.log(clipped_hierarchical)
                + (1.0 - local_target) * np.log(1.0 - clipped_hierarchical)
            )
        )
        primary_brier = float(np.mean((primary_probability[mask] - local_target) ** 2))
        hierarchical_brier = float(
            np.mean((hierarchical_probability[mask] - local_target) ** 2)
        )
        metric_rows.append(
            {
                "cancer_id": cancer,
                "rows": int(mask.sum()),
                "primary_logloss": primary_logloss,
                "hierarchical_logloss": hierarchical_logloss,
                "delta_logloss_vs_primary": primary_logloss - hierarchical_logloss,
                "primary_brier": primary_brier,
                "hierarchical_brier": hierarchical_brier,
                "delta_brier_vs_primary": primary_brier - hierarchical_brier,
                "outer_metric_computed_after_winner_lock": primary_outer_path is not None,
            }
        )
    metrics = pd.DataFrame(metric_rows)
    metrics_path = output / "hierarchical_outer_metrics.parquet"
    metrics.to_parquet(metrics_path, index=False)
    candidate_sha, fold_sha = exact_candidate_and_fold_hashes(frame)
    source_hashes = {"genomic_predictions": prep_manifest["genomic_predictions"]["sha256"]}
    if prep_manifest.get("atac_predictions"):
        source_hashes["atac_predictions"] = prep_manifest["atac_predictions"]["sha256"]
    comparison_contract = {
        "candidate_universe_sha256": candidate_sha,
        "patient_modality_oof_sha256": modality_source_composite_sha256(source_hashes),
        "outer_pair_fold_sha256": fold_sha,
        "seed": int(config["crossfit"]["seed"]),
        "training_budget_id": str(training_budget_id),
    }
    contract_path = output / "COMPARISON_CONTRACT.json"
    contract_path.write_text(json.dumps(comparison_contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    success = {
        "status": "SUCCESS",
        "analysis_version": "CancerLncAtlas_V3.2_HIERARCHICAL_ROUTED_CANDIDATE",
        "architecture_id": "HHGT_HIERARCHICAL_EVIDENCE_GATE_CANDIDATE",
        "preparation_manifest": {
            "path": str(Path(preparation_manifest_path).resolve()),
            "sha256": artifact_sha256(preparation_manifest_path),
        },
        "patient_fold_authority": prep_manifest.get("patient_fold_authority"),
        "device": device,
        "rows": len(frame),
        "oof_predictions": {"path": str(oof_path), "sha256": artifact_sha256(oof_path)},
        "outer_metrics": {
            "path": str(metrics_path),
            "sha256": artifact_sha256(metrics_path),
        },
        "comparison_contract": {"path": str(contract_path), "sha256": artifact_sha256(contract_path)},
        "checkpoints": checkpoint_records,
        "candidate_only": True,
        "formal_v32_primary_unchanged": True,
    }
    if primary_outer_path is not None:
        success["g012_primary_outer_predictions"] = {
            "path": str(primary_outer_path),
            "sha256": artifact_sha256(primary_outer_path),
        }
    (output / "SUCCESS.json").write_text(json.dumps(success, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return success


def materialize_hierarchical_validation_only(
    *,
    repo_root: str | Path,
    config_path: str | Path,
    prepared_root: str | Path,
    checkpoint_pattern: str,
    preparation_manifest_path: str | Path,
    primary_validation_predictions_path: str | Path,
    output_root: str | Path,
    training_budget_id: str,
) -> dict[str, Any]:
    """Infer inner-validation folds without deserializing any test batch."""

    import torch

    from ..gnn import build_model
    from .model import build_v32_hierarchical_evidence_hhgt
    from .validation_prediction_inference import (
        infer_final_logits_for_batches,
        inverse_node_maps,
    )

    root = Path(repo_root).resolve()
    config_file = Path(config_path).resolve()
    prepared = Path(prepared_root).resolve()
    prep_manifest_path = Path(preparation_manifest_path).resolve()
    primary_validation_path = Path(primary_validation_predictions_path).resolve()
    if not primary_validation_path.is_file() or primary_validation_path.stat().st_size <= 0:
        raise HierarchicalInferenceError(
            f"Missing locked G012 validation predictions: {primary_validation_path}"
        )
    output = Path(output_root).resolve()
    staging = output.with_name(f"{output.name}.partial")
    if output.exists() or staging.exists():
        raise HierarchicalInferenceError(
            f"Hierarchical validation inference refuses output/staging reuse: {output}"
        )
    staging.mkdir(parents=True)
    try:
        config = _load_config(config_file)
        primary = config["primary_model"]
        modalities = tuple(primary["evidence_integration"]["modalities"])
        prep_manifest = json.loads(prep_manifest_path.read_text(encoding="utf-8"))
        if prep_manifest.get("status") != "SUCCESS_TRAIN_VALIDATION_ONLY_TEST_REMAINS_SEALED":
            raise HierarchicalInferenceError(
                "Hierarchical preparation does not prove a sealed outer test"
            )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        parts: list[pd.DataFrame] = []
        checkpoint_records: list[dict[str, Any]] = []
        for fold in range(5):
            payload_path = prepared / f"PATIENT_FOLD_{fold}.pt"
            checkpoint_path = Path(checkpoint_pattern.format(fold=fold)).resolve()
            payload = torch.load(payload_path, map_location="cpu", weights_only=False)
            if "test_batches" in payload:
                raise HierarchicalInferenceError(
                    f"Fold {fold} validation payload contains forbidden test batches"
                )
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            if checkpoint.get("architecture_id") != "HHGT_HIERARCHICAL_EVIDENCE_GATE_CANDIDATE":
                raise HierarchicalInferenceError(
                    f"Fold {fold} checkpoint is not hierarchical HHGT"
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
            validation_fold = (fold + 1) % 5
            batches = payload["validation_batches"]
            final_logits = infer_final_logits_for_batches(
                model=model,
                bundle=bundle,
                fixed_graph=payload.get("graph"),
                batches=batches,
                device=device,
                torch=torch,
            )
            for raw, final_logit in zip(batches, final_logits):
                    batch = raw
                    candidate = batch["candidate_batch"]
                    pair_fold = (
                        candidate["outer_pair_fold"]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(int)
                    )
                    if not np.all(pair_fold == validation_fold):
                        raise HierarchicalInferenceError(
                            f"Fold {fold} validation batch contains another pair fold"
                        )
                    l_index = candidate["l"].detach().cpu().numpy().astype(int)
                    p_index = candidate["p"].detach().cpu().numpy().astype(int)
                    c_index = candidate["c"].detach().cpu().numpy().astype(int)
                    modality_probability = (
                        candidate["modality_probability"]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(float)
                    )
                    modality_available = (
                        candidate["modality_available"]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(bool)
                    )
                    if (
                        modality_probability.shape != modality_available.shape
                        or modality_probability.shape[1] != len(modalities)
                    ):
                        raise HierarchicalInferenceError(
                            "Hierarchical validation modality tensors violate the contract"
                        )
                    modality_columns = {
                        f"{modality}_probability": modality_probability[:, index]
                        for index, modality in enumerate(modalities)
                    }
                    modality_columns.update(
                        {
                            f"{modality}_available": modality_available[:, index]
                            for index, modality in enumerate(modalities)
                        }
                    )
                    parts.append(
                        pd.DataFrame(
                            {
                                "cancer_id": [
                                    inverse["cancer"][value].upper()
                                    for value in c_index
                                ],
                                "lncrna_id": [
                                    inverse["lncRNA"][value] for value in l_index
                                ],
                                "pathway_id": [
                                    inverse["pathway"][value] for value in p_index
                                ],
                                FUSION_FOLD_COLUMN: pair_fold,
                                "selection_outer_pair_fold": int(fold),
                                "selection_validation_pair_fold": pair_fold,
                                "outer_test_queried": False,
                                "fusion_target": batch["proxy_label"]
                                .detach()
                                .cpu()
                                .numpy()
                                .astype(float),
                                "primary_probability": torch.sigmoid(
                                    batch["base_logit"]
                                )
                                .detach()
                                .cpu()
                                .numpy(),
                                "hierarchical_probability": (
                                    1.0 / (1.0 + np.exp(-final_logit))
                                ).astype(np.float32),
                                **modality_columns,
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
        frame = (
            pd.concat(parts, ignore_index=True)
            .sort_values(list(TARGET_KEYS), kind="stable")
            .reset_index(drop=True)
        )
        if (
            frame[list(TARGET_KEYS)].duplicated().any()
            or set(frame[FUSION_FOLD_COLUMN]) != set(range(5))
            or not frame.outer_test_queried.eq(False).all()
        ):
            raise HierarchicalInferenceError(
                "Hierarchical validation candidate/fold coverage is invalid"
            )
        # The comparator baseline must be the locked G012 model inferred on
        # the same inner-validation fold, not the L1 offset stored in the
        # hierarchical training payload and not a previously opened outer OOF.
        primary_validation = (
            pd.read_parquet(primary_validation_path)
            .sort_values(list(TARGET_KEYS), kind="stable")
            .reset_index(drop=True)
        )
        required_primary = set(TARGET_KEYS) | {
            FUSION_FOLD_COLUMN,
            "selection_outer_pair_fold",
            "selection_validation_pair_fold",
            "outer_test_queried",
            "fusion_target",
            "primary_probability",
        }
        if missing := sorted(required_primary - set(primary_validation.columns)):
            raise HierarchicalInferenceError(
                f"G012 validation baseline lacks columns: {missing}"
            )
        if (
            len(primary_validation) != len(frame)
            or not primary_validation[list(TARGET_KEYS)].equals(frame[list(TARGET_KEYS)])
            or not primary_validation.outer_test_queried.eq(False).all()
        ):
            raise HierarchicalInferenceError(
                "G012 and hierarchical validation candidate universes differ"
            )
        for column in (
            FUSION_FOLD_COLUMN,
            "selection_outer_pair_fold",
            "selection_validation_pair_fold",
            "fusion_target",
        ):
            left = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
            right = pd.to_numeric(
                primary_validation[column], errors="coerce"
            ).to_numpy(float)
            if not np.allclose(left, right, rtol=0.0, atol=1e-7, equal_nan=False):
                raise HierarchicalInferenceError(
                    f"G012/hierarchical validation parity drift: {column}"
                )
        frame.rename(columns={"primary_probability": "base_l1_probability"}, inplace=True)
        frame["primary_probability"] = pd.to_numeric(
            primary_validation.primary_probability, errors="raise"
        ).astype(np.float32)
        prediction_name = "hierarchical_validation_predictions.PRIVATE.parquet"
        prediction_path = staging / prediction_name
        frame.to_parquet(prediction_path, index=False)
        candidate_sha, fold_sha = exact_candidate_and_fold_hashes(frame)
        contract = {
            "format": "CANCERLNCATLAS_V32_ROUTING_VALIDATION_COMPARISON_CONTRACT_V1",
            "candidate_universe_sha256": candidate_sha,
            "validation_pair_fold_sha256": fold_sha,
            "seed": int(config["crossfit"]["seed"]),
            "training_budget_id": str(training_budget_id),
            "selection_split": "INNER_VALIDATION_ONLY",
            "outer_test_queries": 0,
        }
        contract_name = "VALIDATION_COMPARISON_CONTRACT.json"
        contract_path = staging / contract_name
        contract_path.write_text(
            json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        success = {
            "status": "PASS_VALIDATION_ONLY_HIERARCHICAL",
            "analysis_version": "CancerLncAtlas_V3.2_HIERARCHICAL_ROUTED_CANDIDATE",
            "architecture_id": "HHGT_HIERARCHICAL_EVIDENCE_GATE_CANDIDATE",
            "preparation_manifest": {
                "path": str(prep_manifest_path),
                "sha256": artifact_sha256(prep_manifest_path),
            },
            "patient_fold_authority": prep_manifest.get("patient_fold_authority"),
            "device": device,
            "rows": len(frame),
            "validation_predictions": {
                "path": str(output / prediction_name),
                "sha256": artifact_sha256(prediction_path),
            },
            "validation_comparison_contract": {
                "path": str(output / contract_name),
                "sha256": artifact_sha256(contract_path),
            },
            "checkpoints": checkpoint_records,
            "g012_validation_predictions": {
                "path": str(primary_validation_path),
                "sha256": artifact_sha256(primary_validation_path),
            },
            "outer_test_predictions_written": False,
            "outer_test_metrics_computed": False,
            "architecture_winner_selected": False,
            "candidate_only": True,
            "formal_v32_primary_unchanged": True,
        }
        (staging / "SUCCESS.json").write_text(
            json.dumps(success, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        staging.replace(output)
        return success
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def materialize_hierarchical_outer_after_winner_lock(
    *,
    repo_root: str | Path,
    config_path: str | Path,
    prepared_root: str | Path,
    checkpoint_pattern: str,
    winner_locked_test_manifest_path: str | Path,
    routing_winner_declaration_path: str | Path,
    routing_winner_declaration_sha256: str,
    primary_outer_predictions_path: str | Path,
    output_root: str | Path,
    training_budget_id: str,
) -> dict[str, Any]:
    """Validate the routing lock before delegating to outer HHGT inference."""

    from .routing_validation_winner import WINNER_FORMAT

    manifest_path = Path(winner_locked_test_manifest_path).resolve()
    winner_path = Path(routing_winner_declaration_path).resolve()
    primary_outer_path = Path(primary_outer_predictions_path).resolve()
    expected_winner_sha = str(routing_winner_declaration_sha256).lower()
    if (
        not manifest_path.is_file()
        or not winner_path.is_file()
        or not primary_outer_path.is_file()
        or len(expected_winner_sha) != 64
        or artifact_sha256(winner_path) != expected_winner_sha
    ):
        raise HierarchicalInferenceError(
            "Hierarchical outer inference lacks a hash-bound routing winner"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        winner = json.loads(winner_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HierarchicalInferenceError(
            "Hierarchical winner/outer manifest is unreadable"
        ) from exc
    winner_required = {
        "format": WINNER_FORMAT,
        "status": "PASS_ROUTING_VALIDATION_WINNER_LOCK",
        "winner_id": "hierarchical_end_to_end",
        "selection_split": "validation_only",
        "heldout_test_used_for_selection": False,
        "outer_test_predictions_available_during_selection": False,
        "test_metrics_used_for_selection": False,
        "winner_locked_before_outer_test_inference": True,
    }
    if any(winner.get(key) != value for key, value in winner_required.items()):
        raise HierarchicalInferenceError(
            "Routing winner does not authorize hierarchical outer inference"
        )
    manifest_winner = manifest.get("routing_winner_declaration", {})
    if (
        manifest.get("status")
        != "PASS_HIERARCHICAL_OUTER_PREPARED_AFTER_WINNER_LOCK"
        or manifest.get("outer_payloads_opened_after_routing_winner_lock") is not True
        or manifest.get("test_metrics_computed") is not False
        or manifest_winner.get("sha256") != expected_winner_sha
    ):
        raise HierarchicalInferenceError(
            "Winner-locked hierarchical test manifest contract drift"
        )
    result = materialize_hierarchical_oof(
        repo_root=repo_root,
        config_path=config_path,
        prepared_root=prepared_root,
        checkpoint_pattern=checkpoint_pattern,
        preparation_manifest_path=manifest_path,
        output_root=output_root,
        training_budget_id=training_budget_id,
        primary_outer_predictions_path=primary_outer_path,
    )
    output = Path(output_root).resolve()
    success_path = output / "SUCCESS.json"
    success = json.loads(success_path.read_text(encoding="utf-8"))
    success.update(
        {
            "status": "PASS_HIERARCHICAL_OUTER_AFTER_WINNER_LOCK",
            "routing_winner_declaration": {
                "path": str(winner_path),
                "sha256": expected_winner_sha,
            },
            "outer_inference_started_after_winner_lock": True,
            "test_metrics_used_for_selection": False,
            "g012_primary_outer_predictions": {
                "path": str(primary_outer_path),
                "sha256": artifact_sha256(primary_outer_path),
            },
        }
    )
    temporary = success_path.with_name(".SUCCESS.json.tmp")
    temporary.write_text(
        json.dumps(success, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(success_path)
    return success


__all__ = [
    "HierarchicalInferenceError",
    "materialize_hierarchical_oof",
    "materialize_hierarchical_outer_after_winner_lock",
    "materialize_hierarchical_validation_only",
]
