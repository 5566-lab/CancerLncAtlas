"""Read-only validation inference shared by the formal V3.2 model arms.

This module deliberately accepts only the training payload's validation
batches.  It never accepts a sealed-test manifest or a test-batch payload.
Runtime graph chunks are aggregated with the same registered weights used by
formal training, so a validation score cannot silently fall back to a single
graph chunk.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .multimodal_fusion import FUSION_FOLD_COLUMN, TARGET_KEYS, artifact_sha256
from .routed_fair_comparison import exact_candidate_and_fold_hashes


G012_VALIDATION_STATUS = "PASS_G012_INNER_VALIDATION_PREDICTIONS"


class ValidationPredictionInferenceError(RuntimeError):
    pass


def _move(value: Any, device: str) -> Any:
    if hasattr(value, "to"):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _move(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    return value


def inverse_node_maps(bundle: Any) -> dict[str, dict[int, str]]:
    maps = getattr(bundle, "node_maps", None)
    if not isinstance(maps, Mapping):
        raise ValidationPredictionInferenceError("Prepared graph lacks node_maps")
    result: dict[str, dict[int, str]] = {}
    for kind in ("cancer", "lncRNA", "pathway"):
        values = maps.get(kind)
        if not isinstance(values, Mapping):
            raise ValidationPredictionInferenceError(
                f"Prepared graph lacks the {kind} node map"
            )
        inverse = {int(index): str(identifier) for identifier, index in values.items()}
        if len(inverse) != len(values):
            raise ValidationPredictionInferenceError(
                f"Prepared graph {kind} node map is not one-to-one"
            )
        result[kind] = inverse
    return result


def _kahan_weighted_arrays(
    arrays: Sequence[np.ndarray], weights: Sequence[float]
) -> np.ndarray:
    if not arrays or len(arrays) != len(weights):
        raise ValidationPredictionInferenceError(
            "Runtime chunk aggregation received empty or misaligned values"
        )
    total = np.zeros_like(np.asarray(arrays[0], dtype=np.float64))
    compensation = np.zeros_like(total)
    for raw, weight in zip(arrays, weights):
        value = np.asarray(raw, dtype=np.float64)
        if value.shape != total.shape or not np.isfinite(value).all():
            raise ValidationPredictionInferenceError(
                "Runtime chunk logits changed shape or became non-finite"
            )
        corrected = value * float(weight) - compensation
        updated = total + corrected
        compensation = (updated - total) - corrected
        total = updated
    return total


def infer_final_logits_for_batches(
    *,
    model: Any,
    bundle: Any,
    fixed_graph: Any,
    batches: Sequence[Mapping[str, Any]],
    device: str,
    torch: Any,
) -> list[np.ndarray]:
    """Infer every batch over the complete registered runtime graph estimand."""

    from ..gnn import move_graph
    from .training import _runtime_chunk_contract, _runtime_graph_for_chunk

    if not batches:
        raise ValidationPredictionInferenceError("Validation payload has no batches")
    chunks, weight_map = _runtime_chunk_contract(bundle)
    has_schedule = getattr(bundle, "runtime_schedule", None) is not None
    if has_schedule and fixed_graph is not None:
        raise ValidationPredictionInferenceError(
            "A fixed graph cannot override the registered runtime schedule"
        )
    outputs: dict[int, list[np.ndarray]] = {}
    with torch.no_grad():
        for chunk in chunks:
            graph = (
                _runtime_graph_for_chunk(bundle, int(chunk), device=device)
                if has_schedule
                else (
                    _move(fixed_graph, device)
                    if fixed_graph is not None
                    else move_graph(bundle, device, "cc_hhgt")
                )
            )
            encoded = model.encoder.encode(graph)
            chunk_rows: list[np.ndarray] = []
            for raw in batches:
                batch = _move(raw, device)
                output = model(
                    graph,
                    batch["candidate_batch"],
                    batch["base_logit"],
                    batch["conservation_context"],
                    batch.get("graph_available"),
                    admitted=True,
                    encoded=encoded,
                )
                final = output.get("final_logit")
                if final is None:
                    raise ValidationPredictionInferenceError(
                        "Model inference lacks final_logit"
                    )
                chunk_rows.append(
                    final.detach().cpu().numpy().astype(np.float64).reshape(-1)
                )
            outputs[int(chunk)] = chunk_rows
    if set(outputs) != set(chunks):
        raise ValidationPredictionInferenceError(
            "Validation did not cover every registered runtime chunk"
        )
    counts = {len(value) for value in outputs.values()}
    if counts != {len(batches)}:
        raise ValidationPredictionInferenceError(
            "Runtime chunks produced unequal validation coverage"
        )
    weights = [float(weight_map[int(chunk)]) for chunk in chunks]
    return [
        _kahan_weighted_arrays(
            [outputs[int(chunk)][index] for chunk in chunks], weights
        )
        for index in range(len(batches))
    ]


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationPredictionInferenceError(f"Unreadable {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValidationPredictionInferenceError(f"{label} must be a JSON object")
    return value


def _declared_path(record: Mapping[str, Any], *, relative_to: Path, label: str) -> Path:
    raw = record.get("path")
    if not raw:
        raise ValidationPredictionInferenceError(f"{label} lacks a path")
    path = Path(str(raw))
    if not path.is_absolute():
        path = relative_to / path
    path = path.resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValidationPredictionInferenceError(f"Missing {label}: {path}")
    expected = str(record.get("sha256", "")).lower()
    if len(expected) != 64 or artifact_sha256(path) != expected:
        raise ValidationPredictionInferenceError(f"{label} SHA256 drift")
    return path


def materialize_g012_validation_predictions(
    *,
    repo_root: str | Path,
    prepared_root: str | Path,
    winner_declaration_path: str | Path,
    winner_declaration_sha256: str,
    output_root: str | Path,
) -> dict[str, Any]:
    """Re-infer the locked G012 arm on each fold's inner validation rows."""

    import torch
    import yaml

    from ..gnn import build_model
    from .model import build_v32_cc_hhgt_residual
    from .prepared import PREPARED_FORMAT, validate_prepared_batch
    from .sealed_test_inference import WINNER_DECLARATION_FORMAT

    del repo_root  # retained in the public API for CLI/path symmetry
    prepared = Path(prepared_root).resolve()
    winner_path = Path(winner_declaration_path).resolve()
    expected_winner_sha = str(winner_declaration_sha256).lower()
    if (
        not winner_path.is_file()
        or len(expected_winner_sha) != 64
        or artifact_sha256(winner_path) != expected_winner_sha
    ):
        raise ValidationPredictionInferenceError(
            "G012 validation inference lacks the pinned winner declaration"
        )
    winner = _read_json(winner_path, "G012 winner declaration")
    required = {
        "format": WINNER_DECLARATION_FORMAT,
        "status": "PASS_VALIDATION_ONLY_WINNER_LOCK",
        "selection_scope": "VALIDATION_ONLY",
        "heldout_test_metrics_used_for_selection": False,
        "test_inputs_opened_before_winner_lock": False,
        "winner_or_checkpoint_changed_by_test": False,
        "declaration_locked": True,
    }
    if any(winner.get(key) != value for key, value in required.items()):
        raise ValidationPredictionInferenceError("G012 winner-lock contract drift")
    variant = str(winner.get("graph_variant", ""))
    if variant not in {"G0", "G1", "G2"} or prepared.name != variant:
        raise ValidationPredictionInferenceError("G012 prepared-root variant drift")
    config_path = _declared_path(
        winner.get("config", {}),
        relative_to=winner_path.parent,
        label="G012 winner config",
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValidationPredictionInferenceError("G012 winner config is not a mapping")
    primary = config.get("primary_model", {})
    if primary.get("evidence_integration", {}).get("mode") != "external_router":
        raise ValidationPredictionInferenceError("G012 winner is not the formal core arm")
    prepared_records = {
        int(record.get("patient_fold", -1)): record
        for record in winner.get("prepared_folds", [])
        if isinstance(record, Mapping)
    }
    checkpoint_records = {
        int(record.get("patient_fold", -1)): record
        for record in winner.get("checkpoints", [])
        if isinstance(record, Mapping)
    }
    if set(prepared_records) != set(range(5)) or set(checkpoint_records) != set(range(5)):
        raise ValidationPredictionInferenceError(
            "G012 winner declaration does not bind five folds"
        )

    output = Path(output_root).resolve()
    staging = output.with_name(f"{output.name}.partial")
    if output.exists() or staging.exists():
        raise ValidationPredictionInferenceError(
            f"G012 validation output/staging reuse is forbidden: {output}"
        )
    staging.mkdir(parents=True)
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        parts: list[pd.DataFrame] = []
        fold_lineage: list[dict[str, Any]] = []
        pair_seed = int(winner.get("pair_fold_seed", -1))
        if pair_seed < 0:
            raise ValidationPredictionInferenceError("Invalid pair-fold seed")
        from .multimodal_fusion import pair_blocked_fold

        for fold in range(5):
            declared_prepared = _declared_path(
                prepared_records[fold],
                relative_to=winner_path.parent,
                label=f"G012 prepared fold {fold}",
            )
            expected_prepared = (prepared / f"PATIENT_FOLD_{fold}.pt").resolve()
            if declared_prepared != expected_prepared:
                raise ValidationPredictionInferenceError(
                    f"G012 prepared fold {fold} path drift"
                )
            checkpoint_path = _declared_path(
                checkpoint_records[fold],
                relative_to=winner_path.parent,
                label=f"G012 checkpoint fold {fold}",
            )
            payload = torch.load(
                declared_prepared, map_location="cpu", weights_only=False
            )
            if not isinstance(payload, Mapping):
                raise ValidationPredictionInferenceError(
                    f"G012 prepared fold {fold} is not a mapping"
                )
            if "test_batches" in payload:
                raise ValidationPredictionInferenceError(
                    f"G012 validation payload fold {fold} contains forbidden test batches"
                )
            if (
                payload.get("prepared_format") != PREPARED_FORMAT
                or int(payload.get("patient_fold", -1)) != fold
                or payload.get("formal_graph_variant") != variant
            ):
                raise ValidationPredictionInferenceError(
                    f"G012 prepared fold {fold} contract/variant drift"
                )
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            if (
                not isinstance(checkpoint, Mapping)
                or checkpoint.get("architecture_id")
                != "HHGT_FORMAL_CORE_EXTERNAL_ROUTER"
            ):
                raise ValidationPredictionInferenceError(
                    f"G012 checkpoint fold {fold} architecture drift"
                )
            bundle = payload["bundle"]
            encoder = build_model(
                "cc_hhgt",
                bundle,
                int(payload["feature_dim"]),
                dict(payload["legacy_model_config"]),
            )
            model = build_v32_cc_hhgt_residual(
                encoder,
                hidden_channels=int(primary.get("hidden_channels", 96)),
                context_features=int(payload.get("conservation_context_features", 4)),
                dropout=float(primary.get("dropout", 0.20)),
            )
            model.load_state_dict(checkpoint["model_state"])
            model.to(device).eval()
            batches = payload.get("validation_batches")
            if not isinstance(batches, Sequence) or not batches:
                raise ValidationPredictionInferenceError(
                    f"G012 prepared fold {fold} lacks validation batches"
                )
            for batch in batches:
                validate_prepared_batch(batch)
            logits = infer_final_logits_for_batches(
                model=model,
                bundle=bundle,
                fixed_graph=payload.get("graph"),
                batches=batches,
                device=device,
                torch=torch,
            )
            inverse = inverse_node_maps(bundle)
            validation_fold = (fold + 1) % 5
            for raw, final_logit in zip(batches, logits):
                candidate = raw["candidate_batch"]
                l_index = candidate["l"].detach().cpu().numpy().astype(int)
                p_index = candidate["p"].detach().cpu().numpy().astype(int)
                c_index = candidate["c"].detach().cpu().numpy().astype(int)
                if len(final_logit) != len(l_index):
                    raise ValidationPredictionInferenceError(
                        f"G012 fold {fold} validation output length drift"
                    )
                cancer = [inverse["cancer"][value].upper() for value in c_index]
                lnc = [inverse["lncRNA"][value] for value in l_index]
                pathway = [inverse["pathway"][value] for value in p_index]
                pair_fold = np.fromiter(
                    (
                        pair_blocked_fold(left, right, seed=pair_seed)
                        for left, right in zip(lnc, pathway)
                    ),
                    dtype=np.int16,
                    count=len(lnc),
                )
                if not np.all(pair_fold == validation_fold):
                    raise ValidationPredictionInferenceError(
                        f"G012 fold {fold} validation batch contains another pair fold"
                    )
                target = (
                    raw["proxy_label"].detach().cpu().numpy().astype(float).reshape(-1)
                )
                if not np.isin(target, [0.0, 1.0]).all():
                    raise ValidationPredictionInferenceError(
                        f"G012 fold {fold} validation target is not binary"
                    )
                probability = 1.0 / (1.0 + np.exp(-final_logit))
                parts.append(
                    pd.DataFrame(
                        {
                            "cancer_id": cancer,
                            "lncrna_id": lnc,
                            "pathway_id": pathway,
                            FUSION_FOLD_COLUMN: pair_fold,
                            "selection_outer_pair_fold": int(fold),
                            "selection_validation_pair_fold": pair_fold,
                            "outer_test_queried": False,
                            "fusion_target": target.astype(np.uint8),
                            "primary_probability": probability.astype(np.float32),
                        }
                    )
                )
            fold_lineage.append(
                {
                    "patient_fold": fold,
                    "validation_pair_fold": validation_fold,
                    "prepared_sha256": artifact_sha256(declared_prepared),
                    "checkpoint_sha256": artifact_sha256(checkpoint_path),
                    "runtime_graph_full_registered_estimand": True,
                    "test_batches_deserialized": False,
                }
            )
        frame = (
            pd.concat(parts, ignore_index=True)
            .sort_values(list(TARGET_KEYS), kind="stable")
            .reset_index(drop=True)
        )
        if (
            frame[list(TARGET_KEYS)].duplicated().any()
            or set(frame[FUSION_FOLD_COLUMN].astype(int)) != set(range(5))
            or not frame.outer_test_queried.eq(False).all()
        ):
            raise ValidationPredictionInferenceError(
                "G012 validation predictions do not form one exact five-fold universe"
            )
        candidate_sha, fold_sha = exact_candidate_and_fold_hashes(frame)
        prediction_name = "G012_VALIDATION_PREDICTIONS.PRIVATE.parquet"
        prediction_path = staging / prediction_name
        frame.to_parquet(
            prediction_path, index=False, compression="zstd", row_group_size=100_000
        )
        lineage = {
            "status": G012_VALIDATION_STATUS,
            "graph_variant": variant,
            "winner_declaration": {
                "path": str(winner_path),
                "sha256": expected_winner_sha,
            },
            "winner_config": {
                "path": str(config_path),
                "sha256": artifact_sha256(config_path),
            },
            "folds": fold_lineage,
            "rows": int(len(frame)),
            "candidate_universe_sha256": candidate_sha,
            "validation_pair_fold_sha256": fold_sha,
            "outer_test_predictions_written": False,
            "outer_test_metrics_computed": False,
            "test_batches_deserialized": False,
            "device": device,
        }
        lineage_path = staging / "LINEAGE.json"
        lineage_path.write_text(
            json.dumps(lineage, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        success = {
            "status": G012_VALIDATION_STATUS,
            "graph_variant": variant,
            "rows": int(len(frame)),
            "validation_predictions": {
                "path": str(output / prediction_name),
                "sha256": artifact_sha256(prediction_path),
            },
            "lineage_sha256": artifact_sha256(lineage_path),
            "outer_test_predictions_written": False,
            "outer_test_metrics_computed": False,
            "test_batches_deserialized": False,
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
    "G012_VALIDATION_STATUS",
    "ValidationPredictionInferenceError",
    "infer_final_logits_for_batches",
    "inverse_node_maps",
    "materialize_g012_validation_predictions",
]
