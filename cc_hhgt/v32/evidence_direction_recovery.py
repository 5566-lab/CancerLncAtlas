"""Recover public direction probabilities from fresh V3.2 Evidence heads.

The original formal inference retained only the argmax direction even though
the private EventSet head produced a three-class softmax.  This module performs
a new, deterministic MC-dropout inference from the hash-pinned V3.2
checkpoints.  It does not train, mutate the exact-pathway primary score, or use
historical predictions/checkpoints.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import duckdb
import numpy as np
import pandas as pd

from .evidence_training import (
    ANALYSIS_VERSION,
    EVENT_FEATURE_FIELDS,
    EXACT_KEYS,
    N_FOLDS,
    PRIVATE_CHECKPOINT_FORMAT,
    _batch_tensors,
    build_bag_examples,
    build_fresh_private_eventset_head,
    file_sha256,
    load_core_feature_bundle,
    model_parameter_sha256,
)


FORMAT = "CC_HHGT_V3_2_EVIDENCE_DIRECTION_PROBABILITY_BINDING_V1"
OUTPUT_FORMAT = "CC_HHGT_V3_2_EVIDENCE_DIRECTION_PROBABILITIES_V1"
EXPECTED_BINDING_FORMAT = "CC_HHGT_V3_2_EVIDENCE_OUTPUT_BINDING_V1"
EXPECTED_BINDING_STATUS = "SUCCESS_FRESH_EVIDENCE_OUTPUTS_HASH_BOUND"
PROBABILITY_COLUMNS = (
    "direction_negative_probability",
    "direction_neutral_probability",
    "direction_positive_probability",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TRAINING_DROPOUT = 0.20
_TRAINING_MAX_EVENTS = 64


class EvidenceDirectionRecoveryError(RuntimeError):
    """Raised when recovered probabilities cannot be proven V3.2-only."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceDirectionRecoveryError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceDirectionRecoveryError(f"Expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''").replace("\\", "/")


def _require_bound_file(record: Mapping[str, Any], label: str) -> Path:
    if not isinstance(record, Mapping):
        raise EvidenceDirectionRecoveryError(f"Bound {label} declaration is missing")
    path = Path(str(record.get("path", ""))).resolve()
    expected = str(record.get("sha256", "")).lower()
    if (
        not _SHA256.fullmatch(expected)
        or not path.is_file()
        or path.is_symlink()
        or file_sha256(path) != expected
    ):
        raise EvidenceDirectionRecoveryError(f"Hash drift in bound {label}: {path}")
    return path


def _load_authority(
    evidence_binding_path: Path, expected_binding_sha256: str
) -> tuple[dict[str, Any], Path, Path, dict[str, Any]]:
    binding_path = evidence_binding_path.resolve()
    expected_binding_sha256 = str(expected_binding_sha256).lower()
    if (
        not _SHA256.fullmatch(expected_binding_sha256)
        or binding_path.is_symlink()
        or not binding_path.is_file()
        or file_sha256(binding_path) != expected_binding_sha256
    ):
        raise EvidenceDirectionRecoveryError("Evidence binding SHA256 mismatch")
    binding = _read_json(binding_path)
    required = {
        "format": EXPECTED_BINDING_FORMAT,
        "status": EXPECTED_BINDING_STATUS,
        "analysis_version": ANALYSIS_VERSION,
        "five_fresh_private_heads_verified": True,
        "historical_checkpoints_used": False,
        "historical_predictions_used": False,
        "historical_rankings_used": False,
        "family_to_exact_broadcast": False,
    }
    for key, expected in required.items():
        if binding.get(key) != expected or type(binding.get(key)) is not type(expected):
            raise EvidenceDirectionRecoveryError(
                f"Evidence authority violates {key}: {binding.get(key)!r}"
            )
    predictions = _require_bound_file(
        binding.get("artifacts", {}).get("evidence_predictions", {}),
        "evidence predictions",
    )
    event_lineage = _require_bound_file(
        binding.get("artifacts", {}).get("event_lineage", {}),
        "event lineage",
    )
    training_manifest_path = _require_bound_file(
        binding.get("authorities", {}).get("training_manifest", {}),
        "training manifest",
    )
    bound_training_code = _require_bound_file(
        binding.get("authorities", {}).get("evidence_code", {}),
        "Evidence training code",
    )
    _require_bound_file(
        binding.get("authorities", {}).get("evidence_runner", {}),
        "Evidence training runner",
    )
    imported_training_code = Path(build_bag_examples.__code__.co_filename).resolve()
    if (
        imported_training_code != bound_training_code
        or file_sha256(imported_training_code) != file_sha256(bound_training_code)
    ):
        raise EvidenceDirectionRecoveryError(
            "Imported Evidence implementation is not the hash-bound V3.2 training code"
        )
    training_manifest = _read_json(training_manifest_path)
    if (
        training_manifest.get("analysis_version") != ANALYSIS_VERSION
        or training_manifest.get("module_id") != "evidence_private_eventset"
        or training_manifest.get("status") != "SUCCESS_NEWLY_TRAINED"
        or training_manifest.get("training_generation") != "V3.2"
        or training_manifest.get("checkpoint_format") != PRIVATE_CHECKPOINT_FORMAT
        or training_manifest.get("historical_evidence_checkpoint_loaded") is not False
        or training_manifest.get("historical_evidence_result_loaded") is not False
        or training_manifest.get("family_to_exact_broadcast_used") is not False
        or training_manifest.get("core_frozen") is not True
        or training_manifest.get("core_detached") is not True
        or training_manifest.get("main_ranking_modified") is not False
        or training_manifest.get("prediction_role") != "AUXILIARY_CONFIDENCE_ONLY"
        or training_manifest.get("counts", {}).get("trained_private_heads") != N_FOLDS
    ):
        raise EvidenceDirectionRecoveryError("Training manifest is not fresh V3.2 Evidence")
    return binding, predictions, event_lineage, training_manifest


def _checkpoint_payload(
    *,
    binding: Mapping[str, Any],
    training_manifest: Mapping[str, Any],
    fold: int,
    core_manifest_sha256: str,
) -> tuple[Path, dict[str, Any]]:
    record = binding.get("checkpoints", {}).get(str(fold))
    if not isinstance(record, Mapping):
        raise EvidenceDirectionRecoveryError(f"Evidence binding lacks fold {fold}")
    path = _require_bound_file(record, f"fold {fold} checkpoint")
    torch = __import__("torch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise EvidenceDirectionRecoveryError(f"Fold {fold} checkpoint is not a mapping")
    expected = {
        "checkpoint_format": PRIVATE_CHECKPOINT_FORMAT,
        "patient_fold": fold,
        "private_parameters_fresh_init": True,
        "initialized_from_checkpoint": False,
        "historical_evidence_checkpoint_allowed": False,
        "historical_evidence_result_allowed": False,
        "contains_core_parameters": False,
        "contains_primary_ranking_parameters": False,
        "direction_supervision_available": True,
        "core_manifest_sha256": core_manifest_sha256,
    }
    for key, value in expected.items():
        if payload.get(key) != value or type(payload.get(key)) is not type(value):
            raise EvidenceDirectionRecoveryError(
                f"Fold {fold} checkpoint violates {key}: {payload.get(key)!r}"
            )
    manifest_fold = training_manifest.get("folds", {}).get(str(fold), {})
    for key in (
        "initial_parameter_sha256",
        "final_parameter_sha256",
        "optimizer_steps",
        "core_checkpoint_sha256",
        "core_parameter_sha256",
    ):
        if payload.get(key) != manifest_fold.get(key) or payload.get(key) != record.get(key):
            raise EvidenceDirectionRecoveryError(
                f"Fold {fold} checkpoint/manifest mismatch for {key}"
            )
    if int(payload.get("optimizer_steps", 0)) < 1:
        raise EvidenceDirectionRecoveryError(f"Fold {fold} has no optimizer steps")
    if payload.get("initial_parameter_sha256") == payload.get("final_parameter_sha256"):
        raise EvidenceDirectionRecoveryError(f"Fold {fold} parameters did not change")
    return path, payload


def _event_columns(event_lineage: Path) -> list[str]:
    connection = duckdb.connect()
    try:
        available = {
            str(row[0])
            for row in connection.execute(
                "DESCRIBE SELECT * FROM read_parquet(?)", [str(event_lineage)]
            ).fetchall()
        }
    finally:
        connection.close()
    required = set(
        EXACT_KEYS
        + ["event_id", "leakage_fold", "confidence_target", "direction_target"]
        + list(EVENT_FEATURE_FIELDS)
        + ["is_experimental", "is_computational", "is_physical", "pmid"]
    )
    missing = sorted(required - available)
    if missing:
        raise EvidenceDirectionRecoveryError(
            f"Bound event lineage lacks inference columns: {missing}"
        )
    return sorted(required)


def _infer_fold(
    *,
    events: pd.DataFrame,
    core_embedding_root: Path,
    fold: int,
    checkpoint_payload: Mapping[str, Any],
    batch_size: int,
    mc_samples: int,
    dropout: float,
    inference_seed: int,
    device: str | None,
) -> pd.DataFrame:
    if mc_samples < 2:
        raise EvidenceDirectionRecoveryError("mc_samples must be at least two")
    torch = __import__("torch")
    core = load_core_feature_bundle(core_embedding_root, fold)
    state = checkpoint_payload.get("private_model_state")
    if not isinstance(state, Mapping):
        raise EvidenceDirectionRecoveryError(f"Fold {fold} lacks private model state")
    event_weight = state.get("event_projection.0.weight")
    core_weight = state.get("core_query.0.weight")
    if event_weight is None or core_weight is None:
        raise EvidenceDirectionRecoveryError(f"Fold {fold} lacks architecture tensors")
    hidden_dim, event_feature_dim = map(int, event_weight.shape)
    checkpoint_core_dim = int(core_weight.shape[1])
    if core.combined_dim != checkpoint_core_dim:
        raise EvidenceDirectionRecoveryError(
            f"Fold {fold} core dimension mismatch: {core.combined_dim} != {checkpoint_core_dim}"
        )
    examples, missing_core = build_bag_examples(
        events,
        core,
        event_feature_dim=event_feature_dim,
        max_events=_TRAINING_MAX_EVENTS,
    )
    if not missing_core.empty:
        raise EvidenceDirectionRecoveryError(
            f"Fold {fold} has {len(missing_core)} missing core bags"
        )
    model = build_fresh_private_eventset_head(
        event_feature_dim=event_feature_dim,
        core_feature_dim=core.combined_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        seed=int(checkpoint_payload.get("seed", 0)),
    )
    model.load_state_dict(state, strict=True)
    if model_parameter_sha256(model) != checkpoint_payload.get("final_parameter_sha256"):
        raise EvidenceDirectionRecoveryError(f"Fold {fold} parameter SHA256 mismatch")
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(torch_device)
    torch.manual_seed(inference_seed + fold * 1009)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(inference_seed + fold * 1009)
    direction_names = np.asarray(["negative", "neutral", "positive"], dtype=object)
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            subset = examples[start : start + batch_size]
            batch = _batch_tensors(
                subset, list(range(len(subset))), torch_device
            )
            draws: list[np.ndarray] = []
            for _ in range(mc_samples):
                model.train()
                result = model(batch["events"], batch["mask"], batch["core"])
                draws.append(
                    torch.softmax(result["direction_logits"], dim=-1)
                    .detach()
                    .cpu()
                    .numpy()
                )
            mean_probability = np.stack(draws).mean(axis=0)
            for index, example in enumerate(subset):
                probability = mean_probability[index].astype(float)
                entropy = -float(
                    np.sum(probability * np.log(np.clip(probability, 1.0e-12, 1.0)))
                ) / math.log(3.0)
                records.append(
                    {
                        **dict(zip(EXACT_KEYS, example.key, strict=True)),
                        PROBABILITY_COLUMNS[0]: float(probability[0]),
                        PROBABILITY_COLUMNS[1]: float(probability[1]),
                        PROBABILITY_COLUMNS[2]: float(probability[2]),
                        "recovered_direction": str(
                            direction_names[int(np.argmax(probability))]
                        ),
                        "direction_entropy": float(entropy),
                        "evidence_fold": fold,
                        "event_count": int(example.event_count),
                    }
                )
    model.eval()
    return pd.DataFrame.from_records(records)


def recover_evidence_direction_probabilities(
    *,
    evidence_binding_path: str | Path,
    expected_evidence_binding_sha256: str,
    core_embedding_root: str | Path,
    output_root: str | Path,
    batch_size: int = 512,
    mc_samples: int = 16,
    dropout: float = 0.20,
    inference_seed: int = 20270826,
    device: str | None = None,
) -> dict[str, Any]:
    """Materialize a full 3.3M typed-null direction-probability release."""

    if batch_size < 1 or mc_samples < 2 or not 0.0 <= dropout < 1.0:
        raise EvidenceDirectionRecoveryError("Invalid recovery inference parameters")
    if dropout != _TRAINING_DROPOUT:
        raise EvidenceDirectionRecoveryError(
            "Formal recovery requires the V3.2 training-head dropout value 0.20"
        )
    torch = __import__("torch")
    if device == "cuda" and not torch.cuda.is_available():
        raise EvidenceDirectionRecoveryError("CUDA recovery requested but CUDA is unavailable")
    effective_device = str(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    output = Path(output_root).resolve()
    if output.exists():
        raise EvidenceDirectionRecoveryError(f"Output root already exists: {output}")
    output.mkdir(parents=True)
    parts_root = output / "available_parts"
    parts_root.mkdir()
    binding_path = Path(evidence_binding_path).resolve()
    core_root = Path(core_embedding_root).resolve()
    core_manifest_path = core_root / "CORE_EMBEDDING_MANIFEST.json"
    if not core_manifest_path.is_file():
        raise EvidenceDirectionRecoveryError("V3.2 core embedding manifest is missing")
    core_manifest_sha256 = file_sha256(core_manifest_path)
    binding, predictions, event_lineage, training_manifest = _load_authority(
        binding_path, expected_evidence_binding_sha256
    )
    event_columns = _event_columns(event_lineage)
    partition_records: list[dict[str, Any]] = []
    expected_available = int(binding.get("counts", {}).get("evidence_predictions", 0))
    expected_available = int(
        training_manifest.get("counts", {}).get("available_predictions", expected_available)
    )
    recovered_rows = 0
    for fold in range(N_FOLDS):
        checkpoint_path, checkpoint_payload = _checkpoint_payload(
            binding=binding,
            training_manifest=training_manifest,
            fold=fold,
            core_manifest_sha256=core_manifest_sha256,
        )
        events = pd.read_parquet(
            event_lineage,
            columns=event_columns,
            filters=[("leakage_fold", "==", fold)],
        )
        if events.empty or set(pd.to_numeric(events.leakage_fold).astype(int)) != {fold}:
            raise EvidenceDirectionRecoveryError(f"Fold {fold} event filter failed")
        part = _infer_fold(
            events=events,
            core_embedding_root=core_root,
            fold=fold,
            checkpoint_payload=checkpoint_payload,
            batch_size=batch_size,
            mc_samples=mc_samples,
            dropout=dropout,
            inference_seed=inference_seed,
            device=effective_device,
        )
        expected_fold_rows = int(
            training_manifest.get("folds", {}).get(str(fold), {}).get("n_heldout_bags", -1)
        )
        if len(part) != expected_fold_rows or part.duplicated(EXACT_KEYS).any():
            raise EvidenceDirectionRecoveryError(
                f"Fold {fold} rows differ from training authority: {len(part)} != {expected_fold_rows}"
            )
        probabilities = part[list(PROBABILITY_COLUMNS)].to_numpy(float)
        if (
            not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or (probabilities > 1).any()
            or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1.0e-5)
        ):
            raise EvidenceDirectionRecoveryError(f"Fold {fold} probabilities are invalid")
        part_path = parts_root / f"evidence_fold={fold}.parquet"
        part.to_parquet(part_path, index=False, compression="zstd")
        partition_records.append(
            {
                "fold": fold,
                "path": str(part_path),
                "rows": int(len(part)),
                "sha256": file_sha256(part_path),
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": file_sha256(checkpoint_path),
                "final_parameter_sha256": checkpoint_payload["final_parameter_sha256"],
            }
        )
        recovered_rows += int(len(part))
    if recovered_rows != expected_available:
        raise EvidenceDirectionRecoveryError(
            f"Recovered rows differ from available predictions: {recovered_rows} != {expected_available}"
        )

    final_path = output / "evidence_direction_probabilities.parquet"
    temporary_path = output / ".evidence_direction_probabilities.tmp.parquet"
    connection = duckdb.connect()
    try:
        connection.execute("SET threads=4")
        connection.execute("SET memory_limit='8GB'")
        connection.execute("SET preserve_insertion_order=false")
        connection.execute(
            f"""
            COPY (
              SELECT
                p.cancer_id,
                p.lncrna_id,
                p.pathway_id,
                p.direction AS published_direction,
                r.recovered_direction,
                CAST(r.direction_negative_probability AS FLOAT)
                  AS direction_negative_probability,
                CAST(r.direction_neutral_probability AS FLOAT)
                  AS direction_neutral_probability,
                CAST(r.direction_positive_probability AS FLOAT)
                  AS direction_positive_probability,
                CAST(r.direction_entropy AS FLOAT) AS direction_entropy,
                (r.cancer_id IS NOT NULL) AS direction_probability_available,
                CASE WHEN r.cancer_id IS NULL THEN
                  coalesce(nullif(p.unavailable_reason, ''), 'NO_RECOVERABLE_RAW_EVENT_BAG')
                ELSE NULL END AS direction_probability_unavailable_reason,
                CASE WHEN r.cancer_id IS NULL THEN NULL
                     ELSE p.direction = r.recovered_direction END
                  AS published_direction_matches_recovered_argmax,
                p.availability AS evidence_confidence_available,
                r.evidence_fold,
                r.event_count,
                '{ANALYSIS_VERSION}' AS analysis_version,
                '{OUTPUT_FORMAT}' AS output_format,
                FALSE AS changes_primary_ranking,
                FALSE AS changes_discovery_ranking,
                FALSE AS used_for_fusion,
                FALSE AS historical_checkpoint_used,
                FALSE AS historical_prediction_used
              FROM read_parquet('{_sql_path(predictions)}') p
              LEFT JOIN read_parquet('{_sql_path(parts_root / '*.parquet')}') r
              USING (cancer_id, lncrna_id, pathway_id)
            ) TO '{_sql_path(temporary_path)}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        counts = connection.execute(
            f"""
            SELECT count(*) total_rows,
                   count_if(t.direction_probability_available) available_rows,
                   count_if(NOT t.direction_probability_available) unavailable_rows,
                   count_if(t.direction_probability_available AND
                     abs(t.direction_negative_probability + t.direction_neutral_probability +
                         t.direction_positive_probability - 1.0) > 1e-5) bad_sum,
                   count_if(t.direction_probability_available AND
                     NOT t.published_direction_matches_recovered_argmax) argmax_mismatch_rows,
                   count_if(NOT t.direction_probability_available AND
                     (t.direction_negative_probability IS NOT NULL OR
                      t.direction_neutral_probability IS NOT NULL OR
                      t.direction_positive_probability IS NOT NULL)) null_policy_violations,
                   count_if(t.direction_probability_available AND (
                     NOT p.availability OR p.direction IS NULL OR
                     p.direction NOT IN ('negative', 'neutral', 'positive') OR
                     p.evidence_fold <> r.evidence_fold OR p.event_count <> r.event_count
                   )) source_alignment_violations,
                   count_if(NOT t.direction_probability_available AND (
                     t.recovered_direction IS NOT NULL OR t.direction_entropy IS NOT NULL OR
                     t.evidence_fold IS NOT NULL OR t.event_count IS NOT NULL OR
                     t.direction_probability_unavailable_reason IS NULL OR
                     t.direction_probability_unavailable_reason = ''
                   )) typed_null_violations,
                   count(DISTINCT t.cancer_id) cancers
            FROM read_parquet('{_sql_path(temporary_path)}') t
            JOIN read_parquet('{_sql_path(predictions)}') p
            USING (cancer_id, lncrna_id, pathway_id)
            LEFT JOIN read_parquet('{_sql_path(parts_root / '*.parquet')}') r
            USING (cancer_id, lncrna_id, pathway_id)
            """
        ).fetchone()
    finally:
        connection.close()
    if counts is None:
        raise EvidenceDirectionRecoveryError("Direction output audit returned no row")
    (
        total_rows,
        available_rows,
        unavailable_rows,
        bad_sum,
        mismatches,
        null_violations,
        source_alignment_violations,
        typed_null_violations,
        cancers,
    ) = map(
        int, counts
    )
    if (
        total_rows != int(binding.get("counts", {}).get("evidence_predictions", 0))
        or available_rows != expected_available
        or cancers != 33
        or bad_sum != 0
        or null_violations != 0
        or source_alignment_violations != 0
        or typed_null_violations != 0
    ):
        raise EvidenceDirectionRecoveryError(f"Final direction audit failed: {counts}")
    temporary_path.replace(final_path)
    run_id = "V32-EVIDENCE-DIRECTION-REINFERENCE-" + _json_sha256(
        {
            "evidence_binding_sha256": expected_evidence_binding_sha256,
            "core_manifest_sha256": core_manifest_sha256,
            "mc_samples": mc_samples,
            "dropout": dropout,
            "inference_seed": inference_seed,
            "batch_size": batch_size,
            "device": effective_device,
        }
    )[:16]
    release = {
        "format": FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "status": "SUCCESS_V32_CHECKPOINT_REINFERENCE",
        "inference_run_id": run_id,
        "training_performed": False,
        "source_checkpoints_newly_trained_v32": True,
        "old_checkpoint_loaded": False,
        "old_predictions_used_as_features": False,
        "historical_rankings_used": False,
        "primary_ranking_unchanged": True,
        "discovery_ranking_unchanged": True,
        "used_for_fusion": False,
        "stochastic_reinference_policy": (
            "FIXED_SEED_MC_DROPOUT_FROM_HASH_PINNED_V32_PRIVATE_HEADS"
        ),
        "published_argmax_is_original_inference": True,
        "recovered_argmax_is_fixed_seed_reinference": True,
        "argmax_mismatch_is_not_checkpoint_drift": True,
        "parameters": {
            "batch_size": batch_size,
            "mc_samples": mc_samples,
            "dropout": dropout,
            "inference_seed": inference_seed,
            "max_events": _TRAINING_MAX_EVENTS,
            "device": effective_device,
        },
        "authority": {
            "evidence_binding_path": str(binding_path),
            "evidence_binding_sha256": file_sha256(binding_path),
            "core_manifest_path": str(core_manifest_path),
            "core_manifest_sha256": core_manifest_sha256,
            "event_lineage_path": str(event_lineage),
            "event_lineage_sha256": file_sha256(event_lineage),
            "original_predictions_path": str(predictions),
            "original_predictions_sha256": file_sha256(predictions),
        },
        "partitions": partition_records,
        "artifact": {
            "path": str(final_path),
            "sha256": file_sha256(final_path),
            "bytes": final_path.stat().st_size,
            "rows": total_rows,
            "columns": list(EXACT_KEYS)
            + [
                "published_direction",
                "recovered_direction",
                *PROBABILITY_COLUMNS,
                "direction_entropy",
                "direction_probability_available",
                "direction_probability_unavailable_reason",
                "published_direction_matches_recovered_argmax",
                "evidence_confidence_available",
                "evidence_fold",
                "event_count",
                "analysis_version",
                "output_format",
                "changes_primary_ranking",
                "changes_discovery_ranking",
                "used_for_fusion",
                "historical_checkpoint_used",
                "historical_prediction_used",
            ],
        },
        "counts": {
            "total_rows": total_rows,
            "available_rows": available_rows,
            "unavailable_rows": unavailable_rows,
            "cancers": cancers,
            "probability_sum_violations": bad_sum,
            "null_policy_violations": null_violations,
            "source_alignment_violations": source_alignment_violations,
            "typed_null_violations": typed_null_violations,
            "published_recovered_argmax_mismatch_rows": mismatches,
            "published_recovered_argmax_match_rows": available_rows - mismatches,
        },
        "production_deployed": False,
        "release_ready": False,
    }
    release_path = output / "EVIDENCE_DIRECTION_PROBABILITY_BINDING.json"
    _write_json(release_path, release)
    success = {
        "status": "SUCCESS_V32_CHECKPOINT_REINFERENCE",
        "binding_path": str(release_path),
        "binding_sha256": file_sha256(release_path),
        "rows": total_rows,
        "available_rows": available_rows,
        "production_deployed": False,
        "release_ready": False,
    }
    _write_json(output / "SUCCESS.json", success)
    return success


__all__ = [
    "EvidenceDirectionRecoveryError",
    "FORMAT",
    "OUTPUT_FORMAT",
    "PROBABILITY_COLUMNS",
    "recover_evidence_direction_probabilities",
]
