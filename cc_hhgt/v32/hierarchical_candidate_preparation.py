"""Enrich formal prepared folds for fair hierarchical-gate retraining."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .multimodal_fusion import TARGET_KEYS, artifact_sha256, pair_blocked_fold


MODALITIES = ("mutation", "cnv", "atac")
from .training import PREPARED_FORMAT


class HierarchicalPreparationError(RuntimeError):
    pass


def _canonical_predictions(
    genomic: pd.DataFrame, atac: pd.DataFrame | None
) -> pd.DataFrame:
    if missing := sorted(set(TARGET_KEYS) - set(genomic.columns)):
        raise HierarchicalPreparationError(f"Genomic predictions lack exact keys: {missing}")
    genomic = genomic.copy()
    genomic["cancer_id"] = genomic.cancer_id.astype(str).str.upper()
    for column in ("lncrna_id", "pathway_id"):
        genomic[column] = genomic[column].astype(str)
    genomic = genomic.sort_values(list(TARGET_KEYS), kind="stable").reset_index(drop=True)
    result = genomic[list(TARGET_KEYS)].copy()
    for modality in ("mutation", "cnv"):
        probability = f"{modality}_context_probability"
        availability = f"{modality}_available"
        if missing := sorted({probability, availability} - set(genomic.columns)):
            raise HierarchicalPreparationError(f"Genomic predictions lack {modality}: {missing}")
        result[f"{modality}_probability"] = pd.to_numeric(genomic[probability], errors="coerce")
        result[f"{modality}_available"] = genomic[availability].astype(bool)
    if atac is None:
        result["atac_probability"] = np.nan
        result["atac_available"] = False
    else:
        if missing := sorted(
            set(TARGET_KEYS + ("atac_context_probability", "atac_available")) - set(atac.columns)
        ):
            raise HierarchicalPreparationError(f"ATAC predictions lack: {missing}")
        atac = atac.copy()
        atac["cancer_id"] = atac.cancer_id.astype(str).str.upper()
        for column in ("lncrna_id", "pathway_id"):
            atac[column] = atac[column].astype(str)
        atac = atac.sort_values(list(TARGET_KEYS), kind="stable").reset_index(drop=True)
        if len(atac) != len(genomic) or not atac[list(TARGET_KEYS)].equals(genomic[list(TARGET_KEYS)]):
            raise HierarchicalPreparationError("ATAC and genomic candidate universes differ")
        result["atac_probability"] = pd.to_numeric(atac.atac_context_probability, errors="coerce")
        result["atac_available"] = atac.atac_available.astype(bool)
    for modality in MODALITIES:
        available = result[f"{modality}_available"].to_numpy(bool)
        probability = result[f"{modality}_probability"].to_numpy(float)
        if np.isnan(probability[available]).any() or np.isfinite(probability[~available]).any():
            raise HierarchicalPreparationError(f"{modality} violates typed missingness")
        finite = probability[available]
        if ((finite < 0) | (finite > 1)).any():
            raise HierarchicalPreparationError(f"{modality} probability is outside [0, 1]")
    if result[list(TARGET_KEYS)].duplicated().any():
        raise HierarchicalPreparationError("Modality predictions have duplicate exact keys")
    indexed = result.set_index(list(TARGET_KEYS))
    if not indexed.index.is_unique:
        raise HierarchicalPreparationError("Modality predictions have duplicate exact keys")
    return indexed


def _inverse_node_maps(bundle: Any) -> dict[str, dict[int, str]]:
    maps = getattr(bundle, "node_maps", None)
    if not isinstance(maps, Mapping):
        raise HierarchicalPreparationError("Prepared graph bundle lacks node_maps")
    result: dict[str, dict[int, str]] = {}
    for kind in ("cancer", "lncRNA", "pathway"):
        if kind not in maps:
            raise HierarchicalPreparationError(f"Prepared graph lacks {kind} node map")
        result[kind] = {int(index): str(identifier) for identifier, index in maps[kind].items()}
    return result


def _slice(value: Any, mask, torch):
    if hasattr(value, "shape") and getattr(value, "shape", ()) and int(value.shape[0]) == len(mask):
        return value[mask]
    if isinstance(value, Mapping):
        return {key: _slice(item, mask, torch) for key, item in value.items()}
    if isinstance(value, list) and len(value) == len(mask):
        selected = mask.detach().cpu().numpy().astype(bool)
        return [item for item, keep in zip(value, selected) if keep]
    return value


def _enrich_batches(
    batches: list[Mapping[str, Any]],
    *,
    split: str,
    heldout_pair_fold: int,
    predictions: pd.DataFrame,
    inverse: Mapping[str, Mapping[int, str]],
    torch,
) -> list[dict[str, Any]]:
    validation_pair_fold = (heldout_pair_fold + 1) % 5
    output: list[dict[str, Any]] = []
    for raw in batches:
        batch = dict(raw)
        candidate = dict(batch["candidate_batch"])
        l_index = candidate["l"].detach().cpu().numpy().astype(int)
        p_index = candidate["p"].detach().cpu().numpy().astype(int)
        c_index = candidate["c"].detach().cpu().numpy().astype(int)
        keys = pd.DataFrame(
            {
                "cancer_id": [inverse["cancer"][value].upper() for value in c_index],
                "lncrna_id": [inverse["lncRNA"][value] for value in l_index],
                "pathway_id": [inverse["pathway"][value] for value in p_index],
            }
        )
        pair_folds = np.fromiter(
            (pair_blocked_fold(lnc, pathway) for lnc, pathway in zip(keys.lncrna_id, keys.pathway_id)),
            dtype=np.int16,
            count=len(keys),
        )
        if split == "train":
            keep = ~np.isin(pair_folds, [heldout_pair_fold, validation_pair_fold])
        elif split == "validation":
            keep = pair_folds == validation_pair_fold
        elif split == "test":
            keep = pair_folds == heldout_pair_fold
        else:
            raise ValueError(split)
        if not keep.any():
            continue
        joined = predictions.reindex(pd.MultiIndex.from_frame(keys[list(TARGET_KEYS)]))
        if len(joined) != len(keys):
            raise HierarchicalPreparationError("Modality reindex changed batch length")
        availability_columns = [f"{modality}_available" for modality in MODALITIES]
        if joined[availability_columns].isna().any().any():
            raise HierarchicalPreparationError("Prepared batch contains a key absent from modality predictions")
        probability = np.column_stack(
            [joined[f"{modality}_probability"].to_numpy(float) for modality in MODALITIES]
        )
        availability = np.column_stack(
            [joined[f"{modality}_available"].to_numpy(bool) for modality in MODALITIES]
        )
        if np.isnan(probability[availability]).any():
            raise HierarchicalPreparationError("Available modality is missing after batch join")
        keep_tensor = torch.tensor(keep, dtype=torch.bool)
        batch = _slice(batch, keep_tensor, torch)
        candidate = dict(batch["candidate_batch"])
        candidate["modality_probability"] = torch.tensor(probability[keep], dtype=torch.float32)
        candidate["modality_available"] = torch.tensor(availability[keep], dtype=torch.bool)
        candidate["outer_pair_fold"] = torch.tensor(pair_folds[keep], dtype=torch.int16)
        batch["candidate_batch"] = candidate
        output.append(batch)
    if not output:
        raise HierarchicalPreparationError(f"Pair-blocked {split} produced no batches")
    return output


def prepare_hierarchical_candidate_folds(
    *,
    formal_prepared_root: str | Path,
    genomic_predictions_path: str | Path,
    genomic_lineage_path: str | Path,
    output_root: str | Path,
    atac_predictions_path: str | Path | None = None,
    atac_lineage_path: str | Path | None = None,
    seed: int = 20260726,
    expected_graph_variant: str = "G2",
) -> dict[str, Any]:
    import torch

    source_root = Path(formal_prepared_root).resolve()
    from .patient_fold_authority import (
        FROZEN_V32_RECEIPT_SHA256,
        FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256,
        validate_frozen_v32_prepared_fold_binding,
    )

    prepared_patient_binding = validate_frozen_v32_prepared_fold_binding(source_root)
    if expected_graph_variant not in {"G0", "G1", "G2"}:
        raise HierarchicalPreparationError("Expected graph variant must be G0, G1 or G2")
    variant_marker_path = source_root / "FORMAL_GRAPH_VARIANT.json"
    try:
        variant_marker = json.loads(variant_marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HierarchicalPreparationError("Prepared root lacks a graph-variant marker") from exc
    if (
        variant_marker.get("format")
        != "CANCERLNCATLAS_V32_FORMAL_GRAPH_VARIANT_ROOT_V1"
        or variant_marker.get("variant") != expected_graph_variant
        or variant_marker.get("legacy_root_fold_payloads_allowed") is not False
    ):
        raise HierarchicalPreparationError("Prepared root graph variant drift")
    genomic_path = Path(genomic_predictions_path).resolve()
    genomic_lineage = Path(genomic_lineage_path).resolve()
    atac_path = Path(atac_predictions_path).resolve() if atac_predictions_path else None
    atac_lineage = Path(atac_lineage_path).resolve() if atac_lineage_path else None
    if (atac_path is None) != (atac_lineage is None):
        raise HierarchicalPreparationError(
            "ATAC predictions and lineage must be supplied together"
        )
    from .cancer_modality_router import (
        validate_atac_oof_lineage,
        validate_patient_oof_lineage,
    )

    validate_patient_oof_lineage(
        json.loads(genomic_lineage.read_text(encoding="utf-8")),
        predictions_sha256=artifact_sha256(genomic_path),
    )
    if atac_path is not None and atac_lineage is not None:
        validate_atac_oof_lineage(
            json.loads(atac_lineage.read_text(encoding="utf-8")),
            predictions_sha256=artifact_sha256(atac_path),
        )
    output = Path(output_root).resolve()
    if output.exists():
        raise HierarchicalPreparationError(f"Hierarchical preparation refuses output reuse: {output}")
    output.mkdir(parents=True)
    for filename in (
        "SAMPLE_PATIENT_FOLD_MAP.tsv",
        "PATIENT_FOLD_AUTHORITY_RECEIPT.json",
        "PATIENT_FOLD_BINDING.json",
    ):
        (output / filename).write_bytes((source_root / filename).read_bytes())
    genomic = pd.read_parquet(genomic_path)
    atac = pd.read_parquet(atac_path) if atac_path else None
    predictions = _canonical_predictions(genomic, atac)
    records: list[dict[str, Any]] = []
    for fold in range(5):
        source = source_root / f"PATIENT_FOLD_{fold}.pt"
        if not source.is_file():
            raise HierarchicalPreparationError(f"Missing formal prepared fold: {source}")
        payload = torch.load(source, map_location="cpu", weights_only=False)
        if (
            payload.get("prepared_format") != PREPARED_FORMAT
            or int(payload.get("patient_fold", -1)) != fold
            or payload.get("formal_graph_variant") != expected_graph_variant
            or "test_batches" in payload
        ):
            raise HierarchicalPreparationError(f"Prepared fold {fold} contract drift")
        payload_binding = payload.get("patient_fold_authority")
        if (
            not isinstance(payload_binding, Mapping)
            or payload_binding.get("sample_patient_fold_map", {}).get("sha256")
            != FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256
            or payload_binding.get("authority_receipt", {}).get("sha256")
            != FROZEN_V32_RECEIPT_SHA256
            or payload_binding.get("sample_id_patient_fallback_used") is not False
            or payload_binding.get("legacy_patient_fold_manifest_used") is not False
        ):
            raise HierarchicalPreparationError(
                f"Prepared fold {fold} lacks the frozen patient authority binding"
            )
        inverse = _inverse_node_maps(payload["bundle"])
        enriched = dict(payload)
        for split in ("train", "validation"):
            enriched[f"{split}_batches"] = _enrich_batches(
                payload[f"{split}_batches"],
                split=split,
                heldout_pair_fold=fold,
                predictions=predictions,
                inverse=inverse,
                torch=torch,
            )
        enriched["modality_contract"] = {
            "modality_names": list(MODALITIES),
            "source_scope": "PATIENT_LEVEL_OOF",
            "patient_folds": 5,
            "old_predictions_used": False,
            "missing_values_typed_unavailable": True,
            "outer_pair_blocked": True,
            "outer_pair_fold": fold,
            "validation_pair_fold": (fold + 1) % 5,
            "modality_source_sha256": artifact_sha256(genomic_path),
            "atac_source_sha256": artifact_sha256(atac_path) if atac_path else None,
            "sealed_test_batches_present": False,
            "test_inference_requires_post_winner_lock_materializer": True,
        }
        scope = dict(enriched.get("input_scope", {}))
        scope.update(
            {
                "outer_pair_blocked": True,
                "outer_pair_fold": fold,
                "validation_pair_fold": (fold + 1) % 5,
                "candidate_sampling_budget_per_cancer": 100000,
            }
        )
        enriched["input_scope"] = scope
        destination = output / f"PATIENT_FOLD_{fold}.pt"
        torch.save(enriched, destination)
        records.append(
            {
                "patient_fold": fold,
                "outer_pair_fold": fold,
                "path": str(destination),
                "sha256": artifact_sha256(destination),
                "train_batches": len(enriched["train_batches"]),
                "validation_batches": len(enriched["validation_batches"]),
                "test_batches_present": False,
            }
        )
    fold_contract = hashlib.sha256(
        json.dumps(
            [(row["patient_fold"], row["outer_pair_fold"]) for row in records],
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    manifest = {
        "status": "SUCCESS_TRAIN_VALIDATION_ONLY_TEST_REMAINS_SEALED",
        "analysis_version": "CancerLncAtlas_V3.2_HIERARCHICAL_ROUTED_CANDIDATE",
        "seed": int(seed),
        "modalities": list(MODALITIES),
        "source_scope": "PATIENT_LEVEL_OOF",
        "patient_fold_authority": prepared_patient_binding,
        "graph_variant": expected_graph_variant,
        "same_outer_pair_fold_function_as_external_router": True,
        "candidate_only": True,
        "test_labels_absent_from_training_payload": True,
        "post_winner_lock_sealed_test_materializer_required": True,
        "formal_v32_prepared_artifacts_overwritten": False,
        "genomic_predictions": {"path": str(genomic_path), "sha256": artifact_sha256(genomic_path)},
        "genomic_lineage": {"path": str(genomic_lineage), "sha256": artifact_sha256(genomic_lineage)},
        "atac_predictions": (
            {"path": str(atac_path), "sha256": artifact_sha256(atac_path)} if atac_path else None
        ),
        "atac_lineage": (
            {"path": str(atac_lineage), "sha256": artifact_sha256(atac_lineage)}
            if atac_lineage else None
        ),
        "outer_pair_fold_sha256": fold_contract,
        "records": records,
    }
    manifest_path = output / "PREPARATION_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def prepare_hierarchical_outer_after_winner_lock(
    *,
    hierarchical_prepared_root: str | Path,
    hierarchical_preparation_manifest_path: str | Path,
    sealed_test_manifest_path: str | Path,
    genomic_predictions_path: str | Path,
    genomic_lineage_path: str | Path,
    routing_winner_declaration_path: str | Path,
    routing_winner_declaration_sha256: str,
    hierarchical_validation_success_path: str | Path,
    output_root: str | Path,
    atac_predictions_path: str | Path | None = None,
    atac_lineage_path: str | Path | None = None,
) -> dict[str, Any]:
    """Enrich sealed outer batches only after hierarchical wins validation."""

    import torch

    from .cancer_modality_router import (
        validate_atac_oof_lineage,
        validate_patient_oof_lineage,
    )
    from .routing_validation_winner import WINNER_FORMAT

    prepared_root = Path(hierarchical_prepared_root).resolve()
    prep_manifest_path = Path(hierarchical_preparation_manifest_path).resolve()
    sealed_manifest_path = Path(sealed_test_manifest_path).resolve()
    genomic_path = Path(genomic_predictions_path).resolve()
    genomic_lineage_path = Path(genomic_lineage_path).resolve()
    winner_path = Path(routing_winner_declaration_path).resolve()
    validation_success_path = Path(hierarchical_validation_success_path).resolve()
    atac_path = Path(atac_predictions_path).resolve() if atac_predictions_path else None
    atac_lineage_path = Path(atac_lineage_path).resolve() if atac_lineage_path else None
    if (atac_path is None) != (atac_lineage_path is None):
        raise HierarchicalPreparationError(
            "ATAC predictions and lineage must be supplied together"
        )
    required = (
        prep_manifest_path,
        sealed_manifest_path,
        genomic_path,
        genomic_lineage_path,
        winner_path,
        validation_success_path,
    ) + ((atac_path, atac_lineage_path) if atac_path is not None else ())
    for path in required:
        assert path is not None
        if not path.is_file() or path.stat().st_size <= 0:
            raise HierarchicalPreparationError(f"Missing winner-locked input: {path}")
    expected_winner_sha = str(routing_winner_declaration_sha256).lower()
    if len(expected_winner_sha) != 64 or artifact_sha256(winner_path) != expected_winner_sha:
        raise HierarchicalPreparationError("Routing winner declaration SHA256 drift")
    try:
        winner = json.loads(winner_path.read_text(encoding="utf-8"))
        validation_success = json.loads(
            validation_success_path.read_text(encoding="utf-8")
        )
        prep_manifest = json.loads(prep_manifest_path.read_text(encoding="utf-8"))
        sealed_manifest = json.loads(sealed_manifest_path.read_text(encoding="utf-8"))
        genomic_lineage = json.loads(genomic_lineage_path.read_text(encoding="utf-8"))
        atac_lineage = (
            json.loads(atac_lineage_path.read_text(encoding="utf-8"))
            if atac_lineage_path is not None
            else None
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise HierarchicalPreparationError(
            "Winner-locked hierarchical input receipt is unreadable"
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
        raise HierarchicalPreparationError(
            "Routing winner does not authorize hierarchical outer inference"
        )
    winner_success = winner.get("sources", {}).get("hierarchical_success", {})
    if (
        winner_success.get("sha256") != artifact_sha256(validation_success_path)
        or validation_success.get("status")
        != "PASS_VALIDATION_ONLY_HIERARCHICAL"
        or validation_success.get("outer_test_predictions_written") is not False
        or validation_success.get("outer_test_metrics_computed") is not False
        or validation_success.get("architecture_winner_selected") is not False
    ):
        raise HierarchicalPreparationError(
            "Hierarchical validation success is not bound to the winner"
        )
    if (
        prep_manifest.get("status")
        != "SUCCESS_TRAIN_VALIDATION_ONLY_TEST_REMAINS_SEALED"
        or prep_manifest.get("test_labels_absent_from_training_payload") is not True
        or prep_manifest.get("post_winner_lock_sealed_test_materializer_required")
        is not True
    ):
        raise HierarchicalPreparationError("Hierarchical preparation firewall drift")
    if (
        sealed_manifest.get("status") != "PASS_SEALED_TEST_AUTHORITY_UNOPENED"
        or sealed_manifest.get("payloads_opened_before_winner_lock") is not False
        or sealed_manifest.get("test_labels_absent_from_training_payload") is not True
    ):
        raise HierarchicalPreparationError("Sealed test manifest firewall drift")
    records = sealed_manifest.get("folds")
    if not isinstance(records, list) or len(records) != 5:
        raise HierarchicalPreparationError("Sealed test manifest lacks five folds")
    record_lookup: dict[int, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise HierarchicalPreparationError("Sealed test fold record is invalid")
        fold = int(record.get("patient_fold", -1))
        if fold in record_lookup:
            raise HierarchicalPreparationError("Duplicate sealed test fold record")
        test_record = record.get("test_payload")
        if not isinstance(test_record, Mapping):
            raise HierarchicalPreparationError("Sealed test fold lacks payload record")
        payload_path = Path(str(test_record.get("path", ""))).resolve()
        if (
            not payload_path.is_file()
            or artifact_sha256(payload_path) != test_record.get("sha256")
        ):
            raise HierarchicalPreparationError(
                f"Sealed test payload SHA256 drift: fold {fold}"
            )
        record_lookup[fold] = record
    if set(record_lookup) != set(range(5)):
        raise HierarchicalPreparationError("Sealed test folds are not 0..4")
    validate_patient_oof_lineage(
        genomic_lineage, predictions_sha256=artifact_sha256(genomic_path)
    )
    if atac_path is not None and atac_lineage is not None:
        validate_atac_oof_lineage(
            atac_lineage, predictions_sha256=artifact_sha256(atac_path)
        )
    predictions = _canonical_predictions(
        pd.read_parquet(genomic_path),
        pd.read_parquet(atac_path) if atac_path is not None else None,
    )
    output = Path(output_root).resolve()
    staging = output.with_name(f"{output.name}.partial")
    if output.exists() or staging.exists():
        raise HierarchicalPreparationError(
            f"Winner-locked hierarchical outer preparation refuses reuse: {output}"
        )
    staging.mkdir(parents=True)
    try:
        output_records: list[dict[str, Any]] = []
        for fold in range(5):
            training_payload_path = prepared_root / f"PATIENT_FOLD_{fold}.pt"
            if not training_payload_path.is_file():
                raise HierarchicalPreparationError(
                    f"Missing hierarchical training payload: fold {fold}"
                )
            training_payload = torch.load(
                training_payload_path, map_location="cpu", weights_only=False
            )
            if "test_batches" in training_payload:
                raise HierarchicalPreparationError(
                    f"Hierarchical training payload already contains test: fold {fold}"
                )
            test_record = record_lookup[fold]["test_payload"]
            sealed_payload_path = Path(str(test_record["path"])).resolve()
            sealed_payload = torch.load(
                sealed_payload_path, map_location="cpu", weights_only=False
            )
            if (
                int(sealed_payload.get("patient_fold", -1)) != fold
                or sealed_payload.get("contains_test_labels") is not True
                or sealed_payload.get("contains_train_or_validation_batches") is not False
                or set(sealed_payload).intersection({"train_batches", "validation_batches"})
            ):
                raise HierarchicalPreparationError(
                    f"Sealed test payload contract drift: fold {fold}"
                )
            bundle = training_payload["bundle"]
            inverse = _inverse_node_maps(bundle)
            test_batches = _enrich_batches(
                sealed_payload["test_batches"],
                split="test",
                heldout_pair_fold=fold,
                predictions=predictions,
                inverse=inverse,
                torch=torch,
            )
            output_payload = {
                "sealed_hierarchical_test_format": (
                    "CANCERLNCATLAS_V32_HIERARCHICAL_WINNER_LOCKED_TEST_V1"
                ),
                "patient_fold": fold,
                "formal_graph_variant": prep_manifest.get("graph_variant"),
                "bundle": bundle,
                "feature_dim": training_payload["feature_dim"],
                "legacy_model_config": training_payload["legacy_model_config"],
                "conservation_context_features": training_payload.get(
                    "conservation_context_features", 4
                ),
                "graph": training_payload.get("graph"),
                "test_batches": test_batches,
                "contains_train_or_validation_batches": False,
                "routing_winner_declaration_sha256": expected_winner_sha,
                "outer_inference_authorized_after_winner_lock": True,
                "modality_contract": {
                    "modality_names": list(MODALITIES),
                    "source_scope": "PATIENT_LEVEL_OOF",
                    "outer_pair_fold": fold,
                    "genomic_predictions_sha256": artifact_sha256(genomic_path),
                    "atac_predictions_sha256": (
                        artifact_sha256(atac_path) if atac_path is not None else None
                    ),
                },
            }
            destination = staging / f"PATIENT_FOLD_{fold}.pt"
            torch.save(output_payload, destination)
            output_records.append(
                {
                    "patient_fold": fold,
                    "outer_pair_fold": fold,
                    "path": str(output / destination.name),
                    "sha256": artifact_sha256(destination),
                    "test_batches": len(test_batches),
                    "contains_train_or_validation_batches": False,
                }
            )
        manifest = {
            "status": "PASS_HIERARCHICAL_OUTER_PREPARED_AFTER_WINNER_LOCK",
            "format": "CANCERLNCATLAS_V32_HIERARCHICAL_WINNER_LOCKED_TEST_MANIFEST_V1",
            "routing_winner_declaration": {
                "path": str(winner_path),
                "sha256": expected_winner_sha,
            },
            "hierarchical_preparation_manifest": {
                "path": str(prep_manifest_path),
                "sha256": artifact_sha256(prep_manifest_path),
            },
            "sealed_test_manifest": {
                "path": str(sealed_manifest_path),
                "sha256": artifact_sha256(sealed_manifest_path),
            },
            "genomic_predictions": {
                "path": str(genomic_path),
                "sha256": artifact_sha256(genomic_path),
            },
            "atac_predictions": (
                {"path": str(atac_path), "sha256": artifact_sha256(atac_path)}
                if atac_path is not None
                else None
            ),
            "records": output_records,
            "outer_payloads_opened_after_routing_winner_lock": True,
            "test_metrics_computed": False,
        }
        manifest_path = staging / "WINNER_LOCKED_TEST_MANIFEST.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        staging.replace(output)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = [
    "HierarchicalPreparationError",
    "prepare_hierarchical_candidate_folds",
    "prepare_hierarchical_outer_after_winner_lock",
]
