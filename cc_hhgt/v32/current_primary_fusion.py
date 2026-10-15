"""Bind native validation supervision to current primary predictions, without fitting.

These labels already participated in primary model selection. Pair cross-fitting
on them is fusion development, never independent validation of the whole system.
Sealed test payloads are deliberately not accepted by this adapter.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .current_core_export import sha256, verified_path, validate_checkpoint
from .current_core_prediction import load_current_predictions, candidate_table, prediction_batches
from .evidence_rule_summary import KEYS

FORMAT = "CURRENT_PRIMARY_FUSION_INPUT_BINDING_V1"
SCOPE = "VALIDATION_REUSED_FOR_FUSION_DEVELOPMENT"
LABEL = "validation_proxy_label"
AVAILABLE = "validation_label_available"
REASON = "validation_label_unavailable_reason"


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def extract_validation_labels(payload, candidates, *, fold):
    """Map the actual validation tensors to exact candidates, retaining masks."""
    import torch
    scope = payload.get("input_scope", {})
    if (payload.get("patient_fold") != fold
            or scope.get("candidate_rows_per_split") != len(candidates)
            or scope.get("discovery_features_split") != "train_patients_only"
            or scope.get("replication_labels_splits") != ["validation"]
            or scope.get("held_out_effects_as_features") is not False
            or scope.get("sealed_test_accessed") is not False):
        raise ValueError("Current fusion labels lack native validation-only scope")
    batches = prediction_batches(payload, candidates)
    labels = np.full(len(candidates), np.nan)
    available = np.zeros(len(candidates), dtype=bool)
    for features, batch in zip(batches, payload["validation_batches"]):
        positions = features["candidate_positions"]
        values = batch.get("proxy_label"); mask = batch.get("association_available")
        if (not torch.is_tensor(values) or values.shape != (len(positions),)
                or not torch.is_tensor(mask) or mask.dtype != torch.bool
                or mask.shape != values.shape):
            raise ValueError("Validation supervision requires aligned labels and explicit boolean availability")
        values = values.detach().cpu().numpy().astype(float)
        mask = mask.detach().cpu().numpy()
        if not np.isfinite(values[mask]).all() or not np.isin(values[mask], [0, 1]).all():
            raise ValueError("Available validation proxy labels must be binary and finite")
        # Loss tensors may use zero fillers under a false mask. They are NOT negatives.
        labels[positions] = np.where(mask, values, np.nan)
        available[positions] = mask
    frame = candidates[KEYS].copy()
    frame["patient_fold_id"] = fold  # model's outer fold, not validation patient fold
    frame["validation_patient_fold_id"] = (fold + 1) % 5
    frame[LABEL] = labels; frame[AVAILABLE] = available
    frame[REASON] = np.where(available, None, "NATIVE_ASSOCIATION_UNAVAILABLE_REASON_NOT_SERIALIZED")
    frame["fusion_supervision_scope"] = SCOPE
    return frame


def _native_folds(source, *, expected_core_run_id):
    """Replay native authority/checkpoint validation; no model or optimizer constructed."""
    import torch
    from scripts.export_v32_current_core_embeddings import _validated_payload
    path = verified_path(source, Path.cwd())
    _, binding = load_current_predictions(path, expected_sha=source["sha256"])
    if binding["training_run_id"] != expected_core_run_id:
        raise ValueError("Current primary fusion source belongs to a different core")
    manifest_path = verified_path(binding["input_manifest"], path.parent)
    manifest = _json(manifest_path); root = manifest_path.parent
    candidates = candidate_table(manifest, root)
    for record in sorted(manifest["fold_inputs"], key=lambda r: r["fold"]):
        fold = record["fold"]
        roles = {"config", "input_manifest", "prepared", "checkpoint", "training_receipt"}
        roles |= {k for k in ("graph_overlay", "global_binding_source", "binding_expansion") if k in record}
        paths = {k: verified_path(record[k], root) for k in roles}
        receipt = _json(paths["training_receipt"])
        checkpoint = torch.load(paths["checkpoint"], map_location="cpu", weights_only=False)
        validate_checkpoint(checkpoint, receipt, fold=fold, prepared_sha=record["prepared"]["sha256"],
            config_sha=record["config"]["sha256"], input_manifest_sha=record["input_manifest"]["sha256"])
        candidate_sha = receipt["artifact_hashes"].get("formal_candidate_authority_sha256")
        if candidate_sha is not None and candidate_sha != manifest["candidate_authority"]["sha256"]:
            raise ValueError("Native labels and checkpoint candidate authority differ")
        payload = _validated_payload(paths, record, fold)
        frame = extract_validation_labels(payload, candidates, fold=fold)
        yield frame, {"fold_id": fold, "validation_patient_fold_id": (fold + 1) % 5,
            "prepared_sha256": record["prepared"]["sha256"], "checkpoint_sha256": record["checkpoint"]["sha256"],
            "available_rows": int(frame[AVAILABLE].sum()), "unavailable_rows": int((~frame[AVAILABLE]).sum())}, binding
        del payload, checkpoint


def materialize_primary_fusion_inputs(binding_path, output, *, expected_sha, expected_core_run_id):
    import socket
    if socket.gethostname() != "149":
        raise RuntimeError("Current primary fusion preparation requires host 149")
    source = {"path": str(Path(binding_path).resolve()), "sha256": expected_sha}
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    (output/"LAUNCH_PREFLIGHT.json").write_text(json.dumps({"target_host": "149", "training_started": False,
        "core_run_id": expected_core_run_id, "stage": "NATIVE_VALIDATION_LABEL_BINDING"}))
    folds = []
    for frame, record, primary in _native_folds(source, expected_core_run_id=expected_core_run_id):
        path = output/f"FOLD_{record['fold_id']}.PRIVATE.parquet"
        frame.to_parquet(path, index=False, compression="zstd")
        folds.append({**record, "artifact": {"path": str(path), "sha256": sha256(path)}})
    binding = {"format": FORMAT, "status": "PASS", "target_host": "149", "run_id": expected_core_run_id,
        "primary_source": source, "prediction": primary["prediction"], "folds": folds,
        "supervision_scope": SCOPE, "label_column": LABEL, "availability_column": AVAILABLE,
        "labels_previously_used_for_primary_selection": True, "independent_validation_claimed": False,
        "sealed_test_labels_used": False, "training_started": False,
        "implementation_sha256": sha256(Path(__file__))}
    path = output/"CURRENT_PRIMARY_FUSION_BINDING.json"
    path.write_text(json.dumps(binding, indent=2)+"\n", encoding="utf-8")
    fragment = {"core_run_id": expected_core_run_id, "primary_prediction": binding["prediction"],
        "primary_binding_audit": {"path": str(path), "sha256": sha256(path)}, "primary_folds": folds}
    (output/"FUSION_PRIMARY_INPUTS.json").write_text(json.dumps(fragment, indent=2)+"\n", encoding="utf-8")
    (output/"SUCCESS.json").write_text(json.dumps({"status": "PASS", "binding_sha256": sha256(path), "training_started": False}))
    return path


def load_primary_fusion_inputs(binding_path, *, expected_sha, expected_core_run_id):
    path = verified_path({"path": str(binding_path), "sha256": expected_sha}, Path.cwd())
    binding = _json(path)
    fixed = {"format": FORMAT, "status": "PASS", "target_host": "149", "run_id": expected_core_run_id,
        "supervision_scope": SCOPE, "label_column": LABEL, "availability_column": AVAILABLE,
        "labels_previously_used_for_primary_selection": True, "independent_validation_claimed": False,
        "sealed_test_labels_used": False, "training_started": False}
    if any(binding.get(k) != v or type(binding.get(k)) is not type(v) for k, v in fixed.items()):
        raise ValueError("Invalid current primary fusion supervision binding")
    if binding.get("implementation_sha256") != sha256(Path(__file__)):
        raise ValueError("Native validation label replay requires its recorded implementation")
    success = _json(path.parent/"SUCCESS.json")
    if success.get("status") != "PASS" or success.get("binding_sha256") != expected_sha:
        raise ValueError("Current primary fusion preparation is incomplete")
    records = binding["folds"]
    if len(records) != 5 or {r["fold_id"] for r in records} != set(range(5)):
        raise ValueError("Current primary fusion requires five native validation views")
    for expected, audit, primary in _native_folds(binding["primary_source"], expected_core_run_id=expected_core_run_id):
        item = next(r for r in records if r["fold_id"] == audit["fold_id"])
        if any(item.get(k) != v for k, v in audit.items()) or binding["prediction"] != primary["prediction"]:
            raise ValueError("Current fusion label source identity or coverage drifted")
        actual = pd.read_parquet(verified_path(item["artifact"], path.parent))
        try:
            pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        except AssertionError as error:
            raise ValueError("Current fusion labels differ from native validation replay") from error
    return binding
