"""Post-winner-lock inference for the untouched V3.2 outer test folds.

Training payloads deliberately contain only ``train_batches`` and
``validation_batches``.  Test labels live in a separate sealed authority and
are deserialized only after a validation-only G0/G1/G2 winner declaration and
all five selected checkpoints have passed their physical SHA-256 gates.

The resulting files are private comparison inputs.  This module never trains,
changes a checkpoint, chooses a winner, or emits a public release marker.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .multimodal_fusion import TARGET_KEYS, pair_blocked_fold
from .patient_fold_authority import (
    FROZEN_V32_RECEIPT_SHA256,
    FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256,
    validate_frozen_v32_patient_fold_binding,
    validate_frozen_v32_patient_fold_payload_binding,
    validate_frozen_v32_prepared_fold_binding,
)
from .training import CHECKPOINT_FORMAT, PREPARED_FORMAT


N_FOLDS = 5
GRAPH_VARIANTS = ("G0", "G1", "G2")
PAIR_FOLD_COLUMN = "fusion_pair_fold"
WINNER_DECLARATION_FORMAT = "CC_HHGT_V3_2_G012_VALIDATION_WINNER_LOCK_V1"
SEALED_TEST_MANIFEST_FORMAT = "CC_HHGT_V3_2_SEALED_TEST_AUTHORITY_V1"
SEALED_TEST_PAYLOAD_FORMAT = "CC_HHGT_V3_2_SEALED_TEST_FOLD_V1"
LINEAGE_FORMAT = "CC_HHGT_V3_2_WINNER_LOCK_SEALED_TEST_LINEAGE_V1"
AUDIT_FORMAT = "CC_HHGT_V3_2_WINNER_LOCK_SEALED_TEST_AUDIT_V1"
ACCEPTANCE_FORMAT = "CC_HHGT_V3_2_WINNER_LOCK_SEALED_TEST_ACCEPTANCE_V1"
ACCEPTANCE_STATUS = "PASS_WINNER_LOCK_SEALED_TEST_INFERENCE"
PRIMARY_COLUMNS = (
    *TARGET_KEYS,
    "primary_probability",
    "fusion_target",
    PAIR_FOLD_COLUMN,
    "source_patient_fold",
    "source_split",
)
LOGIT_COLUMNS = (
    *TARGET_KEYS,
    PAIR_FOLD_COLUMN,
    "source_patient_fold",
    "base_logit",
    "final_logit",
    "direction_logit",
    "primary_probability",
    "fusion_target",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class SealedTestInferenceError(RuntimeError):
    """Raised before a sealed test product can be accepted."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha(value: object, label: str) -> str:
    candidate = str(value or "").lower()
    if not _SHA256.fullmatch(candidate):
        raise SealedTestInferenceError(f"{label} is not a SHA-256")
    return candidate


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise SealedTestInferenceError(f"Missing or unsafe {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SealedTestInferenceError(f"Invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise SealedTestInferenceError(f"{label} must be a JSON object")
    return value


def _declared_file(
    declaration: Mapping[str, Any], *, relative_to: Path, label: str
) -> tuple[Path, str]:
    raw = declaration.get("path")
    if not isinstance(raw, str) or not raw.strip():
        raise SealedTestInferenceError(f"{label} lacks a path")
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = relative_to / candidate
    if candidate.is_symlink():
        raise SealedTestInferenceError(f"{label} may not be a symlink")
    candidate = candidate.resolve()
    if not candidate.is_file() or candidate.stat().st_size <= 0:
        raise SealedTestInferenceError(f"Missing {label}: {candidate}")
    expected = _require_sha(declaration.get("sha256"), f"{label} SHA-256")
    observed = sha256_file(candidate)
    if observed != expected:
        raise SealedTestInferenceError(
            f"{label} SHA-256 drift: {observed} != {expected}"
        )
    return candidate, expected


def _records_by_fold(value: object, label: str) -> dict[int, Mapping[str, Any]]:
    if not isinstance(value, list) or len(value) != N_FOLDS:
        raise SealedTestInferenceError(f"{label} must contain five records")
    result: dict[int, Mapping[str, Any]] = {}
    for item in value:
        if not isinstance(item, Mapping):
            raise SealedTestInferenceError(f"{label} contains a non-object record")
        try:
            fold = int(item.get("patient_fold", -1))
        except (TypeError, ValueError) as exc:
            raise SealedTestInferenceError(f"{label} has an invalid fold") from exc
        if fold in result or fold not in range(N_FOLDS):
            raise SealedTestInferenceError(f"{label} fold inventory is invalid")
        result[fold] = item
    if set(result) != set(range(N_FOLDS)):
        raise SealedTestInferenceError(f"{label} does not cover folds 0..4")
    return result


def _authority_identity(payload: Mapping[str, Any], label: str) -> None:
    expected = {
        "sample_patient_fold_map_sha256": FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256,
        "authority_receipt_sha256": FROZEN_V32_RECEIPT_SHA256,
    }
    observed = {key: payload.get(key) for key in expected}
    if observed != expected:
        raise SealedTestInferenceError(f"{label} patient authority SHA drift: {observed}")


def _validate_winner_declaration(
    payload: Mapping[str, Any], *, expected_graph_variant: str
) -> tuple[dict[int, Mapping[str, Any]], dict[int, Mapping[str, Any]]]:
    required = {
        "format": WINNER_DECLARATION_FORMAT,
        "status": "PASS_VALIDATION_ONLY_WINNER_LOCK",
        "compared_graph_variants": list(GRAPH_VARIANTS),
        "winner_id": expected_graph_variant,
        "graph_variant": expected_graph_variant,
        "selection_scope": "VALIDATION_ONLY",
        "heldout_test_metrics_used_for_selection": False,
        "test_inputs_opened_before_winner_lock": False,
        "winner_or_checkpoint_changed_by_test": False,
        "declaration_locked": True,
        "old_checkpoint_loaded": False,
        "old_predictions_used": False,
    }
    drift = {
        key: {"observed": payload.get(key), "required": expected}
        for key, expected in required.items()
        if payload.get(key) != expected
    }
    if drift:
        raise SealedTestInferenceError(f"Winner declaration is not locked: {drift}")
    _authority_identity(payload, "winner declaration")
    _require_sha(payload.get("candidate_authority_sha256"), "candidate authority")
    pair_seed = payload.get("pair_fold_seed")
    if not isinstance(pair_seed, int):
        raise SealedTestInferenceError("Winner declaration lacks an integer pair-fold seed")
    prepared = _records_by_fold(payload.get("prepared_folds"), "prepared_folds")
    checkpoints = _records_by_fold(payload.get("checkpoints"), "checkpoints")
    checkpoint_hashes = []
    for fold in range(N_FOLDS):
        p = prepared[fold]
        c = checkpoints[fold]
        if p.get("graph_variant") != expected_graph_variant:
            raise SealedTestInferenceError(f"Prepared fold {fold} graph variant drift")
        if c.get("graph_variant") != expected_graph_variant:
            raise SealedTestInferenceError(f"Checkpoint fold {fold} graph variant drift")
        prepared_sha = _require_sha(p.get("sha256"), f"prepared fold {fold}")
        if c.get("prepared_fold_sha256") != prepared_sha:
            raise SealedTestInferenceError(f"Checkpoint fold {fold} prepared lineage drift")
        checkpoint_hashes.append(
            _require_sha(c.get("sha256"), f"checkpoint fold {fold}")
        )
        if c.get("selected_using") != "VALIDATION_ONLY":
            raise SealedTestInferenceError(
                f"Checkpoint fold {fold} was not selected using validation only"
            )
        if c.get("heldout_test_metrics_used") is not False:
            raise SealedTestInferenceError(
                f"Checkpoint fold {fold} used held-out test metrics"
            )
        training_hashes = c.get("training_artifact_hashes")
        if not isinstance(training_hashes, Mapping) or not training_hashes:
            raise SealedTestInferenceError(
                f"Checkpoint fold {fold} lacks its guarded training authority"
            )
    if len(set(checkpoint_hashes)) != N_FOLDS:
        raise SealedTestInferenceError("The five winner checkpoints are not physically distinct")
    return prepared, checkpoints


def _contains_training_test_leakage(payload: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    forbidden_top = {
        "test_batches",
        "test_batch",
        "heldout_test_labels",
        "sealed_test_labels",
        "test_logits",
    }
    for key in payload:
        if str(key).lower() in forbidden_top:
            failures.append(str(key))
    for split in ("train_batches", "validation_batches"):
        batches = payload.get(split)
        if not isinstance(batches, Sequence) or not batches:
            failures.append(f"{split}:missing")
            continue
        for index, raw in enumerate(batches):
            if not isinstance(raw, Mapping):
                failures.append(f"{split}[{index}]:malformed")
                continue
            candidate = raw.get("candidate_batch")
            if not isinstance(candidate, Mapping):
                failures.append(f"{split}[{index}].candidate_batch:missing")
                continue
            exposed = {
                "test_effect",
                "test_label",
                "heldout_test_label",
                "replication_effect",
                "replication_direction_label",
            } & set(map(str, candidate))
            failures.extend(f"{split}[{index}].candidate_batch.{key}" for key in exposed)
    scope = payload.get("input_scope")
    if not isinstance(scope, Mapping):
        failures.append("input_scope:missing")
    else:
        if scope.get("sealed_test_accessed") is not False:
            failures.append("input_scope.sealed_test_accessed")
        if set(scope.get("replication_labels_splits", [])) != {"validation"}:
            failures.append("input_scope.replication_labels_splits")
        if scope.get("held_out_effects_as_features") is not False:
            failures.append("input_scope.held_out_effects_as_features")
    label = payload.get("label_contract")
    if not isinstance(label, Mapping):
        failures.append("label_contract:missing")
    else:
        for key in (
            "test_labels_in_training_payload",
            "test_logits_in_training_payload",
            "test_metrics_computed_before_winner_lock",
        ):
            if label.get(key) is not False:
                failures.append(f"label_contract.{key}")
    return failures


def _validate_prepared_payload(
    payload: Mapping[str, Any], *, fold: int, variant: str
) -> None:
    if (
        payload.get("prepared_format") != PREPARED_FORMAT
        or int(payload.get("patient_fold", -1)) != fold
        or payload.get("formal_graph_variant") != variant
        or payload.get("contains_optimizer_state") is not False
        or payload.get("contains_trained_parameters") is not False
    ):
        raise SealedTestInferenceError(f"Prepared fold {fold} identity/format drift")
    try:
        validate_frozen_v32_patient_fold_payload_binding(
            payload.get("patient_fold_authority")
        )
    except Exception as exc:
        raise SealedTestInferenceError(
            f"Prepared fold {fold} patient authority binding drift"
        ) from exc
    try:
        from .formal_graph_authority import validate_formal_graph_payload_binding

        validate_formal_graph_payload_binding(
            payload.get("formal_graph_authority"),
            outer_fold=fold,
            variant=variant,
            bundle=payload.get("bundle"),
        )
    except Exception as exc:
        raise SealedTestInferenceError(
            f"Prepared fold {fold} graph authority lineage drift"
        ) from exc
    leakage = _contains_training_test_leakage(payload)
    if leakage:
        raise SealedTestInferenceError(
            f"Prepared fold {fold} crosses the sealed-test firewall: {leakage}"
        )


def _validate_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    fold: int,
    record: Mapping[str, Any],
    prepared: Mapping[str, Any],
) -> None:
    if checkpoint.get("checkpoint_format") != CHECKPOINT_FORMAT:
        raise SealedTestInferenceError(f"Checkpoint fold {fold} format drift")
    if "model_state" not in checkpoint:
        raise SealedTestInferenceError(f"Checkpoint fold {fold} lacks model_state")
    prepared_input_authority = prepared.get(
        "input_authority_hashes", prepared.get("artifact_hashes")
    )
    if checkpoint.get("input_authority_hashes") != prepared_input_authority:
        raise SealedTestInferenceError(
            f"Checkpoint fold {fold} input-authority hashes differ from prepared"
        )
    if checkpoint.get("artifact_hashes") != record.get("training_artifact_hashes"):
        raise SealedTestInferenceError(
            f"Checkpoint fold {fold} guarded training-authority hashes drift"
        )
    architecture = record.get("architecture_id")
    if checkpoint.get("architecture_id") != architecture:
        raise SealedTestInferenceError(f"Checkpoint fold {fold} architecture drift")
    if architecture not in {
        "HHGT_FORMAL_CORE_EXTERNAL_ROUTER",
        "HHGT_HIERARCHICAL_EVIDENCE_GATE_CANDIDATE",
    }:
        raise SealedTestInferenceError(f"Checkpoint fold {fold} architecture is unsupported")


def _validate_variant_root(prepared_root: Path, expected: str) -> None:
    marker_path = prepared_root / "FORMAL_GRAPH_VARIANT.json"
    marker = _read_json(marker_path, "formal graph variant marker")
    if (
        marker.get("format") != "CANCERLNCATLAS_V32_FORMAL_GRAPH_VARIANT_ROOT_V1"
        or marker.get("variant") != expected
        or marker.get("legacy_root_fold_payloads_allowed") is not False
    ):
        raise SealedTestInferenceError("Prepared root is not the explicit selected G0/G1/G2 arm")


def _candidate_contract(path: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq

    source = pq.ParquetFile(path)
    if missing := sorted(set(TARGET_KEYS) - set(source.schema_arrow.names)):
        raise SealedTestInferenceError(f"Candidate authority lacks exact keys: {missing}")
    digest = hashlib.sha256()
    previous: tuple[str, str, str] | None = None
    counts: Counter[str] = Counter()
    rows = 0
    for batch in source.iter_batches(columns=list(TARGET_KEYS), batch_size=100_000):
        for raw in batch.to_pandas().itertuples(index=False, name=None):
            key = (str(raw[0]).upper(), str(raw[1]), str(raw[2]))
            if not all(key) or key[0] != str(raw[0]):
                raise SealedTestInferenceError("Candidate authority has a non-canonical key")
            if previous is not None and key <= previous:
                raise SealedTestInferenceError("Candidate authority key order/uniqueness drift")
            previous = key
            digest.update(("\t".join(key) + "\n").encode("utf-8"))
            counts[key[0]] += 1
            rows += 1
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": rows,
        "ordered_exact_key_sha256": digest.hexdigest(),
        "cancer_counts": dict(sorted(counts.items())),
    }


def _canonical_frame(frame: pd.DataFrame, *, label: str) -> pd.DataFrame:
    required = set(TARGET_KEYS) | {
        "base_logit",
        "final_logit",
        "direction_logit",
        "fusion_target",
    }
    if missing := sorted(required - set(frame.columns)):
        raise SealedTestInferenceError(f"{label} lacks columns: {missing}")
    result = frame[list(required)].copy()
    result["cancer_id"] = result.cancer_id.astype(str).str.upper()
    for column in ("lncrna_id", "pathway_id"):
        result[column] = result[column].astype(str)
    result = result.sort_values(list(TARGET_KEYS), kind="stable").reset_index(drop=True)
    if result[list(TARGET_KEYS)].duplicated().any():
        raise SealedTestInferenceError(f"{label} duplicates exact keys")
    for column in ("base_logit", "final_logit", "direction_logit"):
        result[column] = pd.to_numeric(result[column], errors="raise").astype(np.float64)
        if not np.isfinite(result[column].to_numpy()).all():
            raise SealedTestInferenceError(f"{label} has non-finite {column}")
    target = pd.to_numeric(result.fusion_target, errors="raise").to_numpy(float)
    if not np.isfinite(target).all() or not np.isin(target, [0.0, 1.0]).all():
        raise SealedTestInferenceError(f"{label} has a non-binary test target")
    result["fusion_target"] = target.astype(np.uint8)
    return result


def _ordered_key_digest(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in frame[list(TARGET_KEYS)].itertuples(index=False, name=None):
        digest.update(("\t".join(map(str, row)) + "\n").encode("utf-8"))
    return digest.hexdigest()


def _validate_test_patient_keys(path: Path, authority: pd.DataFrame, fold: int) -> int:
    observed = pd.read_csv(
        path,
        sep="\t",
        dtype={"cancer_id": str, "sample_id": str, "patient_id": str},
    )
    columns = ("cancer_id", "sample_id", "patient_id", "patient_fold_id", "fold_seed")
    if missing := sorted(set(columns) - set(observed.columns)):
        raise SealedTestInferenceError(f"Fold {fold} test patient keys lack: {missing}")
    expected = authority.loc[authority.patient_fold_id.eq(fold), list(columns)].copy()
    observed = observed[list(columns)].copy()
    for frame in (expected, observed):
        frame["cancer_id"] = frame.cancer_id.astype(str).str.upper()
        for column in ("sample_id", "patient_id"):
            frame[column] = frame[column].astype(str)
        frame["patient_fold_id"] = pd.to_numeric(
            frame.patient_fold_id, errors="raise"
        ).astype(int)
        frame["fold_seed"] = pd.to_numeric(frame.fold_seed, errors="raise").astype(int)
        frame.sort_values(list(columns), kind="stable", inplace=True)
        frame.reset_index(drop=True, inplace=True)
    if not observed.equals(expected):
        raise SealedTestInferenceError(
            f"Fold {fold} sealed patient keys are not the exact frozen test partition"
        )
    return int(len(observed))


def _numpy(value: Any, label: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    try:
        return np.asarray(value)
    except Exception as exc:  # pragma: no cover - defensive tensor boundary
        raise SealedTestInferenceError(f"Cannot convert {label} to an array") from exc


def _inverse_maps(bundle: Any) -> dict[str, dict[int, str]]:
    maps = getattr(bundle, "node_maps", None)
    if not isinstance(maps, Mapping):
        raise SealedTestInferenceError("Prepared graph bundle lacks node_maps")
    result: dict[str, dict[int, str]] = {}
    for kind in ("cancer", "lncRNA", "pathway"):
        if kind not in maps or not isinstance(maps[kind], Mapping):
            raise SealedTestInferenceError(f"Prepared graph lacks {kind} node map")
        inverse = {int(index): str(key) for key, index in maps[kind].items()}
        if len(inverse) != len(maps[kind]):
            raise SealedTestInferenceError(f"Prepared graph {kind} node map is not one-to-one")
        result[kind] = inverse
    return result


def _move(value: Any, device: str) -> Any:
    if hasattr(value, "to"):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _move(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    return value


def _default_payload_loader(path: Path) -> Mapping[str, Any]:
    import torch

    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, Mapping):
        raise SealedTestInferenceError(f"Serialized artifact is not a mapping: {path}")
    return value


def _load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        value = json.loads(text)
    else:
        import yaml

        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise SealedTestInferenceError("Winner config must be a mapping")
    return value


def _default_fold_inferencer(
    *,
    prepared: Mapping[str, Any],
    sealed: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    checkpoint_record: Mapping[str, Any],
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Run read-only inference over every registered runtime graph chunk."""

    import torch

    from ..gnn import build_model, move_graph
    from .model import build_v32_cc_hhgt_residual, build_v32_hierarchical_evidence_hhgt
    from .training import _runtime_chunk_contract, _runtime_graph_for_chunk

    bundle = prepared["bundle"]
    primary = config.get("primary_model", {})
    encoder = build_model(
        "cc_hhgt",
        bundle,
        int(prepared["feature_dim"]),
        prepared["legacy_model_config"],
    )
    architecture = checkpoint_record["architecture_id"]
    if architecture == "HHGT_HIERARCHICAL_EVIDENCE_GATE_CANDIDATE":
        integration = primary.get("evidence_integration", {})
        model = build_v32_hierarchical_evidence_hhgt(
            encoder,
            hidden_channels=int(primary.get("hidden_channels", 96)),
            context_features=int(prepared.get("conservation_context_features", 4)),
            modality_names=tuple(
                integration.get("modalities", ("mutation", "cnv", "atac"))
            ),
            dropout=float(primary.get("dropout", 0.20)),
        )
    else:
        model = build_v32_cc_hhgt_residual(
            encoder,
            hidden_channels=int(primary.get("hidden_channels", 96)),
            context_features=int(prepared.get("conservation_context_features", 4)),
            dropout=float(primary.get("dropout", 0.20)),
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    batches = sealed["test_batches"]
    chunks, weights = _runtime_chunk_contract(bundle)
    has_schedule = getattr(bundle, "runtime_schedule", None) is not None
    fixed_graph = prepared.get("graph")
    if has_schedule and fixed_graph is not None:
        raise SealedTestInferenceError("A fixed graph overrides the runtime schedule")
    inverse = _inverse_maps(bundle)
    output_by_chunk: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    with torch.no_grad():
        active_chunks = chunks if has_schedule else (0,)
        for chunk in active_chunks:
            graph = (
                _runtime_graph_for_chunk(bundle, chunk, device=device)
                if has_schedule
                else (
                    move_graph(fixed_graph, device, "cc_hhgt")
                    if fixed_graph is not None
                    else move_graph(bundle, device, "cc_hhgt")
                )
            )
            encoded = model.encoder.encode(graph)
            chunk_rows: list[tuple[np.ndarray, np.ndarray]] = []
            for raw in batches:
                batch = _move(raw, device)
                result = model(
                    graph,
                    batch["candidate_batch"],
                    batch["base_logit"],
                    batch["conservation_context"],
                    batch.get("graph_available"),
                    admitted=True,
                    encoded=encoded,
                )
                chunk_rows.append(
                    (
                        result["final_logit"].detach().cpu().numpy().astype(np.float64),
                        result["direction_logit"].detach().cpu().numpy().astype(np.float64),
                    )
                )
            output_by_chunk[int(chunk)] = chunk_rows
    rows: list[pd.DataFrame] = []
    for index, raw in enumerate(batches):
        candidate = raw.get("candidate_batch")
        if not isinstance(candidate, Mapping):
            raise SealedTestInferenceError(f"Sealed batch {index} lacks candidate_batch")
        indices = {
            key: _numpy(candidate.get(key), f"candidate_batch.{key}").astype(np.int64)
            for key in ("c", "l", "p")
        }
        size = len(indices["c"])
        if any(len(indices[key]) != size for key in ("l", "p")):
            raise SealedTestInferenceError("Sealed candidate indices differ in length")
        try:
            cancer = [inverse["cancer"][int(value)].upper() for value in indices["c"]]
            lnc = [inverse["lncRNA"][int(value)] for value in indices["l"]]
            pathway = [inverse["pathway"][int(value)] for value in indices["p"]]
        except KeyError as exc:
            raise SealedTestInferenceError("Sealed batch references an absent graph node") from exc
        if has_schedule:
            final = sum(
                output_by_chunk[int(chunk)][index][0] * float(weights[int(chunk)])
                for chunk in chunks
            )
            direction = sum(
                output_by_chunk[int(chunk)][index][1] * float(weights[int(chunk)])
                for chunk in chunks
            )
        else:
            final, direction = output_by_chunk[0][index]
        base = _numpy(raw.get("base_logit"), "base_logit").reshape(-1)
        target = _numpy(raw.get("proxy_label"), "proxy_label").reshape(-1)
        if not (len(base) == len(target) == len(final) == size):
            raise SealedTestInferenceError("Sealed inference arrays differ in length")
        rows.append(
            pd.DataFrame(
                {
                    "cancer_id": cancer,
                    "lncrna_id": lnc,
                    "pathway_id": pathway,
                    "base_logit": base,
                    "final_logit": final,
                    "direction_logit": direction,
                    "fusion_target": target,
                }
            )
        )
    if not rows:
        raise SealedTestInferenceError("Sealed test inference produced no rows")
    return pd.concat(rows, ignore_index=True)


def _validate_sealed_payload(
    payload: Mapping[str, Any],
    *,
    fold: int,
    variant: str,
    candidate_sha256: str,
    training_payload_sha256: str,
) -> None:
    required = {
        "sealed_test_format": SEALED_TEST_PAYLOAD_FORMAT,
        "patient_fold": fold,
        "formal_graph_variant": variant,
        "candidate_authority_sha256": candidate_sha256,
        "training_payload_sha256": training_payload_sha256,
        "contains_test_labels": True,
        "contains_train_or_validation_batches": False,
        "winner_lock_required_before_deserialization": True,
        "accessed_before_winner_lock": False,
    }
    drift = {
        key: {"observed": payload.get(key), "required": value}
        for key, value in required.items()
        if payload.get(key) != value
    }
    if drift:
        raise SealedTestInferenceError(f"Sealed fold {fold} contract drift: {drift}")
    try:
        validate_frozen_v32_patient_fold_payload_binding(
            payload.get("patient_fold_authority")
        )
    except Exception as exc:
        raise SealedTestInferenceError(
            f"Sealed fold {fold} patient authority binding drift"
        ) from exc
    if "train_batches" in payload or "validation_batches" in payload:
        raise SealedTestInferenceError(f"Sealed fold {fold} mixes training data")
    batches = payload.get("test_batches")
    if not isinstance(batches, Sequence) or not batches:
        raise SealedTestInferenceError(f"Sealed fold {fold} lacks test_batches")
    for index, batch in enumerate(batches):
        if not isinstance(batch, Mapping):
            raise SealedTestInferenceError(f"Sealed fold {fold} batch {index} is malformed")
        required_batch = {
            "candidate_batch",
            "base_logit",
            "conservation_context",
            "proxy_label",
        }
        if missing := sorted(required_batch - set(batch)):
            raise SealedTestInferenceError(
                f"Sealed fold {fold} batch {index} lacks: {missing}"
            )


def _artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _make_read_only(path: Path) -> None:
    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def materialize_winner_locked_sealed_test(
    *,
    prepared_root: str | Path,
    winner_declaration_path: str | Path,
    winner_declaration_sha256: str,
    sealed_test_manifest_path: str | Path,
    patient_folds_path: str | Path,
    patient_fold_authority_receipt_path: str | Path,
    output_root: str | Path,
    expected_graph_variant: str = "G2",
    payload_loader: Callable[[Path], Mapping[str, Any]] | None = None,
    fold_inferencer: Callable[..., pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Materialize one immutable OOF test frame after winner lock.

    The optional callbacks exist for focused contract tests.  Formal callers
    must omit them, causing exact Torch checkpoints and batches to be loaded.
    """

    if expected_graph_variant not in GRAPH_VARIANTS:
        raise SealedTestInferenceError("Expected graph variant must be G0, G1 or G2")
    output = Path(output_root).resolve()
    staging = output.with_name(f".{output.name}.staging")
    if output.exists() or staging.exists():
        raise SealedTestInferenceError(f"Sealed-test materializer refuses output reuse: {output}")
    prepared_root = Path(prepared_root).resolve()
    declaration_path = Path(winner_declaration_path).resolve()
    sealed_manifest_path = Path(sealed_test_manifest_path).resolve()
    patient_folds = Path(patient_folds_path).resolve()
    patient_receipt = Path(patient_fold_authority_receipt_path).resolve()

    # Winner lock validation intentionally precedes every sealed payload load.
    expected_declaration_sha = _require_sha(
        winner_declaration_sha256, "winner declaration caller pin"
    )
    if sha256_file(declaration_path) != expected_declaration_sha:
        raise SealedTestInferenceError("Winner declaration caller-pinned SHA drift")
    declaration = _read_json(declaration_path, "winner declaration")
    prepared_records, checkpoint_records = _validate_winner_declaration(
        declaration, expected_graph_variant=expected_graph_variant
    )
    authority_audit = validate_frozen_v32_patient_fold_binding(
        patient_folds, patient_receipt
    )
    validate_frozen_v32_prepared_fold_binding(prepared_root)
    _validate_variant_root(prepared_root, expected_graph_variant)
    loader = payload_loader or _default_payload_loader

    prepared_payloads: dict[int, Mapping[str, Any]] = {}
    checkpoint_payloads: dict[int, Mapping[str, Any]] = {}
    prepared_lineage: list[dict[str, Any]] = []
    checkpoint_lineage: list[dict[str, Any]] = []
    for fold in range(N_FOLDS):
        prepared_path, prepared_sha = _declared_file(
            prepared_records[fold], relative_to=declaration_path.parent,
            label=f"prepared fold {fold}",
        )
        expected_prepared = (prepared_root / f"PATIENT_FOLD_{fold}.pt").resolve()
        if prepared_path != expected_prepared:
            raise SealedTestInferenceError(f"Prepared fold {fold} path/variant drift")
        payload = loader(prepared_path)
        if not isinstance(payload, Mapping):
            raise SealedTestInferenceError(f"Prepared fold {fold} is not a mapping")
        _validate_prepared_payload(payload, fold=fold, variant=expected_graph_variant)
        checkpoint_path, checkpoint_sha = _declared_file(
            checkpoint_records[fold], relative_to=declaration_path.parent,
            label=f"checkpoint fold {fold}",
        )
        checkpoint = loader(checkpoint_path)
        if not isinstance(checkpoint, Mapping):
            raise SealedTestInferenceError(f"Checkpoint fold {fold} is not a mapping")
        _validate_checkpoint(
            checkpoint,
            fold=fold,
            record=checkpoint_records[fold],
            prepared=payload,
        )
        prepared_payloads[fold] = payload
        checkpoint_payloads[fold] = checkpoint
        prepared_lineage.append(
            {"patient_fold": fold, "path": str(prepared_path), "sha256": prepared_sha}
        )
        checkpoint_lineage.append(
            {
                "patient_fold": fold,
                "path": str(checkpoint_path),
                "sha256": checkpoint_sha,
                "architecture_id": checkpoint_records[fold]["architecture_id"],
                "selected_using": "VALIDATION_ONLY",
            }
        )

    # Only after all winner/checkpoint gates have passed may the sealed
    # manifest and payloads become data inputs.
    sealed_manifest = _read_json(sealed_manifest_path, "sealed test manifest")
    required_manifest = {
        "format": SEALED_TEST_MANIFEST_FORMAT,
        "status": "PASS_SEALED_TEST_AUTHORITY_UNOPENED",
        "graph_variant": expected_graph_variant,
        "winner_lock_required_before_payload_deserialization": True,
        "payloads_opened_before_winner_lock": False,
        "test_labels_absent_from_training_payload": True,
        "old_fold_outputs_used": False,
        "pair_fold_seed": declaration["pair_fold_seed"],
    }
    drift = {
        key: {"observed": sealed_manifest.get(key), "required": value}
        for key, value in required_manifest.items()
        if sealed_manifest.get(key) != value
    }
    if drift:
        raise SealedTestInferenceError(f"Sealed test manifest drift: {drift}")
    _authority_identity(sealed_manifest, "sealed test manifest")
    if (
        sealed_manifest.get("winner_declaration_sha256")
        != expected_declaration_sha
    ):
        raise SealedTestInferenceError("Sealed manifest winner declaration SHA drift")
    candidate_path, candidate_sha = _declared_file(
        sealed_manifest.get("candidate_authority", {}),
        relative_to=sealed_manifest_path.parent,
        label="candidate authority",
    )
    if (
        candidate_sha != declaration["candidate_authority_sha256"]
        or sealed_manifest.get("candidate_authority_sha256") != candidate_sha
    ):
        raise SealedTestInferenceError("Candidate authority lineage differs across locks")
    candidate = _candidate_contract(candidate_path)
    sealed_records = _records_by_fold(sealed_manifest.get("folds"), "sealed folds")
    authority_frame = pd.read_csv(
        patient_folds,
        sep="\t",
        dtype={"cancer_id": str, "sample_id": str, "patient_id": str},
    )
    authority_frame["patient_fold_id"] = pd.to_numeric(
        authority_frame.patient_fold_id, errors="raise"
    ).astype(int)
    config_path, config_sha = _declared_file(
        declaration.get("config", {}),
        relative_to=declaration_path.parent,
        label="winner config",
    )
    config = _load_config(config_path)
    infer = fold_inferencer or _default_fold_inferencer

    selected_frames: list[pd.DataFrame] = []
    sealed_lineage: list[dict[str, Any]] = []
    pair_seed = int(declaration["pair_fold_seed"])
    for fold in range(N_FOLDS):
        record = sealed_records[fold]
        if record.get("training_payload_sha256") != prepared_lineage[fold]["sha256"]:
            raise SealedTestInferenceError(f"Sealed fold {fold} training lineage drift")
        test_keys_path, test_keys_sha = _declared_file(
            record.get("test_patient_keys", {}),
            relative_to=sealed_manifest_path.parent,
            label=f"sealed test patient keys fold {fold}",
        )
        test_patient_rows = _validate_test_patient_keys(
            test_keys_path, authority_frame, fold
        )
        payload_path, payload_sha = _declared_file(
            record.get("test_payload", {}),
            relative_to=sealed_manifest_path.parent,
            label=f"sealed test payload fold {fold}",
        )
        sealed = loader(payload_path)
        if not isinstance(sealed, Mapping):
            raise SealedTestInferenceError(f"Sealed test payload fold {fold} is not a mapping")
        _validate_sealed_payload(
            sealed,
            fold=fold,
            variant=expected_graph_variant,
            candidate_sha256=candidate_sha,
            training_payload_sha256=prepared_lineage[fold]["sha256"],
        )
        inferred = _canonical_frame(
            infer(
                prepared=prepared_payloads[fold],
                sealed=sealed,
                checkpoint=checkpoint_payloads[fold],
                checkpoint_record=checkpoint_records[fold],
                config=config,
            ),
            label=f"fold {fold} inference",
        )
        if (
            len(inferred) != candidate["rows"]
            or _ordered_key_digest(inferred) != candidate["ordered_exact_key_sha256"]
        ):
            raise SealedTestInferenceError(
                f"Fold {fold} inference does not cover the exact candidate authority"
            )
        folds = np.fromiter(
            (
                pair_blocked_fold(lnc, pathway, seed=pair_seed)
                for lnc, pathway in inferred[["lncrna_id", "pathway_id"]].itertuples(
                    index=False, name=None
                )
            ),
            dtype=np.int8,
            count=len(inferred),
        )
        keep = folds == fold
        if not keep.any():
            raise SealedTestInferenceError(f"Fold {fold} has no outer pair-test rows")
        selected = inferred.loc[keep].copy()
        selected[PAIR_FOLD_COLUMN] = np.int8(fold)
        selected["source_patient_fold"] = np.int8(fold)
        selected["primary_probability"] = (
            1.0 / (1.0 + np.exp(-selected.final_logit.to_numpy(np.float64)))
        ).astype(np.float32)
        selected_frames.append(selected)
        sealed_lineage.append(
            {
                "patient_fold": fold,
                "test_payload": {
                    "path": str(payload_path),
                    "sha256": payload_sha,
                },
                "test_patient_keys": {
                    "path": str(test_keys_path),
                    "sha256": test_keys_sha,
                    "rows": test_patient_rows,
                },
                "full_candidate_rows_inferred": int(len(inferred)),
                "outer_pair_test_rows_selected": int(keep.sum()),
                "test_payload_deserialized_after_winner_lock": True,
            }
        )

    logits = pd.concat(selected_frames, ignore_index=True).sort_values(
        list(TARGET_KEYS), kind="stable"
    ).reset_index(drop=True)
    if (
        len(logits) != candidate["rows"]
        or logits[list(TARGET_KEYS)].duplicated().any()
        or _ordered_key_digest(logits) != candidate["ordered_exact_key_sha256"]
        or set(logits[PAIR_FOLD_COLUMN].astype(int)) != set(range(N_FOLDS))
    ):
        raise SealedTestInferenceError("Five sealed outer folds do not form one exact OOF universe")
    primary = logits[list(TARGET_KEYS)].copy()
    primary["primary_probability"] = logits.primary_probability.astype(np.float32)
    primary["fusion_target"] = logits.fusion_target.astype(np.uint8)
    primary[PAIR_FOLD_COLUMN] = logits[PAIR_FOLD_COLUMN].astype(np.int8)
    primary["source_patient_fold"] = logits.source_patient_fold.astype(np.int8)
    primary["source_split"] = "test"

    staging.mkdir(parents=True, exist_ok=False)
    logits_path = staging / "TEST_LOGITS.PRIVATE.parquet"
    primary_path = staging / "PRIMARY_FUSION_FRAME.PRIVATE.parquet"
    logits[list(LOGIT_COLUMNS)].to_parquet(
        logits_path, index=False, compression="zstd", row_group_size=100_000
    )
    primary[list(PRIMARY_COLUMNS)].to_parquet(
        primary_path, index=False, compression="zstd", row_group_size=100_000
    )
    final_logits_path = output / logits_path.name
    final_primary_path = output / primary_path.name
    lineage = {
        "format": LINEAGE_FORMAT,
        "status": ACCEPTANCE_STATUS,
        "graph_variant": expected_graph_variant,
        "winner_id": declaration["winner_id"],
        "winner_declaration": {
            "path": str(declaration_path),
            "sha256": expected_declaration_sha,
        },
        "winner_config": {"path": str(config_path), "sha256": config_sha},
        "sealed_test_manifest": {
            "path": str(sealed_manifest_path),
            "sha256": sha256_file(sealed_manifest_path),
        },
        "patient_fold_authority": {
            "sample_patient_fold_map_sha256": FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256,
            "authority_receipt_sha256": FROZEN_V32_RECEIPT_SHA256,
            "patients": int(authority_audit["patients"]),
            "cancers": len(authority_audit["cancers"]),
            "folds": int(authority_audit["folds"]),
        },
        "candidate_authority": candidate,
        "prepared_folds": prepared_lineage,
        "winner_checkpoints": checkpoint_lineage,
        "sealed_folds": sealed_lineage,
        "test_labels_absent_from_training_payload": True,
        "sealed_payloads_deserialized_after_winner_lock": True,
        "test_metrics_used_for_selection": False,
        "winner_or_checkpoint_changed_by_test": False,
        "optimizer_created": False,
        "training_run": False,
        "old_fold_outputs_used": False,
    }
    lineage_path = staging / "LINEAGE.json"
    _write_json(lineage_path, lineage)
    audit = {
        "format": AUDIT_FORMAT,
        "status": ACCEPTANCE_STATUS,
        "checks": {
            "caller_pinned_winner_declaration_sha": True,
            "validation_only_winner_lock": True,
            "five_checkpoint_hashes_match": True,
            "five_prepared_payloads_train_validation_only": True,
            "test_labels_absent_from_training_payload": True,
            "frozen_patient_map_sha_matches": True,
            "frozen_patient_receipt_sha_matches": True,
            "sealed_test_patient_partitions_exact": True,
            "explicit_graph_variant_root": expected_graph_variant,
            "five_outer_pair_folds_form_exact_candidate_universe": True,
            "test_metrics_used_for_winner_selection": False,
            "output_reuse_forbidden": True,
        },
        "rows": int(len(primary)),
        "pair_fold_counts": {
            str(int(key)): int(value)
            for key, value in primary.groupby(PAIR_FOLD_COLUMN, observed=True).size().items()
        },
        "lineage": {"path": str(output / "LINEAGE.json"), "sha256": sha256_file(lineage_path)},
        "test_logits": {"path": str(final_logits_path), "sha256": sha256_file(logits_path)},
        "primary_fusion_frame": {
            "path": str(final_primary_path),
            "sha256": sha256_file(primary_path),
        },
        "production_8260_touched": False,
        "training_run": False,
    }
    audit_path = staging / "SEALED_TEST_INFERENCE_AUDIT.json"
    _write_json(audit_path, audit)
    acceptance = {
        "format": ACCEPTANCE_FORMAT,
        "status": ACCEPTANCE_STATUS,
        "graph_variant": expected_graph_variant,
        "winner_id": declaration["winner_id"],
        "winner_declaration_sha256": expected_declaration_sha,
        "sample_patient_fold_map_sha256": FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256,
        "authority_receipt_sha256": FROZEN_V32_RECEIPT_SHA256,
        "candidate_authority_sha256": candidate_sha,
        "sealed_test_manifest_sha256": sha256_file(sealed_manifest_path),
        "pair_fold_seed": pair_seed,
        "rows": int(len(primary)),
        "test_labels_absent_from_training_payload": True,
        "sealed_payloads_opened_only_after_winner_lock": True,
        "test_metrics_used_for_winner_selection": False,
        "winner_or_checkpoint_changed_by_test": False,
        "old_fold_outputs_used": False,
        "output_reuse_forbidden": True,
        "artifacts_read_only": True,
        "primary_fusion_frame": _artifact_record(primary_path) | {"path": str(final_primary_path)},
        "test_logits": _artifact_record(logits_path) | {"path": str(final_logits_path)},
        "lineage": _artifact_record(lineage_path) | {"path": str(output / "LINEAGE.json")},
        "audit": _artifact_record(audit_path) | {
            "path": str(output / "SEALED_TEST_INFERENCE_AUDIT.json")
        },
        "training_run": False,
        "winner_selection_run": False,
        "production_8260_touched": False,
    }
    acceptance_path = staging / "WINNER_LOCK_SEALED_TEST_INFERENCE_ACCEPTANCE.json"
    _write_json(acceptance_path, acceptance)
    for path in (logits_path, primary_path, lineage_path, audit_path, acceptance_path):
        _make_read_only(path)
    os.replace(staging, output)
    return acceptance


__all__ = [
    "ACCEPTANCE_FORMAT",
    "ACCEPTANCE_STATUS",
    "AUDIT_FORMAT",
    "GRAPH_VARIANTS",
    "LINEAGE_FORMAT",
    "LOGIT_COLUMNS",
    "PAIR_FOLD_COLUMN",
    "PRIMARY_COLUMNS",
    "SEALED_TEST_MANIFEST_FORMAT",
    "SEALED_TEST_PAYLOAD_FORMAT",
    "SealedTestInferenceError",
    "WINNER_DECLARATION_FORMAT",
    "materialize_winner_locked_sealed_test",
    "sha256_file",
]
