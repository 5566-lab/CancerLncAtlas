#!/usr/bin/env python3
"""Independent fail-closed audit of the donor-blocked metacell head.

This module deliberately does not import the training implementation.  It
reconstructs the model initialization, donor isolation, cross-fit aggregation,
BH correction, discovery decisions, and hash lineage directly from frozen
artifacts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import socket
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
import torch
from torch import nn


TRAINING_FORMAT = "CANCERLNCATLAS_V32_SC_METACELL_SIGNED_HEAD_V1"
AUDIT_FORMAT = "CANCERLNCATLAS_V32_SC_METACELL_INDEPENDENT_AUDIT_V1"
BINDING_FORMAT = "CANCERLNCATLAS_V32_SC_METACELL_INDEPENDENT_AUDIT_BINDING_V1"
CONFIG_FORMAT = "CANCERLNCATLAS_V32_SC_METACELL_RUN_CONFIG_V1"
INPUT_LINEAGE_FORMAT = "CANCERLNCATLAS_V32_SC_METACELL_INPUT_LINEAGE_V1"
CHECKPOINT_MANIFEST_FORMAT = (
    "CANCERLNCATLAS_V32_SC_METACELL_CHECKPOINT_MANIFEST_V1"
)
LINEAGE_FORMAT = "CANCERLNCATLAS_V32_SC_METACELL_TRAINING_LINEAGE_V1"
FULL_MATERIALIZATION_STATUS = "PASS_FULL_28_CANCER_MATERIALIZATION_CLOSURE"
FULL_INPUT_SCOPE = "PASS_FULL_28_CANCER_INPUT_LINEAGE"
FULL_LINEAGE_SCOPE = "PASS_FULL_28_CANCER_TRAINING_LINEAGE"
FULL_TRAINING_SCOPE = "PASS_FULL_28_CANCER_TRAINING"
N_FOLDS = 5
ROW_KEY = ["cancer_id", "cell_key", "query_donor_id", "lncrna_id", "pathway_id"]
PAIR_KEY = ["dataset_id", "cancer_id", "cell_key", "lncrna_id", "pathway_id"]
FEATURE_COLUMNS = [
    "source_meta_rho",
    "source_direction_consistency",
    "source_donor_count",
    "n_metacells",
    "lnc_detection_rate",
    "is_bulk_candidate",
    "is_sc_discovery_candidate",
    "measurement_scale_log_normalized",
]
TRAINING_ROW_COLUMNS = [
    "lncrna_id",
    "pathway_id",
    "query_donor_id",
    "rho",
    "n_metacells",
    "lnc_detection_rate",
    "is_bulk_candidate",
    "is_sc_discovery_candidate",
    "source_donor_count",
    "source_meta_rho",
    "source_direction_consistency",
    "dataset_id",
    "cancer_id",
    "cell_key",
    "outer_fold_id",
    "measurement_scale",
    "candidate_selection_query_donor_used",
    "association_method",
    "association_observation_unit",
    "donor_resolved_target",
]
TOP_LEVEL_ARTIFACTS = (
    "LAUNCH_PREFLIGHT.json",
    "RUN_CONFIG.json",
    "INPUT_LINEAGE_AUDIT.json",
    "CHECKPOINT_MANIFEST.json",
    "LINEAGE.json",
    "OOF_DONOR_PREDICTIONS.parquet",
    "CROSSFIT_ASSOCIATIONS.parquet",
    "DISCOVERED_NEW_RELATIONSHIPS.parquet",
    "FULL_MODEL_CONTRACT_VALIDATION.json",
)
RESULT_ARTIFACTS = (
    "OOF_DONOR_PREDICTIONS.parquet",
    "CROSSFIT_ASSOCIATIONS.parquet",
    "DISCOVERED_NEW_RELATIONSHIPS.parquet",
    "FULL_MODEL_CONTRACT_VALIDATION.json",
)
CLOSURE_ARTIFACTS = (
    "RUN_CONFIG.json",
    "INPUT_LINEAGE_AUDIT.json",
    "CHECKPOINT_MANIFEST.json",
    "LINEAGE.json",
)


class IndependentAuditError(RuntimeError):
    """Raised when independent evidence is missing or inconsistent."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise IndependentAuditError(f"missing or unsafe {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise IndependentAuditError(f"{label} is not a JSON object")
    return value


def state_hash(state: Mapping[str, torch.Tensor]) -> str:
    if not state:
        raise IndependentAuditError("empty model state")
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise IndependentAuditError(f"non-tensor state entry: {name}")
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(name).encode())
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def verify_record(
    record: Mapping[str, Any], expected_path: Path, label: str
) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise IndependentAuditError(f"{label} record is not an object")
    declared = Path(str(record.get("path", "")))
    if _resolved(declared) != _resolved(expected_path):
        raise IndependentAuditError(f"{label} path drift")
    if not declared.is_file() or declared.is_symlink():
        raise IndependentAuditError(f"{label} is missing or symlinked")
    observed_sha = sha256(declared)
    if str(record.get("sha256", "")) != observed_sha:
        raise IndependentAuditError(f"{label} hash drift")
    observed_bytes = declared.stat().st_size
    if "bytes" in record and int(record["bytes"]) != observed_bytes:
        raise IndependentAuditError(f"{label} byte count drift")
    return {
        "path": str(declared), "sha256": observed_sha, "bytes": observed_bytes,
    }


class IndependentSignedMetacellHead(nn.Module):
    """Architecture duplicate used only to rebuild the seeded initial state."""

    def __init__(
        self,
        lnc_core: np.ndarray,
        pathway_core: np.ndarray,
        cancer_core: np.ndarray,
        private_lncs: int,
        celltypes: int,
        numeric_features: int,
    ) -> None:
        super().__init__()
        self.register_buffer("lnc_core", torch.as_tensor(lnc_core, dtype=torch.float32))
        self.register_buffer(
            "pathway_core", torch.as_tensor(pathway_core, dtype=torch.float32)
        )
        self.register_buffer(
            "cancer_core", torch.as_tensor(cancer_core, dtype=torch.float32)
        )
        self.private_lnc = nn.Embedding(private_lncs, 32, padding_idx=0)
        self.celltype = nn.Embedding(celltypes, 16, padding_idx=0)
        width = 96 * 3 + 32 + 16 + numeric_features + 1
        self.network = nn.Sequential(
            nn.Linear(width, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )


def rebuild_initial_state_hash(checkpoint: Mapping[str, Any]) -> str:
    state = checkpoint.get("state_dict")
    if not isinstance(state, Mapping):
        raise IndependentAuditError("checkpoint lacks state_dict")
    for name in ("lnc_core", "pathway_core", "cancer_core"):
        if name not in state or not isinstance(state[name], torch.Tensor):
            raise IndependentAuditError(f"checkpoint lacks frozen buffer {name}")
    lncs = checkpoint.get("lncrna_vocabulary")
    celltypes = checkpoint.get("celltype_vocabulary")
    columns = checkpoint.get("numeric_columns")
    if not isinstance(lncs, list) or not isinstance(celltypes, list) or not isinstance(columns, list):
        raise IndependentAuditError("checkpoint vocabularies are absent")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(checkpoint["seed"]))
        model = IndependentSignedMetacellHead(
            state["lnc_core"].detach().cpu().numpy(),
            state["pathway_core"].detach().cpu().numpy(),
            state["cancer_core"].detach().cpu().numpy(),
            len(lncs) + 1,
            len(celltypes) + 1,
            len(columns),
        )
    return state_hash(model.state_dict())


def _bh(values: pd.Series) -> np.ndarray:
    p = values.to_numpy(float)
    if np.isfinite(p).any() and (
        np.nanmin(p) < -1e-12 or np.nanmax(p) > 1.0 + 1e-12
    ):
        raise IndependentAuditError("p-value outside [0, 1]")
    result = np.full(len(p), np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    if not len(finite):
        return result
    order = finite[np.argsort(p[finite], kind="mergesort")]
    ranked = p[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    result[order] = np.minimum(ranked, 1.0)
    return result


def independently_aggregate_oof(predictions: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(
        set(ROW_KEY + PAIR_KEY + [
            "rho", "predicted_rho", "outer_fold_id", "is_bulk_candidate",
            "is_sc_discovery_candidate", "lncrna_core_available", "n_metacells",
        ]) - set(predictions.columns)
    )
    if missing:
        raise IndependentAuditError(f"OOF predictions lack columns {missing}")
    if predictions.empty or predictions.duplicated(ROW_KEY).any():
        raise IndependentAuditError("OOF predictions are empty or duplicated")
    observed = predictions.rho.to_numpy(float)
    predicted = predictions.predicted_rho.to_numpy(float)
    if (
        not np.isfinite(observed).all()
        or not np.isfinite(predicted).all()
        or (np.abs(observed) > 1.000001).any()
        or (np.abs(predicted) > 1.000001).any()
    ):
        raise IndependentAuditError("OOF rho values are non-finite or outside [-1, 1]")
    frame = predictions.copy()
    frame["_fisher"] = np.arctanh(np.clip(observed, -0.999999, 0.999999))
    frame["_fisher_sq"] = frame._fisher ** 2
    frame["_pred_fisher"] = np.arctanh(
        np.clip(predicted, -0.999999, 0.999999)
    )
    frame["_positive"] = frame.rho.gt(0).astype(int)
    frame["_negative"] = frame.rho.lt(0).astype(int)
    grouped = frame.groupby(PAIR_KEY, observed=True).agg(
        n_donors=("query_donor_id", "nunique"),
        observed_rows=("query_donor_id", "size"),
        outer_fold_count=("outer_fold_id", "nunique"),
        fisher_sum=("_fisher", "sum"),
        fisher_sq_sum=("_fisher_sq", "sum"),
        predicted_fisher_sum=("_pred_fisher", "sum"),
        positive_donors=("_positive", "sum"),
        negative_donors=("_negative", "sum"),
        is_bulk_candidate=("is_bulk_candidate", "max"),
        is_sc_discovery_candidate=("is_sc_discovery_candidate", "max"),
        lnc_core_available=("lncrna_core_available", "max"),
        mean_n_metacells=("n_metacells", "mean"),
    ).reset_index()
    if not grouped.observed_rows.eq(grouped.n_donors).all():
        raise IndependentAuditError("one donor contributes repeated pair effects")
    n = grouped.n_donors.to_numpy(float)
    mean = grouped.fisher_sum.to_numpy(float) / n
    variance = np.divide(
        grouped.fisher_sq_sum.to_numpy(float)
        - grouped.fisher_sum.to_numpy(float) ** 2 / n,
        n - 1,
        out=np.full(len(grouped), np.nan),
        where=n > 1,
    )
    standard_error = np.sqrt(np.maximum(variance, 0.0) / n)
    statistic = np.divide(
        mean,
        standard_error,
        out=np.full(len(grouped), np.nan),
        where=standard_error > 1e-12,
    )
    p_value = 2.0 * student_t.sf(np.abs(statistic), np.maximum(n - 1, 1))
    exact = (standard_error <= 1e-12) & (n > 1)
    p_value[exact & (np.abs(mean) > 1e-12)] = 0.0
    p_value[exact & (np.abs(mean) <= 1e-12)] = 1.0
    grouped["observed_meta_rho"] = np.tanh(mean)
    grouped["predicted_meta_rho"] = np.tanh(
        grouped.predicted_fisher_sum.to_numpy(float) / n
    )
    signed = grouped.positive_donors + grouped.negative_donors
    grouped["direction_consistency"] = np.divide(
        np.maximum(grouped.positive_donors, grouped.negative_donors),
        signed,
        out=np.full(len(grouped), np.nan),
        where=signed > 0,
    )
    grouped["meta_p_value"] = p_value
    grouped["meta_fdr"] = np.nan
    for _, rows in grouped.groupby(["cancer_id", "cell_key"], observed=True).groups.items():
        grouped.loc[rows, "meta_fdr"] = _bh(grouped.loc[rows, "meta_p_value"])
    grouped["multiple_testing_family"] = "CANCER_X_CELLTYPE_CROSSFIT_SELECTED_PAIRS"
    grouped["new_relationship_discovered"] = (
        ~grouped.is_bulk_candidate.astype(bool)
        & grouped.is_sc_discovery_candidate.astype(bool)
        & grouped.n_donors.ge(3)
        & grouped.outer_fold_count.ge(2)
        & grouped.meta_fdr.le(0.05)
        & grouped.observed_meta_rho.abs().ge(0.2)
        & grouped.direction_consistency.ge(0.75)
    )
    grouped["discovery_semantics"] = (
        "CROSSFIT_SOURCE_SCREEN_HELDOUT_DONOR_EFFECT_META_ANALYSIS"
    )
    return grouped


def assert_frame_equal_by_key(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    keys: list[str],
    label: str,
) -> None:
    if set(observed.columns) != set(expected.columns):
        raise IndependentAuditError(
            f"{label} column drift: observed={sorted(observed.columns)}, "
            f"expected={sorted(expected.columns)}"
        )
    if observed.duplicated(keys).any() or expected.duplicated(keys).any():
        raise IndependentAuditError(f"{label} has duplicate keys")
    left = observed.sort_values(keys, kind="stable").reset_index(drop=True)
    right = expected.sort_values(keys, kind="stable").reset_index(drop=True)
    left = left[right.columns]
    try:
        pd.testing.assert_frame_equal(
            left,
            right,
            check_dtype=False,
            check_exact=False,
            rtol=1e-10,
            atol=1e-12,
        )
    except AssertionError as error:
        raise IndependentAuditError(f"{label} content drift: {error}") from error


def validate_embedding_routes(predictions: pd.DataFrame) -> dict[str, Any]:
    required = {
        "embedding_route",
        "lncrna_core_available",
        "private_lnc_embedding_available",
    }
    if not required.issubset(predictions.columns):
        raise IndependentAuditError("OOF predictions lack embedding-route evidence")
    for column in ("lncrna_core_available", "private_lnc_embedding_available"):
        if (
            predictions[column].isna().any()
            or not pd.api.types.is_bool_dtype(predictions[column].dtype)
        ):
            raise IndependentAuditError(
                f"OOF embedding availability is not strict boolean: {column}"
            )
    if predictions.embedding_route.isna().any():
        raise IndependentAuditError("OOF embedding route contains null values")
    core_available = predictions.lncrna_core_available.astype(bool).to_numpy()
    private_available = (
        predictions.private_lnc_embedding_available.astype(bool).to_numpy()
    )
    expected_routes = np.select(
        [
            core_available & private_available,
            core_available,
            private_available,
        ],
        [
            "FROZEN_CORE_PLUS_PRIVATE_ID_ADAPTER",
            "FROZEN_CORE_WITH_ZERO_PRIVATE_FALLBACK",
            "SINGLE_CELL_PRIVATE_OOV_ID_EMBEDDING",
        ],
        default="SINGLE_CELL_OOV_ZERO_ID_FALLBACK",
    )
    if not np.array_equal(
        predictions.embedding_route.astype(str).to_numpy(), expected_routes
    ):
        raise IndependentAuditError("OOF embedding route/availability semantics drift")
    counts = (
        predictions.embedding_route.astype(str)
        .value_counts(dropna=False)
        .sort_index()
    )
    return {
        "embedding_routes": {
            str(route): int(count) for route, count in counts.items()
        },
        "zero_identity_fallback_rows": int(
            counts.get("SINGLE_CELL_OOV_ZERO_ID_FALLBACK", 0)
        ),
    }


def replay_history(
    history: Any, *, epochs: int, patience: int
) -> tuple[float, int, int]:
    if not isinstance(history, list) or not history:
        raise IndependentAuditError("checkpoint history is empty")
    if len(history) > epochs:
        raise IndependentAuditError("checkpoint history exceeds configured epochs")
    best = float("inf")
    best_epoch = -1
    stale = 0
    for index, row in enumerate(history):
        if not isinstance(row, Mapping) or int(row.get("epoch", -1)) != index:
            raise IndependentAuditError("history epoch sequence drift")
        train_loss = float(row.get("train_loss", math.nan))
        validation_loss = float(row.get("validation_loss", math.nan))
        if (
            not math.isfinite(train_loss)
            or not math.isfinite(validation_loss)
            or train_loss < 0
            or validation_loss < 0
        ):
            raise IndependentAuditError("history contains invalid losses")
        if validation_loss < best - 1e-6:
            best = validation_loss
            best_epoch = index
            stale = 0
        else:
            stale += 1
            if stale >= patience and index != len(history) - 1:
                raise IndependentAuditError("history continues after patience stop")
    if len(history) < epochs and stale < patience:
        raise IndependentAuditError("short history lacks an early-stop condition")
    return best, best_epoch, stale


def _core_values(path: Path) -> np.ndarray:
    frame = pd.read_parquet(path)
    columns = sorted(column for column in frame if str(column).startswith("core_feature_"))
    if len(columns) != 96:
        raise IndependentAuditError(f"core embedding width drift: {path}")
    values = frame[columns].to_numpy(np.float32)
    if not np.isfinite(values).all():
        raise IndependentAuditError(f"non-finite core embedding: {path}")
    return values


def audit_checkpoint(
    *,
    fold: int,
    checkpoint_path: Path,
    fold_receipt: Mapping[str, Any],
    manifest_record: Mapping[str, Any],
    core_root: Path,
    base_seed: int,
    epochs: int,
    patience: int,
) -> dict[str, Any]:
    checkpoint_record = fold_receipt.get("files", {}).get("checkpoint")
    prediction_record = fold_receipt.get("files", {}).get("predictions")
    verify_record(checkpoint_record, checkpoint_path, f"fold {fold} checkpoint")
    verify_record(
        prediction_record,
        checkpoint_path.parent / "QUERY_PREDICTIONS.parquet",
        f"fold {fold} predictions",
    )
    verify_record(
        manifest_record.get("checkpoint", {}),
        checkpoint_path,
        f"fold {fold} checkpoint manifest",
    )
    verify_record(
        manifest_record.get("predictions", {}),
        checkpoint_path.parent / "QUERY_PREDICTIONS.parquet",
        f"fold {fold} prediction manifest",
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise IndependentAuditError(f"fold {fold} checkpoint is not an object")
    expected_seed = int(base_seed) + fold * 101
    if (
        payload.get("format") != TRAINING_FORMAT
        or int(payload.get("outer_fold_id", -1)) != fold
        or int(payload.get("seed", -1)) != expected_seed
        or int(fold_receipt.get("seed", -1)) != expected_seed
        or int(manifest_record.get("seed", -1)) != expected_seed
    ):
        raise IndependentAuditError(f"fold {fold} checkpoint identity drift")
    if (
        payload.get("fresh_random_initialization") is not True
        or payload.get("old_checkpoint_loaded") is not False
        or payload.get("source_checkpoint_sha256") is not None
        or payload.get("device") != "cpu"
        or payload.get("cuda_used") is not False
    ):
        raise IndependentAuditError(f"fold {fold} initialization/device contract drift")
    if (
        fold_receipt.get("fresh_random_initialization") is not True
        or fold_receipt.get("old_checkpoint_loaded") is not False
        or fold_receipt.get("source_checkpoint_sha256") is not None
        or fold_receipt.get("device") != "cpu"
        or fold_receipt.get("cuda_used") is not False
    ):
        raise IndependentAuditError(f"fold {fold} receipt initialization/device drift")
    if (
        manifest_record.get("fresh_random_initialization") is not True
        or manifest_record.get("old_checkpoint_loaded") is not False
        or manifest_record.get("source_checkpoint_sha256") is not None
        or manifest_record.get("device") != "cpu"
        or manifest_record.get("cuda_used") is not False
    ):
        raise IndependentAuditError(f"fold {fold} manifest old-checkpoint drift")

    lncrna_vocabulary = payload.get("lncrna_vocabulary")
    celltype_vocabulary = payload.get("celltype_vocabulary")
    numeric_columns = payload.get("numeric_columns")
    numeric_mean = np.asarray(payload.get("numeric_mean"), dtype=float)
    numeric_sd = np.asarray(payload.get("numeric_sd"), dtype=float)
    if (
        not isinstance(lncrna_vocabulary, list)
        or lncrna_vocabulary != sorted(set(map(str, lncrna_vocabulary)))
        or not isinstance(celltype_vocabulary, list)
        or celltype_vocabulary != sorted(set(map(str, celltype_vocabulary)))
        or list(numeric_columns or []) != FEATURE_COLUMNS
        or int(payload.get("unknown_identity_index", -1)) != 0
        or numeric_mean.shape != (len(FEATURE_COLUMNS),)
        or numeric_sd.shape != (len(FEATURE_COLUMNS),)
        or not np.isfinite(numeric_mean).all()
        or not np.isfinite(numeric_sd).all()
        or (numeric_sd <= 0).any()
    ):
        raise IndependentAuditError(f"fold {fold} checkpoint preprocessing state drift")

    state = payload.get("state_dict")
    if not isinstance(state, Mapping) or not state:
        raise IndependentAuditError(f"fold {fold} lacks model state")
    for name, tensor in state.items():
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.device.type != "cpu"
            or not torch.isfinite(tensor).all()
        ):
            raise IndependentAuditError(f"fold {fold} has non-finite state {name}")
    for name in ("private_lnc.weight", "celltype.weight"):
        tensor = state.get(name)
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.ndim != 2
            or not tensor.shape[0]
            or not torch.equal(tensor[0], torch.zeros_like(tensor[0]))
        ):
            raise IndependentAuditError(
                f"fold {fold} learned-identity padding row is not fixed zero: {name}"
            )
    final_hash = state_hash(state)
    rebuilt_initial = rebuild_initial_state_hash(payload)
    declared_initial = str(payload.get("initial_parameter_sha256", ""))
    declared_final = str(payload.get("trained_parameter_sha256", ""))
    if (
        rebuilt_initial != declared_initial
        or declared_initial != str(fold_receipt.get("initial_parameter_sha256", ""))
        or declared_initial != str(manifest_record.get("initial_parameter_sha256", ""))
    ):
        raise IndependentAuditError(f"fold {fold} initial parameter hash drift")
    if (
        final_hash != declared_final
        or final_hash != str(fold_receipt.get("trained_parameter_sha256", ""))
        or final_hash != str(manifest_record.get("trained_parameter_sha256", ""))
        or final_hash == declared_initial
    ):
        raise IndependentAuditError(f"fold {fold} trained parameter hash drift")

    frozen = {name: tensor for name, tensor in state.items() if name.endswith("_core")}
    frozen_hash = state_hash(frozen)
    frozen_before = str(payload.get("frozen_core_sha256_before", ""))
    frozen_after = str(payload.get("frozen_core_sha256_after", ""))
    if not (
        frozen_hash == frozen_before == frozen_after
        == str(fold_receipt.get("frozen_core_sha256_before", ""))
        == str(fold_receipt.get("frozen_core_sha256_after", ""))
        == str(manifest_record.get("frozen_core_sha256_before", ""))
        == str(manifest_record.get("frozen_core_sha256_after", ""))
    ):
        raise IndependentAuditError(f"fold {fold} frozen-core hash drift")

    core_dir = core_root / f"patient_fold={fold}"
    core_paths = {
        "lncRNA": core_dir / "lncRNA.parquet",
        "pathway": core_dir / "pathway.parquet",
        "cancer": core_dir / "cancer.parquet",
    }
    core_arrays = {name: _core_values(path) for name, path in core_paths.items()}
    core_manifest = read_json(
        core_root / "CORE_EMBEDDING_MANIFEST.json", "core embedding manifest"
    )
    if (
        core_manifest.get("all_embeddings_from_newly_trained_v32_core") is not True
        or core_manifest.get("historical_checkpoint_loaded") is not False
        or core_manifest.get("historical_prediction_loaded") is not False
    ):
        raise IndependentAuditError("core embedding manifest is not a fresh export")
    manifest_folds = core_manifest.get("folds")
    if not isinstance(manifest_folds, Mapping) or set(manifest_folds) != {
        str(value) for value in range(N_FOLDS)
    }:
        raise IndependentAuditError("core embedding manifest fold closure drift")
    core_fold = manifest_folds[str(fold)]
    if (
        not isinstance(core_fold, Mapping)
        or int(core_fold.get("patient_fold", -1)) != fold
        or core_fold.get("old_checkpoint_loaded") is not False
        or core_fold.get("trained_from_scratch") is not True
        or core_fold.get("partial_chunk_smoke") is not False
    ):
        raise IndependentAuditError(f"fold {fold} frozen-core training lineage drift")
    exports = core_fold.get("exports")
    if not isinstance(exports, Mapping):
        raise IndependentAuditError(f"fold {fold} core export manifest is absent")
    expected_lnc = np.vstack(
        [core_arrays["lncRNA"], np.zeros((1, 96), dtype=np.float32)]
    )
    expected_buffers = {
        "lnc_core": expected_lnc,
        "pathway_core": core_arrays["pathway"],
        "cancer_core": core_arrays["cancer"],
    }
    for name, path in core_paths.items():
        observed_sha = sha256(path)
        if observed_sha != str(fold_receipt.get("core_files", {}).get(name, "")):
            raise IndependentAuditError(f"fold {fold} {name} core file hash drift")
        if observed_sha != str(manifest_record.get("core_files", {}).get(name, "")):
            raise IndependentAuditError(f"fold {fold} {name} manifest core hash drift")
        export = exports.get(name)
        if (
            not isinstance(export, Mapping)
            or _resolved(core_root / str(export.get("path", ""))) != _resolved(path)
            or str(export.get("sha256", "")) != observed_sha
            or int(export.get("features", -1)) != 96
            or int(export.get("rows", -1)) != len(core_arrays[name])
        ):
            raise IndependentAuditError(f"fold {fold} {name} core export lineage drift")
    for name, expected in expected_buffers.items():
        observed = state[name].detach().cpu().numpy()
        if observed.shape != expected.shape or not np.array_equal(observed, expected):
            raise IndependentAuditError(f"fold {fold} frozen buffer content drift: {name}")

    history = payload.get("history")
    best_loss, best_epoch, stale = replay_history(
        history, epochs=epochs, patience=patience
    )
    if history != fold_receipt.get("history"):
        raise IndependentAuditError(f"fold {fold} history differs from receipt")
    if int(fold_receipt.get("epochs_completed", -1)) != len(history):
        raise IndependentAuditError(f"fold {fold} epoch count drift")
    if not math.isclose(
        float(fold_receipt.get("best_validation_loss", math.nan)),
        best_loss,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise IndependentAuditError(f"fold {fold} best loss drift")
    return {
        "outer_fold_id": fold,
        "seed": expected_seed,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "initial_parameter_sha256": declared_initial,
        "independently_rebuilt_initial_parameter_sha256": rebuilt_initial,
        "trained_parameter_sha256": final_hash,
        "frozen_core_sha256": frozen_hash,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "terminal_stale_epochs": stale,
        "all_tensors_finite": True,
        "old_checkpoint_loaded": False,
        "device": "cpu",
    }


def _read_materialization_fold(
    materialization_root: Path, fold: int, filename: str, columns=None
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for cancer_root in sorted((materialization_root / "cancers").iterdir()):
        if not cancer_root.is_dir() or not (cancer_root / "SUCCESS.json").is_file():
            continue
        path = cancer_root / f"outer_fold={fold}" / filename
        if path.is_file():
            frame = pd.read_parquet(path, columns=columns)
            if len(frame):
                parts.append(frame)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def validate_materialization_training_file_hashes(
    materialization_root: Path,
) -> dict[str, int]:
    validation = read_json(
        materialization_root / "MATERIALIZATION_VALIDATION.json",
        "materialization validation",
    )
    reports = validation.get("cancers")
    if not isinstance(reports, list) or len(reports) != 28:
        raise IndependentAuditError("materialization validation lacks 28 cancer reports")
    reports_by_cancer = {
        str(report.get("cancer_id", "")).upper(): report for report in reports
    }
    expected = {
        str(value).upper() for value in validation.get("expected_cancers", [])
    }
    if set(reports_by_cancer) != expected or len(reports_by_cancer) != 28:
        raise IndependentAuditError("materialization cancer report identity drift")

    fold_receipts = 0
    training_files = 0
    for cancer in sorted(expected):
        cancer_root = materialization_root / "cancers" / cancer
        report = reports_by_cancer[cancer]
        cancer_success_path = cancer_root / "SUCCESS.json"
        cancer_success = read_json(cancer_success_path, f"{cancer} materialization SUCCESS")
        if str(report.get("success_sha256", "")) != sha256(cancer_success_path):
            raise IndependentAuditError(f"{cancer} materialization SUCCESS hash drift")
        donor_path = cancer_root / "DONOR_FOLDS.parquet"
        verify_record(
            cancer_success.get("files", {}).get("DONOR_FOLDS.parquet", {}),
            donor_path,
            f"{cancer} donor-fold receipt",
        )
        verify_record(
            report.get("verified_cancer_files", {}).get("DONOR_FOLDS.parquet", {}),
            donor_path,
            f"{cancer} validated donor-fold record",
        )
        donor_folds = pd.read_parquet(
            donor_path, columns=["cancer_id", "donor_id", "outer_fold_id"]
        )
        if (
            donor_folds.empty
            or donor_folds.donor_id.astype(str).duplicated().any()
            or not donor_folds.cancer_id.astype(str).eq(cancer).all()
        ):
            raise IndependentAuditError(f"{cancer} donor-fold authority drift")
        active_folds = sorted(set(donor_folds.outer_fold_id.astype(int)))
        validation_fold_hashes = report.get("fold_success_sha256")
        cancer_fold_hashes = cancer_success.get("fold_receipt_sha256")
        expected_fold_keys = {str(fold) for fold in active_folds}
        if (
            not isinstance(validation_fold_hashes, Mapping)
            or set(validation_fold_hashes) != expected_fold_keys
            or not isinstance(cancer_fold_hashes, Mapping)
            or set(cancer_fold_hashes) != expected_fold_keys
        ):
            raise IndependentAuditError(f"{cancer} fold receipt hash closure drift")
        for fold in active_folds:
            fold_root = cancer_root / f"outer_fold={fold}"
            fold_success_path = fold_root / "SUCCESS.json"
            observed_success_sha = sha256(fold_success_path)
            if (
                str(validation_fold_hashes[str(fold)]) != observed_success_sha
                or str(cancer_fold_hashes[str(fold)]) != observed_success_sha
            ):
                raise IndependentAuditError(
                    f"{cancer} fold {fold} materialization receipt hash drift"
                )
            receipt = read_json(
                fold_success_path, f"{cancer} fold {fold} materialization SUCCESS"
            )
            if (
                receipt.get("target_host") != "149"
                or receipt.get("state") != "FOLD_LOCAL_METACELL_TARGETS_READY"
                or str(receipt.get("cancer_id", "")).upper() != cancer
                or int(receipt.get("outer_fold_id", -1)) != fold
            ):
                raise IndependentAuditError(
                    f"{cancer} fold {fold} materialization receipt identity drift"
                )
            for name in ("TRAIN_ROWS.parquet", "QUERY_ROWS.parquet"):
                verify_record(
                    receipt.get("files", {}).get(name, {}),
                    fold_root / name,
                    f"{cancer} fold {fold} {name}",
                )
                training_files += 1
            fold_receipts += 1
    return {
        "cancer_receipts": 28,
        "fold_receipts": fold_receipts,
        "training_parquet_files": training_files,
    }


def audit_oof_isolation(
    training_root: Path,
    materialization_root: Path,
    fold_receipts: Mapping[int, Mapping[str, Any]],
    *,
    require_receipt_hash_closure: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    materialization_hash_closure = (
        validate_materialization_training_file_hashes(materialization_root)
        if require_receipt_hash_closure
        else None
    )
    donor_fold: dict[tuple[str, str], int] = {}
    cancers = []
    for cancer_root in sorted((materialization_root / "cancers").iterdir()):
        if not cancer_root.is_dir() or not (cancer_root / "SUCCESS.json").is_file():
            continue
        cancer = cancer_root.name
        cancers.append(cancer)
        folds = pd.read_parquet(cancer_root / "DONOR_FOLDS.parquet")
        for row in folds.itertuples(index=False):
            key = (str(row.cancer_id), str(row.donor_id))
            if key in donor_fold:
                raise IndependentAuditError(f"duplicate donor-fold authority: {key}")
            donor_fold[key] = int(row.outer_fold_id)
    if len(cancers) != 28:
        raise IndependentAuditError(
            f"OOF audit requires 28 completed cancers, observed {len(cancers)}"
        )

    prediction_parts: list[pd.DataFrame] = []
    fold_reports: list[dict[str, Any]] = []
    for fold in range(N_FOLDS):
        prediction_path = training_root / f"outer_fold={fold}" / "QUERY_PREDICTIONS.parquet"
        predictions = pd.read_parquet(prediction_path)
        if predictions.empty or predictions.duplicated(ROW_KEY).any():
            raise IndependentAuditError(f"fold {fold} predictions are empty or duplicated")
        if not predictions.outer_fold_id.astype(int).eq(fold).all():
            raise IndependentAuditError(f"fold {fold} prediction fold drift")
        for cancer, donor, assigned in zip(
            predictions.cancer_id.astype(str),
            predictions.query_donor_id.astype(str),
            predictions.outer_fold_id.astype(int),
        ):
            if donor_fold.get((cancer, donor)) != int(assigned):
                raise IndependentAuditError(
                    f"fold {fold} donor is not held out by authority: {cancer}/{donor}"
                )

        expected_query = _read_materialization_fold(
            materialization_root,
            fold,
            "QUERY_ROWS.parquet",
            columns=TRAINING_ROW_COLUMNS,
        )
        if expected_query.empty:
            raise IndependentAuditError(f"fold {fold} has no materialized query rows")
        query_numeric = expected_query[
            [
                "rho",
                "n_metacells",
                "lnc_detection_rate",
                "source_donor_count",
                "source_meta_rho",
                "source_direction_consistency",
            ]
        ].to_numpy(float)
        if (
            not np.isfinite(query_numeric).all()
            or expected_query.rho.astype(float).abs().gt(1.000001).any()
            or expected_query.source_meta_rho.astype(float).abs().gt(1.000001).any()
            or expected_query.candidate_selection_query_donor_used.astype(bool).any()
            or not expected_query.association_method.astype(str).eq(
                "WITHIN_DONOR_METACELL_SPEARMAN"
            ).all()
            or not expected_query.association_observation_unit.astype(str).eq(
                "within_donor_metacell"
            ).all()
            or not expected_query.donor_resolved_target.eq(True).all()
        ):
            raise IndependentAuditError(f"fold {fold} query target semantics drift")
        missing_columns = sorted(set(expected_query.columns) - set(predictions.columns))
        if missing_columns:
            raise IndependentAuditError(
                f"fold {fold} predictions dropped query columns {missing_columns}"
            )
        assert_frame_equal_by_key(
            predictions[expected_query.columns],
            expected_query,
            ROW_KEY,
            f"fold {fold} materialized query/prediction binding",
        )
        train_donors = _read_materialization_fold(
            materialization_root,
            fold,
            "TRAIN_ROWS.parquet",
            columns=["cancer_id", "query_donor_id"],
        ).drop_duplicates()
        train_keys = set(
            zip(train_donors.cancer_id.astype(str), train_donors.query_donor_id.astype(str))
        )
        query_keys = set(
            zip(predictions.cancer_id.astype(str), predictions.query_donor_id.astype(str))
        )
        if train_keys & query_keys:
            raise IndependentAuditError(f"fold {fold} query donor leaked into training rows")
        if any(donor_fold.get(key) == fold for key in train_keys):
            raise IndependentAuditError(f"fold {fold} held-out donor appears in training rows")
        receipt = fold_receipts[fold]
        route_evidence = validate_embedding_routes(predictions)
        if (
            int(receipt.get("query_rows", -1)) != len(predictions)
            or int(receipt.get("query_donors", -1)) != len(query_keys)
            or int(receipt.get("fit_tuning_donor_overlap", -1)) != 0
            or int(receipt.get("query_oov_zero_id_fallback_rows", -1))
            != route_evidence["zero_identity_fallback_rows"]
        ):
            raise IndependentAuditError(f"fold {fold} prediction receipt count drift")
        prediction_parts.append(predictions)
        fold_reports.append(
            {
                "outer_fold_id": fold,
                "prediction_rows": len(predictions),
                "query_donor_groups": len(query_keys),
                "training_donor_groups": len(train_keys),
                "query_training_donor_overlap": 0,
                "prediction_sha256": sha256(prediction_path),
                **route_evidence,
            }
        )
    combined = pd.concat(prediction_parts, ignore_index=True)
    if combined.duplicated(ROW_KEY).any():
        raise IndependentAuditError("OOF donor target is predicted more than once")
    stored = pd.read_parquet(training_root / "OOF_DONOR_PREDICTIONS.parquet")
    assert_frame_equal_by_key(stored, combined, ROW_KEY, "combined OOF predictions")
    return stored, {
        "cancers": sorted(cancers),
        "cancer_count": len(cancers),
        "donor_fold_authority_rows": len(donor_fold),
        "oof_rows": len(stored),
        "oof_duplicate_keys": 0,
        "folds": fold_reports,
        "materialization_hash_closure": materialization_hash_closure,
    }


def _settings(config: Mapping[str, Any]) -> Mapping[str, Any]:
    value = config.get("training", config)
    if not isinstance(value, Mapping):
        raise IndependentAuditError("RUN_CONFIG training settings are absent")
    return value


def _pre_lineage_artifact_names() -> set[str]:
    names = {name for name in TOP_LEVEL_ARTIFACTS if name != "LINEAGE.json"}
    for fold in range(N_FOLDS):
        names.update(
            {
                f"outer_fold={fold}/best_model_state.pt",
                f"outer_fold={fold}/QUERY_PREDICTIONS.parquet",
            }
        )
    return names


def verify_artifact_manifest(
    records: Any,
    expected_names: set[str],
    training_root: Path,
    label: str,
) -> None:
    if not isinstance(records, Mapping) or set(records) != expected_names:
        observed = sorted(records) if isinstance(records, Mapping) else []
        raise IndependentAuditError(
            f"{label} artifact set drift: observed={observed}, "
            f"expected={sorted(expected_names)}"
        )
    for name in sorted(expected_names):
        verify_record(records[name], training_root / name, f"{label}/{name}")


def validate_full_input_documents(
    materialization_root: Path, core_root: Path
) -> tuple[Path, Path, Path, dict[str, Any]]:
    validation_path = materialization_root / "MATERIALIZATION_VALIDATION.json"
    preprocessing_path = materialization_root / "SINGLE_CELL_PREPROCESSING_AUDIT.json"
    core_manifest_path = core_root / "CORE_EMBEDDING_MANIFEST.json"
    validation = read_json(validation_path, "materialization validation")
    preprocessing = read_json(preprocessing_path, "single-cell preprocessing audit")
    core_manifest = read_json(core_manifest_path, "core embedding manifest")

    expected = [str(value).upper() for value in validation.get("expected_cancers", [])]
    validated = [str(value).upper() for value in validation.get("validated_cancers", [])]
    complete = sorted(
        path.name.upper()
        for path in (materialization_root / "cancers").glob("*")
        if path.is_dir() and (path / "SUCCESS.json").is_file()
    )
    if (
        validation.get("status") != "PASS"
        or validation.get("closure_status") != FULL_MATERIALIZATION_STATUS
        or validation.get("validation_scope") != "FULL"
        or int(validation.get("expected_cancer_count", -1)) != 28
        or int(validation.get("validated_cancer_count", -1)) != 28
        or len(expected) != 28
        or len(set(expected)) != 28
        or sorted(expected) != sorted(validated)
        or complete != sorted(expected)
    ):
        raise IndependentAuditError("materialization full-closure document drift")
    active = [str(value).upper() for value in preprocessing.get("active_cancers", [])]
    if (
        preprocessing.get("status") != "PASS"
        or int(preprocessing.get("active_cancer_count", -1)) != 28
        or sorted(active) != sorted(expected)
        or int(preprocessing.get("eligible_training_cells", 0)) <= 0
        or int(
            preprocessing.get("eligible_cells_failing_preprocessing_pass_qc", -1)
        )
        != 0
        or int(
            preprocessing.get("eligible_cells_failing_batch_correction_applied", -1)
        )
        != 0
        or int(
            preprocessing.get("eligible_cells_failing_annotation_completed", -1)
        )
        != 0
    ):
        raise IndependentAuditError("single-cell preprocessing full closure drift")
    folds = core_manifest.get("folds")
    if (
        core_manifest.get("all_embeddings_from_newly_trained_v32_core") is not True
        or core_manifest.get("historical_checkpoint_loaded") is not False
        or core_manifest.get("historical_prediction_loaded") is not False
        or not isinstance(folds, Mapping)
        or set(folds) != {str(value) for value in range(N_FOLDS)}
    ):
        raise IndependentAuditError("frozen-core manifest full closure drift")
    return validation_path, preprocessing_path, core_manifest_path, {
        "expected_cancers": sorted(expected),
        "eligible_training_cells": int(preprocessing["eligible_training_cells"]),
        "core_manifest": core_manifest,
    }


def validate_lineage_closure(
    training_root: Path,
    materialization_root: Path,
    core_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    documents = {
        name: read_json(training_root / name, name)
        for name in (
            "LAUNCH_PREFLIGHT.json",
            "RUN_CONFIG.json",
            "INPUT_LINEAGE_AUDIT.json",
            "CHECKPOINT_MANIFEST.json",
            "LINEAGE.json",
            "FULL_MODEL_CONTRACT_VALIDATION.json",
            "SUCCESS.json",
        )
    }
    preflight = documents["LAUNCH_PREFLIGHT.json"]
    config = documents["RUN_CONFIG.json"]
    inputs = documents["INPUT_LINEAGE_AUDIT.json"]
    manifest = documents["CHECKPOINT_MANIFEST.json"]
    lineage = documents["LINEAGE.json"]
    contract = documents["FULL_MODEL_CONTRACT_VALIDATION.json"]
    success = documents["SUCCESS.json"]

    (
        material_validation,
        preprocessing_audit,
        core_manifest,
        input_evidence,
    ) = validate_full_input_documents(materialization_root, core_root)

    if (
        preflight.get("format") != TRAINING_FORMAT
        or preflight.get("target_host") != "149"
        or preflight.get("cpu_only") is not True
        or preflight.get("paid_compute") is not False
        or preflight.get("allow_partial_materialization") is not False
        or _resolved(Path(str(preflight.get("materialization_root", ""))))
        != _resolved(materialization_root)
        or _resolved(Path(str(preflight.get("core_root", "")))) != _resolved(core_root)
        or preflight.get("fresh_random_initialization") is not True
        or preflight.get("old_checkpoint_loaded") is not False
        or preflight.get("source_checkpoint_sha256") is not None
        or preflight.get("device") != "cpu"
        or preflight.get("cuda_used") is not False
    ):
        raise IndependentAuditError("training launch preflight drift")
    verify_record(
        preflight.get("run_config", {}),
        training_root / "RUN_CONFIG.json",
        "preflight RUN_CONFIG",
    )
    verify_record(
        preflight.get("input_lineage_audit", {}),
        training_root / "INPUT_LINEAGE_AUDIT.json",
        "preflight input lineage",
    )
    trainer_record = preflight.get("trainer_script")
    if not isinstance(trainer_record, Mapping):
        raise IndependentAuditError("preflight trainer record is absent")
    trainer_path = Path(str(trainer_record.get("path", "")))
    verified_trainer = verify_record(
        trainer_record, trainer_path, "preflight trainer script"
    )
    full_input = preflight.get("full_input_contract")
    if not isinstance(full_input, Mapping):
        raise IndependentAuditError("launch preflight lacks full input contract")
    if (
        full_input.get("materialization_closure_status") != FULL_MATERIALIZATION_STATUS
        or _resolved(
            Path(str(full_input.get("materialization_validation_path", "")))
        )
        != _resolved(material_validation)
        or str(full_input.get("materialization_validation_sha256", ""))
        != sha256(material_validation)
        or _resolved(Path(str(full_input.get("preprocessing_audit_path", ""))))
        != _resolved(preprocessing_audit)
        or str(full_input.get("preprocessing_audit_sha256", ""))
        != sha256(preprocessing_audit)
        or sorted(
            str(value).upper()
            for value in full_input.get("materialization_expected_cancers", [])
        )
        != input_evidence["expected_cancers"]
        or int(full_input.get("preprocessing_audit_eligible_cells", -1))
        != input_evidence["eligible_training_cells"]
    ):
        raise IndependentAuditError("launch full-input binding drift")

    if (
        config.get("format") != CONFIG_FORMAT
        or config.get("target_host") != "149"
        or config.get("allow_partial_materialization") is not False
        or config.get("fresh_random_initialization") is not True
        or config.get("old_checkpoint_loaded") is not False
        or config.get("source_checkpoint_sha256") is not None
    ):
        raise IndependentAuditError("RUN_CONFIG format drift")
    settings = _settings(config)
    required_settings = (
        "seed", "epochs", "patience", "batch_size", "max_train_rows",
        "learning_rate",
    )
    if any(name not in settings for name in required_settings):
        raise IndependentAuditError("RUN_CONFIG lacks training hyperparameters")
    expected_seeds = [
        int(settings["seed"]) + fold * 101 for fold in range(N_FOLDS)
    ]
    if (
        int(settings.get("base_seed", -1)) != int(settings["seed"])
        or list(map(int, config.get("fold_seeds", []))) != expected_seeds
        or int(config.get("outer_folds", -1)) != N_FOLDS
        or list(config.get("numeric_feature_columns", [])) != FEATURE_COLUMNS
        or config.get("device") != "cpu"
        or config.get("cuda_used") is not False
        or config.get("old_checkpoint_loaded") is not False
        or int(settings["epochs"]) <= 0
        or int(settings["patience"]) <= 0
        or int(settings["batch_size"]) <= 0
        or int(settings["max_train_rows"]) <= 0
        or not math.isfinite(float(settings["learning_rate"]))
        or float(settings["learning_rate"]) <= 0
    ):
        raise IndependentAuditError("RUN_CONFIG model/runtime contract drift")
    architecture = config.get("architecture")
    if (
        not isinstance(architecture, Mapping)
        or int(architecture.get("frozen_lnc_core_dimensions", -1)) != 96
        or int(architecture.get("frozen_pathway_core_dimensions", -1)) != 96
        or int(architecture.get("frozen_cancer_core_dimensions", -1)) != 96
        or int(architecture.get("private_lnc_embedding_dimensions", -1)) != 32
        or int(architecture.get("celltype_embedding_dimensions", -1)) != 16
        or int(architecture.get("core_availability_flag_dimensions", -1)) != 1
        or int(architecture.get("input_dimensions", -1)) != 345
    ):
        raise IndependentAuditError("RUN_CONFIG architecture drift")
    thresholds = config.get("discovery_thresholds")
    if not isinstance(thresholds, Mapping) or {
        "min_donors": thresholds.get("min_donors"),
        "min_outer_folds": thresholds.get("min_outer_folds"),
        "max_fdr": thresholds.get("max_fdr"),
        "min_abs_meta_rho": thresholds.get("min_abs_meta_rho"),
        "min_direction_consistency": thresholds.get("min_direction_consistency"),
    } != {
        "min_donors": 3,
        "min_outer_folds": 2,
        "max_fdr": 0.05,
        "min_abs_meta_rho": 0.2,
        "min_direction_consistency": 0.75,
    }:
        raise IndependentAuditError("RUN_CONFIG discovery thresholds drift")

    if (
        inputs.get("format") != INPUT_LINEAGE_FORMAT
        or inputs.get("status") != "PASS"
        or inputs.get("scope") != FULL_INPUT_SCOPE
        or inputs.get("target_host") != "149"
        or _resolved(Path(str(inputs.get("materialization_validation_path", ""))))
        != _resolved(material_validation)
        or str(inputs.get("materialization_validation_sha256", ""))
        != sha256(material_validation)
        or _resolved(Path(str(inputs.get("preprocessing_audit_path", ""))))
        != _resolved(preprocessing_audit)
        or str(inputs.get("preprocessing_audit_sha256", ""))
        != sha256(preprocessing_audit)
        or _resolved(Path(str(inputs.get("core_manifest_path", ""))))
        != _resolved(core_manifest)
        or str(inputs.get("core_manifest_sha256", "")) != sha256(core_manifest)
        or inputs.get("full_input_contract") != full_input
        or inputs.get("old_checkpoint_loaded") is not False
        or inputs.get("old_predictions_loaded") is not False
        or inputs.get("old_rankings_loaded") is not False
        or inputs.get("old_training_outputs_reused") is not False
    ):
        raise IndependentAuditError("input lineage audit drift")
    materialization_record = inputs.get("materialization_validation")
    preprocessing_record = inputs.get("preprocessing_audit")
    core_record = inputs.get("core_manifest")
    verify_record(
        materialization_record, material_validation, "input materialization validation"
    )
    verify_record(
        preprocessing_record, preprocessing_audit, "input preprocessing audit"
    )
    verify_record(core_record, core_manifest, "input core manifest")
    if (
        materialization_record.get("status") != "PASS"
        or materialization_record.get("closure_status")
        != FULL_MATERIALIZATION_STATUS
        or materialization_record.get("validation_scope") != "FULL"
        or int(materialization_record.get("expected_cancer_count", -1)) != 28
        or int(materialization_record.get("validated_cancer_count", -1)) != 28
        or preprocessing_record.get("status") != "PASS"
        or int(preprocessing_record.get("active_cancer_count", -1)) != 28
        or int(preprocessing_record.get("eligible_training_cells", -1))
        != input_evidence["eligible_training_cells"]
        or core_record.get("status") != "PASS_FRESH_FROZEN_CORE_EMBEDDINGS"
        or core_record.get("all_embeddings_from_newly_trained_v32_core") is not True
        or core_record.get("historical_checkpoint_loaded") is not False
        or core_record.get("historical_prediction_loaded") is not False
    ):
        raise IndependentAuditError("input lineage evidence semantics drift")

    records = manifest.get("records")
    if (
        manifest.get("format") != CHECKPOINT_MANIFEST_FORMAT
        or manifest.get("status") != "SUCCESS"
        or manifest.get("target_host") != "149"
        or int(manifest.get("outer_folds", -1)) != N_FOLDS
        or int(manifest.get("models_trained", -1)) != N_FOLDS
        or list(map(int, manifest.get("fold_seeds", []))) != expected_seeds
        or manifest.get("fresh_random_initialization") is not True
        or manifest.get("old_checkpoint_loaded") is not False
        or manifest.get("source_checkpoint_sha256") is not None
        or manifest.get("device") != "cpu"
        or manifest.get("cuda_used") is not False
        or not isinstance(records, list)
        or len(records) != N_FOLDS
    ):
        raise IndependentAuditError("checkpoint manifest closure drift")
    records_by_fold = {int(row.get("outer_fold_id", -1)): row for row in records}
    if set(records_by_fold) != set(range(N_FOLDS)):
        raise IndependentAuditError("checkpoint manifest fold identity drift")

    required_false = (
        "old_checkpoint_loaded",
        "old_predictions_used_as_features",
        "old_rankings_used_as_outputs",
        "old_family_support_used",
    )
    if (
        lineage.get("format") != LINEAGE_FORMAT
        or lineage.get("status") != "SUCCESS"
        or lineage.get("scope") != FULL_LINEAGE_SCOPE
        or lineage.get("target_host") != "149"
        or lineage.get("cpu_only") is not True
        or lineage.get("paid_compute") is not False
        or lineage.get("cuda_used") is not False
        or lineage.get("device") != "cpu"
        or int(lineage.get("models_trained", -1)) != N_FOLDS
        or list(map(int, lineage.get("seeds", []))) != expected_seeds
        or list(map(int, lineage.get("fold_seeds", []))) != expected_seeds
        or not str(lineage.get("training_run_id", ""))
        or any(lineage.get(name) is not False for name in required_false)
        or lineage.get("core_parameters_frozen") is not True
        or lineage.get("fresh_random_initialization") is not True
        or lineage.get("source_checkpoint_sha256") is not None
    ):
        raise IndependentAuditError("training lineage semantic drift")
    bound_hashes = {
        "launch_preflight_sha256": training_root / "LAUNCH_PREFLIGHT.json",
        "run_config_sha256": training_root / "RUN_CONFIG.json",
        "input_lineage_audit_sha256": training_root / "INPUT_LINEAGE_AUDIT.json",
        "checkpoint_manifest_sha256": training_root / "CHECKPOINT_MANIFEST.json",
    }
    for field, path in bound_hashes.items():
        if str(lineage.get(field, "")) != sha256(path):
            raise IndependentAuditError(f"lineage {field} drift")
    if (
        _resolved(Path(str(lineage.get("trainer_path", ""))))
        != _resolved(trainer_path)
        or str(lineage.get("trainer_sha256", "")) != verified_trainer["sha256"]
        or str(lineage.get("trainer_script_sha256", ""))
        != verified_trainer["sha256"]
    ):
        raise IndependentAuditError("lineage trainer hash drift")
    verify_record(lineage.get("trainer_script", {}), trainer_path, "lineage trainer")
    for field, name in (
        ("launch_preflight", "LAUNCH_PREFLIGHT.json"),
        ("run_config", "RUN_CONFIG.json"),
        ("input_lineage_audit", "INPUT_LINEAGE_AUDIT.json"),
        ("checkpoint_manifest", "CHECKPOINT_MANIFEST.json"),
        ("contract_validation", "FULL_MODEL_CONTRACT_VALIDATION.json"),
    ):
        verify_record(lineage.get(field, {}), training_root / name, f"lineage {field}")

    success_files = success.get("files")
    lineage_outputs = lineage.get("outputs")
    if not isinstance(success_files, Mapping) or not isinstance(lineage_outputs, Mapping):
        raise IndependentAuditError("SUCCESS/LINEAGE output records are absent")
    verify_artifact_manifest(
        success_files, set(TOP_LEVEL_ARTIFACTS), training_root, "SUCCESS files"
    )
    verify_artifact_manifest(
        lineage_outputs, set(RESULT_ARTIFACTS), training_root, "LINEAGE outputs"
    )
    verify_artifact_manifest(
        lineage.get("output_artifacts_before_lineage"),
        _pre_lineage_artifact_names(),
        training_root,
        "LINEAGE pre-lineage manifest",
    )
    closure_files = success.get("required_closure_files")
    if not isinstance(closure_files, Mapping) or set(closure_files) != set(
        CLOSURE_ARTIFACTS
    ):
        raise IndependentAuditError("SUCCESS required closure artifact set drift")
    for name in CLOSURE_ARTIFACTS:
        verify_record(
            closure_files[name], training_root / name, f"SUCCESS closure/{name}"
        )
        if closure_files[name] != success_files[name]:
            raise IndependentAuditError(f"SUCCESS closure record differs for {name}")
    if str(success.get("lineage_sha256", "")) != sha256(training_root / "LINEAGE.json"):
        raise IndependentAuditError("SUCCESS does not bind LINEAGE")
    if (
        success.get("format") != TRAINING_FORMAT
        or success.get("status") != "SUCCESS"
        or success.get("scope") != FULL_TRAINING_SCOPE
        or success.get("target_host") != "149"
        or success.get("state")
        != "FIVE_FOLD_DONOR_BLOCKED_METACELL_TRAINING_COMPLETE"
        or int(success.get("folds", -1)) != N_FOLDS
        or int(success.get("models_trained", -1)) != N_FOLDS
        or list(map(int, success.get("fold_seeds", []))) != expected_seeds
        or success.get("device") != "cpu"
        or success.get("cuda_used") is not False
        or success.get("fresh_random_initialization") is not True
        or success.get("old_checkpoint_loaded") is not False
        or success.get("source_checkpoint_sha256") is not None
    ):
        raise IndependentAuditError("training SUCCESS identity drift")
    if (
        contract.get("format") != TRAINING_FORMAT
        or contract.get("target_host") != "149"
        or contract.get("status") != "PASS_TECHNICAL_BASELINE"
        or int(contract.get("outer_folds", -1)) != N_FOLDS
        or contract.get("signed_target") is not True
        or contract.get("donor_blocked") is not True
        or contract.get("metacell_resolved") is not True
        or contract.get("core_detached_and_frozen") is not True
        or contract.get("oov_private_embedding_enabled") is not True
        or contract.get("fresh_random_initialization") is not True
        or contract.get("old_checkpoint_loaded") is not False
        or contract.get("source_checkpoint_sha256") is not None
        or contract.get("device") != "cpu"
        or contract.get("cuda_used") is not False
    ):
        raise IndependentAuditError("self contract validation drift")
    for field, name in (
        ("run_config", "RUN_CONFIG.json"),
        ("input_lineage_audit", "INPUT_LINEAGE_AUDIT.json"),
        ("checkpoint_manifest", "CHECKPOINT_MANIFEST.json"),
    ):
        verify_record(contract.get(field, {}), training_root / name, f"contract {field}")
    contract_folds = contract.get("folds")
    if not isinstance(contract_folds, list) or len(contract_folds) != N_FOLDS:
        raise IndependentAuditError("contract fold reports are absent")
    contract_by_fold = {
        int(record.get("outer_fold_id", -1)): record for record in contract_folds
    }
    if set(contract_by_fold) != set(range(N_FOLDS)):
        raise IndependentAuditError("contract fold report identity drift")
    for fold in range(N_FOLDS):
        receipt = read_json(
            training_root / f"outer_fold={fold}" / "SUCCESS.json",
            f"fold {fold} SUCCESS",
        )
        if contract_by_fold[fold] != receipt:
            raise IndependentAuditError(f"contract fold {fold} differs from receipt")
    return documents, {
        "settings": dict(settings),
        "expected_seeds": expected_seeds,
        "checkpoint_records": records_by_fold,
        "trainer_path": str(trainer_path),
        "trainer_sha256": verified_trainer["sha256"],
    }


def snapshot(paths: list[Path]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted({_resolved(path) for path in paths}, key=str):
        if not path.is_file() or path.is_symlink():
            raise IndependentAuditError(f"snapshot artifact is missing or unsafe: {path}")
        result[str(path)] = {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
    return result


def run_independent_audit(
    *,
    training_root: Path,
    materialization_root: Path,
    core_root: Path,
    output: Path,
    expected_training_run_id: str | None,
) -> dict[str, Any]:
    if socket.gethostname() != "149":
        raise IndependentAuditError("independent audit must execute on host 149")
    if torch.cuda.is_available():
        raise IndependentAuditError("independent audit requires the host149 CPU runtime")
    roots = [training_root, materialization_root, core_root]
    if any(output == root or root in output.parents for root in roots):
        raise IndependentAuditError("audit output must be outside authoritative roots")
    if output.exists():
        raise IndependentAuditError(f"audit output already exists: {output}")
    output.mkdir(parents=True)
    started = datetime.now(timezone.utc).isoformat()

    authoritative_paths = [training_root / name for name in TOP_LEVEL_ARTIFACTS]
    authoritative_paths.append(training_root / "SUCCESS.json")
    authoritative_paths.extend(
        [
            materialization_root / "MATERIALIZATION_VALIDATION.json",
            materialization_root / "SINGLE_CELL_PREPROCESSING_AUDIT.json",
            core_root / "CORE_EMBEDDING_MANIFEST.json",
        ]
    )
    for fold in range(N_FOLDS):
        fold_root = training_root / f"outer_fold={fold}"
        authoritative_paths.extend(
            [
                fold_root / "SUCCESS.json",
                fold_root / "best_model_state.pt",
                fold_root / "QUERY_PREDICTIONS.parquet",
            ]
        )
        core_fold = core_root / f"patient_fold={fold}"
        authoritative_paths.extend(
            [
                core_fold / "lncRNA.parquet",
                core_fold / "pathway.parquet",
                core_fold / "cancer.parquet",
            ]
        )
    source_before = snapshot(authoritative_paths)
    documents, closure = validate_lineage_closure(
        training_root, materialization_root, core_root
    )
    lineage = documents["LINEAGE.json"]
    success = documents["SUCCESS.json"]
    contract = documents["FULL_MODEL_CONTRACT_VALIDATION.json"]
    manifest_records = closure["checkpoint_records"]
    settings = closure["settings"]
    training_run_id = str(lineage.get("training_run_id", ""))
    if not training_run_id:
        raise IndependentAuditError("training lineage lacks training_run_id")
    if expected_training_run_id and training_run_id != expected_training_run_id:
        raise IndependentAuditError("training_run_id differs from the expected run")

    fold_receipts: dict[int, Mapping[str, Any]] = {}
    checkpoint_audit = []
    for fold in range(N_FOLDS):
        fold_root = training_root / f"outer_fold={fold}"
        receipt_path = fold_root / "SUCCESS.json"
        receipt = read_json(receipt_path, f"fold {fold} SUCCESS")
        if (
            receipt.get("format") != TRAINING_FORMAT
            or receipt.get("target_host") != "149"
            or receipt.get("state") != "FOLD_TRAINED_AND_OOF_PREDICTED"
            or int(receipt.get("outer_fold_id", -1)) != fold
        ):
            raise IndependentAuditError(f"fold {fold} SUCCESS identity drift")
        manifest_record = manifest_records[fold]
        verify_record(
            manifest_record.get("fold_success", {}),
            receipt_path,
            f"fold {fold} SUCCESS manifest",
        )
        checkpoint_audit.append(
            audit_checkpoint(
                fold=fold,
                checkpoint_path=fold_root / "best_model_state.pt",
                fold_receipt=receipt,
                manifest_record=manifest_record,
                core_root=core_root,
                base_seed=int(settings["seed"]),
                epochs=int(settings["epochs"]),
                patience=int(settings["patience"]),
            )
        )
        fold_receipts[fold] = receipt
    initial_hashes = {
        row["initial_parameter_sha256"] for row in checkpoint_audit
    }
    final_hashes = {row["trained_parameter_sha256"] for row in checkpoint_audit}
    if len(initial_hashes) != N_FOLDS or len(final_hashes) != N_FOLDS:
        raise IndependentAuditError("five folds do not have independent model states")

    predictions, oof_audit = audit_oof_isolation(
        training_root,
        materialization_root,
        fold_receipts,
        require_receipt_hash_closure=True,
    )
    oof_audit.update(validate_embedding_routes(predictions))
    if not np.allclose(
        predictions.predicted_association_strength.to_numpy(float),
        predictions.predicted_rho.abs().to_numpy(float),
        rtol=0.0,
        atol=1e-7,
    ):
        raise IndependentAuditError("predicted association-strength sidecar drift")

    recomputed_associations = independently_aggregate_oof(predictions)
    stored_associations = pd.read_parquet(
        training_root / "CROSSFIT_ASSOCIATIONS.parquet"
    )
    assert_frame_equal_by_key(
        stored_associations,
        recomputed_associations,
        PAIR_KEY,
        "cross-fit associations",
    )
    recomputed_discoveries = recomputed_associations.loc[
        recomputed_associations.new_relationship_discovered
    ].copy()
    stored_discoveries = pd.read_parquet(
        training_root / "DISCOVERED_NEW_RELATIONSHIPS.parquet"
    )
    assert_frame_equal_by_key(
        stored_discoveries,
        recomputed_discoveries,
        PAIR_KEY,
        "new relationship discoveries",
    )

    counts = {
        "oof_rows": len(predictions),
        "association_rows": len(recomputed_associations),
        "new_relationships": len(recomputed_discoveries),
    }
    for field, observed in counts.items():
        if int(success.get(field, -1)) != observed or int(contract.get(field, -1)) != observed:
            raise IndependentAuditError(f"training aggregate count drift: {field}")
    if int(contract.get("oof_duplicate_keys", -1)) != 0:
        raise IndependentAuditError("self contract reports duplicate OOF keys")
    if int(success.get("folds", -1)) != N_FOLDS:
        raise IndependentAuditError("SUCCESS fold count drift")

    materialization_hash_recheck = validate_materialization_training_file_hashes(
        materialization_root
    )
    if materialization_hash_recheck != oof_audit["materialization_hash_closure"]:
        raise IndependentAuditError("materialization input closure changed during audit")
    source_after = snapshot(authoritative_paths)
    if source_before != source_after:
        raise IndependentAuditError("authoritative artifacts changed during audit")
    report = {
        "format": AUDIT_FORMAT,
        "status": "PASS",
        "fail_closed": True,
        "audit_id": f"{training_run_id}-INDEPENDENT-AUDIT",
        "training_run_id": training_run_id,
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "independence": {
            "training_invoked": False,
            "trainer_module_imported": False,
            "trainer_helpers_called": False,
            "training_artifacts_modified": False,
            "validation_method": (
                "INDEPENDENT_CHECKPOINT_RECONSTRUCTION_DONOR_OOF_AND_"
                "CROSSFIT_AGGREGATE_RECOMPUTATION"
            ),
        },
        "runtime": {
            "target_host": "149",
            "device": "cpu",
            "torch_version": torch.__version__,
            "cuda_available": False,
            "paid_compute": False,
        },
        "paths": {
            "training_root": str(training_root),
            "materialization_root": str(materialization_root),
            "core_root": str(core_root),
        },
        "lineage_closure": {
            "trainer_path": closure["trainer_path"],
            "trainer_sha256": closure["trainer_sha256"],
            "launch_preflight_sha256": sha256(training_root / "LAUNCH_PREFLIGHT.json"),
            "run_config_sha256": sha256(training_root / "RUN_CONFIG.json"),
            "input_lineage_audit_sha256": sha256(
                training_root / "INPUT_LINEAGE_AUDIT.json"
            ),
            "checkpoint_manifest_sha256": sha256(
                training_root / "CHECKPOINT_MANIFEST.json"
            ),
            "training_lineage_sha256": sha256(training_root / "LINEAGE.json"),
            "training_success_sha256": sha256(training_root / "SUCCESS.json"),
            "old_artifact_flags_all_false": True,
        },
        "checkpoint_audit": checkpoint_audit,
        "oof_audit": oof_audit,
        "aggregation_audit": {
            **counts,
            "stored_associations_sha256": sha256(
                training_root / "CROSSFIT_ASSOCIATIONS.parquet"
            ),
            "stored_discoveries_sha256": sha256(
                training_root / "DISCOVERED_NEW_RELATIONSHIPS.parquet"
            ),
            "bh_family": "CANCER_X_CELLTYPE_CROSSFIT_SELECTED_PAIRS",
            "discovery_thresholds": {
                "min_donors": 3,
                "min_outer_folds": 2,
                "max_fdr": 0.05,
                "min_abs_meta_rho": 0.2,
                "min_direction_consistency": 0.75,
            },
            "exactly_reproduced": True,
        },
        "authoritative_snapshot_before": source_before,
        "authoritative_snapshot_after": source_after,
    }
    report_path = output / "INDEPENDENT_AUDIT.json"
    atomic_json(report_path, report)
    binding = {
        "format": BINDING_FORMAT,
        "status": "PASS",
        "fail_closed": True,
        "training_run_id": training_run_id,
        "audit_report_path": str(report_path),
        "audit_report_sha256": sha256(report_path),
        "auditor_path": str(Path(__file__).resolve()),
        "auditor_sha256": sha256(Path(__file__).resolve()),
        "authoritative_snapshot_sha256": canonical_hash(source_after),
        "checkpoint_final_parameter_sha256": {
            str(row["outer_fold_id"]): row["trained_parameter_sha256"]
            for row in checkpoint_audit
        },
        "release_decision": {
            "formal_training_closure_accepted": True,
            "pointer_promotion_authorized": False,
        },
    }
    binding_path = output / "INDEPENDENT_AUDIT_BINDING.json"
    atomic_json(binding_path, binding)
    completion = {
        "format": BINDING_FORMAT,
        "status": "PASS",
        "training_run_id": training_run_id,
        "audit_report_path": str(report_path),
        "audit_report_sha256": sha256(report_path),
        "binding_path": str(binding_path),
        "binding_sha256": sha256(binding_path),
        "all_checks_passed": True,
        "pointer_updated": False,
    }
    atomic_json(output / "SUCCESS.json", completion)
    return completion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-root", required=True, type=Path)
    parser.add_argument("--materialization-root", required=True, type=Path)
    parser.add_argument("--core-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-training-run-id")
    args = parser.parse_args()
    output = _resolved(args.output)
    if output.exists():
        raise SystemExit("OUTPUT_ALREADY_EXISTS")
    try:
        result = run_independent_audit(
            training_root=_resolved(args.training_root),
            materialization_root=_resolved(args.materialization_root),
            core_root=_resolved(args.core_root),
            output=output,
            expected_training_run_id=args.expected_training_run_id,
        )
    except Exception as error:
        if not output.exists():
            output.mkdir(parents=True)
        atomic_json(
            output / "FAILURE.json",
            {
                "format": AUDIT_FORMAT,
                "status": "FAIL",
                "fail_closed": True,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
