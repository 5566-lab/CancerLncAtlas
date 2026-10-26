#!/usr/bin/env python3
"""Lazy five-fold donor-isolated SC-HHGT training on host 149.

The optimizer consumes hash-bound episode assets prepared ahead of time. Graph
episodes are loaded one at a time; shared cancer-static membership and external
edges are referenced rather than copied into every episode. Importing this
module never starts training.
"""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import random
import socket
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from cc_hhgt.v32.single_cell_hhgt import (
    LASSO_EDGE_TYPES,
    NODE_TYPES,
    ContextGraph,
    SingleCellHHGT,
    build_context_graph,
)


GRAPH_ASSET_FORMAT = "CANCERLNCATLAS_V32_SC_HHGT_GRAPH_ASSETS_V1"
STATIC_FORMAT = "CANCERLNCATLAS_V32_SC_HHGT_EPISODE_STATIC_V1"
EPISODE_MANIFEST_FORMAT = "CANCERLNCATLAS_V32_SC_HHGT_EPISODE_MANIFEST_V1"
EPISODE_FORMAT = "CANCERLNCATLAS_V32_SC_HHGT_EPISODE_V1"
TRAINING_FORMAT = "CANCERLNCATLAS_V32_SC_HHGT_TRAINING_V1"
CONFIG_FORMAT = "CANCERLNCATLAS_V32_SC_HHGT_RUN_CONFIG_V1"
LINEAGE_FORMAT = "CANCERLNCATLAS_V32_SC_HHGT_LINEAGE_V1"
CHECKPOINT_FORMAT = "CANCERLNCATLAS_V32_SC_HHGT_CHECKPOINT_V1"
CHECKPOINT_MANIFEST_FORMAT = "CANCERLNCATLAS_V32_SC_HHGT_CHECKPOINT_MANIFEST_V1"
SUCCESS_FORMAT = "CANCERLNCATLAS_V32_SC_HHGT_SUCCESS_V1"
N_OUTER_FOLDS = 5
ARMS = ("with_lasso", "without_lasso")
ROLES = ("fit", "validation", "query")
ROW_ID_COLUMNS = (
    "dataset_id", "cancer_id", "cell_key", "query_donor_id",
    "lncrna_id", "pathway_id",
)
SOURCE_BINDING_FIELDS = (
    "feature_source_donors", "lasso_source_donors", "pair_feature_source_donors",
)
STATIC_DUPLICATE_FIELDS = (
    "gene_pathway_membership", "pathway_family_membership",
    "external_relation_types", "external_edge_batches",
)


@dataclass(frozen=True)
class TrainingSettings:
    episode_root: Path
    output_root: Path
    graph_assets_root: Path | None = None
    seed: int = 20261005
    epochs: int = 40
    patience: int = 8
    minimum_validation_improvement: float = 0.0
    prediction_batch_size: int = 2048
    max_rows_per_episode_step: int = 2048
    max_fit_episodes_per_epoch: int = 64
    training_workers: int = 1
    threads_per_worker: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_channels: int = 64
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    max_gradient_norm: float = 5.0
    prepare_only: bool = False


@dataclass(frozen=True)
class EpisodeDescriptor:
    path: Path
    record: dict[str, Any]
    static_path: Path
    static_record: dict[str, Any]
    graph_asset_static_record: dict[str, Any]
    matrix_identity_record: dict[str, Any]
    episode_id: str
    cancer_id: str
    outer_fold_id: int
    role: str
    row_count: int
    pair_feature_count: int


@dataclass
class LoadedEpisode:
    descriptor: EpisodeDescriptor
    inner_fold_id: int | None
    cell_key: str
    label_donors: tuple[str, ...]
    source_donors: tuple[str, ...]
    excluded_donors: tuple[str, ...]
    outer_train_donors: tuple[str, ...]
    outer_query_donors: tuple[str, ...]
    inner_target_donors: tuple[str, ...]
    validation_donors: tuple[str, ...]
    geometry_isolation: str
    graph: ContextGraph
    lnc_index: torch.Tensor
    pathway_index: torch.Tensor
    pair_features: torch.Tensor
    targets: torch.Tensor
    rows: pd.DataFrame
    pair_feature_names: tuple[str, ...]
    lasso_pair_feature_mask: tuple[bool, ...]

    def audit(self, arm: str) -> dict[str, Any]:
        lasso_edges = sum(
            int(self.graph.data[edge_type].edge_index.shape[1])
            for edge_type in LASSO_EDGE_TYPES
        )
        external_edges = sum(
            int(self.graph.data[edge_type].edge_index.shape[1])
            for edge_type in self.graph.edge_schema
            if edge_type[1].startswith("external_")
        )
        mask = torch.as_tensor(self.lasso_pair_feature_mask, dtype=torch.bool)
        lasso_nonzero = (
            int(torch.count_nonzero(self.pair_features[:, mask])) if bool(mask.any()) else 0
        )
        return {
            "episode_id": self.descriptor.episode_id,
            "role": self.descriptor.role,
            "cancer_id": self.descriptor.cancer_id,
            "outer_fold_id": self.descriptor.outer_fold_id,
            "inner_fold_id": self.inner_fold_id,
            "cell_key": self.cell_key,
            "label_donors": list(self.label_donors),
            "source_donors": list(self.source_donors),
            "excluded_donors": list(self.excluded_donors),
            "outer_train_donors": list(self.outer_train_donors),
            "outer_query_donors": list(self.outer_query_donors),
            "inner_target_donors": list(self.inner_target_donors),
            "validation_donors": list(self.validation_donors),
            "validation_provider": bool(self.descriptor.role == "validation"),
            "strict_validation_bound": bool(
                self.descriptor.role == "fit" and self.validation_donors
            ),
            "support_limited_fit": bool(
                self.descriptor.role == "fit" and not self.validation_donors
            ),
            "geometry_isolation": self.geometry_isolation,
            "row_count": len(self.rows),
            "edge_schema_sha256": canonical_hash(
                [list(value) for value in self.graph.edge_schema]
            ),
            "pair_feature_schema_sha256": canonical_hash(
                {"names": self.pair_feature_names, "lasso_mask": self.lasso_pair_feature_mask}
            ),
            "lasso_pair_feature_columns": int(sum(self.lasso_pair_feature_mask)),
            "lasso_pair_feature_nonzero_values": lasso_nonzero,
            "lasso_edge_count": lasso_edges,
            "external_edge_count": external_edges,
            "episode_asset": self.descriptor.record,
            "static_asset": self.descriptor.static_record,
            "graph_asset_static_success": self.descriptor.graph_asset_static_record,
            "matrix_identity": self.descriptor.matrix_identity_record,
        }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    torch.save(dict(value), temporary)
    os.replace(temporary, path)


def read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"missing or unsafe {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} is not a JSON object")
    return value


def require_host_149(hostname: str | None = None) -> str:
    observed = str(hostname if hostname is not None else socket.gethostname()).strip()
    if observed != "149":
        raise RuntimeError(f"SC_HHGT_TRAINING_REQUIRES_HOST_149: observed={observed}")
    return observed


def _path_from_record(record: Mapping[str, Any], base: Path) -> Path:
    raw = str(record.get("path", "")).strip()
    if not raw:
        raise RuntimeError("file record lacks path")
    path = Path(raw).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def verify_file_record(
    record: Mapping[str, Any], *, base: Path, label: str,
    require_within: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    if not isinstance(record, Mapping):
        raise RuntimeError(f"{label} record is not an object")
    path = _path_from_record(record, base)
    if require_within is not None:
        try:
            path.relative_to(require_within.resolve())
        except ValueError as exc:
            raise RuntimeError(f"{label} escapes its authorized root") from exc
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{label} is missing or symlinked: {path}")
    digest = sha256(path)
    if digest != str(record.get("sha256", "")):
        raise RuntimeError(f"{label} hash drift")
    byte_count = path.stat().st_size
    if int(record.get("bytes", -1)) != byte_count:
        raise RuntimeError(f"{label} byte count drift")
    return path, {"path": str(path), "bytes": byte_count, "sha256": digest}


def verify_file_record_metadata(
    record: Mapping[str, Any], *, base: Path, label: str,
    require_within: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Verify an already-bound large asset without streaming it for hashing."""
    if not isinstance(record, Mapping):
        raise RuntimeError(f"{label} record is not an object")
    path = _path_from_record(record, base)
    if require_within is not None:
        try:
            path.relative_to(require_within.resolve())
        except ValueError as exc:
            raise RuntimeError(f"{label} escapes its authorized root") from exc
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{label} is missing or symlinked: {path}")
    byte_count = path.stat().st_size
    if int(record.get("bytes", -1)) != byte_count:
        raise RuntimeError(f"{label} byte count drift")
    digest = str(record.get("sha256", ""))
    if len(digest) != 64:
        raise RuntimeError(f"{label} receipt hash is invalid")
    return path, {"path": str(path), "bytes": byte_count, "sha256": digest}


def _load_weights_only(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise RuntimeError(f"weights-only asset is not a mapping: {path}")
    return value


def _string_tuple(values: Iterable[Any], label: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise RuntimeError(f"{label} must be an array")
    result = tuple(sorted({str(value).strip() for value in values}))
    if (not allow_empty and not result) or "" in result:
        raise RuntimeError(f"{label} is empty or invalid")
    return result


def _state_hash(state: Mapping[str, torch.Tensor], prefix: str | None = None) -> str:
    values = {k: v for k, v in state.items() if prefix is None or k.startswith(prefix)}
    if not values:
        raise RuntimeError(f"empty state for prefix {prefix!r}")
    digest = hashlib.sha256()
    for name in sorted(values):
        array = values[name].detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _gradient_l1(module: nn.Module) -> float:
    gradients = [
        parameter.grad.detach().abs().sum()
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    if not gradients:
        return 0.0
    value = torch.stack(gradients).sum()
    if not bool(torch.isfinite(value)):
        raise RuntimeError("SC_NONFINITE_GRADIENT")
    return float(value)


def _coo_localized(
    value: Mapping[str, Any], row_global: torch.Tensor, column_global: torch.Tensor,
    label: str,
) -> dict[str, Any]:
    if set(value) != {"row_index", "column_index", "weight", "shape"}:
        raise RuntimeError(f"{label} shared COO schema failed")
    shape = tuple(int(number) for number in value["shape"])
    row = torch.as_tensor(value["row_index"], dtype=torch.long)
    column = torch.as_tensor(value["column_index"], dtype=torch.long)
    weight = torch.as_tensor(value["weight"], dtype=torch.float32)
    if (
        len(shape) != 2 or min(shape) <= 0
        or row.ndim != 1 or column.shape != row.shape or weight.shape != row.shape
        or not bool(torch.isfinite(weight).all())
        or bool((row < 0).any()) or bool((column < 0).any())
        or (row.numel() and int(row.max()) >= shape[0])
        or (column.numel() and int(column.max()) >= shape[1])
    ):
        raise RuntimeError(f"{label} shared COO shape failed")
    row_global = torch.as_tensor(row_global, dtype=torch.long)
    column_global = torch.as_tensor(column_global, dtype=torch.long)
    if (
        row_global.ndim != 1 or column_global.ndim != 1
        or torch.unique(row_global).numel() != row_global.numel()
        or torch.unique(column_global).numel() != column_global.numel()
        or bool((row_global < 0).any()) or bool((column_global < 0).any())
        or (row_global.numel() and int(row_global.max()) >= shape[0])
        or (column_global.numel() and int(column_global.max()) >= shape[1])
    ):
        raise RuntimeError(f"{label} node global identity failed")
    row_lookup = torch.full((shape[0],), -1, dtype=torch.long)
    column_lookup = torch.full((shape[1],), -1, dtype=torch.long)
    row_lookup[row_global] = torch.arange(row_global.numel())
    column_lookup[column_global] = torch.arange(column_global.numel())
    local_row = row_lookup[row]
    local_column = column_lookup[column]
    keep = (local_row >= 0) & (local_column >= 0)
    return {
        "row_index": local_row[keep],
        "column_index": local_column[keep],
        "weight": weight[keep],
        "shape": [row_global.numel(), column_global.numel()],
    }


def _external_localized(
    batches: list[Mapping[str, Any]], node_global: Mapping[str, torch.Tensor],
    node_counts: Mapping[str, int],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    lookup: dict[str, torch.Tensor] = {}
    for node_type in NODE_TYPES[:-1]:
        values = torch.as_tensor(node_global[node_type], dtype=torch.long)
        if (
            values.ndim != 1 or not values.numel() or bool((values < 0).any())
            or torch.unique(values).numel() != values.numel()
            or int(values.max()) >= node_counts[node_type]
        ):
            raise RuntimeError("episode node_global_indices are invalid")
        maximum = int(values.max()) + 1
        local = torch.full((maximum,), -1, dtype=torch.long)
        local[values] = torch.arange(values.numel())
        lookup[node_type] = local
    for batch in batches:
        source_type = str(batch.get("source_type", ""))
        target_type = str(batch.get("target_type", ""))
        if source_type not in lookup or target_type not in lookup:
            raise RuntimeError("static external node type is unsupported")
        index = torch.as_tensor(batch.get("edge_index"), dtype=torch.long)
        weight = torch.as_tensor(batch.get("edge_weight"), dtype=torch.float32)
        if index.ndim != 2 or index.shape[0] != 2 or index.shape[1] != weight.numel():
            raise RuntimeError("static external edge batch shape failed")
        if (
            not bool(torch.isfinite(weight).all())
            or bool((index < 0).any())
            or (index.shape[1] and int(index[0].max()) >= node_counts[source_type])
            or (index.shape[1] and int(index[1].max()) >= node_counts[target_type])
        ):
            raise RuntimeError("static external edge batch global identity failed")
        source_lookup, target_lookup = lookup[source_type], lookup[target_type]
        source_valid = (index[0] >= 0) & (index[0] < source_lookup.numel())
        target_valid = (index[1] >= 0) & (index[1] < target_lookup.numel())
        keep = source_valid & target_valid
        candidate_source = index[0, keep]
        candidate_target = index[1, keep]
        local_source = source_lookup[candidate_source]
        local_target = target_lookup[candidate_target]
        retained = (local_source >= 0) & (local_target >= 0)
        result.append(
            {
                "source_type": source_type,
                "relation": str(batch.get("relation", "")),
                "target_type": target_type,
                "edge_index": torch.stack(
                    (local_source[retained], local_target[retained]), dim=0
                ),
                "edge_weight": weight[keep][retained],
                "available": bool(batch.get("available", True)),
                "source_hash": str(batch.get("source_hash", "")),
                "provenance": str(batch.get("provenance", "")),
                "source_donors": list(batch.get("source_donors", ())),
            }
        )
    return result


def _scan_descriptor(
    episode_path: Path, episode_record: dict[str, Any], fold: int,
    root: Path, static_cache: dict[Path, dict[str, dict[str, Any]]],
) -> EpisodeDescriptor:
    payload = _load_weights_only(episode_path)
    if payload.get("format") != EPISODE_FORMAT:
        raise RuntimeError("episode format mismatch")
    if any(name in payload for name in STATIC_DUPLICATE_FIELDS):
        raise RuntimeError("EPISODE_DUPLICATES_SHARED_STATIC_GRAPH_ASSETS")
    episode_id = str(payload.get("episode_id", "")).strip()
    cancer_id = str(payload.get("cancer_id", "")).upper()
    role = str(payload.get("role", ""))
    outer_fold = int(payload.get("outer_fold_id", -1))
    if not episode_id or not cancer_id or role not in ROLES or outer_fold != fold:
        raise RuntimeError("episode descriptor identity failed")
    static_path, static_record = verify_file_record(
        payload.get("static_graph_asset"), base=episode_path.parent,
        label=f"{episode_id}/static_graph_asset",
    )
    prior = static_cache.get(static_path)
    if prior is not None and prior["static_record"] != static_record:
        raise RuntimeError("shared static asset record drift")
    if prior is None:
        static = _load_weights_only(static_path)
        if (
            static.get("format") != STATIC_FORMAT
            or str(static.get("cancer_id", "")).upper() != cancer_id
        ):
            raise RuntimeError("shared static graph identity failed")
        graph_static_path, graph_static_record = verify_file_record(
            static.get("graph_asset_static_success"), base=static_path.parent,
            label=f"{cancer_id}/graph_asset_static_success",
        )
        matrix_path, matrix_record = verify_file_record(
            static.get("matrix_identity"), base=static_path.parent,
            label=f"{cancer_id}/matrix_identity",
        )
        prior = {
            "static_record": static_record,
            "graph_asset_static_record": graph_static_record,
            "matrix_identity_record": matrix_record,
        }
        static_cache[static_path] = prior
    features = payload.get("features")
    pair_by_arm = payload.get("pair_features_by_arm")
    if not isinstance(features, Mapping) or set(features) != set(NODE_TYPES):
        raise RuntimeError("episode feature schema failed")
    if not isinstance(pair_by_arm, Mapping) or set(pair_by_arm) != set(ARMS):
        raise RuntimeError("episode arm-specific pair features are absent")
    targets = torch.as_tensor(payload.get("target_rho"))
    pair_shapes = {tuple(torch.as_tensor(pair_by_arm[arm]).shape) for arm in ARMS}
    if targets.ndim != 1 or len(pair_shapes) != 1:
        raise RuntimeError("episode row tensor schema failed")
    pair_shape = next(iter(pair_shapes))
    if len(pair_shape) != 2 or pair_shape[0] != targets.numel() or targets.numel() == 0:
        raise RuntimeError("episode pair feature shape failed")
    feature_names = tuple(map(str, payload.get("pair_feature_names", ())))
    lasso_mask = torch.as_tensor(
        payload.get("lasso_derived_pair_feature_mask", ()), dtype=torch.bool
    )
    with_lasso = torch.as_tensor(pair_by_arm["with_lasso"], dtype=torch.float32)
    without_lasso = torch.as_tensor(pair_by_arm["without_lasso"], dtype=torch.float32)
    if (
        len(feature_names) != pair_shape[1] or lasso_mask.shape != (pair_shape[1],)
        or not bool(torch.isfinite(with_lasso).all())
        or not bool(torch.isfinite(without_lasso).all())
        or bool(torch.count_nonzero(without_lasso[:, lasso_mask]))
        or not torch.equal(with_lasso[:, ~lasso_mask], without_lasso[:, ~lasso_mask])
    ):
        raise RuntimeError("matched arm pair feature contract failed")
    return EpisodeDescriptor(
        path=episode_path, record=episode_record, static_path=static_path,
        static_record=static_record,
        graph_asset_static_record=prior["graph_asset_static_record"],
        matrix_identity_record=prior["matrix_identity_record"],
        episode_id=episode_id, cancer_id=cancer_id,
        outer_fold_id=fold, role=role, row_count=targets.numel(),
        pair_feature_count=pair_shape[1],
    )


def _metadata_frame(value: Any, row_count: int) -> pd.DataFrame:
    if not isinstance(value, Mapping):
        raise RuntimeError("episode row_metadata is not a mapping")
    frame = pd.DataFrame(dict(value))
    missing = sorted(set(ROW_ID_COLUMNS) - set(frame))
    if missing or len(frame) != row_count:
        raise RuntimeError(f"episode row metadata failed: {missing}")
    if frame[list(ROW_ID_COLUMNS)].isna().any().any() or frame.duplicated(list(ROW_ID_COLUMNS)).any():
        raise RuntimeError("episode row identity is null or duplicated")
    return frame


def load_episode(descriptor: EpisodeDescriptor, arm: str) -> LoadedEpisode:
    if arm not in ARMS:
        raise RuntimeError("unknown training arm")
    verify_file_record_metadata(
        descriptor.record, base=descriptor.path.parent,
        label="episode asset after preflight",
    )
    verify_file_record_metadata(
        descriptor.static_record, base=descriptor.static_path.parent,
        label="shared static asset after preflight",
    )
    payload = _load_weights_only(descriptor.path)
    static = _load_weights_only(descriptor.static_path)
    if (
        static.get("format") != STATIC_FORMAT
        or str(static.get("cancer_id", "")).upper() != descriptor.cancer_id
        or str(payload.get("episode_id", "")) != descriptor.episode_id
        or str(payload.get("cancer_id", "")).upper() != descriptor.cancer_id
        or int(payload.get("outer_fold_id", -1)) != descriptor.outer_fold_id
        or str(payload.get("role", "")) != descriptor.role
    ):
        raise RuntimeError("shared static graph identity failed")
    node_counts_raw = static.get("node_counts")
    if not isinstance(node_counts_raw, Mapping) or set(node_counts_raw) != set(NODE_TYPES[:-1]):
        raise RuntimeError("shared static node_counts schema failed")
    node_counts = {key: int(value) for key, value in node_counts_raw.items()}
    if any(value <= 0 for value in node_counts.values()):
        raise RuntimeError("shared static node_counts values failed")
    node_global_raw = payload.get("node_global_indices")
    if not isinstance(node_global_raw, Mapping) or set(node_global_raw) != set(NODE_TYPES[:-1]):
        raise RuntimeError("episode node_global_indices schema failed")
    node_global = {
        key: torch.as_tensor(value, dtype=torch.long) for key, value in node_global_raw.items()
    }
    features = payload.get("features")
    if not isinstance(features, Mapping) or set(features) != set(NODE_TYPES):
        raise RuntimeError("episode feature schema failed")
    for node_type in NODE_TYPES[:-1]:
        if torch.as_tensor(features[node_type]).shape[0] != node_global[node_type].numel():
            raise RuntimeError("episode feature/global-index alignment failed")
    gene_pathway = _coo_localized(
        static.get("gene_pathway_membership"), node_global["gene"],
        node_global["pathway"], "gene_pathway_membership",
    )
    pathway_family = _coo_localized(
        static.get("pathway_family_membership"), node_global["pathway"],
        node_global["pathway_family"], "pathway_family_membership",
    )
    if tuple(map(int, static["gene_pathway_membership"]["shape"])) != (
        node_counts["gene"], node_counts["pathway"]
    ) or tuple(map(int, static["pathway_family_membership"]["shape"])) != (
        node_counts["pathway"], node_counts["pathway_family"]
    ):
        raise RuntimeError("shared static membership/node-count identity failed")
    external_schema = static.get("external_relation_types")
    external_batches = static.get("external_edge_batches")
    if not isinstance(external_schema, list) or not external_schema:
        raise RuntimeError("shared static external schema is absent")
    if not isinstance(external_batches, list) or not external_batches:
        raise RuntimeError("shared static external batches are absent")
    localized_external = _external_localized(external_batches, node_global, node_counts)

    outer_train = _string_tuple(payload.get("outer_train_donors", ()), "outer_train_donors")
    outer_query = _string_tuple(payload.get("outer_query_donors", ()), "outer_query_donors")
    source = _string_tuple(payload.get("source_donors", ()), "source_donors")
    excluded = _string_tuple(payload.get("excluded_donors", ()), "excluded_donors")
    labels = _string_tuple(payload.get("label_donors", ()), "label_donors")
    target = _string_tuple(
        payload.get("inner_target_donors", ()), "inner_target_donors", allow_empty=True
    )
    validation = _string_tuple(
        payload.get("validation_donors", ()), "validation_donors", allow_empty=True
    )
    if set(outer_train) & set(outer_query) or set(source) & set(excluded) or set(source) & set(labels):
        raise RuntimeError("episode donor partition leak")
    donor_universe = set(outer_train) | set(outer_query)
    if (
        set(source) | set(excluded) != donor_universe
        or not set(target).issubset(set(outer_train))
        or not set(validation).issubset(set(outer_train))
        or not set(labels).issubset(donor_universe)
    ):
        raise RuntimeError("episode donor universe is incomplete or contains undeclared donors")
    for field in SOURCE_BINDING_FIELDS:
        if _string_tuple(payload.get(field, ()), field) != source:
            raise RuntimeError(f"{field.upper()}_DRIFT")
    geometry_source = _string_tuple(payload.get("geometry_source_donors", ()), "geometry_source_donors")
    geometry_excluded = _string_tuple(payload.get("geometry_excluded_donors", ()), "geometry_excluded_donors")
    if geometry_source != source or set(geometry_excluded) != set(excluded):
        raise RuntimeError("GEOMETRY_DONOR_BINDING_DRIFT")
    role = descriptor.role
    token = str(payload.get("geometry_isolation", ""))
    inner_fold_value = payload.get("inner_fold_id")
    inner_fold = None if inner_fold_value is None else int(inner_fold_value)
    if role == "fit":
        if inner_fold is None or inner_fold == descriptor.outer_fold_id or not target:
            raise RuntimeError("fit episode inner-fold contract failed")
        if not set(labels).issubset(target):
            raise RuntimeError("fit labels are outside inner target fold")
        if validation:
            if (
                inner_fold == (descriptor.outer_fold_id + 1) % N_OUTER_FOLDS
                or len(source) < 3
                or set(target) & set(validation)
                or token != "outer_and_inner_target_and_validation_fold"
            ):
                raise RuntimeError("FIT_EPISODE_REQUIRES_TARGET_AND_VALIDATION_REFIT_GEOMETRY")
            expected_source = set(outer_train) - set(target) - set(validation)
            expected_excluded = set(outer_query) | set(target) | set(validation)
        else:
            if len(source) < 2 or token != "outer_and_inner_target_fold":
                raise RuntimeError("SUPPORT_LIMITED_FIT_REQUIRES_LEAVE_ONE_TARGET_GEOMETRY")
            expected_source = set(outer_train) - set(target)
            expected_excluded = set(outer_query) | set(target)
    elif role == "validation":
        if (
            inner_fold is None or inner_fold != (descriptor.outer_fold_id + 1) % N_OUTER_FOLDS
            or len(source) < 3 or not validation or set(target) != set(validation)
            or not set(labels).issubset(validation)
        ):
            raise RuntimeError("validation episode donor contract failed")
        if token != "outer_and_inner_target_fold":
            raise RuntimeError("VALIDATION_EPISODE_REQUIRES_TARGET_REFIT_GEOMETRY")
        expected_source = set(outer_train) - set(validation)
        expected_excluded = set(outer_query) | set(validation)
    else:
        if (
            inner_fold is not None or target or validation or len(source) < 3
            or token != "outer_fold_only"
        ):
            raise RuntimeError("QUERY_EPISODE_REQUIRES_OUTER_FOLD_ONLY_GEOMETRY")
        if not set(labels).issubset(outer_query):
            raise RuntimeError("query labels are outside outer query fold")
        expected_source = set(outer_train)
        expected_excluded = set(outer_query)
    if set(source) != expected_source or set(excluded) != expected_excluded:
        raise RuntimeError("episode donor source set drift")
    expected_provider = role == "validation"
    expected_strict_bound = role == "fit" and bool(validation)
    expected_support_limited = role == "fit" and not validation
    expected_minimum = 2 if expected_support_limited else 3
    if (
        bool(payload.get("validation_provider")) != expected_provider
        or bool(payload.get("strict_validation_bound")) != expected_strict_bound
        or bool(payload.get("support_limited_fit")) != expected_support_limited
        or int(payload.get("minimum_source_donors", -1)) != expected_minimum
    ):
        raise RuntimeError("EPISODE_VALIDATION_PROVIDER_POLICY_DRIFT")

    sparse_lasso = payload.get("sparse_lasso_edges")
    if not isinstance(sparse_lasso, Mapping):
        raise RuntimeError("episode sparse LASSO COO is absent")
    if "response_index" in sparse_lasso and "lnc_index" not in sparse_lasso:
        sparse_lasso = {
            "lnc_index": sparse_lasso["response_index"],
            "gene_index": sparse_lasso.get("gene_index"),
            "coefficient": sparse_lasso.get("coefficient"),
            "stability": sparse_lasso.get("stability"),
        }
    graph = build_context_graph(
        features, gene_pathway, pathway_family,
        membership_orientation="gene_by_pathway",
        pathway_family_orientation="pathway_by_family",
        sparse_lasso_edges=sparse_lasso if arm == "with_lasso" else None,
        lasso_stability_threshold=float(payload.get("lasso_stability_threshold", 0.5)),
        external_edge_batches=localized_external,
        external_relation_types=external_schema,
        source_donors=source, excluded_donors=excluded,
    )
    pair_by_arm = payload.get("pair_features_by_arm")
    feature_names = tuple(map(str, payload.get("pair_feature_names", ())))
    lasso_mask = tuple(bool(value) for value in payload.get("lasso_derived_pair_feature_mask", ()))
    pair_features = torch.as_tensor(pair_by_arm[arm], dtype=torch.float32)
    if pair_features.ndim != 2 or len(feature_names) != pair_features.shape[1] or len(lasso_mask) != pair_features.shape[1]:
        raise RuntimeError("pair feature schema failed")
    mask = torch.as_tensor(lasso_mask, dtype=torch.bool)
    with_lasso_pairs = torch.as_tensor(pair_by_arm["with_lasso"], dtype=torch.float32)
    without_lasso_pairs = torch.as_tensor(pair_by_arm["without_lasso"], dtype=torch.float32)
    if (
        bool(mask.any()) and bool(torch.count_nonzero(without_lasso_pairs[:, mask]))
    ) or not torch.equal(with_lasso_pairs[:, ~mask], without_lasso_pairs[:, ~mask]):
        raise RuntimeError("MATCHED_ARM_PAIR_FEATURE_DRIFT")
    if arm == "without_lasso" and bool(mask.any()) and bool(torch.count_nonzero(pair_features[:, mask])):
        raise RuntimeError("WITHOUT_LASSO_RECEIVED_LASSO_DERIVED_PAIR_FEATURES")
    lnc_index = torch.as_tensor(payload.get("lnc_index"), dtype=torch.long)
    pathway_index = torch.as_tensor(payload.get("pathway_index"), dtype=torch.long)
    targets = torch.as_tensor(payload.get("target_rho"), dtype=torch.float32)
    count = targets.numel()
    if (
        targets.ndim != 1 or lnc_index.shape != targets.shape
        or pathway_index.shape != targets.shape or pair_features.shape[0] != count
        or count != descriptor.row_count or not bool(torch.isfinite(targets).all())
        or bool((targets.abs() > 1).any()) or not bool(torch.isfinite(pair_features).all())
    ):
        raise RuntimeError("episode row tensor values failed")
    rows = _metadata_frame(payload.get("row_metadata"), count)
    if set(rows.query_donor_id.astype(str)) != set(labels):
        raise RuntimeError("episode row donors disagree with label donors")
    rows = rows.copy()
    rows["rho"] = targets.numpy()
    rows["row_uid"] = [
        canonical_hash([str(row[column]) for column in ROW_ID_COLUMNS])
        for _, row in rows.iterrows()
    ]
    return LoadedEpisode(
        descriptor=descriptor, inner_fold_id=inner_fold,
        cell_key=str(payload.get("cell_key", "")).strip(), label_donors=labels,
        source_donors=source, excluded_donors=excluded,
        outer_train_donors=outer_train, outer_query_donors=outer_query,
        inner_target_donors=target, validation_donors=validation,
        geometry_isolation=token, graph=graph, lnc_index=lnc_index,
        pathway_index=pathway_index, pair_features=pair_features, targets=targets,
        rows=rows, pair_feature_names=feature_names,
        lasso_pair_feature_mask=lasso_mask,
    )


def load_episode_manifest(
    settings: TrainingSettings,
) -> tuple[dict[int, list[EpisodeDescriptor]], dict[str, Any]]:
    root = settings.episode_root.expanduser().resolve()
    manifest_path = root / "SC_HHGT_EPISODE_MANIFEST.json"
    manifest = read_json(manifest_path, "SC_HHGT_EPISODE_MANIFEST")
    if (
        manifest.get("format") != EPISODE_MANIFEST_FORMAT
        or manifest.get("state") != "EPISODES_COMPLETE"
        or str(manifest.get("target_host")) != "149"
    ):
        raise RuntimeError("episode manifest is not complete for host 149")
    graph_path, graph_record = verify_file_record(
        manifest.get("graph_asset_manifest"), base=root,
        label="bound GRAPH_ASSET_MANIFEST",
    )
    graph_manifest = read_json(graph_path, "bound GRAPH_ASSET_MANIFEST")
    if (
        graph_manifest.get("format") != GRAPH_ASSET_FORMAT
        or graph_manifest.get("state") != "GRAPH_ASSETS_COMPLETE"
        or str(graph_manifest.get("target_host")) != "149"
    ):
        raise RuntimeError("bound graph asset manifest is not complete")
    if settings.graph_assets_root is not None and graph_path != (
        settings.graph_assets_root.expanduser().resolve() / "GRAPH_ASSET_MANIFEST.json"
    ):
        raise RuntimeError("configured graph asset root disagrees with episode lineage")
    folds = manifest.get("folds")
    if not isinstance(folds, list):
        raise RuntimeError("episode manifest lacks folds")
    by_fold: dict[int, list[EpisodeDescriptor]] = {fold: [] for fold in range(N_OUTER_FOLDS)}
    episode_records: list[dict[str, Any]] = []
    static_cache: dict[Path, dict[str, Any]] = {}
    seen: set[str] = set()
    for fold_entry in folds:
        fold = int(fold_entry.get("outer_fold_id", -1))
        if fold not in by_fold or not isinstance(fold_entry.get("episodes"), list):
            raise RuntimeError("episode fold entry failed")
        for number, raw_record in enumerate(fold_entry["episodes"]):
            path, record = verify_file_record_metadata(
                raw_record, base=root, label=f"fold={fold}/episode={number}",
                require_within=root,
            )
            descriptor = _scan_descriptor(path, record, fold, root, static_cache)
            if descriptor.episode_id in seen:
                raise RuntimeError("duplicate episode_id")
            seen.add(descriptor.episode_id)
            by_fold[fold].append(descriptor)
            episode_records.append(record)
    for fold, descriptors in by_fold.items():
        roles = {value.role for value in descriptors}
        if roles != set(ROLES):
            raise RuntimeError(f"outer fold {fold} lacks fit, validation, or query episodes")
        counts = {value.pair_feature_count for value in descriptors}
        if len(counts) != 1:
            raise RuntimeError(f"outer fold {fold} pair feature dimension drift")
        descriptors.sort(key=lambda value: value.episode_id)
    return by_fold, {
        "episode_manifest": file_record(manifest_path),
        "graph_asset_manifest": graph_record,
        "episode_assets": episode_records,
        "shared_static_assets": list(static_cache.values()),
    }


def _validate_settings(settings: TrainingSettings) -> None:
    integers=("seed","epochs","patience","prediction_batch_size","max_rows_per_episode_step",
              "max_fit_episodes_per_epoch","training_workers","threads_per_worker","hidden_channels","num_layers","num_heads")
    if any(type(getattr(settings,k)) is not int for k in integers) or type(settings.prepare_only) is not bool:
        raise RuntimeError("SC-HHGT integer settings cannot be Boolean or fractional")
    if not 0<=settings.seed<=2**32-1-4*1009:
        raise RuntimeError("SC-HHGT fold seeds must fit NumPy uint32")
    for key in ("minimum_validation_improvement","learning_rate","weight_decay","dropout","max_gradient_norm"):
        value=getattr(settings,key)
        if type(value) not in (int,float) or not np.isfinite(value):
            raise RuntimeError("SC-HHGT numeric settings must be finite: "+key)
    if settings.epochs <= 0 or settings.patience <= 0:
        raise RuntimeError("epoch/patience settings must be positive")
    if settings.prediction_batch_size <= 0 or settings.max_rows_per_episode_step <= 0:
        raise RuntimeError("bounded row settings must be positive")
    if settings.max_fit_episodes_per_epoch <= 0:
        raise RuntimeError("max fit episodes per epoch must be positive")
    if not 1 <= settings.training_workers <= N_OUTER_FOLDS:
        raise RuntimeError("training workers must be between one and five")
    if settings.threads_per_worker <= 0:
        raise RuntimeError("threads per worker must be positive")
    if settings.training_workers * settings.threads_per_worker > (os.cpu_count() or 1):
        raise RuntimeError("training worker/thread allocation exceeds online CPUs")
    if settings.learning_rate <= 0 or settings.weight_decay < 0:
        raise RuntimeError("optimizer settings failed")
    if settings.minimum_validation_improvement < 0:
        raise RuntimeError("minimum validation improvement must be nonnegative")
    if (
        settings.hidden_channels <= 0 or settings.num_layers <= 0
        or settings.num_heads <= 0 or settings.hidden_channels % settings.num_heads
        or not 0 <= settings.dropout < 1 or settings.max_gradient_norm <= 0
    ):
        raise RuntimeError("model settings failed")


def _model_config(settings: TrainingSettings, pair_features: int) -> dict[str, Any]:
    return {
        "hidden_channels": settings.hidden_channels,
        "num_layers": settings.num_layers,
        "num_heads": settings.num_heads,
        "dropout": settings.dropout,
        "pair_feature_count": pair_features,
    }


def _initialize_model(
    graph: ContextGraph, settings: TrainingSettings, pair_features: int, seed: int,
) -> SingleCellHHGT:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = SingleCellHHGT(graph, **_model_config(settings, pair_features))
    forbidden = ("lnc_core", "pathway_core", "cancer_core", "private_lnc")
    if any(any(token in name for token in forbidden) for name in model.state_dict()):
        raise RuntimeError("SC_HHGT_MUST_NOT_CONTAIN_FROZEN_BULK_CORE_MLP")
    return model


def _sample_rows(total: int, maximum: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if total <= maximum:
        rows = np.arange(total, dtype=np.int64)
        rng.shuffle(rows)
        return rows
    return np.sort(rng.choice(total, size=maximum, replace=False)).astype(np.int64)


def _fit_episode_schedule(
    fit: list[EpisodeDescriptor], maximum: int, seed: int, epoch: int,
) -> tuple[list[EpisodeDescriptor], int, int]:
    """Return one deterministic, non-overlapping chunk of a shuffled cycle."""
    if not fit:
        raise RuntimeError("fold has no fit episodes")
    cycle_length = (len(fit) + maximum - 1) // maximum
    cycle = epoch // cycle_length
    position = epoch % cycle_length
    ordered = list(fit)
    random.Random(seed + cycle * 1_000_003).shuffle(ordered)
    start = position * maximum
    return ordered[start : start + maximum], cycle, cycle_length


def _train_episode_step(
    model: SingleCellHHGT, optimizer: torch.optim.Optimizer,
    episode: LoadedEpisode, rows: np.ndarray, max_gradient_norm: float,
) -> dict[str, float]:
    optimizer.zero_grad(set_to_none=True)
    model.train()
    encoded = model.encode_graph(episode.graph)
    prediction = model.decode(
        encoded, episode.lnc_index[rows], episode.pathway_index[rows],
        episode.pair_features[rows],
    )
    target = episode.targets[rows]
    loss = F.smooth_l1_loss(prediction, target, beta=0.1)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("SC_NONFINITE_LOSS")
    loss.backward()
    report = {
        "loss": float(loss.detach()),
        "hgt_gradient_l1": _gradient_l1(model.encoder.layers),
        "weighted_projection_gradient_l1": _gradient_l1(model.encoder.weighted_projection),
        "continuous_projection_gradient_l1": _gradient_l1(model.encoder.continuous_projection),
        "decoder_gradient_l1": _gradient_l1(model.decoder),
    }
    if report["hgt_gradient_l1"] <= 0:
        raise RuntimeError("SC_HGT_ENCODER_HAS_NO_GRADIENT")
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
    optimizer.step()
    return report


def _evaluate_episode(
    model: SingleCellHHGT, episode: LoadedEpisode, batch_size: int,
) -> tuple[np.ndarray, float]:
    model.eval()
    predictions: list[torch.Tensor] = []
    loss_sum = 0.0
    with torch.no_grad():
        encoded = model.encode_graph(episode.graph)
        for start in range(0, len(episode.rows), batch_size):
            stop = min(start + batch_size, len(episode.rows))
            prediction = model.decode(
                encoded, episode.lnc_index[start:stop], episode.pathway_index[start:stop],
                episode.pair_features[start:stop],
            )
            predictions.append(prediction.cpu())
            loss_sum += float(
                F.smooth_l1_loss(
                    prediction, episode.targets[start:stop], beta=0.1, reduction="sum"
                )
            )
    return torch.cat(predictions).numpy(), loss_sum


def _prediction_frame(
    model: SingleCellHHGT, episode: LoadedEpisode, arm: str, batch_size: int,
) -> pd.DataFrame:
    values, _ = _evaluate_episode(model, episode, batch_size)
    frame = episode.rows.copy()
    frame["predicted_rho"] = values.astype(np.float32)
    frame["predicted_association_strength"] = np.abs(values)
    frame["residual"] = frame.rho - frame.predicted_rho
    frame["arm"] = arm
    frame["outer_fold_id"] = episode.descriptor.outer_fold_id
    frame["episode_id"] = episode.descriptor.episode_id
    return frame


def fit_fold_model(settings, fold, arm, fit, validation, model, seed):
    """Original optimizer/model-selection loop; serialization remains outside."""
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    history: list[dict[str, Any]] = []
    episode_audits: dict[str, dict[str, Any]] = {}
    best_state: dict[str, torch.Tensor] | None = None
    best_validation = float("inf")
    best_epoch = -1
    stale = 0
    first_step: dict[str, Any] | None = None
    fit_episode_ids = {value.episode_id for value in fit}
    seen_fit_episode_ids: set[str] = set()
    coverage_epochs = (
        len(fit) + settings.max_fit_episodes_per_epoch - 1
    ) // settings.max_fit_episodes_per_epoch
    if coverage_epochs > settings.epochs:
        raise RuntimeError(
            "epoch budget cannot cover every fit episode before model selection: "
            f"fold={fold} fit={len(fit)} maximum={settings.max_fit_episodes_per_epoch} "
            f"epochs={settings.epochs} required={coverage_epochs}"
        )
    for epoch in range(settings.epochs):
        torch.manual_seed(seed + 100_000 + epoch)
        ordered, episode_cycle, episode_cycle_length = _fit_episode_schedule(
            fit, settings.max_fit_episodes_per_epoch, seed, epoch
        )
        steps: list[dict[str, float]] = []
        sampled_rows = available_rows = 0
        sample_receipts: list[dict[str, Any]] = []
        for position, descriptor in enumerate(ordered):
            episode = load_episode(descriptor, arm)
            rows = _sample_rows(
                len(episode.rows), settings.max_rows_per_episode_step,
                seed + epoch * 1009 + position,
            )
            before = _state_hash(model.state_dict(), "encoder.layers.") if first_step is None else None
            result = _train_episode_step(
                model, optimizer, episode, rows, settings.max_gradient_norm
            )
            if first_step is None:
                after = _state_hash(model.state_dict(), "encoder.layers.")
                first_step = {
                    "episode_id": descriptor.episode_id,
                    "hgt_parameter_sha256_before": before,
                    "hgt_parameter_sha256_after": after,
                    "hgt_parameters_changed": before != after,
                    "hgt_gradient_l1": result["hgt_gradient_l1"],
                }
                if before == after:
                    raise RuntimeError("SC_HGT_PARAMETERS_DID_NOT_UPDATE")
            steps.append(result)
            sampled_rows += len(rows)
            available_rows += len(episode.rows)
            sample_receipts.append(
                {
                    "episode_id": descriptor.episode_id,
                    "available_rows": len(episode.rows),
                    "sampled_rows": len(rows),
                    "sampling_fraction": len(rows) / len(episode.rows),
                    "sample_index_sha256": hashlib.sha256(rows.tobytes()).hexdigest(),
                }
            )
            episode_audits[descriptor.episode_id] = episode.audit(arm)
            seen_fit_episode_ids.add(descriptor.episode_id)
            del episode
        full_fit_episode_coverage = seen_fit_episode_ids == fit_episode_ids
        validation_loss_sum = 0.0
        validation_rows = 0
        validation_loss: float | None = None
        improved = False
        if full_fit_episode_coverage:
            for descriptor in validation:
                episode = load_episode(descriptor, arm)
                _, loss_sum = _evaluate_episode(
                    model, episode, settings.prediction_batch_size
                )
                validation_loss_sum += loss_sum
                validation_rows += len(episode.rows)
                episode_audits[descriptor.episode_id] = episode.audit(arm)
                del episode
            validation_loss = validation_loss_sum / validation_rows
            improved = (
                validation_loss
                < best_validation - settings.minimum_validation_improvement
            )
            if improved:
                best_validation = validation_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
        history.append(
            {
                "epoch": epoch,
                "optimizer_steps": len(steps),
                "mean_train_loss": float(np.mean([value["loss"] for value in steps])),
                "validation_loss": validation_loss,
                "validation_rows": validation_rows,
                "selected_as_best": improved,
                "stale_epochs": stale,
                "available_training_rows": available_rows,
                "sampled_training_rows": sampled_rows,
                "sampling_coverage": sampled_rows / available_rows,
                "sampling_receipts": sample_receipts,
                "fit_episode_population": len(fit),
                "selected_fit_episode_count": len(ordered),
                "selected_fit_episode_ids_sha256": canonical_hash(
                    [value.episode_id for value in ordered]
                ),
                "cumulative_fit_episode_count": len(seen_fit_episode_ids),
                "cumulative_fit_episode_coverage": (
                    len(seen_fit_episode_ids) / len(fit_episode_ids)
                ),
                "full_fit_episode_coverage": full_fit_episode_coverage,
                "fit_episode_cycle": episode_cycle,
                "fit_episode_cycle_length": episode_cycle_length,
                "model_selection_eligible": full_fit_episode_coverage,
                "validation_skipped_until_full_fit_episode_coverage": (
                    not full_fit_episode_coverage
                ),
                **{
                    f"{name}_{statistic}": float(function([value[name] for value in steps]))
                    for name in (
                        "hgt_gradient_l1", "weighted_projection_gradient_l1",
                        "continuous_projection_gradient_l1", "decoder_gradient_l1",
                    )
                    for statistic, function in (("min", min), ("mean", np.mean), ("max", max))
                },
            }
        )
        if full_fit_episode_coverage and stale >= settings.patience:
            break
    if best_state is None or best_epoch < 0:
        raise RuntimeError("INNER_VALIDATION_DID_NOT_SELECT_A_MODEL")
    model.load_state_dict(best_state, strict=True)
    return dict(history=history, episode_audits=episode_audits, best_epoch=best_epoch,
                best_validation=best_validation, first_step=first_step)


def train_fold_arm(
    settings: TrainingSettings, fold: int, arm: str,
    descriptors: list[EpisodeDescriptor], output_root: Path, seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    fit = [value for value in descriptors if value.role == "fit"]
    validation = [value for value in descriptors if value.role == "validation"]
    query = [value for value in descriptors if value.role == "query"]
    example = load_episode(fit[0], arm)
    model = _initialize_model(example.graph, settings, example.pair_features.shape[1], seed)
    edge_schema = [list(value) for value in example.graph.edge_schema]
    feature_dims = {
        node_type: int(example.graph.data[node_type].continuous_x.shape[1])
        for node_type in NODE_TYPES
    }
    del example
    initial_state = _state_hash(model.state_dict())
    initial_hgt = _state_hash(model.state_dict(), "encoder.layers.")
    fitted = fit_fold_model(settings, fold, arm, fit, validation, model, seed)
    history, episode_audits, best_epoch, best_validation, first_step = (
        fitted[key] for key in ("history", "episode_audits", "best_epoch", "best_validation", "first_step")
    )
    final_state = _state_hash(model.state_dict())
    final_hgt = _state_hash(model.state_dict(), "encoder.layers.")
    if final_state == initial_state or final_hgt == initial_hgt:
        raise RuntimeError("SC_HHGT_TRAINING_DID_NOT_CHANGE_PARAMETERS")
    predictions: list[pd.DataFrame] = []
    for descriptor in query:
        episode = load_episode(descriptor, arm)
        predictions.append(
            _prediction_frame(model, episode, arm, settings.prediction_batch_size)
        )
        episode_audits[descriptor.episode_id] = episode.audit(arm)
        del episode
    prediction_frame = pd.concat(predictions, ignore_index=True)
    audits = [episode_audits[value.episode_id] for value in descriptors]
    external_training = sum(
        value["external_edge_count"] for value in audits if value["role"] == "fit"
    )
    lasso_training = sum(
        value["lasso_edge_count"] for value in audits if value["role"] == "fit"
    )
    if external_training <= 0:
        raise RuntimeError("EXTERNAL_EVIDENCE_WAS_NOT_USED")
    if (arm == "with_lasso" and lasso_training <= 0) or (
        arm == "without_lasso" and lasso_training != 0
    ):
        raise RuntimeError("LASSO_ARM_SEMANTICS_FAILED")

    fold_root = output_root / f"arm={arm}" / f"outer_fold={fold}"
    if fold_root.exists():
        raise RuntimeError(f"fold output already exists: {fold_root}")
    fold_root.mkdir(parents=True)
    checkpoint_path = fold_root / "best_model_state.pt"
    history_path = fold_root / "HISTORY.json"
    predictions_path = fold_root / "QUERY_PREDICTIONS.parquet"
    report_path = fold_root / "TRAINING_REPORT.json"
    config = _model_config(settings, descriptors[0].pair_feature_count)
    checkpoint = {
        "format": CHECKPOINT_FORMAT,
        "architecture_family": "SC_HHGT_WEIGHTED_HGTCONV",
        "arm": arm, "outer_fold_id": fold, "seed": seed,
        "model_config": config, "continuous_feature_dims": feature_dims,
        "edge_schema": edge_schema,
        "edge_schema_sha256": canonical_hash(edge_schema),
        "initial_state_sha256": initial_state,
        "initial_hgt_state_sha256": initial_hgt,
        "final_state_sha256": final_state,
        "final_hgt_state_sha256": final_hgt,
        "best_epoch": best_epoch, "best_validation_loss": best_validation,
        "fresh_random_initialization": True, "old_checkpoint_loaded": False,
        "source_checkpoint_sha256": None, "history": history,
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
    }
    fixture_tags={k:fitted[k] for k in ("synthetic_serialization_fixture","optimizer_actually_run") if k in fitted}
    checkpoint.update(fixture_tags)
    atomic_torch_save(checkpoint_path, checkpoint)
    atomic_json(history_path, {"format": TRAINING_FORMAT, "history": history})
    atomic_parquet(predictions_path, prediction_frame)
    report = {
        "format": TRAINING_FORMAT, "status": "PASS", "target_host": "149",
        "device": "cpu", "cuda_used": False,
        "architecture_family": "SC_HHGT_WEIGHTED_HGTCONV",
        "arm": arm, "outer_fold_id": fold, "seed": seed,
        "loss": {"name": "SmoothL1Loss", "beta": 0.1},
        "model_selection_source": "inner_validation_only",
        "outer_query_used_for_model_selection": False,
        "best_epoch": best_epoch, "best_validation_loss": best_validation,
        "initial_state_sha256": initial_state, "final_state_sha256": final_state,
        "initial_hgt_state_sha256": initial_hgt, "final_hgt_state_sha256": final_hgt,
        "hgt_parameters_changed": final_hgt != initial_hgt,
        "first_optimizer_step": first_step, "history": history,
        "episode_audits": audits,
        "fit_episode_count": len(fit), "validation_episode_count": len(validation),
        "query_episode_count": len(query), "query_prediction_count": len(prediction_frame),
        "max_fit_episodes_per_epoch": settings.max_fit_episodes_per_epoch,
        "fit_episode_coverage_before_model_selection": True,
        "fit_episode_population": len(fit),
        "external_training_edge_count": external_training,
        "lasso_training_edge_count": lasso_training,
        "lazy_episode_loading": True,
        "maximum_live_episode_graphs": 1,
        "one_graph_encode_per_episode_step": True,
    }
    report.update(fixture_tags)
    atomic_json(report_path, report)
    files = {
        "checkpoint": file_record(checkpoint_path), "history": file_record(history_path),
        "predictions": file_record(predictions_path), "report": file_record(report_path),
    }
    success_path = fold_root / "SUCCESS.json"
    atomic_json(
        success_path,
        {"format": SUCCESS_FORMAT, "state": "FOLD_TRAINING_COMPLETE",
         "target_host": "149", "arm": arm, "outer_fold_id": fold,
         "seed": seed, "files": files},
    )
    return {
        **report, "files": files, "success": file_record(success_path),
        "model_config": config, "continuous_feature_dims": feature_dims,
        "edge_schema": edge_schema,
    }


def _run_config(settings: TrainingSettings) -> dict[str, Any]:
    return {
        "format": CONFIG_FORMAT, "target_host": "149", "device": "cpu",
        "cuda_used": False, "base_seed": settings.seed,
        "fold_seeds": [settings.seed + fold * 1009 for fold in range(5)],
        "outer_folds": 5, "arms": list(ARMS), "primary_arm": "with_lasso",
        "ablation_arm": "without_lasso", "matched_arm_initialization": True,
        "epochs": settings.epochs, "patience": settings.patience,
        "minimum_validation_improvement": settings.minimum_validation_improvement,
        "prediction_batch_size": settings.prediction_batch_size,
        "max_rows_per_episode_step": settings.max_rows_per_episode_step,
        "max_fit_episodes_per_epoch": settings.max_fit_episodes_per_epoch,
        "fit_episode_coverage_required_before_model_selection": True,
        "training_workers": settings.training_workers,
        "threads_per_worker": settings.threads_per_worker,
        "learning_rate": settings.learning_rate, "weight_decay": settings.weight_decay,
        "max_gradient_norm": settings.max_gradient_norm,
        "architecture": {
            "family": "SC_HHGT_WEIGHTED_HGTCONV", "node_types": list(NODE_TYPES),
            "hidden_channels": settings.hidden_channels, "num_layers": settings.num_layers,
            "num_heads": settings.num_heads, "dropout": settings.dropout,
            "continuous_node_features": True, "signed_edge_weight_path": True,
            "cell_context_node": True, "frozen_bulk_core": False, "mlp_only": False,
        },
        "loss": {"name": "SmoothL1Loss", "beta": 0.1,
                 "target": "signed_within_donor_metacell_spearman_rho"},
        "episode_policy": {
            "precomputed_hash_bound": True, "lazy_loading": True,
            "maximum_live_episode_graphs": 1, "shared_cancer_static_assets": True,
            "fit_lasso_during_optimization": False,
            "one_graph_encode_per_episode_step": True,
            "bounded_row_sampling": True,
            "outer_query_excluded_from_all_fitting": True,
            "strict_inner_validation": True,
            "outer_query_used_for_model_selection": False,
            "training_geometry_requires_target_and_validation_refit": True,
            "support_limited_fit_geometry": "outer_and_inner_target_fold",
            "support_limited_fit_supplies_validation": False,
            "support_limited_fit_minimum_source_donors": 2,
            "outer_only_geometry_allowed_for_training": False,
            "outer_only_geometry_allowed_for_query": True,
            "arm_specific_pair_features": True,
        },
        "fresh_random_initialization": True, "old_checkpoint_loaded": False,
        "deterministic_algorithms": True,
    }


def _write_checksums(root: Path) -> Path:
    path = root / "SHA256SUMS.tsv"
    files = sorted(
        value for value in root.rglob("*")
        if value.is_file() and value != path and value.name != "SUCCESS.json"
        and not value.name.endswith(".partial")
    )
    text = "\n".join(
        f"{sha256(value)}\t{value.stat().st_size}\t{value.relative_to(root).as_posix()}"
        for value in files
    ) + "\n"
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)
    return path


_MODEL_REPORT_KEYS = (
    "arm", "outer_fold_id", "seed", "initial_state_sha256",
    "final_state_sha256", "initial_hgt_state_sha256", "final_hgt_state_sha256",
    "best_epoch", "best_validation_loss", "model_config",
    "continuous_feature_dims", "edge_schema", "files", "success",
    "query_prediction_count",
)


def _train_fold_job(
    settings: TrainingSettings, fold: int, descriptors: list[EpisodeDescriptor],
    output: Path,
) -> list[dict[str, Any]]:
    torch.set_num_threads(settings.threads_per_worker)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    seed = settings.seed + fold * 1009
    reports: list[dict[str, Any]] = []
    paired: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        report = train_fold_arm(settings, fold, arm, descriptors, output, seed)
        compact = {key: report[key] for key in _MODEL_REPORT_KEYS}
        reports.append(compact)
        paired[arm] = compact
    if paired[ARMS[0]]["initial_state_sha256"] != paired[ARMS[1]]["initial_state_sha256"]:
        raise RuntimeError("matched ablation initialization failed")
    return reports


def _combine_oof_predictions(
    output: Path, reports: list[dict[str, Any]], destination: Path,
) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    temporary = destination.with_name(destination.name + ".partial")
    writer = None
    schema = None
    rows = 0
    try:
        for report in sorted(reports, key=lambda value: (value["arm"], value["outer_fold_id"])):
            record = report["files"]["predictions"]
            path, _ = verify_file_record(
                record, base=output,
                label=f"{report['arm']}/fold={report['outer_fold_id']}/predictions",
                require_within=output,
            )
            for batch in pq.ParquetFile(path).iter_batches(batch_size=131_072):
                table = pa.Table.from_batches([batch])
                if writer is None:
                    schema = table.schema
                    writer = pq.ParquetWriter(temporary, schema, compression="zstd")
                elif not table.schema.equals(schema, check_metadata=False):
                    raise RuntimeError("query prediction parquet schema drift")
                elif table.schema.metadata != schema.metadata:
                    table = table.replace_schema_metadata(schema.metadata)
                writer.write_table(table)
                rows += len(table)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise RuntimeError("no query prediction files were combined")
    expected = sum(int(value["query_prediction_count"]) for value in reports)
    if rows != expected:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"combined OOF row count mismatch: expected={expected} actual={rows}"
        )
    os.replace(temporary, destination)
    return rows


def run_training(settings: TrainingSettings, *, hostname: str | None = None) -> dict[str, Any]:
    require_host_149(hostname)
    _validate_settings(settings)
    torch.use_deterministic_algorithms(True)
    output = settings.output_root.expanduser().resolve()
    if output.exists():
        raise RuntimeError(f"training output already exists: {output}")
    descriptors, input_lineage = load_episode_manifest(settings)
    output.mkdir(parents=True)
    config_path = output / "RUN_CONFIG.json"
    preflight_path = output / "LAUNCH_PREFLIGHT.json"
    lineage_path = output / "INPUT_LINEAGE.json"
    atomic_json(config_path, _run_config(settings))
    atomic_json(
        preflight_path,
        {"format": TRAINING_FORMAT, "status": "PASS", "target_host": "149",
         "device": "cpu", "paid_compute": False,"training_started":False,"prepare_only":settings.prepare_only,
         "episode_manifest": input_lineage["episode_manifest"],
         "graph_asset_manifest": input_lineage["graph_asset_manifest"]},
    )
    atomic_json(
        lineage_path,
        {"format": LINEAGE_FORMAT, "status": "PASS", "target_host": "149",
         **input_lineage, "trainer_source": file_record(Path(__file__).resolve()),
         "model_source": file_record(Path(__file__).resolve().parents[1] / "cc_hhgt" / "v32" / "single_cell_hhgt.py"),
         "old_checkpoint_loaded": False, "source_checkpoint_sha256": None},
    )
    from cc_hhgt.v32.single_cell_numeric_replay import collect_input_preflight
    input_preflight_path=output/"INPUT_PREFLIGHT.json"
    atomic_json(input_preflight_path,collect_input_preflight(settings,descriptors,input_lineage))
    if settings.prepare_only:
        return {"status":"PASS_READY","target_host":"149","training_started":False,
                "input_preflight":file_record(input_preflight_path)}
    reports: list[dict[str, Any]] = []
    if settings.training_workers == 1:
        for fold in range(N_OUTER_FOLDS):
            reports.extend(_train_fold_job(settings, fold, descriptors[fold], output))
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=settings.training_workers, mp_context=context
        ) as executor:
            futures = {
                executor.submit(
                    _train_fold_job, settings, fold, descriptors[fold], output
                ): fold
                for fold in range(N_OUTER_FOLDS)
            }
            for future in as_completed(futures):
                reports.extend(future.result())
    for arm in ARMS:
        if len({value["initial_state_sha256"] for value in reports if value["arm"] == arm}) != 5:
            raise RuntimeError("fold model initializations are not independent")
    models = [
        {key: report[key] for key in (
            "arm", "outer_fold_id", "seed", "initial_state_sha256",
            "final_state_sha256", "initial_hgt_state_sha256", "final_hgt_state_sha256",
            "best_epoch", "best_validation_loss", "model_config",
            "continuous_feature_dims", "edge_schema", "files", "success",
        )}
        for report in sorted(reports, key=lambda value: (value["arm"], value["outer_fold_id"]))
    ]
    checkpoint_path = output / "CHECKPOINT_MANIFEST.json"
    atomic_json(
        checkpoint_path,
        {"format": CHECKPOINT_MANIFEST_FORMAT, "status": "SUCCESS",
         "outer_folds": 5, "arms": list(ARMS), "model_count": 10, "models": models},
    )
    oof_path = output / "OOF_PREDICTIONS.parquet"
    oof_rows = _combine_oof_predictions(output, reports, oof_path)
    checksums = _write_checksums(output)
    success = {
        "format": SUCCESS_FORMAT, "state": "TRAINING_COMPLETE", "target_host": "149",
        "training_format": TRAINING_FORMAT, "outer_folds": 5, "arms": list(ARMS),
        "trained_model_count": 10, "oof_prediction_rows": oof_rows,
        "files": {
            "run_config": file_record(config_path), "preflight": file_record(preflight_path),
            "input_lineage": file_record(lineage_path),
            "input_preflight": file_record(input_preflight_path),
            "checkpoint_manifest": file_record(checkpoint_path),
            "oof_predictions": file_record(oof_path), "checksums": file_record(checksums),
        },
    }
    atomic_json(output / "SUCCESS.json", success)
    return success


def parse_args(argv: list[str] | None = None) -> TrainingSettings:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--graph-assets-root", type=Path)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--minimum-validation-improvement", type=float, default=0.0)
    parser.add_argument("--prediction-batch-size", type=int, default=2048)
    parser.add_argument("--max-rows-per-episode-step", type=int, default=2048)
    parser.add_argument("--max-fit-episodes-per-epoch", type=int, default=64)
    parser.add_argument("--training-workers", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--threads-per-worker", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-gradient-norm", type=float, default=5.0)
    parser.add_argument("--prepare-only", action="store_true")
    values = parser.parse_args(argv)
    return TrainingSettings(**vars(values))


def main(argv: list[str] | None = None) -> int:
    result = run_training(parse_args(argv))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
