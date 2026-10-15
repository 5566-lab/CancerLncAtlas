#!/usr/bin/env python3
"""Independent fail-closed validation of five-fold SC-HHGT training."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch

from cc_hhgt.v32.single_cell_hhgt import NODE_TYPES, SingleCellHHGT, build_context_graph


TRAINING_FORMAT = "CANCERLNCATLAS_V32_SC_HHGT_TRAINING_V1"
CONFIG_FORMAT = "CANCERLNCATLAS_V32_SC_HHGT_RUN_CONFIG_V1"
LINEAGE_FORMAT = "CANCERLNCATLAS_V32_SC_HHGT_LINEAGE_V1"
CHECKPOINT_FORMAT = "CANCERLNCATLAS_V32_SC_HHGT_CHECKPOINT_V1"
CHECKPOINT_MANIFEST_FORMAT = "CANCERLNCATLAS_V32_SC_HHGT_CHECKPOINT_MANIFEST_V1"
SUCCESS_FORMAT = "CANCERLNCATLAS_V32_SC_HHGT_SUCCESS_V1"
VALIDATION_FORMAT = "CANCERLNCATLAS_V32_SC_HHGT_INDEPENDENT_VALIDATION_V1"
EPISODE_FORMAT = "CANCERLNCATLAS_V32_SC_HHGT_EPISODE_V1"
STATIC_FORMAT = "CANCERLNCATLAS_V32_SC_HHGT_EPISODE_STATIC_V1"
N_OUTER_FOLDS = 5
ARMS = ("with_lasso", "without_lasso")
SOURCE_BINDING_FIELDS = (
    "feature_source_donors", "lasso_source_donors", "pair_feature_source_donors",
)
STATIC_DUPLICATE_FIELDS = {
    "gene_pathway_membership", "pathway_family_membership",
    "external_relation_types", "external_edge_batches",
}
ROW_ID_COLUMNS = (
    "dataset_id",
    "cancer_id",
    "cell_key",
    "query_donor_id",
    "lncrna_id",
    "pathway_id",
)


class IndependentValidationError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def state_hash(state: Mapping[str, torch.Tensor], *, prefix: str | None = None) -> str:
    selected = {
        name: value
        for name, value in state.items()
        if prefix is None or name.startswith(prefix)
    }
    if not selected:
        raise IndependentValidationError(f"empty state for prefix {prefix!r}")
    digest = hashlib.sha256()
    for name in sorted(selected):
        value = selected[name]
        if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
            raise IndependentValidationError(f"invalid state tensor: {name}")
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise IndependentValidationError(f"missing or unsafe {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise IndependentValidationError(f"{label} is not a JSON object")
    return value


def require_host_149(hostname: str | None = None) -> str:
    observed = str(hostname if hostname is not None else socket.gethostname()).strip()
    if observed != "149":
        raise IndependentValidationError(
            f"SC_HHGT_VALIDATION_REQUIRES_HOST_149: observed={observed}"
        )
    return observed


def verify_record(
    record: Mapping[str, Any], expected_path: Path, label: str
) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise IndependentValidationError(f"{label} record is not an object")
    declared = Path(str(record.get("path", ""))).expanduser().resolve()
    if declared != expected_path.expanduser().resolve():
        raise IndependentValidationError(f"{label} path drift")
    if not declared.is_file() or declared.is_symlink():
        raise IndependentValidationError(f"{label} is missing or symlinked")
    observed_hash = sha256(declared)
    if str(record.get("sha256", "")) != observed_hash:
        raise IndependentValidationError(f"{label} hash drift")
    observed_bytes = declared.stat().st_size
    if int(record.get("bytes", -1)) != observed_bytes:
        raise IndependentValidationError(f"{label} byte count drift")
    return {"path": str(declared), "bytes": observed_bytes, "sha256": observed_hash}


def _string_set(
    values: Iterable[Any], label: str, *, allow_empty: bool = False
) -> set[str]:
    if isinstance(values, (str, bytes)):
        raise IndependentValidationError(f"{label} is not an array")
    result = {str(value).strip() for value in values}
    if (not allow_empty and not result) or "" in result:
        raise IndependentValidationError(f"{label} is empty or invalid")
    return result


def verify_checksum_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / "SHA256SUMS.tsv"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise IndependentValidationError("SHA256SUMS.tsv is missing or unsafe")
    records: dict[str, tuple[str, int]] = {}
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        parts = line.split("\t")
        if len(parts) != 3:
            raise IndependentValidationError(f"invalid checksum line {line_number}")
        digest, byte_count, relative = parts
        if relative in records or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise IndependentValidationError("unsafe or duplicate checksum path")
        records[relative] = digest, int(byte_count)
    expected = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.name not in {"SUCCESS.json", "SHA256SUMS.tsv", "INDEPENDENT_VALIDATION.json"}
        and not path.name.endswith(".partial")
    }
    if set(records) != expected:
        raise IndependentValidationError("checksum artifact inventory drift")
    for relative, (declared_hash, declared_bytes) in records.items():
        path = root / relative
        if path.is_symlink() or sha256(path) != declared_hash:
            raise IndependentValidationError(f"checksum drift: {relative}")
        if path.stat().st_size != declared_bytes:
            raise IndependentValidationError(f"byte count drift: {relative}")
    return {
        "path": str(manifest_path),
        "sha256": sha256(manifest_path),
        "artifact_count": len(records),
    }


def _example_graph(checkpoint: Mapping[str, Any]):
    dimensions = checkpoint.get("continuous_feature_dims")
    schema = checkpoint.get("edge_schema")
    if not isinstance(dimensions, Mapping) or set(dimensions) != set(NODE_TYPES):
        raise IndependentValidationError("checkpoint feature dimensions are invalid")
    if not isinstance(schema, list):
        raise IndependentValidationError("checkpoint edge schema is invalid")
    external = [tuple(value) for value in schema if str(value[1]).startswith("external_")]
    features = {
        node_type: np.zeros((1, int(dimensions[node_type])), dtype=np.float32)
        for node_type in NODE_TYPES
    }
    graph = build_context_graph(
        features,
        np.ones((1, 1), dtype=np.float32),
        np.ones((1, 1), dtype=np.float32),
        membership_orientation="gene_by_pathway",
        pathway_family_orientation="pathway_by_family",
        external_relation_types=external,
        source_donors=["validator_source"],
        excluded_donors=["validator_query"],
    )
    if [list(value) for value in graph.edge_schema] != schema:
        raise IndependentValidationError("reconstructed edge schema drift")
    return graph


def rebuild_initial_state_hash(checkpoint: Mapping[str, Any]) -> str:
    config = checkpoint.get("model_config")
    if not isinstance(config, Mapping):
        raise IndependentValidationError("checkpoint model config is absent")
    graph = _example_graph(checkpoint)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(checkpoint["seed"]))
        model = SingleCellHHGT(graph, **dict(config))
    return state_hash(model.state_dict())


def _load_checkpoint(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise IndependentValidationError("checkpoint is not a mapping")
    return value


def _episode_input_contract(
    payload: Mapping[str, Any], episode_record: Mapping[str, Any],
    static_cache: dict[Path, dict[str, Any]],
) -> dict[str, Any]:
    if payload.get("format") != EPISODE_FORMAT:
        raise IndependentValidationError("input episode format mismatch")
    duplicated = STATIC_DUPLICATE_FIELDS.intersection(payload)
    if duplicated:
        raise IndependentValidationError("episode duplicates shared static graph assets")
    episode_id = str(payload.get("episode_id", "")).strip()
    cancer_id = str(payload.get("cancer_id", "")).upper()
    role = str(payload.get("role", ""))
    if not episode_id or not cancer_id or role not in {"fit", "validation", "query"}:
        raise IndependentValidationError("input episode identity failed")
    source = _string_set(payload.get("source_donors", ()), "source_donors")
    excluded = _string_set(payload.get("excluded_donors", ()), "excluded_donors")
    for field in SOURCE_BINDING_FIELDS:
        if _string_set(payload.get(field, ()), field) != source:
            raise IndependentValidationError(f"input episode {field} drift")
    if (
        _string_set(payload.get("geometry_source_donors", ()), "geometry_source_donors")
        != source
        or _string_set(
            payload.get("geometry_excluded_donors", ()), "geometry_excluded_donors"
        )
        != excluded
    ):
        raise IndependentValidationError("input episode geometry donor binding drift")

    static_record = payload.get("static_graph_asset")
    if not isinstance(static_record, Mapping):
        raise IndependentValidationError("episode lacks shared static reference")
    static_path = Path(str(static_record.get("path", ""))).expanduser().resolve()
    normalized_static = verify_record(
        static_record, static_path, f"{episode_id}/static_graph_asset"
    )
    static_contract = static_cache.get(static_path)
    if static_contract is None:
        static = _load_checkpoint(static_path)
        if (
            static.get("format") != STATIC_FORMAT
            or str(static.get("cancer_id", "")).upper() != cancer_id
        ):
            raise IndependentValidationError("shared static identity mismatch")
        bindings: dict[str, dict[str, Any]] = {}
        for name in ("graph_asset_static_success", "matrix_identity"):
            record = static.get(name)
            if not isinstance(record, Mapping):
                raise IndependentValidationError(f"shared static lacks {name}")
            path = Path(str(record.get("path", ""))).expanduser().resolve()
            bindings[name] = verify_record(record, path, f"{episode_id}/{name}")
        static_contract = {
            "cancer_id": cancer_id,
            "static_asset": normalized_static,
            **bindings,
        }
        static_cache[static_path] = static_contract
    elif (
        static_contract["static_asset"] != normalized_static
        or static_contract["cancer_id"] != cancer_id
    ):
        raise IndependentValidationError("shared static record drift")

    pair_by_arm = payload.get("pair_features_by_arm")
    if not isinstance(pair_by_arm, Mapping) or set(pair_by_arm) != set(ARMS):
        raise IndependentValidationError("input episode arm pair features are absent")
    with_lasso = torch.as_tensor(pair_by_arm["with_lasso"], dtype=torch.float32)
    without_lasso = torch.as_tensor(pair_by_arm["without_lasso"], dtype=torch.float32)
    names = tuple(map(str, payload.get("pair_feature_names", ())))
    mask = torch.as_tensor(
        payload.get("lasso_derived_pair_feature_mask", ()), dtype=torch.bool
    )
    target = torch.as_tensor(payload.get("target_rho"), dtype=torch.float32)
    if (
        with_lasso.ndim != 2 or without_lasso.shape != with_lasso.shape
        or target.ndim != 1 or with_lasso.shape[0] != target.numel()
        or target.numel() == 0 or len(names) != with_lasso.shape[1]
        or not bool(torch.isfinite(target).all()) or bool((target.abs() > 1).any())
        or mask.shape != (with_lasso.shape[1],)
        or not bool(torch.isfinite(with_lasso).all())
        or not bool(torch.isfinite(without_lasso).all())
        or bool(torch.count_nonzero(without_lasso[:, mask]))
        or not torch.equal(with_lasso[:, ~mask], without_lasso[:, ~mask])
    ):
        raise IndependentValidationError("input episode matched pair feature contract failed")
    metadata = payload.get("row_metadata")
    if not isinstance(metadata, Mapping):
        raise IndependentValidationError("input episode row metadata is absent")
    rows = pd.DataFrame(dict(metadata))
    missing_columns = set(ROW_ID_COLUMNS).difference(rows)
    if (
        missing_columns or len(rows) != target.numel()
        or rows[list(ROW_ID_COLUMNS)].isna().any().any()
        or rows.duplicated(list(ROW_ID_COLUMNS)).any()
    ):
        raise IndependentValidationError("input episode row identity failed")
    label_donors = _string_set(payload.get("label_donors", ()), "label_donors")
    if set(rows.query_donor_id.astype(str)) != label_donors:
        raise IndependentValidationError("input episode row/label donor drift")
    row_uids = (
        [
            canonical_hash([str(row[column]) for column in ROW_ID_COLUMNS])
            for _, row in rows.iterrows()
        ]
        if role == "query"
        else []
    )
    validation_donors = _string_set(
        payload.get("validation_donors", ()), "validation_donors", allow_empty=True
    )
    expected_support_limited = role == "fit" and not validation_donors
    expected_validation_provider = role == "validation"
    expected_strict_bound = role == "fit" and bool(validation_donors)
    expected_minimum_source = 2 if expected_support_limited else 3
    if (
        payload.get("support_limited_fit") is not expected_support_limited
        or payload.get("validation_provider") is not expected_validation_provider
        or payload.get("strict_validation_bound") is not expected_strict_bound
        or int(payload.get("minimum_source_donors", -1)) != expected_minimum_source
    ):
        raise IndependentValidationError("input episode hybrid donor policy drift")
    inner_fold = payload.get("inner_fold_id")
    cell_key = str(payload.get("cell_key", "")).strip()
    if not cell_key:
        raise IndependentValidationError("input episode cell_key is empty")
    return {
        "episode_id": episode_id,
        "role": role,
        "cancer_id": cancer_id,
        "outer_fold_id": int(payload.get("outer_fold_id", -1)),
        "inner_fold_id": None if inner_fold is None else int(inner_fold),
        "cell_key": cell_key,
        "label_donors": sorted(label_donors),
        "source_donors": sorted(source),
        "excluded_donors": sorted(excluded),
        "outer_train_donors": sorted(
            _string_set(payload.get("outer_train_donors", ()), "outer_train_donors")
        ),
        "outer_query_donors": sorted(
            _string_set(payload.get("outer_query_donors", ()), "outer_query_donors")
        ),
        "inner_target_donors": sorted(
            _string_set(
                payload.get("inner_target_donors", ()), "inner_target_donors",
                allow_empty=True,
            )
        ),
        "validation_donors": sorted(validation_donors),
        "validation_provider": expected_validation_provider,
        "strict_validation_bound": expected_strict_bound,
        "support_limited_fit": expected_support_limited,
        "geometry_isolation": str(payload.get("geometry_isolation", "")),
        "row_count": int(target.numel()),
        "pair_feature_schema_sha256": canonical_hash(
            {"names": names, "lasso_mask": tuple(map(bool, mask.tolist()))}
        ),
        "lasso_pair_feature_columns": int(mask.sum()),
        "lasso_nonzero_by_arm": {
            "with_lasso": int(torch.count_nonzero(with_lasso[:, mask])) if bool(mask.any()) else 0,
            "without_lasso": 0,
        },
        "row_uids": row_uids,
        "rho_sha256": (
            hashlib.sha256(target.contiguous().numpy().tobytes()).hexdigest()
            if role == "query" else None
        ),
        "episode_asset": dict(episode_record),
        **static_contract,
    }


def _replay_model_selection(
    history: list[Mapping[str, Any]], config: Mapping[str, Any]
) -> tuple[int, float]:
    maximum_epochs = int(config.get("epochs", 0))
    patience = int(config.get("patience", 0))
    minimum_improvement = float(config.get("minimum_validation_improvement", np.nan))
    if (
        maximum_epochs <= 0 or patience <= 0 or not np.isfinite(minimum_improvement)
        or minimum_improvement < 0 or not 0 < len(history) <= maximum_epochs
    ):
        raise IndependentValidationError("invalid early-stopping configuration")
    best_loss = float("inf")
    best_epoch = -1
    stale = 0
    for index, epoch in enumerate(history):
        eligible = epoch.get("model_selection_eligible", True) is True
        if not eligible:
            if (
                epoch.get("validation_loss") is not None
                or int(epoch.get("validation_rows", -1)) != 0
                or epoch.get("selected_as_best") is not False
                or int(epoch.get("stale_epochs", -1)) != stale
                or epoch.get("full_fit_episode_coverage") is not False
                or epoch.get("validation_skipped_until_full_fit_episode_coverage")
                is not True
            ):
                raise IndependentValidationError(
                    "pre-coverage model selection was not fully disabled"
                )
            continue
        loss = float(epoch.get("validation_loss", np.nan))
        improved = np.isfinite(loss) and loss < best_loss - minimum_improvement
        if improved:
            best_loss = loss
            best_epoch = index
            stale = 0
        else:
            stale += 1
        if epoch.get("selected_as_best") is not improved or int(
            epoch.get("stale_epochs", -1)
        ) != stale:
            raise IndependentValidationError("early-stopping history cannot be replayed")
        if stale >= patience and index != len(history) - 1:
            raise IndependentValidationError("training continued after patience was exhausted")
    if len(history) < maximum_epochs and stale < patience:
        raise IndependentValidationError("training stopped before epochs/patience authority")
    if best_epoch < 0:
        raise IndependentValidationError("early stopping selected no model")
    return best_epoch, best_loss


def _validate_audit_matches_input(
    audit: Mapping[str, Any], contract: Mapping[str, Any], arm: str, fold: int
) -> None:
    episode_id = str(audit.get("episode_id", ""))
    if int(contract.get("outer_fold_id", -1)) != fold:
        raise IndependentValidationError("episode audit is absent from bound inputs")
    for field in (
        "episode_id", "role", "cancer_id", "outer_fold_id", "inner_fold_id",
        "cell_key", "label_donors", "source_donors", "excluded_donors",
        "outer_train_donors", "outer_query_donors", "inner_target_donors",
        "validation_donors", "validation_provider", "strict_validation_bound",
        "support_limited_fit", "geometry_isolation", "row_count",
        "pair_feature_schema_sha256", "lasso_pair_feature_columns",
        "episode_asset", "static_asset", "graph_asset_static_success",
        "matrix_identity",
    ):
        if canonical_hash(audit.get(field)) != canonical_hash(contract.get(field)):
            raise IndependentValidationError(
                f"episode audit/input drift: {episode_id}/{field}"
            )
    if int(audit.get("lasso_pair_feature_nonzero_values", -1)) != int(
        contract["lasso_nonzero_by_arm"][arm]
    ):
        raise IndependentValidationError("episode pair-feature audit/input drift")


def _validate_episode_audit(value: Mapping[str, Any], arm: str, fold: int) -> None:
    if int(value.get("outer_fold_id", -1)) != fold:
        raise IndependentValidationError("episode audit fold drift")
    role = str(value.get("role", ""))
    source = _string_set(value.get("source_donors", ()), "source_donors")
    labels = _string_set(value.get("label_donors", ()), "label_donors")
    excluded = _string_set(value.get("excluded_donors", ()), "excluded_donors")
    outer_train = _string_set(value.get("outer_train_donors", ()), "outer_train_donors")
    outer_query = _string_set(value.get("outer_query_donors", ()), "outer_query_donors")
    target = _string_set(
        value.get("inner_target_donors", ()), "inner_target_donors", allow_empty=True
    )
    validation = _string_set(
        value.get("validation_donors", ()), "validation_donors", allow_empty=True
    )
    inner_fold = value.get("inner_fold_id")
    if (
        outer_train.intersection(outer_query)
        or (target | validation).difference(outer_train)
        or source.intersection(labels)
        or source.intersection(excluded)
        or source.intersection(outer_query)
    ):
        raise IndependentValidationError("episode audit donor leak")
    if not isinstance(value.get("episode_asset"), Mapping) or not isinstance(
        value.get("static_asset"), Mapping
    ):
        raise IndependentValidationError("episode/shared-static asset audit is absent")
    if not isinstance(value.get("graph_asset_static_success"), Mapping) or not isinstance(
        value.get("matrix_identity"), Mapping
    ):
        raise IndependentValidationError("shared-static upstream binding is absent")
    if int(value.get("lasso_pair_feature_columns", -1)) < 0:
        raise IndependentValidationError("pair feature LASSO mask audit is absent")
    if arm == "without_lasso" and int(
        value.get("lasso_pair_feature_nonzero_values", -1)
    ) != 0:
        raise IndependentValidationError("without-LASSO received LASSO pair features")
    if role == "fit":
        if inner_fold is None or int(inner_fold) == fold:
            raise IndependentValidationError("fit inner target fold is invalid")
        if not target:
            raise IndependentValidationError("fit target donor partition failed")
        if not labels.issubset(target):
            raise IndependentValidationError("fit labels are outside inner target fold")
        if validation:
            if (
                int(inner_fold) == (fold + 1) % N_OUTER_FOLDS
                or target.intersection(validation)
                or value.get("geometry_isolation")
                != "outer_and_inner_target_and_validation_fold"
                or value.get("support_limited_fit") is not False
                or value.get("validation_provider") is not False
                or value.get("strict_validation_bound") is not True
                or len(source) < 3
            ):
                raise IndependentValidationError("strict fit target/validation contract failed")
            if source != outer_train - target - validation:
                raise IndependentValidationError("fit source donors include held-out inner donors")
            if excluded != outer_query | target | validation:
                raise IndependentValidationError("fit excluded donor identity drift")
        else:
            if (
                value.get("geometry_isolation") != "outer_and_inner_target_fold"
                or value.get("support_limited_fit") is not True
                or value.get("validation_provider") is not False
                or value.get("strict_validation_bound") is not False
                or len(source) < 2
            ):
                raise IndependentValidationError("support-limited fit contract failed")
            if source != outer_train - target or excluded != outer_query | target:
                raise IndependentValidationError("support-limited donor identity drift")
        count = int(value.get("lasso_edge_count", -1))
        if count < 0 or (arm == "without_lasso" and count != 0):
            raise IndependentValidationError("LASSO arm semantics failed")
    elif role == "validation":
        if inner_fold is None or int(inner_fold) != (fold + 1) % N_OUTER_FOLDS:
            raise IndependentValidationError("validation fold policy drift")
        if value.get("geometry_isolation") != "outer_and_inner_target_fold":
            raise IndependentValidationError("validation geometry is not target-refit")
        if (
            not validation or target != validation or not labels.issubset(validation)
            or value.get("validation_provider") is not True
            or value.get("strict_validation_bound") is not False
            or value.get("support_limited_fit") is not False
            or len(source) < 3
        ):
            raise IndependentValidationError("validation donor identity drift")
        if source != outer_train - validation or excluded != outer_query | validation:
            raise IndependentValidationError("validation source donor identity drift")
        if arm == "without_lasso" and int(value.get("lasso_edge_count", -1)) != 0:
            raise IndependentValidationError("without-LASSO validation graph has LASSO edges")
    elif role == "query":
        if (
            inner_fold is not None or target or validation
            or value.get("geometry_isolation") != "outer_fold_only"
            or source != outer_train or excluded != outer_query
            or not labels.issubset(outer_query)
            or value.get("support_limited_fit") is not False
            or value.get("validation_provider") is not False
            or value.get("strict_validation_bound") is not False
        ):
            raise IndependentValidationError("outer query donor isolation failed")
        if arm == "without_lasso" and int(value.get("lasso_edge_count", -1)) != 0:
            raise IndependentValidationError("without-LASSO query graph has LASSO edges")
    else:
        raise IndependentValidationError("unknown episode audit role")


def validate_training(root: Path, *, hostname: str | None = None) -> dict[str, Any]:
    require_host_149(hostname)
    root = root.expanduser().resolve()
    success = read_json(root / "SUCCESS.json", "SUCCESS")
    if (
        success.get("format") != SUCCESS_FORMAT
        or success.get("state") != "TRAINING_COMPLETE"
        or success.get("target_host") != "149"
        or int(success.get("outer_folds", -1)) != N_OUTER_FOLDS
        or int(success.get("trained_model_count", -1)) != 10
    ):
        raise IndependentValidationError("top-level SUCCESS contract failed")
    files = success.get("files")
    if not isinstance(files, Mapping):
        raise IndependentValidationError("SUCCESS lacks file records")
    expected_top = {
        "run_config": root / "RUN_CONFIG.json",
        "preflight": root / "LAUNCH_PREFLIGHT.json",
        "input_lineage": root / "INPUT_LINEAGE.json",
        "checkpoint_manifest": root / "CHECKPOINT_MANIFEST.json",
        "oof_predictions": root / "OOF_PREDICTIONS.parquet",
        "checksums": root / "SHA256SUMS.tsv",
    }
    for name, path in expected_top.items():
        verify_record(files.get(name), path, f"SUCCESS/{name}")
    checksum_audit = verify_checksum_manifest(root)
    config = read_json(expected_top["run_config"], "RUN_CONFIG")
    lineage = read_json(expected_top["input_lineage"], "INPUT_LINEAGE")
    manifest = read_json(expected_top["checkpoint_manifest"], "CHECKPOINT_MANIFEST")
    if (
        config.get("format") != CONFIG_FORMAT
        or config.get("target_host") != "149"
        or config.get("deterministic_algorithms") is not True
    ):
        raise IndependentValidationError("run config contract failed")
    architecture = config.get("architecture")
    policy = config.get("episode_policy")
    loss_contract = config.get("loss")
    if (
        not isinstance(architecture, Mapping) or not isinstance(policy, Mapping)
        or not isinstance(loss_contract, Mapping)
    ):
        raise IndependentValidationError("run config architecture/policy absent")
    if (
        architecture.get("family") != "SC_HHGT_WEIGHTED_HGTCONV"
        or architecture.get("frozen_bulk_core") is not False
        or architecture.get("mlp_only") is not False
        or architecture.get("continuous_node_features") is not True
        or architecture.get("signed_edge_weight_path") is not True
    ):
        raise IndependentValidationError("run config is not a real weighted SC-HHGT")
    if (
        loss_contract.get("name") != "SmoothL1Loss"
        or not np.isclose(float(loss_contract.get("beta", np.nan)), 0.1)
        or loss_contract.get("target") != "signed_within_donor_metacell_spearman_rho"
    ):
        raise IndependentValidationError("signed-rho SmoothL1 contract failed")
    required_policy = (
        "precomputed_hash_bound", "lazy_loading", "shared_cancer_static_assets",
        "one_graph_encode_per_episode_step", "bounded_row_sampling",
        "outer_query_excluded_from_all_fitting", "strict_inner_validation",
        "training_geometry_requires_target_and_validation_refit",
        "outer_only_geometry_allowed_for_query", "arm_specific_pair_features",
    )
    if any(policy.get(name) is not True for name in required_policy):
        raise IndependentValidationError("episode isolation policy is weakened")
    if (
        policy.get("fit_lasso_during_optimization") is not False
        or policy.get("outer_query_used_for_model_selection") is not False
        or policy.get("outer_only_geometry_allowed_for_training") is not False
        or int(policy.get("maximum_live_episode_graphs", -1)) != 1
        or int(config.get("max_rows_per_episode_step", 0)) <= 0
        or policy.get("support_limited_fit_geometry") != "outer_and_inner_target_fold"
        or policy.get("support_limited_fit_supplies_validation") is not False
        or int(policy.get("support_limited_fit_minimum_source_donors", -1)) != 2
    ):
        raise IndependentValidationError("optimizer refit LASSO")
    if lineage.get("format") != LINEAGE_FORMAT or lineage.get("status") != "PASS":
        raise IndependentValidationError("input lineage contract failed")
    for name in ("episode_manifest", "graph_asset_manifest", "trainer_source", "model_source"):
        record = lineage.get(name)
        if not isinstance(record, Mapping):
            raise IndependentValidationError(f"lineage lacks {name}")
        declared = Path(str(record.get("path", "")))
        verify_record(record, declared, f"lineage/{name}")
    episode_assets = lineage.get("episode_assets")
    shared_static = lineage.get("shared_static_assets")
    if not isinstance(episode_assets, list) or not episode_assets:
        raise IndependentValidationError("lineage lacks episode asset inventory")
    if not isinstance(shared_static, list) or not shared_static:
        raise IndependentValidationError("lineage lacks shared static inventory")
    episode_contracts: dict[str, dict[str, Any]] = {}
    static_contract_cache: dict[Path, dict[str, Any]] = {}
    for index, record in enumerate(episode_assets):
        if not isinstance(record, Mapping):
            raise IndependentValidationError("invalid episode lineage record")
        path = Path(str(record.get("path", "")))
        normalized_record = verify_record(record, path, f"lineage/episode_assets/{index}")
        payload = _load_checkpoint(path)
        contract = _episode_input_contract(
            payload, normalized_record, static_contract_cache
        )
        if contract["episode_id"] in episode_contracts:
            raise IndependentValidationError("duplicate input episode_id")
        episode_contracts[contract["episode_id"]] = contract
    if lineage.get("old_checkpoint_loaded") is not False or lineage.get("source_checkpoint_sha256") is not None:
        raise IndependentValidationError("historical checkpoint lineage detected")

    if (
        manifest.get("format") != CHECKPOINT_MANIFEST_FORMAT
        or manifest.get("status") != "SUCCESS"
        or int(manifest.get("model_count", -1)) != 10
    ):
        raise IndependentValidationError("checkpoint manifest contract failed")
    models = manifest.get("models")
    if not isinstance(models, list) or len(models) != 10:
        raise IndependentValidationError("checkpoint manifest model inventory failed")
    observed = {(str(value.get("arm")), int(value.get("outer_fold_id", -1))) for value in models}
    expected = {(arm, fold) for arm in ARMS for fold in range(N_OUTER_FOLDS)}
    if observed != expected:
        raise IndependentValidationError("checkpoint manifest lacks exact arm/fold grid")

    prediction_frames: list[pd.DataFrame] = []
    initial_by_arm: dict[str, set[str]] = {arm: set() for arm in ARMS}
    initial_by_fold: dict[int, dict[str, str]] = {fold: {} for fold in range(N_OUTER_FOLDS)}
    audit_by_episode: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for record in models:
        arm = str(record["arm"])
        fold = int(record["outer_fold_id"])
        fold_root = root / f"arm={arm}" / f"outer_fold={fold}"
        fold_files = record.get("files")
        if not isinstance(fold_files, Mapping):
            raise IndependentValidationError("model record lacks files")
        paths = {
            "checkpoint": fold_root / "best_model_state.pt",
            "history": fold_root / "HISTORY.json",
            "predictions": fold_root / "QUERY_PREDICTIONS.parquet",
            "report": fold_root / "TRAINING_REPORT.json",
        }
        for name, path in paths.items():
            verify_record(fold_files.get(name), path, f"{arm}/fold={fold}/{name}")
        verify_record(record.get("success"), fold_root / "SUCCESS.json", f"{arm}/fold={fold}/SUCCESS")
        checkpoint = _load_checkpoint(paths["checkpoint"])
        report = read_json(paths["report"], "TRAINING_REPORT")
        if (
            checkpoint.get("format") != CHECKPOINT_FORMAT
            or checkpoint.get("architecture_family") != "SC_HHGT_WEIGHTED_HGTCONV"
            or checkpoint.get("arm") != arm
            or int(checkpoint.get("outer_fold_id", -1)) != fold
            or checkpoint.get("fresh_random_initialization") is not True
            or checkpoint.get("old_checkpoint_loaded") is not False
        ):
            raise IndependentValidationError("checkpoint identity/lineage failed")
        for field in (
            "arm", "outer_fold_id", "seed", "initial_state_sha256",
            "final_state_sha256", "initial_hgt_state_sha256",
            "final_hgt_state_sha256", "best_epoch", "best_validation_loss",
            "model_config", "continuous_feature_dims", "edge_schema",
        ):
            if canonical_hash(record.get(field)) != canonical_hash(checkpoint.get(field)):
                raise IndependentValidationError(f"checkpoint manifest drift: {field}")
        state = checkpoint.get("state_dict")
        if not isinstance(state, Mapping):
            raise IndependentValidationError("checkpoint lacks model state")
        if not any(name.startswith("encoder.layers.") for name in state):
            raise IndependentValidationError("checkpoint lacks HGTConv parameters")
        if any(token in name for name in state for token in ("lnc_core", "cancer_core", "private_lnc")):
            raise IndependentValidationError("checkpoint contains frozen bulk-core MLP state")
        observed_final = state_hash(state)
        observed_final_hgt = state_hash(state, prefix="encoder.layers.")
        if (
            observed_final != checkpoint.get("final_state_sha256")
            or observed_final_hgt != checkpoint.get("final_hgt_state_sha256")
            or observed_final == checkpoint.get("initial_state_sha256")
            or observed_final_hgt == checkpoint.get("initial_hgt_state_sha256")
        ):
            raise IndependentValidationError("checkpoint parameter change proof failed")
        rebuilt = rebuild_initial_state_hash(checkpoint)
        if rebuilt != checkpoint.get("initial_state_sha256"):
            raise IndependentValidationError("fresh seeded initialization cannot be rebuilt")
        model = SingleCellHHGT(_example_graph(checkpoint), **dict(checkpoint["model_config"]))
        model.load_state_dict(state, strict=True)
        history = checkpoint.get("history")
        if not isinstance(history, list) or not history:
            raise IndependentValidationError("checkpoint history is empty")
        seen_fit_episode_ids: set[str] = set()
        fit_episode_population = int(report.get("fit_episode_count", -1))
        expected_fit_episode_ids = {
            episode_id for episode_id, contract in episode_contracts.items()
            if int(contract["outer_fold_id"]) == fold and contract["role"] == "fit"
        }
        if fit_episode_population <= 0 or fit_episode_population != len(
            expected_fit_episode_ids
        ):
            raise IndependentValidationError("fit episode population/input drift")
        for expected_epoch, epoch in enumerate(history):
            available = int(epoch.get("available_training_rows", 0))
            sampled = int(epoch.get("sampled_training_rows", 0))
            coverage = float(epoch.get("sampling_coverage", np.nan))
            receipts = epoch.get("sampling_receipts")
            if (
                int(epoch.get("epoch", -1)) != expected_epoch
                or int(epoch.get("optimizer_steps", 0)) <= 0
                or float(epoch.get("hgt_gradient_l1_min", 0)) <= 0
                or not np.isfinite(float(epoch.get("mean_train_loss", np.nan)))
                or available <= 0 or sampled <= 0 or sampled > available
                or not np.isfinite(coverage) or not 0 < coverage <= 1
                or abs(coverage - sampled / available) > 1e-12
                or not isinstance(receipts, list) or not receipts
            ):
                raise IndependentValidationError("history gradient/loss proof failed")
            if len(receipts) != int(epoch["optimizer_steps"]):
                raise IndependentValidationError("sampling receipt/optimizer step drift")
            selected_ids = [str(value.get("episode_id", "")) for value in receipts]
            if (
                not all(selected_ids) or len(set(selected_ids)) != len(selected_ids)
                or not set(selected_ids).issubset(expected_fit_episode_ids)
                or int(epoch.get("fit_episode_population", -1))
                != fit_episode_population
                or int(epoch.get("selected_fit_episode_count", -1))
                != len(selected_ids)
                or len(selected_ids) > int(config.get("max_fit_episodes_per_epoch", 0))
                or epoch.get("selected_fit_episode_ids_sha256")
                != canonical_hash(selected_ids)
            ):
                raise IndependentValidationError("fit episode schedule receipt failed")
            seen_fit_episode_ids.update(selected_ids)
            full_coverage = len(seen_fit_episode_ids) == fit_episode_population
            if (
                int(epoch.get("cumulative_fit_episode_count", -1))
                != len(seen_fit_episode_ids)
                or not np.isclose(
                    float(epoch.get("cumulative_fit_episode_coverage", np.nan)),
                    len(seen_fit_episode_ids) / fit_episode_population,
                )
                or epoch.get("full_fit_episode_coverage") is not full_coverage
                or epoch.get("model_selection_eligible") is not full_coverage
                or epoch.get("validation_skipped_until_full_fit_episode_coverage")
                is not (not full_coverage)
            ):
                raise IndependentValidationError("fit episode cumulative coverage drift")
            if full_coverage:
                if (
                    not np.isfinite(float(epoch.get("validation_loss", np.nan)))
                    or int(epoch.get("validation_rows", 0)) <= 0
                ):
                    raise IndependentValidationError("eligible validation evidence failed")
            elif epoch.get("validation_loss") is not None or int(
                epoch.get("validation_rows", -1)
            ) != 0:
                raise IndependentValidationError("validation ran before full fit coverage")
            for receipt in receipts:
                receipt_available = int(receipt.get("available_rows", 0))
                receipt_sampled = int(receipt.get("sampled_rows", 0))
                if (
                    receipt_available <= 0 or receipt_sampled <= 0
                    or receipt_sampled > receipt_available
                    or receipt_sampled > int(config["max_rows_per_episode_step"])
                    or len(str(receipt.get("sample_index_sha256", ""))) != 64
                ):
                    raise IndependentValidationError("bounded sampling receipt failed")
        replayed_epoch, replayed_loss = _replay_model_selection(history, config)
        if len(seen_fit_episode_ids) != fit_episode_population:
            raise IndependentValidationError("not every fit episode reached optimization")
        best_epoch = int(checkpoint.get("best_epoch", -1))
        if (
            best_epoch != replayed_epoch
            or not np.isclose(
                float(checkpoint.get("best_validation_loss", np.nan)),
                replayed_loss,
            )
            or report.get("model_selection_source") != "inner_validation_only"
            or report.get("outer_query_used_for_model_selection") is not False
            or int(report.get("best_epoch", -1)) != best_epoch
            or not np.isclose(
                float(report.get("best_validation_loss", np.nan)),
                float(checkpoint["best_validation_loss"]),
            )
            or report.get("history") != history
            or report.get("loss") != {"name": "SmoothL1Loss", "beta": 0.1}
        ):
            raise IndependentValidationError("inner-validation model selection proof failed")
        history_receipt = read_json(paths["history"], "HISTORY")
        if history_receipt.get("history") != history:
            raise IndependentValidationError("checkpoint/history receipt drift")
        first_step = report.get("first_optimizer_step")
        if (
            not isinstance(first_step, Mapping)
            or first_step.get("hgt_parameters_changed") is not True
            or first_step.get("hgt_parameter_sha256_before")
            == first_step.get("hgt_parameter_sha256_after")
            or float(first_step.get("hgt_gradient_l1", 0)) <= 0
        ):
            raise IndependentValidationError("first optimizer-step HGT update proof failed")
        audits = report.get("episode_audits")
        if not isinstance(audits, list) or not audits:
            raise IndependentValidationError("training report lacks episode audits")
        for audit in audits:
            episode_id = str(audit.get("episode_id", ""))
            contract = episode_contracts.get(episode_id)
            if contract is None:
                raise IndependentValidationError("episode audit is absent from bound inputs")
            _validate_audit_matches_input(audit, contract, arm, fold)
            _validate_episode_audit(audit, arm, fold)
            for asset_name in (
                "episode_asset", "static_asset", "graph_asset_static_success",
                "matrix_identity",
            ):
                asset = audit[asset_name]
                declared = Path(str(asset.get("path", "")))
                verify_record(
                    asset, declared,
                    f"{arm}/fold={fold}/{audit['episode_id']}/{asset_name}",
                )
            audit_by_episode[(arm, fold, str(audit["episode_id"]))] = audit
        expected_episode_ids = {
            episode_id for episode_id, contract in episode_contracts.items()
            if int(contract["outer_fold_id"]) == fold
        }
        if {str(value.get("episode_id", "")) for value in audits} != expected_episode_ids:
            raise IndependentValidationError("fold episode audit inventory/input drift")
        observed_roles = {str(value.get("role")) for value in audits}
        if observed_roles != {"fit", "validation", "query"}:
            raise IndependentValidationError("fit/validation/query audit inventory failed")
        role_counts = {
            role_name: sum(value.get("role") == role_name for value in audits)
            for role_name in ("fit", "validation", "query")
        }
        if (
            role_counts["fit"] <= 0 or role_counts["validation"] <= 0
            or role_counts["query"] <= 0
            or int(report.get("fit_episode_count", -1)) != role_counts["fit"]
            or int(report.get("validation_episode_count", -1)) != role_counts["validation"]
            or int(report.get("query_episode_count", -1)) != role_counts["query"]
            or report.get("lazy_episode_loading") is not True
            or int(report.get("maximum_live_episode_graphs", -1)) != 1
            or report.get("one_graph_encode_per_episode_step") is not True
            or report.get("fit_episode_coverage_before_model_selection") is not True
            or int(report.get("max_fit_episodes_per_epoch", -1))
            != int(config.get("max_fit_episodes_per_epoch", -2))
        ):
            raise IndependentValidationError("lazy episode/encode execution proof failed")
        if int(report.get("external_training_edge_count", 0)) <= 0:
            raise IndependentValidationError("external evidence was not used in training")
        lasso_count = int(report.get("lasso_training_edge_count", -1))
        if (arm == "with_lasso" and lasso_count <= 0) or (
            arm == "without_lasso" and lasso_count != 0
        ):
            raise IndependentValidationError("matched LASSO ablation proof failed")
        frame = pd.read_parquet(paths["predictions"])
        missing = set(ROW_ID_COLUMNS + ("rho", "predicted_rho", "row_uid", "episode_id")) - set(frame)
        if missing or frame.empty:
            raise IndependentValidationError(f"prediction schema failed: {sorted(missing)}")
        if int(report.get("query_prediction_count", -1)) != len(frame):
            raise IndependentValidationError("query prediction count receipt drift")
        if (
            not np.isfinite(frame[["rho", "predicted_rho"]].to_numpy(float)).all()
            or (frame.predicted_rho.abs() > 1 + 1e-6).any()
            or (frame.rho.abs() > 1 + 1e-6).any()
        ):
            raise IndependentValidationError("signed prediction values are invalid")
        expected_query_rows = {
            str(value["episode_id"]): int(value["row_count"])
            for value in audits if value.get("role") == "query"
        }
        observed_query_rows = {
            str(episode_id): len(part)
            for episode_id, part in frame.groupby("episode_id", sort=False)
        }
        if observed_query_rows != expected_query_rows:
            raise IndependentValidationError("query prediction episode/row inventory drift")
        for episode_id, part in frame.groupby("episode_id"):
            audit = audit_by_episode.get((arm, fold, str(episode_id)))
            if audit is None or audit.get("role") != "query":
                raise IndependentValidationError("prediction lacks query episode audit")
            contract = episode_contracts[str(episode_id)]
            if (
                part.row_uid.astype(str).tolist() != contract["row_uids"]
                or hashlib.sha256(
                    part.rho.to_numpy(np.float32).tobytes()
                ).hexdigest() != contract["rho_sha256"]
            ):
                raise IndependentValidationError("prediction/input row identity or rho drift")
            if not set(part.query_donor_id.astype(str)).issubset(set(audit["label_donors"])):
                raise IndependentValidationError("prediction donor identity drift")
        prediction_frames.append(frame)
        initial = str(checkpoint["initial_state_sha256"])
        initial_by_arm[arm].add(initial)
        initial_by_fold[fold][arm] = initial

    if any(len(values) != N_OUTER_FOLDS for values in initial_by_arm.values()):
        raise IndependentValidationError("fold initializations are not independent")
    if any(values.get(ARMS[0]) != values.get(ARMS[1]) for values in initial_by_fold.values()):
        raise IndependentValidationError("ablation arms do not have matched initialization")
    for fold in range(N_OUTER_FOLDS):
        paired = [value for value in models if int(value["outer_fold_id"]) == fold]
        if len({canonical_hash(value.get("edge_schema")) for value in paired}) != 1:
            raise IndependentValidationError("ablation edge schemas are not identical")
        if len({canonical_hash(value.get("continuous_feature_dims")) for value in paired}) != 1:
            raise IndependentValidationError("ablation feature schemas are not identical")
        paired_audits = {
            arm: {
                episode_id: value
                for (audit_arm, audit_fold, episode_id), value in audit_by_episode.items()
                if audit_arm == arm and audit_fold == fold
            }
            for arm in ARMS
        }
        if set(paired_audits[ARMS[0]]) != set(paired_audits[ARMS[1]]):
            raise IndependentValidationError("ablation episode inventories are not matched")
        for episode_id in paired_audits[ARMS[0]]:
            primary = paired_audits[ARMS[0]][episode_id]
            ablation = paired_audits[ARMS[1]][episode_id]
            for field in (
                "pair_feature_schema_sha256", "row_count", "episode_asset",
                "static_asset", "graph_asset_static_success", "matrix_identity",
            ):
                if canonical_hash(primary.get(field)) != canonical_hash(ablation.get(field)):
                    raise IndependentValidationError("ablation episode schema/lineage drift")
            if int(ablation.get("lasso_pair_feature_nonzero_values", -1)) != 0:
                raise IndependentValidationError("without-LASSO pair feature leak")
    combined = pd.concat(prediction_frames, ignore_index=True)
    for fold in range(N_OUTER_FOLDS):
        left = set(
            combined.loc[
                combined.arm.eq("with_lasso") & combined.outer_fold_id.eq(fold), "row_uid"
            ]
        )
        right = set(
            combined.loc[
                combined.arm.eq("without_lasso") & combined.outer_fold_id.eq(fold), "row_uid"
            ]
        )
        if not left or left != right:
            raise IndependentValidationError("ablation prediction rows are not matched")
    top_oof = pd.read_parquet(root / "OOF_PREDICTIONS.parquet")
    prediction_key = ["arm", "outer_fold_id", "row_uid"]
    required_oof = set(
        prediction_key + ["episode_id", "rho", "predicted_rho", "query_donor_id"]
    )
    if (
        required_oof.difference(top_oof)
        or len(top_oof) != len(combined)
        or top_oof.duplicated(prediction_key).any()
        or combined.duplicated(prediction_key).any()
    ):
        raise IndependentValidationError("top-level OOF prediction inventory drift")
    left = top_oof.sort_values(prediction_key).reset_index(drop=True)
    right = combined.sort_values(prediction_key).reset_index(drop=True)
    if (
        left[prediction_key + ["episode_id", "query_donor_id"]].astype(str).to_numpy().tolist()
        != right[prediction_key + ["episode_id", "query_donor_id"]].astype(str).to_numpy().tolist()
        or not np.array_equal(
            left[["rho", "predicted_rho"]].to_numpy(np.float32),
            right[["rho", "predicted_rho"]].to_numpy(np.float32),
        )
    ):
        raise IndependentValidationError("top-level OOF prediction value drift")

    return {
        "format": VALIDATION_FORMAT,
        "status": "PASS",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_host": "149",
        "training_root": str(root),
        "training_success_sha256": sha256(root / "SUCCESS.json"),
        "checkpoint_manifest_sha256": sha256(root / "CHECKPOINT_MANIFEST.json"),
        "checksum_audit": checksum_audit,
        "validated_model_count": 10,
        "validated_outer_folds": list(range(N_OUTER_FOLDS)),
        "validated_arms": list(ARMS),
        "prediction_rows": len(top_oof),
        "all_hgt_gradients_positive": True,
        "all_hgt_parameters_changed": True,
        "fresh_fold_initializations_independent": True,
        "ablation_initializations_matched": True,
        "outer_query_donor_isolation": True,
        "inner_training_donor_isolation": True,
        "nested_refit_training_geometry": True,
        "strict_inner_validation_model_selection": True,
        "lazy_episode_loading": True,
        "shared_static_graph_assets": True,
        "one_graph_encode_per_episode_step": True,
        "bounded_row_sampling_audited": True,
        "arm_specific_lasso_pair_features": True,
        "support_limited_fit_without_validation_selection": True,
        "external_evidence_used": True,
        "signed_rho_smooth_l1": True,
        "frozen_bulk_core_mlp_used": False,
    }


def run_validation(
    root: Path,
    *,
    output_path: Path | None = None,
    hostname: str | None = None,
) -> dict[str, Any]:
    result = validate_training(root, hostname=hostname)
    destination = output_path or root.expanduser().resolve() / "INDEPENDENT_VALIDATION.json"
    atomic_json(destination, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    values = parser.parse_args(argv)
    result = run_validation(values.training_root, output_path=values.output)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
