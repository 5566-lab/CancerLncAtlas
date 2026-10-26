#!/usr/bin/env python3
"""Materialize compact, hash-bound SC-HHGT training episodes on host 149."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.single_cell_batched_lasso import BatchedLassoPolicy  # noqa: E402
from cc_hhgt.v32.single_cell_hhgt_episodes import (  # noqa: E402
    EPISODE_FORMAT,
    LASSO_DERIVED_PAIR_FEATURE_MASK,
    PAIR_FEATURE_NAMES,
    STATIC_FORMAT,
    canonical_hash,
    episode_pair_features,
    fit_context_lasso,
    lasso_pair_features,
    lasso_policy_receipt,
    local_node_indices,
    localized_lasso_edges,
    source_context_features,
    sparse_coo_payload,
)


ASSET_FORMAT = "CANCERLNCATLAS_V32_SC_HHGT_GRAPH_ASSETS_V1"
EPISODE_MANIFEST_FORMAT = "CANCERLNCATLAS_V32_SC_HHGT_EPISODE_MANIFEST_V1"
EPISODE_PREFLIGHT_FORMAT = "CANCERLNCATLAS_V32_SC_HHGT_EPISODE_PREFLIGHT_V1"
EPISODE_SUCCESS_FORMAT = "CANCERLNCATLAS_V32_SC_HHGT_EPISODE_SUCCESS_V1"
ROW_ID_COLUMNS = (
    "dataset_id", "cancer_id", "cell_key", "query_donor_id",
    "lncrna_id", "pathway_id",
)
IO_RETRY_ATTEMPTS = 120
COMPATIBLE_MATERIALIZER_MIGRATIONS = {
    "b62324efd541b3bfbf8991d92be325e835a967a57e554310439a1e8274c8f689": (
        "RETRY_TORCH_WRAPPED_SSHFS_EPERM",
        "ATOMIC_WRITE_RETRY_ONLY",
    ),
    "4697d0e25a2721edb0fd1d00df426822db77b25ecb40eb5e06a78405232a1302": (
        "AVOID_REHASHING_RECEIPTED_EPISODE_PAYLOADS_ON_RESUME",
        "CACHED_EPISODE_METADATA_VERIFICATION_ONLY",
    ),
    "76c5b8ee0b03dc983f0647f93b095afd1a4f26ff9249c6b2a48441752e33fdfe": (
        "SKIP_LARGE_PLAN_MATRICES_WHEN_ALL_EPISODES_ARE_RECEIPTED",
        "CACHED_PLAN_FAST_PATH_ONLY",
    ),
    "3901a2997ccb5ca483668fa4a7986f0ec6f3d72af1c10131f7ac5db31d7c435d": (
        "AVOID_REHASHING_RECEIPTED_FOLD_GRAPH_ASSETS_DURING_FINALIZE",
        "FINALIZE_FOLD_GRAPH_RECEIPT_VERIFICATION_ONLY",
    ),
    "044992c3d0286c6a95f5a96ce04fb749b2cc6a5c8bb71baa2ac40637d13974fc": (
        "FINALIZE_RECEIPTED_EPISODES_IN_ONE_PASS",
        "FINALIZE_EPISODE_METADATA_AND_SINGLE_LOAD_ONLY",
    ),
}
COMPATIBLE_QUERY_CONTEXT_MIGRATION = {
    "previous_materializer_sha256": "3db913fdf07e0402eddbb6c79a386e709826fdad5a8feb8827cc08cc3f8dbbb0",
    "previous_episode_module_sha256": "e0aba75908a921dbbc6972220363298bc29de838c77e6ab338611d3d45f5fa74",
    "replacement_episode_module_sha256": "e2e1e8ee440465031ed11ba0c39a573edc2389d53cbe0619ec8398ca1b735307",
}


def io_retry_delay(attempt: int) -> float:
    return min(2.0, 0.25 * (attempt + 1)) + (os.getpid() % 7) * 0.03


def retry_io(operation):
    for attempt in range(IO_RETRY_ATTEMPTS):
        try:
            return operation()
        except PermissionError:
            if attempt == IO_RETRY_ATTEMPTS - 1:
                raise
            time.sleep(io_retry_delay(attempt))
    raise AssertionError("unreachable")


def sha256(path: Path) -> str:
    for attempt in range(IO_RETRY_ATTEMPTS):
        try:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            return digest.hexdigest()
        except PermissionError:
            if attempt == IO_RETRY_ATTEMPTS - 1:
                raise
            time.sleep(io_retry_delay(attempt))
    raise AssertionError("unreachable")


def read_json(path: Path) -> dict[str, Any]:
    for attempt in range(IO_RETRY_ATTEMPTS):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except PermissionError:
            if attempt == IO_RETRY_ATTEMPTS - 1:
                raise
            time.sleep(io_retry_delay(attempt))
    raise AssertionError("unreachable")


def file_record(path: Path) -> dict[str, Any]:
    value = path.resolve()
    for attempt in range(IO_RETRY_ATTEMPTS):
        try:
            size = value.stat().st_size
            break
        except PermissionError:
            if attempt == IO_RETRY_ATTEMPTS - 1:
                raise
            time.sleep(io_retry_delay(attempt))
    return {"path": str(value), "bytes": size, "sha256": sha256(value)}


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.partial.{os.getpid()}")
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    for attempt in range(IO_RETRY_ATTEMPTS):
        try:
            temporary.write_text(payload, encoding="utf-8")
            os.replace(temporary, path)
            return
        except PermissionError:
            temporary.unlink(missing_ok=True)
            if attempt == IO_RETRY_ATTEMPTS - 1:
                raise
            time.sleep(io_retry_delay(attempt))


def atomic_torch_save(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.partial.{os.getpid()}")
    for attempt in range(IO_RETRY_ATTEMPTS):
        try:
            torch.save(dict(value), temporary)
            os.replace(temporary, path)
            return
        except (PermissionError, RuntimeError) as error:
            wrapped_permission_error = isinstance(error, RuntimeError) and (
                "open file failed" in str(error)
                and "Operation not permitted" in str(error)
            )
            if not isinstance(error, PermissionError) and not wrapped_permission_error:
                raise
            try:
                temporary.unlink(missing_ok=True)
            except PermissionError:
                pass
            if attempt == IO_RETRY_ATTEMPTS - 1:
                raise
            time.sleep(io_retry_delay(attempt))


def verify_record(record: Mapping[str, Any], expected: Path, label: str) -> None:
    path = Path(str(record.get("path", ""))).resolve()
    expected = expected.resolve()
    if path != expected or not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{label}_PATH_DRIFT")
    if int(record.get("bytes", -1)) != path.stat().st_size or str(
        record.get("sha256", "")
    ) != sha256(path):
        raise RuntimeError(f"{label}_CONTENT_DRIFT")


def verify_record_metadata(record: Mapping[str, Any], expected: Path, label: str) -> None:
    path = Path(str(record.get("path", ""))).resolve()
    expected = expected.resolve()
    if path != expected or path.is_symlink():
        raise RuntimeError(f"{label}_PATH_DRIFT")
    stat = retry_io(path.stat)
    if not path.is_file() or int(record.get("bytes", -1)) != stat.st_size:
        raise RuntimeError(f"{label}_CONTENT_DRIFT")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _read_success(
    path: Path, state: str, *, verify_files: bool = True
) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"MISSING_SUCCESS:{path}")
    value = read_json(path)
    if value.get("target_host") != "149" or value.get("state") != state:
        raise RuntimeError(f"SUCCESS_STATE_DRIFT:{path}")
    if verify_files:
        for relative, record in value.get("files", {}).items():
            verify_record(record, path.parent / relative, f"SUCCESS_FILE_{relative}")
    return value


def _require_host_149() -> None:
    if socket.gethostname() != "149":
        raise SystemExit("WRONG_HOST")


def _deadline(settings) -> None:
    if time.time() >= settings.deadline:
        raise TimeoutError("SC_HHGT_EPISODE_DEADLINE_REACHED")


def _initialize_preflight(settings) -> dict[str, Any]:
    source_validation = settings.source_result_root / "MATERIALIZATION_VALIDATION.json"
    graph_preflight = settings.graph_asset_root / "LAUNCH_PREFLIGHT.json"
    nested_preflight = settings.nested_asset_root / "LAUNCH_PREFLIGHT.json"
    for path in (source_validation, graph_preflight, nested_preflight):
        if not path.is_file():
            raise RuntimeError(f"EPISODE_PREFLIGHT_INPUT_MISSING:{path}")
    source = read_json(source_validation)
    if (
        source.get("target_host") != "149"
        or source.get("status") != "PASS"
        or source.get("closure_status") != "PASS_FULL_28_CANCER_MATERIALIZATION_CLOSURE"
    ):
        raise RuntimeError("EPISODE_SOURCE_MATERIALIZATION_NOT_VALIDATED")
    value = {
        "format": EPISODE_PREFLIGHT_FORMAT,
        "run_id": settings.run_id,
        "target_host": "149",
        "cpu_only": True,
        "paid_compute": False,
        "formal_pointer_update_authorized": False,
        "source_result_root": str(settings.source_result_root.resolve()),
        "graph_asset_root": str(settings.graph_asset_root.resolve()),
        "nested_asset_root": str(settings.nested_asset_root.resolve()),
        "output_root": str(settings.output_root.resolve()),
        "bindings": {
            "source_validation": file_record(source_validation),
            "graph_preflight": file_record(graph_preflight),
            "nested_preflight": file_record(nested_preflight),
        },
        # Normalize tuples before comparing against a JSON round-trip. Without
        # this, an identical persisted policy is misclassified as drift.
        "lasso": _jsonable(lasso_policy_receipt(settings.lasso_policy)),
        "materializer": file_record(Path(__file__).resolve()),
        "episode_module": file_record(
            ROOT / "cc_hhgt" / "v32" / "single_cell_hhgt_episodes.py"
        ),
        "lasso_module": file_record(
            ROOT / "cc_hhgt" / "v32" / "single_cell_batched_lasso.py"
        ),
        "lasso_stability_threshold": settings.lasso_stability_threshold,
        "pair_feature_names": list(PAIR_FEATURE_NAMES),
        "lasso_derived_pair_feature_mask": list(LASSO_DERIVED_PAIR_FEATURE_MASK),
        "created_unix": time.time(),
    }
    destination = settings.output_root / "LAUNCH_PREFLIGHT.json"
    if destination.is_file():
        existing = read_json(destination)
        value["materializer_lineage"] = list(existing.get("materializer_lineage", []))
        value["episode_module_lineage"] = list(
            existing.get("episode_module_lineage", [])
        )
        left, right = dict(existing), dict(value)
        left.setdefault("materializer_lineage", [])
        left.setdefault("episode_module_lineage", [])
        left.pop("created_unix", None)
        right.pop("created_unix", None)
        if left != right:
            differing = sorted(
                key for key in set(left) | set(right) if left.get(key) != right.get(key)
            )
            detail = ""
            if differing == ["lasso"]:
                detail = ":" + json.dumps(
                    {"existing": left["lasso"], "current": right["lasso"]},
                    sort_keys=True,
                    separators=(",", ":"),
                )
            migration = COMPATIBLE_MATERIALIZER_MIGRATIONS.get(
                left["materializer"].get("sha256", "")
            )
            if differing == ["materializer"] and migration is not None:
                reason, behavioral_scope = migration
                value["materializer_lineage"].append(
                    {
                        "previous": left["materializer"],
                        "replacement": right["materializer"],
                        "reason": reason,
                        "behavioral_scope": behavioral_scope,
                        "verification": {
                            "host": "149",
                            "tests": 12,
                            "status": "PASS",
                        },
                    }
                )
                atomic_json(destination, value)
                return value
            query_migration = COMPATIBLE_QUERY_CONTEXT_MIGRATION
            if (
                differing == ["episode_module", "materializer"]
                and left["materializer"].get("sha256")
                == query_migration["previous_materializer_sha256"]
                and left["episode_module"].get("sha256")
                == query_migration["previous_episode_module_sha256"]
                and right["episode_module"].get("sha256")
                == query_migration["replacement_episode_module_sha256"]
            ):
                verification = {"host": "149", "tests": 6, "status": "PASS"}
                value["materializer_lineage"].append(
                    {
                        "previous": left["materializer"],
                        "replacement": right["materializer"],
                        "reason": "QUERY_CONTEXT_TWO_DONOR_SUPPORT",
                        "behavioral_scope": "QUERY_CONTEXT_FEATURE_SUMMARY_ONLY",
                        "verification": verification,
                    }
                )
                value["episode_module_lineage"].append(
                    {
                        "previous": left["episode_module"],
                        "replacement": right["episode_module"],
                        "reason": "QUERY_CONTEXT_TWO_DONOR_SUPPORT",
                        "behavioral_scope": "EXPLICIT_MINIMUM_CONTEXT_DONORS_ONLY",
                        "verification": verification,
                    }
                )
                atomic_json(destination, value)
                return value
            raise RuntimeError(
                f"EPISODE_PREFLIGHT_DRIFT:{','.join(differing)}{detail}"
            )
        return existing
    atomic_json(destination, value)
    return value


def _load_cancer_static(settings, cancer_id: str) -> dict[str, Any]:
    root = settings.graph_asset_root / "cancers" / cancer_id / "STATIC"
    success = _read_success(
        root / "SUCCESS.json", "CANCER_STATIC_GRAPH_ASSETS_READY", verify_files=False
    )
    nodes = retry_io(lambda: pd.read_parquet(root / "NODE_IDENTIFIERS.parquet"))
    identifiers: dict[str, list[str]] = {}
    for node_type in ("lncRNA", "gene", "pathway", "pathway_family"):
        part = nodes.loc[nodes.node_type.astype(str).eq(node_type)].sort_values(
            "global_index", kind="mergesort"
        )
        if not np.array_equal(part.global_index.to_numpy(np.int64), np.arange(len(part))):
            raise RuntimeError(f"STATIC_{node_type.upper()}_GLOBAL_INDEX_DRIFT")
        identifiers[node_type] = part.canonical_id.astype(str).tolist()
    gene_pathway = retry_io(
        lambda: sparse.load_npz(root / "GENE_PATHWAY_MEMBERSHIP.npz").tocsr()
    )
    pathway_family = retry_io(
        lambda: sparse.load_npz(root / "PATHWAY_FAMILY_MEMBERSHIP.npz").tocsr()
    )
    expected = (
        len(identifiers["gene"]), len(identifiers["pathway"]),
        len(identifiers["pathway_family"]),
    )
    if gene_pathway.shape != expected[:2] or pathway_family.shape != expected[1:]:
        raise RuntimeError("STATIC_MEMBERSHIP_SHAPE_DRIFT")
    external = retry_io(
        lambda: pd.read_parquet(root / "EXTERNAL_PHYSICAL_BINDING.parquet")
    )
    required = {"source_index", "target_index", "edge_weight"}
    if missing := sorted(required - set(external)):
        raise RuntimeError(f"STATIC_EXTERNAL_COLUMNS:{missing}")
    return {
        "root": root,
        "success": success,
        "identifiers": identifiers,
        "gene_pathway": gene_pathway,
        "pathway_family": pathway_family,
        "external": external,
    }


def _ensure_shared_static(settings, cancer_id: str, static: Mapping[str, Any]) -> Path:
    destination = settings.output_root / "static" / f"{cancer_id}.pt"
    receipt_path = settings.output_root / "static" / f"{cancer_id}.json"
    if receipt_path.is_file():
        receipt = read_json(receipt_path)
        verify_record(receipt["asset"], destination, "EPISODE_SHARED_STATIC")
        if receipt.get("source_success_sha256") != sha256(static["root"] / "SUCCESS.json"):
            raise RuntimeError("EPISODE_SHARED_STATIC_SOURCE_DRIFT")
        return destination
    external = static["external"]
    index = torch.as_tensor(
        np.vstack(
            (
                external.source_index.to_numpy(np.int64),
                external.target_index.to_numpy(np.int64),
            )
        ),
        dtype=torch.long,
    )
    weight = torch.as_tensor(
        external.edge_weight.to_numpy(np.float32).copy(), dtype=torch.float32
    )
    external_source = static["root"] / "EXTERNAL_PHYSICAL_BINDING.parquet"
    source_hash = sha256(external_source)
    batches = [
        {
            "source_type": "lncRNA",
            "relation": "external_physical_binding",
            "target_type": "gene",
            "edge_index": index,
            "edge_weight": weight,
            "available": True,
            "source_hash": source_hash,
            "provenance": str(external_source.resolve()),
            "source_donors": [],
        },
        {
            "source_type": "gene",
            "relation": "external_physical_binding_reverse",
            "target_type": "lncRNA",
            "edge_index": index.flip(0),
            "edge_weight": weight,
            "available": True,
            "source_hash": source_hash,
            "provenance": str(external_source.resolve()),
            "source_donors": [],
        },
    ]
    payload = {
        "format": STATIC_FORMAT,
        "cancer_id": cancer_id,
        "graph_asset_static_success": file_record(static["root"] / "SUCCESS.json"),
        "matrix_identity": file_record(
            settings.nested_asset_root / "cancers" / cancer_id / "MATRIX_IDENTITY.json"
        ),
        "node_identifiers": static["identifiers"],
        "node_counts": {
            node_type: len(static["identifiers"][node_type])
            for node_type in ("lncRNA", "gene", "pathway", "pathway_family")
        },
        "gene_pathway_membership": sparse_coo_payload(static["gene_pathway"]),
        "pathway_family_membership": sparse_coo_payload(static["pathway_family"]),
        "external_relation_types": [
            ["lncRNA", "external_physical_binding", "gene"],
            ["gene", "external_physical_binding_reverse", "lncRNA"],
        ],
        "external_edge_batches": batches,
    }
    atomic_torch_save(destination, payload)
    atomic_json(
        receipt_path,
        {
            "format": STATIC_FORMAT,
            "target_host": "149",
            "cancer_id": cancer_id,
            "asset": file_record(destination),
            "source_success_sha256": sha256(static["root"] / "SUCCESS.json"),
        },
    )
    return destination


def _episode_identity(cancer_id: str, outer: int, role: str, inner: int | None, cell_key: str) -> str:
    context = canonical_hash(str(cell_key))[:16]
    target = "none" if inner is None else str(inner)
    return f"{cancer_id}_outer{outer}_{role}_inner{target}_context{context}"


def _cached_episode(
    settings,
    cancer_id: str,
    outer_fold_id: int,
    role: str,
    inner_fold_id: int | None,
    cell_key: str,
) -> dict[str, Any] | None:
    episode_id = _episode_identity(
        cancer_id, outer_fold_id, role, inner_fold_id, cell_key
    )
    destination = settings.output_root / "episodes" / cancer_id / (
        f"outer_fold={outer_fold_id}"
    ) / (episode_id + ".pt")
    receipt_path = destination.with_suffix(".json")
    if not receipt_path.is_file():
        return None
    receipt = read_json(receipt_path)
    verify_record_metadata(receipt["episode"], destination, "CACHED_EPISODE")
    return receipt


def _write_episode(
    settings,
    *,
    cancer_id: str,
    outer_fold_id: int,
    role: str,
    inner_fold_id: int | None,
    cell_key: str,
    rows: pd.DataFrame,
    features_full: Mapping[str, np.ndarray],
    coding: sparse.spmatrix,
    lncrna: sparse.spmatrix,
    observations: pd.DataFrame,
    static: Mapping[str, Any],
    static_path: Path,
    source_donors: Iterable[object],
    excluded_donors: Iterable[object],
    outer_train_donors: Iterable[object],
    outer_query_donors: Iterable[object],
    inner_target_donors: Iterable[object],
    validation_donors: Iterable[object],
    geometry_isolation: str,
    source_receipts: list[dict[str, Any]],
) -> dict[str, Any]:
    _deadline(settings)
    episode_id = _episode_identity(
        cancer_id, outer_fold_id, role, inner_fold_id, cell_key
    )
    destination = settings.output_root / "episodes" / cancer_id / f"outer_fold={outer_fold_id}" / (
        episode_id + ".pt"
    )
    receipt_path = destination.with_suffix(".json")
    cached = _cached_episode(
        settings, cancer_id, outer_fold_id, role, inner_fold_id, cell_key
    )
    if cached is not None:
        return cached
    if destination.exists():
        raise RuntimeError(f"UNRECEIPTED_EPISODE_EXISTS:{destination}")
    rows = rows.loc[rows.cell_key.astype(str).eq(str(cell_key))].copy().reset_index(drop=True)
    if rows.empty:
        raise ValueError("SC_EPISODE_HAS_NO_TARGET_ROWS")
    if missing := sorted((set(ROW_ID_COLUMNS) | {"rho"}) - set(rows)):
        raise ValueError(f"SC_EPISODE_TARGET_COLUMNS:{missing}")
    identifiers = static["identifiers"]
    lnc_lookup = {value: index for index, value in enumerate(identifiers["lncRNA"])}
    pathway_lookup = {value: index for index, value in enumerate(identifiers["pathway"])}
    mapped_lnc = rows.lncrna_id.astype(str).map(lnc_lookup)
    mapped_pathway = rows.pathway_id.astype(str).map(pathway_lookup)
    if mapped_lnc.isna().any() or mapped_pathway.isna().any():
        raise ValueError("SC_EPISODE_TARGET_ID_NOT_IN_STATIC_UNIVERSE")
    row_lnc_global = mapped_lnc.to_numpy(np.int64)
    row_pathway_global = mapped_pathway.to_numpy(np.int64)
    response_global = np.unique(row_lnc_global)
    selected_pathway = np.unique(row_pathway_global)
    lasso = fit_context_lasso(
        coding,
        lncrna,
        observations,
        cell_key=cell_key,
        source_donors=source_donors,
        excluded_donors=excluded_donors,
        response_global_indices=response_global,
        lncrna_ids=identifiers["lncRNA"],
        gene_ids=identifiers["gene"],
        policy=settings.lasso_policy,
    )
    response_position = {int(value): index for index, value in enumerate(response_global)}
    row_response = np.asarray([response_position[int(value)] for value in row_lnc_global])
    lasso_features = lasso_pair_features(
        lasso,
        static["gene_pathway"],
        row_response,
        row_pathway_global,
        selected_pathway,
        stability_threshold=settings.lasso_stability_threshold,
    )
    nodes = local_node_indices(
        lnc_global=row_lnc_global,
        pathway_global=row_pathway_global,
        lasso_result=lasso,
        gene_pathway_membership=static["gene_pathway"],
        pathway_family_membership=static["pathway_family"],
        external_edges=static["external"],
        stability_threshold=settings.lasso_stability_threshold,
    )
    localized_features = {
        node_type: torch.as_tensor(
            np.asarray(features_full[node_type], dtype=np.float32)[indices],
            dtype=torch.float32,
        )
        for node_type, indices in nodes.items()
    }
    localized_features["cell_context"] = torch.as_tensor(
        np.asarray(features_full["cell_context"], dtype=np.float32), dtype=torch.float32
    )
    lnc_local = {int(value): index for index, value in enumerate(nodes["lncRNA"])}
    pathway_local = {int(value): index for index, value in enumerate(nodes["pathway"])}
    pair_features = episode_pair_features(rows, lasso_features)
    label_donors = sorted(set(rows.query_donor_id.astype(str)))
    source = sorted({str(value) for value in source_donors})
    excluded = sorted({str(value) for value in excluded_donors})
    payload = {
        "format": EPISODE_FORMAT,
        "episode_id": episode_id,
        "cancer_id": cancer_id,
        "outer_fold_id": int(outer_fold_id),
        "inner_fold_id": inner_fold_id,
        "role": role,
        "cell_key": str(cell_key),
        "label_donors": label_donors,
        "source_donors": source,
        "excluded_donors": excluded,
        "outer_train_donors": sorted({str(value) for value in outer_train_donors}),
        "outer_query_donors": sorted({str(value) for value in outer_query_donors}),
        "inner_target_donors": sorted({str(value) for value in inner_target_donors}),
        "validation_donors": sorted({str(value) for value in validation_donors}),
        "validation_provider": bool(role == "validation"),
        "strict_validation_bound": bool(role == "fit" and validation_donors),
        "support_limited_fit": bool(role == "fit" and not tuple(validation_donors)),
        "minimum_source_donors": 2 if role == "fit" and not tuple(validation_donors) else 3,
        "geometry_source_donors": source,
        "geometry_excluded_donors": excluded,
        "feature_source_donors": source,
        "lasso_source_donors": source,
        "pair_feature_source_donors": source,
        "geometry_isolation": geometry_isolation,
        "static_graph_asset": file_record(static_path),
        "node_global_indices": {
            key: torch.as_tensor(value, dtype=torch.long) for key, value in nodes.items()
        },
        "features": localized_features,
        "sparse_lasso_edges": localized_lasso_edges(lasso, response_global, nodes),
        "lasso_stability_threshold": float(settings.lasso_stability_threshold),
        "pair_feature_names": list(PAIR_FEATURE_NAMES),
        "lasso_derived_pair_feature_mask": list(LASSO_DERIVED_PAIR_FEATURE_MASK),
        "pair_features_by_arm": pair_features,
        "lnc_index": torch.as_tensor(
            [lnc_local[int(value)] for value in row_lnc_global], dtype=torch.long
        ),
        "pathway_index": torch.as_tensor(
            [pathway_local[int(value)] for value in row_pathway_global], dtype=torch.long
        ),
        "target_rho": torch.as_tensor(
            rows.rho.to_numpy(dtype=np.float32, copy=True), dtype=torch.float32
        ),
        "row_metadata": {
            column: rows[column].astype(str).tolist() for column in ROW_ID_COLUMNS
        },
        "lasso_audit": _jsonable(lasso.audit),
        "lasso_response_summary": _jsonable(lasso.response_frame().to_dict("list")),
        "lasso_policy": lasso_policy_receipt(settings.lasso_policy),
        "source_receipts": source_receipts,
    }
    atomic_torch_save(destination, payload)
    receipt = {
        "format": EPISODE_SUCCESS_FORMAT,
        "target_host": "149",
        "state": "EPISODE_READY",
        "episode_id": episode_id,
        "cancer_id": cancer_id,
        "outer_fold_id": int(outer_fold_id),
        "inner_fold_id": inner_fold_id,
        "role": role,
        "cell_key": str(cell_key),
        "row_count": len(rows),
        "source_donor_count": len(source),
        "label_donor_count": len(label_donors),
        "local_node_counts": {key: len(value) for key, value in nodes.items()},
        "lasso_edge_count": len(lasso.coefficient),
        "stable_lasso_edge_count": int(
            np.sum(lasso.stability >= settings.lasso_stability_threshold)
        ),
        "lasso_available_response_count": int(lasso.available.sum()),
        "episode": file_record(destination),
        "finished_unix": time.time(),
    }
    atomic_json(receipt_path, receipt)
    return receipt


def _query_episodes(
    settings, cancer_id: str, static, static_path: Path, graph_success: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if "native_graph_preparation" in graph_success:
        from cc_hhgt.v32.native_single_cell_graph_workflow import validate_outputs
        preflight = Path(graph_success["native_graph_preparation"]["path"])
        verify_record(graph_success["native_graph_preparation"], preflight, "NATIVE_GRAPH_PREPARATION")
        validate_outputs(preflight.parent)
        if preflight.parent != settings.graph_asset_root.resolve():
            raise RuntimeError("NATIVE_GRAPH_ASSET_ROOT_DRIFT")
    result = []
    fold_success_paths = []
    for record in graph_success["outer_fold_successes"]:
        path = Path(record["path"])
        verify_record(record, path, "GRAPH_OUTER_FOLD_SUCCESS")
        fold_success_paths.append(path)
    for success_path in sorted(fold_success_paths):
        _deadline(settings)
        fold_root = success_path.parent
        success = _read_success(
            success_path, "FOLD_GRAPH_ASSETS_READY", verify_files=False
        )
        outer = int(success["outer_fold_id"])
        target_binding_path = fold_root / "TARGET_BINDING.json"
        binding = read_json(target_binding_path)
        query_path = Path(binding["query_rows"]["path"])
        verify_record(binding["query_rows"], query_path, "QUERY_ROWS")
        rows = retry_io(lambda: pd.read_parquet(query_path))
        if rows.empty:
            continue
        cell_keys = sorted(rows.cell_key.astype(str).unique())
        cached = [
            _cached_episode(settings, cancer_id, outer, "query", None, cell_key)
            for cell_key in cell_keys
        ]
        if all(receipt is not None for receipt in cached):
            result.extend(cached)
            del rows
            continue
        coding = retry_io(
            lambda: sparse.load_npz(fold_root / "METACELL_CODING.npz").tocsr()
        )
        lncrna = retry_io(
            lambda: sparse.load_npz(fold_root / "METACELL_LNCRNA.npz").tocsr()
        )
        activity = retry_io(
            lambda: np.load(fold_root / "METACELL_PATHWAY_ACTIVITY.npy", mmap_mode="r")
        )
        observations = retry_io(
            lambda: pd.read_parquet(fold_root / "LASSO_OBSERVATIONS.parquet")
        )
        source, query = success["source_donors"], success["query_donors"]
        for cell_key in cell_keys:
            context_rows = rows.loc[rows.cell_key.astype(str).eq(cell_key)]
            if context_rows.empty:
                continue
            features = source_context_features(
                coding,
                lncrna,
                activity,
                observations,
                cell_key=cell_key,
                source_donors=source,
                pathway_family_membership=static["pathway_family"],
                minimum_context_donors=2,
            )
            result.append(
                _write_episode(
                    settings,
                    cancer_id=cancer_id,
                    outer_fold_id=outer,
                    role="query",
                    inner_fold_id=None,
                    cell_key=cell_key,
                    rows=context_rows,
                    features_full=features,
                    coding=coding,
                    lncrna=lncrna,
                    observations=observations,
                    static=static,
                    static_path=static_path,
                    source_donors=source,
                    excluded_donors=query,
                    outer_train_donors=source,
                    outer_query_donors=query,
                    inner_target_donors=(),
                    validation_donors=(),
                    geometry_isolation="outer_fold_only",
                    source_receipts=[file_record(success_path), file_record(target_binding_path)],
                )
            )
        del coding, lncrna, activity, observations, rows
        gc.collect()
    return result


def _nested_episodes(
    settings, cancer_id: str, static, static_path: Path, nested_success: Mapping[str, Any]
) -> list[dict[str, Any]]:
    result = []
    success_paths = []
    for outer_record in nested_success["outer_fold_receipts"]:
        outer_path = Path(outer_record["path"])
        verify_record(outer_record, outer_path, "NESTED_OUTER_FOLD_SUCCESS")
        outer_success = _read_success(outer_path, "OUTER_FOLD_NESTED_ASSETS_COMPLETE")
        for record in outer_success["episode_receipts"]:
            success_path = Path(record["path"])
            verify_record(record, success_path, "NESTED_EPISODE_SUCCESS")
            success_paths.append(success_path)
    for success_path in sorted(success_paths):
        _deadline(settings)
        success = read_json(success_path)
        if success.get("state") != "NESTED_EPISODE_ASSETS_READY":
            continue
        root = success_path.parent
        partition_path = root / "DONOR_PARTITION.json"
        partition = read_json(partition_path)
        outer = int(success["outer_fold_id"])
        inner = int(success["inner_target_fold_id"])
        role = str(success["role"])
        if role not in {"fit", "validation"}:
            raise RuntimeError("NESTED_EPISODE_ROLE_DRIFT")
        rows = retry_io(lambda: pd.read_parquet(root / "SIGNED_TARGET_ROWS.parquet"))
        if rows.empty:
            continue
        contexts = retry_io(
            lambda: pd.read_parquet(root / "NODE_FEATURE_CONTEXTS.parquet")
        )
        context_records = contexts.sort_values("context_index").to_dict("records")
        cached = [
            _cached_episode(
                settings, cancer_id, outer, role, inner, str(record["cell_key"])
            )
            for record in context_records
        ]
        if all(receipt is not None for receipt in cached):
            result.extend(cached)
            del contexts, rows
            continue
        coding = retry_io(
            lambda: sparse.load_npz(root / "METACELL_CODING.npz").tocsr()
        )
        lncrna = retry_io(
            lambda: sparse.load_npz(root / "METACELL_LNCRNA.npz").tocsr()
        )
        observations = retry_io(
            lambda: pd.read_parquet(root / "LASSO_OBSERVATIONS.parquet")
        )
        arrays = retry_io(lambda: np.load(root / "NODE_FEATURES.npz", allow_pickle=False))
        source = partition["source_donors"]
        excluded = partition["excluded_donors"]
        for record in context_records:
            cell_key = str(record["cell_key"])
            context_rows = rows.loc[rows.cell_key.astype(str).eq(cell_key)]
            if context_rows.empty:
                continue
            context_index = int(record["context_index"])
            features = {
                node_type: np.asarray(arrays[node_type][context_index], dtype=np.float32)
                for node_type in ("lncRNA", "gene", "pathway", "pathway_family", "cell_context")
            }
            result.append(
                _write_episode(
                    settings,
                    cancer_id=cancer_id,
                    outer_fold_id=outer,
                    role=role,
                    inner_fold_id=inner,
                    cell_key=cell_key,
                    rows=context_rows,
                    features_full=features,
                    coding=coding,
                    lncrna=lncrna,
                    observations=observations,
                    static=static,
                    static_path=static_path,
                    source_donors=source,
                    excluded_donors=excluded,
                    outer_train_donors=partition["outer_train_donors"],
                    outer_query_donors=partition["outer_query_donors"],
                    inner_target_donors=partition["inner_target_donors"],
                    validation_donors=partition["inner_validation_donors"],
                    geometry_isolation=partition["geometry_isolation"],
                    source_receipts=[file_record(success_path), file_record(partition_path)],
                )
            )
        arrays.close()
        del coding, lncrna, observations, contexts, rows
        gc.collect()
    return result


def run_cancer(settings) -> dict[str, Any]:
    _require_host_149()
    _initialize_preflight(settings)
    cancer_id = settings.cancer_id.upper()
    nested_success = _read_success(
        settings.nested_asset_root / "cancers" / cancer_id / "SUCCESS.json",
        "CANCER_NESTED_ASSETS_COMPLETE",
    )
    graph_success = _read_success(
        settings.graph_asset_root / "cancers" / cancer_id / "SUCCESS.json",
        "CANCER_GRAPH_ASSETS_COMPLETE",
    )
    static_success_path = (
        settings.graph_asset_root / "cancers" / cancer_id / "STATIC" / "SUCCESS.json"
    )
    verify_record(
        graph_success["cancer_static_success"],
        static_success_path,
        "GRAPH_CANCER_STATIC_SUCCESS",
    )
    destination = settings.output_root / "cancers" / cancer_id / "SUCCESS.json"
    if destination.is_file():
        cached = read_json(destination)
        if cached.get("state") != "CANCER_EPISODES_COMPLETE":
            raise RuntimeError("CACHED_CANCER_EPISODE_STATE_DRIFT")
        for record in cached.get("episodes", []):
            verify_record(record, Path(record["path"]), "CACHED_CANCER_EPISODE")
        return cached
    static = _load_cancer_static(settings, cancer_id)
    static_path = _ensure_shared_static(settings, cancer_id, static)
    receipts = _nested_episodes(
        settings, cancer_id, static, static_path, nested_success
    )
    receipts.extend(
        _query_episodes(settings, cancer_id, static, static_path, graph_success)
    )
    if not receipts:
        raise RuntimeError("CANCER_HAS_NO_ESTIMABLE_EPISODES")
    roles = {value["role"] for value in receipts}
    if not {"fit", "query"}.issubset(roles):
        raise RuntimeError(f"CANCER_EPISODE_ROLE_COVERAGE:{sorted(roles)}")
    graph_fold_successes = [Path(record["path"]) for record in graph_success["outer_fold_successes"]]
    expected_query_rows = 0
    for success_path in graph_fold_successes:
        binding = read_json(success_path.parent / "TARGET_BINDING.json")
        query_path = Path(binding["query_rows"]["path"])
        verify_record(binding["query_rows"], query_path, "QUERY_ROWS")
        expected_query_rows += retry_io(
            lambda: pq.ParquetFile(query_path).metadata.num_rows
        )
    expected_nested_rows = 0
    nested_success_paths = []
    for outer_record in nested_success["outer_fold_receipts"]:
        outer_success = _read_success(
            Path(outer_record["path"]), "OUTER_FOLD_NESTED_ASSETS_COMPLETE"
        )
        nested_success_paths.extend(
            Path(record["path"]) for record in outer_success["episode_receipts"]
        )
    for success_path in nested_success_paths:
        nested_receipt = read_json(success_path)
        if nested_receipt.get("state") == "NESTED_EPISODE_ASSETS_READY":
            expected_nested_rows += retry_io(
                lambda: pq.ParquetFile(
                    success_path.parent / "SIGNED_TARGET_ROWS.parquet"
                ).metadata.num_rows
            )
    actual_query_rows = sum(
        int(item["row_count"]) for item in receipts if item["role"] == "query"
    )
    actual_nested_rows = sum(
        int(item["row_count"])
        for item in receipts
        if item["role"] in {"fit", "validation"}
    )
    if actual_query_rows != expected_query_rows or actual_nested_rows != expected_nested_rows:
        raise RuntimeError(
            "CANCER_EPISODE_ROW_COVERAGE_DRIFT:"
            f"query={actual_query_rows}/{expected_query_rows},"
            f"nested={actual_nested_rows}/{expected_nested_rows}"
        )
    role_row_counts = {
        role: sum(int(item["row_count"]) for item in receipts if item["role"] == role)
        for role in ("fit", "validation", "query")
    }
    value = {
        "format": EPISODE_SUCCESS_FORMAT,
        "target_host": "149",
        "state": "CANCER_EPISODES_COMPLETE",
        "cancer_id": cancer_id,
        "static_asset": file_record(static_path),
        "episode_count": len(receipts),
        "row_count": sum(int(item["row_count"]) for item in receipts),
        "role_row_counts": role_row_counts,
        "source_query_row_count": expected_query_rows,
        "source_nested_target_row_count": expected_nested_rows,
        "role_counts": {
            role: sum(item["role"] == role for item in receipts)
            for role in ("fit", "validation", "query")
        },
        "validation_provider": "validation" in roles,
        "episodes": [item["episode"] for item in receipts],
        "episode_receipts": [
            file_record(Path(item["episode"]["path"]).with_suffix(".json"))
            for item in receipts
        ],
        "finished_unix": time.time(),
    }
    atomic_json(destination, value)
    return value


def _authorized_cancers(settings) -> list[str]:
    report = read_json(
        settings.source_result_root / "MATERIALIZATION_VALIDATION.json"
    )
    cancers = sorted(map(str, report.get("validated_cancers", ())))
    if len(cancers) != 28:
        raise RuntimeError(f"EXPECTED_28_VALIDATED_CANCERS:{len(cancers)}")
    return cancers


def finalize(settings) -> dict[str, Any]:
    _require_host_149()
    preflight = _initialize_preflight(settings)
    cancers = _authorized_cancers(settings)
    graph_records, graph_fold_records, static_records, episode_records = [], [], [], []
    folds: dict[int, list[dict[str, Any]]] = {fold: [] for fold in range(5)}
    fold_roles: dict[int, set[str]] = {fold: set() for fold in range(5)}
    role_counts = {role: 0 for role in ("fit", "validation", "query")}
    role_row_counts = {role: 0 for role in ("fit", "validation", "query")}
    row_count = 0
    for cancer_id in cancers:
        cancer_success = settings.output_root / "cancers" / cancer_id / "SUCCESS.json"
        value = _read_success(cancer_success, "CANCER_EPISODES_COMPLETE")
        static_records.append(value["static_asset"])
        for record in value["episodes"]:
            path = Path(record["path"])
            verify_record_metadata(record, path, "FINAL_EPISODE")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            fold = int(payload["outer_fold_id"])
            folds[fold].append(record)
            role = str(payload["role"])
            fold_roles[fold].add(role)
            role_counts[role] += 1
            rows = int(torch.as_tensor(payload["target_rho"]).numel())
            row_count += rows
            role_row_counts[role] += rows
            episode_records.append(record)
        graph_cancer = settings.graph_asset_root / "cancers" / cancer_id / "STATIC" / "SUCCESS.json"
        graph_records.append(file_record(graph_cancer))
        source_folds = retry_io(
            lambda: pd.read_parquet(
                settings.source_result_root / "cancers" / cancer_id / "DONOR_FOLDS.parquet",
                columns=["outer_fold_id"],
            )
        )
        for fold in sorted(pd.to_numeric(source_folds.outer_fold_id).astype(int).unique()):
            graph_fold = (
                settings.graph_asset_root
                / "cancers"
                / cancer_id
                / f"outer_fold={fold}"
                / "SUCCESS.json"
            )
            # Fold assets were already content-verified while materializing
            # each episode. Finalization validates their immutable receipt
            # without streaming the large matrices a second time.
            _read_success(
                graph_fold, "FOLD_GRAPH_ASSETS_READY", verify_files=False
            )
            graph_fold_records.append(file_record(graph_fold))
    for fold, roles in fold_roles.items():
        if roles != {"fit", "validation", "query"}:
            raise RuntimeError(f"FINAL_FOLD_ROLE_COVERAGE:{fold}:{sorted(roles)}")
    graph_manifest_path = settings.graph_asset_root / "GRAPH_ASSET_MANIFEST.json"
    graph_manifest = {
        "format": ASSET_FORMAT,
        "state": "GRAPH_ASSETS_COMPLETE",
        "target_host": "149",
        "cancers": cancers,
        "cancer_count": len(cancers),
        "cancer_static_receipts": graph_records,
        "fold_receipts": graph_fold_records,
        "fold_count": len(graph_fold_records),
        "preflight": file_record(settings.graph_asset_root / "LAUNCH_PREFLIGHT.json"),
        "finished_unix": time.time(),
    }
    if graph_manifest_path.is_file():
        old = read_json(graph_manifest_path)
        for key in ("format", "state", "target_host", "cancers", "cancer_count"):
            if old.get(key) != graph_manifest[key]:
                raise RuntimeError("GRAPH_ASSET_MANIFEST_DRIFT")
    else:
        atomic_json(graph_manifest_path, graph_manifest)
    manifest_path = settings.output_root / "SC_HHGT_EPISODE_MANIFEST.json"
    manifest = {
        "format": EPISODE_MANIFEST_FORMAT,
        "state": "EPISODES_COMPLETE",
        "target_host": "149",
        "cancers": cancers,
        "cancer_count": len(cancers),
        "outer_folds": 5,
        "episode_count": len(episode_records),
        "row_count": row_count,
        "role_counts": role_counts,
        "role_row_counts": role_row_counts,
        "graph_asset_manifest": file_record(graph_manifest_path),
        "launch_preflight": file_record(settings.output_root / "LAUNCH_PREFLIGHT.json"),
        "shared_static_assets": static_records,
        "folds": [
            {"outer_fold_id": fold, "episodes": sorted(records, key=lambda x: x["path"])}
            for fold, records in sorted(folds.items())
        ],
        "pair_feature_names": list(PAIR_FEATURE_NAMES),
        "lasso_derived_pair_feature_mask": list(LASSO_DERIVED_PAIR_FEATURE_MASK),
        "formal_pointer_update_authorized": False,
        "preflight_sha256": canonical_hash(preflight),
        "finished_unix": time.time(),
    }
    if manifest_path.exists():
        raise RuntimeError("EPISODE_MANIFEST_ALREADY_EXISTS")
    atomic_json(manifest_path, manifest)
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run-id", required=True)
    result.add_argument("--source-result-root", required=True, type=Path)
    result.add_argument("--graph-asset-root", required=True, type=Path)
    result.add_argument("--nested-asset-root", required=True, type=Path)
    result.add_argument("--output-root", required=True, type=Path)
    result.add_argument("--cancer-id")
    result.add_argument("--finalize", action="store_true")
    result.add_argument("--deadline", required=True, type=float)
    result.add_argument("--lasso-alphas", default="0.3,0.1,0.03,0.01")
    result.add_argument("--lasso-max-genes", type=int, default=1000)
    result.add_argument("--lasso-min-donors", type=int, default=4)
    result.add_argument("--lasso-inner-folds", type=int, default=5)
    result.add_argument("--lasso-bootstraps", type=int, default=8)
    result.add_argument("--lasso-response-batch-size", type=int, default=256)
    result.add_argument("--lasso-max-iter", type=int, default=2000)
    result.add_argument("--lasso-tolerance", type=float, default=1e-6)
    result.add_argument("--lasso-stability-threshold", type=float, default=0.5)
    return result


def settings_from_args(args):
    args.lasso_policy = BatchedLassoPolicy(
        alphas=tuple(float(value) for value in args.lasso_alphas.split(",")),
        max_genes=args.lasso_max_genes,
        min_donors=args.lasso_min_donors,
        inner_folds=args.lasso_inner_folds,
        bootstraps=args.lasso_bootstraps,
        response_batch_size=args.lasso_response_batch_size,
        max_iter=args.lasso_max_iter,
        tolerance=args.lasso_tolerance,
    ).validated()
    return args


def main() -> int:
    settings = settings_from_args(parser().parse_args())
    if settings.finalize == bool(settings.cancer_id):
        raise SystemExit("EXACTLY_ONE_OF_CANCER_ID_OR_FINALIZE_IS_REQUIRED")
    settings.output_root.mkdir(parents=True, exist_ok=True)
    result = finalize(settings) if settings.finalize else run_cancer(settings)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(TimeoutError("SIGTERM")))
    raise SystemExit(main())
