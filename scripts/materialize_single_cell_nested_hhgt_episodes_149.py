#!/usr/bin/env python3
"""Materialize strict nested donor-isolated SC-HHGT episode inputs on 149.

Raw mapped matrices are loaded once per cancer.  Every validation or fit asset
then refits LocalSpace PCA and the grouping radius from its exact source-donor
set.  Outer-query, inner-target, and reserved validation donors are never used
by geometry, discovery, node summaries, or LASSO-ready source observations.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.single_cell_metacell_screen import (  # noqa: E402
    SCREEN_POLICY,
    exact_donor_targets,
    exact_source_pair_effects,
    screen_source_pairs,
)
from cc_hhgt.v32.single_cell_nested_episodes import (  # noqa: E402
    NESTED_ASSET_FORMAT,
    NESTED_PREFLIGHT_FORMAT,
    NestedFoldPlan,
    fit_nested_metacells,
    lasso_observations,
    nested_fold_plans,
    source_node_features,
    validate_nested_isolation,
)


TARGET_POLICY = "SIGNED_WITHIN_DONOR_CELLTYPE_METACELL_SPEARMAN_RHO"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_record(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_script(name: str, filename: str):
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _verify_record(record: dict[str, object], expected: Path, label: str) -> None:
    if not isinstance(record, dict):
        raise RuntimeError(f"{label} file record missing")
    path = Path(str(record.get("path", ""))).resolve()
    expected = expected.resolve()
    if path != expected or not path.is_file():
        raise RuntimeError(f"{label} path drift")
    if int(record.get("bytes", -1)) != path.stat().st_size:
        raise RuntimeError(f"{label} byte count drift")
    if str(record.get("sha256", "")) != sha256(path):
        raise RuntimeError(f"{label} SHA256 drift")


def _verify_record_metadata(record: dict[str, object], expected: Path, label: str) -> None:
    """Check an already-hashed artifact without rereading its full contents."""

    if not isinstance(record, dict):
        raise RuntimeError(f"{label} file record missing")
    path = Path(str(record.get("path", ""))).resolve()
    expected = expected.resolve()
    if path != expected or not path.is_file():
        raise RuntimeError(f"{label} path drift")
    if int(record.get("bytes", -1)) != path.stat().st_size:
        raise RuntimeError(f"{label} byte count drift")
    digest = str(record.get("sha256", ""))
    if len(digest) != 64 or any(value not in "0123456789abcdef" for value in digest):
        raise RuntimeError(f"{label} SHA256 receipt invalid")


def validate_preflight(settings) -> dict[str, object]:
    path = settings.output_root / "LAUNCH_PREFLIGHT.json"
    if not path.is_file():
        raise RuntimeError("NESTED_LAUNCH_PREFLIGHT_MISSING")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if "native_donor_preparation" in receipt:
        from cc_hhgt.v32.native_single_cell_donors_workflow import consumer
        consumer(settings, receipt)
        return receipt
    expected = {
        "format": NESTED_PREFLIGHT_FORMAT,
        "target_host": "149",
        "cpu_only": True,
        "paid_compute": False,
        "formal_pointer_update_authorized": False,
        "source_metacell_result_root": str(settings.source_result_root.resolve()),
        "graph_asset_root": str(settings.graph_asset_root.resolve()),
        "candidate_path": str(settings.candidates.resolve()),
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise RuntimeError(f"NESTED_LAUNCH_PREFLIGHT_{key.upper()}_DRIFT")
    _verify_record(receipt.get("materializer"), Path(__file__), "nested materializer")
    _verify_record(
        receipt.get("nested_module"),
        ROOT / "cc_hhgt" / "v32" / "single_cell_nested_episodes.py",
        "nested module",
    )
    source_validation = settings.source_result_root / "MATERIALIZATION_VALIDATION.json"
    graph_preflight = settings.graph_asset_root / "LAUNCH_PREFLIGHT.json"
    for key, expected_path in (
        ("source_materialization_validation", source_validation),
        ("graph_asset_preflight", graph_preflight),
        ("candidate", settings.candidates),
    ):
        _verify_record(receipt.get("bindings", {}).get(key), expected_path, key)
    source_report = json.loads(source_validation.read_text(encoding="utf-8"))
    if (
        source_report.get("target_host") != "149"
        or source_report.get("status") != "PASS"
        or source_report.get("closure_status")
        != "PASS_FULL_28_CANCER_MATERIALIZATION_CLOSURE"
    ):
        raise RuntimeError("SOURCE_MATERIALIZATION_NOT_FORMALLY_VALIDATED")
    return receipt


def _validate_source_cancer(settings, cancer_id: str) -> tuple[pd.DataFrame, dict]:
    preflight = json.loads((settings.output_root / "LAUNCH_PREFLIGHT.json").read_text(encoding="utf-8"))
    if "native_donor_preparation" in preflight:
        from cc_hhgt.v32.native_single_cell_donors_workflow import consumer
        _, folds, receipt = consumer(settings, preflight)
        if set(folds.cancer_id) != {cancer_id}:
            raise RuntimeError("NATIVE_DONOR_CANCER_ID_DRIFT")
        return folds, receipt
    root = settings.source_result_root / "cancers" / cancer_id
    success = root / "SUCCESS.json"
    folds_path = root / "DONOR_FOLDS.parquet"
    if not success.is_file() or not folds_path.is_file():
        raise RuntimeError("SOURCE_CANCER_MATERIALIZATION_INCOMPLETE")
    receipt = json.loads(success.read_text(encoding="utf-8"))
    if (
        receipt.get("target_host") != "149"
        or receipt.get("state") != "CANCER_METACELL_OOF_INPUTS_COMPLETE"
        or receipt.get("cancer_id") != cancer_id
    ):
        raise RuntimeError("SOURCE_CANCER_RECEIPT_INVALID")
    _verify_record(
        receipt.get("files", {}).get("DONOR_FOLDS.parquet"),
        folds_path,
        "source donor folds",
    )
    folds = pd.read_parquet(folds_path)
    return folds, receipt


def _validate_static(
    settings,
    cancer_id: str,
    ids: dict[str, list[str]],
) -> tuple[Path, dict, sparse.csr_matrix, list[str]]:
    root = settings.graph_asset_root / "cancers" / cancer_id / "STATIC"
    success = root / "SUCCESS.json"
    if not success.is_file():
        raise RuntimeError("CANCER_STATIC_GRAPH_ASSETS_MISSING")
    receipt = json.loads(success.read_text(encoding="utf-8"))
    if "native_static_preparation" in receipt:
        from cc_hhgt.v32.native_single_cell_static_workflow import validate_static
        validate_static(success)
    if (
        receipt.get("target_host") != "149"
        or receipt.get("state") != "CANCER_STATIC_GRAPH_ASSETS_READY"
        or receipt.get("cancer_id") != cancer_id
    ):
        raise RuntimeError("CANCER_STATIC_GRAPH_ASSET_RECEIPT_INVALID")
    required = (
        "NODE_IDENTIFIERS.parquet",
        "GENE_PATHWAY_MEMBERSHIP.npz",
        "PATHWAY_FAMILY_MEMBERSHIP.npz",
        "EXTERNAL_PHYSICAL_BINDING.parquet",
    )
    for name in required:
        _verify_record(receipt.get("files", {}).get(name), root / name, f"static {name}")
    nodes = pd.read_parquet(root / "NODE_IDENTIFIERS.parquet")
    expected = {
        "gene": list(map(str, ids["protein_ids"])),
        "lncRNA": list(map(str, ids["lncrna_ids"])),
        "pathway": list(map(str, ids["pathway_ids"])),
    }
    for node_type, values in expected.items():
        observed = (
            nodes.loc[nodes.node_type.eq(node_type)]
            .sort_values("global_index", kind="mergesort")
            .canonical_id.astype(str)
            .tolist()
        )
        if observed != values:
            raise RuntimeError(f"STATIC_{node_type.upper()}_ID_ORDER_DRIFT")
    families = (
        nodes.loc[nodes.node_type.eq("pathway_family")]
        .sort_values("global_index", kind="mergesort")
        .canonical_id.astype(str)
        .tolist()
    )
    family = sparse.load_npz(root / "PATHWAY_FAMILY_MEMBERSHIP.npz").tocsr()
    if family.shape != (len(expected["pathway"]), len(families)):
        raise RuntimeError("STATIC_PATHWAY_FAMILY_SHAPE_DRIFT")
    return root, receipt, family, families


def _ensure_matrix_identity(
    cancer_root: Path,
    ids: dict[str, list[str]],
    families: list[str],
    mapped_success: Path,
    static_success: Path,
) -> Path:
    destination = cancer_root / "MATRIX_IDENTITY.parquet"
    receipt_path = cancer_root / "MATRIX_IDENTITY.json"
    frames = []
    for node_type, values in (
        ("gene", ids["protein_ids"]),
        ("lncRNA", ids["lncrna_ids"]),
        ("pathway", ids["pathway_ids"]),
        ("pathway_family", families),
    ):
        frames.append(
            pd.DataFrame(
                {
                    "node_type": node_type,
                    "matrix_column": np.arange(len(values), dtype=np.int64),
                    "canonical_id": list(map(str, values)),
                }
            )
        )
    expected = pd.concat(frames, ignore_index=True)
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        _verify_record(receipt.get("matrix_identity"), destination, "matrix identity")
        if receipt.get("id_order_sha256") != canonical_hash(
            {
                "gene": ids["protein_ids"],
                "lncRNA": ids["lncrna_ids"],
                "pathway": ids["pathway_ids"],
                "pathway_family": families,
            }
        ):
            raise RuntimeError("MATRIX_IDENTITY_ORDER_DRIFT")
        return receipt_path
    cancer_root.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".parquet.partial")
    expected.to_parquet(temporary, index=False)
    os.replace(temporary, destination)
    atomic_json(
        receipt_path,
        {
            "format": NESTED_ASSET_FORMAT,
            "target_host": "149",
            "state": "CANCER_MATRIX_IDENTITY_READY",
            "matrix_identity": file_record(destination),
            "mapped_success": file_record(mapped_success),
            "static_success": file_record(static_success),
            "id_order_sha256": canonical_hash(
                {
                    "gene": ids["protein_ids"],
                    "lncRNA": ids["lncrna_ids"],
                    "pathway": ids["pathway_ids"],
                    "pathway_family": families,
                }
            ),
        },
    )
    return receipt_path


def _plan_rows(plans: list[NestedFoldPlan]) -> pd.DataFrame:
    rows = []
    for plan in plans:
        roles = {
            **{donor: "source" for donor in plan.source_donors},
            **{donor: "outer_query" for donor in plan.outer_query_donors},
            **{donor: "inner_target" for donor in plan.inner_target_donors},
        }
        if plan.role == "fit":
            roles.update(
                {donor: "inner_validation" for donor in plan.inner_validation_donors}
            )
        for donor, donor_role in sorted(roles.items()):
            rows.append(
                {
                    "cancer_id": plan.cancer_id,
                    "outer_fold_id": plan.outer_fold_id,
                    "inner_target_fold_id": plan.inner_target_fold_id,
                    "inner_validation_fold_id": plan.inner_validation_fold_id,
                    "role": plan.role,
                    "donor_id": donor,
                    "donor_role": donor_role,
                    "geometry_isolation": plan.geometry_isolation,
                    "validation_provider": plan.validation_provider,
                    "support_limited_fit": plan.support_limited_fit,
                    "minimum_source_donors": plan.minimum_source_donors,
                    "available": plan.available,
                    "unavailable_reason": (
                        None
                        if plan.available
                        else f"INSUFFICIENT_SOURCE_DONORS_LT{plan.minimum_source_donors}"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _prepare_staging(destination: Path, plan: NestedFoldPlan) -> Path:
    if destination.is_dir():
        success = destination / "SUCCESS.json"
        if not success.is_file():
            raise RuntimeError("NESTED_ASSET_DESTINATION_INCOMPLETE")
        return destination
    staging = destination.with_name(destination.name + ".partial")
    if staging.exists():
        running = staging / "RUNNING.json"
        live = False
        if running.is_file():
            try:
                record = json.loads(running.read_text(encoding="utf-8"))
                live = (
                    record.get("target_host") == "149"
                    and Path(f"/proc/{int(record.get('pid', -1))}").exists()
                )
            except (ValueError, TypeError, json.JSONDecodeError):
                live = False
        if live:
            raise RuntimeError("NESTED_ASSET_CONCURRENT_WRITER")
        if staging.parent.resolve() not in staging.resolve().parents:
            raise RuntimeError("NESTED_STAGING_CONTAINMENT_FAILURE")
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    atomic_json(
        staging / "RUNNING.json",
        {
            "format": NESTED_ASSET_FORMAT,
            "target_host": "149",
            "pid": os.getpid(),
            "started_unix": time.time(),
            **plan.as_dict(),
        },
    )
    return staging


def _verify_complete(destination: Path, plan: NestedFoldPlan) -> dict | None:
    success = destination / "SUCCESS.json"
    if not success.is_file():
        return None
    receipt = json.loads(success.read_text(encoding="utf-8"))
    if (
        receipt.get("format") != NESTED_ASSET_FORMAT
        or receipt.get("target_host") != "149"
        or receipt.get("geometry_isolation") != plan.geometry_isolation
        or receipt.get("donor_partition_sha256") != canonical_hash(plan.as_dict())
    ):
        raise RuntimeError("NESTED_ASSET_RECEIPT_DRIFT")
    for relative, record in receipt.get("files", {}).items():
        _verify_record(record, destination / relative, f"nested asset {relative}")
    return receipt


def _verify_complete_metadata(
    destination: Path, plan: NestedFoldPlan
) -> dict | None:
    """Authorize scratch deletion from the atomic receipt and file metadata.

    Payload SHA256 values were computed while finalizing the episode.  Training
    episode consumption and the independent closure audit verify those digests
    again, so rereading every large matrix here would be redundant.
    """

    success = destination / "SUCCESS.json"
    if not success.is_file():
        return None
    receipt = json.loads(success.read_text(encoding="utf-8"))
    if (
        receipt.get("format") != NESTED_ASSET_FORMAT
        or receipt.get("target_host") != "149"
        or receipt.get("geometry_isolation") != plan.geometry_isolation
        or receipt.get("donor_partition_sha256") != canonical_hash(plan.as_dict())
    ):
        raise RuntimeError("NESTED_ASSET_RECEIPT_DRIFT")
    for relative, record in receipt.get("files", {}).items():
        _verify_record_metadata(
            record, destination / relative, f"nested asset {relative}"
        )
    return receipt


def _finish(staging: Path, destination: Path, receipt: dict) -> dict:
    running = staging / "RUNNING.json"
    if running.exists():
        running.unlink()
    files = {
        str(path.relative_to(staging)).replace("\\", "/"): file_record(path)
        for path in sorted(staging.rglob("*"))
        if path.is_file() and path.name != "SUCCESS.json"
    }
    receipt = {**receipt, "files": files}
    atomic_json(staging / "SUCCESS.json", receipt)
    os.replace(staging, destination)
    receipt["files"] = {
        str(path.relative_to(destination)).replace("\\", "/"): file_record(path)
        for path in sorted(destination.rglob("*"))
        if path.is_file() and path.name != "SUCCESS.json"
    }
    atomic_json(destination / "SUCCESS.json", receipt)
    return receipt


def _candidate_universe(
    bulk: pd.DataFrame,
    discovery: pd.DataFrame,
    plan: NestedFoldPlan,
    *,
    dataset_id: str,
    cell_key: str,
    measurement_scale: str,
) -> pd.DataFrame:
    discovered = discovery[["lncrna_id", "pathway_id"]].drop_duplicates()
    result = bulk.assign(is_bulk_candidate=True).merge(
        discovered.assign(is_sc_discovery_candidate=True),
        on=["lncrna_id", "pathway_id"],
        how="outer",
        validate="one_to_one",
    )
    result["is_bulk_candidate"] = result.is_bulk_candidate.eq(True)
    result["is_sc_discovery_candidate"] = result.is_sc_discovery_candidate.eq(True)
    result["universe_role"] = np.select(
        [
            result.is_bulk_candidate & result.is_sc_discovery_candidate,
            result.is_bulk_candidate,
        ],
        ["bulk_and_sc_discovery", "bulk_rescore"],
        default="sc_discovery",
    )
    result["selection_policy"] = np.where(
        result.is_sc_discovery_candidate, SCREEN_POLICY, "FROZEN_BULK_EXACT_PAIR"
    )
    result["candidate_selection_query_donor_used"] = False
    result["candidate_selection_target_donor_used"] = False
    result["candidate_selection_validation_donor_used"] = False
    result["held_out_donor_rows_used"] = 0
    result["dataset_id"] = dataset_id
    result["cancer_id"] = plan.cancer_id
    result["cell_key"] = cell_key
    result["outer_fold_id"] = plan.outer_fold_id
    result["inner_target_fold_id"] = plan.inner_target_fold_id
    result["inner_validation_fold_id"] = plan.inner_validation_fold_id
    result["episode_role"] = plan.role
    result["geometry_isolation"] = plan.geometry_isolation
    result["measurement_scale"] = measurement_scale
    return result


def _context_assets(
    metacell: dict[str, object],
    activity: np.ndarray,
    ids: dict[str, list[str]],
    bulk: pd.DataFrame,
    plan: NestedFoldPlan,
    settings,
    *,
    dataset_id: str,
    measurement_scale: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    metadata = metacell["metadata"]
    lnc = metacell["lncrna"]
    universes: list[pd.DataFrame] = []
    targets: list[pd.DataFrame] = []
    coverage: list[dict[str, object]] = []
    source_set = set(plan.source_donors)
    target_set = set(plan.inner_target_donors)
    for number, cell_key in enumerate(sorted(metadata.cell_type.astype(str).unique())):
        if time.time() >= settings.deadline:
            raise TimeoutError("nested context deadline reached")
        mask = metadata.cell_type.astype(str).eq(cell_key).to_numpy()
        context_meta = metadata.loc[mask].reset_index(drop=True)
        present_source = source_set & set(context_meta.donor_id.astype(str))
        present_target = target_set & set(context_meta.donor_id.astype(str))
        report = {
            "cell_key": cell_key,
            "source_donors_present": len(present_source),
            "target_donors_present": len(present_target),
            "source_metacells": int(context_meta.donor_id.astype(str).isin(source_set).sum()),
            "target_metacells": int(context_meta.donor_id.astype(str).isin(target_set).sum()),
        }
        if not present_target:
            report.update(state="UNAVAILABLE", reason="NO_INNER_TARGET_DONOR_CONTEXT")
            coverage.append(report)
            continue
        if len(present_source) < plan.minimum_source_donors:
            report.update(
                state="UNAVAILABLE",
                reason=f"CONTEXT_SOURCE_DONORS_LT{plan.minimum_source_donors}",
            )
            coverage.append(report)
            continue
        discovery = screen_source_pairs(
            lnc[mask],
            activity[mask],
            context_meta,
            ids["lncrna_ids"],
            ids["pathway_ids"],
            present_source,
            top_k=settings.discovery_top_k,
            sketch_oversample=settings.sketch_oversample,
            sketch_dimensions=settings.sketch_dimensions,
            min_source_donors=plan.minimum_source_donors,
            min_metacells=settings.min_target_metacells,
            min_abs_meta_rho=settings.min_screen_abs_rho,
            seed=settings.seed
            + plan.outer_fold_id * 10000
            + plan.inner_target_fold_id * 1000
            + number,
        )
        universe = _candidate_universe(
            bulk,
            discovery,
            plan,
            dataset_id=dataset_id,
            cell_key=cell_key,
            measurement_scale=measurement_scale,
        )
        if universe.empty:
            report.update(state="UNAVAILABLE", reason="NO_EXACT_CANDIDATE_PAIRS")
            coverage.append(report)
            continue
        pair_columns = universe[["lncrna_id", "pathway_id"]]
        labels = exact_donor_targets(
            lnc[mask],
            activity[mask],
            context_meta,
            ids["lncrna_ids"],
            ids["pathway_ids"],
            pair_columns,
            present_target,
            min_metacells=settings.min_target_metacells,
        )
        source_effects = exact_source_pair_effects(
            lnc[mask],
            activity[mask],
            context_meta,
            ids["lncrna_ids"],
            ids["pathway_ids"],
            pair_columns,
            present_source,
            min_metacells=settings.min_target_metacells,
        )
        if not labels.empty:
            labels = labels.merge(
                universe,
                on=["lncrna_id", "pathway_id"],
                how="inner",
                validate="many_to_one",
            ).merge(
                source_effects,
                on=["lncrna_id", "pathway_id"],
                how="inner",
                validate="many_to_one",
            )
            finite = np.isfinite(
                labels[
                    [
                        "rho", "source_meta_rho", "source_direction_consistency",
                        "source_donor_count",
                    ]
                ].to_numpy(float)
            ).all(axis=1)
            labels = labels.loc[finite & labels.source_donor_count.ge(2)].copy()
            labels["row_role"] = plan.role
        universes.append(universe)
        if not labels.empty:
            targets.append(labels)
        report.update(
            state="READY" if not labels.empty else "UNAVAILABLE",
            reason=None if not labels.empty else "NO_ESTIMABLE_SIGNED_TARGET_ROWS",
            universe_rows=len(universe),
            target_rows=len(labels),
            discovery_rows=len(discovery),
        )
        coverage.append(report)
    universe_frame = (
        pd.concat(universes, ignore_index=True)
        if universes
        else pd.DataFrame(
            columns=[
                "lncrna_id", "pathway_id", "is_bulk_candidate",
                "is_sc_discovery_candidate", "universe_role", "selection_policy",
                "candidate_selection_query_donor_used",
                "candidate_selection_target_donor_used",
                "candidate_selection_validation_donor_used",
                "held_out_donor_rows_used", "dataset_id", "cancer_id", "cell_key",
                "outer_fold_id", "inner_target_fold_id",
                "inner_validation_fold_id", "episode_role", "geometry_isolation",
                "measurement_scale",
            ]
        )
    )
    target_frame = (
        pd.concat(targets, ignore_index=True)
        if targets
        else pd.DataFrame(
            columns=[
                "lncrna_id", "pathway_id", "query_donor_id", "rho",
                "association_target", "signed_target_unit_interval", "n_metacells",
                "lnc_detection_rate", "association_method",
                "association_observation_unit", "association_target_level",
                "donor_resolved_target", "source_mean_rho", "source_meta_rho",
                "source_direction_consistency", "source_donor_count",
                "source_meta_p_value", "dataset_id", "cancer_id", "cell_key",
                "outer_fold_id", "inner_target_fold_id",
                "inner_validation_fold_id", "episode_role", "geometry_isolation",
                "measurement_scale", "row_role",
            ]
        )
    )
    return universe_frame, target_frame, coverage


def materialize_plan(
    settings,
    plan: NestedFoldPlan,
    mapped: dict,
    matrices: dict[str, object],
    ids: dict[str, list[str]],
    bulk: pd.DataFrame,
    family_membership: sparse.csr_matrix,
    static_root: Path,
    matrix_identity_receipt: Path,
) -> dict:
    destination = (
        settings.output_root
        / "cancers"
        / plan.cancer_id
        / f"outer_fold={plan.outer_fold_id}"
        / f"role={plan.role}"
        / f"inner_target_fold={plan.inner_target_fold_id}"
    )
    if receipt := _verify_complete(destination, plan):
        preflight = json.loads((settings.output_root / "LAUNCH_PREFLIGHT.json").read_text(encoding="utf-8"))
        if "native_donor_preparation" in preflight:
            from cc_hhgt.v32.single_cell_nested_numeric_replay import verify_plan
            verify_plan(settings, plan, mapped, matrices, ids, bulk, family_membership,
                        static_root, matrix_identity_receipt, destination, receipt)
        return receipt
    staging = _prepare_staging(destination, plan)
    partition = {
        "format": NESTED_ASSET_FORMAT,
        "target_host": "149",
        **plan.as_dict(),
        "donor_partition_sha256": canonical_hash(plan.as_dict()),
    }
    atomic_json(staging / "DONOR_PARTITION.json", partition)
    common_receipt = {
        "format": NESTED_ASSET_FORMAT,
        "target_host": "149",
        "cancer_id": plan.cancer_id,
        "outer_fold_id": plan.outer_fold_id,
        "inner_target_fold_id": plan.inner_target_fold_id,
        "inner_validation_fold_id": plan.inner_validation_fold_id,
        "role": plan.role,
        "geometry_isolation": plan.geometry_isolation,
        "validation_provider": plan.validation_provider,
        "support_limited_fit": plan.support_limited_fit,
        "minimum_source_donors": plan.minimum_source_donors,
        "source_donors": list(plan.source_donors),
        "target_donors": list(plan.inner_target_donors),
        "validation_donors": list(plan.inner_validation_donors),
        "outer_query_donors": list(plan.outer_query_donors),
        "excluded_donors": list(plan.excluded_donors),
        "donor_partition_sha256": canonical_hash(plan.as_dict()),
        "formal_pointer_update_authorized": False,
    }
    if not plan.available:
        atomic_json(
            staging / "UNAVAILABLE.json",
            {
                **partition,
                "state": "NESTED_EPISODE_UNAVAILABLE",
                "reason": f"INSUFFICIENT_SOURCE_DONORS_LT{plan.minimum_source_donors}",
            },
        )
        return _finish(
            staging,
            destination,
            {
                **common_receipt,
                "state": "NESTED_EPISODE_UNAVAILABLE",
                "unavailable_reason": f"INSUFFICIENT_SOURCE_DONORS_LT{plan.minimum_source_donors}",
                "finished_unix": time.time(),
            },
        )
    if time.time() >= settings.deadline:
        raise TimeoutError("nested episode deadline reached")
    metacell = fit_nested_metacells(
        matrices["coding_log"],
        matrices["coding"],
        matrices["lncrna"],
        matrices["library"],
        matrices["metadata"],
        plan,
        measurement_scale=str(mapped["measurement_scale"]),
        max_genes=settings.pca_genes,
        components=settings.pca_components,
        max_cells_per_donor=settings.pca_cells_per_donor,
        target_size=settings.target_size,
        min_size=settings.min_metacell_size,
        radius_quantile=settings.radius_quantile,
        seed=settings.seed
        + plan.outer_fold_id * 1000
        + plan.inner_target_fold_id * 100,
    )
    base = _load_script("nested_base_materializer", "materialize_single_cell_metacell_149.py")
    activity = base._ucell_from_metacells(
        metacell["coding"],
        matrices["signature"],
        matrices["signature_counts"],
        matrices["missing_counts"],
        settings.deadline,
    ).astype(np.float32, copy=False)
    observations = lasso_observations(metacell["metadata"], plan)
    universe, targets, coverage = _context_assets(
        metacell,
        activity,
        ids,
        bulk,
        plan,
        settings,
        dataset_id=str(mapped["dataset_id"]),
        measurement_scale=str(mapped["measurement_scale"]),
    )
    features, contexts = source_node_features(
        metacell["coding"],
        metacell["lncrna"],
        activity,
        metacell["metadata"],
        plan,
        family_membership,
        target_rows=targets,
    )
    validate_nested_isolation(
        plan, metacell["geometry_audit"], observations, universe, targets
    )

    metacell["membership"].to_parquet(
        staging / "METACELL_MEMBERSHIP.parquet", index=False
    )
    metacell["metadata"].to_parquet(
        staging / "METACELL_METADATA.parquet", index=False
    )
    sparse.save_npz(
        staging / "METACELL_CODING.npz", metacell["coding"], compressed=True
    )
    sparse.save_npz(
        staging / "METACELL_LNCRNA.npz", metacell["lncrna"], compressed=True
    )
    with (staging / "METACELL_PATHWAY_ACTIVITY.npy").open("wb") as handle:
        np.save(handle, activity, allow_pickle=False)
    observations.to_parquet(staging / "LASSO_OBSERVATIONS.parquet", index=False)
    observations.loc[observations.lasso_input_allowed].to_parquet(
        staging / "LASSO_SOURCE_ROWS.parquet", index=False
    )
    universe.to_parquet(staging / "CANDIDATE_UNIVERSE.parquet", index=False)
    targets.to_parquet(staging / "SIGNED_TARGET_ROWS.parquet", index=False)
    contexts.to_parquet(staging / "NODE_FEATURE_CONTEXTS.parquet", index=False)
    with (staging / "NODE_FEATURES.npz").open("wb") as handle:
        np.savez_compressed(handle, **features)
    atomic_json(staging / "GEOMETRY_AUDIT.json", metacell["geometry_audit"])
    atomic_json(
        staging / "CONTEXT_COVERAGE.json",
        {
            "format": NESTED_ASSET_FORMAT,
            "geometry_isolation": plan.geometry_isolation,
            "contexts": coverage,
        },
    )
    atomic_json(
        staging / "STATIC_BINDING.json",
        {
            "format": NESTED_ASSET_FORMAT,
            "graph_asset_static_success": file_record(static_root / "SUCCESS.json"),
            "matrix_identity": file_record(matrix_identity_receipt),
        },
    )
    state = (
        "NESTED_EPISODE_ASSETS_READY"
        if len(targets) and len(contexts)
        else "NESTED_EPISODE_NO_ESTIMABLE_CONTEXTS"
    )
    return _finish(
        staging,
        destination,
        {
            **common_receipt,
            "state": state,
            "unavailable_reason": (
                None if state == "NESTED_EPISODE_ASSETS_READY" else state
            ),
            "measurement_scale": mapped["measurement_scale"],
            "aggregation_policy": metacell["aggregation_policy"],
            "metacells": len(metacell["metadata"]),
            "assigned_cells": int(metacell["membership"].metacell_id.ge(0).sum()),
            "unassigned_cells": int(metacell["membership"].metacell_id.lt(0).sum()),
            "lasso_source_observations": int(observations.lasso_input_allowed.sum()),
            "lasso_excluded_observations": int((~observations.lasso_input_allowed).sum()),
            "candidate_rows": len(universe),
            "signed_target_rows": len(targets),
            "feature_contexts": len(contexts),
            "graph_asset_static_success_sha256": sha256(static_root / "SUCCESS.json"),
            "matrix_identity_sha256": sha256(matrix_identity_receipt),
            "finished_unix": time.time(),
        },
    )


def _selected_ints(value: str, available: list[int], label: str) -> list[int]:
    available = [int(item) for item in available]
    text = str(value).strip().lower()
    if text == "all":
        return list(available)
    try:
        selected = int(text)
    except ValueError as exc:
        raise ValueError(f"{label}_MUST_BE_INTEGER_OR_ALL") from exc
    if selected not in available:
        raise ValueError(f"{label}_NOT_ACTIVE")
    return [selected]


def _write_outer_receipt_if_complete(
    settings, cancer_id: str, outer: int, plans: list[NestedFoldPlan]
) -> None:
    root = settings.output_root / "cancers" / cancer_id / f"outer_fold={outer}"
    records = []
    for plan in plans:
        episode = (
            root
            / f"role={plan.role}"
            / f"inner_target_fold={plan.inner_target_fold_id}"
            / "SUCCESS.json"
        )
        if not episode.is_file():
            return
        records.append(file_record(episode))
    atomic_json(
        root / "SUCCESS.json",
        {
            "format": NESTED_ASSET_FORMAT,
            "target_host": "149",
            "state": "OUTER_FOLD_NESTED_ASSETS_COMPLETE",
            "cancer_id": cancer_id,
            "outer_fold_id": outer,
            "geometry_policy": "STRICT_VALIDATION_WHERE_FEASIBLE_WITH_SUPPORT_LIMITED_LEAVE_ONE_TARGET_FIT",
            "inner_validation_fold_id": plans[0].inner_validation_fold_id,
            "validation_provider": plans[0].validation_provider,
            "support_limited_fit": plans[0].support_limited_fit,
            "episode_receipts": records,
            "formal_pointer_update_authorized": False,
            "finished_unix": time.time(),
        },
    )


def run_cancer(settings) -> list[dict]:
    if socket.gethostname() != "149":
        raise SystemExit("WRONG_HOST")
    if time.time() >= settings.deadline:
        raise SystemExit("DEADLINE_EXPIRED")
    preflight = validate_preflight(settings)
    cancer_id = settings.cancer_id.upper()
    cancer_root = settings.output_root / "cancers" / cancer_id
    if "native_donor_preparation" in preflight and (cancer_root / "SUCCESS.json").is_file():
        from cc_hhgt.v32 import native_single_cell_episode_sources as sources
        from cc_hhgt.v32.formal_expression_sources import read
        donor_path = Path(preflight["native_donor_preparation"]["path"])
        value = read(read(donor_path)["preflight"]["path"])
        sources.nested_inventory(value, donor_path)
        return [read(episode["path"]) for outer in read(cancer_root / "SUCCESS.json")["outer_fold_receipts"]
                for episode in read(outer["path"])["episode_receipts"]]
    folds, source_cancer_receipt = _validate_source_cancer(settings, cancer_id)
    active = sorted(pd.to_numeric(folds.outer_fold_id, errors="raise").astype(int).unique())
    requested_outer = _selected_ints(settings.outer_fold, active, "OUTER_FOLD")

    base = _load_script("nested_base_materializer", "materialize_single_cell_metacell_149.py")
    scratch = settings.scratch_root.resolve() / cancer_id
    if settings.scratch_root.resolve() not in scratch.parents:
        raise RuntimeError("NESTED_SCRATCH_CONTAINMENT_FAILURE")
    if scratch.exists() and not (scratch / "MAPPED_SUCCESS.json").is_file():
        shutil.rmtree(scratch)
    mapped = base._materialize_mapped_matrices(
        cancer_id, scratch, settings.repair_root, settings.deadline
    )
    ids = json.loads((scratch / "IDS.json").read_text(encoding="utf-8"))
    ids = {key: list(map(str, value)) for key, value in ids.items()}
    static_root, _, family_membership, families = _validate_static(
        settings, cancer_id, ids
    )
    cancer_root = settings.output_root / "cancers" / cancer_id
    identity_receipt = _ensure_matrix_identity(
        cancer_root,
        ids,
        families,
        scratch / "MAPPED_SUCCESS.json",
        static_root / "SUCCESS.json",
    )
    matrices = {
        "metadata": pd.read_parquet(scratch / "CELL_METADATA.parquet"),
        "coding": sparse.load_npz(scratch / "CODING_MEASURED.npz").tocsr(),
        "lncrna": sparse.load_npz(scratch / "LNC_MEASURED.npz").tocsr(),
        "coding_log": sparse.load_npz(scratch / "CODING_LOG_FOR_GROUPING.npz").tocsr(),
        "library": np.load(scratch / "ALL_GENE_LIBRARY.npy"),
        "signature": sparse.load_npz(scratch / "EXACT_SIGNATURE_MATRIX.npz").tocsr(),
        "signature_counts": np.load(scratch / "SIGNATURE_GENE_COUNTS.npy"),
        "missing_counts": np.load(scratch / "MISSING_GENE_COUNTS.npy"),
    }
    if "native_donor_preparation" in preflight:
        from cc_hhgt.v32.native_single_cell_donors_workflow import consumer
        matrices["metadata"], _, _ = consumer(settings, preflight)
    bulk = base._bulk_candidates(
        settings.candidates, cancer_id, ids["lncrna_ids"], ids["pathway_ids"]
    )
    receipts: list[dict] = []
    plan_table_parts = []
    for outer in requested_outer:
        plans = nested_fold_plans(folds, cancer_id=cancer_id, outer_fold_id=outer)
        selected_targets = _selected_ints(
            settings.inner_target_fold,
            [plan.inner_target_fold_id for plan in plans],
            "INNER_TARGET_FOLD",
        )
        selected = [
            plan for plan in plans if plan.inner_target_fold_id in selected_targets
        ]
        plan_table_parts.append(_plan_rows(plans))
        outer_root = cancer_root / f"outer_fold={outer}"
        outer_root.mkdir(parents=True, exist_ok=True)
        plan_path = outer_root / "NESTED_PLAN.parquet"
        expected_plan = _plan_rows(plans)
        if plan_path.is_file():
            existing = pd.read_parquet(plan_path)
            if canonical_hash(existing.to_dict("records")) != canonical_hash(
                expected_plan.to_dict("records")
            ):
                raise RuntimeError("NESTED_PLAN_DRIFT")
        else:
            temporary = plan_path.with_suffix(".parquet.partial")
            expected_plan.to_parquet(temporary, index=False)
            os.replace(temporary, plan_path)
        for plan in selected:
            receipts.append(
                materialize_plan(
                    settings,
                    plan,
                    mapped,
                    matrices,
                    ids,
                    bulk,
                    family_membership,
                    static_root,
                    identity_receipt,
                )
            )
            print(
                json.dumps(
                    {
                        "cancer_id": cancer_id,
                        "outer_fold_id": outer,
                        "inner_target_fold_id": plan.inner_target_fold_id,
                        "role": plan.role,
                        "state": receipts[-1]["state"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        _write_outer_receipt_if_complete(settings, cancer_id, outer, plans)
    if str(settings.outer_fold).lower() == "all" and str(
        settings.inner_target_fold
    ).lower() == "all":
        outer_records = []
        for outer in active:
            success = cancer_root / f"outer_fold={outer}" / "SUCCESS.json"
            if not success.is_file():
                break
            outer_records.append(file_record(success))
        else:
            atomic_json(
                cancer_root / "SUCCESS.json",
                {
                    "format": NESTED_ASSET_FORMAT,
                    "target_host": "149",
                    "state": "CANCER_NESTED_ASSETS_COMPLETE",
                    "cancer_id": cancer_id,
                    "source_cancer_success_sha256": sha256(
                        Path(source_cancer_receipt["source_receipt_path"])
                        if "native_donor_preparation" in preflight else
                        settings.source_result_root / "cancers" / cancer_id / "SUCCESS.json"
                    ),
                    "source_kind": "NATIVE_CANONICAL_DONOR_PREPARATION" if "native_donor_preparation" in preflight else "HISTORICAL_METACELL_MATERIALIZATION",
                    "source_dataset_id": source_cancer_receipt.get("dataset_id"),
                    "outer_fold_receipts": outer_records,
                    "matrix_identity": file_record(identity_receipt),
                    "formal_pointer_update_authorized": False,
                    "finished_unix": time.time(),
                },
            )
    if settings.delete_verified_scratch:
        for receipt in receipts:
            destination = (
                settings.output_root
                / "cancers"
                / cancer_id
                / f"outer_fold={receipt['outer_fold_id']}"
                / f"role={receipt['role']}"
                / f"inner_target_fold={receipt['inner_target_fold_id']}"
            )
            if _verify_complete_metadata(
                destination,
                next(
                    plan
                    for outer in requested_outer
                    for plan in nested_fold_plans(
                        folds, cancer_id=cancer_id, outer_fold_id=outer
                    )
                    if plan.outer_fold_id == receipt["outer_fold_id"]
                    and plan.inner_target_fold_id == receipt["inner_target_fold_id"]
                ),
            ) is None:
                raise RuntimeError("REFUSING_TO_DELETE_UNVERIFIED_NESTED_SCRATCH")
        resolved = scratch.resolve()
        root = settings.scratch_root.resolve()
        if resolved == root or root not in resolved.parents:
            raise RuntimeError("REFUSING_UNSAFE_NESTED_SCRATCH_DELETION")
        shutil.rmtree(resolved)
    return receipts


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--cancer-id", required=True)
    result.add_argument("--outer-fold", default="all")
    result.add_argument("--inner-target-fold", default="all")
    result.add_argument("--source-result-root", required=True, type=Path)
    result.add_argument("--graph-asset-root", required=True, type=Path)
    result.add_argument("--output-root", required=True, type=Path)
    result.add_argument("--scratch-root", required=True, type=Path)
    result.add_argument(
        "--repair-root",
        type=Path,
        default=Path("/dsk2/user/dengsc/sc_preprocessing_repair_20261004_r1"),
    )
    result.add_argument("--candidates", required=True, type=Path)
    result.add_argument("--deadline", required=True, type=float)
    result.add_argument("--seed", type=int, default=1729)
    result.add_argument("--pca-genes", type=int, default=1500)
    result.add_argument("--pca-components", type=int, default=20)
    result.add_argument("--pca-cells-per-donor", type=int, default=2000)
    result.add_argument("--target-size", type=int, default=5)
    result.add_argument("--min-metacell-size", type=int, default=3)
    result.add_argument("--radius-quantile", type=float, default=0.9)
    result.add_argument("--min-target-metacells", type=int, default=20)
    result.add_argument("--discovery-top-k", type=int, default=12)
    result.add_argument("--sketch-oversample", type=int, default=4)
    result.add_argument("--sketch-dimensions", type=int, default=64)
    result.add_argument("--min-screen-abs-rho", type=float, default=0.1)
    result.add_argument("--delete-verified-scratch", action="store_true")
    return result


if __name__ == "__main__":
    signal.signal(
        signal.SIGTERM,
        lambda *_: (_ for _ in ()).throw(TimeoutError("SIGTERM")),
    )
    print(json.dumps(run_cancer(parser().parse_args()), sort_keys=True))
