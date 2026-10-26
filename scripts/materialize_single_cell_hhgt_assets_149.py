#!/usr/bin/env python3
"""Reaggregate SC-HHGT graph inputs from frozen metacell membership on 149.

The retained metacell materialization is the geometry authority.  This script
reloads original measurements, aligns them by ``cell_id``, and aggregates with
the frozen assignment.  It never calls PCA or metacell grouping functions.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.single_cell_graph_assets import (  # noqa: E402
    ASSET_FORMAT,
    MEMBERSHIP_ORIENTATION,
    UNASSIGNED_PATHWAY_FAMILY_ID,
    aggregate_frozen_measurements,
    align_frozen_membership,
    context_feature_scope,
    gene_by_pathway_membership,
    lasso_observations,
    normalize_external_evidence,
    normalize_pathway_families,
    canonical_biological_id,
)


PREFLIGHT_FORMAT = "CANCERLNCATLAS_V32_SC_HHGT_GRAPH_ASSET_PREFLIGHT_V1"
EXTERNAL_COLUMNS = [
    "lncrna_id", "partner_id", "partner_type", "relation_type", "direction",
    "experiment_family", "is_experimental", "is_predicted", "pmid",
    "source_database", "mapping_status",
]


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
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_base_materializer():
    path = ROOT / "scripts" / "materialize_single_cell_metacell_149.py"
    spec = importlib.util.spec_from_file_location("frozen_metacell_materializer", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _verify_record(record: dict[str, object], expected: Path, label: str) -> None:
    path = Path(str(record.get("path", ""))).resolve()
    expected = expected.resolve()
    if path != expected or not path.is_file():
        raise RuntimeError(f"{label} path drift")
    if int(record.get("bytes", -1)) != path.stat().st_size:
        raise RuntimeError(f"{label} byte count drift")
    if str(record.get("sha256", "")) != sha256(path):
        raise RuntimeError(f"{label} SHA256 drift")


def validate_preflight(settings) -> dict[str, object]:
    path = settings.output_root / "LAUNCH_PREFLIGHT.json"
    if not path.is_file():
        raise RuntimeError("GRAPH_ASSET_LAUNCH_PREFLIGHT_MISSING")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "format": PREFLIGHT_FORMAT,
        "target_host": "149",
        "cpu_only": True,
        "paid_compute": False,
        "source_metacell_result_root": str(settings.source_result_root.resolve()),
        "pathway_family_path": str(settings.pathway_family.resolve()),
        "external_evidence_path": str(settings.external_evidence.resolve()),
        "formal_pointer_update_authorized": False,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise RuntimeError(f"GRAPH_ASSET_PREFLIGHT_{key.upper()}_DRIFT")
    for prefix, source in (
        ("pathway_family", settings.pathway_family),
        ("external_evidence", settings.external_evidence),
    ):
        if receipt.get(f"{prefix}_sha256") != sha256(source):
            raise RuntimeError(f"GRAPH_ASSET_PREFLIGHT_{prefix.upper()}_SHA256_DRIFT")
    source_validation = settings.source_result_root / "MATERIALIZATION_VALIDATION.json"
    if not source_validation.is_file():
        raise RuntimeError("SOURCE_METACELL_VALIDATION_REPORT_MISSING")
    if receipt.get("source_validation_sha256") != sha256(source_validation):
        raise RuntimeError("SOURCE_METACELL_VALIDATION_SHA256_DRIFT")
    return receipt


def _validate_source_fold(source_fold: Path) -> dict[str, object]:
    success_path = source_fold / "SUCCESS.json"
    if not success_path.is_file():
        raise RuntimeError("SOURCE_METACELL_FOLD_NOT_COMPLETE")
    receipt = json.loads(success_path.read_text(encoding="utf-8"))
    if receipt.get("target_host") != "149" or receipt.get("state") != (
        "FOLD_LOCAL_METACELL_TARGETS_READY"
    ):
        raise RuntimeError("SOURCE_METACELL_FOLD_RECEIPT_INVALID")
    for name in (
        "METACELL_MEMBERSHIP.parquet", "METACELL_METADATA.parquet",
        "TRAIN_ROWS.parquet", "QUERY_ROWS.parquet", "UNIVERSE.parquet",
    ):
        record = receipt.get("files", {}).get(name)
        if not isinstance(record, dict):
            raise RuntimeError(f"SOURCE_METACELL_FOLD_FILE_UNBOUND:{name}")
        _verify_record(record, source_fold / name, f"source {name}")
    return receipt


def _node_identifiers(
    gene_ids: list[str],
    lncrna_ids: list[str],
    pathway_ids: list[str],
    families: pd.DataFrame,
) -> pd.DataFrame:
    frames = []
    for node_type, values in (
        ("gene", gene_ids), ("lncRNA", lncrna_ids), ("pathway", pathway_ids),
    ):
        frames.append(
            pd.DataFrame(
                {
                    "node_type": node_type,
                    "global_index": np.arange(len(values), dtype=np.int64),
                    "canonical_id": list(map(str, values)),
                }
            )
        )
    frames.append(families)
    return pd.concat(frames, ignore_index=True)


def _signature_metadata(
    pathway_ids: list[str],
    membership: sparse.csr_matrix,
    signature_counts: np.ndarray,
    missing_counts: np.ndarray,
) -> pd.DataFrame:
    present = np.asarray(membership.sum(axis=0)).ravel().astype(np.int64)
    complete = np.asarray(signature_counts, dtype=np.int64)
    missing = np.asarray(missing_counts, dtype=np.int64)
    if not (
        len(pathway_ids) == len(present) == len(complete) == len(missing)
        and np.array_equal(present + missing, complete)
    ):
        raise RuntimeError("SIGNATURE_COUNT_CONTRACT_DRIFT")
    return pd.DataFrame(
        {
            "pathway_index": np.arange(len(pathway_ids), dtype=np.int64),
            "pathway_id": pathway_ids,
            "present_gene_count": present,
            "missing_gene_count": missing,
            "complete_signature_gene_count": complete,
            "membership_orientation": MEMBERSHIP_ORIENTATION,
        }
    )


def _stream_eligible_external(
    path: Path, *, batch_size: int = 131072
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Read the legacy table once with bounded peak memory."""

    parts: list[pd.DataFrame] = []
    rows_scanned = 0
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(columns=EXTERNAL_COLUMNS, batch_size=batch_size):
        part = batch.to_pandas()
        rows_scanned += len(part)
        part = part.loc[
            part.mapping_status.astype(str).eq("lncrna_and_partner_mapped")
            & part.is_experimental.fillna(False).astype(bool)
            & ~part.is_predicted.fillna(True).astype(bool)
        ].copy()
        if not part.empty:
            parts.append(part)
    eligible = (
        pd.concat(parts, ignore_index=True)
        if parts
        else pd.DataFrame(columns=EXTERNAL_COLUMNS)
    )
    return eligible, {
        "rows_scanned": rows_scanned,
        "eligible_experimental_rows": len(eligible),
    }


def _primary_physical_binding(raw: pd.DataFrame) -> pd.DataFrame:
    values = raw.loc[
        raw.experiment_family.astype(str).str.lower().eq("physical_binding")
    ].copy()
    values["source_id"] = values.lncrna_id.map(canonical_biological_id)
    values["target_id"] = values.partner_id.map(canonical_biological_id)
    values = values.loc[values.source_id.ne("") & values.target_id.ne("")]
    grouped = values.groupby(
        ["source_id", "target_id"], observed=True, sort=True
    ).agg(
        evidence_record_count=("lncrna_id", "size"),
        distinct_pmid_count=("pmid", lambda item: item.dropna().astype(str).nunique()),
        source_databases=(
            "source_database",
            lambda item: "|".join(sorted(set(item.dropna().astype(str)))),
        ),
    ).reset_index()
    grouped.insert(0, "source_type", "lncRNA")
    grouped.insert(3, "relation", "external_physical_binding")
    grouped.insert(4, "target_type", "gene")
    grouped["edge_weight"] = np.float32(1.0)
    return grouped[
        [
            "source_type", "source_id", "relation", "target_type", "target_id",
            "edge_weight", "evidence_record_count", "distinct_pmid_count",
            "source_databases",
        ]
    ]


def _existing_complete(path: Path, state: str) -> dict[str, object] | None:
    success = path / "SUCCESS.json"
    if not path.exists():
        return None
    if not success.is_file():
        raise RuntimeError(f"INCOMPLETE_EXISTING_ASSET_DIRECTORY:{path}")
    receipt = json.loads(success.read_text(encoding="utf-8"))
    if receipt.get("state") != state:
        raise RuntimeError(f"ASSET_STATE_DRIFT:{path}")
    for name, record in receipt.get("files", {}).items():
        _verify_record(record, path / name, f"cached {state} {name}")
    return receipt


def _finish_directory(
    staging: Path, destination: Path, receipt: dict[str, object]
) -> dict[str, object]:
    receipt["files"] = {
        path.name: file_record(path)
        for path in sorted(staging.iterdir())
        if path.is_file() and path.name != "SUCCESS.json"
    }
    atomic_json(staging / "SUCCESS.json", receipt)
    os.replace(staging, destination)
    receipt["files"] = {
        path.name: file_record(path)
        for path in sorted(destination.iterdir())
        if path.is_file() and path.name != "SUCCESS.json"
    }
    atomic_json(destination / "SUCCESS.json", receipt)
    return receipt


def _ensure_global_external(settings) -> tuple[Path, dict[str, object]]:
    destination = settings.output_root / "STATIC_GLOBAL"
    if receipt := _existing_complete(destination, "GLOBAL_EXTERNAL_EVIDENCE_READY"):
        if receipt.get("source_sha256") != sha256(settings.external_evidence):
            raise RuntimeError("GLOBAL_EXTERNAL_SOURCE_SHA256_DRIFT")
        return destination, receipt
    staging = destination.with_name(f"{destination.name}.partial.{os.getpid()}")
    if staging.exists():
        raise RuntimeError("GLOBAL_EXTERNAL_STAGING_ALREADY_EXISTS")
    staging.mkdir(parents=True)
    raw, scan = _stream_eligible_external(settings.external_evidence)
    source_ids = sorted(raw.lncrna_id.map(canonical_biological_id).unique())
    target_ids = sorted(raw.partner_id.map(canonical_biological_id).unique())
    sensitivity, sensitivity_schema = normalize_external_evidence(
        raw, lncrna_ids=source_ids, gene_ids=target_ids
    )
    sensitivity = sensitivity.drop(columns=["source_index", "target_index"])
    primary = _primary_physical_binding(raw)
    primary_schema = pd.DataFrame(
        [
            {
                "source_type": "lncRNA", "relation": "external_physical_binding",
                "target_type": "gene",
            },
            {
                "source_type": "gene",
                "relation": "external_physical_binding_reverse",
                "target_type": "lncRNA",
            },
        ]
    )
    primary.to_parquet(staging / "EXTERNAL_PHYSICAL_BINDING.parquet", index=False)
    sensitivity.to_parquet(
        staging / "EXTERNAL_TYPED_SENSITIVITY.parquet", index=False
    )
    primary_schema.to_parquet(
        staging / "EXTERNAL_PRIMARY_RELATION_SCHEMA.parquet", index=False
    )
    sensitivity_schema.to_parquet(
        staging / "EXTERNAL_TYPED_RELATION_SCHEMA.parquet", index=False
    )
    receipt = {
        "format": ASSET_FORMAT,
        "target_host": "149",
        "state": "GLOBAL_EXTERNAL_EVIDENCE_READY",
        "source_path": str(settings.external_evidence.resolve()),
        "source_sha256": sha256(settings.external_evidence),
        "scan": scan,
        "primary_policy": (
            "MAPPED_EXPERIMENTAL_NONPREDICTED_PHYSICAL_BINDING_COLLAPSED_LNC_GENE"
        ),
        "primary_edges": len(primary),
        "primary_relation_types": 2,
        "sensitivity_edges": len(sensitivity),
        "sensitivity_relation_types": len(sensitivity_schema),
        "finished_unix": time.time(),
    }
    return destination, _finish_directory(staging, destination, receipt)


def _map_external_to_nodes(
    frame: pd.DataFrame, *, lncrna_ids: list[str], gene_ids: list[str]
) -> pd.DataFrame:
    lnc = {canonical_biological_id(value): (i, str(value)) for i, value in enumerate(lncrna_ids)}
    gene = {canonical_biological_id(value): (i, str(value)) for i, value in enumerate(gene_ids)}
    if len(lnc) != len(lncrna_ids) or len(gene) != len(gene_ids):
        raise RuntimeError("CANONICAL_NODE_ID_COLLISION")
    selected = frame.loc[frame.source_id.isin(lnc) & frame.target_id.isin(gene)].copy()
    selected["source_index"] = selected.source_id.map(lambda value: lnc[value][0]).astype(np.int64)
    selected["target_index"] = selected.target_id.map(lambda value: gene[value][0]).astype(np.int64)
    selected["source_id"] = selected.source_id.map(lambda value: lnc[value][1])
    selected["target_id"] = selected.target_id.map(lambda value: gene[value][1])
    columns = list(selected.columns)
    columns.insert(columns.index("source_id") + 1, columns.pop(columns.index("source_index")))
    columns.insert(columns.index("target_id") + 1, columns.pop(columns.index("target_index")))
    return selected.loc[:, columns].reset_index(drop=True)


def _ensure_cancer_static(
    settings,
    *,
    mapped_success_path: Path,
    gene_ids: list[str],
    lnc_ids: list[str],
    pathway_ids: list[str],
    pathway_by_gene: sparse.csr_matrix,
    signature_counts: np.ndarray,
    missing_counts: np.ndarray,
) -> tuple[Path, dict[str, object]]:
    cancer_id = settings.cancer_id.upper()
    destination = settings.output_root / "cancers" / cancer_id / "STATIC"
    global_root, global_receipt = _ensure_global_external(settings)
    if receipt := _existing_complete(destination, "CANCER_STATIC_GRAPH_ASSETS_READY"):
        if "native_static_preparation" in receipt:
            from cc_hhgt.v32.native_single_cell_static_workflow import validate_static
            validate_static(destination / "SUCCESS.json")
        expected = {
            "mapped_success_sha256": sha256(mapped_success_path),
            "pathway_family_source_sha256": sha256(settings.pathway_family),
            "global_static_success_sha256": sha256(global_root / "SUCCESS.json"),
        }
        for key, value in expected.items():
            if receipt.get(key) != value:
                raise RuntimeError(f"CANCER_STATIC_{key.upper()}_DRIFT")
        return destination, receipt
    staging = destination.with_name(f"{destination.name}.partial.{os.getpid()}")
    if staging.exists():
        raise RuntimeError("CANCER_STATIC_STAGING_ALREADY_EXISTS")
    staging.mkdir(parents=True)
    membership = gene_by_pathway_membership(
        pathway_by_gene, gene_count=len(gene_ids), pathway_count=len(pathway_ids)
    )
    signature_metadata = _signature_metadata(
        pathway_ids, membership, signature_counts, missing_counts
    )
    family_membership, family_hierarchy, families = normalize_pathway_families(
        pd.read_parquet(settings.pathway_family), pathway_ids
    )
    nodes = _node_identifiers(gene_ids, lnc_ids, pathway_ids, families)
    primary = _map_external_to_nodes(
        pd.read_parquet(global_root / "EXTERNAL_PHYSICAL_BINDING.parquet"),
        lncrna_ids=lnc_ids,
        gene_ids=gene_ids,
    )
    sensitivity = _map_external_to_nodes(
        pd.read_parquet(global_root / "EXTERNAL_TYPED_SENSITIVITY.parquet"),
        lncrna_ids=lnc_ids,
        gene_ids=gene_ids,
    )
    sparse.save_npz(staging / "GENE_PATHWAY_MEMBERSHIP.npz", membership, compressed=True)
    sparse.save_npz(
        staging / "PATHWAY_FAMILY_MEMBERSHIP.npz", family_membership, compressed=True
    )
    nodes.to_parquet(staging / "NODE_IDENTIFIERS.parquet", index=False)
    signature_metadata.to_parquet(staging / "SIGNATURE_METADATA.parquet", index=False)
    family_hierarchy.to_parquet(
        staging / "PATHWAY_FAMILY_HIERARCHY.parquet", index=False
    )
    primary.to_parquet(staging / "EXTERNAL_PHYSICAL_BINDING.parquet", index=False)
    sensitivity.to_parquet(
        staging / "EXTERNAL_TYPED_SENSITIVITY.parquet", index=False
    )
    atomic_json(
        staging / "MAPPED_INPUT_BINDING.json",
        {
            "format": ASSET_FORMAT,
            "original_path": str(mapped_success_path.resolve()),
            "original_sha256": sha256(mapped_success_path),
            "mapped_receipt": json.loads(mapped_success_path.read_text(encoding="utf-8")),
        },
    )
    static_binding = {
        "format": ASSET_FORMAT,
        "global_static_success": file_record(global_root / "SUCCESS.json"),
        "primary_relation_schema": file_record(
            global_root / "EXTERNAL_PRIMARY_RELATION_SCHEMA.parquet"
        ),
        "sensitivity_relation_schema": file_record(
            global_root / "EXTERNAL_TYPED_RELATION_SCHEMA.parquet"
        ),
    }
    atomic_json(staging / "GLOBAL_STATIC_BINDING.json", static_binding)
    receipt = {
        "format": ASSET_FORMAT,
        "target_host": "149",
        "state": "CANCER_STATIC_GRAPH_ASSETS_READY",
        "cancer_id": cancer_id,
        "mapped_success_path": str(mapped_success_path.resolve()),
        "mapped_success_sha256": sha256(mapped_success_path),
        "pathway_family_source_path": str(settings.pathway_family.resolve()),
        "pathway_family_source_sha256": sha256(settings.pathway_family),
        "membership_orientation": MEMBERSHIP_ORIENTATION,
        "pathway_family_membership_orientation": "pathway_by_family",
        "coding_genes": len(gene_ids),
        "lncrnas": len(lnc_ids),
        "pathways": len(pathway_ids),
        "pathway_families": len(families),
        "pathway_family_edges": int(family_membership.nnz),
        "pathways_without_family_edges": int(
            (np.asarray(family_membership.getnnz(axis=1)) == 0).sum()
        ),
        "pathways_assigned_to_explicit_unassigned_family": int(
            family_hierarchy.pathway_family_id.astype(str).eq(
                UNASSIGNED_PATHWAY_FAMILY_ID
            ).sum()
        ),
        "primary_external_edges": len(primary),
        "sensitivity_external_edges": len(sensitivity),
        "global_static_success_sha256": sha256(global_root / "SUCCESS.json"),
        "global_primary_edges": global_receipt["primary_edges"],
        "global_sensitivity_edges": global_receipt["sensitivity_edges"],
        "finished_unix": time.time(),
    }
    return destination, _finish_directory(staging, destination, receipt)


def materialize_fold(settings) -> dict[str, object]:
    if socket.gethostname() != "149":
        raise SystemExit("WRONG_HOST")
    if time.time() >= settings.deadline:
        raise SystemExit("DEADLINE_EXPIRED")
    validate_preflight(settings)
    cancer_id = settings.cancer_id.upper()
    fold = int(settings.outer_fold)
    source_fold = (
        settings.source_result_root / "cancers" / cancer_id / f"outer_fold={fold}"
    )
    source_receipt = _validate_source_fold(source_fold)
    destination = settings.output_root / "cancers" / cancer_id / f"outer_fold={fold}"
    if destination.exists():
        success = destination / "SUCCESS.json"
        if not success.is_file():
            raise RuntimeError("GRAPH_ASSET_DESTINATION_EXISTS_INCOMPLETE")
        receipt = json.loads(success.read_text(encoding="utf-8"))
        for name, record in receipt.get("files", {}).items():
            _verify_record(record, destination / name, f"graph asset {name}")
        return receipt
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f"{destination.name}.partial.{os.getpid()}")
    if staging.exists():
        raise RuntimeError("GRAPH_ASSET_STAGING_ALREADY_EXISTS")
    staging.mkdir(parents=True)

    base = _load_base_materializer()
    scratch = settings.scratch_root.resolve() / cancer_id
    if settings.scratch_root.resolve() not in scratch.parents:
        raise RuntimeError("GRAPH_ASSET_SCRATCH_CONTAINMENT_FAILURE")
    mapped_receipt = base._materialize_mapped_matrices(
        cancer_id, scratch, settings.repair_root, settings.deadline
    )
    cell_metadata = pd.read_parquet(scratch / "CELL_METADATA.parquet")
    frozen_membership = pd.read_parquet(source_fold / "METACELL_MEMBERSHIP.parquet")
    frozen_metadata = pd.read_parquet(source_fold / "METACELL_METADATA.parquet")
    assignments = align_frozen_membership(cell_metadata, frozen_membership)

    coding = sparse.load_npz(scratch / "CODING_MEASURED.npz")
    lncrna = sparse.load_npz(scratch / "LNC_MEASURED.npz")
    library = np.load(scratch / "ALL_GENE_LIBRARY.npy")
    coding_mc, lnc_mc, metacell_ids, aggregation_policy = (
        aggregate_frozen_measurements(
            coding,
            lncrna,
            assignments,
            measurement_scale=str(mapped_receipt["measurement_scale"]),
            all_gene_library=library,
        )
    )
    if aggregation_policy != source_receipt.get("aggregation_policy"):
        raise RuntimeError("FROZEN_AGGREGATION_POLICY_DRIFT")
    observations = lasso_observations(
        frozen_metadata,
        metacell_ids,
        source_donors=source_receipt["source_donors"],
        query_donors=source_receipt["query_donors"],
    )

    ids = json.loads((scratch / "IDS.json").read_text(encoding="utf-8"))
    gene_ids = list(map(str, ids["protein_ids"]))
    lnc_ids = list(map(str, ids["lncrna_ids"]))
    pathway_ids = list(map(str, ids["pathway_ids"]))
    pathway_by_gene = sparse.load_npz(scratch / "EXACT_SIGNATURE_MATRIX.npz")
    signature_counts = np.load(scratch / "SIGNATURE_GENE_COUNTS.npy")
    missing_counts = np.load(scratch / "MISSING_GENE_COUNTS.npy")
    static_root, static_receipt = _ensure_cancer_static(
        settings,
        mapped_success_path=scratch / "MAPPED_SUCCESS.json",
        gene_ids=gene_ids,
        lnc_ids=lnc_ids,
        pathway_ids=pathway_ids,
        pathway_by_gene=pathway_by_gene,
        signature_counts=signature_counts,
        missing_counts=missing_counts,
    )
    activity = base._ucell_from_metacells(
        coding_mc,
        pathway_by_gene,
        signature_counts,
        missing_counts,
        settings.deadline,
    ).astype(np.float32, copy=False)

    universe = pd.read_parquet(source_fold / "UNIVERSE.parquet")
    scope = context_feature_scope(universe, lnc_ids, pathway_ids)

    sparse.save_npz(staging / "METACELL_CODING.npz", coding_mc, compressed=True)
    sparse.save_npz(staging / "METACELL_LNCRNA.npz", lnc_mc, compressed=True)
    with (staging / "METACELL_PATHWAY_ACTIVITY.npy").open("wb") as handle:
        np.save(handle, activity, allow_pickle=False)
    observations.to_parquet(staging / "LASSO_OBSERVATIONS.parquet", index=False)
    scope.to_parquet(staging / "CONTEXT_FEATURE_SCOPE.parquet", index=False)
    atomic_json(
        staging / "STATIC_BINDING.json",
        {
            "format": ASSET_FORMAT,
            "cancer_static_success": file_record(static_root / "SUCCESS.json"),
        },
    )
    target_binding = {
        "format": ASSET_FORMAT,
        "target_policy": source_receipt["target_policy"],
        "train_rows": file_record(source_fold / "TRAIN_ROWS.parquet"),
        "query_rows": file_record(source_fold / "QUERY_ROWS.parquet"),
        "universe": file_record(source_fold / "UNIVERSE.parquet"),
    }
    atomic_json(staging / "TARGET_BINDING.json", target_binding)

    receipt = {
        "format": ASSET_FORMAT,
        "target_host": "149",
        "state": "FOLD_GRAPH_ASSETS_READY",
        "cancer_id": cancer_id,
        "outer_fold_id": fold,
        "source_donors": source_receipt["source_donors"],
        "query_donors": source_receipt["query_donors"],
        "source_donor_hash": canonical_hash(source_receipt["source_donors"]),
        "query_donor_hash": canonical_hash(source_receipt["query_donors"]),
        "source_fold_success_path": str((source_fold / "SUCCESS.json").resolve()),
        "source_fold_success_sha256": sha256(source_fold / "SUCCESS.json"),
        "frozen_membership_sha256": sha256(
            source_fold / "METACELL_MEMBERSHIP.parquet"
        ),
        "geometry_recomputed": False,
        "frozen_membership_reused": True,
        "measurement_scale": mapped_receipt["measurement_scale"],
        "aggregation_policy": aggregation_policy,
        "metacells": len(observations),
        "coding_genes": len(gene_ids),
        "lncrnas": len(lnc_ids),
        "pathways": len(pathway_ids),
        "pathway_families": static_receipt["pathway_families"],
        "cancer_static_success_path": str((static_root / "SUCCESS.json").resolve()),
        "cancer_static_success_sha256": sha256(static_root / "SUCCESS.json"),
        "geometry_isolation": "outer_fold_only",
        "contexts": int(observations.cell_key.nunique()),
        "primary_external_edges": static_receipt["primary_external_edges"],
        "sensitivity_external_edges": static_receipt["sensitivity_external_edges"],
        "lasso_coefficients_computed": False,
        "lasso_ready_source_observations": int(
            observations.lasso_input_allowed.sum()
        ),
        "query_observations_forbidden_from_lasso": int(
            (~observations.lasso_input_allowed).sum()
        ),
        "finished_unix": time.time(),
    }
    return _finish_directory(staging, destination, receipt)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--cancer-id", required=True)
    result.add_argument(
        "--outer-fold",
        required=True,
        help="0..4 or 'all'; all reuses one mapped-cell scratch and cancer STATIC cache",
    )
    result.add_argument("--source-result-root", required=True, type=Path)
    result.add_argument("--output-root", required=True, type=Path)
    result.add_argument("--scratch-root", required=True, type=Path)
    result.add_argument(
        "--repair-root",
        type=Path,
        default=Path("/dsk2/user/dengsc/sc_preprocessing_repair_20261004_r1"),
    )
    result.add_argument("--pathway-family", required=True, type=Path)
    result.add_argument("--external-evidence", required=True, type=Path)
    result.add_argument("--deadline", required=True, type=float)
    result.add_argument("--delete-verified-scratch", action="store_true")
    return result


def materialize_requested(settings) -> list[dict[str, object]]:
    requested = str(settings.outer_fold).strip().lower()
    if requested == "all":
        cancer_root = settings.source_result_root / "cancers" / settings.cancer_id.upper()
        folds = sorted(
            int(path.name.split("=", 1)[1])
            for path in cancer_root.glob("outer_fold=*")
            if (path / "SUCCESS.json").is_file()
        )
        if not folds:
            raise RuntimeError("SOURCE_CANCER_HAS_NO_COMPLETE_OUTER_FOLDS")
    else:
        try:
            fold = int(requested)
        except ValueError as exc:
            raise ValueError("OUTER_FOLD_MUST_BE_0_TO_4_OR_ALL") from exc
        if fold not in range(5):
            raise ValueError("OUTER_FOLD_MUST_BE_0_TO_4_OR_ALL")
        folds = [fold]
    receipts = []
    for fold in folds:
        arguments = copy.copy(settings)
        arguments.outer_fold = fold
        receipts.append(materialize_fold(arguments))
    if settings.delete_verified_scratch:
        for fold in folds:
            destination = (
                settings.output_root
                / "cancers"
                / settings.cancer_id.upper()
                / f"outer_fold={fold}"
            )
            if _existing_complete(destination, "FOLD_GRAPH_ASSETS_READY") is None:
                raise RuntimeError("REFUSING_TO_DELETE_UNVERIFIED_GRAPH_SCRATCH")
        scratch = settings.scratch_root.resolve() / settings.cancer_id.upper()
        scratch_root = settings.scratch_root.resolve()
        if scratch == scratch_root or scratch_root not in scratch.parents:
            raise RuntimeError("REFUSING_UNSAFE_GRAPH_SCRATCH_DELETION")
        if scratch.exists():
            shutil.rmtree(scratch)
    return receipts


if __name__ == "__main__":
    print(json.dumps(materialize_requested(parser().parse_args()), sort_keys=True))
