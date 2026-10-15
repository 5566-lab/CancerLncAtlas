#!/usr/bin/env python3
"""Materialize fold-local metacell training/OOF rows on host 149."""
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

import h5py
import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.single_cell_cell_level import (  # noqa: E402
    rank_expression_ucell,
    score_ucell_pathway_block,
)
from cc_hhgt.v32.single_cell_metacell_screen import (  # noqa: E402
    SCREEN_POLICY,
    donor_folds,
    exact_donor_targets,
    screen_source_pairs,
)
from cc_hhgt.v32.single_cell_metacells import (  # noqa: E402
    LocalSpace,
    aggregate_measurements,
    local_disjoint_groups,
)


FORMAT = "CANCERLNCATLAS_V32_SC_METACELL_FOLD_LOCAL_V1"
TARGET_POLICY = "SIGNED_WITHIN_DONOR_CELLTYPE_METACELL_SPEARMAN_RHO"
PYTHON_PACKAGES = Path(
    "/public0/DSC/CancerLncAtlas/runtime/python_packages_v32_drug_20260825"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def file_record(path: Path) -> dict[str, object]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def _load_upstream_bundle(cancer_id: str, repair_root: Path, capsule: Path):
    job = repair_root / "jobs" / cancer_id
    required = [
        job / "RUN_STATUS.json",
        job / "SERVER_PREFLIGHT.json",
        job / "TRAINING_HANDOFF.json",
        job / "dataset_manifest_33c.parquet",
    ]
    if not all(path.is_file() for path in required):
        raise RuntimeError(f"incomplete preprocessing capsule for {cancer_id}")
    active_status_path = Path(
        "/dell_2/DSC/CancerLncAtlas/runtime/"
        "single_cell_cell_level_r11_rescued_20260903_r1/RUN_STATUS.json"
    )
    active = json.loads(active_status_path.read_text())
    record = active.get("per_cancer", {}).get(cancer_id)
    if not record or record.get("formal_eligible") is not True:
        raise RuntimeError(f"active status does not authorize {cancer_id}")
    record = json.loads(json.dumps(record))
    capsule.mkdir(parents=True, exist_ok=True)
    metadata_source = Path(record["metadata_path"])
    if "/processed/sc_tool_input/" not in str(metadata_source).replace("\\", "/"):
        metadata_copy = (
            capsule / "processed" / "sc_tool_input" / cancer_id / "cell_metadata.parquet"
        )
        metadata_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(metadata_source, metadata_copy)
        if sha256(metadata_copy) != str(record["metadata_sha256"]):
            raise RuntimeError("active metadata copy hash drift")
        record["metadata_path"] = str(metadata_copy)
    shutil.copyfile(job / "dataset_manifest_33c.parquet", capsule / "dataset_manifest_33c.parquet")
    shutil.copyfile(job / "TRAINING_HANDOFF.json", capsule / "TRAINING_HANDOFF.json")
    atomic_json(
        capsule / "SERVER_PREFLIGHT.json",
        {
            "format": "CC_HHGT_V3_2_SINGLE_CELL_R11_RESCUED_SERVER_PREFLIGHT_V1",
            "target_host": "149",
            "r7_root": str(capsule.resolve()),
            "all_formal_raw_schemas_pass": True,
            "safe_to_start_fresh_cell_level_compute": True,
            "heavy_recompute_started": False,
        },
    )
    atomic_json(
        capsule / "RUN_STATUS.json",
        {
            "derived_assets_bound": False,
            "formal_eligible_cancer_count": 1,
            "formal_eligible_cancers": [cancer_id],
            "per_cancer": {cancer_id: record},
        },
    )
    runner_path = ROOT / "scripts" / "run_v32_single_cell_verified_streaming.py"
    spec = importlib.util.spec_from_file_location("metacell_upstream_runner", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    arguments = module.build_parser().parse_args(
        [
            "--mode", "plan",
            "--r7-root", str(capsule),
            "--run-status-json", str(capsule / "RUN_STATUS.json"),
            "--server-preflight-json", str(capsule / "SERVER_PREFLIGHT.json"),
            "--expected-server-preflight-sha256", sha256(capsule / "SERVER_PREFLIGHT.json"),
            "--expected-run-status-sha256", sha256(capsule / "RUN_STATUS.json"),
            "--cancer-id", cancer_id,
        ]
    )
    return module, module._load_contract(arguments), capsule


def _materialize_mapped_matrices(
    cancer_id: str,
    scratch: Path,
    repair_root: Path,
    deadline: float,
) -> dict[str, object]:
    success = scratch / "MAPPED_SUCCESS.json"
    if success.is_file():
        receipt = json.loads(success.read_text())
        for record in receipt["files"].values():
            path = Path(record["path"])
            if not path.is_file() or sha256(path) != record["sha256"]:
                raise RuntimeError("mapped matrix cache drift")
        return receipt
    if scratch.exists():
        raise RuntimeError(f"partial scratch exists without success: {scratch}")
    scratch.mkdir(parents=True)
    module, bundle, job = _load_upstream_bundle(
        cancer_id, repair_root, scratch / "UPSTREAM_CAPSULE"
    )
    metadata = bundle["metadata"].loc[bundle["keep_mask"]].reset_index(drop=True).copy()
    metadata["donor_id"] = metadata.patient_id.astype(str)
    metadata["cell_type"] = (
        metadata.compartment.astype(str) + "::" + metadata.cell_type_major.astype(str)
    )
    metadata["state_id"] = "Not_assessed"
    metadata.to_parquet(scratch / "CELL_METADATA.parquet", index=False)
    mapping = bundle["mapping"]
    coding_parts: list[sparse.csr_matrix] = []
    lnc_parts: list[sparse.csr_matrix] = []
    libraries: list[np.ndarray] = []
    h5_path = bundle["input_paths"]["h5"]
    with h5py.File(h5_path, "r") as handle:
        group = handle["matrix"] if "matrix" in handle else handle
        for start in range(0, bundle["shape"][1], 256):
            if time.time() >= deadline:
                raise TimeoutError("materialization deadline reached")
            stop = min(start + 256, bundle["shape"][1])
            keep = bundle["keep_mask"][start:stop]
            if keep.any():
                block = module.read_csc_column_block(
                    data=group["data"],
                    indices=group["indices"],
                    indptr=group["indptr"],
                    shape=bundle["shape"],
                    start=start,
                    stop=stop,
                )[:, keep]
                module._validate_measurement_block(
                    block, measurement_scale=bundle["measurement_scale"]
                )
                coding_parts.append(
                    sparse.csr_matrix((mapping["protein_map"] @ block).T, dtype=np.float32)
                )
                lnc_parts.append(
                    sparse.csr_matrix((mapping["lncrna_map"] @ block).T, dtype=np.float32)
                )
                libraries.append(np.asarray(block.sum(axis=0)).ravel().astype(np.float64))
            if start % 4096 == 0 or stop == bundle["shape"][1]:
                atomic_json(
                    scratch / "STATUS.json",
                    {
                        "target_host": "149",
                        "stage": "MAPPING_RAW_CELLS",
                        "cancer_id": cancer_id,
                        "raw_cells_done": stop,
                        "raw_cells_total": bundle["shape"][1],
                        "time": time.time(),
                    },
                )
    coding = sparse.vstack(coding_parts, format="csr", dtype=np.float32)
    lnc = sparse.vstack(lnc_parts, format="csr", dtype=np.float32)
    library = np.concatenate(libraries)
    if len(metadata) != coding.shape[0] or len(metadata) != lnc.shape[0]:
        raise RuntimeError("eligible metadata/matrix alignment drift")
    scale = bundle["measurement_scale"]
    if scale == "raw_counts":
        if (library <= 0).any():
            raise RuntimeError("raw count cell has empty all-gene library")
        coding_log = sparse.diags(10000.0 / library) @ coding
        coding_log.data = np.log1p(coding_log.data)
        aggregation_policy = "RAW_SUM_ALL_GENE_LIBRARY_1E4_LOG1P"
    elif scale == "log_normalized":
        coding_log = coding.copy()
        aggregation_policy = "PROVIDED_LOG_SCALE_ARITHMETIC_MEAN_NO_INVERSION"
    else:
        raise RuntimeError(f"unsupported measurement scale {scale}")
    sparse.save_npz(scratch / "CODING_MEASURED.npz", coding)
    sparse.save_npz(scratch / "LNC_MEASURED.npz", lnc)
    sparse.save_npz(scratch / "CODING_LOG_FOR_GROUPING.npz", coding_log.tocsr())
    np.save(scratch / "ALL_GENE_LIBRARY.npy", library)
    contract = bundle["pathway_contract"]
    sparse.save_npz(
        scratch / "EXACT_SIGNATURE_MATRIX.npz",
        sparse.csr_matrix(contract.signature_matrix),
    )
    np.save(scratch / "SIGNATURE_GENE_COUNTS.npy", contract.signature_gene_counts)
    np.save(scratch / "MISSING_GENE_COUNTS.npy", contract.missing_gene_counts)
    atomic_json(
        scratch / "IDS.json",
        {
            "protein_ids": list(mapping["protein_ids"]),
            "lncrna_ids": list(mapping["lncrna_ids"]),
            "pathway_ids": list(contract.available_pathway_ids),
        },
    )
    files = {
        path.name: file_record(path)
        for path in sorted(scratch.iterdir())
        if path.is_file() and path.name not in {"STATUS.json", "MAPPED_SUCCESS.json"}
    }
    receipt = {
        "format": FORMAT,
        "target_host": "149",
        "state": "MAPPED_CELL_MEASUREMENTS_READY",
        "cancer_id": cancer_id,
        "dataset_id": bundle["dataset_id"],
        "measurement_scale": scale,
        "aggregation_policy": aggregation_policy,
        "eligible_cells": len(metadata),
        "donors": int(metadata.donor_id.nunique()),
        "cell_types": int(metadata.cell_type.nunique()),
        "coding_genes": coding.shape[1],
        "lncrnas": lnc.shape[1],
        "pathways": len(contract.available_pathway_ids),
        "upstream_job": str(job),
        "upstream_contract_sha256": bundle["contract_sha256"],
        "files": files,
        "finished_unix": time.time(),
    }
    atomic_json(success, receipt)
    return receipt


def _ucell_from_metacells(
    coding: sparse.csr_matrix,
    signature_matrix: sparse.csr_matrix,
    signature_gene_counts: np.ndarray,
    missing_gene_counts: np.ndarray,
    deadline: float,
) -> np.ndarray:
    output = np.empty((coding.shape[0], signature_matrix.shape[0]), dtype=np.float32)
    for start in range(0, coding.shape[0], 128):
        if time.time() >= deadline:
            raise TimeoutError("metacell UCell deadline reached")
        stop = min(start + 128, coding.shape[0])
        ranks = rank_expression_ucell(coding[start:stop].toarray().T, max_rank=1500)
        scores = score_ucell_pathway_block(
            ranks,
            signature_matrix=signature_matrix,
            signature_gene_counts=signature_gene_counts,
            missing_gene_counts=missing_gene_counts,
            max_rank=1500,
        )
        output[start:stop] = scores.T.astype(np.float32, copy=False)
    return output


def _bulk_candidates(path: Path, cancer_id: str, lnc_ids, pathway_ids) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=["cancer_id", "lncrna_id", "pathway_id"])
    frame = frame.loc[frame.cancer_id.astype(str).str.upper().eq(cancer_id)].copy()
    frame["lncrna_id"] = frame.lncrna_id.astype(str).str.replace(
        r"^(?i:LNCRNA:|LNC:)", "", regex=True
    )
    frame["pathway_id"] = frame.pathway_id.astype(str)
    frame = frame.loc[
        frame.lncrna_id.isin(set(map(str, lnc_ids)))
        & frame.pathway_id.isin(set(map(str, pathway_ids)))
    ]
    return frame[["lncrna_id", "pathway_id"]].drop_duplicates().reset_index(drop=True)


def _attach_source_features(
    targets: pd.DataFrame,
    source_donors: set[str],
    query_donors: set[str],
    max_train_donors_per_pair: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["lncrna_id", "pathway_id"]
    local = targets.copy()
    local["is_source_donor"] = local.query_donor_id.astype(str).isin(source_donors)
    source = local.loc[local.is_source_donor].copy()
    if source.empty:
        return source, source
    source["_fisher"] = np.arctanh(np.clip(source.rho.astype(float), -0.999999, 0.999999))
    source["_positive"] = source.rho.gt(0).astype(np.int64)
    aggregate = source.groupby(keys, observed=True).agg(
        _source_fisher_sum=("_fisher", "sum"),
        _source_positive=("_positive", "sum"),
        _source_count=("rho", "size"),
    ).reset_index()
    local = local.merge(aggregate, on=keys, how="inner", validate="many_to_one")
    subtract = local.is_source_donor.astype(np.int64)
    local["source_donor_count"] = local._source_count - subtract
    fisher_sum = local._source_fisher_sum - np.where(
        local.is_source_donor,
        np.arctanh(np.clip(local.rho.astype(float), -0.999999, 0.999999)),
        0.0,
    )
    positive = local._source_positive - np.where(local.is_source_donor, local.rho.gt(0), 0)
    local["source_meta_rho"] = np.tanh(
        fisher_sum / np.maximum(local.source_donor_count, 1)
    )
    local["source_direction_consistency"] = np.maximum(
        positive, local.source_donor_count - positive
    ) / np.maximum(local.source_donor_count, 1)
    local = local.loc[local.source_donor_count.ge(2)].copy()
    local["row_role"] = np.where(
        local.query_donor_id.astype(str).isin(query_donors), "query", "train"
    )
    train = local.loc[local.row_role.eq("train")].copy()
    train["_donor_order"] = train.query_donor_id.map(
        lambda value: hashlib.sha256(str(value).encode()).hexdigest()
    )
    train = train.sort_values(keys + ["_donor_order"], kind="mergesort")
    train = train.groupby(keys, observed=True, sort=False).head(
        max_train_donors_per_pair
    )
    query = local.loc[local.row_role.eq("query")].copy()
    drop = [
        "_fisher", "_positive", "_source_fisher_sum", "_source_positive",
        "_source_count", "_donor_order", "is_source_donor",
    ]
    return train.drop(columns=drop, errors="ignore"), query.drop(columns=drop, errors="ignore")


def _write_fold(
    cancer_id: str,
    fold: int,
    cancer_output: Path,
    scratch: Path,
    candidates_path: Path,
    settings,
) -> dict[str, object]:
    fold_output = cancer_output / f"outer_fold={fold}"
    success = fold_output / "SUCCESS.json"
    if success.is_file():
        receipt = json.loads(success.read_text())
        for record in receipt["files"].values():
            path = Path(record["path"])
            if not path.is_file() or sha256(path) != record["sha256"]:
                raise RuntimeError("fold artifact drift")
        return receipt
    if fold_output.exists():
        resolved = fold_output.resolve()
        if cancer_output.resolve() not in resolved.parents:
            raise RuntimeError("refusing unsafe partial fold cleanup")
        shutil.rmtree(resolved)
    fold_output.mkdir(parents=True)
    metadata = pd.read_parquet(scratch / "CELL_METADATA.parquet")
    coding = sparse.load_npz(scratch / "CODING_MEASURED.npz")
    lnc = sparse.load_npz(scratch / "LNC_MEASURED.npz")
    coding_log = sparse.load_npz(scratch / "CODING_LOG_FOR_GROUPING.npz")
    library = np.load(scratch / "ALL_GENE_LIBRARY.npy")
    ids = json.loads((scratch / "IDS.json").read_text())
    signature = sparse.load_npz(scratch / "EXACT_SIGNATURE_MATRIX.npz")
    signature_counts = np.load(scratch / "SIGNATURE_GENE_COUNTS.npy")
    missing_counts = np.load(scratch / "MISSING_GENE_COUNTS.npy")
    mapped_receipt = json.loads((scratch / "MAPPED_SUCCESS.json").read_text())
    folds = donor_folds(cancer_id, metadata.donor_id, folds=5, seed=settings.seed)
    query_donors = set(
        folds.loc[folds.outer_fold_id.eq(fold), "donor_id"].astype(str)
    )
    if not query_donors:
        raise RuntimeError("requested outer fold has no query donors")
    source_donors = set(folds.donor_id.astype(str)) - query_donors
    if len(source_donors) < 3:
        raise RuntimeError("fewer than three source donors")
    space = LocalSpace.fit(
        coding_log,
        metadata.donor_id,
        source_donors,
        max_genes=settings.pca_genes,
        components=settings.pca_components,
        max_cells_per_donor=settings.pca_cells_per_donor,
        seed=settings.seed + fold,
    )
    coordinates = space.transform(coding_log)
    assignments, geometry = local_disjoint_groups(
        coordinates,
        metadata[["donor_id", "cell_type", "state_id"]],
        target_size=settings.target_size,
        min_size=settings.min_metacell_size,
        radius_quantile=settings.radius_quantile,
        allowed_training_donors=source_donors,
        seed=settings.seed + 100 + fold,
    )
    metacell_coding, group_ids, aggregation_policy = aggregate_measurements(
        coding,
        assignments,
        scale=mapped_receipt["measurement_scale"],
        total_library=library,
    )
    metacell_lnc, other_ids, other_policy = aggregate_measurements(
        lnc,
        assignments,
        scale=mapped_receipt["measurement_scale"],
        total_library=library,
    )
    if not np.array_equal(group_ids, other_ids) or aggregation_policy != other_policy:
        raise RuntimeError("metacell aggregation order/policy mismatch")
    first_rows = np.asarray([np.flatnonzero(assignments == group)[0] for group in group_ids])
    metacell_metadata = metadata.iloc[first_rows][
        ["dataset_id", "cancer_id", "donor_id", "cell_type", "state_id"]
    ].reset_index(drop=True)
    metacell_metadata["metacell_id"] = group_ids
    sizes = np.bincount(assignments[assignments >= 0])
    metacell_metadata["n_cells"] = sizes[group_ids]
    membership = metadata[["cell_id", "donor_id", "cell_type", "state_id"]].copy()
    membership["metacell_id"] = assignments
    membership.to_parquet(fold_output / "METACELL_MEMBERSHIP.parquet", index=False)
    metacell_metadata.to_parquet(fold_output / "METACELL_METADATA.parquet", index=False)
    activity = _ucell_from_metacells(
        metacell_coding,
        signature,
        signature_counts,
        missing_counts,
        settings.deadline,
    )
    bulk = _bulk_candidates(
        candidates_path, cancer_id, ids["lncrna_ids"], ids["pathway_ids"]
    )
    all_train: list[pd.DataFrame] = []
    all_query: list[pd.DataFrame] = []
    all_universe: list[pd.DataFrame] = []
    context_reports: list[dict[str, object]] = []
    for context_number, cell_key in enumerate(sorted(metacell_metadata.cell_type.unique())):
        if time.time() >= settings.deadline:
            raise TimeoutError("fold context deadline reached")
        mask = metacell_metadata.cell_type.eq(cell_key).to_numpy()
        context_meta = metacell_metadata.loc[mask].reset_index(drop=True)
        present_source = source_donors & set(context_meta.donor_id.astype(str))
        present_query = query_donors & set(context_meta.donor_id.astype(str))
        if not present_query:
            continue
        if len(present_source) >= 3:
            discovery = screen_source_pairs(
                metacell_lnc[mask],
                activity[mask],
                context_meta,
                ids["lncrna_ids"],
                ids["pathway_ids"],
                present_source,
                top_k=settings.discovery_top_k,
                sketch_oversample=settings.sketch_oversample,
                sketch_dimensions=settings.sketch_dimensions,
                min_source_donors=3,
                min_metacells=settings.min_target_metacells,
                min_abs_meta_rho=settings.min_screen_abs_rho,
                seed=settings.seed + fold * 1000 + context_number,
            )
        else:
            discovery = pd.DataFrame(columns=["lncrna_id", "pathway_id"])
        discovery_pairs = discovery[["lncrna_id", "pathway_id"]].drop_duplicates()
        universe = bulk.assign(is_bulk_candidate=True).merge(
            discovery_pairs.assign(is_sc_discovery_candidate=True),
            on=["lncrna_id", "pathway_id"],
            how="outer",
            validate="one_to_one",
        )
        if universe.empty:
            continue
        universe["is_bulk_candidate"] = universe.is_bulk_candidate.eq(True)
        universe["is_sc_discovery_candidate"] = (
            universe.is_sc_discovery_candidate.eq(True)
        )
        universe["universe_role"] = np.select(
            [
                universe.is_bulk_candidate & universe.is_sc_discovery_candidate,
                universe.is_bulk_candidate,
            ],
            ["bulk_and_sc_discovery", "bulk_rescore"],
            default="sc_discovery",
        )
        universe["selection_policy"] = np.where(
            universe.is_sc_discovery_candidate, SCREEN_POLICY, "FROZEN_BULK_EXACT_PAIR"
        )
        universe["candidate_selection_query_donor_used"] = False
        targets = exact_donor_targets(
            metacell_lnc[mask],
            activity[mask],
            context_meta,
            ids["lncrna_ids"],
            ids["pathway_ids"],
            universe,
            present_source | present_query,
            min_metacells=settings.min_target_metacells,
        )
        if targets.empty:
            continue
        targets = targets.merge(
            universe,
            on=["lncrna_id", "pathway_id"],
            how="inner",
            validate="many_to_one",
        )
        train, query = _attach_source_features(
            targets,
            source_donors,
            query_donors,
            settings.max_train_donors_per_pair,
        )
        for frame in (train, query, universe):
            frame["dataset_id"] = mapped_receipt["dataset_id"]
            frame["cancer_id"] = cancer_id
            frame["cell_key"] = str(cell_key)
            frame["outer_fold_id"] = fold
            frame["measurement_scale"] = mapped_receipt["measurement_scale"]
        if not train.empty:
            all_train.append(train)
        if not query.empty:
            all_query.append(query)
        all_universe.append(universe)
        context_reports.append(
            {
                "cell_key": str(cell_key),
                "metacells": int(mask.sum()),
                "source_donors_present": len(present_source),
                "query_donors_present": len(present_query),
                "bulk_pairs": int(universe.is_bulk_candidate.sum()),
                "discovery_pairs": int(universe.is_sc_discovery_candidate.sum()),
                "train_rows": len(train),
                "query_rows": len(query),
            }
        )
        atomic_json(
            fold_output / "STATUS.json",
            {
                "target_host": "149",
                "cancer_id": cancer_id,
                "outer_fold_id": fold,
                "stage": "CONTEXT_TARGETS",
                "contexts_done": len(context_reports),
                "last_cell_key": str(cell_key),
                "time": time.time(),
            },
        )
    train_frame = pd.concat(all_train, ignore_index=True) if all_train else pd.DataFrame()
    query_frame = pd.concat(all_query, ignore_index=True) if all_query else pd.DataFrame()
    universe_frame = pd.concat(all_universe, ignore_index=True) if all_universe else pd.DataFrame()
    train_frame.to_parquet(fold_output / "TRAIN_ROWS.parquet", index=False)
    query_frame.to_parquet(fold_output / "QUERY_ROWS.parquet", index=False)
    universe_frame.to_parquet(fold_output / "UNIVERSE.parquet", index=False)
    atomic_json(fold_output / "GEOMETRY_AUDIT.json", geometry)
    atomic_json(fold_output / "CONTEXT_COVERAGE.json", context_reports)
    files = {
        path.name: file_record(path)
        for path in sorted(fold_output.iterdir())
        if path.is_file() and path.name not in {"STATUS.json", "SUCCESS.json"}
    }
    receipt = {
        "format": FORMAT,
        "target_host": "149",
        "state": "FOLD_LOCAL_METACELL_TARGETS_READY",
        "cancer_id": cancer_id,
        "outer_fold_id": fold,
        "source_donors": sorted(source_donors),
        "query_donors": sorted(query_donors),
        "source_donor_hash": canonical_hash(sorted(source_donors)),
        "query_donor_hash": canonical_hash(sorted(query_donors)),
        "limited_replication": bool(folds.limited_replication.any()),
        "metacells": len(metacell_metadata),
        "unassigned_cells": int((assignments < 0).sum()),
        "train_rows": len(train_frame),
        "query_rows": len(query_frame),
        "universe_rows": len(universe_frame),
        "contexts": len(context_reports),
        "target_policy": TARGET_POLICY,
        "aggregation_policy": aggregation_policy,
        "query_donor_used_for_geometry_fit": False,
        "query_donor_used_for_candidate_selection": False,
        "signed_rho_preserved": True,
        "files": files,
        "finished_unix": time.time(),
    }
    atomic_json(success, receipt)
    return receipt


def run_cancer(settings) -> None:
    if socket.gethostname() != "149":
        raise SystemExit("WRONG_HOST")
    if time.time() >= settings.deadline:
        raise SystemExit("DEADLINE_EXPIRED")
    preflight = json.loads((settings.output_root / "LAUNCH_PREFLIGHT.json").read_text())
    if preflight.get("target_host") != "149" or preflight.get("cpu_only") is not True:
        raise SystemExit("INVALID_LAUNCH_PREFLIGHT")
    cancer_id = settings.cancer_id.upper()
    cancer_output = settings.output_root / "cancers" / cancer_id
    cancer_success = cancer_output / "SUCCESS.json"
    if cancer_success.is_file():
        print(json.dumps({"cancer_id": cancer_id, "state": "ALREADY_COMPLETE"}), flush=True)
        return
    cancer_output.mkdir(parents=True, exist_ok=True)
    scratch = settings.scratch_root.resolve() / cancer_id
    if settings.scratch_root.resolve() not in scratch.parents:
        raise RuntimeError("scratch containment failure")
    if scratch.exists() and not (scratch / "MAPPED_SUCCESS.json").is_file():
        shutil.rmtree(scratch)
    mapped = _materialize_mapped_matrices(
        cancer_id, scratch, settings.repair_root, settings.deadline
    )
    metadata = pd.read_parquet(scratch / "CELL_METADATA.parquet", columns=["donor_id"])
    folds = donor_folds(cancer_id, metadata.donor_id, folds=5, seed=settings.seed)
    folds.to_parquet(cancer_output / "DONOR_FOLDS.parquet", index=False)
    receipts = []
    for fold in sorted(folds.outer_fold_id.unique()):
        receipt = _write_fold(
            cancer_id,
            int(fold),
            cancer_output,
            scratch,
            settings.candidates,
            settings,
        )
        receipts.append(receipt)
        print(
            json.dumps(
                {
                    "cancer_id": cancer_id,
                    "outer_fold_id": int(fold),
                    "state": "FOLD_COMPLETE",
                    "query_rows": receipt["query_rows"],
                }
            ),
            flush=True,
        )
    files = {"DONOR_FOLDS.parquet": file_record(cancer_output / "DONOR_FOLDS.parquet")}
    receipt = {
        "format": FORMAT,
        "target_host": "149",
        "state": "CANCER_METACELL_OOF_INPUTS_COMPLETE",
        "cancer_id": cancer_id,
        "dataset_id": mapped["dataset_id"],
        "eligible_cells": mapped["eligible_cells"],
        "donors": mapped["donors"],
        "active_outer_folds": sorted(map(int, folds.outer_fold_id.unique())),
        "limited_replication": bool(folds.limited_replication.any()),
        "fold_receipt_sha256": {
            str(item["outer_fold_id"]): sha256(
                cancer_output / f"outer_fold={item['outer_fold_id']}" / "SUCCESS.json"
            )
            for item in receipts
        },
        "files": files,
        "finished_unix": time.time(),
    }
    atomic_json(cancer_success, receipt)
    if settings.delete_verified_scratch:
        resolved = scratch.resolve()
        root = settings.scratch_root.resolve()
        if root not in resolved.parents or resolved == root:
            raise RuntimeError("refusing unsafe scratch deletion")
        shutil.rmtree(resolved)
    print(json.dumps({"cancer_id": cancer_id, "state": receipt["state"]}), flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--cancer-id", required=True)
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
    result.add_argument("--max-train-donors-per-pair", type=int, default=4)
    result.add_argument("--delete-verified-scratch", action="store_true")
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(TimeoutError("SIGTERM")))
    run_cancer(arguments)
