#!/usr/bin/env python3
"""Fresh, resumable, donor-aware r7 single-cell computation.

The runner reads CSC expression in contiguous cell blocks.  It never writes
cell-level pathway rows and never allocates a complete cells-by-pathways
matrix.  Per-cancer publication is an immutable same-filesystem rename.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32 import single_cell_cell_level as ucell_core  # noqa: E402
from cc_hhgt.v32 import single_cell_r7_streaming as stream_core  # noqa: E402
from cc_hhgt.v32.single_cell_cell_level import (  # noqa: E402
    UCELL_ALGORITHM,
    UCELL_FORMULA,
    UCELL_MISSING_MEMBER_POLICY,
    build_ucell_pathway_contract,
    rank_expression_ucell,
    score_ucell_pathway_block,
)
from cc_hhgt.v32.single_cell_r7_streaming import (  # noqa: E402
    BH_CONSERVATIVE_METHOD,
    INSUFFICIENT_DONOR_REPLICATION,
    AssociationEvidenceChunk,
    StreamingContractError,
    StreamingGroupAccumulator,
    assert_fresh_r7_source,
    atomic_publish_directory,
    classify_compartment,
    estimate_streaming_resources,
    load_checkpoint,
    read_csc_column_block,
    select_smallest_formal_cancer,
    stream_donor_association_chunks,
    write_checkpoint,
)


ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
RUN_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_R7_STREAMING_RUN_V1"
PLAN_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_R7_STREAMING_PLAN_V1"
SUCCESS_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_R7_STREAMING_SUCCESS_V1"
POST_BH_HANDOFF_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_R9_POST_BH_EXEC_HANDOFF_V1"
POST_BH_RELATIVE_PATHS = (
    "lncrna_donor_celltype_summary.parquet",
    "pathway_availability.parquet",
    "pathway_donor_celltype_summary.parquet",
    "association_evidence.parquet",
    "association_lncrna_testability.parquet",
    "association_context_availability.parquet",
    "RESOURCE_ESTIMATE.json",
)
SOURCE_GENERATION = "V3.2_R7_FRESH_FROM_RAW_H5"
FORMAL_COMPARTMENTS = ("malignant", "immune", "stromal")
PARQUET_BATCH_ROWS = 16_384
ASSOCIATION_DUCKDB_MEMORY_LIMIT = os.environ.get(
    "V32_ASSOCIATION_DUCKDB_MEMORY_LIMIT", "64MB"
).strip().upper()
_DUCKDB_MEMORY_MATCH = re.fullmatch(r"([1-9][0-9]{1,3})MB", ASSOCIATION_DUCKDB_MEMORY_LIMIT)
if _DUCKDB_MEMORY_MATCH is None or int(_DUCKDB_MEMORY_MATCH.group(1)) < 64:
    raise RuntimeError(
        "V32_ASSOCIATION_DUCKDB_MEMORY_LIMIT must be an integer MB value >=64MB"
    )
ASSOCIATION_DUCKDB_MEMORY_LIMIT_BYTES = int(_DUCKDB_MEMORY_MATCH.group(1)) * 1024**2
ASSOCIATION_ENGINE = "DUCKDB_EXTERNAL_BH_SORT_PROCESS_ISOLATED_V2"
ASSOCIATION_EXECUTION_STRATEGY = (
    "SERIAL_PER_COMPARTMENT_EXTERNAL_SORT_FRESH_BH_PROCESS_V2"
)
ASSOCIATION_BH_FAMILY = "COMPARTMENT_ORDER"
ASSOCIATION_STAGE_KEY_POLICY = (
    "UNIQUE_COMPARTMENT_LNCRNA_PATHWAY_CARRY_PAYLOAD_NO_PARQUET_ROWID_JOIN"
)
REQUIRED_METADATA = (
    "cell_id",
    "patient_id",
    "cell_type_major",
    "dataset_id",
    "cancer_id",
)
DOUBLETS = (
    "is_doublet",
    "doublet",
    "doublet_flag",
    "qc_doublet",
    "predicted_doublet",
)
AUTHORIZED_OUTPUT_PREFIXES = (
    "./data/CancerLncAtlas/",
    "D:/model/CC_HHGT_v3_2_ranked_subtypes_dev/",
)


class R7StreamingRunError(StreamingContractError):
    """Raised when a formal r7 run cannot prove its contract."""


def _sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise R7StreamingRunError(f"cannot hash absent/unsafe file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise R7StreamingRunError(f"JSON input is absent/unsafe: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _exclusive_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise R7StreamingRunError(f"unsafe JSON temporary exists: {temporary}")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _stable_gene_id(value: object) -> str:
    text = str(value).strip()
    upper = text.upper()
    for prefix in ("GENE:", "PROTEIN:", "LNC:", "LNCRNA:"):
        if upper.startswith(prefix):
            text = text[len(prefix) :]
            break
    return re.sub(r"\.\d+$", "", text)


def _decode(values: Any) -> list[str]:
    return [
        item.decode("utf-8", errors="replace")
        if isinstance(item, (bytes, np.bytes_))
        else str(item)
        for item in values
    ]


def _truthy(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "t", "yes", "y", "doublet"})


def _validate_measurement_block(
    block: sparse.spmatrix, *, measurement_scale: str
) -> None:
    values = np.asarray(block.data, dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise R7StreamingRunError("source expression block contains invalid values")
    if measurement_scale == "raw_counts":
        if not np.allclose(values, np.rint(values), rtol=0.0, atol=1e-8):
            raise R7StreamingRunError("raw-count source contains fractional values")
    elif measurement_scale != "log_normalized":
        raise R7StreamingRunError(f"unsupported measurement scale: {measurement_scale}")


def _authorized_output(path: Path) -> None:
    normalized = str(path.resolve()).replace("\\", "/")
    if not normalized.startswith(AUTHORIZED_OUTPUT_PREFIXES):
        raise R7StreamingRunError(f"output escapes authorized roots: {path}")


def _validate_sha(path: Path, expected: str, role: str) -> str:
    observed = _sha256(path)
    if observed != str(expected).lower():
        raise R7StreamingRunError(f"{role} SHA drift: {observed} != {expected}")
    return observed


def _build_feature_mapping(
    *,
    feature_ids: list[str],
    feature_names: list[str],
    annotation: pd.DataFrame,
    measurement_scale: str,
) -> dict[str, Any]:
    required = {"gene_id", "gene_symbol", "gene_class", "unique_symbol"}
    if missing := sorted(required - set(annotation.columns)):
        raise R7StreamingRunError(f"GENCODE annotation lacks columns: {missing}")
    genes = annotation.loc[:, sorted(required)].copy()
    genes["gene_id"] = genes.gene_id.map(_stable_gene_id)
    if genes.gene_id.duplicated().any():
        raise R7StreamingRunError("GENCODE stable gene IDs are duplicated")
    by_id = genes.set_index("gene_id")
    unique_symbols = genes.loc[genes.unique_symbol.astype(bool)].copy()
    if unique_symbols.gene_symbol.astype(str).duplicated().any():
        raise R7StreamingRunError("GENCODE unique-symbol contract is internally duplicated")
    by_symbol = unique_symbols.set_index("gene_symbol")

    mapped: dict[str, OrderedDict[str, list[int]]] = {
        "protein_coding": OrderedDict(),
        "lncRNA": OrderedDict(),
    }
    symbols: dict[str, str] = {}
    routes = {"DIRECT_STABLE_ID": 0, "UNIQUE_GENCODE_SYMBOL": 0, "UNMAPPED": 0}
    for row_index, (raw_id, raw_name) in enumerate(
        zip(feature_ids, feature_names, strict=True)
    ):
        stable = _stable_gene_id(raw_id)
        if stable in by_id.index:
            record = by_id.loc[stable]
            symbol = str(record["gene_symbol"])
            route = "DIRECT_STABLE_ID"
        elif str(raw_name) in by_symbol.index:
            record = by_symbol.loc[str(raw_name)]
            # ``gene_symbol`` is the index in this branch, so the stable ID
            # must come from the retained annotation column.  Using
            # ``record.name`` here silently changes the identity to a symbol
            # and then makes ``record.gene_symbol`` unavailable.
            stable = str(record["gene_id"])
            symbol = str(record.name)
            route = "UNIQUE_GENCODE_SYMBOL"
        else:
            routes["UNMAPPED"] += 1
            continue
        gene_class = str(record.gene_class)
        if gene_class not in mapped:
            routes["UNMAPPED"] += 1
            continue
        routes[route] += 1
        mapped[gene_class].setdefault(stable, []).append(row_index)
        symbols[stable] = symbol

    duplicate_ids = {
        gene_class: sum(len(rows) > 1 for rows in groups.values())
        for gene_class, groups in mapped.items()
    }
    if str(measurement_scale) != "raw_counts" and any(duplicate_ids.values()):
        raise R7StreamingRunError(
            "normalized source has duplicate mapped stable IDs; summing is undefined"
        )

    def make_map(groups: OrderedDict[str, list[int]]) -> sparse.csr_matrix:
        matrix_rows: list[int] = []
        matrix_columns: list[int] = []
        for unique_row, source_rows in enumerate(groups.values()):
            matrix_rows.extend([unique_row] * len(source_rows))
            matrix_columns.extend(source_rows)
        return sparse.csr_matrix(
            (
                np.ones(len(matrix_rows), dtype=np.float64),
                (
                    np.asarray(matrix_rows, dtype=np.int64),
                    np.asarray(matrix_columns, dtype=np.int64),
                ),
            ),
            shape=(len(groups), len(feature_ids)),
        )

    protein_ids = tuple(mapped["protein_coding"].keys())
    lncrna_ids = tuple(mapped["lncRNA"].keys())
    if not protein_ids or not lncrna_ids:
        raise R7StreamingRunError("feature mapping produced an empty protein/lncRNA universe")
    return {
        "protein_ids": protein_ids,
        "lncrna_ids": lncrna_ids,
        "lncrna_symbols": tuple(symbols[gene] for gene in lncrna_ids),
        "protein_map": make_map(mapped["protein_coding"]),
        "lncrna_map": make_map(mapped["lncRNA"]),
        "mapping_route_counts": routes,
        "duplicate_stable_id_counts": duplicate_ids,
        "duplicate_policy": (
            "SUM_RAW_COUNT_ROWS_BY_STABLE_ID"
            if str(measurement_scale) == "raw_counts"
            else "NO_DUPLICATE_NORMALIZED_ROWS_ALLOWED"
        ),
    }


def _load_contract(args: argparse.Namespace) -> dict[str, Any]:
    r7_root = args.r7_root.resolve()
    preflight_path = args.server_preflight_json.resolve()
    preflight_sha = _validate_sha(
        preflight_path, args.expected_server_preflight_sha256, "server preflight"
    )
    preflight = _load_json(preflight_path)
    if (
        preflight.get("format")
        not in {
            "CC_HHGT_V3_2_SINGLE_CELL_R7_PORTABLE_SERVER_PREFLIGHT_V1",
            "CC_HHGT_V3_2_SINGLE_CELL_R11_RESCUED_SERVER_PREFLIGHT_V1",
        }
        or preflight.get("all_formal_raw_schemas_pass") is not True
        or preflight.get("safe_to_start_fresh_cell_level_compute") is not True
        or preflight.get("heavy_recompute_started") is not False
    ):
        raise R7StreamingRunError("server preflight does not unlock fresh compute")
    if Path(preflight.get("r7_root", "")).resolve() != r7_root:
        raise R7StreamingRunError("server preflight r7 root drift")

    run_status_path = r7_root / "RUN_STATUS.json"
    run_status_sha = _validate_sha(
        run_status_path, args.expected_run_status_sha256, "r7 RUN_STATUS"
    )
    run_status = _load_json(run_status_path)
    formal = set(map(str, run_status.get("formal_eligible_cancers", [])))
    declared_formal_count = int(run_status.get("formal_eligible_cancer_count", len(formal)))
    if (
        not (1 <= len(formal) <= 33)
        or declared_formal_count != len(formal)
        or run_status.get("derived_assets_bound") is not False
    ):
        raise R7StreamingRunError("r7 formal scope/derived-asset gate drift")
    if args.select_smallest_formal:
        selected = select_smallest_formal_cancer(preflight)
        cancer_id = selected["cancer_id"]
    elif args.cancer_id:
        cancer_id = str(args.cancer_id).upper()
    else:
        raise R7StreamingRunError("choose --cancer-id or --select-smallest-formal")
    if cancer_id not in formal:
        raise R7StreamingRunError(f"cancer is not formal in r7: {cancer_id}")
    record = run_status["per_cancer"][cancer_id]
    if record.get("formal_eligible") is not True or record.get("blocking_reason") is not None:
        raise R7StreamingRunError(f"r7 cancer record is not unblocked: {cancer_id}")

    handoff_path = r7_root / "TRAINING_HANDOFF.json"
    handoff = _load_json(handoff_path)
    if handoff.get("historical_assets_relabelled_fresh") is not False:
        raise R7StreamingRunError("r7 handoff permits historical relabelling")
    if any(asset.get("path") is not None for asset in handoff.get("assets", {}).values()):
        raise R7StreamingRunError("r7 handoff unexpectedly binds a derived asset")
    manifest_path = r7_root / "dataset_manifest_33c.parquet"
    _validate_sha(manifest_path, handoff["dataset_manifest_sha256"], "dataset manifest")
    dataset_manifest = pd.read_parquet(manifest_path)
    selected_manifest = dataset_manifest.loc[
        dataset_manifest.cancer_id.astype(str).str.upper().eq(cancer_id)
    ]
    if len(selected_manifest) != 1 or not bool(selected_manifest.iloc[0].formal_eligible):
        raise R7StreamingRunError("dataset manifest formal row is absent/ambiguous")
    manifest_row = selected_manifest.iloc[0]
    measurement_scale = str(manifest_row.measurement_scale)
    if measurement_scale not in {"raw_counts", "log_normalized"}:
        raise R7StreamingRunError(f"unsupported measurement scale: {measurement_scale}")

    input_paths = {
        "h5": Path(record["h5_path"]),
        "metadata": Path(record["metadata_path"]),
        "annotation": Path(record["annotation_path"]),
        "membership": Path(record["exact_membership_path"]),
    }
    for role, path in input_paths.items():
        assert_fresh_r7_source(path, role={"h5": "raw_h5"}.get(role, role))
    input_shas = {
        "h5_sha256": _validate_sha(input_paths["h5"], record["h5_sha256"], "raw H5"),
        "metadata_sha256": _validate_sha(
            input_paths["metadata"], record["metadata_sha256"], "cell metadata"
        ),
        "annotation_sha256": _validate_sha(
            input_paths["annotation"], record["annotation_sha256"], "annotation"
        ),
        "membership_sha256": _validate_sha(
            input_paths["membership"],
            record["exact_membership_sha256"],
            "exact membership",
        ),
    }

    metadata = pd.read_parquet(input_paths["metadata"])
    if missing := sorted(set(REQUIRED_METADATA) - set(metadata.columns)):
        raise R7StreamingRunError(f"metadata lacks columns: {missing}")
    if metadata.cell_id.astype(str).duplicated().any():
        raise R7StreamingRunError("metadata cell IDs are duplicated")
    if set(metadata.cancer_id.astype(str).str.upper()) != {cancer_id}:
        raise R7StreamingRunError("metadata cancer ID drift")
    expected_dataset_id = str(record["dataset_id"])
    metadata_dataset_ids = set(
        metadata.dataset_id.fillna("").astype(str).str.strip()
    )
    if metadata_dataset_ids != {expected_dataset_id}:
        raise R7StreamingRunError("metadata dataset ID drift")
    if "dataset_id" in selected_manifest.columns and str(manifest_row.dataset_id) != expected_dataset_id:
        raise R7StreamingRunError("dataset manifest ID drift")
    doublet_columns = [column for column in DOUBLETS if column in metadata]
    doublet_mask = pd.Series(False, index=metadata.index)
    for column in doublet_columns:
        doublet_mask |= _truthy(metadata[column])

    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - server dependency gate
        raise R7StreamingRunError("h5py is required for fresh r7 streaming") from exc
    with h5py.File(input_paths["h5"], "r") as handle:
        group = handle["matrix"] if "matrix" in handle else handle
        shape = tuple(int(value) for value in np.asarray(group["shape"][:]))
        barcodes = _decode(group["barcodes"][:])
        feature_ids = _decode(group["features"]["id"][:])
        feature_names = _decode(group["features"]["name"][:])
        nnz = int(group["data"].shape[0])
    if shape != (len(feature_ids), len(barcodes)) or len(metadata) != len(barcodes):
        raise R7StreamingRunError("H5/metadata dimensions disagree")
    indexer = pd.Index(metadata.cell_id.astype(str)).get_indexer(barcodes)
    if (indexer < 0).any():
        raise R7StreamingRunError("H5 barcodes are not one-to-one with metadata")
    metadata = metadata.iloc[indexer].reset_index(drop=True)
    doublet_mask = doublet_mask.iloc[indexer].reset_index(drop=True)
    keep_mask = ~doublet_mask.to_numpy(dtype=bool)
    if not keep_mask.any():
        raise R7StreamingRunError("all cells are explicit doublets")
    donor = metadata.patient_id.fillna("").astype(str).str.strip()
    celltype = metadata.cell_type_major.fillna("").astype(str).str.strip()
    if donor.eq("").any() or celltype.eq("").any():
        raise R7StreamingRunError("formal metadata contains blank donor/cell type")
    metadata["compartment"] = celltype.map(classify_compartment)
    kept_metadata = metadata.loc[keep_mask].reset_index(drop=True)
    group_frame = (
        kept_metadata.loc[:, ["patient_id", "cell_type_major", "compartment"]]
        .astype(str)
        .drop_duplicates()
        .sort_values(["patient_id", "cell_type_major"], kind="mergesort")
        .reset_index(drop=True)
    )
    group_lookup = {
        (row.patient_id, row.cell_type_major): index
        for index, row in group_frame.iterrows()
    }
    raw_group_codes = np.full(len(metadata), -1, dtype=np.int64)
    kept_raw_rows = np.flatnonzero(keep_mask)
    raw_group_codes[kept_raw_rows] = np.asarray(
        [
            group_lookup[(str(row.patient_id), str(row.cell_type_major))]
            for row in kept_metadata.itertuples(index=False)
        ],
        dtype=np.int64,
    )

    annotation = pd.read_parquet(input_paths["annotation"])
    mapping = _build_feature_mapping(
        feature_ids=feature_ids,
        feature_names=feature_names,
        annotation=annotation,
        measurement_scale=measurement_scale,
    )
    expected_lnc = int(record["fresh_direct_id_lncrna_feature_count"])
    if len(mapping["lncrna_ids"]) != expected_lnc:
        raise R7StreamingRunError(
            f"fresh lncRNA universe drift: {len(mapping['lncrna_ids'])} != {expected_lnc}"
        )
    membership = pd.read_parquet(input_paths["membership"])
    if not {"pathway_id", "gene_id"}.issubset(membership.columns):
        raise R7StreamingRunError("exact membership schema drift")
    membership = membership.loc[:, ["pathway_id", "gene_id"]].copy()
    membership["pathway_id"] = membership.pathway_id.astype(str)
    membership["gene_id"] = membership.gene_id.map(_stable_gene_id)
    membership = membership.drop_duplicates().reset_index(drop=True)
    if membership.pathway_id.nunique() != 2135:
        raise R7StreamingRunError("exact membership is not 2,135 pathways")
    pathway_contract = build_ucell_pathway_contract(
        membership,
        protein_gene_ids=mapping["protein_ids"],
        max_rank=args.max_rank,
    )
    if not pathway_contract.available_pathway_ids:
        raise R7StreamingRunError("no exact pathway is numerically available")

    resource = estimate_streaming_resources(
        cells=shape[1],
        features=shape[0],
        protein_genes=len(mapping["protein_ids"]),
        lncrnas=len(mapping["lncrna_ids"]),
        pathways=2135,
        donor_celltype_groups=len(group_frame),
        nnz=nnz,
        chunk_cells=args.chunk_cells,
        association_pathway_block=args.association_pathway_block,
        duckdb_external_sort_budget_bytes=ASSOCIATION_DUCKDB_MEMORY_LIMIT_BYTES,
    )
    code_shas = {
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "streaming_core_sha256": _sha256(Path(stream_core.__file__).resolve()),
        "ucell_core_sha256": _sha256(Path(ucell_core.__file__).resolve()),
    }
    contract_payload = {
        "format": RUN_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "source_generation": SOURCE_GENERATION,
        "cancer_id": cancer_id,
        "dataset_id": str(record["dataset_id"]),
        "measurement_scale": measurement_scale,
        "server_preflight_sha256": preflight_sha,
        "run_status_sha256": run_status_sha,
        "input_shas": input_shas,
        "full_input_sha256_verified_before_matrix_access": True,
        "raw_h5_full_sha256_verified": True,
        "code_shas": code_shas,
        "max_rank": int(args.max_rank),
        "chunk_cells": int(args.chunk_cells),
        "checkpoint_every_chunks": int(args.checkpoint_every_chunks),
        "min_donors": int(args.min_donors),
        "min_cells_per_donor_context": int(args.min_cells_per_donor_context),
        "min_lncrna_detect_rate": float(args.min_lncrna_detect_rate),
        "min_lncrna_detecting_donors": int(args.min_lncrna_detecting_donors),
        "min_abs_rho": float(args.min_abs_rho),
        "max_nominal_p": float(args.max_nominal_p),
        "association_pathway_block": int(args.association_pathway_block),
        "association_engine": ASSOCIATION_ENGINE,
        "association_execution_strategy": ASSOCIATION_EXECUTION_STRATEGY,
        "association_bh_family": ASSOCIATION_BH_FAMILY,
        "association_stage_key_policy": ASSOCIATION_STAGE_KEY_POLICY,
        "association_duckdb_memory_limit": ASSOCIATION_DUCKDB_MEMORY_LIMIT,
        "parquet_batch_rows": PARQUET_BATCH_ROWS,
        "association_evidence_accumulated_in_python": False,
        "feature_mapping_policy": (
            "STABLE_ENSEMBL_ID_THEN_UNIQUE_GENCODE_SYMBOL"
        ),
        "gene_id_normalization_policy": (
            "STRIP_KNOWN_TYPE_PREFIX_THEN_TERMINAL_NUMERIC_VERSION_ONLY"
        ),
        "duplicate_policy": mapping["duplicate_policy"],
        "compartment_policy": "EXPLICIT_LABEL_MAP_UNKNOWN_TO_OTHER_UNRESOLVED",
        "biological_replicate": "DONOR",
        "cell_as_independent_replicate": False,
        "cell_level_pathway_matrix_persisted": False,
        "historical_checkpoints_used": False,
        "historical_rankings_used": False,
        "historical_predictions_used": False,
        "historical_sc_trajectory_used": False,
    }
    contract_sha = _canonical_sha(contract_payload)
    return {
        "contract": contract_payload,
        "contract_sha256": contract_sha,
        "cancer_id": cancer_id,
        "dataset_id": str(record["dataset_id"]),
        "measurement_scale": measurement_scale,
        "input_paths": input_paths,
        "input_shas": input_shas,
        "shape": shape,
        "nnz": nnz,
        "metadata": metadata,
        "keep_mask": keep_mask,
        "raw_group_codes": raw_group_codes,
        "group_frame": group_frame,
        "doublet_columns": doublet_columns,
        "explicit_doublets_excluded": int(doublet_mask.sum()),
        "mapping": mapping,
        "pathway_contract": pathway_contract,
        "resource_estimate": resource,
        "preflight_path": str(preflight_path),
        "preflight_sha256": preflight_sha,
        "run_status_path": str(run_status_path),
        "run_status_sha256": run_status_sha,
        "code_shas": code_shas,
    }


def _plan_summary(bundle: dict[str, Any]) -> dict[str, Any]:
    contract = bundle["pathway_contract"]
    metadata = bundle["metadata"]
    selected = metadata.loc[bundle["keep_mask"]]
    return {
        "format": PLAN_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "source_generation": SOURCE_GENERATION,
        "cancer_id": bundle["cancer_id"],
        "dataset_id": bundle["dataset_id"],
        "contract_sha256": bundle["contract_sha256"],
        "raw_cells": int(bundle["shape"][1]),
        "kept_cells": int(len(selected)),
        "explicit_doublets_excluded": bundle["explicit_doublets_excluded"],
        "donors": int(selected.patient_id.astype(str).nunique()),
        "cell_types": int(selected.cell_type_major.astype(str).nunique()),
        "donor_celltype_groups": int(len(bundle["group_frame"])),
        "compartment_cell_counts": selected.compartment.value_counts()
        .sort_index()
        .astype(int)
        .to_dict(),
        "protein_rank_genes": len(bundle["mapping"]["protein_ids"]),
        "lncrnas": len(bundle["mapping"]["lncrna_ids"]),
        "pathways_total": int(len(contract.availability)),
        "pathways_available": int(contract.availability.ucell_available.sum()),
        "pathways_typed_unavailable": int((~contract.availability.ucell_available).sum()),
        "mapping_route_counts": bundle["mapping"]["mapping_route_counts"],
        "duplicate_stable_id_counts": bundle["mapping"]["duplicate_stable_id_counts"],
        "resource_estimate": bundle["resource_estimate"],
        "full_input_sha256_verified_before_matrix_access": True,
        "raw_h5_full_sha256_verified": True,
        "no_cell_level_pathway_output": True,
        "donor_is_biological_replicate": True,
        "safe_to_start_pilot": (
            bundle["resource_estimate"]["estimated_peak_ram_bytes"]
            <= 512 * 1024**2
        ),
        "production_deployed": False,
    }


def _normalize_lnc_block(
    lnc: sparse.spmatrix,
    full_block: sparse.spmatrix,
    *,
    measurement_scale: str,
) -> sparse.csc_matrix:
    values = sparse.csc_matrix(lnc, dtype=np.float64)
    values.eliminate_zeros()
    if measurement_scale == "raw_counts":
        library = np.asarray(full_block.sum(axis=0), dtype=np.float64).reshape(-1)
        if (library <= 0).any():
            raise R7StreamingRunError("raw-count cell has non-positive library size")
        values = sparse.csc_matrix(values @ sparse.diags(10_000.0 / library))
        values.data = np.log1p(values.data)
    elif measurement_scale != "log_normalized":
        raise R7StreamingRunError(f"unsupported measurement scale: {measurement_scale}")
    return values


def _acquire_lock(work: Path, contract_sha256: str) -> Path:
    lock = work / "RUNNING.json"
    if lock.exists():
        previous = _load_json(lock)
        same_host = previous.get("hostname") == socket.gethostname()
        process_id = int(previous.get("process_id", -1))
        active = False
        if same_host and process_id > 0:
            try:
                os.kill(process_id, 0)
                active = True
            except OSError:
                active = False
        if active:
            raise R7StreamingRunError(f"same contract is already running: pid={process_id}")
        stale_sha = _sha256(lock)
        stale = work / f"STALE_LOCK.{stale_sha}.json"
        if stale.exists():
            raise R7StreamingRunError(f"stale lock archive already exists: {stale}")
        os.rename(lock, stale)
    _exclusive_json(
        lock,
        {
            "format": "CC_HHGT_V3_2_SINGLE_CELL_R7_STREAM_LOCK_V1",
            "contract_sha256": contract_sha256,
            "hostname": socket.gethostname(),
            "process_id": os.getpid(),
        },
    )
    return lock


def _run_stream(bundle: dict[str, Any], args: argparse.Namespace) -> tuple[StreamingGroupAccumulator, Path]:
    output_parent = args.output_parent.resolve()
    _authorized_output(output_parent)
    output_parent.mkdir(parents=True, exist_ok=True)
    cancer = bundle["cancer_id"]
    final = output_parent / f"cancer_id={cancer}"
    if final.exists() or final.is_symlink():
        success = final / "SUCCESS.json"
        if not args.resume or not success.is_file() or success.is_symlink():
            raise R7StreamingRunError(f"immutable final output already exists: {final}")
        payload = _load_json(success)
        if payload.get("contract_sha256") != bundle["contract_sha256"]:
            raise R7StreamingRunError("completed output contract differs from requested run")
        raise R7StreamingRunError("completed output already exists and is verified")
    work = output_parent / "_work" / f"cancer_id={cancer}" / bundle["contract_sha256"]
    if work.exists() and not args.resume:
        raise R7StreamingRunError(f"checkpoint exists; explicit --resume required: {work}")
    work.mkdir(parents=True, exist_ok=True)
    contract_path = work / "RUN_CONTRACT.json"
    if contract_path.exists():
        if _load_json(contract_path) != bundle["contract"]:
            raise R7StreamingRunError("work directory contract drift")
    else:
        _exclusive_json(contract_path, bundle["contract"])
    lock = _acquire_lock(work, bundle["contract_sha256"])
    contract = bundle["pathway_contract"]
    accumulator = StreamingGroupAccumulator(
        lncrna_count=len(bundle["mapping"]["lncrna_ids"]),
        pathway_count=len(contract.available_pathway_ids),
        group_count=len(bundle["group_frame"]),
    )
    next_cell = 0
    state_path = work / "CHECKPOINT_STATE.json"
    if state_path.exists():
        if not args.resume:
            raise R7StreamingRunError("checkpoint exists without --resume")
        state, arrays = load_checkpoint(
            work, expected_contract_sha256=bundle["contract_sha256"]
        )
        if int(state["total_cells"]) != int(bundle["shape"][1]):
            raise R7StreamingRunError("checkpoint total-cell count drift")
        accumulator.restore(arrays)
        next_cell = int(state["next_cell"])
    try:
        import h5py
        with h5py.File(bundle["input_paths"]["h5"], "r") as handle:
            group = handle["matrix"] if "matrix" in handle else handle
            chunks_since_checkpoint = 0
            for start in range(next_cell, bundle["shape"][1], args.chunk_cells):
                stop = min(start + args.chunk_cells, bundle["shape"][1])
                block = read_csc_column_block(
                    data=group["data"],
                    indices=group["indices"],
                    indptr=group["indptr"],
                    shape=bundle["shape"],
                    start=start,
                    stop=stop,
                )
                keep = bundle["keep_mask"][start:stop]
                if keep.any():
                    kept_block = sparse.csc_matrix(block[:, keep])
                    _validate_measurement_block(
                        kept_block, measurement_scale=bundle["measurement_scale"]
                    )
                    protein = sparse.csc_matrix(
                        bundle["mapping"]["protein_map"] @ kept_block
                    ).toarray()
                    ranks = rank_expression_ucell(protein, max_rank=args.max_rank)
                    scores = score_ucell_pathway_block(
                        ranks,
                        signature_matrix=contract.signature_matrix,
                        signature_gene_counts=contract.signature_gene_counts,
                        missing_gene_counts=contract.missing_gene_counts,
                        max_rank=args.max_rank,
                    )
                    lnc = sparse.csc_matrix(bundle["mapping"]["lncrna_map"] @ kept_block)
                    lnc = _normalize_lnc_block(
                        lnc,
                        kept_block,
                        measurement_scale=bundle["measurement_scale"],
                    )
                    group_codes = bundle["raw_group_codes"][start:stop][keep]
                    accumulator.update(lnc, scores, group_codes)
                chunks_since_checkpoint += 1
                if (
                    chunks_since_checkpoint >= args.checkpoint_every_chunks
                    or stop == bundle["shape"][1]
                ):
                    write_checkpoint(
                        work,
                        contract_sha256=bundle["contract_sha256"],
                        next_cell=stop,
                        total_cells=bundle["shape"][1],
                        arrays=accumulator.state_arrays(),
                    )
                    chunks_since_checkpoint = 0
        if int(accumulator.cell_counts.sum()) != int(bundle["keep_mask"].sum()):
            raise R7StreamingRunError("streamed kept-cell count does not match metadata")
        return accumulator, work
    finally:
        if lock.exists() and not lock.is_symlink():
            lock.unlink()


def _collapse_donor_compartment(
    *,
    group_frame: pd.DataFrame,
    group_indices: np.ndarray,
    cell_counts: np.ndarray,
    lnc_expression_sums: np.ndarray,
    lnc_detect_counts: np.ndarray,
    pathway_activity_sums: np.ndarray,
    min_cells: int,
) -> dict[str, Any]:
    subset = group_frame.iloc[group_indices]
    donors = sorted(subset.patient_id.astype(str).unique())
    donor_cells: list[int] = []
    lnc_means: list[np.ndarray] = []
    lnc_detect: list[np.ndarray] = []
    pathway_means: list[np.ndarray] = []
    kept_donors: list[str] = []
    for donor in donors:
        local = group_indices[subset.patient_id.astype(str).to_numpy() == donor]
        count = int(cell_counts[local].sum())
        if count < int(min_cells):
            continue
        kept_donors.append(donor)
        donor_cells.append(count)
        lnc_means.append(lnc_expression_sums[:, local].sum(axis=1) / count)
        lnc_detect.append(lnc_detect_counts[:, local].sum(axis=1))
        pathway_means.append(pathway_activity_sums[:, local].sum(axis=1) / count)
    n_lnc = lnc_expression_sums.shape[0]
    n_pathway = pathway_activity_sums.shape[0]
    return {
        "donors": kept_donors,
        "cell_counts": np.asarray(donor_cells, dtype=np.int64),
        "lncrna_means": (
            np.vstack(lnc_means) if lnc_means else np.empty((0, n_lnc), dtype=float)
        ),
        "lncrna_detect_counts": (
            np.vstack(lnc_detect)
            if lnc_detect
            else np.empty((0, n_lnc), dtype=np.int64)
        ),
        "pathway_means": (
            np.vstack(pathway_means)
            if pathway_means
            else np.empty((0, n_pathway), dtype=float)
        ),
        "donors_before_min_cells": len(donors),
    }


def _write_manifest(root: Path, relative_paths: list[str]) -> tuple[Path, str]:
    rows = []
    for relative in sorted(relative_paths):
        path = root / relative
        rows.append(
            {
                "relative_path": relative,
                "bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
        )
    manifest = root / "FILE_MANIFEST.parquet"
    pd.DataFrame(rows).to_parquet(manifest, index=False, compression="zstd")
    return manifest, _sha256(manifest)


def _release_transient_memory() -> None:
    """Return Python and Arrow batch allocations between materialization stages."""

    gc.collect()
    try:
        import pyarrow as pa

        pa.default_memory_pool().release_unused()
    except (ImportError, AttributeError):
        pass
    if sys.platform.startswith("linux"):
        try:
            import ctypes

            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim.argtypes = [ctypes.c_size_t]
            libc.malloc_trim.restype = ctypes.c_int
            libc.malloc_trim(0)
        except (ImportError, OSError, AttributeError):
            pass


def _memory_trace(stage: str, **extra: Any) -> None:
    """Emit auditable process RSS/HWM milestones without retaining a trace table."""

    values: dict[str, Any] = {"stage": str(stage), **extra}
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith(("VmRSS:", "VmHWM:")):
                key, raw = line.split(":", 1)
                values[f"{key.lower()}_bytes"] = int(raw.strip().split()[0]) * 1024
    print(
        "R7_MEMORY "
        + json.dumps(values, ensure_ascii=False, sort_keys=True, default=str),
        flush=True,
    )


def _lnc_summary_schema() -> Any:
    import pyarrow as pa

    return pa.schema(
        [
            ("dataset_id", pa.string()),
            ("cancer_id", pa.string()),
            ("lncrna_id", pa.string()),
            ("lncrna_symbol", pa.string()),
            ("patient_id", pa.string()),
            ("cell_type_major", pa.string()),
            ("compartment", pa.string()),
            ("cell_count", pa.int64()),
            ("detected_cell_count", pa.int64()),
            ("detect_rate", pa.float64()),
            ("mean_expression", pa.float64()),
            ("expression_summary_scale", pa.string()),
            ("source_generation", pa.string()),
        ]
    )


def _pathway_summary_schema() -> Any:
    import pyarrow as pa

    return pa.schema(
        [
            ("dataset_id", pa.string()),
            ("cancer_id", pa.string()),
            ("pathway_id", pa.string()),
            ("patient_id", pa.string()),
            ("cell_type_major", pa.string()),
            ("compartment", pa.string()),
            ("cell_count", pa.int64()),
            ("ucell_available", pa.bool_()),
            ("unavailable_reason", pa.string()),
            ("ucell_score_mean", pa.float32()),
            ("biological_aggregation_unit", pa.string()),
            ("source_generation", pa.string()),
        ]
    )


def _raw_evidence_schema() -> Any:
    import pyarrow as pa

    return pa.schema(
        [
            ("dataset_id", pa.string()),
            ("cancer_id", pa.string()),
            ("compartment_order", pa.int8()),
            ("compartment", pa.string()),
            ("lncrna_id", pa.string()),
            ("lncrna_symbol", pa.string()),
            ("pathway_id", pa.string()),
            ("n_donors", pa.int64()),
            ("spearman_rho", pa.float64()),
            ("nominal_p", pa.float64()),
            ("total_tests", pa.int64()),
        ]
    )


def _testability_schema() -> Any:
    import pyarrow as pa

    return pa.schema(
        [
            ("dataset_id", pa.string()),
            ("cancer_id", pa.string()),
            ("compartment", pa.string()),
            ("lncrna_id", pa.string()),
            ("lncrna_symbol", pa.string()),
            ("eligible_donor_count", pa.int64()),
            ("detecting_donor_count", pa.int64()),
            ("context_detect_rate", pa.float64()),
            ("minimum_context_detect_rate", pa.float64()),
            ("minimum_detecting_donors", pa.int64()),
            ("biological_unit", pa.string()),
            ("cell_as_independent_replicate", pa.bool_()),
            ("source_generation", pa.string()),
            ("test_status", pa.string()),
            ("unavailable_reason", pa.string()),
        ]
    )


def _evidence_schema() -> Any:
    import pyarrow as pa

    return pa.schema(
        [
            ("dataset_id", pa.string()),
            ("cancer_id", pa.string()),
            ("compartment", pa.string()),
            ("lncrna_id", pa.string()),
            ("lncrna_symbol", pa.string()),
            ("pathway_id", pa.string()),
            ("n_donors", pa.int64()),
            ("spearman_rho", pa.float64()),
            ("nominal_p", pa.float64()),
            ("bh_q_global_tests", pa.float64()),
            ("fdr_0_10_pass", pa.bool_()),
            ("multiple_testing_adjustment", pa.string()),
            ("bh_q_is_conservative_upper_bound", pa.bool_()),
            ("biological_unit", pa.string()),
            ("cell_as_independent_replicate", pa.bool_()),
            ("evidence_scope", pa.string()),
            ("source_generation", pa.string()),
        ]
    )


class _ParquetFrameWriter:
    """Write bounded pandas frames under one caller-supplied Arrow schema."""

    def __init__(self, path: Path, *, schema: Any) -> None:
        import pyarrow.parquet as pq

        self.path = path
        self.schema = schema
        self.writer: Any | None = pq.ParquetWriter(
            self.path,
            self.schema,
            compression="zstd",
            use_dictionary=True,
        )
        self.rows = 0

    def write(self, frame: pd.DataFrame) -> None:
        import pyarrow as pa

        if self.writer is None:
            raise R7StreamingRunError("cannot write to a closed Parquet writer")
        expected_columns = self.schema.names
        observed_columns = list(frame.columns)
        if observed_columns != expected_columns:
            raise R7StreamingRunError(
                "Parquet batch columns differ from explicit schema: "
                f"{observed_columns} != {expected_columns}"
            )
        table = pa.Table.from_pandas(
            frame,
            schema=self.schema,
            preserve_index=False,
            safe=True,
        )
        # ``from_pandas`` may add pandas-only metadata; field names and Arrow
        # types must remain exactly the caller-supplied storage contract.
        if not table.schema.equals(self.schema, check_metadata=False):
            raise R7StreamingRunError("Arrow did not preserve the explicit schema")
        if table.num_rows:
            self.writer.write_table(table, row_group_size=PARQUET_BATCH_ROWS)
            self.rows += int(table.num_rows)
        del table

    def close(self) -> int:
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        _release_transient_memory()
        return self.rows


def _sql_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _finalize_association_evidence_in_process(
    *,
    raw_path: Path,
    output_path: Path,
    scratch: Path,
    raw_rows: int,
) -> int:
    """Apply per-compartment conservative BH with serial bounded spill.

    This implementation is invoked in a fresh helper process by the public
    wrapper below.  Keeping DuckDB and Arrow allocator state out of the
    expression-accumulation process prevents retained arenas from stacking
    across the two memory-intensive phases.
    """

    if int(raw_rows) == 0:
        _ParquetFrameWriter(output_path, schema=_evidence_schema()).close()
        if raw_path.exists() and not raw_path.is_symlink():
            raw_path.unlink()
        return 0
    if raw_path.is_symlink() or not raw_path.is_file():
        raise R7StreamingRunError("association evidence spool is absent or unsafe")
    if scratch.exists() or scratch.is_symlink():
        raise R7StreamingRunError("association DuckDB scratch already exists")
    scratch.mkdir()
    import duckdb
    import pyarrow as pa
    import pyarrow.parquet as pq

    raw = _sql_literal(raw_path)
    _memory_trace("BH_HELPER_METADATA_START", raw_rows=int(raw_rows))
    metadata_temp = scratch / "metadata_temp"
    metadata_temp.mkdir()
    connection = duckdb.connect()
    try:
        connection.execute(
            f"SET memory_limit={_sql_literal(ASSOCIATION_DUCKDB_MEMORY_LIMIT)}"
        )
        connection.execute("SET threads=1")
        connection.execute("SET preserve_insertion_order=false")
        connection.execute(f"SET temp_directory={_sql_literal(metadata_temp)}")
        families = connection.execute(
            f"""
            SELECT
                CAST(compartment_order AS INTEGER),
                count(*),
                count(DISTINCT (lncrna_id, pathway_id)),
                min(total_tests),
                max(total_tests),
                min(dataset_id),
                max(dataset_id),
                min(cancer_id),
                max(cancer_id),
                min(compartment),
                max(compartment),
                min(n_donors),
                max(n_donors)
            FROM read_parquet({raw})
            GROUP BY compartment_order
            ORDER BY compartment_order
            """
        ).fetchall()
    finally:
        connection.close()
    if not families:
        raise R7StreamingRunError("non-empty association spool has no BH family")
    if sum(int(row[1]) for row in families) != int(raw_rows):
        raise R7StreamingRunError("association BH family row count drift")
    _memory_trace(
        "BH_HELPER_METADATA_COMPLETE",
        raw_rows=int(raw_rows),
        bh_family_count=len(families),
    )

    def execute_copy(query: str, temporary: Path) -> None:
        temporary.mkdir()
        stage_connection = duckdb.connect()
        try:
            stage_connection.execute(
                f"SET memory_limit={_sql_literal(ASSOCIATION_DUCKDB_MEMORY_LIMIT)}"
            )
            stage_connection.execute("SET threads=1")
            stage_connection.execute("SET preserve_insertion_order=false")
            stage_connection.execute(
                f"SET temp_directory={_sql_literal(temporary)}"
            )
            stage_connection.execute(query)
        finally:
            stage_connection.close()
        _release_transient_memory()

    def parquet_rows(path: Path) -> int:
        parquet = pq.ParquetFile(path)
        try:
            return int(parquet.metadata.num_rows)
        finally:
            parquet.close()

    def external_sort_adjusted(
        *,
        adjusted_path: Path,
        part_path: Path,
        run_root: Path,
        dataset_id: str,
        cancer_id: str,
        compartment: str,
        n_donors: int,
    ) -> int:
        """External merge-sort one adjusted BH family with bounded Arrow runs."""

        import heapq

        run_root.mkdir()
        run_paths: list[Path] = []
        adjusted_parquet = pq.ParquetFile(adjusted_path)
        try:
            for run_index, batch in enumerate(
                adjusted_parquet.iter_batches(batch_size=PARQUET_BATCH_ROWS)
            ):
                table = pa.Table.from_batches([batch])
                sorted_table = table.sort_by(
                    [
                        ("bh_q_global_tests", "ascending"),
                        ("nominal_p", "ascending"),
                        ("lncrna_id", "ascending"),
                        ("pathway_id", "ascending"),
                    ]
                )
                run_path = run_root / f"run_{run_index:06d}.parquet"
                pq.write_table(
                    sorted_table,
                    run_path,
                    compression="zstd",
                    use_dictionary=True,
                    row_group_size=PARQUET_BATCH_ROWS,
                )
                run_paths.append(run_path)
                del sorted_table, table, batch
        finally:
            adjusted_parquet.close()
        if not run_paths:
            raise R7StreamingRunError("non-empty BH family produced no merge run")

        value_columns = (
            "lncrna_id",
            "lncrna_symbol",
            "pathway_id",
            "spearman_rho",
            "nominal_p",
            "bh_q_global_tests",
        )

        def rows(path: Path):
            parquet = pq.ParquetFile(path)
            try:
                for batch in parquet.iter_batches(
                    batch_size=1024,
                    columns=list(value_columns),
                ):
                    columns = [column.to_pylist() for column in batch.columns]
                    for row_index in range(batch.num_rows):
                        yield tuple(column[row_index] for column in columns)
                    del columns, batch
            finally:
                parquet.close()

        iterators = [iter(rows(path)) for path in run_paths]
        heap: list[tuple[Any, ...]] = []
        for run_index, iterator in enumerate(iterators):
            try:
                row = next(iterator)
            except StopIteration:
                continue
            heapq.heappush(
                heap,
                (row[5], row[4], row[0], row[2], run_index, row),
            )

        evidence_schema = _evidence_schema()
        buffers: dict[str, list[Any]] = {
            name: [] for name in evidence_schema.names
        }
        writer = pq.ParquetWriter(
            part_path,
            evidence_schema,
            compression="zstd",
            use_dictionary=True,
        )
        observed_rows = 0

        def flush() -> None:
            nonlocal buffers
            if not buffers["lncrna_id"]:
                return
            table = pa.Table.from_pydict(buffers, schema=evidence_schema)
            writer.write_table(table, row_group_size=PARQUET_BATCH_ROWS)
            del table
            buffers = {name: [] for name in evidence_schema.names}

        try:
            while heap:
                _, _, _, _, run_index, row = heapq.heappop(heap)
                lncrna_id, lncrna_symbol, pathway_id, rho, nominal_p, bh_q = row
                buffers["dataset_id"].append(dataset_id)
                buffers["cancer_id"].append(cancer_id)
                buffers["compartment"].append(compartment)
                buffers["lncrna_id"].append(lncrna_id)
                buffers["lncrna_symbol"].append(lncrna_symbol)
                buffers["pathway_id"].append(pathway_id)
                buffers["n_donors"].append(n_donors)
                buffers["spearman_rho"].append(rho)
                buffers["nominal_p"].append(nominal_p)
                buffers["bh_q_global_tests"].append(bh_q)
                buffers["fdr_0_10_pass"].append(bh_q <= 0.10)
                buffers["multiple_testing_adjustment"].append(
                    BH_CONSERVATIVE_METHOD
                )
                buffers["bh_q_is_conservative_upper_bound"].append(True)
                buffers["biological_unit"].append("DONOR")
                buffers["cell_as_independent_replicate"].append(False)
                buffers["evidence_scope"].append(
                    "EXPLORATORY_DONOR_LEVEL_SPEARMAN_SCREEN"
                )
                buffers["source_generation"].append(SOURCE_GENERATION)
                observed_rows += 1
                if len(buffers["lncrna_id"]) >= PARQUET_BATCH_ROWS:
                    flush()
                try:
                    next_row = next(iterators[run_index])
                except StopIteration:
                    continue
                heapq.heappush(
                    heap,
                    (
                        next_row[5],
                        next_row[4],
                        next_row[0],
                        next_row[2],
                        run_index,
                        next_row,
                    ),
                )
            flush()
        finally:
            writer.close()
            for iterator in iterators:
                iterator.close()
        _release_transient_memory()
        return observed_rows

    part_paths: list[Path] = []
    for (
        compartment_order,
        family_rows,
        exact_key_count,
        minimum_tests,
        maximum_tests,
        minimum_dataset,
        maximum_dataset,
        minimum_cancer,
        maximum_cancer,
        minimum_compartment,
        maximum_compartment,
        minimum_donors,
        maximum_donors,
    ) in families:
        compartment_order = int(compartment_order)
        if compartment_order < 0 or compartment_order >= len(FORMAL_COMPARTMENTS):
            raise R7StreamingRunError("association compartment order is invalid")
        if int(exact_key_count) != int(family_rows):
            raise R7StreamingRunError(
                "association raw exact key is not unique within compartment"
            )
        if (
            minimum_tests is None
            or int(minimum_tests) <= 0
            or int(minimum_tests) != int(maximum_tests)
        ):
            raise R7StreamingRunError(
                "association multiple-testing denominator varies within compartment"
            )
        if (
            minimum_dataset != maximum_dataset
            or minimum_cancer != maximum_cancer
            or minimum_compartment != maximum_compartment
            or int(minimum_donors) != int(maximum_donors)
        ):
            raise R7StreamingRunError(
                "association family metadata varies within compartment"
            )
        rank_path = scratch / f"compartment_order={compartment_order}.rank.parquet"
        adjusted_path = (
            scratch / f"compartment_order={compartment_order}.adjusted.parquet"
        )
        part_path = scratch / f"compartment_order={compartment_order}.parquet"
        rank = _sql_literal(rank_path)
        adjusted = _sql_literal(adjusted_path)
        part = _sql_literal(part_path)
        rank_query = f"""
            COPY (
                SELECT
                    CAST(lncrna_id AS VARCHAR) AS lncrna_id,
                    CAST(lncrna_symbol AS VARCHAR) AS lncrna_symbol,
                    CAST(pathway_id AS VARCHAR) AS pathway_id,
                    CAST(spearman_rho AS DOUBLE) AS spearman_rho,
                    CAST(nominal_p AS DOUBLE) AS nominal_p,
                    CAST(total_tests AS BIGINT) AS total_tests,
                    row_number() OVER (
                        ORDER BY nominal_p, lncrna_id, pathway_id
                    ) AS retained_rank
                FROM read_parquet({raw})
                WHERE compartment_order = {compartment_order}
            ) TO {rank} (
                FORMAT PARQUET,
                COMPRESSION ZSTD,
                ROW_GROUP_SIZE {PARQUET_BATCH_ROWS}
            )
        """
        _memory_trace(
            "BH_HELPER_RANK_START",
            compartment_order=compartment_order,
            family_rows=int(family_rows),
        )
        execute_copy(
            rank_query,
            scratch / f"duckdb_temp_{compartment_order}_rank",
        )
        if parquet_rows(rank_path) != int(family_rows):
            raise R7StreamingRunError("association rank-stage row count drift")
        _memory_trace(
            "BH_HELPER_RANK_COMPLETE",
            compartment_order=compartment_order,
            family_rows=int(family_rows),
        )

        adjusted_query = f"""
            COPY (
                SELECT *, LEAST(
                    1.0,
                    MIN(nominal_p * total_tests / retained_rank) OVER (
                        ORDER BY retained_rank DESC
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                    )
                ) AS bh_q_global_tests
                FROM read_parquet({rank})
            ) TO {adjusted} (
                FORMAT PARQUET,
                COMPRESSION ZSTD,
                ROW_GROUP_SIZE {PARQUET_BATCH_ROWS}
            )
        """
        _memory_trace(
            "BH_HELPER_ADJUST_START",
            compartment_order=compartment_order,
            family_rows=int(family_rows),
        )
        execute_copy(
            adjusted_query,
            scratch / f"duckdb_temp_{compartment_order}_adjusted",
        )
        if parquet_rows(adjusted_path) != int(family_rows):
            raise R7StreamingRunError("association adjusted-stage row count drift")
        _memory_trace(
            "BH_HELPER_ADJUST_COMPLETE",
            compartment_order=compartment_order,
            family_rows=int(family_rows),
        )

        _memory_trace(
            "BH_HELPER_FINAL_SORT_START",
            compartment_order=compartment_order,
            family_rows=int(family_rows),
        )
        observed_part = external_sort_adjusted(
            adjusted_path=adjusted_path,
            part_path=part_path,
            run_root=scratch / f"merge_runs_{compartment_order}",
            dataset_id=str(minimum_dataset),
            cancer_id=str(minimum_cancer),
            compartment=str(minimum_compartment),
            n_donors=int(minimum_donors),
        )
        if observed_part != int(family_rows):
            raise R7StreamingRunError(
                "association evidence compartment row count drift"
            )
        part_paths.append(part_path)
        _release_transient_memory()
        _memory_trace(
            "BH_HELPER_FINAL_SORT_COMPLETE",
            compartment_order=compartment_order,
            family_rows=int(family_rows),
        )

    evidence_schema = _evidence_schema()
    output_writer = pq.ParquetWriter(
        output_path,
        evidence_schema,
        compression="zstd",
        use_dictionary=True,
    )
    observed = 0
    try:
        for part_path in part_paths:
            parquet = pq.ParquetFile(part_path)
            try:
                for batch in parquet.iter_batches(batch_size=PARQUET_BATCH_ROWS):
                    table = pa.Table.from_batches([batch])
                    if not table.schema.equals(evidence_schema, check_metadata=False):
                        table = table.cast(evidence_schema, safe=True)
                    output_writer.write_table(
                        table, row_group_size=PARQUET_BATCH_ROWS
                    )
                    observed += int(table.num_rows)
                    del table, batch
            finally:
                parquet.close()
                del parquet
    finally:
        output_writer.close()
    if observed != int(raw_rows):
        raise R7StreamingRunError(
            f"association evidence row drift: {observed} != {raw_rows}"
        )
    raw_path.unlink()
    shutil.rmtree(scratch)
    _release_transient_memory()
    return observed


def _finalize_association_evidence(
    *,
    raw_path: Path,
    output_path: Path,
    scratch: Path,
    raw_rows: int,
) -> int:
    """Expose the exact BH implementation to deterministic equivalence tests.

    Formal runs never call this wrapper from the expression process; they use
    the SHA-bound os.execve handoff and call the implementation only after the
    old process image has been replaced.
    """

    return _finalize_association_evidence_in_process(
        raw_path=raw_path,
        output_path=output_path,
        scratch=scratch,
        raw_rows=raw_rows,
    )


def _post_bh_exec_publish(handoff: dict[str, Any]) -> dict[str, Any]:
    """Complete BH and atomic publication after an os.execve image replacement."""

    if handoff.get("format") != POST_BH_HANDOFF_FORMAT:
        raise R7StreamingRunError("post-BH exec handoff format drift")
    if handoff.get("association_engine") != ASSOCIATION_ENGINE:
        raise R7StreamingRunError("post-BH exec association engine drift")
    cancer = str(handoff.get("cancer_id", ""))
    lineage_base = handoff.get("lineage_base")
    success_base = handoff.get("success_base")
    if not isinstance(lineage_base, dict) or not isinstance(success_base, dict):
        raise R7StreamingRunError("post-BH lineage/success base is not an object")
    if (
        lineage_base.get("cancer_id") != cancer
        or success_base.get("cancer_id") != cancer
    ):
        raise R7StreamingRunError("post-BH cancer ID differs across handoff payloads")
    contract_sha = str(lineage_base.get("contract_sha256", ""))
    if len(contract_sha) != 64 or not re.fullmatch(r"[0-9a-f]{64}", contract_sha):
        raise R7StreamingRunError("post-BH contract SHA-256 is invalid")
    producer_pid = int(handoff.get("producer_pid", -1))
    declared_publish = Path(handoff["publish"])
    declared_final = Path(handoff["final"])
    declared_raw = Path(handoff["raw_path"])
    declared_output = Path(handoff["evidence_path"])
    declared_scratch = Path(handoff["scratch"])
    if any(
        path.is_symlink()
        for path in (
            declared_publish,
            declared_final,
            declared_raw,
            declared_output,
            declared_scratch,
        )
    ):
        raise R7StreamingRunError("post-BH declared path is a symlink")
    publish = declared_publish.resolve()
    final = declared_final.resolve()
    raw_path = declared_raw.resolve()
    output_path = declared_output.resolve()
    scratch = declared_scratch.resolve()
    for path in (publish, final, raw_path, output_path, scratch):
        _authorized_output(path)
    expected_publish_name = (
        f".cancer_id={cancer}.{contract_sha[:16]}.publish.{producer_pid}"
    )
    if (
        publish.parent != final.parent
        or final.name != f"cancer_id={cancer}"
        or publish.name != expected_publish_name
    ):
        raise R7StreamingRunError("post-BH publish/final topology or name drift")
    if (
        raw_path != publish / ".association_evidence_raw.parquet"
        or output_path != publish / "association_evidence.parquet"
        or scratch != publish / ".association_duckdb_scratch"
    ):
        raise R7StreamingRunError("post-BH raw/evidence/scratch fixed path drift")
    relative_paths = tuple(map(str, handoff.get("relative_paths", [])))
    if relative_paths != POST_BH_RELATIVE_PATHS:
        raise R7StreamingRunError("post-BH fixed release file set/order drift")
    for relative in relative_paths:
        relative_path = Path(relative)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or len(relative_path.parts) != 1
        ):
            raise R7StreamingRunError("post-BH release path is not a safe basename")
        staged = publish / relative
        if staged != output_path and (staged.is_symlink() or not staged.is_file()):
            raise R7StreamingRunError(f"post-BH staged release file absent/unsafe: {relative}")
    if publish.is_symlink() or not publish.is_dir():
        raise R7StreamingRunError("post-BH publish staging directory is absent/unsafe")
    if final.exists() or final.is_symlink():
        raise R7StreamingRunError("post-BH immutable final destination already exists")
    if output_path.exists() or output_path.is_symlink():
        raise R7StreamingRunError("post-BH evidence output unexpectedly exists")
    evidence_rows = _finalize_association_evidence_in_process(
        raw_path=raw_path,
        output_path=output_path,
        scratch=scratch,
        raw_rows=int(handoff["raw_rows"]),
    )
    _memory_trace(
        "ASSOCIATION_BH_SORT_COMPLETE",
        cancer_id=handoff["cancer_id"],
        evidence_rows=evidence_rows,
    )
    _, manifest_sha = _write_manifest(publish, list(relative_paths))
    lineage = {
        **handoff["lineage_base"],
        "file_manifest_sha256": manifest_sha,
        "post_bh_exec_handoff_sha256": handoff["_verified_handoff_sha256"],
    }
    lineage_path = publish / "LINEAGE.json"
    _exclusive_json(lineage_path, lineage)
    success = {
        **handoff["success_base"],
        "association_evidence_rows": int(evidence_rows),
        "file_manifest_sha256": manifest_sha,
        "lineage_sha256": _sha256(lineage_path),
        "post_bh_exec_handoff_sha256": handoff["_verified_handoff_sha256"],
    }
    success_path = publish / "SUCCESS.json"
    _exclusive_json(success_path, success)
    _memory_trace("ATOMIC_PUBLISH_START", cancer_id=handoff["cancer_id"])
    atomic_publish_directory(publish, final)
    return {**success, "output_root": str(final)}


def _materialize_outputs(
    bundle: dict[str, Any],
    accumulator: StreamingGroupAccumulator,
    work: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    output_parent = args.output_parent.resolve()
    cancer = bundle["cancer_id"]
    final = output_parent / f"cancer_id={cancer}"
    publish = output_parent / (
        f".cancer_id={cancer}.{bundle['contract_sha256'][:16]}.publish.{os.getpid()}"
    )
    if publish.exists() or publish.is_symlink() or final.exists() or final.is_symlink():
        raise R7StreamingRunError("publish/final destination already exists")
    publish.mkdir()
    _memory_trace("MATERIALIZE_START", cancer_id=cancer)
    groups = bundle["group_frame"]
    lnc_ids = np.asarray(bundle["mapping"]["lncrna_ids"], dtype=object)
    lnc_symbols = np.asarray(bundle["mapping"]["lncrna_symbols"], dtype=object)
    n_lnc = len(lnc_ids)
    n_groups = len(groups)
    cell_counts = accumulator.cell_counts
    if n_groups <= 0 or (cell_counts <= 0).any():
        raise R7StreamingRunError("donor-celltype groups contain zero cells")
    denominator = cell_counts.astype(np.float64)
    group_patient = groups.patient_id.astype(str).to_numpy()
    group_celltype = groups.cell_type_major.astype(str).to_numpy()
    group_compartment = groups.compartment.astype(str).to_numpy()
    expression_scale = (
        "MEAN_LOG1P_CPM10000"
        if bundle["measurement_scale"] == "raw_counts"
        else "MEAN_SOURCE_LOG_NORMALIZED"
    )

    lnc_path = publish / "lncrna_donor_celltype_summary.parquet"
    lnc_writer = _ParquetFrameWriter(lnc_path, schema=_lnc_summary_schema())
    lnc_batch = max(1, PARQUET_BATCH_ROWS // n_groups)
    for start in range(0, n_lnc, lnc_batch):
        stop = min(start + lnc_batch, n_lnc)
        expression_mean = (
            accumulator.lncrna_expression_sums[start:stop, :] / denominator
        )
        detect_rate = accumulator.lncrna_detect_counts[start:stop, :] / denominator
        frame = pd.DataFrame(
            {
                "dataset_id": bundle["dataset_id"],
                "cancer_id": cancer,
                "lncrna_id": np.repeat(lnc_ids[start:stop], n_groups),
                "lncrna_symbol": np.repeat(lnc_symbols[start:stop], n_groups),
                "patient_id": np.tile(group_patient, stop - start),
                "cell_type_major": np.tile(group_celltype, stop - start),
                "compartment": np.tile(group_compartment, stop - start),
                "cell_count": np.tile(cell_counts, stop - start),
                "detected_cell_count": accumulator.lncrna_detect_counts[
                    start:stop, :
                ].reshape(-1),
                "detect_rate": detect_rate.reshape(-1),
                "mean_expression": expression_mean.reshape(-1),
                "expression_summary_scale": expression_scale,
                "source_generation": SOURCE_GENERATION,
            }
        )
        lnc_writer.write(frame)
        del frame, expression_mean, detect_rate
    lnc_rows = lnc_writer.close()
    if lnc_rows != n_lnc * n_groups:
        raise R7StreamingRunError("lncRNA summary row count drift")
    _memory_trace("LNC_SUMMARY_WRITTEN", cancer_id=cancer, rows=lnc_rows)

    availability = bundle["pathway_contract"].availability.reset_index(drop=True).copy()
    availability.insert(0, "cancer_id", cancer)
    availability.insert(0, "dataset_id", bundle["dataset_id"])
    availability["source_generation"] = SOURCE_GENERATION
    availability["typed_null_enforced"] = ~availability.ucell_available.astype(bool)
    availability_path = publish / "pathway_availability.parquet"
    availability.to_parquet(availability_path, index=False, compression="zstd")
    available_rows = np.flatnonzero(availability.ucell_available.astype(bool).to_numpy())
    available_index = np.full(len(availability), -1, dtype=np.int32)
    available_index[available_rows] = np.arange(len(available_rows), dtype=np.int32)
    pathway_path = publish / "pathway_donor_celltype_summary.parquet"
    pathway_writer = _ParquetFrameWriter(
        pathway_path, schema=_pathway_summary_schema()
    )
    pathway_batch = max(1, PARQUET_BATCH_ROWS // n_groups)
    pathway_ids_all = availability.pathway_id.astype(str).to_numpy()
    pathway_available_all = availability.ucell_available.astype(bool).to_numpy()
    pathway_reason_all = availability.unavailable_reason.to_numpy()
    for start in range(0, len(availability), pathway_batch):
        stop = min(start + pathway_batch, len(availability))
        score = np.full((stop - start, n_groups), np.nan, dtype=np.float32)
        source_rows = available_index[start:stop]
        numeric = source_rows >= 0
        if numeric.any():
            score[numeric, :] = (
                accumulator.pathway_activity_sums[source_rows[numeric], :]
                / denominator
            ).astype(np.float32)
        frame = pd.DataFrame(
            {
                "dataset_id": bundle["dataset_id"],
                "cancer_id": cancer,
                "pathway_id": np.repeat(pathway_ids_all[start:stop], n_groups),
                "patient_id": np.tile(group_patient, stop - start),
                "cell_type_major": np.tile(group_celltype, stop - start),
                "compartment": np.tile(group_compartment, stop - start),
                "cell_count": np.tile(cell_counts, stop - start),
                "ucell_available": np.repeat(
                    pathway_available_all[start:stop], n_groups
                ),
                "unavailable_reason": np.repeat(
                    pathway_reason_all[start:stop], n_groups
                ),
                "ucell_score_mean": pd.array(score.reshape(-1), dtype="Float32"),
                "biological_aggregation_unit": "DONOR_CELLTYPE",
                "source_generation": SOURCE_GENERATION,
            }
        )
        if not frame.loc[~frame.ucell_available, "ucell_score_mean"].isna().all():
            raise R7StreamingRunError(
                "typed-unavailable pathway emitted numeric activity"
            )
        pathway_writer.write(frame)
        del frame, score, source_rows, numeric
    pathway_rows = pathway_writer.close()
    if pathway_rows != len(availability) * n_groups:
        raise R7StreamingRunError("pathway summary row count drift")
    _memory_trace("PATHWAY_SUMMARY_WRITTEN", cancer_id=cancer, rows=pathway_rows)

    raw_evidence_path = publish / ".association_evidence_raw.parquet"
    raw_writer = _ParquetFrameWriter(
        raw_evidence_path, schema=_raw_evidence_schema()
    )
    testability_path = publish / "association_lncrna_testability.parquet"
    testability_writer = _ParquetFrameWriter(
        testability_path, schema=_testability_schema()
    )
    context_rows: list[dict[str, Any]] = []
    available_pathway_ids = np.asarray(
        bundle["pathway_contract"].available_pathway_ids, dtype=object
    )
    groups_compartment = groups.compartment.astype(str).to_numpy()
    for compartment_order, compartment in enumerate(FORMAL_COMPARTMENTS):
        group_indices = np.flatnonzero(
            groups_compartment == compartment
        )
        if len(group_indices):
            donor = _collapse_donor_compartment(
                group_frame=groups,
                group_indices=group_indices,
                cell_counts=cell_counts,
                lnc_expression_sums=accumulator.lncrna_expression_sums,
                lnc_detect_counts=accumulator.lncrna_detect_counts,
                pathway_activity_sums=accumulator.pathway_activity_sums,
                min_cells=args.min_cells_per_donor_context,
            )
        else:
            donor = {
                "donors": [],
                "cell_counts": np.empty(0, dtype=np.int64),
                "lncrna_means": np.empty((0, n_lnc), dtype=float),
                "lncrna_detect_counts": np.empty((0, n_lnc), dtype=np.int64),
                "pathway_means": np.empty(
                    (0, len(available_pathway_ids)), dtype=float
                ),
                "donors_before_min_cells": 0,
            }
        donor_count = len(donor["donors"])
        if donor_count:
            total_detect = donor["lncrna_detect_counts"].sum(axis=0)
            total_cells = int(donor["cell_counts"].sum())
            detect_rate = total_detect / total_cells
            detecting_donors = (donor["lncrna_detect_counts"] > 0).sum(axis=0)
        else:
            detect_rate = np.zeros(n_lnc, dtype=float)
            detecting_donors = np.zeros(n_lnc, dtype=int)
        lnc_eligible = (
            (detect_rate >= args.min_lncrna_detect_rate)
            & (detecting_donors >= args.min_lncrna_detecting_donors)
        )
        testability = pd.DataFrame(
            {
                "dataset_id": bundle["dataset_id"],
                "cancer_id": cancer,
                "compartment": compartment,
                "lncrna_id": lnc_ids,
                "lncrna_symbol": lnc_symbols,
                "eligible_donor_count": donor_count,
                "detecting_donor_count": detecting_donors,
                "context_detect_rate": detect_rate,
                "minimum_context_detect_rate": args.min_lncrna_detect_rate,
                "minimum_detecting_donors": args.min_lncrna_detecting_donors,
                "biological_unit": "DONOR",
                "cell_as_independent_replicate": False,
                "source_generation": SOURCE_GENERATION,
            }
        )
        if donor_count < args.min_donors:
            testability["test_status"] = "TYPED_UNAVAILABLE"
            testability["unavailable_reason"] = INSUFFICIENT_DONOR_REPLICATION
        else:
            testability["test_status"] = np.where(
                lnc_eligible, "ELIGIBLE_FOR_DONOR_ASSOCIATION", "TYPED_UNAVAILABLE"
            )
            testability["unavailable_reason"] = np.where(
                detecting_donors < args.min_lncrna_detecting_donors,
                "INSUFFICIENT_DETECTING_DONORS",
                np.where(
                    detect_rate < args.min_lncrna_detect_rate,
                    "LOW_CONTEXT_DETECT_RATE",
                    None,
                ),
            )
        for text_column in (
            "dataset_id",
            "cancer_id",
            "compartment",
            "lncrna_id",
            "lncrna_symbol",
            "biological_unit",
            "source_generation",
            "test_status",
            "unavailable_reason",
        ):
            testability[text_column] = pd.array(
                testability[text_column], dtype="string"
            )
        testability_writer.write(testability)
        del testability
        if donor_count >= args.min_donors and lnc_eligible.any():
            eligible_ids = lnc_ids[lnc_eligible]
            eligible_symbols = lnc_symbols[lnc_eligible]
            association_lnc = pd.DataFrame(
                donor["lncrna_means"][:, lnc_eligible],
                index=donor["donors"],
                columns=eligible_ids,
            )
            association_pathway = pd.DataFrame(
                donor["pathway_means"],
                index=donor["donors"],
                columns=available_pathway_ids,
            )
            total_tests_hint = int(
                (np.ptp(association_lnc.to_numpy(copy=False), axis=0) > 0).sum()
                * (np.ptp(association_pathway.to_numpy(copy=False), axis=0) > 0).sum()
            )

            def evidence_sink(chunk: AssociationEvidenceChunk) -> None:
                for row_start in range(0, len(chunk.nominal_p), PARQUET_BATCH_ROWS):
                    row_stop = min(
                        row_start + PARQUET_BATCH_ROWS, len(chunk.nominal_p)
                    )
                    lnc_rows = chunk.lncrna_rows[row_start:row_stop]
                    pathway_rows = chunk.pathway_rows[row_start:row_stop]
                    frame = pd.DataFrame(
                        {
                            "dataset_id": bundle["dataset_id"],
                            "cancer_id": cancer,
                            "compartment_order": np.int8(compartment_order),
                            "compartment": compartment,
                            "lncrna_id": eligible_ids[lnc_rows],
                            "lncrna_symbol": eligible_symbols[lnc_rows],
                            "pathway_id": available_pathway_ids[pathway_rows],
                            "n_donors": donor_count,
                            "spearman_rho": chunk.spearman_rho[
                                row_start:row_stop
                            ],
                            "nominal_p": chunk.nominal_p[row_start:row_stop],
                            "total_tests": total_tests_hint,
                        }
                    )
                    raw_writer.write(frame)
                    del frame, lnc_rows, pathway_rows

            context = stream_donor_association_chunks(
                association_lnc,
                association_pathway,
                compartment=compartment,
                evidence_sink=evidence_sink,
                min_donors=args.min_donors,
                min_abs_rho=args.min_abs_rho,
                max_nominal_p=args.max_nominal_p,
                pathway_block=args.association_pathway_block,
            )
            if int(context.get("tested_pair_count", 0)) != total_tests_hint:
                raise R7StreamingRunError(
                    "association multiple-testing denominator drift"
                )
            del association_lnc, association_pathway, eligible_ids, eligible_symbols
        else:
            context = {
                "compartment": compartment,
                "biological_unit": "DONOR",
                "cell_as_independent_replicate": False,
                "donor_count": donor_count,
                "minimum_donors": args.min_donors,
                "status": "TYPED_UNAVAILABLE",
                "unavailable_reason": (
                    INSUFFICIENT_DONOR_REPLICATION
                    if donor_count < args.min_donors
                    else "NO_ELIGIBLE_LNCRNA"
                ),
                "tested_pair_count": 0,
                "retained_evidence_count": 0,
            }
        context.update(
            dataset_id=bundle["dataset_id"],
            cancer_id=cancer,
            donors_before_min_cells=int(donor["donors_before_min_cells"]),
            eligible_donors=donor_count,
            cells_in_eligible_donors=int(donor["cell_counts"].sum()),
            minimum_cells_per_donor_context=args.min_cells_per_donor_context,
            eligible_lncrnas=int(lnc_eligible.sum()),
            available_pathways=len(available_pathway_ids),
            source_generation=SOURCE_GENERATION,
        )
        context_rows.append(context)
        del donor, detect_rate, detecting_donors, lnc_eligible, group_indices
        _release_transient_memory()
        _memory_trace(
            "ASSOCIATION_COMPARTMENT_SPOOLED",
            cancer_id=cancer,
            compartment=compartment,
            retained_evidence_count=int(context["retained_evidence_count"]),
        )

    testability_rows = testability_writer.close()
    raw_evidence_rows = raw_writer.close()
    if testability_rows != n_lnc * len(FORMAL_COMPARTMENTS):
        raise R7StreamingRunError("association testability row count drift")
    _memory_trace(
        "ASSOCIATION_SPOOL_COMPLETE",
        cancer_id=cancer,
        raw_evidence_rows=raw_evidence_rows,
    )
    evidence_path = publish / "association_evidence.parquet"
    context_path = publish / "association_context_availability.parquet"
    pd.DataFrame(context_rows).to_parquet(
        context_path, index=False, compression="zstd"
    )

    resource_path = publish / "RESOURCE_ESTIMATE.json"
    _exclusive_json(resource_path, bundle["resource_estimate"])
    relative_paths = [
        lnc_path.name,
        availability_path.name,
        pathway_path.name,
        evidence_path.name,
        testability_path.name,
        context_path.name,
        resource_path.name,
    ]
    if tuple(relative_paths) != POST_BH_RELATIVE_PATHS:
        raise R7StreamingRunError("post-BH release file construction drift")
    checkpoint_state = work / "CHECKPOINT_STATE.json"
    lineage_base = {
        **bundle["contract"],
        "format": "CC_HHGT_V3_2_SINGLE_CELL_R7_STREAMING_LINEAGE_V1",
        "contract_sha256": bundle["contract_sha256"],
        "input_paths": {key: str(value) for key, value in bundle["input_paths"].items()},
        "input_shas": bundle["input_shas"],
        "code_shas": bundle["code_shas"],
        "checkpoint_state_sha256": _sha256(checkpoint_state),
        "processed_raw_cells": int(bundle["shape"][1]),
        "processed_kept_cells": int(cell_counts.sum()),
        "cell_level_pathway_rows_written": 0,
        "association_engine": ASSOCIATION_ENGINE,
        "association_execution_strategy": ASSOCIATION_EXECUTION_STRATEGY,
        "association_bh_family": ASSOCIATION_BH_FAMILY,
        "association_stage_key_policy": ASSOCIATION_STAGE_KEY_POLICY,
        "association_duckdb_memory_limit": ASSOCIATION_DUCKDB_MEMORY_LIMIT,
        "parquet_batch_rows": PARQUET_BATCH_ROWS,
        "per_cancer_atomic_publish": True,
        "resume_checkpoint_used": bool(args.resume),
        "work_checkpoint_retained_for_audit": True,
        "production_deployed": False,
        "release_ready": False,
    }
    success_base = {
        "format": SUCCESS_FORMAT,
        "status": "SUCCESS",
        "analysis_version": ANALYSIS_VERSION,
        "source_generation": SOURCE_GENERATION,
        "cancer_id": cancer,
        "dataset_id": bundle["dataset_id"],
        "contract_sha256": bundle["contract_sha256"],
        "cells": int(cell_counts.sum()),
        "donors": int(bundle["metadata"].loc[bundle["keep_mask"], "patient_id"].nunique()),
        "lncrnas": n_lnc,
        "pathways_total": int(len(availability)),
        "pathways_available": int(availability.ucell_available.sum()),
        "pathways_typed_unavailable": int((~availability.ucell_available).sum()),
        "association_engine": ASSOCIATION_ENGINE,
        "association_execution_strategy": ASSOCIATION_EXECUTION_STRATEGY,
        "association_bh_family": ASSOCIATION_BH_FAMILY,
        "association_stage_key_policy": ASSOCIATION_STAGE_KEY_POLICY,
        "association_duckdb_memory_limit": ASSOCIATION_DUCKDB_MEMORY_LIMIT,
        "parquet_batch_rows": PARQUET_BATCH_ROWS,
        "formal_compartments": list(FORMAL_COMPARTMENTS),
        "biological_unit": "DONOR",
        "cell_as_independent_replicate": False,
        "cell_level_pathway_matrix_persisted": False,
        "cell_level_pathway_rows_written": 0,
        "historical_derived_results_used": False,
        "full_input_sha256_verified_before_matrix_access": True,
        "raw_h5_full_sha256_verified": True,
        "production_deployed": False,
        "release_ready": False,
    }
    handoff = {
        "format": POST_BH_HANDOFF_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "association_engine": ASSOCIATION_ENGINE,
        "association_execution_strategy": ASSOCIATION_EXECUTION_STRATEGY,
        "cancer_id": cancer,
        "publish": str(publish),
        "final": str(final),
        "raw_path": str(raw_evidence_path),
        "evidence_path": str(evidence_path),
        "scratch": str(publish / ".association_duckdb_scratch"),
        "raw_rows": int(raw_evidence_rows),
        "relative_paths": relative_paths,
        "lineage_base": lineage_base,
        "success_base": success_base,
        "source_runner_path": str(Path(__file__).resolve()),
        "source_runner_sha256": _sha256(Path(__file__).resolve()),
        "producer_pid": os.getpid(),
        "formal_transition": "OS_EXECVE_REPLACES_EXPRESSION_PROCESS_IMAGE",
        "no_parent_helper_overlap": True,
    }
    handoff_path = work / f"POST_BH_HANDOFF.{os.getpid()}.json"
    _exclusive_json(handoff_path, handoff)
    handoff_sha = _sha256(handoff_path)
    released_bytes = 0
    for attribute in (
        "lncrna_expression_sums",
        "lncrna_detect_counts",
        "pathway_activity_sums",
    ):
        array = getattr(accumulator, attribute)
        released_bytes += int(array.nbytes)
        setattr(accumulator, attribute, np.empty((0, 0), dtype=array.dtype))
    del context_rows
    _release_transient_memory()
    _memory_trace(
        "POST_BH_EXEC_START",
        cancer_id=cancer,
        handoff_sha256=handoff_sha,
        released_accumulator_bytes=released_bytes,
    )
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--internal-post-bh-r9",
        "--handoff-json",
        str(handoff_path),
        "--expected-handoff-sha256",
        handoff_sha,
    ]
    os.execve(sys.executable, command, dict(os.environ))
    raise R7StreamingRunError("os.execve unexpectedly returned")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("plan", "run"), required=True)
    parser.add_argument("--r7-root", required=True, type=Path)
    parser.add_argument("--server-preflight-json", required=True, type=Path)
    parser.add_argument("--expected-server-preflight-sha256", required=True)
    parser.add_argument("--expected-run-status-sha256", required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--cancer-id")
    selection.add_argument("--select-smallest-formal", action="store_true")
    parser.add_argument("--plan-output-json", type=Path)
    parser.add_argument("--output-parent", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-rank", type=int, default=1500)
    parser.add_argument("--chunk-cells", type=int, default=64, choices=(32, 64, 128))
    parser.add_argument("--checkpoint-every-chunks", type=int, default=25)
    parser.add_argument("--min-donors", type=int, default=5)
    parser.add_argument("--min-cells-per-donor-context", type=int, default=20)
    parser.add_argument("--min-lncrna-detect-rate", type=float, default=0.01)
    parser.add_argument("--min-lncrna-detecting-donors", type=int, default=3)
    parser.add_argument("--min-abs-rho", type=float, default=0.5)
    parser.add_argument("--max-nominal-p", type=float, default=0.05)
    parser.add_argument("--association-pathway-block", type=int, default=32)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.checkpoint_every_chunks <= 0:
        raise R7StreamingRunError("checkpoint interval must be positive")
    if args.min_donors < 3 or args.min_cells_per_donor_context <= 0:
        raise R7StreamingRunError("donor/context thresholds are invalid")
    if not (0 <= args.min_lncrna_detect_rate <= 1):
        raise R7StreamingRunError("lncRNA detect-rate threshold is invalid")
    if args.min_lncrna_detecting_donors <= 0:
        raise R7StreamingRunError("detecting-donor threshold must be positive")
    if not (0 <= args.min_abs_rho <= 1) or not (0 < args.max_nominal_p <= 1):
        raise R7StreamingRunError("association thresholds are invalid")
    if args.association_pathway_block <= 0 or args.association_pathway_block > 512:
        raise R7StreamingRunError("association pathway block must be in 1..512")
    if args.mode == "plan" and args.plan_output_json is None:
        raise R7StreamingRunError("plan mode requires --plan-output-json")
    if args.mode == "run" and args.output_parent is None:
        raise R7StreamingRunError("run mode requires --output-parent")


def main() -> int:
    args = build_parser().parse_args()
    _validate_args(args)
    bundle = _load_contract(args)
    if args.mode == "plan":
        output = args.plan_output_json.resolve()
        _authorized_output(output)
        if output.exists() or output.is_symlink():
            raise R7StreamingRunError(f"plan output reuse is forbidden: {output}")
        plan = _plan_summary(bundle)
        _exclusive_json(output, plan)
        print(json.dumps({**plan, "plan_output_json": str(output)}, indent=2, sort_keys=True))
        return 0
    accumulator, work = _run_stream(bundle, args)
    result = _materialize_outputs(bundle, accumulator, work, args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _internal_main(argv: list[str]) -> int:
    """Run frozen r9 internal phases that are never accepted as public modes."""

    internal_mode = argv[0]
    if internal_mode == "--internal-finalize-association-r9":
        parser = argparse.ArgumentParser()
        parser.add_argument("--raw-path", required=True, type=Path)
        parser.add_argument("--output-path", required=True, type=Path)
        parser.add_argument("--scratch", required=True, type=Path)
        parser.add_argument("--raw-rows", required=True, type=int)
        args = parser.parse_args(argv[1:])
        for path in (args.raw_path, args.output_path, args.scratch):
            _authorized_output(path.resolve())
        observed = _finalize_association_evidence_in_process(
            raw_path=args.raw_path.resolve(),
            output_path=args.output_path.resolve(),
            scratch=args.scratch.resolve(),
            raw_rows=args.raw_rows,
        )
        print(json.dumps({"status": "SUCCESS", "rows": observed}, sort_keys=True))
        return 0
    if internal_mode == "--internal-post-bh-r9":
        parser = argparse.ArgumentParser()
        parser.add_argument("--handoff-json", required=True, type=Path)
        parser.add_argument("--expected-handoff-sha256", required=True)
        args = parser.parse_args(argv[1:])
        handoff_path = args.handoff_json.resolve()
        _authorized_output(handoff_path)
        observed_sha = _sha256(handoff_path)
        if observed_sha != args.expected_handoff_sha256:
            raise R7StreamingRunError("post-BH exec handoff SHA-256 drift")
        handoff = _load_json(handoff_path)
        declared_runner = Path(str(handoff.get("source_runner_path", "")))
        current_runner = Path(__file__).resolve()
        if declared_runner.is_symlink() or declared_runner.resolve() != current_runner:
            raise R7StreamingRunError("post-BH exec runner path drift")
        if handoff.get("source_runner_sha256") != _sha256(current_runner):
            raise R7StreamingRunError("post-BH exec runner SHA-256 drift")
        if int(handoff.get("producer_pid", -1)) != os.getpid():
            raise R7StreamingRunError("post-BH exec did not preserve producer PID")
        handoff["_verified_handoff_sha256"] = observed_sha
        result = _post_bh_exec_publish(handoff)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    raise R7StreamingRunError(f"unknown internal mode: {internal_mode}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].startswith("--internal-"):
        raise SystemExit(_internal_main(sys.argv[1:]))
    raise SystemExit(main())
