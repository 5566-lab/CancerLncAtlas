#!/usr/bin/env python3
"""Fresh HNSC cell-level UCell small-parity and full-pilot runner.

The runner accepts only a corrected V2 preflight and a passing pinned-official
parity artifact.  It never reads historical trajectory, prediction, ranking,
or checkpoint products.  Pseudotime stays explicitly unavailable when the
preflight has no declared root/order.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.input_lineage import artifact_sha256  # noqa: E402
from cc_hhgt.v32.single_cell_cell_level import (  # noqa: E402
    SIGNATURE_AT_OR_ABOVE_MAX_RANK,
    UCELL_FORMULA,
    UCELL_MISSING_MEMBER_POLICY,
    UCellContractError,
    build_ucell_pathway_contract,
    rank_expression_ucell,
    score_ucell_pathway_block,
)
from cc_hhgt.v32.single_cell_partition_builder import (  # noqa: E402
    _decode,
    _stable_gene_id,
)


PILOT_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_CELL_LEVEL_PILOT_V1"
PREFLIGHT_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_CELL_LEVEL_PREFLIGHT_V2"
OFFICIAL_PARITY_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_UCELL_OFFICIAL_PARITY_V1"
PSEUDOTIME_REASON = "NO_EXPLICIT_TRAJECTORY_ROOT_OR_ORDERED_SOURCE_STATE"
FORBIDDEN_SOURCE_TOKENS = ("sc_trajectory", "prediction", "ranking", "checkpoint")
FORMAL_CANCERS = {
    "ACC", "CHOL", "DLBC", "ESCA", "GBM", "HNSC", "KIRC", "LAML", "LGG",
    "LUSC", "MESO", "PCPG", "READ", "SARC", "SKCM", "THYM", "UCEC",
}


class CellLevelPilotError(RuntimeError):
    """Raised when any input, parity, or output gate fails closed."""


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _validate_sha(path: Path, expected: str, role: str) -> str:
    if not path.is_file():
        raise CellLevelPilotError(f"Missing {role}: {path}")
    observed = artifact_sha256(path)
    if observed.lower() != str(expected).lower():
        raise CellLevelPilotError(f"{role} SHA drift: {observed} != {expected}")
    return observed


def _assert_fresh_source(path: Path, role: str) -> None:
    lowered = str(path).lower().replace("\\", "/")
    matched = [token for token in FORBIDDEN_SOURCE_TOKENS if token in lowered]
    if matched:
        raise CellLevelPilotError(
            f"{role} contains forbidden historical-result token(s): {matched}"
        )


def _truthy(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.fillna("").astype(str).str.strip().str.lower().isin(
        {"1", "true", "yes", "y", "doublet"}
    )


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise CellLevelPilotError(f"Cannot load official module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_official_pyucell(pyucell_root: Path):
    package_root = pyucell_root / "pyucell"
    if not package_root.is_dir():
        raise CellLevelPilotError(f"Official pyUCell package absent: {package_root}")
    anndata = types.ModuleType("anndata")
    anndata.AnnData = type("AnnData", (), {})
    sys.modules["anndata"] = anndata
    package = types.ModuleType("pyucell")
    package.__path__ = [str(package_root)]
    sys.modules["pyucell"] = package
    _load_module("pyucell._torch_utils", package_root / "_torch_utils.py")
    _load_module("pyucell.ranks", package_root / "ranks.py")
    return _load_module("pyucell.scoring", package_root / "scoring.py")


def _boundary_corrected_pyucell_rankings(
    expression_gene_by_cell: np.ndarray,
    *,
    max_rank: int,
) -> tuple[sparse.csr_matrix, dict[str, Any]]:
    """Mirror pyUCell 0.7.3 CPU ranks without its tie-boundary truncation bug.

    The pinned implementation first retains every non-zero gene whose average
    rank is at most ``max_rank`` and then, when ties make that set larger than
    ``max_rank``, incorrectly truncates by feature index.  Current R UCell does
    not perform that second truncation.  This independent reconstruction keeps
    the pinned int32 average-rank cast but removes only that index-order cut.
    """

    values = np.asarray(expression_gene_by_cell, dtype=np.float64)
    rows: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    data: list[np.ndarray] = []
    triggered_cells: list[int] = []
    excess_retained_by_cell: dict[str, int] = {}
    for cell_index in range(values.shape[1]):
        column = values[:, cell_index]
        nonzero = np.flatnonzero(column)
        if not len(nonzero):
            continue
        ranks = rankdata(-column[nonzero], method="average").astype(np.int32)
        keep = ranks <= int(max_rank)
        kept_indices = nonzero[keep]
        kept_ranks = ranks[keep]
        if len(kept_indices) > int(max_rank):
            triggered_cells.append(cell_index)
            excess_retained_by_cell[str(cell_index)] = int(
                len(kept_indices) - int(max_rank)
            )
        rows.append(kept_indices.astype(np.int32))
        columns.append(np.full(len(kept_indices), cell_index, dtype=np.int32))
        data.append(kept_ranks.astype(np.int32))
    if data:
        matrix = sparse.csr_matrix(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(columns))),
            shape=values.shape,
            dtype=np.int32,
        )
    else:
        matrix = sparse.csr_matrix(values.shape, dtype=np.int32)
    return matrix, {
        "pinned_pyucell_boundary_bug_triggered": bool(triggered_cells),
        "triggered_cell_indices": triggered_cells,
        "triggered_cell_count": len(triggered_cells),
        "excess_ranked_genes_beyond_max_rank_by_cell": excess_retained_by_cell,
        "correction": "REMOVE_FEATURE_INDEX_TRUNCATION_AFTER_AVERAGE_TIE_RANK_GATE",
        "rank_dtype_retained_from_pinned_pyucell": "int32",
    }


def _validate_gates(
    *,
    preflight_path: Path,
    expected_preflight_sha256: str,
    official_parity_path: Path,
    expected_official_parity_sha256: str,
    expected_cancer_id: str | None = "HNSC",
    require_pseudotime_unavailable: bool = True,
) -> tuple[dict[str, Any], str, dict[str, Any], str]:
    preflight_sha = _validate_sha(
        preflight_path, expected_preflight_sha256, "corrected cell-level preflight"
    )
    official_sha = _validate_sha(
        official_parity_path,
        expected_official_parity_sha256,
        "official UCell parity",
    )
    preflight = _load_json(preflight_path)
    official_parity = _load_json(official_parity_path)
    if preflight.get("format") != PREFLIGHT_FORMAT:
        raise CellLevelPilotError("Pilot requires corrected V2 preflight")
    cancer_id = str(preflight.get("cancer_id", "")).upper().strip()
    if expected_cancer_id is not None and cancer_id != expected_cancer_id:
        raise CellLevelPilotError("This restricted pilot is HNSC-only")
    if expected_cancer_id is None and cancer_id not in FORMAL_CANCERS:
        raise CellLevelPilotError(
            f"Generalized runner accepts only the 17 formal cancers: {cancer_id}"
        )
    if preflight.get("pilot_safe_to_start") is not True:
        raise CellLevelPilotError("Preflight does not permit a pilot")
    ucell = preflight.get("ucell", {})
    if not ucell.get("complete_exact_signature_length_used_in_denominator"):
        raise CellLevelPilotError("Preflight does not bind complete signature length")
    if ucell.get("missing_gene_policy") != UCELL_MISSING_MEMBER_POLICY:
        raise CellLevelPilotError("Preflight missing-member policy drift")
    if ucell.get("missing_member_treatment_is_expression_data_imputation") is not False:
        raise CellLevelPilotError("Preflight incorrectly declares data imputation")
    if (
        require_pseudotime_unavailable
        and preflight.get("pseudotime", {}).get("numeric_output_permitted") is not False
    ):
        raise CellLevelPilotError("HNSC pseudotime gate unexpectedly became numeric")
    if official_parity.get("format") != OFFICIAL_PARITY_FORMAT:
        raise CellLevelPilotError("Unknown official-parity format")
    if official_parity.get("status") != "PASS":
        raise CellLevelPilotError("Official UCell parity did not pass")
    if not official_parity.get("complete_signature_length_used"):
        raise CellLevelPilotError("Official parity did not use complete signatures")
    if official_parity.get("matrix_missing_member_policy") != UCELL_MISSING_MEMBER_POLICY:
        raise CellLevelPilotError("Official parity missing-member policy drift")
    return preflight, preflight_sha, official_parity, official_sha


def _load_fresh_inputs(preflight: dict[str, Any], *, max_rank: int):
    inputs = preflight["inputs"]
    h5_path = Path(inputs["h5_path"]).resolve()
    metadata_path = Path(inputs["metadata_path"]).resolve()
    annotation_path = Path(inputs["annotation_path"]).resolve()
    membership_path = Path(inputs["exact_membership_path"]).resolve()
    for path, role in (
        (h5_path, "cell H5"),
        (metadata_path, "cell metadata"),
        (annotation_path, "GENCODE annotation"),
        (membership_path, "exact pathway membership"),
    ):
        _assert_fresh_source(path, role)
    observed_input_shas = {
        "h5_sha256": _validate_sha(h5_path, inputs["h5_sha256"], "cell H5"),
        "metadata_sha256": _validate_sha(
            metadata_path, inputs["metadata_sha256"], "cell metadata"
        ),
        "annotation_sha256": _validate_sha(
            annotation_path, inputs["annotation_sha256"], "GENCODE annotation"
        ),
        "exact_membership_sha256": _validate_sha(
            membership_path,
            inputs["exact_membership_sha256"],
            "exact pathway membership",
        ),
    }
    metadata = pd.read_parquet(metadata_path)
    annotation = pd.read_parquet(annotation_path)
    membership = pd.read_parquet(membership_path)
    membership["pathway_id"] = membership.pathway_id.astype(str)
    membership["gene_id"] = membership.gene_id.map(_stable_gene_id)
    membership = membership.drop_duplicates(["pathway_id", "gene_id"])
    if membership.pathway_id.nunique() != 2135:
        raise CellLevelPilotError("Exact membership is not the 2,135-pathway authority")

    annotation["gene_id"] = annotation.gene_id.map(_stable_gene_id)
    by_id = annotation.set_index("gene_id", verify_integrity=True)
    unique_symbol_rows = annotation.loc[annotation.unique_symbol.astype(bool)]
    by_symbol = unique_symbol_rows.set_index("gene_symbol", verify_integrity=True)

    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise CellLevelPilotError("h5py is required") from exc
    with h5py.File(h5_path, "r") as handle:
        group = handle["matrix"] if "matrix" in handle else handle
        shape = tuple(int(value) for value in np.asarray(group["shape"][:]))
        barcodes = _decode(group["barcodes"][:])
        feature_ids = _decode(group["features"]["id"][:])
        feature_names = _decode(group["features"]["name"][:])
        matrix = sparse.csc_matrix(
            (
                np.asarray(group["data"][:]),
                np.asarray(group["indices"][:], dtype=np.int64),
                np.asarray(group["indptr"][:], dtype=np.int64),
            ),
            shape=shape,
        )
    if shape != (len(feature_ids), len(barcodes)):
        raise CellLevelPilotError("H5 dimensions disagree with features/barcodes")
    indexer = pd.Index(metadata.cell_id.astype(str)).get_indexer(barcodes)
    if (indexer < 0).any() or len(metadata) != len(barcodes):
        raise CellLevelPilotError("H5 barcodes and metadata are not one-to-one")
    metadata = metadata.iloc[indexer].reset_index(drop=True)

    protein_rows: list[int] = []
    protein_ids: list[str] = []
    for row_index, (raw_id, raw_name) in enumerate(
        zip(feature_ids, feature_names, strict=True)
    ):
        direct = _stable_gene_id(raw_id)
        if direct in by_id.index:
            row = by_id.loc[direct]
            stable = direct
        elif raw_name in by_symbol.index:
            row = by_symbol.loc[raw_name]
            stable = str(row.gene_id)
        else:
            continue
        if str(row.gene_class) == "protein_coding":
            protein_rows.append(row_index)
            protein_ids.append(stable)
    if len(protein_ids) != len(set(protein_ids)):
        raise CellLevelPilotError("Protein rank universe has duplicate stable IDs")
    if len(protein_ids) != preflight["h5_audit"]["unique_protein_gene_ids"]:
        raise CellLevelPilotError("Protein rank-universe count drifted from preflight")

    doublet_columns = [
        column
        for column in (
            "is_doublet",
            "doublet",
            "doublet_flag",
            "qc_doublet",
            "predicted_doublet",
        )
        if column in metadata
    ]
    doublet_mask = pd.Series(False, index=metadata.index)
    for column in doublet_columns:
        doublet_mask |= _truthy(metadata[column])
    kept_columns = np.flatnonzero(~doublet_mask.to_numpy())
    kept_metadata = metadata.iloc[kept_columns].reset_index(drop=True)
    if kept_metadata.empty:
        raise CellLevelPilotError("All cells were excluded as explicit doublets")
    protein_matrix = matrix[np.asarray(protein_rows, dtype=np.int64), :].tocsc()
    contract = build_ucell_pathway_contract(
        membership,
        protein_gene_ids=protein_ids,
        max_rank=max_rank,
    )
    return {
        "metadata": kept_metadata,
        "cell_columns": kept_columns,
        "protein_matrix": protein_matrix,
        "protein_ids": tuple(protein_ids),
        "membership": membership,
        "contract": contract,
        "input_paths": {
            "h5_path": str(h5_path),
            "metadata_path": str(metadata_path),
            "annotation_path": str(annotation_path),
            "exact_membership_path": str(membership_path),
        },
        "input_shas": observed_input_shas,
        "doublet_columns": doublet_columns,
        "explicit_doublets_excluded": int(doublet_mask.sum()),
    }


def _count_and_gate_audit(bundle: dict[str, Any], preflight: dict[str, Any]) -> dict[str, Any]:
    contract = bundle["contract"]
    membership = bundle["membership"]
    protein_set = set(bundle["protein_ids"])
    sidecar = contract.availability.set_index("pathway_id")
    mismatches: list[dict[str, Any]] = []
    for pathway_id, group in membership.groupby("pathway_id", sort=True, observed=True):
        genes = set(group.gene_id.astype(str))
        present = len(genes & protein_set)
        observed = sidecar.loc[str(pathway_id)]
        expected = (len(genes), present, len(genes) - present)
        actual = (
            int(observed.signature_gene_count),
            int(observed.present_gene_count),
            int(observed.missing_gene_count),
        )
        if expected != actual:
            mismatches.append(
                {"pathway_id": str(pathway_id), "independent": expected, "built": actual}
            )
    reason_counts = (
        contract.availability.loc[
            ~contract.availability.ucell_available, "unavailable_reason"
        ]
        .value_counts()
        .sort_index()
        .astype(int)
        .to_dict()
    )
    n_gate = contract.availability.signature_gene_count.ge(
        int(contract.availability.max_rank.iloc[0])
    )
    present_zero = contract.availability.present_gene_count.eq(0)
    n_gate_pass = bool(
        contract.availability.loc[n_gate & ~present_zero, "unavailable_reason"]
        .eq(SIGNATURE_AT_OR_ABOVE_MAX_RANK)
        .all()
    )
    present_zero_pass = bool(
        contract.availability.loc[present_zero, "ucell_available"].eq(False).all()
    )
    expected_available = int(preflight["ucell"]["pathways_available"])
    expected_unavailable = int(preflight["ucell"]["pathways_unavailable"])
    expected_reason_counts = {
        str(key): int(value)
        for key, value in preflight["ucell"]["unavailable_reason_counts"].items()
    }
    available = int(contract.availability.ucell_available.sum())
    unavailable = int(len(contract.availability) - available)
    count_pass = not mismatches
    preflight_pass = (
        available == expected_available
        and unavailable == expected_unavailable
        and reason_counts == expected_reason_counts
    )
    passed = bool(count_pass and n_gate_pass and present_zero_pass and preflight_pass)
    return {
        "status": "PASS" if passed else "FAIL",
        "pathways": int(len(contract.availability)),
        "pathways_available": available,
        "pathways_unavailable": unavailable,
        "unavailable_reason_counts": reason_counts,
        "complete_signature_n_at_or_above_max_rank": int(n_gate.sum()),
        "present_gene_count_zero": int(present_zero.sum()),
        "independent_present_missing_count_parity": count_pass,
        "independent_count_mismatch_count": len(mismatches),
        "independent_count_mismatches_first_10": mismatches[:10],
        "n_at_or_above_max_rank_gate_pass": n_gate_pass,
        "present_zero_gate_pass": present_zero_pass,
        "preflight_summary_parity_pass": preflight_pass,
        "preflight_unavailable_reason_counts": expected_reason_counts,
    }


def _choose_small_pathways(availability: pd.DataFrame, count: int) -> list[str]:
    available = availability.loc[availability.ucell_available].copy()
    chosen: list[str] = []
    subsets = (
        available.loc[available.missing_gene_count.gt(0)].sort_values("pathway_id"),
        available.loc[available.missing_gene_count.eq(0)].sort_values("pathway_id"),
        available.sort_values(["signature_gene_count", "pathway_id"]),
        available.sort_values(
            ["signature_gene_count", "pathway_id"], ascending=[False, True]
        ),
    )
    for subset in subsets:
        for pathway_id in subset.pathway_id.astype(str).head(max(1, count // 4)):
            if pathway_id not in chosen:
                chosen.append(pathway_id)
    for pathway_id in available.sort_values("pathway_id").pathway_id.astype(str):
        if len(chosen) >= count:
            break
        if pathway_id not in chosen:
            chosen.append(pathway_id)
    return chosen[:count]


def _independent_r_helper_scores(
    expression_gene_by_cell: np.ndarray,
    *,
    protein_ids: tuple[str, ...],
    membership: pd.DataFrame,
    pathway_ids: list[str],
    max_rank: int,
) -> np.ndarray:
    ranks = rankdata(-expression_gene_by_cell, method="average", axis=0)
    ranks = np.minimum(ranks, float(max_rank))
    protein_index = {gene: index for index, gene in enumerate(protein_ids)}
    grouped = {
        str(pathway): tuple(group.gene_id.astype(str))
        for pathway, group in membership.groupby("pathway_id", sort=False, observed=True)
    }
    scores: list[np.ndarray] = []
    for pathway_id in pathway_ids:
        genes = grouped[pathway_id]
        present = [protein_index[gene] for gene in genes if gene in protein_index]
        missing = len(genes) - len(present)
        rank_sum = np.full(expression_gene_by_cell.shape[1], missing * max_rank, dtype=float)
        rank_sum += ranks[present, :].sum(axis=0)
        n = len(genes)
        minimum = n * (n + 1) / 2.0
        scores.append(1.0 - (rank_sum - minimum) / (n * max_rank - minimum))
    return np.vstack(scores)


def _write_manifest(root: Path, relative_paths: list[str]) -> tuple[Path, str]:
    rows = []
    for relative in sorted(relative_paths):
        path = root / relative
        rows.append(
            {
                "relative_path": relative.replace("\\", "/"),
                "size_bytes": int(path.stat().st_size),
                "sha256": artifact_sha256(path),
            }
        )
    manifest = root / "FILE_MANIFEST.parquet"
    pd.DataFrame(rows).to_parquet(manifest, index=False, compression="zstd")
    return manifest, artifact_sha256(manifest)


def _base_provenance(
    *,
    mode: str,
    preflight: dict[str, Any],
    preflight_path: Path,
    preflight_sha: str,
    official_parity_path: Path,
    official_parity_sha: str,
    bundle: dict[str, Any],
    max_rank: int,
) -> dict[str, Any]:
    runner_path = Path(__file__).resolve()
    cell_level_module_path = (
        ROOT / "cc_hhgt" / "v32" / "single_cell_cell_level.py"
    ).resolve()
    return {
        "format": PILOT_FORMAT,
        "mode": mode,
        "process_id": int(os.getpid()),
        "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
        "cancer_id": str(preflight["cancer_id"]).upper(),
        "dataset_id": str(bundle["metadata"].dataset_id.iloc[0]),
        "algorithm": "UCELL_MANN_WHITNEY_U_UPDATED_NORMALIZATION",
        "formula": UCELL_FORMULA,
        "max_rank": int(max_rank),
        "ties_method": "average",
        "complete_exact_signature_length_used": True,
        "matrix_missing_member_policy": UCELL_MISSING_MEMBER_POLICY,
        "matrix_missing_member_is_expression_data_imputation": False,
        "smoothing_used": False,
        "preflight_path": str(preflight_path),
        "preflight_sha256": preflight_sha,
        "official_parity_path": str(official_parity_path),
        "official_parity_sha256": official_parity_sha,
        "runtime_code": {
            "root": str(ROOT),
            "runner_path": str(runner_path),
            "runner_sha256": artifact_sha256(runner_path),
            "cell_level_module_path": str(cell_level_module_path),
            "cell_level_module_sha256": artifact_sha256(cell_level_module_path),
        },
        "r6_authority": preflight["r6_authority"],
        "inputs": bundle["input_paths"] | bundle["input_shas"],
        "protein_rank_universe_genes": len(bundle["protein_ids"]),
        "cells_after_explicit_doublet_exclusion": len(bundle["metadata"]),
        "explicit_doublets_excluded": bundle["explicit_doublets_excluded"],
        "historical_sc_trajectory_used": False,
        "historical_predictions_used": False,
        "historical_rankings_used": False,
        "historical_checkpoints_used": False,
        "training_started": False,
        "single_cell_module_complete": False,
        "release_ready": False,
        "production_deployed": False,
    }


def run_small(
    *,
    staging_root: Path,
    published_root: Path,
    provenance: dict[str, Any],
    bundle: dict[str, Any],
    count_audit: dict[str, Any],
    pyucell_root: Path,
    expected_scoring_sha256: str,
    expected_ranks_sha256: str,
    max_rank: int,
    small_cells: int,
    small_pathways: int,
) -> dict[str, Any]:
    if count_audit["status"] != "PASS":
        raise CellLevelPilotError("Present/missing count gate did not pass")
    scoring_sha = _validate_sha(
        pyucell_root / "pyucell" / "scoring.py",
        expected_scoring_sha256,
        "official pyUCell scoring.py",
    )
    ranks_sha = _validate_sha(
        pyucell_root / "pyucell" / "ranks.py",
        expected_ranks_sha256,
        "official pyUCell ranks.py",
    )
    cancer_id = str(provenance["cancer_id"]).upper()
    contract = bundle["contract"]
    pathway_ids = _choose_small_pathways(contract.availability, small_pathways)
    cell_total = len(bundle["metadata"])
    selected_local = np.unique(
        np.linspace(0, cell_total - 1, num=min(small_cells, cell_total), dtype=int)
    )
    selected_h5 = bundle["cell_columns"][selected_local]
    expression = bundle["protein_matrix"][:, selected_h5].toarray()
    ranks = rank_expression_ucell(expression, max_rank=max_rank)
    available_lookup = {
        pathway: index for index, pathway in enumerate(contract.available_pathway_ids)
    }
    pathway_rows = np.asarray([available_lookup[pathway] for pathway in pathway_ids])
    v32_scores = score_ucell_pathway_block(
        ranks,
        signature_matrix=contract.signature_matrix[pathway_rows, :].tocsr(),
        signature_gene_counts=contract.signature_gene_counts[pathway_rows],
        missing_gene_counts=contract.missing_gene_counts[pathway_rows],
        max_rank=max_rank,
    )
    r_scores = _independent_r_helper_scores(
        expression,
        protein_ids=bundle["protein_ids"],
        membership=bundle["membership"],
        pathway_ids=pathway_ids,
        max_rank=max_rank,
    )

    official = _load_official_pyucell(pyucell_root)
    grouped = {
        str(pathway): list(group.gene_id.astype(str))
        for pathway, group in bundle["membership"].groupby(
            "pathway_id", sort=False, observed=True
        )
    }
    signatures = {pathway: grouped[pathway] for pathway in pathway_ids}
    official_ranks = official.get_rankings(
        expression.T,
        max_rank=max_rank,
        ties_method="average",
        device="cpu",
    )
    official_indices = official._prepare_sig_indices(
        signatures,
        np.asarray(bundle["protein_ids"]),
        missing_genes="impute",
    )
    py_scores_cell_by_pathway = official._score_chunk(
        official_ranks,
        official_indices,
        w_neg=1.0,
        max_rank=max_rank,
    )
    py_scores = py_scores_cell_by_pathway.T.astype(np.float64)
    corrected_official_ranks, pyucell_boundary_audit = (
        _boundary_corrected_pyucell_rankings(expression, max_rank=max_rank)
    )
    corrected_py_scores = official._score_chunk(
        corrected_official_ranks,
        official_indices,
        w_neg=1.0,
        max_rank=max_rank,
    ).T.astype(np.float64)
    official_rank_dense = official_ranks.toarray().astype(np.float64)
    official_rank_dense[official_rank_dense == 0] = float(max_rank)
    scores_from_official_ranks = score_ucell_pathway_block(
        official_rank_dense,
        signature_matrix=contract.signature_matrix[pathway_rows, :].tocsr(),
        signature_gene_counts=contract.signature_gene_counts[pathway_rows],
        missing_gene_counts=contract.missing_gene_counts[pathway_rows],
        max_rank=max_rank,
    )
    official_scores_from_v32_ranks = official._score_chunk(
        sparse.csr_matrix(ranks),
        official_indices,
        w_neg=1.0,
        max_rank=max_rank,
    ).T.astype(np.float64)
    r_diff = float(np.max(np.abs(v32_scores - r_scores)))
    py_diff = float(np.max(np.abs(v32_scores - py_scores)))
    corrected_py_diff = float(np.max(np.abs(v32_scores - corrected_py_scores)))
    rank_only_score_diff = float(
        np.max(np.abs(v32_scores - scores_from_official_ranks))
    )
    scoring_only_diff_on_v32_ranks = float(
        np.max(np.abs(v32_scores - official_scores_from_v32_ranks))
    )
    scoring_only_diff_on_official_ranks = float(
        np.max(np.abs(scores_from_official_ranks - py_scores))
    )
    rank_matrix_max_abs_diff = float(
        np.max(np.abs(ranks - official_rank_dense))
    )
    rank_delta = np.abs(ranks - official_rank_dense)
    max_rank_coordinate = np.unravel_index(
        int(np.argmax(rank_delta)), rank_delta.shape
    )
    debug_gene_index, debug_cell_index = map(int, max_rank_coordinate)
    zero_rank_mismatch_count = int(
        ((expression == 0) & (ranks < float(max_rank))).sum()
    )
    detected_dropped_by_pyucell_count = int(
        ((expression > 0) & (official_rank_dense == float(max_rank))).sum()
    )
    r_pass = r_diff <= 1e-12
    # pyUCell 0.7.3 stores average ranks as int32; the R implementation keeps
    # fractional average ties.  Bound and report that implementation drift.
    py_tolerance = 1e-3
    py_pass = py_diff <= py_tolerance
    corrected_py_pass = corrected_py_diff <= py_tolerance
    official_pyucell_gate_pass = bool(
        py_pass
        or (
            pyucell_boundary_audit["pinned_pyucell_boundary_bug_triggered"]
            and corrected_py_pass
            and scoring_only_diff_on_v32_ranks <= 1e-6
            and scoring_only_diff_on_official_ranks <= 1e-6
        )
    )
    passed = bool(
        r_pass
        and official_pyucell_gate_pass
        and count_audit["status"] == "PASS"
    )

    availability_path = staging_root / "ucell_pathway_availability.parquet"
    contract.availability.to_parquet(availability_path, index=False, compression="zstd")
    selected_meta = bundle["metadata"].iloc[selected_local].reset_index(drop=True)
    score_frame = pd.DataFrame(
        {
            "dataset_id": np.repeat(selected_meta.dataset_id.astype(str).to_numpy(), len(pathway_ids)),
            "cancer_id": cancer_id,
            "cell_id": np.repeat(selected_meta.cell_id.astype(str).to_numpy(), len(pathway_ids)),
            "patient_id": np.repeat(selected_meta.patient_id.astype(str).to_numpy(), len(pathway_ids)),
            "cell_type_major": np.repeat(selected_meta.cell_type_major.astype(str).to_numpy(), len(pathway_ids)),
            "pathway_id": np.tile(np.asarray(pathway_ids), len(selected_meta)),
            "ucell_score": v32_scores.T.reshape(-1).astype(np.float32),
        }
    )
    score_path = staging_root / f"{cancer_id}_SMALL_CHUNK_SCORES.parquet"
    score_frame.to_parquet(score_path, index=False, compression="zstd")
    count_path = staging_root / "PRESENT_MISSING_COUNT_AUDIT.json"
    _atomic_json(count_path, count_audit)
    parity = {
        "status": "PASS" if passed else "FAIL",
        "cells": int(len(selected_meta)),
        "pathways": int(len(pathway_ids)),
        "score_rows": int(len(score_frame)),
        "selected_cell_ids": selected_meta.cell_id.astype(str).tolist(),
        "selected_pathway_ids": pathway_ids,
        "present_missing_count_audit_pass": count_audit["status"] == "PASS",
        "v32_vs_pinned_helper_functions_r_max_abs_diff": r_diff,
        "v32_vs_pinned_helper_functions_r_tolerance": 1e-12,
        "v32_vs_pinned_helper_functions_r_pass": r_pass,
        "v32_vs_official_pyucell_0_7_3_max_abs_diff": py_diff,
        "v32_vs_official_pyucell_0_7_3_tolerance": py_tolerance,
        "v32_vs_official_pyucell_0_7_3_pass": py_pass,
        "v32_vs_boundary_corrected_pyucell_0_7_3_max_abs_diff": (
            corrected_py_diff
        ),
        "v32_vs_boundary_corrected_pyucell_0_7_3_tolerance": py_tolerance,
        "v32_vs_boundary_corrected_pyucell_0_7_3_pass": corrected_py_pass,
        "official_pyucell_gate_pass": official_pyucell_gate_pass,
        "official_pyucell_boundary_audit": pyucell_boundary_audit,
        "diagnostic_rank_only_score_max_abs_diff": rank_only_score_diff,
        "diagnostic_scoring_only_on_v32_ranks_max_abs_diff": (
            scoring_only_diff_on_v32_ranks
        ),
        "diagnostic_scoring_only_on_official_ranks_max_abs_diff": (
            scoring_only_diff_on_official_ranks
        ),
        "diagnostic_rank_matrix_max_abs_diff": rank_matrix_max_abs_diff,
        "diagnostic_rank_max_coordinate": {
            "gene_index": debug_gene_index,
            "cell_index": debug_cell_index,
            "gene_id": str(bundle["protein_ids"][debug_gene_index]),
            "cell_id": str(selected_meta.cell_id.iloc[debug_cell_index]),
            "expression": float(expression[debug_gene_index, debug_cell_index]),
            "v32_rank": float(ranks[debug_gene_index, debug_cell_index]),
            "pyucell_rank_after_zero_tail_fill": float(
                official_rank_dense[debug_gene_index, debug_cell_index]
            ),
        },
        "diagnostic_zero_expression_ranked_below_max_count": (
            zero_rank_mismatch_count
        ),
        "diagnostic_detected_gene_dropped_by_pyucell_count": (
            detected_dropped_by_pyucell_count
        ),
        "diagnostic_detected_genes_per_selected_cell": (
            (expression > 0).sum(axis=0).astype(int).tolist()
        ),
        "pyucell_fractional_average_tie_note": (
            "pyUCell_0.7.3_casts_average_ranks_to_int32; R_HelperFunctions_keeps_fractional_ties"
        ),
        "official_pyucell_scoring_sha256": scoring_sha,
        "official_pyucell_ranks_sha256": ranks_sha,
        "score_min": float(v32_scores.min()),
        "score_max": float(v32_scores.max()),
        "full_cell_level_started": False,
        "full_hnsc_started": False if cancer_id == "HNSC" else None,
        "pseudotime_numeric_output": False,
        "pseudotime_status": "OUT_OF_SCOPE_SEPARATE_DIAGNOSTIC_RUNNER",
    }
    parity_path = staging_root / "SMALL_CHUNK_PARITY.json"
    _atomic_json(parity_path, parity)
    if not passed:
        raise CellLevelPilotError(f"Small-chunk parity failed: {parity}")
    manifest, manifest_sha = _write_manifest(
        staging_root,
        [
            availability_path.name,
            score_path.name,
            count_path.name,
            parity_path.name,
        ],
    )
    handoff = provenance | {
        "status": "PASS",
        "stage": f"{cancer_id}_SMALL_CHUNK_PARITY_ONLY",
        "small_chunk_parity_path": str(published_root / parity_path.name),
        "small_chunk_parity_sha256": artifact_sha256(parity_path),
        "file_manifest_path": str(published_root / manifest.name),
        "file_manifest_sha256": manifest_sha,
        "full_cell_level_started": False,
        "full_hnsc_started": False if cancer_id == "HNSC" else None,
        "pseudotime_numeric_output": False,
        "pseudotime_status": "OUT_OF_SCOPE_SEPARATE_DIAGNOSTIC_RUNNER",
    }
    handoff_path = staging_root / "SMALL_CHUNK_HANDOFF.json"
    _atomic_json(handoff_path, handoff)
    status = handoff | {
        "handoff_path": str(published_root / handoff_path.name),
        "handoff_sha256": artifact_sha256(handoff_path),
    }
    _atomic_json(staging_root / "STATUS.json", status)
    return status


def _typed_pseudotime_outputs(
    root: Path, metadata: pd.DataFrame, pathway_ids: pd.Series
) -> list[str]:
    cells = metadata.loc[
        :, ["dataset_id", "cancer_id", "cell_id", "patient_id", "cell_type_major"]
    ].copy()
    cells["pseudotime_available"] = False
    cells["pseudotime_value"] = pd.Series(
        pd.array([pd.NA] * len(cells), dtype="Float64"), index=cells.index
    )
    cells["unavailable_reason"] = PSEUDOTIME_REASON
    cell_path = root / "pseudotime_cell_availability.parquet"
    cells.to_parquet(cell_path, index=False, compression="zstd")
    pathways = pd.DataFrame({"pathway_id": pathway_ids.astype(str).to_numpy()})
    pathways["pseudotime_available"] = False
    pathways["pseudotime_effect"] = pd.Series(
        pd.array([pd.NA] * len(pathways), dtype="Float64"), index=pathways.index
    )
    pathways["unavailable_reason"] = PSEUDOTIME_REASON
    pathway_path = root / "pseudotime_pathway_availability.parquet"
    pathways.to_parquet(pathway_path, index=False, compression="zstd")
    return [cell_path.name, pathway_path.name]


def _small_pyucell_gate_passes(small: dict[str, Any]) -> bool:
    """Re-evaluate the strict pyUCell gate recorded by the small run.

    A pinned pyUCell 0.7.3 feature-index truncation bug is allowed only when
    the boundary-corrected reconstruction and both scoring-only comparisons
    pass their independently recorded tolerances.  Otherwise the unmodified
    pinned implementation itself must pass.
    """

    if small.get("official_pyucell_gate_pass") is not True:
        return False
    boundary = small.get("official_pyucell_boundary_audit")
    if not isinstance(boundary, dict):
        return False
    triggered = boundary.get("pinned_pyucell_boundary_bug_triggered")
    if triggered is True:
        try:
            v32_scoring_diff = float(
                small["diagnostic_scoring_only_on_v32_ranks_max_abs_diff"]
            )
            official_scoring_diff = float(
                small["diagnostic_scoring_only_on_official_ranks_max_abs_diff"]
            )
            triggered_count = int(boundary["triggered_cell_count"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return False
        return bool(
            small.get("v32_vs_boundary_corrected_pyucell_0_7_3_pass") is True
            and boundary.get("correction")
            == "REMOVE_FEATURE_INDEX_TRUNCATION_AFTER_AVERAGE_TIE_RANK_GATE"
            and boundary.get("rank_dtype_retained_from_pinned_pyucell") == "int32"
            and triggered_count > 0
            and v32_scoring_diff <= 1e-6
            and official_scoring_diff <= 1e-6
        )
    if triggered is False:
        return bool(
            small.get("v32_vs_official_pyucell_0_7_3_pass") is True
            and boundary.get("triggered_cell_count") == 0
        )
    return False


def run_full(
    *,
    staging_root: Path,
    published_root: Path,
    provenance: dict[str, Any],
    bundle: dict[str, Any],
    count_audit: dict[str, Any],
    small_parity_path: Path,
    expected_small_parity_sha256: str,
    max_rank: int,
    chunk_size: int,
) -> dict[str, Any]:
    cancer_id = str(provenance["cancer_id"]).upper()
    small_sha = _validate_sha(
        small_parity_path,
        expected_small_parity_sha256,
        f"{cancer_id} small-chunk parity",
    )
    small = _load_json(small_parity_path)
    full_started = small.get("full_cell_level_started")
    if full_started is None and cancer_id == "HNSC":
        full_started = small.get("full_hnsc_started")
    required_small = (
        small.get("status") == "PASS"
        and small.get("present_missing_count_audit_pass") is True
        and small.get("v32_vs_pinned_helper_functions_r_pass") is True
        and _small_pyucell_gate_passes(small)
        and full_started is False
    )
    if not required_small:
        raise CellLevelPilotError(
            f"Small-chunk artifact does not unlock full {cancer_id}"
        )
    if count_audit["status"] != "PASS":
        raise CellLevelPilotError("Present/missing count gate did not pass")
    contract = bundle["contract"]
    metadata = bundle["metadata"]
    cell_columns = bundle["cell_columns"]
    availability = contract.availability.reset_index(drop=True)
    available_pathway_rows = np.flatnonzero(
        availability.ucell_available.astype(bool).to_numpy()
    )
    if tuple(
        availability.iloc[available_pathway_rows].pathway_id.astype(str)
    ) != contract.available_pathway_ids:
        raise CellLevelPilotError("Available pathway row order disagrees with contract")
    output_parts = staging_root / "cell_level_ucell"
    output_parts.mkdir()
    group_frame = (
        metadata.loc[:, ["patient_id", "cell_type_major"]]
        .astype(str)
        .drop_duplicates()
        .sort_values(["patient_id", "cell_type_major"], kind="mergesort")
        .reset_index(drop=True)
    )
    group_lookup = {
        (row.patient_id, row.cell_type_major): index
        for index, row in group_frame.iterrows()
    }
    cell_group_codes = np.asarray(
        [
            group_lookup[(str(row.patient_id), str(row.cell_type_major))]
            for row in metadata.itertuples(index=False)
        ],
        dtype=np.int64,
    )
    available_pathway_count = len(contract.available_pathway_ids)
    coverage_pathway_count = len(availability)
    group_sums = np.zeros(
        (available_pathway_count, len(group_frame)), dtype=np.float64
    )
    group_counts = np.zeros(len(group_frame), dtype=np.int64)
    score_min = np.inf
    score_max = -np.inf
    coverage_rows = 0
    numeric_score_rows = 0
    part_names: list[str] = []
    for part_index, start in enumerate(range(0, len(metadata), chunk_size)):
        stop = min(start + chunk_size, len(metadata))
        h5_columns = cell_columns[start:stop]
        expression = bundle["protein_matrix"][:, h5_columns].toarray()
        ranks = rank_expression_ucell(expression, max_rank=max_rank)
        scores = score_ucell_pathway_block(
            ranks,
            signature_matrix=contract.signature_matrix,
            signature_gene_counts=contract.signature_gene_counts,
            missing_gene_counts=contract.missing_gene_counts,
            max_rank=max_rank,
        )
        score_min = min(score_min, float(scores.min()))
        score_max = max(score_max, float(scores.max()))
        block_meta = metadata.iloc[start:stop].reset_index(drop=True)
        scores_all = np.full(
            (coverage_pathway_count, len(block_meta)), np.nan, dtype=np.float32
        )
        scores_all[available_pathway_rows, :] = scores.astype(np.float32)
        frame = pd.DataFrame(
            {
                "dataset_id": np.repeat(
                    block_meta.dataset_id.astype(str).to_numpy(),
                    coverage_pathway_count,
                ),
                "cancer_id": cancer_id,
                "cell_id": np.repeat(
                    block_meta.cell_id.astype(str).to_numpy(),
                    coverage_pathway_count,
                ),
                "patient_id": np.repeat(
                    block_meta.patient_id.astype(str).to_numpy(),
                    coverage_pathway_count,
                ),
                "cell_type_major": np.repeat(
                    block_meta.cell_type_major.astype(str).to_numpy(),
                    coverage_pathway_count,
                ),
                "pathway_id": np.tile(
                    availability.pathway_id.astype(str).to_numpy(), len(block_meta)
                ),
                "ucell_available": np.tile(
                    availability.ucell_available.astype(bool).to_numpy(),
                    len(block_meta),
                ),
                "unavailable_reason": np.tile(
                    availability.unavailable_reason.to_numpy(), len(block_meta)
                ),
            }
        )
        frame["ucell_score"] = pd.Series(
            pd.array(scores_all.T.reshape(-1), dtype="Float32"), index=frame.index
        )
        if not frame.loc[frame.ucell_available, "ucell_score"].notna().all():
            raise CellLevelPilotError("Available pathway emitted a null UCell score")
        if not frame.loc[~frame.ucell_available, "ucell_score"].isna().all():
            raise CellLevelPilotError("Typed-unavailable pathway emitted a numeric score")
        relative = f"cell_level_ucell/part-{part_index:05d}.parquet"
        frame.to_parquet(staging_root / relative, index=False, compression="zstd")
        part_names.append(relative)
        coverage_rows += len(frame)
        numeric_score_rows += int(scores.size)
        codes = cell_group_codes[start:stop]
        for code in np.unique(codes):
            mask = codes == code
            group_sums[:, code] += scores[:, mask].sum(axis=1)
            group_counts[code] += int(mask.sum())
    expected_coverage_rows = int(len(metadata) * coverage_pathway_count)
    expected_numeric_rows = int(len(metadata) * available_pathway_count)
    expected_unavailable_rows = expected_coverage_rows - expected_numeric_rows
    if (
        coverage_rows != expected_coverage_rows
        or numeric_score_rows != expected_numeric_rows
        or (group_counts <= 0).any()
    ):
        raise CellLevelPilotError(
            f"Full {cancer_id} output row/group count mismatch"
        )

    availability_path = staging_root / "ucell_pathway_availability.parquet"
    contract.availability.to_parquet(availability_path, index=False, compression="zstd")
    available_group_means = group_sums / group_counts[None, :]
    all_group_means = np.full(
        (coverage_pathway_count, len(group_frame)), np.nan, dtype=np.float32
    )
    all_group_means[available_pathway_rows, :] = available_group_means.astype(
        np.float32
    )
    aggregate = pd.DataFrame(
        {
            "pathway_id": np.repeat(
                availability.pathway_id.astype(str).to_numpy(), len(group_frame)
            ),
            "patient_id": np.tile(
                group_frame.patient_id.astype(str).to_numpy(), coverage_pathway_count
            ),
            "cell_type_major": np.tile(
                group_frame.cell_type_major.astype(str).to_numpy(),
                coverage_pathway_count,
            ),
            "cell_count": np.tile(group_counts, coverage_pathway_count),
            "ucell_available": np.repeat(
                availability.ucell_available.astype(bool).to_numpy(), len(group_frame)
            ),
            "unavailable_reason": np.repeat(
                availability.unavailable_reason.to_numpy(), len(group_frame)
            ),
        }
    )
    aggregate["ucell_score_mean"] = pd.Series(
        pd.array(all_group_means.reshape(-1), dtype="Float32"),
        index=aggregate.index,
    )
    aggregate.insert(0, "cancer_id", cancer_id)
    aggregate.insert(0, "dataset_id", str(metadata.dataset_id.iloc[0]))
    aggregate_path = staging_root / "ucell_donor_celltype.parquet"
    aggregate.to_parquet(aggregate_path, index=False, compression="zstd")
    count_path = staging_root / "PRESENT_MISSING_COUNT_AUDIT.json"
    _atomic_json(count_path, count_audit)
    if cancer_id == "HNSC":
        pseudotime_names = _typed_pseudotime_outputs(
            staging_root, metadata, contract.availability.pathway_id
        )
        pseudotime_status = "TYPED_UNAVAILABLE_NO_EXPLICIT_ROOT"
        pseudotime_reason = PSEUDOTIME_REASON
        pseudotime_null_values_only = True
    else:
        pseudotime_names = []
        pseudotime_status = "OUT_OF_SCOPE_SEPARATE_DIAGNOSTIC_RUNNER"
        pseudotime_reason = "NOT_COMPUTED_IN_CELL_LEVEL_UCELL_RUNNER"
        pseudotime_null_values_only = False
    relative_paths = part_names + [
        availability_path.name,
        aggregate_path.name,
        count_path.name,
        *pseudotime_names,
    ]
    manifest, manifest_sha = _write_manifest(staging_root, relative_paths)
    lineage = provenance | {
        "status": "PASS",
        "stage": f"{cancer_id}_FRESH_CELL_LEVEL_UCELL_LINEAGE",
        "full_cell_level_started": True,
        "full_cell_level_completed": True,
        "full_hnsc_started": True if cancer_id == "HNSC" else None,
        "full_hnsc_completed": True if cancer_id == "HNSC" else None,
        "small_chunk_parity_path": str(small_parity_path),
        "small_chunk_parity_sha256": small_sha,
        "file_manifest_path": str(published_root / manifest.name),
        "file_manifest_sha256": manifest_sha,
        "coverage_pathways": int(coverage_pathway_count),
        "available_pathways": int(available_pathway_count),
        "typed_unavailable_pathways": int(
            coverage_pathway_count - available_pathway_count
        ),
        "pseudotime_numeric_output": False,
        "pseudotime_status": pseudotime_status,
    }
    lineage_path = staging_root / "LINEAGE.json"
    _atomic_json(lineage_path, lineage)
    lineage_sha = artifact_sha256(lineage_path)
    handoff = provenance | {
        "status": "PASS",
        "stage": f"{cancer_id}_FRESH_CELL_LEVEL_UCELL",
        "full_cell_level_started": True,
        "full_cell_level_completed": True,
        "full_hnsc_started": True if cancer_id == "HNSC" else None,
        "full_hnsc_completed": True if cancer_id == "HNSC" else None,
        "small_chunk_parity_path": str(small_parity_path),
        "small_chunk_parity_sha256": small_sha,
        "cells": int(len(metadata)),
        "pathways_total_coverage": int(coverage_pathway_count),
        "pathways_available": int(available_pathway_count),
        "pathways_unavailable": int(
            coverage_pathway_count - available_pathway_count
        ),
        "cell_level_coverage_rows": int(coverage_rows),
        "cell_level_coverage_rows_expected": expected_coverage_rows,
        "cell_level_numeric_score_rows": int(numeric_score_rows),
        "cell_level_numeric_score_rows_expected": expected_numeric_rows,
        "cell_level_typed_unavailable_rows": int(expected_unavailable_rows),
        "cell_level_partitions": len(part_names),
        "donor_celltype_groups": int(len(group_frame)),
        "donor_celltype_rows": int(len(aggregate)),
        "ucell_score_min": score_min,
        "ucell_score_max": score_max,
        "pseudotime_numeric_output": False,
        "pseudotime_status": pseudotime_status,
        "pseudotime_unavailable_reason": pseudotime_reason,
        "pseudotime_null_values_only": pseudotime_null_values_only,
        "pseudotime_outputs_written": bool(pseudotime_names),
        "unavailable_values_filled_with_zero_or_half": False,
        "file_manifest_path": str(published_root / manifest.name),
        "file_manifest_sha256": manifest_sha,
        "lineage_path": str(published_root / lineage_path.name),
        "lineage_sha256": lineage_sha,
    }
    handoff_path = staging_root / "CELL_LEVEL_HANDOFF.json"
    _atomic_json(handoff_path, handoff)
    status = handoff | {
        "handoff_path": str(published_root / handoff_path.name),
        "handoff_sha256": artifact_sha256(handoff_path),
    }
    status_path = staging_root / "STATUS.json"
    _atomic_json(status_path, status)
    success = {
        "format": "CC_HHGT_V3_2_SINGLE_CELL_CELL_LEVEL_SUCCESS_V1",
        "status": "SUCCESS",
        "cancer_id": cancer_id,
        "process_id": int(os.getpid()),
        "full_cell_level_started": True,
        "full_cell_level_completed": True,
        "full_hnsc_started": True if cancer_id == "HNSC" else None,
        "full_hnsc_completed": True if cancer_id == "HNSC" else None,
        "status_path": str(published_root / status_path.name),
        "status_sha256": artifact_sha256(status_path),
        "handoff_path": str(published_root / handoff_path.name),
        "handoff_sha256": artifact_sha256(handoff_path),
        "lineage_path": str(published_root / lineage_path.name),
        "lineage_sha256": lineage_sha,
        "file_manifest_path": str(published_root / manifest.name),
        "file_manifest_sha256": manifest_sha,
        "cells": int(len(metadata)),
        "pathways_total_coverage": int(coverage_pathway_count),
        "pathways_available": int(available_pathway_count),
        "pathways_typed_unavailable": int(
            coverage_pathway_count - available_pathway_count
        ),
        "cell_level_coverage_rows": int(coverage_rows),
        "cell_level_numeric_score_rows": int(numeric_score_rows),
        "cell_level_typed_unavailable_rows": int(expected_unavailable_rows),
        "pseudotime_numeric_output": False,
        "pseudotime_status": pseudotime_status,
        "unavailable_values_filled_with_zero_or_half": False,
        "single_cell_module_complete": False,
        "release_ready": False,
    }
    success_path = staging_root / "SUCCESS.json"
    _atomic_json(success_path, success)
    status["success_path"] = str(published_root / success_path.name)
    status["success_sha256"] = artifact_sha256(success_path)
    return status


def run(
    args: argparse.Namespace,
    *,
    generalized_formal_cancer: bool = False,
) -> dict[str, Any]:
    output = args.output_root.resolve()
    if output.exists():
        raise CellLevelPilotError(f"Output reuse is forbidden: {output}")
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise CellLevelPilotError(f"Temporary output reuse is forbidden: {temporary}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    preflight_path = args.preflight_json.resolve()
    official_parity_path = args.official_parity_json.resolve()
    preflight, preflight_sha, _, official_sha = _validate_gates(
        preflight_path=preflight_path,
        expected_preflight_sha256=args.expected_preflight_sha256,
        official_parity_path=official_parity_path,
        expected_official_parity_sha256=args.expected_official_parity_sha256,
        expected_cancer_id=None if generalized_formal_cancer else "HNSC",
        require_pseudotime_unavailable=not generalized_formal_cancer,
    )
    bundle = _load_fresh_inputs(preflight, max_rank=args.max_rank)
    count_audit = _count_and_gate_audit(bundle, preflight)
    provenance = _base_provenance(
        mode=args.mode,
        preflight=preflight,
        preflight_path=preflight_path,
        preflight_sha=preflight_sha,
        official_parity_path=official_parity_path,
        official_parity_sha=official_sha,
        bundle=bundle,
        max_rank=args.max_rank,
    )
    if args.mode == "small":
        if args.official_pyucell_root is None:
            raise CellLevelPilotError("Small mode requires official pyUCell source")
        result = run_small(
            staging_root=temporary,
            published_root=output,
            provenance=provenance,
            bundle=bundle,
            count_audit=count_audit,
            pyucell_root=args.official_pyucell_root.resolve(),
            expected_scoring_sha256=args.expected_pyucell_scoring_sha256,
            expected_ranks_sha256=args.expected_pyucell_ranks_sha256,
            max_rank=args.max_rank,
            small_cells=args.small_cells,
            small_pathways=args.small_pathways,
        )
    else:
        if args.small_parity_json is None or args.expected_small_parity_sha256 is None:
            raise CellLevelPilotError("Full mode requires a pinned passing small parity")
        result = run_full(
            staging_root=temporary,
            published_root=output,
            provenance=provenance,
            bundle=bundle,
            count_audit=count_audit,
            small_parity_path=args.small_parity_json.resolve(),
            expected_small_parity_sha256=args.expected_small_parity_sha256,
            max_rank=args.max_rank,
            chunk_size=args.chunk_size,
        )
    os.replace(temporary, output)
    result["output_root"] = str(output)
    return result


def run_general(args: argparse.Namespace) -> dict[str, Any]:
    """Run the same frozen UCell contract for any of the 17 formal cancers.

    Pseudotime is deliberately out of scope here and remains owned by the
    separately audited diagnostic runner.  This entry point is not used by the
    historical HNSC-only CLI, so the completed HNSC pilot remains immutable.
    """

    return run(args, generalized_formal_cancer=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generalized-formal-cancer",
        action="store_true",
        help=(
            "Run the frozen UCell contract for any of the 17 formally gated "
            "cancers. Without this flag the historical CLI remains HNSC-only."
        ),
    )
    parser.add_argument("--mode", choices=("small", "full"), required=True)
    parser.add_argument("--preflight-json", required=True, type=Path)
    parser.add_argument("--expected-preflight-sha256", required=True)
    parser.add_argument("--official-parity-json", required=True, type=Path)
    parser.add_argument("--expected-official-parity-sha256", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--max-rank", type=int, default=1500)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--small-cells", type=int, default=16)
    parser.add_argument("--small-pathways", type=int, default=32)
    parser.add_argument("--official-pyucell-root", type=Path)
    parser.add_argument("--expected-pyucell-scoring-sha256")
    parser.add_argument("--expected-pyucell-ranks-sha256")
    parser.add_argument("--small-parity-json", type=Path)
    parser.add_argument("--expected-small-parity-sha256")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_general(args) if args.generalized_formal_cancer else run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
