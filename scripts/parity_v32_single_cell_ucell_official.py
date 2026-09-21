#!/usr/bin/env python3
"""Offline parity against pinned official UCell R source and pyUCell wheel."""
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
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.input_lineage import artifact_sha256  # noqa: E402
from cc_hhgt.v32.single_cell_cell_level import (  # noqa: E402
    UCELL_FORMULA,
    UCELL_MISSING_MEMBER_POLICY,
    rank_expression_ucell,
    score_ucell_from_capped_ranks,
)


PARITY_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_UCELL_OFFICIAL_PARITY_V1"
OFFICIAL_R_URL = (
    "https://raw.githubusercontent.com/carmonalab/UCell/master/R/HelperFunctions.R"
)


class UCellParityError(RuntimeError):
    """Raised when official-source binding or numeric parity fails."""


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _validate_sha(path: Path, expected: str, role: str) -> str:
    if not path.is_file():
        raise UCellParityError(f"Missing {role}: {path}")
    observed = artifact_sha256(path)
    if observed.lower() != str(expected).lower():
        raise UCellParityError(f"{role} SHA drift: {observed} != {expected}")
    return observed


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise UCellParityError(f"Cannot load official module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_official_pyucell(pyucell_root: Path):
    """Load the wheel-extracted modules without installing dependencies."""

    package_root = pyucell_root / "pyucell"
    if not package_root.is_dir():
        raise UCellParityError(f"Official pyUCell package is absent: {package_root}")

    # Official modules only need AnnData for annotations/type checks on this path.
    # A minimal placeholder avoids installing or mutating the production runtime.
    anndata = types.ModuleType("anndata")
    anndata.AnnData = type("AnnData", (), {})
    sys.modules["anndata"] = anndata
    package = types.ModuleType("pyucell")
    package.__path__ = [str(package_root)]
    sys.modules["pyucell"] = package
    _load_module("pyucell._torch_utils", package_root / "_torch_utils.py")
    _load_module("pyucell.ranks", package_root / "ranks.py")
    return _load_module("pyucell.scoring", package_root / "scoring.py")


def _r_helper_reference(
    expression_cell_by_gene: np.ndarray,
    *,
    genes: list[str],
    signature: list[str],
    max_rank: int,
) -> np.ndarray:
    """Independent direct translation of pinned HelperFunctions.R ``u_stat``."""

    # HelperFunctions.R ranks descending with data.table::frankv and average ties.
    ranks = rankdata(-expression_cell_by_gene.T, method="average", axis=0)
    ranks = np.minimum(ranks, float(max_rank))
    gene_index = {gene: index for index, gene in enumerate(genes)}
    present = [gene_index[gene] for gene in signature if gene in gene_index]
    missing = len(signature) - len(present)
    rank_sum = np.full(expression_cell_by_gene.shape[0], missing * max_rank, dtype=float)
    if present:
        rank_sum += ranks[present, :].sum(axis=0)
    n = len(signature)
    rank_sum_min = n * (n + 1) / 2.0
    return 1.0 - (rank_sum - rank_sum_min) / (
        n * max_rank - rank_sum_min
    )


def run_parity(
    *,
    pyucell_wheel: str | Path,
    pyucell_root: str | Path,
    helper_functions_r: str | Path,
    expected_wheel_sha256: str,
    expected_scoring_sha256: str,
    expected_ranks_sha256: str,
    expected_helper_sha256: str,
) -> dict[str, Any]:
    wheel = Path(pyucell_wheel).resolve()
    package_root = Path(pyucell_root).resolve()
    helper_path = Path(helper_functions_r).resolve()
    wheel_sha = _validate_sha(wheel, expected_wheel_sha256, "pyUCell wheel")
    scoring_path = package_root / "pyucell" / "scoring.py"
    ranks_path = package_root / "pyucell" / "ranks.py"
    scoring_sha = _validate_sha(
        scoring_path, expected_scoring_sha256, "pyUCell scoring.py"
    )
    ranks_sha = _validate_sha(ranks_path, expected_ranks_sha256, "pyUCell ranks.py")
    helper_sha = _validate_sha(
        helper_path, expected_helper_sha256, "UCell HelperFunctions.R"
    )

    helper_source = helper_path.read_text(encoding="utf-8")
    required_r_contract = (
        "len_sig <- length(gene_idx)",
        "rank_sum <- rep(length(missing_idx) * maxRank, ncells)",
        "rank_sum_min <- len_sig*(len_sig + 1)/2",
        "ucell_score <- 1 - (rank_sum-rank_sum_min)/(len_sig*maxRank - rank_sum_min)",
        'data_to_ranks_data_table <- function(data, ties.method="average")',
        "frankv(x,ties.method=ties.method,order=c(-1L))",
    )
    absent = [snippet for snippet in required_r_contract if snippet not in helper_source]
    if absent:
        raise UCellParityError(f"Pinned HelperFunctions.R contract drift: {absent}")

    official = _load_official_pyucell(package_root)
    genes = [f"G{index}" for index in range(1, 9)]
    cells = ["C1", "C2", "C3", "C4"]
    expression = np.asarray(
        [
            [10, 8, 8, 8, 4, 3, 0, 0],
            [2, 9, 7, 7, 7, 4, 3, 0],
            [6, 5, 4, 3, 2, 1, 0.5, 0],
            [4, 3, 3, 3, 2, 1, 0, 0],
        ],
        dtype=np.float64,
    )
    max_rank = 5
    signatures = {
        "SIG_MISSING": ["G1", "G2", "NOT_IN_MATRIX"],
        "SIG_THREE_WAY_TIE": ["G2", "G3", "G4"],
        "SIG_TAIL": ["G5", "G6", "G7"],
    }

    official_ranks = official.get_rankings(
        expression,
        max_rank=max_rank,
        ties_method="average",
        device="cpu",
    )
    official_indices = official._prepare_sig_indices(
        signatures,
        np.asarray(genes),
        missing_genes="impute",
    )
    official_scores = official._score_chunk(
        official_ranks,
        official_indices,
        w_neg=1.0,
        max_rank=max_rank,
    )

    our_ranks = rank_expression_ucell(expression.T, max_rank=max_rank)
    gene_index = {gene: index for index, gene in enumerate(genes)}
    our_columns: list[np.ndarray] = []
    r_columns: list[np.ndarray] = []
    for signature in signatures.values():
        present = [gene_index[gene] for gene in signature if gene in gene_index]
        our_columns.append(
            score_ucell_from_capped_ranks(
                our_ranks,
                present_gene_indices=present,
                signature_gene_count=len(signature),
                max_rank=max_rank,
            )
        )
        r_columns.append(
            _r_helper_reference(
                expression,
                genes=genes,
                signature=signature,
                max_rank=max_rank,
            )
        )
    our_scores = np.column_stack(our_columns)
    r_scores = np.column_stack(r_columns)
    r_max_abs_diff = float(np.max(np.abs(our_scores - r_scores)))
    py_max_abs_diff = float(np.max(np.abs(our_scores - official_scores)))
    pass_r = r_max_abs_diff <= 1e-12
    pass_py = py_max_abs_diff <= 1e-6
    passed = bool(pass_r and pass_py)

    result = {
        "format": PARITY_FORMAT,
        "status": "PASS" if passed else "FAIL",
        "algorithm_formula": UCELL_FORMULA,
        "complete_signature_length_used": True,
        "matrix_missing_member_policy": UCELL_MISSING_MEMBER_POLICY,
        "matrix_missing_member_is_numeric_expression_imputation": False,
        "ties_method": "average",
        "max_rank": max_rank,
        "official_r_reference": {
            "url": OFFICIAL_R_URL,
            "path": str(helper_path),
            "sha256": helper_sha,
            "required_contract_snippets_present": True,
        },
        "official_pyucell_reference": {
            "distribution": "pyucell==0.7.3",
            "wheel_path": str(wheel),
            "wheel_sha256": wheel_sha,
            "extracted_root": str(package_root),
            "scoring_sha256": scoring_sha,
            "ranks_sha256": ranks_sha,
            "missing_genes_argument": "impute",
            "installed_into_runtime": False,
        },
        "fixture": {
            "cells": cells,
            "genes": genes,
            "expression_cell_by_gene": expression.tolist(),
            "signatures": signatures,
            "contains_three_way_nonzero_ties": True,
            "contains_matrix_missing_signature_member": True,
            "all_signatures_below_max_rank": True,
        },
        "scores": {
            "columns": list(signatures),
            "v32": our_scores.tolist(),
            "official_r_helper_translation": r_scores.tolist(),
            "official_pyucell_0_7_3": official_scores.astype(float).tolist(),
        },
        "parity": {
            "v32_vs_official_r_max_abs_diff": r_max_abs_diff,
            "v32_vs_official_r_tolerance": 1e-12,
            "v32_vs_official_r_pass": pass_r,
            "v32_vs_official_pyucell_max_abs_diff": py_max_abs_diff,
            "v32_vs_official_pyucell_tolerance": 1e-6,
            "v32_vs_official_pyucell_pass": pass_py,
        },
        "full_hnsc_pilot_started": False,
        "training_started": False,
        "single_cell_module_complete": False,
        "release_ready": False,
    }
    if not passed:
        raise UCellParityError(json.dumps(result["parity"], sort_keys=True))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pyucell-wheel", required=True, type=Path)
    parser.add_argument("--pyucell-root", required=True, type=Path)
    parser.add_argument("--helper-functions-r", required=True, type=Path)
    parser.add_argument("--expected-wheel-sha256", required=True)
    parser.add_argument("--expected-scoring-sha256", required=True)
    parser.add_argument("--expected-ranks-sha256", required=True)
    parser.add_argument("--expected-helper-sha256", required=True)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = args.output_json.resolve()
    if output.exists():
        raise UCellParityError(f"Parity output reuse is forbidden: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = run_parity(
        pyucell_wheel=args.pyucell_wheel,
        pyucell_root=args.pyucell_root,
        helper_functions_r=args.helper_functions_r,
        expected_wheel_sha256=args.expected_wheel_sha256,
        expected_scoring_sha256=args.expected_scoring_sha256,
        expected_ranks_sha256=args.expected_ranks_sha256,
        expected_helper_sha256=args.expected_helper_sha256,
    )
    _atomic_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
