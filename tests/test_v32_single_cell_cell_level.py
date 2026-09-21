from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cc_hhgt.v32.single_cell_cell_level import (
    NO_MEMBER_PROTEIN_IN_MATRIX,
    SIGNATURE_AT_OR_ABOVE_MAX_RANK,
    UCellContractError,
    build_ucell_pathway_contract,
    evaluate_ucell_signature_domain,
    rank_expression_ucell,
    score_ucell_from_capped_ranks,
    score_ucell_pathway_block,
)


def _load_cell_level_runner():
    script = Path(__file__).resolve().parents[1] / "scripts" / (
        "run_v32_single_cell_cell_level_pilot.py"
    )
    spec = importlib.util.spec_from_file_location("v32_cell_level_runner_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ucell_complete_signature_counts_missing_members_at_max_rank() -> None:
    ranks = np.asarray(
        [
            [1.0, 5.0],
            [3.0, 5.0],
            [2.0, 4.0],
        ]
    )
    observed = score_ucell_from_capped_ranks(
        ranks,
        present_gene_indices=[0, 1],
        signature_gene_count=3,
        max_rank=5,
    )
    # Cell 1: rank_sum=1+3+5=9, Smin=6, denominator=9 -> 2/3.
    # Cell 2: all complete-signature members are at maxRank -> 0.
    np.testing.assert_allclose(observed, [2.0 / 3.0, 0.0], atol=1e-12)


def test_ucell_average_ties_are_not_integer_truncated() -> None:
    expression = np.asarray([[10.0], [8.0], [8.0], [7.0], [0.0], [0.0]])
    observed = rank_expression_ucell(expression, max_rank=4)
    np.testing.assert_allclose(
        observed[:, 0],
        [1.0, 2.5, 2.5, 4.0, 4.0, 4.0],
        atol=0.0,
    )


def test_ucell_domain_is_fail_closed_at_max_rank() -> None:
    domain = evaluate_ucell_signature_domain(
        signature_gene_count=1500,
        present_gene_count=1499,
        max_rank=1500,
    )
    assert domain.available is False
    assert domain.unavailable_reason == SIGNATURE_AT_OR_ABOVE_MAX_RANK
    assert domain.missing_gene_count == 1


def test_ucell_domain_is_fail_closed_when_no_member_is_in_matrix() -> None:
    domain = evaluate_ucell_signature_domain(
        signature_gene_count=7,
        present_gene_count=0,
        max_rank=1500,
    )
    assert domain.available is False
    assert domain.unavailable_reason == NO_MEMBER_PROTEIN_IN_MATRIX


def test_ucell_unavailable_signature_cannot_emit_numeric_zero() -> None:
    with pytest.raises(UCellContractError, match="NO_MEMBER_PROTEIN_IN_MATRIX"):
        score_ucell_from_capped_ranks(
            np.ones((3, 2), dtype=float),
            present_gene_indices=[],
            signature_gene_count=3,
            max_rank=1500,
        )


def test_ucell_rejects_negative_expression() -> None:
    with pytest.raises(UCellContractError, match="negative"):
        rank_expression_ucell(np.asarray([[1.0], [-1.0]]), max_rank=1500)


def test_pathway_contract_keeps_complete_counts_and_types_unavailable() -> None:
    membership = pd.DataFrame(
        {
            "pathway_id": ["P1", "P1", "P1", "P2", "P3", "P3", "P3"],
            "gene_id": ["G1", "G2", "MISSING", "MISSING", "G1", "G2", "G3"],
        }
    )
    contract = build_ucell_pathway_contract(
        membership,
        protein_gene_ids=["G1", "G2", "G3"],
        max_rank=3,
    )
    sidecar = contract.availability.set_index("pathway_id")
    assert sidecar.loc["P1", "signature_gene_count"] == 3
    assert sidecar.loc["P1", "missing_gene_count"] == 1
    assert sidecar.loc["P1", "unavailable_reason"] == SIGNATURE_AT_OR_ABOVE_MAX_RANK
    assert sidecar.loc["P2", "unavailable_reason"] == NO_MEMBER_PROTEIN_IN_MATRIX
    assert sidecar.loc["P3", "unavailable_reason"] == SIGNATURE_AT_OR_ABOVE_MAX_RANK
    assert contract.signature_matrix.shape == (0, 3)


def test_vectorized_pathway_scores_match_scalar_complete_signature() -> None:
    membership = pd.DataFrame(
        {
            "pathway_id": ["P1", "P1", "P1", "P2", "P2"],
            "gene_id": ["G1", "G2", "MISSING", "G2", "G3"],
        }
    )
    contract = build_ucell_pathway_contract(
        membership,
        protein_gene_ids=["G1", "G2", "G3"],
        max_rank=5,
    )
    ranks = np.asarray([[1.0, 5.0], [3.0, 5.0], [2.0, 4.0]])
    vector = score_ucell_pathway_block(
        ranks,
        signature_matrix=contract.signature_matrix,
        signature_gene_counts=contract.signature_gene_counts,
        missing_gene_counts=contract.missing_gene_counts,
        max_rank=5,
    )
    scalar = np.vstack(
        [
            score_ucell_from_capped_ranks(
                ranks,
                present_gene_indices=[0, 1],
                signature_gene_count=3,
                max_rank=5,
            ),
            score_ucell_from_capped_ranks(
                ranks,
                present_gene_indices=[1, 2],
                signature_gene_count=2,
                max_rank=5,
            ),
        ]
    )
    np.testing.assert_allclose(vector, scalar, atol=1e-12)


def test_full_unlock_accepts_unmodified_pyucell_when_boundary_bug_absent() -> None:
    runner = _load_cell_level_runner()
    small = {
        "official_pyucell_gate_pass": True,
        "v32_vs_official_pyucell_0_7_3_pass": True,
        "official_pyucell_boundary_audit": {
            "pinned_pyucell_boundary_bug_triggered": False,
            "triggered_cell_count": 0,
        },
    }
    assert runner._small_pyucell_gate_passes(small) is True


def test_full_unlock_accepts_strict_boundary_correction() -> None:
    runner = _load_cell_level_runner()
    small = {
        "official_pyucell_gate_pass": True,
        "v32_vs_official_pyucell_0_7_3_pass": False,
        "v32_vs_boundary_corrected_pyucell_0_7_3_pass": True,
        "diagnostic_scoring_only_on_v32_ranks_max_abs_diff": 3e-8,
        "diagnostic_scoring_only_on_official_ranks_max_abs_diff": 3e-8,
        "official_pyucell_boundary_audit": {
            "pinned_pyucell_boundary_bug_triggered": True,
            "triggered_cell_count": 7,
            "correction": (
                "REMOVE_FEATURE_INDEX_TRUNCATION_AFTER_AVERAGE_TIE_RANK_GATE"
            ),
            "rank_dtype_retained_from_pinned_pyucell": "int32",
        },
    }
    assert runner._small_pyucell_gate_passes(small) is True
    small["diagnostic_scoring_only_on_v32_ranks_max_abs_diff"] = 2e-6
    assert runner._small_pyucell_gate_passes(small) is False


def test_generalized_formal_cancer_cli_is_explicit_opt_in() -> None:
    runner = _load_cell_level_runner()
    common = [
        "--mode",
        "small",
        "--preflight-json",
        "preflight.json",
        "--expected-preflight-sha256",
        "a" * 64,
        "--official-parity-json",
        "parity.json",
        "--expected-official-parity-sha256",
        "b" * 64,
        "--output-root",
        "output",
    ]
    assert runner.build_parser().parse_args(common).generalized_formal_cancer is False
    assert (
        runner.build_parser()
        .parse_args(["--generalized-formal-cancer", *common])
        .generalized_formal_cancer
        is True
    )
