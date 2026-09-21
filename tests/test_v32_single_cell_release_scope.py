from __future__ import annotations

from copy import deepcopy

import pytest

from cc_hhgt.v32.single_cell_release_scope import (
    POLICY_ID,
    SingleCellReleaseScopeError,
    final_binding_scope_is_available,
    load_single_cell_release_scope,
    validate_single_cell_release_scope,
)


def test_policy_is_an_exact_23_plus_10_partition_and_not_a_full33_claim() -> None:
    policy = load_single_cell_release_scope()
    eligible = set(policy["formal_eligible_cancers"])
    unavailable = set(policy["typed_unavailable"])
    assert len(eligible) == 23
    assert len(unavailable) == 10
    assert not eligible & unavailable
    assert len(eligible | unavailable) == 33
    assert policy["full_33_single_cell_coverage_claimed"] is False
    assert policy["full_33_single_cell_completion_is_release_gate"] is False
    assert "CESC" in eligible
    assert "UVM" in eligible
    assert policy["typed_unavailable"]["UCS"] == (
        "GSE299623_RELEASE_CONTAINS_18082_FEATURES_BUT_ONLY_38_FORMAL_LNCRNAS"
    )
    assert policy["evidence_basis"]["new_download_attempted"] is True
    assert policy["evidence_basis"]["fastq_download_attempted"] is False


def test_policy_rejects_overlap_or_missing_typed_reason() -> None:
    policy = load_single_cell_release_scope()
    broken = deepcopy(policy)
    broken["typed_unavailable"][broken["formal_eligible_cancers"][0]] = "OVERLAP"
    with pytest.raises(SingleCellReleaseScopeError):
        validate_single_cell_release_scope(broken)
    broken = deepcopy(policy)
    first = next(iter(broken["typed_unavailable"]))
    broken["typed_unavailable"][first] = ""
    with pytest.raises(SingleCellReleaseScopeError):
        validate_single_cell_release_scope(broken)


def test_final_binding_accepts_23_fresh_plus_10_typed_nulls() -> None:
    module = {
        "status": "AVAILABLE",
        "cancer_count": 33,
        "fresh_derived_cancer_count": 23,
        "fresh_recompute": True,
        "historical_assets_relabelled_fresh": False,
        "scope_policy_id": POLICY_ID,
        "formal_eligible_cancer_count": 23,
        "typed_unavailable_cancer_count": 10,
        "full_33_single_cell_coverage_claimed": False,
        "typed_unavailable_rows_are_null": True,
        "typed_unavailable_changes_primary_score": False,
    }
    assert final_binding_scope_is_available(module)
    module["typed_unavailable_rows_are_null"] = False
    assert not final_binding_scope_is_available(module)


def test_legacy_full33_binding_remains_accepted() -> None:
    assert final_binding_scope_is_available(
        {
            "status": "AVAILABLE",
            "cancer_count": 33,
            "fresh_derived_cancer_count": 33,
            "fresh_recompute": True,
            "historical_assets_relabelled_fresh": False,
        }
    )
