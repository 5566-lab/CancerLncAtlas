from __future__ import annotations

import pytest

from cc_hhgt.v32.atac_readiness import (
    nested_fold_capacity,
    normalize_atac_cancer,
    readiness_class,
)


def test_official_padding_suffix_is_normalized_without_touching_real_codes() -> None:
    assert normalize_atac_cancer("ACCx-uuid") == "ACC"
    assert normalize_atac_cancer("GBMx-uuid") == "GBM"
    assert normalize_atac_cancer("LGGx-uuid") == "LGG"
    assert normalize_atac_cancer("BRCA-uuid") == "BRCA"


def test_fold_capacity_is_nested_and_strict_gate_is_only_an_upper_bound() -> None:
    capacity = nested_fold_capacity([15, 11, 16, 18, 13])
    assert capacity.aligned_patients == 73
    assert capacity.minimum_test_patients == 11
    assert capacity.minimum_nested_train_patients == 39
    assert capacity.conservative_pilot_capacity_pass
    assert not capacity.genomic_analogue_min12_upper_bound_pass
    assert readiness_class(True, capacity) == "PRESENT_PILOTABLE_NOT_FORMAL_ROUTER_READY"


def test_true_absence_and_fold_sparsity_are_not_conflated() -> None:
    sparse = nested_fold_capacity([0, 0, 2, 0, 0])
    assert readiness_class(True, sparse) == "PRESENT_BUT_ALIGNMENT_OR_FOLD_SPARSE"
    assert readiness_class(False, nested_fold_capacity([0] * 5)) == (
        "TRUE_RAW_GAP_CURRENT_OFFICIAL_MATRIX"
    )
    with pytest.raises(ValueError):
        nested_fold_capacity([1, 2])
