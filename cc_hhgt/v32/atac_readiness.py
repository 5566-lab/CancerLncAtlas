"""Pure helpers for auditing fresh V3.2 ATAC readiness."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class AtacFoldCapacity:
    aligned_patients: int
    fold_counts: tuple[int, int, int, int, int]
    minimum_test_patients: int
    minimum_validation_patients: int
    minimum_nested_train_patients: int
    all_folds_nonempty: bool
    genomic_analogue_min12_upper_bound_pass: bool
    conservative_pilot_capacity_pass: bool


def normalize_atac_cancer(value: str) -> str:
    """Normalize the official matrix prefix without changing real TCGA codes."""

    cancer = str(value).split("-", 1)[0].upper()
    return cancer[:-1] if cancer.endswith("X") else cancer


def nested_fold_capacity(counts: Sequence[int]) -> AtacFoldCapacity:
    """Summarize the registered heldout/next-fold-validation schedule.

    The current genomic analogue requires at least 12 callable patients for a
    pair.  ``genomic_analogue_min12_upper_bound_pass`` only checks raw patient
    counts, so it is an optimistic upper bound rather than proof that any ATAC
    pair is callable.
    """

    values = tuple(int(value) for value in counts)
    if len(values) != 5 or min(values) < 0:
        raise ValueError("ATAC fold counts must contain five non-negative values")
    total = sum(values)
    nested_train = tuple(
        total - values[heldout] - values[(heldout + 1) % 5]
        for heldout in range(5)
    )
    minimum = min(values)
    return AtacFoldCapacity(
        aligned_patients=total,
        fold_counts=values,
        minimum_test_patients=minimum,
        minimum_validation_patients=minimum,
        minimum_nested_train_patients=min(nested_train),
        all_folds_nonempty=minimum > 0,
        genomic_analogue_min12_upper_bound_pass=minimum >= 12,
        conservative_pilot_capacity_pass=(
            minimum >= 3 and min(nested_train) >= 12
        ),
    )


def readiness_class(raw_available: bool, capacity: AtacFoldCapacity) -> str:
    if not raw_available:
        return "TRUE_RAW_GAP_CURRENT_OFFICIAL_MATRIX"
    if capacity.aligned_patients < 12 or not capacity.all_folds_nonempty:
        return "PRESENT_BUT_ALIGNMENT_OR_FOLD_SPARSE"
    return "PRESENT_PILOTABLE_NOT_FORMAL_ROUTER_READY"


__all__ = [
    "AtacFoldCapacity",
    "nested_fold_capacity",
    "normalize_atac_cancer",
    "readiness_class",
]
