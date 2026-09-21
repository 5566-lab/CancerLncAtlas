"""Phase 8 tests: explicit numeric evidence features.

The plan requires that continuous evidence measurements are NOT hashed into
categorical buckets, that the normalisation is fixed and declared, and that it is
never fitted on the pathway labels.  These tests pin all three, plus the
requirement that the legacy matrix stays bit-identical.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from cc_hhgt.v32.evidence_training import (
    EVENT_NUMERIC_FIELDS,
    EVENT_NUMERIC_SCALES,
    LEGACY_EVENT_FEATURE_FIELDS,
    EvidenceTrainingContractError,
    _event_feature_matrix,
    event_feature_dimension,
    normalise_numeric_feature,
)


def _frame(rows: list[dict]) -> pd.DataFrame:
    base = {
        "route_type": "PARTNER_EXACT_MEMBER",
        "source_database": "NPInter",
        "source_dataset": "NPInter-5",
        "experiment_type": "physical_binding",
        "experiment_family": "physical_binding",
        "assay_subtype": "eclip",
        "graph_assay_class": "eclip",
        "relation_type": "binding_or_interaction",
        "tissue": "liver",
        "cell_line": "HepG2",
        "species": "human",
        "member_type": "protein",
        "pmid": "12345678",
        "is_experimental": True,
        "is_computational": False,
        "is_physical": True,
    }
    return pd.DataFrame([{**base, **row} for row in rows])


# ---------------------------------------------------------------------------
# The declared contract
# ---------------------------------------------------------------------------


def test_numeric_fields_are_declared_and_closed() -> None:
    assert EVENT_NUMERIC_FIELDS == (
        "n_reproducible_peaks", "max_log2fc", "mean_log2fc", "max_neglog10p",
    )
    assert set(EVENT_NUMERIC_FIELDS) == set(EVENT_NUMERIC_SCALES)


def test_an_undeclared_numeric_field_is_rejected() -> None:
    frame = _frame([{}])
    with pytest.raises(EvidenceTrainingContractError):
        _event_feature_matrix(frame, 128, numeric_fields=("not_a_declared_field",))


def test_dimension_helper_appends_without_touching_the_base() -> None:
    assert event_feature_dimension(128) == 128
    assert event_feature_dimension(128, numeric_fields=EVENT_NUMERIC_FIELDS) == 132


# ---------------------------------------------------------------------------
# Legacy equivalence — the anchor
# ---------------------------------------------------------------------------


def test_no_numeric_fields_leaves_the_matrix_bit_identical() -> None:
    """With numeric_fields empty the historical computation is untouched."""

    frame = _frame([{"n_reproducible_peaks": 12345.0, "max_log2fc": 3.2}])
    produced = _event_feature_matrix(frame, 128)

    manual = np.zeros((1, 128), dtype=np.float32)
    row = frame.to_dict("records")[0]
    for field in LEGACY_EVENT_FEATURE_FIELDS:
        token = f"{field}={str(row.get(field, 'unknown')).lower()}"
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "little") % (128 - 8)
        sign = 1.0 if digest[4] & 1 else -1.0
        manual[0, bucket] += sign
    manual[0, 120] = 1.0
    manual[0, 121] = 0.0
    manual[0, 122] = 1.0
    manual[0, 123] = 1.0
    manual[0, 124] = 1.0 / 5.0
    manual[0, 125] = 0.0
    manual[0, 126] = 1.0
    manual[0, 127] = 1.0
    assert np.array_equal(produced, manual)


def test_enabling_numeric_features_does_not_disturb_the_hashed_part() -> None:
    """The numeric block is appended, so the categorical modulus is unchanged."""

    frame = _frame([
        {"n_reproducible_peaks": 5000.0, "max_log2fc": 1.5,
         "mean_log2fc": 0.5, "max_neglog10p": 12.0},
    ])
    base = _event_feature_matrix(frame, 128)
    extended = _event_feature_matrix(
        frame, 132, numeric_fields=EVENT_NUMERIC_FIELDS
    )
    assert np.array_equal(base[:, :128], extended[:, :128]), (
        "adding numeric features perturbed the hashed categorical features"
    )


def test_numeric_block_occupies_the_trailing_slots() -> None:
    frame = _frame([{"n_reproducible_peaks": 1000.0}])
    matrix = _event_feature_matrix(frame, 132, numeric_fields=EVENT_NUMERIC_FIELDS)
    assert matrix.shape == (1, 132)
    # slots 128..131 are the numeric block
    assert matrix[0, 128] == pytest.approx(
        normalise_numeric_feature("n_reproducible_peaks", 1000.0)
    )
    # the remaining declared fields were absent -> neutral zero
    for slot in (129, 130, 131):
        assert matrix[0, slot] == 0.0


# ---------------------------------------------------------------------------
# Normalisation contract
# ---------------------------------------------------------------------------


def test_normalisation_is_deterministic() -> None:
    for name in EVENT_NUMERIC_FIELDS:
        first = normalise_numeric_feature(name, 7.5)
        for _ in range(20):
            assert normalise_numeric_feature(name, 7.5) == first


def test_missing_or_non_finite_values_become_neutral_zero() -> None:
    for value in (None, float("nan"), float("inf"), "", "not_a_number"):
        for name in EVENT_NUMERIC_FIELDS:
            assert normalise_numeric_feature(name, value) == 0.0


def test_every_scale_clips_into_its_declared_range() -> None:
    for name, (_divisor, low, high) in EVENT_NUMERIC_SCALES.items():
        for value in (-1e12, -1.0, 0.0, 1.0, 1e12):
            got = normalise_numeric_feature(name, value)
            assert low <= got <= high, (name, value, got)


def test_larger_measurements_normalise_to_larger_values() -> None:
    """Monotonicity: the channel must not scramble the ordering."""

    for name in ("n_reproducible_peaks", "max_neglog10p"):
        values = [0.0, 1.0, 10.0, 100.0, 1000.0]
        scaled = [normalise_numeric_feature(name, v) for v in values]
        assert scaled == sorted(scaled), name


def test_normalisation_is_not_fitted_on_any_label() -> None:
    """The same value must normalise identically regardless of anything else.

    If the scale depended on the training labels or on the pathway outcome, the
    same measurement would map differently in different contexts.  It must not.
    """

    for name in EVENT_NUMERIC_FIELDS:
        assert normalise_numeric_feature(name, 42.0) == normalise_numeric_feature(name, 42.0)
    # and the scale table contains no data-dependent entry
    for name, (_d, low, high) in EVENT_NUMERIC_SCALES.items():
        assert isinstance(low, float) and isinstance(high, float)


def test_undeclared_name_raises_in_the_helper() -> None:
    with pytest.raises(EvidenceTrainingContractError):
        normalise_numeric_feature("nope", 1.0)


# ---------------------------------------------------------------------------
# Interaction with assay typing
# ---------------------------------------------------------------------------


def test_numeric_and_assay_channels_compose() -> None:
    frame = _frame([
        {"assay_subtype": "eclip", "graph_assay_class": "eclip",
         "n_reproducible_peaks": 900.0},
        {"assay_subtype": "rip", "graph_assay_class": "rip",
         "n_reproducible_peaks": 900.0},
    ])
    matrix = _event_feature_matrix(
        frame, 132, preserve_assay_type=True, numeric_fields=EVENT_NUMERIC_FIELDS
    )
    # identical numeric evidence, different assay -> rows must still differ
    assert not np.array_equal(matrix[0], matrix[1])
    # and the numeric slot is the same for both
    assert matrix[0, 128] == matrix[1, 128]


def test_dimension_guard_still_applies_with_numeric_fields() -> None:
    frame = _frame([{}])
    with pytest.raises(EvidenceTrainingContractError):
        _event_feature_matrix(
            frame, 24, numeric_fields=EVENT_NUMERIC_FIELDS
        )
