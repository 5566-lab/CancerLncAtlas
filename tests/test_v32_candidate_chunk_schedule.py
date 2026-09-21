from __future__ import annotations

import inspect

import pytest

from cc_hhgt.v32.training import (
    CANDIDATE_CHUNK_SCHEDULE_BALANCED_GROUP_SHARED_ENCODER,
    CANDIDATE_CHUNK_SCHEDULE_BALANCED_SINGLE_PASS,
    CANDIDATE_CHUNK_SCHEDULE_EXACT_CARTESIAN,
    SHARED_ENCODER_GRADIENT_ALGORITHM,
    TWO_PASS_STREAMING_GRADIENT_ALGORITHM,
    _canonical_optimizer_batch_groups,
    _candidate_chunk_cycle_schedules,
    _candidate_chunk_schedule,
    _candidate_chunk_schedule_payload,
    _candidate_chunk_schedule_sha256,
    _completed_rotation_receipt,
    _frozen_chunk_permutation,
    _frozen_pass_offset_permutation,
    _gradient_algorithm_for_schedule_mode,
    _optimizer_group_row_counts,
    _resolve_candidate_chunk_schedule_mode,
    shared_encoder_conventional_group_backward,
)


SEED = 20260726
FOLD = 2


def _cycles(
    *,
    batch_count: int,
    chunks: tuple[int, ...],
    count: int,
    mode: str,
    gradient_accumulation: int = 1,
):
    permutation = _frozen_chunk_permutation(chunks, seed=SEED, fold=FOLD)
    offsets = _frozen_pass_offset_permutation(chunks, seed=SEED, fold=FOLD)
    return _candidate_chunk_cycle_schedules(
        batch_count=batch_count,
        chunks=chunks,
        permutation=permutation,
        pass_offset_permutation=offsets,
        max_cycles=count,
        mode=mode,
        gradient_accumulation=gradient_accumulation,
    )


def test_legacy_exact_mode_is_unchanged() -> None:
    chunks = (0, 1, 2)
    permutation = _frozen_chunk_permutation(chunks, seed=SEED, fold=FOLD)
    oracle = _candidate_chunk_schedule(
        batch_count=4, chunks=chunks, permutation=permutation
    )
    assert _cycles(
        batch_count=4,
        chunks=chunks,
        count=2,
        mode=CANDIDATE_CHUNK_SCHEDULE_EXACT_CARTESIAN,
    ) == (oracle, oracle)


def test_one_balanced_cycle_covers_b403_and_balances_k29() -> None:
    chunks = tuple(range(29))
    cycle = _cycles(
        batch_count=403,
        chunks=chunks,
        count=1,
        mode=CANDIDATE_CHUNK_SCHEDULE_BALANCED_SINGLE_PASS,
    )[0]
    assert len(cycle) == 1
    pairs = cycle[0]
    assert [batch for batch, _ in pairs] == list(range(403))
    counts = {chunk: 0 for chunk in chunks}
    for _, chunk in pairs:
        counts[chunk] += 1
    assert sorted(counts.values()) == [13] * 3 + [14] * 26


def test_first_k_balanced_cycles_are_exact_cartesian_without_duplicates() -> None:
    chunks = (0, 1, 2)
    schedules = _cycles(
        batch_count=7,
        chunks=chunks,
        count=len(chunks),
        mode=CANDIDATE_CHUNK_SCHEDULE_BALANCED_SINGLE_PASS,
    )
    flattened = [
        pair
        for cycle in schedules
        for candidate_pass in cycle
        for pair in candidate_pass
    ]
    expected = {(batch, chunk) for batch in range(7) for chunk in chunks}
    assert len(flattened) == len(expected)
    assert set(flattened) == expected


def test_group_latin_b403_k29_a4_is_uniform_with_exact_first_rotation() -> None:
    chunks = tuple(range(29))
    schedules = _cycles(
        batch_count=403,
        chunks=chunks,
        count=len(chunks),
        mode=CANDIDATE_CHUNK_SCHEDULE_BALANCED_GROUP_SHARED_ENCODER,
        gradient_accumulation=4,
    )
    groups = _canonical_optimizer_batch_groups(403, 4)
    assert len(groups) == 101
    assert groups[-1] == (400, 401, 402)
    flattened = []
    for cycle in schedules:
        assert len(cycle) == 1
        pairs = cycle[0]
        assert [batch for batch, _ in pairs] == list(range(403))
        for group in groups:
            assert len({pairs[batch][1] for batch in group}) == 1
        flattened.extend(pairs)
    expected = {(batch, chunk) for batch in range(403) for chunk in chunks}
    assert len(flattened) == len(expected)
    assert set(flattened) == expected


def test_group_latin_schedule_receipt_binds_full_and_partial_group_rows() -> None:
    chunks = tuple(range(29))
    permutation = _frozen_chunk_permutation(chunks, seed=SEED, fold=FOLD)
    offsets = _frozen_pass_offset_permutation(chunks, seed=SEED, fold=FOLD)
    schedules = _cycles(
        batch_count=403,
        chunks=chunks,
        count=3,
        mode=CANDIDATE_CHUNK_SCHEDULE_BALANCED_GROUP_SHARED_ENCODER,
        gradient_accumulation=4,
    )
    row_counts = [8192] * 402 + [6816]
    groups = _canonical_optimizer_batch_groups(403, 4)
    group_rows = _optimizer_group_row_counts(row_counts, groups)
    assert group_rows == (32768,) * 100 + (23200,)
    payload = _candidate_chunk_schedule_payload(
        mode=CANDIDATE_CHUNK_SCHEDULE_BALANCED_GROUP_SHARED_ENCODER,
        chunks=chunks,
        weights={chunk: 1 / len(chunks) for chunk in chunks},
        permutation=permutation,
        pass_offset_permutation=offsets,
        batch_count=403,
        cycle_schedules=schedules,
        gradient_accumulation=4,
        batch_row_counts=row_counts,
    )
    assert payload["assignment_unit"] == "optimizer_group"
    assert payload["optimizer_group_count"] == 101
    assert payload["full_optimizer_group_rows"] == 32768
    assert payload["final_optimizer_group_rows"] == 23200
    assert payload["gradient_algorithm"] == SHARED_ENCODER_GRADIENT_ALGORITHM


def test_group_latin_ten_of_twenty_nine_cycles_is_a_partial_rotation() -> None:
    chunks = tuple(range(29))
    permutation = _frozen_chunk_permutation(chunks, seed=SEED, fold=FOLD)
    schedules = _cycles(
        batch_count=403,
        chunks=chunks,
        count=10,
        mode=CANDIDATE_CHUNK_SCHEDULE_BALANCED_GROUP_SHARED_ENCODER,
        gradient_accumulation=4,
    )
    receipt = _completed_rotation_receipt(schedules, permutation)
    assert len(receipt["completed_unique_pass_offsets"]) == 10
    assert receipt["completed_rotation_fraction"] == pytest.approx(10 / 29)
    assert receipt["full_pair_rotation_completed"] is False


def test_one_exact_cycle_completes_all_pass_offsets() -> None:
    chunks = tuple(range(5))
    permutation = _frozen_chunk_permutation(chunks, seed=SEED, fold=FOLD)
    schedules = _cycles(
        batch_count=13,
        chunks=chunks,
        count=1,
        mode=CANDIDATE_CHUNK_SCHEDULE_EXACT_CARTESIAN,
    )
    receipt = _completed_rotation_receipt(schedules, permutation)
    assert receipt["completed_unique_pass_offsets"] == list(range(5))
    assert receipt["completed_rotation_fraction"] == 1.0
    assert receipt["full_pair_rotation_completed"] is True


def test_shared_helper_has_one_global_reduction_and_metadata_only_plan() -> None:
    source = inspect.getsource(shared_encoder_conventional_group_backward)
    assert source.count("_global_loss_from_outputs(") == 1
    assert "_loss_plan_metadata_from_outputs(" in source
    assert "_loss_plan_from_probe_outputs(" not in source
    assert "global_loss_calls += 1" in source
    assert "global_loss_calls=global_loss_calls" in source


def test_schedule_is_deterministic_and_arm_independent() -> None:
    chunks = tuple(range(5))
    permutation = _frozen_chunk_permutation(chunks, seed=SEED, fold=FOLD)
    offsets = _frozen_pass_offset_permutation(chunks, seed=SEED, fold=FOLD)
    expected = _cycles(
        batch_count=13,
        chunks=chunks,
        count=8,
        mode=CANDIDATE_CHUNK_SCHEDULE_BALANCED_SINGLE_PASS,
    )
    # No graph-arm value is accepted by the schedule API; all three arms bind
    # the same result for a common B/K, fold and seed.
    hashes = set()
    for _arm in ("G0", "G1", "G2"):
        observed = _cycles(
            batch_count=13,
            chunks=chunks,
            count=8,
            mode=CANDIDATE_CHUNK_SCHEDULE_BALANCED_SINGLE_PASS,
        )
        assert observed == expected
        payload = _candidate_chunk_schedule_payload(
            mode=CANDIDATE_CHUNK_SCHEDULE_BALANCED_SINGLE_PASS,
            chunks=chunks,
            weights={chunk: 0.2 for chunk in chunks},
            permutation=permutation,
            pass_offset_permutation=offsets,
            batch_count=13,
            cycle_schedules=observed,
        )
        assert payload["graph_variant_independent_schedule"] is True
        hashes.add(_candidate_chunk_schedule_sha256(payload))
    assert len(hashes) == 1


def test_schedule_hash_changes_with_mode_or_configured_cycle_count() -> None:
    chunks = (0, 1, 2)
    permutation = _frozen_chunk_permutation(chunks, seed=SEED, fold=FOLD)
    offsets = _frozen_pass_offset_permutation(chunks, seed=SEED, fold=FOLD)

    def digest(mode: str, count: int) -> str:
        schedules = _candidate_chunk_cycle_schedules(
            batch_count=7,
            chunks=chunks,
            permutation=permutation,
            pass_offset_permutation=offsets,
            max_cycles=count,
            mode=mode,
        )
        payload = _candidate_chunk_schedule_payload(
            mode=mode,
            chunks=chunks,
            weights={0: 1 / 3, 1: 1 / 3, 2: 1 / 3},
            permutation=permutation,
            pass_offset_permutation=offsets,
            batch_count=7,
            cycle_schedules=schedules,
        )
        assert len(payload["cycles"]) == count
        return _candidate_chunk_schedule_sha256(payload)

    exact_two = digest(CANDIDATE_CHUNK_SCHEDULE_EXACT_CARTESIAN, 2)
    balanced_two = digest(CANDIDATE_CHUNK_SCHEDULE_BALANCED_SINGLE_PASS, 2)
    balanced_three = digest(CANDIDATE_CHUNK_SCHEDULE_BALANCED_SINGLE_PASS, 3)
    assert len({exact_two, balanced_two, balanced_three}) == 3
    assert balanced_two == digest(CANDIDATE_CHUNK_SCHEDULE_BALANCED_SINGLE_PASS, 2)


def test_estimator_is_explicit_and_unknown_modes_fail_closed() -> None:
    assert _resolve_candidate_chunk_schedule_mode({}) == (
        CANDIDATE_CHUNK_SCHEDULE_EXACT_CARTESIAN
    )
    assert _resolve_candidate_chunk_schedule_mode(
        {"candidate_chunk_schedule_mode": "balanced_cyclic_single_pass_v1"}
    ) == CANDIDATE_CHUNK_SCHEDULE_BALANCED_SINGLE_PASS
    assert _resolve_candidate_chunk_schedule_mode(
        {"candidate_chunk_schedule_mode": "balanced_group_latin_shared_encoder_v1"}
    ) == CANDIDATE_CHUNK_SCHEDULE_BALANCED_GROUP_SHARED_ENCODER
    assert _gradient_algorithm_for_schedule_mode(
        CANDIDATE_CHUNK_SCHEDULE_BALANCED_GROUP_SHARED_ENCODER
    ) == SHARED_ENCODER_GRADIENT_ALGORITHM
    assert _gradient_algorithm_for_schedule_mode(
        CANDIDATE_CHUNK_SCHEDULE_BALANCED_SINGLE_PASS
    ) == TWO_PASS_STREAMING_GRADIENT_ALGORITHM
    with pytest.raises(RuntimeError, match="Unsupported"):
        _resolve_candidate_chunk_schedule_mode(
            {"candidate_chunk_schedule_mode": "implicit_or_unknown"}
        )
    with pytest.raises(ValueError, match="Unknown"):
        _gradient_algorithm_for_schedule_mode("implicit_or_unknown")


def test_balanced_mode_fails_closed_when_chunks_outnumber_batches() -> None:
    with pytest.raises(RuntimeError, match="candidate batches >= runtime chunks"):
        _cycles(
            batch_count=2,
            chunks=(0, 1, 2),
            count=1,
            mode=CANDIDATE_CHUNK_SCHEDULE_BALANCED_SINGLE_PASS,
        )


def test_group_latin_fails_closed_when_chunks_outnumber_optimizer_groups() -> None:
    with pytest.raises(RuntimeError, match="optimizer groups >= runtime chunks"):
        _cycles(
            batch_count=8,
            chunks=(0, 1, 2),
            count=1,
            mode=CANDIDATE_CHUNK_SCHEDULE_BALANCED_GROUP_SHARED_ENCODER,
            gradient_accumulation=4,
        )
