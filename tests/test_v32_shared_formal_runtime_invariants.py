from __future__ import annotations

import copy

import pytest

from cc_hhgt.v32.training import (
    CANDIDATE_CHUNK_SCHEDULE_BALANCED_GROUP_SHARED_ENCODER,
    CANDIDATE_CHUNK_SCHEDULE_BALANCED_SINGLE_PASS,
    CANDIDATE_CHUNK_SCHEDULE_EXACT_CARTESIAN,
    CHECKPOINT_FORMAT,
    LEGACY_CHECKPOINT_FORMAT_V3,
    _resolve_partial_rotation_authorization,
    validate_formal_graph_variant_binding,
    validate_resume_history_and_counters,
)


def _shared_resume_state(cycles: int = 2) -> dict[str, object]:
    history = []
    for cycle in range(cycles):
        completed = cycle + 1
        history.append(
            {
                "cycle": cycle,
                "optimizer_steps_this_cycle": 3,
                "optimizer_steps": completed * 3,
                "encoder_forward_calls": 3,
                "encoder_forward_calls_total": completed * 3,
                "decoder_forward_calls": 10,
                "decoder_forward_calls_total": completed * 10,
                "global_loss_calls": 3,
                "global_loss_calls_total": completed * 3,
                "backward_calls": 3,
                "backward_calls_total": completed * 3,
                "oom_fallback_active": False,
                "oom_fallback_events": 0,
            }
        )
    return {
        "history": history,
        "optimizer_steps": cycles * 3,
        "encoder_forward_calls": cycles * 3,
        "decoder_forward_calls": cycles * 10,
        "global_loss_calls": cycles * 3,
        "backward_calls": cycles * 3,
        "oom_fallback_active": False,
        "oom_fallback_events": 0,
    }


def test_formal_variant_binding_requires_run_config_path_and_payload(tmp_path) -> None:
    prepared = tmp_path / "G2" / "PATIENT_FOLD_0.pt"
    config = {"task_contract": {"graph_variant": "G2"}}
    payload = {"formal_graph_variant": "G2"}
    assert (
        validate_formal_graph_variant_binding(
            run_id="v32-g012-g2-paid-gpu-20260901-r2",
            config=config,
            payload=payload,
            prepared_path=prepared,
        )
        == "G2"
    )
    for bad_run, bad_config, bad_payload, bad_path in (
        (
            "v32-g012-g1-paid-gpu-20260901-r2",
            config,
            payload,
            prepared,
        ),
        (
            "v32-g012-g1-g2-paid-gpu-20260901-r2",
            config,
            payload,
            prepared,
        ),
        (
            "v32-g012-g2-paid-gpu-20260901-r2",
            {"task_contract": {"graph_variant": "G1"}},
            payload,
            prepared,
        ),
        (
            "v32-g012-g2-paid-gpu-20260901-r2",
            config,
            {"formal_graph_variant": "G1"},
            prepared,
        ),
        (
            "v32-g012-g2-paid-gpu-20260901-r2",
            config,
            payload,
            tmp_path / "G1" / "PATIENT_FOLD_0.pt",
        ),
    ):
        with pytest.raises(RuntimeError, match="GRAPH_VARIANT|CONFIG_GRAPH_VARIANT"):
            validate_formal_graph_variant_binding(
                run_id=bad_run,
                config=bad_config,
                payload=bad_payload,
                prepared_path=bad_path,
            )


def test_checkpoint_schema_is_explicitly_v4_and_distinct_from_legacy_v3() -> None:
    assert CHECKPOINT_FORMAT.endswith("V4_GROUP_SCHEDULE")
    assert LEGACY_CHECKPOINT_FORMAT_V3.endswith("V3_STREAMING_BF16")
    assert CHECKPOINT_FORMAT != LEGACY_CHECKPOINT_FORMAT_V3


def test_partial_rotation_requires_literal_opt_in_for_balanced_modes() -> None:
    assert (
        _resolve_partial_rotation_authorization(
            {}, schedule_mode=CANDIDATE_CHUNK_SCHEDULE_EXACT_CARTESIAN
        )
        is False
    )
    for mode in (
        CANDIDATE_CHUNK_SCHEDULE_BALANCED_SINGLE_PASS,
        CANDIDATE_CHUNK_SCHEDULE_BALANCED_GROUP_SHARED_ENCODER,
    ):
        with pytest.raises(RuntimeError, match="EXPLICIT_PARTIAL_AUTHORIZATION"):
            _resolve_partial_rotation_authorization({}, schedule_mode=mode)
        with pytest.raises(RuntimeError, match="EXPLICIT_PARTIAL_AUTHORIZATION"):
            _resolve_partial_rotation_authorization(
                {"allow_partial_candidate_chunk_rotation": 1},
                schedule_mode=mode,
            )
        assert (
            _resolve_partial_rotation_authorization(
                {"allow_partial_candidate_chunk_rotation": True},
                schedule_mode=mode,
            )
            is True
        )


def test_shared_resume_recomputes_every_cycle_and_completed_call_budget() -> None:
    state = _shared_resume_state()
    validate_resume_history_and_counters(
        state,
        checkpoint_cycle=1,
        schedule_mode=CANDIDATE_CHUNK_SCHEDULE_BALANCED_GROUP_SHARED_ENCODER,
        optimizer_group_count=3,
        candidate_batch_count=10,
    )

    for mutation, expected in (
        (("optimizer_steps", 7), "STATE_HISTORY_COUNTER_DRIFT"),
        (("decoder_forward_calls", 21), "STATE_HISTORY_COUNTER_DRIFT"),
    ):
        drifted = copy.deepcopy(state)
        drifted[mutation[0]] = mutation[1]
        with pytest.raises(RuntimeError, match=expected):
            validate_resume_history_and_counters(
                drifted,
                checkpoint_cycle=1,
                schedule_mode=CANDIDATE_CHUNK_SCHEDULE_BALANCED_GROUP_SHARED_ENCODER,
                optimizer_group_count=3,
                candidate_batch_count=10,
            )

    drifted = copy.deepcopy(state)
    drifted["history"][1]["encoder_forward_calls"] = 2
    with pytest.raises(RuntimeError, match="CALL_ACCOUNTING_DRIFT|CALL_BUDGET_DRIFT"):
        validate_resume_history_and_counters(
            drifted,
            checkpoint_cycle=1,
            schedule_mode=CANDIDATE_CHUNK_SCHEDULE_BALANCED_GROUP_SHARED_ENCODER,
            optimizer_group_count=3,
            candidate_batch_count=10,
        )


def test_all_resume_modes_reject_negative_or_nonmonotone_history() -> None:
    state = _shared_resume_state()
    validate_resume_history_and_counters(
        state,
        checkpoint_cycle=1,
        schedule_mode=CANDIDATE_CHUNK_SCHEDULE_EXACT_CARTESIAN,
        optimizer_group_count=3,
        candidate_batch_count=10,
    )
    negative = copy.deepcopy(state)
    negative["history"][0]["backward_calls"] = -1
    with pytest.raises(RuntimeError, match="COUNTER_INVALID"):
        validate_resume_history_and_counters(
            negative,
            checkpoint_cycle=1,
            schedule_mode=CANDIDATE_CHUNK_SCHEDULE_EXACT_CARTESIAN,
            optimizer_group_count=3,
            candidate_batch_count=10,
        )
    nonmonotone = copy.deepcopy(state)
    nonmonotone["history"][1]["oom_fallback_events"] = 0
    nonmonotone["history"][0]["oom_fallback_events"] = 1
    nonmonotone["history"][0]["oom_fallback_active"] = True
    with pytest.raises(RuntimeError, match="OOM_MONOTONICITY_DRIFT"):
        validate_resume_history_and_counters(
            nonmonotone,
            checkpoint_cycle=1,
            schedule_mode=CANDIDATE_CHUNK_SCHEDULE_EXACT_CARTESIAN,
            optimizer_group_count=3,
            candidate_batch_count=10,
        )
