"""Fixture checks for the paired multi-omics comparison harness.

The metric and resampling code is verified on synthetic scores.  Declared
checkpoint hashes in these fixtures are placeholders: no arm was trained, and
``compare_arms`` refusing an untrained pair is exercised explicitly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cc_hhgt.v32 import omics_ablation_comparison as compare

DIGEST = "a" * 64


def _arm(**overrides) -> compare.Arm:
    payload = {
        "arm_id": "degree_only",
        "omics_node_input": False,
        "candidate_sha256": DIGEST,
        "label_sha256": DIGEST,
        "fold_sha256": DIGEST,
        "budget_sha256": DIGEST,
        "init_seed": 7,
        "trained_checkpoint_sha256": None,
    }
    payload.update(overrides)
    return compare.Arm(**payload)


def _pair(**overrides):
    overrides.setdefault("arm_id", "omics")
    overrides.setdefault("omics_node_input", True)
    return [_arm(), _arm(**overrides)]


def _scores(pairs: int = 80, seed: int = 0):
    rng = np.random.default_rng(seed)
    labels = np.tile(np.asarray([True, False]), pairs // 2)
    groups = np.asarray(["C1"] * (pairs // 2) + ["C2"] * (pairs // 2))
    baseline = rng.normal(0.0, 0.1, size=pairs)
    improved = baseline + labels * 1.0
    return improved, baseline, labels, groups


def test_average_precision_matches_the_step_definition() -> None:
    scores = np.asarray([3.0, 2.0, 1.0, 0.0])
    labels = np.asarray([1, 0, 1, 1])
    value = compare.average_precision(scores, labels)
    assert value == pytest.approx(1 / 3 + 2 / 9 + 1 / 4)


def test_average_precision_rejects_a_group_without_positives() -> None:
    with pytest.raises(RuntimeError, match="ABLATION_METRIC_GROUP_EMPTY"):
        compare.average_precision(np.asarray([1.0, 0.0]), np.asarray([0, 0]))


@pytest.mark.parametrize(
    "scores, expected",
    [
        ([3.0, 2.0, 1.0, 0.0], 1.0),
        ([0.0, 1.0, 2.0, 3.0], 0.0),
        ([1.0, 1.0, 1.0, 1.0], 0.5),
    ],
)
def test_rank_auc_handles_ordering_and_ties(scores, expected) -> None:
    positive = np.asarray([scores[0], scores[1]])
    negative = np.asarray([scores[2], scores[3]])
    assert compare._rank_auc(positive, negative) == pytest.approx(expected)


def test_validate_arms_accepts_a_single_variable_pair() -> None:
    record = compare.validate_arms(_pair())
    assert record["without_omics"] == "degree_only"
    assert record["with_omics"] == "omics"
    assert record["single_variable_difference"] == "omics_node_input"
    assert record["trained_arms"] == {"without_omics": False, "with_omics": False}
    assert record["comparison_performed"] is False
    assert record["external_validation"] is False


@pytest.mark.parametrize(
    "arms, message",
    [
        pytest.param([_arm()], "ABLATION_ARM_COUNT", id="one_arm"),
        pytest.param([_arm(), _arm(), _arm()], "ABLATION_ARM_COUNT", id="three_arms"),
        pytest.param(
            [_arm(), _arm(arm_id="degree_only"), _arm(omics_node_input=True)],
            "ABLATION_ARM_COUNT",
            id="three_arms_with_duplicate",
        ),
        pytest.param([_arm(), _arm()], "ABLATION_ARM_ID_DUPLICATE", id="duplicate_id"),
        pytest.param(
            [_arm(), _arm(arm_id="other")],
            "ABLATION_ARMS_DIFFER_IN_MORE_THAN_OMICS",
            id="same_omics_flag",
        ),
        pytest.param(
            [_arm(), _arm(arm_id="omics", omics_node_input=True, candidate_sha256="b" * 64)],
            "ABLATION_ARMS_NOT_PAIRED",
            id="candidate_drift",
        ),
        pytest.param(
            [_arm(), _arm(arm_id="omics", omics_node_input=True, label_sha256="b" * 64)],
            "ABLATION_ARMS_NOT_PAIRED",
            id="label_drift",
        ),
        pytest.param(
            [_arm(), _arm(arm_id="omics", omics_node_input=True, fold_sha256="b" * 64)],
            "ABLATION_ARMS_NOT_PAIRED",
            id="fold_drift",
        ),
        pytest.param(
            [_arm(), _arm(arm_id="omics", omics_node_input=True, budget_sha256="b" * 64)],
            "ABLATION_ARMS_NOT_PAIRED",
            id="budget_drift",
        ),
        pytest.param(
            [_arm(), _arm(arm_id="omics", omics_node_input=True, init_seed=11)],
            "ABLATION_ARMS_NOT_PAIRED",
            id="seed_drift",
        ),
        pytest.param(
            [_arm(), _arm(arm_id="omics", omics_node_input=True, candidate_sha256="not-a-digest")],
            "ABLATION_DIGEST_INVALID",
            id="bad_digest",
        ),
    ],
)
def test_validate_arms_refuses_unpaired_or_ambiguous_pairs(arms, message) -> None:
    with pytest.raises(RuntimeError, match=message):
        compare.validate_arms(arms)


def test_paired_difference_of_identical_scores_is_null() -> None:
    improved, baseline, labels, groups = _scores()
    record = compare.paired_difference(
        baseline, baseline, labels, groups, bootstrap=40, permutations=40, seed=3
    )
    assert record["delta_average_precision"] == 0.0
    assert record["delta_average_precision_ci95"] == [0.0, 0.0]
    assert record["sign_test"]["two_sided_p"] == 1.0
    assert record["sign_flip_p"] == 1.0
    assert record["causal_or_biological_claim"] is False


def test_paired_difference_detects_a_planted_improvement() -> None:
    improved, baseline, labels, groups = _scores()
    record = compare.paired_difference(
        improved, baseline, labels, groups, bootstrap=200, permutations=200, seed=5
    )
    assert record["delta_average_precision"] > 0.3
    assert record["delta_average_precision_ci95"][0] > 0.0
    assert record["sign_test"]["with_omics_higher_pairs"] > 0
    assert record["sign_test"]["two_sided_p"] < 1e-6
    assert record["sign_flip_p"] < 0.01
    assert record["groups"] == 2
    assert set(record["per_group"]) == {"C1", "C2"}
    for group in record["per_group"].values():
        assert group["with_omics"]["positives"] == 20
        assert group["with_omics"]["negatives"] == 20
        assert group["delta_average_precision"] > 0.0


def test_paired_difference_is_deterministic_for_a_fixed_seed() -> None:
    improved, baseline, labels, groups = _scores()
    first = compare.paired_difference(
        improved, baseline, labels, groups, bootstrap=60, permutations=60, seed=11
    )
    second = compare.paired_difference(
        improved, baseline, labels, groups, bootstrap=60, permutations=60, seed=11
    )
    assert first["delta_average_precision_ci95"] == second["delta_average_precision_ci95"]
    assert first["sign_flip_p"] == second["sign_flip_p"]


@pytest.mark.parametrize(
    "call, message",
    [
        (
            lambda: compare.paired_difference([1.0, 0.0], [1.0], [1, 0], ["C1", "C1"]),
            "ABLATION_LABEL_LENGTH_MISMATCH",
        ),
        (
            lambda: compare.paired_difference(
                [1.0, float("nan")], [1.0, 0.0], [1, 0], ["C1", "C1"]
            ),
            "ABLATION_SCORES_NOT_FINITE",
        ),
        (
            lambda: compare.paired_difference(
                [1.0, 0.0], [1.0, 0.0], [1, 0], ["C1", "C1"], bootstrap=1
            ),
            "ABLATION_RESAMPLING_TOO_SMALL",
        ),
    ],
)
def test_paired_difference_refuses_invalid_input(call, message) -> None:
    with pytest.raises(RuntimeError, match=message):
        call()


@pytest.mark.parametrize(
    "positive, negative, expected",
    [(0, 0, 1.0), (3, 3, 1.0), (5, 0, 2 / 32), (0, 5, 2 / 32)],
)
def test_sign_test_is_exact(positive, negative, expected) -> None:
    assert compare._sign_test_p(positive, negative) == pytest.approx(expected)


def test_compare_arms_refuses_until_both_arms_declare_a_checkpoint() -> None:
    record = compare.compare_arms(_pair())
    assert record["status"] == "REFUSED_UNTRAINED_ARMS"
    assert record["refused_reason"] == "ABLATION_REQUIRES_TRAINED_ARMS"
    assert record["comparison_performed"] is False
    assert record["effect_reported"] is False
    assert record["untrained_arms"] == ["with_omics", "without_omics"]
    assert "no arm in this repository" in record["notes"]


def test_compare_arms_requires_scores_once_both_arms_are_trained() -> None:
    trained = _pair(trained_checkpoint_sha256=DIGEST)
    trained[0] = _arm(trained_checkpoint_sha256=DIGEST)
    with pytest.raises(RuntimeError, match="ABLATION_SCORES_MISSING"):
        compare.compare_arms(trained)


def test_compare_arms_computes_a_paired_estimate_for_declared_trained_arms() -> None:
    trained = [_arm(trained_checkpoint_sha256=DIGEST), _pair(trained_checkpoint_sha256=DIGEST)[1]]
    improved, baseline, labels, groups = _scores()
    record = compare.compare_arms(
        trained,
        with_omics_scores=improved,
        without_omics_scores=baseline,
        labels=labels,
        groups=groups,
        bootstrap=40,
        permutations=40,
        seed=2,
    )
    assert record["status"] == "COMPARISON_COMPUTED"
    assert record["comparison_performed"] is True
    assert record["comparison"]["delta_average_precision"] > 0.0
    assert record["comparison"]["external_validation"] is False
    assert record["external_validation"] is False
    assert "not external validation" in record["notes"]


def test_module_exposes_no_fitting_entry_point() -> None:
    forbidden = {"fit", "train", "fit_transform", "optimizer", "backward"}
    assert forbidden.isdisjoint(dir(compare))
    assert compare.SCOPE.endswith("NOT_EXTERNAL_VALIDATION")
