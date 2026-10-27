"""Paired comparison harness for "with multi-omics" versus "without".

The question this module makes executable is: *if* two arms were trained on the
same candidates, labels, folds, budget and initialisation - and differ only in
whether the multi-omics node block is present - how would the difference be
reported, and what would make that report invalid?

Two layers, deliberately separated:

``validate_arms``
    Pure bookkeeping over declared digests.  It refuses arms that are not paired
    (different candidate, label, fold or budget identity) and refuses a claim
    that names anything other than the omics block as the difference.

``paired_difference``
    The estimator: per-cancer average precision and ROC AUC, a paired stratified
    bootstrap interval for the difference, a sign test across cancers and a
    paired sign-flip permutation test.  It is verifiable on synthetic scores
    without any training claim.

``compare_arms``
    The only entry point that would report an effect.  It refuses unless both
    arms declare a trained checkpoint, and even then the record says the result
    is an internal paired comparison, not external validation.

No arm in this repository is trained with multi-omics node input, so
``compare_arms`` always refuses today.  That refusal is the honest state: the
comparison code exists, the comparison does not.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

FORMAT = "CANCERLNCATLAS_V32_OMICS_ABLATION_COMPARISON_V1"
SCOPE = "INTERNAL_PAIRED_COMPARISON_NOT_EXTERNAL_VALIDATION"
REQUIRED_ARM_FIELDS = (
    "arm_id",
    "omics_node_input",
    "candidate_sha256",
    "label_sha256",
    "fold_sha256",
    "budget_sha256",
    "init_seed",
    "trained_checkpoint_sha256",
)
DIGEST_FIELDS = ("candidate_sha256", "label_sha256", "fold_sha256", "budget_sha256")
DEFAULT_BOOTSTRAP = 200
DEFAULT_PERMUTATIONS = 400
SCORE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class Arm:
    """One declared comparison arm; digests are computed by the caller."""

    arm_id: str
    omics_node_input: bool
    candidate_sha256: str
    label_sha256: str
    fold_sha256: str
    budget_sha256: str
    init_seed: int
    trained_checkpoint_sha256: str | None = None

    def digest(self) -> str:
        payload = {field_name: getattr(self, field_name) for field_name in REQUIRED_ARM_FIELDS}
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()


def _require_digest(value: Any, name: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise RuntimeError(f"ABLATION_DIGEST_INVALID: {name}")
    return text


def validate_arms(arms: Sequence[Arm]) -> dict:
    """Refuse any pair that is not a single-variable comparison."""

    arms = list(arms)
    if len(arms) != 2:
        raise RuntimeError(f"ABLATION_ARM_COUNT: {len(arms)}")
    first, second = arms
    if not isinstance(first, Arm) or not isinstance(second, Arm):
        raise RuntimeError("ABLATION_ARM_TYPE")
    if first.arm_id == second.arm_id:
        raise RuntimeError("ABLATION_ARM_ID_DUPLICATE")
    if bool(first.omics_node_input) == bool(second.omics_node_input):
        raise RuntimeError("ABLATION_ARMS_DIFFER_IN_MORE_THAN_OMICS")
    for field_name in DIGEST_FIELDS:
        left = _require_digest(getattr(first, field_name), field_name)
        right = _require_digest(getattr(second, field_name), field_name)
        if left != right:
            raise RuntimeError(f"ABLATION_ARMS_NOT_PAIRED: {field_name}")
    if int(first.init_seed) != int(second.init_seed):
        raise RuntimeError("ABLATION_ARMS_NOT_PAIRED: init_seed")
    ordered = sorted(arms, key=lambda arm: bool(arm.omics_node_input))
    without, with_omics = ordered[0], ordered[1]
    return {
        "format": FORMAT,
        "scope": SCOPE,
        "arm_count": 2,
        "without_omics": without.arm_id,
        "with_omics": with_omics.arm_id,
        "paired_digests": {field_name: getattr(without, field_name) for field_name in DIGEST_FIELDS},
        "init_seed": int(without.init_seed),
        "single_variable_difference": "omics_node_input",
        "trained_arms": {
            "without_omics": bool(without.trained_checkpoint_sha256),
            "with_omics": bool(with_omics.trained_checkpoint_sha256),
        },
        "training_required": True,
        "comparison_performed": False,
        "external_validation": False,
    }


def _rank_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """Mann-Whitney AUC with average ranks for ties."""

    if not len(positive) or not len(negative):
        raise RuntimeError("ABLATION_METRIC_GROUP_EMPTY")
    scores = np.concatenate([positive, negative])
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=float)
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop + 1) / 2.0
        start = stop
    positive_rank_sum = float(ranks[: len(positive)].sum())
    return (positive_rank_sum - len(positive) * (len(positive) + 1) / 2.0) / (
        len(positive) * len(negative)
    )


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Step-wise average precision (the same definition as sklearn's AP)."""

    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(bool)
    if scores.size != labels.size:
        raise RuntimeError("ABLATION_LABEL_LENGTH_MISMATCH")
    positives = int(labels.sum())
    if positives == 0:
        raise RuntimeError("ABLATION_METRIC_GROUP_EMPTY")
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order]
    true_positive = np.cumsum(ranked)
    false_positive = np.cumsum(~ranked)
    precision = true_positive / np.maximum(true_positive + false_positive, 1)
    recall = true_positive / positives
    previous_recall = np.concatenate([[0.0], recall[:-1]])
    return float(((recall - previous_recall) * precision).sum())


def _group_metrics(
    scores: np.ndarray, labels: np.ndarray, groups: np.ndarray, group: Any
) -> dict:
    mask = np.asarray(groups) == group
    selected_scores = np.asarray(scores, dtype=float)[mask]
    selected_labels = np.asarray(labels).astype(bool)[mask]
    positives = int(selected_labels.sum())
    negatives = int((~selected_labels).sum())
    record = {
        "pairs": int(mask.sum()),
        "positives": positives,
        "negatives": negatives,
        "prevalence": float(positives / mask.sum()) if mask.sum() else 0.0,
        "average_precision": None,
        "roc_auc": None,
    }
    if positives and negatives:
        record["average_precision"] = average_precision(selected_scores, selected_labels)
        record["roc_auc"] = _rank_auc(
            selected_scores[selected_labels], selected_scores[~selected_labels]
        )
    return record


def paired_difference(
    with_omics_scores,
    without_omics_scores,
    labels,
    groups,
    *,
    bootstrap: int = DEFAULT_BOOTSTRAP,
    permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = 0,
) -> dict:
    """Paired per-cancer difference with a stratified bootstrap interval."""

    with_omics = np.asarray(with_omics_scores, dtype=float)
    without = np.asarray(without_omics_scores, dtype=float)
    labels = np.asarray(labels).astype(bool)
    groups = np.asarray(groups).astype(str)
    if not (with_omics.size == without.size == labels.size == groups.size):
        raise RuntimeError("ABLATION_LABEL_LENGTH_MISMATCH")
    if not np.isfinite(with_omics).all() or not np.isfinite(without).all():
        raise RuntimeError("ABLATION_SCORES_NOT_FINITE")
    if int(bootstrap) < 2 or int(permutations) < 2:
        raise RuntimeError("ABLATION_RESAMPLING_TOO_SMALL")
    unique_groups = sorted(set(groups.tolist()))
    per_group = {}
    for group in unique_groups:
        left = _group_metrics(with_omics, labels, groups, group)
        right = _group_metrics(without, labels, groups, group)
        per_group[group] = {
            "with_omics": left,
            "without_omics": right,
            "delta_average_precision": (
                None
                if left["average_precision"] is None or right["average_precision"] is None
                else left["average_precision"] - right["average_precision"]
            ),
            "delta_roc_auc": (
                None
                if left["roc_auc"] is None or right["roc_auc"] is None
                else left["roc_auc"] - right["roc_auc"]
            ),
        }
    rng = np.random.default_rng(int(seed))
    positions = {group: np.flatnonzero(groups == group) for group in unique_groups}
    bootstrap_deltas = []
    for _ in range(int(bootstrap)):
        sampled = np.concatenate(
            [rng.choice(positions[group], size=len(positions[group]), replace=True) for group in unique_groups]
        )
        sample_labels = labels[sampled]
        if sample_labels.all() or not sample_labels.any():
            continue
        try:
            bootstrap_deltas.append(
                average_precision(with_omics[sampled], sample_labels)
                - average_precision(without[sampled], sample_labels)
            )
        except RuntimeError:
            continue
    if not bootstrap_deltas:
        raise RuntimeError("ABLATION_BOOTSTRAP_EMPTY")
    bootstrap_deltas = np.asarray(bootstrap_deltas)
    observed = average_precision(with_omics, labels) - average_precision(without, labels)
    deltas = with_omics - without
    positive = int((deltas > SCORE_TOLERANCE).sum())
    negative = int((deltas < -SCORE_TOLERANCE).sum())
    sign_p = _sign_test_p(positive, negative)
    flips = 0
    for _ in range(int(permutations)):
        signs = rng.integers(0, 2, size=deltas.size)
        flipped = np.where(signs == 1, deltas, -deltas)
        if abs(float(flipped.mean())) >= abs(float(deltas.mean())):
            flips += 1
    return {
        "format": FORMAT,
        "scope": SCOPE,
        "pairs": int(labels.size),
        "groups": len(unique_groups),
        "average_precision_with_omics": average_precision(with_omics, labels),
        "average_precision_without_omics": average_precision(without, labels),
        "delta_average_precision": float(observed),
        "bootstrap": int(bootstrap),
        "bootstrap_resamples_used": int(bootstrap_deltas.size),
        "delta_average_precision_ci95": [
            float(np.quantile(bootstrap_deltas, 0.025)),
            float(np.quantile(bootstrap_deltas, 0.975)),
        ],
        "sign_test": {
            "with_omics_higher_pairs": positive,
            "with_omics_lower_pairs": negative,
            "two_sided_p": sign_p,
        },
        "sign_flip_permutations": int(permutations),
        "sign_flip_p": float((flips + 1) / (int(permutations) + 1)),
        "per_group": per_group,
        "causal_or_biological_claim": False,
        "external_validation": False,
        "seed": int(seed),
    }


def _sign_test_p(positive: int, negative: int) -> float:
    total = positive + negative
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, k) for k in range(0, min(positive, negative) + 1))
    return float(min(1.0, 2 * tail / (2**total)))


def compare_arms(
    arms: Sequence[Arm],
    *,
    with_omics_scores=None,
    without_omics_scores=None,
    labels=None,
    groups=None,
    bootstrap: int = DEFAULT_BOOTSTRAP,
    permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = 0,
) -> dict:
    """The only path that would report an effect; refuses until both arms train."""

    record = validate_arms(arms)
    missing = [name for name, trained in record["trained_arms"].items() if not trained]
    if missing:
        record.update(
            status="REFUSED_UNTRAINED_ARMS",
            refused_reason="ABLATION_REQUIRES_TRAINED_ARMS",
            untrained_arms=sorted(missing),
            comparison_performed=False,
            effect_reported=False,
            notes=(
                "The paired comparison code path exists and is fixture-verified, but no arm in "
                "this repository was trained with multi-omics node input, so no effect is reported."
            ),
        )
        return record
    if with_omics_scores is None or without_omics_scores is None:
        raise RuntimeError("ABLATION_SCORES_MISSING")
    comparison = paired_difference(
        with_omics_scores,
        without_omics_scores,
        labels,
        groups,
        bootstrap=bootstrap,
        permutations=permutations,
        seed=seed,
    )
    record.update(
        status="COMPARISON_COMPUTED",
        comparison_performed=True,
        effect_reported=True,
        comparison=comparison,
        notes=(
            "Internal paired comparison of two trained arms. It is not external validation and "
            "cannot support a biological claim on its own."
        ),
    )
    return record


__all__ = [
    "Arm",
    "DEFAULT_BOOTSTRAP",
    "DEFAULT_PERMUTATIONS",
    "FORMAT",
    "REQUIRED_ARM_FIELDS",
    "SCOPE",
    "average_precision",
    "compare_arms",
    "paired_difference",
    "validate_arms",
]
