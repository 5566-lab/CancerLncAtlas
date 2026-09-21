"""Independent revision-3 design oracles; these do not mark production ready."""
from __future__ import annotations

from collections import defaultdict, deque
import hashlib
import math
import random


Edge = tuple[str, str, str]


def _reciprocal(source: str, relation: str, target: str) -> set[Edge]:
    return {
        (source, relation, target),
        (target, f"rev_{relation}", source),
    }


def _ppi_pair(left: str, right: str) -> set[Edge]:
    if left == right:
        raise ValueError("PPI self-loops are forbidden")
    left, right = sorted((left, right))
    return {
        (left, "physical_interaction", right),
        (right, "physical_interaction", left),
    }


def _distance(edges: set[Edge], start: str, target: str) -> int | None:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for source, _, destination in edges:
        adjacency[source].add(destination)
    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        node, depth = queue.popleft()
        if node == target:
            return depth
        for neighbour in adjacency[node]:
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append((neighbour, depth + 1))
    return None


def _edge_hash(edges: set[Edge]) -> str:
    payload = "\n".join("\x1f".join(edge) for edge in sorted(edges)).encode()
    return hashlib.sha256(payload).hexdigest()


def _aggregate_logits(contributions: list[tuple[int, float, float]]) -> float:
    """Canonical float64 sum of (chunk_id, weight, logit)."""

    ordered = sorted(contributions, key=lambda item: item[0])
    if not math.isclose(math.fsum(weight for _, weight, _ in ordered), 1.0):
        raise ValueError("Chunk weights must sum to one")
    return math.fsum(weight * logit for _, weight, logit in ordered)


def _softplus(value: float) -> float:
    return max(value, 0.0) + math.log1p(math.exp(-abs(value)))


def _global_nnpu(
    rows: list[tuple[float, bool, bool]],
    *,
    positive_prior: float = 0.10,
    unlabeled_weight: float = 0.12,
    weak_positive_weight: float = 0.35,
) -> float:
    positive = [(z, weak) for z, label, weak in rows if label]
    unlabeled = [z for z, label, _ in rows if not label]
    positive_risk = 0.0
    positive_as_negative = 0.0
    if positive:
        weights = [weak_positive_weight if weak else 1.0 for _, weak in positive]
        denominator = math.fsum(weights)
        positive_risk = positive_prior * math.fsum(
            weight * _softplus(-logit)
            for (logit, _), weight in zip(positive, weights, strict=True)
        ) / denominator
        positive_as_negative = positive_prior * math.fsum(
            weight * _softplus(logit)
            for (logit, _), weight in zip(positive, weights, strict=True)
        ) / denominator
    unlabeled_risk = (
        math.fsum(_softplus(logit) for logit in unlabeled) / len(unlabeled)
        if unlabeled
        else 0.0
    )
    return positive_risk + unlabeled_weight * max(
        unlabeled_risk - positive_as_negative, 0.0
    )


def test_self_type_ppi_is_exactly_two_messages_in_one_relation() -> None:
    messages = _ppi_pair("protein:B", "protein:A")
    assert len(messages) == 2
    assert {relation for _, relation, _ in messages} == {"physical_interaction"}
    assert ("protein:A", "physical_interaction", "protein:B") in messages
    assert ("protein:B", "physical_interaction", "protein:A") in messages


def test_complete_backbone_preserves_two_three_four_hop_paths_per_chunk() -> None:
    static = set()
    static |= _reciprocal("gene:C0", "member_of_positive", "pathway:P0")
    static |= _reciprocal("gene:G1", "member_of_positive", "pathway:P1")
    static |= _reciprocal("gene:G2", "member_of_positive", "pathway:P2")
    static |= _reciprocal("lncRNA:L1", "binds_protein", "protein:A")
    static |= _reciprocal("protein:A", "encoded_by", "gene:G1")
    static |= _reciprocal("lncRNA:L2", "binds_protein", "protein:A")
    static |= _ppi_pair("protein:A", "protein:B")
    static |= _reciprocal("protein:B", "encoded_by", "gene:G2")
    variable = [
        _reciprocal("lncRNA:L0a", "coexpressed_positive", "gene:C0"),
        _reciprocal("lncRNA:L0b", "coexpressed_negative", "gene:C0"),
    ]
    static_hash = _edge_hash(static)
    for index, coexpression in enumerate(variable):
        chunk = static | coexpression
        assert _distance(chunk, f"lncRNA:L0{'a' if index == 0 else 'b'}", "pathway:P0") == 2
        assert _distance(chunk, "lncRNA:L1", "pathway:P1") == 3
        assert _distance(chunk, "lncRNA:L2", "pathway:P2") == 4
        assert _edge_hash(chunk - coexpression) == static_hash
    assert _edge_hash(static) == static_hash


def test_chunk_order_cannot_change_canonical_logit_aggregate() -> None:
    contributions = [(0, 0.2, -1.25), (1, 0.3, 0.75), (2, 0.5, 2.0)]
    expected = _aggregate_logits(contributions)
    random.Random(20260726).shuffle(contributions)
    assert _aggregate_logits(contributions) == expected


def test_typed_masks_remove_messages_but_retain_schema() -> None:
    schema = {
        "expressed_in",
        "coexpressed_positive",
        "coexpressed_negative",
        "member_of_positive",
        "member_of_negative",
        "member_of_family",
        "binds_protein",
        "encoded_by",
        "physical_interaction",
    }
    counts_g2 = {relation: 1 for relation in schema}
    counts_g1 = counts_g2 | {"physical_interaction": 0}
    counts_g0 = counts_g1 | {"binds_protein": 0, "encoded_by": 0}
    assert set(counts_g0) == set(counts_g1) == set(counts_g2) == schema
    assert counts_g0["binds_protein"] == counts_g0["encoded_by"] == 0
    assert counts_g0["physical_interaction"] == counts_g1["physical_interaction"] == 0
    assert counts_g2["physical_interaction"] == 1


def test_global_nnpu_is_batch_partition_and_row_order_invariant() -> None:
    rows = [
        (-2.0, True, False),
        (0.5, True, True),
        (1.5, False, False),
        (-0.5, False, False),
        (3.0, False, False),
    ]
    expected = _global_nnpu(rows)
    shuffled = rows.copy()
    random.Random(7).shuffle(shuffled)
    # Batches are concatenated before the one global nonlinear nnPU reduction.
    partitions = [shuffled[:1], shuffled[1:4], shuffled[4:]]
    assert math.isclose(_global_nnpu([row for part in partitions for row in part]), expected)
    # The current per-batch-scalar mean is a different, partition-dependent estimand.
    per_batch_mean = math.fsum(_global_nnpu(part) for part in partitions) / len(partitions)
    assert not math.isclose(per_batch_mean, expected)


def test_df_uses_design_rank_and_full_universe_count_is_not_3_3m() -> None:
    n, design_columns, design_rank = 100, 5, 3
    correct_df = n - design_rank - 1
    old_column_count_df = n - design_columns - 2
    assert correct_df == 96
    assert old_column_count_df == 93
    assert correct_df != old_column_count_df
    assert 76_734 * 2_135 == 163_827_090
    assert 3_300_000 / 163_827_090 < 0.021
