"""Fixture checks for unvalidated candidate generation with held-out evaluation.

Synthetic only.  The fixture plants a latent binding rule so the estimator's
power can be checked; nothing here is evidence that any real lncRNA binds any
real protein.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cc_hhgt.v32 import novel_binding_discovery as discovery

WEIGHTS = {"motif_affinity": 1.0, "coexpression_abs": 0.2, "rnbp_domain": 0.5}
STANDARDISATION = {
    "motif_affinity": {"center": 0.0, "scale": 1.0},
    "coexpression_abs": {"center": 0.3, "scale": 0.2},
    "rnbp_domain": {"center": 0.0, "scale": 1.0},
}


def _config(**overrides) -> discovery.DiscoveryConfig:
    payload = {
        "feature_weights": WEIGHTS,
        "standardisation": STANDARDISATION,
        "primary_feature": "motif_affinity",
        "budget": 20,
        "holdout_fraction": 0.3,
        "minimum_holdout": 5,
        "seed": 11,
    }
    payload.update(overrides)
    return discovery.DiscoveryConfig(**payload)


def _pairs() -> pd.DataFrame:
    lncrnas = [f"L{index:02d}" for index in range(20)]
    proteins = [f"P{index:02d}" for index in range(15)]
    return pd.DataFrame(
        [(lncrna, protein) for lncrna in lncrnas for protein in proteins],
        columns=["lncrna_id", "protein_id"],
    )


def _features(seed: int = 0):
    rng = np.random.default_rng(seed)
    frame = _pairs()
    frame["motif_affinity"] = rng.normal(0.0, 1.0, len(frame))
    frame["coexpression_abs"] = np.abs(rng.normal(0.3, 0.2, len(frame)))
    frame["rnbp_domain"] = (rng.random(len(frame)) < 0.7).astype(float)
    return frame


def _known_edges(features: pd.DataFrame, *, arm: str = "experimental") -> pd.DataFrame:
    threshold = features.motif_affinity.quantile(0.75)
    mask = features.motif_affinity.gt(threshold) & features.rnbp_domain.eq(1.0)
    frame = features.loc[mask, ["lncrna_id", "protein_id"]].copy()
    frame["evidence_arm"] = arm
    return frame.reset_index(drop=True)


def _scored_frame():
    return pd.DataFrame(
        {
            "lncrna_id": ["L1", "L1", "L1", "L1"],
            "protein_id": ["P1", "P2", "P3", "P4"],
            "score": [3.0, 1.0, 2.0, 0.0],
        }
    )


def test_validate_config_records_a_declared_unfitted_score() -> None:
    record = discovery.validate_config(_config())
    assert record["format"] == discovery.FORMAT
    assert record["scope"] == discovery.SCOPE
    assert record["score_is_fitted"] is False
    assert record["candidates_are_unvalidated"] is True
    assert record["config_digest"] == _config().digest()
    assert record["weights"]["motif_affinity"] == 1.0


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"feature_weights": {}}, "DISCOVERY_WEIGHTS_MISSING"),
        ({"primary_feature": "other"}, "DISCOVERY_PRIMARY_FEATURE_UNWEIGHTED"),
        ({"holdout_fraction": 1.0}, "DISCOVERY_HOLDOUT_FRACTION_INVALID"),
        ({"holdout_fraction": 0.0}, "DISCOVERY_HOLDOUT_FRACTION_INVALID"),
        ({"budget": 0}, "DISCOVERY_BUDGET_INVALID"),
        ({"minimum_holdout": 0}, "DISCOVERY_MINIMUM_HOLDOUT_INVALID"),
        (
            {"standardisation": {"motif_affinity": {"center": 0.0, "scale": 1.0}}},
            "DISCOVERY_STANDARDISATION_MISSING",
        ),
        (
            {
                "standardisation": {
                    **STANDARDISATION,
                    "rnbp_domain": {"center": 0.0, "scale": -1.0},
                }
            },
            "DISCOVERY_STANDARDISATION_INVALID",
        ),
        (
            {
                "standardisation": {
                    **STANDARDISATION,
                    "rnbp_domain": {"center": float("nan"), "scale": 1.0},
                }
            },
            "DISCOVERY_STANDARDISATION_INVALID",
        ),
        (
            {"feature_weights": {**WEIGHTS, "motif_affinity": float("inf")}},
            "DISCOVERY_WEIGHTS_NOT_FINITE",
        ),
    ],
)
def test_validate_config_fails_closed(overrides, message) -> None:
    with pytest.raises(RuntimeError, match=message):
        discovery.validate_config(_config(**overrides))


def test_split_is_deterministic_and_ignores_scores() -> None:
    known = _known_edges(_features())
    first = discovery.split_known_edges(known, _config())
    second = discovery.split_known_edges(known, _config())
    assert first.equals(second)
    assert set(first.partition) == {"scored", "holdout"}
    observed = float((first.partition == "holdout").mean())
    assert abs(observed - 0.3) <= discovery.HOLDOUT_FRACTION_TOLERANCE
    other_seed = discovery.split_known_edges(known, _config(seed=12))
    assert not first.partition.equals(other_seed.partition)


def test_split_rejects_duplicates_missing_columns_and_a_tiny_holdout() -> None:
    known = _known_edges(_features())
    duplicated = pd.concat([known, known.iloc[[0]]], ignore_index=True)
    with pytest.raises(RuntimeError, match="DISCOVERY_KNOWN_EDGE_DUPLICATE"):
        discovery.split_known_edges(duplicated, _config())
    with pytest.raises(RuntimeError, match="DISCOVERY_KNOWN_COLUMNS_MISSING"):
        discovery.split_known_edges(known.drop(columns=["protein_id"]), _config())
    with pytest.raises(RuntimeError, match="DISCOVERY_HOLDOUT_TOO_SMALL"):
        discovery.split_known_edges(known, _config(minimum_holdout=len(known) + 1))


def test_candidate_universe_excludes_scored_known_edges() -> None:
    known = _known_edges(_features())
    split = discovery.split_known_edges(known, _config())
    train = split.loc[split.partition.eq("scored"), ["lncrna_id", "protein_id"]]
    holdout = split.loc[split.partition.eq("holdout"), ["lncrna_id", "protein_id"]]
    universe = discovery.candidate_universe(split, train)
    universe_pairs = set(zip(universe.lncrna_id, universe.protein_id, strict=True))
    train_pairs = set(zip(train.lncrna_id, train.protein_id, strict=True))
    holdout_pairs = set(zip(holdout.lncrna_id, holdout.protein_id, strict=True))
    assert not (universe_pairs & train_pairs)
    assert holdout_pairs <= universe_pairs
    # The universe is the product of the node identifiers present in the known
    # graph minus the scored partition, never the full identifier cross product.
    expected = (
        len(set(split.lncrna_id.astype(str))) * len(set(split.protein_id.astype(str)))
        - len(train)
    )
    assert len(universe) == expected
    assert len(universe) <= 20 * 15 - len(train)
    assert len(universe) >= len(holdout)


def test_score_is_the_declared_weighted_sum() -> None:
    features = _features()
    universe = _pairs()
    scored = discovery.score_candidates(universe, features, _config())
    row = 0
    expected = (
        (features.motif_affinity.iloc[row] - 0.0) / 1.0 * 1.0
        + (features.coexpression_abs.iloc[row] - 0.3) / 0.2 * 0.2
        + (features.rnbp_domain.iloc[row] - 0.0) / 1.0 * 0.5
    )
    assert scored.score.iloc[row] == pytest.approx(expected)
    assert scored["contribution::motif_affinity"].iloc[row] == pytest.approx(
        features.motif_affinity.iloc[row]
    )
    assert scored.missing_features.iloc[row] == ""


def test_missing_feature_contributes_nothing_and_is_recorded() -> None:
    features = _features()
    features.loc[3, "coexpression_abs"] = np.nan
    scored = discovery.score_candidates(_pairs(), features, _config())
    assert scored["contribution::coexpression_abs"].iloc[3] == 0.0
    assert scored.score.iloc[3] == pytest.approx(
        features.motif_affinity.iloc[3] + 0.5 * features.rnbp_domain.iloc[3]
    )
    assert "coexpression_abs" in scored.missing_features.iloc[3]
    assert scored.attrs["missing_counts"]["coexpression_abs"] == 1


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda frame: frame.assign(label=0), "DISCOVERY_LABEL_LEAKAGE"),
        (lambda frame: frame.drop(columns=["rnbp_domain"]), "DISCOVERY_FEATURE_COLUMNS_MISSING"),
        (lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True), "DISCOVERY_FEATURE_ROW_DUPLICATE"),
        # Every candidate lacks the primary feature: a ranked list would be noise.
        (lambda frame: frame.assign(motif_affinity=np.nan), "DISCOVERY_PRIMARY_FEATURE_ABSENT"),
    ],
)
def test_scoring_refuses_leakage_and_unusable_features(mutate, message) -> None:
    features = mutate(_features())
    with pytest.raises(RuntimeError, match=message):
        discovery.score_candidates(_pairs(), features, _config())


def test_evaluate_computes_fixed_budget_metrics_against_the_degree_prior() -> None:
    scored = _scored_frame()
    holdout = pd.DataFrame(
        {"lncrna_id": ["L1", "L1", "L1"], "protein_id": ["P1", "P3", "P4"]}
    )
    train = pd.DataFrame({"lncrna_id": ["L1"], "protein_id": ["P2"]})
    config = _config(
        budget=2,
        holdout_fraction=0.5,
        minimum_holdout=1,
        feature_weights={"s": 1.0},
        standardisation={"s": {"center": 0.0, "scale": 1.0}},
        primary_feature="s",
    )
    record = discovery.evaluate(scored, holdout, train, config)
    assert record["universe"] == 4
    assert record["holdout_positives"] == 3
    assert record["precision_at_budget"] == pytest.approx(1.0)
    assert record["recall_at_budget"] == pytest.approx(2 / 3)
    assert record["average_precision"] == pytest.approx(1 / 3 + 1 / 3 + 1 / 4)
    assert record["degree_prior"]["precision_at_budget"] == pytest.approx(0.5)
    assert record["degree_prior"]["average_precision"] == pytest.approx(
        1 / 6 + 2 / 9 + 1 / 4
    )
    assert record["lift_average_precision"] > 0
    assert record["candidate_status"] == discovery.CANDIDATE_STATUS
    assert record["discovery_claim"] is False
    assert record["biological_validation"] is False
    assert record["degree_prior"]["uses_holdout_edges"] is False


def test_evaluate_refuses_an_unusable_evaluation() -> None:
    scored = _scored_frame()
    holdout = pd.DataFrame({"lncrna_id": ["L1"], "protein_id": ["P1"]})
    train = pd.DataFrame({"lncrna_id": ["L1"], "protein_id": ["P2"]})
    config = _config(
        budget=2,
        feature_weights={"s": 1.0},
        standardisation={"s": {"center": 0.0, "scale": 1.0}},
        primary_feature="s",
    )
    with pytest.raises(RuntimeError, match="DISCOVERY_HOLDOUT_EMPTY"):
        discovery.evaluate(scored, holdout.iloc[0:0], train, config)
    with pytest.raises(RuntimeError, match="DISCOVERY_PARTITION_OVERLAP"):
        discovery.evaluate(
            scored,
            pd.DataFrame({"lncrna_id": ["L1"], "protein_id": ["P2"]}),
            train,
            config,
        )
    with pytest.raises(RuntimeError, match="DISCOVERY_BUDGET_EXCEEDS_UNIVERSE"):
        discovery.evaluate(
            scored.iloc[0:1], holdout, pd.DataFrame({"lncrna_id": [], "protein_id": []}), config
        )


def test_discovery_pipeline_recovers_the_planted_rule() -> None:
    features = _features()
    known = _known_edges(features)
    candidates, record = discovery.discover_candidates(known, features, _config())
    assert record["held_out_known_edges"] >= 5
    assert record["scored_known_edges"] >= 1
    assert record["candidates_emitted"] == 20
    assert record["average_precision"] > record["holdout_prevalence"] + 0.4
    assert record["precision_at_budget"] > 3 * record["holdout_prevalence"]
    assert record["lift_average_precision"] > 0.2
    assert record["lift_precision_at_budget"] > 0
    assert record["known_train_edges_excluded_from_universe"] is True
    assert record["held_out_edges_used_for_scoring"] is False
    assert record["score_is_fitted"] is False
    assert record["discovery_claim"] is False
    assert record["biological_validation"] is False
    assert set(candidates.candidate_status) == {discovery.CANDIDATE_STATUS}
    assert set(candidates.evidence_required) == {"orthogonal_experimental_validation"}
    assert candidates.score.is_monotonic_decreasing
    assert candidates["rank"].tolist() == list(range(1, 21))
    split = discovery.split_known_edges(known, _config())
    scored_partition = split.loc[split.partition.eq("scored")]
    scored_pairs = set(
        zip(scored_partition.lncrna_id, scored_partition.protein_id, strict=True)
    )
    candidate_pairs = set(
        zip(candidates.lncrna_id, candidates.protein_id, strict=True)
    )
    assert not (candidate_pairs & scored_pairs)


def test_discovery_is_deterministic_and_seed_sensitive() -> None:
    features = _features()
    known = _known_edges(features)
    first, first_record = discovery.discover_candidates(known, features, _config())
    second, second_record = discovery.discover_candidates(known, features, _config())
    assert first.equals(second)
    assert first_record["config_digest"] == second_record["config_digest"]
    assert first_record["average_precision"] == second_record["average_precision"]
    third, third_record = discovery.discover_candidates(known, features, _config(seed=12))
    assert third_record["seed"] == 12
    assert not first.lncrna_id.equals(third.lncrna_id) or not first.protein_id.equals(
        third.protein_id
    )


def test_discovery_filters_by_declared_evidence_arm() -> None:
    features = _features()
    experimental = _known_edges(features)
    predicted = _known_edges(features).assign(evidence_arm="predicted")
    mixed = pd.concat([experimental, predicted], ignore_index=True)
    candidates, record = discovery.discover_candidates(mixed, features, _config())
    assert record["evidence_arm"] == "experimental"
    assert record["held_out_known_edges"] + record["scored_known_edges"] == len(
        experimental.drop_duplicates(["lncrna_id", "protein_id"])
    )


def test_discovery_refuses_a_budget_larger_than_the_universe() -> None:
    features = _features().iloc[0:40]
    known = _known_edges(_features()).iloc[0:12]
    with pytest.raises(RuntimeError):
        discovery.discover_candidates(known, features, _config(budget=500, minimum_holdout=1))


def test_module_exposes_no_fitting_entry_point() -> None:
    forbidden = {"fit", "train", "fit_transform", "optimizer", "backward"}
    assert forbidden.isdisjoint(dir(discovery))
    assert discovery.CANDIDATE_STATUS == "unvalidated_candidate"
    assert discovery.SCOPE.endswith("HELD_OUT_EVALUATION")
