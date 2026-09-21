from __future__ import annotations

import pandas as pd

from cc_hhgt.episodic_loco import EpisodicLOCOPlanner


def test_every_training_cancer_is_pseudoheldout_once_per_cycle() -> None:
    planner = EpisodicLOCOPlanner(["A", "B", "C"], ["TEST", "VAL"], seed=17)
    episodes = [planner.episode(epoch) for epoch in range(1, 4)]
    assert {episode.pseudoheldout_cancer for episode in episodes} == {"A", "B", "C"}
    assert all(set(episode.graph_heldout_cancers) >= {"TEST", "VAL"} for episode in episodes)
    assert all(episode.pair_feature_policy == "fully_masked_like_real_test" for episode in episodes)


def test_episode_loss_contains_only_pseudoheldout_candidates() -> None:
    planner = EpisodicLOCOPlanner(["A", "B"], ["TEST", "VAL"], seed=4)
    episode = planner.episode(1)
    frame = pd.DataFrame({"cancer_id": ["A", "A", "B"], "candidate_id": [1, 2, 3]})
    selected = planner.select_candidate_loss_rows(frame, episode)
    assert selected.cancer_id.eq(episode.pseudoheldout_cancer).all()
