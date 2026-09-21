"""Deterministic episodic pseudoheldout LOCO scheduling."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .graph_contract import (
    DeploymentContract,
    FoldGraphMode,
    FoldGraphResult,
    build_fold_graph,
)


@dataclass(frozen=True)
class PseudoheldoutEpisode:
    epoch: int
    pseudoheldout_cancer: str
    graph_heldout_cancers: tuple[str, ...]
    candidate_policy: str = "loss_only_on_pseudoheldout_cancer"
    pair_feature_policy: str = "fully_masked_like_real_test"


class EpisodicLOCOPlanner:
    """Cycle through every training cancer in a seed-stable random order."""

    def __init__(
        self,
        training_cancers: Iterable[str],
        formal_heldout_cancers: Iterable[str],
        *,
        seed: int,
    ) -> None:
        training = sorted(set(map(str, training_cancers)))
        formal = sorted(set(map(str, formal_heldout_cancers)))
        overlap = set(training) & set(formal)
        if not training:
            raise ValueError("Episodic LOCO needs at least one training cancer")
        if overlap:
            raise ValueError(f"Training/formal heldout cancer overlap: {sorted(overlap)}")
        self.training_cancers = tuple(training)
        self.formal_heldout_cancers = tuple(formal)
        self.seed = int(seed)

    def episode(self, epoch: int) -> PseudoheldoutEpisode:
        if epoch < 1:
            raise ValueError("epoch is one-based")
        cycle = (epoch - 1) // len(self.training_cancers)
        offset = (epoch - 1) % len(self.training_cancers)
        order = np.asarray(self.training_cancers, dtype=object)
        rng = np.random.default_rng(self.seed + cycle * 1_000_003)
        rng.shuffle(order)
        cancer = str(order[offset])
        heldout = tuple(sorted({*self.formal_heldout_cancers, cancer}))
        return PseudoheldoutEpisode(epoch, cancer, heldout)

    def build_graph(
        self,
        edges: pd.DataFrame,
        episode: PseudoheldoutEpisode,
        *,
        contract: DeploymentContract | str,
        reference_only: Iterable[str] = (),
    ) -> FoldGraphResult:
        return build_fold_graph(
            edges,
            heldout_cancers=episode.graph_heldout_cancers,
            mode=FoldGraphMode.PSEUDOHELDOUT,
            contract=contract,
            reference_only=reference_only,
        )

    @staticmethod
    def select_candidate_loss_rows(frame: pd.DataFrame, episode: PseudoheldoutEpisode) -> pd.DataFrame:
        if "cancer_id" not in frame:
            raise ValueError("Candidate frame lacks cancer_id")
        output = frame.loc[frame.cancer_id.astype(str).eq(episode.pseudoheldout_cancer)].copy()
        if output.empty:
            raise RuntimeError(f"Pseudoheldout cancer has no candidate loss rows: {episode.pseudoheldout_cancer}")
        return output


def real_test_graph(
    edges: pd.DataFrame,
    *,
    formal_heldout_cancers: Iterable[str],
    contract: DeploymentContract | str,
    reference_only: Iterable[str] = (),
) -> FoldGraphResult:
    """Formal evaluation wrapper that delegates to the same graph builder."""

    return build_fold_graph(
        edges,
        heldout_cancers=formal_heldout_cancers,
        mode=FoldGraphMode.REAL_TEST,
        contract=contract,
        reference_only=reference_only,
    )
