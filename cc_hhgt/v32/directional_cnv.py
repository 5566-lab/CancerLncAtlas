"""Direction-preserving segment CNV materialisation for local-CNV audits.

The historical compact store retained absolute segment burden.  That is useful
for alteration magnitude but cannot distinguish amplification from deletion.
This module remaps the already staged TCGA segment files and preserves signed
segment mean, amplification, deletion, neutral and unavailable semantics.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DirectionalEntityCNV:
    signed_mean: np.ndarray
    callable: np.ndarray
    amplification: np.ndarray
    deletion: np.ndarray

    @property
    def neutral(self) -> np.ndarray:
        return self.callable & ~self.amplification & ~self.deletion


class DirectionalCNVStore:
    """Read-only memory-mapped view of one directional CNV partition."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.patient_ids = tuple(json.loads((self.root / "patient_ids.json").read_text()))
        self.lncrna_ids = tuple(json.loads((self.root / "lncrna_ids.json").read_text()))
        self.pathway_ids = tuple(json.loads((self.root / "pathway_ids.json").read_text()))
        self.patient_index = {value: index for index, value in enumerate(self.patient_ids)}
        self.lncrna_index = {value: index for index, value in enumerate(self.lncrna_ids)}
        self.pathway_index = {value: index for index, value in enumerate(self.pathway_ids)}
        self.lncrna_signed = np.load(self.root / "lncrna_signed_mean.npy", mmap_mode="r")
        self.lncrna_callable = np.load(self.root / "lncrna_callable.npy", mmap_mode="r")
        self.lncrna_amplification = np.load(self.root / "lncrna_amplification.npy", mmap_mode="r")
        self.lncrna_deletion = np.load(self.root / "lncrna_deletion.npy", mmap_mode="r")
        self.pathway_signed = np.load(self.root / "pathway_signed_mean.npy", mmap_mode="r")
        self.pathway_callable = np.load(self.root / "pathway_callable.npy", mmap_mode="r")
        self.pathway_amplification = np.load(self.root / "pathway_amplification.npy", mmap_mode="r")
        self.pathway_deletion = np.load(self.root / "pathway_deletion.npy", mmap_mode="r")


def map_segments_directional(
    segment: pd.DataFrame,
    intervals: pd.DataFrame,
    entity_ids: Sequence[str],
    *,
    event_threshold: float = 0.30,
) -> DirectionalEntityCNV:
    """Map signed segment means at entity interval midpoints.

    Multiple intervals for one entity are averaged without taking an absolute
    value.  Calls are only made for observed finite segments; unavailable is
    represented by ``callable=False`` and ``signed_mean=NaN``.
    """

    identifiers = tuple(map(str, entity_ids))
    size = len(identifiers)
    total = np.zeros(size, dtype=np.float64)
    counts = np.zeros(size, dtype=np.int32)
    amplification = np.zeros(size, dtype=bool)
    deletion = np.zeros(size, dtype=bool)
    signed = np.full(size, np.nan, dtype=np.float32)
    if not size or segment.empty or intervals.empty:
        return DirectionalEntityCNV(signed, counts > 0, amplification, deletion)

    required_segment = {"chromosome", "start", "end", "value"}
    required_interval = {"entity_id", "chromosome", "start", "end"}
    if missing := sorted(required_segment - set(segment.columns)):
        raise ValueError(f"segment table lacks columns: {missing}")
    if missing := sorted(required_interval - set(intervals.columns)):
        raise ValueError(f"interval table lacks columns: {missing}")

    entity_index = {value: position for position, value in enumerate(identifiers)}
    local_segment = segment.copy()
    local_segment["chromosome"] = (
        local_segment.chromosome.astype(str).str.replace(r"^chr", "", regex=True)
    )
    local_segment["start"] = pd.to_numeric(local_segment.start, errors="raise").astype(np.int64)
    local_segment["end"] = pd.to_numeric(local_segment.end, errors="raise").astype(np.int64)
    local_segment["value"] = pd.to_numeric(local_segment.value, errors="coerce")
    local_intervals = intervals.copy()
    local_intervals["chromosome"] = (
        local_intervals.chromosome.astype(str).str.replace(r"^chr", "", regex=True)
    )
    local_intervals["start"] = pd.to_numeric(local_intervals.start, errors="raise").astype(np.int64)
    local_intervals["end"] = pd.to_numeric(local_intervals.end, errors="raise").astype(np.int64)

    for chromosome, local_iv in local_intervals.groupby(
        "chromosome", observed=True, sort=False
    ):
        local_seg = local_segment.loc[
            local_segment.chromosome.eq(str(chromosome))
        ].sort_values("start", kind="stable")
        if local_seg.empty:
            continue
        starts = local_seg.start.to_numpy(np.int64)
        ends = local_seg.end.to_numpy(np.int64)
        values = local_seg.value.to_numpy(float)
        midpoint = (
            local_iv.start.to_numpy(np.int64) + local_iv.end.to_numpy(np.int64)
        ) // 2
        positions = np.searchsorted(starts, midpoint, side="right") - 1
        safe_positions = np.maximum(positions, 0)
        valid = (
            (positions >= 0)
            & (midpoint <= ends[safe_positions])
            & np.isfinite(values[safe_positions])
        )
        entity_positions = local_iv.entity_id.astype(str).map(entity_index).to_numpy(float)
        valid &= np.isfinite(entity_positions)
        if not valid.any():
            continue
        destination = entity_positions[valid].astype(int)
        observed = values[positions[valid]]
        np.add.at(total, destination, observed)
        np.add.at(counts, destination, 1)
        np.logical_or.at(amplification, destination, observed >= float(event_threshold))
        np.logical_or.at(deletion, destination, observed <= -float(event_threshold))

    callable_value = counts > 0
    signed[callable_value] = (total[callable_value] / counts[callable_value]).astype(np.float32)
    return DirectionalEntityCNV(signed, callable_value, amplification, deletion)


def aggregate_pathway_directional(
    gene_cnv: DirectionalEntityCNV,
    pathway_members: Sequence[np.ndarray],
    *,
    canonical_callable: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Aggregate signed gene CNV while retaining event direction separately."""

    count = len(pathway_members)
    signed = np.full(count, np.nan, dtype=np.float32)
    amplification_fraction = np.full(count, np.nan, dtype=np.float32)
    deletion_fraction = np.full(count, np.nan, dtype=np.float32)
    callable_value = np.zeros(count, dtype=bool)
    amplification = np.zeros(count, dtype=bool)
    deletion = np.zeros(count, dtype=bool)
    for position, members in enumerate(pathway_members):
        members = np.asarray(members, dtype=int)
        if not len(members):
            continue
        observed = gene_cnv.callable[members]
        if not observed.any():
            continue
        use = members[observed]
        signed[position] = float(np.mean(gene_cnv.signed_mean[use]))
        amplification_fraction[position] = float(np.mean(gene_cnv.amplification[use]))
        deletion_fraction[position] = float(np.mean(gene_cnv.deletion[use]))
        amplification[position] = bool(gene_cnv.amplification[use].any())
        deletion[position] = bool(gene_cnv.deletion[use].any())
        callable_value[position] = bool(observed.all())
    if canonical_callable is not None:
        canonical = np.asarray(canonical_callable, dtype=bool)
        if canonical.shape != callable_value.shape:
            raise ValueError("canonical pathway callability is not aligned")
        callable_value = canonical.copy()
        signed[~canonical] = np.nan
        amplification_fraction[~canonical] = np.nan
        deletion_fraction[~canonical] = np.nan
        amplification[~canonical] = False
        deletion[~canonical] = False
    return {
        "signed_mean": signed,
        "callable": callable_value,
        "amplification": amplification,
        "deletion": deletion,
        "neutral": callable_value & ~amplification & ~deletion,
        "amplification_fraction": amplification_fraction,
        "deletion_fraction": deletion_fraction,
    }


__all__ = [
    "DirectionalCNVStore", "DirectionalEntityCNV", "aggregate_pathway_directional",
    "map_segments_directional",
]
