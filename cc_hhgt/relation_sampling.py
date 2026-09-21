"""Relation-aware, non-destructive runtime edge scheduling.

The canonical graph is never rewritten.  Edges are assigned deterministically
to bounded runtime chunks, and the complete schedule is a coverage cycle.  A
cycle therefore retains 100% of available edge count and weighted signal while
limiting peak device residency.  Coexpression ordering is per source and sign
stratified using ``raw_effect``; the absolute ``weight`` alone is not treated as
direction.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd


STRUCTURAL_RELATIONS = frozenset(
    {
        "member_of",
        "member_of_family",
        "encoded_by",
        "targets",
        "physical_interaction",
        "curated_relation",
        "curated_drug_relation",
    }
)


@dataclass(frozen=True)
class RuntimeSamplingPolicy:
    edge_chunk_size: int = 250_000
    source_coverage_min: float = 0.95
    weighted_signal_retention_min: float = 0.90
    coverage_cycle: str = "all_runtime_chunks"
    # Fresh G0/G1/G2 uses a resident static backbone.  The legacy scheduler is
    # retained as the default for historical callers; formal callers must opt
    # in explicitly and name the only relations allowed to rotate.
    resident_backbone: bool = False
    variable_relation_types: tuple[str, ...] = (
        "coexpressed_positive",
        "coexpressed_negative",
    )

    def __post_init__(self) -> None:
        if self.edge_chunk_size <= 0:
            raise ValueError("edge_chunk_size must be positive")
        if not 0 < self.source_coverage_min <= 1:
            raise ValueError("source_coverage_min must be in (0, 1]")
        if not 0 < self.weighted_signal_retention_min <= 1:
            raise ValueError("weighted_signal_retention_min must be in (0, 1]")
        if self.resident_backbone and not self.variable_relation_types:
            raise ValueError("resident_backbone requires at least one variable relation type")
        if len(self.variable_relation_types) != len(set(self.variable_relation_types)):
            raise ValueError("variable_relation_types must be unique")


def _sign(frame: pd.DataFrame) -> pd.Series:
    raw_values = frame["raw_effect"] if "raw_effect" in frame else pd.Series(np.nan, index=frame.index)
    raw = pd.to_numeric(raw_values, errors="coerce")
    relation = frame.relation_type.astype(str)
    inferred = pd.Series(0, index=frame.index, dtype="int8")
    inferred.loc[relation.str.contains("positive|enriched", case=False, regex=True)] = 1
    inferred.loc[relation.str.contains("negative|depleted", case=False, regex=True)] = -1
    inferred.loc[raw.gt(0)] = 1
    inferred.loc[raw.lt(0)] = -1
    return inferred


def _stable_tiebreak(frame: pd.DataFrame) -> pd.Series:
    if "edge_id" in frame:
        return frame.edge_id.astype(str)
    keys = frame[
        [
            c
            for c in (
                "source_type",
                "source_canonical_id",
                "relation_type",
                "target_type",
                "target_canonical_id",
                "cancer_id",
            )
            if c in frame
        ]
    ].astype("string").fillna("")
    return keys.agg("\x1f".join, axis=1).map(lambda x: hashlib.sha1(x.encode("utf-8")).hexdigest())


def schedule_runtime_edges(edges: pd.DataFrame, policy: RuntimeSamplingPolicy) -> pd.DataFrame:
    """Return a scheduling table; no canonical row is dropped.

    Coexpression is ordered per source and sign, strongest first, then spread
    round-robin across runtime chunks.  Other relations are ordered by relation,
    evidence priority and degree-aware stable keys.  ``runtime_chunk`` is an
    execution property, not a new frozen graph.
    """

    if edges.empty:
        return edges.assign(runtime_chunk=pd.Series(dtype="int64"), runtime_order=pd.Series(dtype="int64"))
    output = edges.copy()
    if policy.resident_backbone:
        return _schedule_with_resident_backbone(output, policy)
    output["_weight_abs"] = pd.to_numeric(output.get("weight", 1.0), errors="coerce").fillna(0).abs()
    output["_sign"] = _sign(output)
    output["_tie"] = _stable_tiebreak(output)
    output["_relation_family"] = (
        output.source_type.astype(str)
        + "__"
        + output.relation_type.astype(str)
        + "__"
        + output.target_type.astype(str)
    )

    pieces: list[pd.DataFrame] = []
    for _, group in output.groupby("_relation_family", observed=True, sort=True):
        relation = str(group.relation_type.iloc[0])
        if relation == "coexpressed_with":
            # raw_effect is the only signed field in the frozen graph.  Cancer
            # joins the source key so abundant cancers cannot monopolize early
            # chunks for the same lncRNA.
            source_keys = ["source_canonical_id", "_sign"]
            if "cancer_id" in group:
                source_keys.append("cancer_id")
            ordered = group.sort_values(
                [*source_keys, "_weight_abs", "_tie"],
                ascending=[True] * len(source_keys) + [False, True],
                kind="stable",
            ).copy()
            ordered["_within_source_rank"] = ordered.groupby(source_keys, observed=True).cumcount()
            ordered = ordered.sort_values(
                ["_within_source_rank", "_sign", "source_canonical_id", "_weight_abs", "_tie"],
                ascending=[True, True, True, False, True],
                kind="stable",
            )
        else:
            priority = pd.Series(2, index=group.index)
            source_database = group.get("source_database", pd.Series("", index=group.index)).astype(str).str.lower()
            priority.loc[source_database.str.contains("experiment", regex=False)] = 0
            priority.loc[source_database.str.contains("curated", regex=False)] = 1
            ordered = group.assign(_evidence_priority=priority).sort_values(
                ["_evidence_priority", "_weight_abs", "source_canonical_id", "_tie"],
                ascending=[True, False, True, True],
                kind="stable",
            )
        ordered["_relation_rank"] = np.arange(len(ordered), dtype=np.int64)
        pieces.append(ordered)

    # Mix every relation family into every feasible runtime chunk.  This keeps
    # the CC-HHGT metadata stable and avoids a chunk containing only the very
    # large coexpression family.  Increase the chunk count until the hard peak
    # residency limit is respected (round-robin imbalance is at most one edge
    # per relation family).
    n_chunks = max(1, int(np.ceil(len(output) / policy.edge_chunk_size)))
    while True:
        counts = np.zeros(n_chunks, dtype=np.int64)
        for piece in pieces:
            counts += np.bincount(
                (piece["_relation_rank"].to_numpy(np.int64) % n_chunks),
                minlength=n_chunks,
            )
        if int(counts.max(initial=0)) <= policy.edge_chunk_size:
            break
        n_chunks += 1
    for piece in pieces:
        piece["runtime_chunk"] = piece["_relation_rank"].to_numpy(np.int64) % n_chunks
    scheduled = pd.concat(pieces, ignore_index=False)
    scheduled["runtime_order"] = scheduled.groupby("runtime_chunk", sort=True).cumcount().astype(np.int64)
    scheduled["sampling_strategy"] = np.where(
        scheduled.relation_type.astype(str).eq("coexpressed_with"),
        "per_source_sign_stratified_weight_ordered_runtime_cycle",
        "relation_aware_weight_ordered_runtime_cycle",
    )
    return scheduled.sort_index().drop(
        columns=[c for c in ("_weight_abs", "_sign", "_tie", "_relation_family", "_within_source_rank", "_evidence_priority", "_relation_rank") if c in scheduled]
    )


def _schedule_with_resident_backbone(
    edges: pd.DataFrame,
    policy: RuntimeSamplingPolicy,
) -> pd.DataFrame:
    """Keep every non-variable edge resident and rotate only named relations.

    ``edge_chunk_size`` is intentionally a cap on *variable source rows*, not
    total active rows.  Resident edges are stored once with ``runtime_chunk``
    -1; :func:`runtime_chunk_positions` co-locates them with every variable
    chunk at execution time.  This compact representation avoids multiplying
    a million-row backbone by 16--42 chunks in host memory.
    """

    if "edge_id" not in edges:
        raise ValueError("Resident-backbone scheduling requires canonical edge_id")
    edge_id = edges.edge_id.astype(str)
    if edge_id.eq("").any() or edge_id.duplicated().any():
        raise RuntimeError("Resident-backbone scheduling requires unique non-empty edge_id")
    variable_types = set(map(str, policy.variable_relation_types))
    relation = edges.relation_type.astype(str)
    variable = edges.loc[relation.isin(variable_types)].copy()
    resident = edges.loc[~relation.isin(variable_types)].copy()

    if variable.empty:
        only = resident.copy()
        only["runtime_chunk"] = -1
        only["runtime_order"] = np.arange(len(only), dtype=np.int64)
        only["runtime_residency"] = "resident_backbone"
        only["canonical_edge_id"] = only.edge_id.astype(str)
        only["sampling_strategy"] = "complete_resident_backbone"
        return only.reset_index(drop=True)

    # Canonical order is independent of source parquet row order.  Magnitude
    # breaks ties only after cancer/source/sign, matching the frozen formal
    # schedule.  ``edge_id`` is the final stable tie-break.
    variable["_sign"] = _sign(variable)
    variable["_weight_abs"] = pd.to_numeric(
        variable.get("weight", 1.0), errors="coerce"
    ).fillna(0).abs()
    variable["_cancer"] = variable.get(
        "cancer_id", pd.Series("", index=variable.index)
    ).astype("string").fillna("")
    source_column = (
        "source_canonical_id" if "source_canonical_id" in variable else "source_id"
    )
    variable = variable.sort_values(
        ["_cancer", source_column, "_sign", "_weight_abs", "edge_id"],
        ascending=[True, True, True, False, True],
        kind="stable",
    ).reset_index(drop=True)

    n_chunks = max(1, int(np.ceil(len(variable) / policy.edge_chunk_size)))
    # Contiguous balanced slices give each chunk either floor(N/K) or
    # ceil(N/K) rows.  Unlike relation-wise round-robin, this never exceeds
    # the declared variable-row cap and never creates a tiny final chunk.
    base, extra = divmod(len(variable), n_chunks)
    chunk_id = np.concatenate(
        [np.full(base + (1 if index < extra else 0), index, dtype=np.int64) for index in range(n_chunks)]
    )
    if len(chunk_id) != len(variable):
        raise AssertionError("Variable runtime schedule width drift")
    variable["runtime_chunk"] = chunk_id
    variable["runtime_residency"] = "rotating_variable"
    variable["canonical_edge_id"] = variable.edge_id.astype(str)
    variable["sampling_strategy"] = (
        "canonical_cancer_source_sign_weight_balanced_complete_cycle"
    )

    resident["runtime_chunk"] = -1
    resident["runtime_residency"] = "resident_backbone"
    resident["canonical_edge_id"] = resident.edge_id.astype(str)
    resident["sampling_strategy"] = "complete_resident_backbone_compact"
    scheduled = pd.concat([resident, variable], ignore_index=True, sort=False)
    scheduled = scheduled.sort_values(
        ["runtime_chunk", "source_type", "relation_type", "target_type", "edge_id"],
        kind="stable",
    ).reset_index(drop=True)
    scheduled["runtime_order"] = scheduled.groupby(
        "runtime_chunk", sort=True
    ).cumcount().astype(np.int64)
    scheduled = scheduled.drop(columns=["_sign", "_weight_abs", "_cancer"], errors="ignore")

    rotating = scheduled.loc[scheduled.runtime_residency.eq("rotating_variable")]
    if rotating.canonical_edge_id.duplicated().any() or len(rotating) != len(variable):
        raise RuntimeError("A variable edge was omitted or duplicated across the coverage cycle")
    scheduled_resident = scheduled.loc[
        scheduled.runtime_residency.eq("resident_backbone")
    ]
    if (
        len(scheduled_resident) != len(resident)
        or scheduled_resident.canonical_edge_id.duplicated().any()
        or not scheduled_resident.runtime_chunk.eq(-1).all()
    ):
        raise RuntimeError("Compact resident backbone representation is invalid")
    variable_sizes = rotating.groupby("runtime_chunk", observed=True).size().reindex(
        range(n_chunks), fill_value=0
    )
    if int(variable_sizes.max()) > policy.edge_chunk_size:
        raise RuntimeError("Variable runtime chunk exceeds edge_chunk_size")
    if int(variable_sizes.max() - variable_sizes.min()) > 1:
        raise RuntimeError("Variable runtime chunks are not balanced")
    return scheduled


def runtime_chunk_positions(scheduled: pd.DataFrame) -> dict[int, np.ndarray]:
    """Return active row positions, co-locating compact resident rows."""

    if "runtime_chunk" not in scheduled:
        raise ValueError("Edges have not been scheduled")
    chunks = pd.to_numeric(scheduled.runtime_chunk, errors="raise").astype(int)
    if "runtime_residency" not in scheduled:
        return {
            int(key): np.asarray(value, dtype=np.int64)
            for key, value in chunks.groupby(chunks, sort=True).indices.items()
        }
    residency = scheduled.runtime_residency.astype(str)
    resident_positions = np.flatnonzero(residency.eq("resident_backbone").to_numpy())
    if len(resident_positions) and not chunks.iloc[resident_positions].eq(-1).all():
        raise RuntimeError("Resident rows must use compact runtime_chunk=-1")
    rotating = residency.eq("rotating_variable")
    variable_chunks = sorted(chunks.loc[rotating].unique())
    if variable_chunks:
        if variable_chunks != list(range(len(variable_chunks))):
            raise RuntimeError(f"Variable chunks are not contiguous: {variable_chunks}")
    else:
        variable_chunks = [0]
    positions: dict[int, np.ndarray] = {}
    for chunk in variable_chunks:
        variable_positions = np.flatnonzero(
            (rotating & chunks.eq(int(chunk))).to_numpy()
        )
        positions[int(chunk)] = np.concatenate(
            [resident_positions, variable_positions]
        ).astype(np.int64, copy=False)
    return positions


def iter_runtime_chunks(scheduled: pd.DataFrame) -> Iterator[pd.DataFrame]:
    for positions in runtime_chunk_positions(scheduled).values():
        group = scheduled.iloc[positions].sort_values(
            ["runtime_residency", "runtime_order"], kind="stable"
        )
        yield group.drop(columns=["runtime_chunk", "runtime_order", "sampling_strategy"], errors="ignore")


def signal_retention_audit(
    original: pd.DataFrame,
    available: pd.DataFrame,
    scheduled: pd.DataFrame,
    policy: RuntimeSamplingPolicy,
) -> pd.DataFrame:
    """Audit canonical -> contract-available -> one full runtime coverage cycle."""

    def family(frame: pd.DataFrame) -> pd.Series:
        return (
            frame.source_type.astype(str)
            + "__"
            + frame.relation_type.astype(str)
            + "__"
            + frame.target_type.astype(str)
        )

    frames = []
    original = original.assign(_family=family(original), _sign=_sign(original))
    available = available.assign(_family=family(available), _sign=_sign(available))
    scheduled = scheduled.assign(_family=family(scheduled), _sign=_sign(scheduled))
    # Resident execution rows are intentionally repeated.  Retention is a
    # canonical-edge property, so count each canonical edge once here while
    # preserving the full execution table for peak-residency statistics.
    scheduled_execution = scheduled
    if "canonical_edge_id" in scheduled:
        scheduled = scheduled.drop_duplicates("canonical_edge_id", keep="first")
    families = sorted(set(original._family) | set(available._family) | set(scheduled._family))
    for name in families:
        o = original.loc[original._family.eq(name)]
        a = available.loc[available._family.eq(name)]
        s = scheduled.loc[scheduled._family.eq(name)]
        s_execution = scheduled_execution.loc[scheduled_execution._family.eq(name)]
        o_weight = pd.to_numeric(o.get("weight", 1), errors="coerce").fillna(0).abs().sum()
        a_weight = pd.to_numeric(a.get("weight", 1), errors="coerce").fillna(0).abs().sum()
        s_weight = pd.to_numeric(s.get("weight", 1), errors="coerce").fillna(0).abs().sum()
        source_total = a.source_canonical_id.astype(str).nunique() if len(a) else 0
        source_sampled = s.source_canonical_id.astype(str).nunique() if len(s) else 0
        pos_a = pd.to_numeric(a.loc[a._sign.gt(0), "weight"], errors="coerce").fillna(0).abs().sum()
        pos_s = pd.to_numeric(s.loc[s._sign.gt(0), "weight"], errors="coerce").fillna(0).abs().sum()
        neg_a = pd.to_numeric(a.loc[a._sign.lt(0), "weight"], errors="coerce").fillna(0).abs().sum()
        neg_s = pd.to_numeric(s.loc[s._sign.lt(0), "weight"], errors="coerce").fillna(0).abs().sum()
        source_coverage = source_sampled / source_total if source_total else 1.0
        weighted = s_weight / a_weight if a_weight else 1.0
        positive = pos_s / pos_a if pos_a else 1.0
        negative = neg_s / neg_a if neg_a else 1.0
        frames.append(
            {
                "relation_family": name,
                "original_edge_count": int(len(o)),
                "available_edge_count": int(len(a)),
                "sampled_edge_count": int(len(s)),
                "runtime_chunks": int(s_execution.runtime_chunk.nunique()) if len(s_execution) and "runtime_chunk" in s_execution else 0,
                "peak_runtime_edges": int(s_execution.groupby("runtime_chunk").size().max()) if len(s_execution) and "runtime_chunk" in s_execution else 0,
                "node_coverage": source_coverage,
                "weighted_signal_retention": weighted,
                "positive_mass_retention": positive,
                "negative_mass_retention": negative,
                "canonical_signal_mass": float(o_weight),
                "available_signal_mass": float(a_weight),
                "sampled_signal_mass": float(s_weight),
                "destructive_offline_truncation": False,
                "coverage_cycle": policy.coverage_cycle,
                "audit_status": "PASS"
                if source_coverage >= policy.source_coverage_min
                and weighted >= policy.weighted_signal_retention_min
                and positive >= policy.weighted_signal_retention_min
                and negative >= policy.weighted_signal_retention_min
                else "FAIL",
            }
        )
    return pd.DataFrame(frames)
