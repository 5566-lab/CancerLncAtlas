"""Patient-level intervention on one node of a frozen graph model.

Two registered estimands live here.

**C19 - patient relation shift.** For one patient, one protein node ``P`` and one
declared omics channel, define

    delta_(L, pw)(P) = f_(L, pw)(theta, M; do(P = inactivated))
                     - f_(L, pw)(theta, M; P = observed)

where ``f`` is the frozen membership logit of the graph model ``theta``, ``M``
is the graph and the observed pair features, and *only* the declared channel of
P's encoding gene changes.  Every other input stays byte-identical.  This is a
model-conditional interventional contrast.  It is not a causal effect in
patients, it is not an estimate of what would happen in a tumour, and it cannot
be published as a biological finding without orthogonal evidence.

**C20 - interaction attribution with falsification controls.** A message-passing
model can only move information along edges, so "P affects the L-pathway score"
always provably travels through P's outgoing edges.  What is informative is
*which* outgoing edges carry it, and whether the effect is specific to P's
partners rather than a generic consequence of adding signal to the graph.  This
module therefore reports an attribution share for a declared path set plus two
controls that can falsify the interaction reading:

* ``rewire``: P's outgoing edges are redirected to random nodes of the same
  target type.  If the pair effect survives, it was never about P's actual
  interaction partners.
* ``placebo``: the same intervention is applied to a matched node that has no
  path to the pair.  If the pair effect appears there too, the estimator is
  measuring something global, not the interaction.

The result is one of ``INTERACTION_ATTRIBUTED_UNDER_FROZEN_MODEL``,
``NOT_INTERACTION_SPECIFIC``, ``VALUE_ONLY_EFFECT`` or ``NO_EFFECT``.  None of
them is a proof of mediation: ``biological_mediation_proven`` is always false.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import copy
import hashlib
import json

import numpy as np
import pandas as pd

from . import omics_node_encoding as omics

FORMAT = "CANCERLNCATLAS_V32_PATIENT_INTERVENTION_V1"
SCOPE = "MODEL_CONDITIONAL_INTERVENTIONAL_CONTRAST_NON_CAUSAL"
MODES = ("observed", "inactivated")
VERDICTS = (
    "NO_EFFECT",
    "VALUE_ONLY_EFFECT",
    "NOT_INTERACTION_SPECIFIC",
    "INTERACTION_ATTRIBUTED_UNDER_FROZEN_MODEL",
)
EFFECT_TOLERANCE = 1e-7
ATTRIBUTION_MIN_SHARE = 0.5


@dataclass(frozen=True)
class Intervention:
    """One declared counterfactual state for one node."""

    cancer_id: str
    subject_id: str
    node_type: str
    node_id: str
    channel: str
    mode: str = "inactivated"
    value: float | None = None

    def identifier(self) -> str:
        payload = {
            "cancer_id": self.cancer_id,
            "subject_id": self.subject_id,
            "node_type": self.node_type,
            "node_id": self.node_id,
            "channel": self.channel,
            "mode": self.mode,
            "value": None if self.value is None else float(self.value),
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return digest[:32]


def _require_finite(value, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise RuntimeError(f"PATIENT_CONDITIONING_VALUE_NOT_NUMERIC: {name}") from None
    if not np.isfinite(number):
        raise RuntimeError(f"PATIENT_CONDITIONING_VALUE_NOT_FINITE: {name}")
    return number


def validate_intervention(
    intervention: Intervention, evidence: omics.OmicsEvidence, bundle
) -> dict:
    """Refuse any patient-level claim the inputs cannot carry."""

    if not isinstance(intervention, Intervention):
        raise RuntimeError("PATIENT_CONDITIONING_INTERVENTION_TYPE")
    if evidence.scope != "subject":
        raise RuntimeError("PATIENT_CONDITIONING_SUBJECT_SCOPE_REQUIRED")
    if str(intervention.subject_id) != evidence.subject_id:
        raise RuntimeError("PATIENT_CONDITIONING_SUBJECT_MISMATCH")
    if str(intervention.cancer_id) != evidence.cancer_id:
        raise RuntimeError("PATIENT_CONDITIONING_CANCER_MISMATCH")
    if intervention.mode not in MODES:
        raise RuntimeError(f"PATIENT_CONDITIONING_UNKNOWN_MODE: {intervention.mode}")
    if not str(intervention.node_id or "").strip():
        raise RuntimeError("PATIENT_CONDITIONING_NODE_ID_MISSING")
    node_type = str(intervention.node_type)
    if node_type not in bundle.node_maps:
        raise RuntimeError(f"PATIENT_CONDITIONING_NODE_TYPE_UNKNOWN: {node_type}")
    if str(intervention.node_id) not in bundle.node_maps[node_type]:
        raise RuntimeError(
            f"PATIENT_CONDITIONING_NODE_NOT_IN_GRAPH: {node_type}:{intervention.node_id}"
        )
    if intervention.channel not in evidence.channels:
        raise RuntimeError(
            f"PATIENT_CONDITIONING_CHANNEL_MISSING: {intervention.channel}"
        )
    frame = evidence.standardised
    selected = frame[
        frame.node_type.astype(str).eq(node_type)
        & frame.canonical_id.astype(str).eq(str(intervention.node_id))
        & frame.channel.astype(str).eq(str(intervention.channel))
    ]
    if selected.empty:
        raise RuntimeError(
            "PATIENT_CONDITIONING_PARTICIPANT_UNMEASURED: "
            f"{node_type}:{intervention.node_id} has no {intervention.channel} measurement"
        )
    observed = float(selected.value.iloc[0])
    if intervention.mode == "inactivated":
        if intervention.value is None:
            raise RuntimeError("PATIENT_CONDITIONING_INTERVENTION_VALUE_MISSING")
        value = _require_finite(intervention.value, "intervention value")
        if value == observed:
            raise RuntimeError("PATIENT_CONDITIONING_NO_INTERVENTION_CONTRAST")
    else:
        value = observed
    if int(bundle.node_input_dim) <= omics.DEGREE_CHANNELS:
        raise RuntimeError("PATIENT_CONDITIONING_BUNDLE_NOT_AUGMENTED")
    return {
        "scope": SCOPE,
        "intervention": {
            "cancer_id": str(intervention.cancer_id),
            "subject_id": str(intervention.subject_id),
            "node_type": node_type,
            "node_id": str(intervention.node_id),
            "channel": str(intervention.channel),
            "mode": str(intervention.mode),
            "observed_value": observed,
            "intervention_value": value,
            "counterfactual_value_is_declared": intervention.mode == "inactivated",
        },
        "estimand": (
            "delta = f(theta, M; do("
            + node_type
            + ":"
            + str(intervention.node_id)
            + "."
            + str(intervention.channel)
            + " = "
            + repr(value)
            + ")) - f(theta, M; observed value), with every other input fixed"
        ),
        "subject_identified": True,
        "causal_claim": False,
        "requires_orthogonal_validation": True,
    }


def observed_blocks(bundle, evidence: omics.OmicsEvidence) -> dict[str, np.ndarray]:
    result = omics.node_feature_block(bundle.nodes, evidence)
    return result["blocks"]


def _canonical_rows(bundle, blocks: dict[str, np.ndarray]) -> dict[str, list[str]]:
    order: dict[str, list[str]] = {}
    for node_type in bundle.hetero_data.node_types:
        ids = getattr(bundle.hetero_data[str(node_type)], "canonical_ids", None)
        if ids is None:
            raise RuntimeError(f"PATIENT_CONDITIONING_BUNDLE_IDS_MISSING: {node_type}")
        order[str(node_type)] = [str(value) for value in ids]
        if len(order[str(node_type)]) != np.asarray(blocks[str(node_type)]).shape[0]:
            raise RuntimeError(f"PATIENT_CONDITIONING_BLOCK_ROW_MISMATCH: {node_type}")
    return order


def intervened_blocks(
    bundle, blocks: dict[str, np.ndarray], evidence: omics.OmicsEvidence,
    intervention: Intervention,
) -> tuple[dict[str, np.ndarray], dict]:
    """Graph-aligned version of :func:`intervene_blocks`."""

    order = _canonical_rows(bundle, blocks)
    node_type = str(intervention.node_type)
    channel = str(intervention.channel)
    channels = list(evidence.channels)
    if channel not in channels:
        raise RuntimeError(f"PATIENT_CONDITIONING_CHANNEL_MISSING: {channel}")
    row_ids = order[node_type]
    if str(intervention.node_id) not in row_ids:
        raise RuntimeError(
            f"PATIENT_CONDITIONING_NODE_NOT_IN_GRAPH: {node_type}:{intervention.node_id}"
        )
    row = row_ids.index(str(intervention.node_id))
    column = channels.index(channel) * 2
    updated = {
        key: np.array(value, dtype=np.float64, copy=True) for key, value in blocks.items()
    }
    block = updated[node_type]
    observed = float(block[row, column])
    if not bool(block[row, column + 1] > 0):
        raise RuntimeError("PATIENT_CONDITIONING_PARTICIPANT_UNMEASURED")
    value = observed if intervention.mode == "observed" else _require_finite(
        intervention.value, "intervention value"
    )
    block[row, column] = value
    return updated, {
        "node_type": node_type,
        "node_id": str(intervention.node_id),
        "channel": channel,
        "row": int(row),
        "column": int(column),
        "observed_value": observed,
        "intervention_value": value,
    }


def _snapshot(bundle) -> dict:
    saved = {"hetero": {}, "homogeneous": None}
    for node_type in bundle.hetero_data.node_types:
        saved["hetero"][str(node_type)] = bundle.hetero_data[str(node_type)].x.detach().clone()
    saved["homogeneous"] = bundle.homogeneous["node_x"].detach().clone()
    return saved


def _restore(bundle, saved: dict) -> None:
    for node_type, tensor in saved["hetero"].items():
        bundle.hetero_data[node_type].x = tensor
    bundle.homogeneous["node_x"] = saved["homogeneous"]


def _write_block(bundle, blocks: dict[str, np.ndarray]) -> None:
    """Write the omics columns of an already widened bundle in place."""

    import torch

    width = int(bundle.node_input_dim) - omics.DEGREE_CHANNELS
    for node_type in bundle.hetero_data.node_types:
        block = np.asarray(blocks[str(node_type)], dtype=np.float64)
        if block.shape[1] != width:
            raise RuntimeError(
                f"PATIENT_CONDITIONING_BLOCK_WIDTH: {node_type} {block.shape[1]} != {width}"
            )
        if not np.isfinite(block).all():
            raise RuntimeError(f"PATIENT_CONDITIONING_BLOCK_NOT_FINITE: {node_type}")
        current = bundle.hetero_data[str(node_type)].x
        features = current[:, : omics.DEGREE_CHANNELS].to(torch.float64).numpy()
        bundle.hetero_data[str(node_type)].x = torch.tensor(
            np.column_stack([features, block]), dtype=torch.float32
        )
    mapping = bundle.homogeneous["global_map"]
    written = np.zeros((len(bundle.nodes), int(bundle.node_input_dim)), dtype=np.float32)
    for node_type in bundle.hetero_data.node_types:
        stored = bundle.hetero_data[str(node_type)].x.numpy()
        for row, node_id in enumerate(getattr(bundle.hetero_data[str(node_type)], "canonical_ids")):
            written[mapping[(str(node_type), str(node_id))]] = stored[row]
    bundle.homogeneous["node_x"] = torch.tensor(written, dtype=torch.float32)


@contextmanager
def conditioned(bundle, blocks: dict[str, np.ndarray]):
    """Temporarily install one condition block, then restore the original."""

    saved = _snapshot(bundle)
    _write_block(bundle, blocks)
    try:
        yield bundle
    finally:
        _restore(bundle, saved)


def pair_batch(bundle, pairs: pd.DataFrame, features: list[str], kind: str):
    """Minimal forward batch: node indices plus the declared pair features."""

    import torch

    if not len(pairs):
        raise RuntimeError("PATIENT_CONDITIONING_PAIRS_EMPTY")
    required = ["lncrna_id", "cancer_id", *features]
    missing = [name for name in required if name not in pairs.columns]
    if missing:
        raise RuntimeError(f"PATIENT_CONDITIONING_PAIR_COLUMNS_MISSING: {missing}")
    target_column = "pathway_id" if "pathway_id" in pairs.columns else "pathway_family_id"
    if target_column not in pairs.columns:
        raise RuntimeError("PATIENT_CONDITIONING_PAIR_TARGET_MISSING")
    target_type = "pathway" if target_column == "pathway_id" else "pathway_family"
    valid = (
        pairs.lncrna_id.astype(str).isin(bundle.node_maps.get("lncRNA", {}))
        & pairs[target_column].astype(str).isin(bundle.node_maps.get(target_type, {}))
        & pairs.cancer_id.astype(str).isin(bundle.node_maps.get("cancer", {}))
    )
    if not bool(valid.all()):
        raise RuntimeError("PATIENT_CONDITIONING_PAIR_NODE_MISSING")
    frame = pairs.reset_index(drop=True)
    values = frame[features].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("PATIENT_CONDITIONING_PAIR_FEATURE_INVALID")
    if kind == "rgcn":
        mapping = bundle.homogeneous["global_map"]
        l = torch.tensor(
            [mapping[("lncRNA", str(x))] for x in frame.lncrna_id], dtype=torch.long
        )
        p = torch.tensor(
            [mapping[(target_type, str(x))] for x in frame[target_column]], dtype=torch.long
        )
        c = torch.tensor(
            [mapping[("cancer", str(x))] for x in frame.cancer_id], dtype=torch.long
        )
    else:
        l = torch.tensor(
            [bundle.node_maps["lncRNA"][str(x)] for x in frame.lncrna_id], dtype=torch.long
        )
        p = torch.tensor(
            [bundle.node_maps[target_type][str(x)] for x in frame[target_column]], dtype=torch.long
        )
        c = torch.tensor(
            [bundle.node_maps["cancer"][str(x)] for x in frame.cancer_id], dtype=torch.long
        )
    return frame, {
        "l": l,
        "p": p,
        "c": c,
        "x": torch.tensor(values, dtype=torch.float32),
    }


def score_membership(model, graph, batch) -> np.ndarray:
    """Frozen forward pass; no gradient, no optimizer, eval mode."""

    import torch

    model.eval()
    with torch.no_grad():
        membership, _ = model.forward(graph, batch)
    return membership.detach().cpu().numpy().astype(float)


def _node_position(bundle, node_type: str, node_id: str) -> int:
    mapping = bundle.node_maps.get(str(node_type), {})
    if str(node_id) not in mapping:
        raise RuntimeError(
            f"PATIENT_CONDITIONING_NODE_NOT_IN_GRAPH: {node_type}:{node_id}"
        )
    return int(mapping[str(node_id)])


def remove_incident_edges(graph, bundle, node_type: str, node_id: str, *, direction: str = "out"):
    """Return a graph copy with one node's incident edges removed.

    ``direction = "out"`` removes the edges whose source is the node, which is
    the whole channel through which a message-passing model can let that node
    influence anybody else.
    """

    position = _node_position(bundle, node_type, node_id)
    copy_graph = copy.deepcopy(graph)
    if not hasattr(copy_graph, "edge_types"):
        raise RuntimeError("PATIENT_CONDITIONING_GRAPH_KIND_UNSUPPORTED")
    removed = 0
    for edge_type in copy_graph.edge_types:
        # Node indices are per node type, so an edge type that does not start at
        # the declared node type cannot carry a message from it.
        if direction == "out" and edge_type[0] != str(node_type):
            continue
        if direction == "in" and edge_type[2] != str(node_type):
            continue
        store = copy_graph[edge_type]
        index = store.edge_index
        if index.numel() == 0:
            continue
        mask = index[0] != position if direction == "out" else index[1] != position
        removed += int((~mask).sum())
        store.edge_index = index[:, mask]
        weight = getattr(store, "edge_weight", None)
        if weight is not None:
            store.edge_weight = weight[mask]
    return copy_graph, removed


def rewire_outgoing(graph, bundle, node_type: str, node_id: str, *, seed: int = 0):
    """Redirect one node's outgoing edges to other nodes of the same target type.

    This is a control, not a model change: the number, type and weight of the
    rewired edges stay identical, so any remaining effect cannot be attributed
    to graph density or edge count.
    """

    import torch

    position = _node_position(bundle, node_type, node_id)
    copy_graph = copy.deepcopy(graph)
    if not hasattr(copy_graph, "edge_types"):
        raise RuntimeError("PATIENT_CONDITIONING_GRAPH_KIND_UNSUPPORTED")
    rng = np.random.default_rng(seed)
    rewired = 0
    for edge_type in copy_graph.edge_types:
        if edge_type[0] != str(node_type):
            continue
        store = copy_graph[edge_type]
        index = store.edge_index
        if index.numel() == 0:
            continue
        mask = index[0] == position
        if not bool(mask.any()):
            continue
        target_type = edge_type[2]
        count = int(copy_graph[target_type].num_nodes)
        if count < 2:
            raise RuntimeError(f"PATIENT_CONDITIONING_REWIRE_UNAVAILABLE: {target_type}")
        targets = index[1, mask].numpy()
        replacement = []
        for original in targets:
            for _ in range(64):
                candidate = int(rng.integers(0, count))
                if candidate != int(original):
                    replacement.append(candidate)
                    break
            else:
                raise RuntimeError("PATIENT_CONDITIONING_REWIRE_FAILED")
        updated = index.clone()
        updated[1, mask] = torch.tensor(replacement, dtype=updated.dtype)
        store.edge_index = updated
        rewired += len(replacement)
    return copy_graph, rewired


def _delta_table(
    frame: pd.DataFrame, observed: np.ndarray, intervened: np.ndarray
) -> pd.DataFrame:
    table = frame.copy()
    table["membership_observed"] = observed
    table["membership_intervened"] = intervened
    table["delta"] = intervened - observed
    table["absolute_delta"] = np.abs(table.delta.to_numpy(float))
    table["sign_flip"] = (
        (table.membership_observed > 0) != (table.membership_intervened > 0)
    ) & (table.absolute_delta > EFFECT_TOLERANCE)
    return table


def patient_relation_shift(
    model,
    bundle,
    *,
    evidence: omics.OmicsEvidence,
    intervention: Intervention,
    pairs: pd.DataFrame,
    features: list[str],
    kind: str = "cc_hhgt",
    blocks: dict[str, np.ndarray] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """C19: per-candidate membership shift under one patient intervention."""

    record = validate_intervention(intervention, evidence, bundle)
    observed_blocks_value = observed_blocks(bundle, evidence) if blocks is None else blocks
    intervened, writeback = intervened_blocks(
        bundle, observed_blocks_value, evidence, intervention
    )
    frame, batch = pair_batch(bundle, pairs, features, kind)
    from ..gnn import move_graph

    with conditioned(bundle, observed_blocks_value):
        graph = move_graph(bundle, "cpu", kind)
        baseline = score_membership(model, graph, batch)
    with conditioned(bundle, intervened):
        graph = move_graph(bundle, "cpu", kind)
        shifted = score_membership(model, graph, batch)
    table = _delta_table(frame, baseline, shifted)
    record.update(
        writeback=writeback,
        pair_count=int(len(table)),
        changed_pair_count=int((table.absolute_delta > EFFECT_TOLERANCE).sum()),
        sign_flip_count=int(table.sign_flip.sum()),
        max_absolute_delta=float(table.absolute_delta.max()) if len(table) else 0.0,
        pair_features=list(features),
        rank=rank_summary(table),
        kind=str(kind),
        node_input_dim=int(bundle.node_input_dim),
        intervention_identifier=intervention.identifier(),
    )
    return table, record


def _rank_within_cancer(table: pd.DataFrame, column: str) -> pd.Series:
    return (
        table.groupby("cancer_id", sort=False)[column]
        .rank(ascending=False, method="average")
        .astype(float)
    )


def rank_summary(table: pd.DataFrame, *, top_k: int = 10) -> dict:
    """Rank displacement of each pair inside its cancer, observed vs intervened."""

    frame = table.copy()
    frame["rank_observed"] = _rank_within_cancer(frame, "membership_observed")
    frame["rank_intervened"] = _rank_within_cancer(frame, "membership_intervened")
    frame["rank_shift"] = frame.rank_intervened - frame.rank_observed
    per_cancer = {}
    for cancer, group in frame.groupby("cancer_id", sort=False):
        before = set(group.nlargest(min(top_k, len(group)), "membership_observed").index)
        after = set(group.nlargest(min(top_k, len(group)), "membership_intervened").index)
        per_cancer[str(cancer)] = {
            "pairs": int(len(group)),
            "max_absolute_rank_shift": float(group.rank_shift.abs().max()),
            "top_k": int(min(top_k, len(group))),
            "top_k_overlap": int(len(before & after)),
            "top_k_jaccard": float(len(before & after) / max(1, len(before | after))),
        }
    return {
        "per_cancer": per_cancer,
        "rank_table": frame[
            [
                "cancer_id",
                "lncrna_id",
                "rank_observed",
                "rank_intervened",
                "rank_shift",
                "delta",
            ]
        ].to_dict(orient="records"),
    }


def _effect_under_structure(
    model,
    bundle,
    observed_blocks_value: dict[str, np.ndarray],
    intervened_blocks_value: dict[str, np.ndarray],
    batch,
    kind: str,
    mutate=None,
) -> np.ndarray:
    """Intervention effect recomputed inside one graph structure.

    Comparing an intervened graph against an observed graph of a *different*
    structure would mix the intervention with the structural edit.  Every
    control therefore repeats the whole contrast inside its own structure.
    """

    from ..gnn import move_graph

    scores = []
    for blocks in (observed_blocks_value, intervened_blocks_value):
        with conditioned(bundle, blocks):
            graph = copy.deepcopy(move_graph(bundle, "cpu", kind))
        if mutate is not None:
            graph = mutate(graph)
        scores.append(score_membership(model, graph, batch))
    return scores[1] - scores[0]


def interaction_attribution(
    model,
    bundle,
    *,
    evidence: omics.OmicsEvidence,
    intervention: Intervention,
    pairs: pd.DataFrame,
    features: list[str],
    path_target: str,
    path_target_pathway: str | None = None,
    placebo_node: str | None = None,
    kind: str = "cc_hhgt",
    rewire_seed: int = 0,
    attribution_min_share: float = ATTRIBUTION_MIN_SHARE,
) -> dict:
    """C20: attribute the pair effect to P's edges and falsify the reading.

    ``path_target`` is the lncRNA whose relation to the pathway is being
    explained.  The attribution share is the fraction of the total effect that
    disappears when P's edges into that lncRNA are removed, with the contrast
    recomputed inside the edited graph.  The two controls are the only parts
    that can falsify an interaction reading.
    """

    record = validate_intervention(intervention, evidence, bundle)
    blocks = observed_blocks(bundle, evidence)
    intervened, writeback = intervened_blocks(bundle, blocks, evidence, intervention)
    frame, batch = pair_batch(bundle, pairs, features, kind)
    target_column = "pathway_id" if "pathway_id" in frame.columns else "pathway_family_id"
    target_mask = frame.lncrna_id.astype(str).eq(str(path_target))
    if path_target_pathway is not None:
        target_mask = target_mask & frame[target_column].astype(str).eq(
            str(path_target_pathway)
        )
    target_rows = np.flatnonzero(target_mask.to_numpy(bool))
    if not len(target_rows):
        raise RuntimeError("PATIENT_CONDITIONING_PATH_TARGET_NOT_IN_PAIRS")
    counters: dict[str, int] = {}

    def removed(graph):
        result, count = remove_incident_edges(
            graph, bundle, intervention.node_type, intervention.node_id, direction="out"
        )
        counters["removed"] = int(count)
        return result

    def rewired(graph):
        result, count = rewire_outgoing(
            graph,
            bundle,
            intervention.node_type,
            intervention.node_id,
            seed=rewire_seed,
        )
        counters["rewired"] = int(count)
        return result

    total = _effect_under_structure(model, bundle, blocks, intervened, batch, kind)
    removed_effect = _effect_under_structure(
        model, bundle, blocks, intervened, batch, kind, mutate=removed
    )
    rewired_effect = _effect_under_structure(
        model, bundle, blocks, intervened, batch, kind, mutate=rewired
    )
    total_abs = float(np.max(np.abs(total[target_rows])))
    removed_abs = float(np.max(np.abs(removed_effect[target_rows])))
    rewired_abs = float(np.max(np.abs(rewired_effect[target_rows])))

    placebo_record = None
    placebo_abs = None
    if placebo_node is not None:
        if str(placebo_node) not in bundle.node_maps.get("gene", {}):
            raise RuntimeError(
                f"PATIENT_CONDITIONING_NODE_NOT_IN_GRAPH: gene:{placebo_node}"
            )
        placebo_intervention = Intervention(
            cancer_id=intervention.cancer_id,
            subject_id=intervention.subject_id,
            node_type="gene",
            node_id=str(placebo_node),
            channel=intervention.channel,
            mode=intervention.mode,
            value=intervention.value,
        )
        validate_intervention(placebo_intervention, evidence, bundle)
        placebo_blocks, placebo_writeback = intervened_blocks(
            bundle, blocks, evidence, placebo_intervention
        )
        placebo_effect = _effect_under_structure(
            model, bundle, blocks, placebo_blocks, batch, kind
        )
        placebo_abs = float(np.max(np.abs(placebo_effect[target_rows])))
        placebo_record = {
            "node_id": str(placebo_node),
            "writeback": placebo_writeback,
            "target_max_absolute_delta": placebo_abs,
        }

    attributed = max(0.0, total_abs - removed_abs)
    share = float(attributed / total_abs) if total_abs > EFFECT_TOLERANCE else 0.0
    scale = max(EFFECT_TOLERANCE, 0.1 * total_abs)
    rewired_specific = rewired_abs <= scale
    placebo_clean = placebo_abs is None or placebo_abs <= scale
    if total_abs <= EFFECT_TOLERANCE:
        verdict = "NO_EFFECT"
    elif share < attribution_min_share:
        verdict = "VALUE_ONLY_EFFECT"
    elif not (rewired_specific and placebo_clean):
        verdict = "NOT_INTERACTION_SPECIFIC"
    else:
        verdict = "INTERACTION_ATTRIBUTED_UNDER_FROZEN_MODEL"
    if verdict not in VERDICTS:  # pragma: no cover - defensive
        raise RuntimeError("PATIENT_CONDITIONING_VERDICT_UNKNOWN")
    record.update(
        writeback=writeback,
        path_target=str(path_target),
        path_target_pathway=(
            None if path_target_pathway is None else str(path_target_pathway)
        ),
        target_pair_count=int(len(target_rows)),
        pair_count=int(len(frame)),
        total_effect=total_abs,
        path_removed_effect=removed_abs,
        attributed_effect=attributed,
        attribution_share=share,
        removed_edge_count=int(counters.get("removed", 0)),
        rewired_edge_count=int(counters.get("rewired", 0)),
        rewire_target_max_absolute_delta=rewired_abs,
        rewire_specific=bool(rewired_specific),
        rewired_effect_is_zero=bool(rewired_abs <= EFFECT_TOLERANCE),
        placebo=placebo_record,
        placebo_clean=bool(placebo_clean),
        verdict=verdict,
        verdict_scope="WITHIN_FROZEN_MODEL_ONLY",
        effect_definitions={
            "total": "intervention effect in the unmodified graph",
            "path_removed": "intervention effect recomputed with P's outgoing edges removed",
            "rewired": "intervention effect recomputed with P's outgoing edges rewired to random partners",
            "placebo": "intervention effect of a matched node recomputed in the unmodified graph",
        },
        biological_mediation_proven=False,
        requires_orthogonal_validation=True,
        falsification_controls=["rewired_partner", "matched_degree_placebo_node"],
        total_effect_is_not_a_proof=True,
    )
    return record


__all__ = [
    "ATTRIBUTION_MIN_SHARE",
    "EFFECT_TOLERANCE",
    "FORMAT",
    "Intervention",
    "MODES",
    "SCOPE",
    "VERDICTS",
    "conditioned",
    "interaction_attribution",
    "intervened_blocks",
    "observed_blocks",
    "pair_batch",
    "patient_relation_shift",
    "rank_summary",
    "remove_incident_edges",
    "rewire_outgoing",
    "score_membership",
    "validate_intervention",
]
