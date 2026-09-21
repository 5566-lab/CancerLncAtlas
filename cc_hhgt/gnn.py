from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import (
    LOGGER,
    available_device,
    file_sha256,
    read_table,
    write_json,
    write_table,
)
from .metrics import proxy_binary_metrics
from .training_data import split_for_fold
from .graph_contract import (
    DeploymentContract,
    FoldGraphMode,
    build_fold_graph,
)
from .relation_sampling import (
    RuntimeSamplingPolicy,
    runtime_chunk_positions as build_runtime_chunk_positions,
    schedule_runtime_edges,
)


def require_torch_geometric():
    try:
        import torch
        import torch_geometric
    except ImportError as exc:
        raise RuntimeError("PyTorch and torch-geometric are required. Install requirements/requirements-gnn.txt") from exc
    return torch


def amp_autocast(torch, device: str, enabled: bool):
    """Return a version-compatible autocast context for CUDA execution."""
    use_amp = bool(enabled and torch.device(device).type == "cuda")
    try:
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp)
    except AttributeError:  # PyTorch < 2.0 compatibility
        return torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_amp)


def amp_grad_scaler(torch, enabled: bool):
    """Create a GradScaler across old and new PyTorch AMP APIs."""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


@dataclass
class GraphBundle:
    nodes: pd.DataFrame
    edges: pd.DataFrame
    node_maps: dict[str, dict[str, int]]
    hetero_data: Any
    homogeneous: dict[str, Any]
    active_edges: pd.DataFrame | None = None
    runtime_schedule: pd.DataFrame | None = None
    relation_schema_edges: pd.DataFrame | None = None
    runtime_num_chunks: int = 1
    active_runtime_chunk: int | None = None
    graph_contract: str = DeploymentContract.STRICT.value
    node_degrees: dict[str, np.ndarray] | None = None
    runtime_chunk_positions: dict[int, np.ndarray] | None = None
    cancer_edge_positions: dict[str, np.ndarray] | None = None
    localization_positions: np.ndarray | None = None
    episode_cache: dict[str, dict[str, Any]] = field(default_factory=dict)


def fold_node_degrees(node_group: pd.DataFrame, edges: pd.DataFrame, node_type: str) -> np.ndarray:
    """Build degree features from the already filtered/sampled fold graph.

    Reusing degrees stored in the global graph_node table leaks aggregate
    information from test/validation context edges into LOCO folds.
    """
    source = edges.loc[edges.source_type.astype(str) == node_type]
    target = edges.loc[edges.target_type.astype(str) == node_type]
    out_degree = source.groupby(source.source_canonical_id.astype(str), observed=True).size()
    in_degree = target.groupby(target.target_canonical_id.astype(str), observed=True).size()
    ids = node_group.canonical_id.astype(str)
    return (
        ids.map(out_degree).fillna(0).to_numpy(float)
        + ids.map(in_degree).fillna(0).to_numpy(float)
    )


def fold_node_features(node_group: pd.DataFrame, edges: pd.DataFrame, node_type: str) -> np.ndarray:
    degree = fold_node_degrees(node_group, edges, node_type)
    return np.column_stack(
        [np.log1p(degree), np.ones(len(node_group), dtype=float)]
    ).astype(np.float32)


def fold_node_degree_map(nodes: pd.DataFrame, edges: pd.DataFrame) -> dict[str, np.ndarray]:
    """Compute all node degrees with two global groupbys, not one scan/type."""

    source = edges.groupby(
        [edges.source_type.astype(str), edges.source_canonical_id.astype(str)],
        observed=True,
        sort=False,
    ).size()
    target = edges.groupby(
        [edges.target_type.astype(str), edges.target_canonical_id.astype(str)],
        observed=True,
        sort=False,
    ).size()
    output: dict[str, np.ndarray] = {}
    for node_type, raw_group in nodes.groupby("node_type", sort=False):
        group = raw_group.sort_values("node_index_within_type")
        ids = group.canonical_id.astype(str)
        try:
            out_degree = source.xs(str(node_type), level=0)
        except KeyError:
            out_degree = pd.Series(dtype="int64")
        try:
            in_degree = target.xs(str(node_type), level=0)
        except KeyError:
            in_degree = pd.Series(dtype="int64")
        output[str(node_type)] = (
            ids.map(out_degree).fillna(0).to_numpy(np.float64)
            + ids.map(in_degree).fillna(0).to_numpy(np.float64)
        )
    return output


def _subtract_edge_degrees(
    base: dict[str, np.ndarray],
    removed: pd.DataFrame,
    node_maps: dict[str, dict[str, int]],
) -> dict[str, np.ndarray]:
    """Exact degree update for a pseudoheldout mask without rescanning the graph."""

    output = {node_type: values.copy() for node_type, values in base.items()}
    for type_column, id_column in (
        ("source_type", "source_canonical_id"),
        ("target_type", "target_canonical_id"),
    ):
        for node_type, group in removed.groupby(type_column, observed=True, sort=False):
            node_type = str(node_type)
            mapping = node_maps[node_type]
            positions = group[id_column].astype(str).map(mapping)
            if positions.isna().any():
                raise RuntimeError(f"Removed edge references unknown {node_type} nodes")
            decrement = np.bincount(
                positions.to_numpy(np.int64), minlength=len(output[node_type])
            )
            output[node_type] -= decrement
    for node_type, values in output.items():
        if (values < 0).any():
            raise RuntimeError(f"Pseudoheldout degree update became negative for {node_type}")
    return output


def filter_fold_edges(
    edges: pd.DataFrame,
    *,
    heldout_cancers: set[str] | None = None,
    reference_only: set[str] | None = None,
    contract: DeploymentContract | str = DeploymentContract.STRICT,
    mode: FoldGraphMode | str = FoldGraphMode.REAL_TEST,
    compute_fingerprint: bool = True,
) -> pd.DataFrame:
    """Compatibility wrapper around the registered V3.1 fold builder."""

    heldout = set(map(str, heldout_cancers or set()))
    reference = set(map(str, reference_only or set()))
    if not heldout:
        # Training-free graph utilities may still ask for reference-only
        # removal.  Give the shared builder a sentinel that cannot occur in the
        # graph, then verify that it did not affect the result.
        heldout = {"__NO_HELDOUT_CANCER__"}
    result = build_fold_graph(
        edges,
        heldout_cancers=heldout,
        mode=mode,
        contract=contract,
        reference_only=reference,
        compute_fingerprint=compute_fingerprint,
    )
    return result.edges


def localize_cancer_state_edges(edges: pd.DataFrame) -> pd.DataFrame:
    """Derive cancer-state relation signs from the already filtered fold."""
    out = edges.copy()
    if "requires_fold_localization" not in out:
        return out
    mask = out.requires_fold_localization.eq(True)
    if not mask.any():
        return out
    raw = pd.to_numeric(out.loc[mask, "raw_effect"], errors="coerce")
    if raw.isna().any():
        raise RuntimeError("Fold-local cancer-state normalization received missing raw effects")
    state_ids = out.loc[mask, "target_canonical_id"].astype(str)
    state_counts = state_ids.value_counts()
    if (state_counts < 2).any():
        raise RuntimeError(
            "Fold-local cancer-state normalization needs at least two training cancers per state: "
            f"{state_counts.loc[state_counts < 2].to_dict()}"
        )
    means = raw.groupby(state_ids, observed=True).transform("mean")
    sds = raw.groupby(state_ids, observed=True).transform("std").replace(0, 1).fillna(1)
    standardized = (raw - means) / sds
    out.loc[mask, "weight"] = standardized.abs().to_numpy(float)
    out.loc[mask, "relation_type"] = np.where(
        standardized.ge(0), "state_enriched", "state_depleted"
    )
    out.loc[mask, "fold_local_mean"] = means.to_numpy(float)
    out.loc[mask, "fold_local_sd"] = sds.to_numpy(float)
    out.loc[mask, "fold_local_standardized_effect"] = standardized.to_numpy(float)
    out.loc[mask, "fold_localized"] = True
    if out.loc[mask, "relation_type"].eq("state_profile_requires_fold_localization").any():
        raise RuntimeError("Cancer-state placeholder relation survived fold localization")
    return out


def _materialize_graph_bundle(
    nodes: pd.DataFrame,
    active_edges: pd.DataFrame,
    *,
    feature_edges: pd.DataFrame | None = None,
    canonical_edges: pd.DataFrame | None = None,
    runtime_schedule: pd.DataFrame | None = None,
    relation_schema_edges: pd.DataFrame | None = None,
    node_degree_override: dict[str, np.ndarray] | None = None,
    runtime_chunk_positions: dict[int, np.ndarray] | None = None,
    cancer_edge_positions: dict[str, np.ndarray] | None = None,
    localization_positions: np.ndarray | None = None,
    episode_cache: dict[str, dict[str, Any]] | None = None,
    active_runtime_chunk: int | None = None,
    graph_contract: DeploymentContract | str = DeploymentContract.STRICT,
) -> GraphBundle:
    """Build device tensors for one bounded runtime graph chunk."""

    torch = require_torch_geometric()
    from torch_geometric.data import HeteroData
    feature_edges = active_edges if feature_edges is None else feature_edges
    canonical_edges = active_edges if canonical_edges is None else canonical_edges
    node_maps: dict[str, dict[str, int]] = {}
    data = HeteroData()
    global_index = {node_id: i for i, node_id in enumerate(nodes.node_id.astype(str))}
    homogeneous_node_x = np.zeros((len(nodes), 2), dtype=np.float32)
    node_degrees = (
        fold_node_degree_map(nodes, feature_edges)
        if node_degree_override is None
        else node_degree_override
    )
    for node_type, group in nodes.groupby("node_type", sort=False):
        group = group.sort_values("node_index_within_type")
        ids = group.canonical_id.astype(str).tolist()
        node_maps[str(node_type)] = {x: i for i, x in enumerate(ids)}
        degree = np.asarray(node_degrees[str(node_type)], dtype=np.float64)
        if len(degree) != len(group):
            raise RuntimeError(f"Node degree width mismatch for {node_type}")
        features = np.column_stack(
            [np.log1p(degree), np.ones(len(group), dtype=float)]
        ).astype(np.float32)
        data[str(node_type)].x = torch.tensor(
            features,
            dtype=torch.float32,
        )
        data[str(node_type)].canonical_ids = ids
        positions = [global_index[str(node_id)] for node_id in group.node_id]
        homogeneous_node_x[np.asarray(positions, dtype=np.int64)] = features

    relation_names = []
    homo_edges = []
    relation_counter = 0
    relation_schema_edges = feature_edges if relation_schema_edges is None else relation_schema_edges
    relation_groups = {
        tuple(map(str, key)): group
        for key, group in active_edges.groupby(
            ["source_type", "relation_type", "target_type"], sort=False, observed=True
        )
    }
    relation_keys = sorted(
        {
            tuple(map(str, key))
            for key in relation_schema_edges[["source_type", "relation_type", "target_type"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        }
    )
    relation_contract_groups = {
        tuple(map(str, key)): group
        for key, group in relation_schema_edges.groupby(
            ["source_type", "relation_type", "target_type"], sort=False, observed=True
        )
    }
    for src_type, rel, dst_type in relation_keys:
        group = relation_groups.get((src_type, rel, dst_type), active_edges.iloc[0:0])
        contract_group = relation_contract_groups[(src_type, rel, dst_type)]
        explicit_same_relation = False
        if "symmetric_same_relation" in contract_group:
            flags = contract_group.symmetric_same_relation.dropna().astype(bool)
            explicit_same_relation = bool(len(flags)) and bool(flags.all())
        if "edge_role" in contract_group:
            roles = set(contract_group.edge_role.dropna().astype(str))
            explicit_same_relation = explicit_same_relation or roles == {"static_symmetric_ppi"}
        same_relation_symmetric = (
            explicit_same_relation
            and src_type == dst_type
            and rel == "physical_interaction"
        )
        forward_relation_id = relation_counter; relation_counter += 1
        reverse_relation_id = None
        if not same_relation_symmetric:
            reverse_relation_id = relation_counter; relation_counter += 1
        src_map = node_maps[str(src_type)]
        dst_map = node_maps[str(dst_type)]
        edge_type = (str(src_type), str(rel), str(dst_type))
        reverse_type = (str(dst_type), f"rev_{rel}", str(src_type))
        if group.empty:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_weight_tensor = torch.empty((0,), dtype=torch.float32)
            data[edge_type].edge_index = edge_index
            data[edge_type].edge_weight = edge_weight_tensor
            relation_names.append(edge_type)
            empty = np.asarray([], dtype=np.int64)
            empty_weight = np.asarray([], dtype=np.float32)
            homo_edges.append((forward_relation_id, empty, empty, empty_weight))
            if not same_relation_symmetric:
                data[reverse_type].edge_index = edge_index
                data[reverse_type].edge_weight = edge_weight_tensor
                relation_names.append(reverse_type)
                homo_edges.append((reverse_relation_id, empty, empty, empty_weight))
            continue
        src = [src_map.get(str(x), -1) for x in group.source_canonical_id]
        dst = [dst_map.get(str(x), -1) for x in group.target_canonical_id]
        valid = np.asarray([(a >= 0 and b >= 0) for a, b in zip(src, dst)])
        if not valid.any():
            continue
        edge_index = torch.tensor(np.vstack([np.asarray(src)[valid], np.asarray(dst)[valid]]), dtype=torch.long)
        selected = group.loc[valid]
        edge_weight = pd.to_numeric(selected["weight"], errors="coerce").to_numpy(np.float64)
        if "relation_polarity" in selected:
            polarity = pd.to_numeric(
                selected["relation_polarity"], errors="coerce"
            ).to_numpy(np.float64)
            if not np.isfinite(polarity).all() or not np.isin(polarity, (-1.0, 1.0)).all():
                raise RuntimeError(
                    f"Typed relation {edge_type} has invalid relation_polarity"
                )
            edge_weight = edge_weight * polarity
        if not np.isfinite(edge_weight).all() or np.equal(edge_weight, 0).any():
            raise RuntimeError(f"Typed relation {edge_type} has non-finite/zero runtime weights")
        edge_weight = edge_weight.astype(np.float32)
        if same_relation_symmetric:
            source_values = selected.source_canonical_id.astype(str).to_numpy()
            target_values = selected.target_canonical_id.astype(str).to_numpy()
            if np.equal(source_values, target_values).any():
                raise RuntimeError("Same-relation PPI contains a self-loop")
            canonical = pd.DataFrame(
                {
                    "left": np.minimum(source_values, target_values),
                    "right": np.maximum(source_values, target_values),
                    "source": source_values,
                    "target": target_values,
                    "weight": edge_weight,
                }
            )
            for (left, right), pair in canonical.groupby(["left", "right"], sort=False):
                directions = set(zip(pair.source, pair.target, strict=True))
                if len(pair) != 2 or directions != {(left, right), (right, left)}:
                    raise RuntimeError(
                        "Same-relation PPI requires exactly two reciprocal input rows"
                    )
                if not np.isclose(
                    pair.weight.iloc[0], pair.weight.iloc[1], rtol=0.0, atol=0.0
                ):
                    raise RuntimeError("Same-relation PPI reciprocal weights differ")
        data[edge_type].edge_index = edge_index
        data[edge_type].edge_weight = torch.tensor(edge_weight, dtype=torch.float32)
        relation_names.append(edge_type)
        hg_src = [global_index[x] for x in selected["source_node_id"].astype(str)]
        hg_dst = [global_index[x] for x in selected["target_node_id"].astype(str)]
        homo_edges.append((forward_relation_id, np.asarray(hg_src), np.asarray(hg_dst), edge_weight))
        if not same_relation_symmetric:
            # Explicit reverse relation improves message flow and keeps
            # relation semantics visible.  PPI is the exception: its source
            # already contains both messages under one shared relation.
            data[reverse_type].edge_index = edge_index.flip(0)
            data[reverse_type].edge_weight = data[edge_type].edge_weight
            relation_names.append(reverse_type)
            homo_edges.append((reverse_relation_id, np.asarray(hg_dst), np.asarray(hg_src), edge_weight))

    h_src, h_dst, h_rel, h_weight = [], [], [], []
    for rel_id, src, dst, weight in homo_edges:
        h_src.append(src); h_dst.append(dst); h_rel.append(np.full(len(src), rel_id, dtype=np.int64)); h_weight.append(weight)
    homogeneous = {
        "edge_index": torch.tensor(np.vstack([np.concatenate(h_src), np.concatenate(h_dst)]), dtype=torch.long),
        "edge_type": torch.tensor(np.concatenate(h_rel), dtype=torch.long),
        "edge_weight": torch.tensor(np.concatenate(h_weight), dtype=torch.float32),
        "node_x": torch.tensor(homogeneous_node_x, dtype=torch.float32),
        "num_nodes": len(nodes),
        "num_relations": int(max(np.concatenate(h_rel)) + 1) if h_rel else 1,
        "global_map": {(r.node_type, str(r.canonical_id)): i for i, r in enumerate(nodes.itertuples(index=False))},
    }
    if runtime_schedule is not None and len(runtime_schedule):
        if runtime_chunk_positions is None:
            runtime_chunk_positions = build_runtime_chunk_positions(runtime_schedule)
        if cancer_edge_positions is None:
            cancer_text = runtime_schedule.cancer_id.astype("string").fillna("")
            cancer_edge_positions = {
                str(key): np.asarray(value, dtype=np.int64)
                for key, value in cancer_text.groupby(cancer_text, sort=False).indices.items()
                if str(key)
            }
        if localization_positions is None:
            flag = runtime_schedule.get(
                "requires_fold_localization",
                pd.Series(False, index=runtime_schedule.index),
            ).fillna(False).astype(bool)
            localization_positions = np.flatnonzero(flag.to_numpy())
        runtime_num_chunks = len(runtime_chunk_positions)
    else:
        runtime_num_chunks = 1
    return GraphBundle(
        nodes=nodes,
        edges=canonical_edges,
        node_maps=node_maps,
        hetero_data=data,
        homogeneous=homogeneous,
        active_edges=active_edges,
        runtime_schedule=runtime_schedule,
        relation_schema_edges=relation_schema_edges,
        runtime_num_chunks=runtime_num_chunks,
        active_runtime_chunk=active_runtime_chunk,
        graph_contract=DeploymentContract(graph_contract).value,
        node_degrees=node_degrees,
        runtime_chunk_positions=runtime_chunk_positions,
        cancer_edge_positions=cancer_edge_positions,
        localization_positions=localization_positions,
        episode_cache=episode_cache if episode_cache is not None else {},
    )


def load_graph_bundle(
    cfg: dict[str, Any],
    seed: int,
    excluded_cancers: set[str] | None = None,
    *,
    contract: DeploymentContract | str | None = None,
    mode: FoldGraphMode | str = FoldGraphMode.REAL_TEST,
) -> GraphBundle:
    nodes = read_table(cfg["_results"] / "tables" / "graph_node.parquet")
    edges = read_table(cfg["_results"] / "tables" / "graph_edge.parquet")
    excluded_cancers = excluded_cancers or set()
    reference_only = set(map(str, cfg.get("cancer_scope", {}).get("reference_only", [])))
    reference_only.update(map(str, cfg.get("analysis_cancers", {}).get("reference_only", [])))
    selected_contract = contract or cfg.get("graph_contract", {}).get(
        "primary", DeploymentContract.STRICT.value
    )
    edges = filter_fold_edges(
        edges,
        heldout_cancers=set(map(str, excluded_cancers)),
        reference_only=reference_only,
        contract=selected_contract,
        mode=mode,
        compute_fingerprint=False,
    )
    edges = localize_cancer_state_edges(edges)
    runtime_cfg = cfg.get("runtime_graph_sampling", {})
    runtime_sampling = bool(runtime_cfg.get("enabled", False))
    configured_cap = cfg["training"].get("max_edges_per_relation", 2_000_000)
    if runtime_sampling and configured_cap is not None:
        raise RuntimeError("V3.1 runtime sampling forbids a destructive max_edges_per_relation cap")
    if runtime_sampling:
        policy = RuntimeSamplingPolicy(
            edge_chunk_size=int(runtime_cfg.get("edge_chunk_size", 250_000)),
            source_coverage_min=float(runtime_cfg.get("source_coverage_min", 0.95)),
            weighted_signal_retention_min=float(
                runtime_cfg.get("weighted_signal_retention_min", 0.90)
            ),
            coverage_cycle=str(runtime_cfg.get("coverage_cycle", "all_runtime_chunks")),
            resident_backbone=bool(runtime_cfg.get("resident_backbone", False)),
            variable_relation_types=tuple(
                runtime_cfg.get(
                    "variable_relation_types",
                    ("coexpressed_positive", "coexpressed_negative"),
                )
            ),
        )
        scheduled = schedule_runtime_edges(edges, policy)
        scheduled_positions = build_runtime_chunk_positions(scheduled)
        active = scheduled.iloc[scheduled_positions[0]].copy()
        relation_schema = scheduled[
            ["source_type", "relation_type", "target_type"]
        ].drop_duplicates()
        localization_mask = scheduled.get(
            "requires_fold_localization",
            pd.Series(False, index=scheduled.index),
        ).fillna(False).astype(bool)
        if localization_mask.any():
            localized_schema = scheduled.loc[
                localization_mask, ["source_type", "target_type"]
            ].drop_duplicates()
            localized_schema = pd.concat(
                [
                    localized_schema.assign(relation_type=relation)
                    for relation in ("state_enriched", "state_depleted")
                ],
                ignore_index=True,
            )[["source_type", "relation_type", "target_type"]]
            relation_schema = pd.concat(
                [relation_schema, localized_schema], ignore_index=True
            ).drop_duplicates()
        return _materialize_graph_bundle(
            nodes,
            active,
            # Scheduling may repeat resident execution rows.  Node degree and
            # canonical graph identity are defined by the unique frozen graph,
            # never by the number of runtime chunks.
            feature_edges=edges,
            canonical_edges=edges,
            runtime_schedule=scheduled,
            relation_schema_edges=relation_schema,
            active_runtime_chunk=0,
            runtime_chunk_positions=scheduled_positions,
            graph_contract=selected_contract,
        )

    if configured_cap is not None:
        max_edges = int(configured_cap)
        sampled = []
        for _, group in edges.groupby(["source_type", "relation_type", "target_type"], observed=True):
            if len(group) > max_edges:
                w = group.weight.clip(lower=1e-6).to_numpy(dtype=float)
                w = w / w.sum()
                rng = np.random.default_rng(seed)
                idx = rng.choice(len(group), size=max_edges, replace=False, p=w, shuffle=False)
                group = group.iloc[np.sort(idx)]
            sampled.append(group)
        edges = pd.concat(sampled, ignore_index=True)
    return _materialize_graph_bundle(
        nodes,
        edges,
        feature_edges=edges,
        canonical_edges=edges,
        graph_contract=selected_contract,
    )


def runtime_bundle_for_step(
    bundle: GraphBundle,
    step: int,
    *,
    pseudoheldout_cancer: str | None = None,
    formal_heldout_cancers: set[str] | None = None,
    reference_only: set[str] | None = None,
) -> GraphBundle:
    """Materialize one relation-mixed runtime chunk for an optimizer step."""

    if bundle.runtime_schedule is None:
        return bundle
    chunk = int(step) % int(bundle.runtime_num_chunks)
    if bundle.runtime_chunk_positions is None or bundle.node_degrees is None:
        raise RuntimeError("Runtime bundle is missing precomputed chunk/degree indices")
    active = bundle.runtime_schedule.iloc[
        bundle.runtime_chunk_positions[chunk]
    ].copy()
    episode_node_degrees = bundle.node_degrees
    if pseudoheldout_cancer is not None:
        heldout = set(map(str, formal_heldout_cancers or set()))
        heldout.add(str(pseudoheldout_cancer))
        reference = set(map(str, reference_only or set()))
        cache_key = "|".join(
            [str(pseudoheldout_cancer), *sorted(heldout), bundle.graph_contract]
        )
        cached = bundle.episode_cache.get(cache_key)
        if cached is None:
            cancer_positions = (bundle.cancer_edge_positions or {}).get(
                str(pseudoheldout_cancer), np.asarray([], dtype=np.int64)
            )
            cancer_edges = bundle.runtime_schedule.iloc[cancer_positions].copy()
            cancer_edges["_runtime_position"] = cancer_positions
            kept_cancer_edges = build_fold_graph(
                cancer_edges,
                heldout_cancers=heldout,
                mode=FoldGraphMode.PSEUDOHELDOUT,
                contract=bundle.graph_contract,
                reference_only=reference,
                compute_fingerprint=False,
            ).edges
            kept_positions = set(
                pd.to_numeric(
                    kept_cancer_edges.get(
                        "_runtime_position", pd.Series(dtype="int64")
                    ),
                    errors="raise",
                ).astype(np.int64)
            )
            removed = cancer_edges.loc[
                ~cancer_edges._runtime_position.astype(np.int64).isin(kept_positions)
            ]
            episode_node_degrees = _subtract_edge_degrees(
                bundle.node_degrees, removed, bundle.node_maps
            )

            # Only the explicitly fold-local state rows require a statistic
            # recomputation.  This table is tiny relative to the canonical
            # graph, so the exact masking/localization code remains shared.
            localization_positions = (
                bundle.localization_positions
                if bundle.localization_positions is not None
                else np.asarray([], dtype=np.int64)
            )
            localization_edges = bundle.runtime_schedule.iloc[
                localization_positions
            ].copy()
            if len(localization_edges):
                localization_edges = build_fold_graph(
                    localization_edges,
                    heldout_cancers=heldout,
                    mode=FoldGraphMode.PSEUDOHELDOUT,
                    contract=bundle.graph_contract,
                    reference_only=reference,
                    compute_fingerprint=False,
                ).edges
                localization_edges = localize_cancer_state_edges(localization_edges)
                localized = localization_edges.set_index("edge_id")[[
                    "weight",
                    "relation_type",
                ]]
            else:
                localized = pd.DataFrame(columns=["weight", "relation_type"])
            cached = {
                "node_degrees": episode_node_degrees,
                "localized": localized,
                "removed_edge_count": int(len(removed)),
            }
            bundle.episode_cache[cache_key] = cached
        episode_node_degrees = cached["node_degrees"]
        # All non-c* rows were already filtered for the formal fold when the
        # base schedule was built.  Exercise the shared contract builder on
        # the only rows whose policy changes in this episode, then restore the
        # stable runtime order.  This is exactly equivalent to filtering the
        # entire 250k-row chunk and avoids two full DataFrame copies/epoch.
        active["_active_position"] = np.arange(len(active), dtype=np.int64)
        cstar = active.cancer_id.astype("string").eq(
            str(pseudoheldout_cancer)
        ).fillna(False)
        kept_cstar = build_fold_graph(
            active.loc[cstar],
            heldout_cancers=heldout,
            mode=FoldGraphMode.PSEUDOHELDOUT,
            contract=bundle.graph_contract,
            reference_only=reference,
            compute_fingerprint=False,
        ).edges
        active = pd.concat(
            [active.loc[~cstar], kept_cstar], ignore_index=True
        ).sort_values("_active_position", kind="stable").drop(
            columns="_active_position"
        ).reset_index(drop=True)
        localized = cached["localized"]
        if len(localized) and len(active):
            active_ids = active.edge_id.astype(str)
            update = active_ids.isin(localized.index.astype(str))
            if update.any():
                active.loc[update, "weight"] = active_ids.loc[update].map(
                    localized.weight
                ).to_numpy(float)
                active.loc[update, "relation_type"] = active_ids.loc[update].map(
                    localized.relation_type
                ).astype(str).to_numpy()
    return _materialize_graph_bundle(
        bundle.nodes,
        active,
        feature_edges=active,
        canonical_edges=active,
        runtime_schedule=bundle.runtime_schedule,
        # Keep the model metadata stable even when one fold has no active edge
        # for a relation.  This schema is used only to declare relation types;
        # it is never used for degree features or message passing.
        relation_schema_edges=(
            bundle.relation_schema_edges
            if bundle.relation_schema_edges is not None
            else bundle.runtime_schedule
        ),
        node_degree_override=episode_node_degrees,
        runtime_chunk_positions=bundle.runtime_chunk_positions,
        cancer_edge_positions=bundle.cancer_edge_positions,
        localization_positions=bundle.localization_positions,
        episode_cache=bundle.episode_cache,
        active_runtime_chunk=chunk,
        graph_contract=bundle.graph_contract,
    )


def candidate_tensors(frame: pd.DataFrame, bundle: GraphBundle, features: list[str], device: str, model_kind: str):
    import torch
    target_column = "pathway_id" if "pathway_id" in frame.columns else "pathway_family_id"
    target_node_type = "pathway" if target_column == "pathway_id" else "pathway_family"
    valid = frame.lncrna_id.astype(str).isin(bundle.node_maps.get("lncRNA", {})) & frame[target_column].astype(str).isin(bundle.node_maps.get(target_node_type, {})) & frame.cancer_id.astype(str).isin(bundle.node_maps.get("cancer", {}))
    frame = frame.loc[valid].reset_index(drop=True)
    if model_kind == "rgcn":
        gm = bundle.homogeneous["global_map"]
        l = torch.tensor([gm[("lncRNA", str(x))] for x in frame.lncrna_id], dtype=torch.long, device=device)
        p = torch.tensor([gm[(target_node_type, str(x))] for x in frame[target_column]], dtype=torch.long, device=device)
        c = torch.tensor([gm[("cancer", str(x))] for x in frame.cancer_id], dtype=torch.long, device=device)
    else:
        l = torch.tensor([bundle.node_maps["lncRNA"][str(x)] for x in frame.lncrna_id], dtype=torch.long, device=device)
        p = torch.tensor([bundle.node_maps[target_node_type][str(x)] for x in frame[target_column]], dtype=torch.long, device=device)
        c = torch.tensor([bundle.node_maps["cancer"][str(x)] for x in frame.cancer_id], dtype=torch.long, device=device)
    x = torch.tensor(frame[features].fillna(0.0).to_numpy(np.float32), dtype=torch.float32, device=device)
    y = torch.tensor(frame.proxy_label.to_numpy(np.float32), dtype=torch.float32, device=device)
    weak = torch.tensor((frame.label_class == "weak_positive").to_numpy(), dtype=torch.bool, device=device)
    direction = torch.tensor(frame.direction_label.fillna(-1).to_numpy(np.float32), dtype=torch.float32, device=device)
    batch = {"l": l, "p": p, "c": c, "x": x, "y": y, "weak": weak, "direction": direction}
    if "z_base" in frame:
        base = pd.to_numeric(frame.z_base, errors="coerce")
        if base.isna().any() or (~np.isfinite(base.to_numpy(float))).any():
            raise RuntimeError("Pathway residual base contains non-finite logits")
        batch["base_logit"] = torch.tensor(
            base.to_numpy(np.float32), dtype=torch.float32, device=device
        )
    return frame, batch


def nnpu_loss(logits, y, weak_mask, prior: float, unlabeled_weight: float, weak_weight: float):
    import torch
    import torch.nn.functional as F
    positive = y > 0.5
    unlabeled = ~positive
    if positive.any():
        positive_weight = torch.where(
            weak_mask[positive],
            torch.full_like(logits[positive], float(weak_weight)),
            torch.ones_like(logits[positive]),
        )
        positive_denominator = positive_weight.sum().clamp_min(1e-8)
        positive_loss = F.binary_cross_entropy_with_logits(
            logits[positive], torch.ones_like(logits[positive]), reduction="none"
        )
        positive_as_negative_loss = F.binary_cross_entropy_with_logits(
            logits[positive], torch.zeros_like(logits[positive]), reduction="none"
        )
        pos_risk = prior * (positive_loss * positive_weight).sum() / positive_denominator
        pos_as_negative = prior * (positive_as_negative_loss * positive_weight).sum() / positive_denominator
    else:
        pos_risk = logits.sum() * 0
        pos_as_negative = logits.sum() * 0
    if unlabeled.any():
        unlabeled_negative = F.binary_cross_entropy_with_logits(logits[unlabeled], torch.zeros_like(logits[unlabeled]))
    else:
        unlabeled_negative = logits.sum() * 0
    negative_risk = unlabeled_negative - pos_as_negative
    risk = pos_risk + unlabeled_weight * torch.clamp(negative_risk, min=0.0)
    return risk


def build_model(kind: str, bundle: GraphBundle, feature_dim: int, cfg: dict[str, Any]):
    torch = require_torch_geometric()
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import HGTConv, RGCNConv

    hidden = int(cfg["training"]["hidden_channels"])
    layers = int(cfg["training"]["num_layers"])
    heads = int(cfg["training"]["num_heads"])
    dropout = float(cfg["training"]["dropout"])
    residual_learning = bool(cfg.get("residual_learning", {}).get("enabled", False))
    from .pathway_target import pathway_target_node_type

    registered_pathway_target_node_type = pathway_target_node_type(cfg)

    class Decoder(nn.Module):
        def __init__(self, use_pair: bool):
            super().__init__()
            pair_hidden = int(cfg["training"]["pair_hidden_channels"])
            self.use_pair = use_pair
            self.pair = nn.Sequential(nn.Linear(feature_dim, pair_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(pair_hidden, hidden)) if use_pair else None
            input_dim = hidden * (4 if use_pair else 3)
            self.membership = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
            self.direction = nn.Sequential(nn.Linear(input_dim, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))
            if residual_learning:
                nn.init.zeros_(self.membership[-1].weight)
                nn.init.zeros_(self.membership[-1].bias)
        def forward(self, l, p, c, pair_x):
            pieces = [l, p, c]
            if self.use_pair:
                pieces.append(self.pair(pair_x))
            z = torch.cat(pieces, dim=-1)
            return self.membership(z).squeeze(-1), self.direction(z).squeeze(-1)

    def weighted_homogeneous_aggregate(h, graph):
        source, target = graph["edge_index"]
        weight = graph["edge_weight"].to(dtype=h.dtype)
        aggregate = torch.zeros_like(h)
        denominator = torch.zeros((h.shape[0], 1), dtype=h.dtype, device=h.device)
        aggregate.index_add_(0, target, h[source] * weight.unsqueeze(-1))
        denominator.index_add_(0, target, weight.abs().unsqueeze(-1))
        return aggregate / denominator.clamp_min(1e-8)

    if kind == "rgcn":
        class RGCNModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.pathway_target_node_type = registered_pathway_target_node_type
                # Shared deterministic feature encoder: no cancer/node ID has
                # an untrained free embedding in a held-out fold.
                self.input_projection = nn.Sequential(nn.Linear(2, hidden), nn.LayerNorm(hidden), nn.GELU())
                self.layers = nn.ModuleList([RGCNConv(hidden, hidden, bundle.homogeneous["num_relations"], num_bases=min(32, bundle.homogeneous["num_relations"])) for _ in range(layers)])
                self.weighted_layers = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(layers)])
                self.decoder = Decoder(use_pair=False)
            def encode(self, graph):
                h = self.input_projection(graph["node_x"])
                for conv, weighted in zip(self.layers, self.weighted_layers, strict=True):
                    topology = conv(h, graph["edge_index"], graph["edge_type"])
                    weight_message = weighted(weighted_homogeneous_aggregate(h, graph))
                    h = F.relu(topology + weight_message)
                    h = F.dropout(h, p=dropout, training=self.training)
                return h
            def forward(self, graph, batch):
                h = self.encode(graph)
                return self.decoder(h[batch["l"]], h[batch["p"]], h[batch["c"]], batch["x"])
        return RGCNModel()

    class HeteroModel(nn.Module):
        def __init__(self, use_pair: bool):
            super().__init__()
            self.pathway_target_node_type = registered_pathway_target_node_type
            self.proj = nn.ModuleDict({node_type: nn.Linear(2, hidden) for node_type in bundle.hetero_data.node_types})
            self.layers = nn.ModuleList([HGTConv(hidden, hidden, bundle.hetero_data.metadata(), heads=heads) for _ in range(layers)])
            self.weighted_projection = nn.ModuleList(
                [nn.ModuleDict({node_type: nn.Linear(hidden, hidden, bias=False) for node_type in bundle.hetero_data.node_types}) for _ in range(layers)]
            )
            self.decoder = Decoder(use_pair=use_pair)
        def weighted_aggregate(self, h, data, projection):
            aggregate = {node_type: torch.zeros_like(value) for node_type, value in h.items()}
            denominator = {
                node_type: torch.zeros((value.shape[0], 1), dtype=value.dtype, device=value.device)
                for node_type, value in h.items()
            }
            for edge_type in data.edge_types:
                source_type, _, target_type = edge_type
                edge_index = data[edge_type].edge_index
                weight = data[edge_type].edge_weight.to(dtype=h[source_type].dtype)
                source, target = edge_index
                aggregate[target_type].index_add_(
                    0, target, h[source_type][source] * weight.unsqueeze(-1)
                )
                denominator[target_type].index_add_(0, target, weight.abs().unsqueeze(-1))
            return {
                node_type: projection[node_type](aggregate[node_type] / denominator[node_type].clamp_min(1e-8))
                for node_type in h
            }
        def encode(self, data):
            h = {k: F.relu(self.proj[k](data[k].x)) for k in data.node_types}
            for conv, projection in zip(self.layers, self.weighted_projection, strict=True):
                updated = conv(h, data.edge_index_dict)
                weighted = self.weighted_aggregate(h, data, projection)
                h = {
                    k: F.dropout(F.relu(updated.get(k, h[k]) + weighted[k]), p=dropout, training=self.training)
                    for k in h
                }
            return h
        def forward(self, data, batch):
            h = self.encode(data)
            return self.decoder(h["lncRNA"][batch["l"]], h[self.pathway_target_node_type][batch["p"]], h["cancer"][batch["c"]], batch["x"])
    return HeteroModel(use_pair=(kind == "cc_hhgt"))


def move_graph(bundle: GraphBundle, device: str, kind: str):
    if kind == "rgcn":
        return {
            "edge_index": bundle.homogeneous["edge_index"].to(device),
            "edge_type": bundle.homogeneous["edge_type"].to(device),
            "edge_weight": bundle.homogeneous["edge_weight"].to(device),
            "node_x": bundle.homogeneous["node_x"].to(device),
        }
    return bundle.hetero_data.to(device)


def model_output_root(cfg: dict[str, Any]) -> Path:
    override = cfg.get("_model_output_root")
    return Path(override) if override else cfg["_results"] / "models"


def export_node_embeddings(
    cfg: dict[str, Any],
    model: Any,
    graph: Any,
    bundle: GraphBundle,
    kind: str,
    fold_row: pd.Series,
    model_dir: Path,
    *,
    encoded: Any | None = None,
    runtime_ensemble_chunks: int = 1,
) -> dict[str, Any]:
    """Export float16 node embeddings aligned to graph_node indices."""
    torch = require_torch_geometric()
    if encoded is None:
        model.eval()
        with torch.no_grad():
            encoded = model.encode(graph)
    embeddings: dict[str, Any] = {}
    if kind == "rgcn":
        positioned = bundle.nodes.assign(_global_position=np.arange(len(bundle.nodes)))
        for node_type, group in positioned.groupby("node_type", sort=False):
            group = group.sort_values("node_index_within_type")
            indices = torch.tensor(
                group["_global_position"].to_numpy(np.int64),
                dtype=torch.long,
                device=encoded.device,
            )
            embeddings[str(node_type)] = (
                encoded.index_select(0, indices).detach().cpu().to(torch.float16).contiguous()
            )
    else:
        embeddings = {
            str(node_type): tensor.detach().cpu().to(torch.float16).contiguous()
            for node_type, tensor in encoded.items()
        }
    output = model_dir / "node_embeddings.pt"
    torch.save(
        {
            "format_version": 1,
            "kind": kind,
            "analysis_version": cfg["analysis_version"],
            "fold": fold_row.to_dict(),
            "dtype": "float16",
            "runtime_ensemble_chunks": int(runtime_ensemble_chunks),
            "runtime_ensemble_method": "mean_encoded_representation",
            "index_policy": (
                "Each tensor is keyed by node_type and ordered by "
                "graph_node.node_index_within_type."
            ),
            "node_embeddings": embeddings,
        },
        output,
    )
    metadata = {
        "format_version": 1,
        "kind": kind,
        "fold_id": str(fold_row.fold_id),
        "split_seed": int(fold_row.split_seed),
        "dtype": "float16",
        "runtime_ensemble_chunks": int(runtime_ensemble_chunks),
        "runtime_ensemble_method": "mean_encoded_representation",
        "embedding_dimensions": {
            node_type: [int(value.shape[0]), int(value.shape[1])]
            for node_type, value in embeddings.items()
        },
        "index_policy": (
            "node_type tensor rows align to graph_node.parquet sorted by "
            "node_index_within_type"
        ),
        "sha256": file_sha256(output),
    }
    write_json(metadata, model_dir / "embedding_metadata.json")
    return metadata


def train_gnn_fold(cfg: dict[str, Any], fold_row: pd.Series, kind: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    torch = require_torch_geometric()
    import torch.nn.functional as F

    seed = int(fold_row.split_seed)
    torch.manual_seed(seed)
    device = available_device(cfg["training"].get("device", "auto"))
    if torch.device(device).type == "cuda":
        torch.cuda.manual_seed_all(seed)
    amp_enabled = bool(cfg["training"].get("mixed_precision", False) and torch.device(device).type == "cuda")
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    excluded_cancers = {str(fold_row.test_cancer), str(fold_row.validation_cancer)}
    bundle = load_graph_bundle(cfg, seed, excluded_cancers=excluded_cancers)
    graph = move_graph(bundle, device, kind)
    train, val, test = split_for_fold(cfg, fold_row)
    features = [x for x in cfg["training"]["feature_columns"] if x in train.columns]
    frames, tensors = {}, {}
    for split, frame in [("train", train), ("val", val), ("test", test)]:
        frames[split], tensors[split] = candidate_tensors(frame, bundle, features, device, kind)
        if len(frames[split]) == 0:
            raise RuntimeError(f"No graph-mappable pairs in {fold_row.fold_id}/{split}")
    model = build_model(kind, bundle, len(features), cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"])
    scaler = amp_grad_scaler(torch, amp_enabled)
    best_score = -np.inf
    best_epoch = 0
    history = []
    model_dir = model_output_root(cfg) / kind / fold_row.fold_id
    model_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        with amp_autocast(torch, device, amp_enabled):
            logits, direction_logits = model(graph, tensors["train"])
            loss = nnpu_loss(logits, tensors["train"]["y"], tensors["train"]["weak"], cfg["training"]["pu_prior"], cfg["training"]["unlabeled_weight"], cfg["training"]["weak_positive_weight"])
            known_dir = tensors["train"]["direction"] >= 0
            if known_dir.any():
                direction_loss = F.binary_cross_entropy_with_logits(direction_logits[known_dir], tensors["train"]["direction"][known_dir])
                loss = loss + cfg["training"]["direction_loss_weight"] * direction_loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        model.eval()
        with torch.no_grad():
            with amp_autocast(torch, device, amp_enabled):
                val_logits, _ = model(graph, tensors["val"])
            val_prob = torch.sigmoid(val_logits).float().cpu().numpy()
        metrics = proxy_binary_metrics(frames["val"], val_prob, cfg["calibration"]["bins"])
        score = metrics["auprc"] if np.isfinite(metrics["auprc"]) else -float(loss.detach().cpu())
        history.append({"epoch": epoch, "loss": float(loss.detach().cpu()), **metrics})
        if score > best_score:
            best_score = score; best_epoch = epoch
            torch.save({"model_state": model.state_dict(), "kind": kind, "features": features, "fold": fold_row.to_dict(), "analysis_version": cfg["analysis_version"]}, model_dir / "best.pt")
        elif epoch - best_epoch >= int(cfg["training"]["patience"]):
            break
    checkpoint = torch.load(model_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"]); model.eval()
    pred_rows, metric_rows = [], []
    with torch.no_grad():
        for split in ["train", "val", "test"]:
            with amp_autocast(torch, device, amp_enabled):
                logits, dlogits = model(graph, tensors[split])
            probability = torch.sigmoid(logits).float().cpu().numpy()
            direction_probability = torch.sigmoid(dlogits).float().cpu().numpy()
            frame = frames[split]
            part = frame[[c for c in ["candidate_id", "cancer_id", "lncrna_id", "pathway_id", "pathway_family_id", "label_class", "proxy_label", "direction", "observed_evidence_score"] if c in frame]].copy()
            part["split"] = split; part["model_name"] = kind; part["model_analysis_version"] = cfg["analysis_version"]; part["raw_logit"] = logits.float().cpu().numpy(); part["raw_probability"] = probability; part["direction_probability"] = direction_probability; part["fold_id"] = fold_row.fold_id
            pred_rows.append(part)
            metric_rows.append({"fold_id": fold_row.fold_id, "model_name": kind, "split": split, **proxy_binary_metrics(frame, probability, cfg["calibration"]["bins"])})
    predictions = pd.concat(pred_rows, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    write_table(predictions, model_dir / "prediction_raw.parquet")
    write_table(metrics, model_dir / "metrics.tsv")
    write_table(pd.DataFrame(history), model_dir / "training_history.tsv")
    embedding_metadata = None
    if bool(cfg["training"].get("export_node_embeddings", False)):
        embedding_metadata = export_node_embeddings(
            cfg, model, graph, bundle, kind, fold_row, model_dir
        )
    write_json({
        "features": features,
        "device": device,
        "best_epoch": best_epoch,
        "fold": fold_row.to_dict(),
        "proxy_positive_policy": "strong_positive+weak_positive",
        "weak_positive_weight": cfg["training"]["weak_positive_weight"],
        "node_feature_policy": "degree_recomputed_after_fold_context_exclusion_and_relation_sampling",
        "mixed_precision": amp_enabled,
        "max_edges_per_relation": int(cfg["training"]["max_edges_per_relation"]),
        "cuda_device_name": torch.cuda.get_device_name(device) if torch.device(device).type == "cuda" else None,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if torch.device(device).type == "cuda" else None,
        "experiment_seed": cfg.get("_experiment_seed", cfg.get("random_seed")),
        "node_embeddings_exported": embedding_metadata is not None,
        "embedding_sha256": (
            embedding_metadata["sha256"] if embedding_metadata is not None else None
        ),
    }, model_dir / "metadata.json")
    return predictions, metrics


def score_gnn_candidates(cfg: dict[str, Any], model_dir: Path, candidate: pd.DataFrame) -> pd.DataFrame:
    torch = require_torch_geometric()
    checkpoint = torch.load(model_dir / "best.pt", map_location="cpu", weights_only=False)
    kind = checkpoint["kind"]
    device = available_device(cfg["training"].get("device", "auto"))
    amp_enabled = bool(cfg["training"].get("mixed_precision", False) and torch.device(device).type == "cuda")
    excluded_cancers = {str(checkpoint["fold"]["test_cancer"]), str(checkpoint["fold"]["validation_cancer"])}
    bundle = load_graph_bundle(cfg, int(checkpoint["fold"]["split_seed"]), excluded_cancers=excluded_cancers)
    graph = move_graph(bundle, device, kind)
    model = build_model(kind, bundle, len(checkpoint["features"]), cfg).to(device)
    model.load_state_dict(checkpoint["model_state"]); model.eval()
    candidate = candidate.copy()
    candidate["proxy_label"] = candidate.get("label", 0)
    candidate["direction_label"] = np.nan
    frame, tensors = candidate_tensors(candidate, bundle, checkpoint["features"], device, kind)
    batch_size = int(cfg["scoring"]["batch_size"])
    rows = []
    with torch.no_grad():
        # Encode once, then decode candidates in batches.
        if kind == "rgcn":
            with amp_autocast(torch, device, amp_enabled):
                h = model.encode(graph)
            for start in range(0, len(frame), batch_size):
                sl = slice(start, min(start + batch_size, len(frame)))
                with amp_autocast(torch, device, amp_enabled):
                    logits, d = model.decoder(h[tensors["l"][sl]], h[tensors["p"][sl]], h[tensors["c"][sl]], tensors["x"][sl])
                part = frame.iloc[sl].copy(); part["raw_logit"] = logits.float().cpu().numpy(); part["raw_probability"] = torch.sigmoid(logits).float().cpu().numpy(); part["direction_probability"] = torch.sigmoid(d).float().cpu().numpy(); rows.append(part)
        else:
            with amp_autocast(torch, device, amp_enabled):
                h = model.encode(graph)
            for start in range(0, len(frame), batch_size):
                sl = slice(start, min(start + batch_size, len(frame)))
                with amp_autocast(torch, device, amp_enabled):
                    logits, d = model.decoder(h["lncRNA"][tensors["l"][sl]], h[model.pathway_target_node_type][tensors["p"][sl]], h["cancer"][tensors["c"][sl]], tensors["x"][sl])
                part = frame.iloc[sl].copy(); part["raw_logit"] = logits.float().cpu().numpy(); part["raw_probability"] = torch.sigmoid(logits).float().cpu().numpy(); part["direction_probability"] = torch.sigmoid(d).float().cpu().numpy(); rows.append(part)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
