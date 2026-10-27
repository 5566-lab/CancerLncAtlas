"""Rebuild the model-facing SC-HHGT LASSO inputs from recorded evidence.

Each completed episode records the per-response LASSO summary and the localized
coefficient/stability edges that the HGT model actually consumes. This module
rebuilds candidate nodes, pair features and sparse edges from those recorded
values plus the pinned static graph assets, then compares them to the payload
tensors.

It never fits a LASSO, never runs an optimizer and it does not assert that the
recorded coefficients solve the stated L1 objective: the weighted design matrix
is not re-solved here. Fitted-coefficient provenance therefore remains a
separate, unclosed gap.

Rounding note: the payload stores coefficients and stability as float32 while
the original materializer computed pair features from float64 values. Comparisons
use a documented tolerance, and a stability value that sits within that tolerance
of the recorded threshold is reported separately because the support mask can
legitimately differ by one float32 rounding step.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .cnv_input_sources import require_host
from .single_cell_batched_lasso import BatchedLassoResult
from .single_cell_hhgt_episodes import (
    LASSO_DERIVED_PAIR_FEATURE_MASK,
    PAIR_FEATURE_NAMES,
    lasso_pair_features,
    local_node_indices,
    localized_lasso_edges,
)

FORMAT = "NATIVE_SC_LASSO_MODEL_INPUT_REPLAY_V1"
CODE = (
    "cc_hhgt/v32/single_cell_lasso_replay.py",
    "cc_hhgt/v32/single_cell_hhgt_episodes.py",
    "cc_hhgt/v32/single_cell_batched_lasso.py",
)
SUMMARY_KEYS = (
    "response_index",
    "response_id",
    "available",
    "unavailable_reason",
    "selected_alpha",
    "inner_cv_mse",
    "inner_null_mse",
    "bootstrap_successes",
)
EDGE_KEYS = ("lnc_index", "gene_index", "coefficient", "stability")
TOLERANCE = 1e-6
BOUNDARY_TOLERANCE = 1e-6


def _summary_arrays(summary):
    if not isinstance(summary, dict):
        raise ValueError("SC_LASSO_REPLAY_SUMMARY")
    missing = [key for key in SUMMARY_KEYS if key not in summary]
    if missing:
        raise ValueError("SC_LASSO_REPLAY_SUMMARY_COLUMNS:" + ",".join(missing))
    count = len(summary["response_id"])
    if not count:
        raise ValueError("SC_LASSO_REPLAY_EMPTY_SUMMARY")
    for key in SUMMARY_KEYS:
        if len(summary[key]) != count:
            raise ValueError("SC_LASSO_REPLAY_SUMMARY_LENGTH:" + key)
    if not np.array_equal(np.asarray(summary["response_index"], dtype=np.int64), np.arange(count)):
        raise ValueError("SC_LASSO_REPLAY_RESPONSE_INDEX")
    available = np.asarray(summary["available"], dtype=bool)
    return count, {
        "response_ids": tuple(map(str, summary["response_id"])),
        "available": available,
        "unavailable_reason": np.asarray(summary["unavailable_reason"], dtype=object),
        "selected_alpha": np.asarray(summary["selected_alpha"], dtype=np.float64),
        "inner_cv_mse": np.asarray(summary["inner_cv_mse"], dtype=np.float64),
        "inner_null_mse": np.asarray(summary["inner_null_mse"], dtype=np.float64),
        "bootstrap_successes": np.asarray(summary["bootstrap_successes"], dtype=np.int64),
    }


def _row_indices(payload, identities):
    rows = payload.get("row_metadata")
    if not isinstance(rows, dict) or "lncrna_id" not in rows or "pathway_id" not in rows:
        raise ValueError("SC_LASSO_REPLAY_ROW_METADATA")
    lnc_lookup = {value: index for index, value in enumerate(identities["lncRNA"])}
    pathway_lookup = {value: index for index, value in enumerate(identities["pathway"])}
    try:
        row_lnc = np.asarray([lnc_lookup[str(value)] for value in rows["lncrna_id"]], dtype=np.int64)
    except KeyError as error:
        raise ValueError("SC_LASSO_REPLAY_LNCRNA_IDENTITY") from error
    try:
        row_pathway = np.asarray([pathway_lookup[str(value)] for value in rows["pathway_id"]], dtype=np.int64)
    except KeyError as error:
        raise ValueError("SC_LASSO_REPLAY_PATHWAY_IDENTITY") from error
    if not len(row_lnc) or len(row_lnc) != len(row_pathway):
        raise ValueError("SC_LASSO_REPLAY_ROW_COUNT")
    return row_lnc, row_pathway


def _edges(payload, nodes, response_global, response_count, gene_count):
    edges = payload.get("sparse_lasso_edges")
    if not isinstance(edges, dict):
        raise ValueError("SC_LASSO_REPLAY_EDGE_PAYLOAD")
    missing = [key for key in EDGE_KEYS if key not in edges]
    if missing:
        raise ValueError("SC_LASSO_REPLAY_EDGE_COLUMNS:" + ",".join(missing))
    lnc_local = np.asarray(edges["lnc_index"], dtype=np.int64)
    gene_local = np.asarray(edges["gene_index"], dtype=np.int64)
    coefficient = np.asarray(edges["coefficient"], dtype=np.float64)
    stability = np.asarray(edges["stability"], dtype=np.float64)
    if not (len(lnc_local) == len(gene_local) == len(coefficient) == len(stability)):
        raise ValueError("SC_LASSO_REPLAY_EDGE_LENGTH")
    node_lnc = np.asarray(nodes["lncRNA"], dtype=np.int64)
    node_gene = np.asarray(nodes["gene"], dtype=np.int64)
    if len(lnc_local) and (lnc_local.min() < 0 or gene_local.min() < 0
                           or lnc_local.max() >= len(node_lnc) or gene_local.max() >= len(node_gene)):
        raise ValueError("SC_LASSO_REPLAY_EDGE_LOCAL_INDEX")
    lnc_global = node_lnc[lnc_local]
    gene_global = node_gene[gene_local]
    response_of = {int(value): index for index, value in enumerate(response_global)}
    try:
        response_index = np.asarray([response_of[int(value)] for value in lnc_global], dtype=np.int64)
    except KeyError as error:
        raise ValueError("SC_LASSO_REPLAY_EDGE_RESPONSE") from error
    if not np.isfinite(coefficient).all() or not np.isfinite(stability).all():
        raise ValueError("SC_LASSO_REPLAY_EDGE_NONFINITE")
    if not (abs(coefficient) > 0).all() or not ((0.0 <= stability) & (stability <= 1.0)).all():
        raise ValueError("SC_LASSO_REPLAY_EDGE_DOMAIN")
    if gene_global.size and (gene_global.min() < 0 or gene_global.max() >= gene_count):
        raise ValueError("SC_LASSO_REPLAY_EDGE_GENE_RANGE")
    if response_index.size and (response_index.min() < 0 or response_index.max() >= response_count):
        raise ValueError("SC_LASSO_REPLAY_EDGE_RESPONSE_RANGE")
    order = np.lexsort((gene_global, response_index))
    if not np.array_equal(order, np.arange(len(order))):
        raise ValueError("SC_LASSO_REPLAY_EDGE_ORDER")
    if len(response_index) and len(np.unique(response_index * gene_count + gene_global)) != len(response_index):
        raise ValueError("SC_LASSO_REPLAY_EDGE_DUPLICATE")
    return {
        "response_index": response_index,
        "gene_index": gene_global.astype(np.int64),
        "coefficient": coefficient,
        "stability": stability,
        "lnc_index": lnc_local,
        "gene_local": gene_local,
    }


def _result(summary_count, summary, gene_count, edges, audit):
    return BatchedLassoResult(
        response_count=summary_count,
        gene_count=gene_count,
        response_ids=summary["response_ids"],
        gene_ids=tuple(str(value) for value in range(gene_count)),
        response_index=edges["response_index"],
        gene_index=edges["gene_index"],
        coefficient=edges["coefficient"],
        stability=edges["stability"],
        available=summary["available"],
        unavailable_reason=summary["unavailable_reason"],
        selected_alpha=summary["selected_alpha"],
        inner_cv_mse=summary["inner_cv_mse"],
        inner_null_mse=summary["inner_null_mse"],
        bootstrap_successes=summary["bootstrap_successes"],
        audit=dict(audit or {}),
    )


def _same(observed, expected, name, tolerance=TOLERANCE, exact=False):
    observed = np.asarray(observed)
    expected = np.asarray(expected)
    if observed.shape != expected.shape:
        raise ValueError("SC_LASSO_REPLAY_SHAPE:" + name)
    if exact:
        if not np.array_equal(observed, expected):
            raise ValueError("SC_LASSO_REPLAY_VALUE:" + name)
        return
    if not np.isfinite(observed).all() or not np.isfinite(expected).all():
        raise ValueError("SC_LASSO_REPLAY_NONFINITE:" + name)
    if not np.allclose(observed, expected, rtol=0.0, atol=tolerance):
        raise ValueError("SC_LASSO_REPLAY_VALUE:" + name)


def verify_episode(payload, identities, static, threshold, *, tolerance=TOLERANCE):
    """Rebuild one episode's LASSO-derived tensors and compare them."""
    require_host()
    if not isinstance(payload, dict):
        raise ValueError("SC_LASSO_REPLAY_PAYLOAD")
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("SC_LASSO_REPLAY_THRESHOLD")
    recorded = payload.get("lasso_stability_threshold")
    if recorded is None or float(recorded) != float(threshold):
        raise ValueError("SC_LASSO_REPLAY_THRESHOLD_DIFFERS")
    nodes = payload.get("node_global_indices")
    arms = payload.get("pair_features_by_arm")
    if not isinstance(nodes, dict) or not isinstance(arms, dict):
        raise ValueError("SC_LASSO_REPLAY_PAYLOAD_SECTION")
    summary_count, summary = _summary_arrays(payload.get("lasso_response_summary"))
    gene_count = len(identities["gene"])
    row_lnc, row_pathway = _row_indices(payload, identities)
    response_global = np.unique(row_lnc)
    selected_pathway = np.unique(row_pathway)
    if len(response_global) != summary_count:
        raise ValueError("SC_LASSO_REPLAY_RESPONSE_COUNT")
    if [identities["lncRNA"][int(value)] for value in response_global] != list(summary["response_ids"]):
        raise ValueError("SC_LASSO_REPLAY_RESPONSE_IDENTITY")
    for kind in ("lncRNA", "gene", "pathway", "pathway_family"):
        if kind not in nodes:
            raise ValueError("SC_LASSO_REPLAY_NODE_TYPE:" + kind)
    edges = _edges(payload, nodes, response_global, summary_count, gene_count)
    if len(edges["response_index"]) and not summary["available"][edges["response_index"]].all():
        raise ValueError("SC_LASSO_REPLAY_EDGE_UNAVAILABLE_RESPONSE")
    result = _result(summary_count, summary, gene_count, edges, payload.get("lasso_audit"))
    response_position = {int(value): index for index, value in enumerate(response_global)}
    row_response = np.asarray([response_position[int(value)] for value in row_lnc], dtype=np.int64)
    rebuilt_nodes = local_node_indices(
        lnc_global=row_lnc,
        pathway_global=row_pathway,
        lasso_result=result,
        gene_pathway_membership=static["gene_pathway"],
        pathway_family_membership=static["pathway_family"],
        external_edges=static["external"],
        stability_threshold=float(threshold),
    )
    for kind, expected in rebuilt_nodes.items():
        _same(nodes[kind], expected, "nodes." + kind, exact=True)
    features = lasso_pair_features(
        result,
        static["gene_pathway"],
        row_response,
        row_pathway,
        selected_pathway,
        stability_threshold=float(threshold),
    )
    mask = np.asarray(LASSO_DERIVED_PAIR_FEATURE_MASK, dtype=bool)
    with_lasso = torch.as_tensor(arms.get("with_lasso")).detach().cpu().numpy()
    without_lasso = torch.as_tensor(arms.get("without_lasso")).detach().cpu().numpy()
    if with_lasso.shape != (len(row_lnc), len(PAIR_FEATURE_NAMES)) or without_lasso.shape != with_lasso.shape:
        raise ValueError("SC_LASSO_REPLAY_PAIR_FEATURE_SHAPE")
    boundary = int(np.sum(np.abs(edges["stability"] - float(threshold)) <= BOUNDARY_TOLERANCE))
    try:
        _same(with_lasso[:, mask], features, "pair_features.lasso", tolerance)
    except ValueError as error:
        if boundary:
            raise ValueError(
                "SC_LASSO_REPLAY_BOUNDARY_STABILITY: "
                + str(boundary)
                + " recorded edge(s) sit within "
                + str(BOUNDARY_TOLERANCE)
                + " of the stability threshold, so the float32 payload cannot reproduce the float64 support mask"
            ) from error
        raise
    _same(without_lasso[:, mask], np.zeros_like(features), "pair_features.without_lasso", tolerance)
    keep = ~mask
    _same(without_lasso[:, keep], with_lasso[:, keep], "pair_features.base_columns", tolerance)
    rebuilt_edges = localized_lasso_edges(result, response_global, rebuilt_nodes)
    for key in ("lnc_index", "gene_index"):
        _same(edges[key], torch.as_tensor(rebuilt_edges[key]).detach().cpu().numpy(), "edges." + key, exact=True)
    for key in ("coefficient", "stability"):
        recorded_values = np.asarray(edges[key if key == "stability" else "coefficient"], dtype=np.float64)
        _same(recorded_values, torch.as_tensor(rebuilt_edges[key]).detach().cpu().numpy(), "edges." + key, tolerance)
    return {
        "response_count": int(summary_count),
        "available_response_count": int(summary["available"].sum()),
        "edge_count": int(len(edges["response_index"])),
        "stable_edge_count": int(np.sum(edges["stability"] >= float(threshold))),
        "boundary_stability_edges": boundary,
        "row_count": int(len(row_lnc)),
        "node_counts": {kind: int(len(value)) for kind, value in rebuilt_nodes.items()},
        "pair_feature_columns_rebuilt": int(mask.sum()),
    }


def verify_episodes(episode_manifest, settings, manifest, *, limit=None):
    """Replay every manifest episode; fail closed on any derivation mismatch."""
    require_host()
    from scripts import materialize_single_cell_hhgt_training_episodes_149 as native
    # The finalized materializer records a flat ``episodes`` list; serialized
    # fixtures may instead nest them under ``folds``. Both layouts are accepted,
    # and any other shape is rejected rather than silently skipped.
    records = manifest.get("episodes")
    if not isinstance(records, list) or not records:
        folds = manifest.get("folds")
        records = ([item for fold in folds if isinstance(fold, dict) for item in fold.get("episodes", [])]
                   if isinstance(folds, list) else None)
    if not isinstance(records, list) or not records:
        raise ValueError("SC_LASSO_REPLAY_EPISODE_INVENTORY")
    declared = manifest.get("cancers")
    declared = set(map(str, declared)) if isinstance(declared, list) and declared else None
    threshold = float(settings.lasso_stability_threshold)
    statics = {}
    summaries = []
    for position, item in enumerate(records):
        if limit is not None and position >= int(limit):
            break
        if not isinstance(item, dict) or not {"path", "bytes"} <= set(item):
            raise ValueError("SC_LASSO_REPLAY_EPISODE_RECORD")
        path = Path(item["path"])
        native.verify_record_metadata(item, path, "NATIVE_LASSO_REPLAY_EPISODE")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        cancer = str(payload.get("cancer_id"))
        if declared is not None and cancer not in declared:
            raise ValueError("SC_LASSO_REPLAY_UNAUTHORIZED_CANCER")
        if cancer not in statics:
            statics[cancer] = native._load_cancer_static(settings, cancer)
        summary = verify_episode(payload, statics[cancer]["identifiers"], statics[cancer], threshold)
        summaries.append({"episode": path.name, "cancer_id": cancer, **summary})
        del payload
    return {
        "format": FORMAT,
        "target_host": "149",
        "training_started": False,
        "optimizer_executed": False,
        "lasso_fit_executed": False,
        "episode_manifest_sha256": native.sha256(Path(episode_manifest)),
        "stability_threshold": threshold,
        "episode_count": len(summaries),
        "episodes_in_manifest": len(records),
        "cancers": sorted(statics),
        "cancers_declared_in_manifest": declared is not None,
        "tolerance": TOLERANCE,
        "boundary_tolerance": BOUNDARY_TOLERANCE,
        "episodes": summaries,
        "implementation_sha256": {name: native.sha256(Path(native.ROOT) / name) for name in CODE},
        "fitted_coefficient_objective_replayed": False,
        "fitted_lasso_lineage_verified": False,
        "scope": "RECORDED_LASSO_EVIDENCE_TO_MODEL_INPUT_DERIVATION_ONLY",
    }


__all__ = ["BOUNDARY_TOLERANCE", "CODE", "FORMAT", "TOLERANCE", "verify_episode", "verify_episodes"]
