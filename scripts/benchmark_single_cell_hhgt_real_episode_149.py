#!/usr/bin/env python3
"""Run one hash-bound real SC-HHGT optimizer step on host 149."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import socket
import time

import numpy as np
import torch

from cc_hhgt.v32.single_cell_hhgt import LASSO_EDGE_TYPES, NODE_TYPES
from scripts import train_single_cell_hhgt_149 as trainer


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--episode", required=True, type=Path)
    result.add_argument("--episode-root", required=True, type=Path)
    result.add_argument("--arm", choices=trainer.ARMS, default="with_lasso")
    result.add_argument("--rows", type=int, default=2048)
    result.add_argument("--threads", type=int, default=8)
    result.add_argument("--seed", type=int, default=20261005)
    result.add_argument("--receipt", type=Path)
    return result


def main() -> int:
    args = parser().parse_args()
    trainer.require_host_149(socket.gethostname())
    if args.rows <= 0 or args.threads <= 0:
        raise RuntimeError("rows and threads must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    root = args.episode_root.expanduser().resolve()
    path = args.episode.expanduser().resolve()
    record = trainer.file_record(path)
    started = time.perf_counter()
    header = trainer._load_weights_only(path)
    outer_fold_id = int(header["outer_fold_id"])
    del header
    descriptor = trainer._scan_descriptor(path, record, outer_fold_id, root, {})
    episode = trainer.load_episode(descriptor, args.arm)
    loaded = time.perf_counter()
    settings = trainer.TrainingSettings(
        episode_root=root,
        output_root=Path("/tmp/unused_sc_hhgt_real_episode_benchmark"),
        seed=args.seed,
        hidden_channels=64,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        threads_per_worker=args.threads,
    )
    model = trainer._initialize_model(
        episode.graph, settings, episode.pair_features.shape[1], args.seed
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    rows = trainer._sample_rows(len(episode.rows), args.rows, args.seed)
    before = trainer._state_hash(model.state_dict(), "encoder.layers.")
    step = trainer._train_episode_step(
        model, optimizer, episode, rows, settings.max_gradient_norm
    )
    after = trainer._state_hash(model.state_dict(), "encoder.layers.")
    finished = time.perf_counter()
    edge_counts = {
        "|".join(edge_type): int(episode.graph.data[edge_type].edge_index.shape[1])
        for edge_type in episode.graph.edge_schema
    }
    lasso_edges = sum(
        int(episode.graph.data[edge_type].edge_index.shape[1])
        for edge_type in LASSO_EDGE_TYPES
    )
    external_edges = sum(
        count for key, count in edge_counts.items() if "|external_" in key
    )
    if args.arm == "with_lasso" and lasso_edges <= 0:
        raise RuntimeError("real with-LASSO episode has no LASSO edges")
    if external_edges <= 0:
        raise RuntimeError("real episode has no external evidence edges")
    if int(episode.graph.data["cell_context"].num_nodes) <= 0:
        raise RuntimeError("real episode has no cell_context nodes")
    if before == after or step["hgt_gradient_l1"] <= 0:
        raise RuntimeError("real HGTConv did not receive a parameter update")
    result = {
        "format": "CANCERLNCATLAS_V32_SC_HHGT_REAL_EPISODE_BENCHMARK_V1",
        "status": "PASS",
        "target_host": "149",
        "device": "cpu",
        "cuda_used": False,
        "episode": record,
        "episode_id": descriptor.episode_id,
        "arm": args.arm,
        "row_population": len(episode.rows),
        "sampled_rows": len(rows),
        "sample_index_sha256": hashlib.sha256(
            np.asarray(rows, dtype=np.int64).tobytes()
        ).hexdigest(),
        "node_counts": {
            node_type: int(episode.graph.data[node_type].num_nodes)
            for node_type in NODE_TYPES
        },
        "edge_counts": edge_counts,
        "lasso_edge_count": lasso_edges,
        "external_edge_count": external_edges,
        "pair_feature_count": int(episode.pair_features.shape[1]),
        "hgt_parameter_sha256_before": before,
        "hgt_parameter_sha256_after": after,
        "hgt_parameters_changed": before != after,
        "optimizer_step": step,
        "load_seconds": loaded - started,
        "optimizer_step_seconds": finished - loaded,
        "maximum_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    if args.receipt is not None:
        trainer.atomic_json(args.receipt.expanduser().resolve(), result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
