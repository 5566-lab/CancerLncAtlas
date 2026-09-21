#!/usr/bin/env python3
"""CPU benchmark for the V3.1 non-destructive runtime graph path."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cc_hhgt.common import load_config
from cc_hhgt.gnn import load_graph_bundle, runtime_bundle_for_step


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--asset-results", required=True)
    parser.add_argument("--contract", choices=["CONTRACT-S", "CONTRACT-T"], default="CONTRACT-T")
    parser.add_argument("--test-cancer", default="BRCA")
    parser.add_argument("--validation-cancer", default="CESC")
    parser.add_argument("--pseudoheldout-cancer", default="COAD")
    parser.add_argument("--seed", type=int, default=20260726)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["_results"] = Path(args.asset_results).resolve()
    formal_heldout = {args.test_cancer, args.validation_cancer}
    started = time.perf_counter()
    bundle = load_graph_bundle(
        cfg,
        args.seed,
        excluded_cancers=formal_heldout,
        contract=args.contract,
    )
    load_seconds = time.perf_counter() - started
    episode_times = []
    active_counts = []
    for step in (0, 1):
        started = time.perf_counter()
        episode = runtime_bundle_for_step(
            bundle,
            step,
            pseudoheldout_cancer=args.pseudoheldout_cancer,
            formal_heldout_cancers=formal_heldout,
            reference_only=set(),
        )
        episode_times.append(time.perf_counter() - started)
        active_counts.append(len(episode.active_edges))
    print(
        json.dumps(
            {
                "contract": args.contract,
                "formal_heldout": sorted(formal_heldout),
                "pseudoheldout_cancer": args.pseudoheldout_cancer,
                "canonical_edges": len(bundle.edges),
                "runtime_chunks": bundle.runtime_num_chunks,
                "load_seconds": load_seconds,
                "first_episode_seconds": episode_times[0],
                "cached_episode_seconds": episode_times[1],
                "active_edges": active_counts,
                "episode_cache_entries": len(bundle.episode_cache),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
