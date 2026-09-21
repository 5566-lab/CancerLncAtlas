#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from cc_hhgt.common import configure_logging, load_config, stage_status
from cc_hhgt.graph_build import build_fold_manifests, build_graph_tables
from cc_hhgt.v29_state_graph import build_all_state_assets


def main() -> int:
    parser = argparse.ArgumentParser(description="Build V2.9 state nodes, state edges, candidates and graph")
    parser.add_argument("--config", default="config/model_v2_9_state_graph.yaml")
    parser.add_argument("--cancers", nargs="*")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)
    cfg = load_config(args.config)
    with stage_status(cfg, "41_prepare_v29_state_graph"):
        state = build_all_state_assets(cfg, args.cancers)
        nodes, edges = build_graph_tables(cfg)
        folds = build_fold_manifests(cfg)
        payload = {
            "status": "PASS",
            "state": state,
            "n_nodes": len(nodes),
            "n_edges": len(edges),
            "node_type_counts": nodes.node_type.astype(str).value_counts().to_dict(),
            "state_relation_counts": edges.loc[
                edges.source_type.astype(str).eq("state") | edges.target_type.astype(str).eq("state"),
                "relation_type",
            ].astype(str).value_counts().to_dict(),
            "n_folds": len(folds),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
