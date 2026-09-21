#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from cc_hhgt.common import configure_logging, load_config, stage_status
from cc_hhgt.v29_state_graph import build_pathway_state_edges_v29, build_state_catalog


def main() -> int:
    parser = argparse.ArgumentParser(description="Build covariate-residualized V2.9 pathway-state edges before pathway-family construction")
    parser.add_argument("--config", default="config/model_v2_9_state_graph.yaml")
    parser.add_argument("--cancers", nargs="*")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)
    cfg = load_config(args.config)
    with stage_status(cfg, "04_build_pathway_state_edges_v29"):
        catalog = build_state_catalog(cfg)
        summary = build_pathway_state_edges_v29(cfg, args.cancers)
        payload = {"status": "PASS", "n_states": len(catalog), "summary_rows": len(summary)}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
