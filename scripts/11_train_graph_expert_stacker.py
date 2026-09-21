#!/usr/bin/env python3
"""Train a light OOF stacker over existing R-GCN/HGT/CC-HHGT predictions."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cc_hhgt_v26.graph_stacking import (  # noqa: E402
    build_graph_expert_frame,
    save_graph_stacking,
    train_graph_stacker,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict-root", type=Path, default=ROOT / "results" / "strict_global")
    ap.add_argument("--primary-oof", type=Path, default=ROOT / "results" / "strict_release" / "strict_cross_cancer_oof_prediction.parquet")
    ap.add_argument("--output", type=Path, default=ROOT / "results" / "graph_expert_stacking")
    ap.add_argument("--epochs", type=int, default=int(os.getenv("CC_HHGT_GRAPH_STACK_EPOCHS", "80")))
    ap.add_argument("--patience", type=int, default=int(os.getenv("CC_HHGT_GRAPH_STACK_PATIENCE", "10")))
    ap.add_argument("--seed", type=int, default=int(os.getenv("CC_HHGT_GRAPH_STACK_SEED", "20260731")))
    args = ap.parse_args()

    frame = build_graph_expert_frame(args.strict_root, args.primary_oof)
    result = train_graph_stacker(frame, seed=args.seed, epochs=args.epochs, patience=args.patience)
    save_graph_stacking(result, args.output)
    print(json.dumps({
        "status": "COMPLETED",
        "output": str(args.output),
        "rows": len(result.full),
        "graph_models_retrained_by_this_stage": False,
        "underlying_graph_models_retrained": True,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
