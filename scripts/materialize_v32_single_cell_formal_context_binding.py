#!/usr/bin/env python3
"""Materialize the hash-bound V3.2 17-cancer single-cell context release."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.single_cell_formal_context_query import (  # noqa: E402
    build_single_cell_formal_context_binding,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh-root", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--detection", required=True, type=Path)
    parser.add_argument("--remote-observation", required=True, type=Path)
    parser.add_argument("--fusion-adapter-binding", required=True, type=Path)
    parser.add_argument("--multimodal-checkpoint", required=True, type=Path)
    parser.add_argument("--hnsc-ucell-binding", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = build_single_cell_formal_context_binding(
        fresh_root=args.fresh_root,
        dataset_manifest_path=args.dataset_manifest,
        detection_path=args.detection,
        remote_observation_path=args.remote_observation,
        fusion_adapter_binding_path=args.fusion_adapter_binding,
        multimodal_checkpoint_path=args.multimodal_checkpoint,
        hnsc_ucell_binding_path=args.hnsc_ucell_binding,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "binding_path": result["binding_path"],
                "binding_sha256": result["binding_sha256"],
                "success_path": result["success_path"],
                "status": result["binding"]["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
