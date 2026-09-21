#!/usr/bin/env python3
"""Generate a Linux-path binding overlay and original-to-server lineage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath, PureWindowsPath

from cc_hhgt.v32.portable_rebinding import PortableRebinder, SourceMap


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo-root", type=Path, default=ROOT)
    parser.add_argument("--workspace-root", type=Path, default=ROOT.parent)
    parser.add_argument(
        "--source-repo-prefix",
        default=r"D:\model\CC_HHGT_v3_2_ranked_subtypes_dev",
        help="Original Windows prefix serialized inside V3.2 manifests.",
    )
    parser.add_argument(
        "--workspace-prefix",
        default=r"D:\model",
        help="Original Windows workspace prefix serialized inside manifests.",
    )
    parser.add_argument("--overlay-root", type=Path, required=True)
    parser.add_argument("--target-root", required=True)
    parser.add_argument(
        "--source-unified",
        type=Path,
        default=ROOT / "config/v32_unified_staging_bindings.json",
    )
    args = parser.parse_args()
    source_repo = args.source_repo_root.resolve()
    workspace = args.workspace_root.resolve()
    target = PurePosixPath(args.target_root)
    rebinder = PortableRebinder(
        source_repo_root=source_repo,
        overlay_root=args.overlay_root,
        target_root=str(target),
        source_maps=(
            SourceMap(
                source_prefix=PureWindowsPath(args.source_repo_prefix),
                local_root=source_repo,
                target_relative_root=PurePosixPath("."),
            ),
            SourceMap(
                source_prefix=PureWindowsPath(args.workspace_prefix),
                local_root=workspace,
                target_relative_root=PurePosixPath("_vendor/workspace"),
            ),
        ),
        allowed_server_roots=(
            "./data/CancerLncAtlas",
            "./data/CancerLncAtlas",
        ),
    )
    lineage = rebinder.build_unified(args.source_unified)
    print(json.dumps({
        "candidate_unified": lineage["candidate_unified"],
        "mounted_capability_count": lineage["mounted_capability_count"],
        "pending_capability_count": lineage["pending_capability_count"],
        "unresolved_count": len(lineage["unresolved"]),
        "lineage": str(args.overlay_root.resolve() / "PORTABLE_REBINDING_LINEAGE.json"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
