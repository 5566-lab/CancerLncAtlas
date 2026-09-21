#!/usr/bin/env python
"""Validate a V3.2 staging registry without deploying or modifying the website."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.release_registry import load_release_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="Optional validation receipt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = load_release_registry(args.registry, require_staging=True)
    receipt = {
        "status": "PASS",
        "release_id": registry.release_id,
        "analysis_version": registry.analysis_version,
        "registry_sha256": registry.registry_sha256,
        "environment": registry.manifest["environment"],
        "production_deployed": registry.manifest["production_deployed"],
        "release_ready": registry.manifest["release_ready"],
        "all_non_null_predictions_newly_trained_v32": registry.manifest[
            "all_non_null_predictions_newly_trained_v32"
        ],
        "all_published_results_generated_in_v32": registry.manifest[
            "all_published_results_generated_in_v32"
        ],
        "module_statuses": {
            module_id: entry["status"] for module_id, entry in registry.manifest["modules"].items()
        },
        "website_artifact_sha256": dict(registry.artifact_hashes),
        "mixed_exact_pathway_query_enabled": registry.manifest["capabilities"]
        ["mixed_exact_pathway_query"]["enabled"],
        "supersedes": list(registry.manifest.get("supersedes", [])),
        "supersession_reason": registry.manifest.get("supersession_reason"),
    }
    payload = json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
