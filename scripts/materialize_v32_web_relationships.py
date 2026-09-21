#!/usr/bin/env python3
"""Build a fresh-V3.2-only exact-pathway website staging bundle."""
from __future__ import annotations

import argparse
import json

from cc_hhgt.v32.web_relationship_materialization import (
    materialize_fresh_v32_web_relationships,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--expected-input-manifest-sha256", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--memory-limit",
        default="1GB",
        help="DuckDB hard memory limit, for example 512MB or 1GB (default: 1GB)",
    )
    parser.add_argument(
        "--temp-directory",
        help="Optional DuckDB spill directory on the same authorized server",
    )
    args = parser.parse_args()
    result = materialize_fresh_v32_web_relationships(
        input_manifest_path=args.input_manifest,
        expected_input_manifest_sha256=args.expected_input_manifest_sha256,
        output_root=args.output_root,
        memory_limit=args.memory_limit,
        temp_directory=args.temp_directory,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
