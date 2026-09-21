#!/usr/bin/env python3
"""Materialize the hash-pinned V3.2 single-cell diagnostic publication binding."""
from __future__ import annotations

import argparse
import json

from cc_hhgt.v32.single_cell_diagnostic_publication import (
    materialize_publication_binding,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic-manifest", required=True)
    parser.add_argument("--diagnostic-manifest-sha256", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    result = materialize_publication_binding(
        diagnostic_manifest_path=args.diagnostic_manifest,
        expected_diagnostic_manifest_sha256=args.diagnostic_manifest_sha256,
        output_root=args.output_root,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
