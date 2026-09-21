#!/usr/bin/env python3
"""Run disjoint per-cancer V3.2 single-cell lncRNA/pathway partitions.

This helper intentionally calls only ``analyze`` from the pinned pipeline
source.  The authoritative sequential finalizer remains responsible for the
cross-cancer summary and combined artifact after it reuses these fresh
per-cancer outputs.
"""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("cancers", nargs="+")
    args = parser.parse_args()

    source = args.source.resolve()
    if not source.is_file():
        raise RuntimeError(f"Pinned pipeline source is missing: {source}")
    sys.path.insert(0, str(source.parent))
    namespace = runpy.run_path(str(source))
    analyze = namespace.get("analyze")
    if not callable(analyze):
        raise RuntimeError("Pinned pipeline source does not expose analyze()")

    for cancer in args.cancers:
        print(f"[{cancer}] partitioned lncRNA-pathway associations", flush=True)
        _output, summary = analyze(cancer)
        print(
            f"[{cancer}] retained_rows={summary['retained_rows']} status={summary['status']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
