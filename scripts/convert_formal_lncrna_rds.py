#!/usr/bin/env python3
"""Convert native TMM logCPM RDS matrices, retaining consumed source ancestry."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from cc_hhgt.v32.formal_expression_sources import convert


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "artifacts/input/formal_lnc_logcpm")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/input/formal_lncRNA_expression")
    parser.add_argument("--cancer", action="append", help="Explicit subset; omitted requires all 33 cancers")
    args = parser.parse_args()
    convert(args.source, args.output, cancers=args.cancer, runner=__file__)


if __name__ == "__main__":
    main()
