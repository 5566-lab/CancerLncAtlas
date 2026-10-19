#!/usr/bin/env python3
"""Convert one or more native five-fold R outputs for actual CNV/core consumers."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from cc_hhgt.v32.fold_local_expression import convert


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    convert(args.source, args.output, runner=__file__)


if __name__ == "__main__": main()
