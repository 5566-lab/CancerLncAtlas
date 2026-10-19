#!/usr/bin/env python3
"""Convert five-fold native ssGSEA outputs; execute only on server 149."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cc_hhgt.v32.fold_local_activity import convert


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, help="Native five-fold ssGSEA root; repeat for cancers")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    convert(args.source, args.output, runner=__file__)


if __name__ == "__main__": main()
