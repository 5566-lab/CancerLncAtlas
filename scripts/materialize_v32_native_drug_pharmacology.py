#!/usr/bin/env python3
"""Adapt native DrugCentral and assay protocols using an actual staging release."""
import argparse
from pathlib import Path

from cc_hhgt.v32.drug_pharmacology_native import materialize_native_pharmacology


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = materialize_native_pharmacology(args.manifest, args.output)
    print("PREPARED_NOT_TRAINED "+str(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
