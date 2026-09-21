#!/usr/bin/env python3
"""Export the unique GENCODE lncRNA symbol universe for symbol-only H5 inputs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    annotation = pd.read_parquet(args.annotation.resolve())
    required = {"gene_id", "gene_symbol", "gene_class", "unique_symbol"}
    if required - set(annotation.columns):
        raise RuntimeError(f"annotation schema missing: {sorted(required - set(annotation.columns))}")
    symbols = sorted(set(annotation.loc[
        annotation.gene_class.astype(str).eq("lncRNA")
        & annotation.unique_symbol.astype(bool),
        "gene_symbol",
    ].astype(str)))
    if len(symbols) < 1000:
        raise RuntimeError(f"unique lncRNA symbol universe unexpectedly small: {len(symbols)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.partial.{os.getpid()}")
    temporary.write_text(json.dumps(symbols, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({"status": "PASS", "symbols": len(symbols), "output": str(args.output)}))


if __name__ == "__main__":
    main()
