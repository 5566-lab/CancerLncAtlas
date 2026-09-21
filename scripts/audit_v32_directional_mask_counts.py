#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    args = parser.parse_args()
    root = Path(args.root)
    callable_value = np.load(root / "lncrna_callable.npy", mmap_mode="r")
    packed = np.packbits(callable_value.astype(bool), axis=0).T
    _, counts = np.unique(packed, axis=0, return_counts=True)
    print(json.dumps({
        "shape": list(callable_value.shape),
        "unique_lncrna_callable_masks": int(len(counts)),
        "largest_mask_group": int(counts.max()) if len(counts) else 0,
        "median_mask_group": float(np.median(counts)) if len(counts) else 0.0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
