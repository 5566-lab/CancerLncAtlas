#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cc_hhgt.v32.gdc_segment_cnv import (
    TCGA_CANCERS,
    flatten_tumour_manifest,
    gdc_filter,
    query_gdc_masked_segments,
    reconcile_existing_files,
    reconcile_existing_inventory,
    write_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a candidate-only GDC masked CNV segment manifest")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cancers", nargs="*", default=list(TCGA_CANCERS))
    parser.add_argument("--existing-segment-roots", nargs="*", type=Path, default=[])
    parser.add_argument("--existing-md5-inventory", type=Path)
    parser.add_argument("--page-size", type=int, default=5000)
    args = parser.parse_args()
    requested = tuple(value.upper() for value in args.cancers)
    hits, query_audit = query_gdc_masked_segments(requested, page_size=args.page_size)
    rows, flatten_audit = flatten_tumour_manifest(hits)
    existing_audit = reconcile_existing_files(rows, args.existing_segment_roots)
    if args.existing_md5_inventory:
        inventory = []
        for line in args.existing_md5_inventory.read_text(encoding="utf-8").splitlines():
            checksum, path = line.split(maxsplit=1)
            inventory.append((checksum, path.strip()))
        existing_audit = reconcile_existing_inventory(rows, inventory)
    audit = {
        **query_audit,
        **flatten_audit,
        **existing_audit,
        "requested_cancers": list(requested),
        "queried_cancers": list(requested),
        "existing_segment_roots": [str(path) for path in args.existing_segment_roots],
        "filters": gdc_filter(requested),
    }
    payload = write_manifest(rows, args.output_dir, audit)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
