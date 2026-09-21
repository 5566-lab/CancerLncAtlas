#!/usr/bin/env python3
"""Print non-row Parquet metadata for server-side schema inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()

    import pyarrow.parquet as pq

    records = []
    for raw in args.paths:
        path = Path(raw).resolve()
        parquet = pq.ParquetFile(path)
        records.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "rows": parquet.metadata.num_rows,
                "row_groups": parquet.metadata.num_row_groups,
                "schema": [
                    {"name": field.name, "type": str(field.type), "nullable": field.nullable}
                    for field in parquet.schema_arrow
                ],
            }
        )
    print(json.dumps(records, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
