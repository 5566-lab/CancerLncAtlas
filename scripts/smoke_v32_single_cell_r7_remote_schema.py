#!/usr/bin/env python3
"""Isolated server smoke for the frozen r7 Arrow-schema runner."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


RUNNER = Path(
    "./data/CancerLncAtlas/runtime/tools/"
    "single_cell_r7_streaming_20260829_r8/scripts/"
    "run_v32_single_cell_r7_streaming.py"
)
R4_PATHS = (
    Path(
        "./data/CancerLncAtlas/results/model/"
        "v32_single_cell_r7_fresh_streaming_r8_sarc_20260829_r4"
    ),
    Path(
        "./data/CancerLncAtlas/runtime/audits/"
        "single_cell_r7_r8_sarc_20260829_r4"
    ),
)


def main() -> int:
    spec = importlib.util.spec_from_file_location("r7_remote_schema_smoke", RUNNER)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    schema = pa.schema([("reason", pa.string()), ("score", pa.float32())])
    with tempfile.TemporaryDirectory(
        dir="./data/CancerLncAtlas/runtime"
    ) as temporary:
        output = Path(temporary) / "schema.parquet"
        writer = runner._ParquetFrameWriter(output, schema=schema)
        writer.write(
            pd.DataFrame({"reason": [None, pd.NA], "score": [None, None]})
        )
        writer.write(pd.DataFrame({"reason": [], "score": []}))
        writer.write(
            pd.DataFrame(
                {"reason": ["\u4e2d\u6587", ""], "score": [1.0, 2.0]}
            )
        )
        rows = writer.close()
        table = pq.read_table(output)
        assert rows == 4
        assert table.schema.equals(schema, check_metadata=False)
        reasons = table.column("reason").to_pylist()
        assert reasons == [None, None, "\u4e2d\u6587", ""]
        raw = Path(temporary) / "raw.parquet"
        evidence = Path(temporary) / "evidence.parquet"
        pd.DataFrame(
            {
                "dataset_id": ["D"] * 3,
                "cancer_id": ["SARC"] * 3,
                "compartment_order": [0, 1, 2],
                "compartment": ["malignant", "immune", "stromal"],
                "lncrna_id": ["\u957f\u94fe\u4e00", "L2", "L3"],
                "lncrna_symbol": ["\u7b26\u53f7\u4e00", "S2", "S3"],
                "pathway_id": ["\u901a\u8def\u4e00", "P2", "P3"],
                "n_donors": [8, 7, 6],
                "spearman_rho": [0.8, -0.7, 0.6],
                "nominal_p": [0.001, 0.002, 0.003],
                "total_tests": [100, 20, 30],
            }
        ).to_parquet(raw, index=False)
        assert runner._finalize_association_evidence(
            raw_path=raw,
            output_path=evidence,
            scratch=Path(temporary) / "bh_scratch",
            raw_rows=3,
        ) == 3
        bh = pq.read_table(evidence).to_pandas()
        assert bh.compartment.tolist() == ["malignant", "immune", "stromal"]
        assert bh.fdr_0_10_pass.tolist() == [True, True, True]
    assert (
        runner._pathway_summary_schema().field("unavailable_reason").type
        == pa.string()
    )
    assert all(not path.exists() and not path.is_symlink() for path in R4_PATHS)
    assert runner.ASSOCIATION_DUCKDB_MEMORY_LIMIT == "64MB"
    assert runner.ASSOCIATION_BH_FAMILY == "COMPARTMENT_ORDER"
    assert runner.ASSOCIATION_STAGE_KEY_POLICY.endswith("NO_PARQUET_ROWID_JOIN")
    print(
        json.dumps(
            {
                "r4_paths_absent": True,
                "reasons": reasons,
                "rows": rows,
                "bh_rows": len(bh),
                "runner_sha256": hashlib.sha256(RUNNER.read_bytes()).hexdigest(),
                "status": "PASS",
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
