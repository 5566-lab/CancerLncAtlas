#!/usr/bin/env python3
"""Probe serial per-compartment BH on the immutable r3 raw spool copy."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import time

import pyarrow.parquet as pq


RUNNER = Path(
    "./data/CancerLncAtlas/runtime/tools/"
    "single_cell_r7_streaming_bh_candidate_20260829_r3/scripts/"
    "run_v32_single_cell_r7_streaming.py"
)
SOURCE_RAW = Path(
    "./data/CancerLncAtlas/results/model/"
    "v32_single_cell_r7_fresh_streaming_r7_sarc_20260829_r3/"
    ".cancer_id=SARC.92a219d24d4af3c2.publish.1293948/"
    ".association_evidence_raw.parquet"
)
PROBE_ROOT = Path(
    "./data/CancerLncAtlas/runtime/audits/"
    "single_cell_r7_bh_compartment_probe_20260829_r3"
)


def memory_status() -> dict[str, int]:
    result = {}
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith(("VmRSS:", "VmHWM:")):
            key, value = line.split(":", 1)
            result[f"{key.lower()}_bytes"] = int(value.strip().split()[0]) * 1024
    return result


def exclusive_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    assert not PROBE_ROOT.exists() and not PROBE_ROOT.is_symlink()
    PROBE_ROOT.mkdir(parents=True)
    raw = PROBE_ROOT / "raw_copy.parquet"
    output = PROBE_ROOT / "association_evidence.parquet"
    scratch = PROBE_ROOT / "scratch"
    shutil.copyfile(SOURCE_RAW, raw)
    raw_rows = int(pq.ParquetFile(raw).metadata.num_rows)
    spec = importlib.util.spec_from_file_location("bh_compartment_candidate", RUNNER)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    started = time.time()
    try:
        observed = runner._finalize_association_evidence(
            raw_path=raw,
            output_path=output,
            scratch=scratch,
            raw_rows=raw_rows,
        )
        payload = {
            "association_duckdb_memory_limit": runner.ASSOCIATION_DUCKDB_MEMORY_LIMIT,
            "association_execution_strategy": runner.ASSOCIATION_EXECUTION_STRATEGY,
            "elapsed_seconds": round(time.time() - started, 3),
            **memory_status(),
            "output_rows": int(pq.ParquetFile(output).metadata.num_rows),
            "raw_rows": raw_rows,
            "runner_sha256": hashlib.sha256(RUNNER.read_bytes()).hexdigest(),
            "status": "PASS",
        }
        assert observed == raw_rows == payload["output_rows"]
        exclusive_json(PROBE_ROOT / "RESULT.json", payload)
        print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
        return 0
    except Exception as exc:
        payload = {
            "elapsed_seconds": round(time.time() - started, 3),
            "error_type": type(exc).__name__,
            "message": str(exc),
            **memory_status(),
            "raw_rows": raw_rows,
            "runner_sha256": hashlib.sha256(RUNNER.read_bytes()).hexdigest(),
            "status": "FAIL",
        }
        exclusive_json(PROBE_ROOT / "RESULT.json", payload)
        print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
