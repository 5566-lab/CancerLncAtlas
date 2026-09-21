#!/usr/bin/env python3
"""Independently audit the full-33 V3.2 streaming CNV store."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cc_hhgt.v32.streaming_cnv_independent_audit import audit_streaming_cnv_store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-success", required=True)
    parser.add_argument("--supervisor-success", required=True)
    parser.add_argument("--expected-store-root", required=True)
    parser.add_argument("--expected-context-sha256", required=True)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--expected-streaming-code-sha256", required=True)
    parser.add_argument("--expected-supervisor-sha256", required=True)
    parser.add_argument("--expected-total-patients", required=True, type=int)
    parser.add_argument("--expected-total-processed", required=True, type=int)
    parser.add_argument("--expected-total-typed-unavailable", required=True, type=int)
    parser.add_argument("--max-memory-bytes", type=int, default=536_870_912)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise RuntimeError(f"Independent CNV audit refuses output reuse: {output}")
    report = audit_streaming_cnv_store(
        aggregate_success_path=args.aggregate_success,
        supervisor_success_path=args.supervisor_success,
        expected_store_root=args.expected_store_root,
        expected_context_sha256=args.expected_context_sha256,
        expected_candidate_sha256=args.expected_candidate_sha256,
        expected_streaming_code_sha256=args.expected_streaming_code_sha256,
        expected_supervisor_sha256=args.expected_supervisor_sha256,
        expected_total_patients=args.expected_total_patients,
        expected_total_processed=args.expected_total_processed,
        expected_total_typed_unavailable=args.expected_total_typed_unavailable,
        max_memory_bytes=args.max_memory_bytes,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "output": str(output), "cancers": report["cancer_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
