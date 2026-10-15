#!/usr/bin/env python3
"""Bind native Drug predictions, signed measurements and ablation to selected GeneSets.

CURRENT_DRUG_CONTEXT_INPUT_V1 requires core_run_id, drug_run_id and {path,sha256}
records: core_manifest, selected_analysis, sparse_query_manifest,
attribution_manifest, attribution_audit, response_scan_audit. Native statistics
and frozen ablations are replayed. No optimizer, training or deployment runs.
"""
from __future__ import annotations
import argparse
import json
from cc_hhgt.v32.current_drug_context import materialize_current_drug_context


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",required=True)
    parser.add_argument("--output",required=True)
    args = parser.parse_args(argv)
    print(json.dumps(materialize_current_drug_context(args.manifest,args.output)))
    return 0


if __name__ == "__main__": raise SystemExit(main())
