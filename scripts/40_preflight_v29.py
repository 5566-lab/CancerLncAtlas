#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from cc_hhgt.common import configure_logging, input_path, load_config, write_json
from cc_hhgt.v29_state_graph import DEFAULT_REQUIRED_STATES


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail-closed V2.9 input preflight")
    parser.add_argument("--config", default="config/model_v2_9_state_graph.yaml")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)
    cfg = load_config(args.config)
    checks = []

    def add(name: str, ok: bool, detail: object) -> None:
        checks.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})

    required_paths = {
        "bulk_lnc_expression": input_path(cfg, "bulk_lnc_expression"),
        "bulk_gene_expression": input_path(cfg, "bulk_gene_expression"),
        "bulk_pathway_activity": input_path(cfg, "bulk_pathway_activity"),
        "bulk_covariates": input_path(cfg, "bulk_covariates"),
        "tumor_state": input_path(cfg, "tumor_state"),
        "dim_lncRNA": input_path(cfg, "dim_lncRNA"),
        "dim_gene": input_path(cfg, "dim_gene"),
        "dim_pathway": input_path(cfg, "dim_pathway"),
        "dim_cancer": input_path(cfg, "dim_cancer"),
        "pathway_gene_member": input_path(cfg, "pathway_gene_member"),
    }
    for name, path in required_paths.items():
        add(name, bool(path and path.exists()), str(path))

    state_path = required_paths["tumor_state"]
    if state_path and state_path.exists():
        header = pd.read_csv(state_path, sep="\t", nrows=0).columns.astype(str).tolist() if str(state_path).endswith((".tsv", ".tsv.gz", ".txt", ".txt.gz")) else pd.read_parquet(state_path).columns.astype(str).tolist()
        required_states = cfg.get("state_graph", {}).get("required_states", DEFAULT_REQUIRED_STATES)
        missing = sorted(set(map(str, required_states)) - set(header))
        add("required_state_columns", not missing, {"missing": missing, "required": required_states})

    output = cfg["_results"] / "reports" / "V2_9_PREFLIGHT.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    failures = [row for row in checks if row["status"] == "FAIL"]
    payload = {
        "status": "PASS" if not failures else "FAIL",
        "analysis_version": cfg["analysis_version"],
        "failures": failures,
        "checks": checks,
        "strict_retraining_required": True,
        "models": ["rgcn", "hgt", "cc_hhgt_strict"],
    }
    write_json(payload, output)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if failures:
        raise RuntimeError("V2.9 preflight failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
