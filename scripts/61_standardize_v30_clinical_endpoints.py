#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import pandas as pd

from cc_hhgt.clinical_v30 import build_clinical_covariates, load_all_existing, standardise_endpoint_table
from cc_hhgt.common import input_candidates, load_config, write_json, write_table


def main() -> int:
    parser = argparse.ArgumentParser(description="Standardize OS/DSS/PFI/PFS/DFI/DFS and clinical covariates")
    parser.add_argument("--config", default="config/model_v3_0_clinical.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    paths = input_candidates(cfg, "clinical_survival")
    sources = load_all_existing(paths)
    settings = cfg["clinical"]
    endpoints, summary = standardise_endpoint_table(
        sources,
        settings.get("endpoint_aliases", {}),
        settings.get("required_endpoints", ["OS"]),
    )
    covariates = build_clinical_covariates(sources, endpoints, settings.get("clinical_covariate_aliases", {}))
    for column in settings.get("clinical_covariates", {}).get("numeric", []):
        if column in covariates:
            covariates[column] = pd.to_numeric(covariates[column], errors="coerce")
    for column in settings.get("clinical_covariates", {}).get("categorical", []):
        if column in covariates:
            covariates[column] = covariates[column].astype("string")
    root = cfg["_results"] / "clinical_standardized"
    write_table(endpoints, root / "clinical_endpoints.parquet")
    write_table(covariates, root / "clinical_covariates.parquet")
    write_table(summary, root / "clinical_endpoint_availability.tsv")
    payload = {
        "status": "COMPLETED",
        "endpoint_rows": int(len(endpoints)),
        "patients": int(endpoints.patient_id.nunique()),
        "cancers": int(endpoints.cancer_id.nunique()),
        "available_by_endpoint": endpoints.loc[endpoints.endpoint_available.eq(1)].groupby("endpoint").size().astype(int).to_dict(),
    }
    write_json(payload, root / "SUCCESS.json")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
