from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .common import first_existing_column, input_path, read_table, write_json, write_table


def standardize_cellline_context(cfg: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    model_path = input_path(cfg, "cell_model_passports_model")
    if model_path and model_path.exists():
        model = read_table(model_path)
        ach = first_existing_column(model, ["ModelID", "model_id", "depmap_id", "ACH_ID"], required=False)
        sanger = first_existing_column(model, ["SangerModelID", "sanger_model_id", "model_name"], required=False)
        lineage = first_existing_column(model, ["lineage", "cancer_type", "tissue", "TCGA"], required=False)
        columns = [x for x in [ach, sanger, lineage] if x]
        out = model[columns].copy() if columns else model.copy()
        rename = {}
        if ach: rename[ach] = "depmap_model_id"
        if sanger: rename[sanger] = "sanger_model_id"
        if lineage: rename[lineage] = "lineage"
        out = out.rename(columns=rename).drop_duplicates()
        write_table(out, cfg["_standardized"] / "cell_model_crosswalk.parquet")
        summary["cell_model_crosswalk_rows"] = len(out)
        summary["cell_model_crosswalk_path"] = str(model_path)
    for key in ["cell_model_passports_lnc_expression", "depmap_protein_expression", "depmap_model_metadata"]:
        path = input_path(cfg, key)
        summary[key] = {"path": str(path) if path else None, "exists": bool(path and path.exists())}
        if path and path.exists():
            summary[key]["size_bytes"] = path.stat().st_size if path.is_file() else None
    summary["usage"] = {
        "cell_model_passports_lnc_expression": "upstream lncRNA expression source for PRISM/GDSC associations",
        "depmap_protein_expression": "protein-coding control/pathway covariate source; consumed through model_covariates in standardized drug association results when present",
        "cell_model_crosswalk": "ACH-Sanger-lineage alignment",
    }
    write_json(summary, cfg["_results"] / "tables" / "cellline_context_manifest.json")
    return summary
