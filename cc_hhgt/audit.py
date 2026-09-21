from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .common import LOGGER, dataframe_key_duplicates, input_path, parquet_row_count, parquet_schema, read_table, resolve_input_path, utc_now
from .io import discover_preferred

INPUT_CONTRACT: dict[str, dict[str, Any]] = {
    "bulk_lnc_expression": {"required": True, "columns": ["cancer_id", "sample_id", "lncrna_id", "logcpm"]},
    "bulk_gene_expression": {"required": True, "columns": ["cancer_id", "sample_id", "gene_id", "logcpm"]},
    "bulk_pathway_activity": {"required": True, "columns": ["cancer_id", "sample_id", "pathway_id", "activity_score"]},
    "bulk_lnc_pathway": {"required": True, "columns": ["cancer_id", "lncrna_id", "pathway_id", "rho_adjusted", "fdr_global"]},
    "bulk_covariates": {"required": True, "columns": ["cancer_id", "sample_id"]},
    "dim_lncRNA": {"required": True, "columns": ["lncrna_id", "gene_symbol"]},
    "dim_gene": {"required": True, "columns": ["gene_id", "gene_symbol"]},
    "dim_pathway": {"required": True, "columns": ["pathway_id", "pathway_name"]},
    "dim_cancer": {"required": True, "columns": ["cancer_id"]},
    "dim_drug": {"required": True, "columns": ["drug_id", "drug_name"]},
    "pathway_gene_member": {"required": True, "columns": ["pathway_id", "gene_symbol"]},
    "interaction_relation": {"required": True, "columns": ["interaction_id", "lncrna_id", "partner_id", "partner_type", "source_database"]},
    "evidence_event": {"required": True, "columns": ["evidence_event_id", "lncrna_id", "partner_id"]},
    "interaction_pathway_support": {"required": True, "columns": ["lncrna_id", "pathway_id", "support_type"]},
    "lnc_drug_curated": {"required": False, "columns": ["lncrna_id", "drug_id"]},
    "prism_lnc_drug": {"required": False, "columns": ["cancer_id", "lncrna_id", "drug_id", "fdr"]},
    "gdsc_lnc_drug": {"required": False, "columns": ["cancer_id", "lncrna_id", "drug_id", "fdr"]},
    "drug_replication": {"required": False, "columns": ["cancer_id", "lncrna_id", "drug_id", "replication_status"]},
    "lnc2cancer_external": {"required": True, "columns": ["name", "cancer type", "methods", "pubmed id"]},
    "lncrnadisease_external": {"required": True, "columns": ["ncRNA Symbol", "Disease Name", "Validated Method//Prediction Method", "PubMed ID"]},
    "rnadisease_experimental_external": {"required": True, "columns": ["RDID", "RNA Symbol", "Disease Name", "PMID"]},
    "rnadisease_predicted_external": {"required": True, "columns": ["RDID", "RNA_symbol", "disease_name", "method_name"]},
    "gse85011_sample_metadata": {"required": True, "columns": ["gsm", "cell_line", "target_raw", "is_control", "processed_data_file"]},
}


def inspect_input(cfg: dict[str, Any], key: str, contract: dict[str, Any]) -> dict[str, Any]:
    path = input_path(cfg, key)
    row: dict[str, Any] = {
        "input_key": key,
        "required": contract["required"],
        "path": str(path) if path else None,
        "exists": bool(path and path.exists()),
        "status": "MISSING",
        "row_count": None,
        "missing_columns": None,
        "columns": None,
    }
    if not path or not path.exists():
        row["status"] = "FAIL" if contract["required"] else "OPTIONAL_MISSING"
        return row
    try:
        if path.is_dir() or path.suffix == ".parquet":
            schema = parquet_schema(path)
            columns = [x["name"] for x in schema]
            rows = parquet_row_count(path)
        else:
            head = read_table(path).head(5)
            columns = list(head.columns)
            rows = None
        missing = sorted(set(contract["columns"]) - set(columns))
        row.update({
            "row_count": rows,
            "columns": ";".join(columns),
            "missing_columns": ";".join(missing),
            "status": "PASS" if not missing else ("FAIL" if contract["required"] else "OPTIONAL_SCHEMA_MISMATCH"),
        })
    except Exception as exc:
        row.update({"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"})
    return row


def run_audit(cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = [inspect_input(cfg, key, spec) for key, spec in INPUT_CONTRACT.items()]

    sc_paths, sc_source = discover_preferred(cfg, "sc_lnc_pathway_formal", "sc_lnc_pathway_staging_glob")
    pt_paths, pt_source = discover_preferred(cfg, "sc_pseudotime_pathway_formal", "sc_pseudotime_pathway_staging_glob")
    ucell_pattern = str(resolve_input_path(cfg, cfg["inputs"]["ucell_aggregate_glob"]))
    ucell_paths = [Path(x) for x in glob.glob(ucell_pattern)]
    rows.extend([
        {
            "input_key": "sc_lnc_pathway",
            "required": False,
            "path": ";".join(str(x) for x in sc_paths),
            "exists": bool(sc_paths),
            "status": "PASS" if sc_paths else "OPTIONAL_MISSING",
            "row_count": None,
            "missing_columns": "",
            "columns": "",
            "source_tier": sc_source,
            "cancer_count": len({x.parent.name for x in sc_paths}) if sc_source == "staging" else None,
        },
        {
            "input_key": "sc_pseudotime_pathway",
            "required": False,
            "path": ";".join(str(x) for x in pt_paths),
            "exists": bool(pt_paths),
            "status": "PASS" if pt_paths else "OPTIONAL_MISSING",
            "row_count": None,
            "missing_columns": "",
            "columns": "",
            "source_tier": pt_source,
            "cancer_count": len({x.parent.name for x in pt_paths}) if pt_source == "staging" else None,
        },
        {
            "input_key": "ucell_aggregate",
            "required": False,
            "path": ucell_pattern,
            "exists": bool(ucell_paths),
            "status": "PASS" if ucell_paths else "OPTIONAL_MISSING",
            "row_count": None,
            "missing_columns": "",
            "columns": "",
            "source_tier": "staging",
            "cancer_count": len({x.parent.name for x in ucell_paths}),
        },
    ])
    frame = pd.DataFrame(rows)
    hard_failures = frame.loc[(frame["required"] == True) & (~frame["status"].isin(["PASS"]))]
    summary = {
        "generated_at": utc_now(),
        "analysis_version": cfg["analysis_version"],
        "n_inputs": len(frame),
        "n_pass": int((frame.status == "PASS").sum()),
        "n_hard_failures": len(hard_failures),
        "hard_failure_keys": hard_failures.input_key.tolist(),
        "sc_lnc_pathway_source": sc_source,
        "sc_lnc_pathway_files": len(sc_paths),
        "sc_pseudotime_source": pt_source,
        "sc_pseudotime_files": len(pt_paths),
        "ucell_files": len(ucell_paths),
        "ready_for_feature_building": len(hard_failures) == 0,
        "ready_for_full_33_cancer_release": sc_source == "formal" and pt_source == "formal" and len(ucell_paths) >= 33,
    }
    return frame, summary
