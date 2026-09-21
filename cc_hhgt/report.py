from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Template

from .common import read_table, utc_now, write_json, write_table
from .external_validation import external_report_markdown

REPORT_TEMPLATE = Template("""# CancerLncAtlas CC-HHGT model report

- Generated: {{ generated_at }}
- Analysis version: `{{ version }}`
- Primary cancers: {{ primary_cancers }}
- Reference-only cancers: {{ reference_cancers }}
- Pathway families: {{ n_families }}
- Candidate pairs: {{ n_candidates }}
- Materialized Gene Sets: {{ n_genesets }}

## Input readiness

{{ audit_table }}

## Model evaluation

The labels are weak-supervision/positive-unlabeled proxy labels derived from the evidence layer. AUROC/AUPRC are therefore proxy evaluation metrics, not performance against experimentally complete negatives.

{{ metrics_table }}

## Gene Set output

{{ geneset_table }}

## Important interpretation limits

1. All 33 cancers, including HNSC and LGG, are formal LOCO cohorts.
2. Missing single-cell evidence is represented as unavailable, not as negative evidence.
3. UCell contributes cell-state direction support; patient-level true ssGSEA association remains the main single-cell relation evidence.
4. The default GMT excludes `predicted_candidate` members; an extended GMT is exported separately.
""")


def export_report(cfg: dict[str, Any]) -> Path:
    tables = cfg["_results"] / "tables"
    audit_path = tables / "model_input_audit.tsv"
    audit = read_table(audit_path) if audit_path.exists() else pd.DataFrame()
    fold_path = tables / "fold_manifest.tsv"
    folds = read_table(fold_path) if fold_path.exists() else pd.DataFrame()
    family_path = tables / "pathway_family.parquet"
    families = read_table(family_path) if family_path.exists() else pd.DataFrame()
    cand_path = tables / "candidate_universe_manifest.tsv"
    candidates = read_table(cand_path) if cand_path.exists() else pd.DataFrame()
    gs_path = tables / "geneset_master_model.parquet"
    genesets = read_table(gs_path) if gs_path.exists() else pd.DataFrame()
    metrics_files = list((cfg["_results"] / "models").glob("*/*/metrics_calibrated.tsv")) if (cfg["_results"] / "models").exists() else []
    metrics = pd.concat([read_table(x) for x in metrics_files], ignore_index=True) if metrics_files else pd.DataFrame()
    if not metrics.empty:
        metric_summary = metrics.loc[metrics.split == "test"].groupby("model_name", as_index=False).agg(
            n_folds=("fold_id", "nunique"), auprc_mean=("auprc", "mean"), auprc_sd=("auprc", "std"), auroc_mean=("auroc", "mean"), brier_mean=("brier", "mean"), ece_mean=("ece", "mean")
        )
    else:
        metric_summary = pd.DataFrame()
    reference = cfg["analysis_cancers"].get("reference_only", [])
    primary_n = len(folds) if not folds.empty else 33 - len(reference)
    text = REPORT_TEMPLATE.render(
        generated_at=utc_now(),
        version=cfg["analysis_version"],
        primary_cancers=primary_n,
        reference_cancers=", ".join(reference),
        n_families=len(families),
        n_candidates=int(candidates.n_candidates.sum()) if not candidates.empty and "n_candidates" in candidates else 0,
        n_genesets=len(genesets),
        audit_table=audit[[c for c in ["input_key", "status", "path", "row_count"] if c in audit]].to_markdown(index=False) if not audit.empty else "Not available.",
        metrics_table=metric_summary.to_markdown(index=False) if not metric_summary.empty else "No model metrics available.",
        geneset_table=genesets.head(50).to_markdown(index=False) if not genesets.empty else "No Gene Sets materialized.",
    )
    text = f"{text.rstrip()}\n\n{external_report_markdown(cfg)}\n"
    report_path = cfg["_results"] / "reports" / "CC_HHGT_model_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text, encoding="utf-8")
    write_table(metric_summary, cfg["_results"] / "reports" / "model_metric_summary.tsv")
    write_json({"generated_at": utc_now(), "analysis_version": cfg["analysis_version"], "n_families": len(families), "n_genesets": len(genesets)}, cfg["_results"] / "reports" / "model_report_summary.json")
    return report_path
