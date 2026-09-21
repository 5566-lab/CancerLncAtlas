#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import pandas as pd

from cc_hhgt.common import load_config, write_json, write_table


def _patient_endpoint_summary(patient: pd.DataFrame) -> pd.DataFrame:
    """Summarize one event per patient and endpoint, never per OOF row."""

    required = {
        "cancer_id", "endpoint", "patient_id", "endpoint_available",
        "event", "risk_score", "fold_c_index",
    }
    missing = sorted(required - set(patient.columns))
    if missing:
        raise ValueError(f"Patient OOF table missing columns: {missing}")
    eligible = patient.loc[patient["endpoint_available"].eq(1)].copy()
    eligible["event"] = (
        pd.to_numeric(eligible["event"], errors="coerce")
        .fillna(0)
        .gt(0)
        .astype("int8")
    )
    for column in ("risk_score", "fold_c_index"):
        eligible[column] = pd.to_numeric(eligible[column], errors="coerce")
    patient_level = (
        eligible.groupby(["cancer_id", "endpoint", "patient_id"], observed=True)
        .agg(
            event=("event", "max"),
            risk_score=("risk_score", "mean"),
            fold_c_index=("fold_c_index", "mean"),
        )
        .reset_index()
    )
    summary = (
        patient_level.groupby(["cancer_id", "endpoint"], observed=True)
        .agg(
            n_patients=("patient_id", "nunique"),
            n_events=("event", "sum"),
            mean_risk=("risk_score", "mean"),
            c_index=("fold_c_index", "mean"),
        )
        .reset_index()
    )
    if (summary["n_events"] > summary["n_patients"]).any():
        raise RuntimeError("Clinical endpoint summary has more events than patients")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize V3.0 clinical web tables")
    parser.add_argument("--config", default="config/model_v3_0_clinical.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    result = cfg["_results"]
    web = result / "web_tables"
    metrics = pd.read_csv(result / "clinical_survival_release" / "survival_oof_metrics.tsv", sep="\t")
    write_table(metrics, web / "web_clinical_endpoint_summary.parquet")
    association = pd.read_parquet(result / "clinical_association_release" / "clinical_association_summary.parquet")
    write_table(association.loc[association.subject_type.eq("lncRNA")], web / "web_lncRNA_survival_association.parquet")
    write_table(association.loc[association.subject_type.eq("lncRNA_pathway")], web / "web_lncRNA_pathway_clinical_association.parquet")
    write_table(association.loc[association.subject_type.eq("lncRNA_state")], web / "web_lncRNA_state_clinical_association.parquet")
    patient = pd.read_parquet(result / "clinical_survival_release" / "patient_survival_oof_prediction.parquet")
    patient_summary = _patient_endpoint_summary(patient)
    write_table(patient_summary, web / "web_patient_risk_oof_summary.parquet")
    final = pd.read_parquet(result / "clinical_expert_release" / "v3_0_functional_clinical_table.parquet")
    priority_cols = [c for c in ["cancer_id", "lncrna_id", "pathway_family_id", "discovery_ranking_probability", "fused_confidence_probability", "clinical_relevance_score", "clinical_best_endpoint", "clinical_endpoint_count", "translational_priority_score"] if c in final]
    write_table(final[priority_cols], web / "web_translational_priority.parquet")
    relevance = pd.read_parquet(result / "clinical_expert_release" / "clinical_endpoint_relevance.parquet")
    write_table(relevance, web / "web_clinical_endpoint_relevance.parquet")
    payload = {"status": "COMPLETED", "tables": sorted(path.name for path in web.glob("web_*clinical*.parquet")) + ["web_translational_priority.parquet", "web_patient_risk_oof_summary.parquet"]}
    write_json(payload, web / "V3_0_CLINICAL_WEB_SUCCESS.json")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
