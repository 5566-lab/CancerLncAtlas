#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cc_hhgt.clinical_v30 import load_all_existing, tcga_patient_id
from cc_hhgt.common import input_candidates, input_path, load_config, read_table, write_json, write_table


def first_col(df: pd.DataFrame, aliases: list[str], required: bool = True) -> str | None:
    lower = {str(c).lower(): c for c in df.columns}
    for alias in aliases:
        if alias.lower() in lower:
            return lower[alias.lower()]
    if required:
        raise ValueError(f"Missing aliases {aliases}; observed={list(df.columns)}")
    return None


def standardize_fold_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    cancer = first_col(frame, ["cancer_id", "cancer"])
    fold = first_col(frame, ["fold_id", "patient_fold_id"])
    split = first_col(frame, ["split", "set"])
    patient = first_col(frame, ["patient_id", "case_id"], required=False)
    sample = first_col(frame, ["sample_id", "sample"], required=False)
    if patient is None and sample is None:
        raise ValueError("Fold manifest has neither patient_id nor sample_id")
    out = pd.DataFrame({
        "cancer_id": frame[cancer].astype(str).str.replace("TCGA-", "", regex=False),
        "patient_fold_id": frame[fold].astype(str),
        "split": frame[split].astype(str).str.lower(),
        "patient_id": frame[patient].map(tcga_patient_id) if patient else frame[sample].map(tcga_patient_id),
        "sample_id": frame[sample].astype(str) if sample else pd.NA,
    })
    return out.dropna(subset=["patient_id"]).drop_duplicates()


def select_candidates(v29_final: pd.DataFrame, v29_state: pd.DataFrame, settings: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prob_col = settings.get("candidate_probability_column", "fused_confidence_probability")
    if prob_col not in v29_final:
        prob_col = "discovery_ranking_probability"
    use = v29_final[["cancer_id", "lncrna_id", "pathway_family_id", prob_col]].copy()
    use[prob_col] = pd.to_numeric(use[prob_col], errors="coerce")
    use = use.loc[use[prob_col].ge(float(settings.get("candidate_min_probability", 0.50)))]
    use = use.sort_values(["cancer_id", prob_col], ascending=[True, False])
    pair = use.groupby("cancer_id", observed=True).head(int(settings.get("max_pair_candidates_per_cancer", 256))).copy()
    pair = pair.rename(columns={prob_col: "functional_probability"})

    lnc = (
        use.groupby(["cancer_id", "lncrna_id"], observed=True)[prob_col]
        .max().rename("functional_probability").reset_index()
        .sort_values(["cancer_id", "functional_probability"], ascending=[True, False])
        .groupby("cancer_id", observed=True).head(int(settings.get("max_lncRNAs_per_cancer", 512)))
    )
    pathway = (
        use.groupby(["cancer_id", "pathway_family_id"], observed=True)[prob_col]
        .max().rename("functional_probability").reset_index()
        .sort_values(["cancer_id", "functional_probability"], ascending=[True, False])
        .groupby("cancer_id", observed=True).head(int(settings.get("max_pathways_per_cancer", 96)))
    )
    state = v29_state.copy()
    if not state.empty:
        score_col = "lncrna_state_probability" if "lncrna_state_probability" in state else "state_patient_probability"
        state[score_col] = pd.to_numeric(state[score_col], errors="coerce")
        state = state.sort_values(["cancer_id", score_col], ascending=[True, False]).groupby("cancer_id", observed=True).head(int(settings.get("max_state_candidates_per_cancer", 128)))
    return lnc.reset_index(drop=True), pathway.reset_index(drop=True), pair.reset_index(drop=True), state.reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build V3.0 patient-level clinical feature matrices and candidate manifests")
    parser.add_argument("--config", default="config/model_v3_0_clinical.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    settings = cfg["clinical"]
    v29 = cfg["_root"] / settings["v29_result_root"]
    v29_final = read_table(v29 / "v2_9_downstream" / "v2_9_release" / "final_expert_fusion_table.parquet")
    v29_state_path = v29 / "v2_9_downstream" / "lncrna_state_release" / "lncrna_state_final.parquet"
    v29_state = read_table(v29_state_path) if v29_state_path.exists() else pd.DataFrame()
    lnc_candidates, pathway_candidates, pair_candidates, state_candidates = select_candidates(v29_final, v29_state, settings)

    root = cfg["_results"] / "clinical_data"
    write_table(lnc_candidates, root / "clinical_lncRNA_candidates.parquet")
    write_table(pathway_candidates, root / "clinical_pathway_candidates.parquet")
    write_table(pair_candidates, root / "clinical_pair_candidates.parquet")
    write_table(state_candidates, root / "clinical_state_candidates.parquet")

    endpoints = read_table(cfg["_results"] / "clinical_standardized" / "clinical_endpoints.parquet")
    clinical_cov = read_table(cfg["_results"] / "clinical_standardized" / "clinical_covariates.parquet")
    folds = standardize_fold_manifest(load_all_existing(input_candidates(cfg, "patient_fold_manifest")))

    expr = read_table(input_path(cfg, "bulk_lnc_expression"))
    expr_value = first_col(expr, ["logcpm", "expression", "value", "tpm"])
    expr["patient_id"] = expr["sample_id"].map(tcga_patient_id)
    expr["cancer_id"] = expr.cancer_id.astype(str).str.replace("TCGA-", "", regex=False)

    pathway = read_table(input_path(cfg, "bulk_pathway_activity"))
    pathway_value = first_col(pathway, ["activity_score", "score", "value"])
    pathway["patient_id"] = pathway["sample_id"].map(tcga_patient_id)
    pathway["cancer_id"] = pathway.cancer_id.astype(str).str.replace("TCGA-", "", regex=False)
    member = read_table(v29 / "tables" / "pathway_family_member.parquet")
    member["membership_weight"] = pd.to_numeric(member.get("membership_weight", 1.0), errors="coerce").fillna(1.0)
    pathway = pathway.merge(member[["pathway_id", "pathway_family_id", "membership_weight"]], on="pathway_id", how="inner")
    pathway["weighted"] = pd.to_numeric(pathway[pathway_value], errors="coerce") * pathway.membership_weight
    family = pathway.groupby(["cancer_id", "patient_id", "pathway_family_id"], observed=True).agg(weighted=("weighted", "sum"), weight=("membership_weight", "sum")).reset_index()
    family["activity"] = family.weighted / family.weight.clip(lower=1e-8)

    state_source = load_all_existing(input_candidates(cfg, "tumor_state"))
    state_sample = first_col(state_source, ["sample_id", "sample"], required=False)
    if state_sample:
        state_source["patient_id"] = state_source[state_sample].map(tcga_patient_id)
    patient_col = first_col(state_source, ["patient_id", "case_id"], required=False)
    if "patient_id" not in state_source and patient_col:
        state_source["patient_id"] = state_source[patient_col].map(tcga_patient_id)
    cancer_col = first_col(state_source, ["cancer_id", "cancer", "cancer_type"], required=False)
    if cancer_col:
        state_source["cancer_id"] = state_source[cancer_col].astype(str).str.replace("TCGA-", "", regex=False)
    state_targets = cfg.get("state_graph", {}).get("required_states", [])
    state_cols = [c for c in state_targets if c in state_source]
    state_wide = state_source[["cancer_id", "patient_id", *state_cols]].drop_duplicates(["cancer_id", "patient_id"]) if state_cols else pd.DataFrame(columns=["cancer_id", "patient_id"])

    summaries = []
    cancers = sorted(set(lnc_candidates.cancer_id.astype(str)) & set(endpoints.cancer_id.astype(str)))
    for cancer in cancers:
        lnc_ids = lnc_candidates.loc[lnc_candidates.cancer_id.eq(cancer), "lncrna_id"].astype(str).tolist()
        pf_ids = pathway_candidates.loc[pathway_candidates.cancer_id.eq(cancer), "pathway_family_id"].astype(str).tolist()
        e = expr.loc[(expr.cancer_id.eq(cancer)) & expr.lncrna_id.astype(str).isin(lnc_ids), ["patient_id", "lncrna_id", expr_value]].copy()
        e[expr_value] = pd.to_numeric(e[expr_value], errors="coerce")
        e_wide = e.pivot_table(index="patient_id", columns="lncrna_id", values=expr_value, aggfunc="mean").add_prefix("lnc::").reset_index()
        p = family.loc[(family.cancer_id.eq(cancer)) & family.pathway_family_id.astype(str).isin(pf_ids), ["patient_id", "pathway_family_id", "activity"]]
        p_wide = p.pivot_table(index="patient_id", columns="pathway_family_id", values="activity", aggfunc="mean").add_prefix("pf::").reset_index()
        base = clinical_cov.loc[clinical_cov.cancer_id.eq(cancer)].copy()
        base = base.merge(e_wide, on="patient_id", how="left").merge(p_wide, on="patient_id", how="left")
        if not state_wide.empty:
            base = base.merge(state_wide.loc[state_wide.cancer_id.eq(cancer)].drop(columns="cancer_id"), on="patient_id", how="left")
        base = base.merge(folds.loc[folds.cancer_id.eq(cancer)], on=["cancer_id", "patient_id"], how="inner")
        endpoint_wide = endpoints.loc[endpoints.cancer_id.eq(cancer)].pivot(index="patient_id", columns="endpoint", values=["time_days", "event", "endpoint_available"])
        endpoint_wide.columns = [f"{a}::{b}" for a, b in endpoint_wide.columns]
        base = base.merge(endpoint_wide.reset_index(), on="patient_id", how="left")
        out = root / "patients" / f"cancer_id={cancer}" / "part-0.parquet"
        write_table(base, out)
        summaries.append({
            "cancer_id": cancer,
            "n_patients": int(base.patient_id.nunique()),
            "n_lnc_features": int(sum(c.startswith("lnc::") for c in base.columns)),
            "n_pathway_features": int(sum(c.startswith("pf::") for c in base.columns)),
            "n_state_features": int(sum(c in state_cols for c in base.columns)),
            "n_folds": int(base.patient_fold_id.nunique()),
        })
    summary = pd.DataFrame(summaries)
    write_table(summary, root / "clinical_patient_feature_summary.tsv")
    payload = {
        "status": "COMPLETED",
        "cancers": int(len(summary)),
        "lncRNA_candidates": int(len(lnc_candidates)),
        "pathway_candidates": int(len(pathway_candidates)),
        "pair_candidates": int(len(pair_candidates)),
        "state_candidates": int(len(state_candidates)),
        "patient_rows": int(summary.n_patients.sum()) if len(summary) else 0,
    }
    write_json(payload, root / "SUCCESS.json")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
