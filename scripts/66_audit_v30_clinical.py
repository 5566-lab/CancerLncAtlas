#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cc_hhgt.common import load_config, write_json, write_table


def check(condition: bool, name: str, details: str = "") -> dict:
    return {"check": name, "status": "PASS" if condition else "FAIL", "details": details}


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail-closed V3.0 clinical audit")
    parser.add_argument("--config", default="config/model_v3_0_clinical.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    result = cfg["_results"]
    checks = []
    preflight = json.loads((result / "reports" / "V3_0_CLINICAL_PREFLIGHT.json").read_text(encoding="utf-8"))
    checks.append(check(preflight.get("status") == "PASS", "v30_preflight_pass"))
    endpoints = pd.read_parquet(result / "clinical_standardized" / "clinical_endpoints.parquet")
    checks.append(check((pd.to_numeric(endpoints.loc[endpoints.endpoint_available.eq(1), "time_days"], errors="coerce") > 0).all(), "positive_survival_times"))
    checks.append(check(set(pd.to_numeric(endpoints.loc[endpoints.endpoint_available.eq(1), "event"], errors="coerce").dropna().unique()).issubset({0, 1}), "binary_event_indicators"))
    required = cfg["clinical"].get("required_endpoints", ["OS"])
    for endpoint in required:
        n = int(endpoints.loc[(endpoints.endpoint.eq(endpoint)) & endpoints.endpoint_available.eq(1)].shape[0])
        checks.append(check(n > 0, f"required_endpoint_{endpoint}", f"n={n}"))
    availability = pd.read_csv(result / "clinical_standardized" / "clinical_endpoint_availability.tsv", sep="\t")
    semantic_bad = availability.loc[
        (availability.endpoint.eq("PFS") & availability.time_column.astype(str).str.contains("PFI", case=False, na=False))
        | (availability.endpoint.eq("DFS") & availability.time_column.astype(str).str.contains("DFI", case=False, na=False))
    ]
    checks.append(check(semantic_bad.empty, "pfi_dfi_not_renamed_to_pfs_dfs", semantic_bad.to_dict(orient="records")))

    patient_pred = pd.read_parquet(result / "clinical_survival_release" / "patient_survival_oof_prediction.parquet")
    checks.append(check(patient_pred.patient_id.notna().all(), "patient_oof_has_patient_ids"))
    checks.append(check(patient_pred.endpoint_available.isin([0, 1]).all(), "patient_oof_endpoint_masks"))
    checks.append(check(patient_pred.loc[patient_pred.endpoint_available.eq(1), "risk_score"].notna().all(), "patient_oof_risk_available"))
    checks.append(check(patient_pred.groupby(["cancer_id", "patient_fold_id", "seed", "patient_id", "endpoint"], observed=True).size().max() == 1, "patient_oof_unique_keys"))

    association = pd.read_parquet(result / "clinical_association_release" / "clinical_association_summary.parquet")
    checks.append(check(len(association) > 0, "candidate_clinical_associations_exist"))
    checks.append(check(set(association.subject_type.unique()).issubset({"lncRNA", "lncRNA_pathway", "lncRNA_state"}), "clinical_subject_types_valid"))
    checks.append(check(association.fdr.dropna().between(0, 1).all(), "clinical_fdr_valid"))

    clinical_oof = pd.read_parquet(result / "clinical_expert_release" / "clinical_replication_oof_prediction.parquet")
    checks.append(check(clinical_oof.clinical_survival_probability.between(0, 1).all(), "clinical_expert_probability_valid"))
    checks.append(check(clinical_oof.groupby(["cancer_id", "patient_fold_id", "endpoint", "subject_type", "subject_id", "modifier_id"], dropna=False, observed=True).size().max() == 1, "clinical_expert_oof_unique"))

    final = pd.read_parquet(result / "clinical_expert_release" / "v3_0_functional_clinical_table.parquet")
    v29 = cfg["_root"] / cfg["clinical"]["v29_result_root"] / "v2_9_downstream" / "v2_9_release" / "final_expert_fusion_table.parquet"
    base = pd.read_parquet(v29, columns=["cancer_id", "lncrna_id", "pathway_family_id", "discovery_ranking_probability"])
    compare = final[["cancer_id", "lncrna_id", "pathway_family_id", "discovery_ranking_probability"]].merge(base, on=["cancer_id", "lncrna_id", "pathway_family_id"], suffixes=("_v30", "_v29"), validate="one_to_one")
    delta = np.nanmax(np.abs(compare.discovery_ranking_probability_v30 - compare.discovery_ranking_probability_v29))
    checks.append(check(delta <= 1e-12, "survival_does_not_modify_discovery", f"max_delta={delta}"))
    checks.append(check(final.survival_used_in_discovery.eq(False).all(), "survival_flag_false_in_discovery"))
    checks.append(check(final.translational_priority_score.dropna().between(0, 1).all(), "translational_priority_valid"))

    # Checkpoint contracts: preprocessing must be train-only and survival is not a discovery input.
    checkpoint_paths = sorted((result / "models" / "clinical_survival").rglob("best.pt"))
    checks.append(check(len(checkpoint_paths) > 0, "clinical_survival_checkpoints_exist", f"n={len(checkpoint_paths)}"))
    bad = []
    for path in checkpoint_paths:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if ckpt.get("preprocessing_fit_scope") != "train_patients_only" or ckpt.get("survival_endpoints_used_in_discovery") is not False:
            bad.append(str(path))
    checks.append(check(not bad, "clinical_checkpoint_leakage_contract", f"bad={len(bad)}"))
    replication_paths = sorted((result / "models" / "clinical_replication_expert").rglob("best.pt"))
    replication_bad = []
    for path in replication_paths:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if (
            ckpt.get("crossfit_scope") != "leave_one_cancer_out_meta_expert_on_patient_fold_oof_statistics"
            or int(ckpt.get("target_cancer_rows_used_to_fit", -1)) != 0
        ):
            replication_bad.append(str(path))
    checks.append(check(bool(replication_paths) and not replication_bad, "clinical_replication_nested_oof_contract", f"n={len(replication_paths)}, bad={len(replication_bad)}"))

    audit = pd.DataFrame(checks)
    write_table(audit, result / "reports" / "V3_0_CLINICAL_AUDIT.tsv")
    status = "PASS" if audit.status.eq("PASS").all() else "FAIL"
    payload = {
        "status": status,
        "version": "CC-HHGT_v3.0-clinical",
        "checks": audit.to_dict(orient="records"),
        "failed_checks": audit.loc[audit.status.eq("FAIL"), "check"].tolist(),
        "strict_graph_retrained": False,
        "v29_discovery_probability_preserved": True,
        "clinical_endpoints": sorted(patient_pred.endpoint.unique()),
    }
    write_json(payload, result / "reports" / "V3_0_FINAL_AUDIT.json")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if status != "PASS":
        raise RuntimeError("V3.0 clinical audit failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
