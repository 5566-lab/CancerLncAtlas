#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cc_hhgt.common import load_config, read_table, write_json, write_table

MODELS = ["rgcn", "hgt", "cc_hhgt_strict"]
SEEDS = [20260726, 20261726, 20262726]
REQUIRED_STATES = [
    "EXTEND::published_score",
    "stemness_rna::RNAss",
    "stemness_dna::DNAss",
    "stemness_rna::EREG.EXPss",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail-closed V2.9 final audit")
    parser.add_argument("--config", default="config/model_v2_9_state_graph.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    checks = []

    def add(name: str, ok: bool, detail: object, required: bool = True) -> None:
        checks.append({"requirement": name, "status": "PASS" if ok else "FAIL", "required": required, "detail": detail})

    tables = cfg["_results"] / "tables"
    nodes_path = tables / "graph_node.parquet"
    edges_path = tables / "graph_edge.parquet"
    add("graph_node_exists", nodes_path.exists(), str(nodes_path))
    add("graph_edge_exists", edges_path.exists(), str(edges_path))
    if nodes_path.exists() and edges_path.exists():
        nodes = read_table(nodes_path)
        edges = read_table(edges_path)
        state_nodes = set(nodes.loc[nodes.node_type.astype(str).eq("state"), "canonical_id"].astype(str))
        state_edges = edges.loc[edges.source_type.astype(str).eq("state") | edges.target_type.astype(str).eq("state")]
        add("required_state_nodes", set(REQUIRED_STATES).issubset(state_nodes), {"observed": sorted(state_nodes), "missing": sorted(set(REQUIRED_STATES)-state_nodes)})
        add("state_edges_nonzero", len(state_edges) > 0, {"rows": len(state_edges), "relations": state_edges.relation_type.astype(str).value_counts().to_dict()})
        relation_counts = state_edges.relation_type.astype(str).value_counts().to_dict()
        add("state_relation_diversity", state_edges.relation_type.astype(str).nunique() >= 4, relation_counts)
        relation_set = set(state_edges.relation_type.astype(str))
        add(
            "signed_state_relations",
            {"positively_associated_with_state", "negatively_associated_with_state"}.issubset(relation_set),
            relation_counts,
        )
        direct_target = edges.loc[
            edges.source_type.astype(str).eq("lncRNA")
            & edges.target_type.astype(str).eq("state")
        ]
        add(
            "no_direct_lncrna_state_target_edges",
            direct_target.empty,
            {"n_direct_target_edges": int(len(direct_target))},
        )

        folds = read_table(tables / "fold_manifest.tsv")
        leakage = []
        context = edges.is_context_specific.fillna(False).astype(bool)
        for row in folds.itertuples(index=False):
            excluded = {str(row.test_cancer), str(row.validation_cancer)}
            excluded.update(map(str, cfg.get("analysis_cancers", {}).get("reference_only", [])))
            excluded.update(map(str, cfg.get("analysis_cancers", {}).get("exclude_from_training", [])))
            remaining = edges.loc[~(context & edges.cancer_id.astype(str).isin(excluded))]
            bad = int((remaining.is_context_specific.fillna(False).astype(bool) & remaining.cancer_id.astype(str).isin(excluded)).sum())
            leakage.append({"fold_id": row.fold_id, "violations": bad})
        add("fold_context_edge_masking", all(x["violations"] == 0 for x in leakage), leakage[:10])
    else:
        folds = pd.DataFrame()

    expected_folds = int(len(folds)) if not folds.empty else 0
    expected_tasks = expected_folds * len(MODELS) * len(SEEDS)
    task_success = []
    missing_tasks = []
    for model in MODELS:
        for fold in ([] if folds.empty else folds.fold_id.astype(str)):
            for seed in SEEDS:
                root = cfg["_results"] / "v2_9_strict" / model / fold / f"seed_{seed}"
                ok = all((root / name).exists() for name in ["SUCCESS.json", "CALIBRATION_SUCCESS.json", "best.pt", "prediction_pathway_calibrated.parquet", "prediction_state_calibrated.parquet", "embeddings/state.parquet"])
                task_success.append(ok)
                if not ok:
                    missing_tasks.append(str(root))
                if (root / "embeddings/state.parquet").exists():
                    emb = pd.read_parquet(root / "embeddings/state.parquet", columns=["canonical_id"])
                    if not set(REQUIRED_STATES).issubset(set(emb.canonical_id.astype(str))):
                        missing_tasks.append(str(root / "embeddings/state.parquet") + ":missing_required_states")
                if model == "cc_hhgt_strict" and (root / "best.pt").exists():
                    checkpoint = torch.load(root / "best.pt", map_location="cpu", weights_only=False)
                    policy = checkpoint.get("strict_pair_evidence_policy")
                    mask_probability = float(checkpoint.get("strict_pair_evidence_train_row_mask_probability", -1.0))
                    if policy != "validation_and_test_fully_masked" or not (0.0 < mask_probability <= 1.0):
                        missing_tasks.append(
                            str(root / "best.pt")
                            + f":invalid_strict_pair_policy={policy},mask_probability={mask_probability}"
                        )
    add("strict_task_count", len(task_success) == expected_tasks, {"expected": expected_tasks, "observed": len(task_success)})
    add("strict_all_tasks_complete", bool(task_success) and all(task_success), {"missing_count": len(missing_tasks), "examples": missing_tasks[:20]})

    strict_release = cfg["_results"] / "strict_release"
    add("strict_pathway_oof", (strict_release / "strict_cross_cancer_oof_prediction.parquet").exists(), str(strict_release))
    add("strict_state_oof", (strict_release / "strict_state_oof_prediction.parquet").exists(), str(strict_release))
    if (strict_release / "strict_state_oof_prediction.parquet").exists():
        state_oof = read_table(strict_release / "strict_state_oof_prediction.parquet")
        add("strict_state_all_models", all(f"{m}_probability" in state_oof for m in MODELS), list(state_oof.columns))
        add("strict_state_required_states", set(REQUIRED_STATES).issubset(set(state_oof.state_id.astype(str))), sorted(state_oof.state_id.astype(str).unique()))

    downstream = cfg["_results"] / "v2_9_downstream"
    adapter_root = downstream / "adapter_data"
    adapter_files = sorted(adapter_root.glob("cancer_id=*/part-0.parquet"))
    add("adapter_data_31_cancers", len(adapter_files) >= 31, len(adapter_files))
    required_adapter_cols = {
        "state_mean__EXTEND::published_score",
        "state_available__EXTEND::published_score",
        "state_mean__stemness_dna::DNAss",
        "state_available__stemness_dna::DNAss",
    }
    adapter_missing = []
    for path in adapter_files:
        columns = set(pd.read_parquet(path).columns)
        missing = required_adapter_cols - columns
        if missing:
            adapter_missing.append({"path": str(path), "missing": sorted(missing)})
    add("adapter_extend_dnass_columns", not adapter_missing, adapter_missing[:10])

    checkpoints = sorted((downstream / "models" / "cancer_adapter").glob("*/*/seed_*/best.pt"))
    used_columns = set()
    for path in checkpoints[:50]:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        used_columns.update(map(str, payload.get("patient_feature_columns", [])))
    add("adapter_checkpoint_uses_extend", "state_mean__EXTEND::published_score" in used_columns, sorted(x for x in used_columns if "EXTEND" in x))
    add("adapter_checkpoint_uses_dnass", "state_mean__stemness_dna::DNAss" in used_columns, sorted(x for x in used_columns if "DNAss" in x))

    full_release = downstream / "v2_9_release"
    full_final = full_release / "final_expert_fusion_table.parquet"
    full_audit = full_release / "FULL_MODEL_REQUIREMENTS_AUDIT.json"
    add("full_expert_fusion_table", full_final.exists(), str(full_final))
    add("full_model_requirements_audit", full_audit.exists(), str(full_audit))
    if full_audit.exists():
        full_payload = json.loads(full_audit.read_text(encoding="utf-8"))
        add(
            "full_model_requirements_audit_pass",
            full_payload.get("status") == "PASS",
            full_payload.get("required_failures", []),
        )
    if full_final.exists():
        full_columns = set(pd.read_parquet(full_final).columns)
        add(
            "full_expert_probability_columns",
            {
                "discovery_ranking_probability",
                "fused_confidence_probability",
                "graph_ensemble_probability",
                "cancer_native_probability",
                "indirect_mechanism_probability",
                "direct_evidence_probability",
            }.issubset(full_columns),
            sorted(full_columns),
        )

    state_release = downstream / "lncrna_state_release"
    audit_path = state_release / "LNCRNA_STATE_AUDIT.json"
    add("lncrna_state_audit", audit_path.exists(), str(audit_path))
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        add("lncrna_state_audit_pass", audit.get("status") == "PASS", audit.get("required_failures", []))
    for name in [
        "lncrna_state_final.parquet",
        "lncrna_state_significance_report.parquet",
        "rnass_state_significance_report.parquet",
        "pancancer_lncrna_rnass_association.parquet",
    ]:
        add(name, (state_release / name).exists(), str(state_release / name))
    if (state_release / "lncrna_state_fold_prediction.parquet").exists():
        fold = read_table(state_release / "lncrna_state_fold_prediction.parquet")
        graph_cols = ["state_rgcn_probability", "state_hgt_probability", "state_cc_hhgt_strict_probability"]
        coverage = {c: int(fold[c].notna().sum()) if c in fold else 0 for c in graph_cols}
        add("state_graph_expert_coverage", all(v > 0 for v in coverage.values()), coverage)

    state_data_success = downstream / "lncrna_state_data" / "SUCCESS.json"
    add("state_data_contract", state_data_success.exists(), str(state_data_success))
    if state_data_success.exists():
        contract = json.loads(state_data_success.read_text(encoding="utf-8"))
        add(
            "state_fdr_scope_complete_detectable_universe",
            "all detectable lncRNAs" in str(contract.get("multiple_testing_scope", "")),
            contract.get("multiple_testing_scope"),
        )

    web_root = cfg["_results"] / "web_tables"
    web_required = [
        "web_cancer_overview.parquet",
        "web_cancer_lncRNA_pathway_ranking.parquet",
        "web_cancer_lncRNA_ranking.parquet",
        "web_cancer_pathway_lncRNA_ranking.parquet",
        "web_cancer_pathway_ranking.parquet",
        "web_lncRNA_pan_cancer_ranking.parquet",
        "web_three_probability.parquet",
        "web_model_version.parquet",
        "web_lncRNA_state_ranking.parquet",
        "web_state_lncRNA_ranking.parquet",
        "web_lncRNA_state_matrix.parquet",
        "web_cancer_state_lncRNA_summary.parquet",
        "web_cancer_state_summary.parquet",
        "web_cancer_celltype_summary.parquet",
        "web_cancer_drug_summary.parquet",
        "web_cancer_geneset_summary.parquet",
        "web_pancancer_rnass.parquet",
        "V2_9_WEB_TABLES_SUCCESS.json",
    ]
    web_missing = [name for name in web_required if not (web_root / name).exists()]
    add("state_web_tables", not web_missing, web_missing)

    failures = [row for row in checks if row["required"] and row["status"] != "PASS"]
    payload = {
        "status": "PASS" if not failures else "FAIL",
        "analysis_version": cfg["analysis_version"],
        "required_failures": failures,
        "checks": checks,
        "claims": {
            "state_nodes_in_strict_graph": not any(x["requirement"] == "required_state_nodes" and x["status"] != "PASS" for x in checks),
            "three_strict_models_retrained": bool(task_success) and all(task_success),
            "cancer_specific_lncrna_rnass": (state_release / "rnass_state_significance_report.parquet").exists(),
            "pancancer_lncrna_rnass": (state_release / "pancancer_lncrna_rnass_association.parquet").exists(),
        },
    }
    report_root = cfg["_results"] / "reports"
    report_root.mkdir(parents=True, exist_ok=True)
    write_json(payload, report_root / "V2_9_FINAL_AUDIT.json")
    write_table(pd.DataFrame(checks), report_root / "V2_9_FINAL_AUDIT.tsv")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if failures:
        raise RuntimeError("V2.9 final audit failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
