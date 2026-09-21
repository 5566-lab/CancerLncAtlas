#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import duckdb

from cc_hhgt.common import load_config, read_table
from cc_hhgt.gnn import filter_fold_edges, fold_node_features
from cc_hhgt.pathway_target import (
    EXACT_PATHWAY_TARGET,
    pathway_target_level,
    require_exact_pathway_contract,
)
from cc_hhgt.v30_integrity import atomic_write_bytes, atomic_write_json, file_sha256, merkle_sha256, verify_file_manifest


MODELS = ("rgcn", "hgt", "cc_hhgt")
SEEDS = (20260726, 20261726, 20262726)


def _records(path: Path) -> list[dict[str, Any]]:
    frame = pd.read_csv(path, sep="\t")
    required = {"relative_path", "size_bytes", "sha256"}
    if not required.issubset(frame):
        raise RuntimeError(f"Invalid manifest {path}: missing {sorted(required - set(frame))}")
    return frame.to_dict("records")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail-closed V3.0 state formal training gate")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument(
        "--repo-root",
        help="Relocated frozen code root; contents must match CODE_MANIFEST.tsv",
    )
    parser.add_argument("--output-root", required=True, help="Fresh destination for the configured formal task matrix")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    cfg = load_config(
        args.config,
        project_root_override=run_root,
        create_dirs=False,
    )
    target_level = pathway_target_level(cfg)
    if target_level == EXACT_PATHWAY_TARGET:
        require_exact_pathway_contract(cfg)
    contract = cfg["formal_contract"]
    expected_folds = int(contract["expected_folds"])
    expected_models = int(contract["expected_models"])
    expected_seeds = int(contract["expected_seeds"])
    expected_tasks = int(contract["expected_tasks"])
    input_root = Path(args.input_root).resolve()
    input_manifest = Path(args.input_manifest).resolve()
    training_root = Path(args.output_root).resolve()
    output = Path(args.output).resolve()
    assets = run_root / "assets"
    results = assets / "results"
    tables = results / "tables"
    canonical = run_root / "canonical"
    provenance_root = run_root / "provenance"
    provenance = json.loads((provenance_root / "PROVENANCE.json").read_text(encoding="utf-8"))
    runtime_repo_root = Path(args.repo_root or provenance["repo_root"]).resolve()
    asset_success = json.loads((assets / "SUCCESS.json").read_text(encoding="utf-8"))
    canonical_success = json.loads((canonical / "SUCCESS.json").read_text(encoding="utf-8"))
    if args.run_id != provenance.get("run_id") or args.run_id != asset_success.get("run_id") or args.run_id != canonical_success.get("run_id"):
        raise RuntimeError("run_id lineage disagreement")
    if training_root.exists():
        raise RuntimeError(f"Formal output root must be new: {training_root}")

    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: Any) -> None:
        checks.append({"requirement": name, "status": "PASS" if passed else "FAIL", "detail": detail})

    runtime_graph = cfg.get("runtime_graph_sampling", {})
    graph_contract = cfg.get("graph_contract", {})
    add(
        "v31_full_graph_runtime_contract",
        cfg.get("training", {}).get("max_edges_per_relation", "MISSING") is None
        and bool(runtime_graph.get("enabled", False))
        and runtime_graph.get("destructive_offline_cap", "MISSING") is None
        and runtime_graph.get("coverage_cycle") == "all_runtime_chunks"
        and graph_contract.get("primary") == "CONTRACT-T",
        {
            "max_edges_per_relation": cfg.get("training", {}).get("max_edges_per_relation"),
            "runtime_graph_sampling": runtime_graph,
            "graph_contract_primary": graph_contract.get("primary"),
        },
    )
    state_loss_weight = float(
        cfg.get("state_training", {}).get("auxiliary_loss_weight", float("nan"))
    )
    episodic_fixed_graph = bool(graph_contract.get("episodic_pseudoheldout", False))
    add(
        "v31_fixed_graph_state_auxiliary_loss_weight",
        (not episodic_fixed_graph)
        or np.isclose(state_loss_weight, 0.50, atol=0.0, rtol=0.0),
        {
            "episodic_pseudoheldout": episodic_fixed_graph,
            "auxiliary_loss_weight": state_loss_weight,
            "required_when_enabled": 0.50,
        },
    )
    configured_epoch_cap = int(contract.get("configured_epoch_cap", 0))
    configured_training_epochs = int(cfg.get("training", {}).get("epochs", 0))
    resume_checkpoint_interval = int(
        cfg.get("training", {}).get("resume_checkpoint_every_epochs", 0)
    )
    formal_execution = cfg.get("formal_execution", {})
    cap_selection = cfg.get("convergence_cap_selection", {})
    full_retrain_contract = bool(contract.get("require_full_297_retrain", False))
    task_source = (runtime_repo_root / "scripts" / "42_train_v29_strict_multitask.py").read_text(
        encoding="utf-8"
    )
    multitask_source = (runtime_repo_root / "cc_hhgt" / "v29_multitask.py").read_text(
        encoding="utf-8"
    )
    audit_source = (runtime_repo_root / "scripts" / "72_audit_v30_state_matrix.py").read_text(
        encoding="utf-8"
    )
    add(
        "v31_convergence_and_exact_interruption_resume_contract",
        target_level != EXACT_PATHWAY_TARGET
        or (
            configured_epoch_cap == configured_training_epochs
            and configured_epoch_cap > 500
            and contract.get("convergence_policy")
            == "fail_if_any_task_reaches_hard_epoch_cap"
            and resume_checkpoint_interval == 5
            and "--resume-training" in task_source
            and "last_training_state.pt" in multitask_source
            and "TRAINING_PROGRESS.json" in multitask_source
            and "stopped_by_joint_patience" in audit_source
            and int(formal_execution.get("required_physical_gpus", 0)) == 2
            and int(formal_execution.get("slots_per_device", 0)) == 2
            and int(formal_execution.get("task_visible_gpu_count", 0)) == 1
            and formal_execution.get("isolation") == "CUDA_VISIBLE_DEVICES"
            and float(formal_execution.get("minimum_gpu_memory_gib", 0.0)) >= 31.0
            and "minimum_gpu_memory_gib" in task_source
            and (
                not cap_selection
                or (
                    int(cap_selection.get("selected_safety_cap", 0))
                    == configured_epoch_cap
                    and int(
                        cap_selection.get(
                            "earliest_third_regular_stale_validation_epoch", 0
                        )
                    )
                    < configured_epoch_cap
                    and not bool(
                        cap_selection.get("selection_uses_heldout_metric", True)
                    )
                )
            )
            and (
                not full_retrain_contract
                or (
                    expected_tasks == 297
                    and not bool(contract.get("permit_prior_task_import", True))
                    and not bool(
                        contract.get("permit_prior_checkpoint_reuse", True)
                    )
                )
            )
        ),
        {
            "formal_configured_epoch_cap": configured_epoch_cap,
            "training_epochs": configured_training_epochs,
            "convergence_policy": contract.get("convergence_policy"),
            "resume_checkpoint_every_epochs": resume_checkpoint_interval,
            "guarded_resume_cli": "--resume-training" in task_source,
            "full_training_state_checkpoint": "last_training_state.pt"
            in multitask_source,
            "atomic_live_training_progress": "TRAINING_PROGRESS.json"
            in multitask_source,
            "joint_patience_release_audit": "stopped_by_joint_patience"
            in audit_source,
            "formal_execution": formal_execution,
            "convergence_cap_selection": cap_selection,
            "full_retrain_contract": {
                "required": full_retrain_contract,
                "expected_tasks": expected_tasks,
                "permit_prior_task_import": contract.get(
                    "permit_prior_task_import"
                ),
                "permit_prior_checkpoint_reuse": contract.get(
                    "permit_prior_checkpoint_reuse"
                ),
            },
            "guarded_gpu_memory_check": "minimum_gpu_memory_gib" in task_source,
        },
    )

    manifest_contracts = [
        (input_root, input_manifest),
        (runtime_repo_root, provenance_root / "CODE_MANIFEST.tsv"),
        (assets, assets / "ASSET_MANIFEST.tsv"),
    ]
    manifest_mismatches: list[dict[str, str]] = []
    contracts_payload = []
    for root, manifest in manifest_contracts:
        records = _records(manifest)
        manifest_mismatches.extend(verify_file_manifest(root, records))
        contracts_payload.append(
            {
                "root": str(root),
                "manifest": str(manifest.resolve()),
                "manifest_sha256": file_sha256(manifest),
                "merkle_sha256": merkle_sha256(records),
                "files": len(records),
            }
        )
    add("frozen_manifest_assets_match", not manifest_mismatches, manifest_mismatches[:50])
    add("input_merkle_matches_provenance", contracts_payload[0]["merkle_sha256"] == provenance["input_merkle_sha256"], contracts_payload[0]["merkle_sha256"])
    add("code_merkle_matches_provenance", contracts_payload[1]["merkle_sha256"] == provenance["code_merkle_sha256"], contracts_payload[1]["merkle_sha256"])
    add("asset_merkle_matches_build", contracts_payload[2]["merkle_sha256"] == asset_success["asset_merkle_sha256"], contracts_payload[2]["merkle_sha256"])

    samples = read_table(canonical / "canonical_samples.parquet")
    states = read_table(canonical / "canonical_state_complete_cases.parquet")
    eligibility = read_table(canonical / "state_complete_case_eligibility.tsv")
    model_eligibility = read_table(tables / "state_model_eligibility.tsv")
    add("canonical_sample_unique", not samples.duplicated(["cancer_id", "sample_id"]).any(), len(samples))
    normal_count = int(samples.sample_type_code.astype(int).isin(range(10, 20)).sum())
    add("canonical_zero_normal", normal_count == 0 and samples.is_tumor.astype(bool).all(), normal_count)
    add("state_nan_never_enters_statistics", not states.state_value.isna().any(), int(states.state_value.isna().sum()))
    sample_hashes = samples.sample_universe_sha256.astype(str).unique()
    add("one_sample_universe_hash", len(sample_hashes) == 1 and sample_hashes[0] == asset_success["sample_universe_sha256"], sample_hashes.tolist())

    reference = set(map(str, cfg["cancer_scope"]["reference_only"]))
    required_formal = set(map(str, contract.get("required_formal_cancers", [])))
    targets = set(map(str, cfg["sample_contract"]["target_states"]))
    formal_eligibility = model_eligibility.loc[~model_eligibility.cancer_id.astype(str).isin(reference)]
    coverage = formal_eligibility.groupby("state_id", observed=True).evaluation_eligibility.apply(lambda x: int(x.astype(str).eq("ELIGIBLE").sum())).to_dict()
    add(
        "selected_state_eligibility_complete",
        set(coverage) == targets and all(value / expected_folds >= 0.80 for value in coverage.values()),
        coverage,
    )
    reference_rows = eligibility.loc[eligibility.cancer_id.astype(str).isin(reference)]
    add(
        "configured_reference_only_scope",
        (not reference and reference_rows.empty),
        {"reference_only": sorted(reference), "rows": reference_rows[["cancer_id", "state_id", "eligibility"]].to_dict("records")},
    )
    required_rows = eligibility.loc[eligibility.cancer_id.astype(str).isin(required_formal)]
    add(
        "required_formal_cancers_not_reference_only",
        set(required_rows.cancer_id.astype(str)) == required_formal
        and required_rows.cancer_role.astype(str).eq("FORMAL").all(),
        required_rows[["cancer_id", "state_id", "eligibility", "cancer_role"]].to_dict("records"),
    )
    ov_dnass = model_eligibility.loc[
        model_eligibility.cancer_id.astype(str).eq("OV")
        & model_eligibility.state_id.astype(str).eq("stemness_dna::DNAss")
    ]
    add(
        "ov_dnass_explicitly_unavailable",
        len(ov_dnass) == 1
        and str(ov_dnass.iloc[0].evaluation_eligibility) == "UNAVAILABLE"
        and str(ov_dnass.iloc[0].evaluation_unavailable_reason).startswith("N_OBSERVED_LT_"),
        ov_dnass.to_dict("records"),
    )

    folds = read_table(tables / "fold_manifest.tsv")
    test_cancers = set(folds.test_cancer.astype(str))
    add(
        f"exact_{expected_folds}_formal_folds",
        len(folds) == expected_folds
        and folds.test_cancer.astype(str).nunique() == expected_folds
        and not folds.test_cancer.astype(str).isin(reference).any()
        and required_formal.issubset(test_cancers),
        folds.test_cancer.astype(str).tolist(),
    )
    observed_tasks = len(folds) * len(MODELS) * len(SEEDS)
    add(
        f"exact_{expected_tasks}_tasks",
        len(MODELS) == expected_models and len(SEEDS) == expected_seeds and observed_tasks == expected_tasks,
        observed_tasks,
    )
    validation_ok = set(folds.validation_cancer.astype(str)).issubset(
        set(
            formal_eligibility.loc[
                formal_eligibility.evaluation_eligibility.astype(str).eq("ELIGIBLE")
            ].groupby("cancer_id", observed=True).filter(
                lambda group: group.state_id.astype(str).nunique() == len(targets)
            ).cancer_id.astype(str)
        )
    )
    add("validation_cancers_have_both_classes_for_all_states", validation_ok, sorted(folds.validation_cancer.astype(str).unique()))

    nodes = read_table(tables / "graph_node.parquet")
    edges = read_table(tables / "graph_edge.parquet")
    graph_states = set(nodes.loc[nodes.node_type.astype(str).eq("state"), "canonical_id"].astype(str))
    add("graph_contains_only_selected_states", graph_states == targets, sorted(graph_states))
    add("target_independent_explicit_and_narrow", "target_independent" in edges and edges.loc[edges.target_independent.fillna(False).astype(bool), "relation_type"].astype(str).eq("expressed_in").all(), edges.loc[edges.target_independent.fillna(False).astype(bool), "relation_type"].astype(str).value_counts().to_dict() if "target_independent" in edges else "missing")
    direct_target = edges.source_type.astype(str).eq("lncRNA") & edges.target_type.astype(str).isin(["pathway", "pathway_family", "state"])
    add("direct_discovery_target_edges_zero", int(direct_target.sum()) == 0, int(direct_target.sum()))

    fold_edge_audit = []
    representation_rows = []
    for row in folds.itertuples(index=False):
        heldout = {str(row.test_cancer), str(row.validation_cancer)}
        # Only cancer-context rows can be affected by this policy.  Restricting
        # before the functional filter avoids copying the 8M+ static graph 31
        # times while exercising the exact same filtering implementation.
        relevant = edges.loc[edges.cancer_id.astype("string").isin(heldout | reference)].copy()
        filtered = filter_fold_edges(relevant, heldout_cancers=heldout, reference_only=reference)
        surviving = filtered.loc[filtered.cancer_id.astype("string").isin(heldout) & filtered.is_context_specific.fillna(False).astype(bool)]
        forbidden = int((~surviving.relation_type.astype(str).eq("expressed_in")).sum())
        ref_surviving = int(filtered.cancer_id.astype("string").isin(reference).sum())
        fold_edge_audit.append({"fold_id": str(row.fold_id), "forbidden_heldout": forbidden, "reference_edges": ref_surviving})
        cancer_nodes = nodes.loc[nodes.node_type.astype(str).eq("cancer") & nodes.canonical_id.astype(str).isin(heldout)]
        first = fold_node_features(cancer_nodes, filtered, "cancer")
        second = fold_node_features(cancer_nodes, filtered, "cancer")
        representation_rows.append({"fold_id": str(row.fold_id), "deterministic": bool(np.array_equal(first, second)), "distinguishable": bool(len(first) == 2 and not np.array_equal(first[0], first[1]))})
    add("heldout_only_expression_edges", all(item["forbidden_heldout"] == 0 for item in fold_edge_audit), fold_edge_audit)
    add("reference_context_edges_removed", all(item["reference_edges"] == 0 for item in fold_edge_audit), fold_edge_audit)
    add("heldout_cancer_features_deterministic_distinguishable", all(item["deterministic"] and item["distinguishable"] for item in representation_rows), representation_rows)

    candidate_root = tables / "strict_state_candidate"
    candidate_parts = sorted(candidate_root.glob("cancer_id=*/part-0.parquet"))
    candidate_audit = []
    lineage_required = {
        "run_id", "sample_universe_sha256", "code_merkle_sha256", "config_merkle_sha256",
        "input_merkle_sha256", "schema_sha256", "eligibility", "evaluation_policy",
        "n_observed", "n_missing", "design_rank", "residual_df", "lineage_sha256",
    }
    for part in candidate_parts:
        frame = pd.read_parquet(part)
        missing = sorted(lineage_required - set(frame))
        for state_id, group in frame.groupby("state_id", observed=True):
            candidate_audit.append(
                {
                    "cancer_id": str(group.cancer_id.iloc[0]),
                    "state_id": str(state_id),
                    "rows": len(group),
                    "positive": int(group.proxy_label.astype(float).gt(0.5).sum()),
                    "unlabeled": int(group.proxy_label.astype(float).le(0.5).sum()),
                    "direction_unlabeled": int((group.proxy_label.astype(float).le(0.5) & group.direction_label.notna()).sum()),
                    "fdr_family_match": bool(group.fdr_family_size.astype(int).eq(len(group)).all()),
                    "evaluation_eligibility": str(group.evaluation_eligibility.iloc[0]),
                    "lineage_missing": missing,
                    "probability_ready": bool(group.candidate_id.notna().all()),
                }
            )
    add("candidate_lineage_complete", bool(candidate_audit) and all(not row["lineage_missing"] for row in candidate_audit), candidate_audit)
    add("candidate_fdr_family_predefined", bool(candidate_audit) and all(row["fdr_family_match"] for row in candidate_audit), candidate_audit)
    add("unlabeled_direction_supervision_zero", bool(candidate_audit) and all(row["direction_unlabeled"] == 0 for row in candidate_audit), candidate_audit)
    add(
        "candidate_proxy_classes_available_or_explicitly_unavailable",
        bool(candidate_audit)
        and all(
            (row["positive"] > 0 and row["unlabeled"] > 0)
            if row["evaluation_eligibility"] == "ELIGIBLE"
            else True
            for row in candidate_audit
        ),
        candidate_audit,
    )

    pan_filter_path = tables / "pancancer_lncrna_eligibility.parquet"
    if pan_filter_path.exists():
        pan_filter = read_table(pan_filter_path)
        filter_policy = cfg.get("candidate_universe", {}).get("pancancer_lnc_filter", {})
        min_cancers = int(filter_policy.get("minimum_detected_cancers", 1))
        eligible_pan = pan_filter.loc[pan_filter.eligible.astype(bool)]
        add(
            "pancancer_lncrna_filter_frozen",
            bool(filter_policy.get("enabled", False))
            and not eligible_pan.empty
            and eligible_pan.detected_cancers.astype(int).ge(min_cancers).all()
            and pan_filter.formal_cancer_count.astype(int).eq(expected_folds).all(),
            {
                "rows": len(pan_filter),
                "eligible": len(eligible_pan),
                "minimum_detected_cancers": min_cancers,
                "formal_cancer_counts": sorted(pan_filter.formal_cancer_count.astype(int).unique().tolist()),
            },
        )
    else:
        add("pancancer_lncrna_filter_frozen", False, "missing pancancer_lncrna_eligibility.parquet")

    pathway_manifest = read_table(tables / "candidate_universe_manifest.tsv")
    manifest_size_column = "n_pathways" if target_level == EXACT_PATHWAY_TARGET else "n_families"
    add(
        "pathway_candidates_cover_every_formal_cancer",
        set(pathway_manifest.loc[pathway_manifest.status.astype(str).isin(["PASS", "CACHED"]), "cancer_id"].astype(str))
        .issuperset(test_cancers)
        and pd.to_numeric(pathway_manifest.n_candidates, errors="coerce").fillna(0).gt(0).all()
        and pd.to_numeric(pathway_manifest[manifest_size_column], errors="coerce").fillna(0).gt(0).all(),
        pathway_manifest.to_dict("records"),
    )

    if target_level == EXACT_PATHWAY_TARGET:
        add(
            "exact_pathway_target_registered",
            asset_success.get("pathway_target_level") == "exact_pathway"
            and asset_success.get("pathway_target_column") == "pathway_id"
            and set(pathway_manifest.target_level.astype(str)) == {"exact_pathway"}
            and set(pathway_manifest.target_column.astype(str)) == {"pathway_id"},
            {
                "asset_target_level": asset_success.get("pathway_target_level"),
                "asset_target_column": asset_success.get("pathway_target_column"),
                "manifest_target_levels": sorted(pathway_manifest.target_level.astype(str).unique()),
            },
        )
        pair_path = tables / "pair_evidence.parquet"
        connection = duckdb.connect()
        pair_audit = connection.execute(
            f"""
            SELECT
              count(*) AS rows,
              count(DISTINCT cancer_id || chr(31) || lncrna_id || chr(31) || pathway_id)
                AS unique_targets,
              count_if(pathway_id IS NULL) AS missing_pathway,
              count_if(pathway_family_id IS NULL) AS missing_family,
              sum(association_proxy_label) AS positives,
              count_if(
                association_proxy_label = 1
                AND coalesce(bulk_support, 0) <= 0
                AND coalesce(sc_ssgsea_support, 0) <= 0
              ) AS annotation_only_positives,
              count(DISTINCT label_semantics) AS label_semantics_count,
              min(label_semantics) AS label_semantics
            FROM read_parquet('{pair_path.as_posix()}')
            """
        ).fetchone()
        add(
            "exact_pathway_evidence_unique_and_positive",
            int(pair_audit[0]) == int(pair_audit[1])
            and int(pair_audit[2]) == 0
            and int(pair_audit[3]) == 0
            and int(pair_audit[4]) > 0
            and int(pair_audit[5]) == 0
            and int(pair_audit[6]) == 1
            and str(pair_audit[7]).startswith("EXACT_PATHWAY_PATIENT_ASSOCIATION_V2:"),
            {
                "rows": int(pair_audit[0]),
                "unique_targets": int(pair_audit[1]),
                "missing_pathway": int(pair_audit[2]),
                "missing_family": int(pair_audit[3]),
                "positives": int(pair_audit[4]),
                "annotation_only_positives": int(pair_audit[5]),
                "label_semantics_count": int(pair_audit[6]),
                "label_semantics": str(pair_audit[7]),
            },
        )
        exact_policy = cfg.get("exact_pathway_evidence", {})
        forbidden_pair_features = {
            "interaction_support", "perturbation_support", "drug_support",
            "independent_pmid_score", "independent_dataset_score",
        }
        configured_features = set(map(str, cfg["training"]["feature_columns"]))
        add(
            "exact_association_target_excludes_annotation_labels_and_pair_features",
            exact_policy.get("label_truth_layer") == "patient_statistical_association_only"
            and not bool(exact_policy.get("annotation_sources_may_define_label", True))
            and not (configured_features & forbidden_pair_features)
            and bool(
                cfg.get("state_training", {}).get(
                    "mask_pair_evidence_for_all_pathway_models", False
                )
            )
            and float(cfg.get("state_training", {}).get("strict_pair_evidence_mask_probability", -1)) == 1.0,
            {
                "label_truth_layer": exact_policy.get("label_truth_layer"),
                "annotation_sources_may_define_label": exact_policy.get("annotation_sources_may_define_label"),
                "forbidden_configured_features": sorted(configured_features & forbidden_pair_features),
                "mask_pair_evidence_for_all_pathway_models": cfg.get("state_training", {}).get("mask_pair_evidence_for_all_pathway_models"),
                "strict_pair_evidence_mask_probability": cfg.get("state_training", {}).get("strict_pair_evidence_mask_probability"),
            },
        )
        candidate_glob = (tables / "candidate_universe" / "cancer_id=*" / "part-0.parquet").as_posix()
        eligible_path = tables / "pancancer_lncrna_eligibility.parquet"
        candidate_exact_audit = connection.execute(
            f"""
            WITH candidate AS (
              SELECT * FROM read_parquet('{candidate_glob}', hive_partitioning=1)
            ), eligible AS (
              SELECT lncrna_id FROM read_parquet('{eligible_path.as_posix()}') WHERE eligible
            )
            SELECT
              count(*) AS rows,
              count(DISTINCT candidate_id) AS unique_candidates,
              count(DISTINCT cancer_id) AS cancers,
              count(DISTINCT pathway_id) AS pathways,
              count_if(pathway_id IS NULL) AS missing_pathway,
              count_if(pathway_family_id IS NULL) AS missing_family,
              count_if(eligible.lncrna_id IS NULL) AS ineligible_lnc,
              sum(association_proxy_label) AS positives
            FROM candidate LEFT JOIN eligible USING(lncrna_id)
            """
        ).fetchone()
        connection.close()
        add(
            "exact_pathway_candidates_filtered_unique_complete",
            int(candidate_exact_audit[0]) == int(candidate_exact_audit[1])
            and int(candidate_exact_audit[2]) == expected_folds
            and int(candidate_exact_audit[3]) > 1
            and int(candidate_exact_audit[4]) == 0
            and int(candidate_exact_audit[5]) == 0
            and int(candidate_exact_audit[6]) == 0
            and int(candidate_exact_audit[7]) > 0,
            {
                "rows": int(candidate_exact_audit[0]),
                "unique_candidates": int(candidate_exact_audit[1]),
                "cancers": int(candidate_exact_audit[2]),
                "pathways": int(candidate_exact_audit[3]),
                "missing_pathway": int(candidate_exact_audit[4]),
                "missing_family": int(candidate_exact_audit[5]),
                "ineligible_lnc": int(candidate_exact_audit[6]),
                "positives": int(candidate_exact_audit[7]),
            },
        )
        exact_graph_pathways = set(
            nodes.loc[nodes.node_type.astype(str).eq("pathway"), "canonical_id"].astype(str)
        )
        candidate_pathways = set(
            read_table(tables / "pathway_family_member.parquet").pathway_id.astype(str)
        )
        add(
            "exact_pathway_targets_are_graph_nodes",
            candidate_pathways.issubset(exact_graph_pathways),
            {
                "hierarchy_pathways": len(candidate_pathways),
                "graph_pathways": len(exact_graph_pathways),
                "missing": sorted(candidate_pathways - exact_graph_pathways)[:50],
            },
        )

    source = (runtime_repo_root / "cc_hhgt" / "gnn.py").read_text(encoding="utf-8")
    add("rgcn_has_no_id_embedding", "nn.Embedding" not in source and "self.input_projection" in source, "shared node feature encoder")
    add("edge_weight_consumed", "edge_weight" in source and "weighted_homogeneous_aggregate" in source and "weighted_aggregate" in source, "functional regression tests required and frozen in code snapshot")
    source_common = (runtime_repo_root / "cc_hhgt" / "common.py").read_text(encoding="utf-8")
    add("deterministic_pytorch_enabled", "torch.use_deterministic_algorithms(True)" in source_common, "torch.use_deterministic_algorithms(True)")

    failures = [check for check in checks if check["status"] != "PASS"]
    guarded = [
        assets / "SUCCESS.json",
        assets / "ASSET_MANIFEST.tsv",
        canonical / "SUCCESS.json",
        canonical / "canonical_samples.parquet",
        canonical / "canonical_state_complete_cases.parquet",
        canonical / "state_complete_case_eligibility.tsv",
        tables / "state_model_eligibility.tsv",
        tables / "fold_manifest.tsv",
        tables / "graph_node.parquet",
        tables / "graph_edge.parquet",
        Path(args.config).resolve(),
    ]
    payload = {
        "status": "PASS" if not failures else "FAIL",
        "analysis_version": cfg["analysis_version"],
        "run_id": args.run_id,
        "expected_folds": expected_folds,
        "expected_models": expected_models,
        "expected_seeds": expected_seeds,
        "expected_tasks": expected_tasks,
        "models": list(MODELS),
        "seeds": list(SEEDS),
        "required_failures": failures,
        "checks": checks,
        "paths": {
            "run_root": str(run_root),
            "input_root": str(input_root),
            "input_manifest": str(input_manifest),
            "output_root": str(training_root),
            "asset_results": str(results),
        },
        "manifest_contracts": contracts_payload,
        "guarded_sha256": {str(path): file_sha256(path) for path in guarded},
        "asset_merkle_sha256": asset_success["asset_merkle_sha256"],
        "sample_universe_sha256": canonical_success["sample_universe_sha256"],
        "verify_assets_at_task_start": True,
        "verify_assets_at_task_end": True,
        "trusted_gate_bypass_allowed": False,
        "probability_semantics": "proxy_positive_probability",
        "pathway_target_level": target_level,
        "pathway_target_column": (
            "pathway_id" if target_level == EXACT_PATHWAY_TARGET else "pathway_family_id"
        ),
    }
    atomic_write_json(output, payload)
    atomic_write_bytes(output.with_suffix(".tsv"), pd.DataFrame(checks).to_csv(sep="\t", index=False, lineterminator="\n").encode("utf-8"))
    print(json.dumps({**payload, "checks": f"{len(checks)} checks", "guarded_sha256": f"{len(guarded)} files"}, ensure_ascii=False, indent=2))
    if failures:
        raise RuntimeError(f"V3 formal gate failed: {[item['requirement'] for item in failures]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
