#!/usr/bin/env python3
"""Measure one cancer/fold of the frozen coexpression calculation.

This is a design-time resource probe.  It deliberately writes no graph edge,
checkpoint, prediction, ranking, or success marker and cannot authorize
preparation or training.  The numerical kernels are loaded from the frozen
V3.2 bulk-coexpression module supplied on the command line.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time


THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--bulk-module", type=Path, required=True)
    parser.add_argument("--lncrna-parquet", type=Path, required=True)
    parser.add_argument("--gene-parquet", type=Path, required=True)
    parser.add_argument("--covariates", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--cancer", default="BRCA")
    parser.add_argument("--fold-id", default="LOCO_BRCA__PF00")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--min-samples", type=int, default=40)
    parser.add_argument("--min-variance", type=float, default=0.01)
    parser.add_argument("--min-abs-rho", type=float, default=0.20)
    parser.add_argument("--max-fdr", type=float, default=0.05)
    parser.add_argument("--max-edges-per-direction", type=int, default=75)
    parser.add_argument("--block-size", type=int, default=128)
    return parser.parse_args()


def _load_bulk_module(code_root: Path, module_path: Path):
    sys.path.insert(0, str(code_root.resolve()))
    spec = importlib.util.spec_from_file_location(
        "v32_coexpression_resource_probe_frozen_module", module_path.resolve()
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load frozen module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _elapsed(step_started: float) -> float:
    return round(time.perf_counter() - step_started, 6)


def main() -> int:
    args = _arguments()
    thread_values = {name: os.environ.get(name) for name in THREAD_ENVIRONMENT}
    if any(value != "1" for value in thread_values.values()):
        raise RuntimeError(f"Every numerical thread environment must equal 1: {thread_values}")

    inputs = (
        args.bulk_module,
        args.lncrna_parquet,
        args.gene_parquet,
        args.covariates,
        args.fold_manifest,
    )
    for path in inputs:
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)
    output = args.output_json.resolve()
    allowed_root = Path(
        "./data/CancerLncAtlas/runtime/smoke/v32_g012_coexpression_resource_probe_20260829_r1"
    ).resolve()
    if output.parent != allowed_root:
        raise RuntimeError(f"Probe output must be directly under {allowed_root}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite probe output: {output}")

    total_started = time.perf_counter()
    steps: dict[str, float] = {}
    module_started = time.perf_counter()
    bulk = _load_bulk_module(args.code_root, args.bulk_module)
    import numpy as np
    import pandas as pd
    steps["module_import_seconds"] = _elapsed(module_started)

    fold_started = time.perf_counter()
    fold = pd.read_csv(
        args.fold_manifest,
        sep="\t",
        usecols=[
            "cancer_id",
            "patient_id",
            "fold_id",
            "split",
            "patient_assignment_seed",
        ],
    )
    selected_fold = fold.loc[
        fold["cancer_id"].astype(str).eq(args.cancer)
        & fold["fold_id"].astype(str).eq(args.fold_id)
    ].copy()
    if selected_fold.empty or selected_fold["patient_id"].duplicated().any():
        raise RuntimeError("Selected patient fold is empty or duplicated")
    train_patients = set(
        selected_fold.loc[
            selected_fold["split"].astype(str).eq("train"), "patient_id"
        ].astype(str)
    )
    if len(train_patients) < args.min_samples:
        raise RuntimeError("Selected fold has too few train patients")
    steps["fold_manifest_seconds"] = _elapsed(fold_started)

    lnc_started = time.perf_counter()
    lnc, lnc_audit = bulk._patient_matrix(
        args.lncrna_parquet,
        cancer=args.cancer,
        entity_column="lncrna_id",
    )
    steps["lncrna_load_pivot_seconds"] = _elapsed(lnc_started)

    gene_started = time.perf_counter()
    gene, gene_audit = bulk._patient_matrix(
        args.gene_parquet,
        cancer=args.cancer,
        entity_column="gene_id",
    )
    steps["gene_load_pivot_seconds"] = _elapsed(gene_started)

    patients = sorted(train_patients & set(lnc.index) & set(gene.index))
    if len(patients) < args.min_samples:
        raise RuntimeError("Too few aligned train patients after expression intersection")
    lnc = lnc.reindex(patients)
    gene = gene.reindex(patients)

    filter_started = time.perf_counter()
    lnc, dropped_lnc = bulk._variance_filter(lnc, args.min_variance)
    gene, dropped_gene = bulk._variance_filter(gene, args.min_variance)
    if lnc.shape[1] > 12_000 or gene.shape[1] > 25_000:
        raise RuntimeError("Frozen feature-count ceiling exceeded")
    steps["variance_filter_seconds"] = _elapsed(filter_started)

    covariate_started = time.perf_counter()
    covariates = pd.read_parquet(args.covariates)
    design, covariates_used, missing_covariate_patients = bulk._covariate_design(
        covariates,
        cancer=args.cancer,
        patients=patients,
        columns=bulk.DEFAULT_COVARIATES,
    )
    residual_rank = bulk.design_rank(design)
    if len(patients) - residual_rank - 1 <= 0:
        raise RuntimeError("No residual correlation degrees of freedom")
    steps["covariate_design_seconds"] = _elapsed(covariate_started)

    residual_started = time.perf_counter()
    lnc_values = bulk._residual_rank_matrix(lnc, design)
    gene_values = bulk._residual_rank_matrix(gene, design)
    steps["rank_residual_standardize_seconds"] = _elapsed(residual_started)

    bh_started = time.perf_counter()
    sorted_p, adjusted_p, p_cutoff, total_tests = bulk._conservative_bh_candidates(
        lnc_values,
        gene_values,
        n_patients=len(patients),
        residual_rank=residual_rank,
        block_size=args.block_size,
        min_abs_rho=args.min_abs_rho,
        max_fdr=args.max_fdr,
    )
    steps["correlation_bh_pass_seconds"] = _elapsed(bh_started)

    selection_started = time.perf_counter()
    edges = bulk._select_edges(
        lnc_values,
        gene_values,
        lnc_ids=lnc.columns.to_numpy(str),
        gene_ids=gene.columns.to_numpy(str),
        cancer=args.cancer,
        n_patients=len(patients),
        residual_rank=residual_rank,
        block_size=args.block_size,
        min_abs_rho=args.min_abs_rho,
        max_fdr=args.max_fdr,
        max_edges_per_direction=args.max_edges_per_direction,
        sorted_candidate_p=sorted_p,
        adjusted_candidate_p=adjusted_p,
        p_cutoff=p_cutoff,
        run_id="DESIGN_RESOURCE_PROBE_ONLY",
    )
    steps["correlation_edge_selection_pass_seconds"] = _elapsed(selection_started)

    direction_counts = (
        {str(key): int(value) for key, value in edges["direction"].value_counts().items()}
        if not edges.empty
        else {}
    )
    report = {
        "status": "PASS_RESOURCE_PROBE_ONLY_NOT_AN_AUTHORITY",
        "formal_preparation_started": False,
        "training_started": False,
        "graph_edges_persisted": False,
        "predictions_persisted": False,
        "cancer": args.cancer,
        "fold_id": args.fold_id,
        "fold_seed_values": sorted(
            int(value) for value in selected_fold["patient_assignment_seed"].unique()
        ),
        "thread_environment": thread_values,
        "configuration": {
            "min_samples": args.min_samples,
            "detection_rate": 0.10,
            "detection_scope": "INHERITED_FROM_FRESH_FORMAL_LNCRNA_PARTITION",
            "min_variance": args.min_variance,
            "min_abs_rho": args.min_abs_rho,
            "max_fdr": args.max_fdr,
            "max_edges_per_lncrna_per_sign": args.max_edges_per_direction,
            "block_size": args.block_size,
            "correlation": "COVARIATE_RESIDUALIZED_SPEARMAN",
            "multiple_testing": "CONSERVATIVE_GLOBAL_BH_WITH_FULL_TEST_DENOMINATOR",
        },
        "counts": {
            "fold_rows": int(len(selected_fold)),
            "declared_train_patients": int(len(train_patients)),
            "aligned_train_patients": int(len(patients)),
            "candidate_lncrnas_before_variance": int(lnc_audit["features"]),
            "variable_lncrnas": int(lnc.shape[1]),
            "dropped_lncrnas_variance": int(len(dropped_lnc)),
            "genes_before_variance": int(gene_audit["features"]),
            "variable_genes": int(gene.shape[1]),
            "dropped_genes_variance": int(len(dropped_gene)),
            "total_tests": int(total_tests),
            "bh_candidate_p_values": int(len(sorted_p)),
            "selected_edges": int(len(edges)),
            "selected_lncrnas": int(edges["lncrna_id"].nunique()) if len(edges) else 0,
            "selected_genes": int(edges["gene_id"].nunique()) if len(edges) else 0,
            "direction_counts": direction_counts,
            "residual_design_rank": int(residual_rank),
            "correlation_df": int(len(patients) - residual_rank - 1),
            "missing_covariate_patients": int(missing_covariate_patients),
        },
        "covariates_used": covariates_used,
        "bh_p_cutoff": None if p_cutoff is None else float(p_cutoff),
        "step_wall_seconds": steps,
        "total_wall_seconds": round(time.perf_counter() - total_started, 6),
        "process_max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "input_paths": [str(path.resolve()) for path in inputs],
    }
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output)
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
