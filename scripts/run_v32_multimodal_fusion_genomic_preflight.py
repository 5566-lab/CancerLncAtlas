#!/usr/bin/env python3
"""Run a deterministic real-data V3.2 genomic-only fusion preflight.

This is intentionally not a release.  It verifies the pair-blocked secondary
fusion on current formal exact-pathway OOF labels and current fresh
Mutation/CNV outputs while Drug, Evidence and single-cell heads are still
finishing.  A poor result is retained as DIAGNOSTIC_ONLY, never deleted.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.multimodal_fusion import (  # noqa: E402
    ExpertSpec,
    ResidualFusionConfig,
    artifact_sha256,
    build_pair_aggregated_fusion_frame,
    train_pair_blocked_crossfit_fusion,
)


PRIMARY = ROOT / "artifacts/v32_full_multitask/exact_pathway_release_r2/exact_pathway_five_fold_ensemble.parquet"
GENOMIC = ROOT / "artifacts/v32_full_multitask/genomic_fresh_rerun1/mutation_cnv_typed_predictions.parquet"
FOLD_ROOT = ROOT / "artifacts/formal_release_predictions_1seed"
FORMAL_HASHES = {
    PRIMARY: "4259ecc7453f0c636200087e8393db59a73824414f64c78a9fbc8e3bcb47c3a1",
    GENOMIC: "a0b4d4bfe1399dc7d34968dced86ed62d76ea76b5638236ad11694277a3a24a5",
    FOLD_ROOT / "FOLD_0_PREDICTIONS.parquet": "0a8f3f7661a6a7781c68a7614d5da37659a2f1e497db7d0cdfb2bb09648e2f7b",
    FOLD_ROOT / "FOLD_1_PREDICTIONS.parquet": "11a1c141042436764b37e73641410e787f3ed4ff0bfa0a8b2bc2f90dad8b7f61",
    FOLD_ROOT / "FOLD_2_PREDICTIONS.parquet": "1bfb9bb4dcabec5c8bc59f21aa30d61a1e82e1b382a3bbd2f9574a6a7dfb4943",
    FOLD_ROOT / "FOLD_3_PREDICTIONS.parquet": "eb3d2dc4e21f810ee2959e7f8215d685fce635399e2da276a7afd24cee7deecb",
    FOLD_ROOT / "FOLD_4_PREDICTIONS.parquet": "2350bbc2e6672aadb7fb5fdfc59918170229d45d7b0a7307f0862aa4d16e8fab",
}


def _sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def _sample_clause(modulus: int, remainder: int) -> str:
    return (
        "hash(CAST(lncrna_id AS VARCHAR) || '|' || CAST(pathway_id AS VARCHAR)) "
        f"% {int(modulus)} = {int(remainder)}"
    )


def _read_sample(path: Path, columns: list[str], clause: str):
    projection = ", ".join(columns)
    connection = duckdb.connect(database=":memory:")
    try:
        return connection.execute(
            f"SELECT {projection} FROM read_parquet('{_sql_path(path)}') WHERE {clause}"
        ).fetchdf()
    finally:
        connection.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/v32_multimodal_fusion_genomic_preflight_20260826_r1",
    )
    parser.add_argument("--sample-modulus", type=int, default=20)
    parser.add_argument("--sample-remainder", type=int, default=0)
    args = parser.parse_args()
    if args.sample_modulus < 1 or not 0 <= args.sample_remainder < args.sample_modulus:
        raise SystemExit("Invalid deterministic sample modulus/remainder")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    for path, expected in FORMAL_HASHES.items():
        if not path.is_file():
            raise SystemExit(f"Formal input missing: {path}")
        observed = artifact_sha256(path)
        if observed != expected:
            raise SystemExit(f"Formal input SHA drift: {path}: {observed} != {expected}")
    clause = _sample_clause(args.sample_modulus, args.sample_remainder)
    primary = _read_sample(
        PRIMARY,
        [
            "cancer_id",
            "lncrna_id",
            "pathway_id",
            "association_membership_probability",
            "analysis_version",
        ],
        clause,
    )
    fold_frames = [
        _read_sample(
            FOLD_ROOT / f"FOLD_{fold}_PREDICTIONS.parquet",
            [
                "cancer_id",
                "lncrna_id",
                "pathway_id",
                "patient_fold_id",
                "held_out_proxy_label",
            ],
            clause,
        )
        for fold in range(5)
    ]
    genomic = _read_sample(
        GENOMIC,
        [
            "cancer_id",
            "lncrna_id",
            "pathway_id",
            "mutation_cnv_context_probability",
            "genomic_available",
            "analysis_version",
        ],
        clause,
    )
    if primary.empty:
        raise SystemExit("Deterministic fusion sample is empty")
    spec = ExpertSpec(
        expert_id="mutation_cnv",
        probability_column="mutation_cnv_context_probability",
        availability_column="genomic_available",
        endpoint_roles=("discovery", "confidence"),
        split_unit="five_patient_fold_oof_aggregate",
        source_role="current_v32_oof_prediction",
    )
    frame = build_pair_aggregated_fusion_frame(
        primary,
        fold_frames,
        {spec: genomic},
        seed=20260826,
    )
    config = ResidualFusionConfig(
        seed=20260826,
        max_steps=400,
        patience=30,
        evaluation_interval=10,
        max_train_rows=300_000,
        max_validation_rows=100_000,
        batch_size=32_768,
        learning_rate=0.03,
    )
    result = train_pair_blocked_crossfit_fusion(
        frame,
        (spec,),
        endpoint="discovery",
        config=config,
    )
    private_path = output / "genomic_fusion_frame.PRIVATE.parquet"
    oof_path = output / "genomic_fusion_oof.PRIVATE.parquet"
    metrics_path = output / "genomic_fusion_metrics.parquet"
    checkpoint_path = output / "GENOMIC_FUSION_CHECKPOINT.json"
    frame.to_parquet(private_path, index=False, compression="zstd")
    result.oof_prediction.to_parquet(oof_path, index=False, compression="zstd")
    result.fold_metrics.to_parquet(metrics_path, index=False, compression="zstd")
    checkpoint_path.write_text(
        json.dumps(result.final_model.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    all_metrics = result.fold_metrics.loc[
        result.fold_metrics.heldout_pair_fold.eq("ALL_OOF")
    ].iloc[0]
    summary = {
        "status": "SUCCESS_PREFLIGHT_ONLY",
        "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
        "module_id": "multimodal_fusion_genomic_preflight",
        "sample_modulus": args.sample_modulus,
        "sample_remainder": args.sample_remainder,
        "sample_rows": int(len(frame)),
        "sample_pairs": int(frame[["lncrna_id", "pathway_id"]].drop_duplicates().shape[0]),
        "available_genomic_rows": int(frame.mutation_cnv_available.sum()),
        "primary_logloss": float(all_metrics.primary_logloss),
        "adjusted_logloss": float(all_metrics.adjusted_logloss),
        "primary_brier": float(all_metrics.primary_brier),
        "adjusted_brier": float(all_metrics.adjusted_brier),
        "performance_outcome": result.performance_outcome,
        "scientific_status": result.scientific_status,
        "no_increment_capability_retained": True,
        "mutation_cnv_removed_if_no_increment": False,
        "full_multimodal_fusion_complete": False,
        "genomic_feature_only": True,
        "primary_ranking_unchanged": True,
        "adjusted_ranking_is_secondary": True,
        "old_checkpoint_loaded": False,
        "old_predictions_used_as_features": False,
        "formal_input_sha256": {str(path.resolve()): sha for path, sha in FORMAL_HASHES.items()},
        "config": asdict(config),
        "artifacts": {
            "private_frame": {"path": str(private_path), "sha256": artifact_sha256(private_path)},
            "private_oof": {"path": str(oof_path), "sha256": artifact_sha256(oof_path)},
            "metrics": {"path": str(metrics_path), "sha256": artifact_sha256(metrics_path)},
            "checkpoint": {"path": str(checkpoint_path), "sha256": artifact_sha256(checkpoint_path)},
        },
        "release_ready": False,
        "production_deployed": False,
    }
    summary_path = output / "PREFLIGHT_SUMMARY.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({**summary, "summary_sha256": artifact_sha256(summary_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

