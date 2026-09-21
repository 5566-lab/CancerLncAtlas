#!/usr/bin/env python3
"""Materialize exact-pathway Gene Sets and downstream ranked subtypes by cancer."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SUBTYPE_MEMBER_COLUMNS = (
    "geneset_id",
    "geneset_rank",
    "cancer_id",
    "lncrna_id",
    "pathway_id",
    "pathway_family_id",
    "association_membership_probability",
    "association_direction",
    "shared_or_local_scope",
)


def _load_subtype_members(member_paths: list[Path]) -> pd.DataFrame:
    """Load the exact columns required by the ranked-subtype contract."""

    return pd.concat(
        [pd.read_parquet(path, columns=list(SUBTYPE_MEMBER_COLUMNS)) for path in member_paths],
        ignore_index=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-pattern", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--analysis-version", default="CancerLncAtlas_V3.2_RS_CC-HHGT_1seed_FORMAL_RELEASE")
    args = parser.parse_args()

    from cc_hhgt.v32.genesets import (
        GeneSetMaterializationConfig,
        genesets_to_gmt_lines,
        materialize_ranked_genesets,
    )
    from cc_hhgt.v32.subtypes import RankedSubtypeConfig, infer_ranked_subtypes

    paths = [Path(args.prediction_pattern.format(fold=fold)) for fold in range(5)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"Fold predictions missing: {missing}")
    output = Path(args.output)
    master_root = output / "geneset_master_parts"
    member_root = output / "geneset_member_parts"
    master_root.mkdir(parents=True, exist_ok=True)
    member_root.mkdir(parents=True, exist_ok=True)
    cancers = sorted(pd.read_parquet(paths[0], columns=["cancer_id"]).cancer_id.astype(str).unique())
    config = GeneSetMaterializationConfig(
        expected_folds=5,
        min_members=10,
        max_members=200,
        membership_probability_min=0.5,
        analysis_version=args.analysis_version,
    )
    gmt_path = output / "CancerLncAtlas_V3_2_exact_pathway_ranked.gmt"
    with gmt_path.open("w", encoding="utf-8", newline="\n") as gmt:
        for cancer in cancers:
            frames = [
                pd.read_parquet(path, filters=[("cancer_id", "=", cancer)]).drop(
                    columns=["held_out_proxy_label", "association_direction_probability", "regulatory_evidence_available"],
                    errors="ignore",
                )
                for path in paths
            ]
            fold_predictions = pd.concat(frames, ignore_index=True)
            master, members = materialize_ranked_genesets(fold_predictions, config)
            master.to_parquet(master_root / f"cancer_id={cancer}.parquet", index=False, compression="zstd")
            members.to_parquet(member_root / f"cancer_id={cancer}.parquet", index=False, compression="zstd")
            for line in genesets_to_gmt_lines(master, members):
                gmt.write(line + "\n")
            print(json.dumps({"status": "GENESET_CANCER_DONE", "cancer": cancer, "genesets": len(master), "members": len(members)}), flush=True)

    master_paths = sorted(master_root.glob("*.parquet"))
    member_paths = sorted(member_root.glob("*.parquet"))
    master = pd.concat([pd.read_parquet(path) for path in master_paths], ignore_index=True)
    master.to_parquet(output / "GENESET_MASTER.parquet", index=False, compression="zstd")
    subtype_members = _load_subtype_members(member_paths)
    subtype = infer_ranked_subtypes(
        subtype_members,
        RankedSubtypeConfig(
            top_n=200,
            rbo_p=0.98,
            rbo_weight=0.5,
            weighted_jaccard_weight=0.5,
            k_min=1,
            k_max=4,
            program_k_max=6,
            silhouette_threshold=0.25,
            bootstrap_ari_threshold=0.75,
            bootstrap_replicates=200,
            min_cancers=6,
            min_members=10,
            min_cluster_size=3,
            random_seed=20260726,
        ),
    )
    subtype.pathway_context_subtype.to_parquet(output / "PATHWAY_CONTEXT_SUBTYPE.parquet", index=False)
    subtype.pathway_conservation.to_parquet(output / "PATHWAY_CONSERVATION.parquet", index=False)
    subtype.pathway_pairwise_similarity.to_parquet(output / "PATHWAY_PAIRWISE_SIMILARITY.parquet", index=False)
    subtype.cancer_program_subtype.to_parquet(output / "CANCER_PROGRAM_SUBTYPE.parquet", index=False)
    subtype.cancer_program_similarity.to_parquet(output / "CANCER_PROGRAM_SIMILARITY.parquet", index=False)
    summary = {
        "status": "SUCCESS",
        "analysis_version": args.analysis_version,
        "cancers": len(cancers),
        "genesets": int(len(master)),
        "subtype_member_rows": int(len(subtype_members)),
        "pathway_context_rows": int(len(subtype.pathway_context_subtype)),
        "pathway_conservation_rows": int(len(subtype.pathway_conservation)),
        "cancer_program_rows": int(len(subtype.cancer_program_subtype)),
        "family_used_as_target": False,
        "subtype_feedback_to_model": False,
        "regulatory_evidence_used_for_ranking": False,
    }
    (output / "MATERIALIZATION_SUCCESS.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
