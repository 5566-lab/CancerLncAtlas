#!/usr/bin/env python3
"""Run fresh five-fold V3.2 private Evidence EventSet training."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build exact-pathway raw evidence bags and train five fresh private "
            "EventSet/attention heads on frozen same-fold V3.2 core embeddings."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--evidence-event", type=Path, required=True)
    parser.add_argument("--interaction-relation", type=Path, required=True)
    parser.add_argument("--pathway-members", type=Path, required=True)
    parser.add_argument(
        "--core-embedding-root",
        type=Path,
        default=Path("artifacts/v32_full_multitask/core_embeddings"),
    )
    parser.add_argument("--id-map", type=Path)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("artifacts/formal_prepared/FORMAL_CANDIDATE_UNIVERSE.parquet"),
        help=(
            "Exact-pathway candidate parquet or flat parquet directory. The default "
            "is the formal 3.3M/33-cancer universe; ranking columns are never read."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/v32_full_multitask/evidence_fresh"),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--formal-dry-run",
        action="store_true",
        help=(
            "Audit the strict connected-component split and, when fewer than five "
            "legal components exist, write all candidates as null+reason together "
            "with independent physical facts and raw event lineage."
        ),
    )
    mode.add_argument(
        "--split-preflight",
        action="store_true",
        help=(
            "Recompute raw exact event bags and audit the pair-blocked five-fold "
            "split plus per-fold PMID/source-event exclusion without training or "
            "writing predictions."
        ),
    )
    parser.add_argument("--expected-candidate-rows", type=int, default=3_300_000)
    parser.add_argument("--expected-cancer-count", type=int, default=33)
    parser.add_argument(
        "--expected-candidate-sha256",
        default="cced638f3d17ab9b1b0ff521169ac27dea3de1fc042adb7edefe6100a2ca071f",
        help="Pinned SHA-256 of the authoritative exact V3.2 candidate universe.",
    )
    parser.add_argument(
        "--split-preflight-manifest",
        type=Path,
        help=(
            "Required for real training: successful SPLIT_PREFLIGHT.json produced "
            "from the same raw inputs and exact candidate universe."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-events", type=int, default=64)
    parser.add_argument("--event-feature-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--mc-samples", type=int, default=16)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    return parser


def _resolve(root: Path, path: Path | None) -> Path | None:
    if path is None:
        return None
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    sys.path.insert(0, str(root))
    from cc_hhgt.v32.evidence_training import (
        run_evidence_split_preflight,
        run_evidence_training,
        run_formal_evidence_dry_run,
    )

    common = {
        "evidence_event_path": _resolve(root, args.evidence_event),
        "interaction_relation_path": _resolve(root, args.interaction_relation),
        "pathway_member_path": _resolve(root, args.pathway_members),
        "core_embedding_root": _resolve(root, args.core_embedding_root),
        "output_root": _resolve(root, args.output_root),
        "id_map_path": _resolve(root, args.id_map),
        "candidates_path": _resolve(root, args.candidates),
        "seed": args.seed,
        "event_feature_dim": args.event_feature_dim,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
    }
    if args.formal_dry_run:
        manifest = run_formal_evidence_dry_run(
            **common,
            expected_candidate_rows=args.expected_candidate_rows,
            expected_cancer_count=args.expected_cancer_count,
            expected_candidate_sha256=args.expected_candidate_sha256,
        )
    elif args.split_preflight:
        manifest = run_evidence_split_preflight(
            evidence_event_path=common["evidence_event_path"],
            interaction_relation_path=common["interaction_relation_path"],
            pathway_member_path=common["pathway_member_path"],
            output_root=common["output_root"],
            id_map_path=common["id_map_path"],
            candidates_path=common["candidates_path"],
            seed=args.seed,
            expected_candidate_rows=args.expected_candidate_rows,
            expected_cancer_count=args.expected_cancer_count,
            expected_candidate_sha256=args.expected_candidate_sha256,
        )
    else:
        if args.split_preflight_manifest is None:
            parser.error("--split-preflight-manifest is required for real Evidence training")
        manifest = run_evidence_training(
            **common,
            split_preflight_manifest_path=_resolve(root, args.split_preflight_manifest),
            expected_candidate_rows=args.expected_candidate_rows,
            expected_cancer_count=args.expected_cancer_count,
            expected_candidate_sha256=args.expected_candidate_sha256,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            max_events=args.max_events,
            mc_samples=args.mc_samples,
            device=args.device,
        )
    print(
        json.dumps(
            {
                "status": (
                    "AUDIT_EXECUTION_SUCCESS"
                    if args.formal_dry_run
                    else manifest.get("status", manifest["training_status"])
                ),
                "training_status": manifest["training_status"],
                "output_root": str(_resolve(root, args.output_root)),
                "counts": manifest["counts"],
                "data_gaps": manifest.get("data_gaps"),
                "main_ranking_modified": manifest.get("main_ranking_modified", False),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
