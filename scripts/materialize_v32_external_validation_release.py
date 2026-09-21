#!/usr/bin/env python3
"""Materialize fresh V3.2 known-positive rank-recovery validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from cc_hhgt.v32.external_validation_release import (  # noqa: E402
    materialize_external_validation_release,
)


HISTORICAL_TASK_REPO = REPO.parent / "CC_HHGT_GPU_4070TiS_v2_3_multiseed_external_validation"


def _resolve(root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO)
    parser.add_argument(
        "--historical-task-root", type=Path, default=HISTORICAL_TASK_REPO
    )
    parser.add_argument(
        "--task-definition",
        type=Path,
        default=Path(
            "results/model/cc_hhgt_v2_1_gpu/standardized/"
            "external_validation_evidence.parquet"
        ),
    )
    parser.add_argument(
        "--lnc2cancer",
        type=Path,
        default=Path("input/lnc2cancer3_human_lncRNA.tsv"),
    )
    parser.add_argument(
        "--lncrnadisease",
        type=Path,
        default=Path("input/lncrnadisease3_human_lncRNA.tsv"),
    )
    parser.add_argument(
        "--rnadisease-experimental",
        type=Path,
        default=Path("input/rnadisease4_human_lncRNA_experimental.tsv"),
    )
    parser.add_argument(
        "--rnadisease-predicted",
        type=Path,
        default=Path("input/rnadisease4_lncRNA_predicted.tsv"),
    )
    parser.add_argument(
        "--gse85011",
        type=Path,
        default=Path("input/gse85011_sample_metadata.tsv"),
    )
    parser.add_argument(
        "--training-evidence",
        type=Path,
        default=REPO.parent / "processed" / "evidence_event.parquet",
    )
    parser.add_argument(
        "--training-interaction",
        type=Path,
        default=(
            REPO.parent
            / "CC_HHGT_v2_8_gdc_star"
            / "input_snapshot"
            / "parquet"
            / "interaction_relation.parquet"
        ),
    )
    parser.add_argument(
        "--current-prediction",
        type=Path,
        default=Path(
            "artifacts/v32_full_multitask/exact_pathway_release_r2/"
            "exact_pathway_five_fold_ensemble.parquet"
        ),
    )
    parser.add_argument(
        "--current-prediction-lineage",
        type=Path,
        default=Path(
            "artifacts/v32_full_multitask/exact_pathway_release_r2/"
            "MODULE_LINEAGE.json"
        ),
    )
    parser.add_argument(
        "--current-candidates",
        type=Path,
        default=Path("artifacts/formal_prepared/FORMAL_CANDIDATE_UNIVERSE.parquet"),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--fixture-authority",
        action="store_true",
        help="Allow synthetic fixture hashes/counts; never use for formal materialization.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo_root.resolve()
    task_root = args.historical_task_root.resolve()
    raw_sources = {
        "lnc2cancer": _resolve(task_root, args.lnc2cancer),
        "lncrnadisease": _resolve(task_root, args.lncrnadisease),
        "rnadisease_experimental": _resolve(
            task_root, args.rnadisease_experimental
        ),
        "rnadisease_predicted": _resolve(task_root, args.rnadisease_predicted),
        "gse85011": _resolve(task_root, args.gse85011),
    }
    result = materialize_external_validation_release(
        task_definition_path=_resolve(task_root, args.task_definition),
        raw_source_paths=raw_sources,
        training_evidence_path=_resolve(repo, args.training_evidence),
        training_interaction_path=_resolve(repo, args.training_interaction),
        current_prediction_path=_resolve(repo, args.current_prediction),
        current_prediction_lineage_path=_resolve(
            repo, args.current_prediction_lineage
        ),
        current_candidate_path=_resolve(repo, args.current_candidates),
        output_root=_resolve(repo, args.output),
        runner_path=Path(__file__),
        strict_formal_authority=not args.fixture_authority,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
