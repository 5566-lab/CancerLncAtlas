#!/usr/bin/env python3
"""Build V3.2 Drug raw-long staging directly from native source files."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize provenance-locked native GDSC/PRISM/CMP/DrugCentral inputs for V3.2"
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--gdsc1", required=True)
    parser.add_argument("--gdsc2", required=True)
    parser.add_argument("--prism-matrix", required=True)
    parser.add_argument("--prism-compounds", required=True)
    parser.add_argument("--cmp-expression", required=True)
    parser.add_argument("--cmp-models-json", required=True)
    parser.add_argument("--drugcentral-target", required=True)
    parser.add_argument("--hgnc", required=True)
    parser.add_argument("--exact-candidates", required=True)
    parser.add_argument("--pathway-membership", required=True)
    parser.add_argument(
        "--exact-release-prediction",
        required=True,
        help="Newly trained V3.2 exact-pathway release prediction used as identity proof",
    )
    parser.add_argument(
        "--exact-release-lineage",
        required=True,
        help="MODULE_LINEAGE.json for the exact release prediction proof",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--staging-run-id", required=True)
    parser.add_argument("--csv-chunk-rows", type=int, default=100_000)
    parser.add_argument("--prism-chunk-drugs", type=int, default=250)
    parser.add_argument("--expression-chunk-genes", type=int, default=1_000)
    parser.add_argument("--max-prism-drugs", type=int)
    parser.add_argument("--max-expression-lncrnas", type=int)
    parser.add_argument("--max-models", type=int)
    parser.add_argument(
        "--candidate-storage-mode",
        choices=("auto", "factored", "materialized"),
        default="auto",
        help="Formal auto mode stores a factored universe instead of a huge Cartesian table",
    )
    return parser


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(args.repo_root).resolve()
    sys.path.insert(0, str(root))
    from cc_hhgt.v32.drug_staging import DrugStagingConfig, run_native_drug_staging

    result = run_native_drug_staging(
        gdsc1_path=_resolve(root, args.gdsc1),
        gdsc2_path=_resolve(root, args.gdsc2),
        prism_matrix_path=_resolve(root, args.prism_matrix),
        prism_compound_path=_resolve(root, args.prism_compounds),
        cmp_expression_path=_resolve(root, args.cmp_expression),
        cmp_models_json_path=_resolve(root, args.cmp_models_json),
        drugcentral_target_path=_resolve(root, args.drugcentral_target),
        hgnc_path=_resolve(root, args.hgnc),
        exact_candidates_path=_resolve(root, args.exact_candidates),
        pathway_membership_path=_resolve(root, args.pathway_membership),
        exact_release_prediction_path=_resolve(root, args.exact_release_prediction),
        exact_release_lineage_path=_resolve(root, args.exact_release_lineage),
        output_root=_resolve(root, args.output_root),
        staging_run_id=args.staging_run_id,
        config=DrugStagingConfig(
            csv_chunk_rows=args.csv_chunk_rows,
            prism_chunk_drugs=args.prism_chunk_drugs,
            expression_chunk_genes=args.expression_chunk_genes,
            max_prism_drugs=args.max_prism_drugs,
            max_expression_lncrnas=args.max_expression_lncrnas,
            max_models=args.max_models,
            candidate_storage_mode=args.candidate_storage_mode,
        ),
        execution_code_paths=(Path(__file__).resolve(),),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
