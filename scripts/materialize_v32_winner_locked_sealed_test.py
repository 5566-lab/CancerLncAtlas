#!/usr/bin/env python3
"""Materialize hash-bound V3.2 test logits only after G0/G1/G2 winner lock."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--winner-declaration", required=True)
    parser.add_argument("--winner-declaration-sha256", required=True)
    parser.add_argument("--sealed-test-manifest", required=True)
    parser.add_argument("--patient-folds", required=True)
    parser.add_argument("--patient-fold-authority-receipt", required=True)
    parser.add_argument("--expected-graph-variant", choices=("G0", "G1", "G2"), default="G2")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)

    root = Path(args.repo_root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from cc_hhgt.v32.sealed_test_inference import (
        materialize_winner_locked_sealed_test,
    )

    result = materialize_winner_locked_sealed_test(
        prepared_root=args.prepared_root,
        winner_declaration_path=args.winner_declaration,
        winner_declaration_sha256=args.winner_declaration_sha256,
        sealed_test_manifest_path=args.sealed_test_manifest,
        patient_folds_path=args.patient_folds,
        patient_fold_authority_receipt_path=args.patient_fold_authority_receipt,
        output_root=args.output_root,
        expected_graph_variant=args.expected_graph_variant,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
