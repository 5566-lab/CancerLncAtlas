#!/usr/bin/env python3
"""Re-infer the locked G012 core on inner-validation rows only."""
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
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    root = Path(args.repo_root).resolve()
    sys.path.insert(0, str(root))
    from cc_hhgt.v32.validation_prediction_inference import (
        materialize_g012_validation_predictions,
    )

    result = materialize_g012_validation_predictions(
        repo_root=root,
        prepared_root=args.prepared_root,
        winner_declaration_path=args.winner_declaration,
        winner_declaration_sha256=args.winner_declaration_sha256,
        output_root=args.output_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
