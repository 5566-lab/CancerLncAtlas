#!/usr/bin/env python3
"""Materialize the unchanged primary arm after validation winner lock."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--primary-frame", required=True)
    parser.add_argument("--external-validation-predictions", required=True)
    parser.add_argument("--routing-winner-declaration", required=True)
    parser.add_argument("--routing-winner-declaration-sha256", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    root = Path(args.repo_root).resolve()
    sys.path.insert(0, str(root))
    from cc_hhgt.v32.routing_validation_winner import (
        materialize_primary_outer_after_winner_lock,
    )

    result = materialize_primary_outer_after_winner_lock(
        primary_frame_path=args.primary_frame,
        external_validation_predictions_path=args.external_validation_predictions,
        routing_winner_declaration_path=args.routing_winner_declaration,
        routing_winner_declaration_sha256=args.routing_winner_declaration_sha256,
        output_root=args.output_root,
        run_id=args.run_id,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
