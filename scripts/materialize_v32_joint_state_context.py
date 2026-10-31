#!/usr/bin/env python3
"""Publish/query assets from five completed joint Pathway/State folds."""
from __future__ import annotations

import argparse
import json

from cc_hhgt.v32.current_state_context import materialize_current_state_context


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Materialize joint State tables, descriptive GeneSets and selected-member profiles")
    value.add_argument("--manifest", required=True,
        help="CURRENT_JOINT_STATE_CONTEXT_INPUT_V1 manifest")
    value.add_argument("--output", required=True)
    return value


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    print(json.dumps(materialize_current_state_context(args.manifest, args.output),
        ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
