#!/usr/bin/env python
"""Run the registry-gated V3.2 mixed exact-pathway query from the command line."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.mixed_query import MixedExactPathwayQuery


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--member", action="append", default=[], help="Repeat for each input ID/symbol")
    parser.add_argument("--members-file", type=Path, help="Optional newline/comma-delimited input list")
    parser.add_argument("--cancer-id", help="Optional cancer; omission returns PAN_CANCER_MEAN")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output", type=Path, help="Optional JSON output; stdout when omitted")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    members = list(args.member)
    if args.members_file:
        text = args.members_file.read_text(encoding="utf-8")
        members.extend(item.strip() for item in re.split(r"[,\r\n]+", text) if item.strip())
    engine = MixedExactPathwayQuery.from_registry(args.registry)
    result = engine.query(members, cancer_id=args.cancer_id, top_k=args.top_k)
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
