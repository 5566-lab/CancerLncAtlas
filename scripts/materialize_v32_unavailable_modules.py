#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.unavailable_modules import (
    materialize_drug_permission_audit,
    materialize_interaction_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize honest unavailable V3.2 modules")
    parser.add_argument("--physical-facts", required=True)
    parser.add_argument("--static-drug-target", required=True)
    parser.add_argument("--core-embedding-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    root = Path(args.output_root)
    result = {
        "interaction": materialize_interaction_audit(
            physical_facts_path=args.physical_facts,
            core_manifest_path=args.core_embedding_manifest,
            output_root=root / "interaction",
        ),
        "drug": materialize_drug_permission_audit(
            static_drug_target_path=args.static_drug_target,
            core_manifest_path=args.core_embedding_manifest,
            output_root=root / "drug",
        ),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
