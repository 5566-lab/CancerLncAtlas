#!/usr/bin/env python
"""Independently audit a formal V3.2 Drug mechanism release."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.drug_mechanism_audit import audit_drug_mechanism_release  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--audit-output", required=True, type=Path)
    args = parser.parse_args()
    result = audit_drug_mechanism_release(
        args.manifest,
        expected_manifest_sha256=args.expected_manifest_sha256,
        audit_output=args.audit_output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
