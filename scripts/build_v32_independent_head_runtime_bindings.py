#!/usr/bin/env python3
from __future__ import annotations

import argparse

from cc_hhgt.v32.independent_head_runtime_binding import build_runtime_bindings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--portable-overlay-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--server-root", required=True)
    parser.add_argument("--clinical-expression-server-root")
    parser.add_argument("--clinical-curves-server-root")
    args = parser.parse_args()
    result = build_runtime_bindings(
        portable_overlay_root=args.portable_overlay_root,
        output_root=args.output_root,
        server_root=args.server_root,
        clinical_expression_server_root=args.clinical_expression_server_root,
        clinical_curves_server_root=args.clinical_curves_server_root,
    )
    print(result["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
