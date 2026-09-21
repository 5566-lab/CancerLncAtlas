#!/usr/bin/env python3
'''Create the immutable V3.2 33-cancer single-cell expression audit binding.'''
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
V32_MODULE_ROOT = ROOT / 'cc_hhgt' / 'v32'
if str(V32_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(V32_MODULE_ROOT))

from single_cell_expression_query import (  # noqa: E402
    build_single_cell_expression_binding,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-summary', type=Path, required=True)
    parser.add_argument('--expression-table', type=Path, required=True)
    parser.add_argument('--formal-input-gate', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_single_cell_expression_binding(
        audit_summary_path=args.audit_summary,
        expression_table_path=args.expression_table,
        formal_gate_path=args.formal_input_gate,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                'binding_path': result['binding_path'],
                'binding_sha256': result['binding_sha256'],
                'success_path': result['success_path'],
                'cancer_count': result['binding']['cancer_count'],
                'cell_count': result['binding']['cell_count'],
                'single_cell_module_complete': False,
                'release_ready': False,
                'production_deployed': False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == '__main__':
    main()
