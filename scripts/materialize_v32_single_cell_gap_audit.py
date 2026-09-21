"""Materialize the read-only V3.2 single-cell gap audit release."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.single_cell_gap_audit import (  # noqa: E402
    SingleCellGapInputs,
    materialize_single_cell_gap_audit,
)


def default_inputs(repo_root: Path) -> SingleCellGapInputs:
    artifacts = repo_root / "artifacts"
    explicit = artifacts / "single_cell_lncrna_expression_reaudit_20260826_r2_explicit_ids"
    return SingleCellGapInputs(
        remote_observation=repo_root
        / "config"
        / "v32_single_cell_gap_remote_observation_20260826.json",
        remote_live_recheck=artifacts
        / "v32_single_cell_gap_remote_live_recheck_20260826_r1.json",
        lncrna_table=explicit / "single_cell_lncrna_expression_33c_explicit_ids.tsv",
        lncrna_summary=explicit / "AUDIT_SUMMARY.json",
        lncrna_binding=explicit / "SINGLE_CELL_EXPLICIT_ID_REAUDIT_BINDING.json",
        hnsc_ucell_binding=artifacts
        / "single_cell_cell_level_hnsc_ucell_query_binding_20260826_r1"
        / "HNSC_UCELL_QUERY_BINDING.json",
        figure_manifest=artifacts
        / "v32_single_cell_fresh_20260826_r1"
        / "FIGURE_MANIFEST.json",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = materialize_single_cell_gap_audit(
        inputs=default_inputs(args.repo_root.resolve()),
        output_root=args.output_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
