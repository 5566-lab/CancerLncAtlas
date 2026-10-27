"""Write the machine-checked code acceptance record for the handoff report.

Static bookkeeping only: no scientific computation, no host gate, no training.
"""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.code_acceptance import accept, write_acceptance  # noqa: E402

PROVENANCE = ROOT / "provenance/code_completion_20261015"
DEFAULT_OUTPUT = Path("D:/model/audit/requirements_status_20261026/CODE_ACCEPTANCE.json")


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    output = Path(args[0]).resolve() if args else DEFAULT_OUTPUT
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True)
    value = accept(ROOT / "CODE_COMPLETION_PLAN.json", PROVENANCE / "VALIDATION_SUMMARY.json",
                   PROVENANCE, ROOT, head_commit=head, worktree_status=status)
    write_acceptance(output, value)
    print(json.dumps({key: value[key] for key in (
        "status", "requirement_count", "verified_count", "open_count", "open_requirements",
        "latest_case_count", "cumulative_case_count", "source_pin_count", "checks_applied",
        "head_commit")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
