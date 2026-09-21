#!/usr/bin/env python3
"""Audit an already-running, isolated CancerLncAtlas candidate."""

from __future__ import annotations

import argparse
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_RUNTIME_MODULE_PATH = ROOT / "cc_hhgt/v32/runtime_web_acceptance.py"
_RUNTIME_SPEC = spec_from_file_location(
    "v32_runtime_web_acceptance_standalone", _RUNTIME_MODULE_PATH
)
if _RUNTIME_SPEC is None or _RUNTIME_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"Cannot load runtime acceptance module: {_RUNTIME_MODULE_PATH}")
_RUNTIME_MODULE = module_from_spec(_RUNTIME_SPEC)
sys.modules[_RUNTIME_SPEC.name] = _RUNTIME_MODULE
_RUNTIME_SPEC.loader.exec_module(_RUNTIME_MODULE)
execute_runtime_audit = _RUNTIME_MODULE.execute_runtime_audit
load_final_binding_evidence = _RUNTIME_MODULE.load_final_binding_evidence
render_markdown = _RUNTIME_MODULE.render_markdown
report_exit_code = _RUNTIME_MODULE.report_exit_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bounded 51-route/33-cancer acceptance audit. The target must be "
            "a loopback candidate and port 8260 is always rejected."
        )
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8262")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout-seconds", type=float, default=15.0)
    parser.add_argument("--max-body-mib", type=float, default=2.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New or existing directory for RUNTIME_ACCEPTANCE.json/.md",
    )
    parser.add_argument(
        "--gate",
        choices=("full", "http", "report-only"),
        default="full",
        help=(
            "full also requires scientific artifacts; http still requires the "
            "data-processing invariants; report-only never returns a gate failure"
        ),
    )
    parser.add_argument(
        "--final-binding",
        type=Path,
        help="Hash-bound final V3.2 winner/release binding (required for full PASS)",
    )
    parser.add_argument(
        "--final-binding-sha256",
        help="Expected SHA-256 of --final-binding",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help="Allowed root containing the final binding and all declared artifacts",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    max_body_bytes = int(args.max_body_mib * 1024 * 1024)
    binding_args = (
        args.final_binding,
        args.final_binding_sha256,
        args.artifact_root,
    )
    if any(value is not None for value in binding_args) and not all(
        value is not None for value in binding_args
    ):
        raise SystemExit(
            "--final-binding, --final-binding-sha256 and --artifact-root must be supplied together"
        )
    final_binding_evidence = None
    if args.final_binding is not None:
        final_binding_evidence = load_final_binding_evidence(
            args.final_binding,
            expected_sha256=args.final_binding_sha256,
            artifact_root=args.artifact_root,
        )
    report = execute_runtime_audit(
        args.base_url,
        workers=args.workers,
        timeout_seconds=args.timeout_seconds,
        max_body_bytes=max_body_bytes,
        final_binding_evidence=final_binding_evidence,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "RUNTIME_ACCEPTANCE.json"
    markdown_path = args.output_dir / "RUNTIME_ACCEPTANCE.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "json": str(json_path.resolve()),
                "markdown": str(markdown_path.resolve()),
                "verdicts": report["verdicts"],
                "production_port_8260_touched": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return report_exit_code(report, args.gate)


if __name__ == "__main__":
    raise SystemExit(main())
