"""Materialize the versioned G0/G1/G2 optimizer-guard variant.

The production ``cc_hhgt/v32/training.py`` is deliberately left untouched.
This helper takes a *copy* of that file, replaces only the old
``optimizer_step_with_guards`` implementation with a delegating wrapper, and
copies :mod:`cc_hhgt.v32.optimizer_guard_v2` next to it.  It is intended for
building a new isolated code namespace before static authorization; it never
starts a trainer or imports Torch.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


VARIANT_ID = "v32_g012_zero_delta_guard_20260905_r1"
PATCH_FORMAT = "CC_HHGT_V3_2_ZERO_DELTA_GUARD_MATERIALIZATION_V1"
OLD_FAILURE_MARKER = "ZERO_PARAMETER_DELTA_AFTER_STEP={probe_name}"
GUARD_MODULE_NAME = "optimizer_guard_v2.py"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _function_span(source: str, name: str) -> tuple[int, int]:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            if node.end_lineno is None:  # pragma: no cover - Python >=3.10 guarantee
                raise ValueError(f"AST did not expose end line for {name}")
            return node.lineno - 1, node.end_lineno
    raise ValueError(f"Required function is absent: {name}")


def materialize(
    *,
    training_source: Path,
    training_output: Path,
    guard_module_source: Path,
    guard_module_output: Path,
    receipt_output: Path | None = None,
) -> dict[str, object]:
    """Write a new training file and return a tamper-evident receipt."""

    if training_source.resolve() == training_output.resolve():
        raise ValueError(
            "training source/output must be different; refusing to overwrite the old code"
        )
    if guard_module_source.resolve() == guard_module_output.resolve():
        raise ValueError(
            "guard module source/output must be different; refusing to overwrite the source"
        )
    if training_output.exists() or guard_module_output.exists():
        raise FileExistsError(
            "variant output already exists; choose a fresh versioned namespace"
        )

    source = training_source.read_text(encoding="utf-8")
    if OLD_FAILURE_MARKER not in source:
        raise ValueError(
            "training source does not contain the expected pre-fix guard marker; "
            "refusing to patch an unknown code version"
        )
    if "optimizer_step_with_global_delta_guard" in source:
        raise ValueError("training source already references the V2 guard")

    start, end = _function_span(source, "optimizer_step_with_guards")
    replacement = [
        "from .optimizer_guard_v2 import optimizer_step_with_global_delta_guard",
        "",
        "",
        "def optimizer_step_with_guards(",
        "    model,",
        "    optimizer,",
        "    *,",
        "    torch,",
        "    objective_value: float,",
        ") -> dict[str, Any]:",
        "    \"\"\"Versioned V2 global representable-delta guard wrapper.\"\"\"",
        "",
        "    return optimizer_step_with_global_delta_guard(",
        "        model,",
        "        optimizer,",
        "        torch=torch,",
        "        objective_value=objective_value,",
        "    )",
    ]
    output_lines = source.splitlines(keepends=True)
    output_lines[start:end] = [line + "\n" for line in replacement]
    patched = "".join(output_lines)

    # The import is intentionally local to the generated training module.  The
    # source code's existing ``Any`` import is retained, and all old code is
    # preserved byte-for-byte outside the one function span.
    ast.parse(patched)
    module_source = guard_module_source.read_bytes()
    guard_module_output.parent.mkdir(parents=True, exist_ok=True)
    training_output.parent.mkdir(parents=True, exist_ok=True)
    guard_module_output.write_bytes(module_source)
    training_output.write_text(patched, encoding="utf-8", newline="\n")

    receipt: dict[str, object] = {
        "format": PATCH_FORMAT,
        "variant_id": VARIANT_ID,
        "status": "PASS",
        "source_training": str(training_source),
        "output_training": str(training_output),
        "source_training_sha256": _sha256(training_source),
        "output_training_sha256": _sha256(training_output),
        "guard_module_source": str(guard_module_source),
        "guard_module_output": str(guard_module_output),
        "guard_module_sha256": _sha256(guard_module_output),
        "replaced_function": "optimizer_step_with_guards",
        "old_failure_marker": OLD_FAILURE_MARKER,
        "torch_imported": False,
        "paid_compute_started": False,
    }
    if receipt_output is not None:
        receipt_output.parent.mkdir(parents=True, exist_ok=True)
        receipt_output.write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        receipt["receipt_sha256"] = _sha256(receipt_output)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-source", type=Path, required=True)
    parser.add_argument("--training-output", type=Path, required=True)
    parser.add_argument("--guard-module-source", type=Path, required=True)
    parser.add_argument("--guard-module-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path)
    args = parser.parse_args()
    receipt = materialize(
        training_source=args.training_source,
        training_output=args.training_output,
        guard_module_source=args.guard_module_source,
        guard_module_output=args.guard_module_output,
        receipt_output=args.receipt_output,
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
