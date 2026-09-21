#!/usr/bin/env python3
"""Bind a tested r9 candidate's BH implementation to the final secure runner."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
from typing import Any


FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_R9_FINAL_RUNNER_EQUIVALENCE_BINDING_V1"
FUNCTIONS = (
    "_sql_literal",
    "_release_transient_memory",
    "_evidence_schema",
    "_finalize_association_evidence_in_process",
)
CONSTANTS = (
    "PARQUET_BATCH_ROWS",
    "ASSOCIATION_DUCKDB_MEMORY_LIMIT",
    "ASSOCIATION_ENGINE",
    "ASSOCIATION_EXECUTION_STRATEGY",
    "ASSOCIATION_BH_FAMILY",
    "ASSOCIATION_STAGE_KEY_POLICY",
)


class RunnerBindingError(RuntimeError):
    """Raised when the tested and final BH implementations differ."""


def sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise RunnerBindingError(f"absent or unsafe file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def selected_ast(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions: dict[str, str] = {}
    constants: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in FUNCTIONS:
            functions[node.name] = ast.dump(node, annotate_fields=True, include_attributes=False)
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            for target in targets:
                if isinstance(target, ast.Name) and target.id in CONSTANTS:
                    constants[target.id] = ast.dump(
                        value, annotate_fields=True, include_attributes=False
                    )
    if set(functions) != set(FUNCTIONS):
        raise RunnerBindingError(f"required BH function AST set missing in {path}")
    if set(constants) != set(CONSTANTS):
        raise RunnerBindingError(f"required BH constant AST set missing in {path}")
    return functions, constants


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RunnerBindingError(f"absent or unsafe JSON receipt: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RunnerBindingError(f"JSON receipt is not an object: {path}")
    return value


def exclusive_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tested-runner", required=True, type=Path)
    parser.add_argument("--final-runner", required=True, type=Path)
    parser.add_argument("--auditor-resource-failure", required=True, type=Path)
    parser.add_argument("--independent-receipt", required=True, type=Path)
    parser.add_argument("--negative-gate-receipt", required=True, type=Path)
    parser.add_argument("--interruption-audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise RunnerBindingError(f"output reuse forbidden: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    tested = args.tested_runner.resolve()
    final = args.final_runner.resolve()
    auditor_failure = args.auditor_resource_failure.resolve()
    independent = args.independent_receipt.resolve()
    negative_gate = args.negative_gate_receipt.resolve()
    interruption = args.interruption_audit.resolve()
    if load_json(auditor_failure).get("status") != (
        "AUDITOR_RESOURCE_FAILURE_NOT_DATA_OR_RESULT_FAILURE"
    ):
        raise RunnerBindingError("auditor resource-failure classification is invalid")
    if load_json(independent).get("status") != (
        "PASS_EXACT_LARGE_FIXTURE_EQUIVALENCE"
    ):
        raise RunnerBindingError("independent large-fixture equivalence did not pass")
    if load_json(negative_gate).get("status") != "PASS":
        raise RunnerBindingError("negative handoff gates did not pass")
    if load_json(interruption).get("status") != "PASS":
        raise RunnerBindingError("interruption safety audit did not pass")
    tested_functions, tested_constants = selected_ast(tested)
    final_functions, final_constants = selected_ast(final)
    if tested_functions != final_functions:
        raise RunnerBindingError("tested and final BH function ASTs differ")
    if tested_constants != final_constants:
        raise RunnerBindingError("tested and final BH constants differ")
    function_shas = {
        name: digest_text(tested_functions[name]) for name in FUNCTIONS
    }
    constant_shas = {
        name: digest_text(tested_constants[name]) for name in CONSTANTS
    }
    value = {
        "format": FORMAT,
        "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
        "status": "PASS_FINAL_RUNNER_BH_IMPLEMENTATION_IDENTICAL_TO_TESTED_RUNNER",
        "tested_runner_path": str(tested),
        "tested_runner_sha256": sha256_file(tested),
        "final_runner_path": str(final),
        "final_runner_sha256": sha256_file(final),
        "whole_runner_files_identical": sha256_file(tested) == sha256_file(final),
        "selected_bh_function_asts_identical": True,
        "selected_bh_constant_asts_identical": True,
        "function_ast_sha256": function_shas,
        "constant_ast_sha256": constant_shas,
        "failed_aggregate_auditor_classification_path": str(auditor_failure),
        "failed_aggregate_auditor_classification_sha256": sha256_file(
            auditor_failure
        ),
        "independent_receipt_path": str(independent),
        "independent_receipt_sha256": sha256_file(independent),
        "negative_gate_receipt_path": str(negative_gate),
        "negative_gate_receipt_sha256": sha256_file(negative_gate),
        "interruption_audit_path": str(interruption),
        "interruption_audit_sha256": sha256_file(interruption),
        "binding_scope": (
            "BH_RANK_CONSERVATIVE_ADJUST_EXTERNAL_SORT_SCHEMA_AND_LIMIT_CONSTANTS"
        ),
        "security_hardening_outside_bound_bh_scope_permitted": True,
        "production_deployed": False,
    }
    value["binding_contract_sha256"] = canonical_sha256(value)
    exclusive_json(output, value)
    print(
        json.dumps(
            {**value, "output_sha256": sha256_file(output)},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
