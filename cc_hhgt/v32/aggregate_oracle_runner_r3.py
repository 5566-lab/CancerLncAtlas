"""Aggregate-archive-only entry point for the bounded G2/Fold0 Oracle.

The surrounding launcher authenticates one input tar and one code tar.  This
module deliberately accepts neither an input manifest nor file-level hashes.
It writes one scientific receipt atomically; the launcher then packages that
receipt and the JSONL execution log into the single result tar.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import group_shared_encoder_oracle as oracle


RUN_ID = "v32-g012-g2-group-shared-oracle-paid-gpu-20260901-r3"
TASK_ID = f"{RUN_ID}|PATIENT_FOLD_0|CC-HHGT|20260726"
SHA256_ALPHABET = frozenset("0123456789abcdef")


def _sha256(value: str, label: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(
        character not in SHA256_ALPHABET for character in normalized
    ):
        raise ValueError(f"ORACLE_R3_{label}_SHA256_INVALID={value}")
    return normalized


def _regular_file(path: str | Path, label: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"ORACLE_R3_{label}_NOT_REGULAR={resolved}")
    return resolved


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    if partial.exists() or path.exists():
        raise RuntimeError(f"ORACLE_R3_RESULT_TARGET_ALREADY_EXISTS={path}")
    with partial.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(
            dict(value),
            handle,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    partial.replace(path)


def _stable_archive_member_load(torch, prepared_path: Path) -> Any:
    """Deserialize the authenticated tar member without hashing that member."""

    prepared = prepared_path.resolve()
    if not prepared.is_file() or prepared.is_symlink():
        raise RuntimeError(f"ORACLE_R3_PREPARED_MEMBER_NOT_REGULAR={prepared}")
    with prepared.open("rb") as handle:
        before_stat = os.fstat(handle.fileno())
        before = (
            before_stat.st_dev,
            before_stat.st_ino,
            before_stat.st_mode,
            before_stat.st_size,
            before_stat.st_mtime_ns,
            before_stat.st_ctime_ns,
        )
        payload = torch.load(handle, map_location="cpu", weights_only=False)
        after_stat = os.fstat(handle.fileno())
        after = (
            after_stat.st_dev,
            after_stat.st_ino,
            after_stat.st_mode,
            after_stat.st_size,
            after_stat.st_mtime_ns,
            after_stat.st_ctime_ns,
        )
        if after != before:
            raise RuntimeError("ORACLE_R3_PREPARED_MEMBER_CHANGED_WHILE_LOADING")
    return payload


def _run_archive_only_oracle(context: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt the frozen R2 scientific callable without changing its file.

    R2 remains byte-identical for reproducibility.  In this isolated process,
    the adapter replaces its file-level loaders/checks with the already-verified
    aggregate archive bindings, adds archive lineage to heartbeats, and emits
    the first optimizer-step evidence at the actual guarded update.
    """

    from . import training

    prepared = Path(str(context["prepared_path"])).resolve()
    input_sha = str(context["aggregate_input_archive_sha256"])
    code_sha = str(context["aggregate_code_archive_sha256"])
    original_resolve = training._resolve_prepared_path
    original_load = training.load_prepared_artifact_from_authorized_handle
    original_file_sha = training._file_sha256
    original_step = training.optimizer_step_with_guards
    original_emit = oracle._emit_receipt
    first_step_emitted = False

    def resolve_prepared(_config, _fold, _root):
        return prepared

    def load_prepared(torch, _input_manifest_path, *, fold, prepared_path):
        if int(fold) != oracle.EXPECTED_FOLD or Path(prepared_path).resolve() != prepared:
            raise RuntimeError("ORACLE_R3_PREPARED_MEMBER_BINDING_DRIFT")
        return (
            _stable_archive_member_load(torch, prepared),
            "NOT_COMPUTED_AGGREGATE_INPUT_ARCHIVE_ONLY",
        )

    def aggregate_code_marker(_path):
        return "NOT_COMPUTED_AGGREGATE_CODE_ARCHIVE_ONLY"

    def emit_with_aggregate_lineage(value):
        enriched = dict(value)
        enriched.update(
            {
                "aggregate_input_archive_sha256": input_sha,
                "aggregate_code_archive_sha256": code_sha,
                "integrity_scope": "INPUT_CODE_RESULT_ARCHIVES_ONLY",
                "per_file_hashing_performed": False,
                "per_fold_hashing_performed": False,
            }
        )
        original_emit(enriched)

    def optimizer_step_with_evidence(*args, **kwargs):
        nonlocal first_step_emitted
        result = original_step(*args, **kwargs)
        if not first_step_emitted:
            torch = kwargs.get("torch")
            gpu_name = "UNAVAILABLE"
            if torch is not None and torch.cuda.is_available():
                gpu_name = str(torch.cuda.get_device_name(torch.cuda.current_device()))
            original_emit(
                {
                    **oracle._base_receipt("ORACLE_R3_FIRST_OPTIMIZER_STEP"),
                    "run_id": str(context["run_id"]),
                    "task_id": str(context["task_id"]),
                    "patient_fold": oracle.EXPECTED_FOLD,
                    "seed": oracle.EXPECTED_SEED,
                    "graph_variant": oracle.EXPECTED_VARIANT,
                    "optimizer_step": 1,
                    "objective": float(kwargs["objective_value"]),
                    "grad_finite": bool(result["grad_finite"]),
                    "parameters_finite": bool(result["parameters_finite"]),
                    "parameter_delta_positive": bool(
                        result["parameter_delta_positive"]
                    ),
                    "gpu_identity": {
                        "source": "torch.cuda",
                        "name": gpu_name,
                    },
                    "aggregate_input_archive_sha256": input_sha,
                    "aggregate_code_archive_sha256": code_sha,
                    "integrity_scope": "INPUT_CODE_RESULT_ARCHIVES_ONLY",
                    "per_file_hashing_performed": False,
                    "per_fold_hashing_performed": False,
                }
            )
            first_step_emitted = True
        return result

    training._resolve_prepared_path = resolve_prepared
    training.load_prepared_artifact_from_authorized_handle = load_prepared
    training._file_sha256 = aggregate_code_marker
    training.optimizer_step_with_guards = optimizer_step_with_evidence
    oracle._emit_receipt = emit_with_aggregate_lineage
    try:
        receipt = oracle._run_verified_oracle_comparison(dict(context))
    finally:
        training._resolve_prepared_path = original_resolve
        training.load_prepared_artifact_from_authorized_handle = original_load
        training._file_sha256 = original_file_sha
        training.optimizer_step_with_guards = original_step
        oracle._emit_receipt = original_emit
    receipt.update(
        {
            "prepared_artifact_sha256": (
                "NOT_COMPUTED_AGGREGATE_INPUT_ARCHIVE_ONLY"
            ),
            "preregistration_sha256": (
                "NOT_COMPUTED_AGGREGATE_CODE_ARCHIVE_ONLY"
            ),
            "oracle_runner_sha256": (
                "NOT_COMPUTED_AGGREGATE_CODE_ARCHIVE_ONLY"
            ),
            "production_training_sha256": (
                "NOT_COMPUTED_AGGREGATE_CODE_ARCHIVE_ONLY"
            ),
            "authorization_artifact_hashes": {},
            "aggregate_input_archive_sha256": input_sha,
            "aggregate_code_archive_sha256": code_sha,
            "integrity_scope": "INPUT_CODE_RESULT_ARCHIVES_ONLY",
            "per_file_hashing_performed": False,
            "per_fold_hashing_performed": False,
        }
    )
    return receipt


def run(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    if not repo_root.is_dir():
        raise ValueError(f"ORACLE_R3_REPO_ROOT_MISSING={repo_root}")
    config_path = _regular_file(args.config, "CONFIG")
    task_manifest_path = _regular_file(args.task_manifest, "TASK_MANIFEST")
    prepared_path = _regular_file(args.prepared, "PREPARED_MEMBER")
    if prepared_path.name != "PATIENT_FOLD_0.pt" or prepared_path.parent.name != "G2":
        raise ValueError(
            f"ORACLE_R3_PREPARED_MEMBER_SCOPE_DRIFT={prepared_path}"
        )
    for label, path in (
        ("CONFIG", config_path),
        ("TASK_MANIFEST", task_manifest_path),
    ):
        try:
            path.relative_to(repo_root)
        except ValueError as exc:
            raise ValueError(
                f"ORACLE_R3_{label}_OUTSIDE_CODE_ARCHIVE={path}"
            ) from exc

    input_sha = _sha256(args.input_archive_sha256, "INPUT_ARCHIVE")
    code_sha = _sha256(args.code_archive_sha256, "CODE_ARCHIVE")
    output_path = Path(args.output).resolve()
    context = {
        "repo_root": str(repo_root),
        "config_path": str(config_path),
        "task_manifest_path": str(task_manifest_path),
        "task_id": TASK_ID,
        "run_id": RUN_ID,
        "prepared_path": str(prepared_path),
        "aggregate_archive_only": True,
        "aggregate_input_archive_sha256": input_sha,
        "aggregate_code_archive_sha256": code_sha,
        "input_manifest_path": None,
        "artifact_hashes": {},
    }
    try:
        receipt = _run_archive_only_oracle(context)
    except Exception as exc:
        receipt = {
            **oracle._base_receipt(oracle.RUNTIME_FAIL_STATUS),
            "scientific_pass": False,
            "run_id": RUN_ID,
            "task_id": TASK_ID,
            "patient_fold": oracle.EXPECTED_FOLD,
            "seed": oracle.EXPECTED_SEED,
            "graph_variant": oracle.EXPECTED_VARIANT,
            "error_type": type(exc).__name__,
            "reason": str(exc),
            "aggregate_input_archive_sha256": input_sha,
            "aggregate_code_archive_sha256": code_sha,
            "integrity_scope": "INPUT_CODE_RESULT_ARCHIVES_ONLY",
            "per_file_hashing_performed": False,
            "per_fold_hashing_performed": False,
        }
    _atomic_json(output_path, receipt)
    print(
        json.dumps(
            {
                "format": "CANCERLNCATLAS_V32_ORACLE_AGGREGATE_RUN_R3_V1",
                "status": "ORACLE_R3_SCIENTIFIC_RESULT_WRITTEN",
                "scientific_status": receipt.get("status"),
                "scientific_pass": receipt.get("scientific_pass") is True,
                "output_path": str(output_path),
                "aggregate_input_archive_sha256": input_sha,
                "aggregate_code_archive_sha256": code_sha,
                "per_file_hashing_performed": False,
                "per_fold_hashing_performed": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    return 0 if receipt.get("scientific_pass") is True else oracle.NONZERO_RETURN_CODE


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--task-manifest", required=True)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--input-archive-sha256", required=True)
    parser.add_argument("--code-archive-sha256", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run(_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
