#!/usr/bin/env python3
"""Validate the CPU-created G0/G1/G2 authorization receipt without fold hashing."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from normalize_v32_task_manifest import validate_task_manifest  # noqa: E402


class StaticAuthorizationError(RuntimeError):
    """The no-GPU authorization receipt is absent, stale, or malformed."""


FORMAL_SEED = 20260726


def _stable_bytes(path: Path, label: str) -> bytes:
    """Read one regular file from one descriptor and reject concurrent drift."""

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise StaticAuthorizationError(
            f"STATIC_AUTH_ARTIFACT_OPEN_FAILED={label}:{path}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
            raise StaticAuthorizationError(
                f"STATIC_AUTH_ARTIFACT_NOT_NONEMPTY_REGULAR={label}:{path}"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_after != identity_before:
            raise StaticAuthorizationError(
                f"STATIC_AUTH_ARTIFACT_CHANGED_DURING_READ={label}:{path}"
            )
        data = b"".join(chunks)
        if len(data) != before.st_size:
            raise StaticAuthorizationError(
                f"STATIC_AUTH_ARTIFACT_SHORT_READ={label}:{path}"
            )
        try:
            observed_path = os.stat(path, follow_symlinks=False)
        except OSError as exc:
            raise StaticAuthorizationError(
                f"STATIC_AUTH_ARTIFACT_PATH_DRIFT={label}:{path}"
            ) from exc
        if (observed_path.st_dev, observed_path.st_ino) != (
            before.st_dev,
            before.st_ino,
        ):
            raise StaticAuthorizationError(
                f"STATIC_AUTH_ARTIFACT_PATH_REPLACED={label}:{path}"
            )
        return data
    finally:
        os.close(descriptor)


def _sha256(path: Path) -> str:
    return hashlib.sha256(_stable_bytes(path, str(path))).hexdigest()


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StaticAuthorizationError(f"STATIC_AUTH_MAPPING_REQUIRED={label}")
    return dict(value)


def _load_json(
    path: Path,
    label: str,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    try:
        data = _stable_bytes(path, label)
        if expected_sha256 is not None:
            observed = hashlib.sha256(data).hexdigest()
            if observed != expected_sha256:
                raise StaticAuthorizationError(
                    f"STATIC_AUTH_ARTIFACT_SHA_DRIFT={label}"
                )
        payload = json.loads(data.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StaticAuthorizationError(f"STATIC_AUTH_JSON_INVALID={label}:{path}") from exc
    return _mapping(payload, label)


def validate_static_authorization(
    *,
    receipt_path: Path,
    code_root: Path,
    prepared_parent: Path,
    run_parent: Path,
) -> dict[str, Any]:
    code_root = code_root.resolve()
    prepared_parent = prepared_parent.resolve()
    run_parent = run_parent.resolve()
    receipt = _load_json(receipt_path, "STATIC_AUTH_READY")
    if receipt.get("status") != "STATIC_AUTH_READY":
        raise StaticAuthorizationError("STATIC_AUTH_STATUS_DRIFT")
    if receipt.get("gpu_visible") is not False:
        raise StaticAuthorizationError("STATIC_AUTH_WAS_NOT_CPU_ONLY")
    runtime_contract = _mapping(receipt.get("runtime_contract"), "runtime_contract")
    if runtime_contract.get("cuda_available_during_cpu_gate") is not False:
        raise StaticAuthorizationError("STATIC_AUTH_RUNTIME_WAS_NOT_CPU_ONLY")
    if receipt.get("formal_seed") != FORMAL_SEED:
        raise StaticAuthorizationError("STATIC_AUTH_FORMAL_SEED_DRIFT")
    for key in ("python", "torch", "torch_geometric", "pyyaml"):
        if not str(runtime_contract.get(key, "")).strip():
            raise StaticAuthorizationError(f"STATIC_AUTH_RUNTIME_VERSION_MISSING={key}")
    variants = _mapping(receipt.get("variants"), "variants")
    if set(variants) != {"G0", "G1", "G2"}:
        raise StaticAuthorizationError(f"STATIC_AUTH_VARIANT_DRIFT={sorted(variants)}")

    required_code_contract = {
        "cc_hhgt/v32/training.py",
        "cc_hhgt/v32/training_guard.py",
        "cc_hhgt/v32/gpu_backward_probe.py",
        "scripts/cloud_apply_v32_g012_code_patch_no_gpu_20260901_r1.sh",
        "scripts/cloud_authorize_v32_g012_no_gpu_20260901_r1.sh",
        "scripts/cloud_finalize_v32_g012_no_gpu_20260901_r1.sh",
        "scripts/server_launch_v32_g012_paid_gpu_20260831_r1.sh",
        "scripts/normalize_v32_task_manifest.py",
        "scripts/validate_v32_g012_static_auth_ready.py",
        "config/model_v3_2_g012_paid_gpu_20260831_r1.yaml",
    }
    code_contract = _mapping(receipt.get("code_contract_sha256"), "code_contract_sha256")
    if not required_code_contract.issubset(code_contract):
        missing = sorted(required_code_contract - set(code_contract))
        raise StaticAuthorizationError(f"STATIC_AUTH_CODE_CONTRACT_MISSING={missing}")
    for relative in sorted(required_code_contract):
        path = code_root / relative
        if not path.is_file() or _sha256(path) != code_contract[relative]:
            raise StaticAuthorizationError(f"STATIC_AUTH_CODE_DRIFT={relative}")

    patch_ready = receipt_path.parent / "PATCH_READY.json"
    if not patch_ready.is_file() or patch_ready.stat().st_size == 0:
        raise StaticAuthorizationError(f"STATIC_AUTH_PATCH_READY_MISSING={patch_ready}")
    patch_payload = _load_json(
        patch_ready,
        "PATCH_READY",
        expected_sha256=str(receipt.get("patch_ready_sha256", "")),
    )
    if patch_payload.get("status") != "PATCH_READY":
        raise StaticAuthorizationError("STATIC_AUTH_PATCH_READY_STATUS_DRIFT")
    if patch_payload.get("gpu_visible") is not False:
        raise StaticAuthorizationError("STATIC_AUTH_PATCH_READY_WAS_NOT_CPU_ONLY")
    if patch_payload.get("code_root") != str(code_root):
        raise StaticAuthorizationError("STATIC_AUTH_PATCH_READY_CODE_ROOT_DRIFT")
    overlay_sha256 = str(patch_payload.get("overlay_sha256", ""))
    if len(overlay_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in overlay_sha256
    ):
        raise StaticAuthorizationError("STATIC_AUTH_PATCH_OVERLAY_SHA_INVALID")

    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))
    from cc_hhgt.v32.training_guard import code_tree_sha256

    observed_code_tree_sha256 = code_tree_sha256(code_root)
    if patch_payload.get("code_tree_sha256") != observed_code_tree_sha256:
        raise StaticAuthorizationError("STATIC_AUTH_PATCH_CODE_TREE_SHA_DRIFT")
    if receipt.get("code_tree_sha256") != observed_code_tree_sha256:
        raise StaticAuthorizationError("STATIC_AUTH_CODE_TREE_SHA_DRIFT")

    for variant in ("G0", "G1", "G2"):
        run_id = f"v32-g012-{variant.lower()}-paid-gpu-20260831-r1"
        record = _mapping(variants[variant], f"variants.{variant}")
        expected_record = {
            "run_id": run_id,
            "tasks": 5,
            "folds": list(range(5)),
            "seed": FORMAL_SEED,
        }
        for key, value in expected_record.items():
            if record.get(key) != value:
                raise StaticAuthorizationError(
                    f"STATIC_AUTH_VARIANT_FIELD_DRIFT={variant}:{key}"
                )
        authorization = run_parent / variant / "authorization"
        artifacts = {
            "task_manifest_sha256": authorization / "TASK_MANIFEST.tsv",
            "input_manifest_sha256": authorization / "INPUT_MANIFEST.json",
            "approval_sha256": authorization / "TRAINING_APPROVAL.json",
            "success_sha256": authorization / "SUCCESS.json",
        }
        for key, path in artifacts.items():
            if not path.is_file() or path.stat().st_size == 0:
                raise StaticAuthorizationError(
                    f"STATIC_AUTH_ARTIFACT_MISSING={variant}:{path}"
                )
            if _sha256(path) != record.get(key):
                raise StaticAuthorizationError(
                    f"STATIC_AUTH_ARTIFACT_SHA_DRIFT={variant}:{key}"
                )

        rows = validate_task_manifest(
            artifacts["task_manifest_sha256"],
            run_id=run_id,
            owner="paid_gpu",
            hardware_class="PAID_PREEMPTIBLE_GPU",
            paid_task="true",
            expected_folds=range(5),
        )
        if any(row.get("seed") != str(FORMAL_SEED) for row in rows):
            raise StaticAuthorizationError(f"STATIC_AUTH_TASK_SEED_DRIFT={variant}")
        input_payload = _load_json(
            artifacts["input_manifest_sha256"],
            f"input.{variant}",
            expected_sha256=str(record.get("input_manifest_sha256", "")),
        )
        fold_inputs = input_payload.get("fold_inputs")
        if not isinstance(fold_inputs, list) or len(fold_inputs) != 5:
            raise StaticAuthorizationError(f"STATIC_AUTH_INPUT_COUNT_DRIFT={variant}")
        receipt_fold_inputs = record.get("fold_inputs")
        if not isinstance(receipt_fold_inputs, list) or len(receipt_fold_inputs) != 5:
            raise StaticAuthorizationError(
                f"STATIC_AUTH_INPUT_BINDING_COUNT_DRIFT={variant}"
            )
        receipt_by_fold: dict[int, dict[str, Any]] = {}
        for raw_binding in receipt_fold_inputs:
            binding = _mapping(raw_binding, f"receipt.{variant}.fold")
            try:
                binding_fold = int(binding.get("fold", -1))
            except (TypeError, ValueError) as exc:
                raise StaticAuthorizationError(
                    f"STATIC_AUTH_INPUT_BINDING_FOLD_INVALID={variant}"
                ) from exc
            if binding_fold in receipt_by_fold:
                raise StaticAuthorizationError(
                    f"STATIC_AUTH_INPUT_BINDING_DUPLICATE={variant}:{binding_fold}"
                )
            receipt_by_fold[binding_fold] = binding
        if set(receipt_by_fold) != set(range(5)):
            raise StaticAuthorizationError(
                f"STATIC_AUTH_INPUT_BINDING_FOLD_DRIFT={variant}"
            )

        observed_folds: set[int] = set()
        for item in fold_inputs:
            entry = _mapping(item, f"input.{variant}.fold")
            try:
                fold = int(entry.get("fold", -1))
            except (TypeError, ValueError) as exc:
                raise StaticAuthorizationError(
                    f"STATIC_AUTH_INPUT_FOLD_INVALID={variant}"
                ) from exc
            observed_folds.add(fold)
            expected_path = prepared_parent / variant / f"PATIENT_FOLD_{fold}.pt"
            declared_path = entry.get("path")
            if declared_path != str(expected_path):
                raise StaticAuthorizationError(
                    f"STATIC_AUTH_INPUT_PATH_DRIFT={variant}:{fold}"
                )
            if expected_path.is_symlink() or not expected_path.is_file():
                raise StaticAuthorizationError(
                    f"STATIC_AUTH_INPUT_MISSING_OR_SYMLINK={variant}:{fold}:{expected_path}"
                )
            try:
                resolved_path = expected_path.resolve(strict=True)
            except OSError as exc:
                raise StaticAuthorizationError(
                    f"STATIC_AUTH_INPUT_RESOLVE_FAILED={variant}:{fold}"
                ) from exc
            if resolved_path != expected_path:
                raise StaticAuthorizationError(
                    f"STATIC_AUTH_INPUT_ESCAPES_PREPARED_ROOT={variant}:{fold}"
                )
            observed_stat = expected_path.stat()
            if not stat.S_ISREG(observed_stat.st_mode) or observed_stat.st_size <= 0:
                raise StaticAuthorizationError(
                    f"STATIC_AUTH_INPUT_NOT_NONEMPTY_REGULAR={variant}:{fold}"
                )
            digest = str(entry.get("sha256", ""))
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise StaticAuthorizationError(
                    f"STATIC_AUTH_INPUT_SHA_INVALID={variant}:{fold}"
                )
            binding = receipt_by_fold[fold]
            if (
                binding.get("path") != str(expected_path)
                or binding.get("sha256") != digest
                or binding.get("size_bytes") != observed_stat.st_size
            ):
                raise StaticAuthorizationError(
                    f"STATIC_AUTH_INPUT_BINDING_DRIFT={variant}:{fold}"
                )
        if observed_folds != set(range(5)):
            raise StaticAuthorizationError(
                f"STATIC_AUTH_INPUT_FOLD_DRIFT={variant}:{sorted(observed_folds)}"
            )

        approval = _load_json(
            artifacts["approval_sha256"],
            f"approval.{variant}",
            expected_sha256=str(record.get("approval_sha256", "")),
        )
        if approval.get("training_authorized") is not True:
            raise StaticAuthorizationError(f"STATIC_AUTH_APPROVAL_FALSE={variant}")
        expected_approval = {
            "run_id": run_id,
            "endpoint_id": "paid_gpu",
            "hardware_class": "PAID_PREEMPTIBLE_GPU",
            "authorized_trainer": "cc_hhgt.v32.training:run_authorized_task",
            "approved_task_ids": [row["task_id"] for row in rows],
            "paid_enabled": True,
            "max_paid_hours": 96,
            "max_cost_cny": 210,
        }
        for key, value in expected_approval.items():
            if approval.get(key) != value:
                raise StaticAuthorizationError(
                    f"STATIC_AUTH_APPROVAL_DRIFT={variant}:{key}"
                )
        artifact_hashes = _mapping(approval.get("artifact_hashes"), f"hashes.{variant}")
        if artifact_hashes.get("code_sha256") != observed_code_tree_sha256:
            raise StaticAuthorizationError(
                f"STATIC_AUTH_APPROVAL_CODE_SHA_DRIFT={variant}"
            )
        for key, path in (
            ("task_manifest_sha256", artifacts["task_manifest_sha256"]),
            ("input_manifest_sha256", artifacts["input_manifest_sha256"]),
            ("config_sha256", run_parent / variant / "config.yaml"),
        ):
            if artifact_hashes.get(key) != _sha256(path):
                raise StaticAuthorizationError(
                    f"STATIC_AUTH_APPROVAL_HASH_DRIFT={variant}:{key}"
                )
        success = _load_json(
            artifacts["success_sha256"],
            f"success.{variant}",
            expected_sha256=str(record.get("success_sha256", "")),
        )
        if success.get("status") != "AUTHORIZED_ARTIFACTS_READY":
            raise StaticAuthorizationError(f"STATIC_AUTH_SUCCESS_STATUS_DRIFT={variant}")
        if success.get("artifact_hashes") != artifact_hashes:
            raise StaticAuthorizationError(f"STATIC_AUTH_SUCCESS_HASH_DRIFT={variant}")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--code-root", required=True, type=Path)
    parser.add_argument("--prepared-parent", required=True, type=Path)
    parser.add_argument("--run-parent", required=True, type=Path)
    args = parser.parse_args(argv)
    validate_static_authorization(
        receipt_path=args.receipt.resolve(),
        code_root=args.code_root.resolve(),
        prepared_parent=args.prepared_parent.resolve(),
        run_parent=args.run_parent.resolve(),
    )
    print(f"PASS_STATIC_AUTH_READY={args.receipt.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
