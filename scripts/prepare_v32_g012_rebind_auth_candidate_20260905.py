#!/usr/bin/env python3
"""Bind a completed r5 input rebind to a *blocked* launch candidate.

This utility is deliberately not a training authorizer.  It is intended to run
on host 149 after ``rebind_v32_g012_local_cnv_manifest_20260904.py`` has
written its final ``SUCCESS.json``.  It reads the small rebind manifests and
uses ``lstat`` for the fifteen payloads; it never opens or hashes a ``.pt``
file, never calls a provider API, and never writes ``TRAINING_APPROVAL.json``.

With ``--emit-blocked-candidate`` it writes only an immutable, fail-closed
candidate receipt and a pending preflight.  The command exits 42 after writing
those receipts so callers cannot mistake the candidate for launch approval.
Without that flag it performs the same checks but does not create output.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


HOST_EXPECTED = "149"
WORK_ROOT = Path("${PRIVATE_WORK_ROOT}/CancerLncAtlas_v32_staging_20260904_r3")
PREPARED_ROOT = Path(
    "./data/CancerLncAtlas/inputs/"
    "v32_g012_local_cnv_formal_prepared_20260903_r1"
)
REBIN_ROOT = WORK_ROOT / "g012_rebind_20260904_r5"
DEFAULT_OUTPUT_ROOT = WORK_ROOT / "g012_auth_gate_20260905_r1" / "receipt"
DEFAULT_CODE_ARCHIVE = Path(
    "./data/CancerLncAtlas/runtime/code_archives/"
    "v32_local_cnv_core_corrected_code_20260903_r1.tar.gz"
)
VARIANTS = ("G0", "G1", "G2")
FOLDS = tuple(range(5))
SUCCESS_STATUS = "PASS_INPUT_REBIND_MANIFEST_AND_PREFLIGHT_ONLY"
MANIFEST_STATUS = "PASS_INPUT_HASHES_BOUND_NO_TRAINING"
PREFLIGHT_STATUS = "BLOCKED_INPUT_REBIND_ONLY_NO_LAUNCH"


class GateError(RuntimeError):
    """A fail-closed contract violation."""


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        c in "0123456789abcdef" for c in value
    )


def canonical_json(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )


def digest_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def ensure_no_symlink_components(path: Path) -> None:
    """Reject a symlink in any extant path component."""

    path = Path(os.path.abspath(path))
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            item = os.lstat(current)
        except FileNotFoundError:
            # The caller may be checking a not-yet-created output.  Existing
            # components have already been checked; the missing suffix is
            # handled by the caller.
            break
        if stat.S_ISLNK(item.st_mode):
            raise GateError(f"SYMLINK_COMPONENT_FORBIDDEN={current}")


def ensure_regular(path: Path, label: str, *, allow_missing: bool = False) -> os.stat_result | None:
    ensure_no_symlink_components(path)
    try:
        item = os.lstat(path)
    except FileNotFoundError:
        if allow_missing:
            return None
        raise GateError(f"MISSING_{label}={path}")
    if stat.S_ISLNK(item.st_mode) or not stat.S_ISREG(item.st_mode):
        raise GateError(f"NOT_REGULAR_{label}={path}")
    if item.st_nlink != 1:
        raise GateError(f"HARDLINK_FORBIDDEN_{label}={path}")
    if item.st_size <= 0:
        raise GateError(f"EMPTY_{label}={path}")
    return item


def sha256_file(path: Path, label: str) -> str:
    """Hash a declared small authority file, never a payload."""

    ensure_regular(path, label)
    if path.name.endswith(".pt"):
        raise GateError(f"PAYLOAD_READ_FORBIDDEN={path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path, label: str) -> dict[str, Any]:
    ensure_regular(path, label)
    if path.name.endswith(".pt"):
        raise GateError(f"PAYLOAD_READ_FORBIDDEN={path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GateError(f"INVALID_JSON_{label}={path}") from exc
    if not isinstance(value, dict):
        raise GateError(f"JSON_OBJECT_REQUIRED_{label}={path}")
    return value


def path_from_ref(ref: Any, expected: Path, label: str) -> Path:
    if not isinstance(ref, dict):
        raise GateError(f"REFERENCE_OBJECT_REQUIRED={label}")
    raw = ref.get("path")
    if raw != str(expected):
        raise GateError(f"REFERENCE_PATH_DRIFT={label}:{raw!r}!={str(expected)!r}")
    declared = ref.get("sha256")
    if not is_sha(declared):
        raise GateError(f"REFERENCE_SHA_INVALID={label}")
    observed = sha256_file(expected, label)
    if observed != declared:
        raise GateError(f"REFERENCE_SHA_DRIFT={label}:{observed}")
    return expected


def check_payload_row(row: dict[str, Any], prepared: Path, label: str) -> tuple[str, int]:
    variant = row.get("variant")
    try:
        fold = int(row.get("fold"))
    except (TypeError, ValueError) as exc:
        raise GateError(f"FOLD_ID_INVALID={label}") from exc
    if variant not in VARIANTS or fold not in FOLDS:
        raise GateError(f"FOLD_KEY_INVALID={label}:{variant}:{fold}")
    expected = prepared / str(variant) / f"PATIENT_FOLD_{fold}.pt"
    if row.get("path") != str(expected) or row.get("relative_path") != f"{variant}/PATIENT_FOLD_{fold}.pt":
        raise GateError(f"PAYLOAD_PATH_DRIFT={label}")
    declared_bytes = row.get("bytes")
    if not isinstance(declared_bytes, int) or declared_bytes <= 0:
        raise GateError(f"PAYLOAD_SIZE_INVALID={label}")
    declared_sha = row.get("sha256")
    if not is_sha(declared_sha):
        raise GateError(f"PAYLOAD_SHA_INVALID={label}")
    # Metadata only: this does not read the payload bytes.
    item = ensure_regular(expected, label)
    assert item is not None
    if item.st_size != declared_bytes:
        raise GateError(f"PAYLOAD_SIZE_DRIFT={label}:{item.st_size}!={declared_bytes}")
    if row.get("corrected_local_cnv") is not True:
        raise GateError(f"PAYLOAD_CORRECTION_FLAG_DRIFT={label}")
    return str(variant), fold


def validate_rebind(rebind_root: Path, prepared: Path) -> dict[str, Any]:
    """Validate r5's small receipts and payload metadata without PT reads."""

    success_path = rebind_root / "SUCCESS.json"
    success = read_json(success_path, "REBIND_SUCCESS")
    if success.get("format") != "CANCERLNCATLAS_V32_G012_CORRECTED_LOCAL_CNV_REBIND_SUCCESS_V1":
        raise GateError("REBIND_SUCCESS_FORMAT_DRIFT")
    if success.get("status") != SUCCESS_STATUS:
        raise GateError(f"REBIND_SUCCESS_NOT_FINAL={success.get('status')!r}")
    if success.get("output_root") != str(rebind_root):
        raise GateError("REBIND_SUCCESS_ROOT_DRIFT")
    if success.get("host") != HOST_EXPECTED:
        raise GateError("REBIND_SUCCESS_HOST_DRIFT")
    if success.get("fold_artifact_count") != 15:
        raise GateError(f"REBIND_FOLD_COUNT_DRIFT={success.get('fold_artifact_count')!r}")
    for key in (
        "training_started",
        "gpu_started",
        "sealed_test_read",
        "training_approval_emitted",
        "production_result_written",
    ):
        if success.get(key) is not False:
            raise GateError(f"REBIND_UNSAFE_FLAG={key}:{success.get(key)!r}")

    refs = {
        "manifest": ("rebound_input_manifest", rebind_root / "REBOUND_INPUT_MANIFEST.json"),
        "plan": ("rebind_plan", rebind_root / "REBIND_PLAN.json"),
        "preflight": ("launch_preflight", rebind_root / "LAUNCH_PREFLIGHT.json"),
        "sums": ("sha256sums", rebind_root / "SHA256SUMS.tsv"),
    }
    for label, (key, expected) in refs.items():
        path_from_ref(success.get(key), expected, f"REBIND_{label.upper()}")

    manifest = read_json(rebind_root / "REBOUND_INPUT_MANIFEST.json", "REBIND_MANIFEST")
    if manifest.get("status") != MANIFEST_STATUS:
        raise GateError(f"REBIND_MANIFEST_STATUS_DRIFT={manifest.get('status')!r}")
    if manifest.get("prepared_root") != str(prepared):
        raise GateError("REBIND_MANIFEST_PREPARED_ROOT_DRIFT")
    rows = manifest.get("fold_inputs")
    if not isinstance(rows, list) or len(rows) != 15:
        raise GateError("REBIND_MANIFEST_REQUIRES_EXACT_15")
    keys: set[tuple[str, int]] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise GateError("REBIND_MANIFEST_ROW_OBJECT_REQUIRED")
        key = check_payload_row(row, prepared, "REBIND_MANIFEST_ROW")
        if key in keys:
            raise GateError(f"REBIND_MANIFEST_DUPLICATE={key}")
        keys.add(key)
    if keys != {(variant, fold) for variant in VARIANTS for fold in FOLDS}:
        raise GateError("REBIND_MANIFEST_KEY_SET_DRIFT")

    plan = read_json(rebind_root / "REBIND_PLAN.json", "REBIND_PLAN")
    if plan.get("status") != "INPUT_REBIND_READY_LAUNCH_BLOCKED":
        raise GateError(f"REBIND_PLAN_STATUS_DRIFT={plan.get('status')!r}")
    if plan.get("prepared_root") != str(prepared) or plan.get("fold_artifact_count") != 15:
        raise GateError("REBIND_PLAN_BINDING_DRIFT")
    # r5's immutable plan schema predates the explicit top-level
    # ``launch_permitted`` field; absence is therefore the safe (blocked)
    # value.  Never treat a truthy value as missing—only an absent key gets
    # the fail-closed default.
    if plan.get("training_approval_emitted") is not False or plan.get("launch_permitted", False) is not False:
        raise GateError("REBIND_PLAN_UNSAFE_FLAG")

    preflight = read_json(rebind_root / "LAUNCH_PREFLIGHT.json", "REBIND_PREFLIGHT")
    if preflight.get("format") != "CANCERLNCATLAS_V32_G012_CORRECTED_LOCAL_CNV_LAUNCH_PREFLIGHT_V1":
        raise GateError("REBIND_PREFLIGHT_FORMAT_DRIFT")
    if preflight.get("status") != PREFLIGHT_STATUS:
        raise GateError(f"REBIND_PREFLIGHT_STATUS_DRIFT={preflight.get('status')!r}")
    for key in (
        "training_started",
        "gpu_started",
        "optimizer_progress_observed",
        "gpu_telemetry_observed",
        "sealed_test_read",
        "training_approval_emitted",
        "paid_compute_requested",
        "launch_permitted",
    ):
        if preflight.get(key) is not False:
            raise GateError(f"REBIND_PREFLIGHT_UNSAFE_FLAG={key}:{preflight.get(key)!r}")

    variants: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        variant_dir = rebind_root / variant
        input_path = variant_dir / "INPUT_MANIFEST.json"
        task_path = variant_dir / "TASK_MANIFEST.tsv"
        input_doc = read_json(input_path, f"{variant}_INPUT_MANIFEST")
        input_rows = input_doc.get("fold_inputs")
        if not isinstance(input_rows, list) or len(input_rows) != 5:
            raise GateError(f"{variant}_INPUT_MANIFEST_FOLD_COUNT")
        variant_keys: set[int] = set()
        for row in input_rows:
            if not isinstance(row, dict):
                raise GateError(f"{variant}_INPUT_MANIFEST_ROW_OBJECT")
            if row.get("variant", variant) not in (variant, None):
                raise GateError(f"{variant}_INPUT_MANIFEST_VARIANT_DRIFT")
            key = check_payload_row(
                {
                    **row,
                    "variant": variant,
                    "relative_path": f"{variant}/PATIENT_FOLD_{int(row.get('fold'))}.pt",
                    # The per-variant r5 INPUT_MANIFEST intentionally keeps
                    # this flag at the combined-manifest level.  Reinsert
                    # the already-validated immutable contract field for the
                    # shared row checker; do not infer or alter any payload.
                    "corrected_local_cnv": True,
                },
                prepared,
                f"{variant}_INPUT_MANIFEST_ROW",
            )
            variant_keys.add(key[1])
        if variant_keys != set(FOLDS):
            raise GateError(f"{variant}_INPUT_MANIFEST_FOLD_SET")
        ensure_regular(task_path, f"{variant}_TASK_MANIFEST")
        try:
            with task_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                task_rows = list(reader)
        except (OSError, UnicodeError, csv.Error) as exc:
            raise GateError(f"INVALID_{variant}_TASK_MANIFEST") from exc
        required_columns = [
            "task_id", "run_id", "task_type", "model", "patient_fold", "seed",
            "owner", "hardware_class", "status", "blocked_reason", "paid_task",
        ]
        if reader.fieldnames != required_columns or len(task_rows) != 5:
            raise GateError(f"{variant}_TASK_MANIFEST_SCHEMA")
        for row in task_rows:
            if (
                row.get("owner") != "pending_gpu"
                or row.get("status") != "PENDING_REBOUND_PREFLIGHT"
                or row.get("blocked_reason") != "corrected_input_bound_only; launch_not_authorized"
                or row.get("paid_task") != "false"
            ):
                raise GateError(f"{variant}_TASK_MANIFEST_NOT_BLOCKED")
        variants[variant] = {
            "input_manifest_path": str(input_path),
            "input_manifest_sha256": sha256_file(input_path, f"{variant}_INPUT_MANIFEST"),
            "task_manifest_path": str(task_path),
            "task_manifest_sha256": sha256_file(task_path, f"{variant}_TASK_MANIFEST"),
            "run_id": plan.get("variants", {}).get(variant, {}).get("run_id"),
            "tasks": 5,
            "folds": list(FOLDS),
        }

    # A paid approval must not already be hiding in the rebind output.
    for forbidden in ("TRAINING_APPROVAL.json", "TRAINER_APPROVAL.json"):
        if (rebind_root / forbidden).exists():
            raise GateError(f"UNEXPECTED_PAID_APPROVAL={rebind_root / forbidden}")
        for variant in VARIANTS:
            if (rebind_root / variant / forbidden).exists():
                raise GateError(f"UNEXPECTED_PAID_APPROVAL={rebind_root / variant / forbidden}")

    return {
        "success_path": str(success_path),
        "success_sha256": sha256_file(success_path, "REBIND_SUCCESS"),
        "manifest_path": str(rebind_root / "REBOUND_INPUT_MANIFEST.json"),
        "manifest_sha256": sha256_file(rebind_root / "REBOUND_INPUT_MANIFEST.json", "REBIND_MANIFEST"),
        "plan_path": str(rebind_root / "REBIND_PLAN.json"),
        "plan_sha256": sha256_file(rebind_root / "REBIND_PLAN.json", "REBIND_PLAN"),
        "preflight_path": str(rebind_root / "LAUNCH_PREFLIGHT.json"),
        "preflight_sha256": sha256_file(rebind_root / "LAUNCH_PREFLIGHT.json", "REBIND_PREFLIGHT"),
        "fold_artifact_count": 15,
        "variants": variants,
    }


def optional_file_binding(path: Path | None, label: str) -> dict[str, Any]:
    if path is None:
        return {"status": "MISSING_ARGUMENT", "path": None, "sha256": None}
    if not path.exists():
        return {"status": "MISSING_FILE", "path": str(path), "sha256": None}
    return {"status": "DECLARED_FILE_HASHED", "path": str(path), "sha256": sha256_file(path, label)}


def code_binding(args: argparse.Namespace) -> dict[str, Any]:
    root = args.code_root
    if root is None:
        root_record: dict[str, Any] = {"status": "MISSING_ARGUMENT", "path": None}
    elif not root.is_dir() or root.is_symlink():
        root_record = {"status": "MISSING_OR_INVALID_DIRECTORY", "path": str(root)}
    else:
        ensure_no_symlink_components(root)
        root_record = {"status": "DECLARED_DIRECTORY_ONLY_NOT_RECURSIVELY_HASHED", "path": str(root)}
    tree_sha = args.code_tree_sha256
    if tree_sha is not None and not is_sha(tree_sha):
        raise GateError("CODE_TREE_SHA_INVALID")
    return {
        "code_root": root_record,
        "code_tree_sha256": tree_sha,
        "code_tree_status": "DECLARED_SHA_REQUIRED" if tree_sha is None else "DECLARED_SHA_FORMAT_VALID_NOT_RECOMPUTED",
        "code_archive": optional_file_binding(args.code_archive, "CODE_ARCHIVE"),
        "config": optional_file_binding(args.config, "TRAINING_CONFIG"),
        "launcher": optional_file_binding(args.launcher, "TRAINING_LAUNCHER"),
    }


def write_new(path: Path, raw: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(raw)
    except FileExistsError as exc:
        raise GateError(f"REFUSING_OVERWRITE={path}") from exc
    return digest_bytes(raw)


def emit_receipts(
    output_root: Path,
    *,
    rebind: dict[str, Any],
    prepared: Path,
    rebind_root: Path,
    code: dict[str, Any],
    instance_id: str,
) -> list[Path]:
    if output_root.exists():
        if output_root.is_symlink() or not output_root.is_dir() or any(output_root.iterdir()):
            raise GateError(f"OUTPUT_ROOT_NOT_ABSENT_OR_EMPTY={output_root}")
    ensure_no_symlink_components(output_root.parent)
    output_root.mkdir(parents=True, exist_ok=False)
    stamp = now_utc()
    candidate = {
        "format": "CANCERLNCATLAS_V32_G012_REBIND_AUTHORIZATION_CANDIDATE_V1",
        "status": "BLOCKED_REBIND_BOUND_NO_TRAINING_AUTHORIZATION",
        "created_at_utc": stamp,
        "host_expected": HOST_EXPECTED,
        "host_observed": platform.node().split(".", 1)[0],
        "instance_id": instance_id,
        "instance_state": "NOT_QUERIED_BY_THIS_UTILITY",
        "prepared_root": str(prepared),
        "rebind_root": str(rebind_root),
        "rebind_success": rebind,
        "code_binding": code,
        "training_approval_emitted": False,
        "launch_permitted": False,
        "paid_compute_requested": False,
        "gpu_started": False,
        "sealed_test_read": False,
        "next_required_actions": [
            "Complete a no-GPU code/config/launcher binding with immutable hashes.",
            "Run the separate no-GPU static authorization validator against this corrected root.",
            "Query and verify the stopped RTX 4090 provider state and fresh preflight.",
            "Obtain an explicit project-level training decision before any paid approval.",
        ],
    }
    preflight = {
        "format": "CANCERLNCATLAS_V32_G012_REBIND_LAUNCH_PREFLIGHT_V2",
        "status": "BLOCKED_NO_GPU_STATIC_PREFLIGHT_PENDING",
        "created_at_utc": stamp,
        "host_expected": HOST_EXPECTED,
        "host_observed": platform.node().split(".", 1)[0],
        "instance_id": instance_id,
        "prepared_root": str(prepared),
        "rebind_success_sha256": rebind["success_sha256"],
        "rebound_input_manifest_sha256": rebind["manifest_sha256"],
        "variant_input_task_manifest_hashes": code.get("variant_manifests", rebind["variants"]),
        "code_config_launcher_bound": False,
        "provider_state_checked": False,
        "no_gpu_static_authorization": False,
        "optimizer_progress_observed": False,
        "gpu_telemetry_observed": False,
        "training_approval_emitted": False,
        "paid_compute_requested": False,
        "gpu_started": False,
        "launch_permitted": False,
        "blockers": [
            "candidate is not a TRAINING_APPROVAL and cannot start a job",
            "code/config/launcher must be independently hash-bound",
            "provider stopped-state and no-GPU launch preflight have not been queried",
            "formal training decision, fair-comparison gates, and winner lock remain pending",
        ],
    }
    checklist = {
        "format": "CANCERLNCATLAS_V32_G012_REBIND_AUTH_CHECKLIST_V1",
        "status": "BLOCKED_CHECKLIST_ONLY",
        "r5_success_required": True,
        "exact_fold_count": 15,
        "payload_bytes_read_by_this_utility": 0,
        "provider_api_called_by_this_utility": False,
        "training_approval_written_by_this_utility": False,
        "old_authorization_reuse_permitted": False,
        "instance_id": instance_id,
        "rebind_success_sha256": rebind["success_sha256"],
    }
    files: list[Path] = []
    for name, payload in (
        ("AUTHORIZATION_CANDIDATE_BLOCKED.json", candidate),
        ("LAUNCH_PREFLIGHT_PENDING.json", preflight),
        ("BINDING_CHECKLIST.json", checklist),
    ):
        path = output_root / name
        write_new(path, canonical_json(payload))
        files.append(path)
    sums = []
    for path in files:
        sums.append(f"{sha256_file(path, path.name)}\t{path.stat().st_size}\t{path.name}")
    sums_path = output_root / "SHA256SUMS.tsv"
    write_new(sums_path, ("\n".join(sums) + "\n").encode("utf-8"))
    files.append(sums_path)
    return files


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebind-root", type=Path, default=REBIN_ROOT)
    parser.add_argument("--prepared-root", type=Path, default=PREPARED_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--instance-id", default="uhost-1utsjo3ep1jz")
    parser.add_argument("--code-root", type=Path)
    parser.add_argument("--code-tree-sha256")
    parser.add_argument("--code-archive", type=Path, default=DEFAULT_CODE_ARCHIVE)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--launcher", type=Path)
    parser.add_argument("--emit-blocked-candidate", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    host = platform.node().split(".", 1)[0]
    if host != HOST_EXPECTED:
        raise GateError(f"BLOCKED_WRONG_HOST expected={HOST_EXPECTED} observed={host}")
    if not args.instance_id or any(c.isspace() for c in args.instance_id):
        raise GateError("INSTANCE_ID_INVALID")
    prepared = args.prepared_root.absolute()
    rebind_root = args.rebind_root.absolute()
    output_root = args.output_root.absolute()
    ensure_no_symlink_components(prepared)
    ensure_no_symlink_components(rebind_root)
    ensure_no_symlink_components(output_root.parent)
    if prepared != PREPARED_ROOT.absolute():
        raise GateError(f"PREPARED_ROOT_NOT_CURRENT_CORRECTED={prepared}")
    try:
        rebind_root.relative_to(WORK_ROOT.absolute())
        output_root.relative_to(WORK_ROOT.absolute())
    except ValueError as exc:
        raise GateError("STAGING_ROOT_SCOPE_DRIFT") from exc
    if output_root == rebind_root or rebind_root in output_root.parents:
        raise GateError("OUTPUT_MUST_NOT_BE_INSIDE_REBIND_ROOT")
    rebind = validate_rebind(rebind_root, prepared)
    code = code_binding(args)
    code["variant_manifests"] = rebind["variants"]
    if not args.emit_blocked_candidate:
        print(
            json.dumps(
                {
                    "status": "CHECK_ONLY_R5_REBIND_VALID_NO_AUTHORIZATION_WRITTEN",
                    "host": host,
                    "rebind_success_sha256": rebind["success_sha256"],
                    "fold_artifact_count": 15,
                    "training_approval_emitted": False,
                    "launch_permitted": False,
                },
                sort_keys=True,
            )
        )
        return 0
    files = emit_receipts(
        output_root,
        rebind=rebind,
        prepared=prepared,
        rebind_root=rebind_root,
        code=code,
        instance_id=args.instance_id,
    )
    print(
        json.dumps(
            {
                "status": "BLOCKED_CANDIDATE_WRITTEN_NO_TRAINING_AUTHORIZATION",
                "output_root": str(output_root),
                "files": [str(path) for path in files],
                "rebind_success_sha256": rebind["success_sha256"],
                "launch_permitted": False,
                "training_approval_emitted": False,
                "exit_code": 42,
            },
            sort_keys=True,
        )
    )
    return 42


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GateError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(42)
