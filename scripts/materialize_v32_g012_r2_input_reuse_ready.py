#!/usr/bin/env python3
"""Hash-verify and bind the existing 15 G012 fold payloads for r2 reuse.

This CPU/no-GPU tool performs no copy, transfer, extraction, link creation, or
mutation of the prepared input tree.  Its only optional write is one atomic
``INPUT_REUSE_READY.json`` receipt outside that tree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


RECEIPT_FORMAT = "CC_HHGT_V3_2_G012_R2_INPUT_REUSE_READY_V1"
RECEIPT_STATUS = "INPUT_REUSE_READY_HASH_VERIFIED"
VARIANTS = ("G0", "G1", "G2")
FOLDS = tuple(range(5))
EXPECTED_SOURCE_INPUT_ARCHIVE_SHA256 = (
    "1c17b7be89621e5125c39e87f05ce14beb1f2227afdb481d2e9a20049b56bab0"
)
R1_STATIC_AUTH_SHA256 = (
    "1f5bc54b2a8acc037373bfa380c95c8a717e76fde956852ec481b16a0d8716f0"
)


class InputReuseError(RuntimeError):
    """The existing prepared input tree is incomplete, mutable, or drifted."""


def _absolute_lexical(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _assert_no_symlink_components(path: Path, label: str) -> None:
    target = _absolute_lexical(path)
    for component in reversed([target, *target.parents]):
        try:
            observed = os.lstat(component)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise InputReuseError(f"{label}_COMPONENT_LSTAT_FAILED={component}") from exc
        if stat.S_ISLNK(observed.st_mode) or getattr(
            os.path, "isjunction", lambda _: False
        )(component):
            raise InputReuseError(f"{label}_SYMLINK_COMPONENT_FORBIDDEN={component}")


def _open_regular_no_follow(path: Path, label: str) -> tuple[int, list[int], Path]:
    target = _absolute_lexical(path)
    _assert_no_symlink_components(target, label)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    held_directories: list[int] = []
    supports_openat = os.name != "nt" and os.open in getattr(os, "supports_dir_fd", set())
    try:
        if supports_openat:
            directory_flags = (
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            current = os.open(target.anchor, directory_flags)
            held_directories.append(current)
            for component in target.parts[1:-1]:
                current = os.open(component, directory_flags, dir_fd=current)
                if not stat.S_ISDIR(os.fstat(current).st_mode):
                    raise InputReuseError(f"{label}_PARENT_NOT_DIRECTORY={component}")
                held_directories.append(current)
            descriptor = os.open(target.name, flags, dir_fd=current)
        else:
            descriptor = os.open(target, flags)
    except (OSError, InputReuseError) as exc:
        for directory in reversed(held_directories):
            try:
                os.close(directory)
            except OSError:
                pass
        if isinstance(exc, InputReuseError):
            raise
        if isinstance(exc, FileNotFoundError):
            raise
        raise InputReuseError(f"{label}_OPEN_FAILED={target}") from exc
    return descriptor, held_directories, target


def _read_regular_file(
    path: Path,
    *,
    root: Path,
    capture: bool = False,
    label: str = "INPUT_REUSE",
) -> tuple[str, int, bytes | None]:
    target = _absolute_lexical(path)
    bound_root = _absolute_lexical(root)
    if target != bound_root and bound_root not in target.parents:
        raise InputReuseError(f"{label}_PATH_ESCAPES_ROOT={target}")
    descriptor, held_directories, target = _open_regular_no_follow(target, label)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
            raise InputReuseError(f"{label}_NOT_NONEMPTY_REGULAR={target}")
        if before.st_nlink != 1:
            raise InputReuseError(f"{label}_HARDLINK_FORBIDDEN={target}")
        digest = hashlib.sha256()
        captured = bytearray() if capture else None
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            if captured is not None:
                captured.extend(chunk)
        after = os.fstat(descriptor)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if before_identity != after_identity:
            raise InputReuseError(f"{label}_CHANGED_DURING_READ={target}")
        try:
            path_after = os.stat(target, follow_symlinks=False)
        except OSError as exc:
            raise InputReuseError(f"{label}_PATH_LOST_AFTER_READ={target}") from exc
        if not stat.S_ISREG(path_after.st_mode) or (
            path_after.st_dev,
            path_after.st_ino,
            path_after.st_size,
            path_after.st_mtime_ns,
        ) != after_identity[:4]:
            raise InputReuseError(f"{label}_PATH_REPLACED_DURING_READ={target}")
        return (
            digest.hexdigest(),
            int(before.st_size),
            bytes(captured) if captured is not None else None,
        )
    finally:
        os.close(descriptor)
        for directory in reversed(held_directories):
            os.close(directory)


def _sha256_regular_file(path: Path, *, root: Path) -> tuple[str, int]:
    digest, size, _ = _read_regular_file(path, root=root)
    return digest, size


def _load_marker(path: Path, *, variant: str, root: Path) -> dict[str, Any]:
    _, _, raw = _read_regular_file(
        path,
        root=root,
        capture=True,
        label=f"GRAPH_VARIANT_MARKER_{variant}",
    )
    assert raw is not None
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputReuseError(f"GRAPH_VARIANT_MARKER_JSON_INVALID={variant}") from exc
    if not isinstance(payload, Mapping) or payload.get("variant") != variant:
        raise InputReuseError(f"GRAPH_VARIANT_MARKER_DRIFT={variant}")
    return dict(payload)


def _load_r1_static_authority(path: Path) -> tuple[dict[str, Any], str]:
    authority_root = _absolute_lexical(path).parent
    digest, _, raw = _read_regular_file(
        path,
        root=authority_root,
        capture=True,
        label="R1_STATIC_AUTH_R6",
    )
    if digest != R1_STATIC_AUTH_SHA256:
        raise InputReuseError(f"R1_STATIC_AUTH_R6_SHA256_DRIFT={digest}")
    assert raw is not None
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputReuseError("R1_STATIC_AUTH_R6_JSON_INVALID") from exc
    if not isinstance(payload, Mapping) or payload.get("status") != "STATIC_AUTH_READY":
        raise InputReuseError("R1_STATIC_AUTH_R6_SCHEMA_DRIFT")
    variants = payload.get("variants")
    if not isinstance(variants, Mapping) or set(variants) != set(VARIANTS):
        raise InputReuseError("R1_STATIC_AUTH_R6_VARIANTS_DRIFT")
    return dict(payload), digest


def _authority_fold_records(payload: Mapping[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    records: dict[tuple[str, int], dict[str, Any]] = {}
    variants = payload["variants"]
    for variant in VARIANTS:
        variant_payload = variants.get(variant)
        if not isinstance(variant_payload, Mapping):
            raise InputReuseError(f"R1_STATIC_AUTH_R6_VARIANT_SCHEMA_DRIFT={variant}")
        if variant_payload.get("folds") != list(FOLDS) or variant_payload.get("tasks") != 5:
            raise InputReuseError(f"R1_STATIC_AUTH_R6_FOLD_DECLARATION_DRIFT={variant}")
        fold_inputs = variant_payload.get("fold_inputs")
        if not isinstance(fold_inputs, list) or len(fold_inputs) != 5:
            raise InputReuseError(f"R1_STATIC_AUTH_R6_FOLD_INPUT_COUNT_DRIFT={variant}")
        for raw in fold_inputs:
            if not isinstance(raw, Mapping):
                raise InputReuseError(f"R1_STATIC_AUTH_R6_FOLD_RECORD_INVALID={variant}")
            fold = raw.get("fold")
            if fold not in FOLDS or (variant, int(fold)) in records:
                raise InputReuseError(f"R1_STATIC_AUTH_R6_FOLD_KEY_DRIFT={variant}:{fold}")
            if not isinstance(raw.get("path"), str) or not isinstance(raw.get("size_bytes"), int):
                raise InputReuseError(f"R1_STATIC_AUTH_R6_FOLD_RECORD_DRIFT={variant}:{fold}")
            if not isinstance(raw.get("sha256"), str) or len(raw["sha256"]) != 64:
                raise InputReuseError(f"R1_STATIC_AUTH_R6_FOLD_SHA_INVALID={variant}:{fold}")
            records[(variant, int(fold))] = dict(raw)
    if set(records) != {(variant, fold) for variant in VARIANTS for fold in FOLDS}:
        raise InputReuseError("R1_STATIC_AUTH_R6_FOLD_SET_DRIFT")
    return records


def materialize_receipt(
    *,
    prepared_parent: Path,
    r1_static_auth_path: Path,
    output: Path | None,
    now: datetime | None = None,
) -> dict[str, Any]:
    parent = _absolute_lexical(prepared_parent)
    _assert_no_symlink_components(parent, "PREPARED_PARENT")
    try:
        parent_stat = os.lstat(parent)
    except OSError as exc:
        raise InputReuseError(f"PREPARED_PARENT_INVALID={parent}") from exc
    if not stat.S_ISDIR(parent_stat.st_mode):
        raise InputReuseError(f"PREPARED_PARENT_INVALID={parent}")
    authority, authority_sha = _load_r1_static_authority(r1_static_auth_path)
    authoritative_records = _authority_fold_records(authority)
    records: list[dict[str, Any]] = []
    total_bytes = 0
    for variant in VARIANTS:
        variant_root = parent / variant
        try:
            variant_stat = os.lstat(variant_root)
        except OSError as exc:
            raise InputReuseError(f"PREPARED_VARIANT_ROOT_INVALID={variant}") from exc
        if not stat.S_ISDIR(variant_stat.st_mode):
            raise InputReuseError(f"PREPARED_VARIANT_ROOT_INVALID={variant}")
        _load_marker(
            variant_root / "FORMAL_GRAPH_VARIANT.json",
            variant=variant,
            root=parent,
        )
        for fold in FOLDS:
            path = variant_root / f"PATIENT_FOLD_{fold}.pt"
            digest, size = _sha256_regular_file(path, root=parent)
            authoritative = authoritative_records[(variant, fold)]
            observed_binding = {
                "fold": fold,
                "path": str(path),
                "sha256": digest,
                "size_bytes": size,
            }
            if authoritative != observed_binding:
                raise InputReuseError(f"R1_STATIC_AUTH_R6_FOLD_BINDING_DRIFT={variant}:{fold}")
            total_bytes += size
            records.append(
                {
                    "variant": variant,
                    "fold": fold,
                    "path": str(path),
                    "sha256": digest,
                    "size_bytes": size,
                }
            )
    if len(records) != 15 or {
        (record["variant"], int(record["fold"])) for record in records
    } != {(variant, fold) for variant in VARIANTS for fold in FOLDS}:
        raise InputReuseError("INPUT_REUSE_FOLD_SET_DRIFT")

    timestamp = now or datetime.now(timezone.utc)
    receipt = {
        "format": RECEIPT_FORMAT,
        "status": RECEIPT_STATUS,
        "created_at": timestamp.astimezone(timezone.utc).isoformat(),
        "prepared_parent": str(parent),
        "source_input_archive_sha256": EXPECTED_SOURCE_INPUT_ARCHIVE_SHA256,
        "r1_static_auth_r6_path": str(_absolute_lexical(r1_static_auth_path)),
        "r1_static_auth_r6_sha256": authority_sha,
        "fold_artifact_count": 15,
        "fold_artifact_total_bytes": total_bytes,
        "fold_artifacts": records,
        "reused_existing_prepared_inputs": True,
        "retransfer_performed": False,
        "copy_performed": False,
        "extraction_performed": False,
        "source_files_modified": False,
        "formal_result_artifacts_used": False,
    }
    if output is None:
        return receipt
    target = _absolute_lexical(output)
    if parent == target or parent in target.parents:
        raise InputReuseError("INPUT_REUSE_RECEIPT_MUST_BE_OUTSIDE_PREPARED_TREE")
    _assert_no_symlink_components(target.parent, "OUTPUT_PARENT")
    target.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_symlink_components(target.parent, "OUTPUT_PARENT")
    try:
        os.lstat(target)
    except FileNotFoundError:
        target_present = False
    else:
        target_present = True
    if target_present:
        temporary = target.with_name(f".{target.name}.commit.partial")
        if os.path.lexists(temporary):
            public_stat = os.lstat(target)
            temporary_stat = os.lstat(temporary)
            if (
                not stat.S_ISREG(public_stat.st_mode)
                or not stat.S_ISREG(temporary_stat.st_mode)
                or (public_stat.st_dev, public_stat.st_ino)
                != (temporary_stat.st_dev, temporary_stat.st_ino)
                or public_stat.st_nlink != 2
                or temporary_stat.st_nlink != 2
            ):
                raise InputReuseError(
                    "INPUT_REUSE_RECEIPT_PARTIAL_COMMIT_IDENTITY_DRIFT"
                )
            temporary.unlink()
        try:
            _, _, current_raw = _read_regular_file(
                target,
                root=target.parent,
                capture=True,
                label="EXISTING_INPUT_REUSE_RECEIPT",
            )
            assert current_raw is not None
            current = json.loads(current_raw.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, InputReuseError) as exc:
            raise InputReuseError("EXISTING_INPUT_REUSE_RECEIPT_INVALID") from exc
        comparable = dict(receipt)
        comparable["created_at"] = current.get("created_at")
        if current != comparable:
            raise InputReuseError("EXISTING_INPUT_REUSE_RECEIPT_DRIFT")
        return dict(current)

    # A stable private name makes both possible crash points recoverable:
    # staged-but-not-linked (nlink==1), and linked-but-not-unlinked (nlink==2,
    # normalized by the existing-target branch above).
    temporary = target.with_name(f".{target.name}.commit.partial")
    try:
        os.lstat(temporary)
    except FileNotFoundError:
        temporary_present = False
    else:
        temporary_present = True
    encoded = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if temporary_present:
        staged_stat = os.lstat(temporary)
        if not stat.S_ISREG(staged_stat.st_mode) or staged_stat.st_nlink != 1:
            raise InputReuseError("INPUT_REUSE_PARTIAL_FILE_TYPE_OR_NLINK_INVALID")
        if staged_stat.st_size == 0:
            # A process may die after O_EXCL creation but before its first
            # write.  This exact private, unlinked inode is safe to discard.
            temporary.unlink()
            temporary_present = False
    if temporary_present:
        try:
            _, _, staged_raw = _read_regular_file(
                temporary,
                root=target.parent,
                capture=True,
                label="STAGED_INPUT_REUSE_RECEIPT",
            )
            assert staged_raw is not None
        except InputReuseError:
            # Identity/race failures are not a recoverable short write.
            raise
        try:
            staged = json.loads(staged_raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            # Same-fd/path identity was already proven and nlink is one.  An
            # invalid JSON prefix is therefore an interrupted private write,
            # not a committed receipt; remove it and recreate below.
            temporary.unlink()
            temporary_present = False
    if temporary_present:
        comparable = dict(receipt)
        comparable["created_at"] = staged.get("created_at") if isinstance(staged, Mapping) else None
        if not isinstance(staged, Mapping) or dict(staged) != comparable:
            raise InputReuseError("INPUT_REUSE_PARTIAL_DRIFT")
        try:
            os.link(temporary, target, follow_symlinks=False)
            os.unlink(temporary)
        except OSError as exc:
            raise InputReuseError("INPUT_REUSE_PARTIAL_RECOVERY_FAILED") from exc
        return dict(staged)

    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(temporary, flags, 0o600)
    try:
        offset = 0
        while offset < len(encoded):
            offset += os.write(descriptor, encoded[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, target, follow_symlinks=False)
    except OSError as exc:
        raise InputReuseError("INPUT_REUSE_RECEIPT_COMMIT_FAILED") from exc
    try:
        os.unlink(temporary)
    except OSError as exc:
        raise InputReuseError("INPUT_REUSE_RECEIPT_COMMIT_CLEANUP_FAILED") from exc
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepared-parent",
        type=Path,
        default=Path(
            "./data/CancerLncAtlas/inputs/"
            "v32_g012_patient_first_20260830_r1/"
            "formal_prepared_20260830_r3_affine_pyg280_localtorch"
        ),
    )
    parser.add_argument(
        "--r1-static-auth",
        type=Path,
        default=Path(
            "./data/CancerLncAtlas/runtime/bootstrap/"
            "v32_g012_paid_gpu_20260831_r1/"
            "archive_aborted_runtime_infeasible_20260901_r1/"
            "STATIC_AUTH_READY.live_at_abort.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "./data/CancerLncAtlas/runtime/bootstrap/"
            "v32_g012_paid_gpu_20260901_r2/INPUT_REUSE_READY.json"
        ),
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Hash and print the receipt without writing --output.",
    )
    args = parser.parse_args(argv)
    try:
        receipt = materialize_receipt(
            prepared_parent=args.prepared_parent,
            r1_static_auth_path=args.r1_static_auth,
            output=None if args.audit_only else args.output,
        )
    except (InputReuseError, FileNotFoundError, NotADirectoryError) as exc:
        print(str(exc), file=sys.stderr)
        return 22
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
