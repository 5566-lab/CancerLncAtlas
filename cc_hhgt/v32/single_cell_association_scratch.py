"""Fail-closed local-ext4 scratch contract for the V3.2 single-cell BH phase.

The public result tree remains on ``${PRIVATE_WORK_ROOT}`` and is still published by an
atomic same-filesystem rename.  This module authorizes exactly one explicitly
named, per-run scratch directory on a local ext4 filesystem.  It deliberately
does *not* add ``/tmp`` (or any other scratch parent) to the general output
allowlist.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import stat as stat_module
from typing import Any, Iterable, Mapping


SCRATCH_CONTRACT_FORMAT = (
    "CC_HHGT_V3_2_SINGLE_CELL_R10_ASSOCIATION_SCRATCH_CONTRACT_V1"
)
SCRATCH_FAILURE_FORMAT = (
    "CC_HHGT_V3_2_SINGLE_CELL_R10_ASSOCIATION_SCRATCH_FAILURE_V1"
)
MINIMUM_FREE_BYTES = 8 * 1024**3
ESTIMATE_FREE_MULTIPLIER = 16
WORK_BASENAME = "duckdb_work"

# These roots remain forbidden even when the caller explicitly supplies them.
# In particular, this contract is not a mechanism for silently falling back to
# a shared server temporary filesystem.
FORBIDDEN_TEMP_ROOTS = (
    Path("/tmp"),
    Path("/var/tmp"),
    Path("/dev/shm"),
)


class AssociationScratchContractError(RuntimeError):
    """A typed, machine-readable scratch-contract failure."""

    def __init__(
        self,
        reason_code: str,
        message: str,
        *,
        phase: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.reason_code = str(reason_code)
        self.phase = str(phase)
        self.details = dict(details or {})
        super().__init__(f"{self.reason_code}: {message}")

    def as_payload(self) -> dict[str, Any]:
        return {
            "format": SCRATCH_FAILURE_FORMAT,
            "status": "TYPED_FAILURE",
            "reason_code": self.reason_code,
            "phase": self.phase,
            "message": str(self),
            "details": self.details,
            "scratch_was_added_to_general_output_roots": False,
            "server_tmp_fallback_used": False,
        }


def _raise(
    reason_code: str,
    message: str,
    *,
    phase: str,
    **details: Any,
) -> None:
    raise AssociationScratchContractError(
        reason_code,
        message,
        phase=phase,
        details=details,
    )


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _decode_mount_path(value: str) -> str:
    # mountinfo escapes space, tab, newline and backslash with octal codes.
    return (
        value.replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
    )


def _filesystem_type_and_mount(path: Path) -> tuple[str, Path]:
    """Return the deepest Linux mount and filesystem type containing ``path``."""

    resolved = path.resolve(strict=True)
    mountinfo = Path("/proc/self/mountinfo")
    if not mountinfo.is_file():
        _raise(
            "ASSOCIATION_SCRATCH_MOUNTINFO_UNAVAILABLE",
            "Linux mountinfo is required to prove a local ext4 scratch filesystem",
            phase="PREFLIGHT",
            path=str(resolved),
        )
    matches: list[tuple[int, str, Path]] = []
    for raw_line in mountinfo.read_text(encoding="utf-8").splitlines():
        if " - " not in raw_line:
            continue
        before, after = raw_line.split(" - ", 1)
        left = before.split()
        right = after.split()
        if len(left) < 5 or not right:
            continue
        mount = Path(_decode_mount_path(left[4]))
        if resolved == mount or _is_within(resolved, mount):
            matches.append((len(mount.parts), right[0], mount))
    if not matches:
        _raise(
            "ASSOCIATION_SCRATCH_MOUNT_NOT_FOUND",
            "No containing mount was found for the scratch path",
            phase="PREFLIGHT",
            path=str(resolved),
        )
    _, filesystem_type, mount = max(matches, key=lambda item: item[0])
    return filesystem_type, mount.resolve(strict=True)


def _disk_free_bytes(path: Path) -> int:
    return int(shutil.disk_usage(path).free)


def _effective_ids() -> tuple[int, int]:
    if not hasattr(os, "geteuid") or not hasattr(os, "getegid"):
        # This branch supports local contract unit tests on Windows.  Formal
        # server execution is Linux and always uses the effective IDs above.
        current = Path.cwd().stat()
        return int(current.st_uid), int(current.st_gid)
    return int(os.geteuid()), int(os.getegid())


def _observed_permission_mode(observed: os.stat_result) -> int:
    """Return Unix permission bits (kept injectable for Windows unit tests)."""

    return stat_module.S_IMODE(observed.st_mode)


def _absolute_lexical(path: str | Path, *, phase: str) -> Path:
    value = Path(path)
    if not value.is_absolute():
        _raise(
            "ASSOCIATION_SCRATCH_PATH_NOT_ABSOLUTE",
            "--association-scratch-root must be an absolute path",
            phase=phase,
            path=str(value),
        )
    return Path(os.path.abspath(os.fspath(value)))


def _assert_no_symlink_components(path: Path, *, phase: str) -> None:
    """Reject a symlink at any existing component of an absolute path."""

    if not path.is_absolute():
        _raise(
            "ASSOCIATION_SCRATCH_PATH_NOT_ABSOLUTE",
            "Scratch path is not absolute",
            phase=phase,
            path=str(path),
        )
    anchor = Path(path.anchor)
    current = anchor
    for part in path.parts[1:]:
        current = current / part
        try:
            observed = os.lstat(current)
        except FileNotFoundError:
            continue
        if stat_module.S_ISLNK(observed.st_mode):
            _raise(
                "ASSOCIATION_SCRATCH_SYMLINK_FORBIDDEN",
                "A scratch path component is a symlink",
                phase=phase,
                component=str(current),
            )


def _validate_path_policy(
    requested_root: str | Path,
    *,
    phase: str,
    forbidden_output_roots: Iterable[str | Path] = (),
) -> Path:
    root = _absolute_lexical(requested_root, phase=phase)
    _assert_no_symlink_components(root, phase=phase)
    for forbidden in FORBIDDEN_TEMP_ROOTS:
        if root == forbidden or _is_within(root, forbidden):
            _raise(
                "ASSOCIATION_SCRATCH_SHARED_TMP_FORBIDDEN",
                "Shared server temporary roots are never valid association scratch",
                phase=phase,
                path=str(root),
                forbidden_root=str(forbidden),
            )
    for raw_output in forbidden_output_roots:
        output = _absolute_lexical(raw_output, phase=phase)
        # Disallow either nesting direction so scratch cannot be published or
        # swept up as part of a result tree.
        if root == output or _is_within(root, output) or _is_within(output, root):
            _raise(
                "ASSOCIATION_SCRATCH_OUTPUT_OVERLAP",
                "Scratch and durable output trees must be disjoint",
                phase=phase,
                scratch_root=str(root),
                output_root=str(output),
            )
    return root


def estimate_association_scratch_bytes(resource_estimate: Mapping[str, Any]) -> int:
    """Extract a positive, conservative association spill estimate."""

    keys = (
        "association_workspace_bytes",
        "duckdb_external_sort_budget_bytes",
        "parquet_batch_budget_bytes",
    )
    values: list[int] = []
    for key in keys:
        try:
            value = int(resource_estimate.get(key, 0))
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            values.append(value)
    if not values:
        _raise(
            "ASSOCIATION_SCRATCH_ESTIMATE_MISSING",
            "Resource estimate contains no positive association workspace estimate",
            phase="PREFLIGHT",
            required_keys=list(keys),
        )
    return max(values)


def required_free_bytes(estimate_bytes: int) -> int:
    estimate = int(estimate_bytes)
    if estimate <= 0:
        _raise(
            "ASSOCIATION_SCRATCH_ESTIMATE_INVALID",
            "Association scratch estimate must be positive",
            phase="PREFLIGHT",
            estimate_bytes=estimate,
        )
    return max(MINIMUM_FREE_BYTES, ESTIMATE_FREE_MULTIPLIER * estimate)


def preflight_association_scratch_root(
    requested_root: str | Path,
    *,
    estimate_bytes: int,
    forbidden_output_roots: Iterable[str | Path] = (),
) -> dict[str, Any]:
    """Validate a not-yet-created exact scratch root without writing it."""

    root = _validate_path_policy(
        requested_root,
        phase="PREFLIGHT",
        forbidden_output_roots=forbidden_output_roots,
    )
    if root.exists() or root.is_symlink():
        _raise(
            "ASSOCIATION_SCRATCH_REUSE_FORBIDDEN",
            "The exact association scratch root already exists",
            phase="PREFLIGHT",
            path=str(root),
        )
    parent = root.parent
    if parent.is_symlink() or not parent.is_dir():
        _raise(
            "ASSOCIATION_SCRATCH_PARENT_UNSAFE",
            "The scratch parent must be an existing real directory",
            phase="PREFLIGHT",
            parent=str(parent),
        )
    filesystem_type, mount = _filesystem_type_and_mount(parent)
    if filesystem_type != "ext4":
        _raise(
            "ASSOCIATION_SCRATCH_FILESYSTEM_NOT_EXT4",
            "Association scratch must be on a local ext4 filesystem",
            phase="PREFLIGHT",
            path=str(root),
            filesystem_type=filesystem_type,
            mount_point=str(mount),
        )
    required = required_free_bytes(estimate_bytes)
    available = _disk_free_bytes(parent)
    if available < required:
        _raise(
            "ASSOCIATION_SCRATCH_SPACE_INSUFFICIENT",
            "Scratch free space is below max(8 GiB, 16 x estimate)",
            phase="PREFLIGHT",
            available_bytes=available,
            required_bytes=required,
            estimate_bytes=int(estimate_bytes),
        )
    return {
        "requested_root": str(root),
        "parent": str(parent.resolve(strict=True)),
        "filesystem_type": filesystem_type,
        "mount_point": str(mount),
        "estimate_bytes": int(estimate_bytes),
        "required_free_bytes": int(required),
        "observed_free_bytes": int(available),
        "exact_root_must_not_preexist": True,
        "server_tmp_fallback_permitted": False,
        "general_output_allowlist_extended": False,
    }


def create_association_scratch_contract(
    requested_root: str | Path,
    *,
    estimate_bytes: int,
    forbidden_output_roots: Iterable[str | Path] = (),
) -> dict[str, Any]:
    """Create and pin one exact private scratch directory, exclusively."""

    preflight = preflight_association_scratch_root(
        requested_root,
        estimate_bytes=estimate_bytes,
        forbidden_output_roots=forbidden_output_roots,
    )
    root = Path(preflight["requested_root"])
    try:
        os.mkdir(root, mode=0o700)
    except FileExistsError:
        _raise(
            "ASSOCIATION_SCRATCH_CREATE_RACE",
            "The exact scratch root appeared during exclusive creation",
            phase="CREATE",
            path=str(root),
        )
    except OSError as exc:
        _raise(
            "ASSOCIATION_SCRATCH_CREATE_FAILED",
            "The exact scratch root could not be created",
            phase="CREATE",
            path=str(root),
            errno=exc.errno,
        )
    created_stat = os.lstat(root)
    try:
        if stat_module.S_ISLNK(created_stat.st_mode) or not stat_module.S_ISDIR(
            created_stat.st_mode
        ):
            _raise(
                "ASSOCIATION_SCRATCH_CREATED_OBJECT_UNSAFE",
                "Exclusive creation did not yield a real directory",
                phase="CREATE",
                path=str(root),
            )
        # Enforce exact privacy independently of the process umask.
        os.chmod(root, 0o700)
        created_stat = os.lstat(root)
        uid, gid = _effective_ids()
        mode = _observed_permission_mode(created_stat)
        if mode != 0o700:
            _raise(
                "ASSOCIATION_SCRATCH_MODE_NOT_0700",
                "Scratch mode is not exactly 0700",
                phase="CREATE",
                path=str(root),
                observed_mode=oct(mode),
            )
        if int(created_stat.st_uid) != uid or int(created_stat.st_gid) != gid:
            _raise(
                "ASSOCIATION_SCRATCH_OWNER_MISMATCH",
                "Scratch UID/GID do not match the effective process owner",
                phase="CREATE",
                path=str(root),
                observed_uid=int(created_stat.st_uid),
                observed_gid=int(created_stat.st_gid),
                expected_uid=uid,
                expected_gid=gid,
            )
        filesystem_type, mount = _filesystem_type_and_mount(root)
        if filesystem_type != "ext4":
            _raise(
                "ASSOCIATION_SCRATCH_FILESYSTEM_NOT_EXT4",
                "Created scratch root is not on ext4",
                phase="CREATE",
                filesystem_type=filesystem_type,
                mount_point=str(mount),
            )
        available = _disk_free_bytes(root)
        required = int(preflight["required_free_bytes"])
        if available < required:
            _raise(
                "ASSOCIATION_SCRATCH_SPACE_INSUFFICIENT",
                "Scratch free space fell below the pre-exec gate",
                phase="CREATE",
                available_bytes=available,
                required_bytes=required,
            )
        canonical_root = root.resolve(strict=True)
        contract = {
            "format": SCRATCH_CONTRACT_FORMAT,
            "root": str(canonical_root),
            "work_basename": WORK_BASENAME,
            "filesystem_type": filesystem_type,
            "mount_point": str(mount),
            "mode": mode,
            "uid": int(created_stat.st_uid),
            "gid": int(created_stat.st_gid),
            "st_dev": int(created_stat.st_dev),
            "st_ino": int(created_stat.st_ino),
            "estimate_bytes": int(estimate_bytes),
            "minimum_free_bytes": MINIMUM_FREE_BYTES,
            "estimate_free_multiplier": ESTIMATE_FREE_MULTIPLIER,
            "required_free_bytes": required,
            "free_bytes_before_exec": int(available),
            "exclusive_creation": True,
            "exact_mode_0700": True,
            "effective_uid_gid_pinned": True,
            "device_inode_pinned": True,
            "symlinks_permitted": False,
            "server_tmp_fallback_permitted": False,
            "general_output_allowlist_extended": False,
            "durable_publication_location": "./data/CancerLncAtlas",
        }
        # Revalidate the serialized form before it can cross os.execve.
        validate_association_scratch_contract(contract, phase="PRE_EXEC")
        return contract
    except BaseException:
        # Remove only the exact just-created empty inode.  Never recurse.
        try:
            current = os.lstat(root)
            if (
                int(current.st_dev) == int(created_stat.st_dev)
                and int(current.st_ino) == int(created_stat.st_ino)
                and root.is_dir()
                and not root.is_symlink()
                and not any(root.iterdir())
            ):
                os.rmdir(root)
        except OSError:
            pass
        raise


def validate_association_scratch_contract(
    contract: Mapping[str, Any],
    *,
    phase: str,
    require_empty: bool = True,
) -> Path:
    """Revalidate owner, inode, mount, privacy and space after image changes."""

    if contract.get("format") != SCRATCH_CONTRACT_FORMAT:
        _raise(
            "ASSOCIATION_SCRATCH_CONTRACT_FORMAT_DRIFT",
            "Scratch contract format is invalid",
            phase=phase,
        )
    root = _validate_path_policy(contract.get("root", ""), phase=phase)
    _assert_no_symlink_components(root, phase=phase)
    try:
        observed = os.lstat(root)
    except FileNotFoundError:
        _raise(
            "ASSOCIATION_SCRATCH_ROOT_MISSING",
            "Pinned scratch root is missing",
            phase=phase,
            path=str(root),
        )
    if stat_module.S_ISLNK(observed.st_mode) or not stat_module.S_ISDIR(
        observed.st_mode
    ):
        _raise(
            "ASSOCIATION_SCRATCH_ROOT_UNSAFE",
            "Pinned scratch root is not a real directory",
            phase=phase,
            path=str(root),
        )
    mode = _observed_permission_mode(observed)
    expected_identity = (
        int(contract.get("st_dev", -1)),
        int(contract.get("st_ino", -1)),
    )
    observed_identity = (int(observed.st_dev), int(observed.st_ino))
    if observed_identity != expected_identity:
        _raise(
            "ASSOCIATION_SCRATCH_IDENTITY_DRIFT",
            "Scratch st_dev/st_ino changed",
            phase=phase,
            expected=list(expected_identity),
            observed=list(observed_identity),
        )
    if mode != 0o700 or int(contract.get("mode", -1)) != 0o700:
        _raise(
            "ASSOCIATION_SCRATCH_MODE_NOT_0700",
            "Scratch privacy mode drifted",
            phase=phase,
            observed_mode=oct(mode),
        )
    uid, gid = _effective_ids()
    owner = (int(observed.st_uid), int(observed.st_gid))
    expected_owner = (int(contract.get("uid", -1)), int(contract.get("gid", -1)))
    if owner != expected_owner or owner != (uid, gid):
        _raise(
            "ASSOCIATION_SCRATCH_OWNER_MISMATCH",
            "Scratch UID/GID drifted after creation",
            phase=phase,
            expected=list(expected_owner),
            effective=[uid, gid],
            observed=list(owner),
        )
    filesystem_type, mount = _filesystem_type_and_mount(root)
    if (
        filesystem_type != "ext4"
        or contract.get("filesystem_type") != "ext4"
        or str(mount) != str(contract.get("mount_point"))
    ):
        _raise(
            "ASSOCIATION_SCRATCH_FILESYSTEM_DRIFT",
            "Scratch ext4 mount identity drifted",
            phase=phase,
            filesystem_type=filesystem_type,
            mount_point=str(mount),
        )
    required = int(contract.get("required_free_bytes", -1))
    expected_required = required_free_bytes(int(contract.get("estimate_bytes", -1)))
    if required != expected_required:
        _raise(
            "ASSOCIATION_SCRATCH_SPACE_CONTRACT_DRIFT",
            "Serialized scratch space threshold is invalid",
            phase=phase,
            serialized_required=required,
            recomputed_required=expected_required,
        )
    available = _disk_free_bytes(root)
    if available < required:
        _raise(
            "ASSOCIATION_SCRATCH_SPACE_INSUFFICIENT",
            "Scratch free space is below the pinned threshold",
            phase=phase,
            available_bytes=available,
            required_bytes=required,
        )
    if require_empty and any(root.iterdir()):
        _raise(
            "ASSOCIATION_SCRATCH_NOT_EMPTY",
            "Scratch root is not empty at a contract boundary",
            phase=phase,
            path=str(root),
        )
    return root


def association_work_path(contract: Mapping[str, Any], *, phase: str) -> Path:
    root = validate_association_scratch_contract(contract, phase=phase)
    basename = str(contract.get("work_basename", ""))
    if basename != WORK_BASENAME:
        _raise(
            "ASSOCIATION_SCRATCH_WORK_BASENAME_DRIFT",
            "Scratch work basename is not the frozen value",
            phase=phase,
            observed=basename,
        )
    work = root / WORK_BASENAME
    if work.exists() or work.is_symlink():
        _raise(
            "ASSOCIATION_SCRATCH_WORK_REUSE_FORBIDDEN",
            "Association work directory already exists",
            phase=phase,
            path=str(work),
        )
    return work


def remove_empty_association_scratch_root(
    contract: Mapping[str, Any], *, phase: str
) -> None:
    root = validate_association_scratch_contract(
        contract, phase=phase, require_empty=True
    )
    os.rmdir(root)
    if root.exists() or root.is_symlink():
        _raise(
            "ASSOCIATION_SCRATCH_CLEANUP_FAILED",
            "Empty scratch root still exists after exact rmdir",
            phase=phase,
            path=str(root),
        )


def canonical_contract_sha256(contract: Mapping[str, Any]) -> str:
    import hashlib

    encoded = json.dumps(
        dict(contract),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
