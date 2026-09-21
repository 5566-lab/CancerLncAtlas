#!/usr/bin/env python3
"""Build the frozen oracle overlay and CPU/no-GPU deployment manifest locally.

Run this only after the CPU bootstrap and all six oracle controls have reached
their reviewed final bytes.  It performs no cloud operation.  Existing outputs
are never overwritten.  Formal-r2 draft bytes are deliberately outside this
first CPU trust bundle.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import stat
import sys
import tarfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEPLOYMENT_FORMAT = "CANCERLNCATLAS_COMPSHARE_CPU_NOGPU_BOOTSTRAP_MANIFEST_V1"
DEPLOYMENT_STATUS = "BOOTSTRAP_ARTIFACTS_FROZEN"
ORACLE_BUNDLE_FORMAT = "CC_HHGT_V3_2_GROUP_SHARED_ORACLE_CODE_BUNDLE_MANIFEST_V1"
ORACLE_BUNDLE_STATUS = "ORACLE_CODE_BUNDLE_FROZEN"
INSTANCE_ID = "uhost-1up504geeqzj"
CPU_NAMESPACE = "v32_compshare_cpu_nogpu_bootstrap_20260901_r1"
FORMAL_NAMESPACE = "v32_g012_paid_gpu_20260901_r2"
ORACLE_NAMESPACE = "v32_group_shared_oracle_paid_gpu_20260901_r2"
REMOTE_ROOT = "./data/CancerLncAtlas"
MAX_FILES = 10_000
MAX_BYTES = 256 * 1024 * 1024


class BundleBuildError(RuntimeError):
    pass


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _assert_no_links(path: Path, label: str) -> None:
    target = _absolute(path)
    for component in reversed([target, *target.parents]):
        try:
            observed = os.lstat(component)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(observed.st_mode) or getattr(os.path, "isjunction", lambda _: False)(component):
            raise BundleBuildError(f"{label}_LINK_COMPONENT_FORBIDDEN={component}")


def _read(path: Path, label: str) -> tuple[bytes, str, int]:
    target = _absolute(path)
    _assert_no_links(target, label)
    fd = os.open(
        target,
        os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0 or before.st_nlink != 1:
            raise BundleBuildError(f"{label}_FILE_TYPE_OR_NLINK_INVALID={target}")
        raw = bytearray()
        digest = hashlib.sha256()
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            raw.extend(chunk)
            digest.update(chunk)
        after = os.fstat(fd)
        identity = lambda value: (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
        if identity(before) != identity(after):
            raise BundleBuildError(f"{label}_CHANGED_DURING_READ={target}")
        rebound = os.stat(target, follow_symlinks=False)
        if not stat.S_ISREG(rebound.st_mode) or (
            rebound.st_dev,
            rebound.st_ino,
            rebound.st_size,
            rebound.st_mtime_ns,
        ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise BundleBuildError(f"{label}_PATH_REPLACED={target}")
        return bytes(raw), digest.hexdigest(), int(before.st_size)
    finally:
        os.close(fd)


def _write_new(path: Path, raw: bytes) -> None:
    target = _absolute(path)
    _assert_no_links(target.parent, "OUTPUT_PARENT")
    target.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_links(target.parent, "OUTPUT_PARENT")
    fd = os.open(
        target,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(fd, raw[offset:])
        os.fsync(fd)
    finally:
        os.close(fd)


def _source_files(root: Path) -> list[tuple[str, Path, bytes, str, int, int]]:
    rows: list[tuple[str, Path, bytes, str, int, int]] = []
    total = 0
    required = {
        "cc_hhgt/v32/cli.py",
        "cc_hhgt/v32/training_guard.py",
        "cc_hhgt/v32/gpu_backward_probe.py",
        "cc_hhgt/v32/training.py",
        "cc_hhgt/v32/group_shared_encoder_oracle.py",
        "config/model_v3_2_group_shared_oracle_paid_gpu_20260901_r2.yaml",
        "config/v32_group_shared_oracle_paid_gpu_20260901_r2.TASK_MANIFEST.tsv",
        "scripts/v32_pipeline.py",
        "docs/v32_group_shared_encoder_estimator_preregistration_20260901.md",
        "docs/v32_group_shared_encoder_oracle_decision_20260901_r2.json",
    }
    package_root = root / "cc_hhgt"
    if not package_root.is_dir():
        raise BundleBuildError(f"ORACLE_BUNDLE_SOURCE_ROOT_MISSING={package_root}")
    candidates = [candidate for candidate in sorted(package_root.rglob("*")) if candidate.is_file()]
    candidates.extend(root / relative for relative in sorted(required) if not relative.startswith("cc_hhgt/"))
    seen: set[str] = set()
    for candidate in candidates:
        relative = candidate.relative_to(root)
        relative_text = relative.as_posix()
        if relative_text in seen:
            continue
        seen.add(relative_text)
        if any(part in {"__pycache__", ".pytest_cache", ".git"} for part in relative.parts):
            continue
        if candidate.suffix.lower() in {".pyc", ".pyo"} or candidate.name.endswith(".partial"):
            continue
        if candidate.is_symlink():
            raise BundleBuildError(f"ORACLE_BUNDLE_SOURCE_SYMLINK={relative_text}")
        if not candidate.is_file():
            raise BundleBuildError(f"ORACLE_BUNDLE_REQUIRED_SOURCE_MISSING={relative_text}")
        raw, digest, size = _read(candidate, f"ORACLE_BUNDLE_SOURCE_{relative_text}")
        mode = 0o700 if candidate.suffix.lower() == ".sh" else 0o600
        rows.append((relative_text, candidate, raw, digest, size, mode))
        total += size
        if len(rows) > MAX_FILES or total > MAX_BYTES:
            raise BundleBuildError("ORACLE_BUNDLE_SIZE_CAP_EXCEEDED")
    rows.sort(key=lambda row: row[0])
    present = {row[0] for row in rows}
    if not required.issubset(present):
        raise BundleBuildError(f"ORACLE_BUNDLE_REQUIRED_FILES_MISSING={sorted(required - present)}")
    return rows


def _build_archive(path: Path, rows: Iterable[tuple[str, Path, bytes, str, int, int]]) -> None:
    target = _absolute(path)
    _assert_no_links(target.parent, "ARCHIVE_PARENT")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise BundleBuildError(f"ARCHIVE_OUTPUT_EXISTS={target}")
    partial = target.with_name(f".{target.name}.{os.getpid()}.partial")
    with partial.open("xb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w", format=tarfile.GNU_FORMAT) as archive:
                for relative, _, raw, _, size, mode in rows:
                    info = tarfile.TarInfo(relative)
                    info.size = size
                    info.mode = mode
                    info.mtime = 0
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    import io

                    archive.addfile(info, io.BytesIO(raw))
        raw_handle.flush()
        os.fsync(raw_handle.fileno())
    os.link(partial, target, follow_symlinks=False)
    partial.unlink()


def _relative_to_root(path: Path, root: Path) -> str:
    absolute = _absolute(path)
    try:
        return absolute.relative_to(_absolute(root)).as_posix()
    except ValueError as exc:
        raise BundleBuildError(f"DEPLOYMENT_LOCAL_PATH_OUTSIDE_REPOSITORY={absolute}") from exc


def _artifact(
    *,
    name: str,
    local: Path,
    root: Path,
    stage: str,
    remote: str,
    phase: str,
    mode: int,
) -> dict[str, Any]:
    _, digest, size = _read(local, f"DEPLOYMENT_{name}")
    return {
        "name": name,
        "phase": phase,
        "local_relative_path": _relative_to_root(local, root),
        "stage_relative_path": stage,
        "remote_install_path": remote,
        "sha256": digest,
        "size_bytes": size,
        "mode": mode,
    }


def build(*, repository_root: Path, output_directory: Path, deployment_manifest: Path) -> dict[str, Any]:
    root = _absolute(repository_root)
    _assert_no_links(root, "REPOSITORY_ROOT")
    output = _absolute(output_directory)
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise BundleBuildError("OUTPUT_DIRECTORY_OUTSIDE_REPOSITORY") from exc
    expected_output = root / "artifacts" / CPU_NAMESPACE
    if output != expected_output:
        raise BundleBuildError(f"OUTPUT_DIRECTORY_SCOPE_DRIFT={output}")
    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / "ORACLE_CODE_BUNDLE.tar.gz"
    bundle_manifest_path = output / "ORACLE_CODE_BUNDLE.MANIFEST.json"
    rows = _source_files(root)
    _build_archive(archive_path, rows)
    _, archive_sha, archive_size = _read(archive_path, "BUILT_ORACLE_ARCHIVE")
    bundle_manifest = {
        "format": ORACLE_BUNDLE_FORMAT,
        "status": ORACLE_BUNDLE_STATUS,
        "namespace": ORACLE_NAMESPACE,
        "archive_sha256": archive_sha,
        "archive_size_bytes": archive_size,
        "file_count": len(rows),
        "total_bytes": sum(row[4] for row in rows),
        "files": [
            {
                "relative_path": relative,
                "sha256": digest,
                "size_bytes": size,
                "mode": mode,
            }
            for relative, _, _, digest, size, mode in rows
        ],
    }
    _write_new(
        bundle_manifest_path,
        (json.dumps(bundle_manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )

    cpu_remote = f"{REMOTE_ROOT}/runtime/bootstrap/{CPU_NAMESPACE}"
    formal_bootstrap = f"{REMOTE_ROOT}/runtime/bootstrap/{FORMAL_NAMESPACE}"
    oracle_control = f"{REMOTE_ROOT}/runtime/oracle_gates/{ORACLE_NAMESPACE}"
    oracle_bootstrap = f"{REMOTE_ROOT}/runtime/bootstrap/{ORACLE_NAMESPACE}"
    specs = [
        ("bootstrap_manifest_verifier", root / "scripts/verify_v32_compshare_cpu_nogpu_bootstrap_manifest.py", f"{cpu_remote}/verify_v32_compshare_cpu_nogpu_bootstrap_manifest.py", "both", 0o700),
        ("dynamic_receipt_verifier", root / "scripts/verify_v32_compshare_cpu_nogpu_dynamic_receipt.py", f"{cpu_remote}/verify_v32_compshare_cpu_nogpu_dynamic_receipt.py", "both", 0o700),
        ("remote_cpu_bootstrap_driver", root / "scripts/cloud_bootstrap_v32_compshare_cpu_nogpu_20260901_r1.sh", f"{cpu_remote}/cloud_bootstrap_v32_compshare_cpu_nogpu_20260901_r1.sh", "both", 0o700),
        ("oracle_overlay_verifier", root / "scripts/verify_v32_group_shared_oracle_overlay_bundle.py", f"{cpu_remote}/verify_v32_group_shared_oracle_overlay_bundle.py", "prepare_r2", 0o700),
        ("r1_abort_sealer", root / "scripts/seal_v32_g012_r1_aborted_no_gpu.py", f"{formal_bootstrap}/seal_v32_g012_r1_aborted_no_gpu.py", "prepare_r2", 0o700),
        ("r2_input_reuse_materializer", root / "scripts/materialize_v32_g012_r2_input_reuse_ready.py", f"{formal_bootstrap}/materialize_v32_g012_r2_input_reuse_ready.py", "prepare_r2", 0o700),
        ("shared_jit_budget_generator", root / "scripts/local_materialize_compshare_jit_budget_20260901_r2.ps1", f"{cpu_remote}/local_materialize_compshare_jit_budget_20260901_r2.ps1", "both", 0o600),
        ("oracle_code_bundle", archive_path, f"{oracle_bootstrap}/ORACLE_CODE_BUNDLE.tar.gz", "prepare_r2", 0o600),
        ("oracle_code_bundle_manifest", bundle_manifest_path, f"{oracle_bootstrap}/ORACLE_CODE_BUNDLE.MANIFEST.json", "prepare_r2", 0o600),
        ("oracle_external_deployment_manifest", root / "docs/v32_group_shared_oracle_external_deployment_manifest_20260901_r2.json", f"{oracle_control}/EXTERNAL_DEPLOYMENT_MANIFEST.json", "prepare_r2", 0o600),
        ("oracle_static_verifier", root / "scripts/validate_v32_group_shared_oracle_static_auth_ready_r2.py", f"{oracle_control}/validate_v32_group_shared_oracle_static_auth_ready_r2.py", "prepare_r2", 0o700),
        ("oracle_cpu_authorizer", root / "scripts/cloud_authorize_v32_group_shared_oracle_no_gpu_20260901_r2.sh", f"{oracle_control}/cloud_authorize_v32_group_shared_oracle_no_gpu_20260901_r2.sh", "prepare_r2", 0o700),
        ("oracle_paid_launcher", root / "scripts/server_launch_v32_group_shared_oracle_paid_gpu_20260901_r2.sh", f"{oracle_control}/server_launch_v32_group_shared_oracle_paid_gpu_20260901_r2.sh", "prepare_r2", 0o700),
        ("oracle_local_supervisor", root / "scripts/local_supervise_compshare_group_shared_oracle_paid_gpu_20260901_r2.ps1", f"{oracle_control}/local_supervise_compshare_group_shared_oracle_paid_gpu_20260901_r2.ps1", "prepare_r2", 0o600),
        ("oracle_local_dispatcher", root / "scripts/local_dispatch_compshare_group_shared_oracle_paid_gpu_20260901_r2.ps1", f"{oracle_control}/local_dispatch_compshare_group_shared_oracle_paid_gpu_20260901_r2.ps1", "prepare_r2", 0o600),
        ("oracle_log_guard", root / "scripts/v32_group_shared_oracle_log_guard_r2.psm1", f"{oracle_control}/v32_group_shared_oracle_log_guard_r2.psm1", "prepare_r2", 0o600),
    ]
    artifacts = [
        _artifact(
            name=name,
            local=local,
            root=root,
            stage=f"artifacts/{name}/{local.name}",
            remote=remote,
            phase=phase,
            mode=mode,
        )
        for name, local, remote, phase, mode in specs
    ]
    deployment = {
        "format": DEPLOYMENT_FORMAT,
        "status": DEPLOYMENT_STATUS,
        "namespace": CPU_NAMESPACE,
        "instance_id": INSTANCE_ID,
        "remote_project_root": REMOTE_ROOT,
        "gpu_start_permitted": False,
        "without_gpu_spec": "A",
        "formal_r2_code_promotion_permitted": False,
        "formal_r2_patch_ready_permitted": False,
        "artifacts": artifacts,
    }
    target = _absolute(deployment_manifest)
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise BundleBuildError("DEPLOYMENT_MANIFEST_OUTSIDE_REPOSITORY") from exc
    expected_deployment = root / "docs" / "v32_compshare_cpu_nogpu_bootstrap_manifest_20260901_r1.json"
    if target != expected_deployment:
        raise BundleBuildError(f"DEPLOYMENT_MANIFEST_SCOPE_DRIFT={target}")
    _write_new(target, (json.dumps(deployment, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    _, deployment_sha, _ = _read(target, "DEPLOYMENT_MANIFEST_OUTPUT")
    return {
        "status": "CPU_NOGPU_BOOTSTRAP_BUNDLE_MATERIALIZED_LOCAL_ONLY",
        "deployment_manifest": str(target),
        "deployment_manifest_sha256": deployment_sha,
        "oracle_bundle": str(archive_path),
        "oracle_bundle_sha256": archive_sha,
        "oracle_bundle_manifest": str(bundle_manifest_path),
        "formal_r2_code_promotion_permitted": False,
        "cloud_contacted": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--deployment-manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        payload = build(
            repository_root=args.repository_root,
            output_directory=args.output_directory,
            deployment_manifest=args.deployment_manifest,
        )
    except (BundleBuildError, OSError, tarfile.TarError) as exc:
        print(str(exc), file=sys.stderr)
        return 22
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
