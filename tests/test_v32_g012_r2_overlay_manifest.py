from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
from pathlib import Path

import pytest

from scripts import verify_v32_g012_r2_overlay_manifest as verifier


def _archive_manifest(
    tmp_path: Path, files: dict[str, tuple[bytes, int]]
) -> tuple[Path, Path, str, str]:
    archive = tmp_path / "code.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        for relative, (content, mode) in sorted(files.items()):
            member = tarfile.TarInfo(relative)
            member.size = len(content)
            member.mode = mode
            handle.addfile(member, io.BytesIO(content))
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest = tmp_path / "code.MANIFEST.json"
    payload = {
        "format": verifier.FORMAT,
        "namespace": verifier.NAMESPACE,
        "archive_sha256": archive_sha,
        "files": [
            {
                "path": relative,
                "sha256": hashlib.sha256(content).hexdigest(),
                "size_bytes": len(content),
                "mode": mode,
            }
            for relative, (content, mode) in sorted(files.items())
        ],
    }
    manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    return archive, manifest, archive_sha, manifest_sha


def test_external_verifier_never_imports_or_executes_overlay(tmp_path: Path) -> None:
    sentinel = tmp_path / "MALICIOUS_CODE_EXECUTED"
    malicious = (
        "from pathlib import Path\n"
        f"Path({str(sentinel)!r}).write_text('executed')\n"
        "raise RuntimeError('must never import')\n"
    ).encode()
    archive, manifest, archive_sha, manifest_sha = _archive_manifest(
        tmp_path,
        {
            "cc_hhgt/evil.py": (malicious, 0o644),
            "scripts/launcher.sh": (b"#!/bin/sh\nexit 42\n", 0o755),
        },
    )
    staging = tmp_path / "staging"
    receipt = verifier.verify_and_extract(
        archive_path=archive,
        manifest_path=manifest,
        staging_root=staging,
        expected_archive_sha256=archive_sha,
        expected_manifest_sha256=manifest_sha,
    )
    assert receipt["overlay_code_imported"] is False
    assert receipt["overlay_code_executed"] is False
    assert (staging / "cc_hhgt/evil.py").read_bytes() == malicious
    assert not sentinel.exists()


def test_archive_symlink_member_is_rejected(tmp_path: Path) -> None:
    regular_archive, manifest, _, _ = _archive_manifest(
        tmp_path, {"cc_hhgt/value.py": (b"safe = True\n", 0o644)}
    )
    with tarfile.open(regular_archive, "w:gz") as handle:
        member = tarfile.TarInfo("cc_hhgt/value.py")
        member.type = tarfile.SYMTYPE
        member.linkname = "/etc/passwd"
        handle.addfile(member)
    archive_sha = hashlib.sha256(regular_archive.read_bytes()).hexdigest()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["archive_sha256"] = archive_sha
    manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    with pytest.raises(verifier.OverlayVerificationError, match="TYPE_FORBIDDEN"):
        verifier.verify_and_extract(
            archive_path=regular_archive,
            manifest_path=manifest,
            staging_root=tmp_path / "staging",
            expected_archive_sha256=archive_sha,
            expected_manifest_sha256=manifest_sha,
        )
    assert not (tmp_path / "staging").exists()


def test_same_length_archive_drift_is_rejected_before_tar_parse(tmp_path: Path) -> None:
    archive, manifest, archive_sha, manifest_sha = _archive_manifest(
        tmp_path, {"cc_hhgt/value.py": (b"safe = True\n", 0o644)}
    )
    raw = bytearray(archive.read_bytes())
    raw[len(raw) // 2] ^= 1
    archive.write_bytes(raw)
    assert len(raw) == archive.stat().st_size
    with pytest.raises(verifier.OverlayVerificationError, match="ARCHIVE_SHA256_DRIFT"):
        verifier.verify_and_extract(
            archive_path=archive,
            manifest_path=manifest,
            staging_root=tmp_path / "staging",
            expected_archive_sha256=archive_sha,
            expected_manifest_sha256=manifest_sha,
        )
    assert not (tmp_path / "staging").exists()


def test_manifest_hardlink_is_rejected(tmp_path: Path) -> None:
    archive, manifest, archive_sha, manifest_sha = _archive_manifest(
        tmp_path, {"cc_hhgt/value.py": (b"safe = True\n", 0o644)}
    )
    original = tmp_path / "manifest.original.json"
    os.replace(manifest, original)
    try:
        os.link(original, manifest)
    except OSError:
        pytest.skip("hardlink creation unavailable")
    with pytest.raises(verifier.OverlayVerificationError, match="HARDLINK_FORBIDDEN"):
        verifier.verify_and_extract(
            archive_path=archive,
            manifest_path=manifest,
            staging_root=tmp_path / "staging",
            expected_archive_sha256=archive_sha,
            expected_manifest_sha256=manifest_sha,
        )


def test_manifest_symlinked_parent_component_is_rejected(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    archive, manifest, archive_sha, manifest_sha = _archive_manifest(
        real, {"cc_hhgt/value.py": (b"safe = True\n", 0o644)}
    )
    linked_parent = tmp_path / "linked"
    try:
        linked_parent.symlink_to(real, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink creation unavailable")
    with pytest.raises(verifier.OverlayVerificationError, match="SYMLINK_COMPONENT"):
        verifier.verify_and_extract(
            archive_path=archive,
            manifest_path=linked_parent / manifest.name,
            staging_root=tmp_path / "staging",
            expected_archive_sha256=archive_sha,
            expected_manifest_sha256=manifest_sha,
        )


def test_path_traversal_member_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "bad.tar.gz"
    content = b"escape"
    with tarfile.open(archive, "w:gz") as handle:
        member = tarfile.TarInfo("cc_hhgt/../../escape.py")
        member.size = len(content)
        member.mode = 0o644
        handle.addfile(member, io.BytesIO(content))
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "format": verifier.FORMAT,
                "namespace": verifier.NAMESPACE,
                "archive_sha256": archive_sha,
                "files": [
                    {
                        "path": "cc_hhgt/value.py",
                        "sha256": hashlib.sha256(content).hexdigest(),
                        "size_bytes": len(content),
                        "mode": 0o644,
                    }
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    with pytest.raises(verifier.OverlayVerificationError, match="PATH_UNSAFE"):
        verifier.verify_and_extract(
            archive_path=archive,
            manifest_path=manifest,
            staging_root=tmp_path / "staging",
            expected_archive_sha256=archive_sha,
            expected_manifest_sha256=manifest_sha,
        )


def test_installed_tree_rejects_broken_symlink_in_unused_allowed_prefix(
    tmp_path: Path,
) -> None:
    files = {
        relative: (
            f"fixture:{relative}\n".encode(),
            0o755 if relative.endswith((".sh", ".ps1")) else 0o644,
        )
        for relative in verifier.DEPLOYMENT_ARTIFACTS
    }
    _, manifest, archive_sha, manifest_sha = _archive_manifest(tmp_path, files)
    installed = tmp_path / "installed"
    for relative, (content, mode) in files.items():
        target = installed / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        os.chmod(target, mode)
    docs = installed / "docs"
    try:
        docs.symlink_to(installed / "missing-docs", target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation unavailable")
    with pytest.raises(verifier.OverlayVerificationError, match="INSTALLED_SYMLINK"):
        verifier.verify_installed_tree(
            manifest_path=manifest,
            installed_root=installed,
            expected_archive_sha256=archive_sha,
            expected_manifest_sha256=manifest_sha,
        )


def test_installed_tree_rejects_unmanifested_root_startup_hook(tmp_path: Path) -> None:
    files = {"cc_hhgt/value.py": (b"safe = True\n", 0o644)}
    archive, manifest, archive_sha, manifest_sha = _archive_manifest(tmp_path, files)
    installed = tmp_path / "installed"
    verifier.verify_and_extract(
        archive_path=archive,
        manifest_path=manifest,
        staging_root=installed,
        expected_archive_sha256=archive_sha,
        expected_manifest_sha256=manifest_sha,
    )
    (installed / "sitecustomize.py").write_text(
        "raise SystemExit(99)\n", encoding="utf-8"
    )
    with pytest.raises(verifier.OverlayVerificationError, match="PREFIX_FORBIDDEN"):
        verifier.verify_installed_tree(
            manifest_path=manifest,
            installed_root=installed,
            expected_archive_sha256=archive_sha,
            expected_manifest_sha256=manifest_sha,
        )
