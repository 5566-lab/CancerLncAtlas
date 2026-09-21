from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts import materialize_v32_g012_external_archive_lock_no_gpu_r1 as lockmod


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    archive = tmp_path / "code.tar.gz"
    archive.write_bytes(b"fixed archive bytes\n")
    monkeypatch.setattr(lockmod, "ARCHIVE_PATH", archive)
    bootstrap = tmp_path / lockmod.NAMESPACE
    bootstrap.mkdir()
    verifier = bootstrap / lockmod.VERIFIER_NAME
    verifier.write_bytes(b"#!/usr/bin/env python3\nprint('trusted')\n")
    records = []
    for relative in sorted(lockmod.REQUIRED_MANIFEST_PATHS):
        content = verifier.read_bytes() if relative.endswith(lockmod.VERIFIER_NAME) else (
            f"fixture:{relative}\n".encode()
        )
        records.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(content).hexdigest(),
                "size_bytes": len(content),
                "mode": 0o644,
            }
        )
    manifest = bootstrap / lockmod.MANIFEST_NAME
    manifest.write_text(
        json.dumps(
            {
                "format": lockmod.MANIFEST_FORMAT,
                "namespace": lockmod.NAMESPACE,
                "archive_sha256": _sha(archive),
                "files": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "archive": archive,
        "bootstrap": bootstrap,
        "manifest": manifest,
        "verifier": verifier,
        "output": bootstrap / lockmod.LOCK_NAME,
    }


def test_external_lock_breaks_archive_self_hash_cycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    receipt = lockmod.materialize_lock(
        archive_path=fixture["archive"],
        manifest_path=fixture["manifest"],
        verifier_path=fixture["verifier"],
        output_path=fixture["output"],
        now=datetime(2026, 9, 1, tzinfo=timezone.utc),
    )
    assert receipt["archive_sha256"] == _sha(fixture["archive"])
    assert receipt["manifest_sha256"] == _sha(fixture["manifest"])
    assert receipt["bootstrap_verifier_sha256"] == _sha(fixture["verifier"])
    assert receipt["code_archive_contains_lock"] is False
    assert receipt["hashes_embedded_in_archive"] is False
    assert receipt["formal_training_authorized"] is False
    assert json.loads(fixture["output"].read_text(encoding="utf-8")) == receipt


def test_external_lock_rejects_verifier_not_bound_by_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    fixture["verifier"].write_bytes(b"tampered verifier\n")
    with pytest.raises(lockmod.ExternalLockError, match="VERIFIER_MANIFEST_SHA_DRIFT"):
        lockmod.materialize_lock(
            archive_path=fixture["archive"],
            manifest_path=fixture["manifest"],
            verifier_path=fixture["verifier"],
            output_path=fixture["output"],
        )


def test_external_lock_never_overwrites_existing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    fixture["output"].write_bytes(b"must survive\n")
    with pytest.raises(lockmod.ExternalLockError, match="OUTPUT_ALREADY_EXISTS"):
        lockmod.materialize_lock(
            archive_path=fixture["archive"],
            manifest_path=fixture["manifest"],
            verifier_path=fixture["verifier"],
            output_path=fixture["output"],
        )
    assert fixture["output"].read_bytes() == b"must survive\n"


@pytest.mark.parametrize("crash_point", ["staged_only", "linked_not_unlinked"])
def test_external_lock_commit_crash_is_recoverable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crash_point: str,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    original = lockmod.materialize_lock(
        archive_path=fixture["archive"],
        manifest_path=fixture["manifest"],
        verifier_path=fixture["verifier"],
        output_path=fixture["output"],
    )
    partial = fixture["output"].with_name(
        f".{fixture['output'].name}.commit.partial"
    )
    if crash_point == "staged_only":
        os.replace(fixture["output"], partial)
    else:
        os.link(fixture["output"], partial)

    recovered = lockmod.materialize_lock(
        archive_path=fixture["archive"],
        manifest_path=fixture["manifest"],
        verifier_path=fixture["verifier"],
        output_path=fixture["output"],
    )
    assert recovered == original
    assert fixture["output"].exists()
    assert not partial.exists()
