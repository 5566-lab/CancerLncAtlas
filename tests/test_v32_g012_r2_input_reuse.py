from __future__ import annotations

import json
import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts import materialize_v32_g012_r2_input_reuse_ready as reuse


def _prepared_fixture(tmp_path: Path) -> Path:
    parent = tmp_path / "DSC" / "CancerLncAtlas" / "inputs" / "prepared"
    for variant in ("G0", "G1", "G2"):
        variant_root = parent / variant
        variant_root.mkdir(parents=True)
        (variant_root / "FORMAL_GRAPH_VARIANT.json").write_text(
            json.dumps({"variant": variant}) + "\n", encoding="utf-8"
        )
        for fold in range(5):
            (variant_root / f"PATIENT_FOLD_{fold}.pt").write_bytes(
                f"{variant}-fold-{fold}-immutable".encode("ascii")
            )
    return parent


def _r1_authority(
    prepared: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    variants: dict[str, object] = {}
    for variant in ("G0", "G1", "G2"):
        fold_inputs = []
        for fold in range(5):
            path = (prepared / variant / f"PATIENT_FOLD_{fold}.pt").resolve()
            fold_inputs.append(
                {
                    "fold": fold,
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "size_bytes": path.stat().st_size,
                }
            )
        variants[variant] = {"folds": list(range(5)), "tasks": 5, "fold_inputs": fold_inputs}
    authority = tmp_path / "STATIC_AUTH_READY.r6.json"
    authority.write_text(
        json.dumps({"status": "STATIC_AUTH_READY", "variants": variants}, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        reuse,
        "R1_STATIC_AUTH_SHA256",
        hashlib.sha256(authority.read_bytes()).hexdigest(),
    )
    return authority


def test_reuse_receipt_hashes_exactly_15_existing_folds_without_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepared_fixture(tmp_path)
    authority = _r1_authority(prepared, tmp_path, monkeypatch)
    output = prepared.parents[1] / "runtime" / "bootstrap" / "r2" / "INPUT_REUSE_READY.json"
    before = {
        path.relative_to(prepared).as_posix(): path.read_bytes()
        for path in prepared.rglob("*")
        if path.is_file()
    }
    receipt = reuse.materialize_receipt(
        prepared_parent=prepared,
        r1_static_auth_path=authority,
        output=output,
        now=datetime(2026, 9, 1, 7, 0, tzinfo=timezone.utc),
    )
    after = {
        path.relative_to(prepared).as_posix(): path.read_bytes()
        for path in prepared.rglob("*")
        if path.is_file()
    }

    assert receipt["status"] == "INPUT_REUSE_READY_HASH_VERIFIED"
    assert receipt["fold_artifact_count"] == 15
    assert len(receipt["fold_artifacts"]) == 15
    assert receipt["source_input_archive_sha256"] == reuse.EXPECTED_SOURCE_INPUT_ARCHIVE_SHA256
    assert receipt["r1_static_auth_r6_sha256"] == reuse.R1_STATIC_AUTH_SHA256
    assert receipt["reused_existing_prepared_inputs"] is True
    assert receipt["retransfer_performed"] is False
    assert receipt["copy_performed"] is False
    assert receipt["extraction_performed"] is False
    assert receipt["source_files_modified"] is False
    assert receipt["formal_result_artifacts_used"] is False
    assert before == after
    assert json.loads(output.read_text(encoding="utf-8")) == receipt


def test_missing_fold_fails_without_writing_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepared_fixture(tmp_path)
    authority = _r1_authority(prepared, tmp_path, monkeypatch)
    (prepared / "G2" / "PATIENT_FOLD_4.pt").unlink()
    output = tmp_path / "INPUT_REUSE_READY.json"
    with pytest.raises(FileNotFoundError):
        reuse.materialize_receipt(
            prepared_parent=prepared, r1_static_auth_path=authority, output=output
        )
    assert not output.exists()


def test_graph_variant_marker_drift_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepared_fixture(tmp_path)
    authority = _r1_authority(prepared, tmp_path, monkeypatch)
    (prepared / "G1" / "FORMAL_GRAPH_VARIANT.json").write_text(
        '{"variant":"G2"}\n', encoding="utf-8"
    )
    with pytest.raises(reuse.InputReuseError, match="GRAPH_VARIANT_MARKER_DRIFT=G1"):
        reuse.materialize_receipt(
            prepared_parent=prepared,
            r1_static_auth_path=authority,
            output=tmp_path / "receipt.json",
        )


def test_receipt_inside_prepared_tree_is_forbidden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepared_fixture(tmp_path)
    authority = _r1_authority(prepared, tmp_path, monkeypatch)
    with pytest.raises(reuse.InputReuseError, match="MUST_BE_OUTSIDE"):
        reuse.materialize_receipt(
            prepared_parent=prepared,
            r1_static_auth_path=authority,
            output=prepared / "INPUT_REUSE_READY.json",
        )


def test_existing_receipt_is_idempotent_but_input_drift_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepared_fixture(tmp_path)
    authority = _r1_authority(prepared, tmp_path, monkeypatch)
    output = tmp_path / "bootstrap" / "INPUT_REUSE_READY.json"
    first = reuse.materialize_receipt(
        prepared_parent=prepared, r1_static_auth_path=authority, output=output
    )
    second = reuse.materialize_receipt(
        prepared_parent=prepared, r1_static_auth_path=authority, output=output
    )
    assert second == first

    (prepared / "G0" / "PATIENT_FOLD_0.pt").write_bytes(b"drifted")
    with pytest.raises(reuse.InputReuseError, match="FOLD_BINDING_DRIFT"):
        reuse.materialize_receipt(
            prepared_parent=prepared, r1_static_auth_path=authority, output=output
        )


def test_receipt_recovers_linked_but_not_unlinked_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepared_fixture(tmp_path)
    authority = _r1_authority(prepared, tmp_path, monkeypatch)
    output = tmp_path / "bootstrap" / "INPUT_REUSE_READY.json"
    temporary = output.with_name(f".{output.name}.commit.partial")
    real_unlink = reuse.os.unlink
    injected = False

    def fail_once(path: object, *args: object, **kwargs: object) -> None:
        nonlocal injected
        if Path(path) == temporary and not injected:
            injected = True
            raise OSError("injected commit cleanup failure")
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(reuse.os, "unlink", fail_once)
    with pytest.raises(reuse.InputReuseError, match="COMMIT_CLEANUP_FAILED"):
        reuse.materialize_receipt(
            prepared_parent=prepared, r1_static_auth_path=authority, output=output
        )
    assert os.lstat(output).st_ino == os.lstat(temporary).st_ino
    assert os.lstat(output).st_nlink == 2

    recovered = reuse.materialize_receipt(
        prepared_parent=prepared, r1_static_auth_path=authority, output=output
    )
    assert recovered["status"] == reuse.RECEIPT_STATUS
    assert not temporary.exists()
    assert os.lstat(output).st_nlink == 1


def test_receipt_recovers_staged_but_not_linked_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepared_fixture(tmp_path)
    authority = _r1_authority(prepared, tmp_path, monkeypatch)
    output = tmp_path / "bootstrap" / "INPUT_REUSE_READY.json"
    now = datetime(2026, 9, 1, 7, 0, tzinfo=timezone.utc)
    staged = reuse.materialize_receipt(
        prepared_parent=prepared,
        r1_static_auth_path=authority,
        output=None,
        now=now,
    )
    output.parent.mkdir(parents=True)
    temporary = output.with_name(f".{output.name}.commit.partial")
    temporary.write_text(
        json.dumps(staged, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    recovered = reuse.materialize_receipt(
        prepared_parent=prepared,
        r1_static_auth_path=authority,
        output=output,
        now=now,
    )
    assert recovered == staged
    assert output.is_file()
    assert not temporary.exists()
    assert os.lstat(output).st_nlink == 1


@pytest.mark.parametrize("partial_bytes", [b"", b'{"format":'])
def test_receipt_recovers_interrupted_private_partial_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    partial_bytes: bytes,
) -> None:
    prepared = _prepared_fixture(tmp_path)
    authority = _r1_authority(prepared, tmp_path, monkeypatch)
    output = tmp_path / "bootstrap" / "INPUT_REUSE_READY.json"
    output.parent.mkdir(parents=True)
    temporary = output.with_name(f".{output.name}.commit.partial")
    temporary.write_bytes(partial_bytes)

    recovered = reuse.materialize_receipt(
        prepared_parent=prepared,
        r1_static_auth_path=authority,
        output=output,
        now=datetime(2026, 9, 1, 7, 0, tzinfo=timezone.utc),
    )
    assert recovered["status"] == reuse.RECEIPT_STATUS
    assert output.is_file()
    assert not temporary.exists()
    assert os.lstat(output).st_nlink == 1


def test_same_length_fold_drift_is_rejected_against_r1_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepared_fixture(tmp_path)
    authority = _r1_authority(prepared, tmp_path, monkeypatch)
    target = prepared / "G2/PATIENT_FOLD_0.pt"
    target.write_bytes(b"X" * target.stat().st_size)
    with pytest.raises(reuse.InputReuseError, match="FOLD_BINDING_DRIFT=G2:0"):
        reuse.materialize_receipt(
            prepared_parent=prepared, r1_static_auth_path=authority, output=None
        )


def test_r1_authority_symlink_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepared_fixture(tmp_path)
    authority = _r1_authority(prepared, tmp_path, monkeypatch)
    link = tmp_path / "STATIC_AUTH_LINK.json"
    try:
        link.symlink_to(authority)
    except OSError:
        pytest.skip("symlink creation unavailable")
    with pytest.raises(reuse.InputReuseError, match="SYMLINK_FORBIDDEN"):
        reuse.materialize_receipt(
            prepared_parent=prepared, r1_static_auth_path=link, output=None
        )


def test_prepared_parent_symlink_component_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepared_fixture(tmp_path)
    authority = _r1_authority(prepared, tmp_path, monkeypatch)
    linked = tmp_path / "linked-prepared"
    try:
        linked.symlink_to(prepared, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink creation unavailable")
    with pytest.raises(reuse.InputReuseError, match="SYMLINK_COMPONENT"):
        reuse.materialize_receipt(
            prepared_parent=linked, r1_static_auth_path=authority, output=None
        )
