from __future__ import annotations

from pathlib import Path

from cc_hhgt.v30_integrity import (
    atomic_write_bytes,
    build_file_manifest,
    cache_key_sha256,
    merkle_sha256,
    stable_partition,
    verify_file_manifest,
)


def test_manifest_and_cache_key_change_when_input_changes(tmp_path: Path) -> None:
    path = tmp_path / "input.txt"
    path.write_text("first", encoding="utf-8")
    first = build_file_manifest(tmp_path)
    first_root = merkle_sha256(first)
    first_key = cache_key_sha256(
        code_sha256="code",
        config_sha256="config",
        input_sha256=first_root,
        fold_sha256="fold",
        schema_sha256="schema",
    )
    path.write_text("second", encoding="utf-8")
    assert verify_file_manifest(tmp_path, first)
    second_root = merkle_sha256(build_file_manifest(tmp_path))
    second_key = cache_key_sha256(
        code_sha256="code",
        config_sha256="config",
        input_sha256=second_root,
        fold_sha256="fold",
        schema_sha256="schema",
    )
    assert first_root != second_root
    assert first_key != second_key


def test_atomic_write_replaces_and_leaves_no_temporary_file(tmp_path: Path) -> None:
    destination = tmp_path / "asset.bin"
    atomic_write_bytes(destination, b"one")
    atomic_write_bytes(destination, b"two")
    assert destination.read_bytes() == b"two"
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob(".*.tmp"))


def test_stable_partition_is_process_independent() -> None:
    first = stable_partition("TCGA-XX-0001", 5, 20260810)
    assert first == stable_partition("TCGA-XX-0001", 5, 20260810)
    assert 0 <= first < 5

