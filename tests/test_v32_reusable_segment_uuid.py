from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest

from scripts.materialize_v32_reusable_segment_uuid import (
    ReusableSegmentError,
    materialize_reusable_segments,
)


def _md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()  # noqa: S324 - test fixture follows GDC.


def _write_inputs(root: Path, data: bytes = b"segment\n") -> tuple[Path, Path, Path, Path]:
    source_root = root / "authorized"
    source_root.mkdir()
    source = source_root / "sample.seg.txt"
    source.write_bytes(data)
    file_id = "11111111-2222-3333-4444-555555555555"
    manifest = root / "manifest.tsv"
    columns = [
        "cancer_id", "file_id", "file_name", "md5sum", "file_size",
        "selected_for_patient", "existing_reusable", "existing_path",
    ]
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerow(
            {
                "cancer_id": "BRCA",
                "file_id": file_id,
                "file_name": source.name,
                "md5sum": _md5(data),
                "file_size": len(data),
                "selected_for_patient": "True",
                "existing_reusable": "True",
                "existing_path": str(source),
            }
        )
    retry = root / "retry.tsv"
    with retry.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["id", "filename", "md5", "size", "state"],
            delimiter="\t", lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(
            {"id": file_id, "filename": source.name, "md5": _md5(data), "size": len(data), "state": "submitted"}
        )
    download = root / "download"
    download.mkdir()
    return manifest, retry, source_root, download


def test_materializes_exact_reusable_set_without_replace(tmp_path: Path) -> None:
    manifest, retry, source_root, download = _write_inputs(tmp_path)
    output = tmp_path / "audit"
    result = materialize_reusable_segments(
        manifest_path=manifest,
        retry_path=retry,
        allowed_source_root=source_root,
        download_root=download,
        output_root=output,
    )
    target = download / "11111111-2222-3333-4444-555555555555" / "sample.seg.txt"
    assert result["status"] == "PASS"
    assert result["published"] == 1
    assert target.read_bytes() == b"segment\n"
    assert (output / "SUCCESS.json").is_file()


def test_preflight_rejects_source_md5_before_any_target_write(tmp_path: Path) -> None:
    manifest, retry, source_root, download = _write_inputs(tmp_path)
    (source_root / "sample.seg.txt").write_bytes(b"corrupt")
    with pytest.raises(ReusableSegmentError, match="size/MD5 mismatch"):
        materialize_reusable_segments(
            manifest_path=manifest,
            retry_path=retry,
            allowed_source_root=source_root,
            download_root=download,
            output_root=tmp_path / "audit",
        )
    assert not any(download.rglob("*"))
    assert not (tmp_path / "audit").exists()


def test_refuses_mismatched_existing_target(tmp_path: Path) -> None:
    manifest, retry, source_root, download = _write_inputs(tmp_path)
    target_dir = download / "11111111-2222-3333-4444-555555555555"
    target_dir.mkdir()
    (target_dir / "sample.seg.txt").write_bytes(b"wrong")
    with pytest.raises(ReusableSegmentError, match="mismatched existing target"):
        materialize_reusable_segments(
            manifest_path=manifest,
            retry_path=retry,
            allowed_source_root=source_root,
            download_root=download,
            output_root=tmp_path / "audit",
        )
    assert (tmp_path / "audit" / "INCOMPLETE.json").is_file()


def test_retry_must_equal_frozen_reusable_set(tmp_path: Path) -> None:
    manifest, retry, source_root, download = _write_inputs(tmp_path)
    retry.write_text("id\tfilename\tmd5\tsize\tstate\n", encoding="utf-8")
    with pytest.raises(ReusableSegmentError, match="empty"):
        materialize_reusable_segments(
            manifest_path=manifest,
            retry_path=retry,
            allowed_source_root=source_root,
            download_root=download,
            output_root=tmp_path / "audit",
        )
