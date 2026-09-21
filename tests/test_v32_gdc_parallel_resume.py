from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.prepare_v32_gdc_parallel_resume import main


def _manifest(path: Path) -> None:
    rows = [
        {"id": f"id-{index}", "filename": f"file-{index}.txt", "md5": "0" * 32, "size": str(10 + index), "state": "released"}
        for index in range(9)
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_parallel_resume_shards_only_incomplete_disjoint_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "manifest.tsv"
    _manifest(manifest)
    download = tmp_path / "download"
    download.mkdir()
    for index in (0, 3):
        target = download / f"id-{index}" / f"file-{index}.txt"
        target.parent.mkdir()
        target.write_bytes(b"x" * (10 + index))
    output = tmp_path / "shards"
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare", "prepare", "--manifest", str(manifest),
            "--download-root", str(download), "--output-root", str(output),
            "--shards", "3",
        ],
    )
    assert main() == 0
    observed = set()
    for shard in sorted(output.glob("shard_*.gdc_manifest.tsv")):
        with shard.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        keys = {(row["id"], row["filename"]) for row in rows}
        assert not (observed & keys)
        observed.update(keys)
    assert observed == {
        (f"id-{index}", f"file-{index}.txt")
        for index in range(9)
        if index not in {0, 3}
    }


def test_size_verifier_fails_closed_then_writes_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "manifest.tsv"
    _manifest(manifest)
    download = tmp_path / "download"
    download.mkdir()
    receipt = tmp_path / "receipt.json"
    argv = [
        "verify", "verify", "--manifest", str(manifest),
        "--download-root", str(download), "--receipt", str(receipt),
    ]
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(RuntimeError, match="size-invalid"):
        main()
    assert not receipt.exists()
    for index in range(9):
        target = download / f"id-{index}" / f"file-{index}.txt"
        target.parent.mkdir()
        target.write_bytes(b"x" * (10 + index))
    monkeypatch.setattr("sys.argv", argv)
    assert main() == 0
    assert receipt.is_file()
