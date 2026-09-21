from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import scripts.delete_v32_methylation_raw_after_closure as deletion


def _write_contracts(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    parent = tmp_path / "inputs" / "v32_distal_regulatory_mutation_20260830_r1"
    raw = parent / deletion.EXPECTED_NAME
    raw.mkdir(parents=True)
    (raw / "a.txt").write_bytes(b"abc")
    retained = tmp_path / "methylation_features"
    retained.mkdir()
    methylation_audit = retained / "AUDIT.json"
    methylation_audit.write_text(
        json.dumps(
            {
                "status": "PASS_PATIENT_LNCRNA_METHYLATION_FEATURES",
                "raw_download_delete_ready": True,
                "raw_download_delete_target": str(raw.resolve()),
                "download_files_md5_verified": 1,
                "download_bytes_md5_verified": 3,
            }
        ),
        encoding="utf-8",
    )
    closure = tmp_path / "CLOSURE_AUDIT.json"
    closure.write_text(
        json.dumps(
            {
                "status": "PASS",
                "raw_methylation_delete_authorized": True,
                "raw_methylation_delete_target": str(raw.resolve()),
            }
        ),
        encoding="utf-8",
    )
    return parent, raw, methylation_audit, closure


def test_deletion_removes_only_pinned_raw_and_retains_features(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent, raw, methylation_audit, closure = _write_contracts(tmp_path)
    receipt = tmp_path / "receipts" / "DELETION_RECEIPT.json"
    monkeypatch.setattr(deletion, "EXPECTED_PARENT", parent)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "delete",
            "--closure-audit", str(closure),
            "--methylation-audit", str(methylation_audit),
            "--raw-root", str(raw),
            "--receipt", str(receipt),
        ],
    )
    assert deletion.main() == 0
    assert not raw.exists()
    assert methylation_audit.is_file()
    assert receipt.is_file()
    assert json.loads(receipt.read_text())["status"] == "PASS_DELETED"


def test_deletion_rejects_unpinned_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent, _, methylation_audit, closure = _write_contracts(tmp_path)
    wrong = parent / "another_raw_root"
    wrong.mkdir()
    monkeypatch.setattr(deletion, "EXPECTED_PARENT", parent)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "delete",
            "--closure-audit", str(closure),
            "--methylation-audit", str(methylation_audit),
            "--raw-root", str(wrong),
            "--receipt", str(tmp_path / "receipt.json"),
        ],
    )
    with pytest.raises(RuntimeError, match="outside pinned scope"):
        deletion.main()
    assert wrong.is_dir()
