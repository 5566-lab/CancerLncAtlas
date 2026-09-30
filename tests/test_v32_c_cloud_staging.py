"""Small, offline checks for the C graph transfer gate."""
from __future__ import annotations

import hashlib
import json

import pytest

from scripts.rbp_encode_v33.stage_c_cloud_149 import ROLES, preflight


def make_manifest(tmp_path, *, destination_override=None, fold=0):
    cloud_root = "/root/c_graph_train_20260926_r1"
    files = []
    for role in sorted(ROLES):
        source = tmp_path / role
        source.write_bytes(role.encode())
        files.append({
            "role": role, "source": str(source),
            "destination": (
                destination_override if role == "launcher" and destination_override
                else f"{cloud_root}/{role}"
            ),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "bytes": source.stat().st_size,
        })
    manifest = tmp_path / "CLOUD_TRANSFER_MANIFEST.json"
    manifest.write_text(json.dumps({
        "format": "CANCERLNCATLAS_C_FOLD_CLOUD_TRANSFER_V2",
        "preparation_host": "149", "fold": fold, "gpu_target_root": cloud_root,
        "billing_mode": "Spot", "files": files,
    }), encoding="utf-8")
    return manifest, hashlib.sha256(manifest.read_bytes()).hexdigest()


def test_source_preflight_verifies_every_file_without_cloud_access(tmp_path):
    manifest, digest = make_manifest(tmp_path)
    receipt = tmp_path / "SOURCE_READY.json"
    preflight(manifest, digest, receipt)
    result = json.loads(receipt.read_text(encoding="utf-8"))
    assert result["target_host"] == "149"
    assert result["paid_gpu_started"] is False
    assert set(result["files"]) == ROLES
    with pytest.raises(FileExistsError):
        preflight(manifest, digest, receipt)


def test_source_preflight_stops_on_content_drift(tmp_path):
    manifest, digest = make_manifest(tmp_path)
    (tmp_path / "source_prepared").write_bytes(b"changed")
    receipt = tmp_path / "SOURCE_READY.json"
    with pytest.raises(RuntimeError, match="Source size drift|Source SHA256 mismatch"):
        preflight(manifest, digest, receipt)
    assert not receipt.exists()


def test_source_preflight_rejects_cloud_path_escape(tmp_path):
    manifest, digest = make_manifest(
        tmp_path, destination_override="/root/c_graph_train_20260926_r1/../outside",
    )
    with pytest.raises(RuntimeError, match="Destination escapes cloud root"):
        preflight(manifest, digest, tmp_path / "SOURCE_READY.json")


def test_source_preflight_carries_fold_identity_and_rejects_invalid_fold(tmp_path):
    manifest, digest = make_manifest(tmp_path, fold=3)
    receipt = tmp_path / "SOURCE_READY.json"
    preflight(manifest, digest, receipt)
    assert json.loads(receipt.read_text(encoding="utf-8"))["fold"] == 3

    invalid = tmp_path / "invalid"
    invalid.mkdir()
    bad_manifest, bad_digest = make_manifest(invalid, fold=True)
    with pytest.raises(RuntimeError, match="wrong preparation scope"):
        preflight(bad_manifest, bad_digest, invalid / "SOURCE_READY.json")
