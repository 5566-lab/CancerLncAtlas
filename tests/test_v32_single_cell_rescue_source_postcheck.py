from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
FINALIZER = (
    ROOT
    / "scripts"
    / "server_single_cell_rescue_20260828"
    / "finalize_rescue_audit.py"
)


def load_finalizer():
    spec = importlib.util.spec_from_file_location("rescue_finalizer_postcheck", FINALIZER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def record(path: Path) -> dict:
    observed = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "bytes": observed.st_size,
        "mtime_utc": datetime.fromtimestamp(
            observed.st_mtime, timezone.utc
        ).isoformat(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_full_sha_detects_same_size_same_mtime_content_substitution(tmp_path: Path) -> None:
    finalizer = load_finalizer()
    source = tmp_path / "source.rds"
    source.write_bytes(b"original-bytes")
    original = record(source)
    original_stat = source.stat()
    passed = finalizer.verify_source_record(original)
    assert passed["identity_gate"] == "SIZE_MTIME_AND_FULL_SHA256"

    source.write_bytes(b"modified-bytes")
    assert source.stat().st_size == original["bytes"]
    os.utime(source, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    assert (
        datetime.fromtimestamp(source.stat().st_mtime, timezone.utc).isoformat()
        == original["mtime_utc"]
    )
    with pytest.raises(RuntimeError, match="size/mtime/SHA-256 changed"):
        finalizer.verify_source_record(original)


def test_missing_sha_and_symlink_fail_closed(tmp_path: Path) -> None:
    finalizer = load_finalizer()
    source = tmp_path / "source.h5"
    source.write_bytes(b"fixture")
    missing = record(source)
    missing.pop("sha256")
    with pytest.raises(RuntimeError, match="valid SHA-256"):
        finalizer.verify_source_record(missing)

    link = tmp_path / "link.h5"
    try:
        os.symlink(source, link)
    except OSError:
        pytest.skip("symlink creation is unavailable on this host")
    linked = record(source)
    linked["path"] = str(link)
    with pytest.raises(RuntimeError, match="became unsafe"):
        finalizer.verify_source_record(linked)
