from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from cc_hhgt.v32.single_cell_nested_episodes import NESTED_ASSET_FORMAT


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "materialize_single_cell_nested_hhgt_episodes_149.py"
)
SPEC = importlib.util.spec_from_file_location("nested_materializer", SCRIPT)
MATERIALIZER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MATERIALIZER)


def test_selected_ints_normalizes_numpy_fold_ids_for_json_receipts() -> None:
    selected = MATERIALIZER._selected_ints(
        "all", [np.int64(0), np.int64(1)], "OUTER_FOLD"
    )

    assert selected == [0, 1]
    assert all(type(value) is int for value in selected)


def test_scratch_deletion_check_uses_receipt_metadata_without_rehash(
    tmp_path, monkeypatch
) -> None:
    destination = tmp_path / "episode"
    destination.mkdir()
    payload = destination / "matrix.npz"
    payload.write_bytes(b"already-hashed")
    plan = SimpleNamespace(
        geometry_isolation="outer_and_inner_target_fold",
        as_dict=lambda: {"plan": "identity"},
    )
    receipt = {
        "format": NESTED_ASSET_FORMAT,
        "target_host": "149",
        "geometry_isolation": plan.geometry_isolation,
        "donor_partition_sha256": MATERIALIZER.canonical_hash(plan.as_dict()),
        "files": {
            "matrix.npz": {
                "path": str(payload.resolve()),
                "bytes": payload.stat().st_size,
                "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
            }
        },
    }
    (destination / "SUCCESS.json").write_text(json.dumps(receipt))
    monkeypatch.setattr(
        MATERIALIZER,
        "sha256",
        lambda _path: (_ for _ in ()).throw(AssertionError("unexpected rehash")),
    )

    assert MATERIALIZER._verify_complete_metadata(destination, plan) == receipt
