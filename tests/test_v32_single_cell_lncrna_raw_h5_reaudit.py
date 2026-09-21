from __future__ import annotations

from pathlib import Path
import sys
import types

import numpy as np
import pytest


class FakeDataset:
    def __init__(self, values: object) -> None:
        self.values = np.asarray(values)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape

    @property
    def dtype(self) -> np.dtype[object]:
        return self.values.dtype

    def __getitem__(self, item: object) -> object:
        return self.values[item]


class FakeGroup(dict[str, object]):
    def __init__(self, *args: object, name: str, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.name = name


class FakeHandle:
    def __init__(self, group: FakeGroup) -> None:
        self.group = group

    def __enter__(self) -> "FakeHandle":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def visititems(self, visitor: object) -> None:
        visitor("matrix", self.group)
        visitor("matrix/features", self.group["features"])


fake_h5py = types.ModuleType("h5py")
fake_h5py.Dataset = FakeDataset
fake_h5py.Group = FakeGroup
fake_h5py.File = None
sys.modules.setdefault("h5py", fake_h5py)

from scripts import audit_v32_single_cell_h5_explicit_ids_readonly as explicit
from scripts import audit_v32_single_cell_h5_readonly as count_scan
from scripts import materialize_v32_single_cell_explicit_id_reaudit as materialize


def annotation() -> dict[str, object]:
    by_id = {
        "ENSG000001": "lncRNA",
        "ENSG000002": "protein_coding",
        "ENSG000003": "lncRNA",
        "ENSG000004": "lncRNA",
    }
    symbol_map = {
        "LNC3": ("ENSG000003", "lncRNA"),
        "PC2": ("ENSG000002", "protein_coding"),
    }
    return {
        "by_id": by_id,
        "symbol_map": symbol_map,
        "strict_ids": {"ENSG000001", "ENSG000003", "ENSG000004"},
        "formal_by_id": by_id,
        "formal_symbol_map": symbol_map,
        "audit_symbol_map": {"LNC3": "ENSG000003"},
    }


def fake_h5(path: Path, *, malformed_indptr: bool = False) -> FakeHandle:
    path.parent.mkdir(parents=True)
    path.touch()
    features = FakeGroup(
        {
            "id": FakeDataset(
                [
                    b"ENSG000001.7",
                    b"gene:ENSG000001.2",
                    b"not_an_id",
                    b"ENSG000002.1",
                    b"ENSG000004.3",
                    b"unknown",
                ]
            ),
            "name": FakeDataset(
                [b"x", b"x", b"LNC3", b"PC2", b"x", b"unknown"]
            ),
        },
        name="/matrix/features",
    )
    group = FakeGroup(
        {
            "data": FakeDataset(
                [0.0, -1.0, 7.0, 2.0, np.nan, 3.0, 0.0, 5.0]
            ),
            "indices": FakeDataset(
                np.array([0, 2, 3, 1, 2, 2, 4, 5], dtype=np.int32)
            ),
            "indptr": FakeDataset(
                np.array([0, 3, 5], dtype=np.int64)
                if malformed_indptr
                else np.array([0, 3, 5, 8], dtype=np.int64)
            ),
            "shape": FakeDataset(np.array([6, 3], dtype=np.int64)),
            "barcodes": FakeDataset([b"c1", b"c2", b"c3"]),
            "features": features,
        },
        name="/matrix",
    )
    return FakeHandle(group)


def test_mapping_is_version_stripped_exact_and_case_sensitive() -> None:
    ids = ["ENSG000001.7", "gene:ENSG000001.2", "bad", "bad", "bad"]
    names = ["x", "x", "LNC3", "lnc3", "ENSG000004.9"]
    mapped, methods = explicit.map_features(ids, names, annotation())
    assert mapped == [
        "ENSG000001",
        "ENSG000001",
        "ENSG000003",
        None,
        "ENSG000004",
    ]
    assert methods == {
        "feature_id_stable_ensembl": 2,
        "feature_name_stable_ensembl": 1,
        "unique_full_annotation_symbol": 1,
        "unmapped": 1,
    }


def test_explicit_scan_deduplicates_and_requires_finite_positive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h5_path = tmp_path / "TEST" / "raw_feature_bc_matrix.h5"
    handle = fake_h5(h5_path)
    monkeypatch.setattr(explicit, "SC_ROOT", tmp_path)
    monkeypatch.setattr(explicit.h5py, "File", lambda *_args, **_kwargs: handle)
    row = explicit.scan_cancer("TEST", annotation())
    assert row["cell_count"] == 3
    assert row["feature_count"] == 6
    assert row["matrix_orientation_validated"] is True
    assert row["lncRNA_universe_ids"] == [
        "ENSG000001", "ENSG000003", "ENSG000004"
    ]
    assert row["detected_lncRNA_ids"] == ["ENSG000001", "ENSG000003"]
    assert row["lncRNA_universe_count"] == 3
    assert row["detected_lncRNA_count"] == 2
    assert row["minimum_detected_cell_count"] == 1
    assert row["detection_value_threshold"] == "FINITE_VALUE_GT_0"


def test_explicit_scan_rejects_non_csc_pointer_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h5_path = tmp_path / "TEST" / "raw_feature_bc_matrix.h5"
    handle = fake_h5(h5_path, malformed_indptr=True)
    monkeypatch.setattr(explicit, "SC_ROOT", tmp_path)
    monkeypatch.setattr(explicit.h5py, "File", lambda *_args, **_kwargs: handle)
    with pytest.raises(RuntimeError, match="feature-by-cell CSC"):
        explicit.scan_cancer("TEST", annotation())


def test_count_scanner_formal_policy_matches_explicit_mapping() -> None:
    ids = ["ENSG000001.7", "gene:ENSG000001.2", "bad", "bad"]
    names = ["x", "x", "LNC3", "lnc3"]
    _, formal, _, methods = count_scan.map_features(ids, names, annotation())
    assert formal == ["ENSG000001", "ENSG000001", "ENSG000003", None]
    assert methods == {
        "feature_id_stable_ensembl": 2,
        "unique_full_annotation_symbol": 1,
        "unmapped": 1,
    }


def test_explicit_ids_must_reproduce_r1_count_and_set_hashes() -> None:
    universe = ["ENSG000001", "ENSG000003"]
    detected = ["ENSG000003"]
    row = {
        "cancer_id": "TEST",
        "lncRNA_universe_ids": universe,
        "detected_lncRNA_ids": detected,
        "cell_count": 3,
        "feature_count": 4,
        "stored_nnz": 5,
        "h5_path": "/read/only/test.h5",
        "h5_size_bytes": 10,
        "h5_mtime_utc": "2026-01-01T00:00:00+00:00",
        "matrix_orientation_validated": True,
        "detection_value_threshold": "FINITE_VALUE_GT_0",
        "minimum_detected_cell_count": 1,
    }
    reference = {
        "formal_policy_lncRNA_universe_count": 2,
        "formal_policy_detected_lncRNA_count": 1,
        "formal_policy_universe_sha256": materialize.set_sha256(universe),
        "formal_policy_detected_sha256": materialize.set_sha256(detected),
        "cell_count": 3,
        "feature_count": 4,
        "stored_nnz": 5,
        "h5_path": "/read/only/test.h5",
        "h5_size_bytes": 10,
        "h5_mtime_utc": "2026-01-01T00:00:00+00:00",
    }
    materialize.validate_id_row(row, reference)
    row["detected_lncRNA_ids"] = ["ENSG000001"]
    with pytest.raises(materialize.ExplicitIdAuditError, match="set_sha256"):
        materialize.validate_id_row(row, reference)
