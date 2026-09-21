from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pytest
from fastapi import FastAPI, HTTPException


@pytest.fixture
def site_store_class(monkeypatch: pytest.MonkeyPatch):
    """Load the legacy site store without mounting production-only services."""

    monkeypatch.setenv("CANCERLNCATLAS_REQUIRE_V31_EXACT_PATHWAY", "0")
    monkeypatch.setitem(sys.modules, "psutil", types.ModuleType("psutil"))
    package = types.ModuleType("cc_hhgt_v26")
    api = types.ModuleType("cc_hhgt_v26.api")
    api.app = FastAPI()
    api.engine = lambda: None
    monkeypatch.setitem(sys.modules, "cc_hhgt_v26", package)
    monkeypatch.setitem(sys.modules, "cc_hhgt_v26.api", api)
    sys.modules.pop("website.backend.app", None)
    module = importlib.import_module("website.backend.app")
    try:
        yield module.SiteStore
    finally:
        sys.modules.pop("website.backend.app", None)


def _bare_store(site_store_class, dataset) -> object:
    store = object.__new__(site_store_class)
    store._single_cell_dataset = dataset
    store.cancers = ["LUAD"]
    return store


def test_sc_summary_filters_pathway_before_limit_and_normalizes_cancer(
    tmp_path: Path, site_store_class
) -> None:
    rows = [
        {
            "lncrna_id": "LNC:ENSG000001",
            "cancer_id": "LUAD",
            "pathway_id": "PATHWAY:OTHER",
            "cell_type": "malignant",
            "rho": 0.1,
        }
        for _ in range(600)
    ]
    rows.append(
        {
            "lncrna_id": "LNC:ENSG000001",
            "cancer_id": "LUAD",
            "pathway_id": "PATHWAY:TARGET",
            "cell_type": "malignant",
            "rho": 0.8,
        }
    )
    data_root = tmp_path / "single_cell"
    data_root.mkdir()
    pd.DataFrame(rows).to_parquet(data_root / "part-0.parquet", index=False)
    store = _bare_store(site_store_class, ds.dataset(data_root, format="parquet"))

    result = store.sc_summary(
        " luad ",
        lnc="LNC:ENSG000001",
        pathway_id="PATHWAY:TARGET",
    )

    assert result["status"] == "available"
    assert result["cancer_id"] == "LUAD"
    assert result["returned"] == 1
    assert result["rows"][0]["pathway_id"] == "PATHWAY:TARGET"


def test_single_cell_uses_early_stopping_head_instead_of_full_to_table(
    site_store_class,
) -> None:
    class HeadOnlyDataset:
        schema = pa.schema(
            [
                ("lncrna_id", pa.string()),
                ("cancer_id", pa.string()),
                ("pathway_id", pa.string()),
            ]
        )

        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def head(self, num_rows, *, filter=None, columns=None):
            self.calls.append(
                {"num_rows": num_rows, "filter": str(filter), "columns": columns}
            )
            return pa.table(
                {
                    "lncrna_id": ["LNC:ENSG000001"],
                    "cancer_id": ["LUAD"],
                    "pathway_id": ["PATHWAY:TARGET"],
                }
            )

        def to_table(self, *args, **kwargs):  # pragma: no cover - regression tripwire
            raise AssertionError("single_cell must not materialize the full table")

    dataset = HeadOnlyDataset()
    store = _bare_store(site_store_class, dataset)
    result = store.single_cell(
        "LNC:ENSG000001",
        " luad ",
        7,
        pathway_id="PATHWAY:TARGET",
    )

    assert result["returned"] == 1
    assert result["cancer_id"] == "LUAD"
    assert dataset.calls[0]["num_rows"] == 7
    assert "LUAD" in dataset.calls[0]["filter"]
    assert "PATHWAY:TARGET" in dataset.calls[0]["filter"]


def test_lnc_profile_normalizes_cancer_before_exact_store_lookup(
    site_store_class,
) -> None:
    class ExactStore:
        def __init__(self) -> None:
            self.seen_cancer = None

        def lnc_profile(self, lnc_id, cancer):
            self.seen_cancer = cancer
            return {"top_relationships": [], "cancer_landscape": []}

    store = object.__new__(site_store_class)
    store.exact = ExactStore()
    store.query = SimpleNamespace(
        interaction=pd.DataFrame(columns=["lncrna_id", "n_evidence_events"])
    )
    store.lnc_symbol = {"LNC:ENSG000001": "LINC1"}

    result = store.lnc_profile("LNC:ENSG000001", " luad ")

    assert store.exact.seen_cancer == "LUAD"
    assert result["cancer_id"] == "LUAD"


def test_ambiguous_legacy_lncrna_symbol_is_not_silently_bound(
    site_store_class,
) -> None:
    store = object.__new__(site_store_class)
    store.lnc_lookup = pd.DataFrame(
        [
            {
                "lncrna_id": "LNC:ENSG000001",
                "ensembl_gene_id": "ENSG000001",
                "gene_symbol": "DUPLICATE",
            },
            {
                "lncrna_id": "LNC:ENSG000002",
                "ensembl_gene_id": "ENSG000002",
                "gene_symbol": "DUPLICATE",
            },
        ]
    )

    with pytest.raises(HTTPException) as raised:
        store.resolve_lnc("duplicate")

    assert raised.value.status_code == 409
    assert raised.value.detail["reason"] == "AMBIGUOUS_LNCRNA_IDENTIFIER"
    assert raised.value.detail["candidate_lncrna_ids"] == [
        "LNC:ENSG000001",
        "LNC:ENSG000002",
    ]
