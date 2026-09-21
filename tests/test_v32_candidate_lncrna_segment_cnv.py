from __future__ import annotations

import pandas as pd

from scripts.map_v32_candidate_lncrna_segment_cnv import (
    candidate_lncrna_ids,
)


def test_candidate_catalog_gene_ids_are_promoted_to_formal_lnc_namespace() -> None:
    catalog = pd.DataFrame({"gene_id": ["ENSG000001.7", "ENSG000002"]})
    assert candidate_lncrna_ids(catalog) == {
        "LNC:ENSG000001",
        "LNC:ENSG000002",
    }


def test_existing_formal_lnc_namespace_is_preserved_and_unversioned() -> None:
    catalog = pd.DataFrame({"lncrna_id": ["LNC:ENSG000001.7", "LNC:ENSG000002"]})
    assert candidate_lncrna_ids(catalog) == {
        "LNC:ENSG000001",
        "LNC:ENSG000002",
    }
