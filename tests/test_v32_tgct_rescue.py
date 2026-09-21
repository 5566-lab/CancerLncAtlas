from scripts.materialize_v32_tgct_gse261811_rescue import canonical_gene_id


def test_canonical_gene_id_removes_namespace_and_version() -> None:
    assert canonical_gene_id("LNC:ENSG000001234.7") == "ENSG000001234"
    assert canonical_gene_id("ENSG000005678.2") == "ENSG000005678"
