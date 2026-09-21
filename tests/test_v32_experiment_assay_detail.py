from __future__ import annotations

import pandas as pd

from cc_hhgt.v32.experiment_assay_detail import (
    build_assay_detail,
    classify_assay_text,
    parse_pubmed_xml,
)


def _evidence() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "evidence_event_id": "E1",
                "pmid": "101",
                "lncrna_id": "LNC:1",
                "partner_id": "GENE:1",
                "experiment_family": "functional_perturbation",
            },
            {
                "evidence_event_id": "E2",
                "pmid": "102",
                "lncrna_id": "LNC:2",
                "partner_id": "GENE:2",
                "experiment_family": "functional_perturbation",
            },
            {
                "evidence_event_id": "E3",
                "pmid": "103",
                "lncrna_id": "LNC:3",
                "partner_id": "GENE:3",
                "experiment_family": "functional_perturbation",
            },
            {
                "evidence_event_id": "IGNORED",
                "pmid": "104",
                "lncrna_id": "LNC:1",
                "partner_id": "GENE:1",
                "experiment_family": "physical_binding",
            },
        ]
    )


def _ncpath() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pmid": "101",
                "lnc_symbol_key": "LNCA",
                "gene_symbol_key": "GENEA",
                "assay_detail_source": "siRNA knockdown; western blot",
                "interID": "N1",
                "datasource": "RNAInter",
                "source_context": "A549",
            },
            {
                "pmid": "102",
                "lnc_symbol_key": "OTHER",
                "gene_symbol_key": "OTHER",
                "assay_detail_source": "overexpression; luciferase assay",
                "interID": "N2",
                "datasource": "RNAInter",
                "source_context": "HeLa",
            },
        ]
    )


def test_classify_assay_text_keeps_perturbation_and_readout_separate() -> None:
    value = classify_assay_text("CRISPRi knockdown followed by RNA-seq")
    assert value["perturbation_methods"] == "CRISPR_INTERFERENCE;KNOCKDOWN_OR_SILENCING"
    assert value["readout_assays"] == "RNA_SEQ"


def test_build_assay_detail_routes_exact_unique_and_pubmed() -> None:
    dim_lnc = pd.DataFrame(
        {"lncrna_id": ["LNC:1", "LNC:2", "LNC:3"], "gene_symbol": ["LNCA", "LNCB", "LNCC"]}
    )
    dim_gene = pd.DataFrame(
        {"gene_id": ["GENE:1", "GENE:2", "GENE:3"], "gene_symbol": ["GENEA", "GENEB", "GENEC"]}
    )
    result = build_assay_detail(
        _evidence(),
        _ncpath(),
        dim_lnc,
        dim_gene,
        pubmed_records={
            "103": {
                "title": "CRISPRi screen",
                "abstract": "The lncRNA was silenced and quantified by qRT-PCR.",
            }
        },
    ).set_index("evidence_event_id")
    assert result.index.tolist() == ["E1", "E2", "E3"]
    assert result.loc["E1", "match_route"] == "PMID_LNCRNA_GENE_EXACT"
    assert result.loc["E1", "manual_review_required"] == False  # noqa: E712
    assert result.loc["E2", "match_route"] == "PMID_UNIQUE_SOURCE_ASSAY"
    assert result.loc["E2", "manual_review_required"] == True  # noqa: E712
    assert result.loc["E3", "match_route"] == "PUBMED_TITLE_ABSTRACT_KEYWORDS"
    assert result.loc["E3", "perturbation_method_available"] == True  # noqa: E712
    assert result.changes_primary_ranking.eq(False).all()


def test_parse_pubmed_xml() -> None:
    payload = b"""<PubmedArticleSet><PubmedArticle><MedlineCitation>
    <PMID>123</PMID><Article><ArticleTitle>Title <i>with</i> tag</ArticleTitle>
    <Abstract><AbstractText Label='A'>First.</AbstractText><AbstractText>Second.</AbstractText></Abstract>
    </Article></MedlineCitation></PubmedArticle></PubmedArticleSet>"""
    assert parse_pubmed_xml(payload) == {
        "123": {"title": "Title with tag", "abstract": "First. Second."}
    }
