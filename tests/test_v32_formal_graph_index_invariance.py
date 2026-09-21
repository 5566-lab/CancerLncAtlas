from __future__ import annotations

import pandas as pd

from cc_hhgt.v32.formal_graph import (
    materialize_expressed_in,
    materialize_global_lnc_protein_binding,
    materialize_pathway_hierarchy,
    materialize_protein_gene_encoding,
    materialize_signed_coexpression,
    materialize_signed_membership,
    materialize_symmetric_ppi,
)


def test_materializers_are_invariant_to_nonconsecutive_source_indices() -> None:
    candidate = pd.DataFrame({"cancer_id": ["BRCA"], "lncrna_id": ["L1"]})
    sources = [
        (
            materialize_expressed_in,
            pd.DataFrame({"cancer_id": ["BRCA"], "lncrna_id": ["L1"], "detection_rate": [0.7]}),
            {"candidate_pairs": candidate},
        ),
        (
            materialize_signed_coexpression,
            pd.DataFrame(
                {
                    "cancer_id": ["BRCA", "BRCA"],
                    "lncrna_id": ["L1", "L1"],
                    "gene_id": ["G1", "G2"],
                    "rho": [0.8, -0.6],
                    "source_split": ["train", "train"],
                    "edge_outer_fold": [2, 2],
                }
            ),
            {"outer_fold": 2},
        ),
        (
            materialize_signed_membership,
            pd.DataFrame({"gene_id": ["G1", "G2"], "pathway_id": ["P1", "P1"], "weight": [1.0, -0.5]}),
            {},
        ),
        (
            materialize_pathway_hierarchy,
            pd.DataFrame({"pathway_id": ["P1", "P2"], "pathway_family_id": ["F1", "F2"], "weight": [1.0, 0.5]}),
            {},
        ),
        (
            materialize_global_lnc_protein_binding,
            pd.DataFrame(
                {
                    "lncrna_id": ["L1"],
                    "protein_id": ["PR1"],
                    "weight": [0.9],
                    "cancer_id": [pd.NA],
                    "is_context_specific": [False],
                }
            ),
            {"candidate_lncrnas": ["L1"]},
        ),
        (
            materialize_protein_gene_encoding,
            pd.DataFrame({"protein_id": ["PR1"], "gene_id": ["G1"], "weight": [1.0]}),
            {},
        ),
        (
            materialize_symmetric_ppi,
            pd.DataFrame({"protein_id_a": ["PR1"], "protein_id_b": ["PR2"], "weight": [0.8]}),
            {},
        ),
    ]
    for materializer, source, kwargs in sources:
        expected = materializer(source.copy(), **kwargs)
        shifted = source.copy()
        shifted.index = pd.Index(range(101, 101 + len(shifted)))
        observed = materializer(shifted, **kwargs)
        pd.testing.assert_frame_equal(observed, expected)

