from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from cc_hhgt.v32.single_cell_graph_assets import (
    aggregate_frozen_measurements,
    align_frozen_membership,
    context_feature_scope,
    gene_by_pathway_membership,
    lasso_observations,
    normalize_external_evidence,
    normalize_pathway_families,
    UNASSIGNED_PATHWAY_FAMILY_ID,
)


def test_frozen_membership_is_aligned_by_cell_id_without_regrouping():
    metadata = pd.DataFrame(
        {
            "cell_id": ["c1", "c2", "c3", "c4"],
            "donor_id": ["d1", "d1", "d2", "d2"],
            "cell_type": ["T", "T", "T", "T"],
            "state_id": ["N", "N", "N", "N"],
        }
    )
    frozen = metadata.iloc[[2, 0, 3, 1]].copy()
    frozen["metacell_id"] = [1, 0, 1, 0]
    assignments = align_frozen_membership(metadata, frozen)
    np.testing.assert_array_equal(assignments, [0, 0, 1, 1])

    coding = sparse.csr_matrix([[1.0], [3.0], [4.0], [8.0]])
    lnc = sparse.csr_matrix([[2.0], [6.0], [1.0], [3.0]])
    coding_mc, lnc_mc, ids, policy = aggregate_frozen_measurements(
        coding,
        lnc,
        assignments,
        measurement_scale="log_normalized",
        all_gene_library=np.ones(4),
    )
    np.testing.assert_array_equal(ids, [0, 1])
    np.testing.assert_allclose(coding_mc.toarray().ravel(), [2.0, 6.0])
    np.testing.assert_allclose(lnc_mc.toarray().ravel(), [4.0, 2.0])
    assert policy == "PROVIDED_LOG_SCALE_ARITHMETIC_MEAN_NO_INVERSION"


def test_frozen_membership_rejects_label_or_cell_set_drift():
    metadata = pd.DataFrame(
        {
            "cell_id": ["c1"], "donor_id": ["d1"], "cell_type": ["T"],
            "state_id": ["N"],
        }
    )
    frozen = metadata.assign(metacell_id=0)
    with pytest.raises(ValueError, match="DONOR_ID_DRIFT"):
        align_frozen_membership(metadata, frozen.assign(donor_id="heldout"))
    with pytest.raises(ValueError, match="CELL_SET_DRIFT"):
        align_frozen_membership(metadata, frozen.assign(cell_id="other"))


def test_lasso_observations_mark_query_rows_forbidden():
    frozen = pd.DataFrame(
        {
            "metacell_id": [1, 0],
            "donor_id": ["query", "source"],
            "cell_type": ["T", "T"],
            "state_id": ["N", "N"],
            "n_cells": [4, 5],
        }
    )
    result = lasso_observations(
        frozen, np.array([0, 1]), source_donors=["source"], query_donors=["query"]
    )
    assert result.observation_role.tolist() == ["source", "query"]
    assert result.lasso_input_allowed.tolist() == [True, False]
    with pytest.raises(ValueError, match="SC_GRAPH_DONOR_LEAK"):
        lasso_observations(
            frozen,
            np.array([0, 1]),
            source_donors=["source", "query"],
            query_donors=["query"],
        )


def test_signature_orientation_is_explicit_gene_by_pathway():
    pathway_by_gene = sparse.csr_matrix([[1, 0, 1], [0, 1, 1]])
    result = gene_by_pathway_membership(
        pathway_by_gene, gene_count=3, pathway_count=2
    )
    assert result.shape == (3, 2)
    np.testing.assert_array_equal(result.toarray(), [[1, 0], [0, 1], [1, 1]])
    with pytest.raises(ValueError, match="SHAPE"):
        gene_by_pathway_membership(
            pathway_by_gene.transpose(), gene_count=3, pathway_count=2
        )


def test_family_contract_preserves_hierarchy_and_selects_one_representative():
    hierarchy = pd.DataFrame(
        {
            "pathway_id": ["p1", "p1", "p2"],
            "pathway_family_id": ["f1", "f2", "f2"],
            "membership_weight": [1.0, 0.5, 1.0],
            "is_representative": [True, False, True],
        }
    )
    membership, full, families = normalize_pathway_families(hierarchy, ["p2", "p1"])
    np.testing.assert_allclose(membership.toarray(), [[0.0, 1.0], [1.0, 0.5]])
    assert len(full) == 3
    assert families.canonical_id.tolist() == ["f1", "f2"]


def test_family_contract_assigns_explicit_fallback_to_unmapped_pathway():
    hierarchy = pd.DataFrame(
        {
            "pathway_id": ["p1"],
            "pathway_family_id": ["f1"],
            "membership_weight": [1.0],
            "is_representative": [True],
        }
    )
    membership, full, families = normalize_pathway_families(
        hierarchy, ["p1", "p2"]
    )
    assert membership.getnnz(axis=1).tolist() == [1, 1]
    fallback = full.loc[full.pathway_id.eq("p2")].iloc[0]
    assert fallback.pathway_family_id == UNASSIGNED_PATHWAY_FAMILY_ID
    assert not bool(fallback.is_representative)
    assert UNASSIGNED_PATHWAY_FAMILY_ID in set(families.canonical_id)


def test_context_scope_is_exact_universe_subset_with_stable_local_indices():
    universe = pd.DataFrame(
        {
            "cell_key": ["B", "A", "A"],
            "lncrna_id": ["l2", "l1", "l2"],
            "pathway_id": ["p1", "p2", "p1"],
        }
    )
    scope = context_feature_scope(universe, ["l1", "l2"], ["p1", "p2"])
    a_lnc = scope.loc[scope.cell_key.eq("A") & scope.node_type.eq("lncRNA")]
    assert a_lnc.canonical_id.tolist() == ["l1", "l2"]
    assert a_lnc.context_index.tolist() == [0, 1]


def test_external_edges_accumulate_same_relation_and_schema_is_global():
    raw = pd.DataFrame(
        {
            "lncrna_id": ["LNC:ENSG1", "ENSG1", "ENSG2"],
            "partner_id": ["GENE:ENSG10", "ENSG11", "ENSG12"],
            "partner_type": ["gene", "gene", "gene"],
            "relation_type": ["binding", "binding", "activation"],
            "direction": ["direct", "direct", "positive"],
            "experiment_family": ["physical_binding"] * 3,
            "is_experimental": [True, True, True],
            "is_predicted": [False, False, False],
            "pmid": ["1", "2", "3"],
            "source_database": ["A", "B", "C"],
            "mapping_status": ["lncrna_and_partner_mapped"] * 3,
        }
    )
    edges, schema = normalize_external_evidence(
        raw, lncrna_ids=["ENSG1"], gene_ids=["ENSG10", "ENSG11"]
    )
    assert len(edges) == 2
    assert edges.relation.nunique() == 1
    assert len(schema) == 2  # activation remains registered as an empty relation.
    assert edges.edge_weight.eq(1.0).all()
