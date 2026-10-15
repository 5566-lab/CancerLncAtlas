from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
import torch

from cc_hhgt.v32.single_cell_batched_lasso import BatchedLassoResult
from cc_hhgt.v32.single_cell_hhgt_episodes import (
    LASSO_DERIVED_PAIR_FEATURE_MASK,
    PAIR_FEATURE_NAMES,
    episode_pair_features,
    lasso_pair_features,
    local_node_indices,
    localized_lasso_edges,
    source_context_features,
    sparse_coo_payload,
)


def _result() -> BatchedLassoResult:
    return BatchedLassoResult(
        response_count=2,
        gene_count=4,
        response_ids=("L0", "L1"),
        gene_ids=("G0", "G1", "G2", "G3"),
        response_index=np.asarray([0, 0, 1]),
        gene_index=np.asarray([0, 2, 3]),
        coefficient=np.asarray([2.0, -1.0, 0.5], dtype=np.float32),
        stability=np.asarray([1.0, 0.25, 0.75], dtype=np.float32),
        available=np.asarray([True, False]),
        unavailable_reason=np.asarray(["", "NO_BOOTSTRAP"], dtype=object),
        selected_alpha=np.asarray([0.1, np.nan]),
        inner_cv_mse=np.asarray([0.25, np.nan]),
        inner_null_mse=np.asarray([1.0, np.nan]),
        bootstrap_successes=np.asarray([8, 0]),
        audit={"held_out_donors_used": 0},
    )


def test_source_features_ignore_excluded_rows():
    observations = pd.DataFrame(
        {
            "metacell_row": np.arange(8),
            "donor_id": ["d0", "d0", "d1", "d1", "d2", "d2", "held", "held"],
            "cell_key": ["T"] * 8,
        }
    )
    coding = sparse.csr_matrix(np.arange(24, dtype=np.float32).reshape(8, 3))
    lncrna = sparse.csr_matrix(np.arange(16, dtype=np.float32).reshape(8, 2))
    activity = np.arange(16, dtype=np.float32).reshape(8, 2)
    family = sparse.csr_matrix(np.eye(2, dtype=np.float32))
    first = source_context_features(
        coding,
        lncrna,
        activity,
        observations,
        cell_key="T",
        source_donors=("d0", "d1", "d2"),
        pathway_family_membership=family,
    )
    coding.data[-6:] = 1e6
    lncrna.data[-4:] = 1e6
    activity[-2:] = 1e6
    second = source_context_features(
        coding,
        lncrna,
        activity,
        observations,
        cell_key="T",
        source_donors=("d0", "d1", "d2"),
        pathway_family_membership=family,
    )
    for node_type in first:
        np.testing.assert_allclose(first[node_type], second[node_type])


def test_source_features_allow_two_context_donors_only_when_explicit():
    observations = pd.DataFrame(
        {
            "metacell_row": np.arange(5),
            "donor_id": ["d0", "d0", "d1", "d1", "held"],
            "cell_key": ["NK"] * 5,
        }
    )
    coding = sparse.csr_matrix(np.arange(10, dtype=np.float32).reshape(5, 2))
    lncrna = sparse.csr_matrix(np.arange(5, dtype=np.float32).reshape(5, 1))
    activity = np.arange(5, dtype=np.float32).reshape(5, 1)
    family = sparse.csr_matrix(np.ones((1, 1), dtype=np.float32))
    arguments = {
        "cell_key": "NK",
        "source_donors": ("d0", "d1"),
        "pathway_family_membership": family,
    }
    with pytest.raises(ValueError, match="SOURCE_DONORS_LT3"):
        source_context_features(
            coding, lncrna, activity, observations, **arguments
        )
    value = source_context_features(
        coding,
        lncrna,
        activity,
        observations,
        minimum_context_donors=2,
        **arguments,
    )
    assert value["cell_context"][0, 0] == pytest.approx(np.log1p(2))
    with pytest.raises(ValueError, match="SOURCE_DONORS_LT2"):
        source_context_features(
            coding,
            lncrna,
            activity,
            observations,
            minimum_context_donors=2,
            **{**arguments, "source_donors": ("d0",)},
        )


def test_lasso_pair_features_are_signed_edge_aware_and_response_scoped():
    membership = sparse.csr_matrix(
        np.asarray(
            [
                [1, 0],
                [1, 1],
                [0, 1],
                [0, 1],
            ],
            dtype=np.float32,
        )
    )
    value = lasso_pair_features(
        _result(),
        membership,
        np.asarray([0, 0, 1]),
        np.asarray([0, 1, 1]),
        np.asarray([0, 1]),
        stability_threshold=0.5,
    )
    np.testing.assert_allclose(value[:, 0], [np.log1p(2 / np.sqrt(2)), 0, np.log1p(0.375 / np.sqrt(3))])
    np.testing.assert_allclose(value[:, 1], [0.75, 0.75, 0.0])
    np.testing.assert_allclose(value[:, 2], [1.0, 1.0, 0.0])


def test_local_nodes_and_lasso_edges_preserve_global_identity():
    membership = sparse.csr_matrix(
        np.asarray([[1, 0], [0, 1], [0, 1], [0, 0]], dtype=np.float32)
    )
    family = sparse.csr_matrix(np.asarray([[1, 0], [0, 1]], dtype=np.float32))
    external = pd.DataFrame({"source_index": [4], "target_index": [3], "edge_weight": [1.0]})
    nodes = local_node_indices(
        lnc_global=np.asarray([4, 7]),
        pathway_global=np.asarray([1]),
        lasso_result=_result(),
        gene_pathway_membership=membership,
        pathway_family_membership=family,
        external_edges=external,
        stability_threshold=0.5,
    )
    np.testing.assert_array_equal(nodes["lncRNA"], [4, 7])
    np.testing.assert_array_equal(nodes["gene"], [0, 1, 2, 3])
    np.testing.assert_array_equal(nodes["pathway"], [1])
    np.testing.assert_array_equal(nodes["pathway_family"], [1])
    edges = localized_lasso_edges(_result(), np.asarray([4, 7]), nodes)
    assert edges["lnc_index"].tolist() == [0, 0, 1]
    assert edges["gene_index"].tolist() == [0, 2, 3]
    assert torch.allclose(edges["coefficient"], torch.tensor([2.0, -1.0, 0.5]))


def test_without_lasso_arm_has_no_lasso_derived_pair_signal():
    rows = pd.DataFrame(
        {
            "source_meta_rho": [0.25, -0.5],
            "source_direction_consistency": [0.8, 0.6],
            "source_donor_count": [4, 9],
        }
    )
    value = episode_pair_features(
        rows, np.asarray([[1.0, 0.5, 1.0], [2.0, -0.2, 0.0]], dtype=np.float32)
    )
    assert value["with_lasso"].shape == (2, len(PAIR_FEATURE_NAMES))
    mask = torch.as_tensor(LASSO_DERIVED_PAIR_FEATURE_MASK)
    assert torch.count_nonzero(value["with_lasso"][:, mask]) > 0
    assert torch.count_nonzero(value["without_lasso"][:, mask]) == 0
    assert torch.equal(value["with_lasso"][:, ~mask], value["without_lasso"][:, ~mask])


def test_sparse_coo_payload_keeps_orientation_and_weights():
    matrix = sparse.csr_matrix(np.asarray([[0, 2], [3, 0]], dtype=np.float32))
    value = sparse_coo_payload(matrix)
    assert value["shape"] == [2, 2]
    observed = sparse.coo_matrix(
        (value["weight"].numpy(), (value["row_index"].numpy(), value["column_index"].numpy())),
        shape=value["shape"],
    ).toarray()
    np.testing.assert_array_equal(observed, matrix.toarray())
