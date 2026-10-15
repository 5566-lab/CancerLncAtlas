import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from cc_hhgt.v32.single_cell_metacells import (
    LocalSpace,
    aggregate_measurements,
    local_disjoint_groups,
)


def test_grouping_never_mixes_donor_type_or_state_and_never_reuses_cells():
    rng = np.random.default_rng(4)
    metadata = pd.DataFrame(
        {
            "donor_id": np.repeat(["a", "b"], 120),
            "cell_type": np.tile(np.repeat(["T", "B"], 60), 2),
            "state_id": np.tile(np.repeat(["s1", "s2"], 30), 4),
        }
    )
    groups, audit = local_disjoint_groups(
        rng.normal(size=(240, 5)),
        metadata,
        allowed_training_donors=["a"],
        target_size=5,
    )
    assert audit["overlapping_cells"] == 0
    assert (groups >= 0).any()
    for label in np.unique(groups[groups >= 0]):
        selected = metadata.loc[groups == label]
        assert 3 <= len(selected) <= 5
        assert all(selected[column].nunique() == 1 for column in metadata.columns)
    assert sum(row["unassigned_cells"] for row in audit["contexts"]) == (groups < 0).sum()


def test_query_extremes_cannot_change_source_fitted_coding_space():
    rng = np.random.default_rng(9)
    expression = rng.poisson(2, (90, 12)).astype(float)
    donors = np.repeat(["a", "b", "query"], 30)
    first = LocalSpace.fit(expression, donors, ["a", "b"], components=4)
    expression[donors == "query"] = 100000
    second = LocalSpace.fit(expression, donors, ["a", "b"], components=4)
    np.testing.assert_array_equal(first.genes, second.genes)
    np.testing.assert_array_equal(first.mean, second.mean)
    np.testing.assert_array_equal(first.pca.components_, second.pca.components_)


def test_raw_sum_uses_all_gene_library_and_log_input_is_never_inverted():
    expression = sparse.csr_matrix([[1.0, 3.0], [2.0, 0.0], [4.0, 5.0], [6.0, 7.0]])
    labels = np.array([0, 0, 1, -1])
    library = np.array([10.0, 20.0, 30.0, 40.0])
    raw, _, raw_policy = aggregate_measurements(
        expression, labels, scale="raw_counts", total_library=library
    )
    np.testing.assert_allclose(
        raw.toarray(),
        np.log1p(np.array([[3.0, 3.0], [4.0, 5.0]]) * 10000 / np.array([[30.0], [30.0]])),
    )
    normalized, _, log_policy = aggregate_measurements(
        expression, labels, scale="log_normalized"
    )
    np.testing.assert_allclose(normalized.toarray(), [[1.5, 1.5], [4.0, 5.0]])
    assert "RAW_SUM" in raw_policy
    assert "NO_INVERSION" in log_policy
    with pytest.raises(ValueError, match="LIBRARY"):
        aggregate_measurements(expression, labels, scale="raw_counts")
