import numpy as np
import pandas as pd

from cc_hhgt.gnn import fold_node_features


def test_fold_node_degree_does_not_use_global_node_degree_columns():
    nodes = pd.DataFrame(
        {
            "canonical_id": ["a", "b"],
            "log_degree": [99.0, 99.0],
            "bias_feature": [1.0, 1.0],
        }
    )
    filtered_edges = pd.DataFrame(
        {
            "source_type": ["lncRNA"],
            "target_type": ["gene"],
            "source_canonical_id": ["a"],
            "target_canonical_id": ["g"],
        }
    )
    features = fold_node_features(nodes, filtered_edges, "lncRNA")
    assert np.allclose(features[:, 0], [np.log1p(1), 0.0])
    assert np.allclose(features[:, 1], 1.0)
