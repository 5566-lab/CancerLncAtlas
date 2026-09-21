import pandas as pd

from cc_hhgt.explain import build_adjacency, top_paths


def test_explanation_graph_is_directional_except_explicit_symmetric_relations():
    edges = pd.DataFrame(
        {
            "source_node_id": ["lnc", "g1"],
            "target_node_id": ["g1", "g2"],
            "relation_type": ["coexpressed_with", "physical_interaction"],
            "weight": [0.9, 0.8],
            "edge_id": ["e1", "e2"],
        }
    )
    adjacency = build_adjacency(edges)
    assert [x[0] for x in adjacency["lnc"]] == ["g1"]
    assert "lnc" not in [x[0] for x in adjacency["g1"]]
    assert "g1" in [x[0] for x in adjacency["g2"]]


def test_best_first_path_search_honors_expansion_budget():
    adjacency = {
        "source": [("dead", "rel", 1.0, "e0"), ("target", "rel", 0.8, "e1")],
        "dead": [("dead2", "rel", 1.0, "e2")],
    }
    assert top_paths(adjacency, "source", {"target"}, max_expansions=1)[0][0] == 0.8
    assert top_paths(adjacency, "source", {"missing"}, max_expansions=1) == []
