"""Model-facing SC-HHGT LASSO derivation replay: positive and fault cases."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import sparse

from cc_hhgt.v32.single_cell_batched_lasso import BatchedLassoResult
from cc_hhgt.v32.single_cell_hhgt_episodes import (
    LASSO_DERIVED_PAIR_FEATURE_MASK,
    PAIR_FEATURE_NAMES,
    lasso_pair_features,
    local_node_indices,
    localized_lasso_edges,
)
from cc_hhgt.v32.single_cell_lasso_replay import FORMAT, verify_episode, verify_episodes

THRESHOLD = 0.5
IDENTITIES = {
    "lncRNA": ["LNC:Z1", "LNC:Z2", "LNC:Z3"],
    "gene": ["G1", "G2", "G3", "G4", "G5", "G6"],
    "pathway": ["PATH:a", "PATH:b", "PATH:c"],
    "pathway_family": ["FAM:one", "FAM:two"],
}
ROW_METADATA = {
    "dataset_id": ["D1"] * 4,
    "cancer_id": ["BRCA"] * 4,
    "cell_key": ["ct|s1", "ct|s1", "ct|s2", "ct|s2"],
    "query_donor_id": ["P1", "P1", "P2", "P2"],
    "lncrna_id": ["LNC:Z1", "LNC:Z2", "LNC:Z2", "LNC:Z1"],
    "pathway_id": ["PATH:a", "PATH:b", "PATH:c", "PATH:a"],
}


def _static():
    gene_pathway = np.zeros((6, 3), dtype=np.float32)
    gene_pathway[0, 0] = 1.0
    gene_pathway[1, 1] = 1.0
    gene_pathway[2, 1] = 1.0
    gene_pathway[3, 2] = 1.0
    family = np.zeros((3, 2), dtype=np.float32)
    family[0, 0] = 1.0
    family[1, 0] = 1.0
    family[2, 1] = 1.0
    external = pd.DataFrame({"source_index": [], "target_index": [], "edge_weight": []})
    return {
        "identifiers": IDENTITIES,
        "gene_pathway": sparse.csr_matrix(gene_pathway),
        "pathway_family": sparse.csr_matrix(family),
        "external": external,
    }


def _result():
    """Two responses; recorded evidence only, no LASSO was fitted here."""
    return BatchedLassoResult(
        response_count=2,
        gene_count=6,
        response_ids=tuple(IDENTITIES["lncRNA"][:2]),
        gene_ids=tuple(IDENTITIES["gene"]),
        response_index=np.asarray([0, 0, 1], dtype=np.int64),
        gene_index=np.asarray([0, 2, 1], dtype=np.int64),
        coefficient=np.asarray([0.4, -0.3, 0.2], dtype=np.float64),
        stability=np.asarray([0.9, 0.6, 0.8], dtype=np.float64),
        available=np.asarray([True, True]),
        unavailable_reason=np.asarray(["", ""], dtype=object),
        selected_alpha=np.asarray([0.1, 0.03], dtype=np.float64),
        inner_cv_mse=np.asarray([0.21, 0.33], dtype=np.float64),
        inner_null_mse=np.asarray([0.42, 0.44], dtype=np.float64),
        bootstrap_successes=np.asarray([8, 7], dtype=np.int64),
        audit={"format": "SYNTHETIC_RECORDED_EVIDENCE", "solver": "NOT_EXECUTED"},
    )


def _payload():
    static = _static()
    result = _result()
    row_lnc = np.asarray([0, 1, 1, 0], dtype=np.int64)
    row_pathway = np.asarray([0, 1, 2, 0], dtype=np.int64)
    nodes = local_node_indices(
        lnc_global=row_lnc,
        pathway_global=row_pathway,
        lasso_result=result,
        gene_pathway_membership=static["gene_pathway"],
        pathway_family_membership=static["pathway_family"],
        external_edges=static["external"],
        stability_threshold=THRESHOLD,
    )
    response_global = np.unique(row_lnc)
    edges = localized_lasso_edges(result, response_global, nodes)
    features = lasso_pair_features(
        result,
        static["gene_pathway"],
        np.asarray([0, 1, 1, 0], dtype=np.int64),
        row_pathway,
        np.unique(row_pathway),
        stability_threshold=THRESHOLD,
    )
    mask = np.asarray(LASSO_DERIVED_PAIR_FEATURE_MASK, dtype=bool)
    base = np.asarray([[0.11, 0.62, 1.7], [0.23, 0.71, 1.9], [0.31, 0.55, 1.2], [0.07, 0.83, 2.1]])
    with_lasso = np.column_stack((base, features)).astype(np.float32)
    without_lasso = with_lasso.copy()
    without_lasso[:, mask] = 0.0
    return {
        "lasso_stability_threshold": THRESHOLD,
        "lasso_response_summary": {
            "response_index": [0, 1],
            "response_id": list(result.response_ids),
            "available": [True, True],
            "unavailable_reason": ["", ""],
            "selected_alpha": [0.1, 0.03],
            "inner_cv_mse": [0.21, 0.33],
            "inner_null_mse": [0.42, 0.44],
            "bootstrap_successes": [8, 7],
        },
        "node_global_indices": {kind: torch.as_tensor(value, dtype=torch.long) for kind, value in nodes.items()},
        "sparse_lasso_edges": edges,
        "pair_features_by_arm": {
            "with_lasso": torch.as_tensor(with_lasso, dtype=torch.float32),
            "without_lasso": torch.as_tensor(without_lasso, dtype=torch.float32),
        },
        "row_metadata": {key: list(value) for key, value in ROW_METADATA.items()},
        "lasso_audit": {"format": "SYNTHETIC_RECORDED_EVIDENCE", "solver": "NOT_EXECUTED"},
    }


def _fault(payload, kind):
    value = dict(payload)
    value["sparse_lasso_edges"] = dict(payload["sparse_lasso_edges"])
    value["node_global_indices"] = dict(payload["node_global_indices"])
    value["pair_features_by_arm"] = dict(payload["pair_features_by_arm"])
    value["row_metadata"] = {key: list(items) for key, items in payload["row_metadata"].items()}
    value["lasso_response_summary"] = {
        key: list(items) for key, items in payload["lasso_response_summary"].items()
    }
    edges = value["sparse_lasso_edges"]
    arms = value["pair_features_by_arm"]
    if kind == "coefficient":
        edges["coefficient"] = edges["coefficient"] * 1.37
    elif kind == "stability":
        # Edge 0 is the only stable evidence that reaches pathway "a", so dropping
        # its support must change the recorded pair features.
        edges["stability"] = torch.as_tensor([0.2, 0.6, 0.8], dtype=torch.float32)
    elif kind == "node_order":
        genes = value["node_global_indices"]["gene"].clone()
        value["node_global_indices"]["gene"] = genes.flip(0)
    elif kind == "pair_feature":
        frame = arms["with_lasso"].clone()
        frame[0, int(np.flatnonzero(np.asarray(LASSO_DERIVED_PAIR_FEATURE_MASK))[0])] += 0.25
        arms["with_lasso"] = frame
    elif kind == "without_lasso":
        frame = arms["without_lasso"].clone()
        frame[1, int(np.flatnonzero(np.asarray(LASSO_DERIVED_PAIR_FEATURE_MASK))[1])] = 0.5
        arms["without_lasso"] = frame
    elif kind == "response_id":
        summary = value["lasso_response_summary"]
        summary["response_id"] = [summary["response_id"][1], summary["response_id"][0]]
    elif kind == "response_count":
        summary = value["lasso_response_summary"]
        summary["available"] = [True]
    elif kind == "edge_dropped":
        edges["lnc_index"] = edges["lnc_index"][:2]
        edges["gene_index"] = edges["gene_index"][:2]
        edges["coefficient"] = edges["coefficient"][:2]
        edges["stability"] = edges["stability"][:2]
    elif kind == "threshold":
        value["lasso_stability_threshold"] = 0.6
    elif kind == "unavailable_edge":
        summary = value["lasso_response_summary"]
        summary["available"] = [True, False]
    elif kind == "row_identity":
        value["row_metadata"]["lncrna_id"][0] = "LNC:NOT_IN_UNIVERSE"
    else:
        raise AssertionError(kind)
    return value


def test_recorded_lasso_evidence_rebuilds_model_inputs():
    summary = verify_episode(_payload(), IDENTITIES, _static(), THRESHOLD)
    assert summary["response_count"] == 2
    assert summary["edge_count"] == 3
    assert summary["stable_edge_count"] == 3
    assert summary["row_count"] == 4
    assert summary["pair_feature_columns_rebuilt"] == 3
    assert set(summary["node_counts"]) == {"lncRNA", "gene", "pathway", "pathway_family"}
    assert summary["boundary_stability_edges"] == 0


@pytest.mark.parametrize(
    "kind",
    ["coefficient", "stability", "node_order", "pair_feature", "without_lasso",
     "response_id", "response_count", "edge_dropped", "threshold", "unavailable_edge", "row_identity"],
)
def test_recorded_lasso_derivation_mismatch_is_rejected(kind):
    with pytest.raises(ValueError):
        verify_episode(_fault(_payload(), kind), IDENTITIES, _static(), THRESHOLD)


def test_unavailable_response_rows_do_not_gain_evidence():
    payload = _payload()
    summary = payload["lasso_response_summary"]
    summary["available"] = [True, False]
    summary["unavailable_reason"] = ["", "INSUFFICIENT_SOURCE_DONORS"]
    payload["sparse_lasso_edges"] = {
        key: value[:1] for key, value in payload["sparse_lasso_edges"].items()
    }
    # Dropping the second response's edge changes the localized edge set, so the
    # payload must be rebuilt for a consistent claim; the replay refuses the
    # inconsistent mixture instead of accepting it.
    with pytest.raises(ValueError):
        verify_episode(payload, IDENTITIES, _static(), THRESHOLD)


def test_deterministic_replay_is_repeatable():
    payload = _payload()
    first = verify_episode(payload, IDENTITIES, _static(), THRESHOLD)
    second = verify_episode(payload, IDENTITIES, _static(), THRESHOLD)
    assert first == second


def test_format_marker_is_explicit():
    assert FORMAT == "NATIVE_SC_LASSO_MODEL_INPUT_REPLAY_V1"
    assert len(PAIR_FEATURE_NAMES) == 6


def test_verify_episodes_rejects_incomplete_inventory(tmp_path):
    settings = type("S", (), {"lasso_stability_threshold": THRESHOLD})()
    with pytest.raises(ValueError):
        verify_episodes(tmp_path / "SC_HHGT_EPISODE_MANIFEST.json", settings, {"episodes": [], "cancers": []})
    with pytest.raises(ValueError):
        verify_episodes(tmp_path / "SC_HHGT_EPISODE_MANIFEST.json", settings, {"folds": []})
    # A well-formed but malformed record is rejected before any payload is read.
    with pytest.raises(ValueError):
        verify_episodes(tmp_path / "SC_HHGT_EPISODE_MANIFEST.json", settings, {"episodes": [{}]})
    # A declared record whose file is absent must fail closed, never be skipped.
    with pytest.raises((RuntimeError, FileNotFoundError)):
        verify_episodes(tmp_path / "SC_HHGT_EPISODE_MANIFEST.json", settings,
                        {"cancers": ["BRCA"],
                         "episodes": [{"path": str(tmp_path / "missing.pt"), "bytes": 0}]})
