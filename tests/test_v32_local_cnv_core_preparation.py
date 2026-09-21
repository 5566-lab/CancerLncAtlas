from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from cc_hhgt.v32 import group_shared_encoder_oracle, training
from cc_hhgt.v32.local_cnv_core_preparation import (
    adjusted_association_frame,
    e1_coexpression_frame,
    validate_validation_a1_authority,
)


def _association_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cancer_id": ["BRCA", "BRCA"],
            "lncrna_id": ["L1", "L2"],
            "pathway_id": ["P1", "P2"],
            "fold_id": [2, 2],
            "rho_A1_local": [0.4, np.nan],
            "p_A1_local": [0.001, np.nan],
            "fdr_official_A1_local": [0.01, np.nan],
            "label_A1_local": ["strong_positive", "unlabeled"],
            "direction_A1_local": ["positive", "unavailable"],
            "n_A1_local": [75, 8],
            "availability": ["available", "typed_unavailable"],
            "unavailable_reason": ["", "LOCAL_CNV_INSUFFICIENT_MATCHED_PATIENTS"],
        }
    )


def test_adjusted_association_preserves_typed_unavailable() -> None:
    result = adjusted_association_frame(
        _association_rows(), prefix="A1_local", expected_cancer="BRCA", expected_fold=2
    )
    assert result.association_available.tolist() == [True, False]
    assert result.proxy_label.tolist() == [1, 0]
    assert result.label_class.tolist() == ["strong_positive", "unavailable"]
    assert np.isnan(result.loc[1, "discovery_effect"])
    assert result.loc[1, "association_unavailable_reason"] == (
        "LOCAL_CNV_INSUFFICIENT_MATCHED_PATIENTS"
    )


def test_adjusted_association_rejects_numeric_unavailable() -> None:
    rows = _association_rows()
    rows.loc[1, "rho_A1_local"] = 0.0
    with pytest.raises(RuntimeError, match="numeric effect"):
        adjusted_association_frame(rows, prefix="A1_local")


def test_e1_edges_never_fall_back_to_e0() -> None:
    rows = pd.DataFrame(
        {
            "cancer_id": ["BRCA", "BRCA", "LUAD"],
            "fold_id": [1, 1, 1],
            "lncrna_id": ["L1", "L2", "L3"],
            "gene_id": ["G1", "G2", "G3"],
            "rho_E0": [0.5, -0.6, 0.7],
            "rho_E1": [0.4, np.nan, -0.3],
            "present_E0": [True, True, True],
            "present_E1": [True, False, True],
        }
    )
    result = e1_coexpression_frame(rows, expected_fold=1)
    assert list(result.lncrna_id) == ["L1", "L3"]
    assert result.rho.tolist() == [0.4, -0.3]
    assert result.source_split.eq("train").all()
    assert result.edge_outer_fold.eq(1).all()


def test_validation_a1_authority_requires_firewall_gates(tmp_path) -> None:
    independent = tmp_path / "INDEPENDENT_AUDIT.json"
    independent.write_text('{"status":"PASS"}\n', encoding="utf-8")
    success = tmp_path / "SUCCESS.json"
    success.write_text(
        """{
          "status": "PASS",
          "candidate_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
          "patient_map_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
          "gates": {
            "validation_label_only": true,
            "designated_oof_partition_read": false,
            "sealed_test_accessed": false,
            "mutation_used": false,
            "candidate_closure": true,
            "typed_unavailable_preserved": true
          }
        }\n""",
        encoding="utf-8",
    )
    result = validate_validation_a1_authority(tmp_path)
    assert result["candidate_sha256"] == "a" * 64
    payload = success.read_text(encoding="utf-8").replace(
        '"sealed_test_accessed": false', '"sealed_test_accessed": true'
    )
    success.write_text(payload, encoding="utf-8")
    with pytest.raises(RuntimeError, match="governance contract"):
        validate_validation_a1_authority(tmp_path)


def _loss_inputs(include_mask: bool):
    output = {
        "final_logit": torch.tensor([0.2, 7.0, -0.4], requires_grad=True),
        "direction_logit": torch.tensor([0.1, -9.0, 0.3], requires_grad=True),
        "raw_graph_residual": torch.zeros(3, requires_grad=True),
    }
    batch = {
        "proxy_label": torch.tensor([1.0, 1.0, 0.0]),
        "weak_positive": torch.tensor([False, False, False]),
        "direction_label": torch.tensor([1.0, 0.0, 0.0]),
        "direction_available": torch.tensor([True, True, True]),
    }
    if include_mask:
        batch["association_available"] = torch.tensor([True, False, True])
    return output, batch


def test_association_availability_masks_membership_and_direction_losses() -> None:
    output, batch = _loss_inputs(include_mask=True)
    masked = training._global_loss_from_outputs(
        [output], [batch], torch=torch, direction_loss_weight=0.25, shrinkage=0.0
    )
    subset_output = {key: value[[0, 2]] for key, value in output.items()}
    subset_batch = {key: value[[0, 2]] for key, value in batch.items()}
    subset = training._global_loss_from_outputs(
        [subset_output], [subset_batch], torch=torch,
        direction_loss_weight=0.25, shrinkage=0.0,
    )
    assert torch.allclose(masked, subset)


def test_shared_oracle_and_production_loss_match_with_availability_mask() -> None:
    output, batch = _loss_inputs(include_mask=True)
    production = training._global_loss_from_outputs(
        [output], [batch], torch=torch, direction_loss_weight=0.25, shrinkage=0.01
    )
    oracle = group_shared_encoder_oracle._global_group_loss_components(
        [output], [batch], torch=torch, direction_loss_weight=0.25, shrinkage=0.01
    )
    assert torch.allclose(production, oracle.components["total_objective"])
