from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from cc_hhgt.v32.cancer_modality_router import (
    CancerModalityRouterError,
    CancerRouterConfig,
    build_cancer_router_frame,
    materialize_cancer_modality_router_validation_only,
    materialize_external_router_outer_after_winner_lock,
    train_cancer_modality_router,
    train_cancer_modality_router_validation_only,
    validate_atac_oof_lineage,
    validate_patient_oof_lineage,
)
from cc_hhgt.v32.multimodal_fusion import FUSION_FOLD_COLUMN
from cc_hhgt.v32.multimodal_fusion import artifact_sha256
from cc_hhgt.v32.routing_validation_winner import lock_routing_validation_winner


def _frames():
    rows = []
    for cancer in ("BRCA", "COAD"):
        for fold in range(5):
            for index in range(20):
                positive = index % 2 == 0
                rows.append(
                    {
                        "cancer_id": cancer,
                        "lncrna_id": f"L{fold}_{index}",
                        "pathway_id": f"P{index}",
                        "primary_probability": 0.5,
                        "fusion_target": 1.0 if positive else 0.0,
                        FUSION_FOLD_COLUMN: fold,
                    }
                )
    primary = pd.DataFrame(rows)
    genomic = primary[["cancer_id", "lncrna_id", "pathway_id"]].copy()
    genomic["mutation_context_probability"] = np.where(primary.fusion_target.eq(1), 0.9, 0.1)
    genomic["mutation_available"] = True
    genomic["mutation_unavailable_reason"] = pd.NA
    genomic["cnv_context_probability"] = np.nan
    genomic["cnv_available"] = False
    genomic["cnv_unavailable_reason"] = "CNV_NOT_AVAILABLE_FOR_CANCER"
    return primary, genomic


def test_nested_router_splits_experts_and_never_reverse_selects_heldout() -> None:
    primary, genomic = _frames()
    frame = build_cancer_router_frame(primary, genomic)
    assert not frame.atac_available.any()
    assert frame.atac_probability.isna().all()
    result = train_cancer_modality_router(
        frame,
        config=CancerRouterConfig(
            min_train_rows=20,
            min_validation_rows=10,
            min_available_train_rows=10,
            min_available_validation_rows=5,
            validation_logloss_delta=0.0,
            frozen_gate_support_folds=1,
            max_steps=80,
            patience=4,
            evaluation_interval=5,
            batch_size=32,
            max_train_rows=1000,
            max_validation_rows=1000,
        ),
    )
    assert len(result.oof_private) == len(primary)
    assert all(item["heldout_used_for_gate_selection"] is False for item in result.fold_gates)
    assert result.oof_private.cnv_fusion_weight.eq(0).all()
    assert result.oof_private.atac_fusion_weight.eq(0).all()
    assert result.oof_private.mutation_gate_enabled.any()
    assert "fusion_target" not in result.routed_candidate
    assert result.metrics.heldout_used_for_gate_selection.eq(False).all()


def _small_config() -> CancerRouterConfig:
    return CancerRouterConfig(
        min_train_rows=20,
        min_validation_rows=10,
        min_available_train_rows=10,
        min_available_validation_rows=5,
        validation_logloss_delta=0.0,
        frozen_gate_support_folds=1,
        max_steps=40,
        patience=3,
        evaluation_interval=5,
        batch_size=32,
        max_train_rows=1000,
        max_validation_rows=1000,
    )


def test_validation_only_router_never_queries_outer_and_is_outer_label_invariant(
    tmp_path: Path,
) -> None:
    primary, genomic = _frames()
    frame = build_cancer_router_frame(primary, genomic)
    first = train_cancer_modality_router_validation_only(frame, config=_small_config())
    assert len(first.validation_private) == len(primary)
    assert not first.validation_private[["cancer_id", "lncrna_id", "pathway_id"]].duplicated().any()
    assert first.validation_private.outer_test_queried.eq(False).all()
    assert (
        first.validation_private.selection_validation_pair_fold
        == first.validation_private[FUSION_FOLD_COLUMN]
    ).all()

    mutated = frame.copy()
    mask = mutated.cancer_id.eq("BRCA") & mutated[FUSION_FOLD_COLUMN].eq(0)
    mutated.loc[mask, "fusion_target"] = 1.0 - mutated.loc[mask, "fusion_target"]
    second = train_cancer_modality_router_validation_only(mutated, config=_small_config())
    first_gate = next(
        item for item in first.fold_gates
        if item["cancer_id"] == "BRCA" and item["heldout_pair_fold"] == 0
    )
    second_gate = next(
        item for item in second.fold_gates
        if item["cancer_id"] == "BRCA" and item["heldout_pair_fold"] == 0
    )
    assert first_gate == second_gate
    first_fold = first.validation_private.loc[
        first.validation_private.cancer_id.eq("BRCA")
        & first.validation_private.selection_outer_pair_fold.eq(0)
    ].reset_index(drop=True)
    second_fold = second.validation_private.loc[
        second.validation_private.cancer_id.eq("BRCA")
        & second.validation_private.selection_outer_pair_fold.eq(0)
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(first_fold, second_fold, check_dtype=False)

    sources = {}
    for name in ("primary", "genomic", "lineage"):
        path = tmp_path / f"{name}.txt"
        path.write_text(name, encoding="utf-8")
        sources[name] = path
    output = tmp_path / "validation-only"
    success = materialize_cancer_modality_router_validation_only(
        first,
        output_root=output,
        run_id="v32-router-validation-test",
        config=_small_config(),
        source_artifacts=sources,
        training_budget_id="equal-budget-test",
    )
    assert success["outer_test_predictions_written"] is False
    assert success["outer_test_metrics_computed"] is False
    assert success["architecture_winner_selected"] is False
    assert not (output / "router_metrics.parquet").exists()
    assert not (output / "cancer_modality_oof.PRIVATE.parquet").exists()


def test_external_outer_inference_requires_hash_bound_validation_winner(
    tmp_path: Path,
) -> None:
    primary, genomic = _frames()
    frame = build_cancer_router_frame(primary, genomic)
    config = _small_config()
    validation = train_cancer_modality_router_validation_only(frame, config=config)
    source = tmp_path / "source.txt"
    source.write_text("source", encoding="utf-8")
    external_root = tmp_path / "external-validation"
    materialize_cancer_modality_router_validation_only(
        validation,
        output_root=external_root,
        run_id="v32-router-validation-outer-test",
        config=config,
        source_artifacts={"source": source},
        training_budget_id="equal-budget-test",
    )
    external_predictions = external_root / "router_validation_predictions.PRIVATE.parquet"
    external_contract = external_root / "VALIDATION_COMPARISON_CONTRACT.json"
    external_success = external_root / "SUCCESS.json"
    hierarchical_root = tmp_path / "hierarchical-validation"
    hierarchical_root.mkdir()
    hierarchical_predictions = hierarchical_root / "hierarchical_validation_predictions.PRIVATE.parquet"
    hierarchical = pd.read_parquet(external_predictions).drop(
        columns="discovery_adjusted_probability"
    )
    hierarchical["hierarchical_probability"] = hierarchical.primary_probability
    hierarchical.to_parquet(hierarchical_predictions, index=False)
    hierarchical_contract = hierarchical_root / "VALIDATION_COMPARISON_CONTRACT.json"
    hierarchical_contract.write_bytes(external_contract.read_bytes())
    hierarchical_success = hierarchical_root / "SUCCESS.json"
    hierarchical_success.write_text(
        json.dumps(
            {
                "status": "PASS_VALIDATION_ONLY_HIERARCHICAL",
                "validation_predictions": {
                    "sha256": artifact_sha256(hierarchical_predictions)
                },
                "validation_comparison_contract": {
                    "sha256": artifact_sha256(hierarchical_contract)
                },
                "outer_test_predictions_written": False,
                "outer_test_metrics_computed": False,
                "architecture_winner_selected": False,
            }
        ),
        encoding="utf-8",
    )
    winner_root = tmp_path / "winner"
    lock_routing_validation_winner(
        external_predictions_path=external_predictions,
        hierarchical_predictions_path=hierarchical_predictions,
        external_contract_path=external_contract,
        hierarchical_contract_path=hierarchical_contract,
        external_success_path=external_success,
        hierarchical_success_path=hierarchical_success,
        output_root=winner_root,
        run_id="v32-routing-validation-outer-test",
        minimum_logloss_improvement=0.0,
    )
    winner = winner_root / "ROUTING_VALIDATION_WINNER_LOCK.json"
    assert json.loads(winner.read_text(encoding="utf-8"))["winner_id"] == "external_router"
    output = tmp_path / "outer"
    success = materialize_external_router_outer_after_winner_lock(
        frame,
        validation_gates_path=external_root / "VALIDATION_GATES.json",
        external_validation_predictions_path=external_predictions,
        external_validation_success_path=external_success,
        winner_declaration_path=winner,
        winner_declaration_sha256=artifact_sha256(winner),
        output_root=output,
        run_id="v32-external-outer-test",
    )
    assert success["status"] == "PASS_EXTERNAL_ROUTER_OUTER_AFTER_WINNER_LOCK"
    predictions = pd.read_parquet(
        output / "external_router_winner_locked_oof.PRIVATE.parquet"
    )
    assert len(predictions) == len(frame)
    assert predictions.outer_inference_after_winner_lock.eq(True).all()
    assert (output / "external_router_winner_locked_metrics.parquet").is_file()
    ablation = pd.read_parquet(
        output / "external_router_modality_ablation_metrics.parquet"
    )
    assert set(ablation.modality) == {"mutation", "cnv", "atac"}
    assert ablation.outer_metric_computed_after_winner_lock.eq(True).all()
    assert ablation.used_for_model_or_gate_selection.eq(False).all()
    with pytest.raises(CancerModalityRouterError, match="SHA256 drift"):
        materialize_external_router_outer_after_winner_lock(
            frame,
            validation_gates_path=external_root / "VALIDATION_GATES.json",
            external_validation_predictions_path=external_predictions,
            external_validation_success_path=external_success,
            winner_declaration_path=winner,
            winner_declaration_sha256="f" * 64,
            output_root=tmp_path / "outer-invalid",
            run_id="v32-external-outer-invalid-test",
        )


def _atac_lineage() -> dict:
    return {
        "analysis_version": "CancerLncAtlas_V3.2_ATAC_CANDIDATE",
        "modality": "atac",
        "folds": 5,
        "patient_level_modality_oof": True,
        "patient_fold_oof_predictions_not_fold_averaged": True,
        "outer_test_patients_used_in_any_upstream_fit": 0,
        "old_checkpoint_loaded": False,
        "old_predictions_used_as_features": False,
        "predictions_sha256": "a" * 64,
        "fold_status": [
            {
                "patient_fold": fold,
                "status": "SUCCESS",
                "heldout_patients_used_for_fit": False,
            }
            for fold in range(5)
        ],
    }


def _genomic_lineage() -> dict:
    return {
        "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
        "training_status": "SUCCESS",
        "folds": 5,
        "old_checkpoint_loaded": False,
        "old_predictions_used_as_features": False,
        "private_head_trained_from_scratch": True,
        "core_parameters_frozen": True,
        "old_rankings_used_as_outputs": False,
        "prediction_sha256": "b" * 64,
        "modalities": {
            modality: [
                {"patient_fold": fold, "status": "SUCCESS"} for fold in range(5)
            ]
            for modality in ("mutation", "cnv")
        },
    }


def test_genomic_lineage_requires_fresh_hash_bound_successful_patient_oof() -> None:
    validate_patient_oof_lineage(_genomic_lineage(), predictions_sha256="b" * 64)
    invalid = _genomic_lineage()
    invalid["modalities"]["cnv"][2]["status"] = "FAILED"
    with pytest.raises(CancerModalityRouterError, match="not successful"):
        validate_patient_oof_lineage(invalid, predictions_sha256="b" * 64)
    with pytest.raises(CancerModalityRouterError, match="SHA256"):
        validate_patient_oof_lineage(_genomic_lineage(), predictions_sha256="c" * 64)


def test_atac_lineage_requires_fresh_hash_bound_patient_oof() -> None:
    validate_atac_oof_lineage(_atac_lineage(), predictions_sha256="a" * 64)
    invalid = _atac_lineage()
    invalid["old_checkpoint_loaded"] = True
    with pytest.raises(CancerModalityRouterError, match="Old ATAC"):
        validate_atac_oof_lineage(invalid, predictions_sha256="a" * 64)
    with pytest.raises(CancerModalityRouterError, match="SHA256"):
        validate_atac_oof_lineage(_atac_lineage(), predictions_sha256="b" * 64)


def test_atac_availability_and_unavailable_reason_are_typed() -> None:
    primary, genomic = _frames()
    atac = primary[["cancer_id", "lncrna_id", "pathway_id"]].copy()
    atac["atac_context_probability"] = np.nan
    atac["atac_available"] = False
    atac["atac_unavailable_reason"] = "ATAC_NOT_AVAILABLE_FOR_CANCER"
    frame = build_cancer_router_frame(primary, genomic, atac)
    assert not frame.atac_available.any()
    invalid = atac.copy()
    invalid["atac_available"] = invalid["atac_available"].astype("object")
    invalid.loc[invalid.index[0], "atac_available"] = np.nan
    with pytest.raises(CancerModalityRouterError, match="availability contains null"):
        build_cancer_router_frame(primary, genomic, invalid)
    invalid = atac.copy()
    invalid.loc[invalid.index[0], "atac_unavailable_reason"] = pd.NA
    with pytest.raises(CancerModalityRouterError, match="lack an explicit reason"):
        build_cancer_router_frame(primary, genomic, invalid)
