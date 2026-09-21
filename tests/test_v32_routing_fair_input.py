from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cc_hhgt.v32.routing_fair_input import (
    BUDGET_FORMAT,
    FREEZE_FORMAT,
    RoutingFairInputError,
    artifact_sha256,
    pair_blocked_fold,
    stage_routing_fair_inputs,
    validate_staged_routing_fair_inputs,
)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rows(cancers: tuple[str, ...], rows_per_cancer: int, pair_seed: int) -> pd.DataFrame:
    rows = []
    for cancer in cancers:
        candidates = []
        index = 0
        while len(candidates) < rows_per_cancer:
            lnc = f"L{index:04d}"
            pathway = f"P{(index * 7 + 3) % 97:04d}"
            candidates.append(
                {
                    "cancer_id": cancer,
                    "lncrna_id": lnc,
                    "pathway_id": pathway,
                    "fusion_pair_fold": pair_blocked_fold(lnc, pathway, seed=pair_seed),
                }
            )
            index += 1
        if set(item["fusion_pair_fold"] for item in candidates) != set(range(5)):
            raise AssertionError("Synthetic scope did not cover five folds")
        rows.extend(candidates)
    return pd.DataFrame(rows)


def _fixture(tmp_path: Path) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    cancers = ("BRCA", "COAD")
    rows_per_cancer = 40
    patient_seed = 17
    pair_seed = 29
    optimizer_seed = 31
    base = _rows(cancers, rows_per_cancer, pair_seed)
    candidates = tmp_path / "candidates.parquet"
    base[list(("cancer_id", "lncrna_id", "pathway_id"))].to_parquet(candidates, index=False)
    primary = base.copy()
    primary["primary_probability"] = np.where(primary.index % 2 == 0, 0.55, 0.45)
    primary["fusion_target"] = (primary.index % 2 == 0).astype(float)
    primary_path = tmp_path / "primary.parquet"
    primary.to_parquet(primary_path, index=False)
    genomic = base[["cancer_id", "lncrna_id", "pathway_id"]].copy()
    genomic["mutation_context_probability"] = np.where(primary.fusion_target.eq(1), 0.8, 0.2)
    genomic["mutation_available"] = True
    genomic["mutation_unavailable_reason"] = pd.Series(pd.NA, index=genomic.index, dtype="string")
    cnv_available = genomic.index % 3 != 0
    genomic["cnv_context_probability"] = np.where(cnv_available, 0.6, np.nan)
    genomic["cnv_available"] = cnv_available
    genomic["cnv_unavailable_reason"] = pd.Series(
        np.where(cnv_available, None, "CNV_ENTITY_NOT_CALLABLE"), dtype="string"
    )
    genomic_path = tmp_path / "genomic.parquet"
    genomic.to_parquet(genomic_path, index=False)
    atac = base[["cancer_id", "lncrna_id", "pathway_id"]].copy()
    atac_available = atac.index % 4 != 0
    atac["atac_context_probability"] = np.where(atac_available, 0.52, np.nan)
    atac["atac_available"] = atac_available
    atac["atac_unavailable_reason"] = pd.Series(
        np.where(atac_available, None, "ATAC_LNCRNA_NOT_PROMOTER_MAPPED"), dtype="string"
    )
    atac_path = tmp_path / "atac.parquet"
    atac.to_parquet(atac_path, index=False)
    patient_rows = []
    for cancer in cancers:
        for fold in range(5):
            patient_rows.append(
                {
                    "cancer_id": cancer,
                    "sample_id": f"TCGA-{cancer[:2]}-{fold:04d}-01A",
                    "patient_id": f"TCGA-{cancer[:2]}-{fold:04d}",
                    "patient_fold_id": fold,
                    "fold_seed": patient_seed,
                }
            )
    patient_path = tmp_path / "PATIENT_FOLD_MANIFEST.tsv"
    pd.DataFrame(patient_rows).to_csv(patient_path, sep="\t", index=False)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    for fold in range(5):
        (prepared / f"PATIENT_FOLD_{fold}.pt").write_bytes(f"fold={fold}\n".encode())

    genomic_lineage_path = tmp_path / "GENOMIC_LINEAGE.json"
    genomic_lineage = {
        "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
        "training_status": "SUCCESS",
        "folds": 5,
        "old_checkpoint_loaded": False,
        "old_predictions_used_as_features": False,
        "old_rankings_used_as_outputs": False,
        "private_head_trained_from_scratch": True,
        "core_parameters_frozen": True,
        "patient_fold_oof_predictions_not_fold_averaged": True,
        "prediction_sha256": artifact_sha256(genomic_path),
        "modalities": {
            modality: [
                {"patient_fold": fold, "status": "SUCCESS"} for fold in range(5)
            ]
            for modality in ("mutation", "cnv")
        },
    }
    _write_json(genomic_lineage_path, genomic_lineage)
    genomic_success_path = tmp_path / "GENOMIC_SUCCESS.json"
    _write_json(
        genomic_success_path,
        {
            "status": "SUCCESS",
            "prediction_sha256": artifact_sha256(genomic_path),
            "lineage_sha256": artifact_sha256(genomic_lineage_path),
            "candidate_rows_preserved": len(base),
            "null_is_never_zero_or_wildtype": True,
        },
    )
    atac_lineage_path = tmp_path / "ATAC_LINEAGE.json"
    atac_lineage = {
        "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
        "modality": "atac",
        "folds": 5,
        "patient_level_modality_oof": True,
        "patient_fold_oof_predictions_not_fold_averaged": True,
        "outer_test_patients_used_in_any_upstream_fit": 0,
        "old_checkpoint_loaded": False,
        "old_predictions_used_as_features": False,
        "old_rankings_used_as_outputs": False,
        "predictions_sha256": artifact_sha256(atac_path),
        "fold_status": [
            {
                "patient_fold": fold,
                "status": "SUCCESS",
                "heldout_patients_used_for_fit": False,
            }
            for fold in range(5)
        ],
    }
    _write_json(atac_lineage_path, atac_lineage)
    atac_training_success_path = tmp_path / "ATAC_SUCCESS.json"
    _write_json(
        atac_training_success_path,
        {
            "status": "SUCCESS",
            "success_written_last": True,
            "prediction_sha256": artifact_sha256(atac_path),
            "lineage_sha256": artifact_sha256(atac_lineage_path),
            "prediction_rows": len(base),
        },
    )
    atac_audit_path = tmp_path / "ATAC_AUDIT.json"
    _write_json(
        atac_audit_path,
        {
            "status": "PASS",
            "candidate_authority_sha256": artifact_sha256(candidates),
            "prediction_sha256": artifact_sha256(atac_path),
            "prediction_rows": len(base),
            "exact_candidate_keys_match_authority": True,
            "typed_null_never_zero": True,
            "old_predictions_checkpoints_or_rankings_used": False,
            "formal_positive_contribution_claimed": False,
        },
    )
    atac_audit_success_path = tmp_path / "ATAC_AUDIT_SUCCESS.json"
    _write_json(
        atac_audit_success_path,
        {
            "status": "PASS",
            "audit_sha256": artifact_sha256(atac_audit_path),
            "prediction_sha256": artifact_sha256(atac_path),
        },
    )
    budget_path = tmp_path / "BUDGET.json"
    budget = {
        "format": BUDGET_FORMAT,
        "budget_id": "synthetic-fair-budget",
        "scope": {
            "candidate_rows_per_cancer": rows_per_cancer,
            "formal_cancers": list(cancers),
            "modalities": ["mutation", "cnv", "atac"],
        },
        "seeds": {
            "patient_fold_seed": patient_seed,
            "pair_fold_seed": pair_seed,
            "optimizer_seed": optimizer_seed,
        },
        "selection_policy": {
            "outer_folds": 5,
            "validation_offset": 1,
            "training_folds_per_outer": 3,
            "validation_folds_per_outer": 1,
            "selection_scope": "INNER_VALIDATION_FOLD_ONLY",
            "outer_test_queries_during_selection": 0,
            "outer_test_evaluations_per_arm_fold": 1,
            "test_policy": "ONE_FINAL_EVALUATION_AFTER_ARM_SELECTION_IS_FROZEN",
            "winner_selection_queries": 0,
            "promotion_or_winner_selection_enabled": False,
        },
        "architecture_optimizer_ceilings": {
            "external_router": {
                "max_candidate_coverage_cycles": 10,
                "max_validation_checkpoints": 10,
                "patience_validation_checkpoints": 3,
            },
            "hierarchical_end_to_end": {
                "max_candidate_coverage_cycles": 10,
                "max_validation_checkpoints": 10,
                "patience_validation_checkpoints": 3,
            },
        },
        "shared_optimization_budget": {
            "max_candidate_coverage_cycles": 10,
            "max_validation_checkpoints": 10,
            "patience_validation_checkpoints": 3,
        },
    }
    _write_json(budget_path, budget)
    freeze_path = tmp_path / "FREEZE.json"
    _write_json(
        freeze_path,
        {
            "format": FREEZE_FORMAT,
            "candidate_authority_sha256": artifact_sha256(candidates),
            "prediction_sha256": artifact_sha256(atac_path),
            "lineage_sha256": artifact_sha256(atac_lineage_path),
            "training_success_sha256": artifact_sha256(atac_training_success_path),
            "independent_audit_sha256": artifact_sha256(atac_audit_path),
            "independent_audit_success_sha256": artifact_sha256(atac_audit_success_path),
            "prediction_rows": len(base),
        },
    )
    return {
        "candidate": candidates,
        "patient": patient_path,
        "prepared": prepared,
        "primary": primary_path,
        "genomic": genomic_path,
        "genomic_lineage": genomic_lineage_path,
        "genomic_success": genomic_success_path,
        "atac": atac_path,
        "atac_lineage": atac_lineage_path,
        "atac_success": atac_training_success_path,
        "atac_audit": atac_audit_path,
        "atac_audit_success": atac_audit_success_path,
        "budget": budget_path,
        "freeze": freeze_path,
    }


def _stage(paths: dict[str, Path], output: Path) -> dict:
    return stage_routing_fair_inputs(
        candidate_authority_path=paths["candidate"],
        patient_fold_manifest_path=paths["patient"],
        formal_prepared_root=paths["prepared"],
        primary_oof_path=paths["primary"],
        genomic_predictions_path=paths["genomic"],
        genomic_lineage_path=paths["genomic_lineage"],
        genomic_success_path=paths["genomic_success"],
        atac_predictions_path=paths["atac"],
        atac_lineage_path=paths["atac_lineage"],
        atac_training_success_path=paths["atac_success"],
        atac_audit_path=paths["atac_audit"],
        atac_audit_success_path=paths["atac_audit_success"],
        budget_contract_path=paths["budget"],
        atac_freeze_contract_path=paths["freeze"],
        output_root=output,
    )


def test_stages_and_independently_validates_identical_arm_inputs(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    staged = tmp_path / "staged"
    success = _stage(paths, staged)
    assert success["status"] == "READY_FOR_ARM_TRAINING"
    assert success["winner_selection_run"] is False
    assert success["atac_r3_predictions_modified"] is False
    contract = json.loads((staged / "COMPARISON_INPUT_CONTRACT.json").read_text())
    assert contract["candidate_rows"] == 80
    assert contract["seeds"] == {
        "patient_fold_seed": 17,
        "pair_fold_seed": 29,
        "optimizer_seed": 31,
    }
    external = json.loads((staged / "EXTERNAL_ROUTER_INPUT_CONTRACT.json").read_text())
    hierarchical = json.loads(
        (staged / "HIERARCHICAL_END_TO_END_INPUT_CONTRACT.json").read_text()
    )
    for key in (
        "shared_fair_input_sha256",
        "ordered_exact_candidate_key_sha256",
        "outer_pair_fold_sha256",
        "patient_fold_manifest_sha256",
        "training_budget_sha256",
        "split_policy_sha256",
        "seeds",
        "selection_scope",
        "outer_test_policy",
    ):
        assert external[key] == hierarchical[key]
    validated = validate_staged_routing_fair_inputs(
        staging_success_path=staged / "SUCCESS.json",
        output_root=tmp_path / "validation",
    )
    assert validated["status"] == "PASS_INPUTS_ONLY_WINNER_NOT_RUN"
    assert validated["winner_selection_run"] is False
    assert validated["comparison_metrics_computed"] is False


def test_split_policy_uses_validation_only_and_test_exactly_once(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    staged = tmp_path / "staged"
    _stage(paths, staged)
    split = json.loads((staged / "SPLIT_POLICY.json").read_text())
    assert split["winner_selection_enabled"] is False
    for record in split["fold_records"]:
        train = set(record["training_folds"])
        validation = {record["validation_fold"]}
        test = {record["outer_test_fold"]}
        assert not train & validation
        assert not train & test
        assert not validation & test
        assert train | validation | test == set(range(5))
        assert record["selection_may_read_outer_test"] is False
        access = record["consumer_fold_access"]
        for consumer in (
            "scaling_fit",
            "early_stopping",
            "checkpoint_choice",
            "route_threshold_selection",
        ):
            assert record["outer_test_fold"] not in access[consumer]
        assert access["scaling_fit"] == record["training_folds"]
        assert access["early_stopping"] == [record["validation_fold"]]
        assert access["checkpoint_choice"] == [record["validation_fold"]]
        assert access["route_threshold_selection"] == [record["validation_fold"]]
        assert access["final_test_evaluation"] == [record["outer_test_fold"]]
        assert record["outer_test_keys_allowed_consumers"] == ["final_test_evaluation"]
        assert record["outer_test_labels_allowed_consumers"] == ["final_test_evaluation"]
        assert record["outer_test_evaluations_allowed"] == 1
        assert record["outer_test_may_change_checkpoint_gate_or_hyperparameters"] is False


def test_arm_shared_contracts_are_canonical_byte_identical(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    staged = tmp_path / "staged"
    _stage(paths, staged)
    arm_paths = {
        "external_router": staged / "EXTERNAL_ROUTER_INPUT_CONTRACT.json",
        "hierarchical_end_to_end": staged / "HIERARCHICAL_END_TO_END_INPUT_CONTRACT.json",
    }
    payloads = {}
    for name, path in arm_paths.items():
        value = json.loads(path.read_text())
        assert value["architecture_id"] == name
        value.pop("architecture_id")
        value.pop("implementation_sha256")
        payloads[name] = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    assert len(set(payloads.values())) == 1

    hierarchical = json.loads(arm_paths["hierarchical_end_to_end"].read_text())
    hierarchical["unexpected_shared_drift"] = True
    _write_json(arm_paths["hierarchical_end_to_end"], hierarchical)
    success_path = staged / "SUCCESS.json"
    success = json.loads(success_path.read_text())
    success["arm_input_contracts"]["hierarchical_end_to_end"]["sha256"] = artifact_sha256(
        arm_paths["hierarchical_end_to_end"]
    )
    _write_json(success_path, success)
    with pytest.raises(RoutingFairInputError, match="shared contract bytes differ"):
        validate_staged_routing_fair_inputs(
            staging_success_path=success_path,
            output_root=tmp_path / "validation_bad_arm",
        )


def test_fails_closed_until_fresh_cnv_success(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    success = json.loads(paths["genomic_success"].read_text())
    success["status"] = "RUNNING"
    _write_json(paths["genomic_success"], success)
    with pytest.raises(RoutingFairInputError, match="Mutation/CNV fresh training"):
        _stage(paths, tmp_path / "staged")


def test_frozen_atac_r3_hash_drift_is_rejected(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    atac = pd.read_parquet(paths["atac"])
    atac.loc[atac.atac_available, "atac_context_probability"] += 0.01
    atac.to_parquet(paths["atac"], index=False)
    with pytest.raises(RoutingFairInputError, match="Frozen ATAC r3 mismatch"):
        _stage(paths, tmp_path / "staged")


def test_pair_fold_seed_drift_and_output_reuse_are_rejected(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    primary = pd.read_parquet(paths["primary"])
    primary["fusion_pair_fold"] = [
        pair_blocked_fold(row.lncrna_id, row.pathway_id, seed=999)
        for row in primary.itertuples()
    ]
    primary.to_parquet(paths["primary"], index=False)
    with pytest.raises(RoutingFairInputError, match="pair-fold assignment/seed drift"):
        _stage(paths, tmp_path / "staged_bad")

    paths = _fixture(tmp_path / "second")
    output = tmp_path / "staged"
    _stage(paths, output)
    with pytest.raises(RoutingFairInputError, match="refuses output/staging reuse"):
        _stage(paths, output)


def test_server_launcher_contains_no_training_inference_or_winner_command() -> None:
    launcher = (
        Path(__file__).resolve().parents[1]
        / "scripts/server_prepare_v32_routing_fair_inputs_r3.sh"
    ).read_text(encoding="utf-8")
    for forbidden in (
        "compare_v32_routing_architectures.py",
        "run_v32_cancer_modality_router.py",
        "prepare_v32_hierarchical_candidate.py",
        "infer_v32_hierarchical_candidate.py",
        "run-shard",
    ):
        assert forbidden not in launcher
