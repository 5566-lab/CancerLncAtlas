from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cc_hhgt.v32.input_lineage import artifact_sha256
from cc_hhgt.v32.single_cell_context_v2 import (
    CONTEXT_FEATURES,
    DONOR_FOLD_MANIFEST_FORMAT,
    FOLD_LOCAL_FEATURE_MANIFEST_FORMAT,
    PREFLIGHT_FORMAT,
    SingleCellContextV2Error,
    _canonical_sha256,
    _execution_code_binding,
    audit_donor_resolved_association_source,
    assert_leakage_free_feature_schema,
    assert_not_historical_source,
    build_context_domain_feature_frame,
    build_signed_evidence_sidecar,
    build_strict_donor_folds,
    compartment_from_cell_key,
    context_feature_matrix,
    require_donor_blocked_targets,
    run_context_v2_preflight,
    validate_context_donor_folds,
    validate_training_preflight_contract,
)


def test_context_activity_retains_celltype_global_and_delta() -> None:
    associations = pd.DataFrame(
        {
            "dataset_id": ["DS1", "DS1"],
            "cancer_id": ["BRCA", "BRCA"],
            "cell_type_major": ["B cell", "Malignant_candidate"],
            "lncrna_id": ["L1", "L1"],
            "pathway_id": ["P1", "P1"],
            # These labels must have no effect on the feature matrix.
            "association_effect": [-0.9, 0.8],
            "association_fdr": [0.01, 0.02],
        }
    )
    lnc = pd.DataFrame(
        {
            "dataset_id": ["DS1", "DS1"],
            "cancer_id": ["BRCA", "BRCA"],
            "lncrna_id": ["L1", "L1"],
            "cell_key": ["b_cell", "malignant_candidate"],
            "lnc_detection_rate": [0.2, 0.8],
            "lnc_mean_log_expression": [1.0, 2.0],
            "lnc_specificity_tau": [0.3, 0.7],
            "lnc_n_cells": [10, 30],
        }
    )
    activity = pd.DataFrame(
        {
            "dataset_id": ["DS1", "DS1"],
            "cancer_id": ["BRCA", "BRCA"],
            "cell_key": ["b_cell", "malignant_candidate"],
            "pathway_id": ["P1", "P1"],
            "activity_kind": ["activity", "activity"],
            "activity_value": [1.0, 3.0],
            "n_cells": [10, 30],
        }
    )
    features = build_context_domain_feature_frame(associations, lnc, activity)
    assert features.pathway_activity_context.tolist() == [1.0, 3.0]
    np.testing.assert_allclose(features.pathway_activity_global, [2.5, 2.5])
    np.testing.assert_allclose(features.pathway_activity_delta, [-1.5, 0.5])
    assert features.compartment.tolist() == ["immune", "malignant"]
    assert features.compartment_immune.tolist() == [1.0, 0.0]
    assert features.compartment_malignant.tolist() == [0.0, 1.0]
    assert context_feature_matrix(features).shape == (2, len(CONTEXT_FEATURES))

    changed_labels = associations.assign(
        association_effect=[0.01, -0.01], association_fdr=[0.99, 0.99]
    )
    changed = context_feature_matrix(
        build_context_domain_feature_frame(changed_labels, lnc, activity)
    )
    np.testing.assert_array_equal(changed, context_feature_matrix(features))


def test_compartment_encoding_has_four_fail_safe_classes() -> None:
    assert compartment_from_cell_key("Malignant") == "malignant"
    assert compartment_from_cell_key("Malignant_candidate") == "malignant"
    assert compartment_from_cell_key("Myeloid") == "immune"
    assert compartment_from_cell_key("Fibroblast_stromal") == "stromal"
    assert compartment_from_cell_key("Epithelial") == "other"


def test_signed_rho_and_fdr_stay_in_evidence_sidecar() -> None:
    associations = pd.DataFrame(
        {
            "dataset_id": ["DS", "DS", "DS"],
            "cancer_id": ["BRCA"] * 3,
            "donor_id": ["D1", "D2", "D3"],
            "cell_type_major": ["B_cell"] * 3,
            "lncrna_id": ["L1"] * 3,
            "pathway_id": ["P1"] * 3,
            "association_effect": [-0.6, 0.4, 0.0],
            "association_fdr": [0.1, 0.2, 0.3],
        }
    )
    sidecar = build_signed_evidence_sidecar(associations)
    assert sidecar.observed_direction.tolist() == ["negative", "positive", "zero"]
    np.testing.assert_allclose(
        sidecar.observed_signed_rho_x_one_minus_fdr, [-0.54, 0.32, 0.0]
    )
    assert set(CONTEXT_FEATURES).isdisjoint(sidecar.columns)
    with pytest.raises(SingleCellContextV2Error, match="cannot enter"):
        assert_leakage_free_feature_schema([*CONTEXT_FEATURES, "association_fdr"])


def test_strict_donor_folds_do_not_silently_fallback_to_dataset() -> None:
    aggregate = pd.DataFrame(
        {
            "dataset_id": ["DS1"] * 5,
            "cancer_id": ["BRCA"] * 5,
            "donor_id": [""] * 5,
            "cell_type_major": ["B_cell"] * 5,
        }
    )
    audit = require_donor_blocked_targets(aggregate)
    assert audit["ready"] is False
    assert audit["fallback_used"] is False
    with pytest.raises(SingleCellContextV2Error, match="donor-resolved"):
        build_strict_donor_folds(
            aggregate,
            source_semantic_audit={
                "ready": False,
                "reasons": ["SOURCE_SEMANTICS_NOT_PROVEN"],
            },
        )

    donor_rows = aggregate.assign(donor_id=[f"D{i}" for i in range(5)])
    folds = build_strict_donor_folds(
        donor_rows, source_semantic_audit={"ready": True, "reasons": []}, seed=7
    )
    assert folds.groupby("donor_id").single_cell_fold_id.nunique().max() == 1
    assert not folds.block_id.str.contains("dataset:", regex=False).any()


def _core_manifest(path: Path) -> None:
    folds = {}
    for fold in range(5):
        folds[str(fold)] = {
            "patient_fold": fold,
            "checkpoint_sha256": hashlib.sha256(
                f"checkpoint-{fold}".encode()
            ).hexdigest(),
            "core_parameter_sha256": hashlib.sha256(
                f"parameter-{fold}".encode()
            ).hexdigest(),
            "old_checkpoint_loaded": False,
            "trained_from_scratch": True,
            "exports": {},
        }
    path.write_text(
        json.dumps(
            {
                "export_format": "CC_HHGT_V3_2_FROZEN_CORE_EMBEDDINGS_V1",
                "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
                "all_embeddings_from_newly_trained_v32_core": True,
                "historical_checkpoint_loaded": False,
                "historical_prediction_loaded": False,
                "folds": folds,
            }
        ),
        encoding="utf-8",
    )


def test_preflight_recomputes_contexts_and_preserves_v1_binding(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.parquet"
    manifest = tmp_path / "dataset_manifest.parquet"
    association = tmp_path / "fresh_sc_association.parquet"
    lnc = tmp_path / "fresh_lnc_celltype.parquet"
    activity = tmp_path / "fresh_activity.parquet"
    core = tmp_path / "fresh_core_manifest.json"
    binding = tmp_path / "SINGLE_CELL_FORMAL_CONTEXT_BINDING.json"
    pd.DataFrame(
        {
            "cancer_id": ["BRCA", "BRCA"],
            "lncrna_id": ["L1", "L1"],
            "pathway_id": ["P1", "P1"],
        }
    ).drop_duplicates().to_parquet(candidates, index=False)
    pd.DataFrame(
        {
            "dataset_id": ["DS1"],
            "cancer_id": ["BRCA"],
            "formal_eligible": [True],
            "source_tier": ["primary_raw"],
            "quality_status": ["PASS"],
            "donor_metadata_available": [True],
        }
    ).to_parquet(manifest, index=False)
    pd.DataFrame(
        {
            "dataset_id": ["DS1", "DS1"],
            "cancer_id": ["BRCA", "BRCA"],
            "cell_population": ["B cell", "Malignant_candidate"],
            "analysis_context": ["overall", "overall"],
            "lncrna_id": ["L1", "L1"],
            "pathway_id": ["P1", "P1"],
            "rho": [-0.4, 0.7],
            "fdr": [0.1, 0.02],
            "source_tier": ["primary_raw", "primary_raw"],
            "n_patients": [8, 8],
        }
    ).to_parquet(association, index=False)
    pd.DataFrame(
        {
            "dataset_id": ["DS1", "DS1"],
            "cancer_id": ["BRCA", "BRCA"],
            "lncrna_id": ["L1", "L1"],
            "cell_type_major": ["B cell", "Malignant_candidate"],
            "detection_rate": [0.2, 0.8],
            "mean_log_expression": [1.0, 2.0],
            "specificity_tau": [0.3, 0.7],
            "n_cells": [10, 20],
        }
    ).to_parquet(lnc, index=False)
    pd.DataFrame(
        {
            "dataset_id": ["DS1", "DS1", "DS1"],
            "cancer_id": ["BRCA", "BRCA", "BRCA"],
            "donor_id": ["D1", "D1", "D1"],
            "cell_type": ["B cell", "Malignant_candidate", "Endothelial"],
            "pathway_id": ["P1", "P1", "P1"],
            "activity": [1.0, 3.0, 2.0],
            "n_cells": [10, 20, 5],
            "source_tier": ["primary_raw"] * 3,
        }
    ).to_parquet(activity, index=False)
    _core_manifest(core)
    binding.write_text('{"format":"FORMAL_V1_PROTECTED"}\n', encoding="utf-8")
    binding_hash = artifact_sha256(binding)

    result = run_context_v2_preflight(
        candidates_path=candidates,
        dataset_manifest_path=manifest,
        association_path=association,
        lnc_celltype_path=lnc,
        activity_path=activity,
        core_embedding_manifest_path=core,
        formal_v1_binding_path=binding,
        output_root=tmp_path / "context_v2_preflight",
    )
    assert result["context_counts"] == {
        "association_distinct_contexts": 2,
        "activity_available_distinct_contexts": 3,
        "matched_association_contexts": 2,
        "missing_association_contexts": 0,
    }
    assert result["context_gate_pass"] is True
    assert result["full_training_ready"] is False
    assert result["long_training_started"] is False
    assert result["historical_tables_read"] is False
    assert artifact_sha256(binding) == binding_hash
    lineage = json.loads(
        (tmp_path / "context_v2_preflight" / "LINEAGE_CONTRACT.json").read_text()
    )
    assert lineage["formal_v1_binding"]["unchanged"] is True
    assert lineage["signed_evidence_is_feature"] is False
    assert lineage["abs_only_claimed_as_direction"] is False


def test_known_v30_derived_tables_are_rejected_before_use() -> None:
    with pytest.raises(SingleCellContextV2Error, match="Historical V3.0"):
        assert_not_historical_source(Path("audit") / "v3_0" / "ucell_aggregate.parquet")


def test_aggregate_spearman_cannot_be_unlocked_by_injected_donor_ids() -> None:
    raw = pd.DataFrame(
        {
            "dataset_id": ["DS1"] * 5,
            "cancer_id": ["BRCA"] * 5,
            "donor_id": [f"D{i}" for i in range(5)],
            "cell_population": ["B_cell"] * 5,
            "lncrna_id": ["L1"] * 5,
            "pathway_id": ["P1"] * 5,
            "rho": [0.6] * 5,
            "fdr": [0.01] * 5,
            "association_method": [
                "V3.2_FRESH_DONOR_PSEUDOBULK_SPEARMAN_EXACT_PATHWAY_V1"
            ] * 5,
            # Even forged positive attestations cannot override the aggregate
            # method or the broadcast target signature.
            "association_observation_unit": ["donor"] * 5,
            "association_target_level": [
                "donor_x_celltype_x_lncrna_x_exact_pathway"
            ] * 5,
            "donor_resolved_target": [True] * 5,
            "donor_target_row_id": [f"ROW{i}" for i in range(5)],
            "generation": ["V3.2_FRESH"] * 5,
        }
    )
    audit = audit_donor_resolved_association_source(raw)
    assert audit["ready"] is False
    assert audit["aggregate_method_rows"] == 5
    assert "CROSS_DONOR_AGGREGATE_METHOD_FORBIDDEN" in audit["reasons"]


def test_held_out_donor_crossing_folds_and_context_gap_are_blocked() -> None:
    rows = []
    for fold in range(5):
        rows.append(
            {
                "dataset_id": "DS1",
                "cancer_id": "BRCA",
                "donor_id": f"D{fold}",
                "cell_key": "b_cell",
                "block_id": f"DS1|donor:D{fold}",
                "single_cell_fold_id": fold,
            }
        )
    # A second context lacks fold 4, and D0 is then maliciously copied to a
    # different held-out fold.
    for fold in range(4):
        rows.append(
            {
                "dataset_id": "DS1",
                "cancer_id": "BRCA",
                "donor_id": f"D{fold}",
                "cell_key": "myeloid",
                "block_id": f"DS1|donor:D{fold}",
                "single_cell_fold_id": fold,
            }
        )
    rows.append(
        {
            "dataset_id": "DS1",
            "cancer_id": "BRCA",
            "donor_id": "D0",
            "cell_key": "b_cell",
            "block_id": "DS1|donor:D0",
            "single_cell_fold_id": 1,
        }
    )
    audit = validate_context_donor_folds(pd.DataFrame(rows))
    assert audit["ready"] is False
    assert audit["donor_blocks_crossing_folds"] == 1
    assert audit["contexts_failing"] >= 1


def test_training_contract_detects_input_sha_drift(tmp_path: Path) -> None:
    artifact_ids = (
        "v32_candidates",
        "single_cell_dataset_manifest",
        "single_cell_association_target",
        "single_cell_lnc_celltype",
        "single_cell_activity",
        "fresh_v32_core_manifest",
        "protected_formal_v1_binding",
    )
    supplied: dict[str, Path] = {}
    bindings: dict[str, dict[str, str]] = {}
    for artifact_id in artifact_ids:
        path = tmp_path / f"{artifact_id}.txt"
        path.write_text(f"bound:{artifact_id}\n", encoding="utf-8")
        supplied[artifact_id] = path
        bindings[artifact_id] = {
            "path": str(path.resolve()),
            "sha256": artifact_sha256(path),
        }
    input_binding_sha = _canonical_sha256(bindings)

    assignments = pd.DataFrame(
        {
            "dataset_id": ["DS1"] * 5,
            "cancer_id": ["BRCA"] * 5,
            "donor_id": [f"D{i}" for i in range(5)],
            "cell_key": ["b_cell"] * 5,
            "block_id": [f"DS1|donor:D{i}" for i in range(5)],
            "single_cell_fold_id": list(range(5)),
        }
    )
    assignment_path = tmp_path / "fold_assignments.parquet"
    assignments.to_parquet(assignment_path, index=False)
    fold_manifest_path = tmp_path / "DONOR_FOLD_MANIFEST.json"
    fold_manifest_path.write_text(
        json.dumps(
            {
                "format": DONOR_FOLD_MANIFEST_FORMAT,
                "status": "READY",
                "ready": True,
                "dataset_fallback_used": False,
                "association_input_sha256": bindings[
                    "single_cell_association_target"
                ]["sha256"],
                "assignment_path": str(assignment_path.resolve()),
                "assignment_sha256": artifact_sha256(assignment_path),
            }
        ),
        encoding="utf-8",
    )
    fold_manifest_sha = artifact_sha256(fold_manifest_path)

    feature_records = {}
    for fold in range(5):
        feature_path = tmp_path / f"features_fold_{fold}.txt"
        feature_path.write_text(f"fold={fold}\n", encoding="utf-8")
        feature_records[str(fold)] = {
            "path": str(feature_path.resolve()),
            "sha256": artifact_sha256(feature_path),
            "training_donors_only": True,
            "held_out_donor_rows_used": 0,
        }
    feature_manifest_path = tmp_path / "FOLD_LOCAL_FEATURE_MANIFEST.json"
    feature_manifest_path.write_text(
        json.dumps(
            {
                "format": FOLD_LOCAL_FEATURE_MANIFEST_FORMAT,
                "status": "READY",
                "ready": True,
                "training_donors_only_materialization": True,
                "held_out_donor_rows_used": 0,
                "donor_fold_manifest_sha256": fold_manifest_sha,
                "input_binding_sha256": input_binding_sha,
                "folds": feature_records,
            }
        ),
        encoding="utf-8",
    )
    feature_manifest_sha = artifact_sha256(feature_manifest_path)

    execution_code, execution_code_sha = _execution_code_binding()
    lineage_path = tmp_path / "LINEAGE_CONTRACT.json"
    lineage_path.write_text(
        json.dumps(
            {
                "full_training_ready": True,
                "execution_code": execution_code,
                "execution_code_sha256": execution_code_sha,
                "input_artifact_bindings": bindings,
                "input_binding_sha256": input_binding_sha,
            }
        ),
        encoding="utf-8",
    )
    preflight_path = tmp_path / "PREFLIGHT.json"
    preflight_path.write_text(
        json.dumps(
            {
                "format": PREFLIGHT_FORMAT,
                "full_training_ready": True,
                "execution_code_sha256": execution_code_sha,
                "input_binding_sha256": input_binding_sha,
                "lineage_contract_path": str(lineage_path.resolve()),
                "lineage_contract_sha256": artifact_sha256(lineage_path),
                "donor_fold_manifest_path": str(fold_manifest_path.resolve()),
                "donor_fold_manifest_sha256": fold_manifest_sha,
                "fold_local_feature_manifest_path": str(
                    feature_manifest_path.resolve()
                ),
                "fold_local_feature_manifest_sha256": feature_manifest_sha,
            }
        ),
        encoding="utf-8",
    )
    preflight_sha = artifact_sha256(preflight_path)
    result = validate_training_preflight_contract(
        preflight_path=preflight_path,
        expected_preflight_sha256=preflight_sha,
        expected_execution_code_sha256=execution_code_sha,
        donor_fold_manifest_path=fold_manifest_path,
        expected_donor_fold_manifest_sha256=fold_manifest_sha,
        fold_local_feature_manifest_path=feature_manifest_path,
        expected_fold_local_feature_manifest_sha256=feature_manifest_sha,
        supplied_input_paths=supplied,
    )
    assert result["status"] == "PASS"

    supplied["single_cell_activity"].write_text("tampered\n", encoding="utf-8")
    with pytest.raises(SingleCellContextV2Error, match="input SHA256 drift"):
        validate_training_preflight_contract(
            preflight_path=preflight_path,
            expected_preflight_sha256=preflight_sha,
            expected_execution_code_sha256=execution_code_sha,
            donor_fold_manifest_path=fold_manifest_path,
            expected_donor_fold_manifest_sha256=fold_manifest_sha,
            fold_local_feature_manifest_path=feature_manifest_path,
            expected_fold_local_feature_manifest_sha256=feature_manifest_sha,
            supplied_input_paths=supplied,
        )
