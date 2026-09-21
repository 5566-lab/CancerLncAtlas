from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import pandas as pd
import pytest

from cc_hhgt.v32.full_model_contract import (
    DRUG_CROSS_DATASET_ASSOCIATION_POLICY,
    DRUG_EXPRESSION_MOMENT_POLICY,
    FULL_CONTRACT_VERSION,
    MODULE_CONTRACTS,
    FullModelContractError,
    validate_full_release_lineage,
    validate_module_lineage,
    validate_public_module_frame,
)


CORE_HASH = "c" * 64
DRUG_SPARSE_ARTIFACT_NAMES = (
    "exact_candidates",
    "pathway_drug_edges",
    "assay_fold_eligibility",
    "expression_fold_coverage",
    "core_entity_availability",
    "available_predictions",
)
EXACT_BINDING_KEYS = (
    "cancer_id",
    "lncrna_id",
    "pathway_id",
    "pathway_family_id",
)
DRUG_STAGING_REQUIRED_FIELDS = (
    "staging_revalidation_audit_path",
    "staging_revalidation_audit_sha256",
    "staging_revalidation_audit_status",
    "staging_manifest_path",
    "staging_manifest_sha256",
    "staging_provenance_snapshot_path",
    "staging_provenance_snapshot_sha256",
    "staging_input_artifacts_rehashed",
    "staging_execution_code_rehashed",
    "staging_runtime_fingerprint_sha256",
    "exact_release_lineage_contract_validated",
    "exact_candidate_four_key_binding_recomputed",
    "exact_release_prediction_sha256",
    "exact_release_lineage_sha256",
    "expected_staging_manifest_sha256",
    "expected_exact_release_prediction_sha256",
    "expected_exact_release_lineage_sha256",
    "exact_candidate_source_binding",
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), default=str
        ).encode()
    ).hexdigest()


def _artifact(kind: str = "raw_data", sha256: str = "a" * 64) -> dict[str, str]:
    return {"path": "canonical/input.parquet", "sha256": sha256, "artifact_kind": kind}


def _single_cell_target_artifact() -> dict[str, object]:
    return {
        "path": "single_cell/fresh_v32_association.parquet",
        "sha256": "b" * 64,
        "artifact_kind": "training_label",
        "generation": "V3.2_FRESH_FROM_RAW_SINGLE_CELL_INPUT",
        "source_role": "training_label",
        "outcome_derived": True,
        "fold_fitted": False,
        "use_role": "training_target",
    }


def _drug_sparse_release_attestations() -> dict[str, object]:
    artifacts = {
        name: {
            "path": f"drug_sparse/{name}.parquet",
            "sha256": format(index + 10, "064x"),
            "rows": 25 if name == "available_predictions" else 10 + index,
        }
        for index, name in enumerate(DRUG_SPARSE_ARTIFACT_NAMES)
    }
    runtime_fingerprint = {
        "python": "3.12.0",
        "packages": {"duckdb": "1.5.0", "pyarrow": "25.0.0"},
    }
    staging_manifest_sha256 = "a" * 64
    staging_provenance_sha256 = "f" * 64
    exact_candidate_sha256 = "0" * 64
    exact_prediction_sha256 = "b" * 64
    exact_lineage_sha256 = "c" * 64
    input_hashes = {
        "exact_candidates": exact_candidate_sha256,
        "exact_release_prediction_proof": exact_prediction_sha256,
        "exact_release_lineage_proof": exact_lineage_sha256,
        "staging_manifest": staging_manifest_sha256,
        "staging_provenance_snapshot": staging_provenance_sha256,
    }
    code_hashes = {
        "drug_training_streaming.py": "1" * 64,
        "full_model_contract.py": "2" * 64,
    }
    exact_binding = {
        "status": "PASS",
        "binding_policy": (
            "LITERAL_FOUR_KEY_BIDIRECTIONAL_EXCEPT_AND_ROW_UNIQUENESS"
        ),
        "key_columns": list(EXACT_BINDING_KEYS),
        "candidate_path": "staging/v32_exact_candidates.parquet",
        "candidate_sha256": exact_candidate_sha256,
        "candidate_rows": 100,
        "candidate_unique_keys": 100,
        "release_prediction_path": "exact_r2/five_fold_ensemble.parquet",
        "release_prediction_sha256": exact_prediction_sha256,
        "release_prediction_rows": 100,
        "release_prediction_unique_keys": 100,
        "release_lineage_path": "exact_r2/MODULE_LINEAGE.json",
        "release_lineage_sha256": exact_lineage_sha256,
        "lineage_training_run_id": "v32-exact-pathway-r2",
        "lineage_newly_trained_v32": True,
        "candidate_null_keys": 0,
        "prediction_null_keys": 0,
        "candidate_minus_prediction": 0,
        "prediction_minus_candidate": 0,
    }
    return {
        "drug_sparse_query_manifest_path": "drug_sparse/DRUG_SPARSE_QUERY_MANIFEST.json",
        "drug_sparse_query_manifest_sha256": "d" * 64,
        "drug_sparse_query_manifest_format": "CC_HHGT_V3_2_DRUG_SPARSE_QUERY_BUNDLE_V1",
        "drug_sparse_query_manifest_status": "SUCCESS",
        "drug_sparse_query_validation_audit_path": (
            "drug_sparse/SPARSE_QUERY_VALIDATION_AUDIT.json"
        ),
        "drug_sparse_query_validation_audit_sha256": "e" * 64,
        "drug_sparse_query_validation_audit_status": "PASS",
        "drug_sparse_query_artifacts": artifacts,
        "cross_dataset_association_policy": DRUG_CROSS_DATASET_ASSOCIATION_POLICY,
        "expression_moment_policy": DRUG_EXPRESSION_MOMENT_POLICY,
        "typed_absence_resolver": True,
        "absent_key_means_unavailable_not_zero": True,
        "dense_candidate_table_materialized": False,
        "conceptual_candidate_rows": 100,
        "available_rows": 25,
        "unavailable_rows": 75,
        "available_keys_unique": True,
        "available_keys_subset_of_conceptual": True,
        "all_prediction_fold_masks_valid": True,
        "all_contributing_folds_same_fold_supported": True,
        "resolver_requires_expected_manifest_sha256": True,
        "all_five_folds_have_optimizer_updates": True,
        "optimizer_steps_total": 50,
        "staging_revalidation_audit_path": "staging/STAGING_REVALIDATION_AUDIT.json",
        "staging_revalidation_audit_sha256": "3" * 64,
        "staging_revalidation_audit_status": "PASS",
        "staging_manifest_path": "staging/STAGING_MANIFEST.json",
        "staging_manifest_sha256": staging_manifest_sha256,
        "staging_provenance_snapshot_path": (
            "staging/STAGING_PROVENANCE_SNAPSHOT.json"
        ),
        "staging_provenance_snapshot_sha256": staging_provenance_sha256,
        "staging_input_artifacts_rehashed": True,
        "staging_execution_code_rehashed": True,
        "staging_runtime_fingerprint_sha256": "4" * 64,
        "exact_release_lineage_contract_validated": True,
        "exact_candidate_four_key_binding_recomputed": True,
        "exact_release_prediction_sha256": exact_prediction_sha256,
        "exact_release_lineage_sha256": exact_lineage_sha256,
        "expected_staging_manifest_sha256": staging_manifest_sha256,
        "expected_exact_release_prediction_sha256": exact_prediction_sha256,
        "expected_exact_release_lineage_sha256": exact_lineage_sha256,
        "exact_candidate_source_binding": exact_binding,
        "input_hashes_start": dict(input_hashes),
        "input_hashes_end": dict(input_hashes),
        "code_hashes_start": dict(code_hashes),
        "code_hashes_end": dict(code_hashes),
        "runtime_fingerprint": runtime_fingerprint,
        "runtime_fingerprint_sha256": _canonical_sha256(runtime_fingerprint),
        "code_hashes_unchanged": True,
        "input_hashes_unchanged": True,
        "release_ready": True,
        "partial_not_publishable": False,
    }


def _lineage(module_id: str) -> dict[str, object]:
    contract = MODULE_CONTRACTS[module_id]
    lineage: dict[str, object] = {
        "module_id": module_id,
        "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
        "training_run_id": f"v32-{module_id}-run",
        "training_status": "SUCCESS",
        "initialization_policy": contract.core_policy,
        "folds": 5,
        "seeds": [20260726],
        "code_sha256": "1" * 64,
        "config_sha256": "2" * 64,
        "input_manifest_sha256": "3" * 64,
        "checkpoint_manifest_sha256": "4" * 64,
        "old_checkpoint_loaded": False,
        "old_predictions_used_as_features": False,
        "old_rankings_used_as_outputs": False,
        "input_artifacts": [_artifact()],
    }
    if module_id == "exact_pathway":
        lineage["trained_from_scratch"] = True
        lineage.update(
            {
                "ensemble_source_manifest_path": "release/ENSEMBLE_SOURCE_MANIFEST.json",
                "ensemble_source_manifest_sha256": "5" * 64,
                "ensemble_columns": [
                    "association_membership_probability",
                    "association_direction_probability",
                    "l1_probability",
                    "ridge_probability",
                    "graph_residual",
                    "graph_gate",
                ],
                "ensemble_formula": "arithmetic_mean_of_five_patient_fold_predictions",
                "ensemble_source_records": [
                    {
                        "patient_fold": fold,
                        "seed": 20260726,
                        "prediction_path": f"fold_{fold}.parquet",
                        "prediction_sha256": "6" * 64,
                        "prediction_rows": 10,
                        "metrics_path": f"fold_{fold}_metrics.json",
                        "metrics_sha256": "7" * 64,
                        "checkpoint_path": f"fold_{fold}.pt",
                        "checkpoint_sha256": "8" * 64,
                        "success_path": f"fold_{fold}_SUCCESS.json",
                        "success_sha256": "9" * 64,
                        "candidate_alignment": "FULL_ROW_EXACT",
                        "checkpoint_cycle": 9,
                        "completed_cycles": 10,
                        "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
                        "trained_from_random_initialization": True,
                        "old_checkpoint_loaded": False,
                        "old_predictions_used_as_features": False,
                        "old_rankings_used_as_outputs": False,
                    }
                    for fold in range(5)
                ],
            }
        )
    else:
        lineage.update(
            {
                "private_head_trained_from_scratch": True,
                "core_parameters_frozen": True,
                "v32_core_checkpoint_sha256": CORE_HASH,
                "core_parameters_before_sha256": CORE_HASH,
                "core_parameters_after_sha256": CORE_HASH,
            }
        )
        lineage["input_artifacts"] = [
            _artifact(),
            _artifact("v32_core_checkpoint", CORE_HASH),
        ]
        if module_id == "drug":
            lineage.update(_drug_sparse_release_attestations())
        if module_id == "single_cell":
            lineage["input_artifacts"] = [
                _single_cell_target_artifact(),
                _artifact("v32_core_checkpoint", CORE_HASH),
            ]
            lineage.update(
                {
                    "formal_datasets": ["SC_BRCA_A"],
                    "trained_folds": 5,
                    "checkpoint_files": 5,
                    "available_rows": 25,
                    "release_ready": True,
                    "core_embedding_usability": [
                        {
                            "single_cell_fold": fold,
                            "lncrna": {
                                "usable_for_node_discrimination": False,
                                "status": "CONSTANT_EMBEDDING_MASKED",
                            },
                            "pathway": {
                                "usable_for_node_discrimination": True,
                                "status": "USABLE",
                            },
                            "lncrna_fallback": (
                                "MASK_CONSTANT_CORE_AND_USE_FRESH_SINGLE_CELL_LNCRNA_FEATURES"
                            ),
                        }
                        for fold in range(5)
                    ],
                    "constant_lncrna_core_masked": True,
                    "lncrna_identity_feature_source": (
                        "fresh_single_cell_detection_expression_specificity"
                    ),
                }
            )
    return lineage


def _release() -> dict[str, object]:
    return {
        "contract_version": FULL_CONTRACT_VERSION,
        "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
        "all_results_newly_trained_v32": True,
        "all_non_null_predictions_newly_trained_v32": True,
        "historical_results_present_in_public_outputs": False,
        "release_ready": True,
        "all_required_modules_statistically_trained": True,
        "modules": {module_id: _lineage(module_id) for module_id in MODULE_CONTRACTS},
    }


def _public_frame(module_id: str) -> pd.DataFrame:
    contract = MODULE_CONTRACTS[module_id]
    row: dict[str, object] = {key: f"{key}-1" for key in contract.target_keys}
    row.update({column: 0.8 for column in contract.probability_columns})
    row.update(
        {
            "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
            "training_run_id": f"v32-{module_id}-run",
        }
    )
    return pd.DataFrame([row])


def test_complete_newly_trained_release_is_accepted() -> None:
    validate_full_release_lineage(_release())


def test_exact_pathway_requires_five_hashed_current_v32_fold_sources() -> None:
    missing = _lineage("exact_pathway")
    missing["ensemble_source_records"] = missing["ensemble_source_records"][:4]
    with pytest.raises(FullModelContractError, match="exactly five source records"):
        validate_module_lineage("exact_pathway", missing)

    old_source = _lineage("exact_pathway")
    old_source["ensemble_source_records"][0]["old_checkpoint_loaded"] = True
    with pytest.raises(FullModelContractError, match="invalid provenance flag"):
        validate_module_lineage("exact_pathway", old_source)

    invalid_hash = _lineage("exact_pathway")
    invalid_hash["ensemble_source_records"][0]["prediction_sha256"] = "not-a-sha"
    with pytest.raises(FullModelContractError, match="not SHA-256"):
        validate_module_lineage("exact_pathway", invalid_hash)


def test_missing_historical_capability_blocks_release() -> None:
    release = _release()
    del release["modules"]["mutation_cnv"]  # type: ignore[index]
    with pytest.raises(FullModelContractError, match="missing=.*mutation_cnv"):
        validate_full_release_lineage(release)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("old_checkpoint_loaded", True, "old checkpoints"),
        ("old_predictions_used_as_features", True, "old predictions"),
        ("old_rankings_used_as_outputs", True, "old rankings"),
    ],
)
def test_old_results_are_rejected(field: str, value: bool, match: str) -> None:
    lineage = _lineage("state")
    lineage[field] = value
    with pytest.raises(FullModelContractError, match=match):
        validate_module_lineage("state", lineage)


def test_renamed_historical_result_input_is_rejected_by_kind() -> None:
    lineage = _lineage("drug")
    lineage["input_artifacts"] = [
        _artifact("historical_prediction"),
        _artifact("v32_core_checkpoint", CORE_HASH),
    ]
    with pytest.raises(FullModelContractError, match="forbidden old result"):
        validate_module_lineage("drug", lineage)


def test_successful_drug_requires_complete_validated_sparse_release() -> None:
    validate_module_lineage("drug", _lineage("drug"))


@pytest.mark.parametrize("field", DRUG_STAGING_REQUIRED_FIELDS)
def test_successful_drug_requires_every_staging_exact_revalidation_field(
    field: str,
) -> None:
    lineage = _lineage("drug")
    lineage.pop(field)
    with pytest.raises(
        FullModelContractError, match="staging/Exact revalidation lacks required fields"
    ):
        validate_module_lineage("drug", lineage)


@pytest.mark.parametrize(
    "field",
    [
        "staging_revalidation_audit_sha256",
        "staging_manifest_sha256",
        "staging_provenance_snapshot_sha256",
        "staging_runtime_fingerprint_sha256",
        "exact_release_prediction_sha256",
        "exact_release_lineage_sha256",
        "expected_staging_manifest_sha256",
        "expected_exact_release_prediction_sha256",
        "expected_exact_release_lineage_sha256",
        "runtime_fingerprint_sha256",
    ],
)
def test_successful_drug_rejects_invalid_provenance_sha256(field: str) -> None:
    lineage = _lineage("drug")
    lineage[field] = "not-a-sha"
    with pytest.raises(FullModelContractError, match="strict 64-hex"):
        validate_module_lineage("drug", lineage)


@pytest.mark.parametrize(
    ("actual_field", "expected_field"),
    [
        ("staging_manifest_sha256", "expected_staging_manifest_sha256"),
        (
            "exact_release_prediction_sha256",
            "expected_exact_release_prediction_sha256",
        ),
        ("exact_release_lineage_sha256", "expected_exact_release_lineage_sha256"),
    ],
)
def test_successful_drug_requires_actual_release_hashes_to_match_expected(
    actual_field: str, expected_field: str
) -> None:
    lineage = _lineage("drug")
    lineage[expected_field] = "d" * 64
    with pytest.raises(
        FullModelContractError, match=f"{actual_field} does not match {expected_field}"
    ):
        validate_module_lineage("drug", lineage)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        (
            "staging_revalidation_audit_status",
            "SKIPPED",
            "audit status must be PASS",
        ),
        (
            "staging_input_artifacts_rehashed",
            False,
            "staging_input_artifacts_rehashed=true",
        ),
        (
            "staging_execution_code_rehashed",
            False,
            "staging_execution_code_rehashed=true",
        ),
        (
            "exact_release_lineage_contract_validated",
            False,
            "exact_release_lineage_contract_validated=true",
        ),
        (
            "exact_candidate_four_key_binding_recomputed",
            False,
            "exact_candidate_four_key_binding_recomputed=true",
        ),
    ],
)
def test_successful_drug_rejects_false_staging_revalidation_attestations(
    field: str, value: object, match: str
) -> None:
    lineage = _lineage("drug")
    lineage[field] = value
    with pytest.raises(FullModelContractError, match=match):
        validate_module_lineage("drug", lineage)


@pytest.mark.parametrize("scope", ["input", "code"])
def test_successful_drug_requires_nonempty_hash_snapshot_mappings(scope: str) -> None:
    lineage = _lineage("drug")
    lineage[f"{scope}_hashes_start"] = {}
    with pytest.raises(FullModelContractError, match="must be a non-empty mapping"):
        validate_module_lineage("drug", lineage)


@pytest.mark.parametrize("scope", ["input", "code"])
def test_successful_drug_requires_identical_start_end_hash_snapshots(scope: str) -> None:
    lineage = _lineage("drug")
    lineage[f"{scope}_hashes_end"] = dict(lineage[f"{scope}_hashes_end"])
    lineage[f"{scope}_hashes_end"]["drifted"] = "d" * 64
    with pytest.raises(FullModelContractError, match="hashes_start == .*hashes_end"):
        validate_module_lineage("drug", lineage)


@pytest.mark.parametrize("scope", ["input", "code"])
def test_successful_drug_validates_every_snapshot_digest(scope: str) -> None:
    lineage = _lineage("drug")
    lineage[f"{scope}_hashes_start"] = {"artifact": "bad"}
    lineage[f"{scope}_hashes_end"] = {"artifact": "bad"}
    with pytest.raises(FullModelContractError, match="strict 64-hex"):
        validate_module_lineage("drug", lineage)


def test_successful_drug_requires_canonical_runtime_fingerprint_hash() -> None:
    not_a_mapping = _lineage("drug")
    not_a_mapping["runtime_fingerprint"] = "python=unknown"
    with pytest.raises(FullModelContractError, match="must be a non-empty mapping"):
        validate_module_lineage("drug", not_a_mapping)

    drifted = _lineage("drug")
    drifted["runtime_fingerprint"] = dict(drifted["runtime_fingerprint"])
    drifted["runtime_fingerprint"]["python"] = "changed"
    with pytest.raises(FullModelContractError, match="canonical runtime fingerprint"):
        validate_module_lineage("drug", drifted)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("status", "SKIPPED", "binding status must be PASS"),
        ("binding_policy", "CLAIM_ONLY", "binding policy drifted"),
        ("key_columns", list(reversed(EXACT_BINDING_KEYS)), "key order drifted"),
        (
            "lineage_newly_trained_v32",
            False,
            "lacks newly-trained V3.2 lineage",
        ),
    ],
)
def test_successful_drug_rejects_invalid_exact_binding_identity(
    field: str, value: object, match: str
) -> None:
    lineage = _lineage("drug")
    lineage["exact_candidate_source_binding"][field] = value
    with pytest.raises(FullModelContractError, match=match):
        validate_module_lineage("drug", lineage)


@pytest.mark.parametrize(
    "field",
    [
        "candidate_rows",
        "candidate_unique_keys",
        "release_prediction_rows",
        "release_prediction_unique_keys",
    ],
)
def test_successful_drug_requires_all_exact_binding_counts_positive(field: str) -> None:
    lineage = _lineage("drug")
    lineage["exact_candidate_source_binding"][field] = 0
    with pytest.raises(FullModelContractError, match="must be a positive integer"):
        validate_module_lineage("drug", lineage)


def test_successful_drug_requires_exact_binding_counts_equal() -> None:
    lineage = _lineage("drug")
    lineage["exact_candidate_source_binding"]["release_prediction_rows"] = 99
    with pytest.raises(FullModelContractError, match="counts must all be equal"):
        validate_module_lineage("drug", lineage)


@pytest.mark.parametrize(
    "field",
    [
        "candidate_null_keys",
        "prediction_null_keys",
        "candidate_minus_prediction",
        "prediction_minus_candidate",
    ],
)
def test_successful_drug_requires_zero_exact_binding_nulls_and_diffs(
    field: str,
) -> None:
    lineage = _lineage("drug")
    lineage["exact_candidate_source_binding"][field] = 1
    with pytest.raises(FullModelContractError, match="must be integer zero"):
        validate_module_lineage("drug", lineage)


@pytest.mark.parametrize(
    ("binding_field", "match"),
    [
        ("release_prediction_sha256", "prediction SHA-256 does not match"),
        ("release_lineage_sha256", "lineage SHA-256 does not match"),
        ("candidate_sha256", "candidate SHA-256 is absent"),
    ],
)
def test_successful_drug_binds_exact_hashes_to_release_and_input_snapshot(
    binding_field: str, match: str
) -> None:
    lineage = _lineage("drug")
    lineage["exact_candidate_source_binding"][binding_field] = "d" * 64
    with pytest.raises(FullModelContractError, match=match):
        validate_module_lineage("drug", lineage)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("drug_sparse_query_manifest_sha256", "not-a-sha", "strict 64-hex"),
        ("drug_sparse_query_validation_audit_sha256", "f" * 63, "strict 64-hex"),
        ("drug_sparse_query_manifest_format", "V0", "format drifted"),
        ("drug_sparse_query_manifest_status", "AUDITED_UNAVAILABLE", "must be SUCCESS"),
        ("drug_sparse_query_validation_audit_status", "SKIPPED", "must be PASS"),
        (
            "cross_dataset_association_policy",
            "FISHER_Z_FOR_OVERLAPPING_MODELS",
            "cross-dataset association policy drifted",
        ),
        (
            "expression_moment_policy",
            "DATASET_DUPLICATED_EXPRESSION_MOMENTS",
            "expression moment policy drifted",
        ),
        ("typed_absence_resolver", False, "typed_absence_resolver=true"),
        (
            "absent_key_means_unavailable_not_zero",
            False,
            "absent_key_means_unavailable_not_zero=true",
        ),
        ("dense_candidate_table_materialized", True, "dense_candidate_table_materialized=false"),
        ("available_keys_unique", False, "available_keys_unique=true"),
        (
            "available_keys_subset_of_conceptual",
            False,
            "available_keys_subset_of_conceptual=true",
        ),
        ("all_prediction_fold_masks_valid", False, "all_prediction_fold_masks_valid=true"),
        (
            "all_contributing_folds_same_fold_supported",
            False,
            "all_contributing_folds_same_fold_supported=true",
        ),
        (
            "resolver_requires_expected_manifest_sha256",
            False,
            "resolver_requires_expected_manifest_sha256=true",
        ),
        (
            "all_five_folds_have_optimizer_updates",
            False,
            "all_five_folds_have_optimizer_updates=true",
        ),
        ("optimizer_steps_total", 0, "positive integer"),
        ("code_hashes_unchanged", False, "code_hashes_unchanged=true"),
        ("input_hashes_unchanged", False, "input_hashes_unchanged=true"),
        ("release_ready", False, "release_ready=true"),
        ("partial_not_publishable", True, "partial_not_publishable=false"),
    ],
)
def test_successful_drug_sparse_release_rejects_invalid_attestations(
    field: str, value: object, match: str
) -> None:
    lineage = _lineage("drug")
    lineage[field] = value
    with pytest.raises(FullModelContractError, match=match):
        validate_module_lineage("drug", lineage)


def test_successful_drug_sparse_release_requires_exact_hashed_artifact_set() -> None:
    missing = _lineage("drug")
    del missing["drug_sparse_query_artifacts"]["expression_fold_coverage"]  # type: ignore[index]
    with pytest.raises(FullModelContractError, match="artifact set mismatch"):
        validate_module_lineage("drug", missing)

    extra = _lineage("drug")
    extra["drug_sparse_query_artifacts"]["dense_candidates"] = {  # type: ignore[index]
        "path": "forbidden.parquet",
        "sha256": "f" * 64,
        "rows": 1,
    }
    with pytest.raises(FullModelContractError, match="artifact set mismatch"):
        validate_module_lineage("drug", extra)

    invalid_hash = _lineage("drug")
    invalid_hash["drug_sparse_query_artifacts"]["available_predictions"][  # type: ignore[index]
        "sha256"
    ] = "g" * 64
    with pytest.raises(FullModelContractError, match="strict 64-hex"):
        validate_module_lineage("drug", invalid_hash)


@pytest.mark.parametrize(
    "updates,match",
    [
        ({"available_rows": 0, "unavailable_rows": 100}, "positive integer"),
        ({"conceptual_candidate_rows": 24, "available_rows": 25}, "conceptual_candidate_rows"),
        ({"unavailable_rows": 74}, "must equal"),
    ],
)
def test_successful_drug_sparse_release_rejects_invalid_row_accounting(
    updates: dict[str, int], match: str
) -> None:
    lineage = _lineage("drug")
    lineage.update(updates)
    with pytest.raises(FullModelContractError, match=match):
        validate_module_lineage("drug", lineage)


def test_successful_drug_sparse_release_matches_available_artifact_rows() -> None:
    lineage = _lineage("drug")
    lineage["drug_sparse_query_artifacts"]["available_predictions"][  # type: ignore[index]
        "rows"
    ] = 24
    with pytest.raises(FullModelContractError, match="must equal available_rows"):
        validate_module_lineage("drug", lineage)


def test_audited_unavailable_drug_does_not_require_sparse_success_fields() -> None:
    lineage = _lineage("drug")
    for field in _drug_sparse_release_attestations():
        lineage.pop(field, None)
    lineage.update(
        {
            "training_status": "AUDITED_UNAVAILABLE",
            "private_head_trained_from_scratch": False,
            "trained_folds": 0,
            "checkpoint_files": 0,
            "release_ready": False,
            "all_probabilities_null": True,
            "all_unavailable_rows_have_reason": True,
        }
    )
    validate_module_lineage("drug", lineage)


def test_auxiliary_parent_must_be_the_declared_v32_core() -> None:
    lineage = _lineage("clinical")
    lineage["input_artifacts"][1]["sha256"] = "d" * 64  # type: ignore[index]
    with pytest.raises(FullModelContractError, match="parent core checkpoint hash"):
        validate_module_lineage("clinical", lineage)


def test_auxiliary_training_cannot_change_core() -> None:
    lineage = _lineage("single_cell")
    lineage["core_parameters_after_sha256"] = "e" * 64
    with pytest.raises(FullModelContractError, match="changed frozen"):
        validate_module_lineage("single_cell", lineage)


def test_successful_single_cell_must_mask_constant_lncrna_core() -> None:
    lineage = _lineage("single_cell")
    lineage["constant_lncrna_core_masked"] = False
    with pytest.raises(FullModelContractError, match="was not masked"):
        validate_module_lineage("single_cell", lineage)

    lineage = _lineage("single_cell")
    lineage["core_embedding_usability"][0]["lncrna_fallback"] = None
    with pytest.raises(FullModelContractError, match="approved fallback"):
        validate_module_lineage("single_cell", lineage)


def test_single_cell_outcome_target_cannot_masquerade_as_raw_data() -> None:
    lineage = _lineage("single_cell")
    target = lineage["input_artifacts"][0]
    target["source_role"] = "raw_data"
    target["artifact_kind"] = "raw_data"
    with pytest.raises(FullModelContractError, match="training_label, not raw_data"):
        validate_module_lineage("single_cell", lineage)


def test_single_cell_historical_training_label_cannot_unlock() -> None:
    lineage = _lineage("single_cell")
    lineage["input_artifacts"][0]["generation"] = "V2.5_CELLTYPE_STAGING"
    with pytest.raises(FullModelContractError, match="regenerated inside V3.2"):
        validate_module_lineage("single_cell", lineage)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("formal_datasets", [], "formal_datasets=\\[\\]"),
        ("trained_folds", 0, "requires five trained folds"),
    ],
)
def test_single_cell_empty_formal_data_or_zero_folds_cannot_unlock(
    field: str, value: object, match: str
) -> None:
    lineage = _lineage("single_cell")
    lineage[field] = value
    lineage["release_ready"] = True
    with pytest.raises(FullModelContractError, match=match):
        validate_module_lineage("single_cell", lineage)


def test_formally_unavailable_private_head_is_honest_and_fail_closed() -> None:
    lineage = _lineage("evidence")
    lineage.update(
        {
            "statistical_training_status": "UNAVAILABLE_STRICT_COMPONENT_ISOLATION",
            "failure_reason": "STRICT_CONNECTED_COMPONENT_COUNT_1_LT_5",
            "private_head_trained_from_scratch": False,
            "private_head_initialized_from_scratch": True,
            "private_head_optimizer_steps": 0,
        }
    )
    validate_module_lineage("evidence", lineage)

    false_attestation = deepcopy(lineage)
    false_attestation["private_head_trained_from_scratch"] = True
    with pytest.raises(FullModelContractError, match="falsely attests"):
        validate_module_lineage("evidence", false_attestation)


def test_audited_unavailable_module_requires_zero_training_and_all_null_attestation() -> None:
    lineage = _lineage("single_cell")
    lineage.update(
        {
            "training_status": "AUDITED_UNAVAILABLE",
            "private_head_trained_from_scratch": False,
            "trained_folds": 0,
            "checkpoint_files": 0,
            "release_ready": False,
            "all_probabilities_null": True,
            "all_unavailable_rows_have_reason": True,
        }
    )
    validate_module_lineage("single_cell", lineage)

    false_training = deepcopy(lineage)
    false_training["trained_folds"] = 1
    with pytest.raises(FullModelContractError, match="zero trained folds"):
        validate_module_lineage("single_cell", false_training)


def test_full_registry_with_audited_unavailable_module_cannot_be_release_ready() -> None:
    release = _release()
    module = release["modules"]["single_cell"]  # type: ignore[index]
    module.update(
        {
            "training_status": "AUDITED_UNAVAILABLE",
            "private_head_trained_from_scratch": False,
            "trained_folds": 0,
            "checkpoint_files": 0,
            "release_ready": False,
            "all_probabilities_null": True,
            "all_unavailable_rows_have_reason": True,
        }
    )
    release["release_ready"] = False
    release["all_required_modules_statistically_trained"] = False
    validate_full_release_lineage(release)

    release["release_ready"] = True
    with pytest.raises(FullModelContractError, match="release_ready=false"):
        validate_full_release_lineage(release)


def test_dense_drug_development_lineage_is_valid_but_cannot_enter_release() -> None:
    lineage = _lineage("drug")
    for key in _drug_sparse_release_attestations():
        lineage.pop(key, None)
    lineage.update(
        {
            "formal_release_eligible": False,
            "dense_development_only": True,
            "release_ready": False,
            "partial_not_publishable": True,
        }
    )
    validate_module_lineage("drug", lineage)

    release = _release()
    release["modules"]["drug"] = lineage  # type: ignore[index]
    with pytest.raises(FullModelContractError, match="non-releaseable development"):
        validate_full_release_lineage(release)


@pytest.mark.parametrize("module_id", list(MODULE_CONTRACTS))
def test_each_module_has_a_valid_public_schema(module_id: str) -> None:
    validate_public_module_frame(module_id, _public_frame(module_id))


def test_public_output_rejects_labels_and_old_probabilities() -> None:
    labels = _public_frame("exact_pathway")
    labels["held_out_proxy_label"] = 1
    with pytest.raises(FullModelContractError, match="leaks labels"):
        validate_public_module_frame("exact_pathway", labels)

    old = _public_frame("state")
    old["legacy_probability_v2_9"] = 0.9
    with pytest.raises(FullModelContractError, match="historical result columns"):
        validate_public_module_frame("state", old)


def test_auxiliary_layer_cannot_change_primary_ranking() -> None:
    frame = _public_frame("evidence")
    frame["changes_primary_ranking"] = True
    with pytest.raises(FullModelContractError, match="changes primary ranking"):
        validate_public_module_frame("evidence", frame)


def test_unavailable_rows_are_null_with_an_explicit_reason() -> None:
    frame = _public_frame("clinical")
    frame["availability"] = False
    frame["failure_reason"] = "NO_DISTINCT_DFS_SOURCE"
    frame["clinical_relevance_probability"] = float("nan")
    validate_public_module_frame("clinical", frame)

    fake = frame.copy()
    fake["clinical_relevance_probability"] = 0.5
    with pytest.raises(FullModelContractError, match="must be null"):
        validate_public_module_frame("clinical", fake)


def test_module_specific_unavailability_columns_are_enforced() -> None:
    frame = _public_frame("single_cell")
    frame["single_cell_available"] = False
    frame["single_cell_unavailable_reason"] = "STAGING_SOURCE_NOT_FORMAL"
    frame["single_cell_replication_probability"] = float("nan")
    validate_public_module_frame("single_cell", frame)

    fake = frame.copy()
    fake["single_cell_replication_probability"] = 0.5
    with pytest.raises(FullModelContractError, match="must be null"):
        validate_public_module_frame("single_cell", fake)


def test_release_attestation_is_mandatory() -> None:
    release = deepcopy(_release())
    release["all_results_newly_trained_v32"] = False
    release["all_non_null_predictions_newly_trained_v32"] = False
    with pytest.raises(FullModelContractError, match="all-non-null-predictions-newly-trained"):
        validate_full_release_lineage(release)
