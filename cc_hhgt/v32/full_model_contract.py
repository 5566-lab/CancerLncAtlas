"""Fail-closed contract for the complete, newly trained V3.2 model.

The exact-pathway discovery model and every companion endpoint belong to one
release, but they remain separate estimands.  Historical data and task
definitions may be reused after hashing and re-standardisation.  Historical
checkpoints, probabilities, rankings, and release tables may never be used as
V3.2 predictions or training features.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


FULL_CONTRACT_VERSION = "3.2.0-full-model-v1"
V32_ANALYSIS_PREFIX = "CancerLncAtlas_V3.2"
N_FOLDS = 5
DRUG_CROSS_DATASET_ASSOCIATION_POLICY = (
    "NO_OVERLAP_FISHER_Z_N_MINUS_3__OVERLAP_UNIQUE_CANONICAL_MODEL_"
    "MIDRANK_PERCENTILE_CONSENSUS_SPEARMAN_V1"
)
DRUG_EXPRESSION_MOMENT_POLICY = (
    "UNIQUE_CANONICAL_MODEL__DATASET_SPECIFIC_ARITHMETIC_MEAN_CONSENSUS_V1"
)


@dataclass(frozen=True)
class ModuleContract:
    module_id: str
    target_level: str
    target_keys: tuple[str, ...]
    probability_columns: tuple[str, ...]
    split_unit: str
    model_family: str
    core_policy: str
    availability_column: str | None = "availability"
    failure_reason_column: str | None = "failure_reason"
    may_change_primary_ranking: bool = False


MODULE_CONTRACTS: dict[str, ModuleContract] = {
    "exact_pathway": ModuleContract(
        module_id="exact_pathway",
        target_level="cancer_x_lncrna_x_exact_pathway",
        target_keys=("cancer_id", "lncrna_id", "pathway_id"),
        probability_columns=("association_membership_probability",),
        split_unit="patient",
        model_family="cc_hhgt_bounded_l1_residual",
        core_policy="TRAIN_FROM_RANDOM_INITIALIZATION",
        availability_column=None,
        failure_reason_column=None,
        may_change_primary_ranking=True,
    ),
    "state": ModuleContract(
        module_id="state",
        target_level="cancer_x_lncrna_x_tumor_state",
        target_keys=("cancer_id", "lncrna_id", "state_id"),
        probability_columns=("state_membership_probability",),
        split_unit="patient",
        model_family="v32_frozen_core_private_state_head",
        core_policy="FROZEN_V32_CORE_PRIVATE_HEAD_FROM_SCRATCH",
        availability_column="availability",
        failure_reason_column="availability_reason",
    ),
    "clinical": ModuleContract(
        module_id="clinical",
        target_level="cancer_x_subject_x_clinical_endpoint",
        target_keys=("cancer_id", "subject_type", "subject_id", "clinical_endpoint"),
        probability_columns=("clinical_relevance_probability",),
        split_unit="patient",
        model_family="v32_frozen_core_private_clinical_head",
        core_policy="FROZEN_V32_CORE_PRIVATE_HEAD_FROM_SCRATCH",
    ),
    "single_cell": ModuleContract(
        module_id="single_cell",
        target_level="dataset_x_celltype_x_lncrna_x_exact_pathway",
        target_keys=("dataset_id", "cell_type", "lncrna_id", "pathway_id"),
        probability_columns=("single_cell_replication_probability",),
        split_unit="dataset_or_donor",
        model_family="v32_frozen_core_private_single_cell_head",
        core_policy="FROZEN_V32_CORE_PRIVATE_HEAD_FROM_SCRATCH",
        availability_column="single_cell_available",
        failure_reason_column="single_cell_unavailable_reason",
    ),
    "mutation_cnv": ModuleContract(
        module_id="mutation_cnv",
        target_level="cancer_x_lncrna_x_exact_pathway_genomic_context",
        target_keys=("cancer_id", "lncrna_id", "pathway_id"),
        probability_columns=("mutation_cnv_context_probability",),
        split_unit="patient",
        model_family="v32_frozen_core_private_genomic_head",
        core_policy="FROZEN_V32_CORE_PRIVATE_HEAD_FROM_SCRATCH",
        availability_column="genomic_available",
        failure_reason_column="genomic_unavailable_reason",
    ),
    "drug": ModuleContract(
        module_id="drug",
        target_level="cancer_x_lncrna_x_drug_response",
        target_keys=("cancer_id", "lncrna_id", "drug_id"),
        probability_columns=("drug_response_association_probability",),
        split_unit="cell_line_or_dataset",
        model_family="v32_frozen_core_private_drug_head",
        core_policy="FROZEN_V32_CORE_PRIVATE_HEAD_FROM_SCRATCH",
    ),
    "interaction": ModuleContract(
        module_id="interaction",
        target_level="lncrna_x_protein_physical_interaction",
        target_keys=("lncrna_id", "protein_id"),
        probability_columns=("physical_interaction_probability",),
        split_unit="publication_and_pair",
        model_family="v32_frozen_core_private_interaction_head",
        core_policy="FROZEN_V32_CORE_PRIVATE_HEAD_FROM_SCRATCH",
    ),
    "evidence": ModuleContract(
        module_id="evidence",
        target_level="cancer_x_lncrna_x_exact_pathway_evidence_confidence",
        target_keys=("cancer_id", "lncrna_id", "pathway_id"),
        probability_columns=("evidence_confidence_probability",),
        split_unit="publication_dataset_and_pair",
        model_family="v32_exact_pathway_evidence_transformer",
        core_policy="FROZEN_V32_CORE_PRIVATE_HEAD_FROM_SCRATCH",
    ),
}

REQUIRED_MODULES = tuple(MODULE_CONTRACTS)

ALLOWED_INPUT_KINDS = frozenset(
    {
        "raw_data",
        "standardized_input",
        "annotation",
        "split_manifest",
        "training_label",
        "v32_core_checkpoint",
    }
)
FORBIDDEN_HISTORICAL_RESULT_KINDS = frozenset(
    {
        "historical_checkpoint",
        "historical_prediction",
        "historical_probability",
        "historical_ranking",
        "historical_release_table",
        "historical_web_table",
    }
)
FORBIDDEN_PUBLIC_COLUMNS = frozenset(
    {
        "label",
        "label_class",
        "proxy_label",
        "association_proxy_label",
        "strong_association_label",
        "strong_evidence_label",
        "sample_weight",
        "held_out_proxy_label",
    }
)
HISTORICAL_RESULT_COLUMN_TOKENS = (
    "_v2_6",
    "_v2_7",
    "_v2_8",
    "_v2_9",
    "_v3_0",
    "_v3_1",
    "legacy_probability",
    "historical_probability",
)

DRUG_SPARSE_QUERY_MANIFEST_FORMAT = "CC_HHGT_V3_2_DRUG_SPARSE_QUERY_BUNDLE_V1"
DRUG_SPARSE_QUERY_ARTIFACTS = frozenset(
    {
        "exact_candidates",
        "pathway_drug_edges",
        "assay_fold_eligibility",
        "expression_fold_coverage",
        "core_entity_availability",
        "available_predictions",
    }
)
DRUG_EXACT_BINDING_POLICY = (
    "LITERAL_FOUR_KEY_BIDIRECTIONAL_EXCEPT_AND_ROW_UNIQUENESS"
)
DRUG_EXACT_BINDING_KEYS = (
    "cancer_id",
    "lncrna_id",
    "pathway_id",
    "pathway_family_id",
)


class FullModelContractError(RuntimeError):
    """Raised when a purported complete V3.2 release is not newly trained."""


def contract_manifest() -> dict[str, Any]:
    """Return a JSON-serialisable description of the frozen module contract."""

    return {
        "contract_version": FULL_CONTRACT_VERSION,
        "analysis_version_prefix": V32_ANALYSIS_PREFIX,
        "required_modules": list(REQUIRED_MODULES),
        "modules": {name: asdict(contract) for name, contract in MODULE_CONTRACTS.items()},
        "historical_policy": {
            "raw_or_restandardized_inputs_allowed": True,
            "old_code_as_reviewed_reference_allowed": True,
            "historical_checkpoints_allowed": False,
            "historical_predictions_as_features_allowed": False,
            "historical_predictions_as_release_outputs_allowed": False,
        },
    }


def _require_nonempty(mapping: Mapping[str, Any], keys: Sequence[str], context: str) -> None:
    missing = [key for key in keys if key not in mapping or mapping[key] in (None, "", [])]
    if missing:
        raise FullModelContractError(f"{context} lacks required fields: {missing}")


def _require_sha256(value: Any, context: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        char not in "0123456789abcdef" for char in value.lower()
    ):
        raise FullModelContractError(f"{context} is not a strict 64-hex SHA-256")


def _require_positive_int(value: Any, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise FullModelContractError(f"{context} must be a positive integer")
    return value


def _canonical_sha256(value: Any) -> str:
    """Match the canonical JSON hashing used by the training entrypoint."""

    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    except (TypeError, ValueError) as error:
        raise FullModelContractError(
            "Drug runtime fingerprint is not canonically serialisable"
        ) from error
    return hashlib.sha256(encoded).hexdigest()


def _require_hash_mapping(value: Any, context: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise FullModelContractError(f"{context} must be a non-empty mapping")
    result: dict[str, str] = {}
    for raw_key, digest in value.items():
        if not isinstance(raw_key, str) or not raw_key.strip():
            raise FullModelContractError(f"{context} contains an invalid artifact key")
        _require_sha256(digest, f"{context} artifact {raw_key}")
        result[raw_key] = digest
    return result


def _validate_drug_hash_snapshots(lineage: Mapping[str, Any]) -> dict[str, str]:
    snapshots: dict[str, dict[str, str]] = {}
    for scope in ("input", "code"):
        start_key = f"{scope}_hashes_start"
        end_key = f"{scope}_hashes_end"
        start = _require_hash_mapping(
            lineage.get(start_key), f"Drug {scope} start hashes"
        )
        end = _require_hash_mapping(
            lineage.get(end_key), f"Drug {scope} end hashes"
        )
        if start != end:
            raise FullModelContractError(
                f"Successful Drug sparse release requires {start_key} == {end_key}"
            )
        snapshots[scope] = start
    return snapshots["input"]


def _validate_successful_drug_staging_revalidation(
    lineage: Mapping[str, Any],
) -> None:
    context = "successful Drug staging/Exact revalidation"
    required = (
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
        "input_hashes_start",
        "input_hashes_end",
        "code_hashes_start",
        "code_hashes_end",
        "runtime_fingerprint",
        "runtime_fingerprint_sha256",
    )
    _require_nonempty(lineage, required, context)

    for path_field in (
        "staging_revalidation_audit_path",
        "staging_manifest_path",
        "staging_provenance_snapshot_path",
    ):
        if not isinstance(lineage[path_field], str) or not lineage[path_field].strip():
            raise FullModelContractError(f"Drug {path_field} must be a non-empty path")

    sha_fields = (
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
    )
    for field in sha_fields:
        _require_sha256(lineage[field], f"Drug {field}")

    if lineage["staging_revalidation_audit_status"] != "PASS":
        raise FullModelContractError(
            "Drug staging revalidation audit status must be PASS"
        )
    for flag in (
        "staging_input_artifacts_rehashed",
        "staging_execution_code_rehashed",
        "exact_release_lineage_contract_validated",
        "exact_candidate_four_key_binding_recomputed",
    ):
        if lineage.get(flag) is not True:
            raise FullModelContractError(
                f"Successful Drug staging revalidation requires {flag}=true"
            )

    actual_expected_pairs = (
        ("staging_manifest_sha256", "expected_staging_manifest_sha256"),
        (
            "exact_release_prediction_sha256",
            "expected_exact_release_prediction_sha256",
        ),
        ("exact_release_lineage_sha256", "expected_exact_release_lineage_sha256"),
    )
    for actual_field, expected_field in actual_expected_pairs:
        if lineage[actual_field] != lineage[expected_field]:
            raise FullModelContractError(
                f"Drug {actual_field} does not match {expected_field}"
            )

    input_hashes = _validate_drug_hash_snapshots(lineage)
    required_input_hashes = {
        "staging_manifest": lineage["staging_manifest_sha256"],
        "staging_provenance_snapshot": lineage[
            "staging_provenance_snapshot_sha256"
        ],
        "exact_release_prediction_proof": lineage[
            "exact_release_prediction_sha256"
        ],
        "exact_release_lineage_proof": lineage["exact_release_lineage_sha256"],
    }
    for role, expected_sha256 in required_input_hashes.items():
        if input_hashes.get(role) != expected_sha256:
            raise FullModelContractError(
                f"Drug input hash snapshot does not bind {role} to its attested SHA-256"
            )

    runtime = lineage["runtime_fingerprint"]
    if not isinstance(runtime, Mapping) or not runtime:
        raise FullModelContractError(
            "Drug runtime_fingerprint must be a non-empty mapping"
        )
    if _canonical_sha256(runtime) != lineage["runtime_fingerprint_sha256"]:
        raise FullModelContractError(
            "Drug runtime_fingerprint_sha256 does not match the canonical runtime fingerprint"
        )

    binding = lineage["exact_candidate_source_binding"]
    if not isinstance(binding, Mapping):
        raise FullModelContractError(
            "Drug exact_candidate_source_binding must be a mapping"
        )
    _require_nonempty(
        binding,
        (
            "status",
            "binding_policy",
            "key_columns",
            "candidate_path",
            "candidate_sha256",
            "candidate_rows",
            "candidate_unique_keys",
            "release_prediction_path",
            "release_prediction_sha256",
            "release_prediction_rows",
            "release_prediction_unique_keys",
            "release_lineage_path",
            "release_lineage_sha256",
            "lineage_training_run_id",
            "lineage_newly_trained_v32",
            "candidate_null_keys",
            "prediction_null_keys",
            "candidate_minus_prediction",
            "prediction_minus_candidate",
        ),
        "Drug exact candidate source binding",
    )
    if binding["status"] != "PASS":
        raise FullModelContractError("Drug exact candidate source binding status must be PASS")
    if binding["binding_policy"] != DRUG_EXACT_BINDING_POLICY:
        raise FullModelContractError("Drug exact candidate source binding policy drifted")
    key_columns = binding["key_columns"]
    if (
        not isinstance(key_columns, Sequence)
        or isinstance(key_columns, (str, bytes))
        or list(key_columns) != list(DRUG_EXACT_BINDING_KEYS)
    ):
        raise FullModelContractError(
            "Drug exact candidate source binding key order drifted"
        )
    if binding.get("lineage_newly_trained_v32") is not True:
        raise FullModelContractError(
            "Drug exact candidate source binding lacks newly-trained V3.2 lineage"
        )
    for path_field in (
        "candidate_path",
        "release_prediction_path",
        "release_lineage_path",
        "lineage_training_run_id",
    ):
        if not isinstance(binding[path_field], str) or not binding[path_field].strip():
            raise FullModelContractError(
                f"Drug exact candidate source binding {path_field} must be non-empty"
            )
    for sha_field in (
        "candidate_sha256",
        "release_prediction_sha256",
        "release_lineage_sha256",
    ):
        _require_sha256(
            binding[sha_field],
            f"Drug exact candidate source binding {sha_field}",
        )
    if binding["release_prediction_sha256"] != lineage["exact_release_prediction_sha256"]:
        raise FullModelContractError(
            "Drug exact binding prediction SHA-256 does not match the attested release"
        )
    if binding["release_lineage_sha256"] != lineage["exact_release_lineage_sha256"]:
        raise FullModelContractError(
            "Drug exact binding lineage SHA-256 does not match the attested release"
        )
    if input_hashes.get("exact_candidates") != binding["candidate_sha256"]:
        raise FullModelContractError(
            "Drug exact binding candidate SHA-256 is absent from the input hash snapshot"
        )

    positive_count_fields = (
        "candidate_rows",
        "candidate_unique_keys",
        "release_prediction_rows",
        "release_prediction_unique_keys",
    )
    counts = [
        _require_positive_int(
            binding[field], f"Drug exact candidate source binding {field}"
        )
        for field in positive_count_fields
    ]
    if len(set(counts)) != 1:
        raise FullModelContractError(
            "Drug exact candidate source binding row/unique counts must all be equal"
        )
    for field in (
        "candidate_null_keys",
        "prediction_null_keys",
        "candidate_minus_prediction",
        "prediction_minus_candidate",
    ):
        value = binding[field]
        if not isinstance(value, int) or isinstance(value, bool) or value != 0:
            raise FullModelContractError(
                f"Drug exact candidate source binding {field} must be integer zero"
            )


def _validate_successful_drug_sparse_release(lineage: Mapping[str, Any]) -> None:
    """Block a successful Drug release unless its sparse resolver is auditable."""

    _validate_successful_drug_staging_revalidation(lineage)
    context = "successful Drug sparse release"
    _require_nonempty(
        lineage,
        (
            "drug_sparse_query_manifest_path",
            "drug_sparse_query_manifest_sha256",
            "drug_sparse_query_manifest_format",
            "drug_sparse_query_manifest_status",
            "drug_sparse_query_validation_audit_path",
            "drug_sparse_query_validation_audit_sha256",
            "drug_sparse_query_validation_audit_status",
            "drug_sparse_query_artifacts",
            "cross_dataset_association_policy",
            "expression_moment_policy",
            "conceptual_candidate_rows",
            "available_rows",
            "unavailable_rows",
            "optimizer_steps_total",
        ),
        context,
    )
    _require_sha256(
        lineage["drug_sparse_query_manifest_sha256"],
        "Drug sparse query manifest SHA-256",
    )
    _require_sha256(
        lineage["drug_sparse_query_validation_audit_sha256"],
        "Drug sparse query validation audit SHA-256",
    )
    if lineage["drug_sparse_query_manifest_format"] != DRUG_SPARSE_QUERY_MANIFEST_FORMAT:
        raise FullModelContractError("Drug sparse query manifest format drifted")
    if lineage["drug_sparse_query_manifest_status"] != "SUCCESS":
        raise FullModelContractError("Drug sparse query manifest status must be SUCCESS")
    if lineage["drug_sparse_query_validation_audit_status"] != "PASS":
        raise FullModelContractError("Drug sparse query validation audit status must be PASS")
    if (
        lineage["cross_dataset_association_policy"]
        != DRUG_CROSS_DATASET_ASSOCIATION_POLICY
    ):
        raise FullModelContractError(
            "Drug cross-dataset association policy drifted from unique-SIDM consensus"
        )
    if lineage["expression_moment_policy"] != DRUG_EXPRESSION_MOMENT_POLICY:
        raise FullModelContractError(
            "Drug expression moment policy drifted from unique canonical-model moments"
        )

    artifacts = lineage["drug_sparse_query_artifacts"]
    if not isinstance(artifacts, Mapping):
        raise FullModelContractError("Drug sparse query artifacts must be a mapping")
    observed_artifacts = set(artifacts)
    if observed_artifacts != DRUG_SPARSE_QUERY_ARTIFACTS:
        missing = sorted(DRUG_SPARSE_QUERY_ARTIFACTS - observed_artifacts)
        extra = sorted(observed_artifacts - DRUG_SPARSE_QUERY_ARTIFACTS)
        raise FullModelContractError(
            "Drug sparse query artifact set mismatch: "
            f"missing={missing}, extra={extra}"
        )
    artifact_rows: dict[str, int] = {}
    for artifact_name in sorted(DRUG_SPARSE_QUERY_ARTIFACTS):
        artifact = artifacts[artifact_name]
        if not isinstance(artifact, Mapping):
            raise FullModelContractError(
                f"Drug sparse query artifact {artifact_name} is not a mapping"
            )
        _require_nonempty(
            artifact,
            ("path", "sha256", "rows"),
            f"Drug sparse query artifact {artifact_name}",
        )
        _require_sha256(
            artifact["sha256"],
            f"Drug sparse query artifact {artifact_name} SHA-256",
        )
        artifact_rows[artifact_name] = _require_positive_int(
            artifact["rows"], f"Drug sparse query artifact {artifact_name} rows"
        )

    required_true_flags = (
        "typed_absence_resolver",
        "absent_key_means_unavailable_not_zero",
        "available_keys_unique",
        "available_keys_subset_of_conceptual",
        "all_prediction_fold_masks_valid",
        "all_contributing_folds_same_fold_supported",
        "resolver_requires_expected_manifest_sha256",
        "all_five_folds_have_optimizer_updates",
        "code_hashes_unchanged",
        "input_hashes_unchanged",
        "release_ready",
    )
    for flag in required_true_flags:
        if lineage.get(flag) is not True:
            raise FullModelContractError(f"Successful Drug sparse release requires {flag}=true")
    required_false_flags = (
        "dense_candidate_table_materialized",
        "partial_not_publishable",
    )
    for flag in required_false_flags:
        if lineage.get(flag) is not False:
            raise FullModelContractError(f"Successful Drug sparse release requires {flag}=false")

    conceptual_rows = _require_positive_int(
        lineage["conceptual_candidate_rows"], "Drug conceptual_candidate_rows"
    )
    available_rows = _require_positive_int(
        lineage["available_rows"], "Drug available_rows"
    )
    unavailable_rows = lineage["unavailable_rows"]
    if not isinstance(unavailable_rows, int) or isinstance(unavailable_rows, bool):
        raise FullModelContractError("Drug unavailable_rows must be an integer")
    if conceptual_rows < available_rows:
        raise FullModelContractError(
            "Drug row counts require conceptual_candidate_rows >= available_rows >= 1"
        )
    if unavailable_rows != conceptual_rows - available_rows:
        raise FullModelContractError(
            "Drug unavailable_rows must equal conceptual_candidate_rows - available_rows"
        )
    if artifact_rows["available_predictions"] != available_rows:
        raise FullModelContractError(
            "Drug available_predictions artifact rows must equal available_rows"
        )
    _require_positive_int(lineage["optimizer_steps_total"], "Drug optimizer_steps_total")


def validate_module_lineage(module_id: str, lineage: Mapping[str, Any]) -> None:
    """Validate that one module was newly trained inside the V3.2 lineage."""

    if module_id not in MODULE_CONTRACTS:
        raise FullModelContractError(f"Unknown V3.2 module: {module_id}")
    contract = MODULE_CONTRACTS[module_id]
    _require_nonempty(
        lineage,
        (
            "module_id",
            "analysis_version",
            "training_run_id",
            "training_status",
            "initialization_policy",
            "folds",
            "seeds",
            "code_sha256",
            "config_sha256",
            "input_manifest_sha256",
            "checkpoint_manifest_sha256",
            "input_artifacts",
        ),
        f"module {module_id}",
    )
    if str(lineage["module_id"]) != module_id:
        raise FullModelContractError(f"Module ID drift for {module_id}")
    if not str(lineage["analysis_version"]).startswith(V32_ANALYSIS_PREFIX):
        raise FullModelContractError(f"{module_id} is not a V3.2 analysis")
    training_status = str(lineage["training_status"]).upper()
    if training_status not in {"SUCCESS", "AUDITED_UNAVAILABLE"}:
        raise FullModelContractError(
            f"{module_id} training status must be SUCCESS or AUDITED_UNAVAILABLE"
        )
    audited_unavailable = training_status == "AUDITED_UNAVAILABLE"
    if audited_unavailable and module_id == "exact_pathway":
        raise FullModelContractError("The exact-pathway core may not be audited-unavailable")
    if int(lineage["folds"]) != N_FOLDS:
        raise FullModelContractError(f"{module_id} requires exactly {N_FOLDS} folds")
    if not list(lineage["seeds"]):
        raise FullModelContractError(f"{module_id} has no recorded training seed")
    if lineage.get("old_checkpoint_loaded") is not False:
        raise FullModelContractError(f"{module_id} did not prove old checkpoints were excluded")
    if lineage.get("old_predictions_used_as_features") is not False:
        raise FullModelContractError(f"{module_id} did not prove old predictions were excluded")
    if lineage.get("old_rankings_used_as_outputs") is not False:
        raise FullModelContractError(f"{module_id} did not prove old rankings were excluded")
    if str(lineage["initialization_policy"]) != contract.core_policy:
        raise FullModelContractError(
            f"{module_id} initialization policy must be {contract.core_policy}"
        )
    if module_id == "exact_pathway":
        if lineage.get("trained_from_scratch") is not True:
            raise FullModelContractError("The exact-pathway core was not trained from scratch")
        _require_nonempty(
            lineage,
            (
                "ensemble_source_manifest_path",
                "ensemble_source_manifest_sha256",
                "ensemble_source_records",
                "ensemble_columns",
                "ensemble_formula",
            ),
            "exact-pathway five-fold ensemble sources",
        )
        manifest_hash = str(lineage["ensemble_source_manifest_sha256"])
        if len(manifest_hash) != 64 or any(
            char not in "0123456789abcdef" for char in manifest_hash.lower()
        ):
            raise FullModelContractError(
                "Exact-pathway ensemble source manifest hash is not SHA-256"
            )
        records = lineage["ensemble_source_records"]
        if (
            not isinstance(records, Sequence)
            or isinstance(records, (str, bytes))
            or len(records) != N_FOLDS
        ):
            raise FullModelContractError(
                "Exact-pathway ensemble requires exactly five source records"
            )
        expected_columns = [
            "association_membership_probability",
            "association_direction_probability",
            "l1_probability",
            "ridge_probability",
            "graph_residual",
            "graph_gate",
        ]
        if list(lineage["ensemble_columns"]) != expected_columns:
            raise FullModelContractError("Exact-pathway ensemble columns drifted")
        if lineage["ensemble_formula"] != "arithmetic_mean_of_five_patient_fold_predictions":
            raise FullModelContractError(
                "Exact-pathway ensemble formula is not the five-fold mean"
            )
        required_source_fields = (
            "patient_fold",
            "seed",
            "prediction_path",
            "prediction_sha256",
            "prediction_rows",
            "metrics_path",
            "metrics_sha256",
            "checkpoint_path",
            "checkpoint_sha256",
            "success_path",
            "success_sha256",
            "candidate_alignment",
            "checkpoint_cycle",
            "completed_cycles",
            "analysis_version",
        )
        for expected_fold, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise FullModelContractError(
                    "Exact-pathway ensemble source record is not a mapping"
                )
            _require_nonempty(
                record,
                required_source_fields,
                f"exact-pathway fold {expected_fold} source",
            )
            if int(record["patient_fold"]) != expected_fold:
                raise FullModelContractError(
                    "Exact-pathway ensemble fold order or identity drifted"
                )
            if not str(record["analysis_version"]).startswith(V32_ANALYSIS_PREFIX):
                raise FullModelContractError(
                    f"Exact-pathway fold {expected_fold} is not a V3.2 source"
                )
            if record["candidate_alignment"] != "FULL_ROW_EXACT":
                raise FullModelContractError(
                    f"Exact-pathway fold {expected_fold} is not FULL_ROW_EXACT"
                )
            if int(record["prediction_rows"]) <= 0:
                raise FullModelContractError(
                    f"Exact-pathway fold {expected_fold} has no predictions"
                )
            checkpoint_cycle = int(record["checkpoint_cycle"])
            completed_cycles = int(record["completed_cycles"])
            if checkpoint_cycle < 0 or checkpoint_cycle >= completed_cycles:
                raise FullModelContractError(
                    f"Exact-pathway fold {expected_fold} checkpoint cycle is outside training"
                )
            for hash_field in (
                "prediction_sha256",
                "metrics_sha256",
                "checkpoint_sha256",
                "success_sha256",
            ):
                value = str(record[hash_field])
                if len(value) != 64 or any(
                    char not in "0123456789abcdef" for char in value.lower()
                ):
                    raise FullModelContractError(
                        f"Exact-pathway fold {expected_fold} {hash_field} is not SHA-256"
                    )
            for flag in (
                "trained_from_random_initialization",
                "old_checkpoint_loaded",
                "old_predictions_used_as_features",
                "old_rankings_used_as_outputs",
            ):
                expected = flag == "trained_from_random_initialization"
                if record.get(flag) is not expected:
                    raise FullModelContractError(
                        f"Exact-pathway fold {expected_fold} has invalid provenance flag {flag}"
                    )
    else:
        statistical_status = str(lineage.get("statistical_training_status", "TRAINED")).upper()
        formally_unavailable = statistical_status.startswith("UNAVAILABLE_")
        if audited_unavailable:
            if lineage.get("private_head_trained_from_scratch") is not False:
                raise FullModelContractError(
                    f"{module_id} audited-unavailable lineage falsely attests a trained head"
                )
            if int(lineage.get("trained_folds", -1)) != 0:
                raise FullModelContractError(
                    f"{module_id} audited-unavailable lineage must have zero trained folds"
                )
            checkpoint_count = lineage.get("checkpoint_files", lineage.get("checkpoint_file_count", -1))
            if int(checkpoint_count) != 0:
                raise FullModelContractError(
                    f"{module_id} audited-unavailable lineage must have zero checkpoints"
                )
            if lineage.get("release_ready") is not False:
                raise FullModelContractError(
                    f"{module_id} audited-unavailable lineage must be release_ready=false"
                )
            if lineage.get("all_probabilities_null") is not True:
                raise FullModelContractError(
                    f"{module_id} audited-unavailable lineage must prove all probabilities are null"
                )
            if lineage.get("all_unavailable_rows_have_reason") is not True:
                raise FullModelContractError(
                    f"{module_id} audited-unavailable lineage lacks row-level failure reasons"
                )
        elif formally_unavailable:
            if lineage.get("private_head_trained_from_scratch") is not False:
                raise FullModelContractError(
                    f"{module_id} unavailable lineage falsely attests that its private head was trained"
                )
            if lineage.get("private_head_initialized_from_scratch") is not True:
                raise FullModelContractError(
                    f"{module_id} unavailable private head lacks a fresh initialization audit"
                )
            if int(lineage.get("private_head_optimizer_steps", -1)) != 0:
                raise FullModelContractError(
                    f"{module_id} unavailable private head must have zero optimizer steps"
                )
            if not str(lineage.get("failure_reason", "")).strip():
                raise FullModelContractError(
                    f"{module_id} unavailable lineage lacks a failure reason"
                )
        elif lineage.get("private_head_trained_from_scratch") is not True:
            raise FullModelContractError(f"{module_id} private head was not trained from scratch")
        if lineage.get("core_parameters_frozen") is not True:
            raise FullModelContractError(f"{module_id} must freeze the V3.2 core")
        _require_nonempty(
            lineage,
            ("v32_core_checkpoint_sha256", "core_parameters_before_sha256", "core_parameters_after_sha256"),
            f"module {module_id} frozen core",
        )
        if lineage["core_parameters_before_sha256"] != lineage["core_parameters_after_sha256"]:
            raise FullModelContractError(f"{module_id} changed frozen V3.2 core parameters")
        if module_id == "drug" and training_status == "SUCCESS":
            if lineage.get("formal_release_eligible") is False:
                if lineage.get("dense_development_only") is not True:
                    raise FullModelContractError(
                        "Non-releaseable Drug lineage must be dense_development_only=true"
                    )
                if lineage.get("release_ready") is not False:
                    raise FullModelContractError(
                        "Non-releaseable Drug lineage must be release_ready=false"
                    )
                if lineage.get("partial_not_publishable") is not True:
                    raise FullModelContractError(
                        "Non-releaseable Drug lineage must be partial_not_publishable=true"
                    )
            else:
                _validate_successful_drug_sparse_release(lineage)
        if module_id == "single_cell" and training_status == "SUCCESS":
            usability = lineage.get("core_embedding_usability")
            if (
                not isinstance(usability, Sequence)
                or isinstance(usability, (str, bytes))
                or len(usability) != N_FOLDS
            ):
                raise FullModelContractError(
                    "Successful single_cell lineage requires five core usability audits"
                )
            observed_folds: set[int] = set()
            unusable_lncrna = False
            for record in usability:
                if not isinstance(record, Mapping):
                    raise FullModelContractError(
                        "single_cell core usability record is not a mapping"
                    )
                observed_folds.add(int(record.get("single_cell_fold", -1)))
                lncrna = record.get("lncrna")
                pathway = record.get("pathway")
                if not isinstance(lncrna, Mapping) or not isinstance(pathway, Mapping):
                    raise FullModelContractError(
                        "single_cell core usability audit lacks lncRNA/pathway records"
                    )
                if pathway.get("usable_for_node_discrimination") is not True:
                    raise FullModelContractError(
                        "single_cell exact-pathway core embedding is not discriminative"
                    )
                if lncrna.get("usable_for_node_discrimination") is not True:
                    unusable_lncrna = True
                    if record.get("lncrna_fallback") != (
                        "MASK_CONSTANT_CORE_AND_USE_FRESH_SINGLE_CELL_LNCRNA_FEATURES"
                    ):
                        raise FullModelContractError(
                            "single_cell constant lncRNA core lacks the approved fallback"
                        )
            if observed_folds != set(range(N_FOLDS)):
                raise FullModelContractError(
                    "single_cell core usability audits do not cover folds 0..4"
                )
            if unusable_lncrna and (
                lineage.get("constant_lncrna_core_masked") is not True
                or lineage.get("lncrna_identity_feature_source")
                != "fresh_single_cell_detection_expression_specificity"
            ):
                raise FullModelContractError(
                    "single_cell constant lncRNA core was not masked with a fresh-data fallback"
                )

    for artifact in lineage["input_artifacts"]:
        if not isinstance(artifact, Mapping):
            raise FullModelContractError(f"{module_id} input artifact is not a mapping")
        _require_nonempty(artifact, ("path", "sha256", "artifact_kind"), f"{module_id} input artifact")
        kind = str(artifact["artifact_kind"])
        if kind in FORBIDDEN_HISTORICAL_RESULT_KINDS:
            raise FullModelContractError(f"{module_id} consumes forbidden old result: {kind}")
        if kind not in ALLOWED_INPUT_KINDS:
            raise FullModelContractError(f"{module_id} has unregistered input kind: {kind}")
        if kind == "v32_core_checkpoint" and module_id == "exact_pathway":
            raise FullModelContractError("The V3.2 core may not consume a parent checkpoint")

    if module_id == "single_cell":
        # Single-cell association tables are outcome-derived labels, not raw
        # assays.  This module-specific check is deliberately stricter than the
        # generic input gate, which must continue to allow raw response targets
        # for modules such as Drug.
        training_targets = [
            artifact
            for artifact in lineage["input_artifacts"]
            if str(artifact.get("use_role", "")) == "training_target"
        ]
        if not training_targets:
            raise FullModelContractError(
                "single_cell lineage lacks a declared V3.2 training target"
            )
        for artifact in training_targets:
            if artifact.get("outcome_derived") is not True:
                raise FullModelContractError(
                    "single_cell training target must be explicitly outcome_derived=true"
                )
            if (
                str(artifact.get("source_role", "")) != "training_label"
                or str(artifact.get("artifact_kind", "")) != "training_label"
            ):
                raise FullModelContractError(
                    "single_cell outcome-derived training target must be training_label, not raw_data"
                )
            generation = str(artifact.get("generation", "")).upper()
            if not any(marker in generation for marker in ("V3.2", "V3_2", "V3-2")):
                raise FullModelContractError(
                    "single_cell training label must be regenerated inside V3.2"
                )

        formal_datasets = lineage.get("formal_datasets", [])
        if (
            not isinstance(formal_datasets, Sequence)
            or isinstance(formal_datasets, (str, bytes))
        ):
            raise FullModelContractError(
                "single_cell formal_datasets must be an explicit sequence"
            )
        trained_folds = int(lineage.get("trained_folds", -1))
        checkpoint_count = int(
            lineage.get("checkpoint_files", lineage.get("checkpoint_file_count", -1))
        )
        release_ready = lineage.get("release_ready") is True
        if training_status == "SUCCESS":
            if not formal_datasets:
                raise FullModelContractError(
                    "single_cell cannot report SUCCESS with formal_datasets=[]"
                )
            if trained_folds != N_FOLDS:
                raise FullModelContractError(
                    "single_cell SUCCESS requires five trained folds"
                )
            if checkpoint_count != N_FOLDS:
                raise FullModelContractError(
                    "single_cell SUCCESS requires five checkpoints"
                )
        if release_ready:
            if training_status != "SUCCESS":
                raise FullModelContractError(
                    "single_cell release cannot unlock without SUCCESS"
                )
            if not formal_datasets or trained_folds != N_FOLDS:
                raise FullModelContractError(
                    "single_cell release cannot unlock with no formal datasets or zero/incomplete folds"
                )
            if int(lineage.get("available_rows", 0)) <= 0:
                raise FullModelContractError(
                    "single_cell release cannot unlock without available rows"
                )

    core_inputs = [
        artifact
        for artifact in lineage["input_artifacts"]
        if str(artifact["artifact_kind"]) == "v32_core_checkpoint"
    ]
    if module_id != "exact_pathway":
        if len(core_inputs) != 1:
            raise FullModelContractError(
                f"{module_id} must consume exactly one current V3.2 core checkpoint"
            )
        if core_inputs[0]["sha256"] != lineage["v32_core_checkpoint_sha256"]:
            raise FullModelContractError(
                f"{module_id} parent core checkpoint hash does not match its input artifact"
            )


def validate_full_release_lineage(release: Mapping[str, Any]) -> None:
    """Require every historical capability to have a fresh V3.2 training lineage."""

    _require_nonempty(release, ("contract_version", "analysis_version", "modules"), "full release")
    if release["contract_version"] != FULL_CONTRACT_VERSION:
        raise FullModelContractError("Full V3.2 contract version mismatch")
    if not str(release["analysis_version"]).startswith(V32_ANALYSIS_PREFIX):
        raise FullModelContractError("Full release analysis_version is not V3.2")
    modules = release["modules"]
    if not isinstance(modules, Mapping):
        raise FullModelContractError("Full release modules must be a mapping")
    missing = sorted(set(REQUIRED_MODULES) - set(modules))
    extra = sorted(set(modules) - set(REQUIRED_MODULES))
    if missing or extra:
        raise FullModelContractError(f"Full release module mismatch: missing={missing}, extra={extra}")
    for module_id in REQUIRED_MODULES:
        validate_module_lineage(module_id, modules[module_id])
        if modules[module_id].get("formal_release_eligible") is False:
            raise FullModelContractError(
                f"Full release includes non-releaseable development module: {module_id}"
            )
    if (
        release.get("all_non_null_predictions_newly_trained_v32") is not True
        and release.get("all_results_newly_trained_v32") is not True
    ):
        raise FullModelContractError(
            "Release lacks the all-non-null-predictions-newly-trained attestation"
        )
    if release.get("historical_results_present_in_public_outputs") is not False:
        raise FullModelContractError("Historical model results remain in public outputs")
    unavailable_modules = [
        module_id
        for module_id, lineage in modules.items()
        if str(lineage.get("training_status", "")).upper() == "AUDITED_UNAVAILABLE"
        or str(lineage.get("statistical_training_status", "")).upper().startswith("UNAVAILABLE_")
    ]
    if unavailable_modules and release.get("release_ready") is not False:
        raise FullModelContractError(
            "A release with audited-unavailable modules must be release_ready=false: "
            f"{sorted(unavailable_modules)}"
        )
    if release.get("all_required_modules_statistically_trained") is True and unavailable_modules:
        raise FullModelContractError(
            "Release falsely attests all modules trained despite audited-unavailable modules"
        )


def validate_public_module_frame(module_id: str, frame: pd.DataFrame) -> None:
    """Validate a public prediction table without trusting its filename."""

    if module_id not in MODULE_CONTRACTS:
        raise FullModelContractError(f"Unknown V3.2 module: {module_id}")
    contract = MODULE_CONTRACTS[module_id]
    required = set(contract.target_keys) | set(contract.probability_columns) | {
        "analysis_version",
        "training_run_id",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise FullModelContractError(f"{module_id} public table lacks columns: {missing}")
    forbidden = sorted(FORBIDDEN_PUBLIC_COLUMNS.intersection(frame.columns))
    if forbidden:
        raise FullModelContractError(f"{module_id} public table leaks labels: {forbidden}")
    historical = sorted(
        column
        for column in frame.columns
        if any(token in column.lower() for token in HISTORICAL_RESULT_COLUMN_TOKENS)
    )
    if historical:
        raise FullModelContractError(
            f"{module_id} public table contains historical result columns: {historical}"
        )
    if frame.empty:
        raise FullModelContractError(f"{module_id} public prediction table is empty")
    if frame[list(contract.target_keys)].astype(str).duplicated().any():
        raise FullModelContractError(f"{module_id} public target keys are duplicated")
    versions = frame["analysis_version"].dropna().astype(str)
    if versions.empty or not versions.str.startswith(V32_ANALYSIS_PREFIX).all():
        raise FullModelContractError(f"{module_id} public rows are not uniformly V3.2")
    availability_column = contract.availability_column
    reason_column = contract.failure_reason_column
    if availability_column is not None and availability_column in frame:
        availability = frame[availability_column].fillna(False).astype(bool)
        unavailable = ~availability
        if unavailable.any():
            if reason_column is None or reason_column not in frame:
                raise FullModelContractError(
                    f"{module_id} unavailable rows lack {reason_column or 'failure_reason'}"
                )
            reasons = frame.loc[unavailable, reason_column].fillna("").astype(str).str.strip()
            if reasons.eq("").any():
                raise FullModelContractError(
                    f"{module_id} unavailable rows lack a non-empty failure_reason"
                )
    else:
        availability = pd.Series(True, index=frame.index, dtype=bool)
        unavailable = ~availability
    for column in contract.probability_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        available_values = values.loc[availability]
        if available_values.isna().any() or not np.isfinite(
            available_values.to_numpy(float)
        ).all():
            raise FullModelContractError(f"{module_id}.{column} is not finite when available")
        if not available_values.between(0.0, 1.0).all():
            raise FullModelContractError(f"{module_id}.{column} is outside [0, 1]")
        if unavailable.any() and values.loc[unavailable].notna().any():
            raise FullModelContractError(
                f"{module_id}.{column} must be null when unavailable"
            )
    if module_id != "exact_pathway" and "changes_primary_ranking" in frame:
        if frame["changes_primary_ranking"].fillna(False).astype(bool).any():
            raise FullModelContractError(f"{module_id} illegally changes primary ranking")
