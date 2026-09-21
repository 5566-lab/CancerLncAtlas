from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from cc_hhgt.v32.full_model_contract import MODULE_CONTRACTS
from cc_hhgt.v32.mixed_query import (
    MixedExactPathwayQuery,
    MixedQueryAssetError,
    MixedQueryInputError,
    hypergeometric_overrepresentation,
)
from cc_hhgt.v32.release_registry import (
    REGISTRY_SCHEMA_VERSION,
    ReleaseRegistryError,
    artifact_sha256,
    load_release_registry,
)
from cc_hhgt.v32.staging_query import StagingQueryAssetError, V32StagingQuery
from website.backend import v32_staging_api
from website.backend.v32_staging_api import create_staging_app


VERSION = "CancerLncAtlas_V3.2.test"
RUN_ID = "V32_FRESH_TEST_RUN"


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _exact_lineage() -> dict:
    return {
        "module_id": "exact_pathway",
        "analysis_version": VERSION,
        "training_run_id": RUN_ID,
        "training_status": "SUCCESS",
        "initialization_policy": "TRAIN_FROM_RANDOM_INITIALIZATION",
        "folds": 5,
        "seeds": [17],
        "code_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "input_manifest_sha256": "c" * 64,
        "checkpoint_manifest_sha256": "d" * 64,
        "input_artifacts": [
            {"path": "static/pathway_membership.parquet", "sha256": "e" * 64, "artifact_kind": "annotation"}
        ],
        "old_checkpoint_loaded": False,
        "old_predictions_used_as_features": False,
        "old_rankings_used_as_outputs": False,
        "trained_from_scratch": True,
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
                "seed": 17,
                "prediction_path": f"fold_{fold}.parquet",
                "prediction_sha256": "6" * 64,
                "prediction_rows": 12,
                "metrics_path": f"fold_{fold}_metrics.json",
                "metrics_sha256": "7" * 64,
                "checkpoint_path": f"fold_{fold}.pt",
                "checkpoint_sha256": "8" * 64,
                "success_path": f"fold_{fold}_SUCCESS.json",
                "success_sha256": "9" * 64,
                "candidate_alignment": "FULL_ROW_EXACT",
                "checkpoint_cycle": 9,
                "completed_cycles": 10,
                "analysis_version": VERSION,
                "trained_from_random_initialization": True,
                "old_checkpoint_loaded": False,
                "old_predictions_used_as_features": False,
                "old_rankings_used_as_outputs": False,
            }
            for fold in range(5)
        ],
    }


def _unavailable_lineage(module_id: str) -> dict:
    core_hash = "f" * 64
    input_artifacts = [
        {
            "path": "v32/core/CHECKPOINT_MANIFEST.json",
            "sha256": core_hash,
            "artifact_kind": "v32_core_checkpoint",
        }
    ]
    if module_id == "single_cell":
        input_artifacts.append(
            {
                "path": "v32/single_cell/TRAINING_LABELS.parquet",
                "sha256": "1" * 64,
                "artifact_kind": "training_label",
                "generation": "V3.2_TEST_FIXTURE",
                "source_role": "training_label",
                "outcome_derived": True,
                "use_role": "training_target",
            }
        )
    return {
        "module_id": module_id,
        "analysis_version": VERSION,
        "training_run_id": f"V32_{module_id.upper()}_UNAVAILABLE_TEST",
        "training_status": "AUDITED_UNAVAILABLE",
        "initialization_policy": MODULE_CONTRACTS[module_id].core_policy,
        "folds": 5,
        "seeds": [17, 18, 19, 20, 21],
        "code_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "input_manifest_sha256": "c" * 64,
        "checkpoint_manifest_sha256": "d" * 64,
        "input_artifacts": input_artifacts,
        "old_checkpoint_loaded": False,
        "old_predictions_used_as_features": False,
        "old_rankings_used_as_outputs": False,
        "private_head_trained_from_scratch": False,
        "trained_folds": 0,
        "checkpoint_files": 0,
        "release_ready": False,
        "all_probabilities_null": True,
        "all_unavailable_rows_have_reason": True,
        "core_parameters_frozen": True,
        "v32_core_checkpoint_sha256": core_hash,
        "core_parameters_before_sha256": core_hash,
        "core_parameters_after_sha256": core_hash,
    }


def _success_aux_lineage(module_id: str) -> dict:
    core_hash = "f" * 64
    return {
        "module_id": module_id,
        "analysis_version": VERSION,
        "training_run_id": f"V32_{module_id.upper()}_FRESH_TEST",
        "training_status": "SUCCESS",
        "initialization_policy": MODULE_CONTRACTS[module_id].core_policy,
        "folds": 5,
        "seeds": [17],
        "code_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "input_manifest_sha256": "c" * 64,
        "checkpoint_manifest_sha256": "d" * 64,
        "input_artifacts": [
            {
                "path": "v32/core/CHECKPOINT_MANIFEST.json",
                "sha256": core_hash,
                "artifact_kind": "v32_core_checkpoint",
            }
        ],
        "old_checkpoint_loaded": False,
        "old_predictions_used_as_features": False,
        "old_rankings_used_as_outputs": False,
        "private_head_trained_from_scratch": True,
        "core_parameters_frozen": True,
        "v32_core_checkpoint_sha256": core_hash,
        "core_parameters_before_sha256": core_hash,
        "core_parameters_after_sha256": core_hash,
    }


def _build_registry(root: Path, *, with_mixed_manifest: bool = True) -> Path:
    identifier = pd.DataFrame(
        [
            {
                "identifier": "ENSG00000100001",
                "canonical_id": "LNC:ENSG00000100001",
                "entity_type": "lncRNA",
                "gene_symbol": "LINC-ONE",
                "aliases": "LINC1",
            },
            {
                "identifier": "ENSG00000100002",
                "canonical_id": "LNC:ENSG00000100002",
                "entity_type": "lncRNA",
                "gene_symbol": "LINC-TWO",
                "aliases": "LINC2",
            },
            {
                "identifier": "ENSG00000200001",
                "canonical_id": "ENSG00000200001",
                "entity_type": "protein_coding_gene",
                "gene_symbol": "TP53",
                "aliases": "P53",
            },
            {
                "identifier": "ENSG00000200002",
                "canonical_id": "ENSG00000200002",
                "entity_type": "protein_coding_gene",
                "gene_symbol": "EGFR",
                "aliases": "ERBB1",
            },
            {
                "identifier": "ENSG00000200003",
                "canonical_id": "ENSG00000200003",
                "entity_type": "protein_coding_gene",
                "gene_symbol": "KRAS",
                "aliases": "K-RAS",
            },
            {
                "identifier": "ENSG00000200004",
                "canonical_id": "ENSG00000200004",
                "entity_type": "protein_coding_gene",
                "gene_symbol": "BRAF",
                "aliases": "B-RAF",
            },
        ]
    )
    membership = pd.DataFrame(
        [
            {"pathway_id": "MSIGDB:REACTOME:P1", "gene_id": "ENSG00000200001"},
            {"pathway_id": "MSIGDB:REACTOME:P1", "gene_id": "ENSG00000200002"},
            {"pathway_id": "MSIGDB:REACTOME:P2", "gene_id": "ENSG00000200003"},
            {"pathway_id": "MSIGDB:REACTOME:P2", "gene_id": "ENSG00000200004"},
            {"pathway_id": "MSIGDB:REACTOME:P3", "gene_id": "ENSG00000200001"},
            {"pathway_id": "MSIGDB:REACTOME:P3", "gene_id": "ENSG00000200003"},
        ]
    )
    association_rows = []
    values = {
        "LUAD": {
            "LNC:ENSG00000100001": [0.90, 0.20, 0.40],
            "LNC:ENSG00000100002": [0.60, 0.30, 0.50],
        },
        "BRCA": {
            "LNC:ENSG00000100001": [0.70, 0.10, 0.30],
            "LNC:ENSG00000100002": [0.40, 0.20, 0.60],
        },
    }
    pathways = ["MSIGDB:REACTOME:P1", "MSIGDB:REACTOME:P2", "MSIGDB:REACTOME:P3"]
    for cancer, lnc_values in values.items():
        for lnc_id, probabilities in lnc_values.items():
            for pathway_id, probability in zip(pathways, probabilities, strict=True):
                association_rows.append(
                    {
                        "cancer_id": cancer,
                        "lncrna_id": lnc_id,
                        "pathway_id": pathway_id,
                        "association_membership_probability": probability,
                        "analysis_version": VERSION,
                        "training_run_id": RUN_ID,
                        "pathway_target_level": "exact_pathway",
                        "availability": True,
                        "eligible_for_mixed_query": True,
                    }
                )
    association = pd.DataFrame(association_rows)
    metadata = pd.DataFrame(
        {
            "pathway_id": pathways,
            "pathway_name": ["Pathway one", "Pathway two", "Pathway three"],
            "pathway_source": ["REACTOME"] * 3,
        }
    )

    assets = root / "assets"
    assets.mkdir(parents=True)
    identifier.to_parquet(assets / "identifier.parquet", index=False)
    membership.to_parquet(assets / "membership.parquet", index=False)
    association.to_parquet(assets / "association.parquet", index=False)
    metadata.to_parquet(assets / "metadata.parquet", index=False)

    lineage_path = root / "lineage" / "exact_pathway.json"
    exact_lineage = _exact_lineage()
    exact_lineage["prediction_sha256"] = artifact_sha256(assets / "association.parquet")
    _write_json(lineage_path, exact_lineage)
    modules: dict[str, dict] = {
        "exact_pathway": {
            "status": "SUCCESS_NEWLY_TRAINED",
            "analysis_version": VERSION,
            "training_run_id": RUN_ID,
            "lineage_path": str(lineage_path.relative_to(root)),
            "lineage_sha256": artifact_sha256(lineage_path),
            "new_training_attestation": True,
            "old_checkpoint_loaded": False,
            "old_predictions_used_as_features": False,
            "old_rankings_used_as_outputs": False,
        }
    }
    for module_id in MODULE_CONTRACTS:
        if module_id == "exact_pathway":
            continue
        if module_id == "clinical":
            clinical_path = root / "lineage" / "clinical.json"
            clinical_lineage = _success_aux_lineage("clinical")
            _write_json(clinical_path, clinical_lineage)
            modules[module_id] = {
                "status": "SUCCESS_NEWLY_TRAINED",
                "analysis_version": VERSION,
                "training_run_id": clinical_lineage["training_run_id"],
                "lineage_path": str(clinical_path.relative_to(root)),
                "lineage_sha256": artifact_sha256(clinical_path),
                "new_training_attestation": True,
                "old_checkpoint_loaded": False,
                "old_predictions_used_as_features": False,
                "old_rankings_used_as_outputs": False,
            }
            continue
        audit_path = root / "lineage" / f"{module_id}.json"
        reason = f"NO_{module_id.upper()}_STAGING_FIXTURE"
        _write_json(audit_path, _unavailable_lineage(module_id))
        modules[module_id] = {
            "status": "AUDITED_UNAVAILABLE",
            "analysis_version": VERSION,
            "reason_code": reason,
            "lineage_path": str(audit_path.relative_to(root)),
            "lineage_sha256": artifact_sha256(audit_path),
        }
    website_artifacts = []
    for role, filename, kind in (
        ("mixed_query_identifier_map", "identifier.parquet", "static_annotation"),
        ("mixed_query_exact_pathway_membership", "membership.parquet", "static_annotation"),
        (
            "mixed_query_v32_lnc_exact_association",
            "association.parquet",
            "v32_public_prediction",
        ),
        ("mixed_query_pathway_metadata", "metadata.parquet", "v32_public_metadata"),
    ):
        path = assets / filename
        website_artifacts.append(
            {
                "role": role,
                "path": str(path.relative_to(root)),
                "sha256": artifact_sha256(path),
                "artifact_kind": kind,
                "generation": VERSION,
                "source_module": "exact_pathway",
            }
        )
    registry = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "release_id": "V32_STAGING_TEST",
        "analysis_version": VERSION,
        "environment": "staging",
        "production_deployed": False,
        "release_ready": False,
        "all_non_null_predictions_newly_trained_v32": True,
        "all_published_results_generated_in_v32": True,
        "modules": modules,
        "website_artifacts": website_artifacts,
        "capabilities": {
            "mixed_exact_pathway_query": {
                "enabled": True,
                "endpoint": "/v3.2-staging/enrichment/mixed-exact-pathway",
            },
            "clinical": {
                "patient_risk": {"status": "NOT_REGISTERED", "artifact_role": None},
                "entity_association": {"status": "NOT_REGISTERED", "enabled": False},
            },
        },
    }
    registry_path = root / "RELEASE_REGISTRY.json"
    _write_json(registry_path, registry)
    if with_mixed_manifest:
        _enable_mixed_asset_manifest(root, registry_path)
    return registry_path


def _rewrite_registry(path: Path, mutate) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    _write_json(path, value)


def _enable_mixed_asset_manifest(root: Path, registry_path: Path) -> Path:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    exact_module = registry["modules"]["exact_pathway"]
    exact_lineage = json.loads((root / exact_module["lineage_path"]).read_text(encoding="utf-8"))
    declared = {
        artifact["role"]: {
            "filename": Path(artifact["path"]).name,
            "sha256": artifact["sha256"],
        }
        for artifact in registry["website_artifacts"]
        if artifact["role"].startswith("mixed_query_")
    }
    manifest_path = root / "assets" / "ASSET_MANIFEST.json"
    _write_json(
        manifest_path,
        {
            "status": "SUCCESS",
            "analysis_version": VERSION,
            "training_run_id": RUN_ID,
            "pathway_target_level": "exact_pathway",
            "source_exact_lineage_sha256": exact_module["lineage_sha256"],
            "source_exact_association_sha256": exact_lineage["prediction_sha256"],
            "historical_predictions_used": False,
            "historical_rankings_used": False,
            "historical_checkpoints_used": False,
            "static_annotation_only_no_predictions_or_rankings": True,
            "pathway_family_broadcast": False,
            "membership_restricted_to_current_v32_exact_pathways": True,
            "key_coverage": {
                "format": "CANCERLNCATLAS_V32_MIXED_QUERY_KEY_COVERAGE_V1",
                "scope": "ELIGIBLE_CANCER_LNCRNA_X_EXACT_PATHWAY_CARTESIAN",
                "eligible_pair_authority_sha256": exact_lineage["prediction_sha256"],
                "eligible_pair_count": 4,
                "pathway_count": 3,
                "expected_key_count": 12,
                "observed_key_count": 12,
                "missing_key_count": 0,
                "duplicate_key_count": 0,
                "complete_key_coverage": True,
                "unscored_key_encoding": "NULL_WITH_TYPED_NOT_EVALUATED_REASON",
            },
            "artifacts": declared,
        },
    )

    def mutate(value: dict) -> None:
        value["website_artifacts"].append(
            {
                "role": "mixed_query_asset_manifest",
                "path": str(manifest_path.relative_to(root)),
                "sha256": artifact_sha256(manifest_path),
                "artifact_kind": "v32_public_metadata",
                "generation": VERSION,
                "source_module": "exact_pathway",
            }
        )

    _rewrite_registry(registry_path, mutate)
    return manifest_path


def _refresh_mixed_asset_bindings(
    root: Path,
    registry_path: Path,
    *roles: str,
) -> None:
    """Rebind intentionally mutated test assets through the complete hash chain."""

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registered = {item["role"]: item for item in registry["website_artifacts"]}
    manifest_entry = registered["mixed_query_asset_manifest"]
    manifest_path = root / manifest_entry["path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for role in roles:
        manifest["artifacts"][role]["sha256"] = registered[role]["sha256"]

    association_role = "mixed_query_v32_lnc_exact_association"
    if association_role in roles:
        exact_module = registry["modules"]["exact_pathway"]
        lineage_path = root / exact_module["lineage_path"]
        lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
        association_sha = registered[association_role]["sha256"]
        lineage["prediction_sha256"] = association_sha
        _write_json(lineage_path, lineage)
        exact_module["lineage_sha256"] = artifact_sha256(lineage_path)
        manifest["source_exact_lineage_sha256"] = exact_module["lineage_sha256"]
        manifest["source_exact_association_sha256"] = association_sha
        manifest["key_coverage"]["eligible_pair_authority_sha256"] = association_sha

    _write_json(manifest_path, manifest)
    manifest_entry["sha256"] = artifact_sha256(manifest_path)
    _write_json(registry_path, registry)


def _enable_clinical_entity_extension(root: Path, registry_path: Path) -> None:
    entity_path = root / "assets" / "clinical_entity.parquet"
    pd.DataFrame(
        [
            {
                "cancer_id": "LUAD",
                "subject_type": "lncRNA",
                "subject_id": "LNC:ENSG00000100001",
                "clinical_endpoint": "OS",
                "clinical_relevance_score": 0.91,
                "availability": True,
                "failure_reason": None,
                "analysis_version": "CancerLncAtlas_V3.2_CLINICAL_ENTITY_ASSOCIATION",
            }
        ]
    ).to_parquet(entity_path, index=False)
    lineage_path = root / "lineage" / "clinical_entity.json"
    _write_json(
        lineage_path,
        {
            "analysis_version": "CancerLncAtlas_V3.2_CLINICAL_ENTITY_ASSOCIATION",
            "module_id": "clinical_entity_association_extension",
            "training_status": "SUCCESS",
            "old_checkpoint_loaded": False,
            "old_predictions_used_as_features": False,
            "old_rankings_used_as_outputs": False,
            "fresh_statistical_calculation": True,
            "summary_sha256": artifact_sha256(entity_path),
        },
    )
    validation_path = root / "lineage" / "clinical_entity_validation.json"
    _write_json(
        validation_path,
        {
            "status": "PASS",
            "all_non_null_results_newly_computed_v32": True,
            "historical_results_used": False,
            "changes_primary_ranking": False,
            "summary_sha256": artifact_sha256(entity_path),
            "lineage_sha256": artifact_sha256(lineage_path),
        },
    )

    def mutate(value: dict) -> None:
        for role, artifact_path, kind in (
            ("clinical_entity_association_public", entity_path, "v32_public_statistical_result"),
            ("clinical_entity_association_lineage", lineage_path, "v32_public_metadata"),
            ("clinical_entity_association_validation", validation_path, "v32_public_metadata"),
        ):
            value["website_artifacts"].append(
                {
                    "role": role,
                    "path": str(artifact_path.relative_to(root)),
                    "sha256": artifact_sha256(artifact_path),
                    "artifact_kind": kind,
                    "generation": VERSION,
                    "source_module": "clinical",
                }
            )
        value["capabilities"]["clinical"]["entity_association"] = {
            "status": "STAGING_ARTIFACT_REGISTERED",
            "enabled": True,
            "artifact_role": "clinical_entity_association_public",
            "lineage_role": "clinical_entity_association_lineage",
            "validation_role": "clinical_entity_association_validation",
        }

    _rewrite_registry(registry_path, mutate)


def test_registry_requires_all_eight_modules_and_current_hashes(tmp_path: Path) -> None:
    path = _build_registry(tmp_path)
    registry = load_release_registry(path)
    assert set(registry.manifest["modules"]) == set(MODULE_CONTRACTS)
    assert registry.manifest["modules"]["exact_pathway"]["status"] == "SUCCESS_NEWLY_TRAINED"
    assert len(registry.artifact_hashes) == 5
    assert registry.manifest["production_deployed"] is False


def test_registry_rejects_hash_tampering(tmp_path: Path) -> None:
    path = _build_registry(tmp_path)
    frame = pd.read_parquet(tmp_path / "assets" / "association.parquet")
    frame.loc[0, "association_membership_probability"] = 0.01
    frame.to_parquet(tmp_path / "assets" / "association.parquet", index=False)
    with pytest.raises(ReleaseRegistryError, match="hash mismatch"):
        load_release_registry(path)


def test_registry_rejects_old_prediction_or_ranking_path(tmp_path: Path) -> None:
    path = _build_registry(tmp_path)
    old_dir = tmp_path / "v3_1" / "predictions"
    old_dir.mkdir(parents=True)
    old_path = old_dir / "ranked.parquet"
    pd.read_parquet(tmp_path / "assets" / "association.parquet").to_parquet(old_path, index=False)

    def mutate(value: dict) -> None:
        artifact = next(
            item
            for item in value["website_artifacts"]
            if item["role"] == "mixed_query_v32_lnc_exact_association"
        )
        artifact["path"] = str(old_path.relative_to(tmp_path))
        artifact["sha256"] = artifact_sha256(old_path)

    _rewrite_registry(path, mutate)
    with pytest.raises(ReleaseRegistryError, match="old checkpoint/prediction/ranking"):
        load_release_registry(path)


def test_registry_rejects_success_status_when_old_checkpoint_was_loaded(tmp_path: Path) -> None:
    path = _build_registry(tmp_path)

    def mutate(value: dict) -> None:
        value["modules"]["exact_pathway"]["old_checkpoint_loaded"] = True

    _rewrite_registry(path, mutate)
    with pytest.raises(ReleaseRegistryError, match="historical result reuse"):
        load_release_registry(path)


def test_registry_with_unavailable_modules_cannot_be_release_ready(tmp_path: Path) -> None:
    path = _build_registry(tmp_path)

    def mutate(value: dict) -> None:
        value["release_ready"] = True

    _rewrite_registry(path, mutate)
    with pytest.raises(ReleaseRegistryError, match="must have release_ready=false"):
        load_release_registry(path)


def test_registry_requires_non_null_v32_prediction_attestation(tmp_path: Path) -> None:
    path = _build_registry(tmp_path)

    def mutate(value: dict) -> None:
        value["all_non_null_predictions_newly_trained_v32"] = False

    _rewrite_registry(path, mutate)
    with pytest.raises(ReleaseRegistryError, match="every non-null prediction"):
        load_release_registry(path)


def test_registry_requires_all_published_results_generated_in_v32(tmp_path: Path) -> None:
    path = _build_registry(tmp_path)

    def mutate(value: dict) -> None:
        value["all_published_results_generated_in_v32"] = False

    _rewrite_registry(path, mutate)
    with pytest.raises(ReleaseRegistryError, match="every published result was generated in V3.2"):
        load_release_registry(path)


def test_mixed_asset_manifest_must_bind_registered_exact_lineage(tmp_path: Path) -> None:
    path = _build_registry(tmp_path, with_mixed_manifest=False)
    manifest_path = _enable_mixed_asset_manifest(tmp_path, path)
    load_release_registry(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_exact_lineage_sha256"] = "0" * 64
    _write_json(manifest_path, manifest)

    def mutate(value: dict) -> None:
        next(
            item for item in value["website_artifacts"]
            if item["role"] == "mixed_query_asset_manifest"
        )["sha256"] = artifact_sha256(manifest_path)

    _rewrite_registry(path, mutate)
    with pytest.raises(ReleaseRegistryError, match="not bound to the registered exact-pathway lineage"):
        load_release_registry(path)


def test_clinical_entity_statistical_extension_uses_dedicated_contract(tmp_path: Path) -> None:
    path = _build_registry(tmp_path)
    _enable_clinical_entity_extension(tmp_path, path)
    registry = load_release_registry(path)
    assert registry.manifest["capabilities"]["clinical"]["entity_association"]["enabled"] is True


def test_clinical_entity_result_hash_cannot_drift_from_validation(tmp_path: Path) -> None:
    path = _build_registry(tmp_path)
    _enable_clinical_entity_extension(tmp_path, path)
    entity_path = tmp_path / "assets" / "clinical_entity.parquet"
    frame = pd.read_parquet(entity_path)
    frame.loc[0, "clinical_relevance_score"] = 0.01
    frame.to_parquet(entity_path, index=False)

    def mutate(value: dict) -> None:
        artifact = next(
            item
            for item in value["website_artifacts"]
            if item["role"] == "clinical_entity_association_public"
        )
        artifact["sha256"] = artifact_sha256(entity_path)

    _rewrite_registry(path, mutate)
    with pytest.raises(ReleaseRegistryError, match="differs from validation"):
        load_release_registry(path)


def test_clinical_entity_old_result_path_is_rejected(tmp_path: Path) -> None:
    path = _build_registry(tmp_path)
    _enable_clinical_entity_extension(tmp_path, path)
    old_path = tmp_path / "v3_0" / "clinical_results" / "entity.parquet"
    old_path.parent.mkdir(parents=True)
    pd.read_parquet(tmp_path / "assets" / "clinical_entity.parquet").to_parquet(old_path, index=False)

    def mutate(value: dict) -> None:
        artifact = next(
            item
            for item in value["website_artifacts"]
            if item["role"] == "clinical_entity_association_public"
        )
        artifact["path"] = str(old_path.relative_to(tmp_path))
        artifact["sha256"] = artifact_sha256(old_path)

    _rewrite_registry(path, mutate)
    with pytest.raises(ReleaseRegistryError, match="old checkpoint/prediction/ranking"):
        load_release_registry(path)


def test_mixed_query_keeps_ora_and_v32_lnc_channels_separate(tmp_path: Path) -> None:
    engine = MixedExactPathwayQuery.from_registry(_build_registry(tmp_path))
    value = engine.query(
        ["LNC:ENSG00000100001", "TP53", "ENSG00000200002"],
        cancer_id="luad",
        top_k=3,
    )
    assert value["query_scope"] == "LUAD"
    assert value["results"][0]["pathway_id"] == "MSIGDB:REACTOME:P1"
    first = value["results"][0]
    assert first["protein_overlap_count"] == 2
    assert first["lncrna_association_probability_mean"] == pytest.approx(0.9)
    assert first["evidence_channels"] == "protein_ora+v32_lncrna_model"
    assert 0 <= first["combined_evidence_score"] <= 1
    assert value["semantics"]["native_kegg_lncrna_annotation"] is False
    assert value["semantics"]["pathway_family_score_broadcast"] is False
    assert "pathway_family_id" not in first


def test_lnc_symbol_hgnc_symbol_and_pan_cancer_mean_are_supported(tmp_path: Path) -> None:
    engine = MixedExactPathwayQuery.from_registry(_build_registry(tmp_path))
    value = engine.query(["LINC-ONE", "EGFR"], top_k=3)
    first = next(row for row in value["results"] if row["pathway_id"] == "MSIGDB:REACTOME:P1")
    assert value["query_scope"] == "PAN_CANCER_MEAN"
    assert first["lncrna_association_probability_mean"] == pytest.approx(0.8)
    assert value["classification"]["lncRNA"][0]["canonical_id"] == "LNC:ENSG00000100001"
    assert value["classification"]["protein_coding_gene"][0]["canonical_id"] == "ENSG00000200002"


def test_ambiguous_symbol_is_not_silently_typed(tmp_path: Path) -> None:
    path = _build_registry(tmp_path)
    identifier_path = tmp_path / "assets" / "identifier.parquet"
    frame = pd.read_parquet(identifier_path)
    frame.loc[frame.entity_type.eq("protein_coding_gene").idxmax(), "gene_symbol"] = "LINC-ONE"
    frame.to_parquet(identifier_path, index=False)

    def mutate(value: dict) -> None:
        artifact = next(
            item for item in value["website_artifacts"] if item["role"] == "mixed_query_identifier_map"
        )
        artifact["sha256"] = artifact_sha256(identifier_path)

    _rewrite_registry(path, mutate)
    _refresh_mixed_asset_bindings(tmp_path, path, "mixed_query_identifier_map")
    engine = MixedExactPathwayQuery.from_registry(path)
    with pytest.raises(MixedQueryInputError, match="No uniquely mapped"):
        engine.query(["LINC-ONE"])


def test_every_predicted_lncrna_must_be_reachable_through_identifier_map(
    tmp_path: Path,
) -> None:
    path = _build_registry(tmp_path)
    identifier_path = tmp_path / "assets" / "identifier.parquet"
    frame = pd.read_parquet(identifier_path)
    frame = frame.loc[
        ~frame.canonical_id.eq("LNC:ENSG00000100001")
    ].copy()
    frame.to_parquet(identifier_path, index=False)

    def mutate(value: dict) -> None:
        artifact = next(
            item
            for item in value["website_artifacts"]
            if item["role"] == "mixed_query_identifier_map"
        )
        artifact["sha256"] = artifact_sha256(identifier_path)

    _rewrite_registry(path, mutate)
    _refresh_mixed_asset_bindings(tmp_path, path, "mixed_query_identifier_map")
    with pytest.raises(MixedQueryAssetError, match="cannot be reached"):
        MixedExactPathwayQuery.from_registry(path)


def test_protein_ora_membership_rejects_lncrna_identifiers(tmp_path: Path) -> None:
    path = _build_registry(tmp_path)
    membership_path = tmp_path / "assets" / "membership.parquet"
    frame = pd.read_parquet(membership_path)
    frame.loc[0, "gene_id"] = "ENSG00000100001"
    frame.to_parquet(membership_path, index=False)

    def mutate(value: dict) -> None:
        artifact = next(
            item
            for item in value["website_artifacts"]
            if item["role"] == "mixed_query_exact_pathway_membership"
        )
        artifact["sha256"] = artifact_sha256(membership_path)

    _rewrite_registry(path, mutate)
    _refresh_mixed_asset_bindings(
        tmp_path, path, "mixed_query_exact_pathway_membership"
    )
    with pytest.raises(MixedQueryAssetError, match="not registered protein-coding"):
        MixedExactPathwayQuery.from_registry(path)


def test_incomplete_cartesian_key_coverage_fails_closed(tmp_path: Path) -> None:
    path = _build_registry(tmp_path)
    registry = json.loads(path.read_text(encoding="utf-8"))
    manifest_entry = next(
        item
        for item in registry["website_artifacts"]
        if item["role"] == "mixed_query_asset_manifest"
    )
    manifest_path = tmp_path / manifest_entry["path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["key_coverage"]["complete_key_coverage"] = False
    manifest["key_coverage"]["missing_key_count"] = 1
    _write_json(manifest_path, manifest)
    manifest_entry["sha256"] = artifact_sha256(manifest_path)
    _write_json(path, registry)

    with pytest.raises(ReleaseRegistryError, match="complete typed key-coverage"):
        MixedExactPathwayQuery.from_registry(path)


def test_typed_not_evaluated_key_keeps_null_reason_and_denominator(
    tmp_path: Path,
) -> None:
    path = _build_registry(tmp_path)
    association_path = tmp_path / "assets" / "association.parquet"
    frame = pd.read_parquet(association_path)
    target = (
        frame.cancer_id.eq("LUAD")
        & frame.lncrna_id.eq("LNC:ENSG00000100001")
        & frame.pathway_id.eq("MSIGDB:REACTOME:P1")
    )
    frame["availability_reason"] = None
    frame.loc[target, "association_membership_probability"] = None
    frame.loc[target, "availability"] = False
    frame.loc[target, "availability_reason"] = "NOT_EVALUATED_MODEL_FOLD_GAP"
    frame.to_parquet(association_path, index=False)

    def mutate(value: dict) -> None:
        artifact = next(
            item
            for item in value["website_artifacts"]
            if item["role"] == "mixed_query_v32_lnc_exact_association"
        )
        artifact["sha256"] = artifact_sha256(association_path)

    _rewrite_registry(path, mutate)
    _refresh_mixed_asset_bindings(
        tmp_path, path, "mixed_query_v32_lnc_exact_association"
    )
    result = MixedExactPathwayQuery.from_registry(path).query(
        ["LNC:ENSG00000100001"], cancer_id="LUAD", top_k=3
    )
    assert result["coverage"]["lncrna_expected_key_count"] == 3
    assert result["coverage"]["lncrna_evaluated_key_count"] == 2
    assert result["coverage"]["lncrna_not_evaluated_key_count"] == 1
    assert result["coverage"]["lncrna_not_evaluated_reason_counts"] == {
        "NOT_EVALUATED_MODEL_FOLD_GAP": 1
    }
    assert "MSIGDB:REACTOME:P1" not in {
        row["pathway_id"] for row in result["results"]
    }
    assert result["semantics"]["unscored_key_encoding"] == (
        "NULL_WITH_TYPED_NOT_EVALUATED_REASON"
    )


@pytest.mark.parametrize("column", ["held_out_proxy_label", "patient_fold_id"])
def test_private_fold_or_label_columns_are_rejected(tmp_path: Path, column: str) -> None:
    path = _build_registry(tmp_path)
    association_path = tmp_path / "assets" / "association.parquet"
    frame = pd.read_parquet(association_path)
    frame[column] = 0
    frame.to_parquet(association_path, index=False)

    def mutate(value: dict) -> None:
        artifact = next(
            item
            for item in value["website_artifacts"]
            if item["role"] == "mixed_query_v32_lnc_exact_association"
        )
        artifact["sha256"] = artifact_sha256(association_path)

    _rewrite_registry(path, mutate)
    _refresh_mixed_asset_bindings(
        tmp_path, path, "mixed_query_v32_lnc_exact_association"
    )
    with pytest.raises(MixedQueryAssetError, match="private/old/extra"):
        MixedExactPathwayQuery.from_registry(path)


def test_family_to_exact_broadcast_is_rejected(tmp_path: Path) -> None:
    path = _build_registry(tmp_path)
    association_path = tmp_path / "assets" / "association.parquet"
    frame = pd.read_parquet(association_path)
    frame["family_to_exact_broadcast"] = True
    frame.to_parquet(association_path, index=False)

    def mutate(value: dict) -> None:
        artifact = next(
            item
            for item in value["website_artifacts"]
            if item["role"] == "mixed_query_v32_lnc_exact_association"
        )
        artifact["sha256"] = artifact_sha256(association_path)

    _rewrite_registry(path, mutate)
    _refresh_mixed_asset_bindings(
        tmp_path, path, "mixed_query_v32_lnc_exact_association"
    )
    with pytest.raises(MixedQueryAssetError, match="broadcasting is forbidden"):
        MixedExactPathwayQuery.from_registry(path)


def test_extra_static_pathway_outside_v32_prediction_universe_is_rejected(tmp_path: Path) -> None:
    path = _build_registry(tmp_path)
    membership_path = tmp_path / "assets" / "membership.parquet"
    frame = pd.read_parquet(membership_path)
    frame.loc[len(frame)] = {
        "pathway_id": "MSIGDB:REACTOME:EXTRA_NOT_TRAINED_IN_V32",
        "gene_id": "ENSG00000200001",
    }
    frame.to_parquet(membership_path, index=False)

    def mutate(value: dict) -> None:
        artifact = next(
            item
            for item in value["website_artifacts"]
            if item["role"] == "mixed_query_exact_pathway_membership"
        )
        artifact["sha256"] = artifact_sha256(membership_path)

    _rewrite_registry(path, mutate)
    _refresh_mixed_asset_bindings(
        tmp_path, path, "mixed_query_exact_pathway_membership"
    )
    with pytest.raises(MixedQueryAssetError, match="membership exceeds"):
        MixedExactPathwayQuery.from_registry(path)


def test_hypergeometric_ora_matches_closed_form() -> None:
    # Drawing both members of a size-2 pathway in a four-gene universe: 1/C(4,2).
    assert hypergeometric_overrepresentation(
        overlap=2, pathway_size=2, query_size=2, universe_size=4
    ) == pytest.approx(1 / 6)


def test_staging_api_is_separate_and_callable(tmp_path: Path) -> None:
    app = create_staging_app(_build_registry(tmp_path))
    paths = {route.path for route in app.routes}
    assert "/v3.2-staging/enrichment/mixed-exact-pathway" in paths
    with TestClient(app) as client:
        health = client.get("/v3.2-staging/health")
        assert health.status_code == 200
        assert health.json()["production_deployed"] is False
        response = client.post(
            "/v3.2-staging/enrichment/mixed-exact-pathway",
            json={"members": ["LNC:ENSG00000100001", "TP53"], "cancer_id": "LUAD"},
        )
        assert response.status_code == 200
        assert response.json()["semantics"]["target_level"] == "exact_pathway_only"


def test_partition_download_route_resolves_only_the_selected_part(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import website.backend.v32_staging_api as staging_api_module

    part_path = tmp_path / "part-1.tsv"
    part_path.write_bytes(b"selected\n")
    calls: list[tuple[str, int]] = []

    class SelectedPartCatalog:
        audit = {"pass_count": 1}

        def __init__(self, *args, **kwargs) -> None:
            pass

        def resolve(self, download_id: str) -> dict:
            raise AssertionError("partition route must not resolve and hash the full tree")

        def resolve_part(self, download_id: str, part_index: int) -> dict:
            calls.append((download_id, part_index))
            return {
                "kind": "part",
                "download_id": download_id,
                "part_index": part_index,
                "path": str(part_path),
                "relative_name": part_path.name,
                "sha256": artifact_sha256(part_path),
                "sha256_tree": "b" * 64,
                "bytes": part_path.stat().st_size,
            }

    monkeypatch.setattr(
        staging_api_module, "AuditedDownloadCatalog", SelectedPartCatalog
    )
    app = staging_api_module.create_staging_app(
        _build_registry(tmp_path / "registry"),
        download_catalog_binding_path=tmp_path / "fake-binding.json",
        download_catalog_binding_sha256="a" * 64,
        download_catalog_audit_binding_path=tmp_path / "fake-audit.json",
        download_catalog_audit_binding_sha256="c" * 64,
    )
    with TestClient(app) as client:
        response = client.get(
            "/v3.2-staging/downloads/drug_response_predictions/parts/1"
        )
    assert response.status_code == 200
    assert response.content == b"selected\n"
    assert response.headers["x-partition-tree-sha256"] == "b" * 64
    assert calls == [("drug_response_predictions", 1)]


def _enable_staging_multimodule_query(root: Path, registry_path: Path) -> None:
    assets = root / "assets"
    state_path = assets / "state_typed_predictions.parquet"
    clinical_patient_path = assets / "clinical_patient_risk.parquet"
    clinical_entity_path = assets / "entity_clinical_associations.parquet"
    genomic_path = assets / "mutation_cnv_typed_predictions.parquet"

    pd.DataFrame(
        [
            {
                "cancer_id": "LUAD",
                "lncrna_id": "LNC:ENSG00000100001",
                "state_id": "stemness_rna::RNAss",
                "state_membership_probability": 0.88,
                "state_effect": 0.31,
                "association_direction": "positive",
                "availability": True,
                "availability_reason": None,
                "folds_available": 5,
                "folds_expected": 5,
                "model_version": "V3.2",
                "analysis_version": VERSION,
                "training_run_id": "V32_STATE_FRESH_TEST",
                "changes_primary_ranking": False,
            },
            {
                "cancer_id": "BRCA",
                "lncrna_id": "LNC:ENSG00000100002",
                "state_id": "stemness_dna::DNAss",
                "state_membership_probability": None,
                "state_effect": None,
                "association_direction": "unavailable",
                "availability": False,
                "availability_reason": "STATE_NOT_AVAILABLE_FOR_CANCER",
                "folds_available": 0,
                "folds_expected": 5,
                "model_version": "V3.2",
                "analysis_version": VERSION,
                "training_run_id": "V32_STATE_FRESH_TEST",
                "changes_primary_ranking": False,
            },
        ]
    ).to_parquet(state_path, index=False)

    pd.DataFrame(
        [
            {
                "cancer_id": "LUAD",
                "subject_type": "patient",
                "subject_id": "TCGA-TEST-0001",
                "clinical_endpoint": "OS",
                "clinical_relevance_probability": 0.81,
                "availability": True,
                "failure_reason": "",
                "patient_fold_id": 0,
                "model_artifact_id": "clinical-fold-0",
                "analysis_version": VERSION,
                "training_run_id": "V32_CLINICAL_FRESH_TEST",
                "changes_primary_ranking": False,
            },
            {
                "cancer_id": "LUAD",
                "subject_type": "patient",
                "subject_id": "TCGA-TEST-0002",
                "clinical_endpoint": "OS",
                "clinical_relevance_probability": 0.73,
                "availability": True,
                "failure_reason": "",
                "patient_fold_id": 1,
                "model_artifact_id": "clinical-fold-1",
                "analysis_version": VERSION,
                "training_run_id": "V32_CLINICAL_FRESH_TEST",
                "changes_primary_ranking": False,
            },
        ]
    ).to_parquet(clinical_patient_path, index=False)

    entity_version = "CancerLncAtlas_V3.2.test.clinical_entity"
    pd.DataFrame(
        [
            {
                "cancer_id": "LUAD",
                "subject_type": "lncRNA",
                "subject_id": "LNC:ENSG00000100001",
                "clinical_endpoint": "OS",
                "clinical_relevance_probability": 0.91,
                "availability": True,
                "failure_reason": "",
                "direction": "risk",
                "analysis_version": entity_version,
                "old_checkpoint_loaded": False,
                "old_predictions_used_as_features": False,
                "changes_primary_ranking": False,
            }
        ]
    ).to_parquet(clinical_entity_path, index=False)

    pd.DataFrame(
        [
            {
                "cancer_id": "LUAD",
                "lncrna_id": "LNC:ENSG00000100001",
                "pathway_id": "MSIGDB:REACTOME:P1",
                "mutation_context_probability": 0.84,
                "mutation_available": True,
                "mutation_unavailable_reason": "",
                "mutation_patient_folds_with_prediction": 5,
                "cnv_context_probability": 0.24,
                "cnv_available": True,
                "cnv_unavailable_reason": "",
                "cnv_patient_folds_with_prediction": 5,
                "analysis_version": VERSION,
                "training_run_id": "V32_MUTATION_CNV_FRESH_TEST",
                "module_id": "mutation_cnv",
                "target_level": "cancer_x_lncrna_x_exact_pathway_genomic_context",
                "prediction_format": "CC_HHGT_V3_2_GENOMIC_TYPED_PREDICTIONS_V1",
                "changes_primary_ranking": False,
            },
            {
                "cancer_id": "LUAD",
                "lncrna_id": "LNC:ENSG00000100002",
                "pathway_id": "MSIGDB:REACTOME:P2",
                "mutation_context_probability": None,
                "mutation_available": False,
                "mutation_unavailable_reason": "MUTATION_NOT_AVAILABLE_FOR_PAIR",
                "mutation_patient_folds_with_prediction": 0,
                "cnv_context_probability": 0.92,
                "cnv_available": True,
                "cnv_unavailable_reason": "",
                "cnv_patient_folds_with_prediction": 5,
                "analysis_version": VERSION,
                "training_run_id": "V32_MUTATION_CNV_FRESH_TEST",
                "module_id": "mutation_cnv",
                "target_level": "cancer_x_lncrna_x_exact_pathway_genomic_context",
                "prediction_format": "CC_HHGT_V3_2_GENOMIC_TYPED_PREDICTIONS_V1",
                "changes_primary_ranking": False,
            },
        ]
    ).to_parquet(genomic_path, index=False)

    state_lineage_path = root / "lineage" / "state.json"
    state_lineage = _success_aux_lineage("state")
    state_lineage.update(
        prediction_sha256=artifact_sha256(state_path),
        prediction_rows=2,
    )
    _write_json(state_lineage_path, state_lineage)

    genomic_lineage_path = root / "lineage" / "mutation_cnv.json"
    genomic_lineage = _success_aux_lineage("mutation_cnv")
    genomic_lineage.update(
        prediction_sha256=artifact_sha256(genomic_path),
        prediction_rows=2,
    )
    _write_json(genomic_lineage_path, genomic_lineage)

    clinical_lineage_path = root / "lineage" / "clinical.json"
    clinical_lineage = json.loads(clinical_lineage_path.read_text(encoding="utf-8"))
    clinical_lineage["public_prediction_rows"] = 2
    _write_json(clinical_lineage_path, clinical_lineage)

    entity_lineage_path = root / "lineage" / "clinical_entity_query.json"
    _write_json(
        entity_lineage_path,
        {
            "analysis_version": entity_version,
            "module_id": "clinical_entity_association_extension",
            "training_status": "SUCCESS",
            "old_checkpoint_loaded": False,
            "old_predictions_used_as_features": False,
            "old_rankings_used_as_outputs": False,
            "fresh_statistical_calculation": True,
            "summary_sha256": artifact_sha256(clinical_entity_path),
            "summary_rows": 1,
        },
    )
    entity_validation_path = root / "lineage" / "clinical_entity_query_validation.json"
    _write_json(
        entity_validation_path,
        {
            "status": "PASS",
            "all_non_null_results_newly_computed_v32": True,
            "historical_results_used": False,
            "changes_primary_ranking": False,
            "summary_sha256": artifact_sha256(clinical_entity_path),
            "lineage_sha256": artifact_sha256(entity_lineage_path),
        },
    )

    def mutate(value: dict) -> None:
        for module_id, lineage in (
            ("state", state_lineage),
            ("mutation_cnv", genomic_lineage),
        ):
            lineage_path = root / "lineage" / f"{module_id}.json"
            value["modules"][module_id] = {
                "status": "SUCCESS_NEWLY_TRAINED",
                "analysis_version": VERSION,
                "training_run_id": lineage["training_run_id"],
                "lineage_path": str(lineage_path.relative_to(root)),
                "lineage_sha256": artifact_sha256(lineage_path),
                "new_training_attestation": True,
                "old_checkpoint_loaded": False,
                "old_predictions_used_as_features": False,
                "old_rankings_used_as_outputs": False,
            }
        value["modules"]["clinical"]["lineage_sha256"] = artifact_sha256(
            clinical_lineage_path
        )
        for role, path, kind, module_id in (
            ("state_typed_predictions", state_path, "v32_public_prediction", "state"),
            (
                "clinical_patient_risk",
                clinical_patient_path,
                "v32_public_prediction",
                "clinical",
            ),
            (
                "clinical_entity_association_public",
                clinical_entity_path,
                "v32_public_statistical_result",
                "clinical",
            ),
            (
                "clinical_entity_association_lineage",
                entity_lineage_path,
                "v32_public_metadata",
                "clinical",
            ),
            (
                "clinical_entity_association_validation",
                entity_validation_path,
                "v32_public_metadata",
                "clinical",
            ),
            (
                "mutation_cnv_typed_predictions",
                genomic_path,
                "v32_public_prediction",
                "mutation_cnv",
            ),
        ):
            value["website_artifacts"].append(
                {
                    "role": role,
                    "path": str(path.relative_to(root)),
                    "sha256": artifact_sha256(path),
                    "artifact_kind": kind,
                    "generation": VERSION,
                    "source_module": module_id,
                }
            )
        value["capabilities"]["clinical"] = {
            "patient_risk": {
                "status": "STAGING_ARTIFACT_REGISTERED",
                "artifact_role": "clinical_patient_risk",
            },
            "entity_association": {
                "status": "STAGING_ARTIFACT_REGISTERED",
                "enabled": True,
                "artifact_role": "clinical_entity_association_public",
                "lineage_role": "clinical_entity_association_lineage",
                "validation_role": "clinical_entity_association_validation",
            },
        }

    _rewrite_registry(registry_path, mutate)


def test_multimodule_staging_routes_use_only_hash_bound_v32_results(tmp_path: Path) -> None:
    registry_path = _build_registry(tmp_path)
    _enable_staging_multimodule_query(tmp_path, registry_path)
    app = create_staging_app(registry_path)
    paths = {route.path for route in app.routes}
    assert {
        "/v3.2-staging/state",
        "/v3.2-staging/clinical",
        "/v3.2-staging/genomic",
    }.issubset(paths)

    with TestClient(app) as client:
        health = client.get("/v3.2-staging/health").json()
        assert health["auxiliary_queries"]["state"]["enabled"] is True
        state = client.get(
            "/v3.2-staging/state",
            params={"cancer_id": "luad", "state_id": "stemness_rna::RNAss"},
        )
        assert state.status_code == 200
        assert state.json()["results"][0]["training_run_id"] == "V32_STATE_FRESH_TEST"

        clinical = client.get(
            "/v3.2-staging/clinical",
            params={"clinical_endpoint": "OS", "cancer_id": "LUAD", "limit": 2},
        )
        assert clinical.status_code == 200
        clinical_value = clinical.json()
        assert clinical_value["entity_results"]["results"][0]["subject_type"] == "lncRNA"
        assert clinical_value["patient_results"]["returned_rows"] == 2
        assert "patient_result_id" in clinical_value["patient_results"]["results"][0]
        assert "subject_id" not in clinical_value["patient_results"]["results"][0]
        assert "TCGA-TEST" not in clinical.text
        assert "patient_fold_id" not in clinical.text
        assert "model_artifact_id" not in clinical.text

        mutation = client.get(
            "/v3.2-staging/genomic",
            params={"modality": "mutation", "cancer_id": "LUAD", "availability": True},
        )
        cnv = client.get(
            "/v3.2-staging/genomic",
            params={"modality": "cnv", "cancer_id": "LUAD", "availability": True},
        )
        assert mutation.status_code == 200
        assert mutation.json()["modality"] == "mutation"
        assert mutation.json()["total_rows"] == 1
        assert mutation.json()["results"][0]["context_probability"] == pytest.approx(0.84)
        assert cnv.status_code == 503
        assert "superseded combined CNV score is disabled" in cnv.json()["detail"]


def test_directional_cnv_binding_replaces_only_cnv_route(tmp_path: Path) -> None:
    from test_v32_directional_cnv_query import write_directional_cnv_binding

    registry_path = _build_registry(tmp_path / "registry")
    _enable_staging_multimodule_query(tmp_path / "registry", registry_path)
    binding, digest, audit, audit_digest = write_directional_cnv_binding(
        tmp_path / "cnv"
    )
    app = create_staging_app(
        registry_path,
        directional_cnv_binding_path=binding,
        directional_cnv_binding_sha256=digest,
        directional_cnv_audit_binding_path=audit,
        directional_cnv_audit_binding_sha256=audit_digest,
    )

    with TestClient(app) as client:
        response = client.get(
            "/v3.2-staging/genomic",
            params={"modality": "cnv", "cancer_id": "LUAD"},
        )
        coverage = client.get(
            "/v3.2-staging/cnv/coverage", params={"cancer_id": "UCS"}
        )
        entry = client.get("/v3.2-staging/downloads/cnv_context")

    assert response.status_code == 200
    assert response.json()["provenance"]["superseded_combined_cnv_used"] is False
    assert coverage.status_code == 200
    assert coverage.json()["rows"][0]["typed_unavailable_rows"] == 1
    assert entry.status_code == 200
    assert entry.json()["source_generation"] == "CURRENT_V3.2_DIRECTIONAL_CNV_ONLY"


def test_multimodule_query_rechecks_artifact_hash_on_every_request(tmp_path: Path) -> None:
    registry_path = _build_registry(tmp_path)
    _enable_staging_multimodule_query(tmp_path, registry_path)
    app = create_staging_app(registry_path)
    state_path = tmp_path / "assets" / "state_typed_predictions.parquet"
    frame = pd.read_parquet(state_path)
    frame.loc[0, "state_membership_probability"] = 0.01
    frame.to_parquet(state_path, index=False)
    with TestClient(app) as client:
        response = client.get("/v3.2-staging/state", params={"cancer_id": "LUAD"})
    assert response.status_code == 503
    assert "artifact hash drift" in response.json()["detail"]


def test_multimodule_query_rejects_old_generation_rows_even_when_rehashed(tmp_path: Path) -> None:
    registry_path = _build_registry(tmp_path)
    _enable_staging_multimodule_query(tmp_path, registry_path)
    state_path = tmp_path / "assets" / "state_typed_predictions.parquet"
    frame = pd.read_parquet(state_path)
    frame.loc[0, "analysis_version"] = "CancerLncAtlas_V2.9_STATE"
    frame.to_parquet(state_path, index=False)
    lineage_path = tmp_path / "lineage" / "state.json"
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    lineage["prediction_sha256"] = artifact_sha256(state_path)
    _write_json(lineage_path, lineage)

    def mutate(value: dict) -> None:
        next(
            item
            for item in value["website_artifacts"]
            if item["role"] == "state_typed_predictions"
        )["sha256"] = artifact_sha256(state_path)
        value["modules"]["state"]["lineage_sha256"] = artifact_sha256(lineage_path)

    _rewrite_registry(registry_path, mutate)
    with pytest.raises(StagingQueryAssetError, match="typed provenance validation"):
        V32StagingQuery.from_registry(registry_path)


def test_unregistered_auxiliary_module_route_fails_closed_without_fallback(tmp_path: Path) -> None:
    app = create_staging_app(_build_registry(tmp_path))
    with TestClient(app) as client:
        response = client.get("/v3.2-staging/state", params={"cancer_id": "LUAD"})
    assert response.status_code == 503
    assert "staging query is disabled" in response.json()["detail"]


def test_formal23_binding_supersedes_legacy_single_cell_routes(
    tmp_path: Path, monkeypatch
) -> None:
    success_path = tmp_path / "FORMAL23_SUCCESS.json"
    download_part_path = tmp_path / "BRCA.association_evidence.parquet"
    download_part_path.write_bytes(b"formal23-part\n")

    class FakeFormal23:
        def __init__(self, supplied_path, *, expected_sha256):
            assert supplied_path == success_path
            assert expected_sha256 == "a" * 64

        @staticmethod
        def capability_status():
            return {
                "module": "single_cell_formal23",
                "formal_eligible_cancer_count": 23,
                "typed_unavailable_cancer_count": 10,
                "learned_celltype_probability_available": False,
            }

        @staticmethod
        def query_lncrna_celltype(**kwargs):
            return {"query_kind": "formal23_expression", "filters": kwargs}

        @staticmethod
        def query_pathway_activity(**kwargs):
            return {"query_kind": "formal23_activity", "filters": kwargs}

        @staticmethod
        def query_associations(**kwargs):
            return {
                "query_kind": "formal23_association",
                "learned_association": False,
                "filters": kwargs,
            }

        @staticmethod
        def coverage_summary(**kwargs):
            return {"query_kind": "formal23_coverage", "filters": kwargs}

        @staticmethod
        def download_entry(download_id):
            if download_id in {
                "single_cell_pseudotime",
                "single_cell_figures",
            }:
                return {
                    "download_id": download_id,
                    "status": "GAP_TYPED_UNAVAILABLE",
                    "unavailable_reason": "FORMAL23_PRODUCT_NOT_GENERATED",
                }
            return {
                "download_id": download_id,
                "status": "READY_PARTS",
                "source_generation": "V3.2_R7_FRESH_FROM_RAW_H5",
                "parts": [{"cancer_id": "BRCA"}],
            }

        @staticmethod
        def resolve_download_part(download_id, part_index):
            assert download_id == "single_cell_associations"
            assert part_index == 0
            return {
                "kind": "part",
                "download_id": download_id,
                "part_index": part_index,
                "path": str(download_part_path),
                "relative_name": download_part_path.name,
                "sha256": artifact_sha256(download_part_path),
                "sha256_tree": "d" * 64,
                "bytes": download_part_path.stat().st_size,
                "cancer_id": "BRCA",
            }

    monkeypatch.setattr(v32_staging_api, "SingleCellFormal23Query", FakeFormal23)
    client = TestClient(
        create_staging_app(
            _build_registry(tmp_path / "registry"),
            single_cell_formal23_success_path=success_path,
            single_cell_formal23_success_sha256="a" * 64,
        )
    )

    health = client.get("/v3.2-staging/health").json()
    assert health[
        "single_cell_formal23_query"
    ] == "ENABLED_23_ELIGIBLE_PLUS_10_TYPED_UNAVAILABLE"
    assert health["single_cell_formal23_diagnostic"] == (
        "TYPED_UNAVAILABLE_NO_PSEUDOTIME_OR_FIGURES"
    )
    assert "single_cell_ucell_17c_query" not in health
    assert "single_cell_diagnostic_query" not in health
    assert health["legacy_aliases"] == {
        "single_cell_ucell_17c_query": "ROUTE_SHIM_TO_FORMAL23_DONOR_ACTIVITY",
        "single_cell_diagnostic_query": "TYPED_UNAVAILABLE_FORMAL23",
    }
    capability = client.get("/v3.2-staging/single-cell/context/capability")
    assert capability.status_code == 200
    assert capability.json()["formal_eligible_cancer_count"] == 23

    expression = client.get(
        "/v3.2-staging/single-cell/context/lncrna-celltype",
        params={"cancer_id": "BRCA"},
    )
    assert expression.status_code == 200
    assert expression.json()["query_kind"] == "formal23_expression"

    association = client.get(
        "/v3.2-staging/single-cell/exact-pathway-associations",
        params={"cancer_id": "BRCA", "availability": True},
    )
    assert association.status_code == 200
    assert association.json()["learned_association"] is False
    assert association.json()["filters"]["availability"] == "AVAILABLE"

    assert client.get(
        "/v3.2-staging/single-cell/exact-pathway-associations"
    ).status_code == 422
    assert client.get(
        "/v3.2-staging/single-cell/exact-pathway-associations",
        params={"cancer_id": "BRCA", "min_probability": 0.5},
    ).status_code == 422
    assert client.get(
        "/v3.2-staging/single-cell/context/pathway-activity",
        params={"cancer_id": "BRCA", "donor_id": "D1"},
    ).status_code == 422

    coverage = client.get(
        "/v3.2-staging/single-cell/lncrna-expression-summary",
        params={"limit": 33},
    )
    assert coverage.status_code == 200
    assert coverage.json()["query_kind"] == "formal23_coverage"

    entry = client.get("/v3.2-staging/downloads/single_cell_associations")
    assert entry.status_code == 200
    assert entry.json()["status"] == "READY_PARTS"
    assert entry.json()["parts"][0]["download_url"].endswith("/parts/0")
    part = client.get(
        "/v3.2-staging/downloads/single_cell_associations/parts/0"
    )
    assert part.status_code == 200
    assert part.content == b"formal23-part\n"
    assert part.headers["x-single-cell-generation"] == "FORMAL23-FRESH"
    assert client.get(
        "/v3.2-staging/downloads/single_cell_pseudotime/parts/0"
    ).status_code == 409

    for route in (
        "/v3.2-staging/single-cell/diagnostic/capability",
        "/v3.2-staging/single-cell/diagnostic/BRCA/download-manifest",
        "/v3.2-staging/single-cell/diagnostic/BRCA/download/trajectory.parquet",
    ):
        response = client.get(route)
        assert response.status_code == 200
        typed_gap = response.json()
        assert typed_gap["status"] == "GAP_TYPED_UNAVAILABLE"
        assert typed_gap["availability_status"] == "TYPED_UNAVAILABLE"
        assert typed_gap["availability"] is False
        assert typed_gap["numeric_values_available"] is False
        assert typed_gap["primary_score_weight"] is None
        assert typed_gap["secondary_score_weight"] is None
        assert typed_gap["formal_eligible_cancer_count"] == 23
        assert typed_gap["typed_unavailable_cancer_count"] == 10
