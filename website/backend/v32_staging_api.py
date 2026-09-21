"""Standalone, non-production API for registered V3.2 staging results.

This module is intentionally not imported by ``website/backend/app.py``.  Run
it on a separate staging port only after setting
``CANCERLNCATLAS_V32_STAGING_REGISTRY`` to a validated registry JSON.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from cc_hhgt.v32.bulk_expression_query import (
    MAX_QUERY_LIMIT as BULK_EXPRESSION_MAX_QUERY_LIMIT,
    MAX_QUERY_OFFSET as BULK_EXPRESSION_MAX_QUERY_OFFSET,
    BulkExpressionQueryAssetError,
    BulkExpressionQueryInputError,
    BulkExpressionReleaseQuery,
)
from cc_hhgt.v32.bulk_coexpression_query import (
    MAX_QUERY_LIMIT as BULK_COEXPRESSION_MAX_QUERY_LIMIT,
    MAX_QUERY_OFFSET as BULK_COEXPRESSION_MAX_QUERY_OFFSET,
    BulkCoexpressionQueryAssetError,
    BulkCoexpressionQueryInputError,
    BulkCoexpressionReleaseQuery,
)
from cc_hhgt.v32.clinical_km_query import (
    MAX_QUERY_LIMIT as CLINICAL_KM_MAX_QUERY_LIMIT,
    MAX_QUERY_OFFSET as CLINICAL_KM_MAX_QUERY_OFFSET,
    ClinicalKMQueryAssetError,
    ClinicalKMQueryInputError,
    ClinicalKMReleaseQuery,
)
from cc_hhgt.v32.continuous_activity_query import (
    MAX_QUERY_LIMIT as CONTINUOUS_ACTIVITY_MAX_QUERY_LIMIT,
    MAX_QUERY_OFFSET as CONTINUOUS_ACTIVITY_MAX_QUERY_OFFSET,
    ContinuousActivityQueryAssetError,
    ContinuousActivityQueryInputError,
    ContinuousActivityReleaseQuery,
)
from cc_hhgt.v32.directional_cnv_query import (
    CNV_DOWNLOAD_IDS,
    MAX_QUERY_LIMIT as DIRECTIONAL_CNV_MAX_QUERY_LIMIT,
    MAX_QUERY_OFFSET as DIRECTIONAL_CNV_MAX_QUERY_OFFSET,
    DirectionalCNVQueryAssetError,
    DirectionalCNVQueryInputError,
    DirectionalCNVReleaseQuery,
)
from cc_hhgt.v32.mixed_query import (
    MAX_MEMBERS,
    MAX_TOP_K,
    MixedExactPathwayQuery,
    MixedQueryError,
)
from cc_hhgt.v32.multimodal_fusion_query import (
    MAX_QUERY_LIMIT as MULTIMODAL_FUSION_MAX_QUERY_LIMIT,
    MAX_QUERY_OFFSET as MULTIMODAL_FUSION_MAX_QUERY_OFFSET,
    MultimodalFusionQueryAssetError,
    MultimodalFusionQueryInputError,
    MultimodalFusionReleaseQuery,
)
from cc_hhgt.v32.network_query import (
    MAX_QUERY_LIMIT as NETWORK_MAX_QUERY_LIMIT,
    MAX_QUERY_OFFSET as NETWORK_MAX_QUERY_OFFSET,
    NetworkQueryAssetError,
    NetworkQueryInputError,
    UnifiedNetworkQuery,
)
from cc_hhgt.v32.drug_mechanism_query import (
    MAX_QUERY_LIMIT as DRUG_MECHANISM_MAX_QUERY_LIMIT,
    MAX_QUERY_OFFSET as DRUG_MECHANISM_MAX_QUERY_OFFSET,
    DrugMechanismQueryAssetError,
    DrugMechanismQueryInputError,
    DrugMechanismReleaseQuery,
)
from cc_hhgt.v32.drug_sparse_query import (
    DrugSparseAssetError,
    DrugSparseInputError,
    DrugSparseQueryBundle,
    load_drug_sparse_query_bundle,
)
from cc_hhgt.v32.download_catalog import (
    AuditedDownloadCatalog,
    DownloadCatalogError,
)
from cc_hhgt.v32.evidence_query import (
    MAX_QUERY_LIMIT as EVIDENCE_MAX_QUERY_LIMIT,
    MAX_QUERY_OFFSET as EVIDENCE_MAX_QUERY_OFFSET,
    EvidenceBindingQuery,
    EvidenceQueryAssetError,
    EvidenceQueryInputError,
)
from cc_hhgt.v32.evidence_direction_query import (
    MAX_QUERY_LIMIT as EVIDENCE_DIRECTION_MAX_QUERY_LIMIT,
    MAX_QUERY_OFFSET as EVIDENCE_DIRECTION_MAX_QUERY_OFFSET,
    EvidenceDirectionProbabilityQuery,
    EvidenceDirectionQueryAssetError,
    EvidenceDirectionQueryInputError,
)
from cc_hhgt.v32.exact_pathway_report_query import (
    ExactPathwayReportAssetError,
    ExactPathwayReportQuery,
)
from cc_hhgt.v32.experiment_perturbation_query import (
    MAX_QUERY_LIMIT as EXPERIMENT_PERTURBATION_MAX_QUERY_LIMIT,
    MAX_QUERY_OFFSET as EXPERIMENT_PERTURBATION_MAX_QUERY_OFFSET,
    ExperimentPerturbationQueryAssetError,
    ExperimentPerturbationQueryInputError,
    ExperimentPerturbationReleaseQuery,
)
from cc_hhgt.v32.experiment_perturbation_bridge_query import (
    MAX_QUERY_LIMIT as EXPERIMENT_BRIDGE_MAX_QUERY_LIMIT,
    MAX_QUERY_OFFSET as EXPERIMENT_BRIDGE_MAX_QUERY_OFFSET,
    ExperimentPerturbationBridgeAssetError,
    ExperimentPerturbationBridgeInputError,
    ExperimentPerturbationBridgeQuery,
)
from cc_hhgt.v32.external_validation_query import (
    MAX_QUERY_LIMIT as EXTERNAL_VALIDATION_MAX_QUERY_LIMIT,
    MAX_QUERY_OFFSET as EXTERNAL_VALIDATION_MAX_QUERY_OFFSET,
    ExternalValidationQueryAssetError,
    ExternalValidationQueryInputError,
    ExternalValidationReleaseQuery,
)
from cc_hhgt.v32.interaction_query import (
    MAX_QUERY_LIMIT as INTERACTION_MAX_QUERY_LIMIT,
    MAX_QUERY_OFFSET as INTERACTION_MAX_QUERY_OFFSET,
    InteractionQueryAssetError,
    InteractionQueryInputError,
    InteractionReleaseQuery,
)
from cc_hhgt.v32.hnsc_ucell_query import (
    MAX_PAGE_LIMIT as HNSC_UCELL_MAX_PAGE_LIMIT,
    HNSCUCellAssetError,
    HNSCUCellInputError,
    HNSCUCellPilotQuery,
)
from cc_hhgt.v32.single_cell_ucell_17c_query import (
    MAX_QUERY_LIMIT as SINGLE_CELL_UCELL_17C_MAX_QUERY_LIMIT,
    SingleCellUCell17CAssetError,
    SingleCellUCell17CInputError,
    SingleCellUCell17CQuery,
)
from cc_hhgt.v32.single_cell_diagnostic_publication import (
    MAX_QUERY_LIMIT as SINGLE_CELL_DIAGNOSTIC_MAX_QUERY_LIMIT,
    SingleCellDiagnosticAssetError,
    SingleCellDiagnosticInputError,
    SingleCellDiagnosticPublicationQuery,
)
from cc_hhgt.v32.historical_artifact_remediation import (
    HistoricalArtifactQuery,
    HistoricalArtifactRemediationError,
)
from cc_hhgt.v32.gene_set_subtype_query import (
    MAX_MEMBER_LIMIT as GENE_SET_SUBTYPE_MAX_MEMBER_LIMIT,
    MAX_QUERY_LIMIT as GENE_SET_SUBTYPE_MAX_QUERY_LIMIT,
    MAX_QUERY_OFFSET as GENE_SET_SUBTYPE_MAX_QUERY_OFFSET,
    GeneSetSubtypeQueryAssetError,
    GeneSetSubtypeQueryInputError,
    GeneSetSubtypeQueryNotFoundError,
    GeneSetSubtypeReleaseQuery,
)
from cc_hhgt.v32.release_registry import artifact_sha256, load_release_registry
from cc_hhgt.v32.score_architecture import score_architecture_manifest
from cc_hhgt.v32.single_cell_expression_query import (
    MAX_QUERY_LIMIT as SINGLE_CELL_EXPRESSION_MAX_QUERY_LIMIT,
    MAX_QUERY_OFFSET as SINGLE_CELL_EXPRESSION_MAX_QUERY_OFFSET,
    SingleCellExpressionAssetError,
    SingleCellExpressionAuditQuery,
    SingleCellExpressionInputError,
)
from cc_hhgt.v32.single_cell_fusion_query import (
    MAX_QUERY_LIMIT as SINGLE_CELL_FUSION_MAX_QUERY_LIMIT,
    MAX_QUERY_OFFSET as SINGLE_CELL_FUSION_MAX_QUERY_OFFSET,
    SingleCellFusionQueryAssetError,
    SingleCellFusionQueryInputError,
    SingleCellFusionReleaseQuery,
)
from cc_hhgt.v32.single_cell_formal_context_query import (
    MAX_QUERY_LIMIT as SINGLE_CELL_FORMAL_CONTEXT_MAX_QUERY_LIMIT,
    MAX_QUERY_OFFSET as SINGLE_CELL_FORMAL_CONTEXT_MAX_QUERY_OFFSET,
    SingleCellFormalContextAssetError,
    SingleCellFormalContextInputError,
    SingleCellFormalContextQuery,
)
from cc_hhgt.v32.single_cell_formal23_query import (
    MAX_QUERY_LIMIT as SINGLE_CELL_FORMAL23_MAX_QUERY_LIMIT,
    MAX_QUERY_OFFSET as SINGLE_CELL_FORMAL23_MAX_QUERY_OFFSET,
    SINGLE_CELL_FORMAL23_DOWNLOAD_IDS,
    SINGLE_CELL_SUPERSEDED_DOWNLOAD_IDS,
    SingleCellFormal23AssetError,
    SingleCellFormal23InputError,
    SingleCellFormal23Query,
)
from cc_hhgt.v32.single_cell_gap_query import (
    MAX_QUERY_LIMIT as SINGLE_CELL_GAP_MAX_QUERY_LIMIT,
    MAX_QUERY_OFFSET as SINGLE_CELL_GAP_MAX_QUERY_OFFSET,
    SingleCellGapAuditQuery,
    SingleCellGapQueryAssetError,
    SingleCellGapQueryInputError,
)
from cc_hhgt.v32.state_gene_set_query import (
    MAX_GENE_SET_LIMIT as STATE_GENE_SET_MAX_LIMIT,
    MAX_MEMBER_LIMIT as STATE_GENE_SET_MAX_MEMBER_LIMIT,
    MAX_QUERY_OFFSET as STATE_GENE_SET_MAX_OFFSET,
    StateGeneSetQueryAssetError,
    StateGeneSetQueryInputError,
    StateGeneSetReleaseQuery,
)
from cc_hhgt.v32.staging_query import (
    MAX_QUERY_LIMIT,
    MAX_QUERY_OFFSET,
    StagingQueryAssetError,
    StagingQueryInputError,
    V32StagingQuery,
)


class MixedExactPathwayRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    members: list[str] = Field(min_length=1, max_length=MAX_MEMBERS)
    cancer_id: str | None = None
    top_k: int = Field(default=20, ge=1, le=MAX_TOP_K)


class MutationContextRequest(BaseModel):
    """Historical POST contract backed only by current V3.2 genomic rows."""

    model_config = ConfigDict(extra="forbid")

    cancer_id: str = Field(min_length=2, max_length=20)
    lncrna_id: str | None = Field(default=None, max_length=64)
    pathway_id: str | None = Field(default=None, max_length=160)
    availability: bool | None = None
    top_k: int = Field(default=50, ge=1, le=MAX_QUERY_LIMIT)


class LegacyMutationContextRequest(BaseModel):
    """Compatibility request for the retired V2.7 Mutation URL."""

    model_config = ConfigDict(extra="forbid")
    cancer_id: str = Field(min_length=2, max_length=20)
    lncrna_id: str | None = Field(default=None, max_length=64)
    pathway_id: str | None = Field(default=None, max_length=160)
    pathway_family_id: str | None = Field(default=None, max_length=160)
    availability: bool | None = None
    top_k: int = Field(default=20, ge=1, le=MAX_QUERY_LIMIT)


def _validate_experiment_bridge_audit(
    *,
    bridge_binding_path: str | Path,
    bridge_binding_sha256: str,
    audit_binding_path: str | Path,
    audit_binding_sha256: str,
) -> dict[str, Any]:
    """Require the independent post-audit before declaring the bridge final."""

    bridge = Path(bridge_binding_path).resolve()
    audit = Path(audit_binding_path).resolve()
    for path, label in ((bridge, "bridge binding"), (audit, "bridge audit binding")):
        if not path.is_file() or path.is_symlink():
            raise ExperimentPerturbationBridgeAssetError(
                f"Experiment {label} is missing or unsafe: {path}"
            )
    observed_audit = artifact_sha256(audit)
    if observed_audit != str(audit_binding_sha256 or "").lower():
        raise ExperimentPerturbationBridgeAssetError(
            "Experiment bridge independent-audit binding SHA mismatch"
        )
    try:
        payload = json.loads(audit.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentPerturbationBridgeAssetError(
            "Experiment bridge independent-audit binding is invalid JSON"
        ) from exc
    required = {
        "format": "CC_HHGT_V3_2_EXPERIMENT_EVIDENCE_BRIDGE_INDEPENDENT_AUDIT_BINDING_V1",
        "status": "PASS",
        "accepted_for_api_integration": True,
        "production_deployed": False,
    }
    if not isinstance(payload, dict) or any(
        payload.get(key) != value for key, value in required.items()
    ):
        raise ExperimentPerturbationBridgeAssetError(
            "Experiment bridge independent-audit binding was not accepted for API integration"
        )
    declaration = payload.get("bridge_binding")
    if (
        not isinstance(declaration, dict)
        or Path(str(declaration.get("path", ""))).resolve() != bridge
        or declaration.get("sha256") != str(bridge_binding_sha256 or "").lower()
        or artifact_sha256(bridge) != declaration.get("sha256")
    ):
        raise ExperimentPerturbationBridgeAssetError(
            "Experiment bridge independent audit is tied to a different binding"
        )
    report_declaration = payload.get("report")
    if not isinstance(report_declaration, dict):
        raise ExperimentPerturbationBridgeAssetError(
            "Experiment bridge independent audit lacks its report"
        )
    report_path = Path(str(report_declaration.get("path", ""))).resolve()
    if (
        not report_path.is_file()
        or report_path.is_symlink()
        or artifact_sha256(report_path) != report_declaration.get("sha256")
    ):
        raise ExperimentPerturbationBridgeAssetError(
            "Experiment bridge independent-audit report SHA drift"
        )
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentPerturbationBridgeAssetError(
            "Experiment bridge independent-audit report is invalid JSON"
        ) from exc
    if not isinstance(report, dict):
        raise ExperimentPerturbationBridgeAssetError(
            "Experiment bridge independent-audit report must be a JSON object"
        )
    checks = report.get("checks")
    if (
        report.get("audit_format")
        != "CC_HHGT_V3_2_EXPERIMENT_EVIDENCE_BRIDGE_INDEPENDENT_AUDIT_V1"
        or report.get("status") != "PASS"
        or report.get("accepted_for_api_integration") is not True
        or report.get("production_deployed") is not False
        or report.get("checks_failed") != 0
        or not isinstance(report.get("checks_passed"), int)
        or report["checks_passed"] < 49
        or not isinstance(checks, list)
        or len(checks) != report["checks_passed"]
        or any(not isinstance(item, dict) or item.get("status") != "PASS" for item in checks)
        or report.get("bridge_binding_sha256") != declaration.get("sha256")
        or report.get("separate_fusion_forbidden") is not True
    ):
        raise ExperimentPerturbationBridgeAssetError(
            "Experiment bridge independent-audit report did not pass the full gate"
        )
    return {
        "status": "PASS",
        "accepted_for_api_integration": True,
        "checks_passed": report["checks_passed"],
        "binding_path": str(audit),
        "binding_sha256": observed_audit,
        "report_path": str(report_path),
        "report_sha256": report_declaration["sha256"],
    }


def _validate_network_independent_audit(
    *,
    network_manifest_path: str | Path,
    network_manifest_sha256: str,
    audit_binding_path: str | Path,
    audit_binding_sha256: str,
) -> dict[str, Any]:
    """Require the accepted 168/168 independent Network audit."""

    manifest = Path(network_manifest_path).resolve()
    audit = Path(audit_binding_path).resolve()
    for path, label in ((manifest, "manifest"), (audit, "audit binding")):
        if not path.is_file() or path.is_symlink():
            raise NetworkQueryAssetError(
                f"Unified Network {label} is missing or unsafe: {path}"
            )
    expected_manifest_sha = str(network_manifest_sha256 or "").lower()
    expected_audit_sha = str(audit_binding_sha256 or "").lower()
    if artifact_sha256(manifest) != expected_manifest_sha:
        raise NetworkQueryAssetError("Unified Network manifest SHA256 mismatch")
    observed_audit_sha = artifact_sha256(audit)
    if observed_audit_sha != expected_audit_sha:
        raise NetworkQueryAssetError(
            "Unified Network independent-audit binding SHA256 mismatch"
        )
    try:
        payload = json.loads(audit.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NetworkQueryAssetError(
            "Unified Network independent-audit binding is invalid JSON"
        ) from exc
    required = {
        "format": "CC_HHGT_V3_2_UNIFIED_NETWORK_INDEPENDENT_AUDIT_BINDING_V1",
        "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
        "status": "PASS",
        "release_ready": False,
        "production_deployed": False,
        "accepted_for_api_integration": True,
        "pass_count": 168,
        "fail_count": 0,
        "primary_probability_bitwise_preserved": True,
        "native_missingness_to_zero": False,
        "family_to_exact_broadcast": False,
        "experiment_separate_fusion_created": False,
    }
    if not isinstance(payload, dict) or any(
        payload.get(key) != value for key, value in required.items()
    ):
        raise NetworkQueryAssetError(
            "Unified Network independent audit was not accepted at 168/168"
        )
    declaration = payload.get("network_manifest")
    if (
        not isinstance(declaration, dict)
        or Path(str(declaration.get("path", ""))).resolve() != manifest
        or declaration.get("sha256") != expected_manifest_sha
    ):
        raise NetworkQueryAssetError(
            "Unified Network independent audit is tied to a different manifest"
        )
    report_declaration = payload.get("report")
    if not isinstance(report_declaration, dict):
        raise NetworkQueryAssetError(
            "Unified Network independent audit lacks its report"
        )
    report_path = Path(str(report_declaration.get("path", ""))).resolve()
    if (
        not report_path.is_file()
        or report_path.is_symlink()
        or artifact_sha256(report_path) != report_declaration.get("sha256")
    ):
        raise NetworkQueryAssetError(
            "Unified Network independent-audit report SHA256 drift"
        )
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NetworkQueryAssetError(
            "Unified Network independent-audit report is invalid JSON"
        ) from exc
    checks = report.get("checks") if isinstance(report, dict) else None
    report_manifest = report.get("manifest") if isinstance(report, dict) else None
    if (
        not isinstance(report, dict)
        or report.get("format")
        != "CC_HHGT_V3_2_UNIFIED_NETWORK_INDEPENDENT_AUDIT_V1"
        or report.get("analysis_version") != "CancerLncAtlas_V3.2_FULL_MULTITASK"
        or report.get("status") != "PASS"
        or report.get("release_ready") is not False
        or report.get("production_deployed") is not False
        or report.get("accepted_for_api_integration") is not True
        or report.get("independent_of_materializer_implementation") is not True
        or report.get("materializer_modules_imported") is not False
        or report.get("pass_count") != 168
        or report.get("fail_count") != 0
        or not isinstance(checks, list)
        or len(checks) != 168
        or any(not isinstance(item, dict) or item.get("status") != "PASS" for item in checks)
        or not isinstance(report_manifest, dict)
        or Path(str(report_manifest.get("path", ""))).resolve() != manifest
        or report_manifest.get("sha256") != expected_manifest_sha
    ):
        raise NetworkQueryAssetError(
            "Unified Network independent-audit report did not pass the full 168/168 gate"
        )
    return {
        "status": "PASS",
        "accepted_for_api_integration": True,
        "pass_count": 168,
        "fail_count": 0,
        "binding_path": str(audit),
        "binding_sha256": observed_audit_sha,
        "report_path": str(report_path),
        "report_sha256": report_declaration["sha256"],
        "network_manifest_sha256": expected_manifest_sha,
    }


def _validate_single_cell_formal_context_independent_audit(
    *,
    context_binding_path: str | Path,
    context_binding_sha256: str,
    audit_binding_path: str | Path,
    audit_binding_sha256: str,
) -> dict[str, Any]:
    """Require the independent 25/25 gate for the formal context sidecar."""

    context = Path(context_binding_path).resolve()
    audit = Path(audit_binding_path).resolve()
    for path, label in ((context, "context binding"), (audit, "audit binding")):
        if not path.is_file() or path.is_symlink():
            raise SingleCellFormalContextAssetError(
                f"Single-cell formal-context {label} is missing or unsafe: {path}"
            )
    expected_context_sha = str(context_binding_sha256 or "").lower()
    expected_audit_sha = str(audit_binding_sha256 or "").lower()
    if artifact_sha256(context) != expected_context_sha:
        raise SingleCellFormalContextAssetError(
            "Single-cell formal-context binding SHA256 mismatch"
        )
    observed_audit_sha = artifact_sha256(audit)
    if observed_audit_sha != expected_audit_sha:
        raise SingleCellFormalContextAssetError(
            "Single-cell formal-context independent-audit binding SHA256 mismatch"
        )
    try:
        payload = json.loads(audit.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SingleCellFormalContextAssetError(
            "Single-cell formal-context independent-audit binding is invalid JSON"
        ) from exc
    required = {
        "format": "CC_HHGT_V3_2_SINGLE_CELL_FORMAL_CONTEXT_INDEPENDENT_AUDIT_BINDING_V1",
        "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
        "status": "PASS",
        "check_count": 25,
        "failure_count": 0,
        "release_ready": False,
        "production_deployed": False,
    }
    if not isinstance(payload, dict) or any(
        payload.get(key) != value for key, value in required.items()
    ):
        raise SingleCellFormalContextAssetError(
            "Single-cell formal-context independent audit was not accepted at 25/25"
        )
    if (
        Path(str(payload.get("source_binding_path", ""))).resolve() != context
        or payload.get("source_binding_sha256") != expected_context_sha
    ):
        raise SingleCellFormalContextAssetError(
            "Single-cell formal-context independent audit is tied to a different binding"
        )
    report_path = Path(str(payload.get("audit_report_path", ""))).resolve()
    if (
        not report_path.is_file()
        or report_path.is_symlink()
        or artifact_sha256(report_path) != payload.get("audit_report_sha256")
    ):
        raise SingleCellFormalContextAssetError(
            "Single-cell formal-context independent-audit report SHA256 drift"
        )
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SingleCellFormalContextAssetError(
            "Single-cell formal-context independent-audit report is invalid JSON"
        ) from exc
    checks = report.get("checks") if isinstance(report, dict) else None
    if (
        not isinstance(report, dict)
        or report.get("format")
        != "CC_HHGT_V3_2_SINGLE_CELL_FORMAL_CONTEXT_INDEPENDENT_AUDIT_V1"
        or report.get("analysis_version") != "CancerLncAtlas_V3.2_FULL_MULTITASK"
        or report.get("status") != "PASS"
        or report.get("check_count") != 25
        or report.get("failure_count") != 0
        or Path(str(report.get("source_binding_path", ""))).resolve() != context
        or report.get("source_binding_sha256") != expected_context_sha
        or not isinstance(checks, list)
        or len(checks) != 25
        or any(not isinstance(item, dict) or item.get("status") != "PASS" for item in checks)
    ):
        raise SingleCellFormalContextAssetError(
            "Single-cell formal-context independent-audit report did not pass 25/25"
        )
    success_path = audit.parent / "SUCCESS.json"
    if not success_path.is_file() or success_path.is_symlink():
        raise SingleCellFormalContextAssetError(
            "Single-cell formal-context independent-audit SUCCESS is missing or unsafe"
        )
    try:
        success = json.loads(success_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SingleCellFormalContextAssetError(
            "Single-cell formal-context independent-audit SUCCESS is invalid JSON"
        ) from exc
    if (
        not isinstance(success, dict)
        or success.get("binding") != audit.name
        or success.get("binding_sha256") != expected_audit_sha
        or success.get("status") != "PASS"
    ):
        raise SingleCellFormalContextAssetError(
            "Single-cell formal-context independent-audit SUCCESS is stale"
        )
    return {
        "status": "PASS",
        "accepted_for_api_integration": True,
        "check_count": 25,
        "failure_count": 0,
        "binding_path": str(audit),
        "binding_sha256": observed_audit_sha,
        "report_path": str(report_path),
        "report_sha256": payload["audit_report_sha256"],
        "source_binding_sha256": expected_context_sha,
        "release_ready": False,
        "production_deployed": False,
    }


def create_staging_app(
    registry_path: str | Path,
    *,
    network_manifest_path: str | Path | None = None,
    network_manifest_sha256: str | None = None,
    network_audit_binding_path: str | Path | None = None,
    network_audit_binding_sha256: str | None = None,
    gene_set_subtype_binding_path: str | Path | None = None,
    gene_set_subtype_binding_sha256: str | None = None,
    gene_set_subtype_audit_binding_path: str | Path | None = None,
    gene_set_subtype_audit_binding_sha256: str | None = None,
    exact_pathway_report_binding_path: str | Path | None = None,
    exact_pathway_report_binding_sha256: str | None = None,
    exact_pathway_report_audit_binding_path: str | Path | None = None,
    exact_pathway_report_audit_binding_sha256: str | None = None,
    historical_remediation_binding_path: str | Path | None = None,
    historical_remediation_binding_sha256: str | None = None,
    historical_remediation_audit_binding_path: str | Path | None = None,
    historical_remediation_audit_binding_sha256: str | None = None,
    download_catalog_binding_path: str | Path | None = None,
    download_catalog_binding_sha256: str | None = None,
    download_catalog_audit_binding_path: str | Path | None = None,
    download_catalog_audit_binding_sha256: str | None = None,
    interaction_manifest_path: str | Path | None = None,
    interaction_manifest_sha256: str | None = None,
    drug_mechanism_manifest_path: str | Path | None = None,
    drug_mechanism_manifest_sha256: str | None = None,
    drug_sparse_manifest_path: str | Path | None = None,
    drug_sparse_manifest_sha256: str | None = None,
    evidence_binding_path: str | Path | None = None,
    evidence_binding_sha256: str | None = None,
    evidence_direction_binding_path: str | Path | None = None,
    evidence_direction_binding_sha256: str | None = None,
    evidence_direction_audit_binding_path: str | Path | None = None,
    evidence_direction_audit_binding_sha256: str | None = None,
    experiment_perturbation_binding_path: str | Path | None = None,
    experiment_perturbation_binding_sha256: str | None = None,
    experiment_perturbation_bridge_binding_path: str | Path | None = None,
    experiment_perturbation_bridge_binding_sha256: str | None = None,
    experiment_perturbation_bridge_audit_binding_path: str | Path | None = None,
    experiment_perturbation_bridge_audit_binding_sha256: str | None = None,
    single_cell_fusion_binding_path: str | Path | None = None,
    single_cell_fusion_binding_sha256: str | None = None,
    single_cell_expression_binding_path: str | Path | None = None,
    single_cell_expression_binding_sha256: str | None = None,
    single_cell_gap_binding_path: str | Path | None = None,
    single_cell_gap_binding_sha256: str | None = None,
    single_cell_gap_audit_binding_path: str | Path | None = None,
    single_cell_gap_audit_binding_sha256: str | None = None,
    single_cell_formal_context_binding_path: str | Path | None = None,
    single_cell_formal_context_binding_sha256: str | None = None,
    single_cell_formal_context_audit_binding_path: str | Path | None = None,
    single_cell_formal_context_audit_binding_sha256: str | None = None,
    single_cell_formal23_success_path: str | Path | None = None,
    single_cell_formal23_success_sha256: str | None = None,
    bulk_expression_binding_path: str | Path | None = None,
    bulk_expression_binding_sha256: str | None = None,
    hnsc_ucell_binding_path: str | Path | None = None,
    hnsc_ucell_binding_sha256: str | None = None,
    single_cell_ucell_17c_binding_path: str | Path | None = None,
    single_cell_ucell_17c_binding_sha256: str | None = None,
    single_cell_diagnostic_binding_path: str | Path | None = None,
    single_cell_diagnostic_binding_sha256: str | None = None,
    state_gene_set_binding_path: str | Path | None = None,
    state_gene_set_binding_sha256: str | None = None,
    clinical_km_binding_path: str | Path | None = None,
    clinical_km_binding_sha256: str | None = None,
    bulk_coexpression_binding_path: str | Path | None = None,
    bulk_coexpression_binding_sha256: str | None = None,
    external_validation_binding_path: str | Path | None = None,
    external_validation_binding_sha256: str | None = None,
    multimodal_fusion_binding_path: str | Path | None = None,
    multimodal_fusion_binding_sha256: str | None = None,
    continuous_activity_binding_path: str | Path | None = None,
    continuous_activity_binding_sha256: str | None = None,
    directional_cnv_binding_path: str | Path | None = None,
    directional_cnv_binding_sha256: str | None = None,
    directional_cnv_audit_binding_path: str | Path | None = None,
    directional_cnv_audit_binding_sha256: str | None = None,
) -> FastAPI:
    """Create an isolated API whose artifacts are fixed by one registry hash."""

    registry = load_release_registry(registry_path, require_staging=True)
    engine = MixedExactPathwayQuery(registry)
    auxiliary = V32StagingQuery(registry)
    if (network_manifest_path is None) != (network_manifest_sha256 is None):
        raise NetworkQueryAssetError(
            "Unified Network staging requires both manifest path and expected SHA256"
        )
    if (network_audit_binding_path is None) != (
        network_audit_binding_sha256 is None
    ):
        raise NetworkQueryAssetError(
            "Unified Network staging requires both audit binding path and expected SHA256"
        )
    if (network_manifest_path is None) != (network_audit_binding_path is None):
        raise NetworkQueryAssetError(
            "Unified Network staging requires both the release manifest and independent-audit binding"
        )
    if (gene_set_subtype_binding_path is None) != (
        gene_set_subtype_binding_sha256 is None
    ):
        raise GeneSetSubtypeQueryAssetError(
            "Gene Set/ranked-subtype staging requires both binding path and expected SHA256"
        )
    if (gene_set_subtype_audit_binding_path is None) != (
        gene_set_subtype_audit_binding_sha256 is None
    ):
        raise GeneSetSubtypeQueryAssetError(
            "Gene Set/ranked-subtype staging requires both audit binding path and expected SHA256"
        )
    if (gene_set_subtype_binding_path is None) != (
        gene_set_subtype_audit_binding_path is None
    ):
        raise GeneSetSubtypeQueryAssetError(
            "Gene Set/ranked-subtype staging requires both the release and independent-audit bindings"
        )
    if (exact_pathway_report_binding_path is None) != (
        exact_pathway_report_binding_sha256 is None
    ):
        raise ExactPathwayReportAssetError(
            "Exact-pathway report requires both binding path and expected SHA256"
        )
    if (exact_pathway_report_audit_binding_path is None) != (
        exact_pathway_report_audit_binding_sha256 is None
    ):
        raise ExactPathwayReportAssetError(
            "Exact-pathway report requires both audit binding path and expected SHA256"
        )
    if (exact_pathway_report_binding_path is None) != (
        exact_pathway_report_audit_binding_path is None
    ):
        raise ExactPathwayReportAssetError(
            "Exact-pathway report requires both release and independent-audit bindings"
        )
    if (historical_remediation_binding_path is None) != (
        historical_remediation_binding_sha256 is None
    ):
        raise HistoricalArtifactRemediationError(
            "Historical remediation staging requires both binding path and expected SHA256"
        )
    if (historical_remediation_audit_binding_path is None) != (
        historical_remediation_audit_binding_sha256 is None
    ):
        raise HistoricalArtifactRemediationError(
            "Historical remediation staging requires both audit path and expected SHA256"
        )
    if (historical_remediation_binding_path is None) != (
        historical_remediation_audit_binding_path is None
    ):
        raise HistoricalArtifactRemediationError(
            "Historical remediation staging requires both release and independent-audit bindings"
        )
    if (download_catalog_binding_path is None) != (
        download_catalog_binding_sha256 is None
    ):
        raise DownloadCatalogError(
            "Download catalog staging requires both binding path and expected SHA256"
        )
    if (download_catalog_audit_binding_path is None) != (
        download_catalog_audit_binding_sha256 is None
    ):
        raise DownloadCatalogError(
            "Download catalog staging requires both audit path and expected SHA256"
        )
    if (download_catalog_binding_path is None) != (
        download_catalog_audit_binding_path is None
    ):
        raise DownloadCatalogError(
            "Download catalog staging requires both release and independent-audit bindings"
        )
    if (interaction_manifest_path is None) != (interaction_manifest_sha256 is None):
        raise InteractionQueryAssetError(
            "Interaction staging requires both manifest path and expected SHA256"
        )
    if (drug_mechanism_manifest_path is None) != (drug_mechanism_manifest_sha256 is None):
        raise DrugMechanismQueryAssetError(
            "Drug mechanism staging requires both manifest path and expected SHA256"
        )
    if (drug_sparse_manifest_path is None) != (drug_sparse_manifest_sha256 is None):
        raise DrugSparseAssetError(
            "Drug sparse staging requires both manifest path and expected SHA256"
        )
    if (evidence_binding_path is None) != (evidence_binding_sha256 is None):
        raise EvidenceQueryAssetError(
            "Evidence staging requires both binding path and expected SHA256"
        )
    if (evidence_direction_binding_path is None) != (
        evidence_direction_binding_sha256 is None
    ):
        raise EvidenceDirectionQueryAssetError(
            "Evidence direction staging requires both release binding path and expected SHA256"
        )
    if (evidence_direction_audit_binding_path is None) != (
        evidence_direction_audit_binding_sha256 is None
    ):
        raise EvidenceDirectionQueryAssetError(
            "Evidence direction staging requires both audit binding path and expected SHA256"
        )
    if (evidence_direction_binding_path is None) != (
        evidence_direction_audit_binding_path is None
    ):
        raise EvidenceDirectionQueryAssetError(
            "Evidence direction staging requires both release and independent-audit bindings"
        )
    if (experiment_perturbation_binding_path is None) != (
        experiment_perturbation_binding_sha256 is None
    ):
        raise ExperimentPerturbationQueryAssetError(
            "Experiment perturbation staging requires both binding path and expected SHA256"
        )
    if (experiment_perturbation_bridge_binding_path is None) != (
        experiment_perturbation_bridge_binding_sha256 is None
    ):
        raise ExperimentPerturbationBridgeAssetError(
            "Experiment perturbation Evidence bridge staging requires both binding path and expected SHA256"
        )
    if (experiment_perturbation_bridge_audit_binding_path is None) != (
        experiment_perturbation_bridge_audit_binding_sha256 is None
    ):
        raise ExperimentPerturbationBridgeAssetError(
            "Experiment perturbation bridge audit staging requires both binding path and expected SHA256"
        )
    if (experiment_perturbation_bridge_binding_path is None) != (
        experiment_perturbation_bridge_audit_binding_path is None
    ):
        raise ExperimentPerturbationBridgeAssetError(
            "Final Experiment bridge staging requires both the bridge and independent-audit bindings"
        )
    if (single_cell_fusion_binding_path is None) != (
        single_cell_fusion_binding_sha256 is None
    ):
        raise SingleCellFusionQueryAssetError(
            "Single-cell fusion staging requires both binding path and expected SHA256"
        )
    if (single_cell_expression_binding_path is None) != (
        single_cell_expression_binding_sha256 is None
    ):
        raise SingleCellExpressionAssetError(
            "Single-cell expression staging requires both binding path and expected SHA256"
        )
    if (single_cell_gap_binding_path is None) != (
        single_cell_gap_binding_sha256 is None
    ):
        raise SingleCellGapQueryAssetError(
            "Single-cell gap staging requires both release binding path and expected SHA256"
        )
    if (single_cell_gap_audit_binding_path is None) != (
        single_cell_gap_audit_binding_sha256 is None
    ):
        raise SingleCellGapQueryAssetError(
            "Single-cell gap staging requires both audit binding path and expected SHA256"
        )
    if (single_cell_gap_binding_path is None) != (
        single_cell_gap_audit_binding_path is None
    ):
        raise SingleCellGapQueryAssetError(
            "Single-cell gap staging requires both release and independent-audit bindings"
        )
    if (single_cell_formal_context_binding_path is None) != (
        single_cell_formal_context_binding_sha256 is None
    ):
        raise SingleCellFormalContextAssetError(
            "Single-cell formal-context staging requires both binding path and expected SHA256"
        )
    if (single_cell_formal_context_audit_binding_path is None) != (
        single_cell_formal_context_audit_binding_sha256 is None
    ):
        raise SingleCellFormalContextAssetError(
            "Single-cell formal-context staging requires both audit binding path and expected SHA256"
        )
    if (single_cell_formal_context_binding_path is None) != (
        single_cell_formal_context_audit_binding_path is None
    ):
        raise SingleCellFormalContextAssetError(
            "Single-cell formal-context staging requires both release and independent-audit bindings"
        )
    if (single_cell_formal23_success_path is None) != (
        single_cell_formal23_success_sha256 is None
    ):
        raise SingleCellFormal23AssetError(
            "Single-cell formal-23 staging requires SUCCESS path and expected SHA256"
        )
    if (bulk_expression_binding_path is None) != (bulk_expression_binding_sha256 is None):
        raise BulkExpressionQueryAssetError(
            "Bulk expression staging requires both binding path and expected SHA256"
        )
    if (hnsc_ucell_binding_path is None) != (hnsc_ucell_binding_sha256 is None):
        raise HNSCUCellAssetError(
            "HNSC UCell staging requires both binding path and expected SHA256"
        )
    if (single_cell_ucell_17c_binding_path is None) != (
        single_cell_ucell_17c_binding_sha256 is None
    ):
        raise SingleCellUCell17CAssetError(
            "Legacy cell-level UCell staging requires both binding path and expected SHA256"
        )
    if (single_cell_diagnostic_binding_path is None) != (
        single_cell_diagnostic_binding_sha256 is None
    ):
        raise SingleCellDiagnosticAssetError(
            "Legacy single-cell diagnostic staging requires both binding path and expected SHA256"
        )
    if (state_gene_set_binding_path is None) != (state_gene_set_binding_sha256 is None):
        raise StateGeneSetQueryAssetError(
            "State gene-set staging requires both binding path and expected SHA256"
        )
    if (clinical_km_binding_path is None) != (clinical_km_binding_sha256 is None):
        raise ClinicalKMQueryAssetError(
            "Clinical KM staging requires both binding path and expected SHA256"
        )
    if (bulk_coexpression_binding_path is None) != (
        bulk_coexpression_binding_sha256 is None
    ):
        raise BulkCoexpressionQueryAssetError(
            "Bulk coexpression staging requires both binding path and expected SHA256"
        )
    if (external_validation_binding_path is None) != (
        external_validation_binding_sha256 is None
    ):
        raise ExternalValidationQueryAssetError(
            "External validation staging requires both binding path and expected SHA256"
        )
    if (multimodal_fusion_binding_path is None) != (
        multimodal_fusion_binding_sha256 is None
    ):
        raise MultimodalFusionQueryAssetError(
            "Multimodal fusion staging requires both binding path and expected SHA256"
        )
    if (continuous_activity_binding_path is None) != (
        continuous_activity_binding_sha256 is None
    ):
        raise ContinuousActivityQueryAssetError(
            "Continuous activity staging requires both binding path and expected SHA256"
        )
    if (directional_cnv_binding_path is None) != (
        directional_cnv_binding_sha256 is None
    ):
        raise DirectionalCNVQueryAssetError(
            "Directional CNV staging requires both binding path and expected SHA256"
        )
    if (directional_cnv_audit_binding_path is None) != (
        directional_cnv_audit_binding_sha256 is None
    ):
        raise DirectionalCNVQueryAssetError(
            "Directional CNV staging requires both audit binding path and expected SHA256"
        )
    if (directional_cnv_binding_path is None) != (
        directional_cnv_audit_binding_path is None
    ):
        raise DirectionalCNVQueryAssetError(
            "Directional CNV staging requires release and independent-audit bindings"
        )
    network_audit = (
        _validate_network_independent_audit(
            network_manifest_path=network_manifest_path,
            network_manifest_sha256=network_manifest_sha256,
            audit_binding_path=network_audit_binding_path,
            audit_binding_sha256=network_audit_binding_sha256,
        )
        if network_manifest_path is not None
        else None
    )
    network = (
        UnifiedNetworkQuery(
            network_manifest_path,
            expected_manifest_sha256=network_manifest_sha256,
        )
        if network_manifest_path is not None
        else None
    )
    gene_set_subtypes = (
        GeneSetSubtypeReleaseQuery(
            gene_set_subtype_binding_path,
            expected_binding_sha256=gene_set_subtype_binding_sha256,
            audit_binding_path=gene_set_subtype_audit_binding_path,
            expected_audit_binding_sha256=gene_set_subtype_audit_binding_sha256,
        )
        if gene_set_subtype_binding_path is not None
        else None
    )
    exact_pathway_report = (
        ExactPathwayReportQuery(
            exact_pathway_report_binding_path,
            expected_binding_sha256=exact_pathway_report_binding_sha256,
            audit_binding_path=exact_pathway_report_audit_binding_path,
            expected_audit_binding_sha256=exact_pathway_report_audit_binding_sha256,
        )
        if exact_pathway_report_binding_path is not None
        else None
    )
    historical_remediation = (
        HistoricalArtifactQuery(
            Path(historical_remediation_binding_path).resolve().parent,
            expected_binding_sha256=historical_remediation_binding_sha256,
            independent_audit_binding_path=historical_remediation_audit_binding_path,
            expected_independent_audit_sha256=(
                historical_remediation_audit_binding_sha256
            ),
        )
        if historical_remediation_binding_path is not None
        else None
    )
    download_catalog = (
        AuditedDownloadCatalog(
            download_catalog_binding_path,
            expected_binding_sha256=download_catalog_binding_sha256,
            audit_binding_path=download_catalog_audit_binding_path,
            expected_audit_binding_sha256=download_catalog_audit_binding_sha256,
            repository_root=Path(__file__).resolve().parents[2],
        )
        if download_catalog_binding_path is not None
        else None
    )
    interaction = (
        InteractionReleaseQuery(
            interaction_manifest_path,
            expected_manifest_sha256=interaction_manifest_sha256,
        )
        if interaction_manifest_path is not None
        else None
    )
    drug_mechanism = (
        DrugMechanismReleaseQuery(
            drug_mechanism_manifest_path,
            expected_manifest_sha256=drug_mechanism_manifest_sha256,
        )
        if drug_mechanism_manifest_path is not None
        else None
    )
    drug_sparse: DrugSparseQueryBundle | None = (
        load_drug_sparse_query_bundle(
            drug_sparse_manifest_path,
            expected_manifest_sha256=drug_sparse_manifest_sha256,
        )
        if drug_sparse_manifest_path is not None
        else None
    )
    evidence = (
        EvidenceBindingQuery(
            evidence_binding_path,
            expected_binding_sha256=evidence_binding_sha256,
        )
        if evidence_binding_path is not None
        else None
    )
    evidence_direction = (
        EvidenceDirectionProbabilityQuery(
            evidence_direction_binding_path,
            expected_binding_sha256=evidence_direction_binding_sha256,
            audit_binding_path=evidence_direction_audit_binding_path,
            expected_audit_binding_sha256=(
                evidence_direction_audit_binding_sha256
            ),
        )
        if evidence_direction_binding_path is not None
        else None
    )
    experiment_perturbation = (
        ExperimentPerturbationReleaseQuery(
            experiment_perturbation_binding_path,
            expected_binding_sha256=experiment_perturbation_binding_sha256,
        )
        if experiment_perturbation_binding_path is not None
        else None
    )
    experiment_perturbation_bridge_audit = (
        _validate_experiment_bridge_audit(
            bridge_binding_path=experiment_perturbation_bridge_binding_path,
            bridge_binding_sha256=experiment_perturbation_bridge_binding_sha256,
            audit_binding_path=experiment_perturbation_bridge_audit_binding_path,
            audit_binding_sha256=experiment_perturbation_bridge_audit_binding_sha256,
        )
        if experiment_perturbation_bridge_binding_path is not None
        else None
    )
    experiment_perturbation_bridge = (
        ExperimentPerturbationBridgeQuery(
            experiment_perturbation_bridge_binding_path,
            expected_binding_sha256=experiment_perturbation_bridge_binding_sha256,
        )
        if experiment_perturbation_bridge_binding_path is not None
        else None
    )
    single_cell_fusion = (
        SingleCellFusionReleaseQuery(
            single_cell_fusion_binding_path,
            expected_binding_sha256=single_cell_fusion_binding_sha256,
        )
        if single_cell_fusion_binding_path is not None
        else None
    )
    single_cell_expression = (
        SingleCellExpressionAuditQuery(
            single_cell_expression_binding_path,
            expected_binding_sha256=single_cell_expression_binding_sha256,
        )
        if single_cell_expression_binding_path is not None
        else None
    )
    single_cell_gap = (
        SingleCellGapAuditQuery(
            single_cell_gap_binding_path,
            expected_binding_sha256=single_cell_gap_binding_sha256,
            audit_binding_path=single_cell_gap_audit_binding_path,
            expected_audit_binding_sha256=single_cell_gap_audit_binding_sha256,
        )
        if single_cell_gap_binding_path is not None
        else None
    )
    single_cell_formal_context_audit = (
        _validate_single_cell_formal_context_independent_audit(
            context_binding_path=single_cell_formal_context_binding_path,
            context_binding_sha256=single_cell_formal_context_binding_sha256,
            audit_binding_path=single_cell_formal_context_audit_binding_path,
            audit_binding_sha256=single_cell_formal_context_audit_binding_sha256,
        )
        if single_cell_formal_context_binding_path is not None
        else None
    )
    single_cell_formal_context = (
        SingleCellFormalContextQuery(
            single_cell_formal_context_binding_path,
            expected_binding_sha256=single_cell_formal_context_binding_sha256,
        )
        if single_cell_formal_context_audit is not None
        else None
    )
    single_cell_formal23 = (
        SingleCellFormal23Query(
            single_cell_formal23_success_path,
            expected_sha256=single_cell_formal23_success_sha256,
        )
        if single_cell_formal23_success_path is not None
        else None
    )
    bulk_expression = (
        BulkExpressionReleaseQuery(
            bulk_expression_binding_path,
            expected_binding_sha256=bulk_expression_binding_sha256,
        )
        if bulk_expression_binding_path is not None
        else None
    )
    hnsc_ucell = (
        HNSCUCellPilotQuery(
            hnsc_ucell_binding_path,
            expected_binding_sha256=hnsc_ucell_binding_sha256,
        )
        if hnsc_ucell_binding_path is not None
        else None
    )
    single_cell_ucell_17c = (
        SingleCellUCell17CQuery(
            single_cell_ucell_17c_binding_path,
            expected_binding_sha256=single_cell_ucell_17c_binding_sha256,
        )
        if single_cell_ucell_17c_binding_path is not None
        else None
    )
    single_cell_diagnostic = (
        SingleCellDiagnosticPublicationQuery(
            single_cell_diagnostic_binding_path,
            expected_binding_sha256=single_cell_diagnostic_binding_sha256,
        )
        if single_cell_diagnostic_binding_path is not None
        else None
    )
    state_gene_sets = (
        StateGeneSetReleaseQuery(
            state_gene_set_binding_path,
            expected_binding_sha256=state_gene_set_binding_sha256,
        )
        if state_gene_set_binding_path is not None
        else None
    )
    clinical_km = (
        ClinicalKMReleaseQuery(
            clinical_km_binding_path,
            expected_binding_sha256=clinical_km_binding_sha256,
        )
        if clinical_km_binding_path is not None
        else None
    )
    bulk_coexpression = (
        BulkCoexpressionReleaseQuery(
            bulk_coexpression_binding_path,
            expected_binding_sha256=bulk_coexpression_binding_sha256,
        )
        if bulk_coexpression_binding_path is not None
        else None
    )
    external_validation = (
        ExternalValidationReleaseQuery(
            external_validation_binding_path,
            expected_binding_sha256=external_validation_binding_sha256,
        )
        if external_validation_binding_path is not None
        else None
    )
    multimodal_fusion = (
        MultimodalFusionReleaseQuery(
            multimodal_fusion_binding_path,
            expected_binding_sha256=multimodal_fusion_binding_sha256,
        )
        if multimodal_fusion_binding_path is not None
        else None
    )
    continuous_activity = (
        ContinuousActivityReleaseQuery(
            continuous_activity_binding_path,
            expected_binding_sha256=continuous_activity_binding_sha256,
        )
        if continuous_activity_binding_path is not None
        else None
    )
    directional_cnv = (
        DirectionalCNVReleaseQuery(
            directional_cnv_binding_path,
            expected_sha256=directional_cnv_binding_sha256,
            audit_binding_path=directional_cnv_audit_binding_path,
            expected_audit_sha256=directional_cnv_audit_binding_sha256,
        )
        if directional_cnv_binding_path is not None
        else None
    )
    staging = FastAPI(
        title="CancerLncAtlas V3.2 unified staging API",
        version="3.2-staging",
        docs_url="/v3.2-staging/docs",
        openapi_url="/v3.2-staging/openapi.json",
    )

    @staging.get("/v3.2-staging/health")
    def health() -> dict[str, Any]:
        payload = {
            "status": "PASS",
            "environment": "staging",
            "production_deployed": False,
            "release_ready": engine.registry.manifest["release_ready"],
            "all_non_null_predictions_newly_trained_v32": engine.registry.manifest[
                "all_non_null_predictions_newly_trained_v32"
            ],
            "all_published_results_generated_in_v32": engine.registry.manifest[
                "all_published_results_generated_in_v32"
            ],
            "release_id": engine.registry.release_id,
            "analysis_version": engine.registry.analysis_version,
            "registry_sha256": engine.registry.registry_sha256,
            "mixed_exact_pathway_query": "ENABLED",
            "auxiliary_queries": auxiliary.capability_status(),
            "unified_network_query": (
                "ENABLED_UNIFIED_NETWORK" if network else "NOT_MOUNTED"
            ),
            "unified_network_independent_audit": (
                "PASS_API_INTEGRATION_ACCEPTED_168_OF_168"
                if network_audit
                else "NOT_MOUNTED"
            ),
            "exact_pathway_browse_query": (
                "ENABLED_CURRENT_V32_HASH_PINNED"
                if gene_set_subtypes
                else "NOT_MOUNTED"
            ),
            "exact_pathway_report_query": (
                "ENABLED_HASH_BOUND_INDEPENDENTLY_AUDITED_66_OF_66"
                if exact_pathway_report
                else "NOT_MOUNTED"
            ),
            "gene_set_query": (
                "ENABLED_DETERMINISTIC_V32_DERIVED_HEAD"
                if gene_set_subtypes
                else "NOT_MOUNTED"
            ),
            "ranked_subtype_query": (
                "ENABLED_DETERMINISTIC_V32_DERIVED_HEAD"
                if gene_set_subtypes
                else "NOT_MOUNTED"
            ),
            "gene_set_ranked_subtype_independent_audit": (
                f"PASS_API_INTEGRATION_ACCEPTED_{gene_set_subtypes.audit['pass_count']}_OF_{gene_set_subtypes.audit['pass_count']}"
                if gene_set_subtypes
                else "NOT_MOUNTED"
            ),
            "historical_artifact_remediation_query": (
                "ENABLED_HASH_PINNED_WITH_INDEPENDENT_AUDIT"
                if historical_remediation
                else "NOT_MOUNTED"
            ),
            "clinical_translational_priority_query": (
                "ENABLED_FRESH_V32_DERIVED_ARTIFACT"
                if historical_remediation
                else "NOT_MOUNTED"
            ),
            "mutation_subgroup_query": (
                "ENABLED_RETAINED_NO_INCREMENT_DIAGNOSTIC"
                if historical_remediation
                else "NOT_MOUNTED"
            ),
            "cnv_coverage_query": (
                "ENABLED_DIRECTIONAL_CNV_ONLY_HASH_PINNED_TYPED_NULL"
                if directional_cnv
                else "PENDING_DIRECTIONAL_CNV_BINDING_SUPERSEDED_DISABLED"
            ),
            "directional_cnv_query": (
                "ENABLED_SIGNED_CNV_ONLY_INDEPENDENTLY_AUDITED"
                if directional_cnv
                else "NOT_MOUNTED"
            ),
            "download_catalog_query": (
                f"ENABLED_54_IDS_AUDITED_{download_catalog.audit['pass_count']}_OF_{download_catalog.audit['pass_count']}"
                if download_catalog
                else "NOT_MOUNTED"
            ),
            "physical_interaction_query": "ENABLED" if interaction else "NOT_MOUNTED",
            "physical_interaction_unmapped_query": (
                "ENABLED_HASH_PINNED" if interaction else "NOT_MOUNTED"
            ),
            "drug_mechanism_query": "ENABLED" if drug_mechanism else "NOT_MOUNTED",
            "drug_response_actionability_query": (
                "ENABLED_NATIVE_SPARSE" if drug_sparse else "NOT_MOUNTED"
            ),
            "evidence_query": "ENABLED" if evidence else "NOT_MOUNTED",
            "evidence_direction_probability_query": (
                "ENABLED_THREE_CLASS_HASH_PINNED_TYPED_NULL"
                if evidence_direction
                else "NOT_MOUNTED"
            ),
            "experiment_perturbation_query": (
                "ENABLED_FINAL_EVIDENCE_BRIDGE"
                if experiment_perturbation_bridge
                else "RAW_FACTS_ONLY_PENDING_FINAL_BRIDGE"
                if experiment_perturbation
                else "NOT_MOUNTED"
            ),
            "experiment_perturbation_raw_facts_query": (
                "ENABLED_CONFIDENCE_ONLY_PENDING_ABLATION"
                if experiment_perturbation
                else "NOT_MOUNTED"
            ),
            "experiment_perturbation_bridge_query": (
                "ENABLED_ALREADY_ABSORBED_IN_EVIDENCE_TRANSFORMER"
                if experiment_perturbation_bridge
                else "NOT_MOUNTED"
            ),
            "experiment_perturbation_bridge_independent_audit": (
                "PASS_API_INTEGRATION_ACCEPTED"
                if experiment_perturbation_bridge_audit
                else "NOT_MOUNTED"
            ),
            "single_cell_exact_pathway_query": (
                "ENABLED_FORMAL23_DONOR_ASSOCIATION"
                if single_cell_formal23
                else "ENABLED_FORMAL_FUSION_ADAPTER"
                if single_cell_fusion
                else "NOT_MOUNTED"
            ),
            "single_cell_expression_query": (
                "ENABLED_FORMAL23_DONOR_EXPRESSION"
                if single_cell_formal23
                else "ENABLED" if single_cell_expression else "NOT_MOUNTED"
            ),
            "single_cell_gap_audit_query": (
                "ENABLED_FORMAL23_COVERAGE_23_PLUS_10_TYPED_GAPS"
                if single_cell_formal23
                else "ENABLED_33_CANCER_COVERAGE_TYPED_GAPS_68_OF_68"
                if single_cell_gap
                else "NOT_MOUNTED"
            ),
            "single_cell_formal_context_query": (
                "ENABLED_23_CANCER_DONOR_BLOCKED_HASH_PINNED_AUDITED"
                if single_cell_formal23
                else "ENABLED_17_CANCER_HASH_PINNED_AUDITED_25_OF_25"
                if single_cell_formal_context
                else "NOT_MOUNTED"
            ),
            "single_cell_formal23_query": (
                "ENABLED_23_ELIGIBLE_PLUS_10_TYPED_UNAVAILABLE"
                if single_cell_formal23
                else "NOT_MOUNTED"
            ),
            "single_cell_formal23_activity_query": (
                "ENABLED_DONOR_AGGREGATED_ACTIVITY_23_ELIGIBLE"
                if single_cell_formal23
                else "NOT_MOUNTED"
            ),
            "single_cell_formal23_diagnostic": (
                "TYPED_UNAVAILABLE_NO_PSEUDOTIME_OR_FIGURES"
                if single_cell_formal23
                else "NOT_MOUNTED"
            ),
            "single_cell_formal_context_independent_audit": (
                "PASS_HASH_PINNED_FORMAL23_AUTHORITY"
                if single_cell_formal23
                else "PASS_API_INTEGRATION_ACCEPTED_25_OF_25"
                if single_cell_formal_context_audit
                else "NOT_MOUNTED"
            ),
            "single_cell_formal23_independent_audit": (
                "PASS_HASH_PINNED_FORMAL23_AUTHORITY"
                if single_cell_formal23
                else "NOT_MOUNTED"
            ),
            "bulk_expression_query": "ENABLED" if bulk_expression else "NOT_MOUNTED",
            "hnsc_ucell_query": "ENABLED_HNSC_PILOT" if hnsc_ucell else "NOT_MOUNTED",
            "single_cell_ucell_17c_query": (
                "LEGACY_ROUTE_SHIM_TO_FORMAL23_DONOR_ACTIVITY"
                if single_cell_formal23
                else "ENABLED_17_CANCER_HASH_PINNED_AUDITED"
                if single_cell_ucell_17c
                else "NOT_MOUNTED"
            ),
            "single_cell_diagnostic_query": (
                "TYPED_UNAVAILABLE_FORMAL23_NO_PSEUDOTIME_OR_FIGURES"
                if single_cell_formal23
                else "ENABLED_LEGACY_NUMERIC_PSEUDOTIME_AND_FIGURES_ZERO_WEIGHT"
                if single_cell_diagnostic
                else "NOT_MOUNTED"
            ),
            "state_gene_set_query": "ENABLED" if state_gene_sets else "NOT_MOUNTED",
            "clinical_km_query": "ENABLED" if clinical_km else "NOT_MOUNTED",
            "bulk_coexpression_query": (
                "ENABLED" if bulk_coexpression else "NOT_MOUNTED"
            ),
            "external_validation_query": (
                "ENABLED" if external_validation else "NOT_MOUNTED"
            ),
            "multimodal_fusion_query": (
                "ENABLED_SECONDARY_ONLY" if multimodal_fusion else "NOT_MOUNTED"
            ),
            "continuous_activity_query": (
                "ENABLED_FRESH_V32" if continuous_activity else "NOT_MOUNTED"
            ),
            "score_architecture_contract": "ENABLED",
        }
        if single_cell_formal23 is not None:
            # Do not expose the superseded 17-cancer key as a current
            # capability.  Keep a machine-readable alias map for clients
            # that still call the historical route names.
            payload.pop("single_cell_ucell_17c_query", None)
            payload.pop("single_cell_diagnostic_query", None)
            payload["legacy_aliases"] = {
                "single_cell_ucell_17c_query": "ROUTE_SHIM_TO_FORMAL23_DONOR_ACTIVITY",
                "single_cell_diagnostic_query": "TYPED_UNAVAILABLE_FORMAL23",
            }
        return payload

    @staging.get("/v3.2-staging/release")
    def release() -> dict[str, Any]:
        return {
            "release_id": engine.registry.release_id,
            "analysis_version": engine.registry.analysis_version,
            "environment": engine.registry.manifest["environment"],
            "production_deployed": engine.registry.manifest["production_deployed"],
            "release_ready": engine.registry.manifest["release_ready"],
            "all_non_null_predictions_newly_trained_v32": engine.registry.manifest[
                "all_non_null_predictions_newly_trained_v32"
            ],
            "all_published_results_generated_in_v32": engine.registry.manifest[
                "all_published_results_generated_in_v32"
            ],
            "module_statuses": {
                module_id: value["status"]
                for module_id, value in engine.registry.manifest["modules"].items()
            },
            "website_artifact_sha256": dict(engine.registry.artifact_hashes),
            "staging_query_capabilities": auxiliary.capability_status(),
            "sidecar_query_capabilities": {
                "unified_network": (
                    "ENABLED_UNIFIED_NETWORK" if network else "NOT_MOUNTED"
                ),
                "unified_network_independent_audit": (
                    "PASS_API_INTEGRATION_ACCEPTED_168_OF_168"
                    if network_audit
                    else "NOT_MOUNTED"
                ),
                "exact_pathway_browse": (
                    "ENABLED_CURRENT_V32_HASH_PINNED"
                    if gene_set_subtypes
                    else "NOT_MOUNTED"
                ),
                "gene_sets": (
                    "ENABLED_DETERMINISTIC_V32_DERIVED_HEAD"
                    if gene_set_subtypes
                    else "NOT_MOUNTED"
                ),
                "ranked_subtypes": (
                    "ENABLED_DETERMINISTIC_V32_DERIVED_HEAD"
                    if gene_set_subtypes
                    else "NOT_MOUNTED"
                ),
                "gene_set_ranked_subtype_independent_audit": (
                    f"PASS_API_INTEGRATION_ACCEPTED_{gene_set_subtypes.audit['pass_count']}_OF_{gene_set_subtypes.audit['pass_count']}"
                    if gene_set_subtypes
                    else "NOT_MOUNTED"
                ),
                "historical_artifact_remediation": (
                    "ENABLED_HASH_PINNED_WITH_INDEPENDENT_AUDIT"
                    if historical_remediation
                    else "NOT_MOUNTED"
                ),
                "clinical_translational_priority": (
                    "ENABLED_FRESH_V32_DERIVED_ARTIFACT"
                    if historical_remediation
                    else "NOT_MOUNTED"
                ),
                "mutation_subgroup": (
                    "ENABLED_RETAINED_NO_INCREMENT_DIAGNOSTIC"
                    if historical_remediation
                    else "NOT_MOUNTED"
                ),
                "cnv_coverage": (
                    "ENABLED_TYPED_NULL_WITH_REASON"
                    if directional_cnv
                    else "NOT_MOUNTED"
                ),
                "downloads": (
                    f"ENABLED_54_IDS_AUDITED_{download_catalog.audit['pass_count']}_OF_{download_catalog.audit['pass_count']}"
                    if download_catalog
                    else "NOT_MOUNTED"
                ),
                "physical_interaction": "ENABLED" if interaction else "NOT_MOUNTED",
                "physical_interaction_unmapped_partners": (
                    "ENABLED_HASH_PINNED" if interaction else "NOT_MOUNTED"
                ),
                "drug_structural_mechanism": "ENABLED" if drug_mechanism else "NOT_MOUNTED",
                "drug_response_actionability": (
                    "ENABLED_NATIVE_SPARSE" if drug_sparse else "NOT_MOUNTED"
                ),
                "evidence_confidence": "ENABLED" if evidence else "NOT_MOUNTED",
                "evidence_direction_probabilities": (
                    "ENABLED_THREE_CLASS_HASH_PINNED_TYPED_NULL"
                    if evidence_direction
                    else "NOT_MOUNTED"
                ),
                "experiment_perturbation": (
                    "ENABLED_FINAL_EVIDENCE_BRIDGE"
                    if experiment_perturbation_bridge
                    else "RAW_FACTS_ONLY_PENDING_FINAL_BRIDGE"
                    if experiment_perturbation
                    else "NOT_MOUNTED"
                ),
                "experiment_perturbation_raw_facts": (
                    "ENABLED_CONFIDENCE_ONLY_PENDING_ABLATION"
                    if experiment_perturbation
                    else "NOT_MOUNTED"
                ),
                "experiment_perturbation_evidence_bridge": (
                    "ENABLED_ALREADY_ABSORBED_IN_EVIDENCE_TRANSFORMER"
                    if experiment_perturbation_bridge
                    else "NOT_MOUNTED"
                ),
                "experiment_perturbation_bridge_independent_audit": (
                    "PASS_API_INTEGRATION_ACCEPTED"
                    if experiment_perturbation_bridge_audit
                    else "NOT_MOUNTED"
                ),
                "single_cell_exact_pathway_association": (
                    "ENABLED_FORMAL23_DONOR_ASSOCIATION"
                    if single_cell_formal23
                    else "ENABLED_FORMAL_FUSION_ADAPTER"
                    if single_cell_fusion
                    else "NOT_MOUNTED"
                ),
                "single_cell_expression_audit": (
                    "ENABLED_FORMAL23_DONOR_EXPRESSION"
                    if single_cell_formal23
                    else "ENABLED" if single_cell_expression else "NOT_MOUNTED"
                ),
                "single_cell_coverage_and_typed_gaps": (
                    "ENABLED_FORMAL23_COVERAGE_23_PLUS_10_TYPED_GAPS"
                    if single_cell_formal23
                    else "ENABLED_33_CANCER_COVERAGE_TYPED_GAPS_68_OF_68"
                    if single_cell_gap
                    else "NOT_MOUNTED"
                ),
                "single_cell_formal_context": (
                    "ENABLED_23_CANCER_DONOR_BLOCKED_HASH_PINNED_AUDITED"
                    if single_cell_formal23
                    else "ENABLED_17_CANCER_HASH_PINNED_AUDITED_25_OF_25"
                    if single_cell_formal_context
                    else "NOT_MOUNTED"
                ),
                "single_cell_formal_context_independent_audit": (
                    "PASS_HASH_PINNED_FORMAL23_AUTHORITY"
                    if single_cell_formal23
                    else "PASS_API_INTEGRATION_ACCEPTED_25_OF_25"
                    if single_cell_formal_context_audit
                    else "NOT_MOUNTED"
                ),
                "single_cell_formal23_query": (
                    "ENABLED_23_ELIGIBLE_PLUS_10_TYPED_UNAVAILABLE"
                    if single_cell_formal23
                    else "NOT_MOUNTED"
                ),
                "single_cell_formal23_donor_activity": (
                    "ENABLED_DONOR_AGGREGATED_ACTIVITY_23_ELIGIBLE"
                    if single_cell_formal23
                    else "NOT_MOUNTED"
                ),
                "single_cell_formal23_diagnostic": (
                    "TYPED_UNAVAILABLE_NO_PSEUDOTIME_OR_FIGURES"
                    if single_cell_formal23
                    else "NOT_MOUNTED"
                ),
                "bulk_expression_landscape": (
                    "ENABLED" if bulk_expression else "NOT_MOUNTED"
                ),
                "hnsc_cell_level_ucell_pilot": (
                    "ENABLED_HNSC_PILOT" if hnsc_ucell else "NOT_MOUNTED"
                ),
                # Legacy route names remain accepted by the API, but are
                # deliberately nested and marked as aliases so the release
                # surface cannot mistake them for the current formal-23
                # products.
                "legacy_aliases": {
                    "single_cell_ucell_17c": (
                        "ROUTE_SHIM_TO_FORMAL23_DONOR_ACTIVITY"
                        if single_cell_formal23
                        else "MOUNTED_LEGACY_CELL_LEVEL_UCELL"
                        if single_cell_ucell_17c
                        else "NOT_MOUNTED"
                    ),
                    "single_cell_diagnostic": (
                        "TYPED_UNAVAILABLE_FORMAL23"
                        if single_cell_formal23
                        else "MOUNTED_LEGACY_DIAGNOSTIC"
                        if single_cell_diagnostic
                        else "NOT_MOUNTED"
                    ),
                },
                "state_gene_sets": "ENABLED" if state_gene_sets else "NOT_MOUNTED",
                "clinical_kaplan_meier": "ENABLED" if clinical_km else "NOT_MOUNTED",
                "bulk_coexpression": (
                    "ENABLED" if bulk_coexpression else "NOT_MOUNTED"
                ),
                "external_validation": (
                    "ENABLED" if external_validation else "NOT_MOUNTED"
                ),
                "multimodal_secondary_scores": (
                    "ENABLED_SECONDARY_ONLY" if multimodal_fusion else "NOT_MOUNTED"
                ),
                "continuous_pathway_activity": (
                    "ENABLED_FRESH_V32" if continuous_activity else "NOT_MOUNTED"
                ),
            },
            "supersedes": list(engine.registry.manifest.get("supersedes", [])),
            "supersession_reason": engine.registry.manifest.get("supersession_reason"),
        }

    @staging.post("/v3.2-staging/enrichment/mixed-exact-pathway")
    def mixed_exact_pathway(request: MixedExactPathwayRequest) -> dict[str, Any]:
        try:
            return engine.query(
                request.members,
                cancer_id=request.cancer_id,
                top_k=request.top_k,
            )
        except MixedQueryError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @staging.get("/v3.2-staging/scoring/architecture")
    def scoring_architecture() -> dict[str, Any]:
        return score_architecture_manifest()

    @staging.get(
        "/v3.2-staging/historical-remediation/{capability_id}/capability"
    )
    def historical_remediation_capability(capability_id: str) -> dict[str, Any]:
        if historical_remediation is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 historical-artifact remediation release is not mounted",
            )
        try:
            return historical_remediation.capability_status(capability_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail=f"Unknown remediation capability: {capability_id}"
            ) from exc
        except HistoricalArtifactRemediationError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @staging.get("/v3.2-staging/clinical/translational-priority")
    def clinical_translational_priority(
        cancer_id: str,
        endpoint: str | None = None,
        subject_type: str | None = None,
        limit: int = Query(default=100, ge=1, le=1000),
    ) -> dict[str, Any]:
        if historical_remediation is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 Clinical translational-priority release is not mounted",
            )
        try:
            rows = historical_remediation.clinical_priority(
                cancer_id=cancer_id,
                endpoint=endpoint,
                subject_type=subject_type,
                limit=limit,
            )
        except (HistoricalArtifactRemediationError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "module": "clinical",
            "artifact_id": "v32_clinical_translational_priority",
            "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
            "changes_primary_ranking": False,
            "count": len(rows),
            "rows": rows,
        }

    @staging.get("/v3.2-staging/mutation/subgroups")
    def mutation_subgroups(
        cancer_id: str,
        lncrna_id: str | None = None,
        pathway_id: str | None = None,
        limit: int = Query(default=100, ge=1, le=1000),
    ) -> dict[str, Any]:
        if historical_remediation is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 retained Mutation subgroup release is not mounted",
            )
        try:
            rows = historical_remediation.mutation_subgroup(
                cancer_id=cancer_id,
                lncrna_id=lncrna_id,
                pathway_id=pathway_id,
                limit=limit,
            )
        except (HistoricalArtifactRemediationError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "module": "mutation",
            "artifact_id": "v32_mutation_subgroup_probability",
            "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
            "scientific_status": "NO_INCREMENT_DIAGNOSTIC_ONLY_RETAINED",
            "changes_primary_ranking": False,
            "count": len(rows),
            "rows": rows,
        }

    @staging.get("/v3.2-staging/cnv/coverage")
    def cnv_coverage(cancer_id: str | None = None) -> dict[str, Any]:
        if directional_cnv is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "V3.2 directional CNV replacement is pending formal binding; "
                    "the superseded historical coverage artifact is disabled"
                ),
            )
        try:
            return directional_cnv.coverage(cancer_id=cancer_id)
        except DirectionalCNVQueryInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except DirectionalCNVQueryAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    def _pending_cnv_download(download_id: str) -> dict[str, Any]:
        return {
            "download_id": download_id,
            "capability_ids": ["cnv"],
            "status": "PENDING_FORMAL_SUCCESS",
            "download_implemented": False,
            "data_present": False,
            "contract_complete": False,
            "model_version": "V3.2",
            "availability_encoding": "null_with_reason",
            "unavailable_fill_value": None,
            "supersedes_historical_combined_cnv": True,
            "unavailable_reason": (
                "DIRECTIONAL_CNV_33C_LOCAL_AUDIT_AND_FIVE_FOLD_OOF_"
                "FORMAL_BINDING_PENDING"
            ),
        }

    def _current_cnv_download(download_id: str) -> dict[str, Any]:
        if directional_cnv is None:
            return _pending_cnv_download(download_id)
        try:
            return directional_cnv.download_entry(download_id)
        except DirectionalCNVQueryInputError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except DirectionalCNVQueryAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    def _pending_single_cell_download(download_id: str) -> dict[str, Any]:
        historical = download_id in {
            "single_cell_pseudotime",
            "single_cell_figures",
        }
        return {
            "download_id": download_id,
            "capability_ids": ["single_cell"],
            "status": (
                "GAP_TYPED_UNAVAILABLE" if historical else "PENDING_FORMAL_SUCCESS"
            ),
            "download_implemented": False,
            "data_present": False,
            "contract_complete": historical,
            "model_version": "V3.2",
            "supersedes_historical_single_cell": True,
            "unavailable_reason": (
                "FORMAL23_PSEUDOTIME_OR_FIGURE_PRODUCT_NOT_GENERATED"
                if historical
                else "AUDITED_FORMAL23_SINGLE_CELL_SUCCESS_NOT_MOUNTED"
            ),
        }

    def _current_single_cell_download(download_id: str) -> dict[str, Any]:
        if single_cell_formal23 is None:
            return _pending_single_cell_download(download_id)
        try:
            return single_cell_formal23.download_entry(download_id)
        except SingleCellFormal23InputError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except SingleCellFormal23AssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @staging.get("/v3.2-staging/downloads")
    def downloads() -> dict[str, Any]:
        if download_catalog is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 independently audited download catalog is not mounted",
            )
        payload = download_catalog.public_catalog()
        entries = payload.get("catalog")
        if not isinstance(entries, list):
            raise HTTPException(
                status_code=503,
                detail="V3.2 download catalog has an invalid public shape",
            )
        by_id = {
            str(entry.get("download_id")): index
            for index, entry in enumerate(entries)
            if isinstance(entry, dict)
        }
        required_overlays = CNV_DOWNLOAD_IDS | SINGLE_CELL_SUPERSEDED_DOWNLOAD_IDS
        if not required_overlays.issubset(by_id):
            raise HTTPException(
                status_code=503,
                detail=(
                    "V3.2 download catalog lacks the complete corrected CNV/"
                    "single-cell contract IDs"
                ),
            )
        for download_id in sorted(CNV_DOWNLOAD_IDS):
            entries[by_id[download_id]] = _current_cnv_download(download_id)
        for download_id in sorted(SINGLE_CELL_SUPERSEDED_DOWNLOAD_IDS):
            entries[by_id[download_id]] = _current_single_cell_download(download_id)
        status_counts: dict[str, int] = {}
        for entry in entries:
            status = str(entry.get("status", "UNKNOWN"))
            status_counts[status] = status_counts.get(status, 0) + 1
        payload["status_counts"] = status_counts
        payload["directional_cnv_overlay"] = {
            "status": "READY" if directional_cnv else "PENDING_FORMAL_BINDING",
            "superseded_combined_cnv_exposed": False,
        }
        payload["single_cell_formal23_overlay"] = {
            "status": (
                "READY_23_PLUS_10_TYPED_UNAVAILABLE"
                if single_cell_formal23
                else "PENDING_FORMAL23_MOUNT"
            ),
            "historical_single_cell_files_exposed": False,
            "full_33_single_cell_coverage_claimed": False,
        }
        return payload

    @staging.get("/v3.2-staging/downloads/{download_id}")
    def download_entry(download_id: str) -> dict[str, Any]:
        if download_id in CNV_DOWNLOAD_IDS:
            entry = _current_cnv_download(download_id)
            if entry.get("status") == "READY_FILE":
                entry["download_url"] = (
                    f"/v3.2-staging/downloads/{download_id}/file"
                )
            return entry
        if download_id in SINGLE_CELL_SUPERSEDED_DOWNLOAD_IDS:
            entry = _current_single_cell_download(download_id)
            if entry.get("status") == "READY_PARTS":
                for index, part in enumerate(entry.get("parts", [])):
                    part["download_url"] = (
                        f"/v3.2-staging/downloads/{download_id}/parts/{index}"
                    )
            return entry
        if download_catalog is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 independently audited download catalog is not mounted",
            )
        try:
            entry = download_catalog.entry(download_id)
        except DownloadCatalogError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        status = entry.get("status")
        if status == "READY_FILE":
            entry["download_url"] = f"/v3.2-staging/downloads/{download_id}/file"
        elif status in {"READY_PARTS", "PARTIAL_READY_PARTS"}:
            for index, part in enumerate(entry.get("parts", [])):
                part["download_url"] = (
                    f"/v3.2-staging/downloads/{download_id}/parts/{index}"
                )
        return entry

    @staging.get("/v3.2-staging/downloads/{download_id}/file")
    def download_file(download_id: str):
        if download_id in CNV_DOWNLOAD_IDS:
            if directional_cnv is None:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"Download {download_id} is unavailable: corrected "
                        "directional CNV formal binding is pending"
                    ),
                )
            try:
                payload = directional_cnv.resolve_download(download_id)
            except DirectionalCNVQueryInputError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except DirectionalCNVQueryAssetError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            return FileResponse(
                payload["path"],
                filename=Path(str(payload["relative_name"])).name,
                headers={
                    "X-Content-SHA256": str(payload["sha256"]),
                    "X-CancerLncAtlas-Version": "V3.2",
                    "X-CNV-Generation": "DIRECTIONAL-CNV-ONLY",
                },
            )
        if download_id in SINGLE_CELL_SUPERSEDED_DOWNLOAD_IDS:
            entry = _current_single_cell_download(download_id)
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Download {download_id} is partitioned by cancer"
                    if entry.get("status") == "READY_PARTS"
                    else f"Download {download_id} is unavailable: "
                    f"{entry.get('unavailable_reason')}"
                ),
            )
        if download_catalog is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 independently audited download catalog is not mounted",
            )
        try:
            payload = download_catalog.resolve(download_id)
        except DownloadCatalogError as exc:
            message = str(exc)
            status_code = 409 if "is unavailable" in message else 503
            raise HTTPException(status_code=status_code, detail=message) from exc
        if payload.get("kind") != "file":
            raise HTTPException(
                status_code=409,
                detail=f"Download {download_id} is not a single-file payload",
            )
        return FileResponse(
            payload["path"],
            filename=Path(str(payload["relative_name"])).name,
            headers={
                "X-Content-SHA256": str(payload["sha256"]),
                "X-CancerLncAtlas-Version": "V3.2",
            },
        )

    @staging.get("/v3.2-staging/downloads/{download_id}/parts/{part_index}")
    def download_part(download_id: str, part_index: int):
        if download_id in CNV_DOWNLOAD_IDS:
            raise HTTPException(
                status_code=409,
                detail=f"Download {download_id} is a single-file payload",
            )
        if download_id in SINGLE_CELL_SUPERSEDED_DOWNLOAD_IDS:
            if single_cell_formal23 is None:
                entry = _pending_single_cell_download(download_id)
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"Download {download_id} is unavailable: "
                        f"{entry.get('unavailable_reason')}"
                    ),
                )
            entry = _current_single_cell_download(download_id)
            if entry.get("status") != "READY_PARTS":
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"Download {download_id} is unavailable: "
                        f"{entry.get('unavailable_reason')}"
                    ),
                )
            try:
                payload = single_cell_formal23.resolve_download_part(
                    download_id, part_index
                )
            except SingleCellFormal23InputError as exc:
                message = str(exc)
                status_code = 404 if "unknown" in message else 409
                raise HTTPException(status_code=status_code, detail=message) from exc
            except SingleCellFormal23AssetError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            return FileResponse(
                payload["path"],
                filename=Path(str(payload["relative_name"])).name,
                headers={
                    "X-Content-SHA256": str(payload["sha256"]),
                    "X-Partition-Tree-SHA256": str(payload["sha256_tree"]),
                    "X-Cancer-ID": str(payload["cancer_id"]),
                    "X-CancerLncAtlas-Version": "V3.2",
                    "X-Single-Cell-Generation": "FORMAL23-FRESH",
                },
            )
        if download_catalog is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 independently audited download catalog is not mounted",
            )
        try:
            payload = download_catalog.resolve_part(download_id, part_index)
        except DownloadCatalogError as exc:
            message = str(exc)
            if "Unknown download part index" in message:
                status_code = 404
            elif "is unavailable" in message or "not a partitioned payload" in message:
                status_code = 409
            else:
                status_code = 503
            raise HTTPException(status_code=status_code, detail=message) from exc
        if payload.get("kind") != "part":
            raise HTTPException(
                status_code=409,
                detail=f"Download {download_id} is not a partitioned payload",
            )
        return FileResponse(
            payload["path"],
            filename=Path(str(payload["relative_name"])).name,
            headers={
                "X-Content-SHA256": str(payload["sha256"]),
                "X-Partition-Tree-SHA256": str(payload["sha256_tree"]),
                "X-CancerLncAtlas-Version": "V3.2",
            },
        )

    @staging.get("/v3.2-staging/exact-pathway/capability")
    def exact_pathway_capability() -> dict[str, Any]:
        if gene_set_subtypes is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 audited exact-pathway/Gene Set/subtype release is not mounted",
            )
        return gene_set_subtypes.exact_pathway_capability()

    @staging.get("/v3.2-staging/exact-pathway/report")
    def exact_pathway_report_manifest() -> dict[str, Any]:
        if exact_pathway_report is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 exact-pathway report manifest is not mounted",
            )
        return exact_pathway_report.capability()

    @staging.get("/v3.2-staging/search")
    def exact_pathway_search(
        q: str = Query(min_length=2, max_length=100),
        limit: int = Query(default=20, ge=1, le=100),
    ) -> dict[str, Any]:
        if gene_set_subtypes is None:
            raise HTTPException(status_code=503, detail="V3.2 exact-pathway release is not mounted")
        try:
            return gene_set_subtypes.query_exact_pathway_search(query=q, limit=limit)
        except GeneSetSubtypeQueryInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except GeneSetSubtypeQueryAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @staging.get("/v3.2-staging/datasets")
    def exact_pathway_datasets() -> dict[str, Any]:
        if gene_set_subtypes is None:
            raise HTTPException(status_code=503, detail="V3.2 exact-pathway release is not mounted")
        return gene_set_subtypes.query_exact_pathway_datasets()

    @staging.get("/v3.2-staging/cancers")
    def exact_pathway_cancers() -> dict[str, Any]:
        if gene_set_subtypes is None:
            raise HTTPException(status_code=503, detail="V3.2 exact-pathway release is not mounted")
        return gene_set_subtypes.query_exact_pathway_cancers()

    @staging.get("/v3.2-staging/exact-pathway/associations")
    def exact_pathway_associations(
        cancer_id: str | None = None,
        lncrna_id: str | None = None,
        pathway_id: str | None = None,
        availability: bool | None = None,
        limit: int = Query(default=50, ge=1, le=GENE_SET_SUBTYPE_MAX_QUERY_LIMIT),
        offset: int = Query(default=0, ge=0, le=GENE_SET_SUBTYPE_MAX_QUERY_OFFSET),
    ) -> dict[str, Any]:
        if gene_set_subtypes is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 audited exact-pathway/Gene Set/subtype release is not mounted",
            )
        try:
            return gene_set_subtypes.query_exact_pathway_associations(
                cancer_id=cancer_id,
                lncrna_id=lncrna_id,
                pathway_id=pathway_id,
                availability=availability,
                limit=limit,
                offset=offset,
            )
        except GeneSetSubtypeQueryInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except GeneSetSubtypeQueryAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @staging.get("/v3.2-staging/exact-pathway/stats")
    def exact_pathway_stats(cancer_id: str | None = None) -> dict[str, Any]:
        if gene_set_subtypes is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 audited exact-pathway/Gene Set/subtype release is not mounted",
            )
        try:
            return gene_set_subtypes.query_exact_pathway_stats(cancer_id=cancer_id)
        except GeneSetSubtypeQueryInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except GeneSetSubtypeQueryAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @staging.get("/v3.2-staging/gene-sets")
    def gene_set_catalog(
        cancer_id: str | None = None,
        pathway_id: str | None = None,
        direction: str | None = None,
        min_member_count: int | None = Query(default=None, ge=1),
        limit: int = Query(default=50, ge=1, le=GENE_SET_SUBTYPE_MAX_QUERY_LIMIT),
        offset: int = Query(default=0, ge=0, le=GENE_SET_SUBTYPE_MAX_QUERY_OFFSET),
    ) -> dict[str, Any]:
        if gene_set_subtypes is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 audited Gene Set/ranked-subtype release is not mounted",
            )
        try:
            return gene_set_subtypes.query_gene_set_catalog(
                cancer_id=cancer_id,
                pathway_id=pathway_id,
                direction=direction,
                min_member_count=min_member_count,
                limit=limit,
                offset=offset,
            )
        except GeneSetSubtypeQueryInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except GeneSetSubtypeQueryAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @staging.get("/v3.2-staging/gene-sets/enrichment")
    def gene_set_enrichment(
        gene_set_id: str | None = None,
        cancer_id: str | None = None,
        pathway_id: str | None = None,
        min_support_fraction: float | None = Query(default=None, ge=0, le=1),
        limit: int = Query(default=50, ge=1, le=GENE_SET_SUBTYPE_MAX_QUERY_LIMIT),
        offset: int = Query(default=0, ge=0, le=GENE_SET_SUBTYPE_MAX_QUERY_OFFSET),
    ) -> dict[str, Any]:
        if gene_set_subtypes is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 audited Gene Set/ranked-subtype release is not mounted",
            )
        try:
            return gene_set_subtypes.query_gene_set_enrichment(
                gene_set_id=gene_set_id,
                cancer_id=cancer_id,
                pathway_id=pathway_id,
                min_support_fraction=min_support_fraction,
                limit=limit,
                offset=offset,
            )
        except GeneSetSubtypeQueryInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except GeneSetSubtypeQueryAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @staging.get("/v3.2-staging/gene-sets/{gene_set_id}")
    def gene_set_detail(
        gene_set_id: str,
        member_limit: int = Query(
            default=200, ge=1, le=GENE_SET_SUBTYPE_MAX_MEMBER_LIMIT
        ),
        member_offset: int = Query(
            default=0, ge=0, le=GENE_SET_SUBTYPE_MAX_QUERY_OFFSET
        ),
    ) -> dict[str, Any]:
        if gene_set_subtypes is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 audited Gene Set/ranked-subtype release is not mounted",
            )
        try:
            return gene_set_subtypes.query_gene_set_detail(
                gene_set_id=gene_set_id,
                member_limit=member_limit,
                member_offset=member_offset,
            )
        except GeneSetSubtypeQueryNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except GeneSetSubtypeQueryInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except GeneSetSubtypeQueryAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @staging.get("/v3.2-staging/ranked-subtypes/overview")
    def ranked_subtype_overview(cancer_id: str | None = None) -> dict[str, Any]:
        if gene_set_subtypes is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 audited Gene Set/ranked-subtype release is not mounted",
            )
        try:
            return gene_set_subtypes.query_subtype_overview(cancer_id=cancer_id)
        except GeneSetSubtypeQueryInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except GeneSetSubtypeQueryAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @staging.get("/v3.2-staging/ranked-subtypes/detail")
    def ranked_subtype_detail(
        cancer_id: str | None = None,
        pathway_id: str | None = None,
        classification_status: str | None = None,
        limit: int = Query(default=100, ge=1, le=GENE_SET_SUBTYPE_MAX_QUERY_LIMIT),
        offset: int = Query(default=0, ge=0, le=GENE_SET_SUBTYPE_MAX_QUERY_OFFSET),
    ) -> dict[str, Any]:
        if gene_set_subtypes is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 audited Gene Set/ranked-subtype release is not mounted",
            )
        try:
            return gene_set_subtypes.query_subtype_detail(
                cancer_id=cancer_id,
                pathway_id=pathway_id,
                classification_status=classification_status,
                limit=limit,
                offset=offset,
            )
        except GeneSetSubtypeQueryInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except GeneSetSubtypeQueryAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @staging.get("/v3.2-staging/ranked-subtypes/stability")
    def ranked_subtype_stability(
        pathway_id: str,
        cancer_id: str | None = None,
        limit: int = Query(default=200, ge=1, le=GENE_SET_SUBTYPE_MAX_QUERY_LIMIT),
        offset: int = Query(default=0, ge=0, le=GENE_SET_SUBTYPE_MAX_QUERY_OFFSET),
    ) -> dict[str, Any]:
        if gene_set_subtypes is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 audited Gene Set/ranked-subtype release is not mounted",
            )
        try:
            return gene_set_subtypes.query_subtype_stability(
                pathway_id=pathway_id,
                cancer_id=cancer_id,
                limit=limit,
                offset=offset,
            )
        except GeneSetSubtypeQueryInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except GeneSetSubtypeQueryAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @staging.get("/v3.2-staging/state")
    def state(
        cancer_id: str | None = None,
        lncrna_id: str | None = None,
        state_id: str | None = None,
        availability: bool | None = None,
        limit: int = Query(default=50, ge=1, le=MAX_QUERY_LIMIT),
        offset: int = Query(default=0, ge=0, le=MAX_QUERY_OFFSET),
    ) -> dict[str, Any]:
        try:
            return auxiliary.query_state(
                cancer_id=cancer_id,
                lncrna_id=lncrna_id,
                state_id=state_id,
                availability=availability,
                limit=limit,
                offset=offset,
            )
        except StagingQueryInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except StagingQueryAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @staging.get("/v3.2-staging/clinical")
    def clinical(
        clinical_endpoint: str,
        cancer_id: str | None = None,
        subject_type: str | None = None,
        subject_id: str | None = None,
        availability: bool | None = None,
        limit: int = Query(default=50, ge=1, le=MAX_QUERY_LIMIT),
        offset: int = Query(default=0, ge=0, le=MAX_QUERY_OFFSET),
    ) -> dict[str, Any]:
        try:
            return auxiliary.query_clinical(
                clinical_endpoint=clinical_endpoint,
                cancer_id=cancer_id,
                subject_type=subject_type,
                subject_id=subject_id,
                availability=availability,
                limit=limit,
                offset=offset,
            )
        except StagingQueryInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except StagingQueryAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @staging.get("/v3.2-staging/genomic")
    def genomic(
        modality: str,
        cancer_id: str | None = None,
        lncrna_id: str | None = None,
        pathway_id: str | None = None,
        availability: bool | None = None,
        limit: int = Query(default=50, ge=1, le=MAX_QUERY_LIMIT),
        offset: int = Query(default=0, ge=0, le=MAX_QUERY_OFFSET),
    ) -> dict[str, Any]:
        if modality.strip().upper() == "CNV":
            if directional_cnv is None:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "V3.2 directional CNV replacement is pending formal binding; "
                        "the superseded combined CNV score is disabled"
                    ),
                )
            if limit > DIRECTIONAL_CNV_MAX_QUERY_LIMIT:
                raise HTTPException(status_code=422, detail="limit is outside the allowed range")
            if offset > DIRECTIONAL_CNV_MAX_QUERY_OFFSET:
                raise HTTPException(status_code=422, detail="offset is outside the allowed range")
            try:
                return directional_cnv.query(
                    cancer_id=cancer_id,
                    lncrna_id=lncrna_id,
                    pathway_id=pathway_id,
                    availability=availability,
                    limit=limit,
                    offset=offset,
                )
            except DirectionalCNVQueryInputError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            except DirectionalCNVQueryAssetError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        try:
            return auxiliary.query_genomic(
                modality=modality,
                cancer_id=cancer_id,
                lncrna_id=lncrna_id,
                pathway_id=pathway_id,
                availability=availability,
                limit=limit,
                offset=offset,
            )
        except StagingQueryInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except StagingQueryAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    def _resolve_compat_lnc(value: str) -> str:
        """Resolve a legacy symbol/identifier without guessing or broadcasting."""

        classification = engine.classify([value])
        mapped = classification.get("lncRNA", [])
        if len(mapped) == 1:
            return str(mapped[0]["canonical_id"])
        if classification.get("ambiguous"):
            raise HTTPException(
                status_code=422,
                detail={
                    "status": "ambiguous_identifier",
                    "input": value,
                    "classification": classification,
                },
            )
        if classification.get("protein_coding_gene"):
            raise HTTPException(
                status_code=422,
                detail=f"Identifier is protein-coding, not lncRNA: {value}",
            )
        raise HTTPException(status_code=404, detail=f"lncRNA not found: {value}")

    def _query_v32_mutation(
        *,
        cancer_id: str | None = None,
        lncrna_id: str | None = None,
        pathway_id: str | None = None,
        availability: bool | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        try:
            result = auxiliary.query_genomic(
                modality="MUTATION",
                cancer_id=cancer_id,
                lncrna_id=lncrna_id,
                pathway_id=pathway_id,
                availability=availability,
                limit=limit,
                offset=0,
            )
        except StagingQueryInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except StagingQueryAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        provenance = result.get("provenance", {})
        freshly_trained = (
            provenance.get("new_training_attestation") is True
            and provenance.get("old_checkpoint_loaded") is False
            and provenance.get("old_predictions_used_as_features") is False
            and provenance.get("old_rankings_used_as_outputs") is False
        )
        if not freshly_trained:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Mutation compatibility query requires a complete fresh-V3.2 "
                    "training attestation"
                ),
            )
        rows = []
        for source in result.get("results", []):
            row = dict(source)
            row["mutation_adjusted_probability"] = row.get(
                "context_probability"
            )
            row["probability_semantics"] = (
                "V3.2 mutation_context_probability; diagnostic evidence only"
            )
            rows.append(row)
        result["results"] = rows
        result["status"] = "available" if result.get("total_rows", 0) else "no_evidence"
        result["request_contract"] = "V3.2_MUTATION_COMPATIBILITY_QUERY_V1"
        result["new_training"] = freshly_trained
        result["old_predictions_used"] = False
        result["changes_primary_ranking"] = False
        result["legacy_v27_path_only"] = True
        return result

    def _mutation_compatibility_status() -> dict[str, Any]:
        capability = auxiliary.capability_status().get("mutation_cnv", {})
        module = registry.manifest.get("modules", {}).get("mutation_cnv", {})
        fresh_attestation = (
            module.get("new_training_attestation") is True
            and module.get("old_checkpoint_loaded") is False
            and module.get("old_predictions_used_as_features") is False
            and module.get("old_rankings_used_as_outputs") is False
        )
        enabled = bool(capability.get("enabled", False) and fresh_attestation)
        return {
            "status": "available" if enabled else "release_unavailable",
            "mutation_release_available": enabled,
            "mutation_release_status": (
                "available" if enabled else "release_unavailable"
            ),
            "mutation_model_release": module.get(
                "analysis_version", registry.analysis_version
            ),
            "release_id": registry.release_id,
            "release_registry_sha256": registry.registry_sha256,
            "new_training": fresh_attestation,
            "old_predictions_used": False,
            "changes_primary_ranking": False,
            "legacy_v27_release_retired": True,
            "compatibility_source": "V3.2 hash-bound mutation_cnv staging module",
            "capability": capability,
        }

    @staging.get("/api/site/mutation/status")
    def site_mutation_status_v32_compatibility() -> dict[str, Any]:
        return _mutation_compatibility_status()

    @staging.get("/v2.7/health")
    def mutation_v27_health_v32_compatibility() -> dict[str, Any]:
        payload = _mutation_compatibility_status()
        payload["compatibility_route"] = "/v2.7/health"
        return payload

    @staging.get("/v2.7/version")
    def mutation_v27_version_v32_compatibility() -> dict[str, Any]:
        return {
            **_mutation_compatibility_status(),
            "analysis_version": registry.analysis_version,
            "reference_genome": "GRCh38",
            "compatibility_route": "/v2.7/version",
        }

    @staging.get("/api/site/mutation/cancer/{cancer_id}")
    def site_mutation_cancer_v32_compatibility(
        cancer_id: str,
        limit: int = Query(default=100, ge=1, le=MAX_QUERY_LIMIT),
    ) -> dict[str, Any]:
        result = _query_v32_mutation(cancer_id=cancer_id, limit=limit)
        normalized_cancer = str(cancer_id).strip().upper()
        return {
            "status": result["status"],
            "cancer_id": normalized_cancer,
            "summary": {
                "cancer_id": normalized_cancer,
                "n_v32_mutation_context_relations": result["total_rows"],
            },
            "variant_types": [],
            "gene_frequency": [],
            "lncrna_pathways": result["results"],
            "drug_context": [],
            "component_status": {
                "lncrna_pathways": "available_v32_new_training",
                "variant_types": "not_in_v32_mutation_context_contract",
                "gene_frequency": "not_in_v32_mutation_context_contract",
                "drug_context": "separate_v32_drug_module",
            },
            "v32_query": result,
        }

    @staging.get("/api/site/mutation/lncrna/{lncrna}")
    def site_mutation_lncrna_v32_compatibility(
        lncrna: str,
        cancer: str | None = None,
        limit: int = Query(default=100, ge=1, le=MAX_QUERY_LIMIT),
    ) -> dict[str, Any]:
        lncrna_id = _resolve_compat_lnc(lncrna)
        result = _query_v32_mutation(
            cancer_id=cancer,
            lncrna_id=lncrna_id,
            limit=limit,
        )
        return {
            "status": result["status"],
            "lncrna_id": lncrna_id,
            "cancer_id": str(cancer).strip().upper() if cancer else None,
            "summary": [],
            "pathways": result["results"],
            "positions": [],
            "semantics": {
                "absence_is_wildtype": False,
                "locus_overlap": "No functional locus-overlap effect is claimed",
                "context_probability": (
                    "Diagnostic V3.2 evidence; primary ranking unchanged"
                ),
            },
            "v32_query": result,
        }

    @staging.get("/v2.7/mutation/cancer/{cancer_id}")
    def mutation_v27_cancer_v32_compatibility(
        cancer_id: str,
        limit: int = Query(default=100, ge=1, le=MAX_QUERY_LIMIT),
    ) -> dict[str, Any]:
        return site_mutation_cancer_v32_compatibility(cancer_id, limit)

    @staging.get("/v2.7/mutation/lncrna/{lncrna_id}")
    def mutation_v27_lncrna_v32_compatibility(
        lncrna_id: str,
        cancer: str | None = None,
        limit: int = Query(default=100, ge=1, le=MAX_QUERY_LIMIT),
    ) -> dict[str, Any]:
        return site_mutation_lncrna_v32_compatibility(lncrna_id, cancer, limit)

    @staging.post("/v2.7/predict/mutation-context")
    def mutation_v27_predict_v32_compatibility(
        request: LegacyMutationContextRequest,
    ) -> dict[str, Any]:
        if request.pathway_family_id and not request.pathway_id:
            raise HTTPException(
                status_code=422,
                detail=(
                    "V3.2 requires an exact pathway_id; pathway-family broadcast "
                    "is intentionally disabled"
                ),
            )
        lncrna_id = (
            _resolve_compat_lnc(request.lncrna_id)
            if request.lncrna_id
            else None
        )
        return _query_v32_mutation(
            cancer_id=request.cancer_id,
            lncrna_id=lncrna_id,
            pathway_id=request.pathway_id,
            availability=request.availability,
            limit=request.top_k,
        )

    @staging.post("/v3.2-staging/predict/mutation-context")
    def mutation_context(request: MutationContextRequest) -> dict[str, Any]:
        """Retain the historical Mutation POST workflow on fresh V3.2 rows."""

        result = _query_v32_mutation(
            cancer_id=request.cancer_id,
            lncrna_id=request.lncrna_id,
            pathway_id=request.pathway_id,
            availability=request.availability,
            limit=request.top_k,
        )
        result["request_contract"] = "V3.2_MUTATION_CONTEXT_POST_V1"
        result["mutation_retained_despite_no_increment"] = True
        result["changes_primary_ranking"] = False
        result["old_predictions_used"] = False
        return result

    @staging.get("/v3.2-staging/network/capability")
    def unified_network_capability() -> dict[str, Any]:
        if network is None or network_audit is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 audited unified Network release is not mounted",
            )
        return {
            "status": "ENABLED_UNIFIED_NETWORK",
            "module_id": "unified_network",
            "analysis_version": network.manifest["analysis_version"],
            "manifest_path": str(network.manifest_path),
            "manifest_sha256": network.manifest_sha256,
            "release_ready": False,
            "production_deployed": False,
            "accepted_for_api_integration": True,
            "independent_audit": dict(network_audit),
            "counts": dict(network.manifest["counts"]),
            "node_types": list(network.manifest["node_types"]),
            "edge_types": list(network.manifest["edge_types"]),
            "primary_frozen": network.manifest["primary_frozen"],
            "primary_ranking_unchanged": network.manifest[
                "primary_ranking_unchanged"
            ],
            "native_expert_probabilities_public": network.manifest[
                "native_expert_probabilities_public"
            ],
            "native_missingness_encoding": network.manifest[
                "native_missingness_encoding"
            ],
            "family_to_exact_broadcast": network.manifest[
                "family_to_exact_broadcast"
            ],
            "experiment_separate_fusion_created": network.manifest[
                "experiment_separate_fusion_created"
            ],
        }

    @staging.get("/v3.2-staging/network/lncrna/{lncrna_id}")
    def unified_network_lncrna_associations(
        lncrna_id: str,
        cancer_id: str | None = None,
        pathway_id: str | None = None,
        limit: int = Query(default=100, ge=1, le=NETWORK_MAX_QUERY_LIMIT),
        offset: int = Query(default=0, ge=0, le=NETWORK_MAX_QUERY_OFFSET),
    ) -> dict[str, Any]:
        if network is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 audited unified Network release is not mounted",
            )
        try:
            return network.query_model_associations(
                lncrna_id=lncrna_id,
                cancer_id=cancer_id,
                pathway_id=pathway_id,
                limit=limit,
                offset=offset,
            )
        except NetworkQueryInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except NetworkQueryAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @staging.get("/v3.2-staging/network/neighborhood")
    def unified_network_neighborhood(
        node_id: str,
        node_type: str | None = None,
        edge_type: str | None = None,
        cancer_id: str | None = None,
        limit: int = Query(default=100, ge=1, le=NETWORK_MAX_QUERY_LIMIT),
        offset: int = Query(default=0, ge=0, le=NETWORK_MAX_QUERY_OFFSET),
    ) -> dict[str, Any]:
        if network is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 audited unified Network release is not mounted",
            )
        try:
            return network.query_neighborhood(
                node_id=node_id,
                node_type=node_type,
                edge_type=edge_type,
                cancer_id=cancer_id,
                limit=limit,
                offset=offset,
            )
        except NetworkQueryInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except NetworkQueryAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @staging.get("/v3.2-staging/interaction/relationships")
    def interaction_relationships(
        lncrna_id: str,
        cancer_scope: str | None = None,
        partner_gene_id: str | None = None,
        limit: int = Query(default=50, ge=1, le=INTERACTION_MAX_QUERY_LIMIT),
        offset: int = Query(default=0, ge=0, le=INTERACTION_MAX_QUERY_OFFSET),
    ) -> dict[str, Any]:
        if interaction is None:
            raise HTTPException(status_code=503, detail="V3.2 physical interaction release is not mounted")
        try:
            return interaction.query_relationships(
                lncrna_id=lncrna_id,
                cancer_scope=cancer_scope,
                partner_gene_id=partner_gene_id,
                limit=limit,
                offset=offset,
            )
        except InteractionQueryInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except InteractionQueryAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @staging.get("/v3.2-staging/interaction/exact-pathway-enrichment")
    def interaction_enrichment(
        lncrna_id: str,
        cancer_scope: str | None = None,
        max_fdr: float | None = Query(default=None, ge=0, le=1),
        limit: int = Query(default=50, ge=1, le=INTERACTION_MAX_QUERY_LIMIT),
        offset: int = Query(default=0, ge=0, le=INTERACTION_MAX_QUERY_OFFSET),
    ) -> dict[str, Any]:
        if interaction is None:
            raise HTTPException(status_code=503, detail="V3.2 physical interaction release is not mounted")
        try:
            return interaction.query_enrichment(
                lncrna_id=lncrna_id,
                cancer_scope=cancer_scope,
                max_fdr=max_fdr,
                limit=limit,
                offset=offset,
            )
        except InteractionQueryInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except InteractionQueryAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @staging.get("/v3.2-staging/interaction/evidence")
    def interaction_evidence(
        relationship_id: str,
        limit: int = Query(default=200, ge=1, le=INTERACTION_MAX_QUERY_LIMIT),
        offset: int = Query(default=0, ge=0, le=INTERACTION_MAX_QUERY_OFFSET),
    ) -> dict[str, Any]:
        if interaction is None:
            raise HTTPException(status_code=503, detail="V3.2 physical interaction release is not mounted")
        try:
            return interaction.query_evidence(
                relationship_id=relationship_id,
                limit=limit,
                offset=offset,
            )
        except InteractionQueryInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except InteractionQueryAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @staging.get("/v3.2-staging/interaction/unmapped-partners")
    def interaction_unmapped_partners(
        lncrna_id: str | None = None,
        cancer_scope: str | None = None,
        partner_gene_id: str | None = None,
        limit: int = Query(default=50, ge=1, le=INTERACTION_MAX_QUERY_LIMIT),
        offset: int = Query(default=0, ge=0, le=INTERACTION_MAX_QUERY_OFFSET),
    ) -> dict[str, Any]:
        if interaction is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 physical interaction release is not mounted",
            )
        try:
            return interaction.query_unmapped_partners(
                lncrna_id=lncrna_id,
                cancer_scope=cancer_scope,
                partner_gene_id=partner_gene_id,
                limit=limit,
                offset=offset,
            )
        except InteractionQueryInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except InteractionQueryAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @staging.get("/v3.2-staging/drug/structural-mechanisms")
    def drug_structural_mechanisms(
        lncrna_id: str,
        cancer_id: str | None = None,
        drug_id: str | None = None,
        pathway_id: str | None = None,
        target_gene_id: str | None = None,
        min_probability: float | None = Query(default=None, ge=0, le=1),
        limit: int = Query(default=50, ge=1, le=DRUG_MECHANISM_MAX_QUERY_LIMIT),
        offset: int = Query(default=0, ge=0, le=DRUG_MECHANISM_MAX_QUERY_OFFSET),
    ) -> dict[str, Any]:
        if drug_mechanism is None:
            raise HTTPException(status_code=503, detail="V3.2 Drug mechanism release is not mounted")
        try:
            return drug_mechanism.query_mechanisms(
                lncrna_id=lncrna_id,
                cancer_id=cancer_id,
                drug_id=drug_id,
                pathway_id=pathway_id,
                target_gene_id=target_gene_id,
                min_probability=min_probability,
                limit=limit,
                offset=offset,
            )
        except DrugMechanismQueryInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except DrugMechanismQueryAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @staging.get("/v3.2-staging/drug/structural-mechanisms/download-manifest")
    def drug_structural_mechanism_download_manifest() -> dict[str, Any]:
        if drug_mechanism is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 Drug mechanism release is not mounted",
            )
        try:
            return drug_mechanism.download_manifest()
        except DrugMechanismQueryAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @staging.get("/v3.2-staging/drug/structural-mechanisms/download")
    def drug_structural_mechanism_download() -> FileResponse:
        if drug_mechanism is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 Drug mechanism release is not mounted",
            )
        try:
            resolved = drug_mechanism.resolve_download()
        except DrugMechanismQueryAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return FileResponse(
            resolved["path"],
            filename=resolved["download_name"],
            media_type="application/octet-stream",
            headers={
                "X-V32-SHA256": resolved["sha256"],
                "X-V32-Manifest-SHA256": resolved["manifest_sha256"],
            },
        )

    @staging.get("/v3.2-staging/drug/response-actionability")
    def drug_response_actionability(
        cancer_id: str,
        lncrna_id: str,
        drug_id: str,
    ) -> dict[str, Any]:
        if drug_sparse is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 native sparse Drug response release is not mounted",
            )
        try:
            return dict(drug_sparse.resolve(cancer_id, lncrna_id, drug_id))
        except DrugSparseInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except DrugSparseAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @staging.get("/v3.2-staging/evidence/confidence")
    def evidence_confidence(
        lncrna_id: str,
        cancer_id: str | None = None,
        pathway_id: str | None = None,
        availability: bool | None = None,
        min_confidence: float | None = Query(default=None, ge=0, le=1),
        limit: int = Query(default=50, ge=1, le=EVIDENCE_MAX_QUERY_LIMIT),
        offset: int = Query(default=0, ge=0, le=EVIDENCE_MAX_QUERY_OFFSET),
    ) -> dict[str, Any]:
        if evidence is None:
            raise HTTPException(status_code=503, detail="V3.2 Evidence binding is not mounted")
        try:
            return evidence.query_confidence(
                lncrna_id=lncrna_id,
                cancer_id=cancer_id,
                pathway_id=pathway_id,
                availability=availability,
                min_confidence=min_confidence,
                limit=limit,
                offset=offset,
            )
        except EvidenceQueryInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except EvidenceQueryAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @staging.get("/v3.2-staging/evidence/direction/capability")
    def evidence_direction_capability() -> dict[str, Any]:
        if evidence_direction is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "V3.2 independently audited Evidence direction probability "
                    "release is not mounted"
                ),
            )
        try:
            return evidence_direction.capability()
        except EvidenceDirectionQueryAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @staging.get("/v3.2-staging/evidence/direction/probabilities")
    def evidence_direction_probabilities(
        cancer_id: str | None = None,
        lncrna_id: str | None = None,
        pathway_id: str | None = None,
        available: bool | None = None,
        limit: int = Query(default=100, ge=1, le=EVIDENCE_DIRECTION_MAX_QUERY_LIMIT),
        offset: int = Query(default=0, ge=0, le=EVIDENCE_DIRECTION_MAX_QUERY_OFFSET),
    ) -> dict[str, Any]:
        if evidence_direction is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "V3.2 independently audited Evidence direction probability "
                    "release is not mounted"
                ),
            )
        try:
            return evidence_direction.query(
                cancer_id=cancer_id,
                lncrna_id=lncrna_id,
                pathway_id=pathway_id,
                available=available,
                limit=limit,
                offset=offset,
            )
        except EvidenceDirectionQueryInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except EvidenceDirectionQueryAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @staging.get("/v3.2-staging/evidence/events")
    def evidence_events(
        cancer_id: str,
        lncrna_id: str,
        pathway_id: str,
        limit: int = Query(default=200, ge=1, le=EVIDENCE_MAX_QUERY_LIMIT),
        offset: int = Query(default=0, ge=0, le=EVIDENCE_MAX_QUERY_OFFSET),
    ) -> dict[str, Any]:
        if evidence is None:
            raise HTTPException(status_code=503, detail="V3.2 Evidence binding is not mounted")
        try:
            return evidence.query_events(
                cancer_id=cancer_id,
                lncrna_id=lncrna_id,
                pathway_id=pathway_id,
                limit=limit,
                offset=offset,
            )
        except EvidenceQueryInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except EvidenceQueryAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @staging.get("/v3.2-staging/experiment/perturbation/capability")
    def experiment_perturbation_capability() -> dict[str, Any]:
        if experiment_perturbation is None and experiment_perturbation_bridge is None:
            raise HTTPException(
                status_code=503,
                detail="No V3.2 experiment perturbation binding is mounted",
            )
        return {
            "module_id": "experiment_perturbation",
            "final_status": (
                "ALREADY_ABSORBED_IN_EVIDENCE_TRANSFORMER"
                if experiment_perturbation_bridge
                else "RAW_FACTS_ONLY_PENDING_FINAL_BRIDGE"
            ),
            "raw_facts": (
                experiment_perturbation.capability_status()
                if experiment_perturbation
                else {"status": "NOT_MOUNTED"}
            ),
            "final_evidence_bridge": (
                {
                    **experiment_perturbation_bridge.capability_status(),
                    "independent_post_audit": experiment_perturbation_bridge_audit,
                }
                if experiment_perturbation_bridge
                else {"status": "NOT_MOUNTED"}
            ),
            "raw_pending_does_not_imply_final_integration": True,
        }

    @staging.get("/v3.2-staging/experiment/perturbation/raw-facts/capability")
    def experiment_perturbation_raw_facts_capability() -> dict[str, Any]:
        if experiment_perturbation is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 raw experiment perturbation binding is not mounted",
            )
        return experiment_perturbation.capability_status()

    @staging.get("/v3.2-staging/experiment/perturbation/raw-facts/exact-pathway")
    @staging.get("/v3.2-staging/experiment/perturbation/exact-pathway")
    def experiment_perturbation_exact_pathway(
        cancer_id: str | None = None,
        lncrna_id: str | None = None,
        pathway_id: str | None = None,
        partner_id: str | None = None,
        direction_class: str | None = None,
        limit: int = Query(
            default=50,
            ge=1,
            le=EXPERIMENT_PERTURBATION_MAX_QUERY_LIMIT,
        ),
        offset: int = Query(
            default=0,
            ge=0,
            le=EXPERIMENT_PERTURBATION_MAX_QUERY_OFFSET,
        ),
    ) -> dict[str, Any]:
        if experiment_perturbation is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 experiment perturbation binding is not mounted",
            )
        try:
            return experiment_perturbation.query_exact_pathway_facts(
                cancer_id=cancer_id,
                lncrna_id=lncrna_id,
                pathway_id=pathway_id,
                partner_id=partner_id,
                direction_class=direction_class,
                limit=limit,
                offset=offset,
            )
        except ExperimentPerturbationQueryInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ExperimentPerturbationQueryAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @staging.get("/v3.2-staging/experiment/perturbation/evidence-bridge/capability")
    def experiment_perturbation_bridge_capability() -> dict[str, Any]:
        if experiment_perturbation_bridge is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 final experiment/Evidence bridge is not mounted",
            )
        return {
            **experiment_perturbation_bridge.capability_status(),
            "independent_post_audit": experiment_perturbation_bridge_audit,
        }

    @staging.get("/v3.2-staging/experiment/perturbation/evidence-bridge")
    def experiment_perturbation_bridge_exact_keys(
        cancer_id: str | None = None,
        lncrna_id: str | None = None,
        pathway_id: str | None = None,
        limit: int = Query(default=50, ge=1, le=EXPERIMENT_BRIDGE_MAX_QUERY_LIMIT),
        offset: int = Query(default=0, ge=0, le=EXPERIMENT_BRIDGE_MAX_QUERY_OFFSET),
    ) -> dict[str, Any]:
        if experiment_perturbation_bridge is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 final experiment/Evidence bridge is not mounted",
            )
        try:
            return experiment_perturbation_bridge.query_exact_keys(
                cancer_id=cancer_id,
                lncrna_id=lncrna_id,
                pathway_id=pathway_id,
                limit=limit,
                offset=offset,
            )
        except ExperimentPerturbationBridgeInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ExperimentPerturbationBridgeAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @staging.get("/v3.2-staging/experiment/perturbation/evidence-bridge/events")
    def experiment_perturbation_bridge_events(
        cancer_id: str | None = None,
        lncrna_id: str | None = None,
        pathway_id: str | None = None,
        partner_id: str | None = None,
        limit: int = Query(default=50, ge=1, le=EXPERIMENT_BRIDGE_MAX_QUERY_LIMIT),
        offset: int = Query(default=0, ge=0, le=EXPERIMENT_BRIDGE_MAX_QUERY_OFFSET),
    ) -> dict[str, Any]:
        if experiment_perturbation_bridge is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 final experiment/Evidence bridge is not mounted",
            )
        try:
            return experiment_perturbation_bridge.query_native_events(
                cancer_id=cancer_id,
                lncrna_id=lncrna_id,
                pathway_id=pathway_id,
                partner_id=partner_id,
                limit=limit,
                offset=offset,
            )
        except ExperimentPerturbationBridgeInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ExperimentPerturbationBridgeAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @staging.get("/v3.2-staging/single-cell/context/capability")
    def single_cell_formal_context_capability() -> dict[str, Any]:
        if single_cell_formal23 is not None:
            return single_cell_formal23.capability_status()
        if single_cell_formal_context is None or single_cell_formal_context_audit is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 independently audited formal single-cell context is not mounted",
            )
        return {
            **single_cell_formal_context.capability_status(),
            "independent_audit": single_cell_formal_context_audit,
        }

    @staging.get("/v3.2-staging/single-cell/context/lncrna-celltype")
    def single_cell_formal_context_lncrna_celltype(
        cancer_id: str,
        lncrna_id: str | None = None,
        cell_type: str | None = None,
        compartment: str | None = None,
        availability: str = "ALL",
        limit: int = Query(
            default=100, ge=1, le=SINGLE_CELL_FORMAL_CONTEXT_MAX_QUERY_LIMIT
        ),
        offset: int = Query(
            default=0, ge=0, le=SINGLE_CELL_FORMAL_CONTEXT_MAX_QUERY_OFFSET
        ),
    ) -> dict[str, Any]:
        if single_cell_formal23 is not None:
            try:
                return single_cell_formal23.query_lncrna_celltype(
                    cancer_id=cancer_id,
                    lncrna_id=lncrna_id,
                    cell_type=cell_type,
                    compartment=compartment,
                    availability=availability,
                    limit=limit,
                    offset=offset,
                )
            except SingleCellFormal23InputError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            except SingleCellFormal23AssetError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        if single_cell_formal_context is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 independently audited formal single-cell context is not mounted",
            )
        try:
            return single_cell_formal_context.query_lncrna_celltype(
                cancer_id=cancer_id,
                lncrna_id=lncrna_id,
                cell_type=cell_type,
                compartment=compartment,
                availability=availability,
                limit=limit,
                offset=offset,
            )
        except SingleCellFormalContextInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except SingleCellFormalContextAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @staging.get("/v3.2-staging/single-cell/context/pathway-activity")
    def single_cell_formal_context_pathway_activity(
        cancer_id: str,
        pathway_id: str | None = None,
        donor_id: str | None = None,
        cell_type: str | None = None,
        compartment: str | None = None,
        availability: str = "ALL",
        limit: int = Query(
            default=100, ge=1, le=SINGLE_CELL_FORMAL_CONTEXT_MAX_QUERY_LIMIT
        ),
        offset: int = Query(
            default=0, ge=0, le=SINGLE_CELL_FORMAL_CONTEXT_MAX_QUERY_OFFSET
        ),
    ) -> dict[str, Any]:
        if single_cell_formal23 is not None:
            if donor_id is not None:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "donor_id filtering is not exposed by the public formal-23 "
                        "donor-aggregated release"
                    ),
                )
            try:
                return single_cell_formal23.query_pathway_activity(
                    cancer_id=cancer_id,
                    pathway_id=pathway_id,
                    cell_type=cell_type,
                    compartment=compartment,
                    availability=availability,
                    limit=limit,
                    offset=offset,
                )
            except SingleCellFormal23InputError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            except SingleCellFormal23AssetError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        if single_cell_formal_context is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 independently audited formal single-cell context is not mounted",
            )
        try:
            return single_cell_formal_context.query_pathway_activity(
                cancer_id=cancer_id,
                pathway_id=pathway_id,
                donor_id=donor_id,
                cell_type=cell_type,
                compartment=compartment,
                availability=availability,
                limit=limit,
                offset=offset,
            )
        except SingleCellFormalContextInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except SingleCellFormalContextAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @staging.get("/v3.2-staging/single-cell/context/celltype-predictions")
    def single_cell_formal_context_celltype_predictions(
        cancer_id: str,
        lncrna_id: str | None = None,
        pathway_id: str | None = None,
        cell_type: str | None = None,
        compartment: str | None = None,
        availability: str = "ALL",
        limit: int = Query(
            default=100, ge=1, le=SINGLE_CELL_FORMAL_CONTEXT_MAX_QUERY_LIMIT
        ),
        offset: int = Query(
            default=0, ge=0, le=SINGLE_CELL_FORMAL_CONTEXT_MAX_QUERY_OFFSET
        ),
    ) -> dict[str, Any]:
        if single_cell_formal23 is not None:
            if cell_type is not None:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "the formal-23 association evidence is compartment-resolved, "
                        "not cell-type-resolved"
                    ),
                )
            try:
                return single_cell_formal23.query_associations(
                    cancer_id=cancer_id,
                    lncrna_id=lncrna_id,
                    pathway_id=pathway_id,
                    compartment=compartment,
                    availability=availability,
                    limit=limit,
                    offset=offset,
                )
            except SingleCellFormal23InputError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            except SingleCellFormal23AssetError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        if single_cell_formal_context is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 independently audited formal single-cell context is not mounted",
            )
        try:
            return single_cell_formal_context.query_celltype_predictions(
                cancer_id=cancer_id,
                lncrna_id=lncrna_id,
                pathway_id=pathway_id,
                cell_type=cell_type,
                compartment=compartment,
                availability=availability,
                limit=limit,
                offset=offset,
            )
        except SingleCellFormalContextInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except SingleCellFormalContextAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @staging.get("/v3.2-staging/single-cell/exact-pathway/capability")
    def single_cell_exact_pathway_capability() -> dict[str, Any]:
        if single_cell_formal23 is not None:
            return single_cell_formal23.capability_status()
        if single_cell_fusion is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 formal single-cell fusion binding is not mounted",
            )
        return single_cell_fusion.capability_status()

    @staging.get("/v3.2-staging/single-cell/exact-pathway-associations")
    def single_cell_exact_pathway_associations(
        cancer_id: str | None = None,
        lncrna_id: str | None = None,
        pathway_id: str | None = None,
        availability: bool | None = None,
        min_probability: float | None = Query(default=None, ge=0, le=1),
        limit: int = Query(default=50, ge=1, le=SINGLE_CELL_FUSION_MAX_QUERY_LIMIT),
        offset: int = Query(default=0, ge=0, le=SINGLE_CELL_FUSION_MAX_QUERY_OFFSET),
    ) -> dict[str, Any]:
        if single_cell_formal23 is not None:
            if cancer_id is None:
                raise HTTPException(
                    status_code=422,
                    detail="cancer_id is required by the partitioned formal-23 release",
                )
            if min_probability is not None:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "min_probability is unavailable because formal-23 exposes "
                        "donor-pseudobulk association evidence, not a learned probability"
                    ),
                )
            typed_availability = (
                "ALL"
                if availability is None
                else "AVAILABLE"
                if availability
                else "UNAVAILABLE"
            )
            try:
                return single_cell_formal23.query_associations(
                    cancer_id=cancer_id,
                    lncrna_id=lncrna_id,
                    pathway_id=pathway_id,
                    availability=typed_availability,
                    limit=limit,
                    offset=offset,
                )
            except SingleCellFormal23InputError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            except SingleCellFormal23AssetError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        if single_cell_fusion is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 formal single-cell fusion binding is not mounted",
            )
        try:
            return single_cell_fusion.query_associations(
                cancer_id=cancer_id,
                lncrna_id=lncrna_id,
                pathway_id=pathway_id,
                availability=availability,
                min_probability=min_probability,
                limit=limit,
                offset=offset,
            )
        except SingleCellFusionQueryInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except SingleCellFusionQueryAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @staging.get("/v3.2-staging/single-cell/lncrna-expression-summary")
    def single_cell_lncrna_expression_summary(
        cancer_id: str | None = None,
        formal_input_eligible: bool | None = None,
        limit: int = Query(
            default=33,
            ge=1,
            le=SINGLE_CELL_EXPRESSION_MAX_QUERY_LIMIT,
        ),
        offset: int = Query(
            default=0,
            ge=0,
            le=SINGLE_CELL_EXPRESSION_MAX_QUERY_OFFSET,
        ),
    ) -> dict[str, Any]:
        if single_cell_formal23 is not None:
            try:
                return single_cell_formal23.coverage_summary(
                    cancer_id=cancer_id,
                    formal_input_eligible=formal_input_eligible,
                    limit=limit,
                    offset=offset,
                )
            except SingleCellFormal23InputError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            except SingleCellFormal23AssetError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        if single_cell_expression is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 single-cell expression audit binding is not mounted",
            )
        try:
            return single_cell_expression.query_summary(
                cancer_id=cancer_id,
                formal_input_eligible=formal_input_eligible,
                limit=limit,
                offset=offset,
            )
        except SingleCellExpressionInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except SingleCellExpressionAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @staging.get("/v3.2-staging/single-cell/audit/capability")
    def single_cell_gap_capability() -> dict[str, Any]:
        if single_cell_formal23 is not None:
            capability = single_cell_formal23.capability_status()
            return {
                **capability,
                "module": "single_cell_formal23_audit",
                "audit_surface": "FORMAL23_SUCCESS_AND_INDEPENDENT_AUDIT",
                "typed_gap_policy": "EXPLICIT_NULL_NOT_ZERO",
            }
        if single_cell_gap is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 independently audited single-cell gap binding is not mounted",
            )
        return single_cell_gap.capability()

    @staging.get("/v3.2-staging/single-cell/audit/coverage")
    def single_cell_gap_coverage(
        cancer_id: str | None = None,
        formal_eligible: bool | None = None,
        limit: int = Query(default=33, ge=1, le=SINGLE_CELL_GAP_MAX_QUERY_LIMIT),
        offset: int = Query(default=0, ge=0, le=SINGLE_CELL_GAP_MAX_QUERY_OFFSET),
    ) -> dict[str, Any]:
        if single_cell_formal23 is not None:
            try:
                return single_cell_formal23.coverage_summary(
                    cancer_id=cancer_id,
                    formal_input_eligible=formal_eligible,
                    limit=limit,
                    offset=offset,
                )
            except SingleCellFormal23InputError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            except SingleCellFormal23AssetError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        if single_cell_gap is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 independently audited single-cell gap binding is not mounted",
            )
        try:
            return single_cell_gap.query_coverage(
                cancer_id=cancer_id,
                formal_eligible=formal_eligible,
                limit=limit,
                offset=offset,
            )
        except SingleCellGapQueryInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except SingleCellGapQueryAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @staging.get("/v3.2-staging/single-cell/audit/gaps")
    def single_cell_typed_gaps() -> dict[str, Any]:
        if single_cell_formal23 is not None:
            capability = single_cell_formal23.capability_status()
            return {
                "module": "single_cell_formal23_audit",
                "query_kind": "typed_unavailable_gaps",
                "formal_eligible_cancers": capability["formal_eligible_cancers"],
                "typed_unavailable": capability["typed_unavailable"],
                "typed_unavailable_cancer_count": capability[
                    "typed_unavailable_cancer_count"
                ],
                "full_33_single_cell_coverage_claimed": False,
                "typed_unavailable_rows_are_null": True,
                "changes_exact_primary_score": False,
                "production_deployed": False,
                "provenance": {
                    "success_sha256": single_cell_formal23.success_sha256,
                    "binding_sha256": capability["binding_sha256"],
                    "independent_audit_sha256": capability[
                        "independent_audit_sha256"
                    ],
                },
            }
        if single_cell_gap is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 independently audited single-cell gap binding is not mounted",
            )
        return single_cell_gap.gaps()

    @staging.get("/v3.2-staging/single-cell/activity/{cancer_id}")
    def single_cell_activity_compat(
        cancer_id: str,
        pathway_id: str | None = None,
        limit: int = Query(default=100, ge=1, le=HNSC_UCELL_MAX_PAGE_LIMIT),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        cancer = cancer_id.strip().upper()
        if single_cell_formal23 is not None:
            try:
                # The formal-23 release is donor-aggregated.  Expose it through
                # this generic activity route, but never label it as a
                # cell-level UCell result.
                return single_cell_formal23.query_pathway_activity(
                    cancer_id=cancer,
                    pathway_id=pathway_id,
                    limit=limit,
                    offset=offset,
                )
            except SingleCellFormal23InputError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            except SingleCellFormal23AssetError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        if single_cell_ucell_17c is not None:
            try:
                return single_cell_ucell_17c.query_donor_celltype_scores(
                    cancer_id=cancer,
                    pathway_id=pathway_id,
                    limit=limit,
                    offset=offset,
                )
            except SingleCellUCell17CInputError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            except SingleCellUCell17CAssetError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        if cancer != "HNSC" or hnsc_ucell is None:
            return {
                "module": "single_cell_context",
                "query_kind": "cell_type_pathway_activity",
                "cancer_id": cancer,
                "availability": False,
                "rows": [],
                "returned_rows": 0,
                "unavailable_reason": "UCELL_FORMAL_OUTPUT_AVAILABLE_FOR_HNSC_ONLY",
                "production_deployed": False,
            }
        try:
            return hnsc_ucell.query_donor_celltype_scores(
                pathway_id=pathway_id,
                limit=limit,
                offset=offset,
            )
        except HNSCUCellInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except HNSCUCellAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @staging.get("/v3.2-staging/single-cell/trajectory/{cancer_id}")
    def single_cell_trajectory_compat(
        cancer_id: str,
        level: str = "PATHWAY",
        pathway_id: str | None = None,
        limit: int = Query(default=100, ge=1, le=SINGLE_CELL_DIAGNOSTIC_MAX_QUERY_LIMIT),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        cancer = cancer_id.strip().upper()
        if single_cell_formal23 is not None:
            entry = single_cell_formal23.download_entry("single_cell_pseudotime")
            return {
                "module": "single_cell_formal23",
                "query_kind": "pseudotime_availability",
                "cancer_id": cancer,
                "availability": False,
                "rows": [],
                "returned_rows": 0,
                "pseudotime_numeric_values": None,
                "numeric_values_available": False,
                "primary_score_weight": None,
                "secondary_score_weight": None,
                "unavailable_reason": entry["unavailable_reason"],
                "download_status": entry["status"],
                "production_deployed": False,
                "provenance": {
                    "success_sha256": single_cell_formal23.success_sha256,
                    "binding_sha256": single_cell_formal23.binding_sha256,
                    "independent_audit_sha256": single_cell_formal23.audit_sha256,
                },
            }
        if single_cell_diagnostic is not None:
            try:
                return single_cell_diagnostic.query_trajectory(
                    cancer_id=cancer,
                    level=level,
                    pathway_id=pathway_id,
                    limit=limit,
                    offset=offset,
                )
            except SingleCellDiagnosticInputError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            except SingleCellDiagnosticAssetError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        if single_cell_ucell_17c is not None:
            try:
                return single_cell_ucell_17c.pseudotime_status(cancer)
            except SingleCellUCell17CInputError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        if cancer != "HNSC" or hnsc_ucell is None:
            return {
                "module": "single_cell_context",
                "query_kind": "pseudotime_availability",
                "cancer_id": cancer,
                "availability": False,
                "rows": [],
                "returned_rows": 0,
                "pseudotime_numeric_values": None,
                "numeric_values_available": False,
                "unavailable_reason": "NO_EXPLICIT_TRAJECTORY_ROOT_OR_ORDERED_SOURCE_STATE",
                "production_deployed": False,
            }
        try:
            return hnsc_ucell.query_pseudotime_availability(
                level=level,
                pathway_id=pathway_id,
                limit=limit,
                offset=offset,
            )
        except HNSCUCellInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except HNSCUCellAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @staging.get("/v3.2-staging/single-cell/figures/{cancer_id}")
    def single_cell_figures_compat(cancer_id: str) -> dict[str, Any]:
        if single_cell_formal23 is not None:
            entry = single_cell_formal23.download_entry("single_cell_figures")
            return {
                "module": "single_cell_formal23",
                "query_kind": "formal_figure_availability",
                "cancer_id": cancer_id.strip().upper(),
                "availability": False,
                "rows": [],
                "returned_rows": 0,
                "figure_count": None,
                "numeric_values_available": False,
                "unavailable_reason": entry["unavailable_reason"],
                "download_status": entry["status"],
                "production_deployed": False,
                "provenance": {
                    "success_sha256": single_cell_formal23.success_sha256,
                    "binding_sha256": single_cell_formal23.binding_sha256,
                    "independent_audit_sha256": single_cell_formal23.audit_sha256,
                },
            }
        if single_cell_diagnostic is not None:
            try:
                return single_cell_diagnostic.figure_manifest(cancer_id=cancer_id)
            except SingleCellDiagnosticInputError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            except SingleCellDiagnosticAssetError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        if single_cell_gap is None:
            raise HTTPException(status_code=503, detail="Single-cell gap authority is not mounted")
        figure_status = single_cell_gap.capability()["figures"]
        return {
            "module": "single_cell_context",
            "query_kind": "formal_figure_availability",
            "cancer_id": cancer_id.strip().upper(),
            "availability": False,
            "rows": [],
            "returned_rows": 0,
            **figure_status,
            "production_deployed": False,
        }

    @staging.get("/v3.2-staging/single-cell/figure/{cancer_id}/{figure_id}")
    def single_cell_figure_compat(cancer_id: str, figure_id: str):
        if single_cell_formal23 is not None:
            entry = single_cell_formal23.download_entry("single_cell_figures")
            return {
                "module": "single_cell_formal23",
                "query_kind": "formal_figure_file",
                "cancer_id": cancer_id.strip().upper(),
                "figure_id": figure_id.strip(),
                "availability": False,
                "file": None,
                "numeric_values_available": False,
                "unavailable_reason": entry["unavailable_reason"],
                "download_status": entry["status"],
                "legacy_route": True,
                "production_deployed": False,
                "provenance": {
                    "success_sha256": single_cell_formal23.success_sha256,
                    "binding_sha256": single_cell_formal23.binding_sha256,
                    "independent_audit_sha256": single_cell_formal23.audit_sha256,
                },
            }
        if single_cell_diagnostic is not None:
            try:
                resolved = single_cell_diagnostic.resolve_figure(
                    cancer_id=cancer_id, figure_id=figure_id
                )
            except SingleCellDiagnosticInputError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            except SingleCellDiagnosticAssetError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            if resolved is not None:
                return FileResponse(
                    resolved["path"],
                    media_type=resolved["media_type"],
                    filename=resolved["file_name"],
                    headers={
                        "X-Artifact-SHA256": resolved["sha256"],
                        "X-Root-Provenance": "INFERRED_CYTOTRACE2_UCELL_CONSENSUS",
                    },
                )
            return {
                "module": "single_cell_diagnostic",
                "query_kind": "formal_figure_file",
                "cancer_id": cancer_id.strip().upper(),
                "figure_id": figure_id.strip(),
                "availability": False,
                "file": None,
                "unavailable_reason": "FIGURE_ID_NOT_DECLARED_IN_HASH_PINNED_MANIFEST",
                "production_deployed": False,
            }
        return {
            "module": "single_cell_context",
            "query_kind": "formal_figure_file",
            "cancer_id": cancer_id.strip().upper(),
            "figure_id": figure_id.strip(),
            "availability": False,
            "file": None,
            "unavailable_reason": "NO_FORMAL_V32_SINGLE_CELL_FIGURE_FILES",
            "production_deployed": False,
        }

    def _formal23_diagnostic_unavailable(
        *,
        download_id: str,
        query_kind: str,
        cancer_id: str | None = None,
        relative_path: str | None = None,
    ) -> dict[str, Any]:
        """Return an explicit typed gap for diagnostic products.

        Formal-23 currently publishes donor-level activity and association
        tables only.  Pseudotime and figure files are intentionally typed
        unavailable; returning a 200 payload with null-valued measurements
        keeps the route discoverable without fabricating zeros or falling
        back to the superseded legacy publication.
        """

        if single_cell_formal23 is None:  # pragma: no cover - caller guard
            raise RuntimeError("formal-23 sidecar is not mounted")
        entry = single_cell_formal23.download_entry(download_id)
        success_sha256 = getattr(single_cell_formal23, "success_sha256", None)
        binding_sha256 = getattr(single_cell_formal23, "binding_sha256", None)
        audit_sha256 = getattr(single_cell_formal23, "audit_sha256", None)
        payload: dict[str, Any] = {
            "module": "single_cell_formal23",
            "query_kind": query_kind,
            "status": "GAP_TYPED_UNAVAILABLE",
            "availability_status": "TYPED_UNAVAILABLE",
            "availability": False,
            "data_present": False,
            "download_implemented": False,
            "contract_complete": True,
            "formal_eligible_cancer_count": 23,
            "typed_unavailable_cancer_count": 10,
            "full_33_single_cell_coverage_claimed": False,
            "typed_unavailable": True,
            "numeric_values_available": False,
            "primary_score_weight": None,
            "secondary_score_weight": None,
            "unavailable_reason": entry["unavailable_reason"],
            "download_status": entry["status"],
            "legacy_route": True,
            "production_deployed": False,
            "provenance": {
                "success_sha256": success_sha256,
                "binding_sha256": binding_sha256,
                "independent_audit_sha256": audit_sha256,
            },
        }
        if cancer_id is not None:
            payload["cancer_id"] = cancer_id.strip().upper()
        if relative_path is not None:
            payload["relative_path"] = relative_path
        return payload

    @staging.get("/v3.2-staging/single-cell/diagnostic/capability")
    def single_cell_diagnostic_capability() -> dict[str, Any]:
        if single_cell_formal23 is not None:
            payload = _formal23_diagnostic_unavailable(
                download_id="single_cell_pseudotime",
                query_kind="diagnostic_capability",
            )
            payload.update(
                {
                    "cancer_count": None,
                    "rows": [],
                    "returned_rows": 0,
                }
            )
            return payload
        if single_cell_diagnostic is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 single-cell diagnostic publication is not mounted",
            )
        return single_cell_diagnostic.capability_status()

    @staging.get(
        "/v3.2-staging/single-cell/diagnostic/{cancer_id}/download-manifest"
    )
    def single_cell_diagnostic_download_manifest(cancer_id: str) -> dict[str, Any]:
        if single_cell_formal23 is not None:
            return _formal23_diagnostic_unavailable(
                download_id="single_cell_pseudotime",
                query_kind="diagnostic_download_manifest",
                cancer_id=cancer_id,
            )
        if single_cell_diagnostic is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 single-cell diagnostic publication is not mounted",
            )
        try:
            return single_cell_diagnostic.download_manifest(cancer_id=cancer_id)
        except SingleCellDiagnosticInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except SingleCellDiagnosticAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @staging.get(
        "/v3.2-staging/single-cell/diagnostic/{cancer_id}/download/{relative_path:path}",
        response_model=None,
    )
    def single_cell_diagnostic_download(
        cancer_id: str, relative_path: str
    ) -> FileResponse | dict[str, Any]:
        if single_cell_formal23 is not None:
            return _formal23_diagnostic_unavailable(
                download_id="single_cell_pseudotime",
                query_kind="diagnostic_download",
                cancer_id=cancer_id,
                relative_path=relative_path,
            )
        if single_cell_diagnostic is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 single-cell diagnostic publication is not mounted",
            )
        try:
            resolved = single_cell_diagnostic.resolve_download(
                cancer_id=cancer_id, relative_path=relative_path
            )
        except SingleCellDiagnosticInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except SingleCellDiagnosticAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return FileResponse(
            resolved["path"],
            media_type=resolved["media_type"],
            filename=resolved["file_name"],
            headers={"X-Artifact-SHA256": resolved["sha256"]},
        )

    @staging.get("/v3.2-staging/lncrna/{lncrna_id}/expression")
    @staging.get("/v3.2-staging/lncrna/{lncrna_id}/coverage")
    def bulk_lncrna_expression(
        lncrna_id: str,
        cancer_id: str | None = None,
        availability: bool | None = None,
        limit: int = Query(default=33, ge=1, le=BULK_EXPRESSION_MAX_QUERY_LIMIT),
        offset: int = Query(default=0, ge=0, le=BULK_EXPRESSION_MAX_QUERY_OFFSET),
    ) -> dict[str, Any]:
        if bulk_expression is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 bulk expression release is not mounted",
            )
        try:
            return bulk_expression.query_lncrna(
                lncrna_id=lncrna_id,
                cancer_id=cancer_id,
                availability=availability,
                limit=limit,
                offset=offset,
            )
        except BulkExpressionQueryInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except BulkExpressionQueryAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @staging.get("/v3.2-staging/expression/cancer-coverage")
    def bulk_expression_cancer_coverage(
        cancer_id: str | None = None,
        limit: int = Query(default=33, ge=1, le=BULK_EXPRESSION_MAX_QUERY_LIMIT),
        offset: int = Query(default=0, ge=0, le=BULK_EXPRESSION_MAX_QUERY_OFFSET),
    ) -> dict[str, Any]:
        if bulk_expression is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 bulk expression release is not mounted",
            )
        try:
            return bulk_expression.query_cancer_coverage(
                cancer_id=cancer_id,
                limit=limit,
                offset=offset,
            )
        except BulkExpressionQueryInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except BulkExpressionQueryAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @staging.get("/v3.2-staging/lncrna/{lncrna_id}/survival")
    def clinical_lncrna_survival(
        lncrna_id: str,
        cancer_id: str | None = None,
        clinical_endpoint: str | None = None,
        availability: bool | None = None,
        include_curves: bool = True,
        limit: int = Query(default=198, ge=1, le=CLINICAL_KM_MAX_QUERY_LIMIT),
        offset: int = Query(default=0, ge=0, le=CLINICAL_KM_MAX_QUERY_OFFSET),
    ) -> dict[str, Any]:
        if clinical_km is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 fresh Clinical KM release is not mounted",
            )
        try:
            return clinical_km.query_lncrna_survival(
                lncrna_id=lncrna_id,
                cancer_id=cancer_id,
                clinical_endpoint=clinical_endpoint,
                availability=availability,
                include_curves=include_curves,
                limit=limit,
                offset=offset,
            )
        except ClinicalKMQueryInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ClinicalKMQueryAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @staging.get("/v3.2-staging/lncrna/{lncrna_id}/coexpression")
    def bulk_lncrna_coexpression(
        lncrna_id: str,
        cancer_id: str | None = None,
        gene_id: str | None = None,
        direction: str | None = None,
        min_abs_rho: float | None = Query(default=None, ge=0, le=1),
        max_fdr: float | None = Query(default=None, ge=0, le=1),
        limit: int = Query(default=100, ge=1, le=BULK_COEXPRESSION_MAX_QUERY_LIMIT),
        offset: int = Query(
            default=0,
            ge=0,
            le=BULK_COEXPRESSION_MAX_QUERY_OFFSET,
        ),
    ) -> dict[str, Any]:
        if bulk_coexpression is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 fresh bulk coexpression release is not mounted",
            )
        try:
            return bulk_coexpression.query_lncrna(
                lncrna_id=lncrna_id,
                cancer_id=cancer_id,
                gene_id=gene_id,
                direction=direction,
                min_abs_rho=min_abs_rho,
                max_fdr=max_fdr,
                limit=limit,
                offset=offset,
            )
        except BulkCoexpressionQueryInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except BulkCoexpressionQueryAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @staging.get("/v3.2-staging/coexpression/clusters")
    def bulk_coexpression_clusters(
        cancer_id: str | None = None,
        cluster_id: str | None = None,
        lncrna_id: str | None = None,
        limit: int = Query(default=100, ge=1, le=BULK_COEXPRESSION_MAX_QUERY_LIMIT),
        offset: int = Query(
            default=0,
            ge=0,
            le=BULK_COEXPRESSION_MAX_QUERY_OFFSET,
        ),
    ) -> dict[str, Any]:
        if bulk_coexpression is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 fresh bulk coexpression release is not mounted",
            )
        try:
            return bulk_coexpression.query_clusters(
                cancer_id=cancer_id,
                cluster_id=cluster_id,
                lncrna_id=lncrna_id,
                limit=limit,
                offset=offset,
            )
        except BulkCoexpressionQueryInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except BulkCoexpressionQueryAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @staging.get("/v3.2-staging/validation/external")
    def external_validation_metrics(
        validation_role: str | None = None,
        source_database: str | None = None,
        cancer_id: str | None = None,
        k: int | None = None,
        availability: bool | None = None,
        limit: int = Query(default=500, ge=1, le=EXTERNAL_VALIDATION_MAX_QUERY_LIMIT),
        offset: int = Query(
            default=0,
            ge=0,
            le=EXTERNAL_VALIDATION_MAX_QUERY_OFFSET,
        ),
    ) -> dict[str, Any]:
        if external_validation is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 fresh external-validation release is not mounted",
            )
        try:
            return external_validation.query_metrics(
                validation_role=validation_role,
                source_database=source_database,
                cancer_id=cancer_id,
                k=k,
                availability=availability,
                limit=limit,
                offset=offset,
            )
        except ExternalValidationQueryInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ExternalValidationQueryAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @staging.get("/v3.2-staging/lncrna/{lncrna_id}/external-validation")
    def external_validation_lncrna(
        lncrna_id: str,
        cancer_id: str | None = None,
        validation_role: str | None = None,
        source_database: str | None = None,
        limit: int = Query(default=100, ge=1, le=EXTERNAL_VALIDATION_MAX_QUERY_LIMIT),
        offset: int = Query(
            default=0,
            ge=0,
            le=EXTERNAL_VALIDATION_MAX_QUERY_OFFSET,
        ),
    ) -> dict[str, Any]:
        if external_validation is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 fresh external-validation release is not mounted",
            )
        try:
            return external_validation.query_lncrna(
                lncrna_id=lncrna_id,
                cancer_id=cancer_id,
                validation_role=validation_role,
                source_database=source_database,
                limit=limit,
                offset=offset,
            )
        except ExternalValidationQueryInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ExternalValidationQueryAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @staging.get("/v3.2-staging/validation/external/pmid-overlap")
    def external_validation_pmid_overlap(
        pmid: str | None = None,
        source_database: str | None = None,
        training_pmid_overlap: bool | None = None,
        limit: int = Query(default=100, ge=1, le=EXTERNAL_VALIDATION_MAX_QUERY_LIMIT),
        offset: int = Query(
            default=0,
            ge=0,
            le=EXTERNAL_VALIDATION_MAX_QUERY_OFFSET,
        ),
    ) -> dict[str, Any]:
        if external_validation is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 fresh external-validation release is not mounted",
            )
        try:
            return external_validation.query_overlap(
                pmid=pmid,
                source_database=source_database,
                training_pmid_overlap=training_pmid_overlap,
                limit=limit,
                offset=offset,
            )
        except ExternalValidationQueryInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ExternalValidationQueryAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @staging.get("/v3.2-staging/fusion/capability")
    def multimodal_fusion_capability() -> dict[str, Any]:
        if multimodal_fusion is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 multimodal secondary-fusion release is not mounted",
            )
        return multimodal_fusion.capability_status()

    @staging.get("/v3.2-staging/activity/continuous/capability")
    def continuous_activity_capability() -> dict[str, Any]:
        if continuous_activity is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 fresh continuous pathway-activity release is not mounted",
            )
        return continuous_activity.capability_status()

    @staging.get("/v3.2-staging/activity/continuous/{cancer_id}")
    def continuous_activity_predictions(
        cancer_id: str,
        pathway_id: str | None = None,
        sample_id: str | None = None,
        limit: int = Query(default=100, ge=1, le=CONTINUOUS_ACTIVITY_MAX_QUERY_LIMIT),
        offset: int = Query(default=0, ge=0, le=CONTINUOUS_ACTIVITY_MAX_QUERY_OFFSET),
    ) -> dict[str, Any]:
        if continuous_activity is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 fresh continuous pathway-activity release is not mounted",
            )
        try:
            return continuous_activity.query_oof(
                cancer_id=cancer_id,
                pathway_id=pathway_id,
                sample_id=sample_id,
                limit=limit,
                offset=offset,
            )
        except ContinuousActivityQueryInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ContinuousActivityQueryAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @staging.get("/v3.2-staging/activity/continuous/{cancer_id}/metrics")
    def continuous_activity_metrics(
        cancer_id: str,
        pathway_id: str | None = None,
        patient_fold_id: int | None = Query(default=None, ge=0, le=4),
        limit: int = Query(default=100, ge=1, le=CONTINUOUS_ACTIVITY_MAX_QUERY_LIMIT),
        offset: int = Query(default=0, ge=0, le=CONTINUOUS_ACTIVITY_MAX_QUERY_OFFSET),
    ) -> dict[str, Any]:
        if continuous_activity is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 fresh continuous pathway-activity release is not mounted",
            )
        try:
            return continuous_activity.query_metrics(
                cancer_id=cancer_id,
                pathway_id=pathway_id,
                patient_fold_id=patient_fold_id,
                limit=limit,
                offset=offset,
            )
        except ContinuousActivityQueryInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ContinuousActivityQueryAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @staging.get("/v3.2-staging/activity/continuous/{cancer_id}/attributions")
    def continuous_activity_attributions(
        cancer_id: str,
        pathway_id: str | None = None,
        lncrna_id: str | None = None,
        limit: int = Query(default=100, ge=1, le=CONTINUOUS_ACTIVITY_MAX_QUERY_LIMIT),
        offset: int = Query(default=0, ge=0, le=CONTINUOUS_ACTIVITY_MAX_QUERY_OFFSET),
    ) -> dict[str, Any]:
        if continuous_activity is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 fresh continuous pathway-activity release is not mounted",
            )
        try:
            return continuous_activity.query_attributions(
                cancer_id=cancer_id,
                pathway_id=pathway_id,
                lncrna_id=lncrna_id,
                limit=limit,
                offset=offset,
            )
        except ContinuousActivityQueryInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ContinuousActivityQueryAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @staging.get("/v3.2-staging/lncrna/{lncrna_id}/multimodal-scores")
    def multimodal_lncrna_scores(
        lncrna_id: str,
        cancer_id: str | None = None,
        pathway_id: str | None = None,
        order_by: str = "discovery",
        limit: int = Query(default=100, ge=1, le=MULTIMODAL_FUSION_MAX_QUERY_LIMIT),
        offset: int = Query(
            default=0,
            ge=0,
            le=MULTIMODAL_FUSION_MAX_QUERY_OFFSET,
        ),
    ) -> dict[str, Any]:
        if multimodal_fusion is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 multimodal secondary-fusion release is not mounted",
            )
        try:
            return multimodal_fusion.query_scores(
                lncrna_id=lncrna_id,
                cancer_id=cancer_id,
                pathway_id=pathway_id,
                order_by=order_by,
                limit=limit,
                offset=offset,
            )
        except MultimodalFusionQueryInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except MultimodalFusionQueryAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @staging.get("/v3.2-staging/single-cell/ucell/capability")
    def single_cell_ucell_17c_capability(
        cancer_id: str | None = None,
    ) -> dict[str, Any]:
        if single_cell_formal23 is not None:
            capability = single_cell_formal23.capability_status()
            return {
                **capability,
                "module": "single_cell_formal23",
                "cell_level_ucell_available": False,
                "unavailable_reason": (
                    "FORMAL23_RELEASE_IS_DONOR_AGGREGATED; "
                    "CELL_LEVEL_UCELL_IS_NOT_EXPOSED"
                ),
                "requested_cancer_id": cancer_id,
            }
        if single_cell_ucell_17c is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 legacy cell-level UCell route is not mounted",
            )
        try:
            return single_cell_ucell_17c.capability_status(cancer_id=cancer_id)
        except SingleCellUCell17CInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @staging.get("/v3.2-staging/single-cell/ucell/{cancer_id}/cell-scores")
    def single_cell_ucell_17c_cell_scores(
        cancer_id: str,
        pathway_id: str | None = None,
        cell_id: str | None = None,
        cell_type_major: str | None = None,
        patient_id: str | None = None,
        availability: str = "ALL",
        limit: int = Query(default=100, ge=1, le=SINGLE_CELL_UCELL_17C_MAX_QUERY_LIMIT),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        if single_cell_formal23 is not None:
            return {
                "module": "single_cell_formal23",
                "query_kind": "cell_level_ucell_scores",
                "cancer_id": cancer_id.strip().upper(),
                "availability": False,
                "rows": [],
                "returned_rows": 0,
                "unavailable_reason": (
                    "FORMAL23_RELEASE_IS_DONOR_AGGREGATED; "
                    "CELL_LEVEL_UCELL_IS_NOT_EXPOSED"
                ),
                "production_deployed": False,
            }
        if single_cell_ucell_17c is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 legacy cell-level UCell route is not mounted",
            )
        try:
            return single_cell_ucell_17c.query_cell_scores(
                cancer_id=cancer_id,
                pathway_id=pathway_id,
                cell_id=cell_id,
                cell_type_major=cell_type_major,
                patient_id=patient_id,
                availability=availability,
                limit=limit,
                offset=offset,
            )
        except SingleCellUCell17CInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except SingleCellUCell17CAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @staging.get("/v3.2-staging/single-cell/ucell/{cancer_id}/donor-celltype-scores")
    def single_cell_ucell_17c_donor_celltype_scores(
        cancer_id: str,
        pathway_id: str | None = None,
        cell_type_major: str | None = None,
        patient_id: str | None = None,
        availability: str = "ALL",
        limit: int = Query(default=100, ge=1, le=SINGLE_CELL_UCELL_17C_MAX_QUERY_LIMIT),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        if single_cell_formal23 is not None:
            if patient_id is not None:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "patient_id filtering is not exposed by the formal-23 "
                        "donor-aggregated release"
                    ),
                )
            try:
                # This route is named for the historical cell-level UCell
                # product.  The current formal release has a compatible
                # donor/pathway table, so return that table with its explicit
                # donor-level semantics instead of falling back to HNSC-only
                # legacy assets.
                return single_cell_formal23.query_pathway_activity(
                    cancer_id=cancer_id,
                    pathway_id=pathway_id,
                    cell_type=cell_type_major,
                    limit=limit,
                    offset=offset,
                )
            except SingleCellFormal23InputError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            except SingleCellFormal23AssetError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        if single_cell_ucell_17c is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 legacy cell-level UCell route is not mounted",
            )
        try:
            return single_cell_ucell_17c.query_donor_celltype_scores(
                cancer_id=cancer_id,
                pathway_id=pathway_id,
                cell_type_major=cell_type_major,
                patient_id=patient_id,
                availability=availability,
                limit=limit,
                offset=offset,
            )
        except SingleCellUCell17CInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except SingleCellUCell17CAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @staging.get("/v3.2-staging/single-cell/ucell/{cancer_id}/pathway-availability")
    def single_cell_ucell_17c_pathway_availability(
        cancer_id: str,
        pathway_id: str | None = None,
        availability: str = "ALL",
        limit: int = Query(default=100, ge=1, le=SINGLE_CELL_UCELL_17C_MAX_QUERY_LIMIT),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        if single_cell_formal23 is not None:
            try:
                return single_cell_formal23.query_pathway_activity(
                    cancer_id=cancer_id,
                    pathway_id=pathway_id,
                    availability=availability,
                    limit=limit,
                    offset=offset,
                )
            except SingleCellFormal23InputError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            except SingleCellFormal23AssetError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        if single_cell_ucell_17c is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 legacy cell-level UCell route is not mounted",
            )
        try:
            return single_cell_ucell_17c.query_pathway_availability(
                cancer_id=cancer_id,
                pathway_id=pathway_id,
                availability=availability,
                limit=limit,
                offset=offset,
            )
        except SingleCellUCell17CInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except SingleCellUCell17CAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @staging.get("/v3.2-staging/single-cell/ucell/{cancer_id}/download-manifest")
    def single_cell_ucell_17c_download_manifest(
        cancer_id: str,
    ) -> dict[str, Any]:
        if single_cell_formal23 is not None:
            entry = single_cell_formal23.download_entry("single_cell_activity")
            return {
                **entry,
                "cancer_id": cancer_id.strip().upper(),
                "legacy_route": True,
                "cell_level_ucell_available": False,
                "route_note": "formal23 activity download is donor-aggregated",
            }
        if single_cell_ucell_17c is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 legacy cell-level UCell route is not mounted",
            )
        try:
            return single_cell_ucell_17c.download_manifest(cancer_id=cancer_id)
        except SingleCellUCell17CInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except SingleCellUCell17CAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @staging.get(
        "/v3.2-staging/single-cell/ucell/{cancer_id}/download/{relative_path:path}",
        response_model=None,
    )
    def single_cell_ucell_17c_download(
        cancer_id: str,
        relative_path: str,
    ) -> FileResponse | dict[str, Any]:
        if single_cell_formal23 is not None:
            entry = single_cell_formal23.download_entry("single_cell_activity")
            return {
                **entry,
                "cancer_id": cancer_id.strip().upper(),
                "relative_path": relative_path,
                "downloaded": False,
                "legacy_route": True,
                "cell_level_ucell_available": False,
                "unavailable_reason": (
                    "FORMAL23_ACTIVITY_PARTS_REQUIRE_PART_INDEX_ENDPOINT; "
                    "CELL_LEVEL_UCELL_ROUTE_IS_NOT_A_DIRECT_FILE_ROUTE"
                ),
            }
        if single_cell_ucell_17c is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 legacy cell-level UCell route is not mounted",
            )
        try:
            resolved = single_cell_ucell_17c.resolve_download(
                cancer_id=cancer_id,
                relative_path=relative_path,
            )
        except SingleCellUCell17CInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except SingleCellUCell17CAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return FileResponse(
            resolved["path"],
            filename=resolved["download_name"],
            media_type="application/octet-stream",
            headers={
                "X-V32-SHA256": resolved["sha256"],
                "X-V32-Binding-SHA256": resolved["binding_sha256"],
            },
        )

    @staging.get("/v3.2-staging/single-cell/hnsc/ucell/capability")
    def hnsc_ucell_capability() -> dict[str, Any]:
        if hnsc_ucell is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 HNSC cell-level UCell pilot is not mounted",
            )
        return hnsc_ucell.capability_status()

    @staging.get("/v3.2-staging/single-cell/hnsc/ucell/cell-scores")
    def hnsc_ucell_cell_scores(
        pathway_id: str | None = None,
        cell_id: str | None = None,
        cell_type_major: str | None = None,
        patient_id: str | None = None,
        availability: str = "ALL",
        limit: int = Query(default=100, ge=1, le=HNSC_UCELL_MAX_PAGE_LIMIT),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        if hnsc_ucell is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 HNSC cell-level UCell pilot is not mounted",
            )
        try:
            return hnsc_ucell.query_cell_scores(
                pathway_id=pathway_id,
                cell_id=cell_id,
                cell_type_major=cell_type_major,
                patient_id=patient_id,
                availability=availability,
                limit=limit,
                offset=offset,
            )
        except HNSCUCellInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except HNSCUCellAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @staging.get("/v3.2-staging/single-cell/hnsc/ucell/donor-celltype-scores")
    def hnsc_ucell_donor_celltype_scores(
        pathway_id: str | None = None,
        cell_type_major: str | None = None,
        patient_id: str | None = None,
        availability: str = "ALL",
        limit: int = Query(default=100, ge=1, le=HNSC_UCELL_MAX_PAGE_LIMIT),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        if hnsc_ucell is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 HNSC cell-level UCell pilot is not mounted",
            )
        try:
            return hnsc_ucell.query_donor_celltype_scores(
                pathway_id=pathway_id,
                cell_type_major=cell_type_major,
                patient_id=patient_id,
                availability=availability,
                limit=limit,
                offset=offset,
            )
        except HNSCUCellInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except HNSCUCellAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @staging.get("/v3.2-staging/single-cell/hnsc/ucell/pathway-availability")
    def hnsc_ucell_pathway_availability(
        pathway_id: str | None = None,
        availability: str = "ALL",
        limit: int = Query(default=100, ge=1, le=HNSC_UCELL_MAX_PAGE_LIMIT),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        if hnsc_ucell is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 HNSC cell-level UCell pilot is not mounted",
            )
        try:
            return hnsc_ucell.query_pathway_availability(
                pathway_id=pathway_id,
                availability=availability,
                limit=limit,
                offset=offset,
            )
        except HNSCUCellInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except HNSCUCellAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @staging.get("/v3.2-staging/single-cell/hnsc/pseudotime-availability")
    def hnsc_pseudotime_availability(
        level: str,
        pathway_id: str | None = None,
        cell_id: str | None = None,
        cell_type_major: str | None = None,
        patient_id: str | None = None,
        limit: int = Query(default=100, ge=1, le=HNSC_UCELL_MAX_PAGE_LIMIT),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        if hnsc_ucell is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 HNSC cell-level UCell pilot is not mounted",
            )
        try:
            return hnsc_ucell.query_pseudotime_availability(
                level=level,
                pathway_id=pathway_id,
                cell_id=cell_id,
                cell_type_major=cell_type_major,
                patient_id=patient_id,
                limit=limit,
                offset=offset,
            )
        except HNSCUCellInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except HNSCUCellAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @staging.get("/v3.2-staging/state-gene-sets")
    def state_gene_set_catalog(
        state_id: str | None = None,
        cancer_id: str | None = None,
        scope: str | None = None,
        direction: str | None = None,
        limit: int = Query(default=50, ge=1, le=STATE_GENE_SET_MAX_LIMIT),
        offset: int = Query(default=0, ge=0, le=STATE_GENE_SET_MAX_OFFSET),
    ) -> dict[str, Any]:
        if state_gene_sets is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 State gene-set release is not mounted",
            )
        try:
            return state_gene_sets.query_gene_sets(
                state_id=state_id,
                cancer_id=cancer_id,
                scope=scope,
                direction=direction,
                limit=limit,
                offset=offset,
            )
        except StateGeneSetQueryInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except StateGeneSetQueryAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @staging.get("/v3.2-staging/state-gene-sets/{gene_set_id}/members")
    def state_gene_set_members(
        gene_set_id: str,
        limit: int = Query(default=200, ge=1, le=STATE_GENE_SET_MAX_MEMBER_LIMIT),
        offset: int = Query(default=0, ge=0, le=STATE_GENE_SET_MAX_OFFSET),
    ) -> dict[str, Any]:
        if state_gene_sets is None:
            raise HTTPException(
                status_code=503,
                detail="V3.2 State gene-set release is not mounted",
            )
        try:
            return state_gene_sets.query_members(
                gene_set_id=gene_set_id,
                limit=limit,
                offset=offset,
            )
        except StateGeneSetQueryInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except StateGeneSetQueryAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return staging


def _unconfigured_app() -> FastAPI:
    staging = FastAPI(title="CancerLncAtlas V3.2 staging API (unconfigured)")

    @staging.get("/v3.2-staging/health")
    def health() -> dict[str, Any]:
        raise HTTPException(
            status_code=503,
            detail="CANCERLNCATLAS_V32_STAGING_REGISTRY is not configured",
        )

    return staging


_configured_registry = os.getenv("CANCERLNCATLAS_V32_STAGING_REGISTRY")
app = (
    create_staging_app(
        _configured_registry,
        network_manifest_path=os.getenv(
            "CANCERLNCATLAS_V32_NETWORK_MANIFEST"
        ),
        network_manifest_sha256=os.getenv(
            "CANCERLNCATLAS_V32_NETWORK_MANIFEST_SHA256"
        ),
        network_audit_binding_path=os.getenv(
            "CANCERLNCATLAS_V32_NETWORK_AUDIT_BINDING"
        ),
        network_audit_binding_sha256=os.getenv(
            "CANCERLNCATLAS_V32_NETWORK_AUDIT_BINDING_SHA256"
        ),
        gene_set_subtype_binding_path=os.getenv(
            "CANCERLNCATLAS_V32_GENE_SET_SUBTYPE_BINDING"
        ),
        gene_set_subtype_binding_sha256=os.getenv(
            "CANCERLNCATLAS_V32_GENE_SET_SUBTYPE_BINDING_SHA256"
        ),
        gene_set_subtype_audit_binding_path=os.getenv(
            "CANCERLNCATLAS_V32_GENE_SET_SUBTYPE_AUDIT_BINDING"
        ),
        gene_set_subtype_audit_binding_sha256=os.getenv(
            "CANCERLNCATLAS_V32_GENE_SET_SUBTYPE_AUDIT_BINDING_SHA256"
        ),
        exact_pathway_report_binding_path=os.getenv(
            "CANCERLNCATLAS_V32_EXACT_PATHWAY_REPORT_BINDING"
        ),
        exact_pathway_report_binding_sha256=os.getenv(
            "CANCERLNCATLAS_V32_EXACT_PATHWAY_REPORT_BINDING_SHA256"
        ),
        exact_pathway_report_audit_binding_path=os.getenv(
            "CANCERLNCATLAS_V32_EXACT_PATHWAY_REPORT_AUDIT_BINDING"
        ),
        exact_pathway_report_audit_binding_sha256=os.getenv(
            "CANCERLNCATLAS_V32_EXACT_PATHWAY_REPORT_AUDIT_BINDING_SHA256"
        ),
        historical_remediation_binding_path=os.getenv(
            "CANCERLNCATLAS_V32_HISTORICAL_REMEDIATION_BINDING"
        ),
        historical_remediation_binding_sha256=os.getenv(
            "CANCERLNCATLAS_V32_HISTORICAL_REMEDIATION_BINDING_SHA256"
        ),
        historical_remediation_audit_binding_path=os.getenv(
            "CANCERLNCATLAS_V32_HISTORICAL_REMEDIATION_AUDIT_BINDING"
        ),
        historical_remediation_audit_binding_sha256=os.getenv(
            "CANCERLNCATLAS_V32_HISTORICAL_REMEDIATION_AUDIT_BINDING_SHA256"
        ),
        download_catalog_binding_path=os.getenv(
            "CANCERLNCATLAS_V32_DOWNLOAD_CATALOG_BINDING"
        ),
        download_catalog_binding_sha256=os.getenv(
            "CANCERLNCATLAS_V32_DOWNLOAD_CATALOG_BINDING_SHA256"
        ),
        download_catalog_audit_binding_path=os.getenv(
            "CANCERLNCATLAS_V32_DOWNLOAD_CATALOG_AUDIT_BINDING"
        ),
        download_catalog_audit_binding_sha256=os.getenv(
            "CANCERLNCATLAS_V32_DOWNLOAD_CATALOG_AUDIT_BINDING_SHA256"
        ),
        interaction_manifest_path=os.getenv("CANCERLNCATLAS_V32_INTERACTION_MANIFEST"),
        interaction_manifest_sha256=os.getenv("CANCERLNCATLAS_V32_INTERACTION_MANIFEST_SHA256"),
        drug_mechanism_manifest_path=os.getenv("CANCERLNCATLAS_V32_DRUG_MECHANISM_MANIFEST"),
        drug_mechanism_manifest_sha256=os.getenv(
            "CANCERLNCATLAS_V32_DRUG_MECHANISM_MANIFEST_SHA256"
        ),
        drug_sparse_manifest_path=os.getenv(
            "CANCERLNCATLAS_V32_DRUG_SPARSE_MANIFEST"
        ),
        drug_sparse_manifest_sha256=os.getenv(
            "CANCERLNCATLAS_V32_DRUG_SPARSE_MANIFEST_SHA256"
        ),
        evidence_binding_path=os.getenv("CANCERLNCATLAS_V32_EVIDENCE_BINDING"),
        evidence_binding_sha256=os.getenv("CANCERLNCATLAS_V32_EVIDENCE_BINDING_SHA256"),
        evidence_direction_binding_path=os.getenv(
            "CANCERLNCATLAS_V32_EVIDENCE_DIRECTION_BINDING"
        ),
        evidence_direction_binding_sha256=os.getenv(
            "CANCERLNCATLAS_V32_EVIDENCE_DIRECTION_BINDING_SHA256"
        ),
        evidence_direction_audit_binding_path=os.getenv(
            "CANCERLNCATLAS_V32_EVIDENCE_DIRECTION_AUDIT_BINDING"
        ),
        evidence_direction_audit_binding_sha256=os.getenv(
            "CANCERLNCATLAS_V32_EVIDENCE_DIRECTION_AUDIT_BINDING_SHA256"
        ),
        experiment_perturbation_binding_path=os.getenv(
            "CANCERLNCATLAS_V32_EXPERIMENT_PERTURBATION_BINDING"
        ),
        experiment_perturbation_binding_sha256=os.getenv(
            "CANCERLNCATLAS_V32_EXPERIMENT_PERTURBATION_BINDING_SHA256"
        ),
        experiment_perturbation_bridge_binding_path=os.getenv(
            "CANCERLNCATLAS_V32_EXPERIMENT_PERTURBATION_BRIDGE_BINDING"
        ),
        experiment_perturbation_bridge_binding_sha256=os.getenv(
            "CANCERLNCATLAS_V32_EXPERIMENT_PERTURBATION_BRIDGE_BINDING_SHA256"
        ),
        experiment_perturbation_bridge_audit_binding_path=os.getenv(
            "CANCERLNCATLAS_V32_EXPERIMENT_PERTURBATION_BRIDGE_AUDIT_BINDING"
        ),
        experiment_perturbation_bridge_audit_binding_sha256=os.getenv(
            "CANCERLNCATLAS_V32_EXPERIMENT_PERTURBATION_BRIDGE_AUDIT_BINDING_SHA256"
        ),
        single_cell_fusion_binding_path=os.getenv(
            "CANCERLNCATLAS_V32_SINGLE_CELL_FUSION_BINDING"
        ),
        single_cell_fusion_binding_sha256=os.getenv(
            "CANCERLNCATLAS_V32_SINGLE_CELL_FUSION_BINDING_SHA256"
        ),
        single_cell_expression_binding_path=os.getenv(
            "CANCERLNCATLAS_V32_SINGLE_CELL_EXPRESSION_BINDING"
        ),
        single_cell_expression_binding_sha256=os.getenv(
            "CANCERLNCATLAS_V32_SINGLE_CELL_EXPRESSION_BINDING_SHA256"
        ),
        single_cell_gap_binding_path=os.getenv(
            "CANCERLNCATLAS_V32_SINGLE_CELL_GAP_BINDING"
        ),
        single_cell_gap_binding_sha256=os.getenv(
            "CANCERLNCATLAS_V32_SINGLE_CELL_GAP_BINDING_SHA256"
        ),
        single_cell_gap_audit_binding_path=os.getenv(
            "CANCERLNCATLAS_V32_SINGLE_CELL_GAP_AUDIT_BINDING"
        ),
        single_cell_gap_audit_binding_sha256=os.getenv(
            "CANCERLNCATLAS_V32_SINGLE_CELL_GAP_AUDIT_BINDING_SHA256"
        ),
        single_cell_formal_context_binding_path=os.getenv(
            "CANCERLNCATLAS_V32_SINGLE_CELL_FORMAL_CONTEXT_BINDING"
        ),
        single_cell_formal_context_binding_sha256=os.getenv(
            "CANCERLNCATLAS_V32_SINGLE_CELL_FORMAL_CONTEXT_BINDING_SHA256"
        ),
        single_cell_formal_context_audit_binding_path=os.getenv(
            "CANCERLNCATLAS_V32_SINGLE_CELL_FORMAL_CONTEXT_AUDIT_BINDING"
        ),
        single_cell_formal_context_audit_binding_sha256=os.getenv(
            "CANCERLNCATLAS_V32_SINGLE_CELL_FORMAL_CONTEXT_AUDIT_BINDING_SHA256"
        ),
        bulk_expression_binding_path=os.getenv(
            "CANCERLNCATLAS_V32_BULK_EXPRESSION_BINDING"
        ),
        bulk_expression_binding_sha256=os.getenv(
            "CANCERLNCATLAS_V32_BULK_EXPRESSION_BINDING_SHA256"
        ),
        hnsc_ucell_binding_path=os.getenv(
            "CANCERLNCATLAS_V32_HNSC_UCELL_BINDING"
        ),
        hnsc_ucell_binding_sha256=os.getenv(
            "CANCERLNCATLAS_V32_HNSC_UCELL_BINDING_SHA256"
        ),
        single_cell_ucell_17c_binding_path=os.getenv(
            "CANCERLNCATLAS_V32_SINGLE_CELL_UCELL_17C_BINDING"
        ),
        single_cell_ucell_17c_binding_sha256=os.getenv(
            "CANCERLNCATLAS_V32_SINGLE_CELL_UCELL_17C_BINDING_SHA256"
        ),
        single_cell_diagnostic_binding_path=os.getenv(
            "CANCERLNCATLAS_V32_SINGLE_CELL_DIAGNOSTIC_BINDING"
        ),
        single_cell_diagnostic_binding_sha256=os.getenv(
            "CANCERLNCATLAS_V32_SINGLE_CELL_DIAGNOSTIC_BINDING_SHA256"
        ),
        single_cell_formal23_success_path=os.getenv(
            "CANCERLNCATLAS_V32_SINGLE_CELL_FORMAL23_SUCCESS"
        ),
        single_cell_formal23_success_sha256=os.getenv(
            "CANCERLNCATLAS_V32_SINGLE_CELL_FORMAL23_SUCCESS_SHA256"
        ),
        state_gene_set_binding_path=os.getenv(
            "CANCERLNCATLAS_V32_STATE_GENE_SET_BINDING"
        ),
        state_gene_set_binding_sha256=os.getenv(
            "CANCERLNCATLAS_V32_STATE_GENE_SET_BINDING_SHA256"
        ),
        clinical_km_binding_path=os.getenv(
            "CANCERLNCATLAS_V32_CLINICAL_KM_BINDING"
        ),
        clinical_km_binding_sha256=os.getenv(
            "CANCERLNCATLAS_V32_CLINICAL_KM_BINDING_SHA256"
        ),
        bulk_coexpression_binding_path=os.getenv(
            "CANCERLNCATLAS_V32_BULK_COEXPRESSION_BINDING"
        ),
        bulk_coexpression_binding_sha256=os.getenv(
            "CANCERLNCATLAS_V32_BULK_COEXPRESSION_BINDING_SHA256"
        ),
        external_validation_binding_path=os.getenv(
            "CANCERLNCATLAS_V32_EXTERNAL_VALIDATION_BINDING"
        ),
        external_validation_binding_sha256=os.getenv(
            "CANCERLNCATLAS_V32_EXTERNAL_VALIDATION_BINDING_SHA256"
        ),
        multimodal_fusion_binding_path=os.getenv(
            "CANCERLNCATLAS_V32_MULTIMODAL_FUSION_BINDING"
        ),
        multimodal_fusion_binding_sha256=os.getenv(
            "CANCERLNCATLAS_V32_MULTIMODAL_FUSION_BINDING_SHA256"
        ),
        continuous_activity_binding_path=os.getenv(
            "CANCERLNCATLAS_V32_CONTINUOUS_ACTIVITY_BINDING"
        ),
        continuous_activity_binding_sha256=os.getenv(
            "CANCERLNCATLAS_V32_CONTINUOUS_ACTIVITY_BINDING_SHA256"
        ),
        directional_cnv_binding_path=os.getenv(
            "CANCERLNCATLAS_V32_DIRECTIONAL_CNV_BINDING"
        ),
        directional_cnv_binding_sha256=os.getenv(
            "CANCERLNCATLAS_V32_DIRECTIONAL_CNV_BINDING_SHA256"
        ),
        directional_cnv_audit_binding_path=os.getenv(
            "CANCERLNCATLAS_V32_DIRECTIONAL_CNV_AUDIT_BINDING"
        ),
        directional_cnv_audit_binding_sha256=os.getenv(
            "CANCERLNCATLAS_V32_DIRECTIONAL_CNV_AUDIT_BINDING_SHA256"
        ),
    )
    if _configured_registry
    else _unconfigured_app()
)


__all__ = ["MixedExactPathwayRequest", "app", "create_staging_app"]
