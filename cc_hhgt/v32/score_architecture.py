"""Authoritative V3.2 score-routing contract.

The complete model has several heads with different statistical targets.  A
single scalar would silently mix association, evidence confidence, cell
context, and drug response.  This contract keeps those estimands separate and
states the only permitted transformations into public scores.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
CONTRACT_VERSION = "CC_HHGT_V3_2_SCORE_ARCHITECTURE_V1"


class ScoreArchitectureError(RuntimeError):
    """Raised when the declared score routing violates V3.2 semantics."""


@dataclass(frozen=True)
class ScoreEndpoint:
    endpoint_id: str
    target_keys: tuple[str, ...]
    purpose: str
    authoritative_ranking: bool = False
    derived_from_primary_offset: bool = False


@dataclass(frozen=True)
class ModuleRoute:
    module_id: str
    native_target_keys: tuple[str, ...]
    affects_endpoints: tuple[str, ...]
    primary_mutation_allowed: bool
    exact_pathway_aggregation: str | None = None
    direct_target_evidence: bool = False
    missing_value_policy: str = "NULL_WITH_REASON"


ENDPOINTS: dict[str, ScoreEndpoint] = {
    "exact_pathway_primary": ScoreEndpoint(
        endpoint_id="exact_pathway_primary",
        target_keys=("cancer_id", "lncrna_id", "pathway_id"),
        purpose="authoritative exact-pathway association probability",
        authoritative_ranking=True,
    ),
    "exact_pathway_secondary_discovery": ScoreEndpoint(
        endpoint_id="exact_pathway_secondary_discovery",
        target_keys=("cancer_id", "lncrna_id", "pathway_id"),
        purpose="pair-blocked auxiliary discovery score with exact-primary fallback",
        derived_from_primary_offset=True,
    ),
    "exact_pathway_confidence": ScoreEndpoint(
        endpoint_id="exact_pathway_confidence",
        target_keys=("cancer_id", "lncrna_id", "pathway_id"),
        purpose="evidence-supported confidence, separate from discovery ranking",
        derived_from_primary_offset=True,
    ),
    "drug_actionability": ScoreEndpoint(
        endpoint_id="drug_actionability",
        target_keys=("cancer_id", "lncrna_id", "drug_id"),
        purpose="drug-response association and mechanism-supported actionability",
    ),
    "single_cell_context": ScoreEndpoint(
        endpoint_id="single_cell_context",
        target_keys=("dataset_id", "cell_type", "lncrna_id", "pathway_id"),
        purpose="cell-type-specific replication and pathway activity",
    ),
    "continuous_pathway_activity": ScoreEndpoint(
        endpoint_id="continuous_pathway_activity",
        target_keys=("cancer_id", "sample_id", "pathway_id"),
        purpose="independent patient-level continuous pathway activity",
    ),
    "state": ScoreEndpoint(
        endpoint_id="state",
        target_keys=("cancer_id", "lncrna_id", "state_id"),
        purpose="independent tumour-state association",
    ),
    "clinical": ScoreEndpoint(
        endpoint_id="clinical",
        target_keys=("cancer_id", "subject_type", "subject_id", "clinical_endpoint"),
        purpose="independent clinical endpoint",
    ),
}


MODULE_ROUTES: dict[str, ModuleRoute] = {
    "exact_pathway": ModuleRoute(
        module_id="exact_pathway",
        native_target_keys=("cancer_id", "lncrna_id", "pathway_id"),
        affects_endpoints=("exact_pathway_primary",),
        primary_mutation_allowed=True,
    ),
    "mutation_cnv": ModuleRoute(
        module_id="mutation_cnv",
        native_target_keys=("cancer_id", "lncrna_id", "pathway_id"),
        affects_endpoints=(
            "exact_pathway_secondary_discovery",
            "exact_pathway_confidence",
        ),
        primary_mutation_allowed=False,
    ),
    "single_cell": ModuleRoute(
        module_id="single_cell",
        native_target_keys=("dataset_id", "cell_type", "lncrna_id", "pathway_id"),
        affects_endpoints=(
            "single_cell_context",
            "exact_pathway_secondary_discovery",
            "exact_pathway_confidence",
        ),
        primary_mutation_allowed=False,
        exact_pathway_aggregation=(
            "donor_or_dataset_blocked_predictions_then_mean_by_"
            "cancer_id_lncrna_id_exact_pathway_id"
        ),
    ),
    "evidence_transformer": ModuleRoute(
        module_id="evidence_transformer",
        native_target_keys=("cancer_id", "lncrna_id", "pathway_id"),
        affects_endpoints=("exact_pathway_confidence",),
        primary_mutation_allowed=False,
        direct_target_evidence=True,
    ),
    "physical_interaction": ModuleRoute(
        module_id="physical_interaction",
        native_target_keys=("lncrna_id", "protein_id"),
        affects_endpoints=("exact_pathway_confidence",),
        primary_mutation_allowed=False,
        exact_pathway_aggregation="literal_gene_membership_enrichment_no_family_broadcast",
        direct_target_evidence=True,
    ),
    "experiment_perturbation": ModuleRoute(
        module_id="experiment_perturbation",
        native_target_keys=("lncrna_id", "experiment_id", "target_id"),
        affects_endpoints=("exact_pathway_confidence",),
        primary_mutation_allowed=False,
        exact_pathway_aggregation="literal_exact_pathway_mapping_no_family_broadcast",
        direct_target_evidence=True,
    ),
    "drug": ModuleRoute(
        module_id="drug",
        native_target_keys=("cancer_id", "lncrna_id", "drug_id"),
        affects_endpoints=("drug_actionability",),
        primary_mutation_allowed=False,
    ),
    "continuous_pathway_activity": ModuleRoute(
        module_id="continuous_pathway_activity",
        native_target_keys=("cancer_id", "sample_id", "pathway_id"),
        affects_endpoints=("continuous_pathway_activity",),
        primary_mutation_allowed=False,
    ),
    "state": ModuleRoute(
        module_id="state",
        native_target_keys=("cancer_id", "lncrna_id", "state_id"),
        affects_endpoints=("state",),
        primary_mutation_allowed=False,
    ),
    "clinical": ModuleRoute(
        module_id="clinical",
        native_target_keys=("cancer_id", "subject_type", "subject_id", "clinical_endpoint"),
        affects_endpoints=("clinical",),
        primary_mutation_allowed=False,
    ),
}


def validate_score_architecture() -> None:
    """Fail closed if a future edit silently reroutes an auxiliary head."""

    if set(ENDPOINTS) != {endpoint.endpoint_id for endpoint in ENDPOINTS.values()}:
        raise ScoreArchitectureError("Endpoint identifiers are inconsistent")
    for module_id, route in MODULE_ROUTES.items():
        if module_id != route.module_id or not route.native_target_keys:
            raise ScoreArchitectureError(f"Invalid module route: {module_id}")
        if route.missing_value_policy != "NULL_WITH_REASON":
            raise ScoreArchitectureError(f"{module_id} may encode missing evidence as a score")
        unknown = set(route.affects_endpoints) - set(ENDPOINTS)
        if unknown:
            raise ScoreArchitectureError(f"{module_id} routes to unknown endpoints: {unknown}")

    primary_writers = {
        module_id
        for module_id, route in MODULE_ROUTES.items()
        if "exact_pathway_primary" in route.affects_endpoints
        or route.primary_mutation_allowed
    }
    if primary_writers != {"exact_pathway"}:
        raise ScoreArchitectureError("Only the exact-pathway core may write primary")
    if MODULE_ROUTES["drug"].affects_endpoints != ("drug_actionability",):
        raise ScoreArchitectureError("Drug must remain a native actionability endpoint")
    if MODULE_ROUTES["evidence_transformer"].affects_endpoints != (
        "exact_pathway_confidence",
    ):
        raise ScoreArchitectureError("Evidence Transformer must remain confidence-only")
    single_cell = MODULE_ROUTES["single_cell"]
    if not single_cell.exact_pathway_aggregation or "single_cell_context" not in (
        single_cell.affects_endpoints
    ):
        raise ScoreArchitectureError("Single-cell routing lacks its context/aggregation path")
    if any(
        route.direct_target_evidence
        and "exact_pathway_secondary_discovery" in route.affects_endpoints
        for route in MODULE_ROUTES.values()
    ):
        raise ScoreArchitectureError("Direct target evidence may not enter discovery")


def score_architecture_manifest() -> dict[str, Any]:
    validate_score_architecture()
    return {
        "contract_version": CONTRACT_VERSION,
        "analysis_version": ANALYSIS_VERSION,
        "status": "PASS",
        "primary_score_immutable": True,
        "single_total_score_forbidden": True,
        "missing_auxiliary_policy": "NULL_WITH_REASON_AND_EXACT_PRIMARY_FALLBACK",
        "direct_target_evidence_in_discovery": False,
        "endpoints": {key: asdict(value) for key, value in ENDPOINTS.items()},
        "module_routes": {key: asdict(value) for key, value in MODULE_ROUTES.items()},
    }


__all__ = [
    "ANALYSIS_VERSION",
    "CONTRACT_VERSION",
    "ENDPOINTS",
    "MODULE_ROUTES",
    "ModuleRoute",
    "ScoreArchitectureError",
    "ScoreEndpoint",
    "score_architecture_manifest",
    "validate_score_architecture",
]
