#!/usr/bin/env python
"""Load every pinned V3.2 staging sidecar and smoke its capability routes."""
from __future__ import annotations

import json
import argparse
import hashlib
import os
from pathlib import Path
import sys

from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.unified_staging_bindings import create_app_from_unified_bindings


def _manifest_from_args() -> Path:
    """Resolve the hash-pinned staging manifest without assuming an old name."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Hash-pinned unified staging manifest (relative to the checkout or absolute)",
    )
    args = parser.parse_args()
    if args.manifest is not None:
        candidate = args.manifest
        return candidate if candidate.is_absolute() else (ROOT / candidate)
    configured = os.environ.get("V32_STAGING_MANIFEST")
    if configured:
        candidate = Path(configured)
        return candidate if candidate.is_absolute() else (ROOT / candidate)
    current = ROOT / "config" / "v32_server_unified_staging_bindings_20260904_r1.json"
    if current.is_file():
        return current
    return ROOT / "config" / "v32_unified_staging_bindings.json"
REQUIRED_STATE_IDS = (
    "stemness_rna::RNAss",
    "stemness_dna::DNAss",
    "EXTEND::published_score",
    "stemness_rna::EREG.EXPss",
    "stemness_dna::DMPss",
    "stemness_dna::ENHss",
    "stemness_dna::EREG-METHss",
)
REQUIRED_CLINICAL_ENDPOINTS = ("OS", "DSS", "PFI", "PFS", "DFI", "DFS")
CAPABILITY_ROUTES = (
    "/v3.2-staging/health",
    "/v3.2-staging/network/capability",
    "/v3.2-staging/scoring/architecture",
    "/v3.2-staging/exact-pathway/capability",
    "/v3.2-staging/exact-pathway/report",
    "/v3.2-staging/search?q=BRCA&limit=5",
    "/v3.2-staging/datasets",
    "/v3.2-staging/cancers",
    "/v3.2-staging/historical-remediation/clinical/capability",
    "/v3.2-staging/historical-remediation/mutation/capability",
    "/v3.2-staging/historical-remediation/cnv/capability",
    "/v3.2-staging/downloads",
    "/v3.2-staging/evidence/direction/capability",
    "/v3.2-staging/evidence/direction/probabilities?available=true&limit=1",
    "/v3.2-staging/evidence/direction/probabilities?available=false&limit=1",
    "/v3.2-staging/single-cell/exact-pathway/capability",
    "/v3.2-staging/single-cell/audit/capability",
    "/v3.2-staging/single-cell/audit/coverage?cancer_id=HNSC",
    "/v3.2-staging/single-cell/audit/gaps",
    "/v3.2-staging/single-cell/activity/HNSC?limit=2",
    "/v3.2-staging/single-cell/trajectory/HNSC?level=PATHWAY&limit=2",
    "/v3.2-staging/single-cell/figures/HNSC",
    "/v3.2-staging/experiment/perturbation/raw-facts/capability",
    "/v3.2-staging/experiment/perturbation/evidence-bridge/capability",
    "/v3.2-staging/fusion/capability",
    "/v3.2-staging/activity/continuous/capability",
    "/v3.2-staging/single-cell/hnsc/ucell/capability",
)


def _clinical_endpoint_semantics(
    all_rows: dict, available_rows: dict
) -> tuple[bool, int, int]:
    all_total = sum(
        int(all_rows.get(section, {}).get("total_rows", 0))
        for section in ("entity_results", "patient_results")
    )
    available_total = sum(
        int(available_rows.get(section, {}).get("total_rows", 0))
        for section in ("entity_results", "patient_results")
    )
    available_page = [
        row
        for section in ("entity_results", "patient_results")
        for row in available_rows.get(section, {}).get("results", [])
    ]
    all_page = [
        row
        for section in ("entity_results", "patient_results")
        for row in all_rows.get(section, {}).get("results", [])
    ]
    if available_total > 0:
        semantic_ok = any(
            row.get("availability") is True
            and row.get("clinical_relevance_probability") is not None
            for row in available_page
        )
    else:
        semantic_ok = any(
            row.get("availability") is False and bool(row.get("failure_reason"))
            for row in all_page
        )
    return all_total > 0 and semantic_ok, all_total, available_total


def main() -> None:
    manifest = _manifest_from_args()
    app = create_app_from_unified_bindings(manifest, repo_root=ROOT)
    results: dict[str, int] = {}
    with TestClient(app) as client:
        for route in CAPABILITY_ROUTES:
            results[route] = client.get(route).status_code
        health = client.get("/v3.2-staging/health").json()
        available_direction = client.get(
            "/v3.2-staging/evidence/direction/probabilities",
            params={"available": "true", "limit": 1},
        ).json()
        unavailable_direction = client.get(
            "/v3.2-staging/evidence/direction/probabilities",
            params={"available": "false", "limit": 1},
        ).json()
        single_cell_gap = client.get(
            "/v3.2-staging/single-cell/audit/capability"
        ).json()
        state_results = {}
        for state_id in REQUIRED_STATE_IDS:
            response = client.get(
                "/v3.2-staging/state",
                params={"state_id": state_id, "limit": 1},
            )
            state_results[state_id] = {
                "http_status": response.status_code,
                "body": response.json(),
            }
            available_response = client.get(
                "/v3.2-staging/state",
                params={"state_id": state_id, "availability": "true", "limit": 1},
            )
            state_results[state_id]["available_http_status"] = (
                available_response.status_code
            )
            state_results[state_id]["available_body"] = available_response.json()
        clinical_results = {}
        for endpoint in REQUIRED_CLINICAL_ENDPOINTS:
            response = client.get(
                "/v3.2-staging/clinical",
                params={"clinical_endpoint": endpoint, "limit": 1},
            )
            clinical_results[endpoint] = {
                "http_status": response.status_code,
                "body": response.json(),
            }
            available_response = client.get(
                "/v3.2-staging/clinical",
                params={
                    "clinical_endpoint": endpoint,
                    "availability": "true",
                    "limit": 1,
                },
            )
            clinical_results[endpoint]["available_http_status"] = (
                available_response.status_code
            )
            clinical_results[endpoint]["available_body"] = available_response.json()
        mutation_response = client.get(
            "/v3.2-staging/genomic",
            params={"modality": "mutation", "limit": 1},
        )
        mutation_subgroup_response = client.get(
            "/v3.2-staging/mutation/subgroups",
            params={"cancer_id": "BRCA", "limit": 1},
        )
        mutation_post_response = client.post(
            "/v3.2-staging/predict/mutation-context",
            json={"cancer_id": "BRCA", "top_k": 1},
        )
        cnv_response = client.get(
            "/v3.2-staging/genomic",
            params={"modality": "cnv", "limit": 1},
        )
        cnv_coverage_response = client.get("/v3.2-staging/cnv/coverage")
    failed = {route: status for route, status in results.items() if status != 200}
    available_row = available_direction.get("rows", [{}])[0]
    unavailable_row = unavailable_direction.get("rows", [{}])[0]
    probability_sum = sum(
        float(available_row.get(column, 0.0))
        for column in (
            "direction_negative_probability",
            "direction_neutral_probability",
            "direction_positive_probability",
        )
    )
    direction_semantics_pass = (
        available_row.get("direction_probability_available") is True
        and abs(probability_sum - 1.0) <= 1.0e-5
        and unavailable_row.get("direction_probability_available") is False
        and unavailable_row.get("direction_negative_probability") is None
        and unavailable_row.get("direction_neutral_probability") is None
        and unavailable_row.get("direction_positive_probability") is None
        and bool(unavailable_row.get("direction_probability_unavailable_reason"))
    )
    # The current formal-23 overlay has a different, narrower capability
    # contract from the historical 33-cancer gap auditor.  Accept either
    # contract explicitly, but never infer coverage from a missing field.
    formal23_gap_semantics_pass = (
        single_cell_gap.get("module") == "single_cell_formal23_audit"
        and single_cell_gap.get("formal_eligible_cancer_count") == 23
        and single_cell_gap.get("typed_unavailable_cancer_count") == 10
        and single_cell_gap.get("full_33_single_cell_coverage_claimed") is False
        and single_cell_gap.get("typed_unavailable_rows_are_null") is True
        and single_cell_gap.get("changes_exact_primary_score") is False
        and bool(single_cell_gap.get("independent_audit_sha256"))
    )
    legacy_gap_semantics_pass = (
        single_cell_gap.get("changes_primary_score") is False
        and single_cell_gap.get("changes_fusion_score") is False
        and single_cell_gap.get("lncrna_detection_counts", {}).get("cancers") == 33
        and single_cell_gap.get("independent_audit", {}).get("checks") == 68
        and single_cell_gap.get("independent_audit", {}).get("failed_checks") == 0
    )
    single_cell_gap_semantics_pass = (
        formal23_gap_semantics_pass or legacy_gap_semantics_pass
    )
    state_semantics_pass = all(
        result["http_status"] == 200
        and result["body"].get("total_rows", 0) > 0
        and result["body"].get("results")
        and result["body"]["results"][0].get("state_id") == state_id
        and result["available_http_status"] == 200
        and result["available_body"].get("total_rows", 0) > 0
        and result["available_body"].get("results")
        and result["available_body"]["results"][0].get("availability") is True
        and result["available_body"]["results"][0].get(
            "state_membership_probability"
        )
        is not None
        and result["body"].get("provenance", {}).get("old_checkpoint_loaded") is False
        and result["body"].get("provenance", {}).get(
            "old_predictions_used_as_features"
        )
        is False
        for state_id, result in state_results.items()
    )
    clinical_endpoint_audits = {
        endpoint: _clinical_endpoint_semantics(
            result["body"], result["available_body"]
        )
        for endpoint, result in clinical_results.items()
    }
    clinical_semantics_pass = all(
        result["http_status"] == 200
        and result["available_http_status"] == 200
        and clinical_endpoint_audits[endpoint][0]
        and result["body"].get("filters", {}).get("clinical_endpoint") == endpoint
        and result["body"].get("provenance", {}).get("old_checkpoint_loaded") is False
        and result["body"].get("provenance", {}).get(
            "old_predictions_used_as_features"
        )
        is False
        for endpoint, result in clinical_results.items()
    )
    mutation_body = mutation_response.json()
    mutation_subgroup_body = mutation_subgroup_response.json()
    mutation_post_body = mutation_post_response.json()
    mutation_semantics_pass = (
        mutation_response.status_code == 200
        and mutation_body.get("total_rows", 0) > 0
        and mutation_body.get("provenance", {}).get("old_checkpoint_loaded") is False
        and mutation_body.get("provenance", {}).get(
            "old_predictions_used_as_features"
        )
        is False
        and mutation_subgroup_response.status_code == 200
        and mutation_subgroup_body.get("scientific_status")
        == "NO_INCREMENT_DIAGNOSTIC_ONLY_RETAINED"
        and mutation_subgroup_body.get("changes_primary_ranking") is False
        and mutation_post_response.status_code == 200
        and mutation_post_body.get("request_contract")
        == "V3.2_MUTATION_CONTEXT_POST_V1"
        and mutation_post_body.get("mutation_retained_despite_no_increment") is True
        and mutation_post_body.get("changes_primary_ranking") is False
        and mutation_post_body.get("old_predictions_used") is False
    )
    cnv_body = cnv_response.json()
    cnv_semantics_pass = (
        cnv_response.status_code == 200
        and cnv_body.get("total_rows", 0) > 0
        and cnv_body.get("provenance", {}).get("old_checkpoint_loaded") is False
        and cnv_coverage_response.status_code == 200
    )
    semantic_gates_pass = all(
        (
            direction_semantics_pass,
            single_cell_gap_semantics_pass,
            state_semantics_pass,
            clinical_semantics_pass,
            mutation_semantics_pass,
            cnv_semantics_pass,
        )
    )
    report = {
        "status": (
            "PASS"
            if not failed and semantic_gates_pass
            else "FAIL"
        ),
        "manifest": str(manifest.resolve()),
        "manifest_sha256": hashlib.sha256(manifest.resolve().read_bytes()).hexdigest(),
        "production_deployed": health.get("production_deployed"),
        "release_ready": health.get("release_ready"),
        "route_count": len(app.routes),
        "capability_routes": results,
        "drug_response_actionability_query": health.get(
            "drug_response_actionability_query"
        ),
        "single_cell_exact_pathway_query": health.get(
            "single_cell_exact_pathway_query"
        ),
        "single_cell_gap_audit_query": health.get("single_cell_gap_audit_query"),
        "single_cell_gap_semantics_pass": single_cell_gap_semantics_pass,
        "single_cell_gap_contract": (
            "FORMAL23"
            if formal23_gap_semantics_pass
            else "LEGACY_33_CANCER_GAP_AUDIT"
            if legacy_gap_semantics_pass
            else "FAIL"
        ),
        "required_state_ids": {
            state_id: {
                "http_status": result["http_status"],
                "total_rows": result["body"].get("total_rows"),
                "available_rows": result["available_body"].get("total_rows"),
            }
            for state_id, result in state_results.items()
        },
        "state_semantics_pass": state_semantics_pass,
        "required_clinical_endpoints": {
            endpoint: {
                "http_status": result["http_status"],
                "entity_rows": result["body"].get("entity_results", {}).get(
                    "total_rows"
                ),
                "patient_rows": result["body"].get("patient_results", {}).get(
                    "total_rows"
                ),
                "available_rows": clinical_endpoint_audits[endpoint][2],
                "typed_null_only": clinical_endpoint_audits[endpoint][2] == 0,
            }
            for endpoint, result in clinical_results.items()
        },
        "clinical_semantics_pass": clinical_semantics_pass,
        "mutation_semantics_pass": mutation_semantics_pass,
        "cnv_semantics_pass": cnv_semantics_pass,
        "experiment_perturbation_query": health.get("experiment_perturbation_query"),
        "evidence_query": health.get("evidence_query"),
        "evidence_direction_probability_query": health.get(
            "evidence_direction_probability_query"
        ),
        "evidence_direction_semantics_pass": direction_semantics_pass,
        "physical_interaction_query": health.get("physical_interaction_query"),
        "unified_network_query": health.get("unified_network_query"),
        "exact_pathway_browse_query": health.get("exact_pathway_browse_query"),
        "gene_set_query": health.get("gene_set_query"),
        "ranked_subtype_query": health.get("ranked_subtype_query"),
        "historical_artifact_remediation_query": health.get(
            "historical_artifact_remediation_query"
        ),
        "download_catalog_query": health.get("download_catalog_query"),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if failed or not semantic_gates_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
