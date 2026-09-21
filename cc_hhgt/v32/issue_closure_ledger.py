"""Immutable, evidence-bound closure ledger for CancerLncAtlas V3.2 issues."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


LEDGER_FORMAT = "CANCERLNCATLAS_V32_ISSUE_CLOSURE_LEDGER_V1"
ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"

EXPECTED_TRUTH_IDS = (
    "HTML_SC_SUMMARY_33_OF_33_500",
    "HTML_CANCER_DETAIL_27_OF_33_500",
    "HTML_LNCRNA_VISUALS_UNUSABLE",
    "HTML_GENESET_DETAIL_UNUSABLE",
    "HTML_SC_UMAP_ALL_500",
    "HTML_CLINICAL_VISUALS_BLCA_500",
    "HTML_LEGACY_CANCER_BLCA_500",
    "HTML_MUTATION_RELEASE_UNAVAILABLE",
    "HTML_DOWNLOAD_LIST_KEYS_404",
    "HTML_SINGLE_V26_MISSING_ROOT_CAUSE",
    "CLINICAL_EVENT_GT_PATIENT",
    "PREDICTED_CANDIDATE_RELATIONS_ALL_ZERO",
    "SIGNIFICANT_RELATION_COUNT_OUTLIERS",
    "SC_IMMUNE_CANDIDATE_COUNT_OUTLIERS",
    "CHINESE_CANCER_SEARCH_EMPTY",
    "SC_LOW_LNCRNA_COVERAGE",
    "UCS_WRONG_CANCER_SAMPLE_INCLUDED",
    "STALE_RELEASE_PATH_BINDING",
    "CNV_ONLY_BRCA_COAD_KIRP_HAVE_DATA",
    "CNV_MISSING_ASSUMED_NEUTRAL_ZERO",
    "GISTIC_IS_FULL_LNCRNA_CNV",
    "CNV_TUMOUR_ROWS_EQUAL_PATIENT_FILES",
    "SC_RDS_H5_MISMATCH_MEANS_CORRUPTION",
    "CESC_LOW_LNCRNA_IS_INTRINSIC",
    "SINGLE_CELL_DATA_ONLY_17_CANCERS",
    "SC_DONOR_REPLICATION_GAPS",
    "SC_LEGACY_PATIENT_MAPPING_UNVERIFIED",
    "SC_UCELL_FRESH_ALL33_AVAILABLE",
    "SC_FRESH_PSEUDOTIME_AND_FIGURES_AVAILABLE",
    "PRODUCTION_BINDINGS_PORTABLE_TO_LINUX_AS_IS",
    "SMALL_REPO_APP_SAFE_PRODUCTION_REPLACEMENT",
    "V32_ROUTE_CONFLICTS_REQUIRE_EXPLICIT_OVERRIDE",
    "EXPERIMENT_ASSAY_LABELS_ARE_EXACT",
    "FAMILY_LEVEL_ASSAY_CAN_BE_EXACTLY_COMPLETED",
    "SC_IDENTIFIER_MAPPING_DOES_NOT_AFFECT_COUNTS",
    "WEB_OPTIONAL_COMPONENT_FAILURE_MUST_FAIL_WHOLE_PAGE",
    "DOWNLOAD_CATALOG_CACHE_REFLECTS_CURRENT_FILES",
)

SYNTHETIC_IDS = (
    "STAGING_CATALOG_STATIC_UCELL_17_OF_17",
    "WEB_RELATIONSHIP_WINNER_NOT_SELECTED",
)

CLOSED = {
    "HTML_SINGLE_V26_MISSING_ROOT_CAUSE",
    "GISTIC_IS_FULL_LNCRNA_CNV",
    "SC_RDS_H5_MISMATCH_MEANS_CORRUPTION",
    "SINGLE_CELL_DATA_ONLY_17_CANCERS",
    "SMALL_REPO_APP_SAFE_PRODUCTION_REPLACEMENT",
    "WEB_OPTIONAL_COMPONENT_FAILURE_MUST_FAIL_WHOLE_PAGE",
}

BLOCKED = {
    "HTML_SC_UMAP_ALL_500",
    "HTML_MUTATION_RELEASE_UNAVAILABLE",
    "PREDICTED_CANDIDATE_RELATIONS_ALL_ZERO",
    "SIGNIFICANT_RELATION_COUNT_OUTLIERS",
    "STALE_RELEASE_PATH_BINDING",
    "CNV_MISSING_ASSUMED_NEUTRAL_ZERO",
    "SC_DONOR_REPLICATION_GAPS",
    "SC_LEGACY_PATIENT_MAPPING_UNVERIFIED",
    "SC_UCELL_FRESH_ALL33_AVAILABLE",
    "SC_FRESH_PSEUDOTIME_AND_FIGURES_AVAILABLE",
    "PRODUCTION_BINDINGS_PORTABLE_TO_LINUX_AS_IS",
    "FAMILY_LEVEL_ASSAY_CAN_BE_EXACTLY_COMPLETED",
    "WEB_RELATIONSHIP_WINNER_NOT_SELECTED",
}

HTTP_STATUS = {
    "HTML_SC_SUMMARY_33_OF_33_500": "HTTP_33_OF_33_200_TYPED_PARTIAL",
    "HTML_CANCER_DETAIL_27_OF_33_500": "HTTP_33_OF_33_200_TYPED_PARTIAL",
    "HTML_LNCRNA_VISUALS_UNUSABLE": "HTTP_200_FAIL_SOFT",
    "HTML_GENESET_DETAIL_UNUSABLE": "HTTP_200_EXACT_QUERY",
    "HTML_SC_UMAP_ALL_500": "HTTP_33_OF_33_200_BUT_FRESH_LINEAGE_UNPROVED",
    "HTML_CLINICAL_VISUALS_BLCA_500": "HTTP_33_OF_33_200_MIXED_SCIENCE_STATE",
    "HTML_LEGACY_CANCER_BLCA_500": "HTTP_33_OF_33_200_TYPED_PARTIAL",
    "HTML_MUTATION_RELEASE_UNAVAILABLE": "STATUS_HTTP_200_SWEEP_0_OF_66_2XX",
    "HTML_DOWNLOAD_LIST_KEYS_404": "HTTP_9_OF_9_RANGE_PASS",
    "CLINICAL_EVENT_GT_PATIENT": "HTTP_200_INVARIANT_PASS_93_ROWS",
    "PREDICTED_CANDIDATE_RELATIONS_ALL_ZERO": "HTTP_33_OF_33_200_ZERO_SEMANTIC_ROWS",
    "CHINESE_CANCER_SEARCH_EMPTY": "HTTP_200_LUAD_AND_LUSC_RESOLVED",
    "WEB_OPTIONAL_COMPONENT_FAILURE_MUST_FAIL_WHOLE_PAGE": "HTTP_200_TYPED_COMPONENT_PARTIAL",
    "DOWNLOAD_CATALOG_CACHE_REFLECTS_CURRENT_FILES": "HTTP_9_OF_9_RANGE_PASS",
}

CURRENT_CONCLUSIONS = {
    "HTML_SC_UMAP_ALL_500": "R4 served 32 apparently available UMAP responses, but current R11 formal-23 combined output remains unbound; lineage conflict blocks scientific closure.",
    "HTML_MUTATION_RELEASE_UNAVAILABLE": "Mutation remains scientifically unavailable and the 66 cancer-route probes return no 2xx response.",
    "HTML_DOWNLOAD_LIST_KEYS_404": "Every listed key resolves over HTTP, but the catalog lacks per-file V3.2 scientific attestation and final binding.",
    "CLINICAL_EVENT_GT_PATIENT": "Candidate materialization passes events <= patients and uniqueness for 93 rows; final winner/release binding is still absent.",
    "PREDICTED_CANDIDATE_RELATIONS_ALL_ZERO": "R4 overview and 33-cancer sweep both contain zero predicted-candidate relations; the validation-only routing winner is not selected.",
    "CHINESE_CANCER_SEARCH_EMPTY": "The isolated candidate resolves the Chinese lung-cancer query to LUAD/LUSC, but the fix is not final-release bound.",
    "SC_UCELL_FRESH_ALL33_AVAILABLE": "The current R11 rescued authority supersedes the old HNSC diagnostic: current combined fresh outputs remain unbound; 23 cancers are formally eligible and 10 are typed unavailable.",
    "SC_FRESH_PSEUDOTIME_AND_FIGURES_AVAILABLE": "Current R11 formal-23 pseudotime and figures remain unbound pending the combined independent audit.",
    "SINGLE_CELL_DATA_ONLY_17_CANCERS": "Closed as a scope-semantics error: raw records cover 33 cancers; after public-source rescue, 23 are formally eligible and 10 remain typed unavailable.",
    "SC_RDS_H5_MISMATCH_MEANS_CORRUPTION": "Closed only for the corruption allegation after barcode normalization; patient provenance remains a separate blocked issue.",
    "GISTIC_IS_FULL_LNCRNA_CNV": "Closed as a prohibited claim: GISTIC must not be represented as full lncRNA CNV coverage.",
    "WEB_OPTIONAL_COMPONENT_FAILURE_MUST_FAIL_WHOLE_PAGE": "Candidate fail-soft behavior is directly verified; typed unavailability is retained and is not promoted to scientific availability.",
    "STAGING_CATALOG_STATIC_UCELL_17_OF_17": "Local builder and static catalog now report the R11 23+10 scope and keep fresh combined outputs at 0/23 until formal audit; runtime/deployment verification is pending.",
    "WEB_RELATIONSHIP_WINNER_NOT_SELECTED": "Fresh-binding readiness is blocked: no hash-pinned fair validation-only external-router versus hierarchical winner exists.",
}

NEXT_EVIDENCE = {
    "HTML_SC_SUMMARY_33_OF_33_500": ["Fresh, hash-bound single-cell summary artifacts with scientifically available responses for the declared release scope."],
    "HTML_CANCER_DETAIL_27_OF_33_500": ["Final winner-bound 33-cancer detail materialization with non-partial scientific payloads."],
    "HTML_LNCRNA_VISUALS_UNUSABLE": ["Winner-bound survival/KM artifacts and a final runtime rerun; fail-soft HTTP alone is insufficient."],
    "HTML_GENESET_DETAIL_UNUSABLE": ["Final exact-gene-set binding and representative plus 33-cancer release acceptance."],
    "HTML_SC_UMAP_ALL_500": ["Fresh R11 formal-23 UMAP recomputation hashes and runtime marker proof that no superseded UMAP is served."],
    "HTML_CLINICAL_VISUALS_BLCA_500": ["Winner-bound clinical visual artifacts for the currently partial/unavailable cancers."],
    "HTML_LEGACY_CANCER_BLCA_500": ["Final 33-cancer release binding replacing typed partial legacy responses."],
    "HTML_MUTATION_RELEASE_UNAVAILABLE": ["Fresh V3.2 mutation binding plus 33 cancers x both mutation routes scientifically available."],
    "HTML_DOWNLOAD_LIST_KEYS_404": ["Per-download V3.2 scientific status, SHA-256 and final download-catalog binding."],
    "CLINICAL_EVENT_GT_PATIENT": ["Final selected-winner clinical binding and rerun of the invariant against the bound files."],
    "PREDICTED_CANDIDATE_RELATIONS_ALL_ZERO": ["Validation-only routing winner followed by nonzero, mutually exclusive and overview/sweep-identical 33-cancer counts."],
    "SIGNIFICANT_RELATION_COUNT_OUTLIERS": ["Winner-bound unique-key relation materialization and 33-cancer count distribution audit."],
    "SC_IMMUNE_CANDIDATE_COUNT_OUTLIERS": ["Authoritative generator schema separating cell/donor/association counts, followed by release rebuild."],
    "CHINESE_CANCER_SEARCH_EMPTY": ["Final release binding and runtime acceptance of the deployed candidate."],
    "SC_LOW_LNCRNA_COVERAGE": ["Source-specific denominators, completed rescued matrices and cell-type-aware formal release tables."],
    "UCS_WRONG_CANCER_SAMPLE_INCLUDED": ["Fresh full UCS matrix/materialization after exclusion, with donor and sample provenance hashes."],
    "STALE_RELEASE_PATH_BINDING": ["Server-native final binding selected from the validation-only winner and echoed by the runtime marker."],
    "CNV_ONLY_BRCA_COAD_KIRP_HAVE_DATA": ["Full segment-to-lncRNA mapping, patient-deduplicated 33-cancer inputs and retraining acceptance."],
    "CNV_MISSING_ASSUMED_NEUTRAL_ZERO": ["Retraining that preserves missingness masks and demonstrates no neutral-zero imputation."],
    "CNV_TUMOUR_ROWS_EQUAL_PATIENT_FILES": ["Bound patient-level selection manifest consumed by the final CNV training run."],
    "CESC_LOW_LNCRNA_IS_INTRINSIC": ["Rebuilt CESC H5 with cell types and formal evaluation of the rescued source."],
    "SC_DONOR_REPLICATION_GAPS": ["Additional independently identified donors meeting the declared replication threshold."],
    "SC_LEGACY_PATIENT_MAPPING_UNVERIFIED": ["Source-verified sample-to-patient mappings for the ten legacy candidates."],
    "SC_UCELL_FRESH_ALL33_AVAILABLE": ["Fresh combined recomputation and independent audit for all 23 eligible cancers; the 10 typed-unavailable rows must remain null."],
    "SC_FRESH_PSEUDOTIME_AND_FIGURES_AVAILABLE": ["Explicit trajectory roots/ordering, fresh pseudotime outputs and hash-bound figures."],
    "PRODUCTION_BINDINGS_PORTABLE_TO_LINUX_AS_IS": ["Portable server-native final bindings entirely inside the authorized artifact root."],
    "V32_ROUTE_CONFLICTS_REQUIRE_EXPLICIT_OVERRIDE": ["Final composed-app route inventory proving deterministic override of every conflict."],
    "EXPERIMENT_ASSAY_LABELS_ARE_EXACT": ["Manual/primary-source review for inferred assay labels and a server-native typed-confidence binding."],
    "FAMILY_LEVEL_ASSAY_CAN_BE_EXACTLY_COMPLETED": ["Independent exact-pathway assay evidence; family broadcast remains forbidden."],
    "SC_IDENTIFIER_MAPPING_DOES_NOT_AFFECT_COUNTS": ["Release-table rebuild consuming the explicit-ID audit and hash comparison at publication."],
    "DOWNLOAD_CATALOG_CACHE_REFLECTS_CURRENT_FILES": ["Final scientifically attested catalog plus a rerun after controlled file removal/cache refresh."],
    "STAGING_CATALOG_STATIC_UCELL_17_OF_17": ["Regenerated/deployed catalog and runtime check against the current R11 formal-23 binding."],
    "WEB_RELATIONSHIP_WINNER_NOT_SELECTED": ["Hash-pinned fair comparison on identical universe/folds/OOF/seed/budget, validation-only winner, five fold predictions and five checkpoints."],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_hash_bound(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    expected = expected_sha256.lower()
    if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
        raise ValueError(f"Invalid expected SHA-256 for {label}")
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing {label}: {resolved}")
    observed = sha256_file(resolved)
    if observed != expected:
        raise ValueError(f"{label} SHA-256 drift: expected {expected}, observed {observed}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _validate_authorities(
    truth: Mapping[str, Any],
    runtime: Mapping[str, Any],
    readiness: Mapping[str, Any],
    handoff: Mapping[str, Any],
    run_status: Mapping[str, Any],
    supersession: Mapping[str, Any],
    catalog: Mapping[str, Any],
) -> None:
    items = truth.get("items")
    ids = [row.get("id") for row in items] if isinstance(items, list) else []
    if (
        truth.get("format") != "CANCERLNCATLAS_DATA_PROCESSING_TRUTH_MATRIX_V2_EXPANDED"
        or truth.get("summary", {}).get("item_count") != 37
        or tuple(ids) != EXPECTED_TRUTH_IDS
    ):
        raise ValueError("Truth matrix does not match the exact 37-item r2 authority")
    checks = runtime.get("checks", {})
    coverage = runtime.get("coverage", {})
    verdicts = runtime.get("verdicts", {})
    if (
        runtime.get("format") != "CANCERLNCATLAS_V32_RUNTIME_ACCEPTANCE_V1"
        or runtime.get("target", {}).get("production_port_8260_touched") is not False
        or coverage.get("historical_routes_observed") != 51
        or coverage.get("cancer_matrix_probes_observed") != 297
        or verdicts.get("full_candidate_acceptance_pass") is not False
        or checks.get("clinical_invariant", {}).get("pass") is not True
        or checks.get("clinical_invariant", {}).get("violation_count") != 0
        or checks.get("candidate_classification", {}).get("overview_predicted_total") != 0
        or checks.get("candidate_classification", {}).get("sweep_predicted_total") != 0
        or checks.get("mutation", {}).get("scientific_available") is not False
        or checks.get("downloads", {}).get("pass") is not True
        or checks.get("geneset_exact", {}).get("pass") is not True
    ):
        raise ValueError("Runtime r4 authority no longer matches the recorded failed acceptance")
    if (
        readiness.get("status") != "BLOCKED_WAITING_FOR_VALIDATION_ONLY_ROUTING_WINNER"
        or readiness.get("may_claim_final_selected") is not False
        or readiness.get("formal_input_manifest_emitted") is not False
        or readiness.get("sanitized_folds_materialized") is not False
        or readiness.get("full_3_3m_inputs_scanned") is not False
    ):
        raise ValueError("Winner-readiness authority no longer supports a blocked state")
    assets = handoff.get("assets")
    per_cancer = run_status.get("per_cancer")
    typed_unavailable = run_status.get("typed_unavailable")
    if (
        handoff.get("format") != "CC_HHGT_V3_2_SINGLE_CELL_R11_TRAINING_HANDOFF_V1"
        or handoff.get("status") != "PASS"
        or handoff.get("historical_assets_relabelled_fresh") is not False
        or not isinstance(assets, Mapping)
        or any(row.get("path") is not None for row in assets.values())
        or run_status.get("format") != "CC_HHGT_V3_2_SINGLE_CELL_R11_RESCUED_RUN_STATUS_V1"
        or run_status.get("status") != "PASS_INPUT_CONTRACT_READY"
        or not isinstance(per_cancer, Mapping)
        or len(per_cancer) != 33
        or run_status.get("formal_eligible_cancer_count") != 23
        or run_status.get("typed_unavailable_cancer_count") != 10
        or not isinstance(typed_unavailable, Mapping)
        or len(typed_unavailable) != 10
        or run_status.get("derived_assets_bound") is not False
        or supersession.get("format") != "CC_HHGT_V3_2_SINGLE_CELL_R11_RESCUED_CONTRACT_SUCCESS_V1"
        or supersession.get("status") != "SUCCESS"
        or supersession.get("formal_eligible_cancer_count") != 23
        or supersession.get("typed_unavailable_cancer_count") != 10
    ):
        raise ValueError("Current R11 authority no longer supports the pending 23+10 scope")
    capabilities = catalog.get("capabilities")
    single_cell = next(
        (row for row in capabilities or [] if row.get("capability_id") == "single_cell"),
        None,
    )
    delivery = single_cell.get("formal_context_delivery", {}) if single_cell else {}
    authority = delivery.get("fresh_authority", {})
    if (
        not single_cell
        or single_cell.get("staging_status") != "PENDING_FORMAL_BINDING"
        or authority.get("fresh_ucell_cancers") != 0
        or authority.get("formal_eligible_cancers") != 23
        or authority.get("remaining_all_cancers") != 33
        or authority.get("remaining_formal_cancers") != 23
        or authority.get("superseded_diagnostic", {}).get("publishable_as_current") is not False
        or delivery.get("ready_downloads") != []
    ):
        raise ValueError("Static catalog has not adopted the current R11 23+10 authority")


def _evidence_refs(issue_id: str) -> list[dict[str, str]]:
    refs = [{"source": "truth_matrix_r2", "locator": f"/items/{issue_id}"}]
    if issue_id.startswith("HTML_") or issue_id in {
        "CLINICAL_EVENT_GT_PATIENT",
        "PREDICTED_CANDIDATE_RELATIONS_ALL_ZERO",
        "CHINESE_CANCER_SEARCH_EMPTY",
        "WEB_OPTIONAL_COMPONENT_FAILURE_MUST_FAIL_WHOLE_PAGE",
        "DOWNLOAD_CATALOG_CACHE_REFLECTS_CURRENT_FILES",
    }:
        refs.append({"source": "runtime_acceptance_r4", "locator": "/checks and /probe_results"})
    if issue_id.startswith("SC_") or issue_id in {
        "SINGLE_CELL_DATA_ONLY_17_CANCERS",
        "CESC_LOW_LNCRNA_IS_INTRINSIC",
        "UCS_WRONG_CANCER_SAMPLE_INCLUDED",
        "STAGING_CATALOG_STATIC_UCELL_17_OF_17",
    }:
        refs.append({"source": "single_cell_r7", "locator": "/assets, /formal_eligible_cancers, /per_cancer"})
    if issue_id in {"STALE_RELEASE_PATH_BINDING", "WEB_RELATIONSHIP_WINNER_NOT_SELECTED"}:
        refs.append({"source": "winner_readiness_r2", "locator": "/status and /required_next_evidence"})
    if issue_id == "STAGING_CATALOG_STATIC_UCELL_17_OF_17":
        refs.append({"source": "staging_catalog", "locator": "/capabilities/single_cell/formal_context_delivery"})
    return refs


def build_ledger(
    *,
    source_records: Mapping[str, Mapping[str, Any]],
    truth: Mapping[str, Any],
    runtime: Mapping[str, Any],
    readiness: Mapping[str, Any],
    handoff: Mapping[str, Any],
    run_status: Mapping[str, Any],
    supersession: Mapping[str, Any],
    catalog: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_authorities(truth, runtime, readiness, handoff, run_status, supersession, catalog)
    truth_by_id = {row["id"]: row for row in truth["items"]}
    rows: list[dict[str, Any]] = []
    for issue_id in EXPECTED_TRUTH_IDS:
        source = truth_by_id[issue_id]
        status = "CLOSED" if issue_id in CLOSED else "BLOCKED" if issue_id in BLOCKED else "PARTIAL"
        rows.append(
            {
                "issue_id": issue_id,
                "title": source.get("title"),
                "origin": source.get("origin"),
                "root_cause_class": source.get("classification"),
                "intrinsic_data_problem": source.get("classification") == "REAL_DATA_GAP",
                "truth_matrix_claim_status": source.get("claim_status"),
                "closure_status": status,
                "closure_scope": "ISSUE_SPECIFIC_NOT_PRODUCTION_PROMOTION",
                "http_status": HTTP_STATUS.get(issue_id, "NOT_APPLICABLE_OR_NOT_DIRECTLY_PROBED"),
                "scientific_conclusion": CURRENT_CONCLUSIONS.get(
                    issue_id,
                    "Processing or binding remediation is incomplete or not final-release bound."
                    if status != "CLOSED"
                    else "The issue-specific claim is closed; this does not imply whole-module scientific availability.",
                ),
                "evidence": _evidence_refs(issue_id),
                "missing_evidence": [] if status == "CLOSED" else NEXT_EVIDENCE[issue_id],
                "production_deployed": False,
            }
        )
    for issue_id, title, classification in (
        (
            "STAGING_CATALOG_STATIC_UCELL_17_OF_17",
            "Static catalog hard-coded a completed 17/17 UCell state",
            "PROCESSING_BUG",
        ),
        (
            "WEB_RELATIONSHIP_WINNER_NOT_SELECTED",
            "External router versus hierarchical end-to-end winner is not selected",
            "PUBLICATION_BINDING",
        ),
    ):
        status = "BLOCKED" if issue_id in BLOCKED else "PARTIAL"
        rows.append(
            {
                "issue_id": issue_id,
                "title": title,
                "origin": "CLOSURE_LEDGER_AUDIT",
                "root_cause_class": classification,
                "intrinsic_data_problem": False,
                "truth_matrix_claim_status": "NEW",
                "closure_status": status,
                "closure_scope": "ISSUE_SPECIFIC_NOT_PRODUCTION_PROMOTION",
                "http_status": "NOT_DEPLOYED_OR_NOT_DIRECTLY_PROBED",
                "scientific_conclusion": CURRENT_CONCLUSIONS[issue_id],
                "evidence": _evidence_refs(issue_id),
                "missing_evidence": NEXT_EVIDENCE[issue_id],
                "production_deployed": False,
            }
        )
    counts = {
        status: sum(row["closure_status"] == status for row in rows)
        for status in ("CLOSED", "PARTIAL", "BLOCKED")
    }
    classes: dict[str, int] = {}
    for row in rows:
        key = row["root_cause_class"]
        classes[key] = classes.get(key, 0) + 1
    return {
        "format": LEDGER_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "immutability": {
            "output_directory_must_not_preexist": True,
            "source_artifacts_hash_verified": True,
            "production_port_8260_touched": False,
            "service_started": False,
        },
        "status_contract": {
            "CLOSED": "All evidence needed for this issue-specific claim is present; never implies whole-release completion.",
            "PARTIAL": "A correction or direct check exists, but final scientific/release evidence is missing.",
            "BLOCKED": "An explicit missing authority, source fact or fresh recomputation prevents closure.",
            "http_200_semantics": "HTTP 200 proves transport/fail-soft behavior only and is never promoted to scientific completion.",
        },
        "authority_precedence": [
            "single_cell_r7 supersedes the old HNSC-only diagnostic binding",
            "winner_readiness_r2 blocks every final-selected/fresh web relationship claim",
            "runtime_acceptance_r4 is candidate evidence, not production or final scientific acceptance",
            "truth_matrix_r2 supplies issue classification, not automatic closure status",
        ],
        "source_artifacts": dict(source_records),
        "current_facts": {
            "runtime_routes_observed": 51,
            "runtime_cancer_route_probes_observed": 297,
            "runtime_full_candidate_acceptance_pass": False,
            "clinical_rows_checked": 93,
            "clinical_invariant_violations": 0,
            "predicted_candidate_overview_total": 0,
            "predicted_candidate_sweep_total": 0,
            "downloads_http_range_pass": "9/9",
            "downloads_scientifically_attested": False,
            "mutation_scientifically_available": False,
            "single_cell_raw_scope": 33,
            "single_cell_formal_eligible": 23,
            "single_cell_current_fresh_derived_outputs": 0,
            "single_cell_remaining_all_cancers": 33,
            "single_cell_remaining_formal_cancers": 23,
            "old_hnsc_one_cancer_result": "SUPERSEDED_DIAGNOSTIC_NOT_CURRENT_RELEASE",
            "web_relationship_winner_selected": False,
            "final_binding_present": False,
            "production_port_8260_touched": False,
        },
        "items": rows,
        "summary": {
            "item_count": len(rows),
            "truth_matrix_items": len(EXPECTED_TRUTH_IDS),
            "new_ledger_items": len(SYNTHETIC_IDS),
            "closure_status_counts": counts,
            "root_cause_class_counts": dict(sorted(classes.items())),
            "pure_processing_or_binding_items": classes.get("PROCESSING_BUG", 0)
            + classes.get("PUBLICATION_BINDING", 0),
            "mixed_items": classes.get("MIXED", 0),
            "intrinsic_real_data_gap_items": classes.get("REAL_DATA_GAP", 0),
            "release_complete": False,
        },
    }


def render_markdown(ledger: Mapping[str, Any]) -> str:
    summary = ledger["summary"]
    counts = summary["closure_status_counts"]
    lines = [
        "# CancerLncAtlas V3.2 issue closure ledger",
        "",
        f"Generated: `{ledger['generated_at_utc']}`",
        "",
        "> HTTP 200 is transport/fail-soft evidence only. It is never treated as scientific completion.",
        "",
        "## Current verdict",
        "",
        f"- CLOSED: {counts['CLOSED']}",
        f"- PARTIAL: {counts['PARTIAL']}",
        f"- BLOCKED: {counts['BLOCKED']}",
        f"- Pure processing/publication-binding issues: {summary['pure_processing_or_binding_items']}",
        f"- Mixed issues: {summary['mixed_items']}",
        f"- Intrinsic real-data gaps: {summary['intrinsic_real_data_gap_items']}",
        "- Production port 8260 touched: NO",
        "- Release complete: NO",
        "",
        "## Itemized ledger",
        "",
        "| Issue | Root cause | Closure | HTTP | Missing evidence |",
        "|---|---|---|---|---|",
    ]
    for row in ledger["items"]:
        missing = "<br>".join(row["missing_evidence"]) or "None for issue-specific scope"
        title = str(row.get("title") or "").replace("|", "\\|")
        lines.append(
            f"| `{row['issue_id']}`<br>{title} | {row['root_cause_class']} | "
            f"**{row['closure_status']}** | {row['http_status']} | {missing} |"
        )
    lines.extend(
        [
            "",
            "## Binding facts",
            "",
            "- Current single-cell R11: combined fresh outputs pending formal audit; formal-eligible inputs 23/33 and typed-unavailable 10/33.",
            "- The older HNSC 1/33 result is superseded diagnostic evidence, not a current release asset.",
            "- Predicted-candidate totals remain 0 in both overview and the 33-cancer sweep.",
            "- No validation-only external-router versus hierarchical winner is bound.",
            "- Downloads passed 9/9 bounded HTTP probes but lack final scientific attestation.",
        ]
    )
    return "\n".join(lines) + "\n"
