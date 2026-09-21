#!/usr/bin/env python
"""Build the transparent V3.2 staging UI catalog from the parity contract."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.capability_parity import load_parity_config
from cc_hhgt.v32.single_cell_formal23_query import SingleCellFormal23Query
from cc_hhgt.v32.single_cell_release_scope import load_single_cell_release_scope


READY = "QUERYABLE_STAGING"
PARTIAL = "PARTIAL_STAGING"
PENDING = "PENDING_FORMAL_BINDING"
_VALID_STAGING_STATUSES = frozenset({READY, PARTIAL, PENDING})

# A catalog is often built before the complete API asset tree is copied to the
# server.  Historically the static table below was treated as if every route
# were mounted, which made a catalog advertise 25 queryable capabilities while
# the runtime returned 503 for most of them.  A hash-bound capability-scope
# receipt lets the server build state from the *actual* mounted/probed routes;
# the legacy no-receipt mode is retained for backwards-compatible unit tests.
CAPABILITY_SCOPE_SCHEMA = "CANCERLNCATLAS_V32_STAGING_CAPABILITY_SCOPE_V1"

SINGLE_CELL_GAP_BINDING = (
    ROOT
    / "artifacts/v32_single_cell_gap_audit_20260826_r2_codebound/"
    "SINGLE_CELL_GAP_AUDIT_BINDING.json"
)
SINGLE_CELL_GAP_BINDING_SHA256 = (
    "26e94b0e0a9aa7c217d50494c2b92ac477f4c01fdcb2a2147d61dc89eab76ccf"
)
SINGLE_CELL_GAP_AUDIT_BINDING = (
    ROOT
    / "artifacts/v32_single_cell_gap_audit_20260826_r2_codebound_independent_audit/"
    "INDEPENDENT_AUDIT_BINDING.json"
)
SINGLE_CELL_GAP_AUDIT_BINDING_SHA256 = (
    "1e909bc81a2180f1ab301233bf57d77911305120627f6443f7acafce7cbc1784"
)
SINGLE_CELL_R7_ROOT = (
    ROOT / "artifacts/single_cell_cell_level_r7_portable_supersession_20260829"
)
SINGLE_CELL_R7_HANDOFF = SINGLE_CELL_R7_ROOT / "TRAINING_HANDOFF.json"
SINGLE_CELL_R7_HANDOFF_SHA256 = (
    "4fa20c8579463dd394cc7b45ee227f5b7d83cf1f6605f180104534c224c80408"
)
SINGLE_CELL_R7_RUN_STATUS = SINGLE_CELL_R7_ROOT / "RUN_STATUS.json"
SINGLE_CELL_R7_RUN_STATUS_SHA256 = (
    "963268d03448ec514646dc199e3c4648faa1d47884f62faacfdd72aa23d6e2be"
)
SINGLE_CELL_R7_SUPERSESSION = SINGLE_CELL_R7_ROOT / "SUPERSESSION.json"
SINGLE_CELL_R7_SUPERSESSION_SHA256 = (
    "6a1f886b126b6505a777fcc5a5eda542b77760e3900ba132e7e1d983cc58b29c"
)
SINGLE_CELL_R11_ROOT = (
    ROOT / "artifacts/v32_single_cell_r11_rescued_contract_20260903_r1_server_manifest"
)
SINGLE_CELL_R11_HANDOFF = SINGLE_CELL_R11_ROOT / "TRAINING_HANDOFF.json"
SINGLE_CELL_R11_HANDOFF_SHA256 = (
    "ab2eccf729a34930ea3e3ec58068132fb1f39eba28dcce461a82b3f53b55336c"
)
SINGLE_CELL_R11_RUN_STATUS = SINGLE_CELL_R11_ROOT / "RUN_STATUS.json"
SINGLE_CELL_R11_RUN_STATUS_SHA256 = (
    "3c01080cabe27c0031c9f508f6c96d67ae6c388e48d96c59e03aa16c8db45a59"
)
SINGLE_CELL_R11_SUCCESS = SINGLE_CELL_R11_ROOT / "SUCCESS.json"
SINGLE_CELL_R11_SUCCESS_SHA256 = (
    "92826b85e1dd981e57ef3a68586798981692796d70199dd0a602b69b0fc5fb68"
)

STATE_ROUTES = [
    "GET /v3.2-staging/state",
    "GET /v3.2-staging/state-gene-sets",
]

STAGING_BINDINGS = {
    "exact_pathway": (
        READY,
        [
            "GET /v3.2-staging/exact-pathway/associations",
            "GET /v3.2-staging/exact-pathway/stats",
            "GET /v3.2-staging/exact-pathway/report",
            "GET /v3.2-staging/search",
            "GET /v3.2-staging/datasets",
            "GET /v3.2-staging/cancers",
            "POST /v3.2-staging/enrichment/mixed-exact-pathway",
        ],
    ),
    "gene_set": (
        READY,
        [
            "GET /v3.2-staging/gene-sets",
            "GET /v3.2-staging/gene-sets/{gene_set_id}",
            "GET /v3.2-staging/gene-sets/enrichment",
        ],
    ),
    "ranked_subtype": (
        READY,
        [
            "GET /v3.2-staging/ranked-subtypes/overview",
            "GET /v3.2-staging/ranked-subtypes/detail",
            "GET /v3.2-staging/ranked-subtypes/stability",
        ],
    ),
    "network": (
        READY,
        [
            "GET /v3.2-staging/network/lncrna/{lncrna_id}",
            "GET /v3.2-staging/network/neighborhood",
        ],
    ),
    "state_rnass": (READY, STATE_ROUTES),
    "state_dnass": (READY, STATE_ROUTES),
    "state_extend": (READY, STATE_ROUTES),
    "state_ereg_expss": (READY, STATE_ROUTES),
    "state_dmpss": (READY, STATE_ROUTES),
    "state_enhss": (READY, STATE_ROUTES),
    "state_ereg_methss": (READY, STATE_ROUTES),
    "clinical": (
        READY,
        [
            "GET /v3.2-staging/clinical",
            "GET /v3.2-staging/clinical/translational-priority",
        ],
    ),
    "expression_landscape": (
        READY,
        [
            "GET /v3.2-staging/lncrna/{lncrna_id}/expression",
            "GET /v3.2-staging/expression/cancer-coverage",
        ],
    ),
    "survival_kaplan_meier": (
        READY,
        ["GET /v3.2-staging/lncrna/{lncrna_id}/survival"],
    ),
    "bulk_coexpression": (
        READY,
        [
            "GET /v3.2-staging/lncrna/{lncrna_id}/coexpression",
            "GET /v3.2-staging/coexpression/clusters",
        ],
    ),
    "external_validation": (
        READY,
        [
            "GET /v3.2-staging/validation/external",
            "GET /v3.2-staging/lncrna/{lncrna_id}/external-validation",
        ],
    ),
    "continuous_pathway_activity": (
        READY,
        ["GET /v3.2-staging/activity/continuous/{cancer_id}"],
    ),
    "single_cell": (
        READY,
        [
            "GET /v3.2-staging/single-cell/context/capability",
            "GET /v3.2-staging/single-cell/context/lncrna-celltype",
            "GET /v3.2-staging/single-cell/context/pathway-activity",
            "GET /v3.2-staging/single-cell/context/celltype-predictions",
            "GET /v3.2-staging/single-cell/exact-pathway-associations",
            "GET /v3.2-staging/single-cell/lncrna-expression-summary",
            "GET /v3.2-staging/single-cell/audit/capability",
            "GET /v3.2-staging/single-cell/audit/coverage",
            "GET /v3.2-staging/single-cell/audit/gaps",
            "GET /v3.2-staging/single-cell/ucell/capability",
            "GET /v3.2-staging/single-cell/ucell/{cancer_id}/cell-scores",
            "GET /v3.2-staging/single-cell/ucell/{cancer_id}/donor-celltype-scores",
            "GET /v3.2-staging/single-cell/ucell/{cancer_id}/pathway-availability",
            "GET /v3.2-staging/single-cell/ucell/{cancer_id}/download-manifest",
            "GET /v3.2-staging/single-cell/ucell/{cancer_id}/download/{relative_path:path}",
            "GET /v3.2-staging/single-cell/hnsc/ucell/cell-scores",
            "GET /v3.2-staging/single-cell/hnsc/pseudotime-availability",
            "GET /v3.2-staging/single-cell/activity/{cancer_id}",
            "GET /v3.2-staging/single-cell/trajectory/{cancer_id}",
            "GET /v3.2-staging/single-cell/figures/{cancer_id}",
            "GET /v3.2-staging/single-cell/figure/{cancer_id}/{figure_id}",
            "GET /v3.2-staging/single-cell/diagnostic/{cancer_id}/download-manifest",
            "GET /v3.2-staging/single-cell/diagnostic/{cancer_id}/download/{relative_path:path}",
        ],
    ),
    "mutation": (
        READY,
        [
            "GET /v3.2-staging/genomic?modality=mutation",
            "GET /v3.2-staging/mutation/subgroups",
            "POST /v3.2-staging/predict/mutation-context",
        ],
    ),
    "cnv": (
        PENDING,
        [
            "GET /v3.2-staging/genomic?modality=cnv",
            "GET /v3.2-staging/cnv/coverage",
        ],
    ),
    "drug": (
        READY,
        [
            "GET /v3.2-staging/drug/response-actionability",
            "GET /v3.2-staging/drug/structural-mechanisms",
            "GET /v3.2-staging/drug/structural-mechanisms/download-manifest",
            "GET /v3.2-staging/drug/structural-mechanisms/download",
        ],
    ),
    "physical_interaction": (
        READY,
        [
            "GET /v3.2-staging/interaction/relationships",
            "GET /v3.2-staging/interaction/exact-pathway-enrichment",
            "GET /v3.2-staging/interaction/evidence",
            "GET /v3.2-staging/interaction/unmapped-partners",
        ],
    ),
    "experiment_perturbation": (
        READY,
        [
            "GET /v3.2-staging/experiment/perturbation/raw-facts/exact-pathway",
            "GET /v3.2-staging/experiment/perturbation/evidence-bridge",
            "GET /v3.2-staging/experiment/perturbation/evidence-bridge/events",
        ],
    ),
    "evidence_transformer": (
        READY,
        [
            "GET /v3.2-staging/evidence/confidence",
            "GET /v3.2-staging/evidence/events",
            "GET /v3.2-staging/evidence/direction/probabilities",
        ],
    ),
    "mixed_lncrna_protein_pathway_query": (
        READY,
        ["POST /v3.2-staging/enrichment/mixed-exact-pathway"],
    ),
}

KNOWN_GAPS = {
    "cnv": (
        "The prior combined genomic/CNV artifact is superseded because it did not "
        "preserve signed continuous local CNV semantics. The 33-cancer local-confounding "
        "audit and replacement directional CNV-only five-fold OOF are complete on the "
        "server; this checkout has no mounted binding, so run the server-only staging "
        "binding recipe. Do not rerun the CNV computation."
    ),
}

INTERPRETATION_NOTES = {
    "cnv": (
        "The directional CNV overlay keeps amplification and deletion signed, preserves "
        "local-locus callability and pathway-CNV context, and uses typed-unavailable "
        "rather than zero for missing values. It is an independent overlay and does not "
        "change the primary score."
    ),
    "drug": "R6 five-fold response associations and structural mechanism hypotheses are fresh, hash-pinned and queryable. They are diagnostic actionability evidence: no TCGA patient response, efficacy, causal mechanism or primary-ranking claim is made.",
    "single_cell": "The accepted V3.2 formal single-cell overlay is bound for 23 eligible cancers; 10 cancers remain explicitly typed-unavailable. Full 33-cancer single-cell coverage is not claimed and is not a release gate. Associations, activity and UCell downloads are hash-pinned; pseudotime and figures remain typed-unavailable. The overlay has zero fusion weight and does not change the primary score.",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_hash_bound(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    resolved = path.resolve()
    expected = expected_sha256.strip().lower()
    if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
        raise RuntimeError(f"Invalid expected SHA-256 for {label}")
    if not resolved.is_file():
        raise RuntimeError(f"Missing {label}: {resolved}")
    observed = _sha256(resolved)
    if observed != expected:
        raise RuntimeError(
            f"{label} SHA-256 drift: expected {expected}, observed {observed}"
        )
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return payload


def load_capability_scope(
    scope_path: Path | None, scope_sha256: str | None
) -> dict[str, Any] | None:
    """Load a server-side, hash-pinned route/capability scope receipt.

    The receipt is deliberately separate from the scientific registry: it
    attests only which API routes were mounted and smoke-tested in this
    staging runtime.  It must enumerate the complete parity universe so an
    omitted capability cannot silently inherit the optimistic static default.
    """

    if (scope_path is None) != (scope_sha256 is None):
        raise RuntimeError(
            "capability scope binding requires both path and SHA-256"
        )
    if scope_path is None:
        return None
    payload = _load_hash_bound(scope_path, str(scope_sha256), "capability scope")
    if (
        payload.get("schema_version") != CAPABILITY_SCOPE_SCHEMA
        or payload.get("environment") != "staging"
        or str(payload.get("host", "")) != "149"
        or payload.get("status") != "PASS"
        or payload.get("production_deployed") is not False
        or payload.get("release_ready") is not False
    ):
        raise RuntimeError("capability scope receipt is not a server-149 staging PASS")
    declared = payload.get("capabilities")
    if not isinstance(declared, Mapping):
        raise RuntimeError("capability scope lacks a capabilities mapping")
    expected = set(STAGING_BINDINGS)
    if set(declared) != expected:
        raise RuntimeError(
            "capability scope must enumerate every parity capability: "
            f"missing={sorted(expected - set(declared))}, "
            f"extra={sorted(set(declared) - expected)}"
        )
    normalized: dict[str, Any] = {}
    for capability_id, base in STAGING_BINDINGS.items():
        item = declared[capability_id]
        if not isinstance(item, Mapping):
            raise RuntimeError(f"capability scope entry is invalid: {capability_id}")
        status = str(item.get("staging_status", "")).strip()
        if status not in _VALID_STAGING_STATUSES:
            raise RuntimeError(f"invalid staging status for {capability_id}: {status}")
        endpoints = item.get("verified_endpoints", [])
        if not isinstance(endpoints, list) or any(
            not isinstance(route, str) or not route.strip() for route in endpoints
        ):
            raise RuntimeError(f"verified_endpoints is invalid: {capability_id}")
        base_endpoints = set(base[1])
        if not set(endpoints) <= base_endpoints:
            raise RuntimeError(
                f"capability scope advertises unknown endpoint(s): {capability_id}"
            )
        if status in {READY, PARTIAL} and not endpoints:
            raise RuntimeError(
                f"{capability_id} marked {status} without verified endpoints"
            )
        if status == PENDING and endpoints:
            raise RuntimeError(
                f"{capability_id} marked pending with verified endpoints"
            )
        gap = item.get("known_gap")
        if gap is not None and (not isinstance(gap, str) or not gap.strip()):
            raise RuntimeError(f"known_gap is invalid: {capability_id}")
        normalized[capability_id] = {
            "staging_status": status,
            "verified_endpoints": list(endpoints),
            "known_gap": gap,
            "probe_id": item.get("probe_id"),
        }
    normalized_payload = dict(payload)
    normalized_payload["capabilities"] = normalized
    normalized_payload["scope_path"] = str(scope_path.resolve())
    normalized_payload["scope_sha256"] = str(scope_sha256).strip().lower()
    return normalized_payload


def _resolve_declared_file(declaration: Mapping[str, Any], label: str) -> Path:
    raw = str(declaration.get("path") or "").strip()
    expected = str(declaration.get("sha256") or "").strip().lower()
    if not raw:
        raise RuntimeError(f"{label} lacks a declared path")
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    _load_hash_bound(path, expected, label)
    return path.resolve()


def load_directional_cnv_authority(
    binding_path: Path | None,
    binding_sha256: str | None,
    audit_binding_path: Path | None,
    audit_binding_sha256: str | None,
) -> dict[str, Any] | None:
    """Validate the replacement CNV query binding before advertising it."""

    if (binding_path is None) != (binding_sha256 is None):
        raise RuntimeError(
            "Directional CNV catalog state requires both binding path and SHA-256"
        )
    if (audit_binding_path is None) != (audit_binding_sha256 is None):
        raise RuntimeError(
            "Directional CNV catalog state requires both audit binding path and SHA-256"
        )
    if (binding_path is None) != (audit_binding_path is None):
        raise RuntimeError(
            "Directional CNV catalog state requires release and independent-audit bindings"
        )
    if binding_path is None:
        return None
    # Lazy import keeps the base catalog builder usable before the DuckDB-backed
    # directional release exists.  Once supplied, the same validator as the API
    # verifies every artifact hash, typed-null rule and 33-cancer closure.
    from cc_hhgt.v32.directional_cnv_query import DirectionalCNVReleaseQuery

    query = DirectionalCNVReleaseQuery(
        binding_path,
        expected_sha256=str(binding_sha256),
        audit_binding_path=audit_binding_path,
        expected_audit_sha256=str(audit_binding_sha256),
    )
    predictions = query.binding["artifacts"]["predictions"]
    coverage = query.binding["artifacts"]["coverage"]
    return {
        "status": "PASS_HASH_PINNED_INDEPENDENT_AUDIT",
        "binding_path": str(query.binding_path),
        "binding_sha256": query.binding_sha256,
        "independent_audit_binding_path": str(query.audit_binding_path),
        "independent_audit_binding_sha256": query.audit_binding_sha256,
        "prediction_sha256": str(predictions["sha256"]),
        "prediction_rows": int(predictions["rows"]),
        "coverage_sha256": str(coverage["sha256"]),
        "coverage_rows": int(coverage["rows"]),
        "source_generation": "CURRENT_V3.2_DIRECTIONAL_CNV_ONLY",
        "superseded_combined_cnv_exposed": False,
        "changes_primary_ranking": False,
        "release_ready": False,
    }


def _load_single_cell_r7_fresh_authority(
    *,
    current_handoff_path: Path = SINGLE_CELL_R7_HANDOFF,
    current_handoff_sha256: str = SINGLE_CELL_R7_HANDOFF_SHA256,
    current_run_status_path: Path = SINGLE_CELL_R7_RUN_STATUS,
    current_run_status_sha256: str = SINGLE_CELL_R7_RUN_STATUS_SHA256,
    current_supersession_path: Path = SINGLE_CELL_R7_SUPERSESSION,
    current_supersession_sha256: str = SINGLE_CELL_R7_SUPERSESSION_SHA256,
    release_binding_path: Path | None = SINGLE_CELL_GAP_BINDING,
    release_binding_sha256: str | None = SINGLE_CELL_GAP_BINDING_SHA256,
    audit_binding_path: Path | None = SINGLE_CELL_GAP_AUDIT_BINDING,
    audit_binding_sha256: str | None = SINGLE_CELL_GAP_AUDIT_BINDING_SHA256,
) -> dict[str, Any]:
    """Load the newest single-cell authority and fail closed on drift.

    The staging catalog is a presentation artifact, not scientific authority.
    R7 explicitly supersedes the R6 controls and requires every derived asset
    to be recomputed.  The earlier HNSC gap binding remains useful diagnostic
    history, but cannot be counted as an output of the current release.
    """

    handoff = _load_hash_bound(
        current_handoff_path,
        current_handoff_sha256,
        "single-cell R7 training handoff",
    )
    run_status = _load_hash_bound(
        current_run_status_path,
        current_run_status_sha256,
        "single-cell R7 run status",
    )
    supersession = _load_hash_bound(
        current_supersession_path,
        current_supersession_sha256,
        "single-cell R7 supersession",
    )
    expected_assets = {
        "activity",
        "association",
        "exact_pathway_availability",
        "expression_facts",
        "lnc_celltype",
    }
    assets = handoff.get("assets")
    if (
        handoff.get("format")
        != "CC_HHGT_V3_2_SINGLE_CELL_R7_PORTABLE_TRAINING_HANDOFF_V1"
        or handoff.get("analysis_version")
        != "CancerLncAtlas_V3.2_FULL_MULTITASK"
        or handoff.get("run_id")
        != "v32-single-cell-cell-level-r7-portable-20260829"
        or handoff.get("assets_reused_by_reference_from_fresh_r5_partitions")
        is not False
        or handoff.get("historical_assets_relabelled_fresh") is not False
        or handoff.get("fresh_ucell_status")
        != "PENDING_CELL_LEVEL_FRESH_RECOMPUTE"
        or handoff.get("fresh_pseudotime_status")
        != "PENDING_CELL_LEVEL_FRESH_RECOMPUTE"
        or handoff.get("single_cell_module_complete") is not False
        or handoff.get("training_started") is not False
        or handoff.get("release_ready") is not False
        or not isinstance(assets, Mapping)
        or set(assets) != expected_assets
        or any(
            not isinstance(value, Mapping)
            or value.get("status") != "UNBOUND_FRESH_RECOMPUTE_REQUIRED"
            or value.get("path") is not None
            or value.get("historical_asset_relabelled_fresh") is not False
            for value in assets.values()
        )
    ):
        raise RuntimeError("R7 handoff does not support an unbound fresh-recompute state")
    formal_cancers = run_status.get("formal_eligible_cancers")
    per_cancer = run_status.get("per_cancer")
    if (
        run_status.get("format")
        != "CC_HHGT_V3_2_SINGLE_CELL_R7_PORTABLE_RUN_STATUS_V1"
        or run_status.get("run_id") != handoff.get("run_id")
        or run_status.get("scope_count") != 33
        or run_status.get("formal_eligible_count") != 17
        or not isinstance(formal_cancers, list)
        or len(formal_cancers) != 17
        or len(set(formal_cancers)) != 17
        or run_status.get("derived_assets_bound") is not False
        or run_status.get("derived_assets_recompute_required") is not True
        or run_status.get("fresh_ucell_status")
        != "PENDING_CELL_LEVEL_FRESH_RECOMPUTE"
        or run_status.get("single_cell_module_complete") is not False
        or run_status.get("training_started") is not False
        or run_status.get("release_ready") is not False
        or not isinstance(per_cancer, Mapping)
        or len(per_cancer) != 33
        or any(
            not isinstance(value, Mapping)
            or value.get("derived_assets_bound") is not False
            or value.get("historical_predictions_used") is not False
            or value.get("historical_rankings_used") is not False
            for value in per_cancer.values()
        )
    ):
        raise RuntimeError("R7 run status does not support 33/17 unbound scope")
    if (
        supersession.get("format")
        != "CC_HHGT_V3_2_SINGLE_CELL_R7_PORTABLE_SUPERSESSION_V1"
        or supersession.get("run_id") != handoff.get("run_id")
        or supersession.get("supersedes_run_id")
        != run_status.get("supersedes_run_id")
        or supersession.get("derived_assets_copied") is not False
        or supersession.get("derived_assets_recompute_required") is not True
        or supersession.get("derived_assets_relabelled_fresh") is not False
        or supersession.get("release_ready") is not False
    ):
        raise RuntimeError("R7 supersession authority is incomplete or inconsistent")

    release_path = release_binding_path.resolve()
    release = _load_hash_bound(
        release_path, release_binding_sha256, "single-cell gap release binding"
    )
    audit = _load_hash_bound(
        audit_binding_path,
        audit_binding_sha256,
        "single-cell gap independent-audit binding",
    )
    if (
        release.get("format") != "CC_HHGT_V3_2_SINGLE_CELL_GAP_AUDIT_BINDING_V1"
        or release.get("analysis_version")
        != "CancerLncAtlas_V3.2_FULL_MULTITASK"
        or release.get("historical_results_promoted") is not False
        or release.get("candidate_results_promoted") is not False
        or release.get("release_ready") is not False
        or release.get("production_deployed") is not False
    ):
        raise RuntimeError("Single-cell gap release binding violates fresh V3.2 policy")
    if (
        audit.get("format")
        != "CC_HHGT_V3_2_SINGLE_CELL_GAP_INDEPENDENT_AUDIT_BINDING_V1"
        or audit.get("status") != "PASS_HASH_BOUND"
        or audit.get("failed_checks") != 0
        or str(audit.get("release_binding_sha256") or "").lower()
        != release_binding_sha256.lower()
    ):
        raise RuntimeError("Single-cell independent audit does not attest the release")

    report_declaration = release.get("outputs", {}).get("SINGLE_CELL_GAP_AUDIT.json")
    if not isinstance(report_declaration, Mapping):
        raise RuntimeError("Single-cell gap release lacks its report declaration")
    report_path = _resolve_declared_file(report_declaration, "single-cell gap report")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    counts = report.get("lncrna_detection", {}).get("counts", {})
    ucell = report.get("ucell", {})
    pseudotime = report.get("pseudotime", {})
    figures = report.get("figures", {})
    if (
        report.get("format") != "CC_HHGT_V3_2_SINGLE_CELL_GAP_AUDIT_V1"
        or report.get("analysis_version")
        != "CancerLncAtlas_V3.2_FULL_MULTITASK"
        or report.get("historical_predictions_used") is not False
        or report.get("historical_rankings_used") is not False
        or report.get("release_ready") is not False
        or counts.get("cancers") != 33
        or ucell.get("status") != "PARTIAL_SCOPE_HNSC_ONLY"
        or ucell.get("covered_cancers") != ["HNSC"]
        or ucell.get("formal_cancers_covered") != 1
        or ucell.get("formal_cancers_total") != 33
        or pseudotime.get("numeric_rows") != 0
        or figures.get("current_v32_figure_files") != 0
    ):
        raise RuntimeError("Single-cell gap report no longer supports HNSC-only fresh scope")
    return {
        "authority_precedence": "R7_PORTABLE_SUPERSESSION_IS_CURRENT",
        "current_handoff_path": str(current_handoff_path.resolve()),
        "current_handoff_sha256": current_handoff_sha256.lower(),
        "current_run_status_path": str(current_run_status_path.resolve()),
        "current_run_status_sha256": current_run_status_sha256.lower(),
        "current_supersession_path": str(current_supersession_path.resolve()),
        "current_supersession_sha256": current_supersession_sha256.lower(),
        "current_run_id": str(handoff["run_id"]),
        "raw_h5_cancers": int(run_status["scope_count"]),
        "formal_eligible_cancers": int(run_status["formal_eligible_count"]),
        "formal_eligible_cancer_ids": list(formal_cancers),
        "current_fresh_derived_cancers": 0,
        "fresh_ucell_cancers": 0,
        "fresh_ucell_total_cancers": int(run_status["scope_count"]),
        "fresh_ucell_covered_cancers": [],
        "remaining_all_cancers": int(run_status["scope_count"]),
        "remaining_formal_cancers": int(run_status["formal_eligible_count"]),
        "fresh_pseudotime_rows": 0,
        "fresh_figure_files": 0,
        "current_assets": dict(assets),
        "current_fresh_ucell_status": str(handoff["fresh_ucell_status"]),
        "current_fresh_pseudotime_status": str(
            handoff["fresh_pseudotime_status"]
        ),
        "derived_assets_bound": False,
        "derived_assets_recompute_required": True,
        "release_ready": False,
        "superseded_diagnostic": {
            "publishable_as_current": False,
            "release_binding_path": str(release_path),
            "release_binding_sha256": release_binding_sha256.lower(),
            "audit_binding_path": str(audit_binding_path.resolve()),
            "audit_binding_sha256": audit_binding_sha256.lower(),
            "audit_checks": int(audit.get("checks", 0)),
            "audit_failed_checks": int(audit.get("failed_checks", -1)),
            "report_path": str(report_path),
            "report_sha256": str(report_declaration["sha256"]).lower(),
            "historical_scope": "HNSC_ONLY_1_OF_33",
            "historical_covered_cancers": list(ucell["covered_cancers"]),
            "historical_pseudotime_rows": int(pseudotime["numeric_rows"]),
            "historical_figure_files": int(figures["current_v32_figure_files"]),
            "typed_gaps": list(report.get("typed_gaps", [])),
        },
    }


def load_single_cell_fresh_authority(
    *,
    current_handoff_path: Path = SINGLE_CELL_R11_HANDOFF,
    current_handoff_sha256: str = SINGLE_CELL_R11_HANDOFF_SHA256,
    current_run_status_path: Path = SINGLE_CELL_R11_RUN_STATUS,
    current_run_status_sha256: str = SINGLE_CELL_R11_RUN_STATUS_SHA256,
    current_supersession_path: Path = SINGLE_CELL_R11_SUCCESS,
    current_supersession_sha256: str = SINGLE_CELL_R11_SUCCESS_SHA256,
    release_binding_path: Path = SINGLE_CELL_GAP_BINDING,
    release_binding_sha256: str = SINGLE_CELL_GAP_BINDING_SHA256,
    audit_binding_path: Path = SINGLE_CELL_GAP_AUDIT_BINDING,
    audit_binding_sha256: str = SINGLE_CELL_GAP_AUDIT_BINDING_SHA256,
) -> dict[str, Any]:
    """Load the R11 23+10 input authority without claiming derived results."""

    handoff = _load_hash_bound(
        current_handoff_path, current_handoff_sha256, "single-cell R11 handoff"
    )
    run_status = _load_hash_bound(
        current_run_status_path,
        current_run_status_sha256,
        "single-cell R11 run status",
    )
    success = _load_hash_bound(
        current_supersession_path,
        current_supersession_sha256,
        "single-cell R11 success receipt",
    )
    assets = handoff.get("assets")
    expected_assets = {
        "historical_checkpoints",
        "historical_predictions",
        "historical_rankings",
        "historical_trajectory",
    }
    if (
        handoff.get("format")
        != "CC_HHGT_V3_2_SINGLE_CELL_R11_TRAINING_HANDOFF_V1"
        or handoff.get("status") != "PASS"
        or handoff.get("historical_assets_relabelled_fresh") is not False
        or not isinstance(assets, Mapping)
        or set(assets) != expected_assets
        or any(
            not isinstance(value, Mapping) or value.get("path") is not None
            for value in assets.values()
        )
    ):
        raise RuntimeError("R11 handoff does not support a fresh unbound state")
    formal_cancers = run_status.get("formal_eligible_cancers")
    typed_unavailable = run_status.get("typed_unavailable")
    per_cancer = run_status.get("per_cancer")
    if (
        run_status.get("format")
        != "CC_HHGT_V3_2_SINGLE_CELL_R11_RESCUED_RUN_STATUS_V1"
        or run_status.get("status") != "PASS_INPUT_CONTRACT_READY"
        or run_status.get("formal_eligible_cancer_count") != 23
        or run_status.get("typed_unavailable_cancer_count") != 10
        or run_status.get("derived_assets_bound") is not False
        or run_status.get("historical_assets_relabelled_fresh") is not False
        or not isinstance(formal_cancers, list)
        or len(formal_cancers) != 23
        or len(set(formal_cancers)) != 23
        or not isinstance(typed_unavailable, Mapping)
        or len(typed_unavailable) != 10
        or set(formal_cancers) & set(typed_unavailable)
        or not isinstance(per_cancer, Mapping)
        or len(per_cancer) != 33
        or set(per_cancer) != set(formal_cancers) | set(typed_unavailable)
        or any(
            not isinstance(value, Mapping)
            or value.get("formal_eligible") is not (cancer in formal_cancers)
            for cancer, value in per_cancer.items()
        )
        or any(
            value.get(key) is not False
            for value in per_cancer.values()
            for key in ("historical_predictions_used", "historical_rankings_used")
            if key in value
        )
    ):
        raise RuntimeError("R11 run status does not prove the 33-cancer 23+10 input scope")
    manifest_path = current_handoff_path.parent / "dataset_manifest_33c.parquet"
    preflight_path = current_handoff_path.parent / "SERVER_PREFLIGHT.json"
    manifest_sha = _sha256(manifest_path)
    preflight_sha = _sha256(preflight_path)
    if (
        success.get("format")
        != "CC_HHGT_V3_2_SINGLE_CELL_R11_RESCUED_CONTRACT_SUCCESS_V1"
        or success.get("status") != "SUCCESS"
        or success.get("formal_eligible_cancer_count") != 23
        or success.get("typed_unavailable_cancer_count") != 10
        or success.get("run_status_sha256") != current_run_status_sha256.lower()
        or success.get("preflight_sha256") != preflight_sha
        or handoff.get("dataset_manifest_sha256") != manifest_sha
    ):
        raise RuntimeError("R11 success receipt or companion authority hash drift")

    # R11 is the current authority.  The superseded R7/HNSC gap artifacts are
    # useful history when present, but are not required to bind the formal
    # 23-cancer overlay (and may contain paths from a different checkout).
    # An explicitly omitted complete pair is valid for a formal23-only
    # staging bundle.  If legacy bindings are supplied, keep validating them
    # as a complete path/SHA pair for compatibility with older candidates.
    if (release_binding_path is None) != (audit_binding_path is None) or (
        release_binding_path is None
    ) != (release_binding_sha256 is None) or (
        audit_binding_path is None
    ) != (audit_binding_sha256 is None):
        raise RuntimeError(
            "single-cell legacy gap bindings must be supplied as a complete path/SHA pair"
        )
    if release_binding_path is None:
        superseded_diagnostic = {
            "publishable_as_current": False,
            "status": "SUPERSEDED_DIAGNOSTIC_AUTHORITY_NOT_MOUNTED",
            "historical_scope": "HNSC_ONLY_1_OF_33",
            "historical_covered_cancers": ["HNSC"],
            "historical_pseudotime_rows": 0,
            "historical_figure_files": 0,
            "typed_gaps": [],
        }
    else:
        try:
            legacy = _load_single_cell_r7_fresh_authority(
                release_binding_path=release_binding_path,
                release_binding_sha256=str(release_binding_sha256),
                audit_binding_path=audit_binding_path,
                audit_binding_sha256=str(audit_binding_sha256),
            )
            superseded_diagnostic = legacy["superseded_diagnostic"]
        except (OSError, RuntimeError):
            superseded_diagnostic = {
                "publishable_as_current": False,
                "status": "SUPERSEDED_DIAGNOSTIC_AUTHORITY_NOT_MOUNTED",
                "historical_scope": "HNSC_ONLY_1_OF_33",
                "historical_covered_cancers": ["HNSC"],
                "historical_pseudotime_rows": 0,
                "historical_figure_files": 0,
                "typed_gaps": [],
            }
    return {
        "authority_precedence": "R11_RESCUED_INPUT_CONTRACT_IS_CURRENT",
        "current_handoff_path": str(current_handoff_path.resolve()),
        "current_handoff_sha256": current_handoff_sha256.lower(),
        "current_run_status_path": str(current_run_status_path.resolve()),
        "current_run_status_sha256": current_run_status_sha256.lower(),
        "current_supersession_path": str(current_supersession_path.resolve()),
        "current_supersession_sha256": current_supersession_sha256.lower(),
        "current_run_id": "v32-single-cell-r11-rescued-20260903",
        "raw_h5_cancers": len(per_cancer),
        "formal_eligible_cancers": len(formal_cancers),
        "formal_eligible_cancer_ids": list(formal_cancers),
        "typed_unavailable_cancers": len(typed_unavailable),
        "typed_unavailable": dict(typed_unavailable),
        "current_fresh_derived_cancers": 0,
        "fresh_ucell_cancers": 0,
        "fresh_ucell_total_cancers": len(per_cancer),
        "fresh_ucell_covered_cancers": [],
        "remaining_all_cancers": len(per_cancer),
        "remaining_formal_cancers": len(formal_cancers),
        "fresh_pseudotime_rows": 0,
        "fresh_figure_files": 0,
        "current_assets": dict(assets),
        # The R11 receipt is an input-scope authority.  The accepted formal23
        # derived overlay is produced and hash-bound on server 149; when this
        # local checkout is built without that sidecar, report a mount gap,
        # never a request to repeat the rescue/recompute.
        "current_fresh_ucell_status": "FORMAL23_OVERLAY_SERVER_BINDING_REQUIRED",
        "current_fresh_pseudotime_status": "TYPED_UNAVAILABLE_BY_ACCEPTED_SCOPE",
        "derived_assets_bound": False,
        "derived_assets_recompute_required": False,
        "formal23_overlay_server_binding_required": True,
        "release_ready": False,
        "superseded_diagnostic": superseded_diagnostic,
    }


def load_single_cell_formal23_authority(
    success_path: Path | None, success_sha256: str | None
) -> tuple[SingleCellFormal23Query | None, dict[str, Any] | None]:
    if (success_path is None) != (success_sha256 is None):
        raise RuntimeError(
            "formal-23 single-cell catalog binding requires both SUCCESS and SHA256"
        )
    if success_path is None:
        return None, None
    query = SingleCellFormal23Query(
        success_path, expected_sha256=str(success_sha256)
    )
    capability = query.capability_status()
    downloads = [
        query.download_entry(download_id)
        for download_id in (
            "single_cell_associations",
            "single_cell_activity",
            "single_cell_ucell",
            "single_cell_pseudotime",
            "single_cell_figures",
        )
    ]
    return query, {
        "status": capability["status"],
        "scientific_status": "PASS_23_PLUS_10_TYPED_UNAVAILABLE",
        "module_ready_for_binding": True,
        "production_deployed": False,
        "formal_eligible_cancer_count": 23,
        "typed_unavailable_cancer_count": 10,
        "full_33_single_cell_coverage_claimed": False,
        "total_cells": capability["total_cells"],
        "total_association_evidence_rows": capability[
            "total_association_evidence_rows"
        ],
        "ready_downloads": [
            {
                "download_id": item["download_id"],
                "status": item["status"],
                "part_count": len(item.get("parts", [])),
                "sha256_tree": item.get("sha256_tree"),
            }
            for item in downloads
            if item["status"] == "READY_PARTS"
        ],
        "typed_gap_downloads": [
            {
                "download_id": item["download_id"],
                "status": item["status"],
                "reason": item.get("unavailable_reason"),
            }
            for item in downloads
            if item["status"] == "GAP_TYPED_UNAVAILABLE"
        ],
        "success_path": str(Path(success_path).resolve()),
        "success_sha256": str(success_sha256).lower(),
        "binding_sha256": capability["binding_sha256"],
        "independent_audit_sha256": capability["independent_audit_sha256"],
        "single_cell_fusion_weight": 0.0,
        "changes_exact_primary": False,
    }


def build(
    *,
    single_cell_current_handoff_path: Path = SINGLE_CELL_R11_HANDOFF,
    single_cell_current_handoff_sha256: str = SINGLE_CELL_R11_HANDOFF_SHA256,
    single_cell_current_run_status_path: Path = SINGLE_CELL_R11_RUN_STATUS,
    single_cell_current_run_status_sha256: str = SINGLE_CELL_R11_RUN_STATUS_SHA256,
    single_cell_current_supersession_path: Path = SINGLE_CELL_R11_SUCCESS,
    single_cell_current_supersession_sha256: str = SINGLE_CELL_R11_SUCCESS_SHA256,
    single_cell_release_binding_path: Path | None = SINGLE_CELL_GAP_BINDING,
    single_cell_release_binding_sha256: str | None = SINGLE_CELL_GAP_BINDING_SHA256,
    single_cell_audit_binding_path: Path | None = SINGLE_CELL_GAP_AUDIT_BINDING,
    single_cell_audit_binding_sha256: str | None = SINGLE_CELL_GAP_AUDIT_BINDING_SHA256,
    single_cell_formal23_success_path: Path | None = None,
    single_cell_formal23_success_sha256: str | None = None,
    directional_cnv_binding_path: Path | None = None,
    directional_cnv_binding_sha256: str | None = None,
    directional_cnv_audit_binding_path: Path | None = None,
    directional_cnv_audit_binding_sha256: str | None = None,
    capability_scope_path: Path | None = None,
    capability_scope_sha256: str | None = None,
) -> dict:
    contract = load_parity_config()
    # Validate the current formal23 sidecar first.  Once it is present, the
    # superseded R7/HNSC gap bindings are optional history and must not make a
    # formal23-only staging candidate fail closed.
    _, single_cell_formal23_authority = load_single_cell_formal23_authority(
        single_cell_formal23_success_path,
        single_cell_formal23_success_sha256,
    )
    legacy_gap_values = (
        single_cell_release_binding_path,
        single_cell_release_binding_sha256,
        single_cell_audit_binding_path,
        single_cell_audit_binding_sha256,
    )
    if any(value is not None for value in legacy_gap_values) and not all(
        value is not None for value in legacy_gap_values
    ):
        raise RuntimeError(
            "single-cell legacy gap bindings must be supplied as a complete path/SHA pair"
        )
    if (
        single_cell_formal23_authority is None
        and single_cell_release_binding_path is None
    ):
        raise RuntimeError(
            "legacy single-cell gap bindings may be omitted only when formal23 is bound"
        )
    legacy_release_binding_path = single_cell_release_binding_path
    legacy_release_binding_sha256 = single_cell_release_binding_sha256
    legacy_audit_binding_path = single_cell_audit_binding_path
    legacy_audit_binding_sha256 = single_cell_audit_binding_sha256
    if single_cell_formal23_authority is not None:
        legacy_release_binding_path = None
        legacy_release_binding_sha256 = None
        legacy_audit_binding_path = None
        legacy_audit_binding_sha256 = None
    single_cell_authority = load_single_cell_fresh_authority(
        current_handoff_path=single_cell_current_handoff_path,
        current_handoff_sha256=single_cell_current_handoff_sha256,
        current_run_status_path=single_cell_current_run_status_path,
        current_run_status_sha256=single_cell_current_run_status_sha256,
        current_supersession_path=single_cell_current_supersession_path,
        current_supersession_sha256=single_cell_current_supersession_sha256,
        release_binding_path=legacy_release_binding_path,
        release_binding_sha256=legacy_release_binding_sha256,
        audit_binding_path=legacy_audit_binding_path,
        audit_binding_sha256=legacy_audit_binding_sha256,
    )
    single_cell_scope = load_single_cell_release_scope()
    directional_cnv_authority = load_directional_cnv_authority(
        directional_cnv_binding_path,
        directional_cnv_binding_sha256,
        directional_cnv_audit_binding_path,
        directional_cnv_audit_binding_sha256,
    )
    capability_scope = load_capability_scope(
        capability_scope_path, capability_scope_sha256
    )
    capabilities = contract["capabilities"]
    if set(STAGING_BINDINGS) != set(capabilities):
        raise RuntimeError(
            "Staging UI mapping differs from the parity contract: "
            f"missing={sorted(set(capabilities) - set(STAGING_BINDINGS))}, "
            f"extra={sorted(set(STAGING_BINDINGS) - set(capabilities))}"
        )
    rows = []
    for capability_id, value in capabilities.items():
        status, endpoints = STAGING_BINDINGS[capability_id]
        scope_entry = (
            capability_scope["capabilities"][capability_id]
            if capability_scope is not None
            else None
        )
        if scope_entry is not None:
            status = scope_entry["staging_status"]
            endpoints = scope_entry["verified_endpoints"]
        if capability_id == "single_cell":
            if single_cell_formal23_authority is None:
                status = PENDING
                endpoints = []
            elif scope_entry is None:
                status = READY
        elif capability_id == "cnv":
            if directional_cnv_authority is None:
                status = PENDING
                endpoints = []
            elif scope_entry is None:
                status = READY
        if status in {READY, PARTIAL} and not endpoints:
            # An authority sidecar alone is not evidence that its API routes
            # are mounted.  This guard also catches an inconsistent scope
            # receipt after the single-cell/CNV sidecar checks above.
            raise RuntimeError(
                f"{capability_id} is advertised as {status} without endpoints"
            )
        gates = value["gates"]
        known_gap = (
            scope_entry["known_gap"]
            if scope_entry is not None and scope_entry.get("known_gap") is not None
            else None
        )
        if known_gap is None:
            known_gap = (
                (
                    "Accepted formal scope: 23 eligible cancers plus 10 typed-unavailable "
                    "cancers. Full 33-cancer single-cell coverage is not claimed and is "
                    "not a release gate. The older HNSC-only result is superseded "
                    "diagnostic evidence. The formal23 derived overlay is hash-bound "
                    "and queryable on server 149; pseudotime and figure downloads "
                    "remain typed-unavailable."
                )
                if capability_id == "single_cell" and single_cell_formal23_authority is not None
                else (
                    "Accepted formal scope: 23 eligible cancers plus 10 typed-unavailable "
                    "cancers. Full 33-cancer single-cell coverage is not claimed and is "
                    "not a release gate. The formal23 derived overlay is available only "
                    "through the hash-bound server-149 sidecar in this staging build; "
                    "do not start another rescue or recomputation."
                    if capability_id == "single_cell"
                    else None
                    if capability_id == "cnv" and directional_cnv_authority is not None
                    else KNOWN_GAPS.get(capability_id)
                )
            )
        if scope_entry is not None and status == PENDING and known_gap is None:
            known_gap = (
                "The scientific authority may exist, but this route is not mounted "
                "in the isolated staging runtime. It remains explicitly unavailable "
                "until a server-side route probe and hash-bound binding receipt pass."
            )
        rows.append(
            {
                "capability_id": capability_id,
                "description": value.get("description", capability_id),
                "target_level": value["target_level"],
                "staging_status": status,
                "all_four_parity_gates_pass": False,
                "staging_endpoints": endpoints,
                "known_gap": known_gap,
                "interpretation_note": INTERPRETATION_NOTES.get(capability_id),
                **(
                    {
                        "coverage_scope": [
                            {
                                "label": "Raw H5 audited",
                                "available": single_cell_authority["raw_h5_cancers"],
                                "total": 33,
                            },
                            {
                                "label": "Formal eligible inputs",
                                "available": single_cell_authority[
                                    "formal_eligible_cancers"
                                ],
                                "total": 33,
                            },
                            {
                                "label": "Typed unavailable by accepted policy",
                                "available": single_cell_scope[
                                    "typed_unavailable_cancer_count"
                                ],
                                "total": 33,
                            },
                            {
                                "label": "Current fresh derived assets / eligible scope",
                                "available": (
                                    23
                                    if single_cell_formal23_authority is not None
                                    else single_cell_authority["fresh_ucell_cancers"]
                                ),
                                "total": single_cell_scope[
                                    "formal_eligible_cancer_count"
                                ],
                            },
                            {
                                "label": "Fresh pseudotime + figures / eligible scope",
                                "available": 0,
                                "total": single_cell_scope[
                                    "formal_eligible_cancer_count"
                                ],
                            },
                        ],
                        "accepted_release_scope": single_cell_scope,
                        "formal_context_delivery": (
                            single_cell_formal23_authority
                            if single_cell_formal23_authority is not None
                            else {
                            "status": "UNBOUND_SERVER_FORMAL23_MOUNT_REQUIRED",
                            "scientific_status": "PASS_INPUT_SCOPE_NOT_MOUNTED",
                            "binding_action": "RUN_SERVER_ONLY_STAGING_BINDING",
                            "recompute_required": False,
                            "release_ready": False,
                            "fresh_authority": single_cell_authority,
                            "ready_downloads": [],
                            "partial_downloads": [],
                            "typed_gap_downloads": [
                                {
                                    "download_id": "single_cell_associations",
                                    "status": "SERVER_FORMAL23_BINDING_REQUIRED",
                                    "reason": "CURRENT_R11_FORMAL23_ASSOCIATION_AND_LNCRNA_CELLTYPE_ASSETS_NOT_MOUNTED_IN_THIS_CHECKOUT",
                                },
                                {
                                    "download_id": "single_cell_activity",
                                    "status": "SERVER_FORMAL23_BINDING_REQUIRED",
                                    "reason": "CURRENT_R11_FORMAL23_ACTIVITY_ASSET_NOT_MOUNTED_IN_THIS_CHECKOUT",
                                },
                                {
                                    "download_id": "single_cell_ucell",
                                    "status": "SERVER_FORMAL23_BINDING_REQUIRED",
                                    "reason": "CURRENT_R11_FORMAL23_UCELL_ASSET_NOT_MOUNTED_IN_THIS_CHECKOUT",
                                },
                                {
                                    "download_id": "single_cell_pseudotime",
                                    "status": "GAP_TYPED_UNAVAILABLE",
                                    "reason": "PSEUDOTIME_IS_TYPED_UNAVAILABLE_BY_ACCEPTED_23_PLUS_10_SCOPE",
                                },
                                {
                                    "download_id": "single_cell_figures",
                                    "status": "GAP_TYPED_UNAVAILABLE",
                                    "reason": "NO_HASH_BOUND_FRESH_V32_SINGLE_CELL_FIGURE_FILES",
                                },
                            ],
                            "remaining_all_cancers": single_cell_authority[
                                "remaining_all_cancers"
                            ],
                            "formal_eligible_cancers": single_cell_authority[
                                "formal_eligible_cancers"
                            ],
                            "remaining_formal_cancers": single_cell_authority[
                                "remaining_formal_cancers"
                            ],
                            "formal_figure_files": single_cell_authority[
                                "fresh_figure_files"
                            ],
                            "fresh_pseudotime_numeric_rows": single_cell_authority[
                                "fresh_pseudotime_rows"
                            ],
                            "single_cell_fusion_weight": 0.0,
                            "changes_exact_primary": False,
                            "production_deployed": False,
                            }
                        ),
                    }
                    if capability_id == "single_cell"
                    else {}
                ),
                **(
                    {"directional_cnv_delivery": directional_cnv_authority}
                    if capability_id == "cnv"
                    and directional_cnv_authority is not None
                    else {}
                ),
                "required_api_contract": list(gates["api"]["ids"]),
                "ui_surface_ids": list(gates["ui"]["ids"]),
                "download_ids": list(gates["download"]["ids"]),
            }
        )
    return {
        "schema_version": "CANCERLNCATLAS_V32_STAGING_WEB_CATALOG_V1",
        "model_version": "V3.2",
        "environment": "staging",
        "production_deployed": False,
        "release_ready": False,
        "status_scope": "staging queryability only; never implies artifact/API/UI/download parity",
        **(
            {
                "capability_scope": {
                    "path": capability_scope["scope_path"],
                    "sha256": capability_scope["scope_sha256"],
                    "probe_status": capability_scope["status"],
                }
            }
            if capability_scope is not None
            else {}
        ),
        "capability_count": len(rows),
        "capabilities": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the staging catalog from hash-bound scientific authorities"
    )
    parser.add_argument(
        "--single-cell-current-handoff",
        type=Path,
        default=SINGLE_CELL_R11_HANDOFF,
    )
    parser.add_argument(
        "--single-cell-current-handoff-sha256",
        default=SINGLE_CELL_R11_HANDOFF_SHA256,
    )
    parser.add_argument(
        "--single-cell-current-run-status",
        type=Path,
        default=SINGLE_CELL_R11_RUN_STATUS,
    )
    parser.add_argument(
        "--single-cell-current-run-status-sha256",
        default=SINGLE_CELL_R11_RUN_STATUS_SHA256,
    )
    parser.add_argument(
        "--single-cell-current-supersession",
        type=Path,
        default=SINGLE_CELL_R11_SUCCESS,
    )
    parser.add_argument(
        "--single-cell-current-supersession-sha256",
        default=SINGLE_CELL_R11_SUCCESS_SHA256,
    )
    parser.add_argument(
        "--single-cell-release-binding",
        type=Path,
        default=SINGLE_CELL_GAP_BINDING,
    )
    parser.add_argument(
        "--single-cell-release-binding-sha256",
        default=SINGLE_CELL_GAP_BINDING_SHA256,
    )
    parser.add_argument(
        "--single-cell-audit-binding",
        type=Path,
        default=SINGLE_CELL_GAP_AUDIT_BINDING,
    )
    parser.add_argument(
        "--single-cell-audit-binding-sha256",
        default=SINGLE_CELL_GAP_AUDIT_BINDING_SHA256,
    )
    parser.add_argument("--single-cell-formal23-success", type=Path)
    parser.add_argument("--single-cell-formal23-success-sha256")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "website" / "frontend" / "v32-capability-catalog.json",
    )
    parser.add_argument("--directional-cnv-binding", type=Path)
    parser.add_argument("--directional-cnv-binding-sha256")
    parser.add_argument("--directional-cnv-audit-binding", type=Path)
    parser.add_argument("--directional-cnv-audit-binding-sha256")
    parser.add_argument(
        "--capability-scope",
        type=Path,
        help=(
            "server-149 hash-bound route scope; when supplied, capabilities "
            "not listed as probed are fail-closed"
        ),
    )
    parser.add_argument("--capability-scope-sha256")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    payload = build(
        single_cell_current_handoff_path=args.single_cell_current_handoff,
        single_cell_current_handoff_sha256=(
            args.single_cell_current_handoff_sha256
        ),
        single_cell_current_run_status_path=args.single_cell_current_run_status,
        single_cell_current_run_status_sha256=(
            args.single_cell_current_run_status_sha256
        ),
        single_cell_current_supersession_path=(
            args.single_cell_current_supersession
        ),
        single_cell_current_supersession_sha256=(
            args.single_cell_current_supersession_sha256
        ),
        single_cell_release_binding_path=args.single_cell_release_binding,
        single_cell_release_binding_sha256=args.single_cell_release_binding_sha256,
        single_cell_audit_binding_path=args.single_cell_audit_binding,
        single_cell_audit_binding_sha256=args.single_cell_audit_binding_sha256,
        single_cell_formal23_success_path=args.single_cell_formal23_success,
        single_cell_formal23_success_sha256=(
            args.single_cell_formal23_success_sha256
        ),
        directional_cnv_binding_path=args.directional_cnv_binding,
        directional_cnv_binding_sha256=args.directional_cnv_binding_sha256,
        directional_cnv_audit_binding_path=args.directional_cnv_audit_binding,
        directional_cnv_audit_binding_sha256=(
            args.directional_cnv_audit_binding_sha256
        ),
        capability_scope_path=args.capability_scope,
        capability_scope_sha256=args.capability_scope_sha256,
    )
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
