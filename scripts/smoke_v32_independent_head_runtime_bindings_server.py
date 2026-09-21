#!/usr/bin/env python3
"""Strict, hash-pinned runtime smoke for every V3.2 independent head."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

from cc_hhgt.v32.clinical_km_query import ClinicalKMReleaseQuery
from cc_hhgt.v32.evidence_direction_query import EvidenceDirectionProbabilityQuery
from cc_hhgt.v32.evidence_query import EvidenceBindingQuery
from cc_hhgt.v32.gene_set_subtype_query import GeneSetSubtypeReleaseQuery
from cc_hhgt.v32.gene_set_subtype_release import TYPED_UNAVAILABLE
from cc_hhgt.v32.state_gene_set_query import StateGeneSetReleaseQuery


RUNTIME_BINDING_FORMAT = "CANCERLNCATLAS_V32_INDEPENDENT_HEAD_RUNTIME_BINDINGS_V1"
RUNTIME_BINDING_STATUS = "SERVER_UPLOAD_PENDING_RUNTIME_VALIDATION"
GENE_SET_METADATA_CLOSED = "RUNTIME_TRANSITIVE_METADATA_CLOSED"
GENE_SET_CLOSURE_FORMAT = (
    "CANCERLNCATLAS_V32_GENE_SET_RUNTIME_METADATA_CLOSURE_V1"
)
SMOKE_FORMAT = "CANCERLNCATLAS_V32_INDEPENDENT_HEAD_RUNTIME_SMOKE_V2"
ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
PROBE_LNCRNA = "LNC:ENSG00000099869"
EXPECTED_COMPONENTS = {
    "clinical": "secondary_fresh_lncrna_survival_statistics",
    "evidence": "partial_not_publishable",
    "evidence_direction": "auxiliary_direction_probability",
    "gene_set_ranked_subtype": (
        "deterministic_v32_exact_pathway_derived_functional_head"
    ),
    "state_gene_set": "secondary_state_gene_set_and_report",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class RuntimeSmokeError(RuntimeError):
    """A pinned runtime closure or an exact query probe failed closed."""


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeSmokeError(message)


def _nonempty_rows(result: Mapping[str, Any], count_key: str, label: str) -> list[dict[str, Any]]:
    rows = result.get("rows")
    declared = result.get(count_key)
    _require(isinstance(rows, list), f"{label} did not return a row list")
    _require(
        isinstance(declared, int) and declared == len(rows) and declared > 0,
        f"{label} exact probe returned no rows or a contradictory row count",
    )
    _require(all(isinstance(row, dict) for row in rows), f"{label} returned malformed rows")
    return rows


def _validate_declaration(role: str, declaration: Any) -> dict[str, Any]:
    _require(isinstance(declaration, dict), f"Missing runtime binding: {role}")
    expected_status = EXPECTED_COMPONENTS[role]
    _require(
        declaration.get("scientific_status") == expected_status,
        f"{role} scientific status drifted",
    )
    for flag in ("production_deployed", "release_ready"):
        _require(declaration.get(flag) is False, f"{role} invalid {flag}")
    for key in ("binding_path", "binding_sha256"):
        _require(str(declaration.get(key, "")).strip(), f"{role} lacks {key}")
    _require(
        _SHA256.fullmatch(str(declaration["binding_sha256"])),
        f"{role} binding SHA256 is invalid",
    )
    if role in {"evidence_direction", "gene_set_ranked_subtype"}:
        for key in ("audit_binding_path", "audit_binding_sha256"):
            _require(str(declaration.get(key, "")).strip(), f"{role} lacks {key}")
        _require(
            _SHA256.fullmatch(str(declaration["audit_binding_sha256"])),
            f"{role} audit binding SHA256 is invalid",
        )
    return declaration


def load_runtime_manifest(path: Path, expected_sha256: str) -> dict[str, Any]:
    expected = str(expected_sha256).lower()
    _require(_SHA256.fullmatch(expected) is not None, "Expected runtime-binding SHA256 is invalid")
    _require(not path.is_symlink() and path.is_file(), "Runtime binding is missing or a symlink")
    observed = sha(path)
    _require(observed == expected, f"Runtime-binding SHA256 mismatch: {observed} != {expected}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeSmokeError("Runtime binding is invalid JSON") from exc
    _require(isinstance(manifest, dict), "Runtime binding must be a JSON object")
    _require(manifest.get("format") == RUNTIME_BINDING_FORMAT, "Runtime-binding format drifted")
    _require(manifest.get("status") == RUNTIME_BINDING_STATUS, "Runtime-binding status drifted")
    _require(manifest.get("main_score_changed") is False, "Runtime binding changes main score")
    _require(manifest.get("production_deployed") is False, "Runtime binding claims deployment")
    _require(manifest.get("release_ready") is False, "Runtime binding claims release readiness")
    server_root = str(manifest.get("server_root", ""))
    _require(
        any(
            server_root.startswith(prefix)
            for prefix in (
                "./data/CancerLncAtlas/runtime/authorized_bindings/",
                "./data/CancerLncAtlas/runtime/authorized_acceptance/",
            )
        ),
        "Runtime-binding server root is outside the authorized closure",
    )

    gene_summary = manifest.get("gene_set_ranked_subtype")
    _require(isinstance(gene_summary, dict), "Gene Set metadata status is missing")
    _require(
        gene_summary.get("status") == GENE_SET_METADATA_CLOSED,
        "Gene Set transitive metadata is not closed; runtime validation must fail",
    )
    _require(
        gene_summary.get("production_deployed") is False
        and gene_summary.get("release_ready") is False,
        "Gene Set metadata closure has invalid publication flags",
    )

    closure_ref = gene_summary.get("closure_receipt")
    _require(isinstance(closure_ref, dict), "Gene Set closure receipt is missing")
    closure_path = Path(str(closure_ref.get("path", "")))
    closure_sha = str(closure_ref.get("sha256", "")).lower()
    _require(
        _SHA256.fullmatch(closure_sha) is not None
        and closure_path.is_file()
        and not closure_path.is_symlink(),
        "Gene Set closure receipt path/SHA is invalid",
    )
    _require(sha(closure_path) == closure_sha, "Gene Set closure receipt SHA drifted")
    try:
        closure = json.loads(closure_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeSmokeError("Gene Set closure receipt is invalid JSON") from exc
    metadata = closure.get("metadata_closure", {}) if isinstance(closure, dict) else {}
    probes = closure.get("exact_probes", {}) if isinstance(closure, dict) else {}
    resource = closure.get("resource_gate", {}) if isinstance(closure, dict) else {}
    _require(
        isinstance(closure, dict)
        and closure.get("format") == GENE_SET_CLOSURE_FORMAT
        and closure.get("status") == "PASS"
        and closure_ref.get("format") == GENE_SET_CLOSURE_FORMAT
        and closure_ref.get("status") == "PASS"
        and metadata.get("transitive_metadata_closed") is True
        and metadata.get("release_records_hash_validated") == 47
        and metadata.get("upstream_gene_set_records_hash_validated") == 47
        and metadata.get("upstream_gene_set_unique_records") == 44
        and metadata.get("legacy_gene_set_transitive_unresolved") == 24
        and metadata.get("legacy_ranked_subtype_transitive_unresolved") == 72
        and metadata.get("legacy_local_unresolved") == 0
        and metadata.get("portable_rebinder_pending_was_runtime_false_negative")
        is True
        and probes.get("representative_member_rows") == 1
        and probes.get("cancers_typed") == 33
        and probes.get("available_cancers") == 31
        and probes.get("typed_unavailable") == TYPED_UNAVAILABLE
        and resource.get("passed") is True
        and 0 < int(resource.get("kernel_peak_rss_kB") or 0)
        <= int(resource.get("max_peak_rss_kB") or 0)
        <= 2 * 1024 * 1024
        and closure.get("family_to_exact_broadcast") is False
        and closure.get("changes_primary_ranking") is False
        and closure.get("production_deployed") is False
        and closure.get("release_ready") is False,
        "Gene Set closure receipt did not pass the exact metadata contract",
    )

    bindings = manifest.get("bindings")
    _require(isinstance(bindings, dict), "Runtime bindings are missing")
    _require(set(bindings) == set(EXPECTED_COMPONENTS), "Runtime component set is incomplete or has extras")
    for role in sorted(EXPECTED_COMPONENTS):
        _validate_declaration(role, bindings[role])
    _require(
        closure.get("runtime_binding_declaration")
        == bindings["gene_set_ranked_subtype"],
        "Gene Set closure is tied to another runtime declaration",
    )
    return manifest


def run_smoke(manifest_path: Path, expected_sha256: str) -> dict[str, Any]:
    manifest = load_runtime_manifest(manifest_path, expected_sha256)
    bindings = manifest["bindings"]

    evidence_decl = bindings["evidence"]
    evidence = EvidenceBindingQuery(
        evidence_decl["binding_path"],
        expected_binding_sha256=evidence_decl["binding_sha256"],
    )
    evidence_candidates = evidence.query_confidence(
        lncrna_id=PROBE_LNCRNA, availability=True, limit=50
    )
    evidence_rows = _nonempty_rows(
        evidence_candidates, "returned_rows", "Evidence confidence"
    )
    evidence_row = next(
        (row for row in evidence_rows if int(row.get("event_count") or 0) > 0),
        None,
    )
    _require(evidence_row is not None, "Evidence confidence probe has no event-backed exact key")
    evidence_event = evidence.query_events(
        cancer_id=evidence_row["cancer_id"],
        lncrna_id=evidence_row["lncrna_id"],
        pathway_id=evidence_row["pathway_id"],
        limit=1,
    )
    event_rows = _nonempty_rows(evidence_event, "returned_rows", "Evidence event lineage")
    event_row = event_rows[0]
    for key in ("cancer_id", "lncrna_id", "pathway_id"):
        _require(event_row.get(key) == evidence_row.get(key), f"Evidence event exact key drifted: {key}")
    _require(event_row.get("family_broadcast_used") is False, "Evidence event used family broadcast")

    direction_decl = bindings["evidence_direction"]
    direction = EvidenceDirectionProbabilityQuery(
        direction_decl["binding_path"],
        expected_binding_sha256=direction_decl["binding_sha256"],
        audit_binding_path=direction_decl["audit_binding_path"],
        expected_audit_binding_sha256=direction_decl["audit_binding_sha256"],
    )
    direction_probe = direction.query(available=True, limit=1)
    direction_rows = _nonempty_rows(
        direction_probe, "returned_rows", "Evidence direction"
    )
    direction_row = direction_rows[0]
    probabilities = [
        float(direction_row[name])
        for name in (
            "direction_negative_probability",
            "direction_neutral_probability",
            "direction_positive_probability",
        )
    ]
    _require(all(math.isfinite(value) and 0 <= value <= 1 for value in probabilities), "Direction probabilities are invalid")
    _require(abs(sum(probabilities) - 1.0) <= 1e-5, "Direction probabilities do not sum to one")
    for key in ("changes_primary_ranking", "changes_discovery_ranking", "used_for_fusion"):
        _require(direction_row.get(key) is False, f"Direction probe invalid {key}")

    clinical_decl = bindings["clinical"]
    clinical = ClinicalKMReleaseQuery(
        clinical_decl["binding_path"],
        expected_binding_sha256=clinical_decl["binding_sha256"],
    )
    clinical_probe = clinical.query_pair(
        cancer_id="BLCA",
        lncrna_id=PROBE_LNCRNA,
        clinical_endpoint="OS",
        include_curves=True,
    )
    clinical_statistics = clinical_probe.get("statistics")
    clinical_curves = clinical_probe.get("curves")
    _require(
        clinical_probe.get("returned_statistics_rows") == 1
        and isinstance(clinical_statistics, list)
        and len(clinical_statistics) == 1,
        "Clinical BLCA/OS exact probe returned no statistic",
    )
    _require(
        clinical_probe.get("returned_curve_rows") == 12
        and isinstance(clinical_curves, list)
        and len(clinical_curves) == 12,
        "Clinical BLCA/OS exact probe did not return the twelve KM curve points",
    )
    clinical_row = clinical_statistics[0]
    _require(clinical_row.get("availability") is True, "Clinical BLCA/OS probe is unavailable")
    _require(
        0 <= int(clinical_row["n_events"]) <= int(clinical_row["n_endpoint_patients"]),
        "Clinical event count exceeds endpoint-patient count",
    )
    _require(
        int(clinical_row["events_high"]) <= int(clinical_row["n_high"])
        and int(clinical_row["events_low"]) <= int(clinical_row["n_low"]),
        "Clinical group event count exceeds group-patient count",
    )

    state_decl = bindings["state_gene_set"]
    state = StateGeneSetReleaseQuery(
        state_decl["binding_path"],
        expected_binding_sha256=state_decl["binding_sha256"],
    )
    state_probe = state.query_gene_sets(limit=1)
    state_rows = _nonempty_rows(state_probe, "returned_rows", "State Gene Set")
    state_id = str(state_rows[0]["gene_set_id"])
    state_members = state.query_members(gene_set_id=state_id, limit=1)
    member_rows = _nonempty_rows(state_members, "returned_rows", "State Gene Set member")
    _require(member_rows[0].get("gene_set_id") == state_id, "State member exact key drifted")

    gene_decl = bindings["gene_set_ranked_subtype"]
    gene_query = GeneSetSubtypeReleaseQuery(
        gene_decl["binding_path"],
        expected_binding_sha256=gene_decl["binding_sha256"],
        audit_binding_path=gene_decl["audit_binding_path"],
        expected_audit_binding_sha256=gene_decl["audit_binding_sha256"],
    )
    gene_catalog = gene_query.query_gene_set_catalog(cancer_id="BRCA", limit=1)
    gene_rows = _nonempty_rows(gene_catalog, "returned_rows", "Ranked Gene Set catalog")
    _require(gene_catalog.get("availability", {}).get("available") is True, "BRCA Gene Set is not typed available")
    gene_set_id = str(gene_rows[0]["geneset_id"])
    gene_detail = gene_query.query_gene_set_detail(
        gene_set_id=gene_set_id, member_limit=1
    )
    _require(gene_detail.get("member_returned_rows") == 1, "Representative Gene Set has no member")
    _require(
        gene_detail.get("gene_set", {}).get("pathway_target_level") == "exact_pathway",
        "Representative Gene Set is not exact-pathway level",
    )
    overview = gene_query.query_subtype_overview()
    overview_rows = _nonempty_rows(
        overview, "returned_rows", "33-cancer ranked-subtype overview"
    )
    _require(len(overview_rows) == 33, "Ranked-subtype overview is not 33-cancer complete")
    _require(
        len({row.get("cancer_id") for row in overview_rows}) == 33,
        "Ranked-subtype overview has duplicate cancers",
    )
    observed_unavailable = {
        str(row["cancer_id"]): str(row["unavailable_reason"])
        for row in overview_rows
        if row.get("available") is False
    }
    _require(
        observed_unavailable == TYPED_UNAVAILABLE,
        "Ranked-subtype typed-unavailable cancer map drifted",
    )
    _require(
        sum(row.get("available") is True for row in overview_rows) == 31,
        "Ranked-subtype available-cancer count is not 31",
    )

    return {
        "format": SMOKE_FORMAT,
        "status": "PASS",
        "runtime_bindings": {
            "path": str(manifest_path),
            "sha256": sha(manifest_path),
            "format": manifest["format"],
            "status": manifest["status"],
            "exact_component_set_validated": True,
        },
        "components": {
            "evidence": {
                "binding_sha256": evidence.binding_sha256,
                "confidence_exact_key": {
                    key: evidence_row[key]
                    for key in ("cancer_id", "lncrna_id", "pathway_id")
                },
                "event_probe_rows": evidence_event["returned_rows"],
                "family_broadcast_used": False,
                "scientific_status": EXPECTED_COMPONENTS["evidence"],
            },
            "evidence_direction": {
                "binding_sha256": direction.binding_sha256,
                "audit_binding_sha256": direction.audit_binding_sha256,
                "probe_rows": direction_probe["returned_rows"],
                "probability_sum_validated": True,
                "scientific_status": EXPECTED_COMPONENTS["evidence_direction"],
            },
            "clinical": {
                "binding_sha256": clinical.binding_sha256,
                "exact_probe": {
                    "cancer_id": "BLCA",
                    "lncrna_id": PROBE_LNCRNA,
                    "clinical_endpoint": "OS",
                },
                "probe_rows": 1,
                "curve_rows": 12,
                "events_lte_patients_validated": True,
                "scientific_status": EXPECTED_COMPONENTS["clinical"],
            },
            "state_gene_set": {
                "binding_sha256": state.binding_sha256,
                "gene_set_id": state_id,
                "member_probe_rows": state_members["returned_rows"],
                "scientific_status": EXPECTED_COMPONENTS["state_gene_set"],
            },
            "gene_set_ranked_subtype": {
                "binding_sha256": gene_query.binding_sha256,
                "audit_binding_sha256": gene_query.audit_binding_sha256,
                "representative_gene_set_id": gene_set_id,
                "representative_member_rows": gene_detail["member_returned_rows"],
                "cancers_typed": 33,
                "available_cancers": 31,
                "typed_unavailable": observed_unavailable,
                "transitive_metadata_closed": True,
                "scientific_status": EXPECTED_COMPONENTS[
                    "gene_set_ranked_subtype"
                ],
            },
        },
        "manual_assay_exactness_closed": False,
        "family_level_assay_gap_closed": False,
        "web_route_acceptance_claimed": False,
        "main_score_changed": False,
        "production_port_8260_touched": False,
        "production_deployed": False,
        "release_ready": False,
    }


def _write_receipt(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    with path.open("xb") as stream:
        stream.write(encoded)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-bindings", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest_path = Path(args.runtime_bindings).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite runtime smoke: {output}")
    try:
        payload = run_smoke(manifest_path, args.sha256)
    except Exception as exc:
        payload = {
            "format": SMOKE_FORMAT,
            "status": "FAIL",
            "runtime_bindings": {
                "path": str(manifest_path),
                "expected_sha256": str(args.sha256).lower(),
            },
            "failure_type": type(exc).__name__,
            "failure": str(exc),
            "manual_assay_exactness_closed": False,
            "family_level_assay_gap_closed": False,
            "web_route_acceptance_claimed": False,
            "main_score_changed": False,
            "production_port_8260_touched": False,
            "production_deployed": False,
            "release_ready": False,
        }
        _write_receipt(output, payload)
        print("FAIL")
        return 1
    _write_receipt(output, payload)
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
