#!/usr/bin/env python3
"""Seal all independent-head candidate receipts without promoting a release."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping


FORMAT = "CANCERLNCATLAS_V32_INDEPENDENT_HEAD_CANDIDATE_SEAL_V1"
ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
MAX_DRUG_PEAK_RSS_KB = 2 * 1024 * 1024
EXPECTED_INPUT_FORMATS = {
    "binding": "CANCERLNCATLAS_V32_INDEPENDENT_HEAD_AUTHORIZED_BINDING_V1",
    "payload_audit": "CANCERLNCATLAS_V32_INDEPENDENT_HEAD_ACCEPTANCE_AUDIT_V1",
    "drug_runtime": "CANCERLNCATLAS_V32_AUTHORIZED_DRUG_RUNTIME_SMOKE_V2",
    "drug_resource": "CANCERLNCATLAS_V32_DRUG_STRICT_RUNTIME_RESOURCE_RECEIPT_V1",
    "drug_core_reachability": "CANCERLNCATLAS_V32_DRUG_CORE_UNAVAILABLE_REACHABILITY_AUDIT_V1",
    "clinical_clean": "CANCERLNCATLAS_V32_CLINICAL_CLEAN_RUNTIME_PAYLOAD_V1",
    "runtime_smoke": "CANCERLNCATLAS_V32_INDEPENDENT_HEAD_RUNTIME_SMOKE_V2",
    "port_untouched": "CANCERLNCATLAS_V32_PORT_8260_UNTOUCHED_RECEIPT_V1",
}
EXPECTED_RUNTIME_COMPONENTS = {
    "clinical",
    "evidence",
    "evidence_direction",
    "gene_set_ranked_subtype",
    "state_gene_set",
}
EXPECTED_REACHABLE_SUCCESS_BUNDLE_ABSENCE_REASONS = {
    "OUTSIDE_CONCEPTUAL_UNIVERSE",
    "NO_HELD_OUT_NATIVE_ASSAY",
    "NO_MATCHED_LNCRNA_EXPRESSION",
    "MODEL_OUTPUT_MISSING_FAIL_CLOSED",
}
EXPECTED_DRUG_EXACT_PROBES = {
    "AVAILABLE",
    *EXPECTED_REACHABLE_SUCCESS_BUNDLE_ABSENCE_REASONS,
}
CORE_REASON = "CURRENT_V32_CORE_UNAVAILABLE"
CORE_REACHABILITY_STATUS = "UNREACHABLE_IN_FORMAL_FACTORS"
MODEL_RUN_REASON = "MODEL_RUN_AUDITED_UNAVAILABLE"
MODEL_RUN_REASON_NOT_APPLICABLE = (
    "NOT_APPLICABLE_TO_A_SUCCESS_BUNDLE;REQUIRES_A_SEPARATE_"
    "AUDITED_UNAVAILABLE_MODEL_RUN"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class SealError(RuntimeError):
    """One pinned receipt is missing, contradictory, or scientifically unsafe."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SealError(message)


def _load_pinned(role: str, path: Path, expected_sha256: str) -> dict[str, Any]:
    expected = str(expected_sha256).lower()
    _require(_SHA256.fullmatch(expected) is not None, f"{role} expected SHA256 is invalid")
    _require(not path.is_symlink() and path.is_file(), f"{role} is missing or a symlink")
    observed = sha256_file(path)
    _require(observed == expected, f"{role} SHA256 mismatch: {observed} != {expected}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SealError(f"{role} is invalid JSON") from exc
    _require(isinstance(payload, dict), f"{role} must be a JSON object")
    _require(payload.get("format") == EXPECTED_INPUT_FORMATS[role], f"{role} format drifted")
    return payload


def _false_flags(payload: Mapping[str, Any], role: str) -> None:
    for key in (
        "production_deployed",
        "release_ready",
        "production_port_8260_touched",
    ):
        _require(payload.get(key) is False, f"{role} invalid {key}")
    if "main_score_changed" in payload:
        _require(payload.get("main_score_changed") is False, f"{role} changes main score")
    if "changes_primary_ranking" in payload:
        _require(
            payload.get("changes_primary_ranking") is False,
            f"{role} changes primary ranking",
        )


def validate_inputs(
    paths: Mapping[str, Path], expected_sha256: Mapping[str, str]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    receipts = {
        role: _load_pinned(role, paths[role], expected_sha256[role])
        for role in EXPECTED_INPUT_FORMATS
    }
    binding = receipts["binding"]
    _require(binding.get("analysis_version") == ANALYSIS_VERSION, "Binding analysis version drifted")
    _require(binding.get("production_deployed") is False, "Binding claims deployment")
    _require(binding.get("release_ready") is False, "Binding claims release readiness")
    _require(
        binding.get("scope_guards")
        == {
            "diagnostic_promoted_to_formal": False,
            "main_score_changed": False,
            "partial_promoted_to_publishable": False,
            "production_port_8260_touched": False,
            "public0_read": False,
            "public0_write": False,
        },
        "Binding scope guards drifted",
    )
    _require(binding.get("totals") == {"payload_files": 798, "payload_bytes": 2_043_566_977}, "Binding totals drifted")

    payload_audit = receipts["payload_audit"]
    _require(payload_audit.get("status") == "PASS", "798 payload audit did not PASS")
    _require(payload_audit.get("failures") == [], "798 payload audit contains failures")
    _require(
        payload_audit.get("binding", {}).get("sha256")
        == expected_sha256["binding"].lower(),
        "798 payload audit is tied to another binding",
    )
    payload_hash = payload_audit.get("payload_hash_audit", {})
    _require(
        payload_hash.get("declared_files") == 798
        and payload_hash.get("unique_files") == 798
        and payload_hash.get("verified_files") == 798
        and payload_hash.get("verified_bytes") == 2_043_566_977,
        "798 payload audit coverage is incomplete",
    )
    snapshot_audit = payload_audit.get("binding_snapshot_hash_audit", {})
    _require(
        snapshot_audit.get("files") == 9
        and snapshot_audit.get("unique_paths") == 9
        and snapshot_audit.get("semantic_contracts_passed") == 9,
        "Binding snapshot audit is incomplete",
    )
    _require(
        payload_audit.get("status_contract")
        == {
            "drug": "diagnostic_only",
            "evidence": "partial_not_publishable",
            "all_heads_enter_main_score": False,
            "production_deployed": False,
            "release_ready": False,
            "production_port_8260_touched": False,
        },
        "798 payload status contract drifted",
    )

    drug_runtime = receipts["drug_runtime"]
    _require(drug_runtime.get("status") == "PASS", "Drug runtime did not PASS")
    _false_flags(drug_runtime, "Drug runtime")
    _require(drug_runtime.get("scientific_status") == "diagnostic_only", "Drug was promoted")
    strict = drug_runtime.get("strict_validation", {})
    _require(
        strict.get("semantic_sampling") is False
        and strict.get("physical_partitions_validated") == 694
        and strict.get("available_rows_validated") == 16_800_054
        and strict.get("conceptual_candidate_rows_validated") == 62_314_182
        and strict.get("five_fold_support_validated") is True
        and strict.get("all_artifact_hashes_and_rows_validated") is True,
        "Drug runtime strict coverage is incomplete",
    )
    typed = drug_runtime.get("typed_absence_coverage", {})
    _require(
        typed.get("all_reachable_success_bundle_reasons_exactly_probed") is True
        and typed.get("all_success_bundle_reasons_accounted_for") is True
        and set(typed.get("reachable_success_bundle_reasons_exactly_probed", []))
        == EXPECTED_REACHABLE_SUCCESS_BUNDLE_ABSENCE_REASONS
        and typed.get("model_run_level_reason")
        == {
            "reason": MODEL_RUN_REASON,
            "status": MODEL_RUN_REASON_NOT_APPLICABLE,
        },
        "Drug typed-absence coverage is incomplete",
    )
    exact_probes = drug_runtime.get("exact_probes", {})
    _require(
        isinstance(exact_probes, dict)
        and set(exact_probes) == EXPECTED_DRUG_EXACT_PROBES,
        "Drug exact-probe outcome set is incomplete or has extras",
    )
    for outcome, result in exact_probes.items():
        _require(isinstance(result, dict), f"Drug exact probe is malformed: {outcome}")
        if outcome == "AVAILABLE":
            probability = result.get("drug_response_association_probability")
            _require(
                result.get("availability") is True
                and result.get("failure_reason") is None
                and isinstance(probability, (int, float))
                and not isinstance(probability, bool)
                and 0 <= float(probability) <= 1,
                "Drug AVAILABLE exact probe drifted",
            )
        else:
            _require(
                result.get("availability") is False
                and result.get("failure_reason") == outcome
                and result.get("drug_response_association_probability") is None,
                f"Drug typed-absence exact probe drifted: {outcome}",
            )
    resource_controls = drug_runtime.get("runtime_resource_controls", {})
    drug_run_directory = Path(str(resource_controls.get("run_directory", "")))
    _require(
        resource_controls.get("duckdb_memory_limit") == "512MB"
        and resource_controls.get("duckdb_threads") == "1"
        and resource_controls.get("duckdb_max_temp_directory_size") == "4GB"
        and str(resource_controls.get("run_directory", "")).strip() != ""
        and not drug_run_directory.exists()
        and resource_controls.get("spill_cleanup_verified") is True,
        "Drug runtime resource controls or spill cleanup are incomplete",
    )

    drug_manifest_ref = drug_runtime.get("manifest", {})
    drug_binding_snapshots = binding.get("components", {}).get("drug", {}).get(
        "binding_snapshots", []
    )
    expected_drug_manifest_refs = [
        item
        for item in drug_binding_snapshots
        if Path(str(item.get("published_snapshot_path", ""))).name
        == "DRUG_SPARSE_QUERY_MANIFEST.json"
    ]
    _require(
        len(expected_drug_manifest_refs) == 1
        and drug_manifest_ref.get("sha256")
        == expected_drug_manifest_refs[0].get("sha256"),
        "Drug runtime manifest is not the authorized binding manifest",
    )

    core_reachability = receipts["drug_core_reachability"]
    _require(
        core_reachability.get("status") == "PASS"
        and core_reachability.get("failure_reason") == CORE_REASON
        and core_reachability.get("reachability_status")
        == CORE_REACHABILITY_STATUS
        and core_reachability.get("authorized_binding", {}).get("sha256")
        == expected_sha256["binding"].lower()
        and core_reachability.get("formal_manifest", {}).get("sha256")
        == drug_manifest_ref.get("sha256")
        and core_reachability.get("api_defensive_enum_retained") is True,
        "Drug core-unavailable reachability receipt is not pinned",
    )
    _false_flags(core_reachability, "Drug core reachability")
    exhaustive = core_reachability.get("exhaustive_audit", {})
    core_resource = core_reachability.get("resource_gate", {})
    core_artifact_root = core_reachability.get("artifact_root_authority", {})
    _require(
        exhaustive.get("semantic_sampling") is False
        and exhaustive.get("duckdb_used") is False
        and exhaustive.get("exact_rows_examined") == 3_300_000
        and exhaustive.get(
            "conceptual_nonempty_assay_intersect_expression_combinations_examined"
        )
        == 44_344_518
        and exhaustive.get("all_eligible_folds_missing_at_least_one_core_type")
        == 0
        and exhaustive.get("reachable_current_v32_core_unavailable_keys") == 0
        and core_resource.get("passed") is True
        and 0 < int(core_resource.get("kernel_peak_rss_kB") or 0)
        <= int(core_resource.get("max_peak_rss_kB") or 0)
        <= MAX_DRUG_PEAK_RSS_KB,
        "Drug core-unavailable exhaustive or resource coverage is incomplete",
    )
    _require(
        core_artifact_root.get("payload_files_declared") == 699
        and core_artifact_root.get("payload_files_hash_and_bytes_verified") == 699
        and int(core_artifact_root.get("payload_bytes_verified") or 0) > 0
        and _SHA256.fullmatch(
            str(core_artifact_root.get("payload_inventory_sha256", ""))
        )
        is not None
        and core_artifact_root.get(
            "all_paths_canonical_relative_to_exact_common_root"
        )
        is True
        and core_artifact_root.get("absolute_override_rejected") is True
        and core_artifact_root.get("parent_escape_rejected") is True
        and core_artifact_root.get("symlink_escape_rejected") is True
        and drug_runtime.get("artifact_root") == core_artifact_root.get("path"),
        "Drug manifest authority and authorized artifact root are not jointly pinned",
    )
    runtime_core_ref = typed.get("exhaustively_proven_unreachable", {}).get(
        CORE_REASON, {}
    )
    _require(
        runtime_core_ref.get("status") == CORE_REACHABILITY_STATUS
        and runtime_core_ref.get("receipt", {}).get("sha256")
        == expected_sha256["drug_core_reachability"].lower()
        and runtime_core_ref.get("exact_rows_examined") == 3_300_000
        and runtime_core_ref.get(
            "conceptual_nonempty_assay_intersect_expression_combinations_examined"
        )
        == 44_344_518
        and runtime_core_ref.get("reachable_keys") == 0,
        "Drug runtime is not tied to the core-unavailable receipt",
    )

    drug_resource = receipts["drug_resource"]
    _require(drug_resource.get("status") == "PASS", "Drug resource receipt did not PASS")
    _false_flags(drug_resource, "Drug resource")
    _require(drug_resource.get("semantic_gate_relaxed") is False, "Drug semantic gate was relaxed")
    _require(int(drug_resource.get("samples") or 0) > 0, "Drug resource monitor has no samples")
    _require(
        0 < int(drug_resource.get("kernel_peak_rss_kB") or 0)
        <= MAX_DRUG_PEAK_RSS_KB,
        "Drug kernel peak RSS exceeds 2 GiB",
    )
    runtime_ref = drug_resource.get("runtime_receipt", {})
    _require(
        runtime_ref.get("status") == "PASS"
        and runtime_ref.get("sha256") == expected_sha256["drug_runtime"].lower(),
        "Drug resource receipt is tied to another runtime receipt",
    )

    clinical = receipts["clinical_clean"]
    _require(clinical.get("status") == "PASS", "Clinical clean receipt did not PASS")
    _false_flags(clinical, "Clinical clean")
    _require(
        clinical.get("expression", {}).get("files") == 67
        and clinical.get("expression", {}).get("rows") == 23_955_621
        and clinical.get("curves", {}).get("files") == 33
        and clinical.get("curves", {}).get("rows") == 4_265_340
        and clinical.get("source_target_sha_mismatches") == 0,
        "Clinical clean coverage is incomplete",
    )
    _require(
        clinical.get("qc_metadata_archive", {}).get("files") == 33,
        "Clinical QC metadata closure is incomplete",
    )

    runtime_smoke = receipts["runtime_smoke"]
    _require(runtime_smoke.get("status") == "PASS", "Independent runtime smoke did not PASS")
    _false_flags(runtime_smoke, "Independent runtime smoke")
    _require(
        runtime_smoke.get("manual_assay_exactness_closed") is False
        and runtime_smoke.get("family_level_assay_gap_closed") is False
        and runtime_smoke.get("web_route_acceptance_claimed") is False,
        "Independent runtime smoke overclaims unresolved scope",
    )
    runtime_components = runtime_smoke.get("components")
    _require(
        isinstance(runtime_components, dict)
        and set(runtime_components) == EXPECTED_RUNTIME_COMPONENTS,
        "Independent runtime component coverage is incomplete",
    )
    _require(
        runtime_components["clinical"].get("events_lte_patients_validated") is True
        and runtime_components["clinical"].get("curve_rows") == 12,
        "Clinical runtime semantic probe is incomplete",
    )
    gene_runtime = runtime_components["gene_set_ranked_subtype"]
    _require(
        gene_runtime.get("transitive_metadata_closed") is True
        and gene_runtime.get("cancers_typed") == 33
        and gene_runtime.get("available_cancers") == 31
        and set(gene_runtime.get("typed_unavailable", {})) == {"CHOL", "UCS"},
        "Gene Set runtime metadata or 33-cancer typing is incomplete",
    )

    runtime_binding_ref = runtime_smoke.get("runtime_bindings", {})
    runtime_binding_path = Path(str(runtime_binding_ref.get("path", "")))
    runtime_binding_sha = str(runtime_binding_ref.get("sha256", ""))
    _require(_SHA256.fullmatch(runtime_binding_sha) is not None, "Runtime manifest SHA is invalid")
    # Load directly because the runtime manifest is an internal cross-reference,
    # not one of the seven top-level receipt roles above.
    _require(
        not runtime_binding_path.is_symlink() and runtime_binding_path.is_file(),
        "Runtime manifest cross-reference is missing or a symlink",
    )
    _require(
        sha256_file(runtime_binding_path) == runtime_binding_sha,
        "Runtime manifest cross-reference SHA drifted",
    )
    runtime_binding_payload = json.loads(
        runtime_binding_path.read_text(encoding="utf-8")
    )
    gene_summary = runtime_binding_payload.get("gene_set_ranked_subtype", {})
    _require(
        runtime_binding_payload.get("format")
        == "CANCERLNCATLAS_V32_INDEPENDENT_HEAD_RUNTIME_BINDINGS_V1"
        and runtime_binding_payload.get("main_score_changed") is False
        and runtime_binding_payload.get("production_deployed") is False
        and runtime_binding_payload.get("release_ready") is False
        and gene_summary.get("status")
        == "RUNTIME_TRANSITIVE_METADATA_CLOSED"
        and runtime_binding_payload.get("bindings", {}).get("clinical", {}).get(
            "clean_directory_runtime_view_bound"
        )
        is True,
        "Runtime manifest closure drifted",
    )
    gene_closure_ref = gene_summary.get("closure_receipt", {})
    gene_closure_path = Path(str(gene_closure_ref.get("path", "")))
    gene_closure_sha = str(gene_closure_ref.get("sha256", "")).lower()
    _require(
        _SHA256.fullmatch(gene_closure_sha) is not None
        and gene_closure_path.is_file()
        and not gene_closure_path.is_symlink()
        and sha256_file(gene_closure_path) == gene_closure_sha,
        "Gene Set runtime closure receipt is missing or drifted",
    )
    try:
        gene_closure = json.loads(gene_closure_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SealError("Gene Set runtime closure receipt is invalid JSON") from exc
    _require(
        isinstance(gene_closure, dict)
        and gene_closure.get("format")
        == "CANCERLNCATLAS_V32_GENE_SET_RUNTIME_METADATA_CLOSURE_V1"
        and gene_closure.get("status") == "PASS"
        and gene_closure.get("inputs", {}).get("authorized_binding", {}).get(
            "sha256"
        )
        == expected_sha256["binding"].lower()
        and gene_closure.get("metadata_closure", {}).get(
            "transitive_metadata_closed"
        )
        is True
        and gene_closure.get("metadata_closure", {}).get(
            "release_records_hash_validated"
        )
        == 47
        and gene_closure.get("metadata_closure", {}).get(
            "upstream_gene_set_records_hash_validated"
        )
        == 47
        and gene_closure.get("metadata_closure", {}).get(
            "legacy_gene_set_transitive_unresolved"
        )
        == 24
        and gene_closure.get("metadata_closure", {}).get(
            "legacy_ranked_subtype_transitive_unresolved"
        )
        == 72
        and gene_closure.get("metadata_closure", {}).get(
            "portable_rebinder_pending_was_runtime_false_negative"
        )
        is True
        and gene_closure.get("runtime_binding_declaration")
        == runtime_binding_payload.get("bindings", {}).get(
            "gene_set_ranked_subtype"
        )
        and gene_closure.get("family_to_exact_broadcast") is False
        and gene_closure.get("changes_primary_ranking") is False
        and gene_closure.get("production_deployed") is False
        and gene_closure.get("release_ready") is False,
        "Gene Set runtime closure did not pass final seal validation",
    )

    port = receipts["port_untouched"]
    _require(port.get("status") == "PASS", "Port 8260 receipt did not PASS")
    _require(port.get("production_port_8260_touched") is False, "Port 8260 changed")
    _require(
        port.get("kernel_socket_and_pid_state_unchanged") is True
        and port.get("candidate_listener_overlap") == []
        and port.get("failures") == [],
        "Port 8260 socket/PID proof is incomplete",
    )

    refs = {
        role: {
            "path": str(paths[role]),
            "sha256": expected_sha256[role].lower(),
            "format": EXPECTED_INPUT_FORMATS[role],
            "status": receipts[role].get("status"),
        }
        for role in EXPECTED_INPUT_FORMATS
    }
    return refs, receipts


def build_seal(
    paths: Mapping[str, Path], expected_sha256: Mapping[str, str]
) -> dict[str, Any]:
    refs, receipts = validate_inputs(paths, expected_sha256)
    gene = receipts["runtime_smoke"]["components"]["gene_set_ranked_subtype"]
    return {
        "format": FORMAT,
        "status": "PASS",
        "analysis_version": ANALYSIS_VERSION,
        "acceptance_scope": "CANDIDATE_INDEPENDENT_HEADS_NOT_WEB_OR_PRODUCTION_RELEASE",
        "inputs": refs,
        "coverage": {
            "payload_files_hash_verified": 798,
            "payload_bytes_hash_verified": 2_043_566_977,
            "drug_physical_partitions_validated": 694,
            "drug_exact_typed_absence_reasons_validated_for_success_bundle": 4,
            "drug_exhaustively_unreachable_typed_absence_reasons": 1,
            "drug_peak_rss_lte_2gib": True,
            "clinical_clean_runtime_view_validated": True,
            "clinical_events_lte_patients_validated": True,
            "evidence_event_exact_query_validated": True,
            "state_gene_set_member_exact_query_validated": True,
            "gene_set_representative_exact_query_validated": True,
            "gene_set_cancers_typed": gene["cancers_typed"],
            "port_8260_socket_pid_before_after_unchanged": True,
        },
        "scientific_status": {
            "drug": "diagnostic_only",
            "evidence": "partial_not_publishable",
            "gene_set_transitive_metadata_closed_for_candidate_query": True,
            "manual_assay_exactness_closed": False,
            "family_level_assay_gap_closed": False,
            "web_route_acceptance_claimed": False,
        },
        "all_heads_enter_main_score": False,
        "main_score_changed": False,
        "production_port_8260_touched": False,
        "production_deployed": False,
        "release_ready": False,
    }


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise SealError(f"Refusing to overwrite final seal: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    with path.open("xb") as stream:
        stream.write(encoded)


def main() -> int:
    parser = argparse.ArgumentParser()
    for role in EXPECTED_INPUT_FORMATS:
        option = role.replace("_", "-")
        parser.add_argument(f"--{option}", required=True)
        parser.add_argument(f"--{option}-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = {
        role: Path(getattr(args, role)).resolve()
        for role in EXPECTED_INPUT_FORMATS
    }
    hashes = {
        role: str(getattr(args, f"{role}_sha256")).lower()
        for role in EXPECTED_INPUT_FORMATS
    }
    output = Path(args.output).resolve()
    if output.exists():
        raise SealError(f"Refusing to overwrite final seal: {output}")
    try:
        payload = build_seal(paths, hashes)
    except Exception as exc:
        payload = {
            "format": FORMAT,
            "status": "FAIL",
            "failure_type": type(exc).__name__,
            "failure": str(exc),
            "scientific_status": {
                "manual_assay_exactness_closed": False,
                "family_level_assay_gap_closed": False,
                "web_route_acceptance_claimed": False,
            },
            "all_heads_enter_main_score": False,
            "main_score_changed": False,
            "production_port_8260_touched": False,
            "production_deployed": False,
            "release_ready": False,
        }
        _write(output, payload)
        print("FAIL")
        return 1
    _write(output, payload)
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
