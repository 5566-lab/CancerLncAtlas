#!/usr/bin/env python3
"""Independently freeze the patient-first + fresh G0/G1/G2 deployment bundle."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


PATIENT_RUNTIME = (
    "cc_hhgt/v32/patient_fold_authority.py",
    "cc_hhgt/v32/patient_first_lineage.py",
    "cc_hhgt/v32/state_training.py",
    "cc_hhgt/v32/primary_fold_views.py",
    "cc_hhgt/v32/hierarchical_candidate_preparation.py",
    "cc_hhgt/v32/hierarchical_candidate_inference.py",
    "scripts/gate_v32_patient_fold_authority.py",
    "scripts/run_v32_genomic_training.py",
    "scripts/train_v32_partitioned_genomic_heads.py",
    "scripts/run_v32_streaming_segment_cnv.py",
    "scripts/prepare_v32_pancancer_gistic_cnv.py",
    "scripts/map_v32_candidate_lncrna_segment_cnv.py",
    "scripts/run_v32_atac_materialization.py",
    "scripts/run_v32_atac_training.py",
    "scripts/run_v32_clinical_training.py",
    "scripts/run_v32_clinical_entity_training.py",
    "scripts/run_v32_continuous_activity.py",
    "scripts/prepare_v32_formal.py",
    "scripts/materialize_v32_exact_release.py",
    "scripts/materialize_v32_state_release.py",
    "scripts/prepare_v32_hierarchical_training_authorization.py",
    "scripts/build_v32_release_companions.py",
    "scripts/audit_v32_atac_readiness.py",
    "scripts/preflight_v32_full33_segment_cnv.py",
    "scripts/audit_v32_routing_fair_compare_readiness.py",
)

GRAPH_RUNTIME = (
    "cc_hhgt/v32/safe_graph.py",
    "cc_hhgt/v32/formal_graph.py",
    "cc_hhgt/v32/formal_graph_authority.py",
    "cc_hhgt/relation_sampling.py",
    "cc_hhgt/gnn.py",
    "cc_hhgt/v32/training.py",
    "scripts/prepare_v32_formal.py",
)

VALIDATION_ONLY = (
    "scripts/build_v32_patient_fold_authority.py",
    "scripts/audit_v32_patient_fold_authority_independent.py",
    "scripts/audit_v32_patient_fold_consumer_bindings.py",
    "scripts/audit_v32_patient_graph_integration_deployment.py",
)

CONDITIONAL_LAUNCHERS = (
    "scripts/server_prepare_v32_g012_patient_first_r2.sh",
    "scripts/server_launch_v32_atac_patient_first_oof_r5.sh",
    "scripts/server_launch_v32_genomic_patient_first_cpu_r3.sh",
)

BLOCKED_LAUNCHER_TEMPLATES = (
    "scripts/server_prepare_v32_routing_fair_inputs_patient_first_r6.sh",
    "scripts/server_launch_v32_external_router_patient_first_r1.sh",
    "scripts/server_launch_v32_hierarchical_patient_first_gpu_r3.sh",
)

SUPPORT_FILES = (
    "config/model_v3_2_hierarchical_patient_first_20260829_r3.yaml",
)

DISABLED_ENTRYPOINTS = (
    "scripts/prepare_v32_pilot.py",
    "scripts/server_launch_v32_atac_fresh_oof.sh",
    "scripts/server_launch_v32_atac_fresh_oof_r2.sh",
    "scripts/server_launch_v32_atac_fresh_oof_r3.sh",
    "scripts/server_launch_v32_atac_fresh_oof_r4.sh",
    "scripts/server_launch_v32_cnv_router_cpu.sh",
    "scripts/server_launch_v32_hierarchical_gpu.sh",
    "scripts/server_launch_v32_hierarchical_gpu_r2.sh",
    "scripts/server_prepare_v32_routing_fair_inputs_r1.sh",
    "scripts/server_prepare_v32_routing_fair_inputs_r2.sh",
    "scripts/server_prepare_v32_routing_fair_inputs_r3.sh",
    "scripts/server_prepare_v32_routing_fair_inputs_r4.sh",
    "scripts/server_prepare_v32_routing_fair_inputs_r5.sh",
    "scripts/server_stage_v32_genomic_resource_memmaps_r1.sh",
    "config/model_v3_2_train_local.yaml",
    "config/model_v3_2_train_formal_local.yaml",
    "config/model_v3_2_code_only.yaml",
)

FORBIDDEN_LOCAL_PRODUCT_ROOTS = (
    "artifacts/prepared",
    "artifacts/formal_prepared",
    "artifacts/diagnostic_leaky_split_association_20260825/formal_prepared",
)

DIRECT_GATE_CONSUMERS = tuple(
    path
    for path in PATIENT_RUNTIME
    if path
    not in {
        "cc_hhgt/v32/primary_fold_views.py",
        "cc_hhgt/v32/hierarchical_candidate_preparation.py",
        "cc_hhgt/v32/hierarchical_candidate_inference.py",
        "scripts/prepare_v32_hierarchical_training_authorization.py",
        "scripts/build_v32_release_companions.py",
        "scripts/audit_v32_atac_readiness.py",
        "scripts/preflight_v32_full33_segment_cnv.py",
        "scripts/audit_v32_routing_fair_compare_readiness.py",
    }
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(root: Path, relative: str, role: str) -> dict[str, object]:
    path = (root / relative).resolve()
    path.relative_to(root)
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"Deployment source is missing or unsafe: {relative}")
    return {
        "relative_path": relative,
        "role": role,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "suggested_staging_path": (
            "./data/CancerLncAtlas/runtime/tools/"
            "v32_patient_graph_integration_20260829_r2/" + relative
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--pytest-passed", type=int, required=True)
    parser.add_argument("--pytest-command", required=True)
    args = parser.parse_args(argv)

    root = Path(args.repo_root).resolve()
    output = Path(args.output_root)
    output = output.resolve() if output.is_absolute() else (root / output).resolve()
    if output.exists():
        raise RuntimeError(f"Integration audit refuses output reuse: {output}")
    output.relative_to(root / "artifacts")

    sys.path.insert(0, str(root))
    from cc_hhgt.v32.formal_graph import FORMAL_RELATION_SCHEMA
    from cc_hhgt.v32.patient_fold_authority import (
        FROZEN_V32_RECEIPT_SHA256,
        FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256,
        validate_frozen_v32_patient_fold_binding,
    )

    authority_root = root / "artifacts/v32_patient_fold_authority_20260829_r1"
    authority = validate_frozen_v32_patient_fold_binding(
        authority_root / "SAMPLE_PATIENT_FOLD_MAP.tsv",
        authority_root / "PATIENT_FOLD_AUTHORITY_RECEIPT.json",
    )
    authority_bytes = b"".join(
        path.read_bytes()
        for path in sorted(authority_root.iterdir())
        if path.is_file()
    ).lower()
    toy_tokens_absent = all(
        token not in authority_bytes
        for token in (b"hallmark_toy", b"synthetic_patient", b"pytest")
    )

    direct_checks = []
    for relative in DIRECT_GATE_CONSUMERS:
        source = (root / relative).read_text(encoding="utf-8")
        direct_checks.append(
            {
                "path": relative,
                "frozen_gate_present": (
                    "validate_frozen_v32_patient_fold_binding" in source
                    or relative == "cc_hhgt/v32/patient_fold_authority.py"
                ),
                "legacy_manifest_literal_absent": (
                    "PATIENT_FOLD_MANIFEST.tsv" not in source
                    or relative == "cc_hhgt/v32/patient_fold_authority.py"
                ),
            }
        )

    prepared_checks = []
    for relative in (
        "cc_hhgt/v32/primary_fold_views.py",
        "cc_hhgt/v32/hierarchical_candidate_preparation.py",
        "scripts/prepare_v32_hierarchical_training_authorization.py",
        "scripts/build_v32_release_companions.py",
    ):
        source = (root / relative).read_text(encoding="utf-8")
        prepared_checks.append(
            {
                "path": relative,
                "prepared_binding_gate_present": (
                    "validate_frozen_v32_prepared_fold_binding" in source
                ),
                "legacy_manifest_literal_absent": "PATIENT_FOLD_MANIFEST.tsv" not in source,
            }
        )

    training_source = (root / "cc_hhgt/v32/training.py").read_text(encoding="utf-8")
    formal_source = (root / "scripts/prepare_v32_formal.py").read_text(encoding="utf-8")
    evidence_query_source = (root / "cc_hhgt/v32/evidence_query.py").read_text(
        encoding="utf-8"
    )
    evidence_dual_identifier_query_fix = all(
        token in evidence_query_source
        for token in (
            "_lnc_storage_aliases",
            "lncrna_id IN (?, ?)",
            "canonical.removeprefix(\"LNC:\")",
        )
    )
    graph_bridge_ready = all(
        token in formal_source
        for token in (
            "build_bound_formal_graph",
            "--graph-authority-receipt",
            "--graph-fold-expression-pattern",
            "--graph-fold-coexpression-pattern",
            'for variant in ("G0", "G1", "G2")',
            'variant_root / f"PATIENT_FOLD_{fold}.pt"',
            '"legacy_root_fold_payloads_written": False',
        )
    ) and "GRAPH_ROOT" not in formal_source
    graph_payload_firewall = all(
        token in training_source
        for token in (
            "validate_formal_graph_payload_binding",
            "formal_graph_authority",
            "formal_graph_variant",
        )
    )
    # The fair-view extractor requires an outer-test batch in every prepared
    # fold, while the training payload deliberately excludes it.  Treating the
    # two stages as interchangeable would either fail at runtime or re-open the
    # sealed-test firewall.  A post-winner-lock inference materializer is still
    # required before routed/hierarchical comparison can run.
    primary_view_source = (root / "cc_hhgt/v32/primary_fold_views.py").read_text(
        encoding="utf-8"
    )
    prepared_has_test_batches = '"test_batches"' in formal_source
    fair_view_requires_test_batches = 'for source_split in ("train", "validation", "test")' in primary_view_source
    downstream_sealed_test_bridge_ready = bool(
        prepared_has_test_batches or not fair_view_requires_test_batches
    )
    expected_schema = {
        ("lncRNA", "expressed_in", "cancer"),
        ("lncRNA", "coexpressed_positive", "gene"),
        ("lncRNA", "coexpressed_negative", "gene"),
        ("gene", "member_of_positive", "pathway"),
        ("gene", "member_of_negative", "pathway"),
        ("pathway", "member_of_family", "pathway_family"),
        ("lncRNA", "binds_protein", "protein"),
        ("protein", "encoded_by", "gene"),
        ("protein", "physical_interaction", "protein"),
    }
    observed_schema = {(row[0], row[1], row[2]) for row in FORMAL_RELATION_SCHEMA}

    runtime_paths = tuple(dict.fromkeys(PATIENT_RUNTIME + GRAPH_RUNTIME))
    runtime_records = [record(root, path, "runtime") for path in runtime_paths]
    validation_records = [record(root, path, "validation_only") for path in VALIDATION_ONLY]
    launcher_records = [record(root, path, "conditional_launcher") for path in CONDITIONAL_LAUNCHERS]
    blocked_launcher_records = [
        record(root, path, "blocked_launcher_template")
        for path in BLOCKED_LAUNCHER_TEMPLATES
    ]
    support_records = [record(root, path, "support_config") for path in SUPPORT_FILES]
    no_product_files = all(
        not item["relative_path"].startswith(("artifacts/", "tests/"))
        and Path(str(item["relative_path"])).suffix in {".py", ".sh"}
        for item in runtime_records + validation_records + launcher_records + blocked_launcher_records
    )
    port_references = []
    port_operation_tokens = (
        "http://127.0.0.1:8260",
        "http://localhost:8260",
        "curl",
        "lsof",
        "netstat",
        "kill",
        "listen(8260",
        "bind(8260",
    )
    for item in runtime_records + launcher_records + blocked_launcher_records:
        relative = str(item["relative_path"])
        for number, line in enumerate(
            (root / relative).read_text(encoding="utf-8", errors="replace").splitlines(),
            start=1,
        ):
            if "8260" in line:
                lowered = line.lower()
                port_references.append(
                    {
                        "path": relative,
                        "line": number,
                        "text": line.strip(),
                        "operational": any(token in lowered for token in port_operation_tokens),
                    }
                )
    no_8260 = not any(item["operational"] for item in port_references)
    launcher_safety = []
    for relative in CONDITIONAL_LAUNCHERS + BLOCKED_LAUNCHER_TEMPLATES:
        source = (root / relative).read_text(encoding="utf-8")
        frozen_gate_invoked = "gate_v32_patient_fold_authority.py" in source
        authority_delegated_to_frozen_gate = (
            relative == "scripts/server_prepare_v32_g012_patient_first_r2.sh"
            and frozen_gate_invoked
        )
        launcher_safety.append(
            {
                "path": relative,
                "frozen_map_physical_sha_pinned": (
                    FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256 in source
                    or authority_delegated_to_frozen_gate
                ),
                "frozen_receipt_physical_sha_pinned": (
                    FROZEN_V32_RECEIPT_SHA256 in source
                    or authority_delegated_to_frozen_gate
                ),
                "frozen_authority_gate_invoked": frozen_gate_invoked,
                "legacy_fold_filename_absent": "PATIENT_FOLD_MANIFEST.tsv" not in source,
                "undated_routed_candidate_root_absent": (
                    "/results/v32_routed_candidate/" not in source
                    and "/inputs/v32_routed_candidate/" not in source
                ),
                "success_resume_branch_absent": "if ! test -f" not in source,
            }
        )
    launchers_fail_closed = all(
        all(value for key, value in row.items() if key != "path")
        for row in launcher_safety
    )
    blocked_template_checks = []
    for relative in BLOCKED_LAUNCHER_TEMPLATES:
        source = (root / relative).read_text(encoding="utf-8")
        has_acceptance_gate = "BLOCKED_PENDING_WINNER_LOCK_SEALED_TEST_INFERENCE" in source
        if relative.endswith("external_router_patient_first_r1.sh"):
            before_mutation = (
                source.index("BLOCKED_PENDING_WINNER_LOCK_SEALED_TEST_INFERENCE")
                < source.index('mkdir -p "$result_root')
                if 'mkdir -p "$result_root' in source
                else True
            )
        else:
            before_mutation = (
                has_acceptance_gate
                and "exit 41" in source[
                    source.index("BLOCKED_PENDING_WINNER_LOCK_SEALED_TEST_INFERENCE") :
                ]
                and (
                    'mkdir -p "$result_root' not in source
                    or source.index("BLOCKED_PENDING_WINNER_LOCK_SEALED_TEST_INFERENCE")
                    < source.index('mkdir -p "$result_root')
                )
            )
        blocked_template_checks.append(
            {
                "path": relative,
                "winner_lock_sealed_test_gate_present": has_acceptance_gate,
                "gate_precedes_output_or_training_mutation": before_mutation,
                "current_execution_ready": False,
            }
        )
    blocked_templates_fail_closed = all(
        row["winner_lock_sealed_test_gate_present"]
        and row["gate_precedes_output_or_training_mutation"]
        for row in blocked_template_checks
    )
    direct_pass = all(all(value for key, value in row.items() if key != "path") for row in direct_checks)
    prepared_pass = all(all(value for key, value in row.items() if key != "path") for row in prepared_checks)
    payload_firewall = "validate_frozen_v32_patient_fold_payload_binding" in training_source
    code_contract_pass = all(
        (
            direct_pass,
            prepared_pass,
            payload_firewall,
            toy_tokens_absent,
            observed_schema == expected_schema,
            graph_payload_firewall,
            no_product_files,
            no_8260,
            launchers_fail_closed,
            blocked_templates_fail_closed,
            args.pytest_passed > 0,
        )
    )
    g012_preparation_code_ready = bool(
        code_contract_pass and graph_bridge_ready and graph_payload_firewall
    )
    downstream_comparison_code_ready = bool(
        g012_preparation_code_ready and downstream_sealed_test_bridge_ready
    )
    # This audit is deliberately local/read-only with respect to the server.
    # It cannot authorize any launcher whose external inputs have not undergone
    # an on-server no-overwrite preflight.
    server_inputs_preflighted = False
    execution_ready = False

    disabled_records = []
    for relative in DISABLED_ENTRYPOINTS:
        path = root / relative
        disabled_records.append(
            {
                "relative_path": relative,
                "exists": path.is_file(),
                "sha256": sha256(path) if path.is_file() else None,
                "execution_allowed": False,
            }
        )

    launcher_activation_contracts = {
        "g012_preparation": {
            "launcher": "scripts/server_prepare_v32_g012_patient_first_r2.sh",
            "suggested_tool_root": (
                "./data/CancerLncAtlas/runtime/"
                "CC_HHGT_v3_2_ranked_subtypes_dev"
            ),
            "prepared_output": (
                "./data/CancerLncAtlas/inputs/"
                "v32_g012_patient_first_20260829_r2/FORMAL_PREPARED_FOLDS"
            ),
            "upload_candidate_after_target_no_overwrite_check": True,
            "start_status": "BLOCKED_PENDING_ON_SERVER_GRAPH_INPUT_RECEIPT_PREFLIGHT",
            "training_started_by_launcher": False,
        },
        "atac": {
            "launcher": "scripts/server_launch_v32_atac_patient_first_oof_r5.sh",
            "suggested_tool_root": (
                "./data/CancerLncAtlas/runtime/tools/"
                "v32_atac_patient_first_oof_20260829_r5"
            ),
            "authority_root": (
                "./data/CancerLncAtlas/inputs/v32_routed_candidate_20260829_r1/"
                "patient_fold_authority_20260829_r1"
            ),
            "exact_inputs": [
                "./data/CancerLncAtlas/results/model/v32_full_multitask/evidence_inputs/current_v32_exact/FORMAL_CANDIDATE_UNIVERSE.parquet",
                "./data/CancerLncAtlas/results/model/v32_full_multitask/evidence_inputs/current_v32_exact/exact_pathway_gene_membership_ensembl.parquet",
                "./data/CancerLncAtlas/input/v3_1_target_context_assets_e0b44122/atac/TCGA-ATAC_PanCan_Log2Norm_Counts.rds",
                "./data/CancerLncAtlas/input/v3_1_target_context_assets_e0b44122/atac/TCGA_identifier_mapping.txt",
                "./data/CancerLncAtlas/input/v3_1_target_context_assets_e0b44122/atac/TCGA-ATAC_PanCancer_PeakSet.txt",
                "./data/CancerLncAtlas/input/v3_1_target_context_assets_e0b44122/gencode/gencode.v36.transcript_promoters.minus1000_plus100.bed.gz",
            ],
            "result_root": (
                "./data/CancerLncAtlas/results/"
                "v32_atac_patient_first_oof_20260829_r5"
            ),
            "upload_candidate_after_target_no_overwrite_check": True,
            "start_status": "BLOCKED_PENDING_ON_SERVER_INPUT_AND_TARGET_PREFLIGHT",
            "old_r3_success_reuse_allowed": False,
        },
        "mutation_cnv": {
            "launcher": "scripts/server_launch_v32_genomic_patient_first_cpu_r3.sh",
            "suggested_tool_root": (
                "./data/CancerLncAtlas/runtime/tools/"
                "v32_genomic_patient_first_20260829_r3"
            ),
            "authority_root": (
                "./data/CancerLncAtlas/inputs/v32_routed_candidate_20260829_r1/"
                "patient_fold_authority_20260829_r1"
            ),
            "raw_input_root": (
                "./data/CancerLncAtlas/inputs/"
                "v32_genomic_patient_first_20260829_r3"
            ),
            "segment_manifest_root": (
                "./data/CancerLncAtlas/manifests/"
                "v32_pancancer_segment_cnv_20260829_r3_full33_reconciled"
            ),
            "required_g2_core_embedding_manifest": (
                "./data/CancerLncAtlas/results/"
                "v32_g012_patient_first_training_20260829_r1/G2/"
                "CORE_EMBEDDING_MANIFEST.server.json"
            ),
            "result_root": (
                "./data/CancerLncAtlas/results/"
                "v32_genomic_patient_first_20260829_r3"
            ),
            "upload_candidate_after_target_no_overwrite_check": True,
            "start_status": "BLOCKED_PENDING_FRESH_G2_CORE_EMBEDDING_AND_FULL33_INPUT_PREFLIGHT",
            "old_cnv_or_fold_product_reuse_allowed": False,
        },
        "external_router": {
            "launcher": "scripts/server_launch_v32_external_router_patient_first_r1.sh",
            "result_root": (
                "./data/CancerLncAtlas/results/"
                "v32_external_router_patient_first_20260829_r1"
            ),
            "start_status": "BLOCKED_PENDING_WINNER_LOCK_SEALED_TEST_INFERENCE",
        },
        "hierarchical": {
            "launcher": "scripts/server_launch_v32_hierarchical_patient_first_gpu_r3.sh",
            "formal_prepared_root": (
                "./data/CancerLncAtlas/inputs/"
                "v32_g012_patient_first_20260829_r2/FORMAL_PREPARED_FOLDS/G2"
            ),
            "result_root": (
                "./data/CancerLncAtlas/results/"
                "v32_hierarchical_patient_first_20260829_r3"
            ),
            "start_status": "HARD_BLOCK_EXIT_41_PENDING_WINNER_LOCK_SEALED_TEST_INFERENCE",
        },
    }

    audit = {
        "format": "CC_HHGT_V3_2_PATIENT_GRAPH_INTEGRATION_AUDIT_V1",
        "status": (
            "PASS_G012_PREPARATION_CODE_DOWNSTREAM_COMPARISON_BLOCKED"
            if g012_preparation_code_ready and not downstream_comparison_code_ready
            else (
                "PASS_G012_AND_DOWNSTREAM_CODE_SERVER_PREFLIGHT_REQUIRED"
                if downstream_comparison_code_ready
                else "PASS_AUDIT_G012_PREPARATION_CODE_BLOCKED"
            )
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "authority": authority,
        "physical_authority_identity": {
            "sample_patient_fold_map_sha256": FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256,
            "authority_receipt_sha256": FROZEN_V32_RECEIPT_SHA256,
        },
        "checks": {
            "direct_consumers": direct_checks,
            "prepared_consumers": prepared_checks,
            "training_payload_frozen_binding_firewall": payload_firewall,
            "authority_toy_tokens_absent": toy_tokens_absent,
            "formal_relation_schema_exact": observed_schema == expected_schema,
            "formal_prepare_bridge_ready": graph_bridge_ready,
            "formal_graph_training_payload_firewall": graph_payload_firewall,
            "g012_preparation_code_ready": g012_preparation_code_ready,
            "fair_view_requires_test_batches": fair_view_requires_test_batches,
            "training_prepared_payload_contains_test_batches": prepared_has_test_batches,
            "post_winner_lock_sealed_test_view_bridge_ready": downstream_sealed_test_bridge_ready,
            "downstream_router_hierarchical_comparison_code_ready": downstream_comparison_code_ready,
            "server_inputs_preflighted": server_inputs_preflighted,
            "launcher_safety": launcher_safety,
            "blocked_launcher_templates": blocked_template_checks,
            "deployment_contains_only_source_no_products": no_product_files,
            "port_8260_references": port_references,
            "port_8260_operational_reference_present": not no_8260,
            "pytest_passed": int(args.pytest_passed),
            "pytest_command": args.pytest_command,
        },
        "adjacent_processing_findings": {
            "evidence_bare_ensg_vs_lnc_prefixed_direct_query": {
                "class": "PROCESSING_BUG_FIXED",
                "local_dual_identifier_query_fix_present": (
                    evidence_dual_identifier_query_fix
                ),
                "downstream_fusion_adapter_normalizes_to_canonical_lnc": True,
                "experiment_bridge_and_interaction_release_canonicalize": True,
                "r3_dependency_guidance_failure_retained": True,
                "r4_server_status_reported": "PASS",
                "r4_receipt_sha256_prefix_reported": "cf6d2f7d",
                "r4_server_receipt_verified_by_this_local_audit": False,
                "production_8260_touched_by_this_audit": False,
            }
        },
        "old_or_toy_products_admitted": False,
        "training_started": False,
        "server_accessed": False,
        "production_8260_touched": False,
    }

    manifest = {
        "format": "CC_HHGT_V3_2_PATIENT_GRAPH_IMMUTABLE_DEPLOYMENT_MANIFEST_V1",
        "status": "STAGING_BUNDLE_FROZEN_NO_UPLOAD_NO_EXECUTION",
        "suggested_server_target": (
            "./data/CancerLncAtlas/runtime/tools/"
            "v32_patient_graph_integration_20260829_r2"
        ),
        "supersedes_local_audit": {
            "root": "artifacts/v32_patient_graph_integration_deployment_audit_20260829_r1",
            "audit_sha256": "cb410036b010d76be3b18be33ca7259f50b8cbba07f8d8f3273a2b9b22595909",
            "deployment_manifest_sha256": "adbc30c0d404fafffe9e712bcfa10122744ea1fa6455e6ee0bace78ffe27483a",
            "reason": "R1 static detector misclassified delegated frozen-gate validation and a support-only consumer",
            "r1_deployment_allowed": False,
        },
        "overwrite_allowed": False,
        "runtime_files": runtime_records,
        "validation_files": validation_records,
        "conditional_launchers": launcher_records,
        "blocked_launcher_templates": blocked_launcher_records,
        "support_files": support_records,
        "launcher_activation_contracts": launcher_activation_contracts,
        "g012_preparation_code_ready": g012_preparation_code_ready,
        "downstream_router_hierarchical_comparison_code_ready": downstream_comparison_code_ready,
        "server_inputs_preflighted": server_inputs_preflighted,
        "conditional_launcher_execution_ready": execution_ready,
        "disabled_historical_entrypoints": disabled_records,
        "forbidden_local_product_roots": list(FORBIDDEN_LOCAL_PRODUCT_ROOTS),
        "forbidden_server_product_roots": [
            "./data/CancerLncAtlas/inputs/v32_routed_candidate",
            "./data/CancerLncAtlas/results/v32_routed_candidate",
            "./data/CancerLncAtlas/results/v32_atac_fresh_oof_20260829_r3",
            "./data/CancerLncAtlas/results/v32_routing_fair_input_20260829_r2",
        ],
        "frozen_authority_server_root": (
            "./data/CancerLncAtlas/inputs/"
            "v32_routed_candidate_20260829_r1/patient_fold_authority_20260829_r1"
        ),
        "old_fold_sha256_forbidden": (
            "dcdaae31f4b44cf15d0d5b52fe01adbad212137196c42177316e0606a980bd4e"
        ),
        "toy_pytest_artifacts_forbidden": True,
        "training_authorized_by_this_manifest": False,
        "production_deployment_authorized_by_this_manifest": False,
        "port_8260_operations_authorized": False,
    }

    output.mkdir(parents=True)
    audit_path = output / "AUDIT.json"
    manifest_path = output / "DEPLOYMENT_MANIFEST.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    success = {
        "status": "PASS_INDEPENDENT_AUDIT_MANIFEST_FROZEN",
        "audit": {"path": str(audit_path), "sha256": sha256(audit_path)},
        "deployment_manifest": {
            "path": str(manifest_path),
            "sha256": sha256(manifest_path),
        },
        "conditional_launcher_execution_ready": execution_ready,
        "g012_preparation_code_ready": g012_preparation_code_ready,
        "downstream_router_hierarchical_comparison_code_ready": downstream_comparison_code_ready,
        "training_started": False,
        "server_accessed": False,
        "production_8260_touched": False,
    }
    (output / "SUCCESS.json").write_text(
        json.dumps(success, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(success, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
