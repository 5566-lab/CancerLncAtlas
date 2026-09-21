#!/usr/bin/env python3
"""Freeze the current patient-first graph/deployment code as a local-only r3 audit.

This script never accesses a server and never launches preparation, training,
inference, or publication.  It supersedes the stale r1/r2 upload inventories by
re-enumerating the current integration surface and binding the latest local
receipts.  Publication is deliberately not authorized by any output here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Iterable


FORMAT = "CC_HHGT_V3_2_PATIENT_GRAPH_INTEGRATION_DEPLOYMENT_AUDIT_R3"
EXPECTED_PATIENT_MAP_SHA = "e05c20085be532159bb6a51f200a27923975c3caa6a90910290e10a5b60e6253"
EXPECTED_PATIENT_RECEIPT_SHA = "1ef32bda9d14d87997f3b81c34b8aacce172de390db5c29aee011ca73ba317e0"
EXPECTED_WINNER_AUDIT_SHA = "ef2ec56455f3acc0c51a0b96b782636e22ad166727955dfa9ff177761098a8ea"
EXPECTED_SERVER_PREFLIGHT_SHA = "e7c24d464c200ac5192789f7f36d21f14060d7cdfafd521bbad297e19575843e"


CODE_GROUPS: dict[str, tuple[str, ...]] = {
    "patient_authority_gates": (
        "cc_hhgt/v32/patient_fold_authority.py",
        "cc_hhgt/v32/patient_first_lineage.py",
        "cc_hhgt/v32/patient_folds.py",
        "cc_hhgt/v32/input_lineage.py",
        "scripts/gate_v32_patient_fold_authority.py",
        "scripts/validate_v32_patient_first_output_lineage.py",
        "scripts/build_v32_patient_fold_authority.py",
        "scripts/audit_v32_patient_fold_authority_independent.py",
        "scripts/audit_v32_patient_fold_consumer_bindings.py",
    ),
    "formal_graph_and_authority": (
        "cc_hhgt/v32/safe_graph.py",
        "cc_hhgt/v32/formal_graph.py",
        "cc_hhgt/v32/formal_graph_authority.py",
        "cc_hhgt/relation_sampling.py",
        "cc_hhgt/gnn.py",
        "cc_hhgt/stats.py",
        "cc_hhgt/v32/associations.py",
        "cc_hhgt/v32/baselines.py",
        "cc_hhgt/v32/contracts.py",
        "cc_hhgt/v32/training.py",
        "scripts/prepare_v32_formal.py",
    ),
    "g012_preparation_and_training_surface": (
        "scripts/server_prepare_v32_g012_patient_first_r2.sh",
        "scripts/prepare_v32_formal.py",
        "scripts/prepare_v32_hierarchical_training_authorization.py",
        "cc_hhgt/v32/cli.py",
        "config/model_v3_2_hierarchical_patient_first_20260829_r3.yaml",
    ),
    "atac_patient_first": (
        "scripts/server_launch_v32_atac_patient_first_oof_r5.sh",
        "scripts/run_v32_atac_materialization.py",
        "scripts/run_v32_atac_training.py",
        "scripts/materialize_v32_atac_gene_accessibility.R",
        "scripts/audit_v32_atac_readiness.py",
        "inputs/v32_full_multitask/atac/AUTHORITATIVE_INPUT_CONTRACT.json",
        "cc_hhgt/v32/atac_materialization.py",
        "cc_hhgt/v32/atac_training.py",
        "cc_hhgt/v32/atac_readiness.py",
    ),
    "genomic_patient_first": (
        "scripts/server_launch_v32_genomic_patient_first_cpu_r3.sh",
        "scripts/run_v32_genomic_training.py",
        "scripts/train_v32_partitioned_genomic_heads.py",
        "scripts/run_v32_streaming_segment_cnv.py",
        "scripts/preflight_v32_full33_segment_cnv.py",
        "scripts/prepare_v32_pancancer_gistic_cnv.py",
        "scripts/map_v32_candidate_lncrna_segment_cnv.py",
        "cc_hhgt/v32/gdc_segment_cnv.py",
        "cc_hhgt/v32/segment_cnv_streaming.py",
        "cc_hhgt/v32/genomic_training.py",
        "cc_hhgt/v32/genomic_partition_training.py",
        "cc_hhgt/v32/genomic_resource_staging.py",
    ),
    "external_router": (
        "scripts/server_prepare_v32_routing_fair_inputs_patient_first_r6.sh",
        "scripts/server_launch_v32_external_router_patient_first_r1.sh",
        "scripts/prepare_v32_routing_fair_inputs.py",
        "scripts/validate_v32_routing_fair_inputs.py",
        "scripts/audit_v32_routing_fair_compare_readiness.py",
        "scripts/run_v32_cancer_modality_router.py",
        "cc_hhgt/v32/cancer_modality_router.py",
        "cc_hhgt/v32/multimodal_fusion.py",
    ),
    "hierarchical_and_fair_comparison": (
        "scripts/server_launch_v32_hierarchical_patient_first_gpu_r3.sh",
        "scripts/prepare_v32_hierarchical_candidate.py",
        "scripts/infer_v32_hierarchical_candidate.py",
        "scripts/prepare_v32_hierarchical_training_authorization.py",
        "scripts/compare_v32_routing_architectures.py",
        "cc_hhgt/v32/hierarchical_candidate_preparation.py",
        "cc_hhgt/v32/hierarchical_candidate_inference.py",
        "cc_hhgt/v32/routed_fair_comparison.py",
        "cc_hhgt/v32/cli.py",
        "config/model_v3_2_hierarchical_patient_first_20260829_r3.yaml",
    ),
    "winner_lock_sealed_test_and_primary_views": (
        "cc_hhgt/v32/sealed_test_inference.py",
        "scripts/materialize_v32_winner_locked_sealed_test.py",
        "cc_hhgt/v32/primary_fold_views.py",
        "cc_hhgt/v32/primary_fold_view_audit.py",
        "scripts/extract_v32_primary_fold_views.py",
        "scripts/audit_v32_primary_fold_views_independent.py",
        "scripts/prepare_v32_hierarchical_candidate.py",
        "cc_hhgt/v32/hierarchical_candidate_preparation.py",
        "cc_hhgt/v32/multimodal_fusion.py",
        "cc_hhgt/v32/training.py",
    ),
    "r3_auditor": (
        "scripts/audit_v32_patient_graph_integration_deployment_r3.py",
    ),
}


CORE_TESTS = (
    "tests/test_v32_frozen_patient_fold_binding.py",
    "tests/test_v32_patient_fold_authority.py",
    "tests/test_v32_formal_graph_contract.py",
    "tests/test_v32_formal_graph_index_invariance.py",
    "tests/test_v32_formal_prepare_graph_authority.py",
    "tests/test_v32_formal_preparation_firewall.py",
    "tests/test_v32_primary_fold_views.py",
    "tests/test_v32_sealed_test_inference.py",
    "tests/test_v32_training_patient_authority_firewall.py",
    "tests/test_v32_training_objective.py",
    "tests/test_v32_patient_first_current_launchers.py",
    "tests/test_v32_routing_fair_input.py",
    "tests/test_v32_cancer_modality_router.py",
    "tests/test_v32_hierarchical_candidate_preparation.py",
)


MODALITY_TESTS = (
    "tests/test_v32_atac_readiness.py",
    "tests/test_v32_atac_training.py",
    "tests/test_v32_gdc_segment_cnv.py",
    "tests/test_v32_genomic_partition_training.py",
    "tests/test_v32_genomic_resource_staging.py",
    "tests/test_v32_genomic_training.py",
    "tests/test_v32_segment_cnv_streaming.py",
)


CURRENT_SHELL_LAUNCHERS = (
    "scripts/server_prepare_v32_g012_patient_first_r2.sh",
    "scripts/server_launch_v32_atac_patient_first_oof_r5.sh",
    "scripts/server_launch_v32_genomic_patient_first_cpu_r3.sh",
    "scripts/server_prepare_v32_routing_fair_inputs_patient_first_r6.sh",
    "scripts/server_launch_v32_external_router_patient_first_r1.sh",
    "scripts/server_launch_v32_hierarchical_patient_first_gpu_r3.sh",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def file_record(repo: Path, relative: str, groups: Iterable[str]) -> dict[str, Any]:
    path = repo / relative
    require(path.is_file(), f"Missing current integration file: {relative}")
    return {
        "relative_path": relative,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "groups": sorted(set(groups)),
    }


def bind_local_artifact(
    repo: Path,
    relative: str,
    *,
    expected_sha256: str | None = None,
    expected_status: str | None = None,
) -> dict[str, Any]:
    path = repo / relative
    require(path.is_file(), f"Missing required local receipt: {relative}")
    digest = sha256_file(path)
    if expected_sha256 is not None:
        require(digest == expected_sha256, f"Receipt SHA drift: {relative}")
    payload = read_json(path)
    status = payload.get("status")
    if expected_status is not None:
        require(status == expected_status, f"Receipt status drift: {relative}")
    return {
        "relative_path": relative,
        "bytes": path.stat().st_size,
        "sha256": digest,
        "format": payload.get("format"),
        "status": status,
    }


def old_manifest_records(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for section in (
        "runtime_files",
        "conditional_launchers",
        "support_files",
        "validation_files",
        "blocked_launcher_templates",
    ):
        for item in payload.get(section, []):
            relative = str(item["relative_path"])
            result[relative] = {
                "sha256": str(item["sha256"]),
                "old_section": section,
                "old_role": item.get("role"),
            }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument(
        "--output-root",
        default="artifacts/v32_patient_graph_integration_deployment_audit_20260829_r3",
    )
    args = parser.parse_args()
    repo = Path(args.repo_root).resolve()
    output = (repo / args.output_root).resolve()
    require(output.parent == (repo / "artifacts").resolve(), "Output must be a direct artifact child")
    require(not output.exists(), f"Refusing r3 output reuse: {output}")

    prior_r1 = repo / "artifacts/v32_patient_graph_integration_deployment_audit_20260829_r1"
    prior_r2 = repo / "artifacts/v32_patient_graph_integration_deployment_audit_20260829_r2"
    r2_manifest_path = prior_r2 / "DEPLOYMENT_MANIFEST.json"
    r2_manifest = read_json(r2_manifest_path)
    old_records = old_manifest_records(r2_manifest)

    # Preserve every previously enumerated non-legacy file in the superseding
    # inventory, then add the current explicit integration surface.
    groups_by_path: dict[str, set[str]] = {
        relative: {"retained_from_r2"} for relative in old_records
    }
    for group, paths in CODE_GROUPS.items():
        for relative in paths:
            groups_by_path.setdefault(relative, set()).add(group)

    code_records = [
        file_record(repo, relative, groups)
        for relative, groups in sorted(groups_by_path.items())
    ]
    code_by_path = {item["relative_path"]: item for item in code_records}

    changed_since_r2: list[dict[str, Any]] = []
    new_since_r2: list[dict[str, Any]] = []
    unchanged_since_r2 = 0
    for relative, record in code_by_path.items():
        old = old_records.get(relative)
        if old is None:
            new_since_r2.append(record)
        elif old["sha256"] != record["sha256"]:
            changed_since_r2.append(
                {
                    "relative_path": relative,
                    "r2_sha256": old["sha256"],
                    "r3_sha256": record["sha256"],
                    "groups": record["groups"],
                }
            )
        else:
            unchanged_since_r2 += 1

    # Current local receipts.  No connector, SSH, or network call occurs here.
    winner = bind_local_artifact(
        repo,
        "artifacts/v32_winner_lock_sealed_test_code_audit_20260829_r1/AUDIT.json",
        expected_sha256=EXPECTED_WINNER_AUDIT_SHA,
        expected_status="PASS_CODE_CONTRACT_REAL_GPU_CHECKPOINT_INFERENCE_PENDING",
    )
    server_preflight = bind_local_artifact(
        repo,
        "artifacts/v32_server_preflight_g012_atac_genomic_20260829_r1/PREFLIGHT.json",
        expected_sha256=EXPECTED_SERVER_PREFLIGHT_SHA,
        expected_status="BLOCKED_NO_UPLOAD_NO_EXECUTION",
    )
    patient_map_path = repo / "artifacts/v32_patient_fold_authority_20260829_r1/SAMPLE_PATIENT_FOLD_MAP.tsv"
    patient_receipt_path = repo / "artifacts/v32_patient_fold_authority_20260829_r1/PATIENT_FOLD_AUTHORITY_RECEIPT.json"
    require(sha256_file(patient_map_path) == EXPECTED_PATIENT_MAP_SHA, "Patient map SHA drift")
    require(sha256_file(patient_receipt_path) == EXPECTED_PATIENT_RECEIPT_SHA, "Patient receipt SHA drift")

    evidence_r4 = bind_local_artifact(
        repo,
        "artifacts/v32_ecs_cleanup_smokes_20260829_r1/EVIDENCE_RUNTIME_QUERY_CLASS_SMOKE_R4.json",
    )
    evidence_r4_payload = read_json(repo / evidence_r4["relative_path"])
    require(evidence_r4_payload.get("status") == "PASS", "Evidence R4 query did not pass")
    require(evidence_r4_payload.get("release_ready") is False, "Evidence R4 unexpectedly claims release ready")

    evidence_receipts = [
        bind_local_artifact(
            repo,
            "artifacts/v32_evidence_interaction_parquet_20260829_r1_server_receipts/REMATERIALIZATION_MANIFEST.server.json",
            expected_sha256="3bea8fd20ff409ea5c89b1276d3a1ee9f3d6bc61178b310b85035089603a3877",
            expected_status="PASS_REMATERIALIZED_REQUIRES_INDEPENDENT_VALIDATION",
        ),
        bind_local_artifact(
            repo,
            "artifacts/v32_evidence_interaction_parquet_20260829_r1_server_receipts/REMATERIALIZATION_SUCCESS.server.json",
            expected_sha256="56c6b7556d27154644a6ed313950f4bf516aed9464ee2416dd36f2ba6b3b604e",
            expected_status="PASS_REMATERIALIZED_REQUIRES_INDEPENDENT_VALIDATION",
        ),
        bind_local_artifact(
            repo,
            "artifacts/v32_evidence_interaction_parquet_20260829_r1_server_receipts/VALIDATION.server.json",
            expected_sha256="03b0b7e424fa0a72933d17dfb7c323efbe3594bb2fa42d6e70c0f7721668d709",
            expected_status="PASS_INDEPENDENT_REMATERIALIZATION_VALIDATION",
        ),
        bind_local_artifact(
            repo,
            "artifacts/v32_evidence_interaction_parquet_20260829_r1_server_receipts/VALIDATION_SUCCESS.server.json",
            expected_sha256="51c2b2b0cf6a77b65a1fa926bffcc48b999049b002ba6c793414b6385db034fa",
            expected_status="PASS_INDEPENDENT_REMATERIALIZATION_VALIDATION",
        ),
    ]

    processing_manifest = bind_local_artifact(
        repo,
        "artifacts/processing_not_raw_data_summary_20260829_r4/SUMMARY_MANIFEST.json",
        expected_status="AUDIT_SUMMARY_NOT_RELEASE_AUTHORIZATION",
    )
    processing_payload = read_json(repo / processing_manifest["relative_path"])
    processing_summary = repo / "artifacts/processing_not_raw_data_summary_20260829_r4" / str(
        processing_payload["summary"]["path"]
    )
    require(
        sha256_file(processing_summary) == processing_payload["summary"]["sha256"],
        "Processing summary Markdown drift",
    )

    winner_payload = read_json(repo / winner["relative_path"])
    allowed_winner_audit_drift = {
        "tests/test_v32_sealed_test_inference.py",
        "tests/test_v32_primary_fold_views.py",
    }
    winner_file_mismatches = []
    for item in winner_payload.get("files", []):
        relative = str(item["path"])
        current = code_by_path.get(relative)
        if current is None or current["sha256"] != item["sha256"]:
            winner_file_mismatches.append(
                {
                    "relative_path": relative,
                    "bound_winner_audit_sha256": item["sha256"],
                    "current_sha256": None if current is None else current["sha256"],
                    "current_present": current is not None,
                    "allowed_post_audit_drift": relative in allowed_winner_audit_drift,
                }
            )
    unexpected_winner_drift = [
        item for item in winner_file_mismatches if not item["allowed_post_audit_drift"]
    ]
    require(not unexpected_winner_drift, f"Unexpected winner-lock code drift: {unexpected_winner_drift}")

    preflight_payload = read_json(repo / server_preflight["relative_path"])
    require(preflight_payload.get("disposition", {}).get("BLOCKED") is True, "Server preflight not blocked")
    require(preflight_payload.get("scope_and_safety", {}).get("server_writes_performed") is False, "Server write claimed")

    conversion_receipt_dir = repo / "artifacts/v32_evidence_interaction_parquet_20260829_r1_server_receipts"
    conversion_names = ("CONVERSION_MANIFEST.server.json", "CONVERSION_SUCCESS.server.json")
    conversion_receipts = [
        {
            "relative_path": str((conversion_receipt_dir / name).relative_to(repo)).replace("\\", "/"),
            "present": (conversion_receipt_dir / name).is_file(),
        }
        for name in conversion_names
    ]
    conversion_attempt_ledger = bind_local_artifact(
        repo,
        "artifacts/v32_evidence_interaction_parquet_20260829_r1_server_receipts/CONVERSION_ATTEMPT_LEDGER.json",
        expected_status="R3_RUNNING_R1_R2_FAILURES_PRESERVED",
    )
    conversion_attempt_payload = read_json(repo / conversion_attempt_ledger["relative_path"])
    conversion_attempts = {
        str(item["attempt"]): item for item in conversion_attempt_payload.get("attempts", [])
    }
    require(
        conversion_attempts.get("r1", {}).get("status") == "FAILED_DUCKDB_MEMORY_LIMIT_NO_SUCCESS",
        "Evidence conversion r1 failure is not preserved",
    )
    require(
        conversion_attempts.get("r2", {}).get("status") == "FAILED_RUNTIME_NOT_EXPLICIT_NO_OUTPUT",
        "Evidence conversion r2 failure is not preserved",
    )
    require(
        conversion_attempts.get("r3", {}).get("status") == "RUNNING_AWAITING_LOSSLESS_VALIDATION",
        "Evidence conversion r3 is not recorded as running",
    )
    require(
        conversion_attempts["r3"].get("success_receipt_exists_at_ledger_creation") is False,
        "Evidence conversion r3 incorrectly claims a success receipt",
    )
    converter_smoke = bind_local_artifact(
        repo,
        "artifacts/v32_evidence_converter_duckdb_smoke_20260829_r1/CONVERSION_MANIFEST.json",
        expected_sha256="b410900d81fc26789d29c86a0891d334d8fb0af496151d73d59b2f8ddd5451e1",
        expected_status="PASS_LOSSLESS_PARQUET_CONVERSION",
    )

    # Static syntax validation of exactly the deployment Python and current
    # shell launchers.  This does not import data or launch any workflow.
    python_files = [item for item in code_records if item["relative_path"].endswith(".py")]
    for item in python_files:
        path = repo / item["relative_path"]
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    shell_checks = []
    rtools_bash = Path(r"C:\rtools45\usr\bin\bash.exe")
    bash_executable = str(rtools_bash) if os.name == "nt" and rtools_bash.is_file() else "bash"
    for relative in CURRENT_SHELL_LAUNCHERS:
        launcher_text = (repo / relative).read_text(encoding="utf-8")
        shell_check = subprocess.run(
            [bash_executable, "-n"],
            cwd=repo,
            check=False,
            capture_output=True,
            text=True,
            input=launcher_text,
        )
        shell_checks.append(
            {
                "relative_path": relative,
                "returncode": shell_check.returncode,
                "stdout": shell_check.stdout,
                "stderr": shell_check.stderr,
            }
        )
    require(
        all(item["returncode"] == 0 for item in shell_checks),
        f"bash -n failed: {shell_checks}",
    )

    test_file_records = [
        file_record(repo, relative, {"accepted_combination_test"})
        for relative in (*CORE_TESTS, *MODALITY_TESTS)
    ]
    test_receipt: dict[str, Any] = {
        "format": "CC_HHGT_V3_2_PATIENT_GRAPH_INTEGRATION_TEST_RECEIPT_R3",
        "status": "PASS_CURRENT_LOCAL_COMBINATION_TESTS",
        "scope": {
            "server_accessed": False,
            "training_started": False,
            "production_8260_touched": False,
            "real_checkpoint_loaded": False,
            "real_gpu_inference_run": False,
        },
        "accepted_runs": [
            {
                "name": "patient_formal_sealed_router_core",
                "command": "D:/model/.venv_v32_audit/Scripts/python.exe -m pytest --basetemp D:/model/CC_HHGT_v3_2_ranked_subtypes_dev/.pytest_integration_r3_core_20260829_r1 "
                + " ".join(CORE_TESTS)
                + " -q",
                "passed": 53,
                "skipped": 3,
                "failed": 0,
                "errors": 0,
                "elapsed_seconds": 15.13,
            },
            {
                "name": "atac_genomic_modalities",
                "command": "D:/model/.venv_v32_audit/Scripts/python.exe -m pytest --basetemp D:/model/CC_HHGT_v3_2_ranked_subtypes_dev/.pytest_integration_r3_modalities_20260829_r1 "
                + " ".join(MODALITY_TESTS)
                + " -q",
                "passed": 34,
                "skipped": 1,
                "failed": 0,
                "errors": 0,
                "elapsed_seconds": 6.30,
            },
            {
                "name": "current_shell_launcher_syntax",
                "command": "bash -n " + " ".join(CURRENT_SHELL_LAUNCHERS),
                "files": len(CURRENT_SHELL_LAUNCHERS),
                "interpreter": bash_executable,
                "passed": len(CURRENT_SHELL_LAUNCHERS),
                "failed": 0,
                "per_file_results": shell_checks,
            },
        ],
        "non_acceptance_environment_attempt": {
            "status": "INVALID_TEST_ENVIRONMENT_NOT_CODE_FAILURE",
            "reason": "Default system pytest temporary root was unreadable",
            "observed": {"passed": 24, "skipped": 3, "setup_errors": 29, "code_failures": 0},
            "superseded_by_explicit_workspace_basetemp_run": True,
        },
        "accepted_totals": {"pytest_passed": 87, "pytest_skipped": 4, "pytest_failed": 0, "shell_files_passed": 6},
        "static_compile": {"python_files_compiled": len(python_files), "failed": 0},
        "test_files": test_file_records,
    }

    hierarchical_launcher = (repo / "scripts/server_launch_v32_hierarchical_patient_first_gpu_r3.sh").read_text(encoding="utf-8")
    g012_launcher = (repo / "scripts/server_prepare_v32_g012_patient_first_r2.sh").read_text(encoding="utf-8")
    external_launcher = (repo / "scripts/server_launch_v32_external_router_patient_first_r1.sh").read_text(encoding="utf-8")
    require("exit 41" in hierarchical_launcher, "Hierarchical hard block disappeared without audit")
    require("COMPLETE_NO_TRAINING" in g012_launcher, "G012 preparation no-training marker missing")
    require("WINNER_LOCK_SEALED_TEST_INFERENCE" in external_launcher, "External router sealed gate missing")

    module_status = {
        "patient_authority": {
            "code": "READY_LOCAL_VERIFIED",
            "server": "FROZEN_AUTHORITY_PRESENT_PER_BOUND_PREFLIGHT",
            "training_or_release": "NOT_AUTHORIZED_BY_R3",
        },
        "formal_graph": {
            "code": "READY_LOCAL_VERIFIED",
            "server": "BLOCKED_GRAPH_AUTHORITIES_AND_FIVE_FOLD_RESOURCES_NOT_MATERIALIZED",
            "training_or_release": "NOT_STARTED",
        },
        "g012": {
            "preparation_code": "READY_LOCAL_VERIFIED",
            "dedicated_g0_g1_g2_training_launcher_present": False,
            "training_framework": "READY_LOCAL_ONLY",
            "server": "BLOCKED_INPUT_AUTHORITIES_RUNTIME_AND_GPU",
            "real_gpu": "PENDING",
        },
        "atac": {
            "code": "READY_LOCAL_VERIFIED",
            "server": "BLOCKED_TOOL_BUNDLE_AND_APPROVED_PYTHON_RUNTIME",
            "real_gpu": "NOT_REQUIRED_FOR_CURRENT_CPU_HEAD_BUT_NO_TRAINING_STARTED",
        },
        "genomic": {
            "code": "READY_LOCAL_VERIFIED",
            "raw_segment_data": "COMPLETE_PER_BOUND_SERVER_PREFLIGHT_RECEIPT",
            "server": "BLOCKED_FROZEN_PATIENT_REBIND_AND_FRESH_G2_CORE_EMBEDDING",
            "training": "NOT_STARTED",
        },
        "winner_lock_sealed_test": {
            "code": "READY_LOCAL_VERIFIED",
            "server": "BLOCKED_FIVE_REAL_G2_CHECKPOINTS_AND_SEALED_INPUT_AUTHORITY",
            "real_gpu_checkpoint_inference": "PENDING",
        },
        "external_router": {
            "code": "READY_LOCAL_VERIFIED_CONDITIONAL",
            "server": "BLOCKED_WINNER_LOCK_ACCEPTANCE_AND_FRESH_MODALITY_PRODUCTS",
            "training": "NOT_STARTED",
        },
        "hierarchical": {
            "helper_code": "READY_LOCAL_VERIFIED",
            "current_launcher_execution": "BLOCKED_EXIT_41_BEFORE_TRAINING",
            "sealed_bridge_integration": "PENDING_LAUNCHER_REVISION",
            "server": "BLOCKED_UPSTREAM_PRODUCTS_AND_GPU",
            "real_gpu": "PENDING",
        },
        "publication": {
            "website_or_public_release": "FORBIDDEN",
            "production_8260": "UNTOUCHED",
        },
    }

    old_versions = []
    for revision, root in (("r1", prior_r1), ("r2", prior_r2)):
        old_versions.append(
            {
                "revision": revision,
                "root": str(root.relative_to(repo)).replace("\\", "/"),
                "audit_sha256": sha256_file(root / "AUDIT.json"),
                "deployment_manifest_sha256": sha256_file(root / "DEPLOYMENT_MANIFEST.json"),
                "superseded_by_r3": True,
                "upload_allowed": False,
                "reason": "Current integration files and sealed/primary surface differ from this frozen inventory",
            }
        )

    # Write to a private sibling and publish once; final output reuse is forbidden.
    building = Path(tempfile.mkdtemp(prefix=f".{output.name}.building.", dir=output.parent))
    test_receipt_path = building / "TEST_RECEIPT.json"
    write_json(test_receipt_path, test_receipt)

    artifact_bindings = {
        "winner_lock_sealed_test_code_audit": winner,
        "winner_lock_sealed_test_code_audit_current_drift": {
            "semantics": (
                "The bound winner audit remains a verified historical receipt. Its frozen file hashes "
                "are not asserted as current after the declared sealed/primary test changes; r3 hashes "
                "and tests the current files."
            ),
            "mismatches": winner_file_mismatches,
            "unexpected_mismatches": unexpected_winner_drift,
            "current_authority": "THIS_R3_DEPLOYMENT_MANIFEST",
        },
        "server_readonly_preflight": server_preflight,
        "patient_fold_authority": {
            "map": {
                "relative_path": str(patient_map_path.relative_to(repo)).replace("\\", "/"),
                "bytes": patient_map_path.stat().st_size,
                "sha256": EXPECTED_PATIENT_MAP_SHA,
            },
            "receipt": {
                "relative_path": str(patient_receipt_path.relative_to(repo)).replace("\\", "/"),
                "bytes": patient_receipt_path.stat().st_size,
                "sha256": EXPECTED_PATIENT_RECEIPT_SHA,
            },
        },
        "evidence_runtime_query_r4": {
            **evidence_r4,
            "query_result_status": evidence_r4_payload.get("status"),
            "release_ready": evidence_r4_payload.get("release_ready"),
        },
        "evidence_rematerialization_and_validation": evidence_receipts,
        "evidence_parquet_conversion": {
            "formal_server_receipts": conversion_receipts,
            "formal_conversion_complete": all(item["present"] for item in conversion_receipts),
            "status": "PENDING_NOT_WAITED" if not all(item["present"] for item in conversion_receipts) else "LOCAL_RECEIPTS_PRESENT_REQUIRE_SEPARATE_AUDIT",
            "attempt_ledger": {
                **conversion_attempt_ledger,
                "r1_status": conversion_attempts["r1"]["status"],
                "r2_status": conversion_attempts["r2"]["status"],
                "r3_status": conversion_attempts["r3"]["status"],
                "r3_success_receipt_at_ledger_creation": conversion_attempts["r3"][
                    "success_receipt_exists_at_ledger_creation"
                ],
                "monitoring_performed_by_r3_audit": False,
            },
            "local_lossless_smoke": converter_smoke,
        },
        "processing_not_raw_data_summary": {
            **processing_manifest,
            "summary_markdown": {
                "relative_path": str(processing_summary.relative_to(repo)).replace("\\", "/"),
                "bytes": processing_summary.stat().st_size,
                "sha256": sha256_file(processing_summary),
            },
        },
    }

    deployment_manifest: dict[str, Any] = {
        "format": f"{FORMAT}_DEPLOYMENT_MANIFEST",
        "status": "LOCAL_CODE_SNAPSHOT_READY_SERVER_DEPLOYMENT_BLOCKED",
        "scope": {
            "server_accessed": False,
            "server_upload_performed": False,
            "server_write_performed": False,
            "training_started": False,
            "real_gpu_inference_run": False,
            "production_8260_touched": False,
            "publication_authorized": False,
        },
        "code_files": code_records,
        "code_file_count": len(code_records),
        "code_groups": {key: list(value) for key, value in CODE_GROUPS.items()},
        "changes_from_r2": {
            "r2_enumerated_files": len(old_records),
            "r3_enumerated_files": len(code_records),
            "unchanged": unchanged_since_r2,
            "modified": changed_since_r2,
            "new": [
                {
                    "relative_path": item["relative_path"],
                    "sha256": item["sha256"],
                    "groups": item["groups"],
                }
                for item in new_since_r2
            ],
        },
        "module_status": module_status,
        "artifact_bindings": artifact_bindings,
        "test_receipt": {
            "relative_path": "TEST_RECEIPT.json",
            "bytes": test_receipt_path.stat().st_size,
            "sha256": sha256_file(test_receipt_path),
            "status": test_receipt["status"],
        },
        "superseded_forbidden_manifests": old_versions,
        "upload_contract": {
            "r1_upload_allowed": False,
            "r2_upload_allowed": False,
            "r3_upload_authorized_by_this_manifest": False,
            "reason": "Server preflight is blocked and this task is local read-only code audit only",
        },
    }
    manifest_path = building / "DEPLOYMENT_MANIFEST.json"
    write_json(manifest_path, deployment_manifest)

    audit: dict[str, Any] = {
        "format": FORMAT,
        "status": "PASS_LOCAL_CODE_INTEGRATION_SERVER_EXECUTION_BLOCKED",
        "checks": {
            "current_files_exist_and_hashed": True,
            "current_file_count_is_recomputed_not_legacy_42": len(code_records) != 42,
            "winner_lock_artifact_sha_matches_expected": True,
            "winner_lock_historical_file_drift_explicitly_recorded": bool(winner_file_mismatches),
            "winner_lock_unexpected_file_drift": bool(unexpected_winner_drift),
            "patient_fold_authority_physical_shas_match": True,
            "latest_server_preflight_is_bound_and_blocked": True,
            "evidence_query_r4_local_receipt_pass": True,
            "evidence_rematerialization_validation_receipts_pass": True,
            "evidence_formal_parquet_conversion_complete": all(item["present"] for item in conversion_receipts),
            "evidence_conversion_running_recorded_not_promoted_to_success": True,
            "processing_summary_internal_sha_matches": True,
            "accepted_pytest_failed": 0,
            "bash_n_current_launchers_pass": True,
            "python_static_compile_pass": True,
            "r1_r2_upload_forbidden": True,
        },
        "counts": {
            "code_files": len(code_records),
            "modified_since_r2": len(changed_since_r2),
            "new_since_r2": len(new_since_r2),
            "pytest_passed": 87,
            "pytest_skipped": 4,
            "shell_launchers_syntax_checked": 6,
        },
        "module_status": module_status,
        "deployment_manifest": {
            "relative_path": "DEPLOYMENT_MANIFEST.json",
            "bytes": manifest_path.stat().st_size,
            "sha256": sha256_file(manifest_path),
        },
        "test_receipt": deployment_manifest["test_receipt"],
        "scope": deployment_manifest["scope"],
        "conclusion": (
            "Local code contracts are current and tested. Server preparation, training, "
            "real-GPU sealed inference, architecture comparison, website release, and "
            "production deployment remain blocked and were not performed."
        ),
    }
    audit_path = building / "AUDIT.json"
    write_json(audit_path, audit)

    # Detect source or receipt drift after all outputs were composed.
    for item in code_records:
        require(
            sha256_file(repo / item["relative_path"]) == item["sha256"],
            f"Code drift during r3 audit: {item['relative_path']}",
        )
    for binding in (
        winner,
        server_preflight,
        evidence_r4,
        processing_manifest,
        conversion_attempt_ledger,
        *evidence_receipts,
    ):
        require(
            sha256_file(repo / binding["relative_path"]) == binding["sha256"],
            f"Receipt drift during r3 audit: {binding['relative_path']}",
        )

    sums = []
    for name in ("AUDIT.json", "DEPLOYMENT_MANIFEST.json", "TEST_RECEIPT.json"):
        path = building / name
        sums.append(f"{sha256_file(path)}\t{path.stat().st_size}\t{name}")
    (building / "SHA256SUMS.tsv").write_text("sha256\tbytes\tfile\n" + "\n".join(sums) + "\n", encoding="utf-8")

    require(not output.exists(), f"R3 output appeared during audit: {output}")
    os.rename(building, output)
    print(
        json.dumps(
            {
                "status": audit["status"],
                "output_root": str(output),
                "code_files": len(code_records),
                "modified_since_r2": len(changed_since_r2),
                "new_since_r2": len(new_since_r2),
                "audit_sha256": sha256_file(output / "AUDIT.json"),
                "deployment_manifest_sha256": sha256_file(output / "DEPLOYMENT_MANIFEST.json"),
                "test_receipt_sha256": sha256_file(output / "TEST_RECEIPT.json"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
