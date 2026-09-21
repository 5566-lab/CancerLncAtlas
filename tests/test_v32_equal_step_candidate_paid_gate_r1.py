from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / "scripts/validate_v32_equal_step_candidate_static_auth_r1.py"


def _load_gate():
    spec = importlib.util.spec_from_file_location("equal_step_gate_r1", VERIFIER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = _load_gate()
EXTERNAL_MANIFEST = ROOT / "docs/v32_equal_step_candidate_external_deployment_manifest_20260901_r1.json"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _oracle() -> dict[str, object]:
    theta = lambda name: {
        "theta": name,
        "pass": True,
        "scientific_fail_reasons": [],
        "branch_gate": {
            "evaluations": 58,
            "constant_across_all_58": True,
            "paired_disagreement_count": 0,
            "pass": True,
        },
        "components": {
            component: {"pass": True}
            for component in (
                "membership_risk", "direction_loss", "shrinkage_penalty",
                "total_objective",
            )
        },
        "complete_mean_gradient": {"pass": True},
        "module_mean_gradients": {"encoder": {"pass": True}},
    }
    receipt = {
        "format": gate.ORACLE_FORMAT,
        "status": gate.ORACLE_PASS,
        "run_id": gate.ORACLE_RUN_ID,
        "task_id": gate.ORACLE_TASK_ID,
        "graph_variant": "G2",
        "patient_fold": 0,
        "seed": gate.SEED,
        "scientific_pass": True,
        "formal_artifacts_written": 0,
        "formal_training_authorized": False,
        "checkpoint_written": False,
        "success_json_written": False,
        "failure_json_written": False,
        "prediction_written": False,
        "winner_selection_input": False,
        "validation_payload_schema_validated": True,
        "architecture_id": "HHGT_FORMAL_CORE_EXTERNAL_ROUTER",
        "prepared_artifact_sha256": gate.G2_F0_SHA256,
        "preregistration_sha256": _sha(
            ROOT / "docs/v32_group_shared_encoder_estimator_preregistration_20260901.md"
        ),
        "oracle_runner_sha256": _sha(
            ROOT / "cc_hhgt/v32/group_shared_encoder_oracle.py"
        ),
        "production_training_sha256": _sha(ROOT / "cc_hhgt/v32/training.py"),
        "candidate_batch_count": 403,
        "comparison_batch_indices": [0, 1, 2, 3],
        "comparison_batch_row_counts": [8192, 8192, 8192, 8192],
        "comparison_group_rows": 32768,
        "comparison_group_input_sha256": "1" * 64,
        "runtime_chunks": list(range(29)),
        "runtime_chunk_permutation": list(reversed(range(29))),
        "canonical_offset_order": list(range(29)),
        "assignment_sha256": "2" * 64,
        "theta_0_model_state_sha256": "3" * 64,
        "theta_0_rng_sha256": "4" * 64,
        "theta_1_model_state_sha256": "5" * 64,
        "theta_1_rng_sha256": "6" * 64,
        "theta_1_construction": {
            "optimizer_steps": 1,
            "grad_finite": True,
            "parameters_finite": True,
            "parameter_delta_positive": True,
        },
        "theta_results": {
            "theta_0": theta("theta_0"),
            "theta_1": theta("theta_1"),
        },
        "gpu_identity": {"name": "NVIDIA GeForce RTX 4090"},
        "peak_reserved_bytes": 1024,
        "elapsed_seconds": 1.0,
        "training_batches_used": 4,
        "training_batch_indices_used": [0, 1, 2, 3],
        "oof_modality_lineage_preserved": True,
        "patient_fold_authority_preserved": True,
        "optimizer_boundary_preserved": True,
        "immutable_batch_composition_preserved": True,
    }
    return receipt


def test_secure_reader_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text("{}\n", encoding="utf-8")
    link = tmp_path / "link.json"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation unavailable")
    with pytest.raises(gate.EqualStepGateError, match="OPEN_FAILED"):
        gate._read(link, "MALICIOUS_SYMLINK")


def test_secure_reader_rejects_hardlink(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text("{}\n", encoding="utf-8")
    hard = tmp_path / "hard.json"
    os.link(target, hard)
    with pytest.raises(gate.EqualStepGateError, match="HARDLINK_FORBIDDEN"):
        gate._read(target, "MALICIOUS_HARDLINK")


def test_component_containment_rejects_nested_symlink(tmp_path: Path) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir(); outside.mkdir()
    try:
        (root / "nested").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink creation unavailable")
    with pytest.raises(gate.EqualStepGateError, match="SYMLINK_COMPONENT"):
        gate._within(root / "nested/authority.json", root, "MALICIOUS_COMPONENT")


def test_oracle_pass_requires_exact_returned_log_line(tmp_path: Path) -> None:
    receipt = tmp_path / "returned.json"
    log = tmp_path / "oracle.jsonl"
    line = json.dumps(_oracle(), sort_keys=True)
    receipt.write_text(line + "\n", encoding="utf-8")
    log.write_text("noise\n" + line + "\n", encoding="utf-8")
    assert gate.validate_oracle_pass(receipt, log)["terminal_line_sha256"]
    log.write_text(line + "\n" + line + "\n", encoding="utf-8")
    with pytest.raises(gate.EqualStepGateError, match="CARDINALITY=2"):
        gate.validate_oracle_pass(receipt, log)
    log.write_text("noise\n" + line + "\n", encoding="utf-8")
    receipt.write_text(line.replace("true", "false", 1) + "\n", encoding="utf-8")
    with pytest.raises(gate.EqualStepGateError, match="NOT_EXACT_LOG_LINE"):
        gate.validate_oracle_pass(receipt, log)
    shallow = _oracle()
    shallow.pop("theta_results")
    shallow_line = json.dumps(shallow, sort_keys=True)
    receipt.write_text(shallow_line + "\n", encoding="utf-8")
    log.write_text(shallow_line + "\n", encoding="utf-8")
    with pytest.raises(gate.EqualStepGateError, match="ORACLE_THETA_RESULTS"):
        gate.validate_oracle_pass(receipt, log)


def test_terminal_log_rejects_pass_substring_and_unsafe_flags(tmp_path: Path) -> None:
    path = tmp_path / "pilot.jsonl"
    path.write_text(f"text {gate.PASS_STATUS}\n", encoding="utf-8")
    with pytest.raises(gate.EqualStepGateError, match="CARDINALITY=0"):
        gate.validate_terminal_log(path, 0)
    unsafe = {
        "schema": gate.TERMINAL_SCHEMA,
        "status": gate.PASS_STATUS,
        "pilot_id": gate.PILOT_ID,
        "patient_fold": 0,
        "seed": gate.SEED,
        "graph_variants": list(gate.VARIANTS),
        "scientific_pass": True,
        "cross_variant_pass": True,
        "comparison_only": True,
        "candidate_only": True,
        "formal_training_authorized_by_this_receipt": False,
        "formal_artifacts_written": 0,
        "formal_checkpoint_written": True,
        "formal_success_marker_written": False,
        "formal_failure_marker_written": False,
        "formal_prediction_written": False,
        "test_or_outer_artifact_written": False,
        "winner_selection_input": False,
        "output_channel": "STDOUT_JSON_ONLY",
        "test_rows_read": 0,
        "test_labels_read": False,
        "outer_metrics_read": False,
        "outer_predictions_read": False,
    }
    path.write_text(json.dumps(unsafe) + "\n", encoding="utf-8")
    with pytest.raises(gate.EqualStepGateError, match="formal_checkpoint_written"):
        gate.validate_terminal_log(path, 0)


def test_no_torch_bootstrap_has_no_top_level_pilot_or_torch_import() -> None:
    source = (ROOT / gate.BOOTSTRAP_RELATIVE).read_text(encoding="utf-8")
    tree = __import__("ast").parse(source)
    top_imports = []
    for node in tree.body:
        if isinstance(node, (__import__("ast").Import, __import__("ast").ImportFrom)):
            top_imports.append(__import__("ast").unparse(node))
    assert not any("torch" in line for line in top_imports)
    assert not any("equal_step_candidate_pilot" in line for line in top_imports)
    assert "EQUAL_STEP_BOOTSTRAP_TORCH_IMPORTED_BEFORE_GUARDS" in source


def test_checked_external_manifest_and_template_hashes_are_frozen_exactly() -> None:
    manifest = json.loads(EXTERNAL_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["format"] == gate.DEPLOYMENT_FORMAT
    assert [item["name"] for item in manifest["artifacts"]] == list(gate.CONTROL_NAMES)
    for item in manifest["artifacts"]:
        path = ROOT / "scripts" / item["name"]
        assert _sha(path) == item["sha256"]
        assert path.stat().st_size == item["size_bytes"]
    template = json.loads((ROOT / gate.TEMPLATE_RELATIVE).read_text(encoding="utf-8"))
    assert template["equal_step_contract"]["external_deployment_manifest_sha256"] == _sha(EXTERNAL_MANIFEST)
    decision = ROOT / gate.DECISION_RELATIVE
    assert template["equal_step_contract"]["decision_sha256"] == _sha(decision)


def test_stub_static_e2e_materializes_three_exact_component_approvals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "CancerLncAtlas"
    code = project / f"runtime/tools/{gate.NAMESPACE}/code"
    bootstrap = project / f"runtime/bootstrap/{gate.NAMESPACE}"
    control = project / f"runtime/equal_step_gates/{gate.NAMESPACE}"
    prepared = project / "inputs/prepared"
    for relative in (
        gate.PILOT_RELATIVE, gate.BOOTSTRAP_RELATIVE, gate.PREREG_RELATIVE,
        gate.DECISION_RELATIVE,
        gate.TRAINING_RELATIVE, gate.ORACLE_RUNNER_RELATIVE,
        "cc_hhgt/v32/training_guard.py", "scripts/v32_pipeline.py",
        gate.TEMPLATE_RELATIVE,
    ):
        source = ROOT / relative
        target = code / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    # The template binds the fixture-local external manifest hash below.
    rows = []
    for variant_index, variant in enumerate(gate.VARIANTS):
        for fold in range(5):
            path = prepared / variant / f"PATIENT_FOLD_{fold}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(bytes([65 + variant_index, 48 + fold]))
            rows.append({
                "variant": variant, "fold": fold, "path": str(path),
                "sha256": _sha(path), "size_bytes": path.stat().st_size,
            })
    variants = {
        variant: {"fold_inputs": [
            {key: value for key, value in row.items() if key != "variant"}
            for row in rows if row["variant"] == variant
        ]}
        for variant in gate.VARIANTS
    }
    r1_payload = {"status": "STATIC_AUTH_READY", "gpu_visible": False, "variants": variants}
    archived = project / f"runtime/bootstrap/{gate.R1_NAMESPACE}/{gate.R1_ARCHIVE_NAMESPACE}/STATIC_AUTH_READY.live_at_abort.json"
    returned = control / "authority/STATIC_AUTH_READY.v32_g012_20260901_r6.json"
    _write_json(archived, r1_payload); returned.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(archived, returned)
    monkeypatch.setattr(gate, "EXPECTED_R1_STATIC_R6_SHA256", _sha(archived))
    _write_json(
        project / f"runtime/bootstrap/{gate.R1_NAMESPACE}/ABORTED.json",
        {
            "format": gate.ABORT_FORMAT,
            "status": gate.ABORT_STATUS,
            "resume_authorized": False,
            "live_authorizations_removed": True,
            "gpu_training_complete_present": False,
            "formal_result_root_examined": False,
            "formal_result_root_modified": False,
            "formal_result_artifacts_reused": False,
            "deletion_performed": False,
            "archived_authorizations": {
                "static_auth_ready": {
                    "archive_path": str(archived),
                    "sha256": _sha(archived),
                    "size_bytes": archived.stat().st_size,
                }
            },
        },
    )
    monkeypatch.setattr(gate, "FOLD0_SIZE_ANCHORS", {variant: 2 for variant in gate.VARIANTS})
    monkeypatch.setattr(gate, "G2_F0_SHA256", next(row["sha256"] for row in rows if row["variant"] == "G2" and row["fold"] == 0))
    reuse = {
        "format": gate.INPUT_FORMAT, "status": gate.INPUT_STATUS,
        "fold_artifact_count": 15, "fold_artifact_total_bytes": 30,
        "fold_artifacts": rows, "prepared_parent": str(prepared),
        "reused_existing_prepared_inputs": True, "retransfer_performed": False,
        "copy_performed": False, "extraction_performed": False,
        "formal_result_artifacts_used": False, "source_files_modified": False,
        "source_input_archive_sha256": gate.SOURCE_INPUT_ARCHIVE_SHA256,
    }
    reuse_path = project / f"runtime/bootstrap/{gate.FORMAL_R2_NAMESPACE}/INPUT_REUSE_READY.json"
    _write_json(reuse_path, reuse)
    oracle_receipt = control / "authority/ORACLE_PASS_RECEIPT.returned.json"
    oracle_log = project / "runtime/bootstrap/v32_group_shared_oracle_paid_gpu_20260901_r2/ORACLE_EXECUTION.stdout.jsonl"
    oracle_log.parent.mkdir(parents=True, exist_ok=True)
    oracle_line = json.dumps(_oracle(), sort_keys=True)
    oracle_receipt.write_text(oracle_line + "\n", encoding="utf-8")
    oracle_log.write_text(oracle_line + "\n", encoding="utf-8")
    now = datetime.now(timezone.utc)
    jit = {
        "format": gate.JIT_FORMAT, "status": gate.JIT_STATUS, "currency": "CNY",
        "source": "COMPSHARE_API_BILLING_SNAPSHOT", "project_budget_cap_cny": 210.0,
        "compute_hourly_cny": 2.05, "gpu_hourly_cny": 2.05,
        "disk_hourly_cny": 0.04, "total_hourly_cny": 2.09,
        "instance_id": "stub-instance-001", "queried_at": now.isoformat(),
        "valid_until": (now + timedelta(minutes=20)).isoformat(),
        "conservative_remaining_cny": 12.5,
        "budget_method": (
            "FULL_2_09_CNY_RATE_FOR_EVERY_SECOND_SINCE_INSTANCE_CREATE_"
            "MINUS_FUTURE_RESERVE"
        ),
        "instance_age_seconds": 331579,
        "worst_case_full_rate_spend_cny": 192.5,
        "future_cleanup_reserve_cny": 5.0,
        "provider_snapshot_sha256": "7" * 64,
        "generator_sha256": "8" * 64,
    }
    jit_path = bootstrap / "JIT_BUDGET_READY.json"; _write_json(jit_path, jit)
    # Deploy exact controls and then freeze their test manifest.
    control.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for name in gate.CONTROL_NAMES:
        source = VERIFIER if name == gate.VALIDATOR_NAME else ROOT / "scripts" / name
        target = control / name; shutil.copyfile(source, target)
        artifacts.append({"name": name, "server_relative_path": name, "sha256": _sha(target), "size_bytes": target.stat().st_size})
    deployment = {
        "format": gate.DEPLOYMENT_FORMAT, "status": gate.DEPLOYMENT_STATUS,
        "namespace": gate.NAMESPACE, "comparison_only": True,
        "candidate_only": True, "formal_training_authorized": False,
        "artifacts": artifacts,
    }
    deployment_path = control / gate.DEPLOYMENT_NAME; _write_json(deployment_path, deployment)
    template_path = code / gate.TEMPLATE_RELATIVE
    template = json.loads(template_path.read_text(encoding="utf-8"))
    template["equal_step_contract"]["external_deployment_manifest_sha256"] = _sha(deployment_path)
    _write_json(template_path, template)
    archive = bootstrap / "EQUAL_STEP_CODE_BUNDLE.tar.gz"
    archive.parent.mkdir(parents=True, exist_ok=True); archive.write_bytes(b"stub sealed code archive")
    code_receipt = bootstrap / "CODE_ARCHIVE_READY.json"
    gate.materialize_code_receipt(code_root=code, bootstrap_root=bootstrap, template_path=template_path, code_archive=archive, output=code_receipt)
    receipt = gate.materialize(
        project_root=project, code_root=code, bootstrap_root=bootstrap,
        prepared_parent=prepared, template_path=template_path,
        source_reuse=reuse_path, returned_r1=returned,
        oracle_pass=oracle_receipt, oracle_log=oracle_log, jit_path=jit_path,
        code_receipt=code_receipt, deployment_manifest=deployment_path,
        deployment_manifest_sha256=_sha(deployment_path),
        external_verifier_sha256=_sha(control / gate.VALIDATOR_NAME), now=now,
    )
    assert receipt["jit_effective_cost_cap_cny"] == 12.5
    assert receipt["formal_training_authorized"] is False
    assert receipt["budget_authorized_by_this_receipt"] is False
    assert [row["variant"] for row in receipt["component_authorizations"]] == list(gate.VARIANTS)
    for row in receipt["component_authorizations"]:
        approval = json.loads(Path(row["approval_path"]).read_text(encoding="utf-8"))
        assert approval["authorized_trainer"] == gate.TRAINER
        assert approval["approved_task_ids"] == [row["task_id"]]
    # Stub a semantically complete terminal and prove the separate CPU-only
    # handoff recomputes hashes/101/K29/init-RNG/scientific gates.  This handoff
    # remains explicitly non-authorizing for formal r2.
    shared_sha_fields = {
        key: hashlib.sha256(key.encode()).hexdigest()
        for key in (
            "initial_model_sha256", "initial_rng_sha256",
            "candidate_batch_row_counts_sha256", "ordered_train_candidate_key_sha256",
            "ordered_train_label_sha256", "ordered_train_model_input_sha256",
            "ordered_train_mask_weight_sha256", "training_loss_payload_sha256",
            "ordered_train_identity_sha256", "runtime_chunk_permutation_sha256",
            "precision_contract_sha256", "ordered_validation_candidate_key_sha256",
            "ordered_validation_label_sha256", "ordered_validation_model_input_sha256",
            "ordered_validation_mask_weight_sha256", "validation_loss_payload_sha256",
            "validation_loss_contract_sha256", "ordered_validation_identity_sha256",
            "optimization_contract_sha256",
        )
    }
    exact_identity_gate_names = {
        "ordered_candidate_key_sha256", "ordered_label_sha256",
        "ordered_identity_sha256", "ordered_model_input_sha256",
        "ordered_mask_weight_sha256", "validation_loss_payload_sha256",
        "validation_loss_contract_sha256", "precision_contract_sha256",
        "fold_role", "runtime_chunk_weights",
        "runtime_chunk_weights_sha256", "candidate_rows",
    }
    terminal_variants = {}
    for component in receipt["component_authorizations"]:
        variant = component["variant"]
        arm = {
            "training_call_telemetry": {"optimizer_steps": 101},
            "validation_runtime_chunks": 29,
            "initial_model_sha256": shared_sha_fields["initial_model_sha256"],
            "initial_rng_sha256": shared_sha_fields["initial_rng_sha256"],
            "final_model_sha256": hashlib.sha256(f"{variant}-model".encode()).hexdigest(),
            "final_rng_sha256": hashlib.sha256(f"{variant}-rng".encode()).hexdigest(),
            "schedule_sha256": "a" * 64,
            "runtime_chunk_weights_sha256": "b" * 64,
        }
        terminal_variants[variant] = {
            "graph_variant": variant,
            "authorization_run_id": component["run_id"],
            "authorization_task_id": component["task_id"],
            "authorization_endpoint_id": gate.ENDPOINT,
            "authorization_hardware_class": gate.HARDWARE,
            "authorization_artifact_hashes": component["artifact_hashes"],
            "patient_fold": 0,
            "seed": gate.SEED,
            "architecture_id": "HHGT_FORMAL_CORE_EXTERNAL_ROUTER",
            "candidate_batch_count": 403,
            "candidate_rows": 3_300_000,
            "optimizer_group_count": 101,
            "runtime_chunks": list(range(29)),
            "validation_fold_role": {"patient_fold": 0, "role": "validation"},
            "validation_candidate_rows": 1000,
            "prepared_artifact_sha256": rows[[r["variant"] for r in rows].index(variant)]["sha256"],
            **shared_sha_fields,
            "reference": dict(arm),
            "proposed": dict(arm),
            "execution_gates": {
                "reference_optimizer_steps_exact_101": True,
                "proposed_optimizer_steps_exact_101": True,
                "reference_exact_29_chunk_validation": True,
                "proposed_exact_29_chunk_validation": True,
                "same_initial_model": True,
                "same_initial_rng": True,
            },
            "comparison": {
                "pass": True,
                "thresholds": {
                    "validation_logloss_absolute_difference_max": 0.002,
                    "mean_logit_pearson_min": 0.995,
                    "mean_logit_spearman_min": 0.995,
                },
                "gates": {
                    "validation_logloss_absolute_difference": True,
                    "mean_logit_pearson": True,
                    "mean_logit_spearman": True,
                    "ordered_candidate_keys_labels_fold_role_chunk_weights_exact": True,
                },
                "exact_identity_field_gates": {
                    key: True for key in exact_identity_gate_names
                },
                "proxy_label_tensor_exact": True,
                "reference_validation_logloss": 1.0,
                "proposed_validation_logloss": 1.0,
                "validation_logloss_absolute_difference": 0.0,
                "mean_logit_pearson": 0.999,
                "mean_logit_spearman": 0.998,
            },
            "variant_scientific_pass": True,
        }
    exact_cross_gate_names = {
        "architecture_id", "initial_model_sha256", "initial_rng_sha256",
        "candidate_batch_row_counts_sha256", "candidate_rows",
        "ordered_train_candidate_key_sha256", "ordered_train_label_sha256",
        "ordered_train_model_input_sha256", "ordered_train_mask_weight_sha256",
        "training_loss_payload_sha256", "ordered_train_identity_sha256",
        "runtime_chunks", "runtime_chunk_permutation_sha256",
        "precision_contract_sha256", "ordered_validation_candidate_key_sha256",
        "ordered_validation_label_sha256", "ordered_validation_model_input_sha256",
        "ordered_validation_mask_weight_sha256", "validation_loss_payload_sha256",
        "validation_loss_contract_sha256", "ordered_validation_identity_sha256",
        "validation_fold_role", "validation_candidate_rows",
        "optimization_contract_sha256", "reference_schedule_byte_identical",
        "reference_chunk_weights_byte_identical",
        "proposed_schedule_byte_identical",
        "proposed_chunk_weights_byte_identical",
    }
    terminal = {
        "schema": gate.TERMINAL_SCHEMA,
        "status": gate.PASS_STATUS,
        "pilot_id": gate.PILOT_ID,
        "patient_fold": 0,
        "seed": gate.SEED,
        "graph_variants": list(gate.VARIANTS),
        "scientific_pass": True,
        "cross_variant_pass": True,
        "cross_variant_gates": {key: True for key in exact_cross_gate_names},
        "per_variant_pass": {variant: True for variant in gate.VARIANTS},
        "variants": terminal_variants,
        "comparison_only": True,
        "candidate_only": True,
        "formal_training_authorized_by_this_receipt": False,
        "formal_artifacts_written": 0,
        "formal_checkpoint_written": False,
        "formal_success_marker_written": False,
        "formal_failure_marker_written": False,
        "formal_prediction_written": False,
        "test_or_outer_artifact_written": False,
        "winner_selection_input": False,
        "output_channel": "STDOUT_JSON_ONLY",
        "test_rows_read": 0,
        "test_labels_read": False,
        "outer_metrics_read": False,
        "outer_predictions_read": False,
    }
    terminal_log = bootstrap / "EQUAL_STEP_EXECUTION.stdout.jsonl"
    terminal_line = json.dumps(terminal, sort_keys=True)
    terminal_log.write_text(terminal_line + "\n", encoding="utf-8")
    static_path = bootstrap / "STATIC_AUTH_READY.json"
    autostop_path = tmp_path / "AUTOSTOP.json"
    _write_json(
        autostop_path,
        {
            "format": "CANCERLNCATLAS_EQUAL_STEP_CANDIDATE_AUTOSTOP_V1",
            "status": "INSTANCE_STOPPED_STATE_CONFIRMED",
            "namespace": gate.NAMESPACE,
            "stop_trigger": "EQUAL_STEP_PASS_OBSERVED_IMMEDIATE_STOP",
            "instance_id": "stub-instance-001",
            "job_id": gate.JOB_ID,
            "provider_state": "Stopped",
            "gpu_billing_active": False,
            "static_auth_sha256": _sha(static_path),
            "jit_budget_receipt_sha256": _sha(jit_path),
            "terminal_line_sha256": hashlib.sha256(terminal_line.encode()).hexdigest(),
            "terminal_receipt_validated_after_stop": True,
            "dispatch_attempt_not_before_unix": 1_000,
            "job_created_time": 1_001,
            "stopped_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    handoff_path = tmp_path / "EQUAL_STEP_FORMAL_GATE_HANDOFF.json"
    handoff = gate.materialize_formal_gate_handoff(
        static_path=static_path,
        jit_path=jit_path,
        terminal_log=terminal_log,
        autostop_path=autostop_path,
        decision_path=code / gate.DECISION_RELATIVE,
        deployment_path=deployment_path,
        output=handoff_path,
    )
    assert handoff["formal_training_authorized"] is False
    assert handoff["scientific_identity"]["same_initial_model_cross_variant"] is True
    assert [item["variant"] for item in handoff["scientific_identity"]["variants"]] == list(gate.VARIANTS)


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash unavailable")
def test_bash_parses_both_entrypoints() -> None:
    for name in (gate.AUTHORIZER_NAME, gate.LAUNCHER_NAME):
        subprocess.run(
            ["bash", "-n", f"scripts/{name}"], check=True, cwd=ROOT
        )


def test_dispatch_and_supervisor_bind_real_provider_schema_and_live_monitor() -> None:
    dispatch = (
        ROOT
        / "scripts/local_dispatch_compshare_equal_step_candidate_paid_gpu_20260901_r1.ps1"
    ).read_text(encoding="utf-8")
    supervisor = (
        ROOT
        / "scripts/local_supervise_compshare_equal_step_candidate_paid_gpu_20260901_r1.ps1"
    ).read_text(encoding="utf-8")
    assert "instance', 'job', 'list'" in dispatch
    assert "$payload.data.job" in dispatch
    assert "CreatedTime" in dispatch
    assert "DispatchAttemptNotBeforeUnix" in dispatch
    assert "Write-DispatchPhase 'SUBMIT_IN_FLIGHT'" in dispatch
    assert "Write-DispatchPhase 'SUBMITTED'" in dispatch
    assert "Assert-SupervisorAlive 'IMMEDIATELY_BEFORE_PAID_START'" in dispatch
    assert "Assert-SupervisorAlive 'IMMEDIATELY_AFTER_PAID_START'" in dispatch
    assert "$jobPayload.data.job" in supervisor
    assert "$logsPayload.data.logs.Stdout" in supervisor
    assert "$logsPayload.data.logs.Stderr" in supervisor
    assert "[long]$Job.CreatedTime -ge $DispatchAttemptNotBeforeUnix" in supervisor
    assert "Test-EqualStepCompletePassProgress" in (
        ROOT / "scripts/v32_equal_step_candidate_log_guard_r1.psm1"
    ).read_text(encoding="utf-8")
    # Windows PowerShell 5.1 Set-Content -Encoding UTF8 emits a BOM. Receipts
    # consumed by the strict Python verifier must use explicit BOM-free UTF-8.
    assert "[System.IO.File]::WriteAllText" in dispatch
    assert "[System.IO.File]::WriteAllText" in supervisor
    assert "Set-Content -LiteralPath $DispatchJitReceiptPath" not in dispatch
    assert "Set-Content -LiteralPath $partial -Encoding UTF8" not in supervisor


@pytest.mark.skipif(shutil.which("powershell") is None, reason="PowerShell unavailable")
def test_powershell_parses_controls_and_old_tail_cannot_reopen_load_window(
    tmp_path: Path,
) -> None:
    controls = [
        ROOT / "scripts/local_supervise_compshare_equal_step_candidate_paid_gpu_20260901_r1.ps1",
        ROOT / "scripts/local_dispatch_compshare_equal_step_candidate_paid_gpu_20260901_r1.ps1",
        ROOT / "scripts/v32_equal_step_candidate_log_guard_r1.psm1",
    ]
    for path in controls:
        command = (
            "$e=$null;[void][System.Management.Automation.Language.Parser]::ParseFile("
            f"'{path}',[ref]$null,[ref]$e);if($e.Count){{$e|% ToString;exit 1}}"
        )
        subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
            check=True,
        )
    variant = "G0"
    run_id = f"{gate.PILOT_ID}-g0"
    task_id = f"{run_id}|PATIENT_FOLD_0|CC-HHGT|{gate.SEED}"
    setup = {
        "schema": gate.TERMINAL_SCHEMA,
        "status": "EQUAL_STEP_VARIANT_SETUP_START",
        "artifact_type": "CANDIDATE_ONLY_EQUAL_STEP_COMPARISON",
        "comparison_only": True,
        "candidate_only": True,
        "formal_training_authorized_by_this_receipt": False,
        "winner_selection_input": False,
        "formal_artifacts_written": 0,
        "formal_checkpoint_written": False,
        "formal_success_marker_written": False,
        "formal_failure_marker_written": False,
        "formal_prediction_written": False,
        "test_or_outer_artifact_written": False,
        "output_channel": "STDOUT_JSON_ONLY",
        "test_rows_read": 0,
        "test_labels_read": False,
        "outer_metrics_read": False,
        "outer_predictions_read": False,
        "pilot_id": gate.PILOT_ID,
        "authorization_run_id": run_id,
        "authorization_task_id": task_id,
        "graph_variant": variant,
        "arm": "none",
        "phase": "VARIANT_SETUP",
        "completed": 0,
        "total": 1,
        "progress_event_id": f"{gate.PILOT_ID}|{run_id}|{task_id}|G0|none|VARIANT_SETUP|0",
        "optimizer_progress_observed": False,
        "training_started": False,
    }
    line_path = tmp_path / "setup.jsonl"
    line_path.write_text(json.dumps(setup, separators=(",", ":")) + "\n", encoding="utf-8")
    terminal = {
        **{
            key: setup[key]
            for key in (
                "schema", "artifact_type", "comparison_only", "candidate_only",
                "formal_training_authorized_by_this_receipt",
                "winner_selection_input", "formal_artifacts_written",
                "formal_checkpoint_written", "formal_success_marker_written",
                "formal_failure_marker_written", "formal_prediction_written",
                "test_or_outer_artifact_written", "output_channel",
                "test_rows_read", "test_labels_read", "outer_metrics_read",
                "outer_predictions_read", "pilot_id",
            )
        },
        "status": gate.PASS_STATUS,
        "scientific_pass": True,
        "patient_fold": 0,
        "seed": gate.SEED,
        "graph_variants": list(gate.VARIANTS),
        "cross_variant_pass": True,
    }
    pass_path = tmp_path / "pass.jsonl"
    pass_path.write_text(
        json.dumps(terminal, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    probe = tmp_path / "probe.ps1"
    probe.write_text(
        "param($Module,$Line,$PassLine)\n"
        "Import-Module $Module -Force\n"
        "$raw=Get-Content -LiteralPath $Line -Raw -Encoding UTF8\n"
        "$h=New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::Ordinal)\n"
        "$e=New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::Ordinal)\n"
        "$m=@{}\n"
        "$one=Read-EqualStepLogWindow -Logs $raw -SeenLineHashes $h -SeenEventIds $e -MaxCompletedByPhase $m\n"
        "$two=Read-EqualStepLogWindow -Logs $raw -SeenLineHashes $h -SeenEventIds $e -MaxCompletedByPhase $m\n"
        "if(@($one.NewSetupVariants).Count -ne 1 -or @($two.NewSetupVariants).Count -ne 0){exit 9}\n"
        "$passRaw=Get-Content -LiteralPath $PassLine -Raw -Encoding UTF8\n"
        "$h2=New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::Ordinal)\n"
        "$e2=New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::Ordinal)\n"
        "$m2=@{}\n"
        "$incomplete=Read-EqualStepLogWindow -Logs $passRaw -SeenLineHashes $h2 -SeenEventIds $e2 -MaxCompletedByPhase $m2\n"
        "if(-not $incomplete.UnsafeTerminalSeen -or $null -ne $incomplete.TerminalLine){exit 10}\n"
        "$h3=New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::Ordinal)\n"
        "$e3=New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::Ordinal)\n"
        "$m3=@{}\n"
        "foreach($v in @('G0','G1','G2')){$m3[\"$v|none|VARIANT_SETUP\"]=0;$m3[\"$v|reference|OPTIMIZER\"]=101;$m3[\"$v|proposed|OPTIMIZER\"]=101;$m3[\"$v|reference|VALIDATION\"]=29;$m3[\"$v|proposed|VALIDATION\"]=29;$m3[\"$v|both|VARIANT_COMPLETE\"]=1}\n"
        "$complete=Read-EqualStepLogWindow -Logs $passRaw -SeenLineHashes $h3 -SeenEventIds $e3 -MaxCompletedByPhase $m3\n"
        "if($complete.UnsafeTerminalSeen -or $null -eq $complete.TerminalLine){exit 11}\n"
        "$replay=Read-EqualStepLogWindow -Logs $passRaw -SeenLineHashes $h3 -SeenEventIds $e3 -MaxCompletedByPhase $m3\n"
        "if($replay.UnsafeTerminalSeen -or $null -ne $replay.TerminalLine){exit 12}\n"
        "$h4=New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::Ordinal)\n"
        "$e4=New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::Ordinal)\n"
        "$duplicate=Read-EqualStepLogWindow -Logs ($passRaw+$passRaw) -SeenLineHashes $h4 -SeenEventIds $e4 -MaxCompletedByPhase $m3\n"
        "if(-not $duplicate.UnsafeTerminalSeen -or $null -ne $duplicate.TerminalLine){exit 13}\n",
        encoding="utf-8",
    )
    subprocess.run(
        [
            "powershell", "-NoProfile", "-NonInteractive", "-File", str(probe),
            "-Module", str(controls[2]), "-Line", str(line_path),
            "-PassLine", str(pass_path),
        ],
        check=True,
    )
    # Directly exercise the marker-then-exit race boundary: once the monitor
    # process has exited, the immediately-pre-start liveness assertion must
    # fail before a paid `instance start` call can be reached.
    alive_probe = tmp_path / "supervisor_alive_probe.ps1"
    alive_probe.write_text(
        "param($Dispatch)\n"
        "$tokens=$null;$errors=$null\n"
        "$ast=[System.Management.Automation.Language.Parser]::ParseFile($Dispatch,[ref]$tokens,[ref]$errors)\n"
        "if($errors.Count){exit 20}\n"
        "$fn=$ast.Find({param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -ceq 'Assert-SupervisorAlive'},$true)\n"
        "if($null -eq $fn){exit 21}\n"
        ". ([scriptblock]::Create($fn.Extent.Text))\n"
        "$script:supervisorProcess=[pscustomobject]@{HasExited=$true;ExitCode=17}\n"
        "try{Assert-SupervisorAlive 'IMMEDIATELY_BEFORE_PAID_START';exit 22}catch{if($_.Exception.Message -notmatch 'SUPERVISOR_NOT_LIVE_IMMEDIATELY_BEFORE_PAID_START=17'){exit 23}}\n",
        encoding="utf-8",
    )
    subprocess.run(
        [
            "powershell", "-NoProfile", "-NonInteractive", "-File",
            str(alive_probe), "-Dispatch", str(controls[1]),
        ],
        check=True,
    )
