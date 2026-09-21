from __future__ import annotations

import json
import copy
from datetime import datetime, timezone
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from scripts import validate_v32_g012_static_auth_ready_r2 as auth


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/model_v3_2_g012_paid_gpu_20260901_r2.yaml"
SCRIPTS = (
    ROOT / "scripts/cloud_apply_v32_g012_code_patch_no_gpu_20260901_r2.sh",
    ROOT / "scripts/cloud_authorize_v32_g012_no_gpu_20260901_r2.sh",
    ROOT / "scripts/cloud_finalize_v32_g012_no_gpu_20260901_r2.sh",
    ROOT / "scripts/server_launch_v32_g012_paid_gpu_20260901_r2.sh",
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _authorized_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Path]:
    project = tmp_path / "DSC/CancerLncAtlas"
    code = project / "runtime/tools" / auth.NAMESPACE / "code"
    bootstrap = project / "runtime/bootstrap" / auth.NAMESPACE
    prepared = project / "inputs/v32_g012_patient_first_20260830_r1/prepared"
    run_parent = project / "results" / auth.RUN_NAMESPACE
    config = code / "config/model_v3_2_g012_paid_gpu_20260901_r2.yaml"
    receipt = bootstrap / "STATIC_AUTH_READY.json"

    for relative in auth.REQUIRED_CODE_ARTIFACTS:
        target = code / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"fixture:{relative}\n", encoding="utf-8")
    schedule = "balanced_cyclic_single_pass_v1"
    archive_sha = "c" * 64
    manifest = bootstrap / "CODE_ARCHIVE.MANIFEST.json"
    config_payload = {
        "contract_version": "3.2.0-g012-paid-gpu-20260901-r2-test",
        "analysis_version": "CancerLncAtlas_V3.2_G012___GRAPH_VARIANT___R2_TEST",
        "estimator_authorization": {
            "status": auth.ESTIMATOR_READY,
            "decision_receipt_sha256": "a" * 64,
            "selected_schedule_mode": schedule,
            "runtime_estimate_status": "RUNTIME_ESTIMATE_ACCEPTED",
        },
        "execution_control": {
            "execution_mode": "TRAINING",
            "training_authorized": True,
            "paid_enabled": True,
            "max_paid_hours": 96,
            "max_cost_cny": 210,
            "candidate_only": True,
            "overwrite_formal_v32": False,
        },
        "runtime_profile": {
            "candidate_chunk_schedule_mode": schedule,
            "max_coverage_cycles": 6,
            "patience_coverage_cycles": 3,
        },
        "task_contract": {"graph_variant": "__GRAPH_VARIANT__"},
        "formal_gate_receipts": {
            "schema_status": auth.FORMAL_GATE_SCHEMA,
        },
        "fresh_launch_contract": {
            "first_launch_requires_absent_output_root": True,
            "preexisting_checkpoint_forbidden": True,
            "resume_requires_same_r2_lineage": True,
            "lineage_format": "CC_HHGT_V3_2_G012_R2_RUN_LINEAGE_V1",
        },
        "code_archive_contract": {
            "status": auth.EXTERNAL_ARCHIVE_LOCK_REQUIRED,
            "required_status_after_lock": auth.EXTERNAL_ARCHIVE_LOCK_REQUIRED,
            "lock_format": auth.EXTERNAL_ARCHIVE_LOCK_FORMAT,
            "lock_path": str((bootstrap / "CODE_ARCHIVE.LOCK.json").resolve()),
            "root_of_trust": "ARCHIVE_EXTERNAL_BOOTSTRAP_LOCK",
            "lock_sha256_authority": (
                "EXTERNAL_BOOTSTRAP_CONTROLLER_ENV_V32_R2_EXTERNAL_LOCK_SHA256"
            ),
            "lock_is_inside_code_archive": False,
            "archive_hash_embedded_in_archive": False,
            "manifest_hash_embedded_in_archive": False,
        },
        "training_io": {
            "prepared_fold_pattern": str(prepared.resolve())
            + "/__GRAPH_VARIANT__/PATIENT_FOLD_{fold}.pt",
            "output_root": str(run_parent.resolve()) + "/__GRAPH_VARIANT__/training",
        },
    }
    config.write_text(yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8")
    manifest_files = []
    for code_file in sorted(path for path in code.rglob("*") if path.is_file()):
        relative = code_file.relative_to(code).as_posix()
        manifest_files.append(
            {
                "path": relative,
                "sha256": _sha(code_file),
                "size_bytes": code_file.stat().st_size,
                "mode": 0o755 if relative.endswith((".sh", ".ps1")) else 0o644,
            }
        )
    _write_json(
        manifest,
        {
            "format": "CC_HHGT_V3_2_G012_R2_CODE_ARCHIVE_MANIFEST_V1",
            "namespace": auth.NAMESPACE,
            "archive_sha256": archive_sha,
            "files": manifest_files,
        },
    )

    records = []
    total = 0
    for variant in auth.VARIANTS:
        for fold in auth.FOLDS:
            fold_path = prepared / variant / f"PATIENT_FOLD_{fold}.pt"
            fold_path.parent.mkdir(parents=True, exist_ok=True)
            fold_path.write_bytes(f"{variant}:{fold}\n".encode())
            size = fold_path.stat().st_size
            total += size
            records.append(
                {
                    "variant": variant,
                    "fold": fold,
                    "path": str(fold_path.resolve()),
                    "sha256": f"{fold + 1:x}" * 64,
                    "size_bytes": size,
                }
            )
    r1_static = (
        bootstrap.parent
        / auth.R1_NAMESPACE
        / auth.R1_ARCHIVE_NAMESPACE
        / "STATIC_AUTH_READY.live_at_abort.json"
    )
    r1_variants = {
        variant: {
            "fold_inputs": [
                {
                    "fold": record["fold"],
                    "path": record["path"],
                    "sha256": record["sha256"],
                    "size_bytes": record["size_bytes"],
                }
                for record in records
                if record["variant"] == variant
            ]
        }
        for variant in auth.VARIANTS
    }
    _write_json(r1_static, {"status": "STATIC_AUTH_READY", "variants": r1_variants})
    r1_static_sha = _sha(r1_static)
    monkeypatch.setattr(auth, "R1_STATIC_AUTH_R6_SHA256", r1_static_sha)
    _write_json(
        bootstrap / "INPUT_REUSE_READY.json",
        {
            "format": auth.INPUT_FORMAT,
            "status": auth.INPUT_STATUS,
            "prepared_parent": str(prepared.resolve()),
            "source_input_archive_sha256": auth.SOURCE_INPUT_ARCHIVE_SHA256,
            "r1_static_auth_r6_path": str(r1_static.resolve()),
            "r1_static_auth_r6_sha256": r1_static_sha,
            "fold_artifact_count": 15,
            "fold_artifact_total_bytes": total,
            "fold_artifacts": records,
            "reused_existing_prepared_inputs": True,
            "retransfer_performed": False,
            "copy_performed": False,
            "extraction_performed": False,
            "source_files_modified": False,
            "formal_result_artifacts_used": False,
        },
    )
    _write_json(
        bootstrap.parent / auth.R1_NAMESPACE / "ABORTED.json",
        {
            "format": auth.ABORT_FORMAT,
            "status": auth.ABORT_STATUS,
            "resume_authorized": False,
            "superseded_by_namespace": auth.NAMESPACE,
            "instance_id": auth.INSTANCE_ID,
            "r1_patch_ready_sha256": auth.R1_PATCH_READY_SHA256,
            "r1_static_auth_r6_sha256": r1_static_sha,
            "formal_result_root_examined": False,
            "formal_result_root_modified": False,
            "formal_result_artifacts_reused": False,
            "deletion_performed": False,
            "live_authorizations_removed": True,
            "gpu_training_complete_present": False,
            "instance_stop_evidence": {
                "instance_id": auth.INSTANCE_ID,
                "observed_state": "Stopped",
                "gpu_billing_active": False,
                "age_seconds_at_seal": 60,
            },
        },
    )
    cpu_ready = bootstrap.parent / auth.R1_NAMESPACE / "CPU_READY"
    cpu_ready.write_text(
        f"{auth.SOURCE_INPUT_ARCHIVE_SHA256}\t{auth.R1_BASE_CODE_ARCHIVE_SHA256}\t15\n",
        encoding="ascii",
    )
    _, deployment_sha = auth._code_artifacts(code)
    code_tree_sha = auth._code_tree_sha256(code)
    bootstrap_verifier = bootstrap / "verify_v32_g012_r2_overlay_manifest.py"
    bootstrap_verifier.write_bytes(
        (code / "scripts/verify_v32_g012_r2_overlay_manifest.py").read_bytes()
    )
    external_lock = bootstrap / "CODE_ARCHIVE.LOCK.json"
    _write_json(
        external_lock,
        {
            "format": auth.EXTERNAL_ARCHIVE_LOCK_FORMAT,
            "status": auth.EXTERNAL_ARCHIVE_LOCK_STATUS,
            "namespace": auth.NAMESPACE,
            "created_at": "2026-09-01T05:59:00+00:00",
            "archive_path": "/tmp/v32_g012_r2_code_archive_20260901_r1.tar.gz",
            "archive_sha256": archive_sha,
            "manifest_path": str(manifest.resolve()),
            "manifest_sha256": _sha(manifest),
            "bootstrap_verifier_path": str(bootstrap_verifier.resolve()),
            "bootstrap_verifier_sha256": _sha(bootstrap_verifier),
            "code_archive_contains_lock": False,
            "hashes_embedded_in_archive": False,
            "formal_training_authorized": False,
        },
    )
    _write_json(
        bootstrap / "PATCH_READY.json",
        {
            "format": auth.PATCH_FORMAT,
            "status": "PATCH_READY",
            "namespace": auth.NAMESPACE,
            "created_at": "2026-09-01T06:00:00+00:00",
            "cpu_only": True,
            "gpu_visible": False,
            "code_root": str(code.resolve()),
            "baseline_root": str(
                (code.parent / "code.baseline_before_archive_r1").resolve()
            ),
            "archive_path": "/tmp/v32_g012_r2_code_archive_20260901_r1.tar.gz",
            "overlay_code_imported": False,
            "overlay_code_executed": False,
            "prepared_inputs_touched": False,
            "formal_result_artifacts_used": False,
            "archive_sha256": archive_sha,
            "manifest_path": str(manifest.resolve()),
            "manifest_sha256": _sha(manifest),
            "bootstrap_verifier_sha256": _sha(
                code / "scripts/verify_v32_g012_r2_overlay_manifest.py"
            ),
            "bootstrap_verifier_path": str(
                bootstrap_verifier.resolve()
            ),
            "external_code_archive_lock_path": str(external_lock.resolve()),
            "external_code_archive_lock_sha256": _sha(external_lock),
            "code_tree_sha256": code_tree_sha,
            "deployment_contract_sha256": deployment_sha,
        },
    )
    return {
        "project": project,
        "code": code,
        "bootstrap": bootstrap,
        "prepared": prepared,
        "run_parent": run_parent,
        "config": config,
        "receipt": receipt,
    }


def _formal_v2_stub() -> dict[str, object]:
    formal: dict[str, object] = {
        "schema": auth.FORMAL_GATE_SCHEMA,
    }
    formal.update(
        {
            field: f"{index + 1:x}"[-1] * 64
            for index, field in enumerate(auth.FORMAL_V2_HASH_FIELDS)
        }
    )
    formal.update({field: True for field in auth.FORMAL_V2_PASS_FIELDS})
    completed = {
        "state": "COMPLETED_BEFORE_FRESH_JIT",
        "hours_cap": 0,
        "cost_cap_cny": 0,
        "completion_receipt_sha256": "d" * 64,
        "completed_at": "2026-09-01T00:00:00+00:00",
    }
    formal["budget_projection"] = {
        "format": auth.FORMAL_V2_BUDGET_FORMAT,
        "projection_receipt_sha256": formal["formal_runtime_projection_sha256"],
        "fresh_jit_receipt_sha256": formal["formal_jit_budget_sha256"],
        "project_budget_cap_cny": 210.0,
        "full_rate_cny_per_hour": 2.09,
        "fresh_jit_conservative_remaining_cny": 100.0,
        "fresh_jit_queried_at": "2026-09-01T01:00:00+00:00",
        "fresh_jit_valid_until": "2026-09-01T01:25:00+00:00",
        "formal_runtime_projection_hours": 40.0,
        "components": {
            "cpu_nogpu_prepare_r2": dict(completed),
            "cpu_nogpu_oracle_authorization_transport": dict(completed),
            "paid_oracle": dict(completed),
            "cpu_nogpu_equal_step_authorization_transport": dict(completed),
            "paid_equal_step": dict(completed),
            "cpu_nogpu_formal_authorization_transport": {
                "state": "FUTURE_CAP_INCLUDED",
                "hours_cap": 0.5,
                "cost_cap_cny": 1.045,
                "completion_receipt_sha256": None,
                "completed_at": None,
            },
            "paid_formal_training": {
                "state": "FUTURE_CAP_INCLUDED",
                "hours_cap": 40.0,
                "cost_cap_cny": 83.6,
                "completion_receipt_sha256": None,
                "completed_at": None,
            },
            "provider_stop_result_return_margin": {
                "state": "FUTURE_CAP_INCLUDED",
                "hours_cap": 0.5,
                "cost_cap_cny": 1.045,
                "completion_receipt_sha256": None,
                "completed_at": None,
            },
        },
        "total_projected_future_hours": 41.0,
        "total_projected_future_cost_cny": 85.69,
        "budget_headroom_cny": 14.31,
        "cleanup_and_validity_reserve_already_excluded": True,
        "all_future_cpu_nogpu_plus_paid_components_included": True,
        "stop_result_return_margin_included": True,
    }
    return formal


def _install_safe_gate_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    formal = _formal_v2_stub()
    monkeypatch.setattr(auth, "_formal_receipt_specs", lambda *args, **kwargs: formal)
    monkeypatch.setattr(
        auth,
        "_runtime_snapshot",
        lambda *, code_tree_sha256: {
            "format": auth.RUNTIME_READY_FORMAT,
            "status": auth.RUNTIME_READY_STATUS,
            "namespace": auth.NAMESPACE,
            "cpu_only": True,
            "gpu_visible": False,
            "cuda_api_inspected": False,
            "torch_imported": False,
            "metadata_api": "importlib.metadata",
            "python_executable": str(Path(sys.executable).resolve()),
            "python_version": "fixture",
            "python_implementation": "fixture",
            "distributions": {
                "torch": "fixture",
                "torch-geometric": "fixture",
                "PyYAML": "fixture",
            },
            "required_distributions_present": True,
            "code_tree_sha256": code_tree_sha256,
        },
    )


def test_checked_in_config_is_an_explicit_zero_budget_blocker() -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert payload["estimator_authorization"]["status"] == auth.BLOCKER
    assert payload["runtime_profile"]["candidate_chunk_schedule_mode"] == auth.BLOCKER
    assert payload["runtime_profile"]["max_coverage_cycles"] == auth.BLOCKER
    assert payload["execution_control"]["training_authorized"] is False
    assert payload["execution_control"]["paid_enabled"] is False
    assert payload["execution_control"]["max_paid_hours"] == 0
    assert payload["execution_control"]["max_cost_cny"] == 0
    formal = payload["formal_gate_receipts"]
    assert formal["schema_status"] == auth.FORMAL_GATE_BLOCKER
    assert formal["required_schema"] == auth.FORMAL_GATE_SCHEMA
    assert formal["oracle_prerequisite_schema"] == auth.ORACLE_PREREQUISITE_SCHEMA
    assert set(formal["required_receipt_families"]) == {
        "real_data_oracle_prerequisite",
        "equal_step_g0_g1_g2_pilot",
        "formal_training_decision",
        "formal_runtime_and_cost_projection",
        "all_future_cpu_nogpu_plus_paid_instance_time_projection",
        "fresh_formal_training_jit_budget",
        "formal_provider_auto_stop_margin",
    }
    assert set(formal["formal_v2_hashes"]) == set(auth.FORMAL_V2_HASH_FIELDS)
    assert set(formal["formal_v2_pass_flags"]) == set(auth.FORMAL_V2_PASS_FIELDS)
    assert set(formal["formal_v2_hashes"].values()) == {auth.FORMAL_GATE_BLOCKER}
    assert all(value is False for value in formal["formal_v2_pass_flags"].values())
    budget = formal["budget_projection_contract"]
    assert budget["format"] == auth.FORMAL_V2_BUDGET_FORMAT
    assert budget["project_budget_cap_cny"] == 210.0
    assert budget["full_rate_cny_per_hour"] == 2.09
    assert set(budget["required_components"]) == set(auth.FORMAL_V2_BUDGET_COMPONENTS)
    assert budget["minimum_future_hours"] == {
        key: float(value) for key, value in auth.FORMAL_V2_MINIMUM_FUTURE_HOURS.items()
    }
    archive_contract = payload["code_archive_contract"]
    assert archive_contract["status"] == auth.EXTERNAL_ARCHIVE_LOCK_REQUIRED
    assert archive_contract["lock_format"] == auth.EXTERNAL_ARCHIVE_LOCK_FORMAT
    assert archive_contract["lock_is_inside_code_archive"] is False
    assert archive_contract["lock_sha256_authority"].endswith(
        "V32_R2_EXTERNAL_LOCK_SHA256"
    )
    assert archive_contract["archive_hash_embedded_in_archive"] is False
    assert archive_contract["manifest_hash_embedded_in_archive"] is False
    assert "archive_sha256" not in archive_contract
    assert "manifest_sha256" not in archive_contract
    patcher_source = SCRIPTS[0].read_text(encoding="utf-8")
    assert "CODE_ARCHIVE_SHA256_NOT_YET_LOCKED" not in patcher_source
    assert "CODE_ARCHIVE_MANIFEST_SHA256_NOT_YET_LOCKED" not in patcher_source
    assert 'external_lock="$bootstrap_root/CODE_ARCHIVE.LOCK.json"' in patcher_source
    assert 'required_external_lock_sha256="${V32_R2_EXTERNAL_LOCK_SHA256:-}"' in patcher_source
    assert "R2_EXTERNAL_LOCK_SHA256_NOT_INJECTED_BY_BOOTSTRAP_ROOT" in patcher_source
    assert auth.RUN_NAMESPACE in payload["training_io"]["output_root"]
    assert auth.R1_RESULT_NAMESPACE not in payload["training_io"]["output_root"]


def test_checked_in_config_cli_exits_typed_42_before_path_dependencies(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/validate_v32_g012_static_auth_ready_r2.py"),
            "--config-only",
            "--config",
            str(CONFIG),
            "--prepared-parent",
            str(tmp_path / "missing-prepared"),
            "--run-parent",
            str(tmp_path / "missing-run"),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 42
    assert result.stderr.strip() == auth.BLOCKER
    assert list(tmp_path.iterdir()) == []


def test_authorized_estimator_cannot_bypass_unlocked_formal_receipt_schema(
    tmp_path: Path,
) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["estimator_authorization"] = {
        "status": auth.ESTIMATOR_READY,
        "decision_receipt_sha256": "a" * 64,
        "selected_schedule_mode": "balanced_cyclic_single_pass_v1",
        "runtime_estimate_status": "RUNTIME_ESTIMATE_ACCEPTED",
    }
    payload["runtime_profile"].update(
        {
            "candidate_chunk_schedule_mode": "balanced_cyclic_single_pass_v1",
            "max_coverage_cycles": 6,
            "patience_coverage_cycles": 3,
        }
    )
    payload["execution_control"].update(
        {
            "training_authorized": True,
            "paid_enabled": True,
            "max_paid_hours": 96,
            "max_cost_cny": 210,
        }
    )
    plausible = tmp_path / "plausible-decision.json"
    _write_json(plausible, {"decision": "PASS", "training_authorized": True})
    payload["formal_gate_receipts"]["decision"] = {
        "path": str(plausible),
        "sha256": _sha(plausible),
    }
    config = tmp_path / "authorized-estimator.yaml"
    config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    gate = auth.validate_estimator_config(config)
    with pytest.raises(auth.R2AuthorizationError, match=auth.FORMAL_GATE_BLOCKER):
        auth._formal_receipt_specs(
            gate["config"],
            code_root=tmp_path / "code",
            bootstrap_root=tmp_path / auth.NAMESPACE,
        )


@pytest.mark.parametrize(
    "schema_status",
    [
        auth.FORMAL_GATE_BLOCKER,
        auth.ORACLE_PREREQUISITE_SCHEMA,
        auth.FORMAL_GATE_SCHEMA,
    ],
)
def test_oracle_only_or_unimplemented_formal_v2_never_materializes_static_auth(
    tmp_path: Path,
    schema_status: str,
) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["estimator_authorization"] = {
        "status": auth.ESTIMATOR_READY,
        "decision_receipt_sha256": "a" * 64,
        "selected_schedule_mode": "balanced_cyclic_single_pass_v1",
        "runtime_estimate_status": "RUNTIME_ESTIMATE_ACCEPTED",
    }
    payload["runtime_profile"].update(
        {
            "candidate_chunk_schedule_mode": "balanced_cyclic_single_pass_v1",
            "max_coverage_cycles": 6,
            "patience_coverage_cycles": 3,
        }
    )
    payload["execution_control"].update(
        {
            "training_authorized": True,
            "paid_enabled": True,
            "max_paid_hours": 96,
            "max_cost_cny": 210,
        }
    )
    payload["formal_gate_receipts"]["schema_status"] = schema_status
    prepared = tmp_path / "prepared"
    run_parent = tmp_path / "results" / auth.RUN_NAMESPACE
    payload["training_io"]["prepared_fold_pattern"] = (
        str(prepared.resolve()) + "/__GRAPH_VARIANT__/PATIENT_FOLD_{fold}.pt"
    )
    payload["training_io"]["output_root"] = (
        str(run_parent.resolve()) + "/__GRAPH_VARIANT__/training"
    )
    config = tmp_path / "authorized-estimator.yaml"
    config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    bootstrap = tmp_path / "bootstrap" / auth.NAMESPACE
    receipt = bootstrap / "STATIC_AUTH_READY.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/validate_v32_g012_static_auth_ready_r2.py"),
            "--materialize",
            "--config",
            str(config),
            "--prepared-parent",
            str(prepared),
            "--run-parent",
            str(run_parent),
            "--bootstrap-root",
            str(bootstrap),
            "--receipt",
            str(receipt),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 42
    assert result.stderr.strip() == auth.FORMAL_GATE_BLOCKER
    assert not receipt.exists()
    assert not (bootstrap / "fresh_authority").exists()


def test_jit_generator_extension_is_accepted_and_budget_is_recomputed(
    tmp_path: Path,
) -> None:
    generator_source = (
        ROOT / "scripts/local_materialize_compshare_jit_budget_20260901_r2.ps1"
    ).read_text(encoding="utf-8")
    generator_path = (
        ROOT / "scripts/local_materialize_compshare_jit_budget_20260901_r2.ps1"
    )
    assert "$ExpectedInstanceId = 'uhost-1up504geeqzj'" in generator_source
    assert "$InstanceId -cne $ExpectedInstanceId" in generator_source
    receipt = tmp_path / "JIT_BUDGET_READY.json"
    payload = {
        "format": auth.JIT_FORMAT,
        "status": auth.JIT_STATUS,
        "currency": "CNY",
        "source": "COMPSHARE_API_BILLING_SNAPSHOT",
        "project_budget_cap_cny": auth.PROJECT_BUDGET_CAP_CNY,
        "compute_hourly_cny": auth.COMPUTE_HOURLY_CNY,
        "gpu_hourly_cny": auth.GPU_HOURLY_CNY,
        "disk_hourly_cny": auth.DISK_HOURLY_CNY,
        "total_hourly_cny": auth.TOTAL_HOURLY_CNY,
        "instance_id": auth.INSTANCE_ID,
        "queried_at": "2026-09-01T06:00:00+00:00",
        "valid_until": "2026-09-01T06:25:00+00:00",
        "conservative_remaining_cny": 202.0391,
        "budget_method": "FULL_2_09_CNY_RATE_FOR_EVERY_SECOND_SINCE_INSTANCE_CREATE_MINUS_FUTURE_RESERVE",
        "instance_age_seconds": 3600,
        "provider_stop_time_unix": 1788242300,
        "provider_query_duration_seconds": 1.0,
        "worst_case_full_rate_spend_cny": 2.09,
        "future_cleanup_reserve_cny": 5.8709,
        "worst_case_spend_rounding": "CEILING_0_0001_CNY",
        "remaining_rounding": "FLOOR_0_0001_CNY",
        "unallocated_rounding_reserve_cny": 0.0,
        "project_scope_format": auth.JIT_PROJECT_SCOPE_FORMAT,
        "project_scope_sha256": auth.JIT_PROJECT_SCOPE_SHA256,
        "profile_name_sha256": auth.JIT_PROFILE_NAME_SHA256,
        "provider_snapshot_sha256": "b" * 64,
        "generator_sha256": _sha(generator_path),
    }
    _write_json(receipt, payload)
    validated = auth._validate_jit_budget(
        receipt,
        _sha(receipt),
        require_hardened_generator=True,
        generator_path=generator_path,
    )
    assert validated["remaining_cny"] == 202.0391

    payload["future_cleanup_reserve_cny"] = 5.0
    payload["conservative_remaining_cny"] = 202.91
    _write_json(receipt, payload)
    with pytest.raises(
        auth.R2AuthorizationError, match="VALIDITY_TRANSPORT_RESERVE_TOO_SMALL"
    ):
        auth._validate_jit_budget(
            receipt,
            _sha(receipt),
            require_hardened_generator=True,
            generator_path=generator_path,
        )

    payload["future_cleanup_reserve_cny"] = 5.8709
    payload["conservative_remaining_cny"] = 202.0392
    _write_json(receipt, payload)
    with pytest.raises(
        auth.R2AuthorizationError, match="CONSERVATIVE_REMAINING_DRIFT"
    ):
        auth._validate_jit_budget(
            receipt,
            _sha(receipt),
            require_hardened_generator=True,
            generator_path=generator_path,
        )


def test_hardened_jit_requires_exact_deployed_generator_source(tmp_path: Path) -> None:
    generator_path = (
        ROOT / "scripts/local_materialize_compshare_jit_budget_20260901_r2.ps1"
    )
    receipt = tmp_path / "JIT_BUDGET_READY.json"
    payload = {
        "format": auth.JIT_FORMAT,
        "status": auth.JIT_STATUS,
        "currency": "CNY",
        "source": "COMPSHARE_API_BILLING_SNAPSHOT",
        "project_budget_cap_cny": auth.PROJECT_BUDGET_CAP_CNY,
        "compute_hourly_cny": auth.COMPUTE_HOURLY_CNY,
        "gpu_hourly_cny": auth.GPU_HOURLY_CNY,
        "disk_hourly_cny": auth.DISK_HOURLY_CNY,
        "total_hourly_cny": auth.TOTAL_HOURLY_CNY,
        "instance_id": auth.INSTANCE_ID,
        "queried_at": "2026-09-01T06:00:00+00:00",
        "valid_until": "2026-09-01T06:25:00+00:00",
        "conservative_remaining_cny": 202.0391,
        "budget_method": "FULL_2_09_CNY_RATE_FOR_EVERY_SECOND_SINCE_INSTANCE_CREATE_MINUS_FUTURE_RESERVE",
        "instance_age_seconds": 3600,
        "provider_stop_time_unix": 1788242300,
        "provider_query_duration_seconds": 1.0,
        "worst_case_full_rate_spend_cny": 2.09,
        "future_cleanup_reserve_cny": 5.8709,
        "worst_case_spend_rounding": "CEILING_0_0001_CNY",
        "remaining_rounding": "FLOOR_0_0001_CNY",
        "unallocated_rounding_reserve_cny": 0.0,
        "project_scope_format": auth.JIT_PROJECT_SCOPE_FORMAT,
        "project_scope_sha256": auth.JIT_PROJECT_SCOPE_SHA256,
        "profile_name_sha256": auth.JIT_PROFILE_NAME_SHA256,
        "provider_snapshot_sha256": "b" * 64,
        "generator_sha256": "c" * 64,
    }
    _write_json(receipt, payload)

    with pytest.raises(
        auth.R2AuthorizationError, match="GENERATOR_SOURCE_BINDING_REQUIRED"
    ):
        auth._validate_jit_budget(
            receipt, _sha(receipt), require_hardened_generator=True
        )
    with pytest.raises(
        auth.R2AuthorizationError, match="GENERATOR_SOURCE_SHA256_DRIFT"
    ):
        auth._validate_jit_budget(
            receipt,
            _sha(receipt),
            require_hardened_generator=True,
            generator_path=generator_path,
        )


@pytest.mark.parametrize(
    "missing_field",
    [*auth.FORMAL_V2_HASH_FIELDS, *auth.FORMAL_V2_PASS_FIELDS],
)
def test_formal_v2_missing_hash_or_pass_flag_never_materializes_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing_field: str,
) -> None:
    fixture = _authorized_fixture(tmp_path, monkeypatch)
    formal = _formal_v2_stub()
    formal.pop(missing_field)
    monkeypatch.setattr(auth, "_formal_receipt_specs", lambda *args, **kwargs: formal)
    with pytest.raises(auth.R2AuthorizationError, match=auth.FORMAL_GATE_BLOCKER):
        auth.materialize_static_authorization(
            receipt_path=fixture["receipt"],
            config_path=fixture["config"],
            code_root=fixture["code"],
            prepared_parent=fixture["prepared"],
            run_parent=fixture["run_parent"],
            bootstrap_root=fixture["bootstrap"],
        )
    assert not fixture["receipt"].exists()
    assert not (fixture["bootstrap"] / "fresh_authority").exists()


@pytest.mark.parametrize("false_flag", auth.FORMAL_V2_PASS_FIELDS)
def test_formal_v2_false_pass_flag_never_materializes_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    false_flag: str,
) -> None:
    fixture = _authorized_fixture(tmp_path, monkeypatch)
    formal = _formal_v2_stub()
    formal[false_flag] = False
    monkeypatch.setattr(auth, "_formal_receipt_specs", lambda *args, **kwargs: formal)
    with pytest.raises(auth.R2AuthorizationError, match=auth.FORMAL_GATE_BLOCKER):
        auth.materialize_static_authorization(
            receipt_path=fixture["receipt"],
            config_path=fixture["config"],
            code_root=fixture["code"],
            prepared_parent=fixture["prepared"],
            run_parent=fixture["run_parent"],
            bootstrap_root=fixture["bootstrap"],
        )
    assert not fixture["receipt"].exists()
    assert not (fixture["bootstrap"] / "fresh_authority").exists()


@pytest.mark.parametrize("oracle_only", [False, True])
def test_formal_v2_unknown_hash_or_oracle_only_mapping_never_materializes_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    oracle_only: bool,
) -> None:
    fixture = _authorized_fixture(tmp_path, monkeypatch)
    if oracle_only:
        formal: dict[str, object] = {
            "schema": auth.ORACLE_PREREQUISITE_SCHEMA,
            "oracle_decision_sha256": "a" * 64,
            "oracle_static_auth_sha256": "b" * 64,
            "oracle_terminal_sha256": "c" * 64,
            "real_data_oracle_pass": True,
        }
    else:
        formal = _formal_v2_stub()
        formal["unknown_receipt_sha256"] = "f" * 64
    monkeypatch.setattr(auth, "_formal_receipt_specs", lambda *args, **kwargs: formal)
    with pytest.raises(auth.R2AuthorizationError, match=auth.FORMAL_GATE_BLOCKER):
        auth.materialize_static_authorization(
            receipt_path=fixture["receipt"],
            config_path=fixture["config"],
            code_root=fixture["code"],
            prepared_parent=fixture["prepared"],
            run_parent=fixture["run_parent"],
            bootstrap_root=fixture["bootstrap"],
        )
    assert not fixture["receipt"].exists()
    assert not (fixture["bootstrap"] / "fresh_authority").exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_projection",
        "old_123_31_three_paid_components_only",
        "cpu_coverage_flag_false",
        "stop_margin_omitted",
        "component_cost_drift",
        "total_exceeds_fresh_jit",
        "formal_runtime_above_50h",
        "formal_cpu_claimed_completed",
        "epsilon_stop_margin",
        "completion_after_fresh_jit",
        "runtime_projection_hours_mismatch",
    ],
)
def test_formal_v2_complete_future_instance_time_budget_is_mandatory(
    mutation: str,
) -> None:
    formal = _formal_v2_stub()
    projection = formal["budget_projection"]
    assert isinstance(projection, dict)
    components = projection["components"]
    assert isinstance(components, dict)

    if mutation == "missing_projection":
        formal.pop("budget_projection")
    elif mutation == "old_123_31_three_paid_components_only":
        projection["components"] = {
            "paid_oracle": {
                "state": "FUTURE_CAP_INCLUDED",
                "hours_cap": 3.0,
                "cost_cap_cny": 6.27,
                "completion_receipt_sha256": None,
                "completed_at": None,
            },
            "paid_equal_step": {
                "state": "FUTURE_CAP_INCLUDED",
                "hours_cap": 6.0,
                "cost_cap_cny": 12.54,
                "completion_receipt_sha256": None,
                "completed_at": None,
            },
            "paid_formal_training": {
                "state": "FUTURE_CAP_INCLUDED",
                "hours_cap": 50.0,
                "cost_cap_cny": 104.5,
                "completion_receipt_sha256": None,
                "completed_at": None,
            },
        }
        projection["formal_runtime_projection_hours"] = 50.0
        projection["total_projected_future_hours"] = 59.0
        projection["total_projected_future_cost_cny"] = 123.31
        projection["fresh_jit_conservative_remaining_cny"] = 125.0784
        projection["budget_headroom_cny"] = 1.7684
    elif mutation == "cpu_coverage_flag_false":
        projection["all_future_cpu_nogpu_plus_paid_components_included"] = False
    elif mutation == "stop_margin_omitted":
        components.pop("provider_stop_result_return_margin")
    elif mutation == "component_cost_drift":
        components["cpu_nogpu_formal_authorization_transport"]["cost_cap_cny"] = 0
    elif mutation == "total_exceeds_fresh_jit":
        projection["fresh_jit_conservative_remaining_cny"] = 85.68
        projection["budget_headroom_cny"] = -0.01
    elif mutation == "formal_runtime_above_50h":
        components["paid_formal_training"]["hours_cap"] = 50.1
        components["paid_formal_training"]["cost_cap_cny"] = 104.709
        projection["formal_runtime_projection_hours"] = 50.1
        projection["total_projected_future_hours"] = 51.1
        projection["total_projected_future_cost_cny"] = 106.799
        projection["fresh_jit_conservative_remaining_cny"] = 120.0
        projection["budget_headroom_cny"] = 13.201
    elif mutation == "formal_cpu_claimed_completed":
        components["cpu_nogpu_formal_authorization_transport"] = {
            "state": "COMPLETED_BEFORE_FRESH_JIT",
            "hours_cap": 0,
            "cost_cap_cny": 0,
            "completion_receipt_sha256": "e" * 64,
            "completed_at": "2026-09-01T00:30:00+00:00",
        }
        projection["total_projected_future_hours"] = 40.5
        projection["total_projected_future_cost_cny"] = 84.645
        projection["budget_headroom_cny"] = 15.355
    elif mutation == "epsilon_stop_margin":
        components["provider_stop_result_return_margin"]["hours_cap"] = 0.0001
        components["provider_stop_result_return_margin"]["cost_cap_cny"] = 0.0003
        projection["total_projected_future_hours"] = 40.5001
        projection["total_projected_future_cost_cny"] = 84.6453
        projection["budget_headroom_cny"] = 15.3547
    elif mutation == "completion_after_fresh_jit":
        components["paid_equal_step"]["completed_at"] = "2026-09-01T01:01:00+00:00"
    elif mutation == "runtime_projection_hours_mismatch":
        projection["formal_runtime_projection_hours"] = 39.0
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(mutation)

    with pytest.raises(auth.R2AuthorizationError, match=auth.FORMAL_GATE_BLOCKER):
        auth._formal_prerequisite_hashes(formal)


def _passing_theta(theta: str) -> dict[str, object]:
    component = {
        "reference_mean": 1.0,
        "proposed_mean": 1.0,
        "absolute_mean_difference": 0.0,
        "identity_tolerance": 1e-6,
        "identity_pass": True,
        "paired_bootstrap_95_ci": [0.0, 0.0],
        "bootstrap_equivalence_margin": 0.01,
        "bootstrap_pass": True,
        "pass": True,
    }
    module = {
        "status": "COMPARABLE_NONZERO",
        "parameter_tensors": 1,
        "reference_norm": 1.0,
        "proposed_norm": 1.0,
        "cosine": 1.0,
        "norm_ratio": 1.0,
        "pass": True,
    }
    conformance = {
        "loss_reference": 1.0,
        "loss_production": 1.0,
        "loss_absolute_difference": 0.0,
        "loss_tolerance": 2e-6,
        "flattened_gradient": {
            "status": "COMPARABLE_NONZERO",
            "cosine": 1.0,
            "relative_l2": 0.0,
        },
        "production_call_telemetry": {
            "unique_chunks": 1,
            "group_rows": 32768,
            "encoder_forward_calls": 1,
            "decoder_forward_calls": 4,
            "global_loss_calls": 1,
            "backward_calls": 1,
        },
        "pass": True,
        "nnpu_branch_identical": True,
        "all_parameter_gradient_tolerances_pass": True,
        "flattened_gradient_cosine_pass": True,
        "production_call_telemetry_pass": True,
        "maximum_parameter_gradient_abs_difference": 0.0,
        "parameter_tensors_checked": 1,
        "failing_parameters": [],
    }
    return {
        "theta": theta,
        "pass": True,
        "scientific_fail_reasons": [],
        "branch_gate": {
            "evaluations": 58,
            "active_count": 58,
            "inactive_count": 0,
            "constant_across_all_58": True,
            "paired_disagreement_count": 0,
            "pass": True,
        },
        "components": {
            name: copy.deepcopy(component)
            for name in (
                "membership_risk",
                "direction_loss",
                "shrinkage_penalty",
                "total_objective",
            )
        },
        "complete_mean_gradient": {
            "status": "COMPARABLE_NONZERO",
            "relative_l2": 0.0,
            "cosine": 1.0,
            "norm_ratio": 1.0,
            "pass": True,
        },
        "module_mean_gradients": {
            name: copy.deepcopy(module)
            for name in ("encoder", "residual_map_output", "gate", "direction_head")
        },
        "reference_mean_gradient_sha256": "a" * 64,
        "proposed_mean_gradient_sha256": "b" * 64,
        "model_state_sha256": "c" * 64,
        "dropout_disabled": True,
        "optimizer_steps": 0,
        "reference_call_counts": {
            "encoder_forward": 116,
            "decoder_forward": 116,
            "global_loss": 29,
            "backward": 29,
        },
        "proposed_call_counts": {
            "encoder_forward": 29,
            "decoder_forward": 116,
            "global_loss": 29,
            "backward": 29,
        },
        "production_shared_api_conformance": conformance,
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["branch_gate"].update({"constant_across_all_58": False}), "BRANCH"),
        (lambda value: value["components"]["membership_risk"].update({"proposed_mean": 1.1}), "SCALAR"),
        (lambda value: value["components"]["membership_risk"].update({"paired_bootstrap_95_ci": [-0.02, 0.0]}), "BOOTSTRAP"),
        (lambda value: value["complete_mean_gradient"].update({"relative_l2": 0.01}), "COMPLETE_GRADIENT"),
        (lambda value: value["module_mean_gradients"]["encoder"].update({"cosine": 0.9}), "MODULE_GRADIENT"),
    ],
)
def test_formal_bridge_recomputes_each_paid_scientific_gate(mutation, message) -> None:
    passing = _passing_theta("theta_0")
    observed = auth._validate_theta_result(passing, "theta_0")
    assert all(observed.values())
    failing = _passing_theta("theta_0")
    mutation(failing)
    with pytest.raises(auth.R2AuthorizationError, match=message):
        auth._validate_theta_result(failing, "theta_0")


def test_formal_bridge_autostop_binds_the_exact_terminal_receipt(tmp_path: Path) -> None:
    created = datetime(2026, 9, 1, 6, 0, tzinfo=timezone.utc)
    static = {"sha256": "a" * 64, "created_at": created}
    jit = {"sha256": "b" * 64}
    terminal = {"sha256": "c" * 64}
    receipt = tmp_path / "ORACLE_AUTOSTOP.json"
    payload = {
        "format": auth.AUTOSTOP_FORMAT,
        "status": auth.AUTOSTOP_STATUS,
        "namespace": auth.ORACLE_NAMESPACE,
        "instance_id": auth.INSTANCE_ID,
        "job_id": "oracle-job",
        "stop_reason": "ORACLE_PASS_OBSERVED_IMMEDIATE_STOP",
        "observed_state": "Stopped",
        "gpu_billing_active": False,
        "comparison_only": True,
        "formal_training_authorized": False,
        "static_auth_sha256": static["sha256"],
        "jit_budget_receipt_sha256": jit["sha256"],
        "terminal_receipt_status": auth.ORACLE_PASS_STATUS,
        "terminal_receipt_sha256": terminal["sha256"],
        "terminal_receipt_validated_after_stop": True,
        "supervisor_elapsed_seconds": 10.0,
        "stopped_at": "2026-09-01T06:01:00+00:00",
        "stop_receipt_written_after_provider_confirmation": True,
    }
    _write_json(receipt, payload)
    result = auth._validate_oracle_autostop(
        receipt,
        _sha(receipt),
        oracle_static=static,
        jit=jit,
        terminal=terminal,
    )
    assert result["sha256"] == _sha(receipt)
    payload["terminal_receipt_sha256"] = "d" * 64
    _write_json(receipt, payload)
    with pytest.raises(auth.R2AuthorizationError, match="terminal_receipt"):
        auth._validate_oracle_autostop(
            receipt,
            _sha(receipt),
            oracle_static=static,
            jit=jit,
            terminal=terminal,
        )


def test_r2_static_receipt_materializes_and_validates_without_rehashing_pt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _authorized_fixture(tmp_path, monkeypatch)
    _install_safe_gate_stubs(monkeypatch)
    real_sha = auth._sha256_regular

    def reject_fold_rehash(
        path: Path, label: str, *, allow_empty: bool = False
    ) -> tuple[str, int]:
        assert path.suffix != ".pt", f"55GB fold was re-hashed: {path}"
        return real_sha(path, label, allow_empty=allow_empty)

    monkeypatch.setattr(auth, "_sha256_regular", reject_fold_rehash)
    payload = auth.materialize_static_authorization(
        receipt_path=fixture["receipt"],
        config_path=fixture["config"],
        code_root=fixture["code"],
        prepared_parent=fixture["prepared"],
        run_parent=fixture["run_parent"],
        bootstrap_root=fixture["bootstrap"],
    )
    assert payload["status"] == "STATIC_AUTH_READY"
    assert payload["prepared_inputs_retransferred"] is False
    assert payload["formal_result_artifacts_used"] is False
    expected_prerequisites = payload["prerequisite_hashes"]
    assert payload["fresh_authority"]["prerequisite_hashes"] == expected_prerequisites
    for variant in auth.VARIANTS:
        approval_path = (
            fixture["bootstrap"]
            / "fresh_authority"
            / variant
            / "TRAINING_APPROVAL.json"
        )
        approval = json.loads(approval_path.read_text(encoding="utf-8"))
        assert approval["prerequisite_hashes"] == expected_prerequisites
    validated = auth.validate_static_authorization(
        receipt_path=fixture["receipt"],
        config_path=fixture["config"],
        code_root=fixture["code"],
        prepared_parent=fixture["prepared"],
        run_parent=fixture["run_parent"],
        bootstrap_root=fixture["bootstrap"],
    )
    assert validated == payload


def test_r2_static_validation_rejects_hardlinked_fold_without_rehashing_pt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _authorized_fixture(tmp_path, monkeypatch)
    _install_safe_gate_stubs(monkeypatch)
    auth.materialize_static_authorization(
        receipt_path=fixture["receipt"],
        config_path=fixture["config"],
        code_root=fixture["code"],
        prepared_parent=fixture["prepared"],
        run_parent=fixture["run_parent"],
        bootstrap_root=fixture["bootstrap"],
    )
    fold = fixture["prepared"] / "G0/PATIENT_FOLD_0.pt"
    original = fold.with_name("PATIENT_FOLD_0.original.pt")
    fold.replace(original)
    try:
        fold.hardlink_to(original)
    except OSError:
        original.replace(fold)
        pytest.skip("hardlink creation unavailable")
    with pytest.raises(auth.R2AuthorizationError, match="HARDLINK_FORBIDDEN"):
        auth.validate_static_authorization(
            receipt_path=fixture["receipt"],
            config_path=fixture["config"],
            code_root=fixture["code"],
            prepared_parent=fixture["prepared"],
            run_parent=fixture["run_parent"],
            bootstrap_root=fixture["bootstrap"],
        )


def test_r2_static_receipt_commit_never_overwrites_a_racing_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _authorized_fixture(tmp_path, monkeypatch)
    _install_safe_gate_stubs(monkeypatch)
    real_link = auth.os.link
    injected = False

    def inject_target_before_commit(
        source: object, target: object, *, follow_symlinks: bool = True
    ) -> None:
        nonlocal injected
        if Path(target) == fixture["receipt"] and not injected:
            injected = True
            fixture["receipt"].write_bytes(b"racing-target-must-survive\n")
        real_link(source, target, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(auth.os, "link", inject_target_before_commit)
    with pytest.raises(auth.R2AuthorizationError, match="EXCLUSIVE_COMMIT_FAILED"):
        auth.materialize_static_authorization(
            receipt_path=fixture["receipt"],
            config_path=fixture["config"],
            code_root=fixture["code"],
            prepared_parent=fixture["prepared"],
            run_parent=fixture["run_parent"],
            bootstrap_root=fixture["bootstrap"],
        )
    assert injected is True
    assert fixture["receipt"].read_bytes() == b"racing-target-must-survive\n"


def test_r2_static_receipt_rejects_any_r1_formal_result_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _authorized_fixture(tmp_path, monkeypatch)
    _install_safe_gate_stubs(monkeypatch)
    auth.materialize_static_authorization(
        receipt_path=fixture["receipt"],
        config_path=fixture["config"],
        code_root=fixture["code"],
        prepared_parent=fixture["prepared"],
        run_parent=fixture["run_parent"],
        bootstrap_root=fixture["bootstrap"],
    )
    payload = json.loads(fixture["receipt"].read_text(encoding="utf-8"))
    payload["forbidden_result"] = f"/results/{auth.R1_RESULT_NAMESPACE}/G0"
    _write_json(fixture["receipt"], payload)
    with pytest.raises(auth.R2AuthorizationError, match="REFERENCES_R1_FORMAL_RESULTS"):
        auth.validate_static_authorization(
            receipt_path=fixture["receipt"],
            config_path=fixture["config"],
            code_root=fixture["code"],
            prepared_parent=fixture["prepared"],
            run_parent=fixture["run_parent"],
            bootstrap_root=fixture["bootstrap"],
        )


@pytest.mark.parametrize(
    "swapped_trainer",
    [auth.ORACLE_TRAINER, "attacker.module:run", None],
)
def test_r2_fresh_approval_rejects_trainer_swap_before_training_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    swapped_trainer: str | None,
) -> None:
    fixture = _authorized_fixture(tmp_path, monkeypatch)
    _install_safe_gate_stubs(monkeypatch)
    auth.materialize_static_authorization(
        receipt_path=fixture["receipt"],
        config_path=fixture["config"],
        code_root=fixture["code"],
        prepared_parent=fixture["prepared"],
        run_parent=fixture["run_parent"],
        bootstrap_root=fixture["bootstrap"],
    )
    approval_path = (
        fixture["bootstrap"] / "fresh_authority/G2/TRAINING_APPROVAL.json"
    )
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    approval["authorized_trainer"] = swapped_trainer
    _write_json(approval_path, approval)
    with pytest.raises(auth.R2AuthorizationError, match="AUTHORIZED_TRAINER_DRIFT"):
        auth.validate_static_authorization(
            receipt_path=fixture["receipt"],
            config_path=fixture["config"],
            code_root=fixture["code"],
            prepared_parent=fixture["prepared"],
            run_parent=fixture["run_parent"],
            bootstrap_root=fixture["bootstrap"],
        )


def test_r2_first_launch_rejects_preexisting_output_or_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _authorized_fixture(tmp_path, monkeypatch)
    _install_safe_gate_stubs(monkeypatch)
    checkpoint = fixture["run_parent"] / "G0/training/checkpoint.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"must-not-be-reused")
    with pytest.raises(auth.R2AuthorizationError, match="PREEXISTS_FRESH_ONLY"):
        auth.materialize_static_authorization(
            receipt_path=fixture["receipt"],
            config_path=fixture["config"],
            code_root=fixture["code"],
            prepared_parent=fixture["prepared"],
            run_parent=fixture["run_parent"],
            bootstrap_root=fixture["bootstrap"],
        )
    assert not fixture["receipt"].exists()
    assert not (fixture["bootstrap"] / "fresh_authority").exists()


def test_r2_resume_requires_exact_same_static_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _authorized_fixture(tmp_path, monkeypatch)
    _install_safe_gate_stubs(monkeypatch)
    payload = auth.materialize_static_authorization(
        receipt_path=fixture["receipt"],
        config_path=fixture["config"],
        code_root=fixture["code"],
        prepared_parent=fixture["prepared"],
        run_parent=fixture["run_parent"],
        bootstrap_root=fixture["bootstrap"],
    )
    fixture["run_parent"].mkdir(parents=True)
    lineage_path = fixture["run_parent"] / "RUN_LINEAGE.json"
    lineage = {
        "format": "CC_HHGT_V3_2_G012_R2_RUN_LINEAGE_V1",
        "status": "R2_RUN_LINEAGE_ACTIVE",
        "namespace": auth.NAMESPACE,
        "run_parent": str(fixture["run_parent"].resolve()),
        "resume_authorized": True,
        "static_auth_ready_sha256": _sha(fixture["receipt"]),
        "config_sha256": payload["config_sha256"],
        "code_tree_sha256": payload["code_tree_sha256"],
        "input_reuse_ready_sha256": payload["input_reuse_ready_sha256"],
        "launcher_sha256": payload["launcher_sha256"],
        "supervisor_sha256": payload["supervisor_sha256"],
        "finalizer_sha256": payload["finalizer_sha256"],
        "fresh_authority_sha256": payload["fresh_authority_sha256"],
        "authorized_trainer": auth.FORMAL_TRAINER,
        "formal_result_artifacts_reused": False,
    }
    _write_json(lineage_path, lineage)
    assert auth.validate_static_authorization(
        receipt_path=fixture["receipt"],
        config_path=fixture["config"],
        code_root=fixture["code"],
        prepared_parent=fixture["prepared"],
        run_parent=fixture["run_parent"],
        bootstrap_root=fixture["bootstrap"],
        allow_resume=True,
    ) == payload
    lineage["static_auth_ready_sha256"] = "f" * 64
    _write_json(lineage_path, lineage)
    with pytest.raises(auth.R2AuthorizationError, match="RESUME_LINEAGE_DRIFT"):
        auth.validate_static_authorization(
            receipt_path=fixture["receipt"],
            config_path=fixture["config"],
            code_root=fixture["code"],
            prepared_parent=fixture["prepared"],
            run_parent=fixture["run_parent"],
            bootstrap_root=fixture["bootstrap"],
            allow_resume=True,
        )
    lineage["static_auth_ready_sha256"] = _sha(fixture["receipt"])
    lineage["unexpected_authority"] = True
    _write_json(lineage_path, lineage)
    with pytest.raises(auth.R2AuthorizationError, match="RESUME_LINEAGE_SCHEMA_DRIFT"):
        auth.validate_static_authorization(
            receipt_path=fixture["receipt"],
            config_path=fixture["config"],
            code_root=fixture["code"],
            prepared_parent=fixture["prepared"],
            run_parent=fixture["run_parent"],
            bootstrap_root=fixture["bootstrap"],
            allow_resume=True,
        )


def test_static_patch_recheck_rejects_unmanifested_root_startup_hook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _authorized_fixture(tmp_path, monkeypatch)
    config_payload = yaml.safe_load(fixture["config"].read_text(encoding="utf-8"))
    (fixture["code"] / "sitecustomize.py").write_text(
        "raise SystemExit(99)\n", encoding="utf-8"
    )
    with pytest.raises(auth.R2AuthorizationError, match="PREFIX_FORBIDDEN"):
        auth._validate_patch_ready(
            fixture["bootstrap"] / "PATCH_READY.json",
            fixture["code"],
            config_payload,
        )


def test_r2_shell_templates_are_isolated_and_fail_closed() -> None:
    for path in SCRIPTS:
        source = path.read_text(encoding="utf-8")
        assert auth.NAMESPACE in source
        assert auth.R1_RESULT_NAMESPACE not in source
        assert "require_escalated" not in source
        assert "scp " not in source
        assert "rsync " not in source
        assert "curl " not in source
        assert "wget " not in source

    launcher = SCRIPTS[-1].read_text(encoding="utf-8")
    assert launcher.index("--config-only") < launcher.index(
        "R2_PAID_LAUNCH_BODY_NOT_YET_LOCKED"
    )
    assert "nvidia-smi" not in launcher
    assert "run-shard" not in launcher
    assert "\nmkdir " not in launcher

    patcher = SCRIPTS[0].read_text(encoding="utf-8")
    assert "code_tree_sha256" in patcher
    assert "deployment_contract_sha256" in patcher
    assert "O_NOFOLLOW" in patcher
    assert "os.fstat" in patcher
    assert "os.read" in patcher
    assert "read_text" not in patcher
    assert "importlib" not in patcher
    assert "PYTHONPATH" not in patcher
    assert "run_verified_verifier()" in patcher
    assert 'exec(compile(bytes(raw), str(path), "exec")' in patcher
    assert '"$python_bin" "$verifier"' not in patcher
    assert "remove_exact_staging_recovery_tree" in patcher
    assert "R2_PROMOTION_RECOVERY_HAS_NO_CODE_OR_BASELINE" in patcher
    assert "code.invalid_recovery.$$.${RANDOM}" in patcher
    assert "R2_PATCH_READY_STALE_PARTIAL_UNSAFE" in patcher

    authorizer = SCRIPTS[1].read_text(encoding="utf-8")
    assert authorizer.index("--config-only") < authorizer.index("nvidia-smi")
    assert "INPUT_REUSE_READY.json" in authorizer
    assert "ABORTED.json" in authorizer
    finalizer = SCRIPTS[2].read_text(encoding="utf-8")
    assert finalizer.index("--config-only") < finalizer.index("nvidia-smi")
    assert "R2_CPU_FINALIZER_BODY_NOT_YET_LOCKED" in finalizer


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is unavailable")
def test_r2_shell_templates_parse() -> None:
    subprocess.run(
        ["bash", "-n", *(path.relative_to(ROOT).as_posix() for path in SCRIPTS)],
        check=True,
        cwd=ROOT,
    )
