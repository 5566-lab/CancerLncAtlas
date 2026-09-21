from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from scripts import validate_v32_group_shared_oracle_static_auth_ready_r2 as gate


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / gate.CONFIG_RELATIVE
TASK_TEMPLATE = ROOT / gate.TASK_TEMPLATE_RELATIVE
VALIDATOR = ROOT / "scripts" / gate.VALIDATOR_NAME
LAUNCHER = ROOT / "scripts" / gate.LAUNCHER_NAME
AUTHORIZER = ROOT / "scripts" / gate.AUTHORIZER_NAME
SUPERVISOR = ROOT / "scripts" / gate.SUPERVISOR_NAME
CONTROLLER = ROOT / "scripts" / gate.DISPATCH_CONTROLLER_NAME
LOG_GUARD = ROOT / "scripts" / gate.LOG_GUARD_NAME
EXTERNAL_MANIFEST = (
    ROOT
    / "docs"
    / "v32_group_shared_oracle_external_deployment_manifest_20260901_r2.json"
)


def _copy_external_control_fixture(project: Path) -> Path:
    control = project / "runtime/oracle_gates" / gate.NAMESPACE
    control.mkdir(parents=True, exist_ok=True)
    sources = {
        gate.VALIDATOR_NAME: VALIDATOR,
        gate.LAUNCHER_NAME: LAUNCHER,
        gate.AUTHORIZER_NAME: AUTHORIZER,
        gate.SUPERVISOR_NAME: SUPERVISOR,
        gate.DISPATCH_CONTROLLER_NAME: CONTROLLER,
        gate.LOG_GUARD_NAME: LOG_GUARD,
    }
    for name, source in sources.items():
        shutil.copy2(source, control / name)
    destination = control / gate.EXTERNAL_DEPLOYMENT_MANIFEST_NAME
    shutil.copy2(EXTERNAL_MANIFEST, destination)
    return destination


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_code_fixture(destination: Path, prepared_parent: Path) -> None:
    def ignored(_directory: str, names: list[str]) -> set[str]:
        return {
            name
            for name in names
            if name in {"__pycache__", ".pytest_cache"}
            or name.endswith((".pyc", ".pyo"))
        }

    shutil.copytree(ROOT / "cc_hhgt", destination / "cc_hhgt", ignore=ignored)
    for relative in (
        "scripts/v32_pipeline.py",
        gate.PREREG_RELATIVE,
        gate.DECISION_RELATIVE,
        gate.TASK_TEMPLATE_RELATIVE,
    ):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["training_io"]["prepared_fold_pattern"] = str(
        prepared_parent.resolve() / "G2" / "PATIENT_FOLD_{fold}.pt"
    )
    target_config = destination / gate.CONFIG_RELATIVE
    target_config.parent.mkdir(parents=True, exist_ok=True)
    target_config.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )


def _lineage_fixture(
    project: Path,
    prepared: Path,
    returned_static: Path,
    source_input: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    variants: dict[str, object] = {}
    reuse_records = []
    total = 0
    g2f0_sha = ""
    g2f0_size = 0
    for variant in ("G0", "G1", "G2"):
        fold_inputs = []
        for fold in range(5):
            path = prepared / variant / f"PATIENT_FOLD_{fold}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((f"fixture:{variant}:{fold}:" * (fold + 1)).encode())
            digest = hashlib.sha256(f"authority:{variant}:{fold}".encode()).hexdigest()
            size = path.stat().st_size
            row = {
                "fold": fold,
                "path": str(path.resolve()),
                "sha256": digest,
                "size_bytes": size,
            }
            fold_inputs.append(row)
            reuse_records.append({"variant": variant, **row})
            total += size
            if (variant, fold) == ("G2", 0):
                g2f0_sha, g2f0_size = digest, size
        variants[variant] = {"fold_inputs": fold_inputs}
    r1_static_payload = {
        "status": "STATIC_AUTH_READY",
        "gpu_visible": False,
        "variants": variants,
    }
    r1_bootstrap = project / "runtime/bootstrap" / gate.R1_NAMESPACE
    archived = (
        r1_bootstrap
        / gate.R1_ARCHIVE_NAMESPACE
        / "STATIC_AUTH_READY.live_at_abort.json"
    )
    _write_json(archived, r1_static_payload)
    returned_static.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(archived, returned_static)
    anchored_sha = _sha(archived)
    monkeypatch.setattr(gate, "EXPECTED_R1_STATIC_AUTH_R6_SHA256", anchored_sha)
    monkeypatch.setattr(gate, "EXPECTED_G2_F0_SHA256", g2f0_sha)
    monkeypatch.setattr(gate, "EXPECTED_G2_F0_SIZE_BYTES", g2f0_size)
    _write_json(
        r1_bootstrap / "ABORTED.json",
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
                    "archive_path": str(archived.resolve()),
                    "sha256": anchored_sha,
                    "size_bytes": archived.stat().st_size,
                }
            },
        },
    )
    _write_json(
        source_input,
        {
            "format": gate.INPUT_FORMAT,
            "status": gate.INPUT_STATUS,
            "source_input_archive_sha256": gate.SOURCE_INPUT_ARCHIVE_SHA256,
            "prepared_parent": str(prepared.resolve()),
            "fold_artifact_count": 15,
            "fold_artifact_total_bytes": total,
            "fold_artifacts": reuse_records,
            "reused_existing_prepared_inputs": True,
            "retransfer_performed": False,
            "copy_performed": False,
            "extraction_performed": False,
            "source_files_modified": False,
            "formal_result_artifacts_used": False,
        },
    )
    return {"anchored_sha": anchored_sha, "g2f0_sha": g2f0_sha}


def _full_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Path | datetime]:
    project = tmp_path / "DSC/CancerLncAtlas"
    code = project / "runtime/tools" / gate.NAMESPACE / "code"
    bootstrap = project / "runtime/bootstrap" / gate.NAMESPACE
    prepared = project / "inputs/v32_g012_patient_first_20260830_r1/prepared"
    returned = (
        project
        / "runtime/oracle_gates"
        / gate.NAMESPACE
        / "authority/STATIC_AUTH_READY.v32_g012_20260901_r6.json"
    )
    source_input = (
        project
        / "runtime/bootstrap"
        / gate.FORMAL_R2_NAMESPACE
        / "INPUT_REUSE_READY.json"
    )
    _lineage_fixture(project, prepared, returned, source_input, monkeypatch)
    _copy_code_fixture(code, prepared)
    external_manifest = _copy_external_control_fixture(project)
    bootstrap.mkdir(parents=True, exist_ok=True)
    archive = bootstrap / "ORACLE_CODE_BUNDLE.tar.gz"
    archive.write_bytes(b"fixture-independent-oracle-code-archive\n")
    code_receipt = bootstrap / "CODE_ARCHIVE_READY.json"
    gate.materialize_code_archive_receipt(
        code_root=code,
        bootstrap_root=bootstrap,
        code_archive=archive,
        output=code_receipt,
    )
    now = datetime(2026, 9, 1, 6, 0, tzinfo=timezone.utc)
    jit = bootstrap / "JIT_BUDGET_READY.json"
    _write_json(
        jit,
        {
            "format": gate.JIT_FORMAT,
            "status": gate.JIT_STATUS,
            "currency": "CNY",
            "source": "COMPSHARE_API_BILLING_SNAPSHOT",
            "project_budget_cap_cny": gate.PROJECT_BUDGET_CAP_CNY,
            "compute_hourly_cny": gate.COMPUTE_HOURLY_CNY,
            "gpu_hourly_cny": gate.GPU_HOURLY_CNY,
            "disk_hourly_cny": gate.DISK_HOURLY_CNY,
            "total_hourly_cny": gate.TOTAL_HOURLY_CNY,
            "profile_name_sha256": gate.EXPECTED_PROFILE_NAME_SHA256,
            "project_scope_sha256": gate.EXPECTED_PROJECT_SCOPE_SHA256,
            "instance_id": "uhost-fixture4090",
            "queried_at": (now - timedelta(minutes=1)).isoformat(),
            "valid_until": (now + timedelta(minutes=44)).isoformat(),
            "conservative_remaining_cny": 5.25,
        },
    )
    return {
        "project": project,
        "code": code,
        "bootstrap": bootstrap,
        "prepared": prepared,
        "returned": returned,
        "source_input": source_input,
        "config": code / gate.CONFIG_RELATIVE,
        "task_template": code / gate.TASK_TEMPLATE_RELATIVE,
        "code_receipt": code_receipt,
        "jit": jit,
        "external_manifest": external_manifest,
        "now": now,
    }


def test_checked_in_contract_is_one_comparison_with_hard_caps_and_no_output() -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    control = payload["execution_control"]
    oracle = payload["oracle_contract"]
    runtime = payload["runtime_profile"]
    assert control["max_paid_hours"] == 3.0
    assert control["max_cost_cny"] == 8.0
    assert control["comparison_only"] is True
    assert control["formal_training_authorized"] is False
    assert "output_root" not in payload["training_io"]
    assert "/results/" not in CONFIG.read_text(encoding="utf-8")
    assert oracle["trainer"] == gate.TRAINER
    assert (oracle["graph_variant"], oracle["patient_fold"], oracle["seed"]) == (
        "G2",
        0,
        20260726,
    )
    assert oracle["equal_step_g0_g1_g2_pilot_still_required"] is True
    assert oracle["external_verifier_sha256"] == _sha(VALIDATOR)
    assert oracle["external_deployment_manifest_sha256"] == _sha(EXTERNAL_MANIFEST)
    assert runtime["supervisor_static_auth_line_seconds"] == 180
    assert runtime["supervisor_initial_heartbeat_minutes"] == 15
    assert runtime["supervisor_assignment_stall_minutes"] == 5
    assert runtime["supervisor_pre_submit_status_grace_seconds"] == 720
    assert runtime["supervisor_post_submit_status_grace_seconds"] == 180
    result = gate.validate_config(CONFIG, code_root=ROOT)
    assert result["oracle"]["runner_sha256"] == _sha(ROOT / gate.RUNNER_RELATIVE)
    assert result["oracle"]["preregistration_sha256"] == _sha(
        ROOT / gate.PREREG_RELATIVE
    )
    assert result["oracle"]["decision_sha256"] == _sha(
        ROOT / gate.DECISION_RELATIVE
    )


def test_task_template_is_exactly_g2_fold0_seed20260726() -> None:
    rows = gate._task_rows(TASK_TEMPLATE.read_bytes())
    assert len(rows) == 1
    assert rows[0]["task_id"] == gate.TASK_ID
    assert rows[0]["patient_fold"] == "0"
    assert rows[0]["seed"] == "20260726"


def test_lineage_crosschecks_all_15_without_reading_pt_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "DSC/CancerLncAtlas"
    prepared = project / "inputs/prepared"
    returned = project / "control/r1-r6.json"
    source = project / "runtime/bootstrap/formal-r2/INPUT_REUSE_READY.json"
    expected = _lineage_fixture(project, prepared, returned, source, monkeypatch)
    real_reader = gate._read_regular

    def reject_pt_read(path: Path, label: str):
        assert path.suffix != ".pt", f"55GB payload was read: {path}"
        return real_reader(path, label)

    monkeypatch.setattr(gate, "_read_regular", reject_pt_read)
    result = gate._validate_r1_and_inputs(
        project_root=project,
        prepared_parent=prepared,
        source_input_receipt=source,
        returned_r1_static_receipt=returned,
    )
    assert result["r1_static_auth_r6_sha256"] == expected["anchored_sha"]
    assert result["g2f0"]["sha256"] == expected["g2f0_sha"]
    assert len(result["fold_records"]) == 15


def test_any_one_of_15_input_hashes_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "DSC/CancerLncAtlas"
    prepared = project / "inputs/prepared"
    returned = project / "control/r1-r6.json"
    source = project / "runtime/bootstrap/formal-r2/INPUT_REUSE_READY.json"
    _lineage_fixture(project, prepared, returned, source, monkeypatch)
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["fold_artifacts"][11]["sha256"] = "f" * 64
    _write_json(source, payload)
    with pytest.raises(gate.OracleGateError, match="INPUT_SHA_CROSSCHECK_DRIFT"):
        gate._validate_r1_and_inputs(
            project_root=project,
            prepared_parent=prepared,
            source_input_receipt=source,
            returned_r1_static_receipt=returned,
        )


def test_static_gate_materializes_one_hash_bound_comparison_only_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _full_fixture(tmp_path, monkeypatch)
    receipt = gate.materialize_static_authorization(
        project_root=fixture["project"],
        code_root=fixture["code"],
        bootstrap_root=fixture["bootstrap"],
        prepared_parent=fixture["prepared"],
        config_path=fixture["config"],
        task_template_path=fixture["task_template"],
        source_input_receipt=fixture["source_input"],
        returned_r1_static_receipt=fixture["returned"],
        jit_budget_path=fixture["jit"],
        code_archive_receipt=fixture["code_receipt"],
        external_deployment_manifest=fixture["external_manifest"],
        externally_observed_verifier_sha256=_sha(VALIDATOR),
        now=fixture["now"],
    )
    assert receipt["status"] == gate.STATIC_STATUS
    assert receipt["jit_effective_cost_cap_cny"] == 5.25
    assert receipt["max_paid_hours"] == 3.0
    assert receipt["local_paid_api_minimum_ttl_seconds"] == 1800
    assert receipt["remote_paid_launch_minimum_ttl_seconds"] == 900
    assert receipt["formal_training_authorized"] is False
    assert receipt["r1_static_auth_r6_sha256"] == gate.EXPECTED_R1_STATIC_AUTH_R6_SHA256
    assert "/results/" not in json.dumps(receipt)
    authorization = fixture["bootstrap"] / "authorization"
    approval = json.loads((authorization / "ORACLE_APPROVAL.json").read_text())
    assert approval["approved_task_ids"] == [gate.TASK_ID]
    assert approval["authorized_trainer"] == gate.TRAINER
    assert approval["formal_training_authorized"] is False
    assert approval["max_paid_hours"] == 3.0
    assert approval["max_cost_cny"] == 8.0
    validated = gate.validate_static_authorization(
        project_root=fixture["project"],
        code_root=fixture["code"],
        bootstrap_root=fixture["bootstrap"],
        prepared_parent=fixture["prepared"],
        config_path=fixture["config"],
        task_template_path=fixture["task_template"],
        source_input_receipt=fixture["source_input"],
        returned_r1_static_receipt=fixture["returned"],
        jit_budget_path=fixture["jit"],
        code_archive_receipt=fixture["code_receipt"],
        external_deployment_manifest=fixture["external_manifest"],
        externally_observed_verifier_sha256=_sha(VALIDATOR),
        now=fixture["now"],
    )
    assert validated == receipt


def test_static_gate_rejects_expired_jit_before_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _full_fixture(tmp_path, monkeypatch)
    with pytest.raises(gate.OracleGateError, match="JIT_BUDGET_STALE"):
        gate.materialize_static_authorization(
            project_root=fixture["project"],
            code_root=fixture["code"],
            bootstrap_root=fixture["bootstrap"],
            prepared_parent=fixture["prepared"],
            config_path=fixture["config"],
            task_template_path=fixture["task_template"],
            source_input_receipt=fixture["source_input"],
            returned_r1_static_receipt=fixture["returned"],
            jit_budget_path=fixture["jit"],
            code_archive_receipt=fixture["code_receipt"],
            external_deployment_manifest=fixture["external_manifest"],
            externally_observed_verifier_sha256=_sha(VALIDATOR),
            now=fixture["now"] + timedelta(hours=1),
        )


def test_paid_launch_ttl_boundary_and_static_valid_until_are_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _full_fixture(tmp_path, monkeypatch)
    now = fixture["now"]
    jit_path = fixture["jit"]
    payload = json.loads(jit_path.read_text(encoding="utf-8"))
    payload["queried_at"] = (now - timedelta(minutes=20)).isoformat()
    payload["valid_until"] = (now + timedelta(seconds=900)).isoformat()
    _write_json(jit_path, payload)
    accepted = gate._validate_jit_budget(jit_path, now=now)
    assert accepted["launch_ttl_seconds"] == 900

    static_path = fixture["bootstrap"] / "STATIC_AUTH_READY.json"
    _write_json(
        static_path,
        {
            "format": gate.STATIC_FORMAT,
            "status": gate.STATIC_STATUS,
            "namespace": gate.NAMESPACE,
            "comparison_only": True,
            "formal_training_authorized": False,
            "instance_id": payload["instance_id"],
            "jit_budget_receipt_sha256": _sha(jit_path),
            "jit_valid_until": payload["valid_until"],
            "local_paid_api_minimum_ttl_seconds": 1800,
            "remote_paid_launch_minimum_ttl_seconds": 900,
        },
    )
    ready = gate.validate_paid_launch_readiness(
        jit_budget_path=jit_path,
        static_auth_path=static_path,
        now=now,
    )
    assert ready["minimum_launch_ttl_seconds"] == 900
    assert ready["local_paid_api_minimum_ttl_seconds"] == 1800
    assert ready["remote_paid_launch_minimum_ttl_seconds"] == 900

    payload["valid_until"] = (now + timedelta(seconds=899)).isoformat()
    _write_json(jit_path, payload)
    with pytest.raises(gate.OracleGateError, match="MINIMUM_PAID_LAUNCH_TTL"):
        gate._validate_jit_budget(jit_path, now=now)

    payload["valid_until"] = (now + timedelta(seconds=900)).isoformat()
    _write_json(jit_path, payload)
    static_payload = json.loads(static_path.read_text(encoding="utf-8"))
    static_payload["jit_budget_receipt_sha256"] = _sha(jit_path)
    static_payload["jit_valid_until"] = (now + timedelta(seconds=901)).isoformat()
    _write_json(static_path, static_payload)
    with pytest.raises(gate.OracleGateError, match="STATIC_JIT_BINDING_DRIFT=jit_valid_until"):
        gate.validate_paid_launch_readiness(
            jit_budget_path=jit_path,
            static_auth_path=static_path,
            now=now,
        )


@pytest.mark.parametrize(
    "artifact_name",
    [
        gate.VALIDATOR_NAME,
        gate.LAUNCHER_NAME,
        gate.AUTHORIZER_NAME,
        gate.SUPERVISOR_NAME,
        gate.DISPATCH_CONTROLLER_NAME,
        gate.LOG_GUARD_NAME,
    ],
)
def test_external_control_tamper_is_rejected_before_static_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_name: str,
) -> None:
    fixture = _full_fixture(tmp_path, monkeypatch)
    control_file = (
        fixture["project"]
        / "runtime/oracle_gates"
        / gate.NAMESPACE
        / artifact_name
    )
    control_file.write_bytes(control_file.read_bytes() + b"\n# malicious replacement\n")
    with pytest.raises(gate.OracleGateError, match="EXTERNAL_CONTROL_ARTIFACT_DRIFT"):
        gate.materialize_static_authorization(
            project_root=fixture["project"],
            code_root=fixture["code"],
            bootstrap_root=fixture["bootstrap"],
            prepared_parent=fixture["prepared"],
            config_path=fixture["config"],
            task_template_path=fixture["task_template"],
            source_input_receipt=fixture["source_input"],
            returned_r1_static_receipt=fixture["returned"],
            jit_budget_path=fixture["jit"],
            code_archive_receipt=fixture["code_receipt"],
            external_deployment_manifest=fixture["external_manifest"],
            externally_observed_verifier_sha256=_sha(VALIDATOR),
            now=fixture["now"],
        )


def _terminal_receipt(status: str, scientific_pass: bool) -> dict[str, object]:
    return {
        "format": gate.TERMINAL_RECEIPT_FORMAT,
        "status": status,
        "run_id": gate.RUN_ID,
        "task_id": gate.TASK_ID,
        "graph_variant": gate.EXPECTED_VARIANT,
        "patient_fold": gate.EXPECTED_FOLD,
        "seed": gate.EXPECTED_SEED,
        "scientific_pass": scientific_pass,
        "formal_training_authorized": False,
        "formal_artifacts_written": 0,
        "checkpoint_written": False,
        "success_json_written": False,
        "failure_json_written": False,
        "prediction_written": False,
        "winner_selection_input": False,
    }


@pytest.mark.parametrize(
    ("status", "exit_code", "scientific_pass"),
    [
        (gate.PASS_STATUS, 0, True),
        ("SCIENTIFIC_FAIL_REAL_DATA_ORACLE_COMPARISON_ONLY", 42, False),
        ("RUNTIME_FAIL_REAL_DATA_ORACLE_COMPARISON_ONLY", 42, False),
    ],
)
def test_terminal_log_accepts_only_one_safe_terminal_receipt(
    tmp_path: Path, status: str, exit_code: int, scientific_pass: bool
) -> None:
    log = tmp_path / "oracle.jsonl"
    heartbeat = {
        "format": gate.TERMINAL_RECEIPT_FORMAT,
        "status": "ORACLE_COMPARISON_HEARTBEAT",
    }
    log.write_text(
        json.dumps(heartbeat)
        + "\n"
        + json.dumps(_terminal_receipt(status, scientific_pass))
        + "\n",
        encoding="utf-8",
    )
    assert gate.validate_terminal_log(log, exit_code)["status"] == status


def test_verifier_is_external_data_only_and_uses_same_fd_nofollow() -> None:
    source = VALIDATOR.read_text(encoding="utf-8")
    assert "O_NOFOLLOW" in source
    assert "os.fstat(descriptor)" in source
    assert "os.read(descriptor" in source
    assert "observed.st_nlink" in source
    assert "import cc_hhgt" not in source
    assert "importlib" not in source
    assert "exec(" not in source
    assert gate.EXPECTED_R1_STATIC_AUTH_R6_SHA256 in source
    assert gate.EXPECTED_G2_F0_SHA256 in source


def test_verifier_rejects_hardlinked_authority_input(tmp_path: Path) -> None:
    original = tmp_path / "authority.json"
    alias = tmp_path / "attacker-alias.json"
    original.write_text('{"ok":true}\n', encoding="utf-8")
    os.link(original, alias)
    with pytest.raises(gate.OracleGateError, match="HARDLINK_FORBIDDEN"):
        gate._read_regular(original, "MALICIOUS_HARDLINK")


def test_shell_gates_are_separate_fail_closed_and_cuda_after_static() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    authorizer = AUTHORIZER.read_text(encoding="utf-8")
    for source in (launcher, authorizer):
        assert "/runtime/oracle_gates/$namespace" in source
        assert "/runtime/tools/$namespace/code" in source
        assert "require_escalated" not in source
        assert "curl " not in source
        assert "wget " not in source
        assert "scp " not in source
        assert "rsync " not in source
        assert "v32_group_shared_oracle_paid_gpu_20260901_r2.lock" in source
        assert "assert_frozen_verifier" in source
        assert "sha256sum \"$verifier\"" in source
        assert "--external-verifier-sha256" in source
    assert launcher.index("--config-only") < launcher.index("flock -n 9")
    assert launcher.index("--code-archive-receipt") < launcher.index("nvidia-smi -L")
    assert launcher.index("--paid-launch-readiness") < launcher.index("nvidia-smi -L")
    assert "remote 900-second lower boundary" in launcher
    assert launcher.index("nvidia-smi -L") < launcher.index("-m cc_hhgt.v32.cli")
    assert gate.TRAINER in launcher
    assert gate.TASK_ID in launcher
    assert 'hard_timeout_seconds=10800' in launcher
    assert "/results/" not in launcher
    assert "output_root" not in launcher
    assert authorizer.index("--config-only") < authorizer.index("nvidia-smi -L")
    assert "STATIC_GATE_REFUSES_CUDA_VISIBLE_HOST" in authorizer
    assert "--materialize" in authorizer


def test_external_supervisor_stops_on_pass_exit42_and_three_hour_timeout() -> None:
    source = SUPERVISOR.read_text(encoding="utf-8")
    assert "$HardHourCap = 3.0" in source
    assert "$HardCostCapCny = 8.0" in source
    assert "[int]$InitialHeartbeatMinutes = 15" in source
    assert "[int]$HeartbeatStallMinutes = 5" in source
    assert "[int]$PreSubmitStatusFailureGraceSeconds = 720" in source
    assert "[int]$StatusFailureGraceSeconds = 180" in source
    assert "Get-DispatchPhase" in source
    assert "Test-FreshExpectedJob" in source
    assert "[string]$StopReceiptPath," in source
    assert "CANCERLNCATLAS_GROUP_SHARED_ORACLE_AUTOSTOP_V1" in source
    assert "status = 'INSTANCE_STOPPED_STATE_CONFIRMED'" in source
    assert "gpu_billing_active = $false" in source
    assert "terminal_receipt_validated_after_stop" in source
    assert "Read-OracleLogWindow" in source
    assert "$seenHeartbeatLineHashes" in source
    assert "$maxCompletedByPhase" in source
    assert "Write-StopReceiptBestEffort -Reason $Reason" in source
    assert "$effectiveCost = [Math]::Min" in source
    assert "$unwatchedSeconds = [Math]::Max" in source
    assert "- $unwatchedSeconds" in source
    assert "$static.jit_budget_receipt_sha256" in source
    assert "$FixedProfile = 'default'" in source
    assert "@('--profile', $FixedProfile, '--json')" in source
    assert "$MinimumPaidLaunchTtlSeconds = 1800.0" in source
    assert "$RemotePaidLaunchMinimumTtlSeconds = 900.0" in source
    assert "$static.local_paid_api_minimum_ttl_seconds" in source
    assert "$static.remote_paid_launch_minimum_ttl_seconds" in source
    assert "$static.jit_valid_until" in source
    assert "ORACLE_PASS_OBSERVED_IMMEDIATE_STOP" in source
    assert "ORACLE_PROCESS_EXIT42_IMMEDIATE_STOP" in source
    assert "ORACLE_3H_OR_JIT_BUDGET_TIMEOUT" in source
    assert "Stop-PaidInstance 'ORACLE_PASS_OBSERVED_IMMEDIATE_STOP'" in source
    assert "Stop-PaidInstance 'ORACLE_PROCESS_EXIT42_IMMEDIATE_STOP'" in source
    assert "Stop-PaidInstance 'ORACLE_3H_OR_JIT_BUDGET_TIMEOUT'" in source
    stop_start = source.index("function Stop-PaidInstance")
    stop_end = source.index("function Read-JsonReceipt", stop_start)
    stop_body = source[stop_start:stop_end]
    assert stop_body.index("Invoke-CompShareJson") < stop_body.index(
        "Write-WatchEventBestEffort"
    )


def test_dispatch_controller_is_one_click_fail_closed_and_provider_scheduled() -> None:
    source = CONTROLLER.read_text(encoding="utf-8")
    assert "$JobId = 'v32-oracle-g2f0-s20260726-r2'" in source
    assert "Assert-FrozenExternalControls" in source
    assert "Get-FileHash" in source
    assert "Start-SupervisorRunspace" in source
    assert "BeginInvoke()" in source
    assert "instance', 'schedule', 'set'" in source
    assert "instance', 'schedule', 'show'" in source
    assert "instance', 'start'" in source
    assert "instance', 'job', 'submit'" in source
    assert "--job-id', $JobId" in source
    assert "AddParameter('InitialHeartbeatMinutes', 15)" in source
    assert "AddParameter('HeartbeatStallMinutes', 5)" in source
    assert "AddParameter('PreSubmitStatusFailureGraceSeconds', 720)" in source
    assert "AddParameter('StatusFailureGraceSeconds', 180)" in source
    assert "Assert-FixedJobIdAbsent" in source
    assert "Get-FreshBoundJob" in source
    assert "$FixedProfile = 'default'" in source
    assert "@('--profile', $FixedProfile, '--json')" in source
    assert "$MinimumPaidLaunchTtlSeconds = 1800.0" in source
    assert "$RemotePaidLaunchMinimumTtlSeconds = 900.0" in source
    assert "$static.local_paid_api_minimum_ttl_seconds" in source
    assert "$static.remote_paid_launch_minimum_ttl_seconds" in source
    assert "Assert-ExactPrepaidStoppedProfile" in source
    assert "Assert-ExactPaidGpuRunningProfile" in source
    assert gate.EXPECTED_PROFILE_NAME_SHA256 in source
    assert gate.EXPECTED_PROJECT_SCOPE_SHA256 in source
    assert "finally {" in source
    assert "Stop-AndConfirmInstance" in source
    assert "Read-Host" not in source
    assert "require_escalated" not in source
    schedule_index = source.index("'instance', 'schedule', 'set'")
    start_index = source.index("'instance', 'start'")
    submit_index = source.index("'instance', 'job', 'submit'")
    assert schedule_index < start_index < submit_index


def test_dispatch_local_paid_api_ttl_boundary_is_exact() -> None:
    powershell = shutil.which("powershell.exe") or shutil.which("powershell")
    if powershell is None:
        return
    source = CONTROLLER.read_text(encoding="utf-8")
    function_start = source.index("function Assert-MinimumPaidLaunchTtl")
    function_end = source.index("\nfunction ", function_start + 1)
    function_source = source[function_start:function_end]
    harness = f"""
$ErrorActionPreference='Stop'
$MinimumPaidLaunchTtlSeconds=1800.0
{function_source}
$now=[DateTimeOffset]::Parse('2026-09-01T00:00:00+00:00')
$rejected1799=$false
try {{
  [void](Assert-MinimumPaidLaunchTtl -ValidUntil $now.AddSeconds(1799) -Gate 'TEST_1799' -Now $now)
}} catch {{
  $rejected1799=$_.Exception.Message -like 'ORACLE_DISPATCH_MINIMUM_PAID_LAUNCH_TTL_FAILED=TEST_1799/*'
}}
$accepted1800=Assert-MinimumPaidLaunchTtl -ValidUntil $now.AddSeconds(1800) -Gate 'TEST_1800' -Now $now
[ordered]@{{ rejected1799=$rejected1799; accepted1800=$accepted1800 }} | ConvertTo-Json -Compress
"""
    completed = subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-Command", harness],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    observed = json.loads(completed.stdout.strip().splitlines()[-1])
    assert observed == {"rejected1799": True, "accepted1800": 1800}


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash unavailable")
def test_oracle_shell_scripts_parse() -> None:
    subprocess.run(
        [
            "bash",
            "-n",
            AUTHORIZER.relative_to(ROOT).as_posix(),
            LAUNCHER.relative_to(ROOT).as_posix(),
        ],
        check=True,
        cwd=ROOT,
    )


def test_oracle_powershell_supervisor_parses() -> None:
    powershell = shutil.which("powershell.exe") or shutil.which("powershell")
    if powershell is None:
        return
    paths = [SUPERVISOR, CONTROLLER, LOG_GUARD]
    escaped = [str(path).replace("'", "''") for path in paths]
    command = (
        "$ErrorActionPreference='Stop'; "
        + " ".join(
            f"[void][scriptblock]::Create([IO.File]::ReadAllText('{path}'));"
            for path in escaped
        )
        + " "
        "'PARSE_OK'"
    )
    completed = subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-Command", command],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "PARSE_OK" in completed.stdout


def test_log_guard_rejects_replayed_old_heartbeat_and_pass_substrings(
    tmp_path: Path,
) -> None:
    powershell = shutil.which("powershell.exe") or shutil.which("powershell")
    if powershell is None:
        return
    heartbeat = {
        "format": gate.TERMINAL_RECEIPT_FORMAT,
        "status": "ORACLE_COMPARISON_HEARTBEAT",
        "artifact_class": "TIMING_AND_ORACLE_COMPARISON_ONLY",
        "comparison_only": True,
        "formal_training_authorized": False,
        "formal_artifacts_written": 0,
        "theta": "theta_0",
        "estimator": "exact_cartesian_v1_group0_mixed_chunk_oracle",
        "completed_assignments": 1,
        "total_assignments": 29,
        "run_id": gate.RUN_ID,
        "task_id": gate.TASK_ID,
        "patient_fold": 0,
        "seed": 20260726,
        "graph_variant": "G2",
        "assignment_sha256": "a" * 64,
    }
    exact_pass = _terminal_receipt(gate.PASS_STATUS, True)
    unsafe_pass = dict(exact_pass)
    unsafe_pass["formal_training_authorized"] = True
    stale_pass = dict(exact_pass)
    stale_pass["task_id"] = "stale-unrelated-task"
    lines = {
        "heartbeat": json.dumps(heartbeat, separators=(",", ":")),
        "heartbeat_reordered": json.dumps(
            dict(reversed(list(heartbeat.items()))), separators=(",", ":")
        ),
        "unsafe": json.dumps(unsafe_pass, separators=(",", ":")),
        "stale": json.dumps(stale_pass, separators=(",", ":")),
        "safe": json.dumps(exact_pass, separators=(",", ":")),
    }
    data = tmp_path / "windows.json"
    _write_json(data, lines)
    harness = tmp_path / "guard.ps1"
    module_escaped = str(LOG_GUARD).replace("'", "''")
    data_escaped = str(data).replace("'", "''")
    harness.write_text(
        f"""
$ErrorActionPreference='Stop'
Import-Module '{module_escaped}' -Force
$x=Get-Content -LiteralPath '{data_escaped}' -Raw | ConvertFrom-Json
$h=[System.Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
$e=[System.Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
$p=@{{}}
$first=Read-OracleLogWindow -Logs $x.heartbeat -SeenLineHashes $h -SeenEventIds $e -MaxCompletedByPhase $p
$repeat=Read-OracleLogWindow -Logs $x.heartbeat -SeenLineHashes $h -SeenEventIds $e -MaxCompletedByPhase $p
$reordered=Read-OracleLogWindow -Logs $x.heartbeat_reordered -SeenLineHashes $h -SeenEventIds $e -MaxCompletedByPhase $p
$substring=Read-OracleLogWindow -Logs ('prefix {gate.PASS_STATUS} suffix') -SeenLineHashes $h -SeenEventIds $e -MaxCompletedByPhase $p
$unsafe=Read-OracleLogWindow -Logs $x.unsafe -SeenLineHashes $h -SeenEventIds $e -MaxCompletedByPhase $p
$stale=Read-OracleLogWindow -Logs $x.stale -SeenLineHashes $h -SeenEventIds $e -MaxCompletedByPhase $p
$safe=Read-OracleLogWindow -Logs $x.safe -SeenLineHashes $h -SeenEventIds $e -MaxCompletedByPhase $p
[ordered]@{{
 first=$first.NewHeartbeatCount
 repeat=$repeat.NewHeartbeatCount
 reordered=$reordered.NewHeartbeatCount
 substring_terminal=$substring.TerminalStatus
 unsafe_seen=$unsafe.UnsafeTerminalSeen
 unsafe_terminal=$unsafe.TerminalStatus
 stale_seen=$stale.UnsafeTerminalSeen
 stale_terminal=$stale.TerminalStatus
 safe_terminal=$safe.TerminalStatus
}} | ConvertTo-Json -Compress
""",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-File", str(harness)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    observed = json.loads(completed.stdout.strip().splitlines()[-1])
    assert observed["first"] == 1
    assert observed["repeat"] == 0
    assert observed["reordered"] == 0
    assert observed["substring_terminal"] is None
    assert observed["unsafe_seen"] is True
    assert observed["unsafe_terminal"] is None
    assert observed["stale_seen"] is True
    assert observed["stale_terminal"] is None
    assert observed["safe_terminal"] == gate.PASS_STATUS


def test_dispatch_stub_orders_schedule_supervisor_start_submit_and_stop(
    tmp_path: Path,
) -> None:
    powershell = shutil.which("powershell.exe") or shutil.which("powershell")
    if powershell is None:
        return
    stub_source = tmp_path / "CompShareStub.cs"
    stub_exe = tmp_path / "compshare-stub.exe"
    stub_source.write_text(
        r'''
using System;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;

public static class CompShareStub {
    static string Root { get { return Environment.GetEnvironmentVariable("ORACLE_STUB_ROOT"); } }
    static string Marker(string name) { return Path.Combine(Root, name); }
    static string Escape(string value) {
        return value.Replace("\\", "\\\\").Replace("\"", "\\\"")
            .Replace("\r", "\\r").Replace("\n", "\\n");
    }
    static void AppendLog(string line) {
        for (int i = 0; i < 100; ++i) {
            try { File.AppendAllText(Marker("commands.log"), line + Environment.NewLine, Encoding.UTF8); return; }
            catch (IOException) { Thread.Sleep(10); }
        }
        throw new IOException("stub log lock timeout");
    }
    static int Ok(string data) { Console.WriteLine("{\"ok\":true,\"data\":" + data + "}"); return 0; }
    public static int Main(string[] original) {
        var args = original.ToList();
        AppendLog(string.Join(" ", args));
        if (args.Count < 3 || args[0] != "--profile" || args[1] != "default" || args[2] != "--json") return 10;
        args.RemoveRange(0, 3);
        if (args.Count < 2 || args[0] != "instance") return 9;
        if (args[1] == "show") {
            bool running = File.Exists(Marker("running"));
            string state = running ? "Running" : "Stopped";
            string host = "{\"UHostId\":\"uhost-stub4090\",\"State\":\"" + state
                + "\",\"ChargeType\":\"Postpay\",\"GpuType\":\"4090\",\"GPU\":" + (running ? "1" : "0")
                + ",\"CPU\":" + (running ? "16" : "2") + ",\"Memory\":" + (running ? "65536" : "4096")
                + ",\"MachineType\":\"" + (running ? "G" : "O") + "\",\"SupportWithoutGpuStart\":true,"
                + "\"InstancePrice\":" + (running ? "2.05" : "0.14") + ",\"DiskPrice\":0.04}";
            return Ok("{\"UHostSet\":[" + host + "]}");
        }
        if (args[1] == "schedule" && args[2] == "set") {
            int index = args.IndexOf("--at");
            File.WriteAllText(Marker("deadline"), args[index + 1]);
            return Ok("{}");
        }
        if (args[1] == "schedule" && args[2] == "show") {
            string deadline = File.ReadAllText(Marker("deadline"));
            return Ok("{\"instance\":\"uhost-stub4090\",\"scheduled\":true,\"scheduler_stop_time\":" + deadline + "}");
        }
        if (args[1] == "start") { File.WriteAllText(Marker("running"), "1"); return Ok("{}"); }
        if (args[1] == "stop") { if (File.Exists(Marker("running"))) File.Delete(Marker("running")); return Ok("{}"); }
        if (args[1] == "job" && args[2] == "submit") {
            if (!File.Exists(Marker("submitted"))) {
                File.WriteAllText(Marker("submitted"), "1");
                File.WriteAllText(Marker("created"), DateTimeOffset.UtcNow.ToUnixTimeSeconds().ToString());
            }
            return Ok("{\"instance\":\"uhost-stub4090\",\"job\":{\"JobId\":\"v32-oracle-g2f0-s20260726-r2\",\"State\":\"Running\"}}");
        }
        if (args[1] == "job" && args[2] == "list") {
            if (!File.Exists(Marker("submitted"))) return Ok("{\"JobSet\":[],\"TotalCount\":0}");
            string created = File.ReadAllText(Marker("created"));
            return Ok("{\"JobSet\":[{\"JobId\":\"v32-oracle-g2f0-s20260726-r2\",\"CreatedTime\":" + created + "}],\"TotalCount\":1}");
        }
        if (args[1] == "job" && args[2] == "show") {
            if (!File.Exists(Marker("submitted"))) { Console.Error.WriteLine("job not found"); return 2; }
            string created = File.ReadAllText(Marker("created"));
            return Ok("{\"job\":{\"JobId\":\"v32-oracle-g2f0-s20260726-r2\",\"Name\":\"v32-oracle-g2f0-r2\",\"Cwd\":\"./data/CancerLncAtlas\",\"Command\":\"bash ./data/CancerLncAtlas/runtime/oracle_gates/v32_group_shared_oracle_paid_gpu_20260901_r2/server_launch_v32_group_shared_oracle_paid_gpu_20260901_r2.sh\",\"CreatedTime\":" + created + ",\"State\":\"Running\",\"ExitCode\":null}}");
        }
        if (args[1] == "job" && args[2] == "logs") {
            string staticSha = Environment.GetEnvironmentVariable("ORACLE_STUB_STATIC_SHA");
            string terminal = Environment.GetEnvironmentVariable("ORACLE_STUB_TERMINAL");
            string text = "ORACLE_STATIC_AUTH_SHA256=" + staticSha + "\n" + terminal + "\n";
            return Ok("{\"logs\":{\"Stdout\":\"" + Escape(text) + "\",\"Stderr\":\"\"}}");
        }
        return 8;
    }
}
''',
        encoding="utf-8",
    )
    compile_script = tmp_path / "compile.ps1"
    stub_source_literal = str(stub_source).replace("'", "''")
    stub_exe_literal = str(stub_exe).replace("'", "''")
    compile_script.write_text(
        "$ErrorActionPreference='Stop'; "
        f"Add-Type -TypeDefinition (Get-Content -LiteralPath '{stub_source_literal}' -Raw) "
        f"-OutputAssembly '{stub_exe_literal}' -OutputType ConsoleApplication",
        encoding="utf-8",
    )
    compiled = subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-File", str(compile_script)],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert compiled.returncode == 0, compiled.stderr

    manifest = json.loads(EXTERNAL_MANIFEST.read_text(encoding="utf-8"))
    local_sources = {
        gate.VALIDATOR_NAME: VALIDATOR,
        gate.LAUNCHER_NAME: LAUNCHER,
        gate.AUTHORIZER_NAME: AUTHORIZER,
        gate.SUPERVISOR_NAME: SUPERVISOR,
        gate.DISPATCH_CONTROLLER_NAME: CONTROLLER,
        gate.LOG_GUARD_NAME: LOG_GUARD,
    }
    for record in manifest["artifacts"]:
        source = local_sources[record["name"]]
        record["sha256"] = _sha(source)
        record["size_bytes"] = source.stat().st_size
    test_manifest = tmp_path / gate.EXTERNAL_DEPLOYMENT_MANIFEST_NAME
    _write_json(test_manifest, manifest)
    instance_id = "uhost-stub4090"
    now = datetime.now(timezone.utc)
    jit_path = tmp_path / "JIT_BUDGET_READY.json"
    _write_json(
        jit_path,
        {
            "format": gate.JIT_FORMAT,
            "status": gate.JIT_STATUS,
            "currency": "CNY",
            "source": "COMPSHARE_API_BILLING_SNAPSHOT",
            "project_budget_cap_cny": gate.PROJECT_BUDGET_CAP_CNY,
            "compute_hourly_cny": gate.COMPUTE_HOURLY_CNY,
            "gpu_hourly_cny": gate.GPU_HOURLY_CNY,
            "disk_hourly_cny": gate.DISK_HOURLY_CNY,
            "total_hourly_cny": gate.TOTAL_HOURLY_CNY,
            "profile_name_sha256": gate.EXPECTED_PROFILE_NAME_SHA256,
            "project_scope_sha256": gate.EXPECTED_PROJECT_SCOPE_SHA256,
            "instance_id": instance_id,
            "queried_at": (now - timedelta(minutes=1)).isoformat(),
            "valid_until": (now + timedelta(minutes=44)).isoformat(),
            "conservative_remaining_cny": 8.0,
        },
    )
    static_path = tmp_path / "STATIC_AUTH_READY.json"
    _write_json(
        static_path,
        {
            "format": gate.STATIC_FORMAT,
            "status": gate.STATIC_STATUS,
            "namespace": gate.NAMESPACE,
            "comparison_only": True,
            "formal_training_authorized": False,
            "max_paid_hours": gate.MAX_HOURS,
            "configured_cost_cap_cny": gate.MAX_COST_CNY,
            "jit_effective_cost_cap_cny": 8.0,
            "instance_id": instance_id,
            "jit_budget_receipt_sha256": _sha(jit_path),
            "jit_valid_until": (now + timedelta(minutes=44)).isoformat(),
            "local_paid_api_minimum_ttl_seconds": 1800,
            "remote_paid_launch_minimum_ttl_seconds": 900,
            "external_verifier_sha256": _sha(VALIDATOR),
            "external_control_deployment_manifest_sha256": _sha(test_manifest),
            "external_control_artifacts": manifest["artifacts"],
        },
    )
    supervisor_log = tmp_path / "supervisor.log"
    stop_receipt = tmp_path / "AUTOSTOP.json"
    dispatch_jit = tmp_path / "DISPATCH_JIT.json"
    command = [
        powershell,
        "-NoProfile",
        "-NonInteractive",
        "-File",
        str(CONTROLLER),
        "-InstanceId",
        instance_id,
        "-StaticAuthReceiptPath",
        str(static_path),
        "-JitBudgetReceiptPath",
        str(jit_path),
        "-ExternalDeploymentManifestPath",
        str(test_manifest),
        "-FrozenVerifierPath",
        str(VALIDATOR),
        "-NoGpuAuthorizerPath",
        str(AUTHORIZER),
        "-ServerLauncherPath",
        str(LAUNCHER),
        "-SupervisorPath",
        str(SUPERVISOR),
        "-LogGuardModulePath",
        str(LOG_GUARD),
        "-SupervisorLogPath",
        str(supervisor_log),
        "-StopReceiptPath",
        str(stop_receipt),
        "-DispatchJitReceiptPath",
        str(dispatch_jit),
        "-CompSharePath",
        str(stub_exe),
    ]
    environment = dict(os.environ)
    environment["ORACLE_STUB_ROOT"] = str(tmp_path)
    environment["ORACLE_STUB_STATIC_SHA"] = _sha(static_path)
    environment["ORACLE_STUB_TERMINAL"] = json.dumps(
        _terminal_receipt(gate.PASS_STATUS, True), separators=(",", ":")
    )
    dispatched = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
        timeout=60,
    )
    assert dispatched.returncode == 0, dispatched.stderr
    commands = (tmp_path / "commands.log").read_text(encoding="utf-8-sig").splitlines()
    assert commands and all(line.startswith("--profile default --json ") for line in commands)
    schedule_set = next(i for i, line in enumerate(commands) if "instance schedule set" in line)
    pre_submit_poll = next(i for i, line in enumerate(commands) if "instance job show" in line)
    start = next(i for i, line in enumerate(commands) if "instance start" in line)
    submit = next(i for i, line in enumerate(commands) if "instance job submit" in line)
    stop = next(i for i, line in enumerate(commands) if "instance stop" in line)
    assert schedule_set < pre_submit_poll < start < submit < stop
    assert not (tmp_path / "running").exists()
    observed_stop = json.loads(stop_receipt.read_text(encoding="utf-8-sig"))
    assert observed_stop["status"] == "INSTANCE_STOPPED_STATE_CONFIRMED"
    assert observed_stop["terminal_receipt_validated_after_stop"] is True
    dispatch_receipt = json.loads(dispatch_jit.read_text(encoding="utf-8-sig"))
    assert dispatch_receipt["profile_name_sha256"] == gate.EXPECTED_PROFILE_NAME_SHA256
    assert dispatch_receipt["project_scope_sha256"] == gate.EXPECTED_PROJECT_SCOPE_SHA256
    assert dispatch_receipt["provider_stopped_profile"] == "NO_GPU_A_POST_START_STOPPED"
    assert dispatch_receipt["provider_compute_hourly_cny"] == 0.14
    assert dispatch_receipt["conservative_budget_total_hourly_cny"] == 2.09
    assert dispatch_receipt["local_paid_api_minimum_ttl_seconds"] == 1800
    assert dispatch_receipt["remote_paid_launch_minimum_ttl_seconds"] == 900

    # A prior fixed-JobId directory with an old creation time must be rejected
    # by the already-attached supervisor before a second paid start or submit.
    (tmp_path / "created").write_text("1", encoding="ascii")
    jit2 = tmp_path / "JIT_BUDGET_READY.second.json"
    jit2_payload = json.loads(jit_path.read_text(encoding="utf-8"))
    now2 = datetime.now(timezone.utc)
    jit2_payload["queried_at"] = (now2 - timedelta(minutes=1)).isoformat()
    jit2_payload["valid_until"] = (now2 + timedelta(minutes=44)).isoformat()
    _write_json(jit2, jit2_payload)
    static2 = tmp_path / "STATIC_AUTH_READY.second.json"
    static2_payload = json.loads(static_path.read_text(encoding="utf-8"))
    static2_payload["jit_budget_receipt_sha256"] = _sha(jit2)
    static2_payload["jit_valid_until"] = jit2_payload["valid_until"]
    _write_json(static2, static2_payload)
    supervisor_log2 = tmp_path / "supervisor.second.log"
    stop_receipt2 = tmp_path / "AUTOSTOP.second.json"
    dispatch_jit2 = tmp_path / "DISPATCH_JIT.second.json"
    replacements = {
        str(static_path): str(static2),
        str(jit_path): str(jit2),
        str(supervisor_log): str(supervisor_log2),
        str(stop_receipt): str(stop_receipt2),
        str(dispatch_jit): str(dispatch_jit2),
    }
    second_command = [replacements.get(value, value) for value in command]
    second_environment = dict(environment)
    second_environment["ORACLE_STUB_STATIC_SHA"] = _sha(static2)
    before_second = len(commands)
    rejected = subprocess.run(
        second_command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=second_environment,
        timeout=60,
    )
    assert rejected.returncode != 0
    second_commands = (tmp_path / "commands.log").read_text(
        encoding="utf-8-sig"
    ).splitlines()[before_second:]
    assert any("instance job show" in line for line in second_commands)
    assert not any("instance start" in line for line in second_commands)
    assert not any("instance job submit" in line for line in second_commands)

    # A locally stale launch window is rejected before every provider API,
    # including the dispatcher's cleanup path.
    now3 = datetime.now(timezone.utc)
    jit3 = tmp_path / "JIT_BUDGET_READY.ttl1799.json"
    jit3_payload = dict(jit2_payload)
    jit3_payload["queried_at"] = (now3 - timedelta(minutes=14)).isoformat()
    jit3_payload["valid_until"] = (now3 + timedelta(seconds=1790)).isoformat()
    _write_json(jit3, jit3_payload)
    static3 = tmp_path / "STATIC_AUTH_READY.ttl1799.json"
    static3_payload = dict(static2_payload)
    static3_payload["jit_budget_receipt_sha256"] = _sha(jit3)
    static3_payload["jit_valid_until"] = jit3_payload["valid_until"]
    _write_json(static3, static3_payload)
    supervisor_log3 = tmp_path / "supervisor.ttl1799.log"
    stop_receipt3 = tmp_path / "AUTOSTOP.ttl1799.json"
    dispatch_jit3 = tmp_path / "DISPATCH_JIT.ttl1799.json"
    third_replacements = {
        str(static_path): str(static3),
        str(jit_path): str(jit3),
        str(supervisor_log): str(supervisor_log3),
        str(stop_receipt): str(stop_receipt3),
        str(dispatch_jit): str(dispatch_jit3),
    }
    third_command = [third_replacements.get(value, value) for value in command]
    before_third = len(
        (tmp_path / "commands.log").read_text(encoding="utf-8-sig").splitlines()
    )
    rejected_ttl = subprocess.run(
        third_command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
        timeout=60,
    )
    assert rejected_ttl.returncode != 0
    assert "MINIMUM_PAID_LAUNCH_TTL" in rejected_ttl.stderr
    after_third = (tmp_path / "commands.log").read_text(
        encoding="utf-8-sig"
    ).splitlines()
    assert len(after_third) == before_third
