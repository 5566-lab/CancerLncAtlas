from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_paid_gpu_launchers_have_no_instance_or_endpoint_defaults() -> None:
    for relative in (
        "scripts/local_submit_g012_native_job_20260902.ps1",
        "runtime/local_submit_g012_native_job_20260902_r5.ps1",
    ):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "[string]$InstanceId = ''" in text
        assert "[string]$TargetHost = ''" in text
        assert "[int]$TargetPort = 0" in text
        assert "uhost-1us0kyd3hicf" not in text
        assert "117.50.198.67" not in text
    gpu_only = (
        ROOT / "scripts/local_submit_g012_gpu_only_20260903.ps1"
    ).read_text(encoding="utf-8")
    assert '[Parameter(Mandatory = $true)]' in gpu_only
    assert "uhost-1us0kyd3hicf" not in gpu_only
    assert "117.50.198.67" not in gpu_only


def test_paid_gpu_launchers_require_explicit_start_switch_before_provider_call() -> None:
    for relative in (
        "scripts/local_submit_g012_native_job_20260902.ps1",
        "runtime/local_submit_g012_native_job_20260902_r5.ps1",
        "scripts/local_submit_g012_gpu_only_20260903.ps1",
    ):
        text = (ROOT / relative).read_text(encoding="utf-8")
        guard = text.index("if (-not $PaidGpuStartAuthorized)")
        first_provider_start = min(
            position
            for token in ("'instance','start'", "'instance', 'start'")
            if (position := text.find(token)) >= 0
        )
        assert guard < first_provider_start
        assert "PAID_GPU_START_REQUIRES_EXPLICIT_AUTHORIZATION_SWITCH" in text


def test_gpu_only_wrapper_refuses_static_auth_drift_and_cpu_prepare() -> None:
    text = (
        ROOT / "scripts/cloud_gpu_only_g012_formal_wrapper_20260903.sh"
    ).read_text(encoding="utf-8")
    assert "expected_static_auth_sha256=" in text
    assert 'sha256sum "$static_auth"' in text
    assert "cloud_post_transfer_g012_20260902.sh" not in text
    assert "cloud_prepare_v32_g012_no_gpu" not in text
    assert 'bash "$launcher"' in text


def test_gpu_only_supervisor_enforces_runtime_start_gate_and_failure_stop() -> None:
    text = (
        ROOT / "scripts/local_submit_g012_gpu_only_20260903.ps1"
    ).read_text(encoding="utf-8")
    assert "GPU_OPTIMIZER_AND_TELEMETRY_START_GATE_PASS" in text
    assert "GPU_OPTIMIZER_AND_TELEMETRY_START_GATE_TIMEOUT" in text
    assert "INSTANCE_STOPPED_AFTER_FAILURE" in text
    assert "INSTANCE_AND_DISK_DELETED_AFTER_VERIFIED_RESULT_RETURN" in text


def test_transfer_supervisor_allows_explicit_current_staged_instance_only() -> None:
    text = (
        ROOT / "scripts/local_supervise_g012_transfer_train_20260902.ps1"
    ).read_text(encoding="utf-8")
    assert "uhost-1up504geeqzj" in text
    assert "uhost-1us0kyd3hicf" not in text
    assert "EXPLICIT_FRESH_INSTANCE_ID_HOST_AND_PORT_REQUIRED" in text
