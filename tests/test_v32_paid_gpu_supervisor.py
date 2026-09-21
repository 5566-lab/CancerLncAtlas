from __future__ import annotations

import base64
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUPERVISOR = ROOT / "scripts/local_supervise_compshare_paid_gpu_20260901_r2.ps1"


def _source() -> str:
    return SUPERVISOR.read_text(encoding="utf-8")


def test_supervisor_has_conservative_heartbeat_and_budget_contract() -> None:
    source = _source()
    assert "[double]$RemainingBudgetCny," in source
    assert "$RemainingBudgetCny =" not in source
    assert "[int]$InitialHeartbeatMinutes = 12" in source
    assert "[int]$HeartbeatStallMinutes = 30" in source
    assert "[int]$GpuTelemetryStallMinutes = 15" in source
    assert "$RequiredOptimizerPattern" in source
    assert '$RequiredGpuTelemetryPattern' in source
    assert '"OPTIMIZER_STEP"' in source
    assert '"GPU_TELEMETRY"' in source
    assert "GPU_PHASE_HEARTBEAT" not in source
    assert "PASS_REAL_BACKWARD_PROBE" not in source
    assert "TRAINING_START_GATE_TIMEOUT_OPTIMIZER_PLUS_GPU_TELEMETRY_REQUIRED" in source
    assert "OPTIMIZER_PROGRESS_STALLED" in source
    assert "GPU_TELEMETRY_STALLED" in source
    assert "$RequiredOptimizerPattern -cne $ExpectedOptimizerPattern" in source
    assert "$RequiredGpuTelemetryPattern -cne $ExpectedGpuTelemetryPattern" in source
    assert "$InitialHeartbeatMinutes -gt 15" in source
    assert "$HeartbeatStallMinutes -gt 30" in source
    assert "$GpuTelemetryStallMinutes -gt 15" in source
    assert "$MissingJobGraceMinutes -gt 5" in source
    assert "$GpuHourlyCny + $DiskHourlyCny" in source
    assert "$StartupUnwatchedMinutes / 60.0" in source
    assert "$RemainingBudgetCny - $startupReserveCny - $BudgetSafetyReserveCny" in source
    assert "PROVIDER_AUTO_STOP_REQUIRED" in source
    assert "$watchStartMessage = (" in source
    assert "Write-WatchEventBestEffort $watchStartMessage" in source


def test_supervisor_bounds_cli_and_stops_before_logging() -> None:
    source = _source()
    assert "$ExpectedInstanceId = 'uhost-1up504geeqzj'" in source
    assert "$InstanceId -cne $ExpectedInstanceId" in source
    assert "$script:instanceScopeValidated = $true" in source
    assert "if ($script:instanceScopeValidated)" in source
    assert "$RemainingBudgetCny -gt 210.0" in source
    assert "($GpuHourlyCny + $DiskHourlyCny) -lt 2.09" in source
    assert "$BudgetSafetyReserveCny -lt 2.0" in source
    assert "$HardDeadlineHours -gt 96.0" in source
    assert "ReadToEndAsync()" in source
    assert "New-Object System.Text.UTF8Encoding($false, $true)" in source
    assert "$startInfo.StandardOutputEncoding = $utf8NoBomStrict" in source
    assert "$startInfo.StandardErrorEncoding = $utf8NoBomStrict" in source
    assert "$process.WaitForExit($TimeoutSeconds * 1000)" in source
    assert "Stop-ProcessTree -TargetProcessId $timedOutProcessId" in source
    assert "COMPSHARE_PROCESS_TIMEOUT=" in source
    assert "'instance', 'show', $InstanceId" in source
    assert "$observedState -eq 'Stopped'" in source

    stop_start = source.index("function Stop-PaidInstance")
    stop_end = source.index("function Find-LatestMatchingLine", stop_start)
    stop_body = source[stop_start:stop_end]
    first_stop = stop_body.index("Invoke-CompShareJson")
    first_log = stop_body.index("Write-WatchEventBestEffort")
    assert first_stop < first_log
    assert "while (-not $script:stopConfirmed)" in stop_body

    catch_start = source.rindex("catch {")
    catch_body = source[catch_start:]
    assert catch_body.index("Stop-PaidInstance") < catch_body.index(
        "Write-WatchEventBestEffort"
    )


def test_supervisor_parses_in_windows_powershell_51() -> None:
    powershell = shutil.which("powershell.exe") or shutil.which("powershell")
    if powershell is None:
        return
    escaped = str(SUPERVISOR).replace("'", "''")
    command = (
        "$ErrorActionPreference='Stop'; "
        f"[void][scriptblock]::Create([IO.File]::ReadAllText('{escaped}')); "
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


def test_windows_powershell_process_capture_preserves_utf8_json(
    tmp_path: Path,
) -> None:
    """Exercise the exact PS 5.1 ProcessStartInfo UTF-8 capture contract."""

    powershell = shutil.which("powershell.exe") or shutil.which("powershell")
    if powershell is None:
        return
    description = "英伟达 RTX 4090 显卡"
    payload = {
        "ok": True,
        "data": {"UHostSet": [{"State": "Stopped", "GPUTypeDesc": description}]},
    }
    emitter = tmp_path / "emit_utf8_compshare_json.py"
    emitter.write_text(
        "import json, sys\n"
        f"payload = {payload!r}\n"
        "sys.stdout.buffer.write(json.dumps(payload, ensure_ascii=False).encode('utf-8'))\n",
        encoding="utf-8",
    )
    executable = str(Path(sys.executable).resolve()).replace("'", "''")
    emitter_path = str(emitter.resolve()).replace("'", "''")
    expected_b64 = base64.b64encode(description.encode("utf-8")).decode("ascii")
    command = (
        "$ErrorActionPreference='Stop'; "
        "$si=New-Object System.Diagnostics.ProcessStartInfo; "
        f"$si.FileName='{executable}'; "
        f"$si.Arguments='\"{emitter_path}\"'; "
        "$si.UseShellExecute=$false; $si.CreateNoWindow=$true; "
        "$si.RedirectStandardOutput=$true; $si.RedirectStandardError=$true; "
        "$utf8=New-Object System.Text.UTF8Encoding($false,$true); "
        "$si.StandardOutputEncoding=$utf8; $si.StandardErrorEncoding=$utf8; "
        "$p=New-Object System.Diagnostics.Process; $p.StartInfo=$si; "
        "[void]$p.Start(); "
        "$outTask=$p.StandardOutput.ReadToEndAsync(); "
        "$errTask=$p.StandardError.ReadToEndAsync(); "
        "$p.WaitForExit(); "
        "$out=$outTask.GetAwaiter().GetResult(); "
        "$err=$errTask.GetAwaiter().GetResult(); "
        "if($p.ExitCode -ne 0){throw $err}; "
        "$parsed=$out | ConvertFrom-Json; "
        f"$expected=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{expected_b64}')); "
        "if($parsed.data.UHostSet[0].State -ne 'Stopped'){throw 'STATE_DRIFT'}; "
        "if($parsed.data.UHostSet[0].GPUTypeDesc -ne $expected){throw 'UTF8_DRIFT'}; "
        "'UTF8_JSON_OK'"
    )
    completed = subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-Command", command],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "UTF8_JSON_OK" in completed.stdout
