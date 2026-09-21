from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

from cc_hhgt.v32 import aggregate_oracle_runner_r3 as runner
from cc_hhgt.v32 import group_shared_encoder_oracle as oracle
from scripts.verify_v32_group_shared_oracle_aggregate_result_r3 import verify


ROOT = Path(__file__).resolve().parents[1]
INPUT_SHA = "1" * 64
CODE_SHA = "2" * 64


class _FakeTorch:
    @staticmethod
    def load(handle, *, map_location, weights_only):
        assert map_location == "cpu"
        assert weights_only is False
        return {"raw": handle.read()}


def test_aggregate_member_loader_uses_stable_handle_without_fold_digest(tmp_path):
    prepared = tmp_path / "G2" / "PATIENT_FOLD_0.pt"
    prepared.parent.mkdir()
    prepared.write_bytes(b"aggregate-member")
    assert runner._stable_archive_member_load(
        _FakeTorch, prepared
    ) == {"raw": b"aggregate-member"}


def test_runner_builds_archive_only_context_and_writes_one_result(
    tmp_path, monkeypatch
):
    repo = tmp_path / "code"
    config = repo / "config" / "model.yaml"
    task = repo / "config" / "task.tsv"
    prepared = tmp_path / "input" / "G2" / "PATIENT_FOLD_0.pt"
    config.parent.mkdir(parents=True)
    prepared.parent.mkdir(parents=True)
    config.write_text("x: 1\n", encoding="utf-8")
    task.write_text("task_id\n", encoding="utf-8")
    prepared.write_bytes(b"pt")
    observed = {}

    def fake_run(context):
        observed.update(context)
        return {
            **oracle._base_receipt(oracle.PASS_STATUS),
            "scientific_pass": True,
            "aggregate_input_archive_sha256": INPUT_SHA,
            "aggregate_code_archive_sha256": CODE_SHA,
            "integrity_scope": "INPUT_CODE_RESULT_ARCHIVES_ONLY",
            "per_file_hashing_performed": False,
            "per_fold_hashing_performed": False,
        }

    monkeypatch.setattr(runner, "_run_archive_only_oracle", fake_run)
    output = tmp_path / "result" / "scientific_result.json"
    args = argparse.Namespace(
        repo_root=str(repo),
        config=str(config),
        task_manifest=str(task),
        prepared=str(prepared),
        input_archive_sha256=INPUT_SHA,
        code_archive_sha256=CODE_SHA,
        output=str(output),
    )
    assert runner.run(args) == 0
    assert observed["aggregate_archive_only"] is True
    assert observed["input_manifest_path"] is None
    assert observed["artifact_hashes"] == {}
    assert observed["aggregate_input_archive_sha256"] == INPUT_SHA
    assert observed["aggregate_code_archive_sha256"] == CODE_SHA
    assert json.loads(output.read_text(encoding="utf-8"))["scientific_pass"] is True
    assert not output.with_name(output.name + ".partial").exists()


def test_archive_adapter_emits_actual_first_step_and_restores_frozen_runtime(
    tmp_path, monkeypatch, capsys
):
    from cc_hhgt.v32 import training

    prepared = tmp_path / "G2" / "PATIENT_FOLD_0.pt"
    prepared.parent.mkdir()
    prepared.write_bytes(b"payload")

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def current_device():
            return 0

        @staticmethod
        def get_device_name(_device):
            return "NVIDIA GeForce RTX 4090"

    class FakeTorch(_FakeTorch):
        cuda = FakeCuda()

    def original_step(*_args, **_kwargs):
        return {
            "grad_finite": True,
            "parameters_finite": True,
            "parameter_delta_positive": True,
        }

    monkeypatch.setattr(training, "optimizer_step_with_guards", original_step)
    original_resolver = training._resolve_prepared_path
    original_loader = training.load_prepared_artifact_from_authorized_handle
    original_file_sha = training._file_sha256
    original_emit = oracle._emit_receipt

    def fake_frozen_oracle(context):
        assert training._resolve_prepared_path({}, 0, Path("unused")) == prepared
        payload, marker = training.load_prepared_artifact_from_authorized_handle(
            FakeTorch,
            None,
            fold=0,
            prepared_path=prepared,
        )
        assert payload == {"raw": b"payload"}
        assert marker == "NOT_COMPUTED_AGGREGATE_INPUT_ARCHIVE_ONLY"
        assert (
            training._file_sha256(Path("never-opened"))
            == "NOT_COMPUTED_AGGREGATE_CODE_ARCHIVE_ONLY"
        )
        training.optimizer_step_with_guards(
            None,
            None,
            torch=FakeTorch,
            objective_value=1.25,
        )
        oracle._emit_receipt(
            {
                "format": oracle.RECEIPT_FORMAT,
                "status": "ORACLE_COMPARISON_HEARTBEAT",
            }
        )
        return {
            **oracle._base_receipt(oracle.PASS_STATUS),
            "scientific_pass": True,
        }

    monkeypatch.setattr(oracle, "_run_verified_oracle_comparison", fake_frozen_oracle)
    context = {
        "run_id": runner.RUN_ID,
        "task_id": runner.TASK_ID,
        "prepared_path": str(prepared),
        "aggregate_input_archive_sha256": INPUT_SHA,
        "aggregate_code_archive_sha256": CODE_SHA,
    }
    result = runner._run_archive_only_oracle(context)
    assert result["scientific_pass"] is True
    assert result["per_file_hashing_performed"] is False
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    step = next(item for item in lines if item["status"] == "ORACLE_R3_FIRST_OPTIMIZER_STEP")
    assert step["optimizer_step"] == 1
    assert step["parameter_delta_positive"] is True
    assert step["gpu_identity"]["name"] == "NVIDIA GeForce RTX 4090"
    heartbeat = next(item for item in lines if item["status"] == "ORACLE_COMPARISON_HEARTBEAT")
    assert heartbeat["aggregate_input_archive_sha256"] == INPUT_SHA
    assert training._resolve_prepared_path is original_resolver
    assert training.load_prepared_artifact_from_authorized_handle is original_loader
    assert training._file_sha256 is original_file_sha
    assert training.optimizer_step_with_guards is original_step
    assert oracle._emit_receipt is original_emit


def _make_result_archive(tmp_path: Path) -> Path:
    result = {
        "format": oracle.RECEIPT_FORMAT,
        "status": oracle.PASS_STATUS,
        "scientific_pass": True,
        "aggregate_input_archive_sha256": INPUT_SHA,
        "aggregate_code_archive_sha256": CODE_SHA,
        "integrity_scope": "INPUT_CODE_RESULT_ARCHIVES_ONLY",
        "per_file_hashing_performed": False,
        "per_fold_hashing_performed": False,
    }
    result_path = tmp_path / "scientific_result.json"
    log_path = tmp_path / "execution.stdout.jsonl"
    result_path.write_text(json.dumps(result) + "\n", encoding="utf-8")
    log_path.write_text(
        json.dumps(
            {
                "format": "CANCERLNCATLAS_V32_ORACLE_AGGREGATE_RUN_R3_V1",
                "status": "ORACLE_R3_NVIDIA_SMI_TELEMETRY",
                "gpu_count": 1,
                "gpu_name": "NVIDIA GeForce RTX 4090",
                "gpu_uuid": "GPU-test",
                "telemetry_source": "nvidia-smi",
            }
        )
        + "\n"
        + json.dumps(
            {
                "format": oracle.RECEIPT_FORMAT,
                "status": "ORACLE_R3_FIRST_OPTIMIZER_STEP",
                "run_id": runner.RUN_ID,
                "task_id": runner.TASK_ID,
                "optimizer_step": 1,
                "grad_finite": True,
                "parameters_finite": True,
                "parameter_delta_positive": True,
                "aggregate_input_archive_sha256": INPUT_SHA,
                "aggregate_code_archive_sha256": CODE_SHA,
                "per_file_hashing_performed": False,
                "per_fold_hashing_performed": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    archive = tmp_path / "result.tar"
    with tarfile.open(archive, "w") as handle:
        handle.add(result_path, arcname=result_path.name)
        handle.add(log_path, arcname=log_path.name)
    return archive


def test_return_verifier_checks_result_tar_once_and_training_start_evidence(
    tmp_path,
):
    archive = _make_result_archive(tmp_path)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    value = verify(
        archive,
        expected_result_sha256=digest,
        expected_input_sha256=INPUT_SHA,
        expected_code_sha256=CODE_SHA,
    )
    assert value["training_start_confirmed"] is True
    assert value["result_archive_sha256"] == digest
    assert value["per_file_hashing_performed"] is False
    assert value["per_fold_hashing_performed"] is False


def test_return_verifier_rejects_aggregate_sha_drift(tmp_path):
    archive = _make_result_archive(tmp_path)
    with pytest.raises(ValueError, match="RESULT_ARCHIVE_SHA256_DRIFT"):
        verify(
            archive,
            expected_result_sha256="f" * 64,
            expected_input_sha256=INPUT_SHA,
            expected_code_sha256=CODE_SHA,
        )


def test_return_verifier_rejects_telemetry_without_optimizer_step(tmp_path):
    valid = _make_result_archive(tmp_path)
    with tarfile.open(valid, "r") as source:
        result_bytes = source.extractfile("scientific_result.json").read()
        log_lines = source.extractfile("execution.stdout.jsonl").read().splitlines()
    bad_root = tmp_path / "missing-step"
    bad_root.mkdir()
    (bad_root / "scientific_result.json").write_bytes(result_bytes)
    (bad_root / "execution.stdout.jsonl").write_bytes(log_lines[0] + b"\n")
    archive = bad_root / "result.tar"
    with tarfile.open(archive, "w") as handle:
        handle.add(bad_root / "scientific_result.json", arcname="scientific_result.json")
        handle.add(bad_root / "execution.stdout.jsonl", arcname="execution.stdout.jsonl")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="VALID_FIRST_OPTIMIZER_STEP"):
        verify(
            archive,
            expected_result_sha256=digest,
            expected_input_sha256=INPUT_SHA,
            expected_code_sha256=CODE_SHA,
        )


def test_r3_shells_parse_and_gpu_query_follows_both_archive_checks():
    preflight = ROOT / "scripts" / "cloud_preflight_v32_group_shared_oracle_aggregate_r3.sh"
    launcher = ROOT / "scripts" / "server_launch_v32_group_shared_oracle_paid_gpu_20260901_r3.sh"
    for path in (preflight, launcher):
        subprocess.run(
            ["bash", "-n", path.relative_to(ROOT).as_posix()],
            cwd=ROOT,
            check=True,
        )
    source = launcher.read_text(encoding="utf-8")
    gpu_query = source.index("command -v nvidia-smi")
    assert 'require_regular "$prepared"' in source[:gpu_query]
    assert 'require_sha "$input_archive" "$input_sha256"' not in source
    assert 'require_sha "$code_archive" "$code_sha256"' not in source
    assert '-xf "$input_archive"' not in source
    assert '-xzf "$code_archive"' not in source
    assert "timeout --signal=TERM --kill-after=30s" in source
    assert "hard_timeout_seconds=10800" in source
    assert 'ORACLE_R3_PREPARED_ROOT_NAMESPACE_BINDING_DRIFT' in source
    assert 'ORACLE_R3_LAUNCHER_BINDING_DRIFT' in source

    preflight_source = preflight.read_text(encoding="utf-8")
    assert 'require_sha "$input_archive" "$input_sha256"' in preflight_source
    assert 'require_sha "$code_archive" "$code_sha256"' in preflight_source
    assert '-xf "$input_archive"' in preflight_source
    assert '-xzf "$code_archive"' in preflight_source
    expected_python = "/usr/local/miniconda3/envs/py312/bin/python"
    assert f'python_bin={expected_python}' in preflight_source
    assert 'PYTHONPATH="$code_root" "$python_bin" -m cc_hhgt.v32.aggregate_oracle_runner_r3' in preflight_source
    assert f"python_bin={expected_python}" in source
    assert "V32_ORACLE_GPU_PYTHON" not in source
    for runtime_module in (
        "cc_hhgt.gnn",
        "cc_hhgt.v32.aggregate_oracle_runner_r3",
        "cc_hhgt.v32.gpu_backward_probe",
        "cc_hhgt.v32.group_shared_encoder_oracle",
        "cc_hhgt.v32.training",
        "torch",
        "torch_geometric",
        "yaml",
    ):
        assert f"import {runtime_module}" in preflight_source
    assert '"paid_gpu_archive_rehashing_required": False' in preflight_source
    assert '"paid_gpu_archive_extraction_required": False' in preflight_source
    assert 'input_allowed = {"G2", "G2/PATIENT_FOLD_0.pt"}' in preflight_source
    assert "set(input_names).issubset(input_allowed)" in preflight_source
    assert 'input_names.get("G2/PATIENT_FOLD_0.pt") != "file"' in preflight_source
    assert 'input_names != {"G2", "G2/PATIENT_FOLD_0.pt"}' not in preflight_source
    assert '"scripts/server_launch_v32_group_shared_oracle_paid_gpu_20260901_r3.sh"' in preflight_source


def test_r3_chain_has_no_legacy_instance_or_time_limited_receipt_gate():
    paths = [
        ROOT / "scripts" / "server_launch_v32_group_shared_oracle_paid_gpu_20260901_r3.sh",
        ROOT / "scripts" / "local_supervise_compshare_group_shared_oracle_paid_gpu_20260901_r3.ps1",
        ROOT / "scripts" / "v32_group_shared_oracle_log_guard_r3.psm1",
    ]
    source = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    supervisor_source = paths[1].read_text(encoding="utf-8")
    assert "[string]$provider.MachineType -ceq 'G'" in supervisor_source
    assert "[string]$provider.MachineType -ceq 'O'" in supervisor_source
    assert "$ready.cpu_extraction_complete -ne $true" in supervisor_source
    assert "$ready.paid_gpu_archive_rehashing_required -ne $false" in supervisor_source
    assert "$ready.paid_gpu_archive_extraction_required -ne $false" in supervisor_source
    for forbidden in (
        "STATIC_AUTH",
        "JIT_BUDGET",
        "ORACLE_APPROVAL",
        "uhost-1up504geeqzj",
        "per_fold_sha",
        "prepared_sha256_authority",
    ):
        assert forbidden not in source
    assert "HardPaidHours = 3.0" in source
    assert "HardCostCapCny = 8.0" in source
    assert "Stop-AndConfirm 'ORACLE_R3_PROGRESS_STALLED'" in source
    assert "ORACLE_R3_TRAINING_START_CONFIRMED=NVIDIA_SMI_PLUS_FIRST_OPTIMIZER_STEP" in source


def test_r3_powershell_log_guard_requires_both_start_signals(tmp_path):
    module = ROOT / "scripts" / "v32_group_shared_oracle_log_guard_r3.psm1"
    probe = tmp_path / "probe.ps1"
    probe.write_text(
        f"""
$ErrorActionPreference = 'Stop'
Import-Module -Force -Name '{module}'
$inputSha = '{INPUT_SHA}'
$codeSha = '{CODE_SHA}'
$telemetry = @{{format='CANCERLNCATLAS_V32_ORACLE_AGGREGATE_RUN_R3_V1';status='ORACLE_R3_NVIDIA_SMI_TELEMETRY';gpu_count=1;gpu_name='NVIDIA GeForce RTX 4090';gpu_uuid='GPU-x';driver_version='1';memory_total_mib=24564;telemetry_source='nvidia-smi';sampled_at='2026-09-01T00:00:00Z'}} | ConvertTo-Json -Compress
$optimizer = @{{format='CC_HHGT_V3_2_GROUP_SHARED_ORACLE_COMPARISON_V1';status='ORACLE_R3_FIRST_OPTIMIZER_STEP';run_id='v32-g012-g2-group-shared-oracle-paid-gpu-20260901-r3';task_id='v32-g012-g2-group-shared-oracle-paid-gpu-20260901-r3|PATIENT_FOLD_0|CC-HHGT|20260726';graph_variant='G2';patient_fold=0;seed=20260726;optimizer_step=1;grad_finite=$true;parameters_finite=$true;parameter_delta_positive=$true;gpu_identity=@{{name='RTX 4090'}};aggregate_input_archive_sha256=$inputSha;aggregate_code_archive_sha256=$codeSha;per_file_hashing_performed=$false;per_fold_hashing_performed=$false}} | ConvertTo-Json -Compress
$seen = New-Object 'System.Collections.Generic.HashSet[string]'
$value = Read-R3OracleLogWindow -Logs ($telemetry + "`n" + $optimizer) -ExpectedInputSha256 $inputSha -ExpectedCodeSha256 $codeSha -SeenProgressIds $seen
@{{telemetry=($null -ne $value.Telemetry);optimizer=($null -ne $value.Optimizer);progress=$value.NewProgressCount;unsafe=$value.UnsafeControlSeen}} | ConvertTo-Json -Compress
""",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-File",
            str(probe),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    value = json.loads(completed.stdout.strip().splitlines()[-1])
    assert value == {
        "telemetry": True,
        "optimizer": True,
        "progress": 1,
        "unsafe": False,
    }


def test_r3_supervisor_stub_sets_cap_confirms_start_evidence_and_stops(tmp_path):
    powershell = shutil.which("powershell.exe") or shutil.which("powershell")
    if powershell is None:
        pytest.skip("Windows PowerShell is unavailable")
    stub_source = tmp_path / "CompShareR3Stub.cs"
    stub_exe = tmp_path / "compshare-r3-stub.exe"
    stub_source.write_text(
        r'''
using System;
using System.IO;
using System.Linq;
using System.Text;

public static class CompShareR3Stub {
    static string Root { get { return Environment.GetEnvironmentVariable("ORACLE_R3_STUB_ROOT"); } }
    static string Marker(string name) { return Path.Combine(Root, name); }
    static string Escape(string value) {
        return value.Replace("\\", "\\\\").Replace("\"", "\\\"")
            .Replace("\r", "\\r").Replace("\n", "\\n");
    }
    static int Ok(string data) { Console.WriteLine("{\"ok\":true,\"data\":" + data + "}"); return 0; }
    static void Log(string[] args) { File.AppendAllText(Marker("commands.log"), string.Join(" ", args) + Environment.NewLine, Encoding.UTF8); }
    public static int Main(string[] original) {
        Log(original);
        var args = original.ToList();
        if (args.Count < 3 || args[0] != "--profile" || args[1] != "default" || args[2] != "--json") return 10;
        args.RemoveRange(0, 3);
        if (args.Count > 0 && args[0] == "--show-sensitive") args.RemoveAt(0);
        if (args.Count < 2 || args[0] != "instance") return 9;
            if (args[1] == "show") {
                bool running = File.Exists(Marker("running"));
                string state = running ? "Running" : "Stopped";
                int gpu = running ? 1 : 0;
                int cpu = running ? 16 : 2;
                int memory = running ? 65536 : 4096;
                return Ok("{\"UHostSet\":[{\"UHostId\":\"uhost-r3-stub\",\"State\":\"" + state
                    + "\",\"ChargeType\":\"Postpay\",\"GpuType\":\"4090\",\"GPU\":" + gpu
                    + ",\"CPU\":" + cpu + ",\"Memory\":" + memory
                    + ",\"MachineType\":\"O\",\"SupportWithoutGpuStart\":true,\"TotalDiskSpace\":200,"
                    + "\"CompShareImageId\":\"image-r3\",\"Region\":\"cn-sh2\",\"Zone\":\"cn-sh2-02\","
                    + "\"DiskSet\":[{\"IsBoot\":\"True\",\"DiskType\":\"CLOUD_SSD\"}],"
                    + "\"InstancePrice\":2.05,\"DiskPrice\":0.04}]}");
            }
            if (args[1] == "price") {
                return Ok("{\"items\":[{\"ChargeType\":\"Postpay\",\"Instance\":2.05,\"SystemDisks\":0.04,\"Disks\":0.04,\"CompShareImage\":null}]}");
            }
        if (args[1] == "schedule" && args[2] == "set") {
            int index = args.IndexOf("--at"); File.WriteAllText(Marker("deadline"), args[index + 1]); return Ok("{}");
        }
        if (args[1] == "schedule" && args[2] == "show") {
            string value = File.ReadAllText(Marker("deadline"));
            return Ok("{\"scheduled\":true,\"scheduler_stop_time\":" + value + "}");
        }
        if (args[1] == "schedule" && args[2] == "cancel") { return Ok("{}"); }
        if (args[1] == "start") { File.WriteAllText(Marker("running"), "1"); return Ok("{}"); }
        if (args[1] == "stop") { if (File.Exists(Marker("running"))) File.Delete(Marker("running")); return Ok("{}"); }
        if (args[1] == "job" && args[2] == "list") { return Ok("{\"items\":[]}"); }
        if (args[1] == "job" && args[2] == "submit") { File.WriteAllText(Marker("submitted"), "1"); return Ok("{}"); }
        if (args[1] == "job" && args[2] == "show") {
            string command = "env V32_ORACLE_R3_RUN_TOKEN=attempt_1 bash ./data/CancerLncAtlas/runtime/prepared/v32_group_shared_oracle_paid_gpu_20260901_r3/code/scripts/server_launch_v32_group_shared_oracle_paid_gpu_20260901_r3.sh --ready-receipt ./data/CancerLncAtlas/runtime/bootstrap/v32_group_shared_oracle_paid_gpu_20260901_r3/AGGREGATE_READY.json";
            return Ok("{\"job\":{\"JobId\":\"v32-oracle-g2f0-s20260726-r3\",\"Name\":\"v32-oracle-g2f0-r3\","
                + "\"Cwd\":\"./data/CancerLncAtlas\",\"Command\":\"" + Escape(command) + "\",\"State\":\"Running\",\"ExitCode\":null}}");
        }
        if (args[1] == "job" && args[2] == "logs") {
            string logs = Environment.GetEnvironmentVariable("ORACLE_R3_STUB_LOGS");
            return Ok("{\"logs\":{\"Stdout\":\"" + Escape(logs) + "\",\"Stderr\":\"\"}}");
        }
        return 8;
    }
}
''',
        encoding="utf-8",
    )
    compile_script = tmp_path / "compile.ps1"
    compile_script.write_text(
        "$ErrorActionPreference='Stop'; "
        f"Add-Type -TypeDefinition (Get-Content -LiteralPath '{stub_source}' -Raw) "
        f"-OutputAssembly '{stub_exe}' -OutputType ConsoleApplication",
        encoding="utf-8",
    )
    compiled = subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-File", str(compile_script)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert compiled.returncode == 0, compiled.stderr

    ready = tmp_path / "AGGREGATE_READY.json"
    ready.write_text(
        json.dumps(
            {
                "format": "CANCERLNCATLAS_V32_ORACLE_AGGREGATE_READY_R3_V1",
                "status": "ORACLE_R3_AGGREGATE_READY_CPU_ONLY",
                "namespace": "v32_group_shared_oracle_paid_gpu_20260901_r3",
                "cpu_only": True,
                "gpu_visible": False,
                "launch_preflight_pass": True,
                "cpu_extraction_complete": True,
                "paid_gpu_archive_rehashing_required": False,
                "paid_gpu_archive_extraction_required": False,
                "prepared_root_path": "./data/CancerLncAtlas/runtime/prepared/v32_group_shared_oracle_paid_gpu_20260901_r3",
                "prepared_member_path": "./data/CancerLncAtlas/runtime/prepared/v32_group_shared_oracle_paid_gpu_20260901_r3/input/G2/PATIENT_FOLD_0.pt",
                "code_root_path": "./data/CancerLncAtlas/runtime/prepared/v32_group_shared_oracle_paid_gpu_20260901_r3/code",
                "config_path": "./data/CancerLncAtlas/runtime/prepared/v32_group_shared_oracle_paid_gpu_20260901_r3/code/config/model_v3_2_group_shared_oracle_paid_gpu_20260901_r3.yaml",
                "task_manifest_path": "./data/CancerLncAtlas/runtime/prepared/v32_group_shared_oracle_paid_gpu_20260901_r3/code/config/v32_group_shared_oracle_paid_gpu_20260901_r3.TASK_MANIFEST.tsv",
                "launcher_path": "./data/CancerLncAtlas/runtime/prepared/v32_group_shared_oracle_paid_gpu_20260901_r3/code/scripts/server_launch_v32_group_shared_oracle_paid_gpu_20260901_r3.sh",
                "python_executable": "/usr/local/miniconda3/envs/py312/bin/python",
                "python_version": "3.12.10",
                "integrity_scope": "INPUT_CODE_RESULT_ARCHIVES_ONLY",
                "per_file_hashing_performed": False,
                "per_fold_hashing_performed": False,
                "input_archive_sha256": INPUT_SHA,
                "code_archive_sha256": CODE_SHA,
                "hard_paid_hours": 3.0,
                "hard_cost_cap_cny": 8.0,
            }
        ),
        encoding="utf-8",
    )
    result_sha = "3" * 64
    result_path = "./data/CancerLncAtlas/runtime/runs/v32_group_shared_oracle_paid_gpu_20260901_r3/attempt_1/ORACLE_G2F0_RESULT_R3.tar"
    common = {
        "format": "CANCERLNCATLAS_V32_ORACLE_AGGREGATE_RUN_R3_V1",
        "aggregate_input_archive_sha256": INPUT_SHA,
        "aggregate_code_archive_sha256": CODE_SHA,
        "integrity_scope": "INPUT_CODE_RESULT_ARCHIVES_ONLY",
        "per_file_hashing_performed": False,
        "per_fold_hashing_performed": False,
    }
    telemetry = {
        "format": common["format"],
        "status": "ORACLE_R3_NVIDIA_SMI_TELEMETRY",
        "gpu_count": 1,
        "gpu_name": "NVIDIA GeForce RTX 4090",
        "gpu_uuid": "GPU-stub",
        "driver_version": "1",
        "memory_total_mib": 24564,
        "telemetry_source": "nvidia-smi",
        "sampled_at": "2026-09-01T00:00:00Z",
    }
    optimizer = {
        **common,
        "format": oracle.RECEIPT_FORMAT,
        "status": "ORACLE_R3_FIRST_OPTIMIZER_STEP",
        "run_id": runner.RUN_ID,
        "task_id": runner.TASK_ID,
        "graph_variant": "G2",
        "patient_fold": 0,
        "seed": 20260726,
        "optimizer_step": 1,
        "grad_finite": True,
        "parameters_finite": True,
        "parameter_delta_positive": True,
        "gpu_identity": {"name": "RTX 4090"},
    }
    result_event = {
        **common,
        "status": "ORACLE_R3_RESULT_ARCHIVE_READY",
        "result_archive_path": result_path,
        "result_archive_sha256": result_sha,
        "result_archive_size_bytes": 10240,
    }
    terminal = {
        **common,
        "status": "ORACLE_R3_PASS",
        "scientific_status": oracle.PASS_STATUS,
        "scientific_pass": True,
        "runner_exit_code": 0,
        "result_archive_path": result_path,
        "result_archive_sha256": result_sha,
        "result_archive_size_bytes": 10240,
    }
    environment = dict(os.environ)
    environment["ORACLE_R3_STUB_ROOT"] = str(tmp_path)
    environment["ORACLE_R3_STUB_LOGS"] = "\n".join(
        json.dumps(item, separators=(",", ":"))
        for item in (telemetry, optimizer, result_event, terminal)
    )
    supervisor_log = tmp_path / "supervisor.log"
    stop_receipt = tmp_path / "STOP.json"
    supervisor = ROOT / "scripts" / "local_supervise_compshare_group_shared_oracle_paid_gpu_20260901_r3.ps1"
    log_guard = ROOT / "scripts" / "v32_group_shared_oracle_log_guard_r3.psm1"
    completed = subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-NonInteractive",
            "-File",
            str(supervisor),
            "-InstanceId",
            "uhost-r3-stub",
            "-AggregateReadyReceiptPath",
            str(ready),
            "-ExpectedInputArchiveSha256",
            INPUT_SHA,
            "-ExpectedCodeArchiveSha256",
            CODE_SHA,
            "-LogPath",
            str(supervisor_log),
            "-StopReceiptPath",
            str(stop_receipt),
            "-CompSharePath",
            str(stub_exe),
            "-LogGuardModulePath",
            str(log_guard),
            "-PollSeconds",
            "5",
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    receipt = json.loads(stop_receipt.read_text(encoding="utf-8-sig"))
    assert receipt["stop_reason"] == "ORACLE_R3_PASS_RESULT_ARCHIVE_READY"
    assert receipt["training_start_confirmed"] is True
    assert receipt["result_archive_sha256"] == result_sha
    assert receipt["observed_total_hourly_cny"] == pytest.approx(2.13)
    commands = (tmp_path / "commands.log").read_text(encoding="utf-8-sig")
    assert commands.index("instance price") < commands.index("instance schedule set")
    assert commands.index("instance schedule set") < commands.index("instance start")
    assert commands.index("instance start") < commands.index("instance job submit")
    assert commands.rindex("instance stop") > commands.index("instance job submit")
    assert "instance schedule cancel uhost-r3-stub --yes" in commands
