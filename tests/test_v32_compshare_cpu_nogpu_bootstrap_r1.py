from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from pathlib import Path

import pytest

from scripts import verify_v32_group_shared_oracle_overlay_bundle as overlay_verifier


ROOT = Path(__file__).resolve().parents[1]
CONTROLLER = ROOT / "scripts/local_bootstrap_compshare_cpu_nogpu_20260901_r1.ps1"
REMOTE_DRIVER = ROOT / "scripts/cloud_bootstrap_v32_compshare_cpu_nogpu_20260901_r1.sh"
MANIFEST_VERIFIER = ROOT / "scripts/verify_v32_compshare_cpu_nogpu_bootstrap_manifest.py"
DYNAMIC_VERIFIER = ROOT / "scripts/verify_v32_compshare_cpu_nogpu_dynamic_receipt.py"
JIT_GENERATOR = ROOT / "scripts/local_materialize_compshare_jit_budget_20260901_r2.ps1"
OVERLAY_VERIFIER = ROOT / "scripts/verify_v32_group_shared_oracle_overlay_bundle.py"
FROZEN_DEPLOYMENT_MANIFEST = ROOT / "docs/v32_compshare_cpu_nogpu_bootstrap_manifest_20260901_r1.json"
FROZEN_BUNDLE_ROOT = ROOT / "artifacts/v32_compshare_cpu_nogpu_bootstrap_20260901_r1"
INSTANCE = "uhost-1up504geeqzj"
CPU_NAMESPACE = "v32_compshare_cpu_nogpu_bootstrap_20260901_r1"
ORACLE_NAMESPACE = "v32_group_shared_oracle_paid_gpu_20260901_r2"
REMOTE_ROOT = "./data/CancerLncAtlas"
SCOPE_FORMAT = "CANCERLNCATLAS_FIXED_INSTANCE_PLUS_200GB_BOOT_DISK_V1"
SCOPE_CANONICAL = (
    '{"instance_id":"uhost-1up504geeqzj","instance_name":"CancerLncAtlas-v32-g012-20260831-r1",'
    '"region":"cn-wlcb","system_disk_count":1,"system_disk_drive":"vda",'
    '"system_disk_encrypted":"false","system_disk_id":"bsi-9fd446ee68",'
    '"system_disk_is_boot":"True",'
    '"system_disk_name":"system_disk_CancerLncAtlas-v32-g012-20260831-r1",'
    '"system_disk_size_gb":200,"system_disk_type":"CLOUD_SSD",'
    '"system_disk_usage":"Boot","total_disk_space_gb":200,"zone":"cn-wlcb-01"}'
)
SCOPE_SHA = hashlib.sha256(SCOPE_CANONICAL.encode()).hexdigest()
PROFILE_SHA = hashlib.sha256(b"default").hexdigest()
PREDECESSOR_TX = "cpu-prepare-1788256262-e82ca68ed50b"
PREDECESSOR_MANIFEST_SHA = "8933dfe267713720d7f89ac4bc81826048d998f262ffd4f6a5e85e0e2f40229a"
PREDECESSOR_CONTROLLER_SHA = "3f900f37bf693193e55827d0e5f773b7fbe51b9467eef8e8d645962d2bb59682"
PREDECESSOR_JIT_SHA = "2cf6e64563b34d54a4c12e130d66d0a0fabc2b3839a730e58c7b0a07a9e34f42"
LATEST_PREDECESSOR_TX = "cpu-prepare-1788260502-9c10683eb07b"
LATEST_PREDECESSOR_MANIFEST_SHA = "f488dc047082063eeb4fef8330c77ae13db0b3fa1ae1f7c215bef61d86927384"
LATEST_PREDECESSOR_CONTROLLER_SHA = "3e265b7c0c6679c98c38388c16debb967d6bfd22406991c7fc28eeef9bb68315"
LATEST_PREDECESSOR_JIT_SHA = "004ab542b61361f46a99b61633532e89101462fdc4d1f2923d956e3477e1575f"
TRANSPORT_PREDECESSOR_TX = "cpu-prepare-1788261780-b7fc0e5f8919"
TRANSPORT_PREDECESSOR_MANIFEST_SHA = "f488dc047082063eeb4fef8330c77ae13db0b3fa1ae1f7c215bef61d86927384"
TRANSPORT_PREDECESSOR_CONTROLLER_SHA = "2f65466f4a9dc4eb2885171f32069da52cf1b06dbe34a34fae082e8d55c4cc95"
TRANSPORT_PREDECESSOR_JIT_SHA = "004ab542b61361f46a99b61633532e89101462fdc4d1f2923d956e3477e1575f"
KNOWN_PREDECESSOR_ACTIVE_B64 = {
    PREDECESSOR_TX: (
        "ew0KICAgICJmb3JtYXQiOiAgIkNBTkNFUkxOQ0FUTEFTX0NPTVBTSEFSRV9DUFVfTk9HUFVfVFJBTlNBQ1RJT05fVjEiLA0KICAgICJzdGF0dXMiOiAgIklOX1BST0dSRVNTX0ZBSUxfQ0xPU0VEIiwNCiAgICAicGhhc2UiOiAgIlByZXBhcmVSMiIsDQogICAgInRyYW5zYWN0aW9uX2lkIjogICJjcHUtcHJlcGFyZS0xNzg4MjU2MjYyLWU4MmNhNjhlZDUwYiIsDQogICAgImluc3RhbmNlX2lkIjogICJ1aG9zdC0xdXA1MDRnZWVxemoiLA0KICAgICJwcm9qZWN0X3Njb3BlX2Zvcm1hdCI6ICAiQ0FOQ0VSTE5DQVRMQVNfRklYRURfSU5TVEFOQ0VfUExVU18yMDBHQl9CT09UX0RJU0tfVjEiLA0KICAgICJwcm9qZWN0X3Njb3BlX3NoYTI1NiI6ICAiNzI2NjMxNjYwNjM5ZjYwMDM1Yjc3NzRmOGJlN2UwOTA2MThiZjA4M2JjNDAwYTE1ODBmNDA1MGYwMmE4NjFmZCIsDQogICAgInByb2ZpbGVfbmFtZV9zaGEyNTYiOiAgIjM3YThlZWMxY2UxOTY4N2QxMzJmZTI5MDUxZGNhNjI5ZDE2NGUyYzQ5NThiYTE0MWQ1ZjQxMzNhMzNmMDY4OGYiLA0KICAgICJtYW5pZmVzdF9zaGEyNTYiOiAgIjg5MzNkZmUyNjc3MTM3MjBkN2Y4OWFjNGJjODE4MjYwNDhkOTk4ZjI2MmZmZDRmNmE1ZTg1ZTBlMmY0MDIyOWEiLA0KICAgICJqaXRfZ2VuZXJhdG9yX3NoYTI1NiI6ICAiMmNmNmU2NDU2M2IzNGQ1NGE0YzEyZTEzMGQ2NmQwYTBmYWJjMmIzODM5YTczMGU1OGM3YjBhMDdhOWUzNGY0MiIsDQogICAgImNyZWF0ZWRfYXQiOiAgIjIwMjYtMDktMDFUMDk6NTE6MDIuMzE0MDY0OSswMDowMCIsDQogICAgImdwdV9zdGFydF9wZXJtaXR0ZWQiOiAgZmFsc2UsDQogICAgIndpdGhvdXRfZ3B1X3NwZWMiOiAgIkEiLA0KICAgICJyYXdfcHJvdmlkZXJfcmVzcG9uc2VfZW1iZWRkZWQiOiAgZmFsc2UsDQogICAgImNyZWRlbnRpYWxzX2VtYmVkZGVkIjogIGZhbHNlDQp9"
    ),
    LATEST_PREDECESSOR_TX: (
        "ew0KICAgICJmb3JtYXQiOiAgIkNBTkNFUkxOQ0FUTEFTX0NPTVBTSEFSRV9DUFVfTk9HUFVfVFJBTlNBQ1RJT05fVjEiLA0KICAgICJzdGF0dXMiOiAgIklOX1BST0dSRVNTX0ZBSUxfQ0xPU0VEIiwNCiAgICAicGhhc2UiOiAgIlByZXBhcmVSMiIsDQogICAgInRyYW5zYWN0aW9uX2lkIjogICJjcHUtcHJlcGFyZS0xNzg4MjYwNTAyLTljMTA2ODNlYjA3YiIsDQogICAgImluc3RhbmNlX2lkIjogICJ1aG9zdC0xdXA1MDRnZWVxemoiLA0KICAgICJwcm9qZWN0X3Njb3BlX2Zvcm1hdCI6ICAiQ0FOQ0VSTE5DQVRMQVNfRklYRURfSU5TVEFOQ0VfUExVU18yMDBHQl9CT09UX0RJU0tfVjEiLA0KICAgICJwcm9qZWN0X3Njb3BlX3NoYTI1NiI6ICAiNzI2NjMxNjYwNjM5ZjYwMDM1Yjc3NzRmOGJlN2UwOTA2MThiZjA4M2JjNDAwYTE1ODBmNDA1MGYwMmE4NjFmZCIsDQogICAgInByb2ZpbGVfbmFtZV9zaGEyNTYiOiAgIjM3YThlZWMxY2UxOTY4N2QxMzJmZTI5MDUxZGNhNjI5ZDE2NGUyYzQ5NThiYTE0MWQ1ZjQxMzNhMzNmMDY4OGYiLA0KICAgICJtYW5pZmVzdF9zaGEyNTYiOiAgImY0ODhkYzA0NzA4MjA2M2VlYjRmZWY4MzMwYzc3YWUxM2RiMGIzZmExYWUxZjdjMjE1YmVmNjFkODY5MjczODQiLA0KICAgICJqaXRfZ2VuZXJhdG9yX3NoYTI1NiI6ICAiMDA0YWI1NDJiNjEzNjFmNDZhOTliNjE2MzM1MzJlODkxMDE0NjJmZGM0ZDFmMjkyM2Q5NTZlMzQ3N2UxNTc1ZiIsDQogICAgImNyZWF0ZWRfYXQiOiAgIjIwMjYtMDktMDFUMTE6MDE6NDIuNjYwMjg5NiswMDowMCIsDQogICAgImdwdV9zdGFydF9wZXJtaXR0ZWQiOiAgZmFsc2UsDQogICAgIndpdGhvdXRfZ3B1X3NwZWMiOiAgIkEiLA0KICAgICJyYXdfcHJvdmlkZXJfcmVzcG9uc2VfZW1iZWRkZWQiOiAgZmFsc2UsDQogICAgImNyZWRlbnRpYWxzX2VtYmVkZGVkIjogIGZhbHNlDQp9"
    ),
    TRANSPORT_PREDECESSOR_TX: (
        "ew0KICAgICJmb3JtYXQiOiAgIkNBTkNFUkxOQ0FUTEFTX0NPTVBTSEFSRV9DUFVfTk9HUFVfVFJBTlNBQ1RJT05fVjEiLA0KICAgICJzdGF0dXMiOiAgIklOX1BST0dSRVNTX0ZBSUxfQ0xPU0VEIiwNCiAgICAicGhhc2UiOiAgIlByZXBhcmVSMiIsDQogICAgInRyYW5zYWN0aW9uX2lkIjogICJjcHUtcHJlcGFyZS0xNzg4MjYxNzgwLWI3ZmMwZTVmODkxOSIsDQogICAgImluc3RhbmNlX2lkIjogICJ1aG9zdC0xdXA1MDRnZWVxemoiLA0KICAgICJwcm9qZWN0X3Njb3BlX2Zvcm1hdCI6ICAiQ0FOQ0VSTE5DQVRMQVNfRklYRURfSU5TVEFOQ0VfUExVU18yMDBHQl9CT09UX0RJU0tfVjEiLA0KICAgICJwcm9qZWN0X3Njb3BlX3NoYTI1NiI6ICAiNzI2NjMxNjYwNjM5ZjYwMDM1Yjc3NzRmOGJlN2UwOTA2MThiZjA4M2JjNDAwYTE1ODBmNDA1MGYwMmE4NjFmZCIsDQogICAgInByb2ZpbGVfbmFtZV9zaGEyNTYiOiAgIjM3YThlZWMxY2UxOTY4N2QxMzJmZTI5MDUxZGNhNjI5ZDE2NGUyYzQ5NThiYTE0MWQ1ZjQxMzNhMzNmMDY4OGYiLA0KICAgICJtYW5pZmVzdF9zaGEyNTYiOiAgImY0ODhkYzA0NzA4MjA2M2VlYjRmZWY4MzMwYzc3YWUxM2RiMGIzZmExYWUxZjdjMjE1YmVmNjFkODY5MjczODQiLA0KICAgICJqaXRfZ2VuZXJhdG9yX3NoYTI1NiI6ICAiMDA0YWI1NDJiNjEzNjFmNDZhOTliNjE2MzM1MzJlODkxMDE0NjJmZGM0ZDFmMjkyM2Q5NTZlMzQ3N2UxNTc1ZiIsDQogICAgImNyZWF0ZWRfYXQiOiAgIjIwMjYtMDktMDFUMTE6MjM6MDAuMzM4OTI2MiswMDowMCIsDQogICAgImdwdV9zdGFydF9wZXJtaXR0ZWQiOiAgZmFsc2UsDQogICAgIndpdGhvdXRfZ3B1X3NwZWMiOiAgIkEiLA0KICAgICJyYXdfcHJvdmlkZXJfcmVzcG9uc2VfZW1iZWRkZWQiOiAgZmFsc2UsDQogICAgImNyZWRlbnRpYWxzX2VtYmVkZGVkIjogIGZhbHNlDQp9"
    ),
}
KNOWN_PREDECESSOR_ACTIVE_SHA = {
    PREDECESSOR_TX: "713c28d20b2a09dff952024154fce23ad7fe4efc2a29f24a8efe21cbfbdebd8f",
    LATEST_PREDECESSOR_TX: "e96020743c739657fee8fea98ac9118150724ccfa36d2faeb7f5ee75ffc0fd64",
    TRANSPORT_PREDECESSOR_TX: "23d420cc8eefb0adf9a9003b2541ca691a377303002d431c1469c973d5906089",
}
KNOWN_PREDECESSOR_PROVENANCE = {
    PREDECESSOR_TX: (PREDECESSOR_CONTROLLER_SHA, PREDECESSOR_MANIFEST_SHA),
    LATEST_PREDECESSOR_TX: (
        LATEST_PREDECESSOR_CONTROLLER_SHA,
        LATEST_PREDECESSOR_MANIFEST_SHA,
    ),
    TRANSPORT_PREDECESSOR_TX: (
        TRANSPORT_PREDECESSOR_CONTROLLER_SHA,
        TRANSPORT_PREDECESSOR_MANIFEST_SHA,
    ),
}
REQUIRED_NAMES = [
    "bootstrap_manifest_verifier",
    "dynamic_receipt_verifier",
    "remote_cpu_bootstrap_driver",
    "oracle_overlay_verifier",
    "r1_abort_sealer",
    "r2_input_reuse_materializer",
    "shared_jit_budget_generator",
    "oracle_code_bundle",
    "oracle_code_bundle_manifest",
    "oracle_external_deployment_manifest",
    "oracle_static_verifier",
    "oracle_cpu_authorizer",
    "oracle_paid_launcher",
    "oracle_local_supervisor",
    "oracle_local_dispatcher",
    "oracle_log_guard",
]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_exact_predecessor_active(path: Path, transaction_id: str) -> None:
    raw = base64.b64decode(KNOWN_PREDECESSOR_ACTIVE_B64[transaction_id], validate=True)
    assert len(raw) == 930
    assert hashlib.sha256(raw).hexdigest() == KNOWN_PREDECESSOR_ACTIVE_SHA[transaction_id]
    path.write_bytes(raw)


def _predecessor_active(manifest_sha: str = PREDECESSOR_MANIFEST_SHA) -> dict[str, object]:
    return {
        "format": "CANCERLNCATLAS_COMPSHARE_CPU_NOGPU_TRANSACTION_V1",
        "status": "IN_PROGRESS_FAIL_CLOSED",
        "phase": "PrepareR2",
        "transaction_id": PREDECESSOR_TX,
        "instance_id": INSTANCE,
        "project_scope_format": SCOPE_FORMAT,
        "project_scope_sha256": SCOPE_SHA,
        "profile_name_sha256": PROFILE_SHA,
        "manifest_sha256": manifest_sha,
        "jit_generator_sha256": PREDECESSOR_JIT_SHA,
        "created_at": "2026-09-01T09:51:02.3140649+00:00",
        "gpu_start_permitted": False,
        "without_gpu_spec": "A",
        "raw_provider_response_embedded": False,
        "credentials_embedded": False,
    }


def _powershell() -> str | None:
    return shutil.which("powershell.exe") or shutil.which("powershell")


def _msys_path(path: Path) -> str:
    if os.name == "nt":
        absolute = path.resolve()
        drive = absolute.drive.rstrip(":").lower()
        tail = absolute.as_posix().split(":", 1)[1]
        wsl = subprocess.run(
            ["bash", "-lc", "test -d /mnt/c"],
            capture_output=True,
            timeout=10,
            check=False,
        )
        prefix = "/mnt/" if wsl.returncode == 0 else "/"
        return f"{prefix}{drive}{tail}"
    converted = subprocess.run(
        ["bash", "-lc", 'cygpath -u "$1"', "--", str(path)],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if converted.returncode != 0:
        raise RuntimeError(converted.stderr)
    return converted.stdout.strip()


def _build_manifest(tmp_path: Path) -> tuple[Path, str]:
    fixed = {
        "bootstrap_manifest_verifier": MANIFEST_VERIFIER,
        "dynamic_receipt_verifier": DYNAMIC_VERIFIER,
        "remote_cpu_bootstrap_driver": REMOTE_DRIVER,
        "shared_jit_budget_generator": JIT_GENERATOR,
    }
    used = set(fixed.values())
    candidates = [
        value
        for value in sorted((ROOT / "scripts").glob("*"))
        if value.is_file() and value.stat().st_size > 0 and value not in used
    ]
    rows = []
    for index, name in enumerate(REQUIRED_NAMES):
        local = fixed.get(name)
        if local is None:
            local = candidates.pop(0)
        used.add(local)
        relative = local.relative_to(ROOT).as_posix()
        rows.append(
            {
                "name": name,
                "phase": "both",
                "local_relative_path": relative,
                "stage_relative_path": f"artifacts/{name}/{local.name}",
                "remote_install_path": (
                    f"{REMOTE_ROOT}/runtime/bootstrap/{CPU_NAMESPACE}/stub-install/"
                    f"{index:02d}-{name}-{local.name}"
                ),
                "sha256": _sha(local),
                "size_bytes": local.stat().st_size,
                "mode": 0o700 if local.suffix in {".py", ".sh"} else 0o600,
            }
        )
    manifest = {
        "format": "CANCERLNCATLAS_COMPSHARE_CPU_NOGPU_BOOTSTRAP_MANIFEST_V1",
        "status": "BOOTSTRAP_ARTIFACTS_FROZEN",
        "namespace": CPU_NAMESPACE,
        "instance_id": INSTANCE,
        "remote_project_root": REMOTE_ROOT,
        "gpu_start_permitted": False,
        "without_gpu_spec": "A",
        "formal_r2_code_promotion_permitted": False,
        "formal_r2_patch_ready_permitted": False,
        "artifacts": rows,
    }
    path = tmp_path / "TEST_DEPLOYMENT_MANIFEST.json"
    _write_json(path, manifest)
    return path, _sha(path)


def _compile_stub(tmp_path: Path, powershell: str) -> Path:
    source = tmp_path / "CompShareCpuNoGpuStub.cs"
    executable = tmp_path / "compshare-stub.exe"
    source.write_text(
        r'''
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading;

public static class CompShareCpuNoGpuStub {
    static string Root { get { return Environment.GetEnvironmentVariable("CPU_NOGPU_STUB_ROOT"); } }
    static string RemoteRoot { get { return Path.Combine(Root, "remote"); } }
    static string Marker(string name) { return Path.Combine(Root, name); }
    static long ReadOrCreateUnix(string name, long initial) {
        string path = Marker(name);
        if (!File.Exists(path)) File.WriteAllText(path, initial.ToString(), Encoding.ASCII);
        return long.Parse(File.ReadAllText(path));
    }
    static string Escape(string value) {
        return value.Replace("\\", "\\\\").Replace("\"", "\\\"")
            .Replace("\r", "\\r").Replace("\n", "\\n");
    }
    static void Log(string value) {
        for (int i = 0; i < 100; ++i) {
            try { File.AppendAllText(Marker("commands.log"), value + Environment.NewLine, Encoding.UTF8); return; }
            catch (IOException) { Thread.Sleep(5); }
        }
        throw new IOException("log lock timeout");
    }
    static int Ok(string data) {
        Console.WriteLine("{\"ok\":true,\"schema_version\":\"1\",\"data\":" + data + "}");
        return 0;
    }
    static bool ConsumeTransient(string environmentName, string markerName) {
        int limit;
        if (!int.TryParse(Environment.GetEnvironmentVariable(environmentName), out limit) || limit <= 0) {
            return false;
        }
        string path = Marker(markerName);
        int observed = File.Exists(path) ? int.Parse(File.ReadAllText(path)) : 0;
        if (observed >= limit) return false;
        File.WriteAllText(path, (observed + 1).ToString(), Encoding.ASCII);
        return true;
    }
    static int TransientTransportFailure(bool includeCredentialSource) {
        string credential = includeCredentialSource ? ",\"credential_source\":\"api\"" : "";
        Console.WriteLine("{\"ok\":false,\"schema_version\":\"1\",\"error\":"
            + "{\"code\":\"ssh_failed\",\"message\":\"Connection closed by remote host\","
            + "\"details\":{\"instance\":\"uhost-1up504geeqzj\",\"phase\":\"ssh\","
            + "\"exit_code\":255,\"stdout\":\"\",\"stderr\":\"Connection closed by remote host\\n\""
            + credential + "}}}");
        return 255;
    }
    static int Remote(bool ok, int code, string stdout) {
        if (!ok) {
            Console.WriteLine("{\"ok\":false,\"schema_version\":\"1\",\"error\":"
                + "{\"code\":\"remote_exit_nonzero\",\"message\":\"stub remote failure\"}}");
            return code;
        }
        return Ok("{\"instance\":\"uhost-1up504geeqzj\",\"phase\":\"completed\"," 
            + "\"exit_code\":0,\"stdout\":\"" + Escape(stdout)
            + "\",\"stderr\":\"\",\"credential_source\":\"api\"}");
    }
    static string MapRemote(string remote) {
        if (!remote.StartsWith("/")) throw new InvalidOperationException("remote path must be absolute");
        using (var algorithm = SHA256.Create()) {
            string key = BitConverter.ToString(
                algorithm.ComputeHash(new UTF8Encoding(false).GetBytes(remote))
            ).Replace("-", "").ToLowerInvariant();
            return Path.Combine(RemoteRoot, key);
        }
    }
    static void WriteRemote(string remote, string text) {
        string path = MapRemote(remote);
        Directory.CreateDirectory(Path.GetDirectoryName(path));
        File.WriteAllText(path, text, new UTF8Encoding(false));
    }
    static string Sha(string path) {
        using (var stream = File.OpenRead(path))
        using (var algorithm = SHA256.Create()) {
            return BitConverter.ToString(algorithm.ComputeHash(stream)).Replace("-", "").ToLowerInvariant();
        }
    }
    static string Artifact(string remote) {
        string local = MapRemote(remote);
        return "{\"path\":\"" + Escape(remote) + "\",\"sha256\":\"" + Sha(local)
            + "\",\"size_bytes\":" + new FileInfo(local).Length + "}";
    }
    static int HandleSsh(List<string> args) {
        int separator = args.IndexOf("--");
        if (separator < 0 || separator + 1 >= args.Count) return 81;
        var remote = args.Skip(separator + 1).ToList();
        if (ConsumeTransient("CPU_NOGPU_STUB_TRANSIENT_SSH_FAILURES", "transient-ssh-count")) {
            return TransientTransportFailure(true);
        }
        if (remote[0] == "bash"
            && Environment.GetEnvironmentVariable("CPU_NOGPU_STUB_TRANSIENT_DRIVER") == "1") {
            return TransientTransportFailure(true);
        }
        if (remote[0] == "true" && remote.Count == 1) return Remote(true, 0, "");
        if (remote[0] == "mkdir") {
            int marker = remote.IndexOf("--");
            foreach (string path in remote.Skip(marker + 1)) Directory.CreateDirectory(MapRemote(path));
            return Remote(true, 0, "");
        }
        if (remote[0] == "sha256sum") {
            string path = MapRemote(remote[1]);
            return Remote(true, 0, Sha(path) + "  " + remote[1] + "\n");
        }
        if (remote[0] == "stat") {
            string requested = remote[remote.Count - 1];
            string path = MapRemote(requested);
            return Remote(true, 0, new FileInfo(path).Length + ":1\n");
        }
        if (remote[0] != "bash" || remote.Count != 12) return Remote(false, 82, "");
        string phase = remote[2];
        string transaction = remote[3];
        string staging = remote[4];
        string dynamic = staging + "/dynamic/" + (phase == "prepare_r2" ? "PROVIDER_STOPPED.json" : "JIT_BUDGET_READY.json");
        string dynamicLocal = MapRemote(dynamic);
        if (!File.Exists(dynamicLocal) || Sha(dynamicLocal) != remote[6]
            || new FileInfo(dynamicLocal).Length.ToString() != remote[7]) return Remote(false, 83, "");
        if (Environment.GetEnvironmentVariable("CPU_NOGPU_STUB_FAIL_REMOTE") == "1") {
            return Remote(false, 42, "");
        }
        var artifacts = new List<string>();
        if (phase == "prepare_r2") {
            string aborted = "./data/CancerLncAtlas/runtime/bootstrap/v32_g012_paid_gpu_20260831_r1/ABORTED.json";
            string reuse = "./data/CancerLncAtlas/runtime/bootstrap/v32_g012_paid_gpu_20260901_r2/INPUT_REUSE_READY.json";
            WriteRemote(aborted, "{\"status\":\"ABORTED_NO_GPU\"}\n");
            WriteRemote(reuse, "{\"status\":\"INPUT_REUSE_READY\"}\n");
            artifacts.Add(Artifact(aborted)); artifacts.Add(Artifact(reuse));
        } else if (phase == "authorize_oracle") {
            string auth = "./data/CancerLncAtlas/runtime/bootstrap/v32_group_shared_oracle_paid_gpu_20260901_r2/STATIC_AUTH_READY.json";
            WriteRemote(auth, "{\"status\":\"ORACLE_STATIC_AUTH_READY_CPU_ONLY\"}\n");
            artifacts.Add(Artifact(auth));
        } else return Remote(false, 84, "");
        string result = "./data/CancerLncAtlas/runtime/bootstrap/v32_compshare_cpu_nogpu_bootstrap_20260901_r1/transactions/"
            + transaction + "/REMOTE_RESULT.json";
        string payload = "{\"format\":\"CANCERLNCATLAS_COMPSHARE_CPU_NOGPU_REMOTE_RESULT_V1\","
            + "\"status\":\"CPU_NOGPU_REMOTE_PHASE_COMPLETE\",\"phase\":\"" + phase
            + "\",\"transaction_id\":\"" + transaction + "\",\"gpu_visible\":false,"
            + "\"gpu_start_performed\":false,\"formal_r2_code_promotion_performed\":false,"
            + "\"formal_r2_patch_ready_materialized\":false,\"created_at\":\"2026-09-01T00:00:00+00:00\","
            + "\"phase_elapsed_seconds\":1,\"dynamic_receipt_kind\":\""
            + (phase == "prepare_r2" ? "stopped" : "jit") + "\",\"dynamic_receipt_sha256\":\""
            + remote[6] + "\",\"dynamic_receipt_size_bytes\":" + remote[7] + ",\"artifacts\":["
            + string.Join(",", artifacts.ToArray()) + "]}\n";
        WriteRemote(result, payload);
        return Remote(true, 0, "CPU_NOGPU_REMOTE_PHASE_COMPLETE=" + phase + "\n");
    }
    public static int Main(string[] original) {
        var args = original.ToList();
        Log(string.Join(" ", args));
        if (args.SequenceEqual(new [] { "--profile", "default", "--json", "--version" })) {
            string version = Environment.GetEnvironmentVariable("CPU_NOGPU_STUB_CLI_VERSION") ?? "0.3.6";
            return Ok("{\"version\":\"" + Escape(version) + "\"}");
        }
        if (args.Count < 5 || args[0] != "--profile" || args[1] != "default" || args[2] != "--json") return 90;
        if (args.Contains("--show-sensitive")) return 91;
        args.RemoveRange(0, 3);
        if (args.Count < 2 || args[0] != "instance") return 92;
        if (args[1] == "show") {
            bool running = File.Exists(Marker("running"));
            bool usedNoGpu = File.Exists(Marker("used-no-gpu"));
            bool noGpuA = running || usedNoGpu;
            string mutation = Environment.GetEnvironmentVariable("CPU_NOGPU_STUB_MUTATION") ?? "";
            long now = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
            long create = ReadOrCreateUnix("create-time", now - 3600);
            long stopTime = ReadOrCreateUnix("stop-time", now);
            string gpu = noGpuA ? "0" : "1";
            string gpuType = "4090";
            string cpu = noGpuA ? "2" : "16";
            string memory = noGpuA ? "4096" : "65536";
            string machineType = noGpuA ? "O" : "G";
            string computePrice = noGpuA ? "0.14" : "2.05";
            if (running && mutation == "running-gpu") gpu = "1";
            if (running && mutation == "running-gputype") gpuType = "A100";
            if (running && mutation == "running-cpu") cpu = "4";
            if (running && mutation == "running-memory") memory = "8192";
            if (running && mutation == "running-machine") machineType = "G";
            if (running && mutation == "running-price") computePrice = "0.15";
            if (!running && mutation == "stopped-price") computePrice = noGpuA ? "0.15" : "2.06";
            if (!running && noGpuA && mutation == "posta-gpu") gpu = "1";
            if (!running && noGpuA && mutation == "posta-gputype") gpuType = "A100";
            if (!running && noGpuA && mutation == "posta-cpu") cpu = "4";
            if (!running && noGpuA && mutation == "posta-memory") memory = "8192";
            if (!running && noGpuA && mutation == "posta-machine") machineType = "G";
            if (!running && noGpuA && mutation == "posta-price") computePrice = "0.15";
            string noGpuCapability = mutation == "stopped-capability" || mutation == "posta-capability" ? "false" : "true";
            string host = "{\"UHostId\":\"uhost-1up504geeqzj\",\"Name\":\"CancerLncAtlas-v32-g012-20260831-r1\","
                + "\"Region\":\"cn-wlcb\",\"Zone\":\"cn-wlcb-01\",\"State\":\"" + (running ? "Running" : "Stopped")
                + "\",\"ChargeType\":\"Postpay\",\"InstancePrice\":" + computePrice + ",\"DiskPrice\":0.04,"
                + "\"SupportWithoutGpuStart\":" + noGpuCapability + ",\"CreateTime\":" + create
                + ",\"StartTime\":" + (create + 1) + ",\"StopTime\":" + stopTime
                + ",\"SchedulerStopTime\":" + stopTime + ",\"GPU\":" + gpu
                + ",\"GpuType\":\"" + gpuType + "\",\"CPU\":" + cpu + ",\"Memory\":" + memory
                + ",\"MachineType\":\"" + machineType + "\",\"TotalDiskSpace\":200,\"DiskSet\":[{"
                + "\"Name\":\"system_disk_CancerLncAtlas-v32-g012-20260831-r1\",\"DiskId\":\"bsi-9fd446ee68\","
                + "\"DiskType\":\"CLOUD_SSD\",\"Type\":\"Boot\",\"Size\":200,\"IsBoot\":\"True\","
                + "\"Encrypted\":\"false\",\"Drive\":\"vda\"}]}";
            return Ok("{\"UHostSet\":[" + host + "]}");
        }
        if (args[1] == "schedule" && args[2] == "set") {
            if (args.Contains("--project-id")) return 93;
            int at = args.IndexOf("--at");
            File.WriteAllText(Marker("deadline"), args[at + 1], Encoding.ASCII);
            return Ok("{}");
        }
        if (args[1] == "schedule" && args[2] == "show") {
            string deadline = File.ReadAllText(Marker("deadline"));
            return Ok("{\"instance\":\"uhost-1up504geeqzj\",\"scheduled\":true,\"scheduler_stop_time\":" + deadline + "}");
        }
        if (args[1] == "start") {
            int without = args.IndexOf("--without-gpu");
            if (without < 0 || args[without + 1] != "A") return 94;
            int delayMs;
            if (int.TryParse(Environment.GetEnvironmentVariable("CPU_NOGPU_STUB_START_DELAY_MS"), out delayMs)
                && delayMs > 0 && delayMs <= 10000) Thread.Sleep(delayMs);
            File.WriteAllText(Marker("running"), "1", Encoding.ASCII);
            File.WriteAllText(Marker("used-no-gpu"), "1", Encoding.ASCII);
            return Ok("{}");
        }
        if (args[1] == "stop") {
            if (File.Exists(Marker("running"))) File.Delete(Marker("running"));
            File.WriteAllText(
                Marker("stop-time"), DateTimeOffset.UtcNow.ToUnixTimeSeconds().ToString(), Encoding.ASCII
            );
            return Ok("{}");
        }
        if (args[1] == "cp") {
            if (ConsumeTransient("CPU_NOGPU_STUB_TRANSIENT_CP_FAILURES", "transient-cp-count")) {
                return TransientTransportFailure(false);
            }
            string source = args[3], target = args[4];
            if (source.StartsWith(":")) source = MapRemote(source.Substring(1));
            if (target.StartsWith(":")) target = MapRemote(target.Substring(1));
            Directory.CreateDirectory(Path.GetDirectoryName(target));
            File.Copy(source, target, false);
            return Ok("{\"instance\":\"uhost-1up504geeqzj\",\"phase\":\"completed\"," 
                + "\"exit_code\":0,\"stdout\":\"\",\"stderr\":\"\"}");
        }
        if (args[1] == "ssh") return HandleSsh(args);
        return 95;
    }
}
''',
        encoding="utf-8",
    )
    compile_script = tmp_path / "compile.ps1"
    source_literal = str(source).replace("'", "''")
    executable_literal = str(executable).replace("'", "''")
    compile_script.write_text(
        "$ErrorActionPreference='Stop'; "
        f"Add-Type -TypeDefinition (Get-Content -LiteralPath '{source_literal}' -Raw) "
        f"-OutputAssembly '{executable_literal}' -OutputType ConsoleApplication",
        encoding="utf-8",
    )
    compiled = subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-File", str(compile_script)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stderr
    return executable


def _controller_command(
    powershell: str,
    stub: Path,
    manifest: Path,
    manifest_sha: str,
    state: Path,
    phase: str,
    jit: Path | None = None,
) -> list[str]:
    command = [
        powershell,
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(CONTROLLER),
        "-Phase",
        phase,
        "-Profile",
        "default",
        "-StateDirectory",
        str(state),
        "-CompSharePath",
        str(stub),
        "-LocalPythonPath",
        sys.executable,
        "-ScheduleMinutes",
        "45" if phase == "PrepareR2" else "25",
        "-TestOnlyManifestPath",
        str(manifest),
        "-TestOnlyManifestSha256",
        manifest_sha,
    ]
    if jit is not None:
        command.extend(["-JitBudgetReceiptPath", str(jit)])
    return command


def _run_extracted_controller_functions(
    tmp_path: Path,
    powershell: str,
    function_names: list[str],
    body: str,
) -> subprocess.CompletedProcess[str]:
    controller_literal = str(CONTROLLER).replace("'", "''")
    names_literal = ", ".join(f"'{name}'" for name in function_names)
    harness = tmp_path / "controller-function-unit.ps1"
    harness.write_text(
        f"""
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$tokens = $null
$parseErrors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile(
    '{controller_literal}', [ref]$tokens, [ref]$parseErrors
)
if ($parseErrors.Count -ne 0) {{ throw 'CONTROLLER_PARSE_FAILED' }}
$wanted = @({names_literal})
foreach ($name in $wanted) {{
    $matches = @($ast.FindAll({{
        param($node)
        $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
        [string]$node.Name -ceq $name
    }}, $true))
    if ($matches.Count -ne 1) {{ throw "FUNCTION_CARDINALITY_DRIFT=$name" }}
    . ([scriptblock]::Create([string]$matches[0].Extent.Text))
}}
{body}
""",
        encoding="utf-8",
    )
    return subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-File", str(harness)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
    )


@pytest.mark.skipif(os.name != "nt", reason="PowerShell AST function unit test")
def test_controller_json_contract_functions_without_main_or_mutex(tmp_path: Path) -> None:
    powershell = _powershell()
    if powershell is None:
        pytest.skip("Windows PowerShell is unavailable")
    success = {
        "ok": True,
        "schema_version": "1",
        "data": {
            "instance": INSTANCE,
            "phase": "completed",
            "exit_code": 0,
            "stdout": "ready\n",
            "stderr": "benign ssh diagnostic\n",
            "credential_source": "api",
        },
    }
    copy_success = {
        "ok": True,
        "schema_version": "1",
        "data": {
            "instance": INSTANCE,
            "phase": "completed",
            "exit_code": 0,
            "stdout": "",
            "stderr": "",
        },
    }
    success_b64 = base64.b64encode(json.dumps(success).encode()).decode()
    copy_b64 = base64.b64encode(json.dumps(copy_success).encode()).decode()
    body = f"""
$FixedInstanceId = '{INSTANCE}'
$goodRaw = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{success_b64}'))
$execution = [pscustomobject]@{{ ExitCode = 0; Stdout = $goodRaw }}
$payload = Convert-CompShareJson -Execution $execution -Label 'UNIT'
$data = Assert-CompShareRemoteSuccessEnvelope -Execution $execution -Payload $payload
if ([string]$data.stdout -cne "ready`n") {{ throw 'REMOTE_SUCCESS_NOT_RETURNED' }}
if ([string]$data.stderr -cne "benign ssh diagnostic`n") {{ throw 'REMOTE_STDERR_STRING_NOT_RETURNED' }}

$legacy = $goodRaw | ConvertFrom-Json
$legacy.data | Add-Member -NotePropertyName ok -NotePropertyValue $true
$legacyRejected = $false
try {{ [void](Assert-CompShareRemoteSuccessEnvelope -Execution $execution -Payload $legacy) }}
catch {{
    if ($_.Exception.Message -notlike '*CPU_BOOTSTRAP_REMOTE_DATA_KEY_SET_DRIFT*') {{ throw }}
    $legacyRejected = $true
}}
if (-not $legacyRejected) {{ throw 'LEGACY_INNER_OK_FALSE_PASS' }}

$missingSchema = [pscustomobject]@{{ ok = $true; data = [pscustomobject]@{{}} }}
$missingExecution = [pscustomobject]@{{ ExitCode = 0; Stdout = ($missingSchema | ConvertTo-Json -Compress) }}
$schemaRejected = $false
try {{ [void](Convert-CompShareJson -Execution $missingExecution -Label 'UNIT') }}
catch {{
    if ($_.Exception.Message -notlike '*CPU_BOOTSTRAP_UNIT_SCHEMA_VERSION_DRIFT*') {{ throw }}
    $schemaRejected = $true
}}
if (-not $schemaRejected) {{ throw 'MISSING_SCHEMA_FALSE_PASS' }}

$copyRaw = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{copy_b64}'))
$copyExecution = [pscustomobject]@{{ ExitCode = 0; Stdout = $copyRaw }}
$copyPayload = Convert-CompShareJson -Execution $copyExecution -Label 'COPY_UNIT'
[void](Assert-CompShareCopySuccessEnvelope -Execution $copyExecution -Payload $copyPayload)
'JSON_CONTRACT_UNIT_OK'
"""
    checked = _run_extracted_controller_functions(
        tmp_path,
        powershell,
        [
            "Convert-CompShareJson",
            "Assert-CompShareRemoteSuccessEnvelope",
            "Assert-CompShareCopySuccessEnvelope",
        ],
        body,
    )
    assert checked.returncode == 0, checked.stderr
    assert "JSON_CONTRACT_UNIT_OK" in checked.stdout


@pytest.mark.skipif(os.name != "nt", reason="PowerShell/C# local stub contract test")
def test_compshare_stub_uses_real_outer_success_envelope_without_controller(
    tmp_path: Path,
) -> None:
    powershell = _powershell()
    if powershell is None:
        pytest.skip("Windows PowerShell is unavailable")
    stub = _compile_stub(tmp_path, powershell)
    stub_root = tmp_path / "stub"
    stub_root.mkdir()
    environment = dict(os.environ)
    environment["CPU_NOGPU_STUB_ROOT"] = str(stub_root)
    version = subprocess.run(
        [str(stub), "--profile", "default", "--json", "--version"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        env=environment,
        timeout=10,
        check=False,
    )
    assert version.returncode == 0, version.stderr
    assert json.loads(version.stdout) == {
        "ok": True,
        "schema_version": "1",
        "data": {"version": "0.3.6"},
    }
    remote = subprocess.run(
        [
            str(stub),
            "--profile",
            "default",
            "--json",
            "instance",
            "ssh",
            INSTANCE,
            "--no-cache",
            "--refresh",
            "--timeout",
            "120",
            "--connect-timeout",
            "30",
            "--",
            "mkdir",
            "-p",
            "--",
            "/tmp/contract-unit",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        env=environment,
        timeout=10,
        check=False,
    )
    assert remote.returncode == 0, remote.stderr
    payload = json.loads(remote.stdout)
    assert payload["ok"] is True and payload["schema_version"] == "1"
    assert set(payload["data"]) == {
        "instance",
        "phase",
        "exit_code",
        "stdout",
        "stderr",
        "credential_source",
    }
    assert "ok" not in payload["data"]
    assert payload["data"]["phase"] == "completed"
    assert payload["data"]["credential_source"] == "api"


@pytest.mark.skipif(os.name != "nt", reason="PowerShell AST function unit test")
def test_exact_predecessor_resolver_without_main_or_mutex(tmp_path: Path) -> None:
    powershell = _powershell()
    if powershell is None:
        pytest.skip("Windows PowerShell is unavailable")
    latest_b64 = KNOWN_PREDECESSOR_ACTIVE_B64[LATEST_PREDECESSOR_TX]
    body = f"""
$FixedInstanceId = '{INSTANCE}'
$ProjectScopeFormat = '{SCOPE_FORMAT}'
$KnownPredecessorTransactions = @(
    [pscustomobject]@{{
        TransactionId = '{PREDECESSOR_TX}'
        ControllerSha256 = '{PREDECESSOR_CONTROLLER_SHA}'
        ManifestSha256 = '{PREDECESSOR_MANIFEST_SHA}'
        JitGeneratorSha256 = '{PREDECESSOR_JIT_SHA}'
        CreatedAt = '2026-09-01T09:51:02.3140649+00:00'
        CreatedUnix = [int64]1788256262
        ActiveSha256 = '{KNOWN_PREDECESSOR_ACTIVE_SHA[PREDECESSOR_TX]}'
        ActiveSizeBytes = [int64]930
    }},
    [pscustomobject]@{{
        TransactionId = '{LATEST_PREDECESSOR_TX}'
        ControllerSha256 = '{LATEST_PREDECESSOR_CONTROLLER_SHA}'
        ManifestSha256 = '{LATEST_PREDECESSOR_MANIFEST_SHA}'
        JitGeneratorSha256 = '{LATEST_PREDECESSOR_JIT_SHA}'
        CreatedAt = '2026-09-01T11:01:42.6602896+00:00'
        CreatedUnix = [int64]1788260502
        ActiveSha256 = '{KNOWN_PREDECESSOR_ACTIVE_SHA[LATEST_PREDECESSOR_TX]}'
        ActiveSizeBytes = [int64]930
    }}
)
$raw = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{latest_b64}'))
$prior = $raw | ConvertFrom-Json
$resolved = Resolve-ExactPredecessorContract -Prior $prior `
    -ActiveSha256 '{KNOWN_PREDECESSOR_ACTIVE_SHA[LATEST_PREDECESSOR_TX]}' `
    -ActiveSizeBytes 930 -ExpectedProjectScopeSha '{SCOPE_SHA}' `
    -ExpectedProfileSha '{PROFILE_SHA}'
if ([string]$resolved.TransactionId -cne '{LATEST_PREDECESSOR_TX}') {{
    throw 'LATEST_PREDECESSOR_NOT_RESOLVED'
}}

$driftRejected = $false
try {{
    [void](Resolve-ExactPredecessorContract -Prior $prior -ActiveSha256 ('0' * 64) `
        -ActiveSizeBytes 930 -ExpectedProjectScopeSha '{SCOPE_SHA}' `
        -ExpectedProfileSha '{PROFILE_SHA}')
}}
catch {{
    if ($_.Exception.Message -notlike '*CPU_BOOTSTRAP_PREDECESSOR_ACTIVE_EXACT_ALLOWLIST_DRIFT*') {{ throw }}
    $driftRejected = $true
}}
if (-not $driftRejected) {{ throw 'KNOWN_PREDECESSOR_BYTE_DRIFT_FALSE_PASS' }}

$unknown = $raw | ConvertFrom-Json
$unknown.manifest_sha256 = ('f' * 64)
$unknownResult = Resolve-ExactPredecessorContract -Prior $unknown -ActiveSha256 ('f' * 64) `
    -ActiveSizeBytes 930 -ExpectedProjectScopeSha '{SCOPE_SHA}' `
    -ExpectedProfileSha '{PROFILE_SHA}'
if ($null -ne $unknownResult) {{ throw 'UNKNOWN_PREDECESSOR_FALSE_ALLOWLIST_MATCH' }}
'PREDECESSOR_CONTRACT_UNIT_OK'
"""
    checked = _run_extracted_controller_functions(
        tmp_path,
        powershell,
        ["Resolve-ExactPredecessorContract"],
        body,
    )
    assert checked.returncode == 0, checked.stderr
    assert "PREDECESSOR_CONTRACT_UNIT_OK" in checked.stdout


@pytest.mark.skipif(os.name != "nt", reason="PowerShell 5/CompShare executable controller test")
def test_controller_stub_two_cpu_sessions_are_profile_bound_nogpu_and_finally_stopped(tmp_path: Path) -> None:
    powershell = _powershell()
    if powershell is None:
        pytest.skip("Windows PowerShell is unavailable")
    stub = _compile_stub(tmp_path, powershell)
    manifest, manifest_sha = _build_manifest(tmp_path)
    state = tmp_path / "state"
    stub_root = tmp_path / "stub"
    stub_root.mkdir()
    environment = dict(os.environ)
    environment["CANCERLNCATLAS_COMPSHARE_STUB_E2E"] = "1"
    environment["CPU_NOGPU_STUB_ROOT"] = str(stub_root)

    prepare = subprocess.run(
        _controller_command(powershell, stub, manifest, manifest_sha, state, "PrepareR2"),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
        timeout=90,
        check=False,
    )
    assert prepare.returncode == 0, prepare.stderr
    generator_sha = _sha(JIT_GENERATOR)
    now = datetime.now(timezone.utc)
    queried_unix = int(now.timestamp())
    create_unix = int((stub_root / "create-time").read_text(encoding="ascii"))
    stop_unix = int((stub_root / "stop-time").read_text(encoding="ascii"))
    age_seconds = queried_unix - create_unix
    spend = (Decimal("2.09") * Decimal(age_seconds) / Decimal(3600)).quantize(
        Decimal("0.0001"), rounding=ROUND_CEILING
    )
    reserve = Decimal("5.7000")
    remaining = (Decimal("210") - spend - reserve).quantize(
        Decimal("0.0001"), rounding=ROUND_FLOOR
    )
    unallocated = Decimal("210") - spend - reserve - remaining
    jit = tmp_path / "JIT_BUDGET_READY.json"
    _write_json(
        jit,
        {
            "format": "CANCERLNCATLAS_COMPSHARE_JIT_BUDGET_V1",
            "status": "JIT_BUDGET_READY_CONSERVATIVE",
            "currency": "CNY",
            "source": "COMPSHARE_API_BILLING_SNAPSHOT",
            "instance_id": INSTANCE,
            "project_scope_format": SCOPE_FORMAT,
            "project_scope_sha256": SCOPE_SHA,
            "profile_name_sha256": PROFILE_SHA,
            "generator_sha256": generator_sha,
            "queried_at": now.isoformat(),
            "valid_until": (now + timedelta(minutes=20)).isoformat(),
            "project_budget_cap_cny": 210.0,
            "compute_hourly_cny": 2.05,
            "gpu_hourly_cny": 2.05,
            "disk_hourly_cny": 0.04,
            "total_hourly_cny": 2.09,
            "conservative_remaining_cny": float(remaining),
            "budget_method": "FULL_2_09_CNY_RATE_FOR_EVERY_SECOND_SINCE_INSTANCE_CREATE_MINUS_FUTURE_RESERVE",
            "instance_age_seconds": age_seconds,
            "provider_stop_time_unix": stop_unix,
            "provider_query_duration_seconds": 0.1,
            "worst_case_full_rate_spend_cny": float(spend),
            "future_cleanup_reserve_cny": float(reserve),
            "worst_case_spend_rounding": "CEILING_0_0001_CNY",
            "remaining_rounding": "FLOOR_0_0001_CNY",
            "unallocated_rounding_reserve_cny": float(unallocated),
            "provider_snapshot_sha256": "d" * 64,
        },
    )
    authorize = subprocess.run(
        _controller_command(
            powershell, stub, manifest, manifest_sha, state, "AuthorizeOracle", jit
        ),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
        timeout=90,
        check=False,
    )
    assert authorize.returncode == 0, authorize.stderr
    commands = (stub_root / "commands.log").read_text(encoding="utf-8-sig").splitlines()
    assert commands
    assert all(line.startswith("--profile default --json ") for line in commands)
    assert all("--show-sensitive" not in line for line in commands)
    assert all("--project-id" not in line for line in commands)
    starts = [index for index, line in enumerate(commands) if " instance start " in f" {line} "]
    schedules = [index for index, line in enumerate(commands) if " instance schedule set " in f" {line} "]
    remote_runs = [index for index, line in enumerate(commands) if " instance ssh " in f" {line} " and " -- bash " in f" {line} "]
    stops = [index for index, line in enumerate(commands) if " instance stop " in f" {line} "]
    assert len(starts) == len(schedules) == len(remote_runs) == len(stops) == 2
    assert all("--without-gpu A" in commands[index] for index in starts)
    for schedule, start, remote, stop in zip(schedules, starts, remote_runs, stops):
        assert schedule < start < remote < stop
    assert not (stub_root / "running").exists()
    prepare_complete = json.loads((state / "PREPARE_R2_COMPLETE.json").read_text(encoding="utf-8"))
    assert prepare_complete["project_scope_sha256"] == SCOPE_SHA
    assert prepare_complete["profile_name_sha256"] == PROFILE_SHA
    assert prepare_complete["jit_generator_sha256"] == generator_sha
    completions = list(state.glob("cpu-oracle-*/LOCAL_COMPLETION.json"))
    assert len(completions) == 1
    completion = json.loads(completions[0].read_text(encoding="utf-8"))
    assert completion["status"] == "ORACLE_STATIC_RETURNED_STOPPED_CONFIRMED"
    assert completion["gpu_start_performed"] is False
    assert completion["formal_r2_code_promotion_performed"] is False
    stopped_receipts = sorted(state.glob("cpu-*/PROVIDER_STOPPED.json"))
    assert len(stopped_receipts) == 2
    observed_profiles = [
        json.loads(path.read_text(encoding="utf-8"))["provider_stopped_profile"]
        for path in stopped_receipts
    ]
    assert observed_profiles == [
        "NO_GPU_A_POST_START_STOPPED",
        "NOMINAL_G_GPU_ATTACHED_STOPPED",
    ] or observed_profiles == [
        "NOMINAL_G_GPU_ATTACHED_STOPPED",
        "NO_GPU_A_POST_START_STOPPED",
    ]
    assert (stub_root / "used-no-gpu").is_file()


@pytest.mark.skipif(os.name != "nt", reason="PowerShell 5/CompShare executable controller test")
def test_controller_retries_exact_transient_readiness_and_cp_then_completes(
    tmp_path: Path,
) -> None:
    powershell = _powershell()
    if powershell is None:
        pytest.skip("Windows PowerShell is unavailable")
    stub = _compile_stub(tmp_path, powershell)
    manifest, manifest_sha = _build_manifest(tmp_path)
    state = tmp_path / "state"
    stub_root = tmp_path / "stub"
    stub_root.mkdir()
    environment = dict(os.environ)
    environment.update(
        {
            "CANCERLNCATLAS_COMPSHARE_STUB_E2E": "1",
            "CPU_NOGPU_STUB_ROOT": str(stub_root),
            "CPU_NOGPU_STUB_TRANSIENT_SSH_FAILURES": "1",
            "CPU_NOGPU_STUB_TRANSIENT_CP_FAILURES": "1",
        }
    )
    completed = subprocess.run(
        _controller_command(powershell, stub, manifest, manifest_sha, state, "PrepareR2"),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Connection closed by remote host" not in completed.stdout
    assert "Connection closed by remote host" not in completed.stderr
    commands = (stub_root / "commands.log").read_text(encoding="utf-8-sig").splitlines()
    readiness = [line for line in commands if " instance ssh " in f" {line} " and line.endswith(" -- true")]
    copies = [line for line in commands if " instance cp " in f" {line} "]
    drivers = [line for line in commands if " instance ssh " in f" {line} " and " -- bash " in f" {line} "]
    assert len(readiness) == 3  # failed+successful initial probe, then pre-driver probe
    assert len(copies) >= 2 and copies[0] == copies[1]
    assert len(drivers) == 1
    assert commands.index(readiness[1]) < commands.index(copies[0]) < commands.index(drivers[0])
    assert (stub_root / "transient-ssh-count").read_text(encoding="ascii") == "1"
    assert (stub_root / "transient-cp-count").read_text(encoding="ascii") == "1"
    assert not (stub_root / "running").exists()
    assert (state / "PREPARE_R2_COMPLETE.json").is_file()


@pytest.mark.skipif(os.name != "nt", reason="PowerShell 5/CompShare executable controller test")
def test_controller_persistent_readiness_transport_failure_exhausts_before_staging_and_stops(
    tmp_path: Path,
) -> None:
    powershell = _powershell()
    if powershell is None:
        pytest.skip("Windows PowerShell is unavailable")
    stub = _compile_stub(tmp_path, powershell)
    manifest, manifest_sha = _build_manifest(tmp_path)
    state = tmp_path / "state"
    stub_root = tmp_path / "stub"
    stub_root.mkdir()
    environment = dict(os.environ)
    environment.update(
        {
            "CANCERLNCATLAS_COMPSHARE_STUB_E2E": "1",
            "CPU_NOGPU_STUB_ROOT": str(stub_root),
            "CPU_NOGPU_STUB_TRANSIENT_SSH_FAILURES": "99",
        }
    )
    failed = subprocess.run(
        _controller_command(powershell, stub, manifest, manifest_sha, state, "PrepareR2"),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
        timeout=150,
        check=False,
    )
    assert failed.returncode != 0
    assert "CPU_BOOTSTRAP_REMOTE_TRANSIENT_TRANSPORT_RETRIES_EXHAUSTED=12" in failed.stderr
    assert "Connection closed by remote host" not in failed.stdout
    assert "Connection closed by remote host" not in failed.stderr
    commands = (stub_root / "commands.log").read_text(encoding="utf-8-sig").splitlines()
    readiness = [line for line in commands if " instance ssh " in f" {line} " and line.endswith(" -- true")]
    assert len(readiness) == 12
    assert not any(" instance cp " in f" {line} " for line in commands)
    assert not any(" -- mkdir " in f" {line} " for line in commands)
    assert not any(" -- bash " in f" {line} " for line in commands)
    assert any(" instance stop " in f" {line} " for line in commands)
    assert not (stub_root / "running").exists()
    assert (state / "ACTIVE_TRANSACTION.json").is_file()


@pytest.mark.skipif(os.name != "nt", reason="PowerShell 5/CompShare executable controller test")
def test_controller_never_replays_driver_after_ambiguous_transient_transport_failure(
    tmp_path: Path,
) -> None:
    controller_source = CONTROLLER.read_text(encoding="utf-8")
    assert ")) -TimeoutSeconds $boundedRemoteTimeout -RetryMode 'None')" in controller_source
    assert "$env:COMPSHARE_INSIGHTS_URL = ''" in controller_source
    assert (
        "Wait-CompShareDeadlineBoundedDelay -DelaySeconds 1 "
        "-Label 'COPY_TO_REMOTE_PACING'"
    ) in controller_source
    powershell = _powershell()
    if powershell is None:
        pytest.skip("Windows PowerShell is unavailable")
    stub = _compile_stub(tmp_path, powershell)
    manifest, manifest_sha = _build_manifest(tmp_path)
    state = tmp_path / "state"
    stub_root = tmp_path / "stub"
    stub_root.mkdir()
    environment = dict(os.environ)
    environment.update(
        {
            "CANCERLNCATLAS_COMPSHARE_STUB_E2E": "1",
            "CPU_NOGPU_STUB_ROOT": str(stub_root),
            "CPU_NOGPU_STUB_TRANSIENT_DRIVER": "1",
        }
    )
    failed = subprocess.run(
        _controller_command(powershell, stub, manifest, manifest_sha, state, "PrepareR2"),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
        timeout=90,
        check=False,
    )
    assert failed.returncode != 0
    assert "CPU_BOOTSTRAP_REMOTE_RESPONSE_NOT_OK" in failed.stderr
    assert "TRANSIENT_TRANSPORT_RETRIES_EXHAUSTED" not in failed.stderr
    assert "Connection closed by remote host" not in failed.stdout
    assert "Connection closed by remote host" not in failed.stderr
    commands = (stub_root / "commands.log").read_text(encoding="utf-8-sig").splitlines()
    drivers = [line for line in commands if " instance ssh " in f" {line} " and " -- bash " in f" {line} "]
    assert len(drivers) == 1
    assert any(" instance stop " in f" {line} " for line in commands)
    assert not (stub_root / "running").exists()
    assert (state / "ACTIVE_TRANSACTION.json").is_file()


@pytest.mark.skipif(os.name != "nt", reason="PowerShell 5/CompShare executable controller test")
def test_controller_stub_remote_failure_still_stops_and_leaves_recoverable_journal(tmp_path: Path) -> None:
    powershell = _powershell()
    if powershell is None:
        pytest.skip("Windows PowerShell is unavailable")
    stub = _compile_stub(tmp_path, powershell)
    manifest, manifest_sha = _build_manifest(tmp_path)
    state = tmp_path / "state"
    stub_root = tmp_path / "stub"
    stub_root.mkdir()
    environment = dict(os.environ)
    environment.update(
        {
            "CANCERLNCATLAS_COMPSHARE_STUB_E2E": "1",
            "CPU_NOGPU_STUB_ROOT": str(stub_root),
            "CPU_NOGPU_STUB_FAIL_REMOTE": "1",
        }
    )
    failed = subprocess.run(
        _controller_command(powershell, stub, manifest, manifest_sha, state, "PrepareR2"),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
        timeout=90,
        check=False,
    )
    assert failed.returncode != 0
    assert not (stub_root / "running").exists()
    commands = (stub_root / "commands.log").read_text(encoding="utf-8-sig").splitlines()
    driver_commands = [
        line
        for line in commands
        if " instance ssh " in f" {line} " and " -- bash " in f" {line} "
    ]
    assert len(driver_commands) == 1
    assert "TRANSIENT_TRANSPORT_RETRIES_EXHAUSTED" not in failed.stderr
    start = next(index for index, line in enumerate(commands) if "instance start" in line)
    stop = next(index for index, line in enumerate(commands) if "instance stop" in line)
    assert start < stop
    journal = json.loads((state / "ACTIVE_TRANSACTION.json").read_text(encoding="utf-8"))
    assert journal["status"] == "IN_PROGRESS_FAIL_CLOSED"
    assert journal["project_scope_sha256"] == SCOPE_SHA
    assert journal["gpu_start_permitted"] is False

    # A retry must first recover the ACTIVE transaction while the provider is
    # persistently stopped in exact A, then create a fresh transaction.
    retry_environment = dict(environment)
    retry_environment.pop("CPU_NOGPU_STUB_FAIL_REMOTE")
    recovered = subprocess.run(
        _controller_command(powershell, stub, manifest, manifest_sha, state, "PrepareR2"),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=retry_environment,
        timeout=90,
        check=False,
    )
    assert recovered.returncode == 0, recovered.stderr
    recovery = json.loads(
        (state / f"RECOVERED_{journal['transaction_id']}.json").read_text(encoding="utf-8")
    )
    assert recovery["observed_state"] == "Stopped"
    assert not (state / "ACTIVE_TRANSACTION.json").exists()
    assert not (stub_root / "running").exists()


@pytest.mark.skipif(os.name != "nt", reason="PowerShell 5/CompShare executable controller test")
@pytest.mark.parametrize(
    "predecessor_tx",
    [PREDECESSOR_TX, LATEST_PREDECESSOR_TX, TRANSPORT_PREDECESSOR_TX],
)
def test_controller_recovers_only_the_exact_known_predecessor_before_new_start(
    tmp_path: Path,
    predecessor_tx: str,
) -> None:
    powershell = _powershell()
    if powershell is None:
        pytest.skip("Windows PowerShell is unavailable")
    stub = _compile_stub(tmp_path, powershell)
    manifest, manifest_sha = _build_manifest(tmp_path)
    state = tmp_path / "state"
    state.mkdir()
    old_transaction = state / predecessor_tx
    old_transaction.mkdir()
    (old_transaction / "FAILURE_EVIDENCE.txt").write_text("preserve\n", encoding="utf-8")
    _write_exact_predecessor_active(state / "ACTIVE_TRANSACTION.json", predecessor_tx)
    stub_root = tmp_path / "stub"
    stub_root.mkdir()
    environment = dict(os.environ)
    environment.update(
        {
            "CANCERLNCATLAS_COMPSHARE_STUB_E2E": "1",
            "CPU_NOGPU_STUB_ROOT": str(stub_root),
        }
    )
    completed = subprocess.run(
        _controller_command(powershell, stub, manifest, manifest_sha, state, "PrepareR2"),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
        timeout=90,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    archive = state / f"ACTIVE_TRANSACTION.{predecessor_tx}.predecessor-recovered.json"
    receipt_path = state / f"RECOVERED_{predecessor_tx}.PREDECESSOR.json"
    assert archive.is_file()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected_controller, expected_manifest = KNOWN_PREDECESSOR_PROVENANCE[predecessor_tx]
    assert receipt["predecessor_controller_sha256"] == expected_controller
    assert receipt["predecessor_manifest_sha256"] == expected_manifest
    assert receipt["predecessor_active_sha256"] == KNOWN_PREDECESSOR_ACTIVE_SHA[predecessor_tx]
    assert receipt["predecessor_active_size_bytes"] == 930
    assert receipt["schedule_or_start_performed_during_recovery"] is False
    assert (old_transaction / "FAILURE_EVIDENCE.txt").read_text(encoding="utf-8") == "preserve\n"
    commands = (stub_root / "commands.log").read_text(encoding="utf-8-sig").splitlines()
    assert sum("instance start" in line for line in commands) == 1
    assert not (stub_root / "running").exists()


@pytest.mark.skipif(os.name != "nt", reason="PowerShell 5/CompShare executable controller test")
def test_controller_rejects_unknown_predecessor_manifest_without_schedule_or_start(
    tmp_path: Path,
) -> None:
    powershell = _powershell()
    if powershell is None:
        pytest.skip("Windows PowerShell is unavailable")
    stub = _compile_stub(tmp_path, powershell)
    manifest, manifest_sha = _build_manifest(tmp_path)
    state = tmp_path / "state"
    state.mkdir()
    _write_json(state / "ACTIVE_TRANSACTION.json", _predecessor_active("f" * 64))
    stub_root = tmp_path / "stub"
    stub_root.mkdir()
    environment = dict(os.environ)
    environment.update(
        {
            "CANCERLNCATLAS_COMPSHARE_STUB_E2E": "1",
            "CPU_NOGPU_STUB_ROOT": str(stub_root),
        }
    )
    rejected = subprocess.run(
        _controller_command(powershell, stub, manifest, manifest_sha, state, "PrepareR2"),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
        timeout=90,
        check=False,
    )
    assert rejected.returncode != 0
    assert "CPU_BOOTSTRAP_ACTIVE_TRANSACTION_SCOPE_DRIFT" in rejected.stderr
    commands = (stub_root / "commands.log").read_text(encoding="utf-8-sig").splitlines()
    assert not any("instance schedule" in line for line in commands)
    assert not any("instance start" in line for line in commands)
    assert (state / "ACTIVE_TRANSACTION.json").is_file()


@pytest.mark.skipif(os.name != "nt", reason="PowerShell 5/CompShare executable controller test")
def test_controller_rejects_cli_version_drift_before_provider_contact(tmp_path: Path) -> None:
    powershell = _powershell()
    if powershell is None:
        pytest.skip("Windows PowerShell is unavailable")
    stub = _compile_stub(tmp_path, powershell)
    manifest, manifest_sha = _build_manifest(tmp_path)
    state = tmp_path / "state"
    stub_root = tmp_path / "stub"
    stub_root.mkdir()
    environment = dict(os.environ)
    environment.update(
        {
            "CANCERLNCATLAS_COMPSHARE_STUB_E2E": "1",
            "CPU_NOGPU_STUB_ROOT": str(stub_root),
            "CPU_NOGPU_STUB_CLI_VERSION": "0.3.7",
        }
    )
    rejected = subprocess.run(
        _controller_command(powershell, stub, manifest, manifest_sha, state, "PrepareR2"),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
        timeout=60,
        check=False,
    )
    assert rejected.returncode != 0
    assert "CPU_BOOTSTRAP_COMPSHARE_VERSION_CONTRACT_DRIFT" in rejected.stderr
    commands = (stub_root / "commands.log").read_text(encoding="utf-8-sig").splitlines()
    assert commands == ["--profile default --json --version"]
    assert not state.exists()


@pytest.mark.skipif(os.name != "nt", reason="PowerShell 5/CompShare executable controller test")
@pytest.mark.parametrize(
    ("mutation", "start_expected", "post_a_initial"),
    [
        ("stopped-price", False, False),
        ("stopped-capability", False, False),
        ("running-gpu", True, False),
        ("running-gputype", True, False),
        ("running-cpu", True, False),
        ("running-memory", True, False),
        ("running-machine", True, False),
        ("running-price", True, False),
        ("posta-gpu", False, True),
        ("posta-gputype", False, True),
        ("posta-cpu", False, True),
        ("posta-memory", False, True),
        ("posta-machine", False, True),
        ("posta-price", False, True),
        ("posta-capability", False, True),
    ],
)
def test_controller_stub_rejects_provider_spec_price_capability_or_running_gpu_drift(
    tmp_path: Path, mutation: str, start_expected: bool, post_a_initial: bool
) -> None:
    powershell = _powershell()
    if powershell is None:
        pytest.skip("Windows PowerShell is unavailable")
    stub = _compile_stub(tmp_path, powershell)
    manifest, manifest_sha = _build_manifest(tmp_path)
    state = tmp_path / "state"
    stub_root = tmp_path / "stub"
    stub_root.mkdir()
    if post_a_initial:
        (stub_root / "used-no-gpu").write_text("1", encoding="ascii")
    environment = dict(os.environ)
    environment.update(
        {
            "CANCERLNCATLAS_COMPSHARE_STUB_E2E": "1",
            "CPU_NOGPU_STUB_ROOT": str(stub_root),
            "CPU_NOGPU_STUB_MUTATION": mutation,
        }
    )
    rejected = subprocess.run(
        _controller_command(powershell, stub, manifest, manifest_sha, state, "PrepareR2"),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
        timeout=90,
        check=False,
    )
    assert rejected.returncode != 0
    commands = (stub_root / "commands.log").read_text(encoding="utf-8-sig").splitlines()
    starts = [line for line in commands if "instance start" in line]
    assert bool(starts) is start_expected
    assert not (stub_root / "running").exists()
    if start_expected:
        assert any("instance stop" in line for line in commands)


@pytest.mark.skipif(os.name != "nt", reason="PowerShell 5/CompShare executable JIT test")
def test_jit_generator_accepts_exact_persistent_a_but_budgets_full_209_and_rejects_drift(
    tmp_path: Path,
) -> None:
    powershell = _powershell()
    if powershell is None:
        pytest.skip("Windows PowerShell is unavailable")
    stub = _compile_stub(tmp_path, powershell)
    stub_root = tmp_path / "stub"
    stub_root.mkdir()
    (stub_root / "used-no-gpu").write_text("1", encoding="ascii")
    environment = dict(os.environ)
    environment["CPU_NOGPU_STUB_ROOT"] = str(stub_root)
    output = tmp_path / "JIT_BUDGET_READY.json"
    command = [
        powershell,
        "-NoProfile",
        "-NonInteractive",
        "-File",
        str(JIT_GENERATOR),
        "-OutputPath",
        str(output),
        "-CompSharePath",
        str(stub),
        "-ValidityMinutes",
        "45",
        "-CliTimeoutSeconds",
        "30",
    ]
    generated = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
        timeout=60,
        check=False,
    )
    assert generated.returncode == 0, generated.stderr
    receipt = json.loads(output.read_text(encoding="utf-8-sig"))
    assert receipt["compute_hourly_cny"] == 2.05
    assert receipt["disk_hourly_cny"] == 0.04
    assert receipt["total_hourly_cny"] == 2.09
    assert receipt["future_cleanup_reserve_cny"] == pytest.approx(6.5675)
    assert (
        datetime.fromisoformat(receipt["valid_until"])
        - datetime.fromisoformat(receipt["queried_at"])
    ).total_seconds() == 45 * 60
    assert (stub_root / "commands.log").read_text(encoding="utf-8-sig").splitlines()[0].startswith(
        "--profile default --json instance show"
    )

    for index, mutation in enumerate(
        ("posta-gpu", "posta-gputype", "posta-cpu", "posta-memory", "posta-machine", "posta-price")
    ):
        mutated_environment = dict(environment)
        mutated_environment["CPU_NOGPU_STUB_MUTATION"] = mutation
        mutated_command = list(command)
        mutated_command[mutated_command.index(str(output))] = str(tmp_path / f"reject-{index}.json")
        rejected = subprocess.run(
            mutated_command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=mutated_environment,
            timeout=60,
            check=False,
        )
        assert rejected.returncode != 0, mutation
        assert "JIT_BUDGET_PROVIDER_EXACT_G_OR_A_STOPPED_PROFILE_DRIFT" in rejected.stderr


@pytest.mark.skipif(os.name != "nt", reason="MSYS Bash integration test")
def test_ambient_stub_environment_cannot_relax_manifest_exact_specs(tmp_path: Path) -> None:
    manifest, manifest_sha = _build_manifest(tmp_path)
    environment = dict(os.environ)
    environment["CANCERLNCATLAS_COMPSHARE_STUB_E2E"] = "1"
    rejected = subprocess.run(
        [
            sys.executable,
            str(MANIFEST_VERIFIER),
            "--manifest",
            str(manifest),
            "--expected-manifest-sha256",
            manifest_sha,
            "--source-root",
            str(ROOT),
            "--phase",
            "prepare_r2",
        ],
        capture_output=True,
        text=True,
        env=environment,
        timeout=30,
        check=False,
    )
    assert rejected.returncode == 22
    assert "BOOTSTRAP_ARTIFACT_EXACT_SPEC_DRIFT" in rejected.stderr


def test_dynamic_binding_install_recovers_link_before_partial_unlink(tmp_path: Path) -> None:
    raw = b'{"format":"TEST_BINDING","status":"EXACT"}\n'
    source = tmp_path / "source.json"
    source.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    target = tmp_path / "PREPARE_R2_COMPLETE.json"
    partial = target.with_name(f".{target.name}.{digest}.dynamic.partial")
    partial.write_bytes(raw)
    try:
        os.link(partial, target)
    except OSError as exc:
        pytest.skip(f"hard links unavailable: {exc}")
    assert target.stat().st_nlink == 2
    recovered = subprocess.run(
        [
            sys.executable,
            str(DYNAMIC_VERIFIER),
            "--kind",
            "binding",
            "--source",
            str(source),
            "--expected-sha256",
            digest,
            "--expected-size",
            str(len(raw)),
            "--install-target",
            str(target),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert recovered.returncode == 0, recovered.stderr
    assert target.read_bytes() == raw
    assert target.stat().st_nlink == 1
    assert not partial.exists()


def test_frozen_deployment_and_oracle_archive_verify_from_final_bytes(tmp_path: Path) -> None:
    deployment_sha = _sha(FROZEN_DEPLOYMENT_MANIFEST)
    controller_source = CONTROLLER.read_text(encoding="utf-8")
    assert f"$LockedManifestSha256 = '{deployment_sha}'" in controller_source
    for phase in ("prepare_r2", "authorize_oracle"):
        checked = subprocess.run(
            [
                sys.executable,
                str(MANIFEST_VERIFIER),
                "--manifest",
                str(FROZEN_DEPLOYMENT_MANIFEST),
                "--expected-manifest-sha256",
                deployment_sha,
                "--source-root",
                str(ROOT),
                "--phase",
                phase,
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert checked.returncode == 0, checked.stderr

    bundle_manifest = FROZEN_BUNDLE_ROOT / "ORACLE_CODE_BUNDLE.MANIFEST.json"
    archive = FROZEN_BUNDLE_ROOT / "ORACLE_CODE_BUNDLE.tar.gz"
    bundle = json.loads(bundle_manifest.read_text(encoding="utf-8"))
    extracted = subprocess.run(
        [
            sys.executable,
            str(OVERLAY_VERIFIER),
            "--manifest",
            str(bundle_manifest),
            "--expected-manifest-sha256",
            _sha(bundle_manifest),
            "--expected-archive-sha256",
            _sha(archive),
            "--archive",
            str(archive),
            "--staging-root",
            str(tmp_path / "extracted"),
        ],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert extracted.returncode == 0, extracted.stderr
    observed = json.loads(extracted.stdout)
    assert observed["status"] == "ORACLE_CODE_BUNDLE_INSTALLED_VERIFIED"
    assert observed["archive_sha256"] == bundle["archive_sha256"] == _sha(archive)


def test_overlay_member_mode_skips_windows_and_remains_fd_bound_on_posix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int]] = []
    monkeypatch.setattr(
        overlay_verifier.os,
        "fchmod",
        lambda descriptor, mode: calls.append((descriptor, mode)),
        raising=False,
    )

    monkeypatch.setattr(overlay_verifier, "_IS_WINDOWS", True)
    overlay_verifier._copy_member_mode_for_platform(17, 0o640)
    assert calls == []

    monkeypatch.setattr(overlay_verifier, "_IS_WINDOWS", False)
    overlay_verifier._copy_member_mode_for_platform(19, 0o600)
    assert calls == [(19, 0o600)]


def test_controller_rejects_time_envelope_over_by_one_before_provider_contact(
    tmp_path: Path,
) -> None:
    powershell = _powershell()
    if powershell is None:
        pytest.skip("Windows PowerShell is unavailable")
    stub = _compile_stub(tmp_path, powershell)
    manifest, manifest_sha = _build_manifest(tmp_path)
    state = tmp_path / "state"
    stub_root = tmp_path / "stub"
    stub_root.mkdir()
    environment = dict(os.environ)
    environment.update(
        {
            "CANCERLNCATLAS_COMPSHARE_STUB_E2E": "1",
            "CPU_NOGPU_STUB_ROOT": str(stub_root),
        }
    )
    command = _controller_command(
        powershell, stub, manifest, manifest_sha, state, "PrepareR2"
    )
    command.extend(["-RemotePhaseTimeoutSeconds", "1801"])
    rejected = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=environment,
        timeout=60,
        check=False,
    )
    assert rejected.returncode != 0
    assert "CPU_BOOTSTRAP_REMOTE_TIMEOUT_INVALID" in rejected.stderr
    assert not (stub_root / "commands.log").exists()


def test_real_remote_driver_promotes_transaction_jit_and_archives_failed_attempt_on_retry(
    tmp_path: Path,
) -> None:
    if shutil.which("bash") is None:
        pytest.skip("MSYS Bash is unavailable")
    test_root = tmp_path / "cpu_nogpu_driver_test_root"
    test_root.mkdir()
    posix_root = _msys_path(test_root)
    python_query = subprocess.run(
        ["bash", "-lc", "command -v python3"],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if python_query.returncode != 0 or not python_query.stdout.strip():
        pytest.skip("MSYS python3 is unavailable")
    python_bin = python_query.stdout.strip()

    fake_installer = test_root / "fake_manifest_verifier.py"
    fake_dynamic = test_root / "fake_dynamic_verifier.py"
    fake_authorizer = test_root / "fake_authorizer.sh"
    fake_nvidia = test_root / "fake-bin/nvidia-smi"
    fake_nvidia.parent.mkdir()
    fake_installer.write_bytes(
        b'''import os, shutil, sys\nfrom pathlib import Path\nroot = Path(os.environ["CANCERLNCATLAS_CPU_BOOTSTRAP_TEST_PROJECT_ROOT"])\nargs = sys.argv[1:]\nstage = Path(args[args.index("--staging-root") + 1])\ncpu = root / "runtime/bootstrap/v32_compshare_cpu_nogpu_bootstrap_20260901_r1"\noracle = root / "runtime/oracle_gates/v32_group_shared_oracle_paid_gpu_20260901_r2"\nitems = [\n (stage / "artifacts/bootstrap_manifest_verifier/verify_v32_compshare_cpu_nogpu_bootstrap_manifest.py", cpu / "verify_v32_compshare_cpu_nogpu_bootstrap_manifest.py"),\n (stage / "artifacts/dynamic_receipt_verifier/verify_v32_compshare_cpu_nogpu_dynamic_receipt.py", cpu / "verify_v32_compshare_cpu_nogpu_dynamic_receipt.py"),\n (stage / "artifacts/remote_cpu_bootstrap_driver/cloud_bootstrap_v32_compshare_cpu_nogpu_20260901_r1.sh", cpu / "cloud_bootstrap_v32_compshare_cpu_nogpu_20260901_r1.sh"),\n (stage / "artifacts/oracle_cpu_authorizer/cloud_authorize_v32_group_shared_oracle_no_gpu_20260901_r2.sh", oracle / "cloud_authorize_v32_group_shared_oracle_no_gpu_20260901_r2.sh"),\n]\nfor source, target in items:\n target.parent.mkdir(parents=True, exist_ok=True)\n shutil.copyfile(source, target)\nprint("{\\"status\\":\\"FAKE_STATIC_INSTALL_COMPLETE\\"}")\n'''
    )
    fake_dynamic.write_bytes(
        b'''import argparse, hashlib, os, shutil\nfrom pathlib import Path\np = argparse.ArgumentParser()\np.add_argument("--kind", required=True); p.add_argument("--source", type=Path, required=True)\np.add_argument("--expected-sha256", required=True); p.add_argument("--expected-size", type=int, required=True)\np.add_argument("--transaction-id", required=True); p.add_argument("--project-scope-sha256", required=True)\np.add_argument("--profile-name-sha256", required=True); p.add_argument("--expected-generator-sha256", required=True)\na = p.parse_args()\nraw = a.source.read_bytes()\nassert a.kind == "jit" and hashlib.sha256(raw).hexdigest() == a.expected_sha256 and len(raw) == a.expected_size\nroot = Path(os.environ["CANCERLNCATLAS_CPU_BOOTSTRAP_TEST_PROJECT_ROOT"])\ntarget = root / "runtime/bootstrap/v32_compshare_cpu_nogpu_bootstrap_20260901_r1/transactions" / a.transaction_id / "JIT_BUDGET_READY.json"\ntarget.parent.mkdir(parents=True, exist_ok=True)\nwith target.open("xb") as handle: handle.write(raw)\nprint("{\\"status\\":\\"FAKE_DYNAMIC_INSTALLED\\"}")\n'''
    )
    fake_authorizer.write_bytes(
        b'''#!/usr/bin/env bash\nset -euo pipefail\nroot=$CANCERLNCATLAS_CPU_BOOTSTRAP_TEST_PROJECT_ROOT\nfixed="$root/runtime/bootstrap/v32_group_shared_oracle_paid_gpu_20260901_r2/JIT_BUDGET_READY.json"\ntest -s "$fixed"\nif test "${CPU_NOGPU_DRIVER_FAKE_AUTHORIZE_FAIL:-}" = 1; then exit 20; fi\nstatic="$root/runtime/bootstrap/v32_group_shared_oracle_paid_gpu_20260901_r2/STATIC_AUTH_READY.json"\ntest ! -e "$static"\njit_sha="$(sha256sum "$fixed" | awk '{print $1}')"\nvalid_until="$("$V32_ORACLE_GATE_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["valid_until"])' "$fixed")"\nprintf '{"format":"CC_HHGT_V3_2_GROUP_SHARED_ORACLE_R2_STATIC_AUTH_V1","status":"ORACLE_STATIC_AUTH_READY_CPU_ONLY","namespace":"v32_group_shared_oracle_paid_gpu_20260901_r2","cpu_only_materialized":true,"gpu_visible_during_materialization":false,"comparison_only":true,"formal_training_authorized":false,"formal_artifacts_allowed":false,"checkpoint_allowed":false,"prediction_allowed":false,"instance_id":"uhost-1up504geeqzj","jit_budget_receipt_sha256":"%s","jit_valid_until":"%s"}\\n' "$jit_sha" "$valid_until" > "$static"\nif test "${CPU_NOGPU_DRIVER_FAKE_AUTHORIZE_FAIL_AFTER_STATIC:-}" = 1; then exit 21; fi\n'''
    )
    fake_nvidia.write_bytes(b"#!/usr/bin/env bash\nexit 0\n")

    manifest = test_root / "DEPLOYMENT_MANIFEST.json"
    _write_json(
        manifest,
        {
            "artifacts": [
                {
                    "name": "bootstrap_manifest_verifier",
                    "phase": "both",
                    "local_relative_path": (
                        "scripts/verify_v32_compshare_cpu_nogpu_bootstrap_manifest.py"
                    ),
                    "stage_relative_path": (
                        "artifacts/bootstrap_manifest_verifier/"
                        "verify_v32_compshare_cpu_nogpu_bootstrap_manifest.py"
                    ),
                    "remote_install_path": (
                        "./data/CancerLncAtlas/runtime/bootstrap/"
                        "v32_compshare_cpu_nogpu_bootstrap_20260901_r1/"
                        "verify_v32_compshare_cpu_nogpu_bootstrap_manifest.py"
                    ),
                    "sha256": _sha(fake_installer),
                    "size_bytes": fake_installer.stat().st_size,
                    "mode": 0o700,
                },
                {
                    "name": "dynamic_receipt_verifier",
                    "sha256": _sha(fake_dynamic),
                    "size_bytes": fake_dynamic.stat().st_size,
                },
                {
                    "name": "oracle_cpu_authorizer",
                    "sha256": _sha(fake_authorizer),
                    "size_bytes": fake_authorizer.stat().st_size,
                },
            ]
        },
    )
    manifest_sha = _sha(manifest)
    environment = dict(os.environ)
    environment.update(
        {
            "CANCERLNCATLAS_CPU_BOOTSTRAP_DRIVER_TEST_ONLY": "1",
            "CANCERLNCATLAS_CPU_BOOTSTRAP_TEST_PROJECT_ROOT": posix_root,
            "CANCERLNCATLAS_CPU_BOOTSTRAP_TEST_FAKE_BIN": _msys_path(fake_nvidia.parent),
        }
    )
    if os.name == "nt" and posix_root.startswith("/mnt/"):
        imported = [
            "CANCERLNCATLAS_CPU_BOOTSTRAP_DRIVER_TEST_ONLY/u",
            "CANCERLNCATLAS_CPU_BOOTSTRAP_TEST_PROJECT_ROOT/u",
            "CANCERLNCATLAS_CPU_BOOTSTRAP_TEST_FAKE_BIN/u",
            "CPU_NOGPU_DRIVER_FAKE_AUTHORIZE_FAIL/u",
            "CPU_NOGPU_DRIVER_FAKE_AUTHORIZE_FAIL_AFTER_STATIC/u",
        ]
        existing_wslenv = environment.get("WSLENV", "")
        environment["WSLENV"] = ":".join(
            ([existing_wslenv] if existing_wslenv else []) + imported
        )
    fake_authorizer.chmod(fake_authorizer.stat().st_mode | 0o111)
    fake_nvidia.chmod(fake_nvidia.stat().st_mode | 0o111)

    def stage(transaction: str, payload: bytes) -> tuple[str, str, int]:
        staging = test_root / "staging" / transaction
        (staging / "artifacts/bootstrap_manifest_verifier").mkdir(parents=True)
        (staging / "artifacts/dynamic_receipt_verifier").mkdir(parents=True)
        (staging / "artifacts/remote_cpu_bootstrap_driver").mkdir(parents=True)
        (staging / "artifacts/oracle_cpu_authorizer").mkdir(parents=True)
        (staging / "dynamic").mkdir()
        shutil.copyfile(
            fake_installer,
            staging
            / "artifacts/bootstrap_manifest_verifier/verify_v32_compshare_cpu_nogpu_bootstrap_manifest.py",
        )
        shutil.copyfile(
            fake_dynamic,
            staging
            / "artifacts/dynamic_receipt_verifier/verify_v32_compshare_cpu_nogpu_dynamic_receipt.py",
        )
        shutil.copyfile(
            REMOTE_DRIVER,
            staging
            / "artifacts/remote_cpu_bootstrap_driver/cloud_bootstrap_v32_compshare_cpu_nogpu_20260901_r1.sh",
        )
        shutil.copyfile(
            fake_authorizer,
            staging
            / "artifacts/oracle_cpu_authorizer/cloud_authorize_v32_group_shared_oracle_no_gpu_20260901_r2.sh",
        )
        shutil.copyfile(manifest, staging / "DEPLOYMENT_MANIFEST.json")
        dynamic = staging / "dynamic/JIT_BUDGET_READY.json"
        dynamic.write_bytes(payload)
        return _msys_path(staging), hashlib.sha256(payload).hexdigest(), len(payload)

    def direct_jit_payload(attempt: int) -> bytes:
        return (
            json.dumps(
                {
                    "attempt": attempt,
                    "format": "CANCERLNCATLAS_COMPSHARE_JIT_BUDGET_V1",
                    "status": "JIT_BUDGET_READY_CONSERVATIVE",
                    "valid_until": (
                        datetime.now(timezone.utc) + timedelta(minutes=20)
                    ).isoformat(),
                },
                sort_keys=True,
            )
            + "\n"
        ).encode()

    first_tx = "cpu-oracle-1788252000-111111111111"
    first_payload = direct_jit_payload(1)
    first_stage, first_sha, first_size = stage(first_tx, first_payload)
    first_environment = dict(environment)
    first_environment["CPU_NOGPU_DRIVER_FAKE_AUTHORIZE_FAIL"] = "1"
    common = [SCOPE_SHA, PROFILE_SHA, "c" * 64, python_bin]
    first = subprocess.run(
        [
            "bash",
            _msys_path(REMOTE_DRIVER),
            "authorize_oracle",
            first_tx,
            first_stage,
            manifest_sha,
            first_sha,
            str(first_size),
            *common,
        ],
        capture_output=True,
        text=True,
        env=first_environment,
        timeout=60,
        check=False,
    )
    assert first.returncode != 0
    oracle_bootstrap = test_root / f"runtime/bootstrap/{ORACLE_NAMESPACE}"
    fixed = oracle_bootstrap / "JIT_BUDGET_READY.json"
    assert fixed.exists(), f"stdout={first.stdout}\nstderr={first.stderr}"
    assert fixed.read_bytes() == first_payload
    assert not (oracle_bootstrap / "STATIC_AUTH_READY.json").exists()

    second_tx = "cpu-oracle-1788252001-222222222222"
    second_payload = direct_jit_payload(2)
    second_stage, second_sha, second_size = stage(second_tx, second_payload)
    second = subprocess.run(
        [
            "bash",
            _msys_path(REMOTE_DRIVER),
            "authorize_oracle",
            second_tx,
            second_stage,
            manifest_sha,
            second_sha,
            str(second_size),
            *common,
        ],
        capture_output=True,
        text=True,
        env=environment,
        timeout=60,
        check=False,
    )
    assert second.returncode == 0, second.stderr
    assert fixed.read_bytes() == second_payload
    archived = oracle_bootstrap / f"jit_history/{second_tx}/PREVIOUS_JIT_BUDGET_READY.json"
    assert archived.read_bytes() == first_payload
    transaction_jit = (
        test_root
        / f"runtime/bootstrap/{CPU_NAMESPACE}/transactions/{second_tx}/JIT_BUDGET_READY.json"
    )
    assert transaction_jit.read_bytes() == second_payload
    result = json.loads(
        (
            test_root
            / f"runtime/bootstrap/{CPU_NAMESPACE}/transactions/{second_tx}/REMOTE_RESULT.json"
        ).read_text(encoding="utf-8")
    )
    assert result["gpu_start_performed"] is False
    assert result["formal_r2_code_promotion_performed"] is False
    assert result["formal_r2_patch_ready_materialized"] is False
    assert result["dynamic_receipt_kind"] == "jit"
    assert result["dynamic_receipt_sha256"] == second_sha

    third_tx = "cpu-oracle-1788252002-333333333333"
    third_payload = direct_jit_payload(3)
    third_stage, third_sha, third_size = stage(third_tx, third_payload)
    third_environment = dict(environment)
    third_environment["CPU_NOGPU_DRIVER_FAKE_AUTHORIZE_FAIL_AFTER_STATIC"] = "1"
    third = subprocess.run(
        [
            "bash",
            _msys_path(REMOTE_DRIVER),
            "authorize_oracle",
            third_tx,
            third_stage,
            manifest_sha,
            third_sha,
            str(third_size),
            *common,
        ],
        capture_output=True,
        text=True,
        env=third_environment,
        timeout=60,
        check=False,
    )
    assert third.returncode != 0
    assert fixed.read_bytes() == third_payload
    third_static = json.loads((oracle_bootstrap / "STATIC_AUTH_READY.json").read_text())
    assert third_static["jit_budget_receipt_sha256"] == third_sha
    archived_second_pair = oracle_bootstrap / f"authorization_history/{third_tx}"
    assert (archived_second_pair / "PRIOR_JIT_BUDGET_READY.json").read_bytes() == second_payload
    assert (archived_second_pair / "PRIOR_STATIC_AUTH_READY.json").is_file()

    fourth_tx = "cpu-oracle-1788252003-444444444444"
    fourth_payload = direct_jit_payload(4)
    fourth_stage, fourth_sha, fourth_size = stage(fourth_tx, fourth_payload)
    fourth = subprocess.run(
        [
            "bash",
            _msys_path(REMOTE_DRIVER),
            "authorize_oracle",
            fourth_tx,
            fourth_stage,
            manifest_sha,
            fourth_sha,
            str(fourth_size),
            *common,
        ],
        capture_output=True,
        text=True,
        env=environment,
        timeout=60,
        check=False,
    )
    assert fourth.returncode == 0, fourth.stderr
    assert fixed.read_bytes() == fourth_payload
    archived_third_pair = oracle_bootstrap / f"authorization_history/{fourth_tx}"
    assert (archived_third_pair / "PRIOR_JIT_BUDGET_READY.json").read_bytes() == third_payload
    fourth_result = json.loads(
        (
            test_root
            / f"runtime/bootstrap/{CPU_NAMESPACE}/transactions/{fourth_tx}/REMOTE_RESULT.json"
        ).read_text(encoding="utf-8")
    )
    assert fourth_result["dynamic_receipt_sha256"] == fourth_sha
