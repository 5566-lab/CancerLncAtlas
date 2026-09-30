"""The final G2 transfer reuses an existing cloud A fold when available."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts.rbp_encode_v33 import stage_global_c_cloud_no_rehash_149 as staging


def test_existing_cloud_parent_is_hard_linked_without_transferring_again(tmp_path, monkeypatch):
    fold = 0
    cloud_root = "/root/CancerLncAtlas_C_GLOBAL_G2_20260928/fold_0"
    old_cloud_parent = "/root/old/fold_0/A.pt"
    sources = {}
    files = []
    for role in sorted(staging.ROLES):
        path = tmp_path / role
        path.write_bytes(role.encode())
        sources[role] = path
        row = {"role": role, "source": str(path),
               "destination": f"{cloud_root}/{role}",
               "bytes": path.stat().st_size}
        if role == "source_prepared":
            row["sha256_reused"] = "a" * 64
            row["reuse_cloud_source"] = old_cloud_parent
        files.append(row)
    manifest = tmp_path / "CLOUD_TRANSFER_MANIFEST.json"
    manifest.write_text(json.dumps({
        "format": "CANCERLNCATLAS_C_GLOBAL_G2_FOLD_CLOUD_TRANSFER_V1",
        "preparation_host": "149", "billing_mode": "Postpay",
        "fold": fold, "gpu_target_root": cloud_root, "files": files,
    }))
    authority = tmp_path / "parent.json"
    authority.write_text(json.dumps({
        "status": "PASS_C_ALL_FOLDS_INPUT_SHA256", "target_host": "149",
        "folds": [{"fold": fold, "source_path": str(sources["source_prepared"]),
                   "source_bytes": sources["source_prepared"].stat().st_size,
                   "source_sha256": "a" * 64}],
    }))
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps({
        "format": "CANCERLNCATLAS_C_FOLD_CLOUD_TRANSFER_V2", "fold": fold,
        "files": [{"role": "source_prepared",
                   "source": str(sources["source_prepared"]),
                   "destination": old_cloud_parent,
                   "bytes": sources["source_prepared"].stat().st_size,
                   "sha256": "a" * 64}],
    }))
    overlay = tmp_path / "overlay.json"
    overlay.write_text(json.dumps({
        "status": "PASS_G2_GLOBAL_BINDING_SIDECAR", "target_host": "149",
        "patient_fold": fold, "binding_context_policy": "GLOBAL_PHYSICAL_BINDING",
        "path": str(sources["c_graph_overlay"]),
        "bytes": sources["c_graph_overlay"].stat().st_size,
        "global_binding": str(sources["global_binding_source"]),
    }))
    cpu = tmp_path / "cpu.json"
    cpu.write_text(json.dumps({
        "status": "PASS_G2_GLOBAL_C_CPU_MODEL_PREFLIGHT", "target_host": "149",
        "fold": fold, "graph_overlay_path": str(sources["c_graph_overlay"]),
        "graph_overlay_bytes": sources["c_graph_overlay"].stat().st_size,
    }))
    key = tmp_path / "id_ed25519"
    key.write_text("placeholder")
    key.chmod(0o600)
    receipt = tmp_path / "stage.json"
    commands, copied = [], []

    def fake_remote(_host, _port, _key, command, **_kwargs):
        commands.append(command)
        if command == "hostname":
            return "cloud-guest"
        if command.startswith("df -B1"):
            return str(10**12)
        if command.startswith("test -f"):
            return str(sources["source_prepared"].stat().st_size)
        if command.startswith("test -e"):
            return "missing"
        if command.startswith("stat -c '%d:%i %s'"):
            size = sources["source_prepared"].stat().st_size
            return f"1:2 {size}\n1:2 {size}"
        if command.startswith("stat -c %s"):
            for role, path in sources.items():
                if f"/{role}.partial" in command:
                    return str(path.stat().st_size)
        return ""

    def fake_copy(command, **_kwargs):
        copied.append(command)
        return None

    monkeypatch.setattr(staging.socket, "gethostname", lambda: "149")
    monkeypatch.setattr(staging.stat, "S_IMODE", lambda _mode: 0o600)
    monkeypatch.setattr(staging, "remote", fake_remote)
    monkeypatch.setattr(staging.subprocess, "run", fake_copy)
    monkeypatch.setattr(sys, "argv", [
        "stage", "--manifest", str(manifest),
        "--fold-authority", str(authority),
        "--prior-transfer-manifest", str(prior),
        "--overlay-receipt", str(overlay), "--cpu-receipt", str(cpu),
        "--instance-id", "uhost-known", "--ssh-host", "cpod-test-w3.podtcp.compshare.cn",
        "--ssh-port", "2222", "--identity-file", str(key),
        "--receipt", str(receipt), "--user-directed-no-rehash",
    ])
    assert staging.main() == 0
    assert any(command.startswith("ln -T --") for command in commands)
    assert all(str(sources["source_prepared"]) not in command for command in copied)
    assert set(json.loads(receipt.read_text())["verified_roles"]) == staging.ROLES
