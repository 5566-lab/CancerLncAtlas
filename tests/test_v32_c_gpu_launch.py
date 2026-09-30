from __future__ import annotations

import io
import tarfile
from argparse import Namespace

import pytest

from scripts.rbp_encode_v33 import launch_c_gpu as launcher
from scripts.rbp_encode_v33 import launch_global_c_gpu as global_launcher


def test_gpu_launcher_refuses_cpu_host_149(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher.socket, "gethostname", lambda: "149")
    with pytest.raises(RuntimeError, match="host 149"):
        launcher.run(Namespace(runtime_root=tmp_path))


def test_global_gpu_launcher_refuses_cpu_host_149(tmp_path, monkeypatch):
    monkeypatch.setattr(global_launcher.socket, "gethostname", lambda: "149")
    with pytest.raises(RuntimeError, match="host 149"):
        global_launcher.run(Namespace(runtime_root=tmp_path))


def test_gpu_launcher_extracts_only_regular_archive_entries(tmp_path):
    archive = tmp_path / "good.tar"
    with tarfile.open(archive, "w") as handle:
        data = b"ok"
        entry = tarfile.TarInfo("code/file.txt")
        entry.size = len(data)
        handle.addfile(entry, io.BytesIO(data))
    destination = tmp_path / "good"
    launcher.unpack(archive, destination)
    assert (destination / "code" / "file.txt").read_bytes() == b"ok"

    bad = tmp_path / "bad.tar"
    with tarfile.open(bad, "w") as handle:
        entry = tarfile.TarInfo("../escape.txt")
        entry.size = 1
        handle.addfile(entry, io.BytesIO(b"x"))
    with pytest.raises(RuntimeError, match="Unsafe cloud archive"):
        launcher.unpack(bad, tmp_path / "bad")
    assert not (tmp_path / "escape.txt").exists()
