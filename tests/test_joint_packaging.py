from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import zipfile

from cla_engine.common import code_digest, digest
from cla_engine.transport import make_code_tar


ROOT = Path(__file__).resolve().parents[1]
CODE_DIRS = {"cla_engine", "cc_hhgt", "scripts", "config"}
CODE_SUFFIXES = {".py", ".r", ".sh", ".yaml", ".yml", ".json"}
SOURCE_MANIFEST = "provenance/joint_mainline_20261101/SOURCE_MANIFEST.json"


def _isolated_import(root: Path) -> None:
    program = f"""
import pathlib, sys
root = pathlib.Path({str(root)!r}).resolve()
sys.path.insert(0, str(root))
import cla, cla_engine.transport, cc_hhgt
for module in (cla, cla_engine.transport, cc_hhgt):
    assert pathlib.Path(module.__file__).resolve().is_relative_to(root), module.__file__
assert callable(cla.main)
assert 'backend-check' in cla.parser()._subparsers._group_actions[0].choices
"""
    subprocess.run(
        [sys.executable, "-I", "-c", program],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    )


def test_code_tar_is_unified_digest_bound_and_importable(tmp_path):
    archive = tmp_path / "code.tar"
    make_code_tar(ROOT, archive)
    with tarfile.open(archive) as capsule:
        members = capsule.getmembers()
        names = {member.name for member in members}
        assert "cla.py" in names
        assert "cla_engine/transport.py" in names
        assert "cc_hhgt/gnn.py" in names
        assert "config/149.json" in names
        assert any(name.startswith("scripts/") for name in names)
        assert not any(name.startswith("code/") for name in names)
        assert not any("gnn_full_r1.py" in name for name in names)
        assert all(not member.issym() and not member.islnk() for member in members)
        assert all(not Path(member.name).is_absolute() and ".." not in Path(member.name).parts for member in members)

        archived = {}
        for member in members:
            path = Path(member.name)
            if not member.isfile():
                continue
            if member.name == "cla.py" or (
                path.parts[0] in CODE_DIRS and path.suffix.lower() in CODE_SUFFIXES
            ):
                archived[member.name] = hashlib.sha256(
                    capsule.extractfile(member).read()
                ).hexdigest()
        assert digest(archived) == code_digest()
        if (ROOT / SOURCE_MANIFEST).is_file():
            assert SOURCE_MANIFEST in names

        extracted = tmp_path / "extracted"
        extracted.mkdir()
        capsule.extractall(extracted)
    _isolated_import(extracted)
    if (ROOT / SOURCE_MANIFEST).is_file():
        from cla_engine.package import verify
        assert verify(extracted)['state']=='PACKAGE_HASH_CHECK_PASS_NOT_TRAINED'


def test_code_tar_carries_current_manifest_without_historical_snapshot(tmp_path):
    package = tmp_path / "package"
    (package / "cla_engine").mkdir(parents=True)
    (package / "provenance/joint_mainline_20261101").mkdir(parents=True)
    (package / "cla.py").write_text("def main(): return 0\n", encoding="utf-8")
    (package / "cla_engine/__init__.py").write_text("", encoding="utf-8")
    from cla_engine.package import source_files
    record={'format':'CLA_JOINT_MAINLINE_SOURCE_V1', 'files':{
        name:{'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
        for name,path in source_files(package).items()}}
    (package / SOURCE_MANIFEST).write_text(json.dumps(record)+"\n", encoding="utf-8")
    historical = package / "provenance/gnn_full_r1.py"
    historical.parent.mkdir(parents=True, exist_ok=True)
    historical.write_text("raise AssertionError('must not ship')\n", encoding="utf-8")

    archive = tmp_path / "fixture.tar"
    make_code_tar(package, archive)
    with tarfile.open(archive) as capsule:
        assert capsule.getnames() == [
            "cla.py",
            "cla_engine/__init__.py",
            SOURCE_MANIFEST,
        ]


def test_wheel_contains_joint_engine_module_and_canonical_entrypoint(tmp_path):
    source = tmp_path / "wheel_source"
    source.mkdir()
    shutil.copy2(ROOT / "pyproject.toml", source / "pyproject.toml")
    shutil.copy2(ROOT / "cla.py", source / "cla.py")
    for package in ("cla_engine", "cc_hhgt"):
        shutil.copytree(
            ROOT / package,
            source / package,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
        )

    wheel_dir = tmp_path / "wheelhouse"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheel_dir),
            str(source),
        ],
        check=True,
        text=True,
        capture_output=True,
        env={**os.environ, "PIP_DISABLE_PIP_VERSION_CHECK": "1"},
    )
    wheels = list(wheel_dir.glob("*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as wheel:
        names = set(wheel.namelist())
        assert "cla.py" in names
        assert "cla_engine/backend_check.py" in names
        assert "cc_hhgt/gnn.py" in names
        entry_name = next(name for name in names if name.endswith(".dist-info/entry_points.txt"))
        entry_points = wheel.read(entry_name).decode("utf-8")
        assert "cancerlncatlas = cla:main" in entry_points
        assert "cchhgt-v32 = cla:main" in entry_points
        installed = tmp_path / "wheel_extracted"
        wheel.extractall(installed)
    _isolated_import(installed)
