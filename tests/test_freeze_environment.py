from __future__ import annotations

import importlib.util
from pathlib import Path

from cc_hhgt.common import load_config


def _load_freezer():
    path = Path(__file__).resolve().parents[1] / "scripts" / "67_freeze_v30_formal_run.py"
    spec = importlib.util.spec_from_file_location("v30_freezer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_optional_environment_command_can_be_unavailable() -> None:
    freezer = _load_freezer()
    result = freezer.command_output(["definitely-missing-v31-environment-command"])
    assert result["returncode"] == 127
    assert result["availability"] == "UNAVAILABLE"
    assert "FileNotFoundError" in result["stderr"]


def test_config_root_can_be_relocated_without_creating_configured_paths(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "project_root: 'D:/nonportable/build/root'\n"
        "results_dir: results/formal\n"
        "cache_dir: results/formal/cache\n"
        "standardized_dir: results/formal/standardized\n",
        encoding="utf-8",
    )
    runtime = tmp_path / "runtime"
    cfg = load_config(
        config,
        project_root_override=runtime,
        create_dirs=False,
    )
    assert cfg["_root"] == runtime.resolve()
    assert cfg["project_root"] == str(runtime.resolve())
    assert cfg["_results"] == runtime.resolve() / "results/formal"
    assert not cfg["_results"].exists()
