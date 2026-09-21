from __future__ import annotations

import json
from pathlib import Path

import pytest

from cc_hhgt.v32.cli import main
from cc_hhgt.v32.orchestration import ExecutionPolicy, build_task_manifest, write_task_manifest
from cc_hhgt.v32.training_guard import (
    APPROVAL_FORMAT,
    TrainingAuthorizationError,
    compute_artifact_hashes,
    guard_training_entry,
)


def _fixture(tmp_path: Path, *, training: bool) -> dict[str, object]:
    repo = tmp_path / "repo"
    (repo / "cc_hhgt" / "v32").mkdir(parents=True)
    (repo / "scripts").mkdir()
    (repo / "cc_hhgt" / "v32" / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "scripts" / "v32_pipeline.py").write_text("# wrapper\n", encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "execution_control": {
                    "execution_mode": "TRAINING" if training else "CODE_ONLY",
                    "training_authorized": training,
                    "paid_enabled": False,
                    "max_paid_hours": 0,
                    "max_cost_cny": 0,
                }
            }
        ),
        encoding="utf-8",
    )
    inputs = tmp_path / "INPUT_MANIFEST.json"
    inputs.write_text('{"format":"test-only"}\n', encoding="utf-8")
    policy = (
        ExecutionPolicy(
            execution_mode="TRAINING", training_authorized=True
        )
        if training
        else ExecutionPolicy()
    )
    tasks = build_task_manifest(run_id="guard-test", policy=policy)
    manifest = write_task_manifest(tmp_path / "TASK_MANIFEST.tsv", tasks)
    task = tasks[0]
    return {
        "repo": repo,
        "config": config,
        "inputs": inputs,
        "manifest": manifest,
        "task": task,
        "approval": tmp_path / "TRAINING_APPROVAL.json",
    }


def _guard_args(paths: dict[str, object], *, allow: bool) -> dict[str, object]:
    task = paths["task"]
    assert isinstance(task, dict)
    return {
        "allow_training": allow,
        "repo_root": paths["repo"],
        "config_path": paths["config"],
        "input_manifest_path": paths["inputs"],
        "task_manifest_path": paths["manifest"],
        "approval_path": paths["approval"],
        "run_id": "guard-test",
        "task_id": task["task_id"],
        "endpoint_id": "local4070",
        "hardware_class": "LOCAL_RTX_4070_TI_SUPER_16GB",
    }


def _write_matching_approval(
    paths: dict[str, object],
    *,
    authorized_trainer: str | None = None,
) -> None:
    task = paths["task"]
    assert isinstance(task, dict)
    hashes = compute_artifact_hashes(
        repo_root=paths["repo"],
        config_path=paths["config"],
        input_manifest_path=paths["inputs"],
        task_manifest_path=paths["manifest"],
    )
    payload = {
        "approval_format": APPROVAL_FORMAT,
        "training_authorized": True,
        "run_id": "guard-test",
        "endpoint_id": "local4070",
        "hardware_class": "LOCAL_RTX_4070_TI_SUPER_16GB",
        "approved_task_ids": [task["task_id"]],
        "artifact_hashes": hashes,
        "paid_enabled": False,
        "max_paid_hours": 0,
        "max_cost_cny": 0,
    }
    if authorized_trainer is not None:
        payload["authorized_trainer"] = authorized_trainer
    approval = paths["approval"]
    assert isinstance(approval, Path)
    approval.write_text(json.dumps(payload), encoding="utf-8")


def test_missing_allow_flag_denies_before_approval_access(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, training=True)
    with pytest.raises(TrainingAuthorizationError, match="--allow-training"):
        guard_training_entry(**_guard_args(paths, allow=False))
    assert not Path(paths["approval"]).exists()


def test_code_only_config_denies_even_if_flag_is_present(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, training=False)
    with pytest.raises(TrainingAuthorizationError, match="CODE_ONLY"):
        guard_training_entry(**_guard_args(paths, allow=True))


def test_missing_approval_denies_authorized_config(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, training=True)
    with pytest.raises(TrainingAuthorizationError, match="approval is absent"):
        guard_training_entry(**_guard_args(paths, allow=True))


def test_matching_approval_returns_read_only_context(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, training=True)
    _write_matching_approval(paths)
    context = guard_training_entry(**_guard_args(paths, allow=True))
    assert context.task_id == paths["task"]["task_id"]
    assert context.policy["training_authorized"] is True
    assert set(context.artifact_hashes) == {
        "code_sha256",
        "config_sha256",
        "input_manifest_sha256",
        "task_manifest_sha256",
    }


def test_any_artifact_change_invalidates_approval(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, training=True)
    _write_matching_approval(paths)
    Path(paths["inputs"]).write_text('{"format":"changed"}\n', encoding="utf-8")
    with pytest.raises(TrainingAuthorizationError, match="artifact hash mismatch"):
        guard_training_entry(**_guard_args(paths, allow=True))


def test_dependency_outside_v32_invalidates_approval(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, training=True)
    dependency = Path(paths["repo"]) / "cc_hhgt" / "gnn.py"
    dependency.write_text("VALUE = 1\n", encoding="utf-8")
    _write_matching_approval(paths)
    dependency.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(TrainingAuthorizationError, match="artifact hash mismatch"):
        guard_training_entry(**_guard_args(paths, allow=True))


def test_cli_blocks_before_dynamic_trainer_import(tmp_path: Path, monkeypatch) -> None:
    paths = _fixture(tmp_path, training=True)
    imports: list[str] = []

    def forbidden_import(name: str):
        imports.append(name)
        raise AssertionError("trainer import must not be reached")

    monkeypatch.setattr("cc_hhgt.v32.cli.importlib.import_module", forbidden_import)
    task = paths["task"]
    assert isinstance(task, dict)
    result = main(
        [
            "run-shard",
            "--allow-training",
            "--repo-root",
            str(paths["repo"]),
            "--config",
            str(paths["config"]),
            "--input-manifest",
            str(paths["inputs"]),
            "--task-manifest",
            str(paths["manifest"]),
            "--approval",
            str(paths["approval"]),
            "--run-id",
            "guard-test",
            "--task-id",
            str(task["task_id"]),
        ]
    )
    assert result == 13
    assert imports == []


def test_cli_binds_exact_trainer_before_dynamic_import(tmp_path: Path, monkeypatch) -> None:
    paths = _fixture(tmp_path, training=True)
    authorized = "cc_hhgt.v32.training:run_authorized_task"
    _write_matching_approval(paths, authorized_trainer=authorized)
    imports: list[str] = []

    def forbidden_import(name: str):
        imports.append(name)
        raise AssertionError("mismatched trainer must be denied before import")

    monkeypatch.setattr("cc_hhgt.v32.cli.importlib.import_module", forbidden_import)
    task = paths["task"]
    assert isinstance(task, dict)
    result = main(
        [
            "run-shard",
            "--allow-training",
            "--repo-root",
            str(paths["repo"]),
            "--config",
            str(paths["config"]),
            "--input-manifest",
            str(paths["inputs"]),
            "--task-manifest",
            str(paths["manifest"]),
            "--approval",
            str(paths["approval"]),
            "--run-id",
            "guard-test",
            "--task-id",
            str(task["task_id"]),
            "--trainer",
            "cc_hhgt.v32.group_shared_encoder_oracle:run_authorized_oracle_comparison",
        ]
    )
    assert result == 13
    assert imports == []


def test_guard_records_exact_authorized_trainer(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, training=True)
    trainer = "cc_hhgt.v32.training:run_authorized_task"
    _write_matching_approval(paths, authorized_trainer=trainer)
    args = _guard_args(paths, allow=True)
    args["trainer_specification"] = trainer
    context = guard_training_entry(**args)
    assert context.authorized_trainer == trainer


def test_production_guard_modules_do_not_import_torch() -> None:
    root = Path(__file__).resolve().parents[1]
    for relative in (
        "cc_hhgt/v32/orchestration.py",
        "cc_hhgt/v32/training_guard.py",
        "cc_hhgt/v32/cli.py",
        "scripts/v32_pipeline.py",
    ):
        source = (root / relative).read_text(encoding="utf-8")
        executable = "\n".join(
            line for line in source.splitlines() if not line.lstrip().startswith("#")
        )
        assert "import torch" not in executable
        assert "torch.cuda" not in executable
