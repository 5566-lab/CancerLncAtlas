from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).with_name("43d_run_v31_strict_matrix_multigpu.py")
if not SCRIPT.exists():
    SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / SCRIPT.name
SPEC = importlib.util.spec_from_file_location("v31_multigpu_runner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_execution_lanes_multiplex_without_changing_visible_device_identity() -> None:
    lanes = MODULE._execution_lanes(["0", "1"], 2)
    assert lanes == [
        {"lane_id": "gpu_0_slot_0", "device": "0"},
        {"lane_id": "gpu_0_slot_1", "device": "0"},
        {"lane_id": "gpu_1_slot_0", "device": "1"},
        {"lane_id": "gpu_1_slot_1", "device": "1"},
    ]


def test_run_one_isolates_the_requested_gpu(monkeypatch, tmp_path: Path) -> None:
    observed = {}

    def fake_run(command, *, cwd, capture_output, text, env):
        observed["command"] = command
        observed["cwd"] = cwd
        observed["cuda"] = env.get("CUDA_VISIBLE_DEVICES")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(MODULE.subprocess, "run", fake_run)
    audit = SimpleNamespace(audit_task=lambda *args: (True, "PASS"))
    output_root = tmp_path / "training"
    shared = {
        "config": "/frozen/config.yaml",
        "run_id": "run-1",
        "input_root": "/frozen/input",
        "input_manifest": "/frozen/INPUT_MANIFEST.tsv",
        "formal_gate": "/frozen/gate.json",
        "output_root": str(output_root),
        "python": "/env/bin/python",
        "repo_root": "/frozen/repo",
        "gate_sha256": "gate-sha",
    }
    task = {"fold_id": "LOCO_ACC", "model": "hgt", "seed": 20260726}
    result = MODULE._run_one(shared, audit, task, "1")
    assert result["status"] == "COMPLETED"
    assert result["device"] == "1"
    assert observed["cuda"] == "1"
    assert observed["cwd"] == "/frozen/repo"
    log = next((output_root / "run_control" / "logs").glob("*.log"))
    assert "CUDA_VISIBLE_DEVICES=1" in log.read_text(encoding="utf-8")


def test_run_one_rejects_existing_task_root(tmp_path: Path) -> None:
    output_root = tmp_path / "training"
    task_root = output_root / "rgcn" / "LOCO_ACC" / "seed_20260726"
    task_root.mkdir(parents=True)
    shared = {
        "output_root": str(output_root),
    }
    task = {"fold_id": "LOCO_ACC", "model": "rgcn", "seed": 20260726}
    result = MODULE._run_one(shared, SimpleNamespace(), task, "0")
    assert result["status"] == "FAILED"
    assert "without an authorized exact checkpoint resume" in result["error"]


def test_resume_skips_only_audit_valid_task(tmp_path: Path) -> None:
    output_root = tmp_path / "training"
    task = {"fold_id": "LOCO_ACC", "model": "rgcn", "seed": 20260726}
    task_root = output_root / "rgcn" / "LOCO_ACC" / "seed_20260726"
    task_root.mkdir(parents=True)
    audit = SimpleNamespace(audit_task=lambda *args: (True, "PASS"))
    shared = {"gate_sha256": "gate-sha", "run_id": "run-1"}
    completed, pending, quarantined = MODULE._partition_resume_tasks(
        [task], output_root, audit, shared, quarantine_incomplete=False
    )
    assert len(completed) == 1
    assert completed[0]["resumed"] is True
    assert pending == []
    assert quarantined == []
    assert task_root.exists()


def test_resume_quarantines_incomplete_without_deleting_it(tmp_path: Path) -> None:
    output_root = tmp_path / "training"
    task = {"fold_id": "LOCO_ACC", "model": "rgcn", "seed": 20260726}
    task_root = output_root / "rgcn" / "LOCO_ACC" / "seed_20260726"
    task_root.mkdir(parents=True)
    (task_root / "partial.txt").write_text("preserve me", encoding="utf-8")
    audit = SimpleNamespace(audit_task=lambda *args: (False, "missing checkpoint"))
    shared = {"gate_sha256": "gate-sha", "run_id": "run-1"}
    completed, pending, quarantined = MODULE._partition_resume_tasks(
        [task], output_root, audit, shared, quarantine_incomplete=True
    )
    assert completed == []
    assert pending == [task]
    assert len(quarantined) == 1
    assert not task_root.exists()
    preserved = Path(quarantined[0]["quarantine_root"])
    assert (preserved / "partial.txt").read_text(encoding="utf-8") == "preserve me"


def test_resume_continues_task_with_exact_checkpoint_in_place(tmp_path: Path) -> None:
    output_root = tmp_path / "training"
    task = {"fold_id": "LOCO_ACC", "model": "cc_hhgt", "seed": 20260726}
    task_root = output_root / "cc_hhgt" / "LOCO_ACC" / "seed_20260726"
    task_root.mkdir(parents=True)
    checkpoint = task_root / "last_training_state.pt"
    checkpoint.write_bytes(b"guarded checkpoint")
    audit = SimpleNamespace(audit_task=lambda *args: (False, "missing final outputs"))
    shared = {"gate_sha256": "gate-sha", "run_id": "run-1"}
    completed, pending, quarantined = MODULE._partition_resume_tasks(
        [task], output_root, audit, shared, quarantine_incomplete=False
    )
    assert completed == []
    assert len(pending) == 1
    assert pending[0]["resume_training"] is True
    assert pending[0]["resume_checkpoint"] == str(checkpoint)
    assert quarantined == []
    assert checkpoint.exists()


def test_run_one_passes_guarded_resume_flag_only_for_exact_checkpoint(
    monkeypatch, tmp_path: Path
) -> None:
    observed = {}

    def fake_run(command, *, cwd, capture_output, text, env):
        observed["command"] = command
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(MODULE.subprocess, "run", fake_run)
    output_root = tmp_path / "training"
    task_root = output_root / "cc_hhgt" / "LOCO_ACC" / "seed_20260726"
    task_root.mkdir(parents=True)
    (task_root / "last_training_state.pt").write_bytes(b"checkpoint")
    shared = {
        "config": "/frozen/config.yaml",
        "run_id": "run-1",
        "input_root": "/frozen/input",
        "input_manifest": "/frozen/INPUT_MANIFEST.tsv",
        "formal_gate": "/frozen/gate.json",
        "output_root": str(output_root),
        "python": "/env/bin/python",
        "repo_root": "/frozen/repo",
        "gate_sha256": "gate-sha",
    }
    task = {
        "fold_id": "LOCO_ACC",
        "model": "cc_hhgt",
        "seed": 20260726,
        "resume_training": True,
    }
    audit = SimpleNamespace(audit_task=lambda *args: (True, "PASS"))
    result = MODULE._run_one(shared, audit, task, "0")
    assert result["status"] == "COMPLETED"
    assert observed["command"][-1] == "--resume-training"
