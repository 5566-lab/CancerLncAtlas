from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from cla_engine.common import Stop, atomic_json, digest, read_json, sha
from cla_engine.queue import job_grid, variant_config


def _plan(tmp_path: Path, cfg: dict):
    admission = tmp_path / "diagnostic"
    admission.mkdir()
    atomic_json(admission / "TRAINING_RECEIPT.json", {"state": "GO_TO_FORMAL_EVALUATION"})
    jobs = job_grid(cfg, tmp_path / "formal")
    for job in jobs:
        prepared = Path(job["prepared"])
        prepared.mkdir(parents=True)
        (prepared / "MANIFEST.json").write_text("{}\n", encoding="utf-8")
        job["sha256"] = sha(prepared / "MANIFEST.json")
        job["config_sha256"] = digest(variant_config(cfg, job["variant"]))
    value = {
        "format": "CLA_FULL_PYG_R62_15_JOB_PLAN_V1",
        "config_sha256": digest(cfg),
        "diagnostic_run": str(admission),
        "diagnostic_receipt_sha256": sha(admission / "TRAINING_RECEIPT.json"),
        "jobs": jobs,
    }
    path = tmp_path / "JOBS.json"
    atomic_json(path, value)
    return path, value


def test_direct_formal_prepare_is_rejected_before_preflight(tmp_path, monkeypatch):
    import cla
    from cla_engine import location

    monkeypatch.setattr(location, "enter_command", lambda *args: None)
    monkeypatch.setattr(
        "sys.argv",
        [
            "cancerlncatlas",
            "prepare",
            "--profile",
            "formal",
            "--allow-formal",
            "--output",
            str(tmp_path / "prepared"),
        ],
    )
    assert cla.main() == 2
    result = read_json(tmp_path / "prepared" / "STATUS.json")
    assert result["error_code"] == "FORMAL_PREPARE_REQUIRES_SIGNED_ADMISSION"
    assert not (tmp_path / "prepared_preflight").exists()


def test_formal_worker_rejects_manual_env_and_lease_before_optimizer(tmp_path, monkeypatch):
    from cla_engine import train as training

    prepared = tmp_path / "prepared"
    output = tmp_path / "run"
    prepared.mkdir()
    output.mkdir()
    atomic_json(
        output / "SUPERVISOR_LEASE.json",
        {"attempt_id": "forged", "expires_unix": time.time() + 3600},
    )
    monkeypatch.setenv("CLA_REMOTE_ATTEMPT_ID", "forged")
    cfg = {"model": {"task_weights": {"pathway": 1.0, "state": 0.5}}}
    monkeypatch.setattr(
        training,
        "validate_prepared",
        lambda path: ({"metadata": {"profile": "formal"}}, {}, cfg, {}),
    )
    monkeypatch.setattr(
        training.torch.optim,
        "AdamW",
        lambda *args, **kwargs: pytest.fail("optimizer must not be created"),
    )
    with pytest.raises(Stop) as error:
        training.train(
            prepared,
            output,
            device="cpu",
            deadline=time.time() + 60,
            lease_required=True,
        )
    assert error.value.code == "FORMAL_PLAN_REQUIRED"


def test_signed_plan_job_binding_accepts_only_exact_prepared_job(tmp_path, monkeypatch):
    from cla_engine import local_execution

    cfg = {"graph": {"variant": "G2"}}
    plan_path, plan = _plan(tmp_path, cfg)
    monkeypatch.setattr(local_execution, "assert_controller", lambda value: {"state": "ok"})
    monkeypatch.setattr(local_execution, "verify_signed_file", lambda *args: None)
    monkeypatch.setattr("cla_engine.queue.check_admission", lambda *args: {"state": "ok"})
    monkeypatch.setattr(local_execution, "verify_manifest", lambda path, expected=None: {})

    selected = plan["jobs"][0]
    prepared_cfg = variant_config(cfg, selected["variant"])
    monkeypatch.setattr(
        local_execution,
        "validate_prepared",
        lambda path: (
            {"metadata": {"profile": "formal"}},
            {"fold": selected["fold"], "graph_variant": selected["variant"]},
            prepared_cfg,
            {},
        ),
    )
    authorization = local_execution.validate_formal_job(
        selected["prepared"], plan_path, selected["job_id"], cfg
    )
    assert authorization["state"] == "FORMAL_JOB_AUTHORIZED_ON_149"
    assert authorization["prepared_sha256"] == selected["sha256"]

    with pytest.raises(Stop) as error:
        local_execution.validate_formal_job(
            selected["prepared"], plan_path, "G2_PF99", cfg
        )
    assert error.value.code == "FORMAL_JOB_NOT_AUTHORIZED"


def test_local_queue_resume_never_resets_persistent_deadline(tmp_path, monkeypatch):
    from cla_engine import queue

    cfg = {"graph": {"variant": "G2"}}
    plan_path, plan = _plan(tmp_path, cfg)
    monkeypatch.setattr(queue, "validate_formal_plan", lambda *args: (plan_path, plan))
    output = tmp_path / "queue"
    output.mkdir()
    original_deadline = time.time() - 10
    atomic_json(
        output / "QUEUE_BUDGET.json",
        {
            "deadline": original_deadline,
            "plan_sha256": sha(plan_path),
            "config_sha256": digest(cfg),
            "authorized_hours": 1.0,
            "device": "cpu",
            "target_host": "149",
        },
    )
    result = queue.run_queue(
        tmp_path, plan_path, cfg, output, hours=96, resume=True, device="cpu"
    )
    assert result["state"] == "QUEUE_BUDGET_EXHAUSTED"
    assert read_json(output / "QUEUE_BUDGET.json")["deadline"] == original_deadline


def test_local_queue_runs_all_signed_jobs_with_one_deadline(tmp_path, monkeypatch):
    from cla_engine import queue

    cfg = {"graph": {"variant": "G2"}}
    plan_path, plan = _plan(tmp_path, cfg)
    monkeypatch.setattr(queue, "validate_formal_plan", lambda *args: (plan_path, plan))
    authorized = []
    calls = []
    monkeypatch.setattr(
        queue,
        "validate_formal_job",
        lambda prepared, path, job_id, value: authorized.append(job_id),
    )

    def fake_train(prepared, run, **kwargs):
        Path(run).mkdir(parents=True, exist_ok=True)
        atomic_json(Path(run) / "TRAINING_RECEIPT.json", {"state": "TRAINED_FORMAL_CANDIDATE"})
        calls.append((Path(prepared), kwargs))
        return {"state": "TRAINED_FORMAL_CANDIDATE"}

    monkeypatch.setattr(queue, "train", fake_train)
    monkeypatch.setattr(queue, "evaluate_fold", lambda prepared, run, out: Path(out).mkdir(parents=True, exist_ok=True))
    monkeypatch.setattr(
        queue,
        "gate",
        lambda paths, value, out: {"state": "PREDICTION_GATE_PASS_NOT_DEPLOYED"},
    )
    output = tmp_path / "queue"
    result = queue.run_queue(
        tmp_path, plan_path, cfg, output, hours=1, device="cpu"
    )
    budget = read_json(output / "QUEUE_BUDGET.json")
    assert result["state"] == "QUEUE_COMPLETE_LOCAL_149"
    assert len(authorized) == len(calls) == 15
    assert {kwargs["deadline"] for _, kwargs in calls} == {budget["deadline"]}
    assert all(kwargs["formal_plan"] == plan_path for _, kwargs in calls)
    assert all(kwargs["formal_config"] is cfg for _, kwargs in calls)
    assert {kwargs["formal_job_id"] for _, kwargs in calls} == {job["job_id"] for job in plan["jobs"]}
    assert result["off_host_execution"] is False
