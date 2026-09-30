"""Offline checks for the provider-enforced C training shutdown lease."""
from __future__ import annotations

from argparse import Namespace

import pytest

from scripts.rbp_encode_v33 import monitor_c_gpu_lease as lease


def test_validation_marker_tracks_completed_batch_across_runtime_chunks():
    assert lease.validation_marker([
        '{"status":"VALIDATION_HEARTBEAT","phase":"START","cycle":0}',
        '{"status":"VALIDATION_HEARTBEAT","cycle":0,"runtime_chunk":8,"completed_batches":400}',
        '{"status":"VALIDATION_HEARTBEAT","cycle":0,"runtime_chunk":8,"completed_batches":403}',
        '{"status":"GPU_TELEMETRY","sequence":12}',
    ]) == (0, 8, 403)


def test_schedule_must_be_for_exact_instance_and_has_integer_shutdown():
    good = {"data": {"instance": "cpod-allowed", "scheduled": True,
                     "scheduler_stop_time": 2000}}
    assert lease.schedule_stop_time(good, "cpod-allowed") == 2000
    with pytest.raises(RuntimeError, match="wrong instance"):
        lease.schedule_stop_time(good, "cpod-other")
    good["data"]["scheduler_stop_time"] = True
    with pytest.raises(RuntimeError, match="missing"):
        lease.schedule_stop_time(good, "cpod-allowed")


def test_extension_never_crosses_hard_deadline_and_reads_back(monkeypatch, tmp_path):
    state = {"stop": 1900}
    calls = []

    def fake_cli(_executable, *args, **_kwargs):
        calls.append(args)
        if args[2] == "show":
            return {"ok": True, "data": {"instance": "cpod-allowed",
                                         "scheduled": True,
                                         "scheduler_stop_time": state["stop"]}}
        assert args[:3] == ("instance", "schedule", "extend")
        state["stop"] += int(args[-1][:-1])
        return {"ok": True, "data": {}}

    monkeypatch.setattr(lease, "cli", fake_cli)
    observed = lease.extend_lease(tmp_path / "compshare", "cpod-allowed",
                                  now=1000, hard_deadline=2500,
                                  renew_seconds=1800, margin_seconds=1000)
    assert observed == 2500
    assert calls[1] == ("instance", "schedule", "extend", "cpod-allowed",
                        "--by", "600s")
    assert lease.extend_lease(tmp_path / "compshare", "cpod-allowed",
                              now=1000, hard_deadline=2500,
                              renew_seconds=1800, margin_seconds=1600) == 2500
    assert sum(args[2] == "extend" for args in calls) == 1


def test_stop_targets_one_instance_and_verifies_platform_state(monkeypatch, tmp_path):
    calls = []

    def fake_cli(_executable, *args, **_kwargs):
        calls.append(args)
        if args[:2] == ("instance", "show"):
            return {"ok": True, "data": {"UHostSet": [
                {"UHostId": "cpod-allowed", "State": "Stopped"}]}}
        return {"ok": True, "data": {}}

    monkeypatch.setattr(lease, "cli", fake_cli)
    receipt = tmp_path / "stop.json"
    lease.stop_one(tmp_path / "compshare", "cpod-allowed", "NO_PROGRESS", receipt)
    assert calls == [
        ("instance", "stop", "cpod-allowed", "--yes", "--wait", "--timeout", "180"),
        ("instance", "show", "cpod-allowed", "--status"),
    ]
    assert '"stopped": true' in receipt.read_text()


def test_monitor_rejects_absent_provider_schedule_before_job_poll(tmp_path, monkeypatch):
    cli_path = tmp_path / "compshare"
    cli_path.write_text("")
    monkeypatch.setattr(lease.socket, "gethostname", lambda: "local-control")
    monkeypatch.setattr(lease, "current_schedule", lambda *_args: 0)
    args = Namespace(
        cli=cli_path, instance_id="cpod-allowed", job_id="job-one",
        instance_created_unix=lease.time.time(), hard_hours=30,
        receipt=tmp_path / "monitor.json", first_step_minutes=45,
        stall_minutes=30, initial_lease_minutes=90, renew_minutes=30,
        renew_margin_minutes=15, poll_seconds=60,
    )
    with pytest.raises(RuntimeError, match="Initial provider shutdown schedule"):
        lease.run(args)
    assert not args.receipt.exists()


def test_live_job_without_optimizer_progress_never_extends_shutdown(tmp_path, monkeypatch):
    cli_path = tmp_path / "compshare"
    cli_path.write_text("")
    clock = {"unix": 1000.0, "mono": 0.0}
    stops = []
    extensions = []
    monkeypatch.setattr(lease.socket, "gethostname", lambda: "local-control")
    monkeypatch.setattr(lease.time, "time", lambda: clock["unix"])
    monkeypatch.setattr(lease.time, "monotonic", lambda: clock["mono"])

    def advance(_seconds):
        clock["unix"] += 600
        clock["mono"] += 600

    monkeypatch.setattr(lease.time, "sleep", advance)
    monkeypatch.setattr(lease, "current_schedule", lambda *_args: 3000)
    monkeypatch.setattr(lease, "extend_lease", lambda *_args, **_kwargs: extensions.append(1))
    monkeypatch.setattr(lease, "cli", lambda *_args, **_kwargs: {
        "ok": True, "data": {"logs": {"State": "Running", "Stdout":
            '{"status":"GPU_TELEMETRY","sequence":1}', "Stderr": ""}}})
    monkeypatch.setattr(lease, "stop_one", lambda _exe, instance, reason, _receipt:
                        stops.append((instance, reason)))
    args = Namespace(
        cli=cli_path, instance_id="cpod-allowed", job_id="job-one",
        instance_created_unix=1000, hard_hours=2,
        receipt=tmp_path / "monitor.json", first_step_minutes=10,
        stall_minutes=10, initial_lease_minutes=90, renew_minutes=30,
        renew_margin_minutes=15, poll_seconds=60,
        initial_optimizer_step=0, initial_gpu_telemetry_sequence=0,
        max_api_errors=3,
    )
    assert lease.run(args) == 1
    assert stops == [("cpod-allowed", "FIRST_OPTIMIZER_STEP_TIMEOUT")]
    assert extensions == []
