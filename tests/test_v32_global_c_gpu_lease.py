"""Offline checks for one provider lease supervising several fold jobs."""
from __future__ import annotations

from argparse import Namespace

from scripts.rbp_encode_v33 import monitor_global_c_gpu_lease as lease


def arguments(tmp_path):
    cli = tmp_path / "compshare"
    cli.write_text("")
    return Namespace(
        cli=cli, instance_id="cpod-allowed", job_id=["fold-0", "fold-1"],
        paid_start_unix=1000, hard_hours=2, receipt=tmp_path / "monitor.json",
        resume_receipt=None,
        first_step_minutes=10, stall_minutes=30,
        initial_lease_minutes=90, renew_minutes=30,
        renew_margin_minutes=15, poll_seconds=60, max_api_errors=3,
    )


def test_one_idle_fold_stops_entire_instance_without_extending_lease(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    clock = {"unix": 1000.0, "mono": 0.0}
    stops, extensions = [], []
    monkeypatch.setattr(lease.socket, "gethostname", lambda: "local-control")
    monkeypatch.setattr(lease.time, "time", lambda: clock["unix"])
    monkeypatch.setattr(lease.time, "monotonic", lambda: clock["mono"])

    def advance(_seconds):
        clock["unix"] += 600
        clock["mono"] += 600

    monkeypatch.setattr(lease.time, "sleep", advance)
    monkeypatch.setattr(lease, "current_schedule", lambda *_args: 3000)
    monkeypatch.setattr(lease, "extend_lease", lambda *_args, **_kw: extensions.append(1))
    monkeypatch.setattr(lease, "cli", lambda _exe, *parts: {
        "data": {"logs": {"State": "Running", "Stdout": (
            '{"status":"OPTIMIZER_STEP","optimizer_step":1}\n'
            '{"status":"GPU_TELEMETRY","sequence":1}'
            if "fold-0" in parts else
            '{"status":"GPU_TELEMETRY","sequence":1}'
        )}},
    })
    monkeypatch.setattr(lease, "stop_one", lambda _exe, instance, reason, _receipt:
                        stops.append((instance, reason)))
    assert lease.run(args) == 1
    assert stops == [("cpod-allowed", "FIRST_OPTIMIZER_STEP_TIMEOUT:fold-1")]
    assert extensions == []


def test_all_jobs_need_optimizer_gpu_and_success_evidence(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    monkeypatch.setattr(lease.socket, "gethostname", lambda: "local-control")
    monkeypatch.setattr(lease.time, "time", lambda: 1000.0)
    monkeypatch.setattr(lease.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(lease, "current_schedule", lambda *_args: 3000)
    monkeypatch.setattr(lease, "cli", lambda _exe, *parts: {
        "data": {"logs": {"State": "Succeeded", "Stdout": (
            '{"status":"OPTIMIZER_STEP","optimizer_step":2}\n'
            '{"status":"GPU_TELEMETRY","sequence":1}\n'
            '{"status":"TRAINING_SUCCEEDED"}'
        )}},
    })
    assert lease.run(args) == 0
    assert 'ALL_GLOBAL_G2_JOBS_SUCCEEDED_RESULT_RETURN_PENDING' in args.receipt.read_text()


def test_validation_batches_keep_long_coverage_check_alive(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    args.job_id = ["fold-0"]
    clock = {"unix": 1000.0, "mono": 0.0, "poll": 0}
    stops = []
    monkeypatch.setattr(lease.socket, "gethostname", lambda: "local-control")
    monkeypatch.setattr(lease.time, "time", lambda: clock["unix"])
    monkeypatch.setattr(lease.time, "monotonic", lambda: clock["mono"])
    monkeypatch.setattr(lease.time, "sleep", lambda _seconds: (
        clock.__setitem__("unix", clock["unix"] + 600),
        clock.__setitem__("mono", clock["mono"] + 600),
        clock.__setitem__("poll", clock["poll"] + 1),
    ))
    monkeypatch.setattr(lease, "current_schedule", lambda *_args: 3000)
    monkeypatch.setattr(lease, "extend_lease", lambda *_args, **_kw: 3000)
    monkeypatch.setattr(lease, "stop_one", lambda _exe, _id, reason, _receipt:
                        stops.append(reason))

    def logs(_exe, *parts):
        done = clock["poll"] >= 5
        return {"data": {"logs": {"State": "Succeeded" if done else "Running",
            "Stdout": (
                '{"status":"OPTIMIZER_STEP","optimizer_step":101}\n'
                '{"status":"GPU_TELEMETRY","sequence":%d}\n' % (clock["poll"] + 2) +
                '{"status":"VALIDATION_HEARTBEAT","cycle":0,'
                '"runtime_chunk":%d,"completed_batches":403}\n' % clock["poll"]
            ) + ('{"status":"TRAINING_SUCCEEDED"}' if done else "")}}}

    monkeypatch.setattr(lease, "cli", logs)
    result = lease.run(args)
    assert result == 0, stops
    assert clock["mono"] >= 3000
    assert stops == []
