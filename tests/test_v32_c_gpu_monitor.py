from __future__ import annotations

from scripts.rbp_encode_v33 import monitor_c_gpu_149 as monitor


log_lines = monitor.log_lines
progress = monitor.progress


def test_comp_share_job_log_envelope_requires_optimizer_and_gpu_evidence():
    output = {"ok": True, "data": {"logs": {
        "State": "Running",
        "Stdout": '\n'.join([
            '{"status":"GPU_TELEMETRY","sequence":3,"utilization_gpu":86}',
            '{"status":"OPTIMIZER_STEP","optimizer_step":12}',
        ]),
        "Stderr": "",
    }}}
    assert progress(log_lines(output)) == (12, 3, False)


def test_bootstrap_messages_are_not_training_progress():
    output = {"data": {"logs": {"Stdout": '\n'.join([
        '{"status":"GPU_TELEMETRY","sequence":1}',
        '{"status":"VALIDATION_HEARTBEAT","completed_batches":1}',
        '{"status":"BOOTSTRAP_READY"}',
    ])}}}
    assert progress(log_lines(output)) == (0, 1, False)


def test_offline_stop_simulation_is_scoped_to_one_instance(tmp_path, monkeypatch):
    calls = []

    def fake_cli(*args, **_kwargs):
        calls.append(args)
        if args[:2] == ("instance", "show"):
            return {"ok": True, "data": {"state": "Stopped"}}
        return {"ok": True, "data": {}}

    monkeypatch.setattr(monitor, "cli", fake_cli)
    receipt = tmp_path / "stop.json"
    monitor.stop_one("cpod-allowed", "NO_PROGRESS", receipt)
    assert calls == [
        ("instance", "stop", "cpod-allowed", "--yes", "--wait", "--timeout", "180"),
        ("instance", "show", "cpod-allowed", "--status"),
    ]
    assert '"stopped": true' in receipt.read_text()
