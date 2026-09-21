from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SUPERVISOR = ROOT / "scripts" / "run_v32_single_cell_r10_isolated_supervisor.py"
CGROUP_SMOKE = ROOT / "scripts" / "smoke_v32_single_cell_cgroup_v2_isolated.py"


def load_supervisor():
    spec = importlib.util.spec_from_file_location("r10_isolated_supervisor", SUPERVISOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def supervisor():
    return load_supervisor()


def proc_stat(pid: int, pgrp: int, starttime: int, command: str = "worker") -> str:
    # Fields after ')' start at Linux proc field 3.  starttime is field 22.
    fields = ["S", "1", str(pgrp)] + ["0"] * 16 + [str(starttime)]
    return f"{pid} ({command}) " + " ".join(fields) + "\n"


def proc_status(rss_kib: int, hwm_kib: int) -> str:
    return f"Name:\tfixture\nVmHWM:\t{hwm_kib} kB\nVmRSS:\t{rss_kib} kB\n"


def snapshot(supervisor, *members, unreadable=(), errors=()):
    return supervisor.GroupMemorySnapshot(
        777,
        tuple(members),
        tuple(unreadable),
        tuple(errors),
        sum(row.rss_bytes for row in members),
        max((row.high_water_bytes for row in members), default=0),
    )


def fallback_observation(supervisor):
    return {
        "mode": "PROC_SINGLE_MEMBER_FALLBACK",
        "helper_overlap_permitted": False,
        "single_member_proof": {
            "claim": "FROZEN_RUNNER_USES_EXEC_REPLACEMENT_AND_SPAWNS_NO_HELPERS",
            "helper_overlap_permitted": False,
            "runner_path": str(SUPERVISOR.resolve()),
            "runner_sha256": "a" * 64,
            "proof_receipt_path": str(SUPERVISOR.resolve()),
            "proof_receipt_sha256": "b" * 64,
        },
    }


def test_control_plane_has_only_stdlib_imports() -> None:
    tree = ast.parse(SUPERVISOR.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.partition(".")[0])
    allowed = {
        "__future__",
        "argparse",
        "hashlib",
        "json",
        "os",
        "pathlib",
        "signal",
        "socket",
        "subprocess",
        "sys",
        "time",
        "typing",
    }
    assert imported <= allowed
    assert imported.isdisjoint(
        {
            "anndata",
            "duckdb",
            "h5py",
            "numpy",
            "pandas",
            "polars",
            "pyarrow",
            "scanpy",
            "scipy",
            "sklearn",
            "torch",
        }
    )


def test_cgroup_smoke_is_stdlib_only_and_never_names_scientific_runner() -> None:
    source = CGROUP_SMOKE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.partition(".")[0])
    assert imported <= {
        "__future__",
        "argparse",
        "hashlib",
        "json",
        "os",
        "pathlib",
        "secrets",
        "signal",
        "subprocess",
        "sys",
        "time",
        "typing",
    }
    assert "run_v32_single_cell_r7_streaming.py" not in source
    assert "run_v32_single_cell_r10_streaming.py" not in source
    assert "MEMORY_LIMIT_BYTES = 512 * 1024**2" in source


def test_two_control_processes_are_clean_and_distinct() -> None:
    command = [sys.executable, str(SUPERVISOR), "prove-stdlib"]
    first = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    second = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    first_stdout, first_stderr = first.communicate(timeout=30)
    second_stdout, second_stderr = second.communicate(timeout=30)
    assert first.returncode == 0, first_stderr.decode(errors="replace")
    assert second.returncode == 0, second_stderr.decode(errors="replace")
    rows = [json.loads(first_stdout), json.loads(second_stdout)]
    assert all(row["scientific_modules_loaded"] == [] for row in rows)
    assert rows[0]["process_id"] != rows[1]["process_id"]
    assert all(row["memory_limit_bytes"] == 512 * 1024**2 for row in rows)


def test_proc_parsers_handle_command_parenthesis_and_exact_units(supervisor) -> None:
    identity = supervisor.parse_proc_stat_identity(
        proc_stat(17, 777, 123456, command="name with ) paren")
    )
    assert identity.process_group_id == 777
    assert identity.start_time_ticks == 123456
    memory = supervisor.parse_proc_status_memory(proc_status(123, 456), 17)
    assert memory.rss_bytes == 123 * 1024
    assert memory.high_water_bytes == 456 * 1024


def test_ucec_false_kill_counterexample_excludes_watchdog_hwm(supervisor) -> None:
    child_hwm = 408_199_168
    old_watchdog_hwm = 129_417_216
    old_incorrect_gate = child_hwm + old_watchdog_hwm
    assert old_incorrect_gate == 537_616_384
    assert old_incorrect_gate > supervisor.MEMORY_LIMIT_BYTES

    current = snapshot(
        supervisor,
        supervisor.ProcessMemory(2001, child_hwm, child_hwm),
    )
    decision = supervisor.evaluate_memory_snapshot(current)
    assert decision.stop is False
    assert decision.formal_gate_value_bytes == child_hwm
    assert decision.memory_limit_bytes == 536_870_912


def test_live_member_hwm_is_max_not_sum(supervisor) -> None:
    one = supervisor.ProcessMemory(1, 100 * 1024**2, 300 * 1024**2)
    two = supervisor.ProcessMemory(2, 100 * 1024**2, 300 * 1024**2)
    current = snapshot(supervisor, one, two)
    assert sum(row.high_water_bytes for row in current.members) > 512 * 1024**2
    decision = supervisor.evaluate_memory_snapshot(current)
    assert decision.stop is False
    assert decision.formal_gate_value_bytes == 300 * 1024**2


def test_fallback_forbids_any_observed_helper_overlap(supervisor) -> None:
    one = supervisor.ProcessMemory(1, 100 * 1024**2, 100 * 1024**2)
    two = supervisor.ProcessMemory(2, 100 * 1024**2, 100 * 1024**2)
    decision = supervisor.evaluate_memory_snapshot(
        snapshot(supervisor, one, two), require_single_live_member=True
    )
    assert decision.stop is True
    assert decision.reason == "HELPER_PROCESS_OVERLAP_FORBIDDEN_WITHOUT_CGROUP_PEAK"


def test_cgroup_peak_catches_between_poll_multi_helper_peak(supervisor) -> None:
    # /proc samples immediately before and after the short overlap can both
    # look safe and single-member.  The exclusive cgroup aggregate peak
    # persists and therefore catches the missed 520 MiB concurrent peak.
    before = snapshot(
        supervisor, supervisor.ProcessMemory(11, 250 * 1024**2, 250 * 1024**2)
    )
    after = snapshot(
        supervisor, supervisor.ProcessMemory(11, 200 * 1024**2, 250 * 1024**2)
    )
    assert supervisor.evaluate_memory_snapshot(
        before, require_single_live_member=True
    ).stop is False
    assert supervisor.evaluate_memory_snapshot(
        after, require_single_live_member=True
    ).stop is False
    cgroup = supervisor.CgroupMemorySnapshot(
        "/sys/fs/cgroup/delegated/UCEC",
        200 * 1024**2,
        520 * 1024**2,
        (11,),
        (("oom", 0), ("oom_kill", 0)),
        (),
    )
    decision = supervisor.evaluate_cgroup_v2_snapshot(
        cgroup, expected_leader_pid=11
    )
    assert decision.stop is True
    assert decision.reason == "OBSERVED_CGROUP_V2_MEMORY_PEAK_EXCEEDED_512_MIB"
    assert decision.formal_gate_value_bytes == 520 * 1024**2


def test_cgroup_oom_event_fails_even_when_numeric_peak_equals_limit(supervisor) -> None:
    cgroup = supervisor.CgroupMemorySnapshot(
        "/sys/fs/cgroup/delegated/UCEC",
        100,
        supervisor.MEMORY_LIMIT_BYTES,
        (11,),
        (("oom", 1), ("oom_kill", 1)),
        (),
    )
    decision = supervisor.evaluate_cgroup_v2_snapshot(
        cgroup, expected_leader_pid=11
    )
    assert decision.stop is True
    assert decision.reason == "CGROUP_V2_MEMORY_OOM_EVENT"


def test_read_cgroup_snapshot_binds_current_peak_limit_and_members(
    tmp_path: Path, supervisor
) -> None:
    cgroup = tmp_path / "job"
    cgroup.mkdir()
    (cgroup / "memory.current").write_text("123\n", encoding="ascii")
    (cgroup / "memory.peak").write_text("456\n", encoding="ascii")
    (cgroup / "memory.max").write_text(
        str(supervisor.MEMORY_LIMIT_BYTES), encoding="ascii"
    )
    (cgroup / "cgroup.procs").write_text("22\n11\n", encoding="ascii")
    (cgroup / "memory.events").write_text(
        "low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\n", encoding="ascii"
    )
    observed = supervisor.read_cgroup_v2_snapshot(cgroup)
    assert observed.current_bytes == 123
    assert observed.peak_bytes == 456
    assert observed.process_ids == (11, 22)
    assert observed.observation_errors == ()


def test_simultaneous_group_rss_sum_crossing_limit_stops(supervisor) -> None:
    one = supervisor.ProcessMemory(1, 300 * 1024**2, 300 * 1024**2)
    two = supervisor.ProcessMemory(2, 220 * 1024**2, 220 * 1024**2)
    decision = supervisor.evaluate_memory_snapshot(snapshot(supervisor, one, two))
    assert decision.stop is True
    assert decision.reason == "OBSERVED_PROCESS_GROUP_MEMORY_EXCEEDED_512_MIB"
    assert decision.formal_gate_value_bytes == 520 * 1024**2


def test_live_member_hwm_crossing_limit_stops_even_after_rss_falls(supervisor) -> None:
    member = supervisor.ProcessMemory(1, 80 * 1024**2, 513 * 1024**2)
    decision = supervisor.evaluate_memory_snapshot(snapshot(supervisor, member))
    assert decision.stop is True
    assert decision.formal_gate_value_bytes == 513 * 1024**2


def test_exact_512_mib_is_allowed_but_one_byte_more_stops(supervisor) -> None:
    exact = supervisor.ProcessMemory(1, supervisor.MEMORY_LIMIT_BYTES, 1)
    above = supervisor.ProcessMemory(1, supervisor.MEMORY_LIMIT_BYTES + 1, 1)
    assert supervisor.evaluate_memory_snapshot(snapshot(supervisor, exact)).stop is False
    assert supervisor.evaluate_memory_snapshot(snapshot(supervisor, above)).stop is True
    with pytest.raises(supervisor.IsolatedSupervisorError, match="immutable"):
        supervisor.evaluate_memory_snapshot(
            snapshot(supervisor, exact), memory_limit_bytes=513 * 1024**2
        )


def test_collector_excludes_exited_or_pid_reused_member_hwm(
    tmp_path: Path, supervisor
) -> None:
    proc = tmp_path / "proc"
    for pid in (100, 101, 102):
        (proc / str(pid)).mkdir(parents=True)
    (proc / "100" / "stat").write_text(proc_stat(100, 777, 1), encoding="utf-8")
    (proc / "100" / "status").write_text(proc_status(10, 20), encoding="utf-8")
    (proc / "101" / "stat").write_text(proc_stat(101, 777, 1), encoding="utf-8")
    (proc / "101" / "status").write_text(
        proc_status(1, 900 * 1024), encoding="utf-8"
    )
    (proc / "102" / "stat").write_text(proc_stat(102, 999, 1), encoding="utf-8")
    (proc / "102" / "status").write_text(
        proc_status(1, 900 * 1024), encoding="utf-8"
    )
    calls: dict[Path, int] = {}

    def changing_reader(path: Path) -> str:
        calls[path] = calls.get(path, 0) + 1
        if path == proc / "101" / "stat" and calls[path] > 1:
            return proc_stat(101, 777, 2)
        return path.read_text(encoding="utf-8")

    current = supervisor.collect_process_group_snapshot(
        777, proc_root=proc, read_text=changing_reader
    )
    assert [row.process_id for row in current.members] == [100]
    assert current.maximum_live_member_hwm_bytes == 20 * 1024
    assert supervisor.evaluate_memory_snapshot(current).stop is False


def test_unreadable_still_live_member_fails_closed(tmp_path: Path, supervisor) -> None:
    proc = tmp_path / "proc"
    member = proc / "100"
    member.mkdir(parents=True)
    (member / "stat").write_text(proc_stat(100, 777, 1), encoding="utf-8")
    current = supervisor.collect_process_group_snapshot(777, proc_root=proc)
    assert current.unreadable_live_member_pids == (100,)
    decision = supervisor.evaluate_memory_snapshot(current)
    assert decision.stop is True
    assert decision.reason == "PROCESS_GROUP_MEMORY_UNOBSERVABLE"


def test_no_live_member_is_not_interpreted_as_zero_memory(supervisor) -> None:
    current = snapshot(supervisor)
    decision = supervisor.evaluate_memory_snapshot(current)
    assert decision.stop is True
    assert decision.reason == "PROCESS_GROUP_HAS_NO_OBSERVABLE_LIVE_MEMBER"


def test_watch_request_cannot_change_limit_or_add_environment(supervisor) -> None:
    request = {
        "format": supervisor.WATCH_REQUEST_FORMAT,
        "cancer_id": "UCEC",
        "memory_limit_bytes": supervisor.MEMORY_LIMIT_BYTES,
        "jobs": [
            {
                "stage": "RUN",
                "argv": ["python", "runner.py"],
                "log_path": str(Path.cwd() / "run.log"),
                "environment": {},
                "memory_observation": fallback_observation(supervisor),
            }
        ],
    }
    supervisor.validate_watch_request(request)
    request["memory_limit_bytes"] += 1
    with pytest.raises(supervisor.IsolatedSupervisorError, match="512 MiB"):
        supervisor.validate_watch_request(request)
    request["memory_limit_bytes"] = supervisor.MEMORY_LIMIT_BYTES
    request["jobs"][0]["environment"] = {"LD_PRELOAD": "anything"}
    with pytest.raises(supervisor.IsolatedSupervisorError, match="forbidden"):
        supervisor.validate_watch_request(request)


def test_fallback_requires_bound_no_helper_proof(supervisor) -> None:
    invalid = {
        "mode": "PROC_SINGLE_MEMBER_FALLBACK",
        "helper_overlap_permitted": False,
    }
    with pytest.raises(supervisor.IsolatedSupervisorError, match="proof"):
        supervisor.validate_memory_observation_contract(invalid)
    assert (
        supervisor.validate_memory_observation_contract(
            fallback_observation(supervisor)
        )
        == "PROC_SINGLE_MEMBER_FALLBACK"
    )


def test_execute_cohort_spawns_one_fresh_watchdog_command_per_cancer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, supervisor
) -> None:
    output_root = tmp_path / "output"
    audit_root = tmp_path / "audit"
    cancers = [
        {"cancer_id": cancer, "jobs": [{"stage": "RUN", "argv": ["x"]}]}
        for cancer in supervisor.REMAINING_EXECUTION_ORDER
    ]
    spec = {
        "output_root": str(output_root),
        "audit_root": str(audit_root),
        "runner_generation": "R10",
        "cancers": cancers,
    }
    preflight = {"format": supervisor.FORMAT, "status": "PASS_READY_BUT_NOT_STARTED"}
    binding = {
        "R9_TYPED_FAILURE": {
            "path": "/immutable/TYPED_FAILURE.json",
            "sha256": "a" * 64,
            "bytes": 1,
        }
    }
    monkeypatch.setattr(supervisor, "validate_launch_spec", lambda _spec: (output_root, audit_root))
    monkeypatch.setattr(supervisor, "verify_immutable_bindings", lambda: binding)
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append(list(command))
        request_path = Path(command[command.index("--request-json") + 1])
        receipt_path = Path(command[command.index("--receipt-json") + 1])
        request = json.loads(request_path.read_text(encoding="utf-8"))
        cancer = request["cancer_id"]
        receipt = {
            "format": supervisor.WATCH_RECEIPT_FORMAT,
            "status": "SUCCESS",
            "cancer_id": cancer,
            "watchdog_pid": 10_000 + len(commands),
            "watchdog_start_time_ticks": 20_000 + len(commands),
        }
        supervisor.exclusive_json(receipt_path, receipt)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(supervisor.subprocess, "run", fake_run)
    result = supervisor.execute_cohort(
        spec,
        preflight,
        expected_supervisor_sha256="b" * 64,
        root_confirmation_token=supervisor.ROOT_CONFIRMATION_TOKEN,
    )
    assert result == 0
    assert len(commands) == 4
    assert all("watch-cancer" in command for command in commands)
    assert len({command[command.index("--request-json") + 1] for command in commands}) == 4
    success = json.loads(
        (audit_root / "ISOLATED_SUPERVISOR_SUCCESS.json").read_text(encoding="utf-8")
    )
    assert success["distinct_watchdog_process_identity_per_cancer"] is True
    assert [row["watchdog_pid"] for row in success["per_cancer_watchdogs"]] == [
        10001,
        10002,
        10003,
        10004,
    ]
    assert success["final_17_cancer_binding_published"] is False


def test_cohort_confirmation_gate_precedes_any_root_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, supervisor
) -> None:
    output_root = tmp_path / "output"
    audit_root = tmp_path / "audit"
    spec = {
        "output_root": str(output_root),
        "audit_root": str(audit_root),
        "cancers": [],
    }
    with pytest.raises(supervisor.IsolatedSupervisorError, match="confirmation"):
        supervisor.execute_cohort(
            spec,
            {},
            expected_supervisor_sha256="a" * 64,
            root_confirmation_token="",
        )
    assert not output_root.exists()
    assert not audit_root.exists()
