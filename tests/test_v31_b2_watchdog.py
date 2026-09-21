from __future__ import annotations

import importlib.util
import os
from pathlib import Path


def _watchdog_module():
    path = Path(__file__).parents[1] / "scripts" / "95_watch_finalize_v31_b2.py"
    spec = importlib.util.spec_from_file_location("v31_b2_watchdog", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pid_probe_is_read_only_for_current_process() -> None:
    watchdog = _watchdog_module()
    assert watchdog._pid_running(os.getpid()) is True
    # The process must still be alive after the probe; reaching this assertion
    # is the regression check for the previous Windows os.kill(pid, 0) bug.
    assert os.getpid() > 0


def test_pid_probe_rejects_nonexistent_process() -> None:
    watchdog = _watchdog_module()
    assert watchdog._pid_running(2_147_483_647) is False
