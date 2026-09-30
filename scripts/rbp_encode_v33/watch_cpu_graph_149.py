#!/usr/bin/env python3
"""Bound a detached C graph build on host 149 without restarting the work."""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import time
from pathlib import Path


def start_ticks(pid: int) -> int | None:
    try:
        # The command name is parenthesized and can contain spaces.
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return int(fields[19])  # /proc stat field 22, with field 3 at index 0.
    except (FileNotFoundError, IndexError, ValueError):
        return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pid", type=int, required=True)
    p.add_argument("--expected-start-ticks", type=int, required=True)
    p.add_argument("--deadline-unix", type=float, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    args = p.parse_args()
    if socket.gethostname() != "149":
        raise RuntimeError("C graph build monitor must run on host 149")
    if args.receipt.exists() or start_ticks(args.pid) != args.expected_start_ticks:
        raise RuntimeError("CPU build monitor target or receipt is not fresh")
    if not 0 < args.deadline_unix - time.time() <= 7200:
        raise RuntimeError("CPU build monitor deadline must be within two hours")
    while start_ticks(args.pid) == args.expected_start_ticks:
        now = time.time()
        if now >= args.deadline_unix:
            os.kill(args.pid, signal.SIGTERM)
            time.sleep(10)
            if start_ticks(args.pid) == args.expected_start_ticks:
                os.kill(args.pid, signal.SIGKILL)
            status = "TERMINATED_AT_CPU_DEADLINE"
            break
        args.receipt.write_text(json.dumps({
            "host": "149", "pid": args.pid, "start_ticks": args.expected_start_ticks,
            "deadline_unix": args.deadline_unix, "status": "WATCHING",
            "checked_unix": now,
        }, sort_keys=True) + "\n")
        time.sleep(30)
    else:
        status = "PROCESS_EXITED"
    args.receipt.write_text(json.dumps({
        "host": "149", "pid": args.pid, "start_ticks": args.expected_start_ticks,
        "deadline_unix": args.deadline_unix, "status": status,
        "checked_unix": time.time(),
    }, sort_keys=True) + "\n")
    return 0 if status == "PROCESS_EXITED" else 124


if __name__ == "__main__":
    raise SystemExit(main())
