#!/usr/bin/env python3
"""Thin V3.2 CLI wrapper; safe to invoke before package installation."""
from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cc_hhgt.v32.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
