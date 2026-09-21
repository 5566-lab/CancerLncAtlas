#!/usr/bin/env python3
"""CLI wrapper for fresh V3.2 seven-State private-head training."""
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.state_training import main


if __name__ == "__main__":
    raise SystemExit(main())
