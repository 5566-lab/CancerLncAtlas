"""Opt-in bootstrap for the hash-pinned CancerLncAtlas Pandas wheel overlay."""
from __future__ import annotations

import os


root = os.environ.get("CANCERLNCATLAS_PANDAS_EXTENSION_ROOT")
if root:
    from pandas_extension_overlay import install

    install(root)
