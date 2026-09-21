from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    value = str(candidate)
    if value not in sys.path:
        sys.path.insert(0, value)

# The delivery container used for code-contract tests may not ship pyarrow.
# Production installs it from requirements-base.txt.  A minimal stub allows
# pure-math unit tests to import cc_hhgt.common without exercising parquet IO.
try:
    import pyarrow  # noqa: F401
except ModuleNotFoundError:
    import types
    pa = types.ModuleType("pyarrow")
    pa.__version__ = "0.0.0"
    pa.lib = types.SimpleNamespace(ArrowTypeError=type("ArrowTypeError", (Exception,), {}))
    pa.Table = type("Table", (), {})
    pa.RecordBatch = type("RecordBatch", (), {})
    pa.Array = type("Array", (), {})
    pa.ChunkedArray = type("ChunkedArray", (), {})
    sys.modules["pyarrow"] = pa
