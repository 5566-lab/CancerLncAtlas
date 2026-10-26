from __future__ import annotations

import sys
import os
from pathlib import Path
import pytest

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


@pytest.fixture(scope="session", autouse=True)
def code_only_atac_verification_guard():
    """Optional session guard: fail before any ATAC/auxiliary parameter fit.

    Enabled by the code-only ATAC verification launcher, including module
    fixtures. Existing explicit frozen-parameter substitutes remain possible.
    Ordinary tests outside that launcher retain their original behavior.
    """
    if os.environ.get("CANCERLNCATLAS_CODE_ONLY_ATAC_CHECKS") != "1":
        yield
        return
    import torch
    from cc_hhgt.v32 import atac_training, cnv_only_training_v2, genomic_training, multimodal_fusion
    def forbidden(*args, **kwargs):
        raise AssertionError("Code-only ATAC verification forbids parameter fitting")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(atac_training, "_fit_logistic", forbidden)
        for module in (cnv_only_training_v2, genomic_training): patch.setattr(module, "_fit_head", forbidden)
        for name in ("fit_residual_fusion_model", "train_pair_blocked_crossfit_fusion"):
            patch.setattr(multimodal_fusion, name, forbidden)
        for name in ("Adam", "AdamW", "SGD"): patch.setattr(torch.optim, name, forbidden)
        yield
