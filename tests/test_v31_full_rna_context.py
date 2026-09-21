from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "86_build_v31_full_rna_context.py"
SPEC = importlib.util.spec_from_file_location("build_v31_full_rna_context", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_registered_loco_cancers_requires_exact_formal_33(tmp_path) -> None:
    cancers = [*[f"C{i:02d}" for i in range(31)], "HNSC", "LGG"]
    path = tmp_path / "folds.tsv"
    pd.DataFrame({"test_cancer": cancers}).to_csv(path, sep="\t", index=False)
    assert MODULE.registered_loco_cancers(path) == tuple(cancers)
    pd.DataFrame({"test_cancer": cancers[:-1]}).to_csv(
        path, sep="\t", index=False
    )
    with pytest.raises(RuntimeError, match="Expected 33"):
        MODULE.registered_loco_cancers(path)
