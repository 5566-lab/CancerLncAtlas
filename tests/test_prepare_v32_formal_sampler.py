from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_v32_formal.py"
SPEC = importlib.util.spec_from_file_location("prepare_v32_formal", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _scope() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cancer_id": ["A", "A", "A", "B", "B"],
            "lncrna_id": ["L1", "L2", "L3", "L1", "L4"],
            "within_cancer_eligible": [True, True, True, True, True],
        }
    )


def test_affine_candidate_sampler_is_deterministic_balanced_and_unique() -> None:
    pathways = [f"P{i}" for i in range(11)]
    first = MODULE.make_candidate_pairs(_scope(), pathways, budget=17, seed=7)
    second = MODULE.make_candidate_pairs(_scope(), pathways, budget=17, seed=7)

    pd.testing.assert_frame_equal(first, second)
    assert first.groupby("cancer_id").size().to_dict() == {"A": 17, "B": 17}
    assert not first.duplicated(["cancer_id", "lncrna_id", "pathway_id"]).any()
    assert set(first.pathway_id).issubset(pathways)
    assert first.groupby(["cancer_id", "lncrna_id"]).size().min() >= 1


def test_affine_candidate_sampler_changes_with_seed() -> None:
    pathways = [f"P{i}" for i in range(31)]
    first = MODULE.make_candidate_pairs(_scope(), pathways, budget=17, seed=7)
    second = MODULE.make_candidate_pairs(_scope(), pathways, budget=17, seed=8)
    assert not first.equals(second)


def test_affine_candidate_sampler_rejects_impossible_budget() -> None:
    with pytest.raises(RuntimeError, match="exceeds the no-replacement"):
        MODULE.make_candidate_pairs(_scope(), ["P1", "P2"], budget=7, seed=7)
