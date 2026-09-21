from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cc_hhgt.v31_exact_baselines import (
    FORBIDDEN_PAIR_EVIDENCE,
    LABEL_DERIVED_AVAILABILITY,
    OUTCOME_INDEPENDENT_NUMERIC,
    fit_sparse_logistic_baseline,
    hashed_design,
)


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "77_train_v31_exact_pathway_sparse_baseline.py"
SPEC = importlib.util.spec_from_file_location("v31_exact_sparse_baseline_runner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def _frame(prefix: str, n: int = 40) -> pd.DataFrame:
    index = np.arange(n)
    label = (index % 2).astype(int)
    return pd.DataFrame(
        {
            "candidate_id": [f"{prefix}:{i}" for i in index],
            "cancer_id": prefix,
            "lncrna_id": np.where(label == 1, "L_POS", "L_NEG"),
            "pathway_id": [f"P{i % 5}" for i in index],
            "pathway_family_id": [f"F{i % 2}" for i in index],
            "proxy_label": label,
            "label_class": np.where(label == 1, "strong_positive", "unlabeled"),
            "bulk_available": True,
            "sc_available": False,
            "bulk_detection_rate": 0.5,
            "sc_detection_rate": 0.0,
            **{column: index.astype(float) for column in FORBIDDEN_PAIR_EVIDENCE},
        }
    )


def test_hash_design_ignores_cancer_and_all_pair_evidence() -> None:
    original = _frame("A")
    changed = original.copy()
    changed["cancer_id"] = "HELD_OUT_CHANGED"
    for column in FORBIDDEN_PAIR_EVIDENCE:
        changed[column] = changed[column] * -999 - 123
    # These flags are created only after an association/evidence row passes
    # its qualification threshold, so they encode the proxy-label observation
    # process and must not enter a matched baseline.
    for column in LABEL_DERIVED_AVAILABILITY:
        changed[column] = np.arange(len(changed)) % 2 == 0
    left = hashed_design(original, n_features=4096)
    right = hashed_design(changed, n_features=4096)
    assert (left != right).nnz == 0


def test_only_expression_detection_rates_are_numeric_predictors() -> None:
    assert OUTCOME_INDEPENDENT_NUMERIC == (
        "bulk_detection_rate",
        "sc_detection_rate",
    )
    assert {"bulk_available", "sc_available"}.issubset(
        set(LABEL_DERIVED_AVAILABILITY)
    )


def test_l1_and_l2_select_on_validation_and_predict_test() -> None:
    train = _frame("TRAIN", 80)
    validation = _frame("VAL", 40)
    test = _frame("TEST", 40)
    for penalty in ("l1", "l2"):
        fit = fit_sparse_logistic_baseline(
            train,
            validation,
            test,
            penalty=penalty,
            c_grid=(0.1, 1.0),
            seed=7,
            n_features=4096,
            max_iter=1000,
        )
        assert fit.selected_c in {0.1, 1.0}
        assert len(fit.test_probability) == len(test)
        assert np.all((fit.test_probability >= 0) & (fit.test_probability <= 1))
        assert fit.feature_contract["test_used_for_tuning"] is False
        assert fit.feature_contract["cancer_id_feature"] is False
        assert fit.feature_contract["numeric_features"] == [
            "bulk_detection_rate",
            "sc_detection_rate",
        ]
        assert {"bulk_available", "sc_available"}.issubset(
            set(fit.feature_contract["forbidden_label_derived_availability"])
        )


def test_candidate_match_includes_exact_pathway_identity_and_label() -> None:
    frame = _frame("MATCH", 20)
    reference = frame[RUNNER.IDENTITY + []].copy()
    digest = RUNNER._assert_matched(frame, reference, "test")
    assert len(digest) == 64
    reference.loc[0, "pathway_id"] = "DRIFT"
    with pytest.raises(RuntimeError, match="differs from graph task"):
        RUNNER._assert_matched(frame, reference, "test")


def test_matrix_audit_must_prove_all_297_converged_tasks(tmp_path: Path) -> None:
    path = tmp_path / "AUDIT.json"
    valid = {
        "status": "PASS",
        "run_id": "RUN",
        "expected_tasks": RUNNER.EXPECTED_TASKS,
        "verified_tasks": RUNNER.EXPECTED_TASKS,
        "release_eligible": True,
        "all_tasks_converged_before_hard_epoch_cap": True,
        "candidate_keys_and_labels_identical_across_model_seed": True,
    }
    path.write_text(json.dumps(valid), encoding="utf-8")
    assert RUNNER._require_matrix_audit(path, "RUN")["status"] == "PASS"
    valid["all_tasks_converged_before_hard_epoch_cap"] = False
    path.write_text(json.dumps(valid), encoding="utf-8")
    with pytest.raises(RuntimeError, match="release-eligible"):
        RUNNER._require_matrix_audit(path, "RUN")
