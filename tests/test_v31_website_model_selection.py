from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "76_select_v31_exact_pathway_website_model.py"
SPEC = importlib.util.spec_from_file_location("v31_website_model_selection", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _validation() -> pd.DataFrame:
    labels = [0, 0, 1, 1]
    scores = {
        "cc_hhgt": [0.05, 0.15, 0.85, 0.95],
        "hgt": [0.10, 0.70, 0.30, 0.90],
        "rgcn": [0.90, 0.80, 0.20, 0.10],
    }
    rows = []
    for fold_index in range(MODULE.EXPECTED_FOLDS):
        fold = f"LOCO_C{fold_index:02d}"
        for model in MODULE.MODELS:
            for seed_index, seed in enumerate(MODULE.SEEDS):
                for candidate_index, label in enumerate(labels):
                    rows.append(
                        {
                            "fold_id": fold,
                            "candidate_id": f"{fold}:P{candidate_index}",
                            "cancer_id": f"C{fold_index:02d}",
                            "lncrna_id": f"L{candidate_index}",
                            "pathway_id": f"PW{candidate_index}",
                            "pathway_family_id": f"PF{candidate_index % 2}",
                            "proxy_label": label,
                            "model": model,
                            "seed": seed,
                            "proxy_positive_probability": scores[model][candidate_index]
                            + seed_index * 1e-4,
                        }
                    )
    return pd.DataFrame(rows)


def test_selects_architecture_from_seed_ensemble_validation_only() -> None:
    metrics, summary, selection = MODULE.select_architecture(_validation())
    assert len(metrics) == MODULE.EXPECTED_FOLDS * len(MODULE.MODELS)
    assert set(metrics.groupby("model").fold_id.nunique()) == {MODULE.EXPECTED_FOLDS}
    assert selection["selected_model"] == "cc_hhgt"
    assert selection["selection_split"] == "val_only"
    assert selection["test_metrics_used"] is False
    assert summary.iloc[0].model == "cc_hhgt"


def test_rejects_candidate_or_label_drift_across_models() -> None:
    frame = _validation()
    hit = (
        frame.fold_id.eq("LOCO_C00")
        & frame.model.eq("hgt")
        & frame.candidate_id.eq("LOCO_C00:P0")
    )
    frame.loc[hit, "proxy_label"] = 1
    with pytest.raises(RuntimeError, match="identities or labels drift"):
        MODULE.select_architecture(frame)


def test_validation_loader_projects_only_registered_validation_rows(tmp_path: Path) -> None:
    task = tmp_path / "task"
    task.mkdir()
    (task / "SUCCESS.json").write_text(
        json.dumps(
            {
                "status": "COMPLETED",
                "run_id": "RUN",
                "hit_hard_epoch_cap": False,
            }
        ),
        encoding="utf-8",
    )
    rows = pd.DataFrame(
        {
            "candidate_id": ["VAL", "TEST"],
            "cancer_id": ["A", "B"],
            "lncrna_id": ["L1", "L2"],
            "pathway_id": ["P1", "P2"],
            "pathway_family_id": ["F1", "F2"],
            "proxy_label": [1, 0],
            "split": ["val", "test"],
            "proxy_positive_probability": [0.8, 0.2],
        }
    )
    rows.to_parquet(task / "prediction_pathway_calibrated.parquet", index=False)
    loaded = MODULE._validation_frame(task, "RUN", "hgt", "LOCO_A", MODULE.SEEDS[0])
    assert loaded.candidate_id.tolist() == ["VAL"]
    assert loaded.split.tolist() == ["val"]


def test_runner_summary_accepts_completed_multigpu_matrix(tmp_path: Path) -> None:
    control = tmp_path / "run_control"
    control.mkdir()
    (control / "PARALLEL_SUMMARY.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "run_id": "RUN",
                "n_tasks": MODULE.EXPECTED_TASKS,
                "completed": MODULE.EXPECTED_TASKS,
                "mode": f"FULL_{MODULE.EXPECTED_TASKS}_MULTIGPU",
            }
        ),
        encoding="utf-8",
    )
    loaded = MODULE._load_runner_summary(tmp_path, "RUN")
    assert loaded["status"] == "PASS"


def test_matrix_audit_must_prove_release_eligibility(tmp_path: Path) -> None:
    path = tmp_path / "AUDIT.json"
    valid = {
        "status": "PASS",
        "run_id": "RUN",
        "expected_tasks": MODULE.EXPECTED_TASKS,
        "verified_tasks": MODULE.EXPECTED_TASKS,
        "release_eligible": True,
        "all_tasks_converged_before_hard_epoch_cap": True,
        "candidate_keys_and_labels_identical_across_model_seed": True,
    }
    path.write_text(json.dumps(valid), encoding="utf-8")
    assert MODULE._load_matrix_audit(path, "RUN")["release_eligible"] is True
    valid["candidate_keys_and_labels_identical_across_model_seed"] = False
    path.write_text(json.dumps(valid), encoding="utf-8")
    with pytest.raises(RuntimeError, match="matrix audit"):
        MODULE._load_matrix_audit(path, "RUN")
