from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from cc_hhgt.v31_current_expected_downstream import (
    DOWNSTREAM_TASKS,
    FROZEN_REFERENCE_SHA256,
    _acceptance,
    _metric_rows,
    _normalize_direction,
    build_current_atlas_scores,
)
from cc_hhgt.v31_phase_c import PILOT_CANCERS, PILOT_SEEDS


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reference_rows = []
    simple_rows = []
    expected_rows = []
    direction_rows = []
    targets = ("stemness_rna::RNAss", "stemness_dna::DNAss")
    for cancer_index, cancer in enumerate(PILOT_CANCERS):
        for target in targets:
            subtype = "RNAss" if "RNAss" in target else "DNAss"
            for fold in range(5):
                for index in range(12):
                    lnc = f"LNC:ENSG{index:011d}"
                    specificity = int(index % 4 == 0)
                    conservation = int(index % 4 == 1)
                    reversal = int(index % 4 == 2)
                    keys = {
                        "cancer_id": cancer,
                        "patient_fold_id": f"LOCO_{cancer}__PF{fold:02d}",
                        "lncrna_id": lnc,
                        "target_id": target,
                    }
                    reference_rows.append(
                        {
                            **keys,
                            "target_type": "state",
                            "observed_effect_signed": (-1.0 if index % 2 else 1.0) * 0.2,
                            "observed_strength_score": (index + 1) / 13,
                            "observed_reproducibility_probability": (index + 1) / 13,
                            "heldout_specificity_label": specificity,
                            "heldout_conservation_label": conservation,
                            "heldout_reversal_label": reversal,
                        }
                    )
                    simple_rows.append(
                        {
                            **keys,
                            "best_simple_specificity_method": "C0",
                            "best_simple_specificity_score": index / 12,
                            "best_simple_conservation_method": "K0",
                            "best_simple_conservation_score": (12 - index) / 12,
                            "best_simple_reversal_method": "R0",
                            "best_simple_reversal_score": float(index % 2),
                        }
                    )
            for seed in PILOT_SEEDS:
                for index in range(12):
                    lnc = f"LNC:ENSG{index:011d}"
                    expected_rows.append(
                        {
                            "loco_cancer": cancer,
                            "seed": seed,
                            "target_subtype": subtype,
                            "split": "test",
                            "lncrna_id": lnc,
                            "state_id": target,
                            "proxy_positive_probability": (
                                (12 - index + cancer_index) / 15
                            ),
                            "model_variant": "R_SELECTED",
                        }
                    )
                    direction_rows.append(
                        {
                            "loco_cancer": cancer,
                            "seed": seed,
                            "target_subtype": subtype,
                            "split": "test",
                            "lncrna_id": lnc,
                            "state_id": target,
                            "direction_positive_probability": 0.8,
                            "prediction_scale": "raw_probability",
                            "module_admitted": True,
                        }
                    )
    return tuple(
        map(pd.DataFrame, (reference_rows, simple_rows, expected_rows, direction_rows))
    )


def test_score_construction_is_invariant_to_heldout_labels() -> None:
    reference, simple, expected, direction = _inputs()
    first, score_first = build_current_atlas_scores(
        reference, simple, expected, direction
    )
    permuted = reference.copy()
    for column in (
        "heldout_specificity_label",
        "heldout_conservation_label",
        "heldout_reversal_label",
    ):
        permuted[column] = permuted[column].sample(
            frac=1.0, random_state=31
        ).to_numpy()
    second, score_second = build_current_atlas_scores(
        permuted, simple, expected, direction
    )
    pd.testing.assert_frame_equal(
        score_first.sort_values(list(score_first.columns[:5])).reset_index(drop=True),
        score_second.sort_values(list(score_second.columns[:5])).reset_index(drop=True),
    )
    assert not first.heldout_specificity_label.equals(
        second.heldout_specificity_label
    )
    assert first.score_construction_used_heldout_labels.eq(False).all()


def test_rejected_graph_direction_uses_neutral_fallback() -> None:
    _, _, _, direction = _inputs()
    direction["module_admitted"] = False
    normalized = _normalize_direction(direction)
    assert normalized.pan_cancer_direction_positive_probability.eq(0.5).all()


def test_exact_three_task_metrics_and_cluster_acceptance() -> None:
    reference, simple, expected, direction = _inputs()
    atlas, _ = build_current_atlas_scores(reference, simple, expected, direction)
    metrics = {task: _metric_rows(atlas, task) for task in DOWNSTREAM_TASKS}
    acceptance = _acceptance(metrics, 200)
    assert set(acceptance.task) == set(DOWNSTREAM_TASKS)
    assert acceptance.n_clusters.eq(15).all()
    assert acceptance.total_cancers.eq(3).all()
    assert acceptance.test_metric_used_for_selection.eq(False).all()
    for frame in metrics.values():
        assert set(frame.cancer_id) == set(PILOT_CANCERS)
        assert set(frame.seed.astype(int)) == set(PILOT_SEEDS)
        assert np.isfinite(frame.auprc).all()
        assert np.isfinite(frame.positive_prevalence).all()


def test_runner_is_hard_go_only_and_has_no_training_entrypoint() -> None:
    root = Path(__file__).parents[1]
    path = root / "scripts" / "100_run_v31_current_expected_downstream.py"
    source = path.read_text(encoding="utf-8")
    module_source = (
        root / "cc_hhgt" / "v31_current_expected_downstream.py"
    ).read_text(encoding="utf-8")
    assert "HARD_GO_B3" in module_source
    assert "full_cancer_training_started" in module_source
    assert "test_labels_used_for_score_construction" in module_source
    assert "torch" not in source.lower()
    spec = importlib.util.spec_from_file_location("current_expected_runner", path)
    loaded = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(loaded)
    assert set(FROZEN_REFERENCE_SHA256) == {
        "tables/ATLAS_REFERENCE_ALL.parquet",
        "SPECIFICITY_SIMPLE_BASELINE_PREDICTIONS.parquet",
        "OUTPUT_SHA256_MANIFEST.tsv",
        "V31_DUAL_AXIS_PILOT_SUMMARY.json",
    }
    assert all(len(value) == 64 for value in FROZEN_REFERENCE_SHA256.values())
