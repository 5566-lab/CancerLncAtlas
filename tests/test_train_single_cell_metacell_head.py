import numpy as np
import pandas as pd
import pytest
import torch
import json
from types import SimpleNamespace

from scripts.train_single_cell_metacell_head_149 import (
    PAIR_KEY,
    SignedMetacellHead,
    _donor_tuning_mask,
    _embedding_routes,
    _indices,
    _sample_donor_split_training,
    aggregate_oof,
    build_checkpoint_manifest,
    build_run_config,
    state_hash,
    validate_full_training_inputs,
)


def test_core_buffers_stay_frozen_and_unknown_embeddings_are_zero():
    rng = np.random.default_rng(3)
    model = SignedMetacellHead(
        rng.normal(size=(3, 96)).astype(np.float32),
        rng.normal(size=(2, 96)).astype(np.float32),
        rng.normal(size=(2, 96)).astype(np.float32),
        private_lncs=3,
        celltypes=3,
        numeric_features=8,
    )
    frozen_before = state_hash({
        key: value for key, value in model.state_dict().items()
        if key.endswith("_core")
    })
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
    prediction = model(
        torch.tensor([0, 2]), torch.tensor([0, 1]), torch.tensor([0, 1]),
        torch.tensor([1, 0]), torch.tensor([1, 0]), torch.zeros((2, 8)),
        torch.tensor([1.0, 0.0]),
    )
    assert torch.isfinite(prediction).all()
    prediction.sum().backward()
    optimizer.step()
    frozen_after = state_hash({
        key: value for key, value in model.state_dict().items()
        if key.endswith("_core")
    })
    assert frozen_after == frozen_before
    assert torch.count_nonzero(model.private_lnc.weight[0]).item() == 0
    assert torch.count_nonzero(model.celltype.weight[0]).item() == 0


def test_train_only_indices_use_zero_fallback_without_losing_core_route():
    frame = pd.DataFrame({
        "lncrna_id": ["LNC_TRAIN", "LNC_CORE_ONLY", "LNC_UNRESOLVED"],
        "pathway_id": ["P1", "P1", "P1"],
        "cancer_id": ["C1", "C1", "C1"],
        "cell_key": ["T", "UNSEEN", "UNSEEN"],
    })
    result = _indices(
        frame,
        ["LNC_TRAIN", "LNC_CORE_ONLY"], ["P1"], ["C1"],
        {"LNC_TRAIN": 1}, {"T": 1},
    )
    assert result["private"].tolist() == [1, 0, 0]
    assert result["private_available"].tolist() == [True, False, False]
    assert result["cell"].tolist() == [1, 0, 0]
    assert result["core_available"].tolist() == [1.0, 1.0, 0.0]
    # OOV core lookup uses the appended final all-zero core row.
    assert result["lnc"].tolist() == [0, 1, 2]
    assert _embedding_routes(
        result["core_available"], result["private_available"]
    ).tolist() == [
        "FROZEN_CORE_PLUS_PRIVATE_ID_ADAPTER",
        "FROZEN_CORE_WITH_ZERO_PRIVATE_FALLBACK",
        "SINGLE_CELL_OOV_ZERO_ID_FALLBACK",
    ]


def test_tuning_split_is_strictly_donor_blocked():
    frame = pd.DataFrame({
        "cancer_id": np.repeat(["A", "B"], 60),
        "query_donor_id": np.tile(np.repeat(["d1", "d2", "d3"], 20), 2),
    })
    mask = _donor_tuning_mask(frame, fold=2)
    fit = set(map(tuple, frame.loc[~mask, ["cancer_id", "query_donor_id"]].to_numpy()))
    tune = set(map(tuple, frame.loc[mask, ["cancer_id", "query_donor_id"]].to_numpy()))
    assert fit
    assert tune
    assert not fit.intersection(tune)


def test_sampling_happens_after_split_and_retains_fit_lnc_anchors():
    rows = []
    for donor in ["d1", "d2", "d3", "d4"]:
        for lnc in ["L_COMMON", f"L_{donor}"]:
            for pathway in range(6):
                rows.append({
                    "cancer_id": "C1",
                    "query_donor_id": donor,
                    "lncrna_id": lnc,
                    "pathway_id": f"P{pathway}",
                    "cell_key": "T",
                })
    frame = pd.DataFrame(rows)
    tuning_mask = frame.query_donor_id.eq("d4").to_numpy()
    sampled, sampled_tuning, receipt = _sample_donor_split_training(
        frame, tuning_mask, maximum=12, seed=7
    )
    fit_before = set(frame.loc[~tuning_mask, "lncrna_id"])
    fit_after = set(sampled.loc[~sampled_tuning, "lncrna_id"])
    assert len(sampled) == 12
    assert fit_after == fit_before
    assert set(sampled.loc[sampled_tuning, "query_donor_id"]) == {"d4"}
    assert receipt["fit_lncrna_anchors_required"] == len(fit_before)
    assert receipt["fit_lncrna_anchors_retained"] == len(fit_before)
    assert receipt["tuning_donor_anchors_required"] == 1
    assert receipt["tuning_donor_anchors_retained"] == 1


def test_sampling_fails_closed_when_required_anchors_exceed_budget():
    frame = pd.DataFrame({
        "cancer_id": ["C1"] * 5,
        "query_donor_id": ["fit"] * 4 + ["tune"],
        "lncrna_id": ["L1", "L2", "L3", "L4", "LT"],
        "pathway_id": ["P1"] * 5,
        "cell_key": ["T"] * 5,
    })
    with pytest.raises(RuntimeError, match="cannot retain every fit lncRNA"):
        _sample_donor_split_training(
            frame,
            frame.query_donor_id.eq("tune").to_numpy(),
            maximum=4,
            seed=9,
        )


def _oof_rows(rhos, *, bulk=False, discovery=True):
    rows = []
    for index, rho in enumerate(rhos):
        rows.append({
            "dataset_id": "D1", "cancer_id": "C1", "cell_key": "T",
            "query_donor_id": f"d{index}", "lncrna_id": "L1",
            "pathway_id": "P1", "outer_fold_id": index % 3,
            "rho": rho, "predicted_rho": rho * 0.9,
            "is_bulk_candidate": bulk,
            "is_sc_discovery_candidate": discovery,
            "lncrna_core_available": False, "n_metacells": 30,
        })
    return pd.DataFrame(rows)


def test_oof_aggregation_preserves_sign_and_discovers_new_pair():
    result = aggregate_oof(_oof_rows([0.70, 0.65, 0.75, 0.68]))
    assert len(result) == 1
    row = result.iloc[0]
    assert row.observed_meta_rho > 0.6
    assert row.predicted_meta_rho > 0
    assert row.direction_consistency == 1.0
    assert row.meta_fdr <= 0.05
    assert bool(row.new_relationship_discovered)
    assert set(PAIR_KEY).issubset(result.columns)


def test_zero_effect_does_not_count_as_negative_direction_support():
    result = aggregate_oof(_oof_rows([0.6, 0.5, 0.0]))
    assert result.iloc[0].direction_consistency == 1.0


def test_oof_duplicate_donor_target_is_rejected():
    frame = _oof_rows([0.4, 0.5, 0.6])
    with pytest.raises(RuntimeError, match="more than once"):
        aggregate_oof(pd.concat([frame, frame.iloc[[0]]], ignore_index=True))


def _write_full_gate(tmp_path):
    cancers = [f"C{index:02d}" for index in range(28)]
    for cancer in cancers:
        folder = tmp_path / "cancers" / cancer
        folder.mkdir(parents=True)
        (folder / "SUCCESS.json").write_text("{}\n", encoding="utf-8")
    validation = {
        "status": "PASS",
        "closure_status": "PASS_FULL_28_CANCER_MATERIALIZATION_CLOSURE",
        "validation_scope": "FULL",
        "expected_cancer_count": 28,
        "validated_cancer_count": 28,
        "expected_cancers": cancers,
        "validated_cancers": list(reversed(cancers)),
    }
    audit = {
        "status": "PASS",
        "active_cancer_count": 28,
        "active_cancers": cancers,
        "eligible_training_cells": 1234,
        "eligible_cells_failing_preprocessing_pass_qc": 0,
        "eligible_cells_failing_batch_correction_applied": 0,
        "eligible_cells_failing_annotation_completed": 0,
    }
    (tmp_path / "MATERIALIZATION_VALIDATION.json").write_text(
        json.dumps(validation), encoding="utf-8"
    )
    (tmp_path / "SINGLE_CELL_PREPROCESSING_AUDIT.json").write_text(
        json.dumps(audit), encoding="utf-8"
    )
    return cancers, validation


def test_full_training_gate_binds_exact_28_cancer_closure(tmp_path):
    cancers, _ = _write_full_gate(tmp_path)
    result = validate_full_training_inputs(tmp_path)
    assert result["materialization_expected_cancers"] == cancers
    assert result["preprocessing_audit_eligible_cells"] == 1234
    assert len(result["materialization_validation_sha256"]) == 64
    assert len(result["preprocessing_audit_sha256"]) == 64


def test_full_training_gate_rejects_partial_validation_masquerading_as_pass(tmp_path):
    _, validation = _write_full_gate(tmp_path)
    validation["closure_status"] = "PASS_SINGLE_CANCER_PILOT_VALIDATION"
    (tmp_path / "MATERIALIZATION_VALIDATION.json").write_text(
        json.dumps(validation), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="not a full closure"):
        validate_full_training_inputs(tmp_path)


def test_run_config_records_architecture_seeds_and_discovery_thresholds():
    settings = SimpleNamespace(
        seed=100, epochs=40, patience=6, batch_size=8192,
        max_train_rows=500000, learning_rate=0.001,
        allow_partial_materialization=False,
    )
    result = build_run_config(settings)
    assert result["format"] == "CANCERLNCATLAS_V32_SC_METACELL_RUN_CONFIG_V1"
    assert result["fold_seeds"] == [100, 201, 302, 403, 504]
    assert result["numeric_feature_columns"]
    assert result["architecture"]["input_dimensions"] == 345
    assert result["discovery_thresholds"]["maximum_meta_fdr"] == 0.05
    assert result["fresh_random_initialization"] is True
    assert result["old_checkpoint_loaded"] is False
    assert result["device"] == "cpu" and result["cuda_used"] is False


def test_checkpoint_manifest_requires_fresh_cpu_frozen_models(tmp_path):
    reports = []
    for fold in range(5):
        folder = tmp_path / f"outer_fold={fold}"
        folder.mkdir()
        (folder / "SUCCESS.json").write_text("{}\n", encoding="utf-8")
        reports.append({
            "outer_fold_id": fold,
            "seed": 10 + fold,
            "files": {
                "checkpoint": {
                    "path": str(folder / "best_model_state.pt"),
                    "sha256": "a" * 64,
                },
                "predictions": {
                    "path": str(folder / "QUERY_PREDICTIONS.parquet"),
                    "sha256": "b" * 64,
                },
            },
            "core_files": {
                "lncRNA": "d" * 64,
                "pathway": "e" * 64,
                "cancer": "f" * 64,
            },
            "initial_parameter_sha256": str(fold) * 64,
            "trained_parameter_sha256": str(fold + 1) * 64,
            "frozen_core_sha256_before": "c" * 64,
            "frozen_core_sha256_after": "c" * 64,
            "fresh_random_initialization": True,
            "old_checkpoint_loaded": False,
            "source_checkpoint_sha256": None,
            "device": "cpu",
            "cuda_used": False,
            "epochs_completed": 3,
            "best_validation_loss": 0.2,
            "history": [{"epoch": 0, "train_loss": 0.3, "validation_loss": 0.2}],
        })
    manifest = build_checkpoint_manifest(reports)
    assert manifest["format"] == (
        "CANCERLNCATLAS_V32_SC_METACELL_CHECKPOINT_MANIFEST_V1"
    )
    assert manifest["status"] == "SUCCESS"
    assert manifest["models_trained"] == 5
    assert manifest["fold_seeds"] == [10, 11, 12, 13, 14]
    assert all(item["history"] for item in manifest["records"])
    assert all("fold_success" in item for item in manifest["records"])
    reports[2]["old_checkpoint_loaded"] = True
    with pytest.raises(RuntimeError, match="lineage contract"):
        build_checkpoint_manifest(reports)
