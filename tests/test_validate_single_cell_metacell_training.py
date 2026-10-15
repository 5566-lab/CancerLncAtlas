from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "validate_single_cell_metacell_training_149.py"
)
SPEC = importlib.util.spec_from_file_location(
    "single_cell_metacell_training_validator", SCRIPT
)
auditor = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(auditor)


def _oof_rows(
    rhos: list[float],
    *,
    folds: list[int] | None = None,
    bulk: bool = False,
    discovery: bool = True,
    pathway: str = "P1",
) -> pd.DataFrame:
    if folds is None:
        folds = [index % 3 for index in range(len(rhos))]
    return pd.DataFrame(
        [
            {
                "dataset_id": "D1",
                "cancer_id": "C1",
                "cell_key": "T",
                "query_donor_id": f"d{index}",
                "lncrna_id": "L1",
                "pathway_id": pathway,
                "outer_fold_id": folds[index],
                "rho": rho,
                "predicted_rho": rho * 0.8,
                "predicted_association_strength": abs(rho * 0.8),
                "residual": rho - rho * 0.8,
                "is_bulk_candidate": bulk,
                "is_sc_discovery_candidate": discovery,
                "lncrna_core_available": False,
                "private_lnc_embedding_available": True,
                "celltype_embedding_available": True,
                "embedding_route": "SINGLE_CELL_PRIVATE_OOV_ID_EMBEDDING",
                "n_metacells": 30,
                "lnc_detection_rate": 0.9,
                "source_donor_count": 4,
                "source_meta_rho": 0.4,
                "source_direction_consistency": 0.8,
                "measurement_scale": "LOG_NORMALIZED_EXPRESSION",
                "candidate_selection_query_donor_used": False,
                "association_method": "WITHIN_DONOR_METACELL_SPEARMAN",
                "association_observation_unit": "within_donor_metacell",
                "donor_resolved_target": True,
            }
            for index, rho in enumerate(rhos)
        ]
    )


def test_independent_bh_is_monotone_and_preserves_nan_positions():
    adjusted = auditor._bh(pd.Series([0.04, 0.01, np.nan, 0.03, 0.20]))
    np.testing.assert_allclose(
        adjusted[[0, 1, 3, 4]], [0.05333333333333334, 0.04, 0.05333333333333334, 0.20]
    )
    assert np.isnan(adjusted[2])

    with pytest.raises(auditor.IndependentAuditError, match="outside"):
        auditor._bh(pd.Series([0.1, 1.01]))


def test_independent_aggregate_recomputes_fisher_effect_and_bh_by_family():
    first = _oof_rows([0.70, 0.65, 0.75, 0.68], pathway="P1")
    second = _oof_rows([0.05, -0.05, 0.04, -0.04], pathway="P2")
    result = auditor.independently_aggregate_oof(
        pd.concat([first, second], ignore_index=True)
    ).set_index("pathway_id")

    expected = np.tanh(np.mean(np.arctanh([0.70, 0.65, 0.75, 0.68])))
    assert result.loc["P1", "observed_meta_rho"] == pytest.approx(expected)
    assert result.loc["P1", "direction_consistency"] == 1.0
    assert result.loc["P1", "meta_fdr"] <= 0.05
    assert bool(result.loc["P1", "new_relationship_discovered"])
    assert not bool(result.loc["P2", "new_relationship_discovered"])
    assert set(auditor.PAIR_KEY).issubset(result.reset_index().columns)


@pytest.mark.parametrize(
    ("frame", "reason"),
    [
        (_oof_rows([0.70, 0.65, 0.75, 0.68], bulk=True), "bulk candidate"),
        (_oof_rows([0.70, 0.65, 0.75, 0.68], discovery=False), "not screened"),
        (_oof_rows([0.70, 0.65]), "fewer than three donors"),
        (
            _oof_rows([0.70, 0.65, 0.75, 0.68], folds=[1, 1, 1, 1]),
            "fewer than two folds",
        ),
        (_oof_rows([0.19, 0.19, 0.19, 0.19]), "effect below threshold"),
        (
            _oof_rows([0.90, 0.90, 0.90, -0.10, -0.10]),
            "direction consistency below threshold",
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_discovery_requires_every_declared_threshold(frame, reason):
    row = auditor.independently_aggregate_oof(frame).iloc[0]
    assert not bool(row.new_relationship_discovered), reason


def _seeded_checkpoint(seed: int = 17) -> dict:
    rng = np.random.default_rng(4)
    lnc_core = rng.normal(size=(3, 96)).astype(np.float32)
    pathway_core = rng.normal(size=(2, 96)).astype(np.float32)
    cancer_core = rng.normal(size=(2, 96)).astype(np.float32)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = auditor.IndependentSignedMetacellHead(
            lnc_core,
            pathway_core,
            cancer_core,
            private_lncs=3,
            celltypes=3,
            numeric_features=8,
        )
    return {
        "seed": seed,
        "state_dict": model.state_dict(),
        "lncrna_vocabulary": ["L1", "L2"],
        "celltype_vocabulary": ["T", "B"],
        "numeric_columns": [f"f{index}" for index in range(8)],
    }


def test_checkpoint_initial_state_is_rebuilt_without_importing_trainer():
    checkpoint = _seeded_checkpoint()
    expected = auditor.state_hash(checkpoint["state_dict"])
    assert auditor.rebuild_initial_state_hash(checkpoint) == expected

    trained = {key: value.clone() for key, value in checkpoint["state_dict"].items()}
    trained["network.0.weight"][0, 0] += 0.25
    assert auditor.state_hash(trained) != expected

    frozen_names = ("lnc_core", "pathway_core", "cancer_core")
    initial_frozen = {name: checkpoint["state_dict"][name] for name in frozen_names}
    trained_frozen = {name: trained[name] for name in frozen_names}
    assert auditor.state_hash(initial_frozen) == auditor.state_hash(trained_frozen)
    assert all(torch.isfinite(value).all() for value in trained.values())


def test_auditor_source_never_imports_training_implementation():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "from scripts.train_single_cell_metacell_head_149" not in source
    assert "import scripts.train_single_cell_metacell_head_149" not in source
    assert "import train_single_cell_metacell_head_149" not in source


def test_checkpoint_rebuild_fails_closed_on_missing_core():
    checkpoint = _seeded_checkpoint()
    del checkpoint["state_dict"]["cancer_core"]
    with pytest.raises(auditor.IndependentAuditError, match="cancer_core"):
        auditor.rebuild_initial_state_hash(checkpoint)


def _record(path: Path) -> dict:
    return {
        "path": str(path),
        "sha256": auditor.sha256(path),
        "bytes": path.stat().st_size,
    }


def _write_core(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        values,
        columns=[f"core_feature_{index:03d}" for index in range(96)],
    )
    frame.insert(0, "entity_id", [f"E{index}" for index in range(len(frame))])
    frame.to_parquet(path, index=False)


def _checkpoint_audit_case(tmp_path: Path) -> dict:
    fold = 2
    base_seed = 10
    seed = base_seed + fold * 101
    rng = np.random.default_rng(8)
    core_root = tmp_path / "core"
    core_by_fold = {}
    manifest_folds = {}
    for core_fold in range(auditor.N_FOLDS):
        core_dir = core_root / f"patient_fold={core_fold}"
        arrays = {
            "lncRNA": rng.normal(size=(2, 96)).astype(np.float32),
            "pathway": rng.normal(size=(2, 96)).astype(np.float32),
            "cancer": rng.normal(size=(2, 96)).astype(np.float32),
        }
        paths = {name: core_dir / f"{name}.parquet" for name in arrays}
        for name, path in paths.items():
            _write_core(path, arrays[name])
        core_by_fold[core_fold] = (arrays, paths)
        manifest_folds[str(core_fold)] = {
            "patient_fold": core_fold,
            "old_checkpoint_loaded": False,
            "trained_from_scratch": True,
            "partial_chunk_smoke": False,
            "exports": {
                name: {
                    "path": f"patient_fold={core_fold}/{name}.parquet",
                    "sha256": auditor.sha256(path),
                    "features": 96,
                    "rows": len(arrays[name]),
                }
                for name, path in paths.items()
            },
        }
    _write_json(
        core_root / "CORE_EMBEDDING_MANIFEST.json",
        {
            "all_embeddings_from_newly_trained_v32_core": True,
            "historical_checkpoint_loaded": False,
            "historical_prediction_loaded": False,
            "folds": manifest_folds,
        },
    )
    core_arrays, core_paths = core_by_fold[fold]

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = auditor.IndependentSignedMetacellHead(
            np.vstack([core_arrays["lncRNA"], np.zeros((1, 96), dtype=np.float32)]),
            core_arrays["pathway"],
            core_arrays["cancer"],
            private_lncs=3,
            celltypes=3,
            numeric_features=8,
        )
    initial_state = {name: value.clone() for name, value in model.state_dict().items()}
    initial_hash = auditor.state_hash(initial_state)
    trained_state = {name: value.clone() for name, value in initial_state.items()}
    trained_state["network.0.weight"][0, 0] += 0.25
    trained_hash = auditor.state_hash(trained_state)
    frozen = {name: value for name, value in trained_state.items() if name.endswith("_core")}
    frozen_hash = auditor.state_hash(frozen)
    history = [
        {"epoch": 0, "train_loss": 0.4, "validation_loss": 0.3},
        {"epoch": 1, "train_loss": 0.2, "validation_loss": 0.1},
    ]
    payload = {
        "format": auditor.TRAINING_FORMAT,
        "outer_fold_id": fold,
        "seed": seed,
        "state_dict": trained_state,
        "lncrna_vocabulary": ["L1", "L2"],
        "celltype_vocabulary": ["B", "T"],
        "numeric_columns": auditor.FEATURE_COLUMNS,
        "numeric_mean": np.zeros(len(auditor.FEATURE_COLUMNS)),
        "numeric_sd": np.ones(len(auditor.FEATURE_COLUMNS)),
        "unknown_identity_index": 0,
        "fresh_random_initialization": True,
        "old_checkpoint_loaded": False,
        "source_checkpoint_sha256": None,
        "device": "cpu",
        "cuda_used": False,
        "initial_parameter_sha256": initial_hash,
        "trained_parameter_sha256": trained_hash,
        "frozen_core_sha256_before": frozen_hash,
        "frozen_core_sha256_after": frozen_hash,
        "history": history,
    }
    fold_dir = tmp_path / f"outer_fold={fold}"
    fold_dir.mkdir()
    checkpoint_path = fold_dir / "MODEL_CHECKPOINT.pt"
    prediction_path = fold_dir / "QUERY_PREDICTIONS.parquet"
    torch.save(payload, checkpoint_path)
    pd.DataFrame({"query_donor_id": ["d1"]}).to_parquet(
        prediction_path, index=False
    )
    core_hashes = {name: auditor.sha256(path) for name, path in core_paths.items()}
    receipt = {
        "seed": seed,
        "fresh_random_initialization": True,
        "old_checkpoint_loaded": False,
        "source_checkpoint_sha256": None,
        "device": "cpu",
        "cuda_used": False,
        "files": {
            "checkpoint": _record(checkpoint_path),
            "predictions": _record(prediction_path),
        },
        "initial_parameter_sha256": initial_hash,
        "trained_parameter_sha256": trained_hash,
        "frozen_core_sha256_before": frozen_hash,
        "frozen_core_sha256_after": frozen_hash,
        "core_files": core_hashes,
        "history": history,
        "epochs_completed": len(history),
        "best_validation_loss": 0.1,
    }
    manifest = {
        "seed": seed,
        "checkpoint": _record(checkpoint_path),
        "predictions": _record(prediction_path),
        "fresh_random_initialization": True,
        "old_checkpoint_loaded": False,
        "source_checkpoint_sha256": None,
        "device": "cpu",
        "cuda_used": False,
        "initial_parameter_sha256": initial_hash,
        "trained_parameter_sha256": trained_hash,
        "frozen_core_sha256_before": frozen_hash,
        "frozen_core_sha256_after": frozen_hash,
        "core_files": core_hashes,
    }
    return {
        "fold": fold,
        "base_seed": base_seed,
        "core_root": core_root,
        "checkpoint_path": checkpoint_path,
        "payload": payload,
        "receipt": receipt,
        "manifest": manifest,
        "initial_state": initial_state,
    }


def _rewrite_checkpoint(case: dict) -> None:
    torch.save(case["payload"], case["checkpoint_path"])
    record = _record(case["checkpoint_path"])
    case["receipt"]["files"]["checkpoint"] = record
    case["manifest"]["checkpoint"] = record


def _audit_checkpoint(case: dict) -> dict:
    return auditor.audit_checkpoint(
        fold=case["fold"],
        checkpoint_path=case["checkpoint_path"],
        fold_receipt=case["receipt"],
        manifest_record=case["manifest"],
        core_root=case["core_root"],
        base_seed=case["base_seed"],
        epochs=2,
        patience=2,
    )


def test_checkpoint_audit_rebuilds_initial_and_checks_frozen_finite_training(tmp_path):
    result = _audit_checkpoint(_checkpoint_audit_case(tmp_path))
    assert result["initial_parameter_sha256"] == result[
        "independently_rebuilt_initial_parameter_sha256"
    ]
    assert result["trained_parameter_sha256"] != result["initial_parameter_sha256"]
    assert result["all_tensors_finite"] is True
    assert result["old_checkpoint_loaded"] is False


def test_checkpoint_audit_rejects_nonfinite_state(tmp_path):
    case = _checkpoint_audit_case(tmp_path)
    case["payload"]["state_dict"]["network.0.weight"][0, 0] = float("nan")
    _rewrite_checkpoint(case)
    with pytest.raises(auditor.IndependentAuditError, match="non-finite state"):
        _audit_checkpoint(case)


def test_checkpoint_audit_rejects_initial_state_masquerading_as_trained(tmp_path):
    case = _checkpoint_audit_case(tmp_path)
    case["payload"]["state_dict"] = case["initial_state"]
    initial_hash = auditor.state_hash(case["initial_state"])
    case["payload"]["trained_parameter_sha256"] = initial_hash
    case["receipt"]["trained_parameter_sha256"] = initial_hash
    case["manifest"]["trained_parameter_sha256"] = initial_hash
    _rewrite_checkpoint(case)
    with pytest.raises(auditor.IndependentAuditError, match="trained parameter hash"):
        _audit_checkpoint(case)


def test_checkpoint_audit_rejects_changed_frozen_core(tmp_path):
    case = _checkpoint_audit_case(tmp_path)
    case["payload"]["frozen_core_sha256_after"] = "0" * 64
    _rewrite_checkpoint(case)
    with pytest.raises(auditor.IndependentAuditError, match="frozen-core hash"):
        _audit_checkpoint(case)


def test_checkpoint_audit_rejects_nonzero_identity_padding_row(tmp_path):
    case = _checkpoint_audit_case(tmp_path)
    case["payload"]["state_dict"]["private_lnc.weight"][0, 0] = 0.5
    trained_hash = auditor.state_hash(case["payload"]["state_dict"])
    case["payload"]["trained_parameter_sha256"] = trained_hash
    case["receipt"]["trained_parameter_sha256"] = trained_hash
    case["manifest"]["trained_parameter_sha256"] = trained_hash
    _rewrite_checkpoint(case)
    with pytest.raises(auditor.IndependentAuditError, match="padding row"):
        _audit_checkpoint(case)


def test_embedding_routes_cover_all_core_private_availability_combinations():
    routes = pd.DataFrame(
        {
            "lncrna_core_available": [True, True, False, False],
            "private_lnc_embedding_available": [True, False, True, False],
            "embedding_route": [
                "FROZEN_CORE_PLUS_PRIVATE_ID_ADAPTER",
                "FROZEN_CORE_WITH_ZERO_PRIVATE_FALLBACK",
                "SINGLE_CELL_PRIVATE_OOV_ID_EMBEDDING",
                "SINGLE_CELL_OOV_ZERO_ID_FALLBACK",
            ],
        }
    )
    report = auditor.validate_embedding_routes(routes)
    assert report["zero_identity_fallback_rows"] == 1
    assert set(report["embedding_routes"]) == set(routes.embedding_route)

    routes.loc[3, "embedding_route"] = "SINGLE_CELL_PRIVATE_OOV_ID_EMBEDDING"
    with pytest.raises(auditor.IndependentAuditError, match="route/availability"):
        auditor.validate_embedding_routes(routes)


def test_oof_duplicate_donor_target_is_rejected():
    frame = _oof_rows([0.4, 0.5, 0.6])
    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(auditor.IndependentAuditError, match="duplicated"):
        auditor.independently_aggregate_oof(duplicated)


def _oof_isolation_case(tmp_path: Path) -> dict:
    training_root = tmp_path / "training"
    materialization_root = tmp_path / "materialization"
    cancers = [f"C{index:02d}" for index in range(28)]
    for index, cancer in enumerate(cancers):
        cancer_root = materialization_root / "cancers" / cancer
        cancer_root.mkdir(parents=True)
        (cancer_root / "SUCCESS.json").write_text("{}\n", encoding="utf-8")
        if cancer == "C00":
            donor_folds = pd.DataFrame(
                {
                    "cancer_id": [cancer] * auditor.N_FOLDS,
                    "donor_id": [f"d{fold}" for fold in range(auditor.N_FOLDS)],
                    "outer_fold_id": list(range(auditor.N_FOLDS)),
                }
            )
        else:
            donor_folds = pd.DataFrame(
                {
                    "cancer_id": [cancer],
                    "donor_id": [f"x{index}"],
                    "outer_fold_id": [index % auditor.N_FOLDS],
                }
            )
        donor_folds.to_parquet(cancer_root / "DONOR_FOLDS.parquet", index=False)

    predictions = []
    receipts = {}
    c00 = materialization_root / "cancers" / "C00"
    for fold in range(auditor.N_FOLDS):
        frame = _oof_rows([0.5], folds=[fold])
        frame["cancer_id"] = "C00"
        frame["query_donor_id"] = f"d{fold}"
        frame["outer_fold_id"] = fold
        prediction_dir = training_root / f"outer_fold={fold}"
        materialized_dir = c00 / f"outer_fold={fold}"
        prediction_dir.mkdir(parents=True)
        materialized_dir.mkdir(parents=True)
        frame.to_parquet(prediction_dir / "QUERY_PREDICTIONS.parquet", index=False)
        frame.drop(columns=["predicted_rho"]).to_parquet(
            materialized_dir / "QUERY_ROWS.parquet", index=False
        )
        train_donors = [f"d{other}" for other in range(auditor.N_FOLDS) if other != fold]
        pd.DataFrame(
            {
                "cancer_id": ["C00"] * len(train_donors),
                "query_donor_id": train_donors,
            }
        ).to_parquet(materialized_dir / "TRAIN_ROWS.parquet", index=False)
        receipts[fold] = {
            "query_rows": 1,
            "query_donors": 1,
            "fit_tuning_donor_overlap": 0,
            "query_oov_zero_id_fallback_rows": 0,
        }
        predictions.append(frame)
    pd.concat(predictions, ignore_index=True).to_parquet(
        training_root / "OOF_DONOR_PREDICTIONS.parquet", index=False
    )
    return {
        "training_root": training_root,
        "materialization_root": materialization_root,
        "receipts": receipts,
    }


def test_oof_audit_enforces_authoritative_donor_fold_isolation(tmp_path):
    case = _oof_isolation_case(tmp_path)
    predictions, report = auditor.audit_oof_isolation(
        case["training_root"], case["materialization_root"], case["receipts"]
    )
    assert len(predictions) == auditor.N_FOLDS
    assert report["cancer_count"] == 28
    assert report["oof_duplicate_keys"] == 0
    assert all(row["query_training_donor_overlap"] == 0 for row in report["folds"])


def test_oof_audit_rejects_query_donor_leaking_into_training(tmp_path):
    case = _oof_isolation_case(tmp_path)
    leaked = (
        case["materialization_root"]
        / "cancers"
        / "C00"
        / "outer_fold=0"
        / "TRAIN_ROWS.parquet"
    )
    pd.DataFrame(
        {"cancer_id": ["C00"], "query_donor_id": ["d0"]}
    ).to_parquet(leaked, index=False)
    with pytest.raises(auditor.IndependentAuditError, match="leaked"):
        auditor.audit_oof_isolation(
            case["training_root"], case["materialization_root"], case["receipts"]
        )


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _lineage_closure_case(tmp_path: Path) -> dict:
    training_root = tmp_path / "training"
    materialization_root = tmp_path / "materialization"
    core_root = tmp_path / "core"
    training_root.mkdir()
    materialization_root.mkdir()
    core_root.mkdir()

    cancers = [f"C{index:02d}" for index in range(28)]
    for cancer in cancers:
        _write_json(
            materialization_root / "cancers" / cancer / "SUCCESS.json",
            {"status": "PASS"},
        )
    materialization_validation = materialization_root / "MATERIALIZATION_VALIDATION.json"
    preprocessing_audit = materialization_root / "SINGLE_CELL_PREPROCESSING_AUDIT.json"
    materialization_document = {
        "status": "PASS",
        "closure_status": auditor.FULL_MATERIALIZATION_STATUS,
        "validation_scope": "FULL",
        "expected_cancer_count": 28,
        "validated_cancer_count": 28,
        "expected_cancers": cancers,
        "validated_cancers": list(reversed(cancers)),
    }
    preprocessing_document = {
        "status": "PASS",
        "active_cancer_count": 28,
        "active_cancers": cancers,
        "eligible_training_cells": 100,
        "eligible_cells_failing_preprocessing_pass_qc": 0,
        "eligible_cells_failing_batch_correction_applied": 0,
        "eligible_cells_failing_annotation_completed": 0,
    }
    _write_json(materialization_validation, materialization_document)
    _write_json(preprocessing_audit, preprocessing_document)

    core_manifest = core_root / "CORE_EMBEDDING_MANIFEST.json"
    rng = np.random.default_rng(31)
    core_folds = {}
    for fold in range(auditor.N_FOLDS):
        exports = {}
        for name in ("lncRNA", "pathway", "cancer"):
            values = rng.normal(size=(2, 96)).astype(np.float32)
            path = core_root / f"patient_fold={fold}" / f"{name}.parquet"
            _write_core(path, values)
            exports[name] = {
                "path": f"patient_fold={fold}/{name}.parquet",
                "sha256": auditor.sha256(path),
                "features": 96,
                "rows": len(values),
            }
        core_folds[str(fold)] = {
            "patient_fold": fold,
            "old_checkpoint_loaded": False,
            "trained_from_scratch": True,
            "partial_chunk_smoke": False,
            "exports": exports,
        }
    core_document = {
        "all_embeddings_from_newly_trained_v32_core": True,
        "historical_checkpoint_loaded": False,
        "historical_prediction_loaded": False,
        "folds": core_folds,
    }
    _write_json(core_manifest, core_document)

    trainer_path = tmp_path / "frozen_trainer.py"
    trainer_path.write_text("# frozen trainer\n", encoding="utf-8")
    settings = {
        "seed": 11,
        "base_seed": 11,
        "epochs": 3,
        "patience": 2,
        "batch_size": 16,
        "max_train_rows": 100,
        "learning_rate": 0.001,
    }
    fold_seeds = [settings["seed"] + fold * 101 for fold in range(auditor.N_FOLDS)]
    config = {
        "format": auditor.CONFIG_FORMAT,
        "target_host": "149",
        "allow_partial_materialization": False,
        "fresh_random_initialization": True,
        "source_checkpoint_sha256": None,
        "outer_folds": auditor.N_FOLDS,
        "fold_seeds": fold_seeds,
        "numeric_feature_columns": auditor.FEATURE_COLUMNS,
        "device": "cpu",
        "cuda_used": False,
        "old_checkpoint_loaded": False,
        "training": settings,
        "architecture": {
            "frozen_lnc_core_dimensions": 96,
            "frozen_pathway_core_dimensions": 96,
            "frozen_cancer_core_dimensions": 96,
            "private_lnc_embedding_dimensions": 32,
            "celltype_embedding_dimensions": 16,
            "core_availability_flag_dimensions": 1,
            "input_dimensions": 345,
        },
        "discovery_thresholds": {
            "min_donors": 3,
            "min_outer_folds": 2,
            "max_fdr": 0.05,
            "min_abs_meta_rho": 0.2,
            "min_direction_consistency": 0.75,
        },
    }
    _write_json(training_root / "RUN_CONFIG.json", config)
    full_input_contract = {
        "materialization_closure_status": auditor.FULL_MATERIALIZATION_STATUS,
        "materialization_validation_path": str(materialization_validation),
        "materialization_validation_sha256": auditor.sha256(
            materialization_validation
        ),
        "preprocessing_audit_path": str(preprocessing_audit),
        "preprocessing_audit_sha256": auditor.sha256(preprocessing_audit),
        "materialization_expected_cancers": cancers,
        "preprocessing_audit_eligible_cells": 100,
    }
    materialization_record = {
        **_record(materialization_validation),
        "status": "PASS",
        "closure_status": auditor.FULL_MATERIALIZATION_STATUS,
        "validation_scope": "FULL",
        "expected_cancer_count": 28,
        "validated_cancer_count": 28,
    }
    preprocessing_record = {
        **_record(preprocessing_audit),
        "status": "PASS",
        "active_cancer_count": 28,
        "eligible_training_cells": 100,
    }
    core_record = {
        **_record(core_manifest),
        "status": "PASS_FRESH_FROZEN_CORE_EMBEDDINGS",
        "all_embeddings_from_newly_trained_v32_core": True,
        "historical_checkpoint_loaded": False,
        "historical_prediction_loaded": False,
    }
    input_lineage = {
        "format": auditor.INPUT_LINEAGE_FORMAT,
        "status": "PASS",
        "scope": "PASS_FULL_28_CANCER_INPUT_LINEAGE",
        "target_host": "149",
        "materialization_validation_path": str(materialization_validation),
        "materialization_validation_sha256": auditor.sha256(
            materialization_validation
        ),
        "preprocessing_audit_path": str(preprocessing_audit),
        "preprocessing_audit_sha256": auditor.sha256(preprocessing_audit),
        "core_manifest_path": str(core_manifest),
        "core_manifest_sha256": auditor.sha256(core_manifest),
        "full_input_contract": full_input_contract,
        "old_checkpoint_loaded": False,
        "old_predictions_loaded": False,
        "old_rankings_loaded": False,
        "old_training_outputs_reused": False,
        "materialization_validation": materialization_record,
        "preprocessing_audit": preprocessing_record,
        "core_manifest": core_record,
    }
    _write_json(training_root / "INPUT_LINEAGE_AUDIT.json", input_lineage)

    fold_receipts = []
    manifest_records = []
    for fold in range(auditor.N_FOLDS):
        fold_root = training_root / f"outer_fold={fold}"
        fold_root.mkdir()
        checkpoint = fold_root / "best_model_state.pt"
        predictions = fold_root / "QUERY_PREDICTIONS.parquet"
        checkpoint.write_bytes(f"checkpoint-{fold}".encode())
        predictions.write_bytes(f"predictions-{fold}".encode())
        receipt = {
            "format": auditor.TRAINING_FORMAT,
            "target_host": "149",
            "state": "FOLD_TRAINED_AND_OOF_PREDICTED",
            "outer_fold_id": fold,
            "seed": fold_seeds[fold],
            "fresh_random_initialization": True,
            "old_checkpoint_loaded": False,
            "source_checkpoint_sha256": None,
            "device": "cpu",
            "cuda_used": False,
        }
        receipt_path = fold_root / "SUCCESS.json"
        _write_json(receipt_path, receipt)
        fold_receipts.append(receipt)
        manifest_records.append(
            {
                "outer_fold_id": fold,
                "seed": fold_seeds[fold],
                "checkpoint": _record(checkpoint),
                "predictions": _record(predictions),
                "fold_success": _record(receipt_path),
                "fresh_random_initialization": True,
                "old_checkpoint_loaded": False,
                "source_checkpoint_sha256": None,
                "device": "cpu",
                "cuda_used": False,
            }
        )
    manifest = {
        "format": auditor.CHECKPOINT_MANIFEST_FORMAT,
        "status": "SUCCESS",
        "target_host": "149",
        "outer_folds": auditor.N_FOLDS,
        "models_trained": auditor.N_FOLDS,
        "fold_seeds": fold_seeds,
        "fresh_random_initialization": True,
        "old_checkpoint_loaded": False,
        "source_checkpoint_sha256": None,
        "device": "cpu",
        "cuda_used": False,
        "records": manifest_records,
    }
    _write_json(training_root / "CHECKPOINT_MANIFEST.json", manifest)

    preflight = {
        "format": auditor.TRAINING_FORMAT,
        "target_host": "149",
        "cpu_only": True,
        "paid_compute": False,
        "allow_partial_materialization": False,
        "materialization_root": str(materialization_root),
        "core_root": str(core_root),
        "fresh_random_initialization": True,
        "old_checkpoint_loaded": False,
        "source_checkpoint_sha256": None,
        "device": "cpu",
        "cuda_used": False,
        "run_config": _record(training_root / "RUN_CONFIG.json"),
        "input_lineage_audit": _record(
            training_root / "INPUT_LINEAGE_AUDIT.json"
        ),
        "trainer_script": _record(trainer_path),
        "full_input_contract": full_input_contract,
    }
    _write_json(training_root / "LAUNCH_PREFLIGHT.json", preflight)

    for name in (
        "OOF_DONOR_PREDICTIONS.parquet",
        "CROSSFIT_ASSOCIATIONS.parquet",
        "DISCOVERED_NEW_RELATIONSHIPS.parquet",
    ):
        (training_root / name).write_bytes(f"placeholder:{name}".encode())
    contract = {
        "format": auditor.TRAINING_FORMAT,
        "target_host": "149",
        "status": "PASS_TECHNICAL_BASELINE",
        "outer_folds": auditor.N_FOLDS,
        "signed_target": True,
        "donor_blocked": True,
        "metacell_resolved": True,
        "core_detached_and_frozen": True,
        "oov_private_embedding_enabled": True,
        "fresh_random_initialization": True,
        "old_checkpoint_loaded": False,
        "source_checkpoint_sha256": None,
        "device": "cpu",
        "cuda_used": False,
        "run_config": _record(training_root / "RUN_CONFIG.json"),
        "input_lineage_audit": _record(
            training_root / "INPUT_LINEAGE_AUDIT.json"
        ),
        "checkpoint_manifest": _record(
            training_root / "CHECKPOINT_MANIFEST.json"
        ),
        "folds": fold_receipts,
    }
    _write_json(training_root / "FULL_MODEL_CONTRACT_VALIDATION.json", contract)

    lineage_output_names = {
        "OOF_DONOR_PREDICTIONS.parquet",
        "CROSSFIT_ASSOCIATIONS.parquet",
        "DISCOVERED_NEW_RELATIONSHIPS.parquet",
        "FULL_MODEL_CONTRACT_VALIDATION.json",
    }
    lineage = {
        "format": auditor.LINEAGE_FORMAT,
        "status": "SUCCESS",
        "scope": "PASS_FULL_28_CANCER_TRAINING_LINEAGE",
        "training_run_id": "synthetic-full-training-run",
        "target_host": "149",
        "cpu_only": True,
        "paid_compute": False,
        "cuda_used": False,
        "device": "cpu",
        "models_trained": auditor.N_FOLDS,
        "seeds": fold_seeds,
        "fold_seeds": fold_seeds,
        "old_checkpoint_loaded": False,
        "old_predictions_used_as_features": False,
        "old_rankings_used_as_outputs": False,
        "old_family_support_used": False,
        "core_parameters_frozen": True,
        "fresh_random_initialization": True,
        "source_checkpoint_sha256": None,
        "launch_preflight_sha256": auditor.sha256(
            training_root / "LAUNCH_PREFLIGHT.json"
        ),
        "run_config_sha256": auditor.sha256(training_root / "RUN_CONFIG.json"),
        "input_lineage_audit_sha256": auditor.sha256(
            training_root / "INPUT_LINEAGE_AUDIT.json"
        ),
        "checkpoint_manifest_sha256": auditor.sha256(
            training_root / "CHECKPOINT_MANIFEST.json"
        ),
        "trainer_path": str(trainer_path),
        "trainer_sha256": auditor.sha256(trainer_path),
        "trainer_script_sha256": auditor.sha256(trainer_path),
        "trainer_script": _record(trainer_path),
        "launch_preflight": _record(training_root / "LAUNCH_PREFLIGHT.json"),
        "run_config": _record(training_root / "RUN_CONFIG.json"),
        "input_lineage_audit": _record(
            training_root / "INPUT_LINEAGE_AUDIT.json"
        ),
        "checkpoint_manifest": _record(
            training_root / "CHECKPOINT_MANIFEST.json"
        ),
        "contract_validation": _record(
            training_root / "FULL_MODEL_CONTRACT_VALIDATION.json"
        ),
        "output_artifacts_before_lineage": {
            name: _record(training_root / name)
            for name in auditor._pre_lineage_artifact_names()
        },
        "outputs": {
            name: _record(training_root / name) for name in lineage_output_names
        },
    }
    _write_json(training_root / "LINEAGE.json", lineage)
    success = {
        "format": auditor.TRAINING_FORMAT,
        "target_host": "149",
        "state": "FIVE_FOLD_DONOR_BLOCKED_METACELL_TRAINING_COMPLETE",
        "status": "SUCCESS",
        "scope": "PASS_FULL_28_CANCER_TRAINING",
        "folds": auditor.N_FOLDS,
        "models_trained": auditor.N_FOLDS,
        "fold_seeds": fold_seeds,
        "device": "cpu",
        "cuda_used": False,
        "fresh_random_initialization": True,
        "old_checkpoint_loaded": False,
        "source_checkpoint_sha256": None,
        "lineage_sha256": auditor.sha256(training_root / "LINEAGE.json"),
        "files": {
            name: _record(training_root / name) for name in auditor.TOP_LEVEL_ARTIFACTS
        },
    }
    success["required_closure_files"] = {
        name: success["files"][name] for name in auditor.CLOSURE_ARTIFACTS
    }
    _write_json(training_root / "SUCCESS.json", success)
    return {
        "training_root": training_root,
        "materialization_root": materialization_root,
        "core_root": core_root,
        "trainer_path": trainer_path,
    }


def test_lineage_closure_accepts_complete_hash_bound_chain(tmp_path):
    case = _lineage_closure_case(tmp_path)
    documents, closure = auditor.validate_lineage_closure(
        case["training_root"], case["materialization_root"], case["core_root"]
    )
    assert documents["LINEAGE.json"]["status"] == "SUCCESS"
    assert set(closure["checkpoint_records"]) == set(range(auditor.N_FOLDS))


@pytest.mark.parametrize(
    "missing_name",
    ["INPUT_LINEAGE_AUDIT.json", "LINEAGE.json", "SUCCESS.json"],
)
def test_lineage_closure_fails_closed_when_required_document_missing(
    tmp_path, missing_name
):
    case = _lineage_closure_case(tmp_path)
    (case["training_root"] / missing_name).unlink()
    with pytest.raises(auditor.IndependentAuditError, match="missing or unsafe"):
        auditor.validate_lineage_closure(
            case["training_root"], case["materialization_root"], case["core_root"]
        )


def test_lineage_closure_fails_closed_on_bound_document_hash_drift(tmp_path):
    case = _lineage_closure_case(tmp_path)
    with (case["training_root"] / "RUN_CONFIG.json").open("a", encoding="utf-8") as handle:
        handle.write(" \n")
    with pytest.raises(
        auditor.IndependentAuditError,
        match="RUN_CONFIG hash drift|run_config_sha256",
    ):
        auditor.validate_lineage_closure(
            case["training_root"], case["materialization_root"], case["core_root"]
        )


def test_lineage_closure_fails_closed_on_trainer_hash_drift(tmp_path):
    case = _lineage_closure_case(tmp_path)
    case["trainer_path"].write_text("# changed trainer\n", encoding="utf-8")
    with pytest.raises(auditor.IndependentAuditError, match="trainer.*hash drift"):
        auditor.validate_lineage_closure(
            case["training_root"], case["materialization_root"], case["core_root"]
        )


@pytest.mark.parametrize(
    ("document_name", "partial_scope"),
    [
        ("INPUT_LINEAGE_AUDIT.json", "PASS_PARTIAL_PILOT_INPUT_LINEAGE"),
        ("LINEAGE.json", "PASS_PARTIAL_PILOT_LINEAGE"),
        ("SUCCESS.json", "PASS_PARTIAL_PILOT_TRAINING"),
    ],
)
def test_lineage_closure_rejects_partial_scope(
    tmp_path, document_name, partial_scope
):
    case = _lineage_closure_case(tmp_path)
    path = case["training_root"] / document_name
    document = json.loads(path.read_text(encoding="utf-8"))
    document["scope"] = partial_scope
    _write_json(path, document)
    with pytest.raises(auditor.IndependentAuditError, match="drift"):
        auditor.validate_lineage_closure(
            case["training_root"], case["materialization_root"], case["core_root"]
        )


def test_oof_audit_rejects_duplicate_donor_target(tmp_path):
    case = _oof_isolation_case(tmp_path)
    path = case["training_root"] / "outer_fold=0" / "QUERY_PREDICTIONS.parquet"
    frame = pd.read_parquet(path)
    pd.concat([frame, frame], ignore_index=True).to_parquet(path, index=False)
    with pytest.raises(auditor.IndependentAuditError, match="duplicated"):
        auditor.audit_oof_isolation(
            case["training_root"], case["materialization_root"], case["receipts"]
        )


def _materialization_hash_closure_case(tmp_path: Path) -> dict:
    root = tmp_path / "materialization"
    cancers = [f"C{index:02d}" for index in range(28)]
    validation_reports = []
    train_paths = []
    for index, cancer in enumerate(cancers):
        fold = index % auditor.N_FOLDS
        cancer_root = root / "cancers" / cancer
        fold_root = cancer_root / f"outer_fold={fold}"
        fold_root.mkdir(parents=True)
        donor_path = cancer_root / "DONOR_FOLDS.parquet"
        pd.DataFrame(
            {
                "cancer_id": [cancer],
                "donor_id": [f"donor-{index}"],
                "outer_fold_id": [fold],
            }
        ).to_parquet(donor_path, index=False)
        train_path = fold_root / "TRAIN_ROWS.parquet"
        query_path = fold_root / "QUERY_ROWS.parquet"
        pd.DataFrame({"row_id": [f"train-{index}"]}).to_parquet(
            train_path, index=False
        )
        pd.DataFrame({"row_id": [f"query-{index}"]}).to_parquet(
            query_path, index=False
        )
        fold_success_path = fold_root / "SUCCESS.json"
        _write_json(
            fold_success_path,
            {
                "target_host": "149",
                "state": "FOLD_LOCAL_METACELL_TARGETS_READY",
                "cancer_id": cancer,
                "outer_fold_id": fold,
                "files": {
                    "TRAIN_ROWS.parquet": _record(train_path),
                    "QUERY_ROWS.parquet": _record(query_path),
                },
            },
        )
        fold_sha = auditor.sha256(fold_success_path)
        cancer_success_path = cancer_root / "SUCCESS.json"
        _write_json(
            cancer_success_path,
            {
                "files": {"DONOR_FOLDS.parquet": _record(donor_path)},
                "fold_receipt_sha256": {str(fold): fold_sha},
            },
        )
        validation_reports.append(
            {
                "cancer_id": cancer,
                "success_sha256": auditor.sha256(cancer_success_path),
                "verified_cancer_files": {
                    "DONOR_FOLDS.parquet": _record(donor_path)
                },
                "fold_success_sha256": {str(fold): fold_sha},
            }
        )
        train_paths.append(train_path)
    _write_json(
        root / "MATERIALIZATION_VALIDATION.json",
        {
            "status": "PASS",
            "closure_status": auditor.FULL_MATERIALIZATION_STATUS,
            "validation_scope": "FULL",
            "expected_cancer_count": 28,
            "validated_cancer_count": 28,
            "expected_cancers": cancers,
            "validated_cancers": cancers,
            "cancers": validation_reports,
        },
    )
    return {"root": root, "train_paths": train_paths}


def test_materialization_training_hash_closure_passes_and_rejects_tampering(
    tmp_path,
):
    case = _materialization_hash_closure_case(tmp_path)
    report = auditor.validate_materialization_training_file_hashes(case["root"])
    assert report == {
        "cancer_receipts": 28,
        "fold_receipts": 28,
        "training_parquet_files": 56,
    }

    case["train_paths"][0].write_bytes(b"tampered")
    with pytest.raises(auditor.IndependentAuditError, match="hash drift"):
        auditor.validate_materialization_training_file_hashes(case["root"])
