from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from cc_hhgt.v32.patient_folds import assign_outer_split
from cc_hhgt.v32.state_training import (
    CORE_CHECKPOINT_FORMAT,
    CORE_EXPORT_FORMAT,
    HISTORICAL_STATE_IDS,
    MAINLINE_REQUIRED_STATE_IDS,
    StateTrainingConfig,
    StateTrainingError,
    compute_fold_local_associations,
    file_sha256,
    load_v32_core_embeddings,
    path_sha256,
    read_table,
    reject_forbidden_input_path,
    run_state_training_from_paths,
    train_state_models,
    validate_core_embedding_frame,
    validate_nullable_predictions,
    validate_state_measurements,
)


def _fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[int, pd.DataFrame]]:
    samples = [f"S{i:02d}" for i in range(15)]
    folds = pd.DataFrame(
        {
            "cancer_id": "TEST",
            "sample_id": samples,
            "patient_id": samples,
            "patient_fold_id": [index % 5 for index in range(len(samples))],
            "fold_seed": 20260726,
        }
    )
    base = np.linspace(-2.0, 2.0, len(samples))
    expression_rows: list[tuple[str, str, str, float]] = []
    for index, sample in enumerate(samples):
        expression_rows.extend(
            [
                ("TEST", sample, "L_POS", float(base[index] + 0.03 * np.sin(index))),
                ("TEST", sample, "L_MIXED", float(np.sin(index * 1.7) + 0.1 * base[index])),
                ("TEST", sample, "L_CONST", 1.0),
            ]
        )
    expression = pd.DataFrame(
        expression_rows, columns=["cancer_id", "sample_id", "lncrna_id", "logcpm"]
    )
    state_rows: list[tuple[str, str, str, str, float, str]] = []
    # Leave EXTEND wholly unmeasured to prove that all seven states remain in
    # the candidate universe with null+reason rather than a 0/0.5 sentinel.
    measured_states = HISTORICAL_STATE_IDS[:-1]
    for state_index, state_id in enumerate(measured_states):
        for sample_index, sample in enumerate(samples):
            value = base[sample_index] * (1.0 + state_index * 0.1)
            value += 0.05 * np.cos(sample_index + state_index)
            state_rows.append((sample, "TEST", sample, state_id, float(value), "continuous"))
    states = pd.DataFrame(
        state_rows,
        columns=["sample_id", "cancer_id", "patient_id", "state_id", "state_value", "state_type"],
    )
    cores: dict[int, pd.DataFrame] = {}
    for fold in range(5):
        cores[fold] = pd.DataFrame(
            {
                "node_id": ["L_CONST", "L_MIXED", "L_POS"],
                "core_feature_000": [0.1 + fold, 0.2 + fold, 0.3 + fold],
                "core_feature_001": [1.1, 1.2, 1.3],
                "core_feature_002": [-0.1, 0.0, 0.1],
                "core_feature_003": [0.5, -0.5, 0.25],
            }
        )
    return states, folds, expression, cores


def _fast_config() -> StateTrainingConfig:
    return StateTrainingConfig(
        seed=17,
        min_pairs=3,
        effect_threshold=0.60,
        max_epochs=3,
        patience=2,
        learning_rate=5.0e-3,
        batch_size=128,
        hidden_features=8,
        dropout=0.0,
        min_available_folds=3,
        device="cpu",
    )


def test_state_contract_is_exactly_the_seven_historical_programs() -> None:
    assert HISTORICAL_STATE_IDS == (
        "stemness_rna::RNAss",
        "stemness_dna::DNAss",
        "stemness_dna::DMPss",
        "stemness_dna::ENHss",
        "stemness_dna::EREG-METHss",
        "stemness_rna::EREG.EXPss",
        "EXTEND::published_score",
    )
    assert MAINLINE_REQUIRED_STATE_IDS == {
        "stemness_rna::RNAss",
        "stemness_dna::DNAss",
        "stemness_rna::EREG.EXPss",
        "EXTEND::published_score",
    }


def test_non_historical_metadata_with_missing_cancer_is_excluded_before_state_id_validation() -> None:
    states, _, _, _ = _fixture()
    metadata = pd.DataFrame(
        [
            {
                "sample_id": "TCGA-00-0000-01",
                "cancer_id": pd.NA,
                "patient_id": "TCGA-00-0000",
                "state_id": "sample_type_code",
                "state_value": 1.0,
                "state_type": "continuous",
            }
        ]
    )
    validated = validate_state_measurements(pd.concat([states, metadata], ignore_index=True))
    assert "sample_type_code" not in set(validated.state_id)
    assert len(validated) == len(states)

    broken = states.copy()
    broken.loc[0, "cancer_id"] = pd.NA
    with pytest.raises(StateTrainingError, match="state.cancer_id"):
        validate_state_measurements(broken)


def test_fold_local_statistics_do_not_read_outer_test_measurements() -> None:
    states, folds, expression, _ = _fixture()
    states = validate_state_measurements(states)
    split = assign_outer_split(folds, 0)
    before = compute_fold_local_associations(
        states,
        expression,
        split,
        split="train",
        min_pairs=3,
        effect_threshold=0.6,
    )
    test_samples = set(split.loc[split.split.eq("test"), "sample_id"])
    changed_states = states.copy()
    changed_states.loc[changed_states.sample_id.isin(test_samples), "state_value"] *= -1000
    changed_expression = expression.copy()
    changed_expression.loc[changed_expression.sample_id.isin(test_samples), "logcpm"] += 5000
    after = compute_fold_local_associations(
        changed_states,
        changed_expression,
        split,
        split="train",
        min_pairs=3,
        effect_threshold=0.6,
    )
    pd.testing.assert_frame_equal(before, after)


def test_five_fresh_private_heads_emit_nullable_seven_state_release() -> None:
    states, folds, expression, cores = _fixture()
    result = train_state_models(states, folds, cores, expression, config=_fast_config())

    assert set(result.fold_heads) == set(range(5))
    assert set(result.release_predictions.state_id) == set(HISTORICAL_STATE_IDS)
    assert result.missing_state_ids == ("EXTEND::published_score",)
    assert result.oof_predictions.patient_fold_id.nunique() == 5
    for fold, lineage in result.fold_lineage.items():
        assert lineage["patient_fold"] == fold
        assert lineage["initialization"]["source_checkpoint_sha256"] is None
        assert lineage["private_head_trained_from_scratch"] is True
        assert lineage["core_embeddings_detached"] is True
        assert lineage["core_parameters_frozen"] is True
        assert lineage["test_measurements_used_for_training"] is False
        assert lineage["initialization"]["initial_parameter_sha256"] != lineage[
            "final_parameter_sha256"
        ]

    extend = result.release_predictions.loc[
        result.release_predictions.state_id.eq("EXTEND::published_score")
    ]
    assert not extend.empty
    assert (~extend.availability).all()
    assert extend.state_membership_probability.isna().all()
    assert extend.state_effect.isna().all()
    assert extend.availability_reason.str.contains("state_measurement_missing").all()
    available = result.release_predictions.loc[result.release_predictions.availability]
    assert not available.empty
    assert available.state_membership_probability.between(0, 1).all()
    validate_nullable_predictions(result.release_predictions)


def test_tcga_state_sample_and_expression_aliquot_align_by_patient() -> None:
    states, folds, expression, cores = _fixture()
    mapping = {sample: f"TCGA-AA-{index:04d}" for index, sample in enumerate(folds.sample_id)}
    folds["sample_id"] = [f"{mapping[value]}-01A-11R-TEST-00" for value in folds.sample_id]
    expression["sample_id"] = [
        f"{mapping[value]}-01A-11R-TEST-00" for value in expression.sample_id
    ]
    states["sample_id"] = [f"{mapping[value]}-01" for value in states.sample_id]
    states["patient_id"] = [mapping[value] for value in states.patient_id]
    config = replace(_fast_config(), max_epochs=1, patience=1)

    result = train_state_models(states, folds, cores, expression, config=config)
    assert not result.release_predictions.loc[
        result.release_predictions.state_id.eq("stemness_rna::RNAss")
    ].empty
    assert result.fold_lineage[0]["alignment_counts"]["analysis_patients"] == 15


@pytest.mark.parametrize(
    "name",
    [
        "strict_state_oof_prediction.parquet",
        "lncrna_state_final.parquet",
        "old_prediction.tsv",
        "legacy_checkpoint.pt",
        "model.ckpt",
    ],
)
def test_historical_result_or_checkpoint_path_is_rejected(tmp_path: Path, name: str) -> None:
    with pytest.raises(StateTrainingError, match="forbidden|checkpoint"):
        reject_forbidden_input_path(tmp_path / name, "test input")


def test_old_prediction_columns_are_rejected() -> None:
    states, _, _, cores = _fixture()
    states["strict_state_oof_prediction"] = 0.5
    with pytest.raises(StateTrainingError, match="forbidden old-result columns"):
        validate_state_measurements(states)

    core = cores[0].copy()
    core["old_prediction_probability"] = 0.5
    with pytest.raises(StateTrainingError, match="forbidden old-result columns"):
        validate_core_embedding_frame(core, 0)


@pytest.mark.parametrize("sentinel", [0.0, 0.5])
def test_unavailable_output_may_not_use_numeric_sentinel(sentinel: float) -> None:
    frame = pd.DataFrame(
        {
            "state_membership_probability": [sentinel],
            "state_effect": [sentinel],
            "availability": [False],
            "availability_reason": ["not measured"],
        }
    )
    with pytest.raises(StateTrainingError, match="must be null"):
        validate_nullable_predictions(frame)


def _write_core_manifest(
    tmp_path: Path, cores: dict[int, pd.DataFrame], *, old_checkpoint: bool = False
) -> Path:
    folds: dict[str, object] = {}
    for fold, frame in cores.items():
        path = tmp_path / f"core_fold_{fold}.csv"
        frame.to_csv(path, index=False)
        folds[str(fold)] = {
            "patient_fold": fold,
            "checkpoint_format": CORE_CHECKPOINT_FORMAT,
            "old_checkpoint_loaded": old_checkpoint if fold == 0 else False,
            "trained_from_scratch": True,
            "core_parameter_sha256": "a" * 64,
            "checkpoint_sha256": "b" * 64,
            "exports": {
                "lncRNA": {
                    "path": path.relative_to(tmp_path).as_posix(),
                    "sha256": file_sha256(path),
                    "rows": len(frame),
                    "features": 4,
                }
            },
        }
    manifest = {
        "export_format": CORE_EXPORT_FORMAT,
        "training_generation": "V3.2",
        "all_embeddings_from_newly_trained_v32_core": True,
        "historical_checkpoint_loaded": False,
        "historical_prediction_loaded": False,
        "folds": folds,
    }
    path = tmp_path / "CORE_EMBEDDING_MANIFEST.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_core_loader_reads_only_fresh_v32_embedding_tables(tmp_path: Path) -> None:
    _, _, _, cores = _fixture()
    manifest = _write_core_manifest(tmp_path, cores)
    loaded, lineage, _ = load_v32_core_embeddings(manifest, repo_root=tmp_path)
    assert set(loaded) == set(range(5))
    assert lineage[0]["core_parameter_sha256"] == "a" * 64

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["folds"]["0"]["old_checkpoint_loaded"] = True
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(StateTrainingError, match="old checkpoint"):
        load_v32_core_embeddings(manifest, repo_root=tmp_path)


def test_partitioned_expression_dataset_is_a_supported_raw_input(tmp_path: Path) -> None:
    dataset = tmp_path / "formal_lncrna_expression"
    first = dataset / "cancer_id=A" / "part-0.parquet"
    second = dataset / "cancer_id=B" / "part-0.parquet"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    pd.DataFrame({"sample_id": ["S1"], "lncrna_id": ["L1"], "logcpm": [1.0]}).to_parquet(
        first, index=False
    )
    pd.DataFrame({"sample_id": ["S2"], "lncrna_id": ["L2"], "logcpm": [2.0]}).to_parquet(
        second, index=False
    )

    loaded = read_table(dataset, "lncRNA expression")
    assert len(loaded) == 2
    assert len(path_sha256(dataset)) == 64


def test_path_runner_writes_new_checkpoints_predictions_and_lineage(tmp_path: Path) -> None:
    states, folds, expression, cores = _fixture()
    state_path = tmp_path / "tumor_state_long.csv"
    fold_path = tmp_path / "PATIENT_FOLD_MANIFEST.tsv"
    expression_path = tmp_path / "lncrna_expression.csv"
    states.to_csv(state_path, index=False)
    folds.to_csv(fold_path, sep="\t", index=False)
    expression.to_csv(expression_path, index=False)
    core_manifest = _write_core_manifest(tmp_path, cores)
    config = replace(_fast_config(), max_epochs=1, patience=1)

    result = run_state_training_from_paths(
        state_measurements_path=state_path,
        fold_manifest_path=fold_path,
        core_embedding_manifest_path=core_manifest,
        lncrna_expression_path=expression_path,
        output_root=tmp_path / "fresh_state_output",
        repo_root=tmp_path,
        run_id="v32-unit-fresh-state",
        config=config,
    )
    lineage = json.loads(Path(result["lineage"]).read_text(encoding="utf-8"))
    assert lineage["training_status"] == "SUCCESS"
    assert lineage["private_head_trained_from_scratch"] is True
    assert lineage["old_checkpoint_loaded"] is False
    assert lineage["old_predictions_used_as_features"] is False
    assert lineage["unavailable_encoding"] == "null_with_reason"
    assert set(lineage["new_private_head_checkpoints"]) == set(map(str, range(5)))
    assert Path(result["oof_predictions"]).is_file()
    assert Path(result["lncrna_state_release"]).is_file()


def test_state_trainer_has_no_checkpoint_loading_path() -> None:
    root = Path(__file__).resolve().parents[1]
    module = (root / "cc_hhgt" / "v32" / "state_training.py").read_text(encoding="utf-8")
    runner = (root / "scripts" / "run_v32_state_training.py").read_text(encoding="utf-8")
    assert "torch.load" not in module
    assert "torch.load" not in runner
    assert "build_private_auxiliary_head" in module
