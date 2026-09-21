from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from cc_hhgt.v32.single_cell_training import (
    EmbeddingLookup,
    FoldCoreEmbeddings,
    SingleCellTrainingConfig,
    SingleCellTrainingError,
    _source_audit,
    build_blocked_folds,
    candidate_core,
    candidate_core_availability,
    coverage_null_scaffold,
    embedding_usability,
    exact_candidate_join,
    fit_private_head,
    normalise_dataset_manifest,
    normalise_single_cell_associations,
    run_single_cell_training,
    split_block_ids,
)


def _formal_manifest() -> pd.DataFrame:
    return normalise_dataset_manifest(
        pd.DataFrame(
            {
                "dataset_id": ["SC_BRCA_A"],
                "cancer_id": ["BRCA"],
                "formal_eligible": [True],
                "source_tier": ["primary_raw"],
                "quality_status": ["PASS"],
            }
        )
    )


def test_donor_and_dataset_blocks_never_cross_folds() -> None:
    rows = []
    for donor in range(12):
        for repeat in range(3):
            rows.append(
                {
                    "dataset_id": "SC_BRCA_A",
                    "cancer_id": "BRCA",
                    "donor_id": f"D{donor}",
                    "repeat": repeat,
                }
            )
    blocked = build_blocked_folds(pd.DataFrame(rows), seed=17)
    assert blocked.groupby("donor_id").single_cell_fold_id.nunique().max() == 1
    assert set(blocked.single_cell_fold_id) == set(range(5))
    for fold in range(5):
        split = split_block_ids(blocked, fold)
        assert not (split["train"] & split["validation"])
        assert not (split["train"] & split["test"])
        assert not (split["validation"] & split["test"])

    dataset_only = pd.DataFrame(
        {
            "dataset_id": np.repeat([f"DS{i}" for i in range(7)], 2),
            "cancer_id": "BRCA",
        }
    )
    dataset_blocked = build_blocked_folds(dataset_only, seed=17)
    assert dataset_blocked.groupby("dataset_id").single_cell_fold_id.nunique().max() == 1

    repeated_donor = build_blocked_folds(
        pd.DataFrame(
            {
                "dataset_id": ["DS_A", "DS_B"],
                "cancer_id": ["BRCA", "BRCA"],
                "donor_id": ["SAME_DONOR", "SAME_DONOR"],
            }
        ),
        seed=17,
    )
    assert repeated_donor.single_cell_fold_id.nunique() == 1


def test_exact_pathway_join_never_broadcasts_family_support() -> None:
    candidates = pd.DataFrame(
        {
            "cancer_id": ["BRCA", "BRCA"],
            "lncrna_id": ["LNC:1", "LNC:1"],
            "pathway_id": ["PW:EXACT_A", "PW:EXACT_B"],
            # Static hierarchy metadata may coexist in the candidate universe;
            # it is ignored and never used to broadcast an association.
            "pathway_family_id": ["FAMILY_X", "FAMILY_X"],
        }
    )
    association = pd.DataFrame(
        {
            "cancer_id": ["BRCA"],
            "lncrna_id": ["LNC:1"],
            "pathway_id": ["PW:EXACT_A"],
            "rho": [0.6],
        }
    )
    joined = exact_candidate_join(association, candidates)
    assert joined.pathway_id.tolist() == ["PW:EXACT_A"]
    assert joined.exact_pathway_join.eq(True).all()

    with pytest.raises(SingleCellTrainingError, match="family"):
        exact_candidate_join(
            association.assign(pathway_family_id="HALLMARK_LIKE_FAMILY"), candidates
        )


def test_typed_lnc_prefix_canonicalises_to_core_node_id() -> None:
    core = FoldCoreEmbeddings(
        fold=0,
        lncrna=EmbeddingLookup(
            values=np.asarray([[1.0, 2.0]], dtype=np.float32),
            index={"ENSG00000123456": 0},
        ),
        pathway=EmbeddingLookup(
            values=np.asarray([[3.0, 4.0]], dtype=np.float32),
            index={"MSIGDB:HALLMARK:HALLMARK_APOPTOSIS": 0},
        ),
        checkpoint_sha256="a" * 64,
        parameter_sha256="b" * 64,
        artifact_hashes={},
    )
    frame = pd.DataFrame(
        {
            "lncrna_id": ["LNC:ENSG00000123456"],
            "pathway_id": ["MSIGDB:HALLMARK:HALLMARK_APOPTOSIS"],
        }
    )
    values, available = candidate_core(frame, core)
    assert available.tolist() == [True]
    np.testing.assert_array_equal(values, [[1.0, 2.0, 3.0, 4.0]])


def test_constant_formal_lncrna_core_is_explicitly_masked() -> None:
    core = FoldCoreEmbeddings(
        fold=0,
        lncrna=EmbeddingLookup(
            values=np.asarray([[7.0, 8.0], [7.0, 8.0]], dtype=np.float32),
            index={"ENSG1": 0, "ENSG2": 1},
        ),
        pathway=EmbeddingLookup(
            values=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            index={"P1": 0, "P2": 1},
        ),
        checkpoint_sha256="a" * 64,
        parameter_sha256="b" * 64,
        artifact_hashes={},
    )
    audit = embedding_usability(core.lncrna)
    assert audit == {
        "rows": 2,
        "features": 2,
        "unique_embedding_vectors": 1,
        "varying_feature_count": 0,
        "usable_for_node_discrimination": False,
        "status": "CONSTANT_EMBEDDING_MASKED",
    }
    values, available = candidate_core(
        pd.DataFrame({"lncrna_id": ["LNC:ENSG1"], "pathway_id": ["P2"]}),
        core,
    )
    assert available.tolist() == [True]
    np.testing.assert_array_equal(values, [[0.0, 0.0, 3.0, 4.0]])
    assert candidate_core_availability(
        pd.DataFrame(
            {
                "lncrna_id": ["LNC:ENSG1", "LNC:MISSING"],
                "pathway_id": ["P2", "P2"],
            }
        ),
        core,
    ).tolist() == [True, False]


def test_staging_is_not_formal_and_missing_cancers_are_null() -> None:
    with pytest.raises(SingleCellTrainingError, match="Staging/nominal"):
        normalise_dataset_manifest(
            pd.DataFrame(
                {
                    "dataset_id": ["STAGING_BRCA"],
                    "cancer_id": ["BRCA"],
                    "formal_eligible": [True],
                    "source_tier": ["staging"],
                    "quality_status": ["PASS"],
                }
            )
        )

    manifest = normalise_dataset_manifest(
        pd.DataFrame(
            {
                "dataset_id": ["STAGING_BRCA"],
                "cancer_id": ["BRCA"],
                "formal_eligible": [False],
                "source_tier": ["staging"],
                "quality_status": ["PASS"],
            }
        )
    )
    candidates = pd.DataFrame(
        {
            "cancer_id": ["BRCA", "ACC"],
            "lncrna_id": ["LNC:1", "LNC:2"],
            "pathway_id": ["PW:A", "PW:B"],
        }
    )
    empty_association = pd.DataFrame(columns=["cancer_id", "formal_row"])
    nulls = coverage_null_scaffold(candidates, manifest, empty_association)
    assert nulls.single_cell_replication_probability.isna().all()
    assert not nulls.single_cell_available.any()
    reasons = dict(zip(nulls.cancer_id, nulls.single_cell_unavailable_reason, strict=True))
    assert reasons["BRCA"] == "STAGING_OR_NOMINAL_SINGLE_CELL_SOURCE_NOT_FORMAL"
    assert reasons["ACC"] == "NO_DECLARED_SINGLE_CELL_DATASET"

    limited = normalise_dataset_manifest(
        pd.DataFrame(
            {
                "dataset_id": ["SC_CESC_LIMITED"],
                "cancer_id": ["CESC"],
                "formal_eligible": [False],
                "source_tier": ["primary_raw_count"],
                "quality_status": ["LIMITED"],
                "feature_universe_status": ["LIMITED"],
                "quality_flags": ["KNOWN_SOURCE_FEATURE_UNIVERSE_LIMITATION"],
                "donor_metadata_available": [True],
            }
        )
    )
    limited_null = coverage_null_scaffold(
        pd.DataFrame(
            {"cancer_id": ["CESC"], "lncrna_id": ["LNC:3"], "pathway_id": ["PW:C"]}
        ),
        limited,
        empty_association,
    )
    assert limited_null.single_cell_unavailable_reason.iloc[0] == (
        "LIMITED_LNCRNA_FEATURE_UNIVERSE"
    )

    with pytest.raises(SingleCellTrainingError, match="Limited feature universes"):
        normalise_dataset_manifest(
            pd.DataFrame(
                {
                    "dataset_id": ["SC_UCS_LIMITED"],
                    "cancer_id": ["UCS"],
                    "formal_eligible": [True],
                    "source_tier": ["primary_raw_count"],
                    "quality_status": ["PASS"],
                    "feature_universe_status": ["LIMITED"],
                    "donor_metadata_available": [True],
                }
            )
        )


def test_association_source_tier_is_checked_per_row() -> None:
    manifest = _formal_manifest()
    raw = pd.DataFrame(
        {
            "dataset_id": ["SC_BRCA_A", "SC_BRCA_A"],
            "cancer_id": ["BRCA", "BRCA"],
            "patient_id": ["P1", "P2"],
            "cell_population": ["Malignant", "Malignant"],
            "analysis_context": ["overall", "overall"],
            "lncrna_id": ["LNC:1", "LNC:1"],
            "pathway_id": ["PW:A", "PW:A"],
            "rho": [0.7, 0.5],
            "fdr": [0.02, 0.04],
            "source_tier": ["primary_raw", "staging"],
            "n_cells": [100, 80],
        }
    )
    normalised = normalise_single_cell_associations(raw, manifest)
    assert normalised.formal_row.tolist() == [True, False]
    assert normalised.association_target.between(0, 1).all()
    assert "association_effect" in normalised


def test_private_head_is_fresh_and_does_not_mutate_core_array() -> None:
    pytest.importorskip("torch")
    rng = np.random.default_rng(11)
    core = rng.normal(size=(20, 8)).astype(np.float32)
    domain = rng.normal(size=(20, 9)).astype(np.float32)
    labels = np.linspace(0.05, 0.95, 20, dtype=np.float32)
    before = hashlib.sha256(core.tobytes()).hexdigest()
    config = SingleCellTrainingConfig(
        seed=31,
        epochs=2,
        batch_size=8,
        hidden_features=8,
        patience=2,
        max_train_rows=100,
        max_validation_rows=100,
    )
    _, metadata, _, _, _ = fit_private_head(
        core[:15], domain[:15], labels[:15],
        core[15:], domain[15:], labels[15:],
        fold=0, config=config,
    )
    after = hashlib.sha256(core.tobytes()).hexdigest()
    assert before == after
    assert metadata["source_checkpoint_sha256"] is None
    assert metadata["initialization_policy"] == "FROZEN_V32_CORE_PRIVATE_HEAD_FROM_SCRATCH"
    assert metadata["core_embedding_detached"] is True
    assert metadata["core_parameters_frozen"] is True
    assert len(metadata["initial_parameter_sha256"]) == 64


def test_null_only_run_materialises_all_companions_and_lineage(tmp_path) -> None:
    candidates_path = tmp_path / "candidates.parquet"
    manifest_path = tmp_path / "dataset_manifest.parquet"
    association_path = tmp_path / "sc_association.parquet"
    lnc_path = tmp_path / "lnc_celltype.parquet"
    pd.DataFrame(
        {
            "cancer_id": ["BRCA", "ACC"],
            "lncrna_id": ["LNC:1", "LNC:2"],
            "pathway_id": ["PW:A", "PW:B"],
        }
    ).to_parquet(candidates_path, index=False)
    pd.DataFrame(
        {
            "dataset_id": ["STAGING_BRCA"],
            "cancer_id": ["BRCA"],
            "formal_eligible": [False],
            "source_tier": ["staging"],
            "quality_status": ["PASS"],
        }
    ).to_parquet(manifest_path, index=False)
    pd.DataFrame(
        {
            "dataset_id": ["STAGING_BRCA"],
            "cancer_id": ["BRCA"],
            "cell_population": ["Malignant"],
            "analysis_context": ["overall"],
            "lncrna_id": ["LNC:1"],
            "pathway_id": ["PW:A"],
            "rho": [0.7],
            "fdr": [0.02],
            "source_tier": ["staging"],
            "n_patients": [8],
        }
    ).to_parquet(association_path, index=False)
    pd.DataFrame(
        {
            "dataset_id": ["STAGING_BRCA"],
            "cancer_id": ["BRCA"],
            "lncrna_id": ["LNC:1"],
            "cell_type_major": ["Malignant"],
            "detection_rate": [0.4],
            "mean_log_expression": [0.2],
            "specificity_tau": [0.8],
            "n_cells": [100],
        }
    ).to_parquet(lnc_path, index=False)

    folds = {}
    for fold in range(5):
        fold_root = tmp_path / f"core_fold_{fold}"
        fold_root.mkdir()
        lnc_core = fold_root / "lncRNA.parquet"
        pathway_core = fold_root / "pathway.parquet"
        pd.DataFrame(
            {"node_id": ["LNC:1", "LNC:2"], "core_feature_0000": [0.1, 0.2]}
        ).to_parquet(lnc_core, index=False)
        pd.DataFrame(
            {"node_id": ["PW:A", "PW:B"], "core_feature_0000": [0.3, 0.4]}
        ).to_parquet(pathway_core, index=False)
        folds[str(fold)] = {
            "patient_fold": fold,
            "checkpoint_sha256": hashlib.sha256(f"checkpoint-{fold}".encode()).hexdigest(),
            "core_parameter_sha256": hashlib.sha256(f"core-{fold}".encode()).hexdigest(),
            "old_checkpoint_loaded": False,
            "trained_from_scratch": True,
            "exports": {
                "lncRNA": {
                    "path": str(lnc_core),
                    "sha256": hashlib.sha256(lnc_core.read_bytes()).hexdigest(),
                },
                "pathway": {
                    "path": str(pathway_core),
                    "sha256": hashlib.sha256(pathway_core.read_bytes()).hexdigest(),
                },
            },
        }
    core_manifest_path = tmp_path / "core_manifest.json"
    core_manifest_path.write_text(
        json.dumps(
            {
                "export_format": "CC_HHGT_V3_2_FROZEN_CORE_EMBEDDINGS_V1",
                "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
                "all_embeddings_from_newly_trained_v32_core": True,
                "historical_checkpoint_loaded": False,
                "historical_prediction_loaded": False,
                "folds": folds,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    success = run_single_cell_training(
        candidates_path=candidates_path,
        dataset_manifest_path=manifest_path,
        association_path=association_path,
        lnc_celltype_path=lnc_path,
        core_embedding_manifest_path=core_manifest_path,
        output_root=output,
        training_run_id="unit-null-run",
        association_generation="V3.2_UNIT_TEST_LABEL_FROM_FRESH_SINGLE_CELL_INPUT",
        config=SingleCellTrainingConfig(epochs=1),
    )
    assert success["trained_folds"] == 0
    assert success["status"] == "AUDITED_UNAVAILABLE"
    assert success["available_rows"] == 0
    assert success["staging_never_promoted"] is True
    for name in (
        "single_cell_typed_predictions.parquet",
        "lnc_celltype_summary.parquet",
        "lnc_exact_pathway.parquet",
        "activity.parquet",
        "FIGURE_MANIFEST.json",
        "CHECKPOINT_MANIFEST.json",
        "LINEAGE.json",
        "SUCCESS.json",
    ):
        assert (output / name).is_file()
    typed = pd.read_parquet(output / "single_cell_typed_predictions.parquet")
    assert len(typed) == 2
    assert typed.single_cell_replication_probability.isna().all()
    checkpoint_manifest = json.loads((output / "CHECKPOINT_MANIFEST.json").read_text())
    assert len(checkpoint_manifest["records"]) == 5
    assert checkpoint_manifest["all_private_heads_random_initialization"] is True
    assert checkpoint_manifest["trained_private_head_count"] == 0
    assert checkpoint_manifest["all_five_fold_private_heads_trained"] is False
    assert not any(row["fresh_random_initialization"] for row in checkpoint_manifest["records"])
    lineage = json.loads((output / "LINEAGE.json").read_text())
    assert lineage["training_status"] == "AUDITED_UNAVAILABLE"
    assert lineage["private_head_trained_from_scratch"] is False
    assert lineage["trained_folds"] == 0
    assert lineage["checkpoint_files"] == 0
    assert lineage["all_probabilities_null"] is True
    assert lineage["all_unavailable_rows_have_reason"] is True
    assert lineage["release_ready"] is False
    association_input = next(
        row
        for row in lineage["input_artifacts"]
        if row["path"] == str(association_path.resolve())
    )
    assert association_input["artifact_kind"] == "training_label"
    assert association_input["source_role"] == "training_label"
    assert association_input["outcome_derived"] is True
    assert association_input["use_role"] == "training_target"


def test_historical_single_cell_association_target_fails_closed(tmp_path) -> None:
    candidates = tmp_path / "candidates.parquet"
    manifest = tmp_path / "manifest.parquet"
    association = tmp_path / "association.parquet"
    lnc_celltype = tmp_path / "lnc_celltype.parquet"
    pd.DataFrame(
        {"cancer_id": ["BRCA"], "lncrna_id": ["LNC:1"], "pathway_id": ["PW:A"]}
    ).to_parquet(candidates, index=False)
    pd.DataFrame(
        {
            "dataset_id": ["SC_BRCA_A"],
            "cancer_id": ["BRCA"],
            "formal_eligible": [True],
            "source_tier": ["primary_raw"],
            "quality_status": ["PASS"],
        }
    ).to_parquet(manifest, index=False)
    pd.DataFrame(
        {
            "dataset_id": ["SC_BRCA_A"],
            "cancer_id": ["BRCA"],
            "lncrna_id": ["LNC:1"],
            "pathway_id": ["PW:A"],
            "rho": [0.7],
            "fdr": [0.01],
        }
    ).to_parquet(association, index=False)
    pd.DataFrame(
        {
            "dataset_id": ["SC_BRCA_A"],
            "cancer_id": ["BRCA"],
            "lncrna_id": ["LNC:1"],
            "cell_type_major": ["Malignant"],
            "detection_rate": [0.5],
            "mean_log_expression": [0.4],
            "specificity_tau": [0.8],
            "n_cells": [100],
        }
    ).to_parquet(lnc_celltype, index=False)

    with pytest.raises(
        SingleCellTrainingError,
        match="HISTORICAL_TRAINING_LABEL_FORBIDDEN",
    ):
        _source_audit(
            candidates=candidates,
            dataset_manifest=manifest,
            association=association,
            association_generation="V2.5_CELLTYPE_STAGING",
            lnc_celltype=lnc_celltype,
            lnc_celltype_generation="V3.2_FRESH_NONPREDICTIVE_INPUT",
            activities=(),
        )
