from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cc_hhgt.v32.atac_training import (
    FORMAL_CANCERS,
    MATERIALIZATION_FORMAT,
    RAW_COVERED_CANCERS,
    RAW_GAP_CANCERS,
    AtacTrainingConfig,
    AtacTrainingError,
    file_sha256,
    normalise_patient_folds,
    run_atac_training,
)
from cc_hhgt.v32.cancer_modality_router import validate_atac_oof_lineage


def _fixture(tmp_path: Path) -> dict[str, Path]:
    rng = np.random.default_rng(20260829)
    pathways = [f"PW:{index:02d}" for index in range(20)]
    lncrnas = [f"LNC:ENSG_LNC_{index:02d}" for index in range(5)]
    pathway_genes = [f"ENSG_GENE_{index:02d}" for index in range(12)]
    membership_rows = []
    for index, pathway in enumerate(pathways):
        for offset in range(3):
            membership_rows.append(
                {"pathway_id": pathway, "gene_id": pathway_genes[(index + offset) % 12]}
            )
    membership = pd.DataFrame(membership_rows).drop_duplicates()
    membership_path = tmp_path / "exact_pathway_gene_membership_ensembl.parquet"
    membership.to_parquet(membership_path, index=False)

    candidates = pd.DataFrame(
        [
            {"cancer_id": cancer, "lncrna_id": lnc, "pathway_id": pathway}
            for cancer in FORMAL_CANCERS
            for lnc in lncrnas
            for pathway in pathways
        ]
    )
    candidates_path = tmp_path / "FORMAL_CANDIDATE_UNIVERSE.parquet"
    candidates.to_parquet(candidates_path, index=False)

    fold_rows = []
    for cancer in FORMAL_CANCERS:
        for fold in range(5):
            fold_rows.append(
                {
                    "cancer_id": cancer,
                    "sample_id": f"TCGA-{cancer[:2]:0<2}-{fold:04d}-01A",
                    "patient_id": f"TCGA-{cancer[:2]:0<2}-{fold:04d}",
                    "patient_fold_id": fold,
                }
            )
    folds = pd.DataFrame(fold_rows)
    folds_path = tmp_path / "PATIENT_FOLD_MANIFEST.tsv"
    folds.to_csv(folds_path, sep="\t", index=False)

    records = []
    matrix_root = tmp_path / "gene_accessibility"
    for cancer_index, cancer in enumerate(RAW_COVERED_CANCERS):
        patient_ids = [f"TCGA-{cancer[:2]:0<2}-{fold:04d}"[:12] for fold in range(5)]
        # Cancer-specific latent accessibility plus independent gene noise gives
        # both concordant and discordant lncRNA/pathway pairs.
        latent = rng.normal(size=5) + cancer_index * 0.01
        gene_ids = [value.removeprefix("LNC:") for value in lncrnas] + pathway_genes
        values = []
        for gene_index, _ in enumerate(gene_ids):
            sign = 1.0 if gene_index % 3 else -1.0
            values.append(sign * latent + rng.normal(scale=0.45, size=5))
        frame = pd.DataFrame(np.asarray(values), columns=patient_ids)
        frame.insert(0, "gene_id", gene_ids)
        path = matrix_root / f"cancer_id={cancer}" / "part-0.parquet"
        path.parent.mkdir(parents=True)
        frame.to_parquet(path, index=False)
        records.append(
            {
                "cancer_id": cancer,
                "path": str(path),
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
                "genes": len(frame),
                "patients": 5,
            }
        )
    materialization_path = tmp_path / "GENE_ACCESSIBILITY_SUCCESS.json"
    materialization_path.write_text(
        json.dumps(
            {
                "format": MATERIALIZATION_FORMAT,
                "status": "SUCCESS",
                "covered_cancers": list(RAW_COVERED_CANCERS),
                "raw_gap_cancers": list(RAW_GAP_CANCERS),
                "records": records,
                "manifest_sha256": "a" * 64,
                "input_artifacts": [],
                "technical_replicate_policy": "MEAN_WITHIN_EXACT_ALIQUOT_ID",
                "multiple_aliquot_policy": "EXPLICIT_EQUAL_ALIQUOT_WEIGHT_MEAN",
                "old_predictions_used": False,
                "old_checkpoints_used": False,
            }
        ),
        encoding="utf-8",
    )
    return {
        "candidates": candidates_path,
        "folds": folds_path,
        "membership": membership_path,
        "materialization": materialization_path,
    }


def test_patient_fold_normalisation_fails_closed_on_conflict() -> None:
    frame = pd.DataFrame(
        {
            "cancer_id": ["ACC"] * 6,
            "sample_id": ["TCGA-OR-0001"] * 2 + [f"TCGA-OR-000{x}" for x in range(2, 6)],
            "patient_id": ["TCGA-OR-0001"] * 2 + [f"TCGA-OR-000{x}" for x in range(2, 6)],
            "patient_fold_id": [0, 1, 1, 2, 3, 4],
        }
    )
    with pytest.raises(AtacTrainingError, match="multiple canonical folds"):
        normalise_patient_folds(frame)


def test_fresh_atac_training_preserves_scope_and_typed_gaps(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    output = tmp_path / "fresh_atac_oof"
    result = run_atac_training(
        candidates_path=paths["candidates"],
        patient_folds_path=paths["folds"],
        pathway_membership_path=paths["membership"],
        materialization_success_path=paths["materialization"],
        output_root=output,
        training_run_id="synthetic-fresh-atac",
        config=AtacTrainingConfig(
            expected_rows_per_cancer=None,
            max_training_rows=2_000,
            prediction_pair_batch_size=25,
            epochs=12,
        ),
    )
    assert result["status"] == "SUCCESS"
    typed = pd.read_parquet(output / "atac_typed_predictions.parquet")
    assert len(typed) == len(FORMAL_CANCERS) * 100
    gap = typed.cancer_id.isin(RAW_GAP_CANCERS)
    assert not typed.loc[gap, "atac_available"].any()
    assert typed.loc[gap, "atac_context_probability"].isna().all()
    assert set(typed.loc[gap, "atac_unavailable_reason"].dropna()) == {
        "ATAC_RAW_NOT_AVAILABLE_FOR_CANCER"
    }
    covered = typed.cancer_id.isin(RAW_COVERED_CANCERS)
    assert typed.loc[covered].groupby("cancer_id").atac_available.any().all()
    assert typed.loc[typed.atac_available, "atac_unavailable_reason"].isna().all()
    assert typed.loc[typed.atac_available, "atac_context_probability"].between(0, 1).all()

    lineage = json.loads((output / "LINEAGE.json").read_text(encoding="utf-8"))
    validate_atac_oof_lineage(
        lineage,
        predictions_sha256=file_sha256(output / "atac_typed_predictions.parquet"),
    )
    assert len(lineage["fold_status"]) == 5
    for fold in range(5):
        assert (output / "checkpoints" / f"atac_patient_fold_{fold}.json").is_file()
        assert (output / "checkpoints" / f"atac_patient_fold_{fold}.SUCCESS.json").is_file()
        fold_table = pd.read_parquet(
            output / "patient_fold_oof_predictions" / f"patient_fold={fold}" / "part-0.parquet"
        )
        assert len(fold_table) == len(typed)
        assert fold_table.patient_fold_id.eq(fold).all()

    with pytest.raises(AtacTrainingError, match="refuses output reuse"):
        run_atac_training(
            candidates_path=paths["candidates"],
            patient_folds_path=paths["folds"],
            pathway_membership_path=paths["membership"],
            materialization_success_path=paths["materialization"],
            output_root=output,
            training_run_id="must-not-overwrite",
            config=AtacTrainingConfig(expected_rows_per_cancer=None),
        )


def test_materialized_matrix_hash_mismatch_fails_before_output(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    payload = json.loads(paths["materialization"].read_text(encoding="utf-8"))
    payload["records"][0]["sha256"] = "0" * 64
    paths["materialization"].write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "must_not_exist"
    with pytest.raises(AtacTrainingError, match="hash mismatch"):
        run_atac_training(
            candidates_path=paths["candidates"],
            patient_folds_path=paths["folds"],
            pathway_membership_path=paths["membership"],
            materialization_success_path=paths["materialization"],
            output_root=output,
            training_run_id="hash-failure",
            config=AtacTrainingConfig(expected_rows_per_cancer=None),
        )
    assert not output.exists()
