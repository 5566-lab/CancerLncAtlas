from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cc_hhgt.v32.atac_training import AtacTrainingError
from cc_hhgt.v32.atac_training import normalise_patient_folds as normalise_atac_folds
from cc_hhgt.v32.clinical_training import ClinicalTrainingError, patient_fold_manifest
from cc_hhgt.v32.genomic_training import GenomicTrainingError
from cc_hhgt.v32.genomic_training import normalise_patient_folds as normalise_genomic_folds
from cc_hhgt.v32.routing_fair_input import RoutingFairInputError, _validate_patient_folds
from cc_hhgt.v32.state_training import StateTrainingError, validate_fold_manifest


def _folds() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cancer_id": ["ACC"] * 5,
            "sample_id": [f"ACC_SAMPLE_{index}" for index in range(5)],
            "patient_id": [f"ACC_PATIENT_{index}" for index in range(5)],
            "patient_fold_id": list(range(5)),
            "fold_seed": [20260726] * 5,
        }
    )


def test_formal_fold_consumers_keep_explicit_patient_identity(tmp_path) -> None:
    frame = _folds()
    expected = set(frame.patient_id)
    assert set(normalise_genomic_folds(frame).patient_id) == expected
    assert set(normalise_atac_folds(frame).patient_id) == expected
    assert set(patient_fold_manifest(frame).patient_id) == expected
    assert set(validate_fold_manifest(frame).patient_id) == expected
    path = tmp_path / "PATIENT_FOLD_MANIFEST.tsv"
    frame.to_csv(path, sep="\t", index=False)
    summary = _validate_patient_folds(path, cancers=["ACC"], expected_seed=20260726)
    assert summary["patients"] == 5


@pytest.mark.parametrize(
    ("consumer", "error"),
    [
        (normalise_genomic_folds, (GenomicTrainingError, ValueError)),
        (normalise_atac_folds, AtacTrainingError),
        (patient_fold_manifest, ClinicalTrainingError),
        (validate_fold_manifest, StateTrainingError),
    ],
)
def test_formal_fold_consumers_reject_missing_patient_id(consumer, error) -> None:
    with pytest.raises(error):
        consumer(_folds().drop(columns="patient_id"))


def test_fair_input_rejects_missing_patient_id(tmp_path) -> None:
    path = tmp_path / "PATIENT_FOLD_MANIFEST.tsv"
    _folds().drop(columns="patient_id").to_csv(path, sep="\t", index=False)
    with pytest.raises(RoutingFairInputError, match="patient_id"):
        _validate_patient_folds(path, cancers=["ACC"], expected_seed=20260726)


def test_preflights_require_receipt_and_never_derive_fold_patient_from_sample() -> None:
    root = Path(__file__).resolve().parents[1]
    atac = (root / "scripts/audit_v32_atac_readiness.py").read_text(encoding="utf-8")
    cnv = (root / "scripts/preflight_v32_full33_segment_cnv.py").read_text(encoding="utf-8")
    routing = (
        root / "scripts/audit_v32_routing_fair_compare_readiness.py"
    ).read_text(encoding="utf-8")
    for source in (atac, cnv, routing):
        assert "patient-fold-authority-receipt" in source
        assert "validate_frozen_v32_patient_fold_binding" in source
    assert "folds.sample_id.astype(str).str[:12]" not in atac
    assert 'patient_column = "patient_id" if "patient_id" in folds else "sample_id"' not in cnv
    assert "fold_sample_id_fallback_used" in atac
    assert "fold_sample_id_fallback_used" in cnv
    # Raw official ATAC Case_ID-to-patient mapping is a source-specific mapping,
    # not permission to infer fold patients from sample IDs.
    assert "mapping.Case_ID.astype(str).str[:12]" in atac


def test_clinical_entity_path_preserves_explicit_patient_ids() -> None:
    root = Path(__file__).resolve().parents[1]
    entity = (root / "cc_hhgt/v32/clinical_entity_training.py").read_text(encoding="utf-8")
    runner = (root / "scripts/run_v32_clinical_entity_training.py").read_text(encoding="utf-8")
    assert "str(value)[:12]" not in entity
    assert "str.slice(0, 12)" not in entity
    assert "str.slice(0, 12)" not in runner
    assert '"source": "EXPLICIT_PATIENT_ID_COLUMN"' in runner
    assert '"sample_id_patient_fallback_used": False' in runner
