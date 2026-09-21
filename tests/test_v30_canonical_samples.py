from __future__ import annotations

import pandas as pd
import pytest

from cc_hhgt.v30_samples import build_canonical_sample_outputs


def fold_manifest() -> pd.DataFrame:
    rows = []
    for patient, sample in [
        ("TCGA-AA-0001", "TCGA-AA-0001-01A-01R-0000-01"),
        ("TCGA-AA-0002", "TCGA-AA-0002-01A-01R-0000-01"),
    ]:
        for fold in range(2):
            rows.append(
                {
                    "cancer_id": "BRCA",
                    "patient_id": patient,
                    "sample_id": sample,
                    "patient_fold_id": f"LOCO_BRCA__PF0{fold}",
                    "split": "test" if fold == (0 if patient.endswith("1") else 1) else "train",
                }
            )
    return pd.DataFrame(rows)


def score_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["TCGA-AA-0001-01", "TCGA-AA-0001-11", "TCGA-AA-0002-01", "LEGACY-AA-1"],
            "patient_id": ["TCGA-AA-0001", "TCGA-AA-0001", "TCGA-AA-0002", "LEGACY-AA"],
            "cancer_type": ["BRCA", "BRCA", "BRCA", "LGG"],
            "sample_type_code": [1, 11, 1, 1],
            "is_tumor": [True, False, True, True],
            "stemness_rna::RNAss": [0.2, 0.9, 0.7, 0.5],
            "stemness_dna::DNAss": [0.1, 0.8, float("nan"), 0.5],
        }
    )


def test_canonical_exact_sample_join_excludes_normal_and_never_imputes_state() -> None:
    canonical, expanded, states, eligibility, audit = build_canonical_sample_outputs(
        fold_manifest(),
        score_table(),
        target_states=["stemness_rna::RNAss", "stemness_dna::DNAss"],
        reference_only=[],
        minimum_observed=2,
    )
    assert len(canonical) == 2
    assert canonical.is_tumor.all()
    assert set(canonical.sample_type_code.astype(int)) == {1}
    assert len(expanded) == 4
    assert not states.state_value.isna().any()
    assert len(states.loc[states.state_id.eq("stemness_rna::RNAss")]) == 2
    assert len(states.loc[states.state_id.eq("stemness_dna::DNAss")]) == 1
    dna = eligibility.loc[eligibility.state_id.eq("stemness_dna::DNAss")].iloc[0]
    assert dna.n_observed == 1
    assert dna.n_missing == 1
    assert dna.eligibility == "UNAVAILABLE"
    assert audit["normal_samples"] == 0
    assert audit["state_nan_rows"] == 0
    assert audit["source_score_invalid_barcode_rows"] == 1
    assert audit["source_score_type11_rows"] == 1


def test_selected_normal_sample_fails_closed() -> None:
    manifest = fold_manifest().copy()
    manifest.loc[manifest.patient_id.eq("TCGA-AA-0001"), "sample_id"] = "TCGA-AA-0001-11A-01R-0000-01"
    with pytest.raises(RuntimeError, match="non-tumor sample types"):
        build_canonical_sample_outputs(
            manifest,
            score_table(),
            target_states=["stemness_rna::RNAss", "stemness_dna::DNAss"],
        )
