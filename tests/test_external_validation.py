import pandas as pd

from cc_hhgt.external_validation import (
    _base_frame,
    _evaluate_role,
    classify_lncrnadisease_method,
    map_cancer_value,
    normalize_direction,
)


def test_external_value_normalizers_handle_missing_values():
    assert normalize_direction(pd.NA) == "unknown"
    assert normalize_direction("expression increased") == "positive"
    assert normalize_direction("knockdown suppressed growth") == "negative"
    assert classify_lncrnadisease_method(pd.NA) == (
        "curated_unspecified",
        False,
        False,
    )
    assert classify_lncrnadisease_method("RT-PCR and western blot") == (
        "experimental_or_clinical",
        True,
        False,
    )
    assert classify_lncrnadisease_method("LDAP prediction") == (
        "computational_prediction",
        False,
        True,
    )


def test_cancer_mapping_does_not_broadcast_ambiguous_disease():
    mapped, candidates, status = map_cancer_value(
        "lung cancer", aliases={}, cancer_hint=pd.NA
    )
    assert pd.isna(mapped)
    assert candidates == "LUAD;LUSC"
    assert status == "ambiguous_multiple_tcga"
    mapped, candidates, status = map_cancer_value(
        "glioblastoma multiforme", aliases={}, cancer_hint=pd.NA
    )
    assert mapped == "GBM"
    assert candidates == "GBM"
    assert status == "mapped_phrase"


def test_base_frame_resets_nonconsecutive_source_indices():
    index = [10, 30]
    frame = _base_frame(
        "GSE85011",
        pd.Series(["GSM1", "GSM2"], index=index),
        pd.Series(["LINC1", "LINC2"], index=index),
        pd.Series(["HeLa", "U87"], index=index),
        pd.Series(["CRISPRi", "CRISPRi"], index=index),
        pd.Series(["hit", "hit"], index=index),
        pd.Series(["27980086", "27980086"], index=index),
        pd.Series(["", ""], index=index),
        "experimental_crispri_growth_modifier",
        True,
        False,
    )
    assert len(frame) == 2
    assert frame.source_row_id.tolist() == ["GSM1", "GSM2"]


def test_rank_validation_uses_external_positives_only():
    ranks = pd.DataFrame(
        {
            "cancer_id": ["BRCA", "BRCA", "BRCA"],
            "lncrna_id": ["L1", "L2", "L3"],
            "rank": [1, 2, 3],
            "percentile": [1.0, 2 / 3, 1 / 3],
            "n_ranked_lncRNAs": [3, 3, 3],
        }
    )
    evidence = pd.DataFrame(
        {
            "source_database": ["Lnc2Cancer", "RNADisease"],
            "cancer_id": ["BRCA", "BRCA"],
            "lncrna_id": ["L1", "L2"],
            "primary_validation_eligible": [True, False],
            "consistency_validation_eligible": [False, True],
        }
    )
    metrics, matches = _evaluate_role(ranks, evidence, "primary", [1])
    assert len(matches) == 1
    assert metrics.loc[0, "hits_at_k"] == 1
    assert metrics.loc[0, "recall_at_k"] == 1
