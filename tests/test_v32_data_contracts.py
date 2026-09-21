from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cc_hhgt.v32.lncrna_scope import build_lncrna_scope
from cc_hhgt.v32.patient_folds import assign_outer_split, build_patient_fold_manifest
from cc_hhgt.v32.pathway_activity import (
    ActivityScaler,
    compute_rank_mean_activity,
    validate_protein_coding_pathway_membership,
)


def test_full_locked_lncrna_counts_and_local_scope() -> None:
    annotation = [f"L{i:05d}" for i in range(16889)]
    cancers = ["C1", "C2", "C3"]
    samples = pd.DataFrame(
        [(cancer, f"{cancer}_S{i}") for cancer in cancers for i in range(10)],
        columns=["cancer_id", "sample_id"],
    )
    rows = []
    for cancer in cancers:
        sample = f"{cancer}_S0"
        rows.extend((cancer, sample, lnc, 1.0) for lnc in annotation[:4712])
    # One cancer-only lncRNA proves that the local branch is retained rather
    # than removed by the shared filter.
    rows.append(("C1", "C1_S0", annotation[4712], 1.0))
    expression = pd.DataFrame(rows, columns=["cancer_id", "sample_id", "lncrna_id", "logcpm"])
    scope = build_lncrna_scope(expression, samples, annotation)
    assert scope.source_universe_n == 16889
    assert scope.shared_universe_n == 4712
    local = scope.eligibility.loc[
        scope.eligibility.cancer_id.eq("C1") & scope.eligibility.lncrna_id.eq(annotation[4712])
    ].iloc[0]
    assert local.shared_or_local_scope == "cancer_local"
    other = scope.eligibility.loc[
        scope.eligibility.cancer_id.eq("C2") & scope.eligibility.lncrna_id.eq(annotation[4712])
    ].iloc[0]
    assert other.shared_or_local_scope == "not_expression_eligible"


def test_patient_folds_are_deterministic_and_disjoint() -> None:
    samples = pd.DataFrame(
        [
            (cancer, f"{cancer}_S{i}", f"{cancer}_P{i // 2}")
            for cancer in ["A", "B"]
            for i in range(30)
        ],
        columns=["cancer_id", "sample_id", "patient_id"],
    )
    first = build_patient_fold_manifest(samples)
    second = build_patient_fold_manifest(samples.sample(frac=1, random_state=3))
    pd.testing.assert_frame_equal(first, second)
    split = assign_outer_split(first, 2)
    assert set(split.split) == {"train", "validation", "test"}
    assert split.sample_id.nunique() == len(split)
    assert split.groupby(["cancer_id", "patient_id"]).patient_fold_id.nunique().eq(1).all()
    assert split.groupby(["cancer_id", "patient_id"]).split.nunique().eq(1).all()


def test_patient_fold_builder_rejects_missing_patient_identity() -> None:
    samples = pd.DataFrame(
        [("A", f"S{i}") for i in range(10)],
        columns=["cancer_id", "sample_id"],
    )
    with pytest.raises(ValueError, match="patient_id"):
        build_patient_fold_manifest(samples)


def test_activity_rejects_lncrna_members_and_scales_from_train_only() -> None:
    annotation = pd.DataFrame(
        {"gene_id": ["G1", "G2", "L1"], "gene_type": ["protein_coding", "protein_coding", "lncRNA"]}
    )
    invalid = pd.DataFrame({"pathway_id": ["P"], "gene_id": ["L1"]})
    with pytest.raises(RuntimeError, match="Non-protein-coding"):
        validate_protein_coding_pathway_membership(invalid, annotation)
    members = pd.DataFrame({"pathway_id": ["P", "P"], "gene_id": ["G1", "G2"]})
    expression = pd.DataFrame(
        [(f"S{i}", gene, float(i + offset)) for i in range(8) for gene, offset in [("G1", 0), ("G2", 2)]],
        columns=["sample_id", "gene_id", "expression"],
    )
    activity = compute_rank_mean_activity(expression, members, annotation)
    scaler = ActivityScaler.fit(activity, {"S0", "S1", "S2", "S3"})
    transformed = scaler.transform(activity)
    assert np.isfinite(transformed.pathway_activity_scaled).all()
