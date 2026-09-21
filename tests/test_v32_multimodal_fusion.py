from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from cc_hhgt.v32.multimodal_fusion import (
    ANALYSIS_VERSION,
    ExpertSpec,
    MultimodalFusionError,
    ResidualFusionConfig,
    ResidualFusionModel,
    apply_residual_fusion_model,
    artifact_sha256,
    build_pair_aggregated_fusion_frame,
    build_pair_aggregated_fusion_frame_from_parquet,
    materialize_multimodal_fusion_release,
    train_pair_blocked_crossfit_fusion,
    validate_source_declarations,
)
from cc_hhgt.v32.multimodal_fusion_query import (
    MultimodalFusionQueryAssetError,
    MultimodalFusionQueryInputError,
    MultimodalFusionReleaseQuery,
)


def _fixtures():
    rows = []
    for lnc_index in range(45):
        for pathway_index in range(4):
            for cancer_index, cancer in enumerate(("ACC", "BRCA")):
                latent = (
                    0.9 * np.sin((lnc_index + 1) * 0.37)
                    + 0.7 * np.cos((pathway_index + 1) * 0.91)
                    + 0.2 * cancer_index
                )
                primary = 1.0 / (1.0 + np.exp(-(0.75 * latent - 0.15)))
                rows.append(
                    {
                        "cancer_id": cancer,
                        "lncrna_id": f"LNC:ENSG{lnc_index:011d}",
                        "pathway_id": f"PATH:{pathway_index:03d}",
                        "latent": latent,
                        "association_membership_probability": primary,
                        "analysis_version": ANALYSIS_VERSION,
                    }
                )
    base = pd.DataFrame(rows)
    primary = base.drop(columns="latent")
    folds = []
    for fold in range(5):
        local = base[["cancer_id", "lncrna_id", "pathway_id"]].copy()
        local["patient_fold_id"] = f"PF_{fold}"
        local["held_out_proxy_label"] = (
            base.latent.to_numpy() + (fold - 2) * 0.08 > 0
        ).astype("int8")
        folds.append(local)

    genomic = base[["cancer_id", "lncrna_id", "pathway_id"]].copy()
    genomic["genomic_probability"] = 1.0 / (
        1.0 + np.exp(-(1.25 * base.latent.to_numpy() + 0.1))
    )
    genomic["genomic_available"] = True
    genomic["analysis_version"] = ANALYSIS_VERSION

    evidence = base[["cancer_id", "lncrna_id", "pathway_id"]].copy()
    evidence["evidence_probability"] = 1.0 / (
        1.0 + np.exp(-(1.45 * base.latent.to_numpy() - 0.05))
    )
    evidence["evidence_available"] = True
    evidence["analysis_version"] = ANALYSIS_VERSION

    # Exercise exact fallback and typed null semantics.
    fallback = np.arange(len(base)) % 19 == 0
    genomic.loc[fallback, "genomic_probability"] = np.nan
    genomic.loc[fallback, "genomic_available"] = False
    evidence.loc[fallback, "evidence_probability"] = np.nan
    evidence.loc[fallback, "evidence_available"] = False
    return primary, folds, genomic, evidence


GENOMIC = ExpertSpec(
    expert_id="genomic",
    probability_column="genomic_probability",
    availability_column="genomic_available",
    endpoint_roles=("discovery", "confidence"),
    split_unit="patient_fold_oof",
    source_role="current_v32_oof_prediction",
)
EVIDENCE = ExpertSpec(
    expert_id="evidence_transformer",
    probability_column="evidence_probability",
    availability_column="evidence_available",
    endpoint_roles=("confidence",),
    split_unit="lncrna_exact_pathway_pair_blocked",
    source_role="current_v32_oof_prediction",
    direct_target_evidence=True,
)


def test_pair_blocked_fusion_preserves_primary_and_trains_secondary_scores():
    primary, folds, genomic, evidence = _fixtures()
    frame = build_pair_aggregated_fusion_frame(
        primary,
        folds,
        {GENOMIC: genomic, EVIDENCE: evidence},
    )
    assert len(frame) == len(primary)
    assert set(frame.fusion_target_fold_count) == {5}
    pair_folds = frame.groupby(["lncrna_id", "pathway_id"]).fusion_pair_fold.nunique()
    assert pair_folds.eq(1).all()
    assert set(frame.fusion_pair_fold) == set(range(5))

    config = ResidualFusionConfig(
        max_steps=160,
        patience=12,
        evaluation_interval=5,
        max_train_rows=10_000,
        max_validation_rows=10_000,
        batch_size=512,
        learning_rate=0.03,
    )
    discovery = train_pair_blocked_crossfit_fusion(
        frame, (GENOMIC, EVIDENCE), endpoint="discovery", config=config
    )
    assert discovery.final_model.expert_ids == ("genomic",)
    assert len(discovery.fold_models) == 5
    assert len(discovery.oof_prediction) == len(primary)
    assert not discovery.oof_prediction[["cancer_id", "lncrna_id", "pathway_id"]].duplicated().any()
    assert all(weight >= 0 for model in discovery.fold_models for weight in model.weights)
    assert all(model.optimiser_steps > 0 for model in discovery.fold_models)
    assert all(
        model.initial_parameter_sha256 != model.final_parameter_sha256
        for model in discovery.fold_models
    )

    public = apply_residual_fusion_model(discovery.final_model, frame)
    fallback = public.primary_fallback
    assert fallback.any()
    assert np.array_equal(
        public.loc[fallback, "discovery_adjusted_probability"].to_numpy(),
        public.loc[fallback, "primary_probability"].to_numpy(),
    )
    assert public.loc[~public.genomic_available, "genomic_logit_contribution"].isna().all()
    assert public.primary_ranking_unchanged.all()
    assert public.adjusted_ranking_is_secondary.all()
    assert not public.used_for_primary_release.any()
    assert not public.old_checkpoint_loaded.any()

    confidence = train_pair_blocked_crossfit_fusion(
        frame, (GENOMIC, EVIDENCE), endpoint="confidence", config=config
    )
    assert confidence.final_model.expert_ids == (
        "genomic",
        "evidence_transformer",
    )
    assert "fused_confidence_probability" in confidence.oof_prediction


def test_zero_weight_available_expert_falls_back_bit_for_bit() -> None:
    frame = pd.DataFrame(
        {
            "cancer_id": ["ACC", "BRCA"],
            "lncrna_id": ["LNC:ENSG1", "LNC:ENSG2"],
            "pathway_id": ["PATH:A", "PATH:B"],
            "primary_probability": [0.123456789012345, 0.876543210987654],
            "genomic_probability": [0.8, 0.2],
            "genomic_available": [True, True],
        }
    )
    model = ResidualFusionModel(
        endpoint="discovery",
        expert_ids=("genomic",),
        centres=(0.0,),
        weights=(0.0,),
        seed=1,
        initial_parameter_sha256="0" * 64,
        final_parameter_sha256="1" * 64,
        optimiser_steps=1,
        validation_logloss=0.5,
    )
    output = apply_residual_fusion_model(model, frame)
    assert output.no_available_auxiliary_expert.eq(False).all()
    assert output.zero_total_contribution.all()
    assert output.primary_fallback.all()
    assert np.array_equal(
        output.discovery_adjusted_probability.to_numpy(),
        output.primary_probability.to_numpy(),
    )


def test_fusion_fails_closed_on_untyped_missing_or_direct_discovery_evidence():
    primary, folds, genomic, evidence = _fixtures()
    bad = evidence.copy()
    index = bad.index[0]
    bad.loc[index, "evidence_available"] = False
    bad.loc[index, "evidence_probability"] = 0.0
    with pytest.raises(MultimodalFusionError, match="fills unavailable"):
        build_pair_aggregated_fusion_frame(
            primary, folds, {GENOMIC: genomic, EVIDENCE: bad}
        )

    with pytest.raises(MultimodalFusionError, match="forbidden in discovery"):
        ExpertSpec(
            expert_id="direct_bad",
            probability_column="p",
            availability_column="a",
            endpoint_roles=("discovery",),
            split_unit="pair",
            direct_target_evidence=True,
        ).validate()


def test_streamed_parquet_builder_matches_in_memory_contract(tmp_path):
    primary, folds, genomic, evidence = _fixtures()
    primary_path = tmp_path / "primary.parquet"
    primary.to_parquet(primary_path, index=False)
    fold_paths = []
    for index, frame in enumerate(folds):
        path = tmp_path / f"fold_{index}.parquet"
        frame.to_parquet(path, index=False)
        fold_paths.append(path)
    genomic_path = tmp_path / "genomic.parquet"
    evidence_path = tmp_path / "evidence.parquet"
    genomic.to_parquet(genomic_path, index=False)
    evidence.to_parquet(evidence_path, index=False)

    expected = build_pair_aggregated_fusion_frame(
        primary,
        folds,
        {GENOMIC: genomic, EVIDENCE: evidence},
    )
    observed, audit = build_pair_aggregated_fusion_frame_from_parquet(
        primary_path,
        fold_paths,
        {GENOMIC: genomic_path, EVIDENCE: evidence_path},
        expected_rows=len(primary),
        temp_directory=tmp_path / "duckdb_tmp",
    )
    pd.testing.assert_frame_equal(
        observed.sort_values(["cancer_id", "lncrna_id", "pathway_id"]).reset_index(drop=True),
        expected.sort_values(["cancer_id", "lncrna_id", "pathway_id"]).reset_index(drop=True),
        check_dtype=False,
    )
    assert audit["status"] == "PASS"
    assert audit["candidate_rows"] == len(primary)
    assert audit["all_five_primary_folds_present"] is True
    assert audit["all_experts_full_candidate_universe"] is True
    assert audit["all_unavailable_probabilities_null"] is True


def test_fusion_rejects_legacy_derived_sources():
    with pytest.raises(MultimodalFusionError, match="Legacy derived"):
        validate_source_declarations(
            [
                {
                    "path": "results/v2_9/final_predictions.parquet",
                    "sha256": "a" * 64,
                    "source_role": "current_v32_oof_prediction",
                    "generation": "current_v32",
                }
            ]
        )


def test_fusion_release_is_hash_bound_and_keeps_oof_targets_private(tmp_path):
    primary, folds, genomic, evidence = _fixtures()
    frame = build_pair_aggregated_fusion_frame(
        primary, folds, {GENOMIC: genomic, EVIDENCE: evidence}
    )
    config = ResidualFusionConfig(
        max_steps=80,
        patience=8,
        evaluation_interval=5,
        max_train_rows=10_000,
        max_validation_rows=10_000,
        batch_size=512,
        learning_rate=0.03,
    )
    discovery = train_pair_blocked_crossfit_fusion(
        frame, (GENOMIC, EVIDENCE), endpoint="discovery", config=config
    )
    confidence = train_pair_blocked_crossfit_fusion(
        frame, (GENOMIC, EVIDENCE), endpoint="confidence", config=config
    )
    source = tmp_path / "current_v32_source.parquet"
    primary.to_parquet(source, index=False)
    result = materialize_multimodal_fusion_release(
        tmp_path / "release",
        frame,
        discovery,
        confidence,
        source_declarations=[
            {
                "path": str(source),
                "sha256": artifact_sha256(source),
                "source_role": "current_v32_oof_prediction",
                "generation": "current_v32",
            }
        ],
        training_run_id="V32-FUSION-SYNTHETIC-TEST",
    )
    assert artifact_sha256(result["binding_path"]) == result["binding_sha256"]
    public = pd.read_parquet(result["prediction_path"])
    assert len(public) == len(primary)
    assert "fusion_target" not in public
    assert "held_out_proxy_label" not in public
    assert public.primary_ranking_unchanged.all()
    assert not public.used_for_primary_release.any()
    assert {
        "genomic_native_probability",
        "genomic_native_available",
        "evidence_transformer_native_probability",
        "evidence_transformer_native_available",
    }.issubset(public.columns)
    with open(result["binding_path"], encoding="utf-8") as handle:
        binding = json.load(handle)
    assert binding["zero_total_contribution_exact_primary_fallback"] is True
    assert binding["native_expert_probabilities_public"] is True
    assert binding["public_native_experts"] == ["genomic", "evidence_transformer"]
    private = pd.read_parquet(
        tmp_path / "release" / "discovery_oof.PRIVATE.parquet"
    )
    assert "fusion_target" in private
    query = MultimodalFusionReleaseQuery(
        result["binding_path"],
        expected_binding_sha256=result["binding_sha256"],
    )
    status = query.capability_status()
    assert status["primary_score_preserved"] is True
    response = query.query_scores(
        lncrna_id="ENSG00000000000",
        cancer_id="ACC",
        order_by="confidence",
        limit=3,
    )
    assert response["returned_rows"] == 3
    assert response["primary_ranking_unchanged"] is True
    assert response["rows"][0]["lncrna_id"] == "LNC:ENSG00000000000"
    with pytest.raises(MultimodalFusionQueryInputError, match="order_by"):
        query.query_scores(lncrna_id="ENSG00000000000", order_by="bad")
    with pytest.raises(MultimodalFusionQueryAssetError, match="SHA mismatch"):
        MultimodalFusionReleaseQuery(
            result["binding_path"], expected_binding_sha256="0" * 64
        )
    with pytest.raises(MultimodalFusionError, match="overwrite"):
        materialize_multimodal_fusion_release(
            tmp_path / "release",
            frame,
            discovery,
            confidence,
            source_declarations=[
                {
                    "path": str(source),
                    "sha256": artifact_sha256(source),
                    "source_role": "current_v32_oof_prediction",
                    "generation": "current_v32",
                }
            ],
            training_run_id="V32-FUSION-SYNTHETIC-TEST",
        )
